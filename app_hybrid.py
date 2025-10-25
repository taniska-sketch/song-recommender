import os
import re
import sys
import json
import time
import hashlib
import logging
import traceback
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import joblib
import requests

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

# Firebase
from firebase_admin import credentials, firestore, initialize_app as fb_initialize_app

# Optional gdown; we’ll import lazily
_GDOWN_AVAILABLE = False
try:
    import gdown  # type: ignore
    _GDOWN_AVAILABLE = True
except Exception:
    _GDOWN_AVAILABLE = False

# ---------------- Logging ----------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("hybrid-app")

# ---------------- FastAPI ----------------
app = FastAPI(title="Hybrid Song Recommender API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Env & Paths ----------------
MODELS_DIR = os.getenv("MODELS_DIR", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

ARTIFACT_FILES = {
    "content":  os.path.join(MODELS_DIR, "content_similarity.pkl"),
    "cf":       os.path.join(MODELS_DIR, "item_similarity_cf_matrix.pkl"),
    "useritem": os.path.join(MODELS_DIR, "user_item_matrix.pkl"),
    "scaler":   os.path.join(MODELS_DIR, "scaler.pkl"),
}

ENV_KEYS: Dict[str, Tuple[str, ...]] = {
    "content":  ("CONTENT_SIMILARITY_URL", "CONTENT_SIM_URL"),
    "cf":       ("CF_SIMILARITY_URL", "ITEM_SIM_CF_URL"),
    "useritem": ("USER_ITEM_MATRIX_URL", "USERITEM_URL"),
    "scaler":   ("SCALER_URL",),
}

DOWNLOAD_RETRIES = int(os.getenv("DOWNLOAD_RETRIES", "3"))
FIREBASE_CREDENTIALS_PATH = os.getenv(
    "FIREBASE_CREDENTIALS_PATH",
    "/opt/render/project/src/.firebase_key.json"
)
HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", "0.5"))

# ---------------- Small utils ----------------
def sha256_short(path: str, n: int = 12) -> Optional[str]:
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]

def fsize_kb(path: str) -> Optional[float]:
    if not os.path.exists(path):
        return None
    return round(os.path.getsize(path) / 1024.0, 2)

def get_env_url(keys: Tuple[str, ...]) -> Optional[str]:
    for k in keys:
        v = os.getenv(k)
        if v:
            return v.strip()
    return None

_DRIVE_ID_RE = re.compile(r"/d/([^/]+)/|id=([^&]+)")
def parse_drive_id(url: str) -> Optional[str]:
    m = _DRIVE_ID_RE.search(url)
    if not m:
        return None
    return m.group(1) or m.group(2)

def drive_direct_url(file_id: str) -> str:
    # Works for direct HTTP fallback
    return f"https://drive.google.com/uc?export=download&id={file_id}"

def looks_like_html(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            head = f.read(256).lower()
        return b"<html" in head or b"<!doctype html" in head or b"google" in head and b"drive" in head
    except Exception:
        return False

def download_artifact(url: str, out_path: str, name_for_error: str) -> None:
    """Download from Google Drive (gdown if present) or direct HTTP.
       Detects HTML (quota/restricted) and errors clearly."""
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        log.info(f"↪ {name_for_error} already present: {out_path} ({fsize_kb(out_path)} KB, sha256={sha256_short(out_path)})")
        return
    if not url:
        raise FileNotFoundError(f"No URL provided for {name_for_error}. Set the env var correctly.")

    last_err = None
    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        try:
            log.info(f"⬇️  Downloading {name_for_error} (attempt {attempt}/{DOWNLOAD_RETRIES})")
            file_id = parse_drive_id(url)
            if _GDOWN_AVAILABLE:
                # gdown handles Drive confirm/virus/quota better
                gdown.download(url if file_id is None else f"https://drive.google.com/uc?id={file_id}",
                               out_path, quiet=False, fuzzy=True)
            else:
                # Fallback direct HTTP
                dl_url = url if file_id is None else drive_direct_url(file_id)
                with requests.get(dl_url, stream=True, timeout=90) as r:
                    r.raise_for_status()
                    with open(out_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=1 << 20):
                            if chunk:
                                f.write(chunk)

            # Sanity checks
            if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
                raise RuntimeError("Downloaded file is empty.")
            if looks_like_html(out_path):
                # Replace vague joblib KeyError: 60 with actionable message
                with open(out_path, "rb") as f:
                    sample = f.read(200).decode("utf-8", "ignore")
                raise RuntimeError(
                    f"{name_for_error} appears to be an HTML page, not a pickle.\n"
                    f"Likely causes: Google Drive link is not set to 'Anyone with the link' OR quota exceeded.\n"
                    f"First 200 bytes:\n{sample}"
                )

            log.info(f"✅ Downloaded {name_for_error}: {out_path} ({fsize_kb(out_path)} KB, sha256={sha256_short(out_path)})")
            return
        except Exception as e:
            last_err = e
            log.warning(f"Download failure for {name_for_error}: {e}")
            if attempt < DOWNLOAD_RETRIES:
                backoff = 2 ** (attempt - 1)
                log.info(f"Retrying {name_for_error} in {backoff}s ...")
                time.sleep(backoff)
    raise RuntimeError(f"Failed to download {name_for_error} after {DOWNLOAD_RETRIES} attempts. Last error: {last_err}")

def load_pickle(path: str):
    with open(path, "rb") as f:
        return joblib.load(f)

def _ensure_df(x, name: str) -> pd.DataFrame:
    if isinstance(x, pd.DataFrame):
        return x
    if isinstance(x, np.ndarray):
        return pd.DataFrame(x)
    try:
        return pd.DataFrame(x)
    except Exception:
        raise TypeError(f"{name} must be DataFrame/ndarray-like; got {type(x)}")

def _user_vector(useritem: pd.DataFrame, user_id: str) -> Optional[pd.Series]:
    if user_id in useritem.index:
        return pd.to_numeric(useritem.loc[user_id], errors="coerce").fillna(0.0)
    if user_id in useritem.columns:
        return pd.to_numeric(useritem[user_id], errors="coerce").fillna(0.0)
    return None

def _stringify_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    return df

def _apply_scaler(v: np.ndarray, scaler) -> np.ndarray:
    try:
        if scaler is not None and hasattr(scaler, "transform"):
            return scaler.transform(v.reshape(-1, 1)).ravel()
    except Exception as e:
        log.warning(f"Scaler transform failed; using raw scores. Reason: {e}")
    return v

# ---------------- Global state ----------------
app.state.artifacts: Dict[str, object] = {}
app.state.firebase_ok: bool = False
app.state.firestore = None

# ---------------- Startup ----------------
@app.on_event("startup")
def startup():
    log.info("🚀 Startup: download → load → Firebase")

    # 1) Downloads
    env_urls = {k: get_env_url(ENV_KEYS[k]) for k in ARTIFACT_FILES.keys()}
    for key, path in ARTIFACT_FILES.items():
        try:
            download_artifact(env_urls[key], path, name_for_error=f"{key}.pkl")
        except Exception as e:
            log.error(f"❌ Download error for {key}: {e}")
            log.error("Traceback:\n" + traceback.format_exc())

    # 2) Load
    try:
        loaded = {}
        for k, p in ARTIFACT_FILES.items():
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing expected file after download: {p}")
            obj = load_pickle(p)
            loaded[k] = obj
            log.info(f"• Loaded {k}: type={type(obj).__name__}, size={fsize_kb(p)} KB")
        app.state.artifacts = loaded
        log.info("✅ All artifacts loaded")
    except Exception as e:
        log.error(f"❌ Failed to load model artifacts: {e}")
        log.error("Traceback:\n" + traceback.format_exc())
        app.state.artifacts = {}

    # 3) Firebase
    try:
        if os.path.exists(FIREBASE_CREDENTIALS_PATH):
            cred = credentials.Certificate(FIREBASE_CREDENTIALS_PATH)
            fb_initialize_app(cred)
            app.state.firestore = firestore.client()
            app.state.firebase_ok = True
            log.info("🔥 Firebase initialized (Firestore ready)")
        else:
            log.warning(f"Firebase key not found at {FIREBASE_CREDENTIALS_PATH}. Skipping Firebase.")
    except Exception as e:
        log.error(f"❌ Firebase init failed: {e}")
        log.error("Traceback:\n" + traceback.format_exc())
        app.state.firebase_ok = False

# ---------------- Routes ----------------
@app.get("/")
def root():
    return {
        "name": "Hybrid Song Recommender",
        "version": "1.0",
        "endpoints": ["/healthz", "/debug/artifacts", "/recommend_hybrid?user_id=U1&top_k=10&alpha=0.5"],
    }

@app.get("/healthz")
def healthz():
    return {
        "ok": True,
        "artifacts_loaded": bool(app.state.artifacts),
        "firebase_ok": app.state.firebase_ok,
        "has_content": "content" in app.state.artifacts,
        "has_cf": "cf" in app.state.artifacts,
        "has_useritem": "useritem" in app.state.artifacts,
        "has_scaler": "scaler" in app.state.artifacts,
    }

@app.get("/debug/artifacts")
def debug_artifacts():
    info = {}
    for key, path in ARTIFACT_FILES.items():
        info[key] = {
            "exists": os.path.exists(path),
            "path": path,
            "size_kb": fsize_kb(path),
            "sha256_12": sha256_short(path),
            "env": get_env_url(ENV_KEYS[key]),
        }
    return info

@app.get("/recommend_hybrid")
def recommend_hybrid(
    user_id: str = Query(...),
    top_k: int = Query(10, ge=1, le=200),
    alpha: float = Query(HYBRID_ALPHA, ge=0.0, le=1.0),
):
    if not app.state.artifacts:
        raise HTTPException(status_code=503, detail="Artifacts not loaded")

    content = _ensure_df(app.state.artifacts.get("content"), "content_similarity")
    cf = _ensure_df(app.state.artifacts.get("cf"), "item_similarity_cf_matrix")
    useritem = _ensure_df(app.state.artifacts.get("useritem"), "user_item_matrix")
    scaler = app.state.artifacts.get("scaler")

    user_vec = _user_vector(useritem, user_id)
    if user_vec is None:
        raise HTTPException(status_code=404, detail=f"user_id '{user_id}' not found in user_item_matrix")

    # Align to similarity matrices
    content = _stringify_index(content)
    cf = _stringify_index(cf)
    user_vec.index = user_vec.index.astype(str)

    # Reindex user vector on matrix items
    u_on_content = user_vec.reindex(content.index, fill_value=0.0)
    u_on_cf = user_vec.reindex(cf.index, fill_value=0.0)

    # Scores
    c_scores = (u_on_content.to_numpy().reshape(1, -1) @ content.to_numpy()).ravel()
    f_scores = (u_on_cf.to_numpy().reshape(1, -1) @ cf.to_numpy()).ravel()

    c_series = pd.Series(c_scores, index=content.columns, name="content")
    f_series = pd.Series(f_scores, index=cf.columns, name="cf")

    all_items = sorted(set(c_series.index) | set(f_series.index))
    c_vec = pd.Series(0.0, index=all_items)
    f_vec = pd.Series(0.0, index=all_items)
    c_vec.loc[c_series.index] = c_series.values
    f_vec.loc[f_series.index] = f_series.values

    h_raw = alpha * c_vec.values + (1.0 - alpha) * f_vec.values
    h = pd.Series(h_raw, index=all_items, name="hybrid_raw")

    # Hide items already interacted with
    already = user_vec[user_vec > 0].index.astype(str)
    h.loc[already] = -np.inf

    # Optional scaling
    h_vals = _apply_scaler(h.replace([-np.inf, np.inf], np.nan).fillna(-1e12).values, scaler)
    h_scaled = pd.Series(h_vals, index=h.index)

    top = h_scaled.sort_values(ascending=False).head(top_k)
    results = [{"item_id": k, "score": float(v)} for k, v in top.items() if np.isfinite(v)]

    # Optional Firestore log
    try:
        if app.state.firebase_ok and app.state.firestore:
            app.state.firestore.collection("reco_logs").add({
                "ts": int(time.time()),
                "user_id": user_id,
                "top_k": top_k,
                "alpha": float(alpha),
                "count": len(results),
            })
    except Exception as e:
        log.debug(f"Firestore log skipped: {e}")

    return {"user_id": user_id, "alpha": float(alpha), "top_k": top_k, "count": len(results), "results": results}

