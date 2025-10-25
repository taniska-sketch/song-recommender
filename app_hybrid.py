import os
import re
import sys
import time
import json
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

# Try importing gdown for Google Drive downloads
_GDOWN = False
try:
    import gdown
    _GDOWN = True
except Exception:
    _GDOWN = False

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

# ---------------- Paths ----------------
MODELS_DIR = os.getenv("MODELS_DIR", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

ARTIFACT_FILES = {
    "content":  os.path.join(MODELS_DIR, "content_similarity.pkl"),
    "cf":       os.path.join(MODELS_DIR, "item_similarity_cf_matrix.pkl"),
    "useritem": os.path.join(MODELS_DIR, "user_item_matrix.pkl"),
    "scaler":   os.path.join(MODELS_DIR, "scaler.pkl"),
}

ENV_KEYS = {
    "content":  ("CONTENT_SIMILARITY_URL",),
    "cf":       ("CF_SIMILARITY_URL",),
    "useritem": ("USER_ITEM_MATRIX_URL",),
    "scaler":   ("SCALER_URL",),
}

FIREBASE_CREDENTIALS_PATH = os.getenv("FIREBASE_CREDENTIALS_PATH", "/opt/render/project/src/.firebase_key.json")
HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", "0.5"))

# -------- Helpers --------
def sha256_short(path): return hashlib.sha256(open(path, "rb").read()).hexdigest()[:12] if os.path.exists(path) else None
def fsize_kb(path): return round(os.path.getsize(path)/1024.0,2) if os.path.exists(path) else None

def env_url(key): 
    for k in key:
        v = os.getenv(k)
        if v: return v.strip()
    return None

def parse_drive_id(url):
    match = re.search(r"/d/([^/]+)/|id=([^&]+)", url)
    return match.group(1) or match.group(2) if match else None

def drive_direct_url(fid): 
    return f"https://drive.google.com/uc?export=download&id={fid}"

def looks_html(file):
    try:
        with open(file,"rb") as f:
            head=f.read(500).lower()
        return b"<html" in head
    except: return False

def download_artifact(url, path, label):
    if os.path.exists(path) and os.path.getsize(path)>0:
        return

    if not url: raise FileNotFoundError(f"Missing URL for {label}")

    last_err=None
    for attempt in range(3):
        try:
            fid=parse_drive_id(url)
            if _GDOWN:
                gdown.download(url if not fid else drive_direct_url(fid),path,fuzzy=True)
            else:
                with requests.get(url if not fid else drive_direct_url(fid),stream=True) as r:
                    r.raise_for_status()
                    with open(path,"wb") as f:
                        for chunk in r.iter_content(1<<20): f.write(chunk)

            if looks_html(path):
                raise RuntimeError(f"HTML returned for {label}: change Drive sharing to 'Anyone with link'")

            return
        except Exception as e:
            last_err=e
            time.sleep(2)
    raise RuntimeError(f"Download failed for {label}: {last_err}")

def load_pkl(path): return joblib.load(path)

def to_df(x,name):
    if isinstance(x,pd.DataFrame): return x
    try: return pd.DataFrame(x)
    except: raise TypeError(f"{name} wrong type")

# -------- Global state --------
app.state.artifacts={}
app.state.firebase_ok=False
app.state.firestore=None

# -------- Startup --------
@app.on_event("startup")
def startup():
    # Download
    for k,p in ARTIFACT_FILES.items():
        download_artifact(env_url(ENV_KEYS[k]),p,k)

    # Load
    loaded={}
    for k,p in ARTIFACT_FILES.items():
        loaded[k]=load_pkl(p)
    app.state.artifacts=loaded

    # Firebase
    try:
        if os.path.exists(FIREBASE_CREDENTIALS_PATH):
            cred=credentials.Certificate(FIREBASE_CREDENTIALS_PATH)
            fb_initialize_app(cred)
            app.state.firestore=firestore.client()
            app.state.firebase_ok=True
    except: pass

# -------- Routes --------
@app.get("/")
def root():
    return {
        "name":"Hybrid Song Recommender",
        "version":"1.0",
        "endpoints":[
            "/healthz",
            "/debug/artifacts",
            "/recommend_hybrid?user_id=U1&top_k=10&alpha=0.5",
            "/debug/users"
        ],
    }

@app.get("/healthz")
def healthz():
    a=app.state.artifacts
    return {
        "ok": True,
        "artifacts_loaded": bool(a),
        "firebase_ok": app.state.firebase_ok,
        "has_content": "content" in a,
        "has_cf": "cf" in a,
        "has_useritem": "useritem" in a,
        "has_scaler": "scaler" in a,
    }

@app.get("/debug/artifacts")
def debug_artifacts():
    d={}
    for k,p in ARTIFACT_FILES.items():
        d[k]={
            "exists": os.path.exists(p),
            "size_kb": fsize_kb(p),
            "sha256_12": sha256_short(p),
            "env": env_url(ENV_KEYS[k]),
        }
    return d

# ✅ New Debug Users Route
@app.get("/debug/users")
def debug_users(n: int = 25):
    if not app.state.artifacts or "useritem" not in app.state.artifacts:
        raise HTTPException(status_code=503, detail="user_item_matrix not loaded")
    df = to_df(app.state.artifacts["useritem"], "user_item_matrix")
    return {
        "total_users": int(df.shape[0]),
        "row_index_sample": [str(x) for x in df.index.astype(str)[:n]],
        "column_sample": [str(x) for x in df.columns.astype(str)[:n]],
    }

@app.get("/recommend_hybrid")
def recommend_hybrid(user_id: str, top_k: int = 10, alpha: float = HYBRID_ALPHA):
    a=app.state.artifacts
    df_useritem = to_df(a["useritem"],"user_item_matrix")
    df_cf = to_df(a["cf"],"cf")
    df_content = to_df(a["content"],"content")

    user_vec = df_useritem.loc[user_id] if user_id in df_useritem.index else None
    if user_vec is None:
        raise HTTPException(status_code=404, detail="user_id not found")

    sc = a.get("scaler")
    u = user_vec.fillna(0)

    cf_scores = u.to_numpy().reshape(1,-1) @ df_cf.to_numpy()
    content_scores = u.to_numpy().reshape(1,-1) @ df_content.to_numpy()

    cf_s = pd.Series(cf_scores.ravel(), index=df_cf.columns)
    ct_s = pd.Series(content_scores.ravel(), index=df_content.columns)

    hybrid = alpha*ct_s + (1-alpha)*cf_s
    hybrid[u>0] = -np.inf

    top = hybrid.sort_values(ascending=False).head(top_k)
    return {
        "user_id": user_id,
        "results": [{"item_id":i,"score":float(v)} for i,v in top.items() if np.isfinite(v)]
    }
