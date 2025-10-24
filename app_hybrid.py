# ----------------------------------------------------------------------------------
# HYBRID RECOMMENDATION API (FastAPI)
# Loads model artifacts from Google Drive (Render-friendly)
# ----------------------------------------------------------------------------------

import os, io, requests, shutil, joblib, pandas as pd, numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
import firebase_admin
from firebase_admin import credentials, firestore
from fastapi.responses import JSONResponse

# ------------------------------------------------------
# 1️⃣ Firebase initialization
# ------------------------------------------------------
try:
    firebase_key_path = os.path.join(os.getcwd(), "firebase_key.json")
    cred = credentials.Certificate(firebase_key_path)
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("✅ Firebase initialized successfully.")
except Exception as e:
    print(f"⚠️ Firebase initialization failed: {e}")
    db = None

# ------------------------------------------------------
# 2️⃣ FastAPI app config
# ------------------------------------------------------
app = FastAPI(title="Hybrid Spotify Recommender API", version="1.0.2")

# ------------------------------------------------------
# 3️⃣ Google Drive helper functions
# ------------------------------------------------------
def download_from_gdrive(url: str, dest: str):
    print(f"📥 Downloading {os.path.basename(dest)} from Drive...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            shutil.copyfileobj(r.raw, f)
    print(f"✅ Saved {dest}")
    return dest

def ensure_artifacts():
    os.makedirs("artifacts", exist_ok=True)
    files = {
        "content_similarity.pkl": os.getenv("GDRIVE_CONTENT_URL", "https://drive.google.com/uc?export=download&id=1b2EYpS7eGY1M9bBv3iCS7JVubwKpoO6F"),
        "item_similarity_cf_matrix.pkl": os.getenv("GDRIVE_CF_URL", "https://drive.google.com/uc?export=download&id=1RnTh7VuPdGQl2yyct1rZp_ttzLSInY8k"),
        "scaler.pkl": os.getenv("GDRIVE_SCALER_URL", "https://drive.google.com/uc?export=download&id=1Hyxort42wZddkLJa_pyJtCygM4XvC5k4"),
        "user_item_matrix.pkl": os.getenv("GDRIVE_USERITEM_URL", "https://drive.google.com/uc?export=download&id=1rmHTgcvsnDfH9SE1kmTpPlpAyv6naloc"),
    }
    paths = {}
    for name, url in files.items():
        local_path = f"artifacts/{name}"
        if not os.path.exists(local_path):
            download_from_gdrive(url, local_path)
        paths[name] = local_path
    return paths

# ------------------------------------------------------
# 4️⃣ Load models and data
# ------------------------------------------------------
try:
    paths = ensure_artifacts()
    CONTENT_SIMILARITY_MATRIX = joblib.load(paths["content_similarity.pkl"])
    CF_SIMILARITY_MATRIX = joblib.load(paths["item_similarity_cf_matrix.pkl"])
    SCALER = joblib.load(paths["scaler.pkl"])
    USER_ITEM_MATRIX = joblib.load(paths["user_item_matrix.pkl"])
    SCALED_FEATURES = pd.read_csv("scaled_feature_sample.csv")

    print("✅ All models loaded successfully from Google Drive!")
except Exception as e:
    print(f"❌ Model load failed: {e}")
    exit(1)

# ------------------------------------------------------
# 5️⃣ Metadata load (Firebase or local)
# ------------------------------------------------------
try:
    if db:
        docs = db.collection("songs").stream()
        data = [d.to_dict() for d in docs]
        METADATA = pd.DataFrame(data)
        METADATA.rename(columns={"title": "track_name", "artist": "artist_name", "url": "track_id"}, inplace=True)
        print(f"✅ Loaded {len(METADATA)} songs from Firestore.")
    else:
        raise Exception("Firestore unavailable")
except Exception:
    METADATA = pd.read_csv("SpotifyFeatures.csv")
    print(f"📁 Loaded {len(METADATA)} songs from local CSV.")

METADATA_COLS_REQUIRED = ["track_id", "track_name", "artist_name"]
METADATA["track_id"] = METADATA["track_id"].astype(str)

TRACK_TO_CF_INDEX = {track: i for i, track in enumerate(USER_ITEM_MATRIX.columns)}
CB_INDICES = pd.Series(SCALED_FEATURES.index, index=SCALED_FEATURES["track_id"])

# ------------------------------------------------------
# 6️⃣ Input schema
# ------------------------------------------------------
class RecommendationRequest(BaseModel):
    track_id: str
    user_id: str
    top_n: int = 10

# ------------------------------------------------------
# 7️⃣ Recommender logic
# ------------------------------------------------------
def get_cb_recommendations(track_id, top_n=10):
    if track_id not in CB_INDICES.index:
        return {}
    idx = CB_INDICES[track_id]
    scores = list(enumerate(CONTENT_SIMILARITY_MATRIX[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1 : top_n + 1]
    tracks = SCALED_FEATURES.iloc[[i[0] for i in scores]]["track_id"].tolist()
    return dict(zip(tracks, [i[1] for i in scores]))

def get_cf_recommendations(user_id, track_id, top_n=10):
    if user_id not in USER_ITEM_MATRIX.index or track_id not in TRACK_TO_CF_INDEX:
        return {}
    idx = TRACK_TO_CF_INDEX[track_id]
    sims = CF_SIMILARITY_MATRIX[idx]
    pairs = sorted(list(enumerate(sims)), key=lambda x: x[1], reverse=True)[1:]
    top = pairs[: top_n * 2]
    tracks = [USER_ITEM_MATRIX.columns[i[0]] for i in top]
    return dict(zip(tracks, [i[1] for i in top]))

@app.post("/recommend_hybrid")
def recommend_hybrid(req: RecommendationRequest):
    cb = get_cb_recommendations(req.track_id, req.top_n * 2)
    cf = get_cf_recommendations(req.user_id, req.track_id, req.top_n * 2)
    df = pd.merge(
        pd.DataFrame(cb.items(), columns=["track_id", "cb_score"]),
        pd.DataFrame(cf.items(), columns=["track_id", "cf_score"]),
        on="track_id", how="outer").fillna(0)
    df["hybrid_score"] = df["cb_score"] * 0.4 + df["cf_score"] * 0.6
    df = df[df["track_id"] != req.track_id].sort_values("hybrid_score", ascending=False).head(req.top_n)
    merged = pd.merge(df, METADATA, on="track_id", how="left")
    return merged.to_dict(orient="records")

@app.get("/health")
def health(): return {"status": "ok", "models": "Hybrid CB+CF from Google Drive"}

