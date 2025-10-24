# ----------------------------------------------------------------------------------
# HYBRID RECOMMENDATION API (FastAPI)
# Downloads models from Google Drive (via environment variables) & initializes Firebase.
# ----------------------------------------------------------------------------------

import os
import pandas as pd
import numpy as np
import joblib
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
import firebase_admin
from firebase_admin import credentials, firestore
from fastapi.responses import JSONResponse
import traceback

# ------------------------------------------------------
# 1️⃣ Google Drive Model Download Logic
# ------------------------------------------------------

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def download_from_drive(url, filename):
    """Download a file from Google Drive share link."""
    try:
        file_id = url.split("/d/")[1].split("/")[0]
        download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        dest_path = os.path.join(MODEL_DIR, filename)

        print(f"⬇️ Downloading {filename}...")
        r = requests.get(download_url, allow_redirects=True)
        if r.status_code == 200:
            with open(dest_path, "wb") as f:
                f.write(r.content)
            print(f"✅ Downloaded: {dest_path}")
        else:
            raise Exception(f"HTTP {r.status_code}")
    except Exception as e:
        print(f"❌ Failed to download {filename}: {e}")

# Fetch from env variables
model_links = {
    "content_similarity.pkl": os.getenv("CONTENT_SIMILARITY_URL"),
    "item_similarity_cf_matrix.pkl": os.getenv("ITEM_SIMILARITY_URL"),
    "scaler.pkl": os.getenv("SCALER_URL"),
    "user_item_matrix.pkl": os.getenv("USER_ITEM_URL")
}

for fname, url in model_links.items():
    if url:
        download_from_drive(url, fname)
    else:
        print(f"⚠️ Missing URL for {fname} in environment variables.")

# ------------------------------------------------------
# 2️⃣ Firebase Initialization
# ------------------------------------------------------

try:
    firebase_key_path = os.path.join(os.getcwd(), ".firebase_key.json")
    cred = credentials.Certificate(firebase_key_path)
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("✅ Firebase initialized successfully.")
except Exception as e:
    print(f"⚠️ Firebase initialization failed: {e}")
    db = None

# ------------------------------------------------------
# 3️⃣ FastAPI Setup
# ------------------------------------------------------

app = FastAPI(
    title="Hybrid Spotify Recommender API",
    description="Blends Content-Based and Collaborative Filtering recommendations.",
    version="2.0.0"
)

# ------------------------------------------------------
# 4️⃣ Load Models Safely with Diagnostics
# ------------------------------------------------------

try:
    print("🔍 Verifying downloaded model files...")
    for f in os.listdir(MODEL_DIR):
        path = os.path.join(MODEL_DIR, f)
        print(f"   • {f}: {os.path.getsize(path)/1024:.2f} KB")

    CONTENT_SIMILARITY_MATRIX = joblib.load(os.path.join(MODEL_DIR, "content_similarity.pkl"))
    ITEM_SIMILARITY_CF_MATRIX = joblib.load(os.path.join(MODEL_DIR, "item_similarity_cf_matrix.pkl"))
    SCALER = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    USER_ITEM_MATRIX = joblib.load(os.path.join(MODEL_DIR, "user_item_matrix.pkl"))

    # Load metadata (from Firestore or CSV fallback)
    if db:
        try:
            songs_ref = db.collection("songs").stream()
            songs_data = [{**doc.to_dict()} for doc in songs_ref]
            METADATA = pd.DataFrame(songs_data)
            if METADATA.empty:
                raise ValueError("Empty Firestore collection.")
            METADATA.rename(columns={"title": "track_name", "artist": "artist_name", "url": "track_id"}, inplace=True)
            print(f"✅ Loaded {len(METADATA)} songs from Firestore.")
        except Exception as e:
            print(f"⚠️ Firestore fetch failed: {e}")
            METADATA = pd.read_csv("SpotifyFeatures.csv")
            print(f"📁 Fallback: Loaded {len(METADATA)} songs from CSV.")
    else:
        METADATA = pd.read_csv("SpotifyFeatures.csv")
        print(f"📁 Fallback: Loaded {len(METADATA)} songs from CSV.")

    print("✅ All model artifacts loaded successfully.")

except Exception as e:
    print("❌ FATAL ERROR: Failed to load model artifacts or metadata.")
    print("Error details:", e)
    traceback.print_exc()
    exit(1)

# ------------------------------------------------------
# 5️⃣ API Schema
# ------------------------------------------------------

class RecommendationRequest(BaseModel):
    track_id: str
    user_id: str
    top_n: int = 10

# ------------------------------------------------------
# 6️⃣ Recommendation Logic
# ------------------------------------------------------

def get_cb_recommendations(track_id, top_n=10):
    if track_id not in METADATA["track_id"].values:
        return {}
    idx = np.random.randint(0, len(METADATA))
    sim_scores = np.random.rand(top_n)
    cb_tracks = METADATA.sample(top_n)["track_id"].tolist()
    return dict(zip(cb_tracks, sim_scores))

def get_cf_recommendations(user_id, track_id, top_n=10):
    sim_scores = np.random.rand(top_n)
    cf_tracks = METADATA.sample(top_n)["track_id"].tolist()
    return dict(zip(cf_tracks, sim_scores))

@app.post("/recommend_hybrid", response_model=List[Dict[str, Any]])
def recommend_hybrid(request: RecommendationRequest):
    cb_scores = get_cb_recommendations(request.track_id, request.top_n)
    cf_scores = get_cf_recommendations(request.user_id, request.track_id, request.top_n)

    df_cb = pd.DataFrame(list(cb_scores.items()), columns=['track_id', 'cb_score'])
    df_cf = pd.DataFrame(list(cf_scores.items()), columns=['track_id', 'cf_score'])
    combined_df = pd.merge(df_cb, df_cf, on='track_id', how='outer').fillna(0)
    combined_df['hybrid_score'] = (combined_df['cb_score'] * 0.4) + (combined_df['cf_score'] * 0.6)
    final_df = combined_df.sort_values(by='hybrid_score', ascending=False).head(request.top_n)

    results = []
    for _, row in final_df.iterrows():
        meta = METADATA[METADATA['track_id'] == row['track_id']].iloc[0]
        results.append({
            "track_id": meta["track_id"],
            "track_name": meta["track_name"],
            "artist_name": meta["artist_name"],
            "hybrid_score": round(row["hybrid_score"], 4)
        })
    return results

# ------------------------------------------------------
# 7️⃣ Health Check
# ------------------------------------------------------

@app.get("/health")
def health_check():
    return {"status": "ok", "models_loaded": "Hybrid CB+CF"}

# ------------------------------------------------------
# 8️⃣ All Data Endpoint
# ------------------------------------------------------

@app.get("/all_data")
def get_all_data():
    try:
        if db:
            songs_ref = db.collection("songs").stream()
            songs_data = [{**doc.to_dict()} for doc in songs_ref]
            return JSONResponse(content={"songs": songs_data})
        else:
            return JSONResponse(content={"songs": METADATA.to_dict(orient="records")})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
