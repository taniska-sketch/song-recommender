# ----------------------------------------------------------------------------------
# HYBRID RECOMMENDATION API (FastAPI)
# Blends Content-Based (CB) and Collaborative Filtering (CF) scores for a final result.
# Loads model artifacts automatically from Google Drive.
# ----------------------------------------------------------------------------------

import os
import requests
import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
from fastapi.responses import JSONResponse

# ------------------------------------------------------
# 1️⃣ Download helper for Google Drive model files
# ------------------------------------------------------
def download_from_gdrive(url, dest_path):
    """Downloads a file from a direct Google Drive URL if it doesn't exist locally."""
    if not os.path.exists(dest_path):
        print(f"⬇️ Downloading {os.path.basename(dest_path)}...")
        response = requests.get(url, allow_redirects=True)
        if response.status_code == 200:
            with open(dest_path, 'wb') as f:
                f.write(response.content)
            print(f"✅ Downloaded: {dest_path}")
        else:
            raise Exception(f"Failed to download {url}: {response.status_code}")

# ------------------------------------------------------
# 2️⃣ Ensure all model artifacts are available
# ------------------------------------------------------
def ensure_artifacts():
    os.makedirs("models", exist_ok=True)
    paths = {
        "content_similarity.pkl": os.path.join("models", "content_similarity.pkl"),
        "item_similarity_cf_matrix.pkl": os.path.join("models", "item_similarity_cf_matrix.pkl"),
        "scaler.pkl": os.path.join("models", "scaler.pkl"),
        "user_item_matrix.pkl": os.path.join("models", "user_item_matrix.pkl")
    }

    urls = {
        "content_similarity.pkl": os.getenv("GDRIVE_CONTENT_URL"),
        "item_similarity_cf_matrix.pkl": os.getenv("GDRIVE_CF_URL"),
        "scaler.pkl": os.getenv("GDRIVE_SCALER_URL"),
        "user_item_matrix.pkl": os.getenv("GDRIVE_USERITEM_URL")
    }

    for key, path in paths.items():
        if urls[key]:
            download_from_gdrive(urls[key], path)
        else:
            print(f"⚠️ Missing environment variable for {key}")

    return paths

# ------------------------------------------------------
# 3️⃣ Initialize FastAPI app
# ------------------------------------------------------
app = FastAPI(
    title="Hybrid Spotify Recommender API",
    description="Blends Content-Based and Collaborative Filtering models to suggest songs.",
    version="2.0.0"
)

# ------------------------------------------------------
# 4️⃣ Load model & metadata
# ------------------------------------------------------
try:
    paths = ensure_artifacts()

    CONTENT_SIMILARITY_MATRIX = joblib.load(paths["content_similarity.pkl"])
    CF_SIMILARITY_MATRIX = joblib.load(paths["item_similarity_cf_matrix.pkl"])
    SCALER = joblib.load(paths["scaler.pkl"])
    USER_ITEM_MATRIX = joblib.load(paths["user_item_matrix.pkl"])

    # Load metadata CSV
    METADATA = pd.read_csv('SpotifyFeatures.csv')
    METADATA.columns = [c.lower() for c in METADATA.columns]
    METADATA_COLS_REQUIRED = ['track_id', 'track_name', 'artist_name']

    for c in METADATA_COLS_REQUIRED:
        if c not in METADATA.columns:
            raise KeyError(f"Missing column: {c}")

    USER_ITEM_MATRIX.columns = USER_ITEM_MATRIX.columns.astype(str)
    METADATA['track_id'] = METADATA['track_id'].astype(str)

    CB_INDICES = pd.Series(METADATA.index, index=METADATA['track_id'])
    TRACK_TO_CF_INDEX = {track: i for i, track in enumerate(USER_ITEM_MATRIX.columns)}

    print("✅ All model artifacts loaded successfully.")

except Exception as e:
    print("❌ FATAL ERROR: Failed to load model artifacts or metadata.")
    print(f"Error details: {e}")
    exit(1)

# ------------------------------------------------------
# 5️⃣ Input schema
# ------------------------------------------------------
class RecommendationRequest(BaseModel):
    track_id: str
    user_id: str
    top_n: int = 10

# ------------------------------------------------------
# 6️⃣ Recommendation functions
# ------------------------------------------------------
def get_cb_recommendations(track_id, top_n=10):
    if track_id not in CB_INDICES.index:
        return {}
    idx = CB_INDICES[track_id]
    sim_scores = list(enumerate(CONTENT_SIMILARITY_MATRIX[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]
    track_indices = [i[0] for i in sim_scores]
    track_scores = [i[1] for i in sim_scores]
    cb_tracks = METADATA.iloc[track_indices]['track_id'].tolist()
    return dict(zip(cb_tracks, track_scores))


def get_cf_recommendations(user_id, track_id, top_n=10):
    if user_id not in USER_ITEM_MATRIX.index or track_id not in TRACK_TO_CF_INDEX:
        return {}
    track_cf_index = TRACK_TO_CF_INDEX[track_id]
    sim_scores = CF_SIMILARITY_MATRIX[track_cf_index]
    sim_scores_paired = sorted(list(enumerate(sim_scores)), key=lambda x: x[1], reverse=True)[1:]
    top_indices = [i[0] for i in sim_scores_paired[:top_n * 2]]
    top_scores = [i[1] for i in sim_scores_paired[:top_n * 2]]
    cf_tracks = [USER_ITEM_MATRIX.columns[idx] for idx in top_indices]
    return dict(zip(cf_tracks, top_scores))

# ------------------------------------------------------
# 7️⃣ Hybrid recommendation endpoint
# ------------------------------------------------------
@app.post("/recommend_hybrid", response_model=List[Dict[str, Any]])
def recommend_hybrid(request: RecommendationRequest):
    cb_scores = get_cb_recommendations(request.track_id, request.top_n * 2)
    cf_scores = get_cf_recommendations(request.user_id, request.track_id, request.top_n * 2)

    df_cb = pd.DataFrame(list(cb_scores.items()), columns=['track_id', 'cb_score'])
    df_cf = pd.DataFrame(list(cf_scores.items()), columns=['track_id', 'cf_score'])

    combined_df = pd.merge(df_cb, df_cf, on='track_id', how='outer').fillna(0)
    combined_df['hybrid_score'] = (combined_df['cb_score'] * 0.4) + (combined_df['cf_score'] * 0.6)
    final_df = combined_df[combined_df['track_id'] != request.track_id].sort_values(
        by='hybrid_score', ascending=False
    ).head(request.top_n)

    result_df = pd.merge(final_df, METADATA, on='track_id', how='left')
    results = result_df.apply(lambda row: {
        "track_id": row['track_id'],
        "track_name": row['track_name'],
        "artist_name": row['artist_name'],
        "hybrid_score": round(row['hybrid_score'], 4)
    }, axis=1).tolist()

    if not results:
        fallback = METADATA.sample(n=request.top_n, random_state=42)
        results = fallback.apply(lambda row: {
            "track_id": row['track_id'],
            "track_name": row['track_name'],
            "artist_name": row['artist_name'],
            "hybrid_score": 0.0
        }, axis=1).tolist()

    return results

# ------------------------------------------------------
# 8️⃣ Health check endpoint
# ------------------------------------------------------
@app.get("/health")
def health_check():
    return {"status": "ok", "models_loaded": "Hybrid CB+CF"}
