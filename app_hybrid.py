# ----------------------------------------------------------------------------------
# HYBRID RECOMMENDATION API (FastAPI)
# Blends Content-Based (CB) and Collaborative Filtering (CF) scores for a final result.
# ----------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
import firebase_admin
from firebase_admin import credentials, firestore
from fastapi.responses import JSONResponse

# ------------------------------------------------------
# 1️⃣ Initialize Firebase connection
# ------------------------------------------------------
cred = credentials.Certificate("C:/Users/tanis/OneDrive/Documents/songrecommender/firebase_key.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# ------------------------------------------------------
# 2️⃣ FastAPI app configuration
# ------------------------------------------------------
app = FastAPI(
    title="Hybrid Spotify Recommender API",
    description="Blends Content-Based and Collaborative Filtering models to suggest songs.",
    version="1.0.0"
)

# ------------------------------------------------------
# 3️⃣ Required metadata columns
# ------------------------------------------------------
METADATA_COLS_REQUIRED = ['track_id', 'track_name', 'artist_name']

# ------------------------------------------------------
# 4️⃣ Load all model and data artifacts
# ------------------------------------------------------
try:
    # --- Content-Based (CB) Artifacts ---
    CONTENT_SIMILARITY_MATRIX = joblib.load('content_similarity.pkl')
    SCALED_FEATURES = pd.read_csv('scaled_feature_sample.csv')
    SCALER = joblib.load('scaler.pkl')

    # --- Collaborative Filtering (CF) Artifacts ---
    CF_SIMILARITY_MATRIX = joblib.load('item_similarity_cf_matrix.pkl')
    USER_ITEM_MATRIX = joblib.load('user_item_matrix.pkl')

    # --- Firestore Data Load ---
    try:
        songs_ref = db.collection("songs").stream()
        songs_data = [{**doc.to_dict()} for doc in songs_ref]
        METADATA = pd.DataFrame(songs_data)

        if not METADATA.empty:
            # ✅ Map Firestore field names to match recommender schema
            METADATA = METADATA.rename(columns={
                "title": "track_name",
                "artist": "artist_name",
                "url": "track_id",
                "category": "genre"
            })
            print(f"✅ Loaded {len(METADATA)} songs from Firestore (fields mapped).")
        else:
            raise ValueError("No data found in Firestore collection 'songs'.")

    except Exception as e:
        print(f"⚠️ Firebase load failed, falling back to local CSV.")
        print(f"Error details: {e}")
        METADATA = pd.read_csv('SpotifyFeatures.csv')

    # --- CRITICAL VALIDATION AND TYPE CASTING ---
    METADATA.columns = [col.lower() for col in METADATA.columns]

    if not all(col in METADATA.columns for col in METADATA_COLS_REQUIRED):
        missing = [col for col in METADATA_COLS_REQUIRED if col not in METADATA.columns]
        raise KeyError(f"Metadata missing required columns: {missing}. Found: {list(METADATA.columns)}")

    USER_ITEM_MATRIX.columns = USER_ITEM_MATRIX.columns.astype(str)
    SCALED_FEATURES['track_id'] = SCALED_FEATURES['track_id'].astype(str)
    METADATA['track_id'] = METADATA['track_id'].astype(str)

    TRACK_TO_CF_INDEX = {track: i for i, track in enumerate(USER_ITEM_MATRIX.columns)}
    CB_INDICES = pd.Series(SCALED_FEATURES.index, index=SCALED_FEATURES['track_id'])
    GUARANTEED_CB_TRACK_ID = SCALED_FEATURES['track_id'].iloc[0]

    print("--- HYBRID API LOADED SUCCESSFULLY ---")
    print(f"CB Matrix Shape: {CONTENT_SIMILARITY_MATRIX.shape}")
    print(f"CF Matrix Shape: {CF_SIMILARITY_MATRIX.shape}")
    print(f"*** TEST TRACK ID (Use this!): {GUARANTEED_CB_TRACK_ID} ***")

except Exception as e:
    print("FATAL ERROR: Failed to load one or more artifacts or validate data.")
    print(f"Error details: {e}")
    exit(1)

# ------------------------------------------------------
# 5️⃣ Input schema definition
# ------------------------------------------------------
class RecommendationRequest(BaseModel):
    track_id: str
    user_id: str
    top_n: int = 10

# ------------------------------------------------------
# 6️⃣ Content-Based Recommendation
# ------------------------------------------------------
def get_cb_recommendations(track_id, top_n=10):
    if track_id not in CB_INDICES.index:
        return {}
    idx = CB_INDICES[track_id]
    sim_scores = list(enumerate(CONTENT_SIMILARITY_MATRIX[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    track_indices = [i[0] for i in sim_scores]
    track_scores = [i[1] for i in sim_scores]
    cb_tracks = SCALED_FEATURES.iloc[track_indices]['track_id'].tolist()
    return dict(zip(cb_tracks, track_scores))

# ------------------------------------------------------
# 7️⃣ Collaborative Filtering Recommendation
# ------------------------------------------------------
def get_cf_recommendations(user_id, track_id, top_n=10):
    if user_id not in USER_ITEM_MATRIX.index or track_id not in TRACK_TO_CF_INDEX:
        return {}
    track_cf_index = TRACK_TO_CF_INDEX[track_id]
    sim_scores = CF_SIMILARITY_MATRIX[track_cf_index]
    sim_scores_paired = sorted(list(enumerate(sim_scores)), key=lambda x: x[1], reverse=True)[1:]
    top_indices = [i[0] for i in sim_scores_paired[:top_n*2]]
    top_scores = [i[1] for i in sim_scores_paired[:top_n*2]]
    cf_tracks = [USER_ITEM_MATRIX.columns[idx] for idx in top_indices]
    return dict(zip(cf_tracks, top_scores))

# ------------------------------------------------------
# 8️⃣ Hybrid Recommendation Endpoint
# ------------------------------------------------------
@app.post("/recommend_hybrid", response_model=List[Dict[str, Any]])
def recommend_hybrid(request: RecommendationRequest):
    cb_scores = get_cb_recommendations(request.track_id, request.top_n*2)
    cf_scores = get_cf_recommendations(request.user_id, request.track_id, request.top_n*2)
    df_cb = pd.DataFrame(list(cb_scores.items()), columns=['track_id', 'cb_score'])
    df_cf = pd.DataFrame(list(cf_scores.items()), columns=['track_id', 'cf_score'])
    df_cb['track_id'] = df_cb['track_id'].astype(str)
    df_cf['track_id'] = df_cf['track_id'].astype(str)
    combined_df = pd.merge(df_cb, df_cf, on='track_id', how='outer').fillna(0)
    combined_df['hybrid_score'] = (combined_df['cb_score'] * 0.4) + (combined_df['cf_score'] * 0.6)
    final_recommendations = combined_df[combined_df['track_id'] != request.track_id]
    final_recommendations = final_recommendations.sort_values(by='hybrid_score', ascending=False)
    final_recommendations = final_recommendations.drop_duplicates(subset=['track_id']).head(request.top_n)
    result_df = pd.merge(final_recommendations, METADATA[METADATA_COLS_REQUIRED], on='track_id', how='left')

    results = result_df.apply(lambda row: {
        "track_id": row['track_id'],
        "track_name": row['track_name'],
        "artist_name": row['artist_name'],
        "popularity": 0,
        "hybrid_score": round(row['hybrid_score'], 4)
    }, axis=1).tolist()

    if not results:
        fallback_sample = METADATA.sample(n=request.top_n, random_state=42)
        results = fallback_sample.apply(lambda row: {
            "track_id": row['track_id'],
            "track_name": row['track_name'],
            "artist_name": row['artist_name'],
            "popularity": 0,
            "hybrid_score": 0.0
        }, axis=1).tolist()
    return results

# ------------------------------------------------------
# 9️⃣ Health check endpoint
# ------------------------------------------------------
@app.get("/health")
def health_check():
    return {"status": "ok", "models_loaded": "Hybrid CB+CF"}

# ------------------------------------------------------
# 🔟 Unified data endpoint (optional)
# ------------------------------------------------------
@app.get("/all_data")
def get_all_data():
    try:
        songs_ref = db.collection("songs").stream()
        songs_data = [{**doc.to_dict()} for doc in songs_ref]
        return JSONResponse(content={"songs": songs_data})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
