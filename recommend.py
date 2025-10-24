import pandas as pd
import joblib

df_sample = pd.read_csv('scaled_feature_sample.csv', index_col='track_id')
similarity_matrix = joblib.load('content_similarity.pkl')
scaler = joblib.load('scaler.pkl')

# Load slim metadata (track_id, track_name, artist_name)
df_meta = pd.read_csv('songs_metadata_for_api.csv')

def recommend(track_id, n=5):
    if track_id not in df_sample.index:
        return [{'track_id': track_id,
                 'track_name': 'Not found',
                 'artist_name': 'Unknown'}]

    idx = df_sample.index.get_loc(track_id)
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    top_indices = [i[0] for i in sim_scores[1:n+1]]
    rec_ids = df_sample.index[top_indices].tolist()

    rec_info = df_meta[df_meta['track_id'].isin(rec_ids)][['track_id',
                                                            'track_name',
                                                            'artist_name']]
    return rec_info.to_dict(orient='records')

