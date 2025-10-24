from flask import Flask, request, jsonify, render_template
from recommend import recommend
import os

app = Flask(__name__)

# 🏠 Home Route — Displays the HTML page
@app.route('/')
def home():
    return render_template('index.html')  # This loads templates/index.html

# 🎧 Recommendation API Endpoint
@app.route('/recommend', methods=['GET'])
def get_recommendations():
    track_id = request.args.get('track_id')
    n = int(request.args.get('n', 10))

    if not track_id:
        return jsonify({'error': 'track_id parameter is required'}), 400

    recs = recommend(track_id, n)
    return jsonify({'recommendations': recs})

#  App Entry Point (for both local + Render hosting)
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
