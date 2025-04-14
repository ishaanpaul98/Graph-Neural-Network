from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import json
import os
import numpy as np
from typing import List, Dict
from NN.mpgnn import MPGNN
from Dataset.download_movielens import MovieLensDownloader

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize the model and load mappings
MODEL_PATH = os.path.join('models', 'mpgnn_model.pth')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the MovieLens data to get mappings and features
downloader = MovieLensDownloader('100k')
dataset_path = downloader.download()
data = downloader.get_dataset_info()
ratings_df = data['ratings']
movies_df = data['movies']

# Create user and movie mappings
user_mapping = {user_id: idx for idx, user_id in enumerate(ratings_df['userId'].unique())}
movie_mapping = {movie_id: idx for idx, movie_id in enumerate(ratings_df['movieId'].unique())}
movie_mapping_reverse = {idx: movie_id for movie_id, idx in movie_mapping.items()}

# Create movie title to ID mapping
movie_title_to_id = {row['title']: row['movieId'] for _, row in movies_df.iterrows()}

# Initialize model
num_users = len(user_mapping)
num_movies = len(movie_mapping)
num_user_features = num_users  # One-hot encoding
num_movie_features = 19  # Number of genre features
hidden_channels = 64
num_classes = 1

model = MPGNN(
    num_user_features=num_user_features,
    num_movie_features=num_movie_features,
    hidden_channels=hidden_channels,
    num_classes=num_classes
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

def prepare_model_input(movie_ids: List[int]) -> tuple:
    """
    Prepare input tensors for the model
    """
    # Create a dummy user (we'll use index 0)
    user_idx = 0
    
    # Create user features (one-hot encoding)
    user_features = torch.zeros(num_users)
    user_features[user_idx] = 1
    
    # Get movie features (genres)
    movie_features = torch.zeros((num_movies, num_movie_features))  # Create tensor for all movies
    for movie_id in movie_mapping:
        movie_row = movies_df[movies_df['movieId'] == movie_id].iloc[0]
        # Convert genre features to float explicitly
        genre_features = movie_row.iloc[-19:].values.astype(np.float32)
        movie_features[movie_mapping[movie_id]] = torch.tensor(genre_features, dtype=torch.float)
    
    # Create edge indices for input movies
    edge_index = torch.tensor([
        [user_idx] * len(movie_ids),  # Source nodes (user)
        [movie_mapping[movie_id] for movie_id in movie_ids]  # Target nodes (movies)
    ], dtype=torch.long)
    
    return user_features, movie_features, edge_index

@app.route('/api/recommend', methods=['POST'])
def get_recommendations():
    try:
        data = request.get_json()
        
        # Validate input
        if not data or 'movies' not in data:
            return jsonify({'error': 'Please provide a list of movies'}), 400
            
        user_movies = data['movies']
        
        if not isinstance(user_movies, list) or len(user_movies) != 5:
            return jsonify({'error': 'Please provide exactly 5 movies'}), 400
            
        # Convert movie titles to IDs
        movie_ids = []
        for movie_title in user_movies:
            if movie_title not in movie_title_to_id:
                return jsonify({'error': f'Movie not found: {movie_title}'}), 400
            movie_ids.append(movie_title_to_id[movie_title])
        
        # Prepare model input
        user_features, movie_features, edge_index = prepare_model_input(movie_ids)
        
        # Get predictions for all movies
        with torch.no_grad():
            # Create edge indices for all possible movie recommendations
            all_movie_indices = torch.arange(num_movies)
            all_edge_index = torch.tensor([
                [0] * num_movies,  # Source nodes (user)
                all_movie_indices  # Target nodes (all movies)
            ], dtype=torch.long)
            
            # Get predictions
            predictions = model.predict(
                user_features.unsqueeze(0),  # Add batch dimension
                movie_features,
                all_edge_index
            )
            
            # Get top 15 recommendations
            top_15_indices = predictions.squeeze().topk(15).indices.tolist()
            recommended_movie_ids = [movie_mapping_reverse[idx] for idx in top_15_indices]
            
            # Get movie titles
            recommended_movies = []
            for movie_id in recommended_movie_ids:
                movie_title = movies_df[movies_df['movieId'] == movie_id]['title'].iloc[0]
                recommended_movies.append(movie_title)
        
        return jsonify({
            'recommendations': recommended_movies
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
