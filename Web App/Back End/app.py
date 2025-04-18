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
# Configure CORS to allow requests from the frontend
CORS(app, resources={
    r"/*": {
        "origins": [
            "http://localhost:5173",
            "http://localhost:5174",
            "http://localhost:3000"
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Accept"],
        "supports_credentials": True
    }
})

# Add request logging middleware
@app.before_request
def log_request_info():
    print('\nReceived request:')
    print('Method:', request.method)
    print('URL:', request.url)
    print('Headers:', dict(request.headers))
    print('Body:', request.get_data())

@app.after_request
def after_request(response):
    print('\nSending response:')
    print('Status:', response.status)
    print('Headers:', dict(response.headers))
    #print('Body:', response.get_data())
    return response

# Initialize the model and load mappings
MODEL_PATH = os.path.join('models', 'mpgnn_model.pth')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model and mappings
if os.path.exists(MODEL_PATH):
    try:
        # Load the saved data with weights_only=False to allow numpy types
        saved_data = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        
        # Extract model and mappings
        model_state_dict = saved_data['model_state_dict']
        user_mapping = saved_data['user_mapping']
        movie_mapping = saved_data['movie_mapping']
        movie_mapping_reverse = saved_data['movie_mapping_reverse']  # Load reverse mapping
        dataset_size = saved_data['dataset_size']
        num_users = saved_data['num_users']
        num_movies = saved_data['num_movies']
        
        print(f"\nLoaded model trained on {dataset_size} dataset")
        print(f"Number of users: {num_users}")
        print(f"Number of movies: {num_movies}")
        
        # Initialize model with correct dimensions
        model = MPGNN(
            num_user_features=num_users,
            num_movie_features=num_movies,
            hidden_channels=8,
            num_classes=1
        )
        
        # Load model state
        model.load_state_dict(model_state_dict)
        print("Model loaded successfully from:", MODEL_PATH)
        
        # At startup, after loading model and mappings:
        movie_id_to_title = saved_data['movie_id_to_title']
        movie_id_to_popularity = saved_data['movie_id_to_popularity']
        movie_title_to_id = {title: mid for mid, title in movie_id_to_title.items()}
        
    except Exception as e:
        print(f"\nError loading model: {str(e)}")
        raise RuntimeError("Failed to load model and mappings. Please train a model first.")
else:
    raise RuntimeError("No trained model found. Please train a model first.")

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
    
    # Create movie features (one-hot encoding)
    movie_features = torch.zeros((num_movies, num_movies))
    for movie_id in movie_mapping:
        movie_idx = movie_mapping[movie_id]
        movie_features[movie_idx, movie_idx] = 1
    
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
            
        print("\nInput movies:", user_movies)
            
        # Convert movie titles to IDs
        movie_ids = []
        for movie_title in user_movies:
            if movie_title not in movie_title_to_id:
                return jsonify({'error': f'Movie not found: {movie_title}'}), 400
            movie_ids.append(movie_title_to_id[movie_title])
        
        print("Converted to movie IDs:", movie_ids)
        
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
            print("\nTop 15 indices from model:", top_15_indices)
            
            recommended_movie_ids = [movie_mapping_reverse[idx] for idx in top_15_indices]
            print("Converted to movie IDs:", recommended_movie_ids)
            
            # Get movie titles with error handling
            recommended_movies = []
            for idx, movie_id in enumerate(recommended_movie_ids):
                movie_title = movie_id_to_title.get(movie_id)
                if movie_title:
                    print(f"Recommendation {idx + 1}: ID {movie_id} -> {movie_title}")
                    recommended_movies.append(movie_title)
                else:
                    print(f"Warning: Movie ID {movie_id} not found in movie_id_to_title")
            if len(recommended_movies) < 15:
                print(f"Warning: Only found {len(recommended_movies)} valid recommendations")
        
        print("\nFinal recommendations:", recommended_movies)
        return jsonify({
            'recommendations': recommended_movies
        })
        
    except Exception as e:
        print(f"Error in get_recommendations: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

@app.route('/api/movies', methods=['GET'])
def get_available_movies():
    try:
        # Only return the top 1000 by popularity
        sorted_movies = sorted(movie_id_to_popularity.items(), key=lambda x: x[1], reverse=True)
        top_1000 = sorted_movies[:1000]
        movies = [{
            'title': movie_id_to_title[movie_id],
            'id': movie_id,
            'popularity': popularity
        } for movie_id, popularity in top_1000]
        return jsonify({'movies': movies})
    except Exception as e:
        print(f"Error in get_available_movies: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
