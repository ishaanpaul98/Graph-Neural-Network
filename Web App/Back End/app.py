from flask import Flask, request, jsonify, redirect, session, url_for
from flask_cors import CORS
import torch
import json
import os
import numpy as np
import secrets
from typing import List, Dict
from dotenv import load_dotenv
from NN.enhanced_mpgnn import EnhancedMPGNN
from trakt_api import trakt_api
from session_manager import session_manager
import time
import threading
from enhanced_recommendations import EnhancedRecommendationEngine

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', secrets.token_hex(16))

# Configure CORS to allow requests from the frontend
CORS(app, resources={
    r"/*": {
        "origins": [
            "http://localhost:5173",  # Frontend development server
            "http://localhost:5174",
            "http://localhost:3000",
            "http://localhost:8000",
            "https://*.amplifyapp.com",  # AWS Amplify domains
            "https://*.amplifyapp.net",  # Alternative Amplify domains
            "https://main.d2p9ieiqdwymip.amplifyapp.com",  # Your main branch Amplify domain
            "https://dev.d2p9ieiqdwymip.amplifyapp.com",  # Your dev branch Amplify domain
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Accept", "X-Session-ID", "Authorization"],
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
    
    # Remove duplicate CORS headers if they exist
    if 'Access-Control-Allow-Origin' in response.headers:
        # Keep only the first one to avoid duplicates
        pass
    
    return response

# Initialize the model and load mappings
MODEL_PATH = os.path.join('models', 'trakt_gnn_model.pth')
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
        
        # Initialize model with correct dimensions for Trakt data

        model = EnhancedMPGNN(
            num_user_features=16,
            num_movie_features=16,
            hidden_channels=64,
            num_classes=1
        )
        
        # Load model state
        model.load_state_dict(model_state_dict)
        model.to(device)
        print("Model loaded successfully from:", MODEL_PATH)
        print(f"Model device: {next(model.parameters()).device}")
        
        # At startup, after loading model and mappings:
        movie_id_to_title = saved_data['movie_id_to_title']
        movie_id_to_popularity = saved_data['movie_id_to_popularity']
        movie_title_to_id = {title: mid for mid, title in movie_id_to_title.items()}
        
        # Create dummy embeddings and features for enhanced recommendations
        # In a real implementation, these would come from your trained model
        movie_embeddings = {mid: torch.randn(64, device=device) for mid in movie_id_to_title.keys()}
        movie_features_dict = {mid: torch.randn(16, device=device) for mid in movie_id_to_title.keys()}
        
        # Validate model dimensions
        print(f"Number of movies in mapping: {len(movie_mapping)}")
        print(f"Number of movies in reverse mapping: {len(movie_mapping_reverse)}")
        print(f"Number of movies in title mapping: {len(movie_id_to_title)}")
        
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
    # Create user features (16-dimensional random features)
    user_features = torch.randn(1, 16, device=device)  # Single user with 16 features
    
    # Create movie features (16-dimensional random features for all movies)
    movie_features = torch.randn(num_movies, 16, device=device)
    
    # Create edge indices for input movies (user -> movie edges)
    edge_indices = []
    for movie_id in movie_ids:
        if movie_id in movie_mapping:
            movie_idx = movie_mapping[movie_id]
            edge_indices.append([0, movie_idx])  # user_idx=0, movie_idx
    
    if not edge_indices:
        # Fallback: create edges to first few movies
        edge_indices = [[0, i] for i in range(min(3, num_movies))]
    
    edge_index = torch.tensor(edge_indices, dtype=torch.long, device=device).t()
    
    return user_features, movie_features, edge_index

@app.route('/api/recommend', methods=['GET', 'POST'])
def get_recommendations():
    try:
        # Handle both GET and POST requests
        if request.method == 'GET':
            # For GET requests, get movies from query parameters
            user_movies = request.args.getlist('movies[]')
        else:
            # For POST requests, get movies from JSON body
            data = request.get_json()
            if not data or 'movies' not in data:
                return jsonify({'error': 'Please provide a list of movies'}), 400
            user_movies = data['movies']
        
        # Validate input
        if not user_movies or not isinstance(user_movies, list) or not (1 <= len(user_movies) <= 15):
            return jsonify({'error': 'Please provide between 1 and 15 movies'}), 400
            
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
        
        print(f"User features shape: {user_features.shape}")
        print(f"Movie features shape: {movie_features.shape}")
        print(f"Edge index shape: {edge_index.shape}")
        print("Getting recommendations...")
        
        # Create enhanced recommendation engine with proper parameters
        enhanced_recommendations = EnhancedRecommendationEngine(
            model=model, 
            movie_embeddings=movie_embeddings,  # You'll need to define this
            movie_features=movie_features_dict,  # You'll need to define this
            movie_popularity=movie_id_to_popularity
        )
        
        # Call with correct parameters: movie titles and title_to_id mapping
        #recommended_movies = enhanced_recommendations.get_diverse_recommendations(user_movies, movie_title_to_id)
        #print(f"Recommended movies: {recommended_movies}")
        scores = enhanced_recommendations._get_base_predictions(user_movies, movie_title_to_id)

        #Get top k recommendations from tensor containing scores
        top_k_indices = scores.topk(10).indices.tolist()
        recommended_movies = [movie_id_to_title[idx] for idx in top_k_indices]
        
        print("\nFinal recommendations:", recommended_movies)
        return jsonify({
            'recommendations': recommended_movies
        })
        
    except Exception as e:
        print(f"Error in get_recommendations: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Provide more specific error messages
        error_msg = str(e)
        if "Tensors must have same number of dimensions" in error_msg:
            error_msg = "Model tensor dimension mismatch. Please check model configuration."
        elif "out of bounds" in error_msg:
            error_msg = "Index out of bounds error. Please check movie mappings."
        elif "CUDA" in error_msg:
            error_msg = "GPU memory error. Please try again or contact support."
        
        return jsonify({'error': error_msg}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

@app.route('/', methods=['GET'])
def home():
    return jsonify({'status': 'healthy', 'message': 'Welcome to the Movie Recommendation API'})


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

# Trakt API Integration Endpoints

# Thread-safe temporary storage for OAuth state (in production, use Redis or database)
oauth_states = {}
oauth_states_lock = threading.Lock()

def cleanup_expired_states():
    """Clean up expired OAuth states"""
    with oauth_states_lock:
        current_time = time.time()
        expired_states = [s for s, data in oauth_states.items() if current_time - data['timestamp'] > 600]
        for expired_state in expired_states:
            del oauth_states[expired_state]
        if expired_states:
            print(f"Cleaned up {len(expired_states)} expired OAuth states")

@app.route('/auth/trakt', methods=['GET'])
def trakt_auth():
    """Redirect user to Trakt OAuth authorization"""
    try:
        # Generate a unique state parameter for security
        state = secrets.token_urlsafe(32)
        
        # Store state with timestamp for cleanup (thread-safe)
        with oauth_states_lock:
            oauth_states[state] = {
                'timestamp': time.time(),
                'origin': request.headers.get('Origin', 'http://localhost:5173')
            }
        
        # Clean up expired states
        cleanup_expired_states()
        
        # Get authorization URL
        auth_url = trakt_api.get_authorization_url(state)
        print(f"Generated OAuth state: {state} for origin: {request.headers.get('Origin')}")
        return jsonify({'auth_url': auth_url})
    except Exception as e:
        print(f"Error in trakt_auth: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/auth/callback', methods=['GET'])
def trakt_callback():
    """Handle OAuth callback from Trakt"""
    try:
        code = request.args.get('code')
        state = request.args.get('state')
        print(f"Callback received - Code: {code}, State: {state}")
        print(f"Origin: {request.headers.get('Origin')}")
        
        # Verify state parameter (thread-safe)
        with oauth_states_lock:
            if state not in oauth_states:
                print(f"State {state} not found in oauth_states")
                print(f"Available states: {list(oauth_states.keys())}")
                return jsonify({'error': 'Invalid state parameter'}), 400
            
            # Get the stored state data
            state_data = oauth_states[state]
            origin = state_data['origin']
            
            # Clean up the used state
            del oauth_states[state]
        
        if not code:
            return jsonify({'error': 'No authorization code received'}), 400
        
        # Exchange code for tokens
        token_data = trakt_api.exchange_code_for_token(code)
        
        # Create session
        session_id = secrets.token_urlsafe(32)
        success = session_manager.create_session(
            session_id=session_id,
            access_token=token_data['access_token'],
            refresh_token=token_data['refresh_token'],
            expires_in=token_data['expires_in']
        )
        
        if not success:
            return jsonify({'error': 'Failed to create session'}), 500
        
        # Redirect to frontend with success and session ID
        if origin == "https://main.d2p9ieiqdwymip.amplifyapp.com":
            return redirect(f'https://main.d2p9ieiqdwymip.amplifyapp.com/auth-success?session_id={session_id}')
        elif origin == "https://dev.d2p9ieiqdwymip.amplifyapp.com":
            return redirect(f'https://dev.d2p9ieiqdwymip.amplifyapp.com/auth-success?session_id={session_id}')
        else:
            # Default redirect for localhost development
            return redirect(f'http://localhost:5173/auth-success?session_id={session_id}')
        
    except Exception as e:
        print(f"Error in trakt_callback: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/trakt/search', methods=['GET'])
def trakt_search():
    """Search for movies and TV shows on Trakt"""
    try:
        query = request.args.get('query', '')
        limit = int(request.args.get('limit', 10))
        
        if not query:
            return jsonify({'error': 'Query parameter is required'}), 400
        
        # Search for movies and shows
        results = trakt_api.search_movies_and_shows(query, limit)
        
        return jsonify({'results': results})
        
    except Exception as e:
        print(f"Error in trakt_search: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/trakt/recommend', methods=['POST'])
def trakt_recommend():
    """Get recommendations using Trakt data and GNN model"""
    try:
        data = request.get_json()
        
        if not data or 'movies' not in data:
            return jsonify({'error': 'Please provide a list of movies'}), 400
        
        user_movies = data['movies']
        
        if not isinstance(user_movies, list) or not (1 <= len(user_movies) <= 15):
            return jsonify({'error': 'Please provide between 1 and 15 movies'}), 400
        
        # Get session ID from request
        session_id = request.headers.get('X-Session-ID')
        if not session_id:
            return jsonify({'error': 'Session ID required'}), 401
        
        # Get access token
        access_token = session_manager.get_access_token(session_id)
        if not access_token:
            return jsonify({'error': 'Invalid or expired session'}), 401
        
        # First, try to get Trakt recommendations
        try:
            print(f"\nGetting Trakt recommendations...")
            print(f"Access token: {access_token[:20]}..." if access_token else "No access token")
            trakt_recommendations = trakt_api.get_movie_recommendations(access_token, 10)
            print(f"Trakt API response: {trakt_recommendations}")
            
            # Handle empty or invalid response
            if not trakt_recommendations or not isinstance(trakt_recommendations, list):
                print("Empty or invalid Trakt response, using empty list")
                trakt_titles = []
            else:
                trakt_titles = []
                for movie in trakt_recommendations:
                    if isinstance(movie, dict) and 'title' in movie:
                        trakt_titles.append(movie['title'])
                    else:
                        print(f"Skipping invalid movie entry: {movie}")
                
            print(f"Trakt titles: {trakt_titles}")
        except Exception as e:
            print(f"Error getting Trakt recommendations: {e}")
            print(f"Error type: {type(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            trakt_titles = []
            
        # If Trakt recommendations failed, try trending movies as fallback
        if not trakt_titles:
            try:
                print("Trakt recommendations failed, trying trending movies as fallback...")
                trending_movies = trakt_api.get_trending_movies(10)
                trakt_titles = [movie['title'] for movie in trending_movies if isinstance(movie, dict) and 'title' in movie]
                print(f"Fallback trending titles: {trakt_titles}")
            except Exception as e:
                print(f"Error getting trending movies fallback: {e}")
                trakt_titles = []
        
                # Then get GNN recommendations
        try:
            # Convert movie titles to IDs for GNN model
            movie_ids = []
            print(f"\nProcessing movies: {user_movies}")
            print(f"Available movies in dataset: {len(movie_title_to_id)}")
            print(f"Sample movies in dataset: {list(movie_title_to_id.keys())[:5]}")
            print(f"Movie title to ID mapping sample: {dict(list(movie_title_to_id.items())[:3])}")
            print(f"Movie mapping reverse sample: {dict(list(movie_mapping_reverse.items())[:3])}")
            print(f"Movie ID to title sample: {dict(list(movie_id_to_title.items())[:3])}")
            
            # Check if we have a proper movie dataset
            if len(movie_title_to_id) < 100:  # Small dataset indicates test data
                print("Warning: Small dataset detected. Using Trakt recommendations only.")
                gnn_recommendations = []
            else:
                for movie_title in user_movies:
                    print(f"\nLooking for: {movie_title}")
                    if movie_title not in movie_title_to_id:
                        print(f"  Not found exactly, searching for similar...")
                        # Try to find similar movie in our dataset
                        similar_movie = None
                        for title, movie_id in movie_title_to_id.items():
                            if movie_title.lower() in title.lower() or title.lower() in movie_title.lower():
                                similar_movie = movie_id
                                print(f"  Found similar: {title} -> {movie_id}")
                                break
                        
                        if similar_movie:
                            movie_ids.append(similar_movie)
                            print(f"  Added similar movie ID: {similar_movie}")
                        else:
                            # Use a random movie from dataset instead of always the same one
                            import random
                            available_ids = list(movie_title_to_id.values())
                            random_movie = random.choice(available_ids)
                            print(f"  No similar found, using random: {random_movie}")
                            movie_ids.append(random_movie)
                    else:
                        movie_id = movie_title_to_id[movie_title]
                        print(f"  Found exact match: {movie_title} -> {movie_id}")
                        movie_ids.append(movie_id)
                        print(f"  Added exact movie ID: {movie_id}")
                
                print(f"Final movie IDs: {movie_ids}")
                
                # Check if movie IDs are in the mapping
                valid_movie_ids = []
                for movie_id in movie_ids:
                    if movie_id in movie_mapping:
                        valid_movie_ids.append(movie_id)
                        print(f"  Valid movie ID: {movie_id} -> mapping index: {movie_mapping[movie_id]}")
                    else:
                        print(f"  Invalid movie ID: {movie_id} not in movie_mapping")
                
                if not valid_movie_ids:
                    print("No valid movie IDs found, using first few movies from dataset")
                    valid_movie_ids = list(movie_title_to_id.values())[:3]
                
                movie_ids = valid_movie_ids
                
                # Ensure we have unique movie IDs to avoid tensor dimension issues
                unique_movie_ids = list(set(movie_ids))
                if len(unique_movie_ids) < len(movie_ids):
                    print(f"Removed duplicates: {movie_ids} -> {unique_movie_ids}")
                    movie_ids = unique_movie_ids
                
                # Get GNN predictions
                user_features, movie_features, edge_index = prepare_model_input(movie_ids)
                
                with torch.no_grad():
                    # Create edge indices for all possible movie recommendations (user -> all movies)
                    all_edge_indices = []
                    for movie_idx in range(num_movies):
                        all_edge_indices.append([0, movie_idx])  # user_idx=0, movie_idx
                    
                    all_edge_index = torch.tensor(all_edge_indices, dtype=torch.long).t()
                    
                    print(f"User features shape: {user_features.shape}")
                    print(f"Movie features shape: {movie_features.shape}")
                    print(f"All edge index shape: {all_edge_index.shape}")
                    
                    predictions = model.predict(
                        user_features,
                        movie_features,
                        all_edge_index
                    )
                    
                    print(f"Predictions shape: {predictions.shape}")
                    print(f"Predictions: {predictions.squeeze()[:10]}")  # Show first 10 predictions
                    
                    # Get top 10 recommendations
                    top_10_indices = predictions.squeeze().topk(10).indices.tolist()
                    print(f"Top 10 indices: {top_10_indices}")
                    
                    gnn_recommendations = []
                    for idx in top_10_indices:
                        if idx in movie_mapping_reverse:
                            movie_id = movie_mapping_reverse[idx]
                            if movie_id in movie_id_to_title:
                                gnn_recommendations.append(movie_id_to_title[movie_id])
                            else:
                                print(f"Warning: Movie ID {movie_id} not found in movie_id_to_title")
                        else:
                            print(f"Warning: Index {idx} not found in movie_mapping_reverse")
                
        except Exception as e:
            print(f"Error getting GNN recommendations: {e}")
            print(f"Error type: {type(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            gnn_recommendations = []
        
        # Combine recommendations (GNN first, then Trakt)
        combined_recommendations = []
        
        # Create a set of input movies for filtering (case-insensitive)
        input_movies_set = {movie.lower() for movie in user_movies}
        
        # Add GNN recommendations first (prioritize our trained model)
        for title in gnn_recommendations:
            # Skip if already in combined list or if it's one of the input movies
            if (title not in combined_recommendations and 
                title.lower() not in input_movies_set and
                len(combined_recommendations) < 15):
                combined_recommendations.append(title)
        
        # Add Trakt recommendations as backup
        for title in trakt_titles:
            # Skip if already in combined list or if it's one of the input movies
            if (title not in combined_recommendations and 
                title.lower() not in input_movies_set and
                len(combined_recommendations) < 15):
                combined_recommendations.append(title)
        
        print(f"\nFinal response:")
        print(f"  Input movies: {user_movies}")
        print(f"  Trakt recommendations: {len(trakt_titles)}")
        print(f"  GNN recommendations: {len(gnn_recommendations)}")
        print(f"  Combined recommendations: {len(combined_recommendations)}")
        print(f"  Combined list: {combined_recommendations}")
        
        # Log any filtered recommendations
        filtered_gnn = [title for title in gnn_recommendations if title.lower() in input_movies_set]
        filtered_trakt = [title for title in trakt_titles if title.lower() in input_movies_set]
        if filtered_gnn or filtered_trakt:
            print(f"  Filtered out (input movies): GNN={filtered_gnn}, Trakt={filtered_trakt}")
        
        response_data = {
            'recommendations': combined_recommendations,
            'trakt_recommendations': trakt_titles,
            'gnn_recommendations': gnn_recommendations
        }
        
        print(f"Returning response: {response_data}")
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error in trakt_recommend: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/trakt/trending', methods=['GET'])
def trakt_trending():
    """Get trending movies from Trakt"""
    try:
        limit = int(request.args.get('limit', 10))
        trending = trakt_api.get_trending_movies(limit)
        return jsonify({'trending': trending})
    except Exception as e:
        print(f"Error in trakt_trending: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/trakt/popular', methods=['GET'])
def trakt_popular():
    """Get popular movies from Trakt"""
    try:
        limit = int(request.args.get('limit', 10))
        popular = trakt_api.get_popular_movies(limit)
        return jsonify({'popular': popular})
    except Exception as e:
        print(f"Error in trakt_popular: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/trakt/collect-data', methods=['POST'])
def collect_trakt_data():
    """Collect Trakt data for training"""
    try:
        from trakt_data_collector import data_collector
        
        data = request.get_json()
        access_tokens = data.get('access_tokens', [])
        usernames = data.get('usernames', [])
        
        if not access_tokens:
            return jsonify({'error': 'Access tokens required'}), 400
        
        # Start data collection
        output_dir = data_collector.collect_all_data_for_training(access_tokens, usernames)
        
        if output_dir:
            return jsonify({
                'message': 'Data collection completed successfully',
                'output_directory': output_dir
            })
        else:
            return jsonify({'error': 'Data collection failed'}), 500
            
    except Exception as e:
        print(f"Error in collect_trakt_data: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/trakt/user-history', methods=['GET'])
def trakt_user_history():
    """Get user's recently watched and favorite movies/shows from Trakt"""
    try:
        # Get session ID from request
        session_id = request.headers.get('X-Session-ID')
        if not session_id:
            return jsonify({'error': 'Session ID required'}), 401

        # Get access token
        access_token = session_manager.get_access_token(session_id)
        if not access_token:
            return jsonify({'error': 'Invalid or expired session'}), 401

        # Get username (optional, default to 'me')
        username = request.args.get('username', 'me')

        # Get recently watched movies and shows
        watched_movies = trakt_api.get_user_watched_movies(access_token, username)
        watched_shows = trakt_api.get_user_watched_shows(access_token, username)

        # Sort by most recent (if possible)
        recently_watched_movies = sorted(watched_movies, key=lambda x: x.get('last_watched_at', ''), reverse=True)[:15]
        recently_watched_shows = sorted(watched_shows, key=lambda x: x.get('last_watched_at', ''), reverse=True)[:15]

        # Get favorite movies and shows (rated >= 8)
        movie_ratings = trakt_api.get_user_ratings(access_token, username, 'movies')
        show_ratings = trakt_api.get_user_ratings(access_token, username, 'shows')
        favorite_movies = [m for m in movie_ratings if m.get('rating', 0) >= 8][:15]
        favorite_shows = [s for s in show_ratings if s.get('rating', 0) >= 8][:15]

        return jsonify({
            'recently_watched_movies': recently_watched_movies,
            'recently_watched_shows': recently_watched_shows,
            'favorite_movies': favorite_movies,
            'favorite_shows': favorite_shows
        })
    except Exception as e:
        print(f"Error in trakt_user_history: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
