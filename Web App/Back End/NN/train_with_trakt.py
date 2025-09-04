import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
import pandas as pd
import numpy as np
import os
import json
from typing import List, Dict, Tuple
from dotenv import load_dotenv
from enhanced_mpgnn import EnhancedMPGNN
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Dataset.trakt_data_collector import data_collector
import time
import random
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

# GPU Configuration
def get_device():
    """Get the best available device (GPU if available, else CPU)"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = torch.device('cpu')
        print("💻 Using CPU")
    return device

# Global device variable
DEVICE = get_device()

class EnhancedTraktDataProcessor:
    def __init__(self, data_dir: str = 'trakt_data'):
        self.data_dir = data_dir
        self.users_df = None
        self.movies_df = None
        self.ratings_df = None
        
    def load_data(self) -> bool:
        """Load the collected Trakt data"""
        try:
            users_path = os.path.join(self.data_dir, 'users.csv')
            movies_path = os.path.join(self.data_dir, 'movies.csv')
            ratings_path = os.path.join(self.data_dir, 'ratings.csv')
            
            if not all(os.path.exists(path) for path in [users_path, movies_path, ratings_path]):
                print("Data files not found. Please run data collection first.")
                return False
            
            self.users_df = pd.read_csv(users_path)
            self.movies_df = pd.read_csv(movies_path)
            self.ratings_df = pd.read_csv(ratings_path)
            
            print(f"Loaded data:")
            print(f"  Users: {len(self.users_df)}")
            print(f"  Movies: {len(self.movies_df)}")
            print(f"  Ratings: {len(self.ratings_df)}")
            
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def preprocess_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """Preprocess data for GNN training"""
        try:
            # Create mappings
            user_id_to_idx = {user_id: idx for idx, user_id in enumerate(self.users_df['user_id'])}
            movie_id_to_idx = {movie_id: idx for idx, movie_id in enumerate(self.movies_df['movie_id'])}
            
            # Create reverse mappings
            user_idx_to_id = {idx: user_id for user_id, idx in user_id_to_idx.items()}
            movie_idx_to_id = {idx: movie_id for movie_id, idx in movie_id_to_idx.items()}
            
            # Create movie title mappings
            movie_id_to_title = {}
            for _, row in self.movies_df.iterrows():
                movie_id_to_title[row['movie_id']] = row['title']
            
            movie_title_to_id = {title: movie_id for movie_id, title in movie_id_to_title.items()}
            
            # Create popularity mapping (based on number of ratings)
            movie_id_to_popularity = {}
            for movie_id in self.movies_df['movie_id']:
                popularity = len(self.ratings_df[self.ratings_df['movie_id'] == movie_id])
                movie_id_to_popularity[movie_id] = popularity
            
            # Prepare edge indices and edge features
            edge_indices = []
            edge_features = []
            
            for _, rating in self.ratings_df.iterrows():
                user_idx = user_id_to_idx[rating['user_id']]
                movie_idx = movie_id_to_idx[rating['movie_id']]
                
                # Add user->movie edge
                edge_indices.append([user_idx, movie_idx])
                edge_features.append([rating['rating'] / 10.0])  # Normalize rating to [0, 1]
                
                # Add movie->user edge (bidirectional)
                edge_indices.append([movie_idx, user_idx])
                edge_features.append([rating['rating'] / 10.0])
            
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_features, dtype=torch.float)
            
            # Create node features
            num_users = len(self.users_df)
            num_movies = len(self.movies_df)
            
            # User features (simple embedding-like features)
            user_features = torch.randn(num_users, 16)  # 16-dimensional features for users
            
            # Movie features (simple embedding-like features)
            movie_features = torch.randn(num_movies, 16)  # 16-dimensional features for movies
            
            # Combine all node features
            node_features = torch.cat([user_features, movie_features], dim=0)
            
            # Create labels (rating predictions)
            labels = []
            for _, rating in self.ratings_df.iterrows():
                user_idx = user_id_to_idx[rating['user_id']]
                movie_idx = movie_id_to_idx[rating['movie_id']] + num_users  # Offset for movie nodes
                labels.append([user_idx, movie_idx, rating['rating'] / 10.0])
            
            labels = torch.tensor(labels, dtype=torch.float)
            
            # Move tensors to GPU if available
            node_features = node_features.to(DEVICE)
            edge_index = edge_index.to(DEVICE)
            edge_attr = edge_attr.to(DEVICE)
            labels = labels.to(DEVICE)
            
            # Create mappings dictionary
            mappings = {
                'user_id_to_idx': user_id_to_idx,
                'movie_id_to_idx': movie_id_to_idx,
                'user_idx_to_id': user_idx_to_id,
                'movie_idx_to_id': movie_idx_to_id,
                'movie_id_to_title': movie_id_to_title,
                'movie_title_to_id': movie_title_to_id,
                'movie_id_to_popularity': movie_id_to_popularity,
                'num_users': num_users,
                'num_movies': num_movies,
                'dataset_size': len(self.ratings_df)
            }
            
            return node_features, edge_index, edge_attr, labels, mappings
            
        except Exception as e:
            print(f"Error preprocessing data: {e}")
            return None, None, None, None, None

class ComprehensiveMovieCollector:
    """Enhanced movie collector to gather 10k movies from every category"""
    
    def __init__(self, output_dir: str = 'trakt_data'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.collected_movies = set()  # Track unique movies by Trakt ID
    
    def collect_movies_by_year(self, start_year: int = 2020, end_year: int = 2024, movies_per_year: int = 500):
        """Collect movies by year range"""
        from trakt_api import trakt_api
        
        print(f"Collecting movies from {start_year} to {end_year}...")
        all_movies = []
        
        for year in range(start_year, end_year + 1):
            try:
                print(f"Collecting movies from year {year}...")
                # Use popular movies endpoint with year filter
                url = f"https://api.trakt.tv/movies/popular"
                params = {'limit': movies_per_year, 'extended': 'full'}
                
                response = trakt_api.get_popular_movies(movies_per_year)
                
                # Filter by year
                year_movies = [movie for movie in response if movie.get('year') == year]
                all_movies.extend(year_movies)
                
                print(f"Collected {len(year_movies)} movies from {year}")
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                print(f"Error collecting movies from {year}: {e}")
                continue
        
        return all_movies
    
    def collect_movies_by_genre(self, genres: List[str], movies_per_genre: int = 500):
        """Collect movies by genre"""
        from trakt_api import trakt_api
        
        print(f"Collecting movies by genres: {genres}")
        all_movies = []
        
        for genre in genres:
            try:
                print(f"Collecting {movies_per_genre} movies for genre: {genre}")
                
                # Search for movies in this genre
                search_terms = [f"{genre} movie", f"{genre} film", genre]
                genre_movies = []
                
                for term in search_terms:
                    try:
                        results = trakt_api.search_movies_and_shows(term, movies_per_genre // len(search_terms))
                        movies = [item for item in results if item['type'] == 'movie']
                        genre_movies.extend(movies)
                        time.sleep(0.5)  # Rate limiting
                    except Exception as e:
                        print(f"Error searching for {term}: {e}")
                        continue
                
                # Remove duplicates
                unique_genre_movies = {}
                for movie in genre_movies:
                    trakt_id = movie['ids'].get('trakt')
                    if trakt_id and trakt_id not in unique_genre_movies:
                        unique_genre_movies[trakt_id] = movie
                
                genre_movies_list = list(unique_genre_movies.values())[:movies_per_genre]
                all_movies.extend(genre_movies_list)
                
                print(f"Collected {len(genre_movies_list)} unique movies for genre: {genre}")
                
            except Exception as e:
                print(f"Error collecting movies for genre {genre}: {e}")
                continue
        
        return all_movies
    
    def collect_movies_by_popularity_tiers(self, tiers: List[int] = [100, 500, 1000, 2000, 5000]):
        """Collect movies from different popularity tiers"""
        from trakt_api import trakt_api
        
        print(f"Collecting movies from popularity tiers: {tiers}")
        all_movies = []
        
        for tier in tiers:
            try:
                print(f"Collecting top {tier} popular movies...")
                movies = trakt_api.get_popular_movies(tier)
                all_movies.extend(movies)
                print(f"Collected {len(movies)} movies from tier {tier}")
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                print(f"Error collecting movies from tier {tier}: {e}")
                continue
        
        return all_movies
    
    def collect_movies_by_trending_and_recent(self, trending_limit: int = 1000, recent_years: int = 3):
        """Collect trending and recent movies"""
        from trakt_api import trakt_api
        
        print("Collecting trending and recent movies...")
        all_movies = []
        
        # Collect trending movies
        try:
            print(f"Collecting {trending_limit} trending movies...")
            trending_movies = trakt_api.get_trending_movies(trending_limit)
            all_movies.extend(trending_movies)
            print(f"Collected {len(trending_movies)} trending movies")
        except Exception as e:
            print(f"Error collecting trending movies: {e}")
        
        # Collect recent movies by year
        current_year = 2024
        for year in range(current_year - recent_years + 1, current_year + 1):
            try:
                print(f"Collecting recent movies from {year}...")
                # Use search to find recent movies
                search_terms = [str(year), f"movie {year}"]
                year_movies = []
                
                for term in search_terms:
                    try:
                        results = trakt_api.search_movies_and_shows(term, 200)
                        movies = [item for item in results if item['type'] == 'movie' and item.get('year') == year]
                        year_movies.extend(movies)
                        time.sleep(0.5)
                    except Exception as e:
                        print(f"Error searching for {term}: {e}")
                        continue
                
                # Remove duplicates
                unique_year_movies = {}
                for movie in year_movies:
                    trakt_id = movie['ids'].get('trakt')
                    if trakt_id and trakt_id not in unique_year_movies:
                        unique_year_movies[trakt_id] = movie
                
                year_movies_list = list(unique_year_movies.values())
                all_movies.extend(year_movies_list)
                print(f"Collected {len(year_movies_list)} movies from {year}")
                
            except Exception as e:
                print(f"Error collecting movies from {year}: {e}")
                continue
        
        return all_movies
    
    def collect_comprehensive_movie_dataset(self, target_movies: int = 10000):
        """Collect comprehensive movie dataset from multiple sources"""
        from trakt_api import trakt_api
        
        print(f"Starting comprehensive movie collection targeting {target_movies} movies...")
        
        all_movies = []
        
        # 1. Collect by popularity tiers
        print("\n1. Collecting by popularity tiers...")
        popularity_movies = self.collect_movies_by_popularity_tiers([100000])
        all_movies.extend(popularity_movies)
        print(f"Total movies so far: {len(all_movies)}")
        
        # 2. Collect by genres
        print("\n2. Collecting by genres...")
        genres = [
            'action', 'comedy', 'drama', 'horror', 'romance', 'sci-fi',
            'thriller', 'adventure', 'fantasy', 'mystery', 'crime',
            'documentary', 'animation', 'family', 'war', 'western',
            'musical', 'biography', 'history', 'sport', 'superhero',
            'zombie', 'vampire', 'alien', 'robot', 'time travel'
        ]
        genre_movies = self.collect_movies_by_genre(genres, 100000)
        all_movies.extend(genre_movies)
        print(f"Total movies so far: {len(all_movies)}")
        
        # 3. Collect by year range
        print("\n3. Collecting by year range...")
        year_movies = self.collect_movies_by_year(2020, 2025, 10000)
        all_movies.extend(year_movies)
        print(f"Total movies so far: {len(all_movies)}")
        
        # 4. Collect trending and recent
        print("\n4. Collecting trending and recent movies...")
        trending_movies = self.collect_movies_by_trending_and_recent(100000, 3)
        all_movies.extend(trending_movies)
        print(f"Total movies so far: {len(all_movies)}")
        
        # 5. Remove duplicates and limit to target
        print("\n5. Removing duplicates and finalizing dataset...")
        unique_movies = {}
        for movie in all_movies:
            trakt_id = movie['ids'].get('trakt')
            if trakt_id and trakt_id not in unique_movies:
                unique_movies[trakt_id] = movie
        
        final_movies = list(unique_movies.values())
        
        # If we don't have enough movies, add more from popular
        if len(final_movies) < target_movies:
            print(f"Need {target_movies - len(final_movies)} more movies...")
            try:
                additional_movies = trakt_api.get_popular_movies(target_movies - len(final_movies))
                for movie in additional_movies:
                    trakt_id = movie['ids'].get('trakt')
                    if trakt_id and trakt_id not in unique_movies:
                        unique_movies[trakt_id] = movie
                final_movies = list(unique_movies.values())
            except Exception as e:
                print(f"Error collecting additional movies: {e}")
        
        # Limit to target size
        final_movies = final_movies[:target_movies]
        
        print(f"Final dataset: {len(final_movies)} unique movies")
        
        # Save comprehensive movie dataset
        filename = f"{self.output_dir}/comprehensive_movies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(final_movies, f, indent=2)
        
        print(f"Comprehensive movie dataset saved to: {filename}")
        return final_movies


def train_gnn_with_comprehensive_data(data_dir: str = 'trakt_data', 
                                     hidden_channels: int = 64,
                                     learning_rate: float = 0.001,
                                     epochs: int = 100,
                                     batch_size: int = 32,
                                     save_path: str = 'models/trakt_gnn_model.pth',
                                     target_movies: int = 10000,
                                     use_mixed_precision: bool = True):
    """Train GNN model with comprehensive Trakt data"""
    
    print("Starting comprehensive GNN training with Trakt data...")
    
    # Initialize data processor
    processor = EnhancedTraktDataProcessor(data_dir)
    
    # Check if we need to collect more data
    if not processor.load_data() or len(processor.movies_df) < target_movies * 0.8:  # 80% of target
        print(f"Insufficient data ({len(processor.movies_df) if processor.movies_df is not None else 0} movies). Collecting comprehensive dataset...")
        
        # Initialize comprehensive collector
        collector = ComprehensiveMovieCollector(data_dir)
        
        # Collect comprehensive movie dataset
        comprehensive_movies = collector.collect_comprehensive_movie_dataset(target_movies)
        
        # Create synthetic user data for training
        print("Creating synthetic user data for training...")
        synthetic_users = []
        synthetic_ratings = []
        
        # Create 1000 synthetic users
        num_users = 1000
        for i in range(num_users):
            user_id = i
            username = f"synthetic_user_{i}"
            synthetic_users.append({
                'user_id': user_id,
                'username': username
            })
            
            # Each user rates 50-200 random movies
            num_ratings = random.randint(50, 200)
            user_movies = random.sample(comprehensive_movies, min(num_ratings, len(comprehensive_movies)))
            
            for j, movie in enumerate(user_movies):
                # Generate realistic ratings (biased towards positive)
                rating = random.choices([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
                                      weights=[0.05, 0.05, 0.1, 0.1, 0.15, 0.15, 0.2, 0.15, 0.05, 0.05])[0]
                
                synthetic_ratings.append({
                    'user_id': user_id,
                    'movie_id': j,  # Use index as movie_id
                    'rating': rating,
                    'rated_at': datetime.now().isoformat()
                })
        
        # Create movies dataframe
        movies_data = []
        for i, movie in enumerate(comprehensive_movies):
            movies_data.append({
                'movie_id': i,
                'trakt_id': movie['ids'].get('trakt', i),
                'title': movie['title'],
                'year': movie.get('year'),
                'rating': movie.get('rating', 0),
                'votes': movie.get('votes', 0)
            })
        
        # Save synthetic dataset
        users_df = pd.DataFrame(synthetic_users)
        movies_df = pd.DataFrame(movies_data)
        ratings_df = pd.DataFrame(synthetic_ratings)
        
        users_df.to_csv(f"{data_dir}/users.csv", index=False)
        movies_df.to_csv(f"{data_dir}/movies.csv", index=False)
        ratings_df.to_csv(f"{data_dir}/ratings.csv", index=False)
        
        print(f"Synthetic dataset created:")
        print(f"  Users: {len(users_df)}")
        print(f"  Movies: {len(movies_df)}")
        print(f"  Ratings: {len(ratings_df)}")
        
        # Reload data
        processor.load_data()
    
    # Preprocess data
    node_features, edge_index, edge_attr, labels, mappings = processor.preprocess_data()
    
    if node_features is None:
        print("Failed to preprocess data. Exiting.")
        return False
    
    # Create PyTorch Geometric Data object
    data = Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=labels
    )
    
    print(f"Data prepared:")
    print(f"  Nodes: {data.x.size(0)}")
    print(f"  Edges: {data.edge_index.size(1)}")
    print(f"  Labels: {data.y.size(0)}")
    
    # Initialize model
    num_users = mappings['num_users']
    num_movies = mappings['num_movies']
    
    """model = MPGNN(
        num_user_features=16,  # Fixed feature dimension for users
        num_movie_features=16,  # Fixed feature dimension for movies
        hidden_channels=hidden_channels,
        num_classes=1
    )"""
    model = EnhancedMPGNN(
        num_user_features=16,
        num_movie_features=16,
        hidden_channels=hidden_channels,
        num_classes=1
    )
    
    # Move model to GPU
    model = model.to(DEVICE)
    
    # Set up optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Set up mixed precision training if available and requested
    scaler = None
    if use_mixed_precision and torch.cuda.is_available():
        scaler = torch.amp.GradScaler('cuda')
        print("🔧 Using mixed precision training for faster GPU computation")
    
    # Training loop
    model.train()
    print(f"\nStarting training for {epochs} epochs...")
    
    # Split node features for user and movie nodes
    num_users = mappings['num_users']
    x_user = data.x[:num_users].to(DEVICE)
    x_movie = data.x[num_users:].to(DEVICE)
    
    # Create proper edge indices for the model directly from ratings data
    # This is simpler and more reliable than trying to match with existing edges
    user_to_movie_edges = []
    target_ratings = []
    
    # Get user and movie mappings
    user_id_to_idx = mappings['user_id_to_idx']
    movie_id_to_idx = mappings['movie_id_to_idx']
    
    # Create edges directly from ratings
    for _, rating in processor.ratings_df.iterrows():
        user_idx = user_id_to_idx[rating['user_id']]
        movie_idx = movie_id_to_idx[rating['movie_id']]
        rating_value = rating['rating'] / 10.0  # Normalize to [0, 1]
        
        user_to_movie_edges.append([user_idx, movie_idx])
        target_ratings.append(rating_value)
    
    if not user_to_movie_edges:
        print("Error: No user->movie edges found. Check data preprocessing.")
        return False
    
    edge_index_adjusted = torch.tensor(user_to_movie_edges, dtype=torch.long).t().contiguous().to(DEVICE)
    target_ratings = torch.tensor(target_ratings, dtype=torch.float).to(DEVICE)
    
    print(f"Adjusted edge indices: {edge_index_adjusted.size(1)} user->movie edges")
    print(f"Target ratings: {target_ratings.size(0)} ratings")
    
    # Training monitoring
    start_time = time.time()
    best_loss = float('inf')
    
    print(f"\n🚀 Starting GPU-accelerated training for {epochs} epochs...")
    if torch.cuda.is_available():
        print(f"   GPU Memory Usage: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass with mixed precision if available
        if scaler is not None:
            with torch.cuda.amp.autocast():
                out = model(x_user, x_movie, edge_index_adjusted)
                loss = F.mse_loss(out.squeeze(), target_ratings)
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard forward and backward pass
            out = model(x_user, x_movie, edge_index_adjusted)
            loss = F.mse_loss(out.squeeze(), target_ratings)
            loss.backward()
            optimizer.step()
        
        # Track best loss
        if loss.item() < best_loss:
            best_loss = loss.item()
        
        if (epoch + 1) % 10 == 0:
            elapsed_time = time.time() - start_time
            avg_time_per_epoch = elapsed_time / (epoch + 1)
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1024**3
                print(f'Epoch {epoch+1:03d}, Loss: {loss.item():.4f}, Best: {best_loss:.4f}, '
                      f'Time/epoch: {avg_time_per_epoch:.2f}s, GPU Memory: {gpu_memory:.2f} GB')
            else:
                print(f'Epoch {epoch+1:03d}, Loss: {loss.item():.4f}, Best: {best_loss:.4f}, '
                      f'Time/epoch: {avg_time_per_epoch:.2f}s')
    
    total_training_time = time.time() - start_time
    print(f"\n✅ Training completed!")
    print(f"   Total time: {total_training_time:.2f} seconds")
    print(f"   Average time per epoch: {total_training_time/epochs:.2f} seconds")
    print(f"   Best loss achieved: {best_loss:.4f}")
    
    if torch.cuda.is_available():
        print(f"   Final GPU Memory Usage: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        # Clear GPU cache
        torch.cuda.empty_cache()
    
    # Save model and mappings
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'user_mapping': mappings['user_id_to_idx'],
        'movie_mapping': mappings['movie_id_to_idx'],
        'movie_mapping_reverse': mappings['movie_idx_to_id'],
        'movie_id_to_title': mappings['movie_id_to_title'],
        'movie_id_to_popularity': mappings['movie_id_to_popularity'],
        'movie_title_to_id': mappings['movie_title_to_id'],
        'dataset_size': mappings['dataset_size'],
        'num_users': mappings['num_users'],
        'num_movies': mappings['num_movies'],
        'hidden_channels': hidden_channels
    }, save_path)
    
    print(f"Model saved to: {save_path}")
    
    # Save training summary
    summary = {
        'training_date': pd.Timestamp.now().isoformat(),
        'data_directory': data_dir,
        'num_users': mappings['num_users'],
        'num_movies': mappings['num_movies'],
        'num_ratings': mappings['dataset_size'],
        'hidden_channels': hidden_channels,
        'learning_rate': learning_rate,
        'epochs': epochs,
        'final_loss': loss.item(),
        'best_loss': best_loss,
        'total_training_time': total_training_time,
        'avg_time_per_epoch': total_training_time/epochs,
        'device_used': str(DEVICE),
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
        'model_path': save_path,
        'target_movies': target_movies
    }
    
    summary_path = os.path.join(data_dir, 'training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Training summary saved to: {summary_path}")
    
    return True

def train_gnn_with_comprehensive_data(data_dir: str = 'trakt_data', 
                                     hidden_channels: int = 64,
                                     learning_rate: float = 0.001,
                                     epochs: int = 100,
                                     batch_size: int = 32,
                                     save_path: str = '../models/trakt_gnn_model.pth',
                                     target_movies: int = 10000,
                                     use_mixed_precision: bool = True,
                                     patience: int = 20,  # Early stopping patience
                                     min_delta: float = 1e-4):  # Minimum improvement threshold
    """Train GNN model with comprehensive Trakt data"""
    
    print("Starting comprehensive GNN training with Trakt data...")
    
    # Initialize data processor
    processor = EnhancedTraktDataProcessor(data_dir)
    
    # Check if we need to collect more data
    if not processor.load_data() or len(processor.movies_df) < target_movies * 0.8:  # 80% of target
        print(f"Insufficient data ({len(processor.movies_df) if processor.movies_df is not None else 0} movies). Collecting comprehensive dataset...")
        
        # Initialize comprehensive collector
        collector = ComprehensiveMovieCollector(data_dir)
        
        # Collect comprehensive movie dataset
        comprehensive_movies = collector.collect_comprehensive_movie_dataset(target_movies)
        
        # Create synthetic user data for training
        print("Creating synthetic user data for training...")
        synthetic_users = []
        synthetic_ratings = []
        
        # Create 1000 synthetic users
        num_users = 1000
        for i in range(num_users):
            user_id = i
            username = f"synthetic_user_{i}"
            synthetic_users.append({
                'user_id': user_id,
                'username': username
            })
            
            # Each user rates 50-200 random movies
            num_ratings = random.randint(50, 200)
            user_movies = random.sample(comprehensive_movies, min(num_ratings, len(comprehensive_movies)))
            
            for j, movie in enumerate(user_movies):
                # Generate realistic ratings (biased towards positive)
                rating = random.choices([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
                                      weights=[0.05, 0.05, 0.1, 0.1, 0.15, 0.15, 0.2, 0.15, 0.05, 0.05])[0]
                
                synthetic_ratings.append({
                    'user_id': user_id,
                    'movie_id': j,  # Use index as movie_id
                    'rating': rating,
                    'rated_at': datetime.now().isoformat()
                })
        
        # Create movies dataframe
        movies_data = []
        for i, movie in enumerate(comprehensive_movies):
            movies_data.append({
                'movie_id': i,
                'trakt_id': movie['ids'].get('trakt', i),
                'title': movie['title'],
                'year': movie.get('year'),
                'rating': movie.get('rating', 0),
                'votes': movie.get('votes', 0)
            })
        
        # Save synthetic dataset
        users_df = pd.DataFrame(synthetic_users)
        movies_df = pd.DataFrame(movies_data)
        ratings_df = pd.DataFrame(synthetic_ratings)
        
        users_df.to_csv(f"{data_dir}/users.csv", index=False)
        movies_df.to_csv(f"{data_dir}/movies.csv", index=False)
        ratings_df.to_csv(f"{data_dir}/ratings.csv", index=False)
        
        print(f"Synthetic dataset created:")
        print(f"  Users: {len(users_df)}")
        print(f"  Movies: {len(movies_df)}")
        print(f"  Ratings: {len(ratings_df)}")
        
        # Reload data
        processor.load_data()
    
    # Preprocess data
    node_features, edge_index, edge_attr, labels, mappings = processor.preprocess_data()
    
    if node_features is None:
        print("Failed to preprocess data. Exiting.")
        return False
    
    # Create PyTorch Geometric Data object
    data = Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=labels
    )
    
    print(f"Data prepared:")
    print(f"  Nodes: {data.x.size(0)}")
    print(f"  Edges: {data.edge_index.size(1)}")
    print(f"  Labels: {data.y.size(0)}")
    
    # Initialize model
    num_users = mappings['num_users']
    num_movies = mappings['num_movies']
    
    """model = MPGNN(
        num_user_features=16,  # Fixed feature dimension for users
        num_movie_features=16,  # Fixed feature dimension for movies
        hidden_channels=hidden_channels,
        num_classes=1
    )"""
    model = EnhancedMPGNN(
        num_user_features=16,
        num_movie_features=16,
        hidden_channels=hidden_channels,
        num_classes=1
    )
    
    # Move model to GPU
    model = model.to(DEVICE)
    
    # Set up optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Set up mixed precision training if available and requested
    scaler = None
    if use_mixed_precision and torch.cuda.is_available():
        scaler = torch.amp.GradScaler('cuda')
        print("🔧 Using mixed precision training for faster GPU computation")
    
    # Training loop
    model.train()
    print(f"\nStarting training for {epochs} epochs...")
    
    # Split node features for user and movie nodes
    num_users = mappings['num_users']
    x_user = data.x[:num_users].to(DEVICE)
    x_movie = data.x[num_users:].to(DEVICE)
    
    # Create proper edge indices for the model directly from ratings data
    user_to_movie_edges = []
    target_ratings = []
    
    # Get user and movie mappings
    user_id_to_idx = mappings['user_id_to_idx']
    movie_id_to_idx = mappings['movie_id_to_idx']
    
    # Create edges directly from ratings
    for _, rating in processor.ratings_df.iterrows():
        user_idx = user_id_to_idx[rating['user_id']]
        movie_idx = movie_id_to_idx[rating['movie_id']]
        rating_value = rating['rating'] / 10.0  # Normalize to [0, 1]
        
        user_to_movie_edges.append([user_idx, movie_idx])
        target_ratings.append(rating_value)
    
    if not user_to_movie_edges:
        print("Error: No user->movie edges found. Check data preprocessing.")
        return False
    
    edge_index_adjusted = torch.tensor(user_to_movie_edges, dtype=torch.long).t().contiguous().to(DEVICE)
    target_ratings = torch.tensor(target_ratings, dtype=torch.float).to(DEVICE)
    
    print(f"Adjusted edge indices: {edge_index_adjusted.size(1)} user->movie edges")
    print(f"Target ratings: {target_ratings.size(0)} ratings")
    
    # Training monitoring
    start_time = time.time()
    best_loss = float('inf')
    patience_counter = 0  # Early stopping counter
    
    print(f"\n🚀 Starting GPU-accelerated training for {epochs} epochs...")
    print(f"   Early stopping patience: {patience} epochs")
    if torch.cuda.is_available():
        print(f"   GPU Memory Usage: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass with mixed precision if available
        if scaler is not None:
            with torch.cuda.amp.autocast():
                out = model(x_user, x_movie, edge_index_adjusted)
                loss = F.mse_loss(out.squeeze(), target_ratings)
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard forward and backward pass
            out = model(x_user, x_movie, edge_index_adjusted)
            loss = F.mse_loss(out.squeeze(), target_ratings)
            loss.backward()
            optimizer.step()
        
        # Track best loss and early stopping
        if loss.item() < best_loss - min_delta:
            best_loss = loss.item()
            patience_counter = 0
            # Save best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'user_mapping': mappings['user_id_to_idx'],
                'movie_mapping': mappings['movie_id_to_idx'],
                'movie_mapping_reverse': mappings['movie_idx_to_id'],
                'movie_id_to_title': mappings['movie_id_to_title'],
                'movie_id_to_popularity': mappings['movie_id_to_popularity'],
                'movie_title_to_id': mappings['movie_title_to_id'],
                'dataset_size': mappings['dataset_size'],
                'num_users': mappings['num_users'],
                'num_movies': mappings['num_movies'],
                'hidden_channels': hidden_channels
            }, save_path.replace('.pth', '_best.pth'))
        else:
            patience_counter += 1
        
        # Early stopping check
        if patience_counter >= patience:
            print(f"\n🛑 Early stopping triggered! No improvement for {patience} epochs.")
            print(f"   Best loss: {best_loss:.4f} at epoch {epoch - patience + 1}")
            break
        
        if (epoch + 1) % 10 == 0:
            elapsed_time = time.time() - start_time
            avg_time_per_epoch = elapsed_time / (epoch + 1)
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1024**3
                print(f'Epoch {epoch+1:03d}, Loss: {loss.item():.4f}, Best: {best_loss:.4f}, '
                      f'Patience: {patience_counter}/{patience}, '
                      f'Time/epoch: {avg_time_per_epoch:.2f}s, GPU Memory: {gpu_memory:.2f} GB')
            else:
                print(f'Epoch {epoch+1:03d}, Loss: {loss.item():.4f}, Best: {best_loss:.4f}, '
                      f'Patience: {patience_counter}/{patience}, '
                      f'Time/epoch: {avg_time_per_epoch:.2f}s')
    
    total_training_time = time.time() - start_time
    print(f"\n✅ Training completed!")
    print(f"   Total time: {total_training_time:.2f} seconds")
    print(f"   Average time per epoch: {total_training_time/epochs:.2f} seconds")
    print(f"   Best loss achieved: {best_loss:.4f}")
    
    if torch.cuda.is_available():
        print(f"   Final GPU Memory Usage: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        # Clear GPU cache
        torch.cuda.empty_cache()
    
    # Save model and mappings
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'user_mapping': mappings['user_id_to_idx'],
        'movie_mapping': mappings['movie_id_to_idx'],
        'movie_mapping_reverse': mappings['movie_idx_to_id'],
        'movie_id_to_title': mappings['movie_id_to_title'],
        'movie_id_to_popularity': mappings['movie_id_to_popularity'],
        'movie_title_to_id': mappings['movie_title_to_id'],
        'dataset_size': mappings['dataset_size'],
        'num_users': mappings['num_users'],
        'num_movies': mappings['num_movies'],
        'hidden_channels': hidden_channels
    }, save_path)
    
    print(f"Model saved to: {save_path}")
    
    # Save training summary
    summary = {
        'training_date': pd.Timestamp.now().isoformat(),
        'data_directory': data_dir,
        'num_users': mappings['num_users'],
        'num_movies': mappings['num_movies'],
        'num_ratings': mappings['dataset_size'],
        'hidden_channels': hidden_channels,
        'learning_rate': learning_rate,
        'epochs': epochs,
        'final_loss': loss.item(),
        'best_loss': best_loss,
        'total_training_time': total_training_time,
        'avg_time_per_epoch': total_training_time/epochs,
        'device_used': str(DEVICE),
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
        'model_path': save_path,
        'target_movies': target_movies
    }
    
    summary_path = os.path.join(data_dir, 'training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Training summary saved to: {summary_path}")
    
    return True


def collect_and_train_comprehensive(access_tokens: List[str] = None, 
                                   usernames: List[str] = None,
                                   data_dir: str = 'trakt_data',
                                   target_movies: int = 10000,
                                   use_mixed_precision: bool = True,
                                   **training_kwargs):
    """Collect comprehensive Trakt data and train the model"""
    
    print("Starting comprehensive data collection and training...")
    print(f"Target: {target_movies} movies from every category possible")
    
    # If access tokens provided, collect user data first
    if access_tokens:
        print("Collecting user data from provided access tokens...")
        for access_token, username in zip(access_tokens, usernames or [f"user_{i}" for i in range(len(access_tokens))]):
            data_collector.collect_user_data(access_token, username)
    
    # Train with comprehensive data collection
    success = train_gnn_with_comprehensive_data(
        data_dir=data_dir, 
        target_movies=target_movies,
        use_mixed_precision=use_mixed_precision,
        **training_kwargs
    )
    
    if success:
        print("Comprehensive data collection and training completed successfully!")
    else:
        print("Training failed.")
    
    return success

if __name__ == "__main__":
    # Example usage
    print("Enhanced Trakt GNN Training Script")
    print("=" * 50)
    print("This script will collect 10k movies from every category possible")
    print("and train a comprehensive GNN model.")
    
    # Train with comprehensive data collection
    success = collect_and_train_comprehensive(
        data_dir='trakt_data',
        target_movies=1000000,  # 1M movies
        hidden_channels=64,
        learning_rate=0.001,
        epochs=1000,
        use_mixed_precision=True  # Enable GPU acceleration with mixed precision
    )
    
    if success:
        print("\n🎉 Training completed successfully!")
        print("Your model is now ready with 10k movies from every category!")
    else:
        print("\n❌ Training failed. Please check the logs above.") 