import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
import pandas as pd
import numpy as np
import os
import json
from typing import List, Dict, Tuple
from dotenv import load_dotenv
from NN.mpgnn import MPGNN
from trakt_data_collector import data_collector

# Load environment variables from .env file
load_dotenv()

class TraktDataProcessor:
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

def train_gnn_with_trakt_data(data_dir: str = 'trakt_data', 
                             hidden_channels: int = 64,
                             learning_rate: float = 0.001,
                             epochs: int = 100,
                             batch_size: int = 32,
                             save_path: str = 'models/trakt_gnn_model.pth'):
    """Train GNN model with Trakt data"""
    
    print("Starting GNN training with Trakt data...")
    
    # Initialize data processor
    processor = TraktDataProcessor(data_dir)
    
    # Load data
    if not processor.load_data():
        print("Failed to load data. Exiting.")
        return False
    
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
    
    model = MPGNN(
        num_user_features=16,  # Fixed feature dimension for users
        num_movie_features=16,  # Fixed feature dimension for movies
        hidden_channels=hidden_channels,
        num_classes=1
    )
    
    # Set up optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    model.train()
    print(f"\nStarting training for {epochs} epochs...")
    
    # Split node features for user and movie nodes
    num_users = mappings['num_users']
    x_user = data.x[:num_users]
    x_movie = data.x[num_users:]
    
    # Create proper edge indices for the model
    # The model expects edge_index[0] = user indices, edge_index[1] = movie indices
    # We need to filter only user->movie edges and adjust movie indices
    user_to_movie_edges = []
    for i in range(data.edge_index.size(1)):
        user_idx = data.edge_index[0, i]
        movie_idx = data.edge_index[1, i]
        
        # Only keep edges where user_idx < num_users and movie_idx >= num_users
        if user_idx < num_users and movie_idx >= num_users:
            movie_idx_adjusted = movie_idx - num_users
            user_to_movie_edges.append([user_idx, movie_idx_adjusted])
    
    edge_index_adjusted = torch.tensor(user_to_movie_edges, dtype=torch.long).t().contiguous()
    
    print(f"Adjusted edge indices: {edge_index_adjusted.size(1)} user->movie edges")
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass
        out = model(x_user, x_movie, edge_index_adjusted)
        
        # Calculate loss (MSE for rating prediction)
        loss = F.mse_loss(out.squeeze(), data.y[:edge_index_adjusted.size(1), 2])
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1:03d}, Loss: {loss.item():.4f}')
    
    print("Training completed!")
    
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
        'model_path': save_path
    }
    
    summary_path = os.path.join(data_dir, 'training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Training summary saved to: {summary_path}")
    
    return True

def collect_and_train(access_tokens: List[str], 
                     usernames: List[str] = None,
                     data_dir: str = 'trakt_data',
                     **training_kwargs):
    """Collect Trakt data and train the model"""
    
    print("Starting comprehensive data collection and training...")
    
    # Collect data
    output_dir = data_collector.collect_all_data_for_training(access_tokens, usernames)
    
    if not output_dir:
        print("Data collection failed. Exiting.")
        return False
    
    # Train model
    success = train_gnn_with_trakt_data(data_dir, **training_kwargs)
    
    if success:
        print("Data collection and training completed successfully!")
    else:
        print("Training failed.")
    
    return success

if __name__ == "__main__":
    # Example usage
    print("Trakt GNN Training Script")
    print("=" * 50)
    
    # Check if data exists
    if os.path.exists('trakt_data'):
        print("Found existing data. Training with existing data...")
        success = train_gnn_with_trakt_data(
            data_dir='trakt_data',
            hidden_channels=64,
            learning_rate=0.001,
            epochs=50
        )
    else:
        print("No existing data found.")
        print("To collect data and train, use:")
        print("collect_and_train(['your_access_token'], ['username'])") 