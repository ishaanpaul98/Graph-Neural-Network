import os
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from NN.mpgnn import MPGNN
from Dataset.download_movielens import MovieLensDownloader
from pathlib import Path

def load_movielens_data(dataset_size='100k'):
    """
    Load and preprocess MovieLens dataset
    """
    # Initialize downloader
    downloader = MovieLensDownloader(dataset_size)
    
    # This will only download if not present
    dataset_path = downloader.download()
    
    # Get dataset info
    data = downloader.get_dataset_info()
    
    ratings_df = data['ratings']
    movies_df = data['movies']
    
    # Create user and movie mappings
    user_mapping = {user_id: idx for idx, user_id in enumerate(ratings_df['userId'].unique())}
    movie_mapping = {movie_id: idx for idx, movie_id in enumerate(ratings_df['movieId'].unique())}
    
    # Convert ratings to edge indices
    edge_index = torch.tensor([
        [user_mapping[user] for user in ratings_df['userId']],
        [movie_mapping[movie] for movie in ratings_df['movieId']]
    ], dtype=torch.long)
    
    # Create node features
    num_users = len(user_mapping)
    num_movies = len(movie_mapping)
    
    # Create user features (one-hot encoding)
    user_features = torch.eye(num_users)
    
    # Create movie features (using genres for ml-100k)
    if dataset_size == '100k':
        # Get genre columns (last 19 columns)
        genre_columns = movies_df.columns[-19:]
        movie_features = torch.tensor(movies_df[genre_columns].values, dtype=torch.float)
    else:
        # For other datasets, use one-hot encoding
        movie_features = torch.eye(num_movies)
    
    # Create edge features (ratings)
    edge_attr = torch.tensor(ratings_df['rating'].values, dtype=torch.float).view(-1, 1)
    
    # Create PyTorch Geometric Data object
    data = Data(
        x_user=user_features,  # User features
        x_movie=movie_features,  # Movie features
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_users=num_users,
        num_movies=num_movies
    )
    
    return data, user_mapping, movie_mapping

def train_model(data, num_epochs=50, batch_size=256, learning_rate=0.01):
    """
    Train the MPGNN model
    """
    # Split data into train and test sets
    train_mask, test_mask = train_test_split(
        range(data.edge_index.size(1)),
        test_size=0.2,
        random_state=42
    )
    
    # Create data loaders
    train_loader = DataLoader(
        [data],
        batch_size=batch_size,
        shuffle=True
    )
    
    # Initialize model with correct input sizes
    model = MPGNN(
        num_user_features=data.x_user.size(1),
        num_movie_features=data.x_movie.size(1),
        hidden_channels=64,
        num_classes=1,
        num_layers=2
    )
    
    # Initialize optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            
            # Forward pass with separate user and movie features
            out = model(batch.x_user, batch.x_movie, batch.edge_index)
            
            # Calculate loss
            loss = criterion(out, batch.edge_attr)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}')
    
    return model

def main():
    # Load data
    print("Loading MovieLens dataset...")
    data, user_mapping, movie_mapping = load_movielens_data('100k')
    
    # Train model
    print("\nTraining MPGNN model...")
    model = train_model(data)
    
    # Save model
    torch.save(model.state_dict(), 'models/mpgnn_model.pth')
    print("\nModel saved to 'models/mpgnn_model.pth'")

if __name__ == "__main__":
    main()
