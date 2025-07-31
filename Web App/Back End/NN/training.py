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
import argparse

def load_movielens_data(dataset_size='100k', sample_size=None):
    """
    Load and preprocess MovieLens dataset
    Args:
        dataset_size: Size of the dataset ('100k', '1m', '10m', '20m', '32m')
        sample_size: Number of ratings to sample (None for full dataset)
    """
    # Initialize downloader
    downloader = MovieLensDownloader(dataset_size)
    
    # Download dataset first
    print("Downloading dataset...")
    dataset_path = downloader.download()
    
    # Get dataset info
    print("Loading dataset info...")
    data = downloader.get_dataset_info()
    if data is None:
        raise RuntimeError("Failed to load dataset info. Please check if the download was successful.")
    
    ratings_df = data['ratings']
    movies_df = data['movies']
    
    # For large datasets, sample users and movies first to ensure consistent dimensions
    if dataset_size in ['10m', '20m', '32m']:
        if sample_size is None:
            sample_size = 500000  # Default sample size for large datasets
        
        print(f"Sampling users and movies to ensure consistent dimensions...")
        
        # Calculate sampling percentages based on dataset size
        if dataset_size == '10m':
            user_sample_percent = 0.3  # Sample 30% of users
            movie_sample_percent = 0.15  # Sample 15% of movies
        elif dataset_size == '20m':
            user_sample_percent = 0.2  # Sample 20% of users
            movie_sample_percent = 0.1  # Sample 10% of movies
        else:  # 32m
            user_sample_percent = 0.15  # Sample 15% of users
            movie_sample_percent = 0.08  # Sample 8% of movies
        
        # Sample users and movies based on percentages
        num_users_to_sample = max(1, int(len(ratings_df['userId'].unique()) * user_sample_percent))
        num_movies_to_sample = max(1, int(len(ratings_df['movieId'].unique()) * movie_sample_percent))
        
        sampled_users = ratings_df['userId'].sample(n=num_users_to_sample, random_state=42)
        sampled_movies = ratings_df['movieId'].sample(n=num_movies_to_sample, random_state=42)
        
        # Filter ratings to include only sampled users and movies
        ratings_df = ratings_df[
            ratings_df['userId'].isin(sampled_users) & 
            ratings_df['movieId'].isin(sampled_movies)
        ]
        
        # Further sample ratings if needed
        if len(ratings_df) > sample_size:
            ratings_df = ratings_df.sample(n=sample_size, random_state=42)
    
    # Create user and movie mappings
    user_mapping = {user_id: idx for idx, user_id in enumerate(sorted(ratings_df['userId'].unique()))}
    movie_mapping = {movie_id: idx for idx, movie_id in enumerate(sorted(ratings_df['movieId'].unique()))}
    
    # Convert ratings to edge indices
    edge_index = torch.tensor([
        [user_mapping[user] for user in ratings_df['userId']],
        [movie_mapping[movie] for movie in ratings_df['movieId']]
    ], dtype=torch.long)
    
    # Create node features
    num_users = len(user_mapping)
    num_movies = len(movie_mapping)
    
    # Create sparse user features (one-hot encoding)
    user_indices = torch.arange(num_users)
    user_features = torch.sparse_coo_tensor(
        torch.stack([user_indices, user_indices]),
        torch.ones(num_users),
        size=(num_users, num_users)
    )
    
    # Create sparse movie features (one-hot encoding)
    movie_indices = torch.arange(num_movies)
    movie_features = torch.sparse_coo_tensor(
        torch.stack([movie_indices, movie_indices]),
        torch.ones(num_movies),
        size=(num_movies, num_movies)
    )
    
    # Create edge features (ratings)
    edge_attr = torch.tensor(ratings_df['rating'].values, dtype=torch.float).view(-1, 1)
    
    # Create PyTorch Geometric Data object
    data = Data(
        x_user=user_features,  # Sparse user features
        x_movie=movie_features,  # Movie features
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_users=num_users,
        num_movies=num_movies,
        num_nodes=num_users + num_movies
    )
    
    # Build movie_id_to_title and movie_id_to_popularity for only movies in movie_mapping
    movie_id_to_title = {int(row['movieId']): row['title'] for _, row in movies_df.iterrows() if int(row['movieId']) in movie_mapping}
    # Calculate popularity (number of ratings) for each movie
    movie_popularity = ratings_df.groupby('movieId').size().to_dict()
    movie_id_to_popularity = {int(mid): int(movie_popularity.get(mid, 0)) for mid in movie_mapping}
    
    return data, user_mapping, movie_mapping, movie_id_to_title, movie_id_to_popularity

def train_model(data, num_epochs=50, batch_size=16, learning_rate=0.01, gradient_accumulation_steps=8):
    """
    Train the MPGNN model with gradient accumulation
    Args:
        data: PyTorch Geometric Data object
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        gradient_accumulation_steps: Number of steps to accumulate gradients
    """
    # Split data into train and test sets
    train_mask, test_mask = train_test_split(
        range(data.edge_index.size(1)),
        test_size=0.2,
        random_state=42
    )
    
    # Create data loaders with smaller batch size
    train_loader = DataLoader(
        [data],
        batch_size=batch_size,
        shuffle=True
    )
    
    # Initialize model with correct input sizes
    model = MPGNN(
        num_user_features=data.x_user.size(1),
        num_movie_features=data.x_movie.size(1),
        hidden_channels=8,  # Further reduced hidden channels
        num_classes=1,
        num_layers=2
    )
    
    # Initialize optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()
    
    # Training loop with gradient accumulation
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        optimizer.zero_grad()
        
        for i, batch in enumerate(train_loader):
            # Forward pass
            out = model(batch.x_user, batch.x_movie, batch.edge_index)
            
            # Calculate loss
            loss = criterion(out, batch.edge_attr)
            loss = loss / gradient_accumulation_steps  # Normalize loss
            
            # Backward pass
            loss.backward()
            
            # Accumulate gradients
            if (i + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * gradient_accumulation_steps
        
        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}')
    
    return model

def main():
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Train MPGNN model on MovieLens dataset')
    parser.add_argument('--dataset', type=str, choices=['100k', '1m', '10m', '20m'], 
                      help='MovieLens dataset size to use')
    parser.add_argument('--epochs', type=int, default=50, 
                      help='Number of training epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=16,
                      help='Batch size for training (default: 16)')
    parser.add_argument('--learning-rate', type=float, default=0.01,
                      help='Learning rate for optimizer (default: 0.01)')
    parser.add_argument('--gradient-steps', type=int, default=8,
                      help='Number of gradient accumulation steps (default: 8)')
    parser.add_argument('--sample-size', type=int, default=500000,
                      help='Number of ratings to sample for large datasets (default: 500000)')
    args = parser.parse_args()
    
    # If dataset size not provided, prompt user
    if args.dataset is None:
        print("Available MovieLens dataset sizes:")
        print("1. 100k (small)")
        print("2. 1M (medium)")
        print("3. 10M (large)")
        print("4. 20M (full)")
        
        while True:
            try:
                choice = input("\nEnter your choice (1-4): ")
                size_map = {
                    '1': '100k',
                    '2': '1m',
                    '3': '10m',
                    '4': '20m'
                }
                if choice in size_map:
                    dataset_size = size_map[choice]
                    break
                else:
                    print("Invalid choice. Please enter a number between 1 and 4.")
            except Exception as e:
                print(f"Error: {str(e)}")
                print("Please try again.")
    else:
        dataset_size = args.dataset
    
    print(f"\nSelected dataset size: {dataset_size}")
    print(f"Number of epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Gradient accumulation steps: {args.gradient_steps}")
    
    # Set sample size for large datasets
    sample_size = None
    if dataset_size in ['10m', '20m', '32m']:
        sample_size = args.sample_size
        print(f"Sample size for large dataset: {sample_size}")
    
    # Load data
    print("\nLoading MovieLens dataset...")
    data, user_mapping, movie_mapping, movie_id_to_title, movie_id_to_popularity = load_movielens_data(dataset_size, sample_size)
    
    # Train model
    print("\nTraining MPGNN model...")
    model = train_model(
        data,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_steps
    )
    
    # Save model and mappings
    save_data = {
        'model_state_dict': model.state_dict(),
        'user_mapping': {int(k): int(v) for k, v in user_mapping.items()},  # Convert to int
        'movie_mapping': {int(k): int(v) for k, v in movie_mapping.items()},  # Convert to int
        'movie_mapping_reverse': {int(v): int(k) for k, v in movie_mapping.items()},  # Add reverse mapping
        'dataset_size': dataset_size,
        'num_users': int(len(user_mapping)),  # Convert to int
        'num_movies': int(len(movie_mapping)),  # Convert to int
        'movie_id_to_title': movie_id_to_title,
        'movie_id_to_popularity': movie_id_to_popularity
    }
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)

    # Save the model and mappings
    torch.save(save_data, 'models/mpgnn_model.pth')
    print("\nModel and mappings saved to 'models/mpgnn_model.pth'")

if __name__ == "__main__":
    main()
