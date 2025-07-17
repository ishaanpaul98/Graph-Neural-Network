import pandas as pd
import numpy as np
import json
import os
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
from dotenv import load_dotenv
from trakt_api import trakt_api
from session_manager import session_manager

# Load environment variables from .env file
load_dotenv()

class TraktDataCollector:
    def __init__(self, output_dir: str = 'trakt_data'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def collect_user_data(self, access_token: str, username: str = 'me') -> Dict:
        """Collect comprehensive user data from Trakt"""
        try:
            print(f"Collecting data for user: {username}")
            
            # Collect watched movies
            print("Collecting watched movies...")
            watched_movies = trakt_api.get_user_watched_movies(access_token, username)
            
            # Collect watched shows
            print("Collecting watched shows...")
            watched_shows = trakt_api.get_user_watched_shows(access_token, username)
            
            # Collect movie ratings
            print("Collecting movie ratings...")
            movie_ratings = trakt_api.get_user_ratings(access_token, username, 'movies')
            
            # Collect show ratings
            print("Collecting show ratings...")
            show_ratings = trakt_api.get_user_ratings(access_token, username, 'shows')
            
            # Collect episode ratings
            print("Collecting episode ratings...")
            episode_ratings = trakt_api.get_user_ratings(access_token, username, 'episodes')
            
            user_data = {
                'username': username,
                'collected_at': datetime.now().isoformat(),
                'watched_movies': watched_movies,
                'watched_shows': watched_shows,
                'movie_ratings': movie_ratings,
                'show_ratings': show_ratings,
                'episode_ratings': episode_ratings
            }
            
            # Save user data
            filename = f"{self.output_dir}/user_{username}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(user_data, f, indent=2)
            
            print(f"User data saved to: {filename}")
            return user_data
            
        except Exception as e:
            print(f"Error collecting user data: {e}")
            return {}
    
    def collect_popular_movies(self, limit: int = 1000) -> List[Dict]:
        """Collect popular movies from Trakt"""
        try:
            print(f"Collecting {limit} popular movies...")
            movies = []
            
            # Collect in batches of 100
            batch_size = 100
            for i in range(0, limit, batch_size):
                batch_limit = min(batch_size, limit - i)
                batch = trakt_api.get_popular_movies(batch_limit)
                movies.extend(batch)
                print(f"Collected {len(movies)} movies so far...")
            
            # Save popular movies
            filename = f"{self.output_dir}/popular_movies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(movies, f, indent=2)
            
            print(f"Popular movies saved to: {filename}")
            return movies
            
        except Exception as e:
            print(f"Error collecting popular movies: {e}")
            return []
    
    def collect_trending_movies(self, limit: int = 100) -> List[Dict]:
        """Collect trending movies from Trakt"""
        try:
            print(f"Collecting {limit} trending movies...")
            movies = trakt_api.get_trending_movies(limit)
            
            # Save trending movies
            filename = f"{self.output_dir}/trending_movies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(movies, f, indent=2)
            
            print(f"Trending movies saved to: {filename}")
            return movies
            
        except Exception as e:
            print(f"Error collecting trending movies: {e}")
            return []
    
    def search_and_collect_movies(self, search_terms: List[str], limit_per_term: int = 50) -> List[Dict]:
        """Search for movies using terms and collect their data"""
        try:
            print(f"Searching and collecting movies for {len(search_terms)} terms...")
            all_movies = []
            
            for term in search_terms:
                print(f"Searching for: {term}")
                results = trakt_api.search_movies_and_shows(term, limit_per_term)
                
                # Filter for movies only
                movies = [item for item in results if item['type'] == 'movie']
                all_movies.extend(movies)
                
                print(f"Found {len(movies)} movies for '{term}'")
            
            # Remove duplicates based on Trakt ID
            unique_movies = {}
            for movie in all_movies:
                trakt_id = movie['ids'].get('trakt')
                if trakt_id and trakt_id not in unique_movies:
                    unique_movies[trakt_id] = movie
            
            unique_movies_list = list(unique_movies.values())
            
            # Save searched movies
            filename = f"{self.output_dir}/searched_movies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(unique_movies_list, f, indent=2)
            
            print(f"Searched movies saved to: {filename}")
            return unique_movies_list
            
        except Exception as e:
            print(f"Error searching and collecting movies: {e}")
            return []
    
    def create_training_dataset(self, user_data_files: List[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create training dataset from collected Trakt data"""
        try:
            print("Creating training dataset...")
            
            # If no files specified, use all files in output directory
            if user_data_files is None:
                user_data_files = [f for f in os.listdir(self.output_dir) if f.startswith('user_') and f.endswith('.json')]
                user_data_files = [os.path.join(self.output_dir, f) for f in user_data_files]
            
            users = []
            movies = []
            ratings = []
            
            movie_id_counter = 0
            user_id_counter = 0
            movie_id_map = {}
            user_id_map = {}
            
            for file_path in user_data_files:
                try:
                    with open(file_path, 'r') as f:
                        user_data = json.load(f)
                    
                    username = user_data['username']
                    
                    # Map username to user ID
                    if username not in user_id_map:
                        user_id_map[username] = user_id_counter
                        users.append({
                            'user_id': user_id_counter,
                            'username': username
                        })
                        user_id_counter += 1
                    
                    user_id = user_id_map[username]
                    
                    # Process movie ratings
                    for rating_data in user_data.get('movie_ratings', []):
                        movie_info = rating_data['movie']
                        movie_title = movie_info['title']
                        trakt_id = movie_info['ids'].get('trakt')
                        
                        # Map movie to movie ID
                        if trakt_id not in movie_id_map:
                            movie_id_map[trakt_id] = movie_id_counter
                            movies.append({
                                'movie_id': movie_id_counter,
                                'trakt_id': trakt_id,
                                'title': movie_title,
                                'year': movie_info.get('year'),
                                'rating': movie_info.get('rating', 0),
                                'votes': movie_info.get('votes', 0)
                            })
                            movie_id_counter += 1
                        
                        movie_id = movie_id_map[trakt_id]
                        
                        # Add rating
                        ratings.append({
                            'user_id': user_id,
                            'movie_id': movie_id,
                            'rating': rating_data['rating'],
                            'rated_at': rating_data.get('rated_at')
                        })
                    
                    # Process watched movies (treat as implicit ratings)
                    for watch_data in user_data.get('watched_movies', []):
                        movie_info = watch_data['movie']
                        trakt_id = movie_info['ids'].get('trakt')
                        
                        if trakt_id not in movie_id_map:
                            movie_id_map[trakt_id] = movie_id_counter
                            movies.append({
                                'movie_id': movie_id_counter,
                                'trakt_id': trakt_id,
                                'title': movie_info['title'],
                                'year': movie_info.get('year'),
                                'rating': movie_info.get('rating', 0),
                                'votes': movie_info.get('votes', 0)
                            })
                            movie_id_counter += 1
                        
                        movie_id = movie_id_map[trakt_id]
                        
                        # Add implicit rating (watched = positive interaction)
                        ratings.append({
                            'user_id': user_id,
                            'movie_id': movie_id,
                            'rating': 1,  # Implicit positive rating
                            'rated_at': watch_data.get('last_watched_at')
                        })
                
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")
                    continue
            
            # Create DataFrames
            users_df = pd.DataFrame(users)
            movies_df = pd.DataFrame(movies)
            ratings_df = pd.DataFrame(ratings)
            
            # Save datasets
            users_df.to_csv(f"{self.output_dir}/users.csv", index=False)
            movies_df.to_csv(f"{self.output_dir}/movies.csv", index=False)
            ratings_df.to_csv(f"{self.output_dir}/ratings.csv", index=False)
            
            print(f"Training dataset created:")
            print(f"  Users: {len(users_df)}")
            print(f"  Movies: {len(movies_df)}")
            print(f"  Ratings: {len(ratings_df)}")
            
            return users_df, movies_df, ratings_df
            
        except Exception as e:
            print(f"Error creating training dataset: {e}")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    def collect_all_data_for_training(self, access_tokens: List[str], usernames: List[str] = None) -> str:
        """Collect all necessary data for training the GNN model"""
        try:
            print("Starting comprehensive data collection for training...")
            
            if usernames is None:
                usernames = [f"user_{i}" for i in range(len(access_tokens))]
            
            # Collect user data
            for access_token, username in zip(access_tokens, usernames):
                self.collect_user_data(access_token, username)
            
            # Collect popular movies
            self.collect_popular_movies(1000)
            
            # Collect trending movies
            self.collect_trending_movies(100)
            
            # Search for additional movies using common terms
            search_terms = [
                'action', 'comedy', 'drama', 'horror', 'romance', 'sci-fi',
                'thriller', 'adventure', 'fantasy', 'mystery', 'crime',
                'documentary', 'animation', 'family', 'war', 'western'
            ]
            self.search_and_collect_movies(search_terms, 30)
            
            # Create training dataset
            users_df, movies_df, ratings_df = self.create_training_dataset()
            
            # Save dataset summary
            summary = {
                'collection_date': datetime.now().isoformat(),
                'num_users': len(users_df),
                'num_movies': len(movies_df),
                'num_ratings': len(ratings_df),
                'output_directory': self.output_dir
            }
            
            summary_file = f"{self.output_dir}/dataset_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"Data collection completed. Summary saved to: {summary_file}")
            return self.output_dir
            
        except Exception as e:
            print(f"Error in comprehensive data collection: {e}")
            return ""

# Global data collector instance
data_collector = TraktDataCollector() 