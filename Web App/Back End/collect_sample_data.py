#!/usr/bin/env python3
"""
Sample Data Collection Script

This script collects sample data from Trakt to test the training pipeline.
It will collect trending movies, popular movies, and some search results.
"""

import os
import json
import random
from datetime import datetime, timedelta
from trakt_data_collector import data_collector

def collect_sample_data():
    """Collect sample data for testing"""
    print("🎬 Collecting Sample Trakt Data")
    print("=" * 50)
    
    try:
        # Create trakt_data directory if it doesn't exist
        os.makedirs('trakt_data', exist_ok=True)
        
        print("📊 Collecting trending movies...")
        trending_movies = data_collector.collect_trending_movies(150)
        print(f"✅ Collected {len(trending_movies)} trending movies")
        
        print("\n📊 Collecting popular movies...")
        popular_movies = data_collector.collect_popular_movies(100)
        print(f"✅ Collected {len(popular_movies)} popular movies")
        
        print("\n🔍 Searching for movies by genre...")
        search_terms = [
            'action', 'comedy', 'drama', 'horror', 'romance', 
            'sci-fi', 'thriller', 'adventure', 'fantasy'
        ]
        
        searched_movies = data_collector.search_and_collect_movies(search_terms, 50)
        print(f"✅ Collected {len(searched_movies)} movies from search")
        
        # Combine all movies
        all_movies = trending_movies + popular_movies + searched_movies
        
        # Remove duplicates based on Trakt ID
        unique_movies = {}
        for movie in all_movies:
            trakt_id = movie['ids'].get('trakt')
            if trakt_id and trakt_id not in unique_movies:
                unique_movies[trakt_id] = movie
        
        unique_movies_list = list(unique_movies.values())
        print(f"\n📈 Total unique movies collected: {len(unique_movies_list)}")
        
        # Create sample user data with realistic ratings
        print("\n👥 Creating sample user data with ratings...")
        create_sample_user_data(unique_movies_list)
        
        print("\n📈 Creating training dataset...")
        users_df, movies_df, ratings_df = data_collector.create_training_dataset()
        
        if len(users_df) > 0 and len(movies_df) > 0 and len(ratings_df) > 0:
            print(f"✅ Training dataset created successfully!")
            print(f"   Users: {len(users_df)}")
            print(f"   Movies: {len(movies_df)}")
            print(f"   Ratings: {len(ratings_df)}")
            
            # Save dataset summary
            summary = {
                'collection_date': datetime.now().isoformat(),
                'num_users': len(users_df),
                'num_movies': len(movies_df),
                'num_ratings': len(ratings_df),
                'trending_movies': len(trending_movies),
                'popular_movies': len(popular_movies),
                'searched_movies': len(searched_movies),
                'unique_movies': len(unique_movies_list),
                'search_terms': search_terms
            }
            
            summary_path = 'Dataset/trakt_data/sample_data_summary.json'
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"\n📄 Summary saved to: {summary_path}")
            print("\n🎉 Sample data collection completed!")
            print("\nNext steps:")
            print("1. Run: python train_with_trakt.py")
            print("2. The model will train on this sample data")
            
        else:
            print("❌ Failed to create training dataset")
            
    except Exception as e:
        print(f"❌ Error collecting sample data: {e}")
        import traceback
        traceback.print_exc()

def create_sample_user_data(movies_list):
    """Create sample user data with realistic ratings"""
    try:
        # Create sample users with different preferences
        sample_users = [
            {
                'username': 'action_fan',
                'preferences': ['action', 'thriller', 'adventure'],
                'rating_range': (7, 10)
            },
            {
                'username': 'comedy_lover',
                'preferences': ['comedy', 'romance', 'family'],
                'rating_range': (6, 9)
            },
            {
                'username': 'drama_critic',
                'preferences': ['drama', 'mystery', 'crime'],
                'rating_range': (5, 10)
            },
            {
                'username': 'horror_buff',
                'preferences': ['horror', 'thriller', 'sci-fi'],
                'rating_range': (6, 9)
            },
            {
                'username': 'general_viewer',
                'preferences': ['action', 'comedy', 'drama', 'romance'],
                'rating_range': (4, 8)
            }
        ]
        
        for user_info in sample_users:
            username = user_info['username']
            preferences = user_info['preferences']
            rating_range = user_info['rating_range']
            
            # Select movies that match user preferences
            user_movies = []
            for movie in movies_list:
                # Check if movie title contains any preference keywords
                title_lower = movie['title'].lower()
                if any(pref in title_lower for pref in preferences):
                    user_movies.append(movie)
            
            # If not enough preference-based movies, add some random ones
            if len(user_movies) < 20:
                remaining_movies = [m for m in movies_list if m not in user_movies]
                user_movies.extend(random.sample(remaining_movies, min(20 - len(user_movies), len(remaining_movies))))
            
            # Create movie ratings
            movie_ratings = []
            watched_movies = []
            
            for movie in user_movies[:30]:  # Limit to 30 movies per user
                # Determine if user watched this movie
                watched = random.random() < 0.8  # 80% chance of watching
                
                if watched:
                    # Generate rating based on preferences
                    if any(pref in movie['title'].lower() for pref in preferences):
                        rating = random.randint(rating_range[0], rating_range[1])
                    else:
                        rating = random.randint(rating_range[0] - 2, rating_range[1] - 1)
                    
                    # Add to ratings
                    movie_ratings.append({
                        'rating': rating,
                        'rated_at': (datetime.now() - timedelta(days=random.randint(1, 365))).isoformat(),
                        'movie': {
                            'title': movie['title'],
                            'year': movie.get('year'),
                            'ids': movie['ids'],
                            'rating': movie.get('rating', 0),
                            'votes': movie.get('votes', 0)
                        }
                    })
                    
                    # Add to watched movies
                    watched_movies.append({
                        'last_watched_at': (datetime.now() - timedelta(days=random.randint(1, 365))).isoformat(),
                        'movie': {
                            'title': movie['title'],
                            'year': movie.get('year'),
                            'ids': movie['ids'],
                            'rating': movie.get('rating', 0),
                            'votes': movie.get('votes', 0)
                        }
                    })
            
            # Create user data
            user_data = {
                'username': username,
                'collected_at': datetime.now().isoformat(),
                'watched_movies': watched_movies,
                'watched_shows': [],
                'movie_ratings': movie_ratings,
                'show_ratings': [],
                'episode_ratings': []
            }
            
            # Save user data
            filename = f'trakt_data/user_{username}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            with open(filename, 'w') as f:
                json.dump(user_data, f, indent=2)
            print(f"   Created: {filename} with {len(movie_ratings)} ratings")
            
    except Exception as e:
        print(f"Error creating sample user data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    collect_sample_data() 