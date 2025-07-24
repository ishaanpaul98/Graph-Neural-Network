import requests
import json
import os
from typing import List, Dict, Optional, Tuple
from urllib.parse import urlencode
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class TraktAPI:
    def __init__(self):
        self.client_id = os.getenv('TRAKT_CLIENT_ID')
        self.client_secret = os.getenv('TRAKT_CLIENT_SECRET')
        
        # Try to get redirect URI from environment, fallback to EC2 domain
        self.redirect_uri = os.getenv('TRAKT_REDIRECT_URI')
        if not self.redirect_uri:
            # Default to EC2 domain if not set
            self.redirect_uri = 'https://api.ishaanpaul.com/auth/callback'
        
        self.base_url = 'https://api.trakt.tv'
        
        print(f"Trakt API initialized with redirect_uri: {self.redirect_uri}")
        
        if not self.client_id or not self.client_secret:
            raise ValueError("TRAKT_CLIENT_ID and TRAKT_CLIENT_SECRET must be set in environment variables")
    
    def get_headers(self, access_token: Optional[str] = None) -> Dict[str, str]:
        """Get headers for Trakt API requests"""
        headers = {
            'Content-Type': 'application/json',
            'trakt-api-version': '2',
            'trakt-api-key': self.client_id,
            'User-Agent': 'MovieRecommendationApp/1.0.0'
        }
        
        if access_token:
            headers['Authorization'] = f'Bearer {access_token}'
        
        return headers
    
    def get_authorization_url(self, state: str = None) -> str:
        """Generate OAuth authorization URL"""
        params = {
            'response_type': 'code',
            'client_id': self.client_id,
            'redirect_uri': self.redirect_uri
        }
        
        if state:
            params['state'] = state
            
        return f"{self.base_url}/oauth/authorize?{urlencode(params)}"
    
    def exchange_code_for_token(self, code: str) -> Dict:
        """Exchange authorization code for access token"""
        url = f"{self.base_url}/oauth/token"
        data = {
            'code': code,
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'redirect_uri': self.redirect_uri,
            'grant_type': 'authorization_code'
        }
        
        response = requests.post(url, json=data, headers=self.get_headers())
        response.raise_for_status()
        return response.json()
    
    def refresh_token(self, refresh_token: str) -> Dict:
        """Refresh access token using refresh token"""
        url = f"{self.base_url}/oauth/token"
        data = {
            'refresh_token': refresh_token,
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'redirect_uri': self.redirect_uri,
            'grant_type': 'refresh_token'
        }
        
        response = requests.post(url, json=data, headers=self.get_headers())
        response.raise_for_status()
        return response.json()
    
    def search_movies_and_shows(self, query: str, limit: int = 10) -> List[Dict]:
        """Search for movies and TV shows"""
        url = f"{self.base_url}/search/movie,show"
        params = {
            'query': query,
            'limit': limit
        }
        
        response = requests.get(url, params=params, headers=self.get_headers())
        response.raise_for_status()
        
        results = []
        for item in response.json():
            if item['type'] == 'movie':
                movie = item['movie']
                results.append({
                    'type': 'movie',
                    'title': movie['title'],
                    'year': movie.get('year'),
                    'ids': movie['ids'],
                    'overview': movie.get('overview', ''),
                    'rating': movie.get('rating'),
                    'votes': movie.get('votes')
                })
            elif item['type'] == 'show':
                show = item['show']
                results.append({
                    'type': 'show',
                    'title': show['title'],
                    'year': show.get('year'),
                    'ids': show['ids'],
                    'overview': show.get('overview', ''),
                    'rating': show.get('rating'),
                    'votes': show.get('votes')
                })
        
        return results
    
    def get_movie_details(self, movie_id: str, id_type: str = 'trakt') -> Dict:
        """Get detailed information about a movie"""
        url = f"{self.base_url}/movies/{id_type}/{movie_id}"
        params = {'extended': 'full'}
        
        response = requests.get(url, params=params, headers=self.get_headers())
        response.raise_for_status()
        return response.json()
    
    def get_show_details(self, show_id: str, id_type: str = 'trakt') -> Dict:
        """Get detailed information about a TV show"""
        url = f"{self.base_url}/shows/{id_type}/{show_id}"
        params = {'extended': 'full'}
        
        response = requests.get(url, params=params, headers=self.get_headers())
        response.raise_for_status()
        return response.json()
    
    def get_user_watched_movies(self, access_token: str, username: str = 'me') -> List[Dict]:
        """Get user's watched movies"""
        url = f"{self.base_url}/users/{username}/watched/movies"
        params = {'extended': 'full'}
        
        response = requests.get(url, params=params, headers=self.get_headers(access_token))
        response.raise_for_status()
        return response.json()
    
    def get_user_watched_shows(self, access_token: str, username: str = 'me') -> List[Dict]:
        """Get user's watched TV shows"""
        url = f"{self.base_url}/users/{username}/watched/shows"
        params = {'extended': 'full'}
        
        response = requests.get(url, params=params, headers=self.get_headers(access_token))
        response.raise_for_status()
        return response.json()
    
    def get_user_ratings(self, access_token: str, username: str = 'me', rating_type: str = 'movies') -> List[Dict]:
        """Get user's ratings"""
        url = f"{self.base_url}/users/{username}/ratings/{rating_type}"
        params = {'extended': 'full'}
        
        response = requests.get(url, params=params, headers=self.get_headers(access_token))
        response.raise_for_status()
        return response.json()
    
    def get_trending_movies(self, limit: int = 10) -> List[Dict]:
        """Get trending movies"""
        url = f"{self.base_url}/movies/trending"
        params = {'limit': limit, 'extended': 'full'}
        
        response = requests.get(url, params=params, headers=self.get_headers())
        response.raise_for_status()
        
        results = []
        for item in response.json():
            movie = item['movie']
            results.append({
                'title': movie['title'],
                'year': movie.get('year'),
                'ids': movie['ids'],
                'overview': movie.get('overview', ''),
                'rating': movie.get('rating'),
                'votes': movie.get('votes'),
                'watchers': item.get('watchers', 0)
            })
        
        return results
    
    def get_popular_movies(self, limit: int = 10) -> List[Dict]:
        """Get popular movies"""
        url = f"{self.base_url}/movies/popular"
        params = {'limit': limit, 'extended': 'full'}
        
        response = requests.get(url, params=params, headers=self.get_headers())
        response.raise_for_status()
        
        results = []
        for movie in response.json():
            results.append({
                'title': movie['title'],
                'year': movie.get('year'),
                'ids': movie['ids'],
                'overview': movie.get('overview', ''),
                'rating': movie.get('rating'),
                'votes': movie.get('votes')
            })
        
        return results
    
    def get_movie_recommendations(self, access_token: str, limit: int = 10) -> List[Dict]:
        """Get personalized movie recommendations for the user"""
        url = f"{self.base_url}/recommendations/movies"
        params = {'limit': limit, 'extended': 'full'}
        
        response = requests.get(url, params=params, headers=self.get_headers(access_token))
        response.raise_for_status()
        
        results = []
        for item in response.json():
            movie = item['movie']
            results.append({
                'title': movie['title'],
                'year': movie.get('year'),
                'ids': movie['ids'],
                'overview': movie.get('overview', ''),
                'rating': movie.get('rating'),
                'votes': movie.get('votes')
            })
        
        return results

# Global instance
trakt_api = TraktAPI() 