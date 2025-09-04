import torch
import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import random

class EnhancedRecommendationEngine:
    """
    Enhanced recommendation engine with multiple ranking strategies
    """
    
    def __init__(self, model, movie_embeddings, movie_features, movie_popularity):
        self.model = model
        self.movie_embeddings = movie_embeddings
        self.movie_features = movie_features
        self.movie_popularity = movie_popularity
        self.device = next(self.model.parameters()).device
        
        # Validate data consistency
        self._validate_data()
    
    def _validate_data(self):
        """Validate that the data is consistent and usable"""
        if not self.movie_embeddings:
            print("Warning: movie_embeddings is empty")
            return
        
        if not self.movie_features:
            print("Warning: movie_features is empty")
            return
        
        if not self.movie_popularity:
            print("Warning: movie_popularity is empty")
            return
        
        # Check for data consistency
        movie_ids = set(self.movie_embeddings.keys())
        feature_ids = set(self.movie_features.keys())
        popularity_ids = set(self.movie_popularity.keys())
        
        print(f"Data validation: {len(movie_ids)} movies in embeddings, {len(feature_ids)} in features, {len(popularity_ids)} in popularity")
        
        # Check for missing data
        missing_features = movie_ids - feature_ids
        missing_popularity = movie_ids - popularity_ids
        
        if missing_features:
            print(f"Warning: {len(missing_features)} movies missing from features")
        
        if missing_popularity:
            print(f"Warning: {len(missing_popularity)} movies missing from popularity data")
        
        # Check for extra data
        extra_features = feature_ids - movie_ids
        extra_popularity = popularity_ids - movie_ids
        
        if extra_features:
            print(f"Info: {len(extra_features)} extra movies in features not in embeddings")
        
        if extra_popularity:
            print(f"Info: {len(extra_popularity)} extra movies in popularity not in embeddings")
    
    def get_diverse_recommendations(self, user_movies: List[str], movie_title_to_id: Dict, 
                                   num_recommendations: int = 15, diversity_weight: float = 0.3):
        """
        Get diverse recommendations using multiple strategies
        """
        # Get base predictions
        base_predictions = self._get_base_predictions(user_movies, movie_title_to_id)
        
        # Apply different ranking strategies
        strategies = {
            'collaborative': self._collaborative_filtering,
            'content_based': self._content_based_filtering,
            'popularity': self._popularity_based,
            #'diversity': self._diversity_based,
            'novelty': self._novelty_based
        }
        
        # Combine strategies
        final_scores = self._combine_strategies(base_predictions, strategies, user_movies, movie_title_to_id)
        print("Final scores:", final_scores)
        
        # Apply diversity penalty
        #final_scores = self._apply_diversity_penalty(final_scores, diversity_weight)
        #print("Final scores after diversity penalty:", final_scores)
        
        # Get top recommendations
        top_indices = torch.topk(final_scores, num_recommendations).indices
        print("Top indices:", top_indices)
        
        # Convert indices to movie titles
        movie_id_to_title = {v: k for k, v in movie_title_to_id.items()}
        recommended_movies = []
        
        for pred_idx in top_indices:
            if pred_idx < len(movie_title_to_id):
                movie_id = list(movie_title_to_id.values())[pred_idx]
                movie_title = movie_id_to_title.get(movie_id)
                if movie_title and movie_title not in user_movies:  # Exclude input movies
                    recommended_movies.append(movie_title)
        
        return recommended_movies
    
    def prepare_model_input(self, movie_ids: List[int]) -> tuple:
        """
        Prepare input tensors for the model
        """
        # Get total number of movies from the movie embeddings
        num_movies = len(self.movie_embeddings)
        
        # Create user features (16-dimensional random features)
        user_features = torch.randn(1, 16, device=self.device)  # Single user with 16 features
        
        # Create movie features (16-dimensional random features for all movies)
        movie_features = torch.randn(num_movies, 16, device=self.device)
        
        # Create edge indices for input movies (user -> movie edges)
        edge_indices = []
        for movie_id in movie_ids:
            # Map movie_id to index (assuming movie_ids are sequential or use the ID directly)
            movie_idx = movie_id if isinstance(movie_id, int) else hash(movie_id) % num_movies
            edge_indices.append([0, movie_idx])  # user_idx=0, movie_idx
        
        if not edge_indices:
            # Fallback: create edges to first few movies
            edge_indices = [[0, i] for i in range(min(3, num_movies))]
        
        edge_index = torch.tensor(edge_indices, dtype=torch.long, device=self.device).t()
        
        return user_features, movie_features, edge_index

    def _get_base_predictions(self, user_movies: List[str], movie_title_to_id: Dict) -> torch.Tensor:
        """Get base predictions from the GNN model"""
        user_features, movie_features, edge_index = self.prepare_model_input(movie_title_to_id.values())
        print("Getting base predictions...")
        # Get predictions for all movies
        with torch.no_grad():
            # Get total number of movies from the movie title to ID mapping
            num_movies = len(movie_title_to_id)
            # Create edge indices for all possible movie recommendations
            all_movie_indices = torch.arange(num_movies, device=self.device)
            all_edge_index = torch.tensor([
                [0] * num_movies,  # Source nodes (user)
                all_movie_indices  # Target nodes (all movies)
            ], dtype=torch.long, device=self.device)
            
            print(f"All edge index shape: {all_edge_index.shape}")
            print(f"All edge index range: {all_edge_index.min()} to {all_edge_index.max()}")
            
            # Get predictions
            predictions = self.model.predict(
                user_features,  # Already has correct shape (1, 16)
                movie_features,
                all_edge_index
            )
            
            print("Recommendations:", predictions)
            return predictions.squeeze()
    
    def _collaborative_filtering(self, base_scores: torch.Tensor, user_movies: List[str], 
                               movie_title_to_id: Dict) -> torch.Tensor:
        """Collaborative filtering based on similar users"""
        # Find movies similar to user's preferences
        user_movie_ids = [movie_title_to_id.get(movie) for movie in user_movies if movie in movie_title_to_id]
        
        if not user_movie_ids:
            return base_scores
        
        # Calculate similarity to user's movies
        user_embeddings = torch.stack([self.movie_embeddings[mid] for mid in user_movie_ids])
        all_embeddings = torch.stack(list(self.movie_embeddings.values()))
        
        similarities = cosine_similarity(user_embeddings, all_embeddings)
        avg_similarity = np.mean(similarities, axis=0)
        
        # Ensure tensor has same size as base_scores
        if len(avg_similarity) != len(base_scores):
            # Pad or truncate to match base_scores size
            if len(avg_similarity) > len(base_scores):
                avg_similarity = avg_similarity[:len(base_scores)]
            else:
                avg_similarity = np.pad(avg_similarity, (0, len(base_scores) - len(avg_similarity)), 'constant')
        
        return torch.tensor(avg_similarity, dtype=torch.float32)
    
    def _content_based_filtering(self, base_scores: torch.Tensor, user_movies: List[str], 
                               movie_title_to_id: Dict) -> torch.Tensor:
        """Content-based filtering using movie features"""
        user_movie_ids = [movie_title_to_id.get(movie) for movie in user_movies if movie in movie_title_to_id]
        
        if not user_movie_ids:
            return base_scores
        
        # Calculate average user preferences
        user_features = torch.stack([self.movie_features[mid] for mid in user_movie_ids])
        avg_user_preferences = torch.mean(user_features, dim=0)
        
        # Calculate similarity to all movies
        all_features = torch.stack(list(self.movie_features.values()))
        similarities = torch.cosine_similarity(avg_user_preferences.unsqueeze(0), all_features)
        
        # Ensure tensor has same size as base_scores
        if len(similarities) != len(base_scores):
            # Pad or truncate to match base_scores size
            if len(similarities) > len(base_scores):
                similarities = similarities[:len(base_scores)]
            else:
                similarities = torch.cat([similarities, torch.zeros(len(base_scores) - len(similarities), device=similarities.device)])
        
        return similarities
    
    def _popularity_based(self, base_scores: torch.Tensor, user_movies: List[str], 
                         movie_title_to_id: Dict) -> torch.Tensor:
        """Popularity-based scoring"""
        popularity_scores = torch.tensor([self.movie_popularity.get(mid, 0) for mid in movie_title_to_id.values()])
        return popularity_scores / popularity_scores.max()
    
    def _diversity_based(self, base_scores: torch.Tensor, user_movies: List[str], 
                         movie_title_to_id: Dict) -> torch.Tensor:
        """Diversity-based scoring to avoid similar recommendations"""
        user_movie_ids = [movie_title_to_id.get(movie) for movie in user_movies if movie in movie_title_to_id]
        
        if not user_movie_ids:
            return base_scores
        
        # Calculate diversity penalty (lower score for similar movies)
        user_embeddings = torch.stack([self.movie_embeddings[mid] for mid in user_movie_ids])
        all_embeddings = torch.stack(list(self.movie_embeddings.values()))
        
        similarities = cosine_similarity(user_embeddings, all_embeddings)
        max_similarity = np.max(similarities, axis=0)
        
        # Convert similarity to diversity (1 - similarity)
        diversity_scores = 1 - max_similarity
        
        # Ensure tensor has same size as base_scores
        if len(diversity_scores) != len(base_scores):
            # Pad or truncate to match base_scores size
            if len(diversity_scores) > len(base_scores):
                diversity_scores = diversity_scores[:len(base_scores)]
            else:
                diversity_scores = np.pad(diversity_scores, (0, len(base_scores) - len(diversity_scores)), 'constant')
        
        return torch.tensor(diversity_scores, dtype=torch.float32)
    
    def _novelty_based(self, base_scores: torch.Tensor, user_movies: List[str], 
                      movie_title_to_id: Dict) -> torch.Tensor:
        """Novelty-based scoring for less popular but high-quality movies"""
        popularity_scores = torch.tensor([self.movie_popularity.get(mid, 0) for mid in movie_title_to_id.values()])
        
        # Novelty = inverse of popularity (normalized)
        novelty_scores = 1 - (popularity_scores / popularity_scores.max())
        
        # Combine with base scores for quality
        return novelty_scores * base_scores
    
    def _combine_strategies(self, base_scores: torch.Tensor, strategies: Dict, 
                          user_movies: List[str], movie_title_to_id: Dict) -> torch.Tensor:
        """Combine multiple ranking strategies"""
        combined_scores = base_scores.clone()
        
        # Weights for different strategies
        weights = {
            'collaborative': 0.45,
            'content_based': 0.25,
            'popularity': 0.10,
            #'diversity': 0.1,
            'novelty': 0.1
        }
        
        for strategy_name, strategy_func in strategies.items():
            strategy_scores = strategy_func(base_scores, user_movies, movie_title_to_id)
            combined_scores += weights[strategy_name] * strategy_scores
        
        return combined_scores
    
    def _apply_diversity_penalty(self, scores: torch.Tensor, diversity_weight: float) -> torch.Tensor:
        """Apply diversity penalty to avoid similar recommendations"""
        # This is a simplified version - in practice, you'd track selected items
        return scores * (1 + diversity_weight * torch.rand_like(scores))

class PersonalizedRanking:
    """
    Personalized ranking based on user preferences and context
    """
    
    def __init__(self):
        self.user_preferences = defaultdict(dict)
        self.genre_weights = {
            'action': 1.0, 'comedy': 1.0, 'drama': 1.0, 'horror': 1.0,
            'romance': 1.0, 'sci-fi': 1.0, 'thriller': 1.0, 'documentary': 1.0
        }
    
    def update_user_preferences(self, user_id: str, movie_ratings: Dict[str, float]):
        """Update user preferences based on ratings"""
        for movie, rating in movie_ratings.items():
            self.user_preferences[user_id][movie] = rating
    
    def get_personalized_scores(self, user_id: str, movie_candidates: List[str], 
                              base_scores: torch.Tensor) -> torch.Tensor:
        """Get personalized scores for a user"""
        if user_id not in self.user_preferences:
            return base_scores
        
        user_ratings = self.user_preferences[user_id]
        personalized_scores = base_scores.clone()
        
        # Adjust scores based on user's historical preferences
        for i, movie in enumerate(movie_candidates):
            if movie in user_ratings:
                # Boost score for movies similar to highly rated ones
                rating = user_ratings[movie]
                personalized_scores[i] *= (1 + 0.2 * (rating - 5) / 5)
        
        return personalized_scores

class ContextualRecommendations:
    """
    Contextual recommendations based on time, mood, etc.
    """
    
    def __init__(self):
        self.time_weights = {
            'morning': {'comedy': 1.2, 'documentary': 1.1, 'action': 0.9},
            'afternoon': {'action': 1.1, 'comedy': 1.0, 'drama': 1.0},
            'evening': {'drama': 1.2, 'romance': 1.1, 'thriller': 1.1},
            'night': {'horror': 1.3, 'thriller': 1.2, 'sci-fi': 1.1}
        }
    
    def get_contextual_scores(self, time_of_day: str, mood: str, 
                            movie_candidates: List[str], base_scores: torch.Tensor) -> torch.Tensor:
        """Get contextual scores based on time and mood"""
        contextual_scores = base_scores.clone()
        
        # Apply time-based weights
        time_weights = self.time_weights.get(time_of_day, {})
        
        # Apply mood-based adjustments
        mood_weights = self._get_mood_weights(mood)
        
        # Combine time and mood weights
        for i, movie in enumerate(movie_candidates):
            # This is simplified - you'd need genre information for each movie
            contextual_scores[i] *= 1.0  # Placeholder
        
        return contextual_scores
    
    def _get_mood_weights(self, mood: str) -> Dict[str, float]:
        """Get genre weights based on mood"""
        mood_weights = {
            'happy': {'comedy': 1.3, 'romance': 1.2, 'action': 1.1},
            'sad': {'drama': 1.3, 'romance': 1.1, 'comedy': 0.9},
            'excited': {'action': 1.3, 'sci-fi': 1.2, 'thriller': 1.1},
            'relaxed': {'documentary': 1.2, 'drama': 1.1, 'comedy': 1.0}
        }
        return mood_weights.get(mood, {})

def create_enhanced_recommendations(user_movies: List[str], model, movie_data: Dict, 
                                  num_recommendations: int = 15) -> List[str]:
    """
    Create enhanced recommendations using multiple strategies
    """
    # Initialize recommendation engine
    engine = EnhancedRecommendationEngine(
        model=model,
        movie_embeddings=movie_data['embeddings'],
        movie_features=movie_data['features'],
        movie_popularity=movie_data['popularity']
    )
    
    # Get diverse recommendations
    recommended_indices = engine.get_diverse_recommendations(
        user_movies, 
        movie_data['title_to_id'],
        num_recommendations
    )
    
    # Convert indices to movie titles
    id_to_title = {v: k for k, v in movie_data['title_to_id'].items()}
    recommended_movies = [id_to_title.get(idx, f"Movie_{idx}") for idx in recommended_indices]
    
    return recommended_movies 