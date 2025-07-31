import React, { useEffect, useState } from 'react';
import { useForm } from 'react-hook-form';
import { 
  Box, 
  Button, 
  Typography,
  CircularProgress,
  Alert,
  Autocomplete,
  TextField,
  Chip,
  Card,
  CardContent,
  IconButton,
  Snackbar,
  Divider
} from '@mui/material';
import { 
  Close as CloseIcon, 
  Login, 
  Logout, 
  TrendingUp, 
  Star,
  Movie
} from '@mui/icons-material';
import axios from 'axios';
import { API_URLS } from '../config/api';

interface MovieFormProps {
  onRecommendations: (recommendations: string[]) => void;
}

interface FormData {
  movies: string[];
}

interface MovieOption {
  title: string;
  id?: number;
  year?: number;
  type?: 'movie' | 'show';
  ids?: {
    trakt?: number;
    imdb?: string;
    tmdb?: number;
  };
  overview?: string;
  rating?: number;
  votes?: number;
  popularity?: number;
}

interface TrendingMovie {
  title: string;
  year?: number;
  ids: {
    trakt?: number;
    imdb?: string;
    tmdb?: number;
  };
  overview?: string;
  rating?: number;
  votes?: number;
  watchers: number;
}

const MovieForm: React.FC<MovieFormProps> = ({ onRecommendations }) => {
  const { handleSubmit } = useForm<FormData>();
  
  // State management
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [movieOptions, setMovieOptions] = useState<MovieOption[]>([]);
  const [selectedMovies, setSelectedMovies] = useState<MovieOption[]>([]);
  const [searchResults, setSearchResults] = useState<MovieOption[]>([]);
  const [recentAndFavs, setRecentAndFavs] = useState<MovieOption[]>([]);
  const [trendingMovies, setTrendingMovies] = useState<TrendingMovie[]>([]);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [searchLoading, setSearchLoading] = useState(false);
  const [duplicateWarning, setDuplicateWarning] = useState(false);
  const [limitWarning, setLimitWarning] = useState(false);

  // Initialize component
  useEffect(() => {
    const storedSessionId = localStorage.getItem('trakt_session_id');
    if (storedSessionId) {
      setSessionId(storedSessionId);
      setIsAuthenticated(true);
      fetchRecentAndFavs(storedSessionId);
    }
    
    fetchTrendingMovies();
    fetchAvailableMovies();
    
    // Listen for authentication success
    const handleMessage = (event: MessageEvent) => {
      if (event.data.type === 'TRAKT_AUTH_SUCCESS') {
        setSessionId(event.data.sessionId);
        setIsAuthenticated(true);
        setError(null);
        setLoading(false);
        fetchRecentAndFavs(event.data.sessionId);
      }
    };

    window.addEventListener('message', handleMessage);
    return () => window.removeEventListener('message', handleMessage);
  }, []);

  // Fetch movies from GNN model (non-authenticated)
  const fetchAvailableMovies = async () => {
    try {
      const response = await axios.get(API_URLS.MOVIES, {
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
        withCredentials: true,
      });
      
      if (response.data?.movies) {
        setMovieOptions(response.data.movies);
      }
    } catch (error) {
      console.error('Error fetching movies:', error);
    }
  };

  // Fetch user's recent and favorite movies from Trakt
  const fetchRecentAndFavs = async (sessionId: string) => {
    try {
      const response = await axios.get(API_URLS.TRAKT_USER_HISTORY, {
        headers: { 'X-Session-ID': sessionId }
      });
      
      if (response.data?.movies) {
        setRecentAndFavs(response.data.movies);
      }
    } catch (error) {
      console.error('Error fetching user history:', error);
    }
  };

  // Fetch trending movies from Trakt
  const fetchTrendingMovies = async () => {
    try {
      const response = await axios.get(API_URLS.TRAKT_TRENDING);
      if (response.data?.movies) {
        setTrendingMovies(response.data.movies);
      }
    } catch (error) {
      console.error('Error fetching trending movies:', error);
    }
  };

  // Handle Trakt authentication
  const handleTraktAuth = async () => {
    try {
      setLoading(true);
      const response = await axios.get(API_URLS.TRAKT_AUTH);
      if (response.data?.auth_url) {
        window.open(response.data.auth_url, 'trakt_auth', 'width=500,height=600');
      }
    } catch (error) {
      setError('Failed to start authentication');
      console.error('Auth error:', error);
    } finally {
      setLoading(false);
    }
  };

  // Handle logout
  const handleLogout = () => {
    localStorage.removeItem('trakt_session_id');
    setSessionId(null);
    setIsAuthenticated(false);
    setRecentAndFavs([]);
  };

  // Search movies (works for both Trakt and GNN movies)
  const handleSearch = async (_: any, query: string) => {
    if (!query || query.length < 2) {
      setSearchResults([]);
      return;
    }

    try {
      setSearchLoading(true);
      
      if (isAuthenticated) {
        // Search Trakt API
        const response = await axios.get(`${API_URLS.TRAKT_SEARCH}?query=${encodeURIComponent(query)}`);
        if (response.data?.movies) {
          setSearchResults(response.data.movies);
        }
      } else {
        // Search local GNN movies
        const filteredMovies = movieOptions.filter(movie =>
          movie.title.toLowerCase().includes(query.toLowerCase())
        );
        setSearchResults(filteredMovies.slice(0, 10));
      }
    } catch (error) {
      console.error('Error searching movies:', error);
    } finally {
      setSearchLoading(false);
    }
  };

  // Handle movie selection
  const handleSelectMovie = (_: any, value: MovieOption | null) => {
    if (!value) return;

    // Check for duplicates
    if (selectedMovies.some(selected => selected.title === value.title)) {
      setDuplicateWarning(true);
      return;
    }

    // Check limit
    if (selectedMovies.length >= 15) {
      setLimitWarning(true);
      return;
    }

    setSelectedMovies(prev => [...prev, value]);
    setSearchResults([]);
  };

  // Add trending movie
  const handleAddTrending = (movie: TrendingMovie) => {
    const movieOption: MovieOption = {
      title: movie.title,
      year: movie.year,
      ids: movie.ids,
      overview: movie.overview,
      rating: movie.rating,
      votes: movie.votes
    };
    handleSelectMovie(null, movieOption);
  };

  // Remove selected movie
  const removeMovie = (title: string) => {
    setSelectedMovies(prev => prev.filter(movie => movie.title !== title));
  };

  // Submit recommendations
  const onSubmit = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const movieTitles = selectedMovies.map(m => m.title);
      console.log('Submitting movies:', movieTitles);

      let response;
      if (isAuthenticated && sessionId) {
        // Use Trakt API
        response = await axios.post(API_URLS.TRAKT_RECOMMEND, {
          movies: movieTitles
        }, {
          headers: { 'X-Session-ID': sessionId }
        });
      } else {
        // Use GNN model
        response = await axios.post(API_URLS.RECOMMEND, {
          movies: movieTitles
        });
      }
      
      console.log('Recommendation response:', response.data);
      onRecommendations(response.data.recommendations);
      setSelectedMovies([]);
    } catch (err: any) {
      console.error('Recommendation error:', err);
      if (err.response) {
        setError(`Failed to get recommendations: ${err.response.data.error || err.response.statusText}`);
      } else {
        setError(`Failed to get recommendations: ${err.message}`);
      }
    } finally {
      setLoading(false);
    }
  };

  // Get available options for search
  const getAvailableOptions = () => {
    if (isAuthenticated) {
      return [...recentAndFavs, ...searchResults];
    } else {
      return [...movieOptions, ...searchResults];
    }
  };

  return (
    <Box sx={{ maxWidth: 800, mx: 'auto', p: 3 }}>
      {/* Header */}
      <Typography 
        variant="h3" 
        gutterBottom 
        align="center" 
        sx={{ 
          fontWeight: 'bold',
          background: 'linear-gradient(45deg, #1e3c72, #2a5298, #4facfe)',
          backgroundClip: 'text',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
          textShadow: '0 2px 4px rgba(0,0,0,0.1)',
          mb: 3
        }}
      >
        🎬 Lights, Camera, Recommendations! 🍿
      </Typography>
      
      <Typography 
        variant="h6" 
        align="center" 
        sx={{ 
          color: 'text.secondary',
          mb: 4,
          fontStyle: 'italic',
          fontWeight: 300
        }}
      >
        Share your favorite movies and let our AI discover your next cinematic adventure! ✨
      </Typography>

      {/* Authentication Section */}
      <Card sx={{ 
        mb: 3, 
        borderRadius: 3,
        border: '1px solid rgba(255, 255, 255, 0.2)',
        boxShadow: '0 8px 32px rgba(0, 0, 0, 0.1)',
        backdropFilter: 'blur(10px)',
        backgroundColor: 'rgba(255, 255, 255, 0.9)'
      }}>
        <CardContent>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <Box>
              <Typography variant="h6" gutterBottom>
                {isAuthenticated ? 'Connected to Trakt' : 'Movie Recommendation Options'}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {isAuthenticated 
                  ? 'Get personalized recommendations based on your watch history and ratings!'
                  : 'Use our AI model for recommendations, or connect Trakt for personalized suggestions.'
                }
              </Typography>
            </Box>
            <Button
              variant={isAuthenticated ? "outlined" : "contained"}
              color={isAuthenticated ? "error" : "primary"}
              startIcon={isAuthenticated ? <Logout /> : <Login />}
              onClick={isAuthenticated ? handleLogout : handleTraktAuth}
              disabled={loading}
            >
              {isAuthenticated ? 'Disconnect' : 'Connect Trakt'}
            </Button>
          </Box>
        </CardContent>
      </Card>

      {/* Error Display */}
      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      {/* Movie Search */}
      <Card sx={{ mb: 3, borderRadius: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Select Movies You Like (1-15)
          </Typography>
          
          <Autocomplete
            options={getAvailableOptions()}
            getOptionLabel={(option) => option.title}
            onChange={handleSelectMovie}
            onInputChange={handleSearch}
            loading={searchLoading}
            renderInput={(params) => (
              <TextField
                {...params}
                label={`Search for movies${isAuthenticated ? ' (Trakt)' : ' (AI Model)'}`}
                variant="outlined"
                margin="normal"
                fullWidth
              />
            )}
            renderOption={(props, option) => (
              <li {...props}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', width: '100%' }}>
                  <Typography>{option.title}</Typography>
                  {option.popularity && (
                    <Chip 
                      label={`${option.popularity} ratings`} 
                      size="small" 
                      color="primary" 
                      variant="outlined"
                    />
                  )}
                </Box>
              </li>
            )}
          />
        </CardContent>
      </Card>

      {/* Trending Movies (Trakt only) */}
      {isAuthenticated && trendingMovies.length > 0 && (
        <Card sx={{ mb: 3, borderRadius: 3 }}>
          <CardContent>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
              <TrendingUp sx={{ mr: 1, color: 'primary.main' }} />
              <Typography variant="h6">Trending Now</Typography>
            </Box>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
              {trendingMovies.slice(0, 10).map((movie, index) => (
                <Chip
                  key={index}
                  label={movie.title}
                  onClick={() => handleAddTrending(movie)}
                  variant="outlined"
                  sx={{ cursor: 'pointer' }}
                />
              ))}
            </Box>
          </CardContent>
        </Card>
      )}

      {/* Selected Movies */}
      {selectedMovies.length > 0 && (
        <Card sx={{ mb: 3, borderRadius: 3 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Selected Movies ({selectedMovies.length}/15)
            </Typography>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2 }}>
              {selectedMovies.map((movie, index) => (
                <Card key={index} sx={{ minWidth: 200, maxWidth: 300 }}>
                  <CardContent sx={{ pb: 1 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                      <Typography variant="body1" sx={{ flex: 1, mr: 1 }}>
                        {movie.title}
                      </Typography>
                      <IconButton
                        size="small"
                        onClick={() => removeMovie(movie.title)}
                        sx={{ mt: -0.5, mr: -0.5 }}
                      >
                        <CloseIcon fontSize="small" />
                      </IconButton>
                    </Box>
                    {movie.popularity && (
                      <Chip 
                        label={`${movie.popularity} ratings`} 
                        size="small" 
                        color="primary" 
                        variant="outlined"
                        sx={{ mt: 1 }}
                      />
                    )}
                  </CardContent>
                </Card>
              ))}
            </Box>
          </CardContent>
        </Card>
      )}

      {/* Submit Button */}
      <Button
        type="submit"
        variant="contained"
        color="primary"
        fullWidth
        sx={{ mt: 2 }}
        disabled={loading || selectedMovies.length === 0 || selectedMovies.length > 15}
        onClick={handleSubmit(onSubmit)}
      >
        {loading ? <CircularProgress size={24} /> : 'Get Recommendations'}
      </Button>

      {/* Warning Snackbars */}
      <Snackbar
        open={duplicateWarning}
        autoHideDuration={3000}
        onClose={() => setDuplicateWarning(false)}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert onClose={() => setDuplicateWarning(false)} severity="warning">
          This movie is already selected!
        </Alert>
      </Snackbar>

      <Snackbar
        open={limitWarning}
        autoHideDuration={3000}
        onClose={() => setLimitWarning(false)}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert onClose={() => setLimitWarning(false)} severity="warning">
          You can only select up to 15 movies!
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default MovieForm; 