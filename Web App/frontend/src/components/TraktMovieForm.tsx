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
  Grid
} from '@mui/material';
import { TrendingUp, Star, Login, Logout } from '@mui/icons-material';
import axios from 'axios';
import { API_URLS } from '../config/api';

interface TraktMovieFormProps {
  onRecommendations: (recommendations: string[]) => void;
}

interface FormData {
  movies: string[];
}

interface MovieOption {
  title: string;
  year?: number;
  type: 'movie' | 'show';
  ids: {
    trakt?: number;
    imdb?: string;
    tmdb?: number;
  };
  overview?: string;
  rating?: number;
  votes?: number;
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

const TraktMovieForm: React.FC<TraktMovieFormProps> = ({ onRecommendations }) => {
  const { setValue, formState: { errors }, handleSubmit } = useForm<FormData>();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [searchResults, setSearchResults] = useState<MovieOption[]>([]);
  const [selectedMovies, setSelectedMovies] = useState<string[]>([]);
  const [trendingMovies, setTrendingMovies] = useState<TrendingMovie[]>([]);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [searchLoading, setSearchLoading] = useState(false);

  useEffect(() => {
    // Check if user is authenticated (check for session ID in localStorage)
    const storedSessionId = localStorage.getItem('trakt_session_id');
    if (storedSessionId) {
      setSessionId(storedSessionId);
      setIsAuthenticated(true);
    }

    // Load trending movies
    loadTrendingMovies();

    // Listen for authentication success message from popup
    const handleMessage = (event: MessageEvent) => {
      if (event.data.type === 'TRAKT_AUTH_SUCCESS') {
        setSessionId(event.data.sessionId);
        setIsAuthenticated(true);
        setError(null);
        setLoading(false);
      }
    };

    window.addEventListener('message', handleMessage);
    return () => window.removeEventListener('message', handleMessage);
  }, []);

  const loadTrendingMovies = async () => {
    try {
      const response = await axios.get(API_URLS.TRAKT_TRENDING, {
        params: { limit: 20 }
      });
      setTrendingMovies(response.data.trending || []);
    } catch (err) {
      console.error('Error loading trending movies:', err);
    }
  };

  const handleTraktAuth = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await axios.get(API_URLS.TRAKT_AUTH);
      const authUrl = response.data.auth_url;
      
      // Open Trakt authorization in a new window
      const authWindow = window.open(authUrl, 'trakt_auth', 'width=600,height=700');
      
      // Listen for the callback
      const checkAuth = setInterval(() => {
        if (authWindow?.closed) {
          clearInterval(checkAuth);
          // Check if we have a session ID (this would be set by the callback)
          const storedSessionId = localStorage.getItem('trakt_session_id');
          if (storedSessionId) {
            setSessionId(storedSessionId);
            setIsAuthenticated(true);
            setError(null);
          } else {
            setError('Authentication was cancelled or failed');
          }
          setLoading(false);
        }
      }, 1000);
      
    } catch (err) {
      setError('Failed to start authentication');
      setLoading(false);
      console.error(err);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('trakt_session_id');
    setSessionId(null);
    setIsAuthenticated(false);
    setSelectedMovies([]);
  };

  const handleSearch = async (query: string) => {
    if (!query.trim()) {
      setSearchResults([]);
      return;
    }

    try {
      setSearchLoading(true);
      const response = await axios.get(API_URLS.TRAKT_SEARCH, {
        params: { query, limit: 20 }
      });
      setSearchResults(response.data.results || []);
    } catch (err) {
      console.error('Error searching:', err);
      setSearchResults([]);
    } finally {
      setSearchLoading(false);
    }
  };

  const onSubmit = async () => {
    try {
      setLoading(true);
      setError(null);
      
      if (!isAuthenticated) {
        setError('Please authenticate with Trakt first');
        return;
      }

      const response = await axios.post(API_URLS.TRAKT_RECOMMEND, {
        movies: selectedMovies
      }, {
        headers: {
          'X-Session-ID': sessionId
        }
      });
      
      onRecommendations(response.data.recommendations);
      setSelectedMovies([]);
    } catch (err) {
      if (axios.isAxiosError(err) && err.response?.status === 401) {
        setError('Authentication expired. Please log in again.');
        handleLogout();
      } else {
        setError('Failed to get recommendations. Please try again.');
      }
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleMovieChange = (index: number, value: MovieOption | null) => {
    try {
      const newSelectedMovies = [...selectedMovies];
      newSelectedMovies[index] = value ? value.title : '';
      setSelectedMovies(newSelectedMovies);
      setValue(`movies.${index}`, value ? value.title : '');
    } catch (err) {
      console.error('Error in handleMovieChange:', err);
    }
  };

  const addFromTrending = (movie: TrendingMovie) => {
    const emptyIndex = selectedMovies.findIndex(movie => !movie);
    if (emptyIndex !== -1) {
      const newSelectedMovies = [...selectedMovies];
      newSelectedMovies[emptyIndex] = movie.title;
      setSelectedMovies(newSelectedMovies);
      setValue(`movies.${emptyIndex}`, movie.title);
    }
  };

  return (
    <Box sx={{ maxWidth: 800, mx: 'auto', p: 3 }}>
      <Typography variant="h4" gutterBottom align="center">
        Movie Recommendations with Trakt
      </Typography>
      
      {/* Authentication Section */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <Box>
              <Typography variant="h6" gutterBottom>
                Trakt Authentication
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {isAuthenticated 
                  ? 'Connected to Trakt - Get personalized recommendations!' 
                  : 'Connect your Trakt account to get personalized recommendations based on your watch history and ratings.'
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

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      {/* Movie Selection Form */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Enter 5 Movies You Like
          </Typography>
          
          {[...Array(5)].map((_, index) => (
            <Autocomplete
              key={index}
              options={searchResults}
              getOptionLabel={(option) => `${option.title}${option.year ? ` (${option.year})` : ''}`}
              value={searchResults.find(option => option.title === selectedMovies[index]) || null}
              onChange={(_, newValue) => handleMovieChange(index, newValue)}
              onInputChange={(_, newInputValue) => {
                if (newInputValue.length > 2) {
                  handleSearch(newInputValue);
                }
              }}
              loading={searchLoading}
              renderInput={(params) => (
                <TextField
                  {...params}
                  label={`Movie ${index + 1}`}
                  variant="outlined"
                  margin="normal"
                  error={!!errors.movies?.[index]}
                  helperText={errors.movies?.[index]?.message}
                  required
                  InputProps={{
                    ...params.InputProps,
                    endAdornment: (
                      <>
                        {searchLoading ? <CircularProgress color="inherit" size={20} /> : null}
                        {params.InputProps.endAdornment}
                      </>
                    ),
                  }}
                />
              )}
              renderOption={(props, option) => (
                <li {...props}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', width: '100%', alignItems: 'center' }}>
                    <Box>
                      <Typography variant="body1">{option.title}</Typography>
                      <Typography variant="body2" color="text.secondary">
                        {option.year} • {option.type} • {option.overview?.substring(0, 100)}...
                      </Typography>
                    </Box>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      {option.rating && (
                        <Chip 
                          icon={<Star />}
                          label={`${option.rating.toFixed(1)}`} 
                          size="small" 
                          color="primary" 
                          variant="outlined"
                        />
                      )}
                      <Chip 
                        label={option.type} 
                        size="small" 
                        color="secondary" 
                        variant="outlined"
                      />
                    </Box>
                  </Box>
                </li>
              )}
              sx={{ mb: 2 }}
            />
          ))}

          <Button
            type="submit"
            variant="contained"
            color="primary"
            fullWidth
            sx={{ mt: 2 }}
            disabled={loading || !isAuthenticated || selectedMovies.length !== 5 || selectedMovies.some(movie => !movie)}
            onClick={handleSubmit(onSubmit)}
          >
            {loading ? <CircularProgress size={24} /> : 'Get Recommendations'}
          </Button>
        </CardContent>
      </Card>

      {/* Trending Movies Section */}
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <TrendingUp />
            Trending Movies
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            Click on a trending movie to add it to your selection
          </Typography>
          
          <Grid container spacing={2}>
            {trendingMovies.slice(0, 10).map((movie, index) => (
              <Grid xs={12} sm={6} md={4} key={index}>
                <Card 
                  variant="outlined" 
                  sx={{ 
                    cursor: 'pointer',
                    '&:hover': { backgroundColor: 'action.hover' }
                  }}
                  onClick={() => addFromTrending(movie)}
                >
                  <CardContent sx={{ p: 2 }}>
                    <Typography variant="subtitle2" noWrap>
                      {movie.title}
                    </Typography>
                    <Typography variant="body2" color="text.secondary" noWrap>
                      {movie.year} • {movie.watchers} watching
                    </Typography>
                    {movie.rating && (
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mt: 1 }}>
                        <Star sx={{ fontSize: 16, color: 'warning.main' }} />
                        <Typography variant="body2">
                          {movie.rating.toFixed(1)}
                        </Typography>
                      </Box>
                    )}
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </CardContent>
      </Card>
    </Box>
  );
};

export default TraktMovieForm; 