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
  Snackbar,
  IconButton
} from '@mui/material';
import { TrendingUp, Star, Login, Logout, Close as CloseIcon } from '@mui/icons-material';
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
  const { handleSubmit } = useForm<FormData>();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [searchResults, setSearchResults] = useState<MovieOption[]>([]);
  const [selectedMovies, setSelectedMovies] = useState<MovieOption[]>([]);
  const [recentAndFavs, setRecentAndFavs] = useState<MovieOption[]>([]);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [searchLoading, setSearchLoading] = useState(false);
  const [duplicateWarning, setDuplicateWarning] = useState(false);
  const [limitWarning, setLimitWarning] = useState(false);
  const [trendingMovies, setTrendingMovies] = useState<MovieOption[]>([]);

  useEffect(() => {
    const storedSessionId = localStorage.getItem('trakt_session_id');
    if (storedSessionId) {
      setSessionId(storedSessionId);
      setIsAuthenticated(true);
    }
    if (storedSessionId) {
      fetchRecentAndFavs(storedSessionId);
    }
    fetchTrendingMovies();
    // Listen for authentication success message from popup
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

  const fetchRecentAndFavs = async (sessionId: string) => {
    try {
      const response = await axios.get(API_URLS.TRAKT_USER_HISTORY, {
        headers: { 'X-Session-ID': sessionId }
      });
      // Flatten and deduplicate by title
      const movies: MovieOption[] = [];
      const seen = new Set();
      [
        ...(response.data.favorite_movies || []),
        ...(response.data.recently_watched_movies || [])
      ].forEach((item: any) => {
        const movie = item.movie || item.show;
        if (movie && !seen.has(movie.title)) {
          seen.add(movie.title);
          movies.push({
            title: movie.title,
            year: movie.year,
            type: item.movie ? 'movie' : 'show',
            ids: movie.ids,
            overview: movie.overview,
            rating: movie.rating,
            votes: movie.votes
          });
        }
      });
      setRecentAndFavs(movies);
    } catch (err) {
      console.error('Error fetching user history:', err);
    }
  };

  const fetchTrendingMovies = async () => {
    try {
      const response = await axios.get(API_URLS.TRAKT_TRENDING, { params: { limit: 20 } });
      setTrendingMovies((response.data.trending || []).map(item => ({
        title: item.title,
        year: item.year,
        type: 'movie',
        ids: item.ids,
        overview: item.overview,
        rating: item.rating,
        votes: item.votes
      })));
    } catch (err) {
      setTrendingMovies([]);
    }
  };

  const handleTraktAuth = async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await axios.get(API_URLS.TRAKT_AUTH);
      const authUrl = response.data.auth_url;
      const authWindow = window.open(authUrl, 'trakt_auth', 'width=600,height=700');
      const checkAuth = setInterval(() => {
        if (authWindow?.closed) {
          clearInterval(checkAuth);
          const storedSessionId = localStorage.getItem('trakt_session_id');
          if (storedSessionId) {
            setSessionId(storedSessionId);
            setIsAuthenticated(true);
            setError(null);
            fetchRecentAndFavs(storedSessionId);
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
    setRecentAndFavs([]);
  };

  const handleSearch = async (_: any, query: string) => {
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
      setSearchResults([]);
    } finally {
      setSearchLoading(false);
    }
  };

  const handleSelectMovie = (_: any, value: MovieOption | null) => {
    if (!value) return;
    if (selectedMovies.some(m => m.title === value.title)) {
      setDuplicateWarning(true);
      return;
    }
    if (selectedMovies.length >= 15) {
      setLimitWarning(true);
      return;
    }
    setSelectedMovies([...selectedMovies, value]);
  };

  const handleAddTrending = (movie: MovieOption) => {
    if (selectedMovies.some(m => m.title === movie.title)) {
      setDuplicateWarning(true);
      return;
    }
    if (selectedMovies.length >= 15) {
      setLimitWarning(true);
      return;
    }
    setSelectedMovies([...selectedMovies, movie]);
  };

  const removeMovie = (title: string) => {
    setSelectedMovies(selectedMovies.filter(m => m.title !== title));
  };

  // Combine recent/favs and search results, deduped, with recent/favs on top
  const combinedOptions = [
    ...recentAndFavs,
    ...searchResults.filter(
      sr => !recentAndFavs.some(rf => rf.title === sr.title)
    )
  ].filter(
    option => !selectedMovies.some(m => m.title === option.title)
  );

  const onSubmit = async () => {
    try {
      setLoading(true);
      setError(null);
      if (!isAuthenticated) {
        setError('Please authenticate with Trakt first');
        return;
      }
      const response = await axios.post(API_URLS.TRAKT_RECOMMEND, {
        movies: selectedMovies.map(m => m.title)
      }, {
        headers: {
          'X-Session-ID': sessionId
        }
      });
      onRecommendations(response.data.recommendations);
      setSelectedMovies([]);
    } catch (err) {
      setError('Failed to get recommendations. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box sx={{ maxWidth: 800, mx: 'auto', p: 3 }}>
      <Typography variant="h4" gutterBottom align="center">
        Movie Recommendations with Trakt
      </Typography>
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
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Add Movies You Like (1–15)
          </Typography>
          <Autocomplete
            options={combinedOptions}
            getOptionLabel={option => `${option.title}${option.year ? ` (${option.year})` : ''}`}
            onInputChange={handleSearch}
            onChange={handleSelectMovie}
            loading={searchLoading}
            renderInput={params => (
              <TextField
                {...params}
                label="Search or select a movie/show"
                variant="outlined"
                margin="normal"
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
                <Box sx={{ display: 'flex', flexDirection: 'column' }}>
                  <Typography variant="body1">{option.title}</Typography>
                  <Typography variant="body2" color="text.secondary">
                    {option.year} • {option.type} {option.overview ? `• ${option.overview.substring(0, 60)}...` : ''}
                  </Typography>
                </Box>
              </li>
            )}
            sx={{ mb: 2 }}
          />
          {/* Trending Movies Section */}
          <Box sx={{ mb: 2 }}>
            <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <TrendingUp /> Trending Movies
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              Click on a trending movie to add it to your selection
            </Typography>
            <Box sx={{
              display: 'grid',
              gridTemplateColumns: { xs: '1fr', sm: 'repeat(2, 1fr)', md: 'repeat(3, 1fr)' },
              gap: 2
            }}>
              {trendingMovies.slice(0, 10).map((movie, index) => (
                <Card
                  key={index}
                  variant="outlined"
                  sx={{ cursor: 'pointer', '&:hover': { backgroundColor: 'action.hover' } }}
                  onClick={() => handleAddTrending(movie)}
                >
                  <CardContent sx={{ p: 2 }}>
                    <Typography variant="subtitle2" noWrap>
                      {movie.title}
                    </Typography>
                    <Typography variant="body2" color="text.secondary" noWrap>
                      {movie.year}
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
              ))}
            </Box>
          </Box>
          {/* Selected Movie Cards */}
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2, mb: 2 }}>
            {selectedMovies.map(movie => (
              <Card key={movie.title} sx={{ minWidth: 200, position: 'relative' }}>
                <CardContent>
                  <Typography variant="subtitle1">{movie.title}</Typography>
                  <Typography variant="body2" color="text.secondary">
                    {movie.year} • {movie.type}
                  </Typography>
                  <IconButton
                    aria-label="remove"
                    onClick={() => removeMovie(movie.title)}
                    sx={{ position: 'absolute', top: 0, right: 0 }}
                  >
                    <CloseIcon />
                  </IconButton>
                </CardContent>
              </Card>
            ))}
          </Box>
          <Snackbar
            open={duplicateWarning}
            autoHideDuration={3000}
            onClose={() => setDuplicateWarning(false)}
            message="This movie is already selected."
            anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
          />
          <Snackbar
            open={limitWarning}
            autoHideDuration={3000}
            onClose={() => setLimitWarning(false)}
            message="Maximum 15 movies allowed."
            anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
          />
          <Button
            type="submit"
            variant="contained"
            color="primary"
            fullWidth
            sx={{ mt: 2 }}
            disabled={
              loading ||
              !isAuthenticated ||
              selectedMovies.length < 1 ||
              selectedMovies.length > 15
            }
            onClick={handleSubmit(onSubmit)}
          >
            {loading ? <CircularProgress size={24} /> : 'Get Recommendations'}
          </Button>
        </CardContent>
      </Card>
    </Box>
  );
};

export default TraktMovieForm; 