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
  Snackbar
} from '@mui/material';
import { Close as CloseIcon } from '@mui/icons-material';
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
  id: number;
  popularity: number;
}

const MovieForm: React.FC<MovieFormProps> = ({ onRecommendations }) => {
  const { handleSubmit } = useForm<FormData>();
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  const [movieOptions, setMovieOptions] = useState<MovieOption[]>([]);
  const [selectedMovies, setSelectedMovies] = useState<MovieOption[]>([]);
  const [searchResults, setSearchResults] = useState<MovieOption[]>([]);
  const [duplicateWarning, setDuplicateWarning] = useState(false);
  const [limitWarning, setLimitWarning] = useState(false);

  useEffect(() => {
    // Fetch available movies from the backend
    const fetchMovies = async () => {
      try {
        console.log('Fetching movies from backend...');
        const response = await axios.get(API_URLS.MOVIES, {
          headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
          },
          withCredentials: true,
        });
        
        console.log('Response received:', response);
        
        if (!response.data) {
          console.error('No data in response');
          setError('No data received from server');
          return;
        }
        
        if (!response.data.movies) {
          console.error('No movies in response data:', response.data);
          setError('Invalid response format from server');
          return;
        }
        
        const movies = response.data.movies;
        console.log('Movies received:', movies);
        
        if (!Array.isArray(movies)) {
          console.error('Movies is not an array:', movies);
          setError('Invalid movies data format');
          return;
        }
        
        if (movies.length === 0) {
          console.error('No movies in the array');
          setError('No movies available');
          return;
        }
        
        console.log('Setting movie options:', movies);
        setMovieOptions(movies);
        
      } catch (error) {
        console.error('Error fetching movies:', error);
        if (axios.isAxiosError(error)) {
          if (error.response) {
            console.error('Response error:', error.response.data);
            console.error('Status code:', error.response.status);
            console.error('Headers:', error.response.headers);
          } else if (error.request) {
            console.error('Request error:', error.request);
          }
        } else {
          console.error('Error:', error);
        }
        setError('Failed to fetch available movies');
      }
    };

    fetchMovies();
  }, []);

  const onSubmit = async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await axios.post(API_URLS.RECOMMEND, {
        movies: selectedMovies.map(movie => movie.title)
      });
      onRecommendations(response.data.recommendations);
      setSelectedMovies([]);
    } catch (err) {
      setError('Failed to get recommendations. Please try again.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleSearch = async (_: any, query: string) => {
    if (!query || query.length < 2) {
      setSearchResults([]);
      return;
    }

    try {
      const filteredMovies = movieOptions.filter(movie =>
        movie.title.toLowerCase().includes(query.toLowerCase())
      );
      setSearchResults(filteredMovies.slice(0, 10)); // Limit to 10 results
    } catch (error) {
      console.error('Error searching movies:', error);
    }
  };

  const handleSelectMovie = (movie: MovieOption | null) => {
    if (!movie) return;

    // Check for duplicates
    if (selectedMovies.some(selected => selected.id === movie.id)) {
      setDuplicateWarning(true);
      return;
    }

    // Check limit
    if (selectedMovies.length >= 15) {
      setLimitWarning(true);
      return;
    }

    setSelectedMovies(prev => [...prev, movie]);
    setSearchResults([]); // Clear search results after selection
  };

  const removeMovie = (movieId: number) => {
    setSelectedMovies(prev => prev.filter(movie => movie.id !== movieId));
  };

  // Combine all available options, filtering out already selected movies
  const combinedOptions = [...movieOptions]
    .filter(movie => !selectedMovies.some(selected => selected.id === movie.id))
    .slice(0, 20); // Limit to 20 options for performance

  return (
    <Box component="form" onSubmit={handleSubmit(onSubmit)} sx={{ maxWidth: 800, mx: 'auto', p: 3 }}>
      <Typography variant="h5" gutterBottom>
        Select Movies You Like (1-15)
      </Typography>
      
      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      {/* Single Search Bar */}
      <Autocomplete
        options={combinedOptions}
        getOptionLabel={(option) => option.title}
        onChange={(_, newValue) => handleSelectMovie(newValue)}
        onInputChange={handleSearch}
        renderInput={(params) => (
          <TextField
            {...params}
            label="Search for movies..."
            variant="outlined"
            margin="normal"
            fullWidth
          />
        )}
        renderOption={(props, option) => (
          <li {...props}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', width: '100%' }}>
              <Typography>{option.title}</Typography>
              <Chip 
                label={`${option.popularity} ratings`} 
                size="small" 
                color="primary" 
                variant="outlined"
              />
            </Box>
          </li>
        )}
        sx={{ mb: 3 }}
      />

      {/* Selected Movies Display */}
      {selectedMovies.length > 0 && (
        <Box sx={{ mb: 3 }}>
          <Typography variant="h6" gutterBottom>
            Selected Movies ({selectedMovies.length}/15)
          </Typography>
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2 }}>
            {selectedMovies.map((movie) => (
              <Card key={movie.id} sx={{ minWidth: 200, maxWidth: 300 }}>
                <CardContent sx={{ pb: 1 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                    <Typography variant="body1" sx={{ flex: 1, mr: 1 }}>
                      {movie.title}
                    </Typography>
                    <IconButton
                      size="small"
                      onClick={() => removeMovie(movie.id)}
                      sx={{ mt: -0.5, mr: -0.5 }}
                    >
                      <CloseIcon fontSize="small" />
                    </IconButton>
                  </Box>
                  <Chip 
                    label={`${movie.popularity} ratings`} 
                    size="small" 
                    color="primary" 
                    variant="outlined"
                    sx={{ mt: 1 }}
                  />
                </CardContent>
              </Card>
            ))}
          </Box>
        </Box>
      )}

      <Button
        type="submit"
        variant="contained"
        color="primary"
        fullWidth
        sx={{ mt: 2 }}
        disabled={loading || selectedMovies.length === 0 || selectedMovies.length > 15}
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