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
  Chip
} from '@mui/material';
import axios from 'axios';

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
  const { setValue, formState: { errors }, handleSubmit } = useForm<FormData>();
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  const [movieOptions, setMovieOptions] = useState<MovieOption[]>([]);
  const [selectedMovies, setSelectedMovies] = useState<string[]>([]);

  useEffect(() => {
    // Fetch available movies from the backend
    const fetchMovies = async () => {
      try {
        console.log('Fetching movies from backend...');
        const response = await axios.get('http://localhost:5000/api/movies', {
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

  const onSubmit = async (data: FormData) => {
    try {
      setLoading(true);
      setError(null);
      const response = await axios.post('http://localhost:5000/api/recommend', {
        movies: selectedMovies
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

  const handleMovieChange = (index: number, value: MovieOption | null) => {
    try {
      console.log('Movie change:', { index, value });
      const newSelectedMovies = [...selectedMovies];
      newSelectedMovies[index] = value ? value.title : '';
      setSelectedMovies(newSelectedMovies);
      setValue(`movies.${index}`, value ? value.title : '');
    } catch (err) {
      console.error('Error in handleMovieChange:', err);
    }
  };

  return (
    <Box component="form" onSubmit={handleSubmit(onSubmit)} sx={{ maxWidth: 600, mx: 'auto', p: 3 }}>
      <Typography variant="h5" gutterBottom>
        Enter 5 Movies You Like
      </Typography>
      
      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      {[...Array(5)].map((_, index) => (
        <Autocomplete
          key={index}
          options={movieOptions}
          getOptionLabel={(option) => option.title}
          value={movieOptions.find(option => option.title === selectedMovies[index]) || null}
          onChange={(_, newValue) => handleMovieChange(index, newValue)}
          renderInput={(params) => (
            <TextField
              {...params}
              label={`Movie ${index + 1}`}
              variant="outlined"
              margin="normal"
              error={!!errors.movies?.[index]}
              helperText={errors.movies?.[index]?.message}
              required
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
          sx={{ mb: 2 }}
        />
      ))}

      <Button
        type="submit"
        variant="contained"
        color="primary"
        fullWidth
        sx={{ mt: 2 }}
        disabled={loading || selectedMovies.length !== 5 || selectedMovies.some(movie => !movie)}
      >
        {loading ? <CircularProgress size={24} /> : 'Get Recommendations'}
      </Button>
    </Box>
  );
};

export default MovieForm; 