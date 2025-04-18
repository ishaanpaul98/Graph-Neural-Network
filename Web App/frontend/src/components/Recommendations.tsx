import React from 'react';
import { Box, Typography, List, ListItem, ListItemText, Paper } from '@mui/material';
import MovieIcon from '@mui/icons-material/Movie';

interface RecommendationsProps {
  recommendations: string[];
}

const Recommendations: React.FC<RecommendationsProps> = ({ recommendations }) => {
  if (!recommendations.length) return null;

  return (
    <Box sx={{ maxWidth: 600, mx: 'auto', mt: 4, p: 3 }}>
      <Typography variant="h5" gutterBottom>
        Recommended Movies
      </Typography>
      <Paper elevation={2}>
        <List>
          {recommendations.map((movie, index) => (
            <ListItem key={index} divider={index < recommendations.length - 1}>
              <MovieIcon sx={{ mr: 2, color: 'primary.main' }} />
              <ListItemText primary={movie} />
            </ListItem>
          ))}
        </List>
      </Paper>
    </Box>
  );
};

export default Recommendations; 