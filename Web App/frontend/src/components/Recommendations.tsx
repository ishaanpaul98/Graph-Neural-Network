import React from 'react';
import { Box, Typography, List, ListItem, ListItemText, Paper, Chip } from '@mui/material';
import MovieIcon from '@mui/icons-material/Movie';
import StarIcon from '@mui/icons-material/Star';

interface RecommendationsProps {
  recommendations: string[];
}

const Recommendations: React.FC<RecommendationsProps> = ({ recommendations }) => {
  if (!recommendations.length) return null;

  return (
    <Box sx={{ maxWidth: 600, mx: 'auto', mt: 4, p: 3 }}>
      <Typography 
        variant="h4" 
        gutterBottom 
        align="center"
        sx={{ 
          fontWeight: 'bold',
          background: 'linear-gradient(45deg, #1e3c72, #2a5298, #4facfe)',
          backgroundClip: 'text',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
          textShadow: '0 2px 4px rgba(0,0,0,0.1)',
          mb: 2
        }}
      >
        🎯 Your Perfect Matches! 🎬
      </Typography>
      <Typography 
        variant="body1" 
        align="center" 
        sx={{ 
          color: 'text.secondary',
          mb: 3,
          fontStyle: 'italic'
        }}
      >
        Based on your taste, here are some amazing films you'll love! ✨
      </Typography>
      <Paper 
        elevation={3}
        sx={{
          borderRadius: 3,
          overflow: 'hidden',
          border: '1px solid rgba(255, 255, 255, 0.2)',
        }}
      >
        <List sx={{ p: 0 }}>
          {recommendations.map((movie, index) => (
            <ListItem 
              key={index} 
              divider={index < recommendations.length - 1}
              sx={{
                transition: 'all 0.3s ease',
                '&:hover': {
                  backgroundColor: 'rgba(25, 118, 210, 0.08)',
                  transform: 'translateX(8px)',
                }
              }}
            >
              <Box sx={{ display: 'flex', alignItems: 'center', width: '100%' }}>
                <Chip
                  icon={<StarIcon />}
                  label={`#${index + 1}`}
                  size="small"
                  sx={{ 
                    mr: 2, 
                    background: 'linear-gradient(45deg, #1e3c72, #2a5298)',
                    color: 'white',
                    fontWeight: 'bold'
                  }}
                />
                <MovieIcon sx={{ mr: 2, color: 'primary.main' }} />
                <ListItemText 
                  primary={movie}
                  primaryTypographyProps={{
                    sx: { fontWeight: 500, fontSize: '1.1rem' }
                  }}
                />
              </Box>
            </ListItem>
          ))}
        </List>
      </Paper>
    </Box>
  );
};

export default Recommendations; 