import React, { useEffect, useState } from 'react';
import { Box, Typography, CircularProgress, Button } from '@mui/material';
import { CheckCircle, Error } from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { sessionManager } from '../utils/sessionManager';

const AuthSuccess: React.FC = () => {
  const [status, setStatus] = useState<'loading' | 'success' | 'error'>('loading');
  const [message, setMessage] = useState('');
  const navigate = useNavigate();

  useEffect(() => {
    // Get session ID from URL parameters
    const urlParams = new URLSearchParams(window.location.search);
    const sessionId = urlParams.get('session_id');
    
    if (sessionId) {
      // Store session ID using session manager
      sessionManager.setSession(sessionId);
      setStatus('success');
      setMessage('Successfully connected to Trakt!');
      
      // Close window if it's a popup, otherwise redirect
      setTimeout(() => {
        if (window.opener) {
          window.opener.postMessage({ type: 'TRAKT_AUTH_SUCCESS', sessionId }, '*');
          window.close();
        } else {
          navigate('/');
        }
      }, 2000);
    } else {
      setStatus('error');
      setMessage('Authentication failed. Please try again.');
    }
  }, [navigate]);

  const handleClose = () => {
    if (window.opener) {
      window.close();
    } else {
      navigate('/');
    }
  };

  return (
    <Box sx={{ 
      display: 'flex', 
      flexDirection: 'column', 
      alignItems: 'center', 
      justifyContent: 'center', 
      minHeight: '100vh',
      p: 3
    }}>
      {status === 'loading' && (
        <>
          <CircularProgress size={60} sx={{ mb: 2 }} />
          <Typography variant="h6">Completing authentication...</Typography>
        </>
      )}
      
      {status === 'success' && (
        <>
          <CheckCircle color="success" sx={{ fontSize: 60, mb: 2 }} />
          <Typography variant="h6" color="success.main" gutterBottom>
            Authentication Successful!
          </Typography>
          <Typography variant="body1" align="center" sx={{ mb: 3 }}>
            {message}
          </Typography>
          <Button variant="contained" onClick={handleClose}>
            Continue
          </Button>
        </>
      )}
      
      {status === 'error' && (
        <>
          <Error color="error" sx={{ fontSize: 60, mb: 2 }} />
          <Typography variant="h6" color="error.main" gutterBottom>
            Authentication Failed
          </Typography>
          <Typography variant="body1" align="center" sx={{ mb: 3 }}>
            {message}
          </Typography>
          <Button variant="contained" onClick={handleClose}>
            Try Again
          </Button>
        </>
      )}
    </Box>
  );
};

export default AuthSuccess; 