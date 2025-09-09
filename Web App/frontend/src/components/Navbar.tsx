import React, { useState, useEffect } from 'react';
import {
  AppBar,
  Toolbar,
  Typography,
  Button,
  IconButton,
  Menu,
  MenuItem,
  Avatar,
  Box,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  CircularProgress,
  Alert,
  Divider,
  Chip
} from '@mui/material';
import {
  AccountCircle,
  Login,
  Logout,
  Movie,
  TrendingUp,
  Star
} from '@mui/icons-material';
import { sessionManager, SessionInfo } from '../utils/sessionManager';
import { API_URLS } from '../config/api';
import axios from 'axios';

interface NavbarProps {
  onAuthChange?: (isAuthenticated: boolean) => void;
}

const Navbar: React.FC<NavbarProps> = ({ onAuthChange }) => {
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  const [loginDialogOpen, setLoginDialogOpen] = useState(false);
  const [sessionInfo, setSessionInfo] = useState<SessionInfo | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Check authentication status on component mount
  useEffect(() => {
    const checkAuthStatus = async () => {
      const isValid = await sessionManager.validateSession();
      if (isValid) {
        const info = await sessionManager.getSessionStatus();
        setSessionInfo(info);
        onAuthChange?.(true);
      } else {
        onAuthChange?.(false);
      }
    };

    checkAuthStatus();
  }, [onAuthChange]);

  const handleMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };

  const handleMenuClose = () => {
    setAnchorEl(null);
  };

  const handleLoginClick = () => {
    setLoginDialogOpen(true);
    setError(null);
  };

  const handleLoginDialogClose = () => {
    setLoginDialogOpen(false);
    setError(null);
  };

  const handleTraktLogin = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await axios.get(API_URLS.TRAKT_AUTH);
      if (response.data?.auth_url) {
        // Open Trakt auth in popup window
        const popup = window.open(
          response.data.auth_url,
          'trakt_auth',
          'width=500,height=600,scrollbars=yes,resizable=yes'
        );

        // Listen for authentication success
        const handleMessage = (event: MessageEvent) => {
          if (event.data.type === 'TRAKT_AUTH_SUCCESS') {
            const sessionId = event.data.sessionId;
            sessionManager.setSession(sessionId);
            
            // Update session info
            sessionManager.getSessionStatus().then((info) => {
              setSessionInfo(info);
              onAuthChange?.(true);
            });
            
            setLoginDialogOpen(false);
            setError(null);
            popup?.close();
          }
        };

        window.addEventListener('message', handleMessage);
        
        // Clean up listener when popup closes
        const checkClosed = setInterval(() => {
          if (popup?.closed) {
            clearInterval(checkClosed);
            window.removeEventListener('message', handleMessage);
            setLoading(false);
          }
        }, 1000);
      }
    } catch (error) {
      console.error('Login error:', error);
      setError('Failed to start authentication. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleLogout = async () => {
    try {
      await sessionManager.logout();
      setSessionInfo(null);
      onAuthChange?.(false);
      handleMenuClose();
    } catch (error) {
      console.error('Logout error:', error);
      // Still clear local state even if server logout fails
      setSessionInfo(null);
      onAuthChange?.(false);
      handleMenuClose();
    }
  };

  const isAuthenticated = sessionManager.isAuthenticated();

  return (
    <>
      <AppBar 
        position="static" 
        sx={{ 
          background: 'linear-gradient(45deg, #1e3c72, #2a5298, #4facfe)',
          boxShadow: '0 4px 20px rgba(0, 0, 0, 0.1)',
          backdropFilter: 'blur(10px)',
        }}
      >
        <Toolbar>
          {/* Logo/Brand */}
          <Box sx={{ display: 'flex', alignItems: 'center', mr: 2 }}>
            <Movie sx={{ mr: 1, fontSize: 28 }} />
            <Typography 
              variant="h6" 
              component="div" 
              sx={{ 
                fontWeight: 'bold',
                background: 'linear-gradient(45deg, #ffffff, #e3f2fd)',
                backgroundClip: 'text',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
              }}
            >
              MovieRec AI
            </Typography>
          </Box>

          {/* Spacer */}
          <Box sx={{ flexGrow: 1 }} />

          {/* Authentication Section */}
          {isAuthenticated ? (
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Chip
                icon={<Star />}
                label={`Welcome, ${sessionInfo?.user_info?.username || 'User'}`}
                color="secondary"
                variant="outlined"
                sx={{ 
                  color: 'white',
                  borderColor: 'rgba(255, 255, 255, 0.3)',
                  '& .MuiChip-icon': { color: 'white' }
                }}
              />
              <IconButton
                size="large"
                edge="end"
                aria-label="account of current user"
                aria-controls="menu-appbar"
                aria-haspopup="true"
                onClick={handleMenuOpen}
                color="inherit"
                sx={{ 
                  bgcolor: 'rgba(255, 255, 255, 0.1)',
                  '&:hover': { bgcolor: 'rgba(255, 255, 255, 0.2)' }
                }}
              >
                <AccountCircle />
              </IconButton>
            </Box>
          ) : (
            <Button
              color="inherit"
              startIcon={<Login />}
              onClick={handleLoginClick}
              sx={{
                bgcolor: 'rgba(255, 255, 255, 0.1)',
                '&:hover': { bgcolor: 'rgba(255, 255, 255, 0.2)' },
                borderRadius: 2,
                px: 3,
                py: 1
              }}
            >
              Login with Trakt
            </Button>
          )}

          {/* User Menu */}
          <Menu
            id="menu-appbar"
            anchorEl={anchorEl}
            anchorOrigin={{
              vertical: 'bottom',
              horizontal: 'right',
            }}
            keepMounted
            transformOrigin={{
              vertical: 'top',
              horizontal: 'right',
            }}
            open={Boolean(anchorEl)}
            onClose={handleMenuClose}
            PaperProps={{
              sx: {
                mt: 1,
                minWidth: 200,
                borderRadius: 2,
                boxShadow: '0 8px 32px rgba(0, 0, 0, 0.1)',
                backdropFilter: 'blur(10px)',
                backgroundColor: 'rgba(255, 255, 255, 0.9)',
              }
            }}
          >
            <MenuItem disabled>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <TrendingUp color="primary" />
                <Typography variant="body2" color="text.secondary">
                  Connected to Trakt
                </Typography>
              </Box>
            </MenuItem>
            <Divider />
            <MenuItem onClick={handleLogout}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Logout color="error" />
                <Typography color="error">Logout</Typography>
              </Box>
            </MenuItem>
          </Menu>
        </Toolbar>
      </AppBar>

      {/* Login Dialog */}
      <Dialog
        open={loginDialogOpen}
        onClose={handleLoginDialogClose}
        maxWidth="sm"
        fullWidth
        PaperProps={{
          sx: {
            borderRadius: 3,
            boxShadow: '0 8px 32px rgba(0, 0, 0, 0.1)',
            backdropFilter: 'blur(10px)',
            backgroundColor: 'rgba(255, 255, 255, 0.9)',
          }
        }}
      >
        <DialogTitle sx={{ textAlign: 'center', pb: 1 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 1 }}>
            <Movie color="primary" sx={{ fontSize: 32 }} />
            <Typography variant="h5" component="div" sx={{ fontWeight: 'bold' }}>
              Connect to Trakt
            </Typography>
          </Box>
        </DialogTitle>
        
        <DialogContent sx={{ textAlign: 'center', py: 2 }}>
          <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
            Connect your Trakt account to get personalized movie recommendations 
            based on your watch history and ratings.
          </Typography>
          
          {error && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {error}
            </Alert>
          )}
          
          <Box sx={{ 
            p: 3, 
            border: '2px dashed #e0e0e0', 
            borderRadius: 2,
            bgcolor: 'rgba(25, 118, 210, 0.05)',
            mb: 2
          }}>
            <Typography variant="h6" gutterBottom>
              What you'll get:
            </Typography>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1, textAlign: 'left' }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Star color="primary" />
                <Typography variant="body2">Personalized recommendations</Typography>
              </Box>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <TrendingUp color="primary" />
                <Typography variant="body2">Based on your watch history</Typography>
              </Box>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Movie color="primary" />
                <Typography variant="body2">Access to trending movies</Typography>
              </Box>
            </Box>
          </Box>
        </DialogContent>
        
        <DialogActions sx={{ justifyContent: 'center', pb: 3, px: 3 }}>
          <Button
            onClick={handleLoginDialogClose}
            variant="outlined"
            sx={{ mr: 2, minWidth: 100 }}
          >
            Cancel
          </Button>
          <Button
            onClick={handleTraktLogin}
            variant="contained"
            startIcon={loading ? <CircularProgress size={20} /> : <Login />}
            disabled={loading}
            sx={{ 
              minWidth: 150,
              background: 'linear-gradient(45deg, #1e3c72, #2a5298)',
              '&:hover': {
                background: 'linear-gradient(45deg, #2a5298, #4facfe)',
              }
            }}
          >
            {loading ? 'Connecting...' : 'Connect to Trakt'}
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
};

export default Navbar;
