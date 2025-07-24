// API Configuration
export const API_CONFIG = {
  // Your EC2 Backend API (handles both your GNN model and Trakt API proxying)
  BACKEND_BASE_URL: import.meta.env.VITE_BACKEND_BASE_URL || 'https://api.ishaanpaul.com',
  
  ENDPOINTS: {
    // Your EC2 Backend endpoints (GNN model)
    MOVIES: '/api/movies',
    RECOMMEND: '/api/recommend',
    
    // Trakt API endpoints (proxied through your backend)
    TRAKT_AUTH: '/auth/trakt',
    TRAKT_SEARCH: '/api/trakt/search',
    TRAKT_RECOMMEND: '/api/trakt/recommend',
    TRAKT_TRENDING: '/api/trakt/trending',
    TRAKT_POPULAR: '/api/trakt/popular',
    TRAKT_USER_HISTORY: '/api/trakt/user-history',
  }
};

// Helper function to build full API URLs
export const buildApiUrl = (endpoint: string): string => {
  return `${API_CONFIG.BACKEND_BASE_URL}${endpoint}`;
};

// Pre-built URLs for common endpoints
export const API_URLS = {
  // Your EC2 Backend endpoints (GNN model)
  MOVIES: buildApiUrl(API_CONFIG.ENDPOINTS.MOVIES),
  RECOMMEND: buildApiUrl(API_CONFIG.ENDPOINTS.RECOMMEND),
  
  // Trakt API endpoints (proxied through your backend)
  TRAKT_AUTH: buildApiUrl(API_CONFIG.ENDPOINTS.TRAKT_AUTH),
  TRAKT_SEARCH: buildApiUrl(API_CONFIG.ENDPOINTS.TRAKT_SEARCH),
  TRAKT_RECOMMEND: buildApiUrl(API_CONFIG.ENDPOINTS.TRAKT_RECOMMEND),
  TRAKT_TRENDING: buildApiUrl(API_CONFIG.ENDPOINTS.TRAKT_TRENDING),
  TRAKT_POPULAR: buildApiUrl(API_CONFIG.ENDPOINTS.TRAKT_POPULAR),
  TRAKT_USER_HISTORY: buildApiUrl(API_CONFIG.ENDPOINTS.TRAKT_USER_HISTORY),
}; 