// API Configuration
export const API_CONFIG = {
  BASE_URL: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000',
  ENDPOINTS: {
    MOVIES: import.meta.env.VITE_API_MOVIES_ENDPOINT || '/api/movies',
    RECOMMEND: import.meta.env.VITE_API_RECOMMEND_ENDPOINT || '/api/recommend',
  }
};

// Helper function to build full API URLs
export const buildApiUrl = (endpoint: string): string => {
  return `${API_CONFIG.BASE_URL}${endpoint}`;
};

// Pre-built URLs for common endpoints
export const API_URLS = {
  MOVIES: buildApiUrl(API_CONFIG.ENDPOINTS.MOVIES),
  RECOMMEND: buildApiUrl(API_CONFIG.ENDPOINTS.RECOMMEND),
}; 