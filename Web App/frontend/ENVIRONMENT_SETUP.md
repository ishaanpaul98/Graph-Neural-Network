# Environment Setup

This document explains how to set up environment variables for the Movie Recommendation Frontend.

## Setup Instructions

### 1. Create Environment File

Copy the example environment file and create your own:

```bash
cp env.example .env
```

### 2. Configure Your Environment Variables

Edit the `.env` file with your actual API endpoints:

```env
# API Configuration
VITE_API_BASE_URL=http://your-api-domain.com
VITE_API_MOVIES_ENDPOINT=/api/movies
VITE_API_RECOMMEND_ENDPOINT=/api/recommend

# Add other environment variables as needed
VITE_APP_NAME=Movie Recommendation App
```

### 3. Environment Variables Reference

| Variable | Description | Example |
|----------|-------------|---------|
| `VITE_API_BASE_URL` | Base URL for your API server | `http://api.ishaanpaul.com` |
| `VITE_API_MOVIES_ENDPOINT` | Endpoint for fetching movies | `/api/movies` |
| `VITE_API_RECOMMEND_ENDPOINT` | Endpoint for getting recommendations | `/api/recommend` |
| `VITE_APP_NAME` | Application name | `Movie Recommendation App` |

### 4. AWS Amplify Deployment

For AWS Amplify deployment, you need to set these environment variables in the Amplify console:

1. Go to your Amplify app dashboard
2. Navigate to **Environment variables**
3. Add each variable from your `.env` file
4. Redeploy your application

### 5. Security Notes

- **Never commit `.env` files to git** - they are already in `.gitignore`
- Use different environment variables for development, staging, and production
- Consider using AWS Secrets Manager for production deployments
- Rotate API keys and endpoints regularly

### 6. Development vs Production

- **Development**: Use `env.example` as a template for your local `.env`
- **Production**: Set environment variables in your deployment platform (AWS Amplify, Vercel, etc.)

## Troubleshooting

### Environment Variables Not Loading

1. Make sure your `.env` file is in the root of the frontend directory
2. Restart your development server after creating/modifying `.env`
3. Check that variable names start with `VITE_` (required for Vite)

### Build Issues

1. Ensure all required environment variables are set in your deployment platform
2. Check that the API endpoints are accessible from your deployment environment
3. Verify CORS settings on your backend allow requests from your frontend domain 