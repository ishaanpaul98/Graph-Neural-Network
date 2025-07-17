# Trakt API Integration for Movie Recommendations

This integration adds Trakt API support to your movie recommendation system, allowing users to:
- Authenticate with their Trakt account via OAuth
- Search for any movie or TV show from Trakt's extensive database
- Get personalized recommendations based on their Trakt watch history and ratings
- Train the GNN model on real user data from Trakt

## Features

### 🔐 OAuth Authentication
- Secure OAuth 2.0 authentication with Trakt
- Automatic token refresh
- Session management

### 🔍 Advanced Search
- Search for movies and TV shows by title
- Real-time search results with ratings and metadata
- Support for both movies and TV shows

### 🎯 Personalized Recommendations
- Combines Trakt's recommendation engine with your GNN model
- Uses user's watch history and ratings for better personalization
- Fallback to GNN-only recommendations if Trakt data is unavailable

### 📊 Data Collection & Training
- Collect user ratings and watch history from Trakt
- Gather trending and popular movies
- Train the GNN model on real user data
- Comprehensive data preprocessing pipeline

## Setup Instructions

### 1. Create Trakt API Application

1. Go to [Trakt OAuth Applications](https://trakt.tv/oauth/applications/new)
2. Create a new application with the following settings:
   - **Name**: Your App Name (e.g., "Movie Recommendation App")
   - **Description**: Brief description of your app
   - **Redirect URI**: `http://localhost:8000/auth/callback`
   - **JavaScript Origins**: `http://localhost:5173`
   - **Scopes**: Select all scopes you need (recommended: `public`, `private`)

3. Note down your `Client ID` and `Client Secret`

### 2. Environment Variables

Create a `.env` file in the `Back End/` directory:

```bash
# Trakt API Configuration
TRAKT_CLIENT_ID=your_client_id_here
TRAKT_CLIENT_SECRET=your_client_secret_here
TRAKT_REDIRECT_URI=http://localhost:8000/auth/callback

# Flask Configuration
FLASK_SECRET_KEY=your_secret_key_here
```

### 3. Install Dependencies

The required dependencies are already included in your `requirements.txt`. Make sure to install them:

```bash
cd "Back End"
pip install -r requirements.txt
```

### 4. Start the Application

1. Start the backend server:
```bash
cd "Back End"
python app.py
```

2. Start the frontend development server:
```bash
cd frontend
npm install
npm run dev
```

3. Open your browser to `http://localhost:5173`

## Usage

### For End Users

1. **Authentication**: Click "Connect Trakt" to authenticate with your Trakt account
2. **Search Movies**: Use the search boxes to find movies and TV shows
3. **Select Movies**: Choose 5 movies you like (you can also click on trending movies)
4. **Get Recommendations**: Click "Get Recommendations" to receive personalized suggestions

### For Developers

#### Data Collection

To collect data from Trakt for training:

```python
from trakt_data_collector import data_collector

# Collect data for specific users
access_tokens = ['user1_access_token', 'user2_access_token']
usernames = ['user1', 'user2']

data_collector.collect_all_data_for_training(access_tokens, usernames)
```

#### Model Training

To train the GNN model with Trakt data:

```python
from train_with_trakt import train_gnn_with_trakt_data

# Train with collected data
success = train_gnn_with_trakt_data(
    data_dir='trakt_data',
    hidden_channels=64,
    learning_rate=0.001,
    epochs=100
)
```

#### API Endpoints

The following new endpoints are available:

- `GET /auth/trakt` - Get OAuth authorization URL
- `GET /auth/callback` - Handle OAuth callback
- `GET /api/trakt/search?query=<search_term>` - Search movies/shows
- `POST /api/trakt/recommend` - Get personalized recommendations
- `GET /api/trakt/trending` - Get trending movies
- `GET /api/trakt/popular` - Get popular movies
- `POST /api/trakt/collect-data` - Collect training data

## Architecture

### Backend Components

1. **`trakt_api.py`** - Trakt API client with OAuth support
2. **`session_manager.py`** - OAuth session management
3. **`trakt_data_collector.py`** - Data collection utilities
4. **`train_with_trakt.py`** - Training script for Trakt data
5. **Updated `app.py`** - Flask endpoints for Trakt integration

### Frontend Components

1. **`TraktMovieForm.tsx`** - New form component with Trakt integration
2. **Updated `App.tsx`** - Tabbed interface for both original and Trakt modes
3. **Updated `api.ts`** - New API endpoints configuration

### Data Flow

1. **Authentication**: User authenticates with Trakt via OAuth
2. **Search**: User searches for movies using Trakt's search API
3. **Selection**: User selects 5 movies from search results
4. **Recommendation**: System combines Trakt recommendations with GNN predictions
5. **Data Collection**: Optional collection of user data for training

## Training Your Model with Trakt Data

### Step 1: Collect Data

```python
# Collect data from multiple users
access_tokens = [
    'user1_access_token',
    'user2_access_token',
    'user3_access_token'
]

usernames = ['user1', 'user2', 'user3']

from trakt_data_collector import data_collector
data_collector.collect_all_data_for_training(access_tokens, usernames)
```

### Step 2: Train Model

```python
from train_with_trakt import train_gnn_with_trakt_data

success = train_gnn_with_trakt_data(
    data_dir='trakt_data',
    hidden_channels=64,
    learning_rate=0.001,
    epochs=100,
    save_path='models/trakt_gnn_model.pth'
)
```

### Step 3: Use Trained Model

The trained model will be saved with all necessary mappings and can be used by the application.

## Data Structure

The collected data is stored in the following format:

```
trakt_data/
├── users.csv              # User information
├── movies.csv             # Movie metadata
├── ratings.csv            # User ratings
├── dataset_summary.json   # Dataset statistics
└── training_summary.json  # Training results
```

## Security Considerations

1. **OAuth Tokens**: Access tokens are stored securely and automatically refreshed
2. **Session Management**: Sessions are managed server-side with proper expiration
3. **Rate Limiting**: Respect Trakt's API rate limits
4. **Data Privacy**: Only collect data with user consent

## Troubleshooting

### Common Issues

1. **OAuth Error**: Make sure your redirect URI matches exactly
2. **Search Not Working**: Check if Trakt API is accessible
3. **Training Fails**: Ensure you have sufficient data (at least 100 ratings)
4. **Model Loading Error**: Verify the model file path and format

### Debug Mode

Enable debug logging by setting environment variables:

```bash
export FLASK_DEBUG=1
export TRAKT_DEBUG=1
```

## API Rate Limits

Trakt API has the following rate limits:
- **Unauthenticated**: 1,000 requests per 5 minutes
- **Authenticated**: 1,000 requests per 5 minutes
- **POST/PUT/DELETE**: 1 request per second

The application includes rate limiting handling and will automatically retry when limits are exceeded.

## Contributing

To extend the Trakt integration:

1. Add new endpoints in `app.py`
2. Extend the `TraktAPI` class in `trakt_api.py`
3. Update the frontend components as needed
4. Add tests for new functionality

## License

This integration follows the same license as your original project.

## Support

For issues related to:
- **Trakt API**: Check [Trakt API Documentation](https://trakt.docs.apiary.io/)
- **OAuth**: Review [Trakt OAuth Guide](https://trakt.docs.apiary.io/#reference/authentication-oauth)
- **This Integration**: Check the troubleshooting section above 