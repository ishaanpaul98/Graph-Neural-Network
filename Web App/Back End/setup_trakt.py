#!/usr/bin/env python3
"""
Trakt Integration Setup Script

This script helps you set up the Trakt API integration for your movie recommendation system.
"""

import os
import sys
import requests
from urllib.parse import urlencode
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def check_environment():
    """Check if required environment variables are set"""
    print("🔍 Checking environment variables...")
    
    required_vars = ['TRAKT_CLIENT_ID', 'TRAKT_CLIENT_SECRET']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        print("\nPlease create a .env file in the Back End directory with:")
        print("TRAKT_CLIENT_ID=your_client_id_here")
        print("TRAKT_CLIENT_SECRET=your_client_secret_here")
        print("TRAKT_REDIRECT_URI=http://localhost:8000/auth/callback")
        print("FLASK_SECRET_KEY=your_secret_key_here")
        return False
    
    print("✅ Environment variables are set")
    return True

def test_trakt_connection():
    """Test connection to Trakt API"""
    print("\n🔍 Testing Trakt API connection...")
    
    try:
        # Test basic API connection
        response = requests.get('https://api.trakt.tv/movies/trending', 
                              headers={
                                  'Content-Type': 'application/json',
                                  'trakt-api-version': '2',
                                  'trakt-api-key': os.getenv('TRAKT_CLIENT_ID'),
                                  'User-Agent': 'MovieRecommendationApp/1.0.0'
                              })
        
        if response.status_code == 200:
            print("✅ Trakt API connection successful")
            return True
        else:
            print(f"❌ Trakt API connection failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Trakt API connection error: {e}")
        return False

def test_oauth_flow():
    """Test OAuth flow setup"""
    print("\n🔍 Testing OAuth flow...")
    
    client_id = os.getenv('TRAKT_CLIENT_ID')
    redirect_uri = os.getenv('TRAKT_REDIRECT_URI', 'http://localhost:8000/auth/callback')
    
    # Generate authorization URL
    params = {
        'response_type': 'code',
        'client_id': client_id,
        'redirect_uri': redirect_uri
    }
    
    auth_url = f"https://api.trakt.tv/oauth/authorize?{urlencode(params)}"
    
    print(f"✅ OAuth authorization URL generated:")
    print(f"   {auth_url}")
    print(f"\n📝 To test the OAuth flow:")
    print(f"   1. Start your Flask server: python app.py")
    print(f"   2. Start your frontend: npm run dev")
    print(f"   3. Open http://localhost:5173")
    print(f"   4. Click 'Connect Trakt'")
    
    return True

def check_dependencies():
    """Check if required Python packages are installed"""
    print("\n🔍 Checking dependencies...")
    
    required_packages = [
        'requests',
        'pandas',
        'torch',
        'torch_geometric',
        'flask',
        'flask_cors'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install them with: pip install -r requirements.txt")
        return False
    
    print("✅ All required packages are installed")
    return True

def create_sample_env():
    """Create a sample .env file"""
    print("\n📝 Creating sample .env file...")
    
    env_content = """# Trakt API Configuration
TRAKT_CLIENT_ID=your_client_id_here
TRAKT_CLIENT_SECRET=your_client_secret_here
TRAKT_REDIRECT_URI=http://localhost:8000/auth/callback

# Flask Configuration
FLASK_SECRET_KEY=your_secret_key_here

# Optional: Debug mode
FLASK_DEBUG=1
"""
    
    env_file = '.env'
    if os.path.exists(env_file):
        print(f"⚠️  {env_file} already exists. Skipping creation.")
    else:
        with open(env_file, 'w') as f:
            f.write(env_content)
        print(f"✅ Created {env_file}")
        print("Please edit it with your actual Trakt API credentials")

def main():
    """Main setup function"""
    print("🚀 Trakt Integration Setup")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists('app.py'):
        print("❌ Please run this script from the Back End directory")
        sys.exit(1)
    
    # Create sample .env file
    create_sample_env()
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Setup incomplete. Please install missing dependencies.")
        sys.exit(1)
    
    # Check environment variables
    if not check_environment():
        print("\n❌ Setup incomplete. Please set environment variables.")
        sys.exit(1)
    
    # Test Trakt connection
    if not test_trakt_connection():
        print("\n❌ Setup incomplete. Trakt API connection failed.")
        sys.exit(1)
    
    # Test OAuth flow
    test_oauth_flow()
    
    print("\n✅ Setup completed successfully!")
    print("\n📋 Next steps:")
    print("   1. Edit the .env file with your Trakt API credentials")
    print("   2. Start the backend: python app.py")
    print("   3. Start the frontend: npm run dev")
    print("   4. Open http://localhost:5173 and test the integration")
    print("\n📚 For more information, see TRAKT_INTEGRATION_README.md")

if __name__ == "__main__":
    main() 