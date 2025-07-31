#!/usr/bin/env python3
"""
Debug script for recommendation API issues
==========================================
This script helps identify why the recommendation API is returning empty responses.
"""

import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_recommendation_api():
    """Test the recommendation API endpoint"""
    
    # Test data
    test_movies = ['The Fantastic Four: First Steps', 'How to Train Your Dragon', 'Lilo & Stitch']
    
    # API endpoint
    api_url = "https://api.ishaanpaul.com/api/trakt/recommend"
    
    # Test session ID (you'll need to replace this with a valid one)
    session_id = "ZpzOBJhhZzsNyBfCYL-xNnpA_gmToQbhkIQM7M3_PSo"
    
    # Request data
    data = {
        "movies": test_movies
    }
    
    headers = {
        "Content-Type": "application/json",
        "X-Session-ID": session_id
    }
    
    print("🔍 Testing Recommendation API")
    print("=" * 50)
    print(f"API URL: {api_url}")
    print(f"Session ID: {session_id}")
    print(f"Movies: {test_movies}")
    print()
    
    try:
        # Make the request
        response = requests.post(api_url, json=data, headers=headers, timeout=30)
        
        print(f"Response Status: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        print(f"Response Body: {response.text}")
        
        if response.status_code == 200:
            try:
                result = response.json()
                print(f"\n✅ Parsed JSON Response:")
                print(json.dumps(result, indent=2))
                
                # Check for empty recommendations
                if 'recommendations' in result:
                    if not result['recommendations']:
                        print("\n❌ Empty recommendations found!")
                        print("Possible causes:")
                        print("1. Session ID is invalid or expired")
                        print("2. Trakt API is not returning recommendations")
                        print("3. GNN model is not working properly")
                        print("4. Movie mappings are incorrect")
                    else:
                        print(f"\n✅ Found {len(result['recommendations'])} recommendations")
                else:
                    print("\n❌ No 'recommendations' field in response")
                    
            except json.JSONDecodeError as e:
                print(f"\n❌ Failed to parse JSON response: {e}")
        else:
            print(f"\n❌ Request failed with status {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Request failed: {e}")

def test_health_endpoint():
    """Test the health endpoint"""
    print("\n🏥 Testing Health Endpoint")
    print("=" * 30)
    
    try:
        response = requests.get("https://api.ishaanpaul.com/api/health", timeout=10)
        print(f"Health Status: {response.status_code}")
        print(f"Health Response: {response.text}")
    except Exception as e:
        print(f"Health check failed: {e}")

def test_session_validation():
    """Test if the session ID is valid"""
    print("\n🔐 Testing Session Validation")
    print("=" * 35)
    
    session_id = "ZpzOBJhhZzsNyBfCYL-xNnpA_gmToQbhkIQM7M3_PSo"
    
    # Try to get user history (requires valid session)
    try:
        response = requests.get(
            f"https://api.ishaanpaul.com/api/trakt/user-history",
            headers={"X-Session-ID": session_id},
            timeout=10
        )
        print(f"Session Test Status: {response.status_code}")
        print(f"Session Test Response: {response.text}")
        
        if response.status_code == 401:
            print("❌ Session ID is invalid or expired")
        elif response.status_code == 200:
            print("✅ Session ID appears to be valid")
        else:
            print(f"⚠️ Unexpected status code: {response.status_code}")
            
    except Exception as e:
        print(f"Session test failed: {e}")

def main():
    """Main debugging function"""
    print("🚀 Recommendation API Debug Tool")
    print("=" * 40)
    
    # Test health endpoint first
    test_health_endpoint()
    
    # Test session validation
    test_session_validation()
    
    # Test recommendation API
    test_recommendation_api()
    
    print("\n" + "=" * 40)
    print("🔧 Debugging Complete")
    print("\nNext steps:")
    print("1. Check if the session ID is valid")
    print("2. Verify Trakt API credentials")
    print("3. Check server logs for errors")
    print("4. Ensure the model is properly loaded")

if __name__ == "__main__":
    main() 