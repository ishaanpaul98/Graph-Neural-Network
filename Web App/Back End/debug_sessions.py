#!/usr/bin/env python3
"""
Debug script for session management issues
=========================================
This script helps identify and fix session-related problems.
"""

import json
import os
from datetime import datetime
from session_manager import session_manager

def check_sessions():
    """Check current session state"""
    print("🔍 Checking Session State")
    print("=" * 40)
    
    # Check if sessions file exists
    if os.path.exists('sessions.json'):
        print("✅ sessions.json file exists")
        with open('sessions.json', 'r') as f:
            sessions = json.load(f)
            print(f"📊 Found {len(sessions)} sessions")
            
            for session_id, session_data in sessions.items():
                expires_at = datetime.fromisoformat(session_data['expires_at'])
                created_at = datetime.fromisoformat(session_data['created_at'])
                is_expired = datetime.now() > expires_at
                
                print(f"\nSession: {session_id[:10]}...")
                print(f"  Created: {created_at}")
                print(f"  Expires: {expires_at}")
                print(f"  Status: {'❌ EXPIRED' if is_expired else '✅ VALID'}")
                print(f"  Has access token: {'✅' if session_data.get('access_token') else '❌'}")
                print(f"  Has refresh token: {'✅' if session_data.get('refresh_token') else '❌'}")
    else:
        print("❌ sessions.json file does not exist")
        print("💡 This means no sessions have been created yet")

def test_session_lookup():
    """Test session lookup with the problematic session IDs"""
    print("\n🔍 Testing Session Lookup")
    print("=" * 40)
    
    # Test the session IDs from the error
    test_session_ids = [
        "ZpzOBJhhZzsNyBfCYL-xNnpA_gmToQbhkIQM7M3_PSo",
        "FaF3FhaqmAoE8yFTimV-fZ3C8fGuCFRjf9LqI9IdbAg"
    ]
    
    for session_id in test_session_ids:
        print(f"\nTesting session: {session_id[:10]}...")
        access_token = session_manager.get_access_token(session_id)
        if access_token:
            print(f"✅ Session found with token: {access_token[:20]}...")
        else:
            print(f"❌ Session not found or expired")

def clear_expired_sessions():
    """Clear all expired sessions"""
    print("\n🧹 Clearing Expired Sessions")
    print("=" * 40)
    
    if os.path.exists('sessions.json'):
        with open('sessions.json', 'r') as f:
            sessions = json.load(f)
        
        expired_count = 0
        for session_id, session_data in sessions.items():
            expires_at = datetime.fromisoformat(session_data['expires_at'])
            if datetime.now() > expires_at:
                expired_count += 1
                print(f"Removing expired session: {session_id[:10]}...")
        
        if expired_count > 0:
            session_manager.cleanup_expired_sessions()
            print(f"✅ Removed {expired_count} expired sessions")
        else:
            print("✅ No expired sessions found")
    else:
        print("ℹ️ No sessions file to clean")

def create_test_session():
    """Create a test session for debugging"""
    print("\n🧪 Creating Test Session")
    print("=" * 40)
    
    # This is just for testing - you'll need real tokens from Trakt OAuth
    test_session_id = "test_session_123"
    test_access_token = "test_access_token"
    test_refresh_token = "test_refresh_token"
    expires_in = 3600  # 1 hour
    
    success = session_manager.create_session(
        session_id=test_session_id,
        access_token=test_access_token,
        refresh_token=test_refresh_token,
        expires_in=expires_in
    )
    
    if success:
        print(f"✅ Created test session: {test_session_id}")
        print("💡 Note: This is just for testing - you need real Trakt tokens")
    else:
        print("❌ Failed to create test session")

def main():
    """Main debug function"""
    print("🐛 Session Debug Tool")
    print("=" * 50)
    
    check_sessions()
    test_session_lookup()
    clear_expired_sessions()
    
    print("\n💡 Recommendations:")
    print("1. Clear your browser's localStorage for the app")
    print("2. Re-authenticate with Trakt to get a fresh session")
    print("3. Check that your Trakt OAuth app is properly configured")
    print("4. Verify your environment variables are set correctly")

if __name__ == "__main__":
    main()