import json
import os
import time
from typing import Dict, Optional
from datetime import datetime, timedelta

class SessionManager:
    def __init__(self, storage_file: str = 'sessions.json'):
        self.storage_file = storage_file
        self.sessions = self._load_sessions()
        print(f"SessionManager initialized with {len(self.sessions)} existing sessions")
    
    def _load_sessions(self) -> Dict:
        """Load sessions from storage file"""
        if os.path.exists(self.storage_file):
            try:
                with open(self.storage_file, 'r') as f:
                    sessions = json.load(f)
                    print(f"Loaded {len(sessions)} sessions from {self.storage_file}")
                    return sessions
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading sessions: {e}")
                return {}
        else:
            print(f"Sessions file {self.storage_file} does not exist, starting with empty sessions")
            return {}
    
    def _save_sessions(self):
        """Save sessions to storage file"""
        try:
            with open(self.storage_file, 'w') as f:
                json.dump(self.sessions, f, indent=2)
            print(f"Saved {len(self.sessions)} sessions to {self.storage_file}")
        except IOError as e:
            print(f"Error saving sessions: {e}")
    
    def create_session(self, session_id: str, access_token: str, refresh_token: str, 
                      expires_in: int, user_info: Dict = None) -> bool:
        """Create a new session with OAuth tokens"""
        try:
            expires_at = datetime.now() + timedelta(seconds=expires_in)
            
            self.sessions[session_id] = {
                'access_token': access_token,
                'refresh_token': refresh_token,
                'expires_at': expires_at.isoformat(),
                'user_info': user_info or {},
                'created_at': datetime.now().isoformat()
            }
            
            self._save_sessions()
            print(f"Created session {session_id[:10]}... (expires: {expires_at})")
            return True
        except Exception as e:
            print(f"Error creating session: {e}")
            return False
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session data by session ID"""
        print(f"Looking up session: {session_id[:10]}...")
        
        if session_id not in self.sessions:
            print(f"Session {session_id[:10]}... not found in {len(self.sessions)} sessions")
            return None
        
        session = self.sessions[session_id]
        expires_at = datetime.fromisoformat(session['expires_at'])
        
        print(f"Session {session_id[:10]}... expires at {expires_at}")
        
        # Check if session has expired
        if datetime.now() > expires_at:
            print(f"Session {session_id[:10]}... has expired, attempting refresh")
            # Try to refresh the token
            if self._refresh_session_token(session_id):
                session = self.sessions[session_id]
                print(f"Session {session_id[:10]}... refreshed successfully")
            else:
                # Remove expired session
                print(f"Session {session_id[:10]}... refresh failed, removing")
                self.remove_session(session_id)
                return None
        else:
            print(f"Session {session_id[:10]}... is valid")
        
        return session
    
    def _refresh_session_token(self, session_id: str) -> bool:
        """Refresh the access token for a session"""
        try:
            from trakt_api import trakt_api
            
            session = self.sessions[session_id]
            refresh_token = session['refresh_token']
            
            print(f"Refreshing token for session {session_id[:10]}...")
            
            # Refresh the token
            token_data = trakt_api.refresh_token(refresh_token)
            
            # Update session with new tokens
            expires_at = datetime.now() + timedelta(seconds=token_data['expires_in'])
            
            self.sessions[session_id].update({
                'access_token': token_data['access_token'],
                'refresh_token': token_data.get('refresh_token', refresh_token),
                'expires_at': expires_at.isoformat()
            })
            
            self._save_sessions()
            print(f"Token refreshed successfully for session {session_id[:10]}...")
            return True
            
        except Exception as e:
            print(f"Error refreshing token for session {session_id[:10]}...: {e}")
            return False
    
    def get_access_token(self, session_id: str) -> Optional[str]:
        """Get valid access token for a session"""
        session = self.get_session(session_id)
        if session:
            print(f"Returning access token for session {session_id[:10]}...")
            return session['access_token']
        else:
            print(f"No valid session found for {session_id[:10]}...")
            return None
    
    def remove_session(self, session_id: str) -> bool:
        """Remove a session"""
        try:
            if session_id in self.sessions:
                del self.sessions[session_id]
                self._save_sessions()
            return True
        except Exception as e:
            print(f"Error removing session: {e}")
            return False
    
    def update_user_info(self, session_id: str, user_info: Dict) -> bool:
        """Update user information for a session"""
        try:
            if session_id in self.sessions:
                self.sessions[session_id]['user_info'] = user_info
                self._save_sessions()
                return True
            return False
        except Exception as e:
            print(f"Error updating user info: {e}")
            return False
    
    def get_all_sessions(self) -> Dict:
        """Get all active sessions"""
        # Clean up expired sessions first
        expired_sessions = []
        for session_id, session in self.sessions.items():
            expires_at = datetime.fromisoformat(session['expires_at'])
            if datetime.now() > expires_at:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            self.remove_session(session_id)
        
        return self.sessions.copy()
    
    def cleanup_expired_sessions(self):
        """Remove all expired sessions"""
        expired_sessions = []
        for session_id, session in self.sessions.items():
            expires_at = datetime.fromisoformat(session['expires_at'])
            if datetime.now() > expires_at:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            self.remove_session(session_id)

# Global session manager instance
session_manager = SessionManager() 