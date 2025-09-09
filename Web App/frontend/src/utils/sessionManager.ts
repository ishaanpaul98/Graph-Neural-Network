// Session management utility for frontend
import axios from 'axios';
import { API_URLS } from '../config/api';

export interface SessionInfo {
  authenticated: boolean;
  session_id?: string;
  user_info?: {
    username?: string;
    [key: string]: any;
  };
  expires_at?: string;
  created_at?: string;
}

class SessionManager {
  private sessionId: string | null = null;
  private sessionInfo: SessionInfo | null = null;

  constructor() {
    // Load session from localStorage on initialization
    this.loadSessionFromStorage();
  }

  private loadSessionFromStorage(): void {
    const storedSessionId = localStorage.getItem('trakt_session_id');
    if (storedSessionId) {
      this.sessionId = storedSessionId;
      this.validateSession();
    }
  }

  private saveSessionToStorage(): void {
    if (this.sessionId) {
      localStorage.setItem('trakt_session_id', this.sessionId);
    } else {
      localStorage.removeItem('trakt_session_id');
    }
  }

  public async validateSession(): Promise<boolean> {
    if (!this.sessionId) {
      return false;
    }

    try {
      const response = await axios.get(API_URLS.SESSION_VALIDATE, {
        headers: {
          'X-Session-ID': this.sessionId,
          'Content-Type': 'application/json',
        },
        withCredentials: true,
      });

      if (response.data.authenticated) {
        this.sessionInfo = response.data;
        return true;
      } else {
        this.clearSession();
        return false;
      }
    } catch (error) {
      console.error('Session validation failed:', error);
      this.clearSession();
      return false;
    }
  }

  public async getSessionStatus(): Promise<SessionInfo | null> {
    if (!this.sessionId) {
      return null;
    }

    try {
      const response = await axios.get(API_URLS.SESSION_STATUS, {
        headers: {
          'X-Session-ID': this.sessionId,
          'Content-Type': 'application/json',
        },
        withCredentials: true,
      });

      this.sessionInfo = response.data;
      return response.data;
    } catch (error) {
      console.error('Failed to get session status:', error);
      this.clearSession();
      return null;
    }
  }

  public async logout(): Promise<boolean> {
    if (!this.sessionId) {
      return true; // Already logged out
    }

    try {
      await axios.post(API_URLS.SESSION_LOGOUT, {}, {
        headers: {
          'X-Session-ID': this.sessionId,
          'Content-Type': 'application/json',
        },
        withCredentials: true,
      });

      this.clearSession();
      return true;
    } catch (error) {
      console.error('Logout failed:', error);
      // Clear session locally even if server logout fails
      this.clearSession();
      return false;
    }
  }

  public setSession(sessionId: string): void {
    this.sessionId = sessionId;
    this.saveSessionToStorage();
  }

  public getSessionId(): string | null {
    return this.sessionId;
  }

  public getSessionInfo(): SessionInfo | null {
    return this.sessionInfo;
  }

  public isAuthenticated(): boolean {
    return this.sessionId !== null && this.sessionInfo?.authenticated === true;
  }

  public clearSession(): void {
    this.sessionId = null;
    this.sessionInfo = null;
    this.saveSessionToStorage();
  }

  public getAuthHeaders(): Record<string, string> {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };

    if (this.sessionId) {
      headers['X-Session-ID'] = this.sessionId;
    }

    return headers;
  }
}

// Export singleton instance
export const sessionManager = new SessionManager();
export default sessionManager;
