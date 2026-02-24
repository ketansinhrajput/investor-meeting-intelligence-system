/**
 * Authentication Context
 *
 * Manages auth state (token, user info) and provides login/logout actions.
 * Listens for 401 events to auto-logout on expired tokens.
 */

import {
  createContext,
  useContext,
  useState,
  useEffect,
  useCallback,
  type ReactNode,
} from 'react';

import { login as apiLogin, logout as apiLogout, getAuthToken, clearAuthToken } from '@/api/client';
import type { AuthUser, LoginRequest } from '@/types/api';

interface AuthContextType {
  user: AuthUser | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  login: (credentials: LoginRequest) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType | null>(null);

/**
 * Decode JWT payload without verification (client-side only).
 * Returns null if the token is malformed.
 */
function decodeTokenPayload(token: string): { sub: string; role: string } | null {
  try {
    const parts = token.split('.');
    if (parts.length !== 3) return null;
    const payload = JSON.parse(atob(parts[1]));
    if (!payload.sub) return null;
    return payload;
  } catch {
    return null;
  }
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // Check existing token on mount
  useEffect(() => {
    const token = getAuthToken();
    if (token) {
      const payload = decodeTokenPayload(token);
      if (payload) {
        // Check if token is expired
        const decoded = JSON.parse(atob(token.split('.')[1]));
        const now = Math.floor(Date.now() / 1000);
        if (decoded.exp && decoded.exp > now) {
          setUser({ username: payload.sub, role: payload.role });
        } else {
          clearAuthToken();
        }
      } else {
        clearAuthToken();
      }
    }
    setIsLoading(false);
  }, []);

  // Listen for auth:logout events (from 401 interceptor)
  useEffect(() => {
    function handleLogout() {
      setUser(null);
    }
    window.addEventListener('auth:logout', handleLogout);
    return () => window.removeEventListener('auth:logout', handleLogout);
  }, []);

  const login = useCallback(async (credentials: LoginRequest) => {
    const response = await apiLogin(credentials);
    setUser({ username: response.username, role: response.role });
  }, []);

  const logout = useCallback(() => {
    apiLogout();
    setUser(null);
  }, []);

  return (
    <AuthContext.Provider
      value={{
        user,
        isAuthenticated: user !== null,
        isLoading,
        login,
        logout,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth(): AuthContextType {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}
