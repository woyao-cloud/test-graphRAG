import React, { useState } from 'react';
import { useAuth } from '../hooks/useAuth';
import { useTheme } from '../App';

const Header: React.FC = () => {
  const { user, isAuthenticated, login, logout, loading, error } = useAuth();
  const { theme, toggleTheme } = useTheme();
  const [showLogin, setShowLogin] = useState(false);
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      await login(username, password);
      setShowLogin(false);
      setUsername('');
      setPassword('');
    } catch {}
  };

  const currentRoute = window.location.hash.replace('#', '') || '/';

  return (
    <header className="header">
      <div className="header-logo">EnterpriseApp</div>

      {isAuthenticated && (
        <nav className="header-nav">
          <a href="#/" className={currentRoute === '/' ? 'active' : ''}>
            Dashboard
          </a>
          <a href="#/users" className={currentRoute === '/users' ? 'active' : ''}>
            Users
          </a>
          <a href="#/settings" className={currentRoute === '/settings' ? 'active' : ''}>
            Settings
          </a>
        </nav>
      )}

      <div className="header-actions">
        <button className="btn btn-ghost btn-sm" onClick={toggleTheme}>
          {theme === 'light' ? '\u{1F319} Dark' : '\u{2600}️ Light'}
        </button>

        {isAuthenticated ? (
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <span style={{ fontSize: 13, color: 'var(--text-secondary)' }}>
              {user?.name}
            </span>
            <button className="btn btn-ghost btn-sm" onClick={logout}>
              Logout
            </button>
          </div>
        ) : (
          <button className="btn btn-primary btn-sm" onClick={() => setShowLogin(true)}>
            Login
          </button>
        )}
      </div>

      {/* Login Modal */}
      {showLogin && (
        <div className="modal-overlay" onClick={() => setShowLogin(false)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>Login</h3>
              <button className="modal-close" onClick={() => setShowLogin(false)}>
                &times;
              </button>
            </div>
            <form onSubmit={handleLogin}>
              <div className="form-group">
                <label>Username</label>
                <input
                  className="form-input"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  placeholder="Enter username"
                />
              </div>
              <div className="form-group">
                <label>Password</label>
                <input
                  className="form-input"
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="Enter password"
                />
              </div>
              {error && <p className="form-error">{error}</p>}
              <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end', marginTop: 16 }}>
                <button type="button" className="btn btn-ghost" onClick={() => setShowLogin(false)}>
                  Cancel
                </button>
                <button type="submit" className="btn btn-primary" disabled={loading}>
                  {loading ? 'Logging in...' : 'Login'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </header>
  );
};

export default React.memo(Header);
