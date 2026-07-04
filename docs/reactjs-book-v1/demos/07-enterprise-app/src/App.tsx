import React, { useState, useEffect, lazy, Suspense, createContext, useContext, useCallback, useMemo } from 'react';
import { AuthProvider, useAuth } from './hooks/useAuth';
import { ToastProvider } from './hooks/useToast';
import Header from './components/Header';
import Sidebar from './components/Sidebar';
import ErrorBoundary from './components/ErrorBoundary';
import ToastContainer from './components/Toast';

const Dashboard = lazy(() => import('./pages/Dashboard'));
const UserManagement = lazy(() => import('./pages/UserManagement'));
const Settings = lazy(() => import('./pages/Settings'));
const NotFound = lazy(() => import('./pages/NotFound'));

/* ── Theme Context ── */
interface ThemeContextType {
  theme: 'light' | 'dark';
  toggleTheme: () => void;
}
const ThemeContext = createContext<ThemeContextType>({ theme: 'light', toggleTheme: () => {} });
export const useTheme = () => useContext(ThemeContext);

function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [theme, setTheme] = useState<'light' | 'dark'>(() => {
    try {
      return (localStorage.getItem('app-theme') as 'light' | 'dark') || 'light';
    } catch {
      return 'light';
    }
  });

  const toggleTheme = useCallback(() => {
    setTheme((t) => {
      const next = t === 'light' ? 'dark' : 'light';
      try { localStorage.setItem('app-theme', next); } catch {}
      return next;
    });
  }, []);

  useEffect(() => {
    document.documentElement.className = theme;
  }, [theme]);

  return <ThemeContext.Provider value={{ theme, toggleTheme }}>{children}</ThemeContext.Provider>;
}

/* ── Router ── */
function getRoute(): string {
  return window.location.hash.replace('#', '') || '/';
}

function Router({ children }: { children: (route: string) => React.ReactNode }) {
  const [route, setRoute] = useState(getRoute);

  useEffect(() => {
    const handler = () => setRoute(getRoute());
    window.addEventListener('hashchange', handler);
    return () => window.removeEventListener('hashchange', handler);
  }, []);

  return <>{children(route)}</>;
}

/* ── Layout ── */
function Layout({ route }: { route: string }) {
  const { user } = useAuth();
  const showSidebar = !!user;

  return (
    <div className="app-layout">
      <Header />
      <div className="app-body">
        {showSidebar && <Sidebar />}
        <main className="app-content">
          <Suspense fallback={<div className="loading">Loading...</div>}>
            {route === '/' && <Dashboard />}
            {route === '/users' && <UserManagement />}
            {route === '/settings' && <Settings />}
            {route !== '/' && route !== '/users' && route !== '/settings' && <NotFound />}
          </Suspense>
        </main>
      </div>
    </div>
  );
}

/* ── App ── */
function App() {
  return (
    <ThemeProvider>
      <AuthProvider>
        <ToastProvider>
          <ErrorBoundary>
            <Router>{(route) => <Layout route={route} />}</Router>
          </ErrorBoundary>
          <ToastContainer />
        </ToastProvider>
      </AuthProvider>
    </ThemeProvider>
  );
}

export default App;
