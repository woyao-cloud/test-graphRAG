import React, { useState, useEffect, useRef } from 'react';
import { useDebounce } from './hooks/useDebounce';
import { useThrottle } from './hooks/useThrottle';
import { usePrevious } from './hooks/usePrevious';
import { useLocalStorage } from './hooks/useLocalStorage';

const styles: Record<string, React.CSSProperties> = {
  container: {
    fontFamily: 'system-ui, sans-serif',
    maxWidth: 800,
    margin: '0 auto',
    padding: 24,
  },
  section: {
    border: '1px solid #e5e7eb',
    borderRadius: 8,
    padding: 16,
    marginBottom: 20,
    background: '#f9fafb',
  },
  h2: { fontSize: 18, margin: '0 0 12px 0', color: '#1f2937' },
  input: {
    padding: '8px 12px',
    borderRadius: 6,
    border: '1px solid #d1d5db',
    fontSize: 14,
    width: 280,
  },
  label: { fontSize: 13, color: '#6b7280', marginTop: 8 },
  value: { fontWeight: 600, fontFamily: 'monospace', fontSize: 14 },
  btn: {
    padding: '8px 16px',
    border: 'none',
    borderRadius: 6,
    cursor: 'pointer',
    fontWeight: 600,
    fontSize: 13,
    background: '#4f46e5',
    color: '#fff',
    marginTop: 8,
  },
};

/* ── Section 1: Debounced Search ── */
function DebouncedSearch() {
  const [query, setQuery] = useState('');
  const debouncedQuery = useDebounce(query, 300);
  const apiCallCount = useRef(0);
  const [results, setResults] = useState<string[]>([]);

  useEffect(() => {
    if (!debouncedQuery.trim()) {
      setResults([]);
      return;
    }
    apiCallCount.current += 1;
    const id = setTimeout(() => {
      setResults(
        Array.from({ length: 3 }, (_, i) => `Result ${i + 1} for "${debouncedQuery}"`)
      );
    }, 400);
    return () => clearTimeout(id);
  }, [debouncedQuery]);

  return (
    <div style={styles.section}>
      <h2 style={styles.h2}>Debounced Search (300ms)</h2>
      <input
        style={styles.input}
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Type to search..."
      />
      <p style={styles.label}>
        Raw: <span style={styles.value}>"{query}"</span> | Debounced:{' '}
        <span style={styles.value}>"{debouncedQuery}"</span>
      </p>
      <p style={styles.label}>
        API calls made: <strong>{apiCallCount.current}</strong>
      </p>
      <ul style={{ marginTop: 8, fontSize: 14 }}>
        {results.map((r, i) => (
          <li key={i}>{r}</li>
        ))}
      </ul>
    </div>
  );
}

/* ── Section 2: Throttled Scroll Position ── */
function ThrottledScroll() {
  const [scrollY, setScrollY] = useState(0);
  const throttledScrollY = useThrottle(scrollY, 200);

  useEffect(() => {
    const handleScroll = () => setScrollY(window.scrollY);
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <div style={styles.section}>
      <h2 style={styles.h2}>Throttled Scroll (200ms)</h2>
      <p style={styles.label}>
        Raw scrollY: <span style={styles.value}>{scrollY}px</span>
      </p>
      <p style={styles.label}>
        Throttled scrollY:{' '}
        <span style={styles.value}>{throttledScrollY}px</span>
      </p>
      <p style={{ fontSize: 12, color: '#9ca3af' }}>
        Scroll the page to see throttling in action
      </p>
    </div>
  );
}

/* ── Section 3: Previous Value ── */
function PreviousValueDemo() {
  const [current, setCurrent] = useState('');
  const prev = usePrevious(current);

  return (
    <div style={styles.section}>
      <h2 style={styles.h2}>Previous Value</h2>
      <input
        style={styles.input}
        value={current}
        onChange={(e) => setCurrent(e.target.value)}
        placeholder="Type to see previous value..."
      />
      <p style={styles.label}>
        Current: <span style={styles.value}>"{current}"</span>
      </p>
      <p style={styles.label}>
        Previous:{' '}
        <span style={styles.value}>"{prev ?? '(none)'}"</span>
      </p>
    </div>
  );
}

/* ── Section 4: Theme Toggle (localStorage) ── */
function ThemeToggle() {
  const [theme, setTheme] = useLocalStorage<'light' | 'dark'>(
    'demo-theme',
    'light'
  );

  useEffect(() => {
    document.documentElement.style.backgroundColor =
      theme === 'dark' ? '#1f2937' : '#fff';
    document.documentElement.style.color =
      theme === 'dark' ? '#f3f4f6' : '#1f2937';
  }, [theme]);

  return (
    <div style={styles.section}>
      <h2 style={styles.h2}>Theme Toggle (persisted to localStorage)</h2>
      <p>
        Current theme: <strong>{theme}</strong>
      </p>
      <button
        style={styles.btn}
        onClick={() => setTheme(theme === 'light' ? 'dark' : 'light')}
      >
        Toggle Theme
      </button>
      <p style={{ fontSize: 12, color: '#9ca3af', marginTop: 8 }}>
        Theme is saved to localStorage. Refresh the page to see it persist.
      </p>
    </div>
  );
}

/* ── App ── */
function App() {
  return (
    <div style={styles.container}>
      <h1 style={{ textAlign: 'center', marginBottom: 4 }}>
        Advanced Hooks
      </h1>
      <p style={{ textAlign: 'center', color: '#666', marginBottom: 24, fontSize: 14 }}>
        useDebounce, useThrottle, usePrevious, useLocalStorage
      </p>
      <DebouncedSearch />
      <ThrottledScroll />
      <PreviousValueDemo />
      <ThemeToggle />
    </div>
  );
}

export default App;
