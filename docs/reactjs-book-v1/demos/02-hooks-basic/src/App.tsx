import React, {
  useState,
  useEffect,
  useRef,
  useMemo,
  useCallback,
} from 'react';

/* ── Styles ── */
const s = {
  container: {
    fontFamily: 'system-ui, sans-serif',
    maxWidth: 900,
    margin: '0 auto',
    padding: 24,
  } as React.CSSProperties,
  section: {
    border: '1px solid #e5e7eb',
    borderRadius: 8,
    padding: 16,
    marginBottom: 20,
    background: '#f9fafb',
  } as React.CSSProperties,
  h2: { fontSize: 18, margin: '0 0 12px 0', color: '#1f2937' } as React.CSSProperties,
  btn: {
    padding: '8px 16px',
    border: 'none',
    borderRadius: 6,
    cursor: 'pointer',
    fontWeight: 600,
    fontSize: 13,
    background: '#4f46e5',
    color: '#fff',
    marginRight: 8,
    marginTop: 8,
  } as React.CSSProperties,
  code: {
    background: '#e5e7eb',
    padding: '2px 6px',
    borderRadius: 4,
    fontFamily: 'monospace',
    fontSize: 13,
  } as React.CSSProperties,
};

/* ── 1. useState + Auto-batching ── */
function UseStateDemo() {
  const [count, setCount] = useState(0);
  const [batchLog, setBatchLog] = useState('');

  const handleBatch = () => {
    setBatchLog('');
    // Three synchronous updates — React 18+ batches them into one render
    setCount((c) => c + 1);
    setCount((c) => c + 1);
    setCount((c) => c + 1);
    setBatchLog('Three setCount calls batched into one render');
  };

  return (
    <div style={s.section}>
      <h2 style={s.h2}>useState + Auto-batching</h2>
      <p>
        Count: <strong>{count}</strong>
      </p>
      <button style={s.btn} onClick={() => setCount(count + 1)}>
        +1
      </button>
      <button style={s.btn} onClick={handleBatch}>
        Batch +3 (sync)
      </button>
      {batchLog && (
        <p style={{ color: '#16a34a', fontSize: 13, marginTop: 8 }}>
          {batchLog}
        </p>
      )}
    </div>
  );
}

/* ── 2. useEffect ── */
function UseEffectDemo() {
  const [width, setWidth] = useState(window.innerWidth);
  const [dep, setDep] = useState(0);

  useEffect(() => {
    console.log('UseEffectDemo mounted');
    const handleResize = () => setWidth(window.innerWidth);
    window.addEventListener('resize', handleResize);
    return () => {
      console.log('UseEffectDemo unmounting');
      window.removeEventListener('resize', handleResize);
    };
  }, []);

  useEffect(() => {
    console.log(`useEffect with [dep] ran — dep = ${dep}`);
  }, [dep]);

  return (
    <div style={s.section}>
      <h2 style={s.h2}>useEffect</h2>
      <p>
        Window width: <strong>{width}px</strong> (resize browser to see)
      </p>
      <p>
        Dependency demo: <strong>{dep}</strong>
      </p>
      <button style={s.btn} onClick={() => setDep((d) => d + 1)}>
        Increment dep
      </button>
      <p style={{ fontSize: 12, color: '#6b7280' }}>
        Open console to see mount / unmount / dep-change logs
      </p>
    </div>
  );
}

/* ── 3. useRef ── */
function UseRefDemo() {
  const inputRef = useRef<HTMLInputElement>(null);
  const renderCount = useRef(0);
  const [value, setValue] = useState('');
  const prevValue = useRef('');

  renderCount.current += 1;

  useEffect(() => {
    prevValue.current = value;
  }, [value]);

  return (
    <div style={s.section}>
      <h2 style={s.h2}>useRef</h2>
      <p>
        Component renders: <strong>{renderCount.current}</strong>
      </p>
      <input
        ref={inputRef}
        value={value}
        onChange={(e) => setValue(e.target.value)}
        placeholder="Type something..."
        style={{ padding: 8, borderRadius: 4, border: '1px solid #d1d5db', width: 250 }}
      />
      <button style={s.btn} onClick={() => inputRef.current?.focus()}>
        Focus Input (useRef)
      </button>
      <p style={{ fontSize: 13 }}>
        Current: <strong>{value || '(empty)'}</strong> | Previous:{' '}
        <strong>{prevValue.current || '(empty)'}</strong>
      </p>
    </div>
  );
}

/* ── 4. useMemo (expensive Fibonacci) ── */
function fib(n: number): number {
  if (n <= 1) return n;
  return fib(n - 1) + fib(n - 2);
}

function UseMemoDemo() {
  const [num, setNum] = useState(38);
  const [toggle, setToggle] = useState(false);

  const startMemo = performance.now();
  const memoResult = useMemo(() => fib(num), [num]);
  const memoTime = performance.now() - startMemo;

  // Without useMemo — recalculates every render
  const noMemoStart = performance.now();
  const noMemoResult = fib(num);
  const noMemoTime = performance.now() - noMemoStart;

  return (
    <div style={s.section}>
      <h2 style={s.h2}>useMemo — Expensive Fibonacci</h2>
      <p>
        fib({num}) = <strong>{memoResult}</strong>
      </p>
      <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
        <button style={s.btn} onClick={() => setNum((n) => n + 1)}>
          +1
        </button>
        <button style={s.btn} onClick={() => setNum(38)}>
          Reset to 38
        </button>
        <button style={s.btn} onClick={() => setToggle((t) => !t)}>
          Toggle (re-render)
        </button>
      </div>
      <table
        style={{
          marginTop: 12,
          borderCollapse: 'collapse',
          fontSize: 13,
          width: '100%',
        }}
      >
        <thead>
          <tr>
            <th style={{ border: '1px solid #d1d5db', padding: 6 }}>Method</th>
            <th style={{ border: '1px solid #d1d5db', padding: 6 }}>Result</th>
            <th style={{ border: '1px solid #d1d5db', padding: 6 }}>Time</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td style={{ border: '1px solid #d1d5db', padding: 6 }}>
              With useMemo
            </td>
            <td style={{ border: '1px solid #d1d5db', padding: 6 }}>
              {memoResult}
            </td>
            <td style={{ border: '1px solid #d1d5db', padding: 6 }}>
              {memoTime.toFixed(2)}ms
            </td>
          </tr>
          <tr>
            <td style={{ border: '1px solid #d1d5db', padding: 6 }}>
              Without useMemo
            </td>
            <td style={{ border: '1px solid #d1d5db', padding: 6 }}>
              {noMemoResult}
            </td>
            <td style={{ border: '1px solid #d1d5db', padding: 6 }}>
              {noMemoTime.toFixed(2)}ms
            </td>
          </tr>
        </tbody>
      </table>
      <p style={{ fontSize: 12, color: '#6b7280', marginTop: 8 }}>
        Toggle re-renders the component. With useMemo, fib only recalculates
        when num changes.
      </p>
    </div>
  );
}

/* ── 5. useCallback ── */
const ChildList: React.FC<{ items: number[]; onRemove: (id: number) => void }> =
  React.memo(({ items, onRemove }) => {
    const renderCount = useRef(0);
    renderCount.current += 1;
    return (
      <div>
        <p style={{ fontSize: 13, color: '#6b7280' }}>
          Child renders: {renderCount.current}
        </p>
        <ul style={{ margin: 0, paddingLeft: 20 }}>
          {items.map((id) => (
            <li key={id} style={{ marginBottom: 4 }}>
              Item {id}{' '}
              <button
                onClick={() => onRemove(id)}
                style={{
                  padding: '2px 8px',
                  fontSize: 12,
                  border: '1px solid #ef4444',
                  borderRadius: 4,
                  color: '#ef4444',
                  cursor: 'pointer',
                  background: 'transparent',
                }}
              >
                Remove
              </button>
            </li>
          ))}
        </ul>
      </div>
    );
  });

function UseCallbackDemo() {
  const [items, setItems] = useState([1, 2, 3, 4, 5]);
  const [count, setCount] = useState(0);

  const addItem = useCallback(() => {
    setItems((prev) => [...prev, prev.length + 1]);
  }, []);

  const removeItem = useCallback((id: number) => {
    setItems((prev) => prev.filter((i) => i !== id));
  }, []);

  // Without useCallback (creates new function each render — breaks memo)
  const unstableRemoveItem = (id: number) => {
    setItems((prev) => prev.filter((i) => i !== id));
  };

  return (
    <div style={s.section}>
      <h2 style={s.h2}>useCallback — Re-render Prevention</h2>
      <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 8 }}>
        <button style={s.btn} onClick={addItem}>
          Add Item
        </button>
        <button style={s.btn} onClick={() => setCount((c) => c + 1)}>
          Re-render parent ({count})
        </button>
      </div>
      <p style={{ fontSize: 13, color: '#6b7280' }}>
        Click "Re-render parent" — Child with useCallback does NOT re-render.
        The "Unstable" version does.
      </p>
      <div style={{ display: 'flex', gap: 24 }}>
        <div>
          <p style={{ fontWeight: 600, fontSize: 13 }}>With useCallback</p>
          <ChildList items={items} onRemove={removeItem} />
        </div>
        <div>
          <p style={{ fontWeight: 600, fontSize: 13 }}>
            Without useCallback (unstable)
          </p>
          <ChildList items={items} onRemove={unstableRemoveItem} />
        </div>
      </div>
    </div>
  );
}

/* ── App ── */
function App() {
  return (
    <div style={s.container}>
      <h1 style={{ textAlign: 'center', marginBottom: 4 }}>
        React Hooks Playground
      </h1>
      <p style={{ textAlign: 'center', color: '#666', marginBottom: 24, fontSize: 14 }}>
        Interactive demos of useState, useEffect, useRef, useMemo, useCallback
      </p>
      <UseStateDemo />
      <UseEffectDemo />
      <UseRefDemo />
      <UseMemoDemo />
      <UseCallbackDemo />
    </div>
  );
}

export default App;
