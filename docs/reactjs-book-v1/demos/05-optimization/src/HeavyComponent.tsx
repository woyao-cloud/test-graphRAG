import React, { useMemo } from 'react';

/* Simulates an expensive component for code-splitting demo */
const HeavyComponent: React.FC = () => {
  const items = useMemo(() => {
    const arr: number[] = [];
    for (let i = 0; i < 5000; i++) {
      arr.push(i);
    }
    return arr;
  }, []);

  /* Busy-work to simulate heaviness */
  const start = performance.now();
  let x = 0;
  for (let i = 0; i < 5000000; i++) {
    x += Math.sqrt(i);
  }
  const elapsed = (performance.now() - start).toFixed(1);

  return (
    <div
      style={{
        border: '2px solid #f59e0b',
        borderRadius: 8,
        padding: 16,
        marginTop: 12,
        background: '#fffbeb',
      }}
    >
      <h3 style={{ margin: '0 0 8px 0', color: '#92400e' }}>
        Heavy Component (Lazy Loaded)
      </h3>
      <p style={{ fontSize: 13, margin: 0 }}>
        This component simulates heavy computation:{' '}
        <strong>{elapsed}ms</strong> of busy work + rendered{' '}
        <strong>{items.length}</strong> items.
      </p>
      <p style={{ fontSize: 12, color: '#92400e', marginTop: 4 }}>
        It was loaded via <code>React.lazy()</code> with a Suspense fallback.
      </p>
    </div>
  );
};

export default HeavyComponent;
