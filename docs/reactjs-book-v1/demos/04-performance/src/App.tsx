import React, { useState, useRef, useCallback, useMemo, useEffect } from 'react';

/* ── Styles ── */
const s: Record<string, React.CSSProperties> = {
  container: { fontFamily: 'system-ui, sans-serif', maxWidth: 900, margin: '0 auto', padding: 24 },
  section: { border: '1px solid #e5e7eb', borderRadius: 8, padding: 16, marginBottom: 20, background: '#f9fafb' },
  h2: { fontSize: 18, margin: '0 0 12px 0', color: '#1f2937' },
  btn: { padding: '8px 16px', border: 'none', borderRadius: 6, cursor: 'pointer', fontWeight: 600, fontSize: 13, background: '#4f46e5', color: '#fff', marginRight: 8, marginTop: 8 },
  toggle: { padding: '8px 16px', border: '2px solid #4f46e5', borderRadius: 6, cursor: 'pointer', fontWeight: 600, fontSize: 13, background: '#fff', color: '#4f46e5', marginRight: 8, marginTop: 8 },
  toggleOn: { padding: '8px 16px', border: '2px solid #4f46e5', borderRadius: 6, cursor: 'pointer', fontWeight: 600, fontSize: 13, background: '#4f46e5', color: '#fff', marginRight: 8, marginTop: 8 },
};

/* ── Generate items ── */
function generateItems(count: number, offset = 0) {
  return Array.from({ length: count }, (_, i) => ({
    id: offset + i,
    label: `Item #${offset + i}`,
    color: `hsl(${(offset + i) * 37 % 360}, 60%, 70%)`,
  }));
}

/* ── ListItem — with and without React.memo ── */
const MemoizedListItem = React.memo<{ item: { id: number; label: string; color: string }; onRender: () => void }>(
  ({ item, onRender }) => {
    const renders = useRef(0);
    renders.current += 1;
    useEffect(() => { onRender(); });
    return (
      <div style={{ display: 'flex', alignItems: 'center', padding: '4px 8px', borderBottom: '1px solid #f3f4f6', fontSize: 13, background: item.color }}>
        <span style={{ flex: 1 }}>{item.label}</span>
        <span style={{ fontSize: 11, color: '#6b7280' }}>R: {renders.current}</span>
      </div>
    );
  }
);

function PlainListItem({ item, onRender }: { item: { id: number; label: string; color: string }; onRender: () => void }) {
  const renders = useRef(0);
  renders.current += 1;
  useEffect(() => { onRender(); });
  return (
    <div style={{ display: 'flex', alignItems: 'center', padding: '4px 8px', borderBottom: '1px solid #f3f4f6', fontSize: 13, background: item.color }}>
      <span style={{ flex: 1 }}>{item.label}</span>
      <span style={{ fontSize: 11, color: '#6b7280' }}>R: {renders.current}</span>
    </div>
  );
}

/* ── Section 1: React.memo comparison ── */
function MemoComparison() {
  const [useMemo_, setUseMemo_] = useState(false);
  const [toggle, setToggle] = useState(false);
  const items = useMemo(() => generateItems(100), []);
  const totalRenders = useRef(0);

  const countRender = useCallback(() => { totalRenders.current += 1; }, []);

  const ListComponent = useMemo_ ? MemoizedListItem : PlainListItem;

  return (
    <div style={s.section}>
      <h2 style={s.h2}>React.memo Comparison (100 items)</h2>
      <button style={useMemo_ ? s.toggleOn : s.toggle} onClick={() => setUseMemo_((v) => !v)}>
        {useMemo_ ? 'With React.memo' : 'Without React.memo'}
      </button>
      <button style={s.btn} onClick={() => setToggle((t) => !t)}>
        Force Re-render (toggle: {String(toggle)})
      </button>
      <p style={{ fontSize: 13, color: '#6b7280' }}>
        Total item renders this session: <strong>{totalRenders.current}</strong>
      </p>
      <div style={{ maxHeight: 300, overflowY: 'auto', border: '1px solid #e5e7eb', borderRadius: 4, marginTop: 8 }}>
        {items.map((item) => (
          <ListComponent key={item.id} item={item} onRender={countRender} />
        ))}
      </div>
    </div>
  );
}

/* ── Section 2: Key optimization ── */
function KeyOptimization() {
  const [items, setItems] = useState(() => generateItems(10));
  const [useStableKey, setUseStableKey] = useState(true);

  const addToFront = () => {
    const newId = Date.now();
    setItems((prev) => [{ id: newId, label: `New #${newId}`, color: `hsl(${newId % 360}, 70%, 80%)` }, ...prev]);
  };

  return (
    <div style={s.section}>
      <h2 style={s.h2}>Key Optimization</h2>
      <button style={s.btn} onClick={addToFront}>Add Item to Front</button>
      <button style={useStableKey ? s.toggleOn : s.toggle} onClick={() => setUseStableKey((v) => !v)}>
        {useStableKey ? 'Stable Keys (id)' : 'Index Keys'}
      </button>
      <p style={{ fontSize: 12, color: '#9ca3af', marginTop: 4 }}>
        {useStableKey
          ? 'Using stable keys — React reuses existing DOM nodes when adding to front.'
          : 'Using index keys — React re-renders ALL items when adding to front.'}
      </p>
      <div style={{ maxHeight: 300, overflowY: 'auto', border: '1px solid #e5e7eb', borderRadius: 4, marginTop: 8 }}>
        {items.map((item, idx) => {
          const ListItem = useStableKey ? MemoizedListItem : PlainListItem;
          return (
            <ListItem
              key={useStableKey ? item.id : idx}
              item={item}
              onRender={() => {}}
            />
          );
        })}
      </div>
    </div>
  );
}

/* ── Section 3: Simple Virtual List ── */
function VirtualList() {
  const allItems = useMemo(() => generateItems(1000), []);
  const ITEM_HEIGHT = 32;
  const CONTAINER_HEIGHT = 400;
  const [scrollTop, setScrollTop] = useState(0);
  const containerRef = useRef<HTMLDivElement>(null);

  const startIdx = Math.floor(scrollTop / ITEM_HEIGHT);
  const endIdx = Math.min(startIdx + Math.ceil(CONTAINER_HEIGHT / ITEM_HEIGHT) + 1, allItems.length);
  const visibleItems = allItems.slice(startIdx, endIdx);
  const totalRenderTime = useRef(0);

  const handleScroll = useCallback(() => {
    if (containerRef.current) {
      const t0 = performance.now();
      setScrollTop(containerRef.current.scrollTop);
      totalRenderTime.current += performance.now() - t0;
    }
  }, []);

  return (
    <div style={s.section}>
      <h2 style={s.h2}>Simple Virtual List (1000 items)</h2>
      <p style={{ fontSize: 13, color: '#6b7280' }}>
        Rendering {visibleItems.length} of {allItems.length} items | Total scroll compute: {totalRenderTime.current.toFixed(1)}ms
      </p>
      <div
        ref={containerRef}
        onScroll={handleScroll}
        style={{ height: CONTAINER_HEIGHT, overflowY: 'auto', border: '1px solid #e5e7eb', borderRadius: 4 }}
      >
        <div style={{ height: allItems.length * ITEM_HEIGHT, position: 'relative' }}>
          <div style={{ position: 'absolute', top: startIdx * ITEM_HEIGHT, left: 0, right: 0 }}>
            {visibleItems.map((item) => (
              <div
                key={item.id}
                style={{ height: ITEM_HEIGHT, display: 'flex', alignItems: 'center', padding: '0 8px', fontSize: 13, borderBottom: '1px solid #f3f4f6', background: item.color }}
              >
                {item.label}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

/* ── App ── */
function App() {
  return (
    <div style={s.container}>
      <h1 style={{ textAlign: 'center', marginBottom: 4 }}>Performance Demos</h1>
      <p style={{ textAlign: 'center', color: '#666', marginBottom: 24, fontSize: 14 }}>
        React.memo, key optimization, virtual list
      </p>
      <MemoComparison />
      <KeyOptimization />
      <VirtualList />
    </div>
  );
}

export default App;
