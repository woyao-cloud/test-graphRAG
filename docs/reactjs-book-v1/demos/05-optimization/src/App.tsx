import React, {
  useState,
  useTransition,
  useDeferredValue,
  useMemo,
  lazy,
  Suspense,
  useCallback,
  useRef,
} from 'react';

const HeavyComponent = lazy(() => import('./HeavyComponent'));

/* ── Styles ── */
const s: Record<string, React.CSSProperties> = {
  container: { fontFamily: 'system-ui, sans-serif', maxWidth: 900, margin: '0 auto', padding: 24 },
  section: { border: '1px solid #e5e7eb', borderRadius: 8, padding: 16, marginBottom: 20, background: '#f9fafb' },
  h2: { fontSize: 18, margin: '0 0 12px 0', color: '#1f2937' },
  input: { padding: '8px 12px', borderRadius: 6, border: '1px solid #d1d5db', fontSize: 14, width: 280 },
  btn: { padding: '8px 16px', border: 'none', borderRadius: 6, cursor: 'pointer', fontWeight: 600, fontSize: 13, background: '#4f46e5', color: '#fff', marginRight: 8, marginTop: 8 },
  code: { background: '#e5e7eb', padding: '2px 6px', borderRadius: 4, fontFamily: 'monospace', fontSize: 13 },
  label: { fontSize: 13, color: '#6b7280', marginTop: 8 },
};

/* ── 1. Code Splitting ── */
function CodeSplittingDemo() {
  const [show, setShow] = useState(false);

  return (
    <div style={s.section}>
      <h2 style={s.h2}>Code Splitting (React.lazy + Suspense)</h2>
      <button style={s.btn} onClick={() => setShow((v) => !v)}>
        {show ? 'Hide' : 'Load'} Heavy Component
      </button>
      {show && (
        <Suspense
          fallback={
            <div style={{ padding: 16, color: '#6b7280', fontSize: 14 }}>
              Loading heavy component...
            </div>
          }
        >
          <HeavyComponent />
        </Suspense>
      )}
    </div>
  );
}

/* ── 2. useTransition ── */
const largeList = Array.from({ length: 10000 }, (_, i) => `Item ${i}: ${Math.random().toString(36).slice(2)}`);

function UseTransitionDemo() {
  const [filter, setFilter] = useState('');
  const [useTrans, setUseTrans] = useState(false);
  const [isPending, startTransition] = useTransition();
  const [filtered, setFiltered] = useState<string[]>([]);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const val = e.target.value;
      setFilter(val);

      const filterFn = () => {
        setFiltered(
          largeList.filter((item) =>
            item.toLowerCase().includes(val.toLowerCase())
          )
        );
      };

      if (useTrans) {
        startTransition(filterFn);
      } else {
        filterFn();
      }
    },
    [useTrans, startTransition]
  );

  return (
    <div style={s.section}>
      <h2 style={s.h2}>useTransition — Slow Filter (10,000 items)</h2>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
        <input
          ref={inputRef}
          style={s.input}
          value={filter}
          onChange={handleChange}
          placeholder="Type to filter..."
        />
        <button
          style={useTrans ? { ...s.btn, background: '#16a34a' } : s.btn}
          onClick={() => setUseTrans((v) => !v)}
        >
          {useTrans ? 'With useTransition' : 'Without useTransition'}
        </button>
      </div>
      <p style={s.label}>
        {isPending ? 'Filtering (transition pending)...' : `Showing ${filtered.length} results`}
      </p>
      <p style={{ fontSize: 12, color: '#9ca3af' }}>
        {useTrans
          ? 'With useTransition: input stays responsive, list updates are deferred.'
          : 'Without useTransition: input lags while filtering.'}
      </p>
      <div style={{ maxHeight: 200, overflowY: 'auto', border: '1px solid #e5e7eb', borderRadius: 4, padding: 8, marginTop: 4, fontSize: 13 }}>
        {filtered.slice(0, 200).map((item, i) => (
          <div key={i} style={{ padding: '2px 0' }}>{item}</div>
        ))}
        {filtered.length > 200 && <div style={{ color: '#9ca3af' }}>...and {filtered.length - 200} more</div>}
      </div>
    </div>
  );
}

/* ── 3. useDeferredValue ── */
function UseDeferredValueDemo() {
  const [text, setText] = useState('');
  const deferredText = useDeferredValue(text);

  const deferredList = useMemo(
    () =>
      largeList.filter((item) =>
        item.toLowerCase().includes(deferredText.toLowerCase())
      ),
    [deferredText]
  );

  const isStale = text !== deferredText;

  return (
    <div style={s.section}>
      <h2 style={s.h2}>useDeferredValue — Deferred List</h2>
      <input
        style={s.input}
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Type to filter..."
      />
      <p style={s.label}>
        Urgent value: <strong>"{text}"</strong> | Deferred:{' '}
        <strong>"{deferredText}"</strong>
        {isStale && (
          <span style={{ color: '#ca8a04', marginLeft: 8 }}>
            (stale — list updating)
          </span>
        )}
      </p>
      <div style={{ maxHeight: 200, overflowY: 'auto', border: `2px solid ${isStale ? '#f59e0b' : '#e5e7eb'}`, borderRadius: 4, padding: 8, marginTop: 4, fontSize: 13, opacity: isStale ? 0.6 : 1, transition: 'opacity 0.2s' }}>
        {deferredList.slice(0, 200).map((item, i) => (
          <div key={i} style={{ padding: '2px 0' }}>{item}</div>
        ))}
        {deferredList.length > 200 && <div style={{ color: '#9ca3af' }}>...and {deferredList.length - 200} more</div>}
      </div>
    </div>
  );
}

/* ── 4. Component Splitting ── */
function ComponentSplittingDemo() {
  return (
    <div style={s.section}>
      <h2 style={s.h2}>Component Splitting — Colocated State</h2>
      <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
        <BigState />
        <ColocatedState />
      </div>
    </div>
  );
}

function BigState() {
  const [count, setCount] = useState(0);
  const [name, setName] = useState('');
  const renders = useRef(0);
  renders.current += 1;

  return (
    <div style={{ flex: 1, border: '2px solid #ef4444', borderRadius: 6, padding: 12, background: '#fef2f2' }}>
      <h3 style={{ margin: '0 0 8px 0', fontSize: 15, color: '#991b1b' }}>One Big State</h3>
      <p style={{ fontSize: 12, color: '#991b1b' }}>Renders: {renders.current}</p>
      <p>Count: {count}</p>
      <button style={s.btn} onClick={() => setCount((c) => c + 1)}>+1</button>
      <div style={{ marginTop: 8 }}>
        <input style={s.input} value={name} onChange={(e) => setName(e.target.value)} placeholder="Name" />
      </div>
      <p style={{ fontSize: 12, color: '#6b7280', marginTop: 8 }}>
        Changing name ALSO re-renders the counter section (everything in one component).
      </p>
    </div>
  );
}

function ColocatedState() {
  return (
    <div style={{ flex: 1, border: '2px solid #16a34a', borderRadius: 6, padding: 12, background: '#f0fdf4' }}>
      <h3 style={{ margin: '0 0 8px 0', fontSize: 15, color: '#166534' }}>Colocated State</h3>
      <CounterSection />
      <NameSection />
    </div>
  );
}

function CounterSection() {
  const [count, setCount] = useState(0);
  const renders = useRef(0);
  renders.current += 1;

  return (
    <div style={{ marginBottom: 8 }}>
      <p style={{ fontSize: 12, color: '#166534' }}>Counter renders: {renders.current}</p>
      <p>Count: {count}</p>
      <button style={s.btn} onClick={() => setCount((c) => c + 1)}>+1</button>
    </div>
  );
}

function NameSection() {
  const [name, setName] = useState('');
  const renders = useRef(0);
  renders.current += 1;

  return (
    <div>
      <p style={{ fontSize: 12, color: '#166534' }}>Name renders: {renders.current}</p>
      <input style={s.input} value={name} onChange={(e) => setName(e.target.value)} placeholder="Name" />
      <p style={{ fontSize: 12, color: '#6b7280', marginTop: 4 }}>
        Changing name does NOT re-render the counter — state is colocated.
      </p>
    </div>
  );
}

/* ── App ── */
function App() {
  return (
    <div style={s.container}>
      <h1 style={{ textAlign: 'center', marginBottom: 4 }}>Optimization Patterns</h1>
      <p style={{ textAlign: 'center', color: '#666', marginBottom: 24, fontSize: 14 }}>
        Code splitting, useTransition, useDeferredValue, component splitting
      </p>
      <CodeSplittingDemo />
      <UseTransitionDemo />
      <UseDeferredValueDemo />
      <ComponentSplittingDemo />
    </div>
  );
}

export default App;
