import React, { useState, useRef, useCallback } from 'react';

/* ── CSS-in-JS styles ── */
const styles: Record<string, React.CSSProperties> = {
  container: {
    fontFamily: 'system-ui, sans-serif',
    maxWidth: 800,
    margin: '0 auto',
    padding: 24,
  },
  title: { textAlign: 'center' as const, marginBottom: 8 },
  subtitle: { textAlign: 'center' as const, color: '#666', marginBottom: 24 },
  toolbar: {
    display: 'flex',
    gap: 12,
    justifyContent: 'center',
    marginBottom: 24,
  },
  btn: {
    padding: '10px 20px',
    border: 'none',
    borderRadius: 6,
    cursor: 'pointer',
    fontWeight: 600,
    fontSize: 14,
  },
  btnPrimary: { background: '#4f46e5', color: '#fff' },
  btnHigh: { background: '#16a34a', color: '#fff' },
  btnLow: { background: '#ca8a04', color: '#fff' },
  tree: {
    border: '2px solid #e5e7eb',
    borderRadius: 8,
    padding: 16,
    background: '#f9fafb',
  },
  fiberNode: (depth: number, highlight: boolean): React.CSSProperties => ({
    marginLeft: depth * 24,
    marginTop: 8,
    marginBottom: 8,
    padding: '10px 14px',
    border: `2px solid ${highlight ? '#4f46e5' : '#d1d5db'}`,
    borderRadius: 6,
    background: highlight ? '#eef2ff' : '#fff',
    transition: 'all 0.2s',
    position: 'relative' as const,
  }),
  nodeHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  nodeName: { fontWeight: 700, fontSize: 15 },
  renderCount: {
    fontSize: 12,
    color: '#6b7280',
    background: '#f3f4f6',
    padding: '2px 8px',
    borderRadius: 4,
  },
  logPanel: {
    marginTop: 16,
    padding: 12,
    background: '#1f2937',
    color: '#22c55e',
    borderRadius: 6,
    fontFamily: 'monospace',
    fontSize: 13,
    maxHeight: 160,
    overflowY: 'auto' as const,
    whiteSpace: 'pre-wrap' as const,
  },
};

/* ── Logger helper ── */
let logs: string[] = [];
function log(msg: string) {
  logs.push(`[${new Date().toLocaleTimeString()}] ${msg}`);
  if (logs.length > 20) logs = logs.slice(-20);
}

/* ── FiberNode component (recursive tree node) ── */
interface FiberNodeProps {
  name: string;
  depth: number;
  children?: React.ReactNode;
  highlight?: boolean;
}

const FiberNode: React.FC<FiberNodeProps> = React.memo(
  ({ name, depth, children, highlight }) => {
    const renderCount = useRef(0);
    renderCount.current += 1;

    return (
      <div style={styles.fiberNode(depth, highlight ?? false)}>
        <div style={styles.nodeHeader}>
          <span style={styles.nodeName}>{name}</span>
          <span style={styles.renderCount}>
            Renders: {renderCount.current}
          </span>
        </div>
        {children && <div style={{ marginTop: 4 }}>{children}</div>}
      </div>
    );
  }
);

/* ── Leaf component ── */
const LeafNode: React.FC<{ label: string }> = React.memo(({ label }) => {
  const renderCount = useRef(0);
  renderCount.current += 1;
  return (
    <div
      style={{
        marginLeft: 24,
        padding: '6px 12px',
        border: '1px dashed #9ca3af',
        borderRadius: 4,
        marginTop: 4,
        fontSize: 13,
        display: 'flex',
        justifyContent: 'space-between',
        background: '#fff',
      }}
    >
      <span style={{ color: '#6b7280' }}>{label}</span>
      <span style={{ fontSize: 11, color: '#9ca3af' }}>
        R: {renderCount.current}
      </span>
    </div>
  );
});

/* ── PriorityDemo ── */
const PriorityDemo: React.FC<{
  onScheduleHigh: () => void;
  onScheduleLow: () => void;
}> = ({ onScheduleHigh, onScheduleLow }) => {
  const renderCount = useRef(0);
  renderCount.current += 1;

  return (
    <div
      style={{
        marginLeft: 24,
        padding: 12,
        border: '2px solid #d1d5db',
        borderRadius: 6,
        marginTop: 8,
        background: '#fff',
      }}
    >
      <div style={styles.nodeHeader}>
        <span style={styles.nodeName}>PriorityDemo</span>
        <span style={styles.renderCount}>Renders: {renderCount.current}</span>
      </div>
      <div style={{ display: 'flex', gap: 8, marginTop: 8 }}>
        <button
          style={{ ...styles.btn, ...styles.btnHigh }}
          onClick={onScheduleHigh}
        >
          Simulate High Priority
        </button>
        <button
          style={{ ...styles.btn, ...styles.btnLow }}
          onClick={onScheduleLow}
        >
          Simulate Low Priority
        </button>
      </div>
    </div>
  );
};

/* ── App root ── */
const App: React.FC = () => {
  const [counter, setCounter] = useState(0);
  const [highPrio, setHighPrio] = useState(0);
  const [lowPrio, setLowPrio] = useState(0);
  const [logsState, setLogsState] = useState<string[]>([]);
  const appRenderCount = useRef(0);
  appRenderCount.current += 1;

  const triggerReRender = useCallback(() => {
    log(`Trigger re-render (counter: ${counter + 1})`);
    setCounter((c) => c + 1);
    setLogsState([...logs]);
  }, [counter]);

  const scheduleHigh = useCallback(() => {
    log('HIGH priority update scheduled');
    setHighPrio((h) => h + 1);
    setCounter((c) => c + 1);
    setTimeout(() => {
      log('HIGH priority update committed');
      setLogsState([...logs]);
    }, 0);
  }, []);

  const scheduleLow = useCallback(() => {
    log('LOW priority update scheduled');
    setLowPrio((l) => l + 1);
    setTimeout(() => {
      log('LOW priority update committed');
      setLogsState([...logs]);
    }, 100);
  }, []);

  return (
    <div style={styles.container}>
      <h1 style={styles.title}>React Fiber Tree Visualizer</h1>
      <p style={styles.subtitle}>
        Each node tracks its own render count. App renders: {appRenderCount.current}
      </p>

      <div style={styles.toolbar}>
        <button
          style={{ ...styles.btn, ...styles.btnPrimary }}
          onClick={triggerReRender}
        >
          Trigger Re-render (Counter: {counter})
        </button>
      </div>

      <div style={styles.tree}>
        <FiberNode name="App (Root)" depth={0} highlight>
          <FiberNode name="Toolbar" depth={1}>
            <LeafNode label="Re-render Button" />
          </FiberNode>

          <FiberNode name="PriorityDemo" depth={1}>
            <PriorityDemo
              onScheduleHigh={scheduleHigh}
              onScheduleLow={scheduleLow}
            />
          </FiberNode>

          <FiberNode name="HighPriority Section" depth={1} highlight={highPrio > 0}>
            <LeafNode label={`High priority updates: ${highPrio}`} />
          </FiberNode>

          <FiberNode name="LowPriority Section" depth={1} highlight={lowPrio > 0}>
            <LeafNode label={`Low priority updates: ${lowPrio}`} />
          </FiberNode>

          <FiberNode name="LogPanel" depth={1}>
            <div style={styles.logPanel}>
              {logsState.length === 0 ? 'No logs yet...' : logsState.join('\n')}
            </div>
          </FiberNode>
        </FiberNode>
      </div>
    </div>
  );
};

export default App;
