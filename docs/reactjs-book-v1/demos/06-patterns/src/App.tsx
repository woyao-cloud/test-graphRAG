import React, {
  Component,
  useState,
  useEffect,
  useRef,
  useCallback,
  createContext,
  useContext,
  createPortal,
} from 'react';

/* ── Styles ── */
const s: Record<string, React.CSSProperties> = {
  container: { fontFamily: 'system-ui, sans-serif', maxWidth: 900, margin: '0 auto', padding: 24 },
  section: { border: '1px solid #e5e7eb', borderRadius: 8, padding: 16, marginBottom: 20, background: '#f9fafb' },
  h2: { fontSize: 18, margin: '0 0 12px 0', color: '#1f2937' },
  btn: { padding: '8px 16px', border: 'none', borderRadius: 6, cursor: 'pointer', fontWeight: 600, fontSize: 13, background: '#4f46e5', color: '#fff', marginRight: 8, marginTop: 8 },
  btnDanger: { padding: '8px 16px', border: 'none', borderRadius: 6, cursor: 'pointer', fontWeight: 600, fontSize: 13, background: '#ef4444', color: '#fff', marginRight: 8, marginTop: 8 },
  input: { padding: '8px 12px', borderRadius: 6, border: '1px solid #d1d5db', fontSize: 14, width: 280 },
};

/* ════════════════════════════════════════════
   1. ErrorBoundary
   ════════════════════════════════════════════ */
interface EBProps { children: React.ReactNode; }
interface EBState { hasError: boolean; error: Error | null; }

class ErrorBoundary extends Component<EBProps, EBState> {
  constructor(props: EBProps) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): EBState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    console.error('ErrorBoundary caught:', error, info);
  }

  handleReset = () => {
    this.setState({ hasError: false, error: null });
  };

  render() {
    if (this.state.hasError) {
      return (
        <div style={{ padding: 16, border: '2px solid #ef4444', borderRadius: 8, background: '#fef2f2', textAlign: 'center' }}>
          <h3 style={{ color: '#991b1b', margin: '0 0 8px 0' }}>Something went wrong</h3>
          <p style={{ fontSize: 13, color: '#dc2626' }}>
            {this.state.error?.message}
          </p>
          <button style={s.btn} onClick={this.handleReset}>
            Reset & Try Again
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

function ThrowButton() {
  const [shouldThrow, setShouldThrow] = useState(false);
  if (shouldThrow) {
    throw new Error('💥 This error was thrown on purpose!');
  }
  return (
    <button style={s.btnDanger} onClick={() => setShouldThrow(true)}>
      Throw Error
    </button>
  );
}

function ErrorBoundaryDemo() {
  return (
    <div style={s.section}>
      <h2 style={s.h2}>ErrorBoundary</h2>
      <ErrorBoundary>
        <p style={{ fontSize: 13, color: '#6b7280', marginBottom: 8 }}>
          Click the button to trigger an error. ErrorBoundary catches it and shows a fallback.
        </p>
        <ThrowButton />
      </ErrorBoundary>
    </div>
  );
}

/* ════════════════════════════════════════════
   2. Portal (Modal)
   ════════════════════════════════════════════ */
function ModalPortal() {
  const [open, setOpen] = useState(false);

  return (
    <div style={s.section}>
      <h2 style={s.h2}>Portal — Modal Overlay</h2>
      <button style={s.btn} onClick={() => setOpen(true)}>
        Open Modal
      </button>
      {open &&
        createPortal(
          <div
            style={{
              position: 'fixed',
              inset: 0,
              background: 'rgba(0,0,0,0.5)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              zIndex: 1000,
            }}
            onClick={() => setOpen(false)}
          >
            <div
              style={{
                background: '#fff',
                borderRadius: 12,
                padding: 24,
                minWidth: 320,
                boxShadow: '0 20px 60px rgba(0,0,0,0.3)',
              }}
              onClick={(e) => e.stopPropagation()}
            >
              <h3 style={{ margin: '0 0 12px 0' }}>Modal (Portal)</h3>
              <p style={{ fontSize: 14, color: '#6b7280', marginBottom: 16 }}>
                This modal is rendered via createPortal into document.body. It
                escapes the parent DOM hierarchy.
              </p>
              <button style={s.btn} onClick={() => setOpen(false)}>
                Close
              </button>
            </div>
          </div>,
          document.body
        )}
    </div>
  );
}

/* ════════════════════════════════════════════
   3. Compound Components (Tabs)
   ════════════════════════════════════════════ */
interface TabsContextType {
  activeTab: string;
  setActiveTab: (tab: string) => void;
}
const TabsContext = createContext<TabsContextType>({
  activeTab: '',
  setActiveTab: () => {},
});

function Tabs({ defaultTab, children }: { defaultTab: string; children: React.ReactNode }) {
  const [activeTab, setActiveTab] = useState(defaultTab);
  return (
    <TabsContext.Provider value={{ activeTab, setActiveTab }}>
      <div style={{ border: '1px solid #e5e7eb', borderRadius: 8, overflow: 'hidden' }}>
        {children}
      </div>
    </TabsContext.Provider>
  );
}

function TabList({ children }: { children: React.ReactNode }) {
  return (
    <div style={{ display: 'flex', borderBottom: '2px solid #e5e7eb', background: '#f9fafb' }}>
      {children}
    </div>
  );
}

function Tab({ label, tabKey }: { label: string; tabKey: string }) {
  const { activeTab, setActiveTab } = useContext(TabsContext);
  const isActive = activeTab === tabKey;
  return (
    <button
      onClick={() => setActiveTab(tabKey)}
      style={{
        flex: 1,
        padding: '10px 16px',
        border: 'none',
        background: isActive ? '#fff' : 'transparent',
        borderBottom: isActive ? '2px solid #4f46e5' : '2px solid transparent',
        cursor: 'pointer',
        fontWeight: isActive ? 700 : 500,
        color: isActive ? '#4f46e5' : '#6b7280',
        fontSize: 14,
        transition: 'all 0.15s',
      }}
    >
      {label}
    </button>
  );
}

function TabPanel({ tabKey, children }: { tabKey: string; children: React.ReactNode }) {
  const { activeTab } = useContext(TabsContext);
  if (activeTab !== tabKey) return null;
  return <div style={{ padding: 16, fontSize: 14 }}>{children}</div>;
}

function CompoundTabsDemo() {
  return (
    <div style={s.section}>
      <h2 style={s.h2}>Compound Components (Tabs)</h2>
      <Tabs defaultTab="tab1">
        <TabList>
          <Tab label="First Tab" tabKey="tab1" />
          <Tab label="Second Tab" tabKey="tab2" />
          <Tab label="Third Tab" tabKey="tab3" />
        </TabList>
        <TabPanel tabKey="tab1">Content for the first tab. React.Children + context pattern.</TabPanel>
        <TabPanel tabKey="tab2">Content for the second tab. Each panel is rendered only when active.</TabPanel>
        <TabPanel tabKey="tab3">Content for the third tab. Clean API for consumers.</TabPanel>
      </Tabs>
    </div>
  );
}

/* ════════════════════════════════════════════
   4. HOC — withLogger
   ════════════════════════════════════════════ */
interface WithLoggerProps {
  loggerName: string;
}

function withLogger<P extends object>(
  Wrapped: React.ComponentType<P>
): React.FC<P & WithLoggerProps> {
  const displayName = Wrapped.displayName || Wrapped.name || 'Component';

  const LoggedComponent: React.FC<P & WithLoggerProps> = (props) => {
    const { loggerName, ...rest } = props;

    useEffect(() => {
      console.log(`[withLogger] ${loggerName || displayName} mounted`);
      return () => console.log(`[withLogger] ${loggerName || displayName} unmounted`);
    }, [loggerName]);

    return <Wrapped {...(rest as unknown as P)} />;
  };

  LoggedComponent.displayName = `withLogger(${displayName})`;
  return LoggedComponent;
}

const LoggedDiv = withLogger(({ text }: { text: string }) => {
  return <div style={{ padding: 8, background: '#f3f4f6', borderRadius: 4, fontSize: 13 }}>{text}</div>;
});

function HOCDemo() {
  const [show, setShow] = useState(true);

  return (
    <div style={s.section}>
      <h2 style={s.h2}>HOC — withLogger</h2>
      <button style={s.btn} onClick={() => setShow((v) => !v)}>
        {show ? 'Unmount' : 'Mount'} Logged Component
      </button>
      <p style={{ fontSize: 12, color: '#9ca3af', marginTop: 4 }}>
        Check the console for mount/unmount logs.
      </p>
      {show && <LoggedDiv loggerName="DemoDiv" text="This component logs mount/unmount to console." />}
    </div>
  );
}

/* ════════════════════════════════════════════
   5. Debounced Search
   ════════════════════════════════════════════ */
function useDebouncedCallback(cb: (val: string) => void, delay: number) {
  const timer = useRef<ReturnType<typeof setTimeout>>();
  return useCallback(
    (val: string) => {
      if (timer.current) clearTimeout(timer.current);
      timer.current = setTimeout(() => cb(val), delay);
    },
    [cb, delay]
  );
}

function DebouncedSearchDemo() {
  const [query, setQuery] = useState('');
  const [apiResult, setApiResult] = useState('');
  const callCount = useRef(0);

  const searchApi = useCallback((val: string) => {
    callCount.current += 1;
    const id = callCount.current;
    setApiResult(`Searching for "${val}"... (API call #${id})`);
    setTimeout(() => {
      setApiResult(`Results for "${val}" (API call #${id} completed)`);
    }, 500);
  }, []);

  const debouncedSearch = useDebouncedCallback(searchApi, 400);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = e.target.value;
    setQuery(val);
    debouncedSearch(val);
  };

  return (
    <div style={s.section}>
      <h2 style={s.h2}>Debounced Search (400ms)</h2>
      <input style={s.input} value={query} onChange={handleChange} placeholder="Type to search..." />
      <p style={{ fontSize: 13, color: '#6b7280', marginTop: 8 }}>{apiResult}</p>
      <p style={{ fontSize: 12, color: '#9ca3af' }}>API calls made: {callCount.current}</p>
    </div>
  );
}

/* ════════════════════════════════════════════
   App
   ════════════════════════════════════════════ */
function App() {
  return (
    <div style={s.container}>
      <h1 style={{ textAlign: 'center', marginBottom: 4 }}>React Patterns</h1>
      <p style={{ textAlign: 'center', color: '#666', marginBottom: 24, fontSize: 14 }}>
        ErrorBoundary, Portal, Compound Components, HOC, Debounced Search
      </p>
      <ErrorBoundaryDemo />
      <ModalPortal />
      <CompoundTabsDemo />
      <HOCDemo />
      <DebouncedSearchDemo />
    </div>
  );
}

export default App;
