# 第4章 自定义 Hooks 与高级模式

本章将深入探讨自定义 Hooks 的设计模式、组合技巧以及在实际项目中的高级应用。通过丰富的代码示例，你将学会如何编写可复用、可组合的自定义 Hooks。

## 4.1 自定义 Hooks 基础

### 4.1.1 什么是自定义 Hook

自定义 Hook 是一个以 `use` 开头的 JavaScript 函数，可以调用其他 Hooks。它让你可以将组件逻辑提取到可复用的函数中。

```jsx
// 自定义 Hook 的基本形式
function useDocumentTitle(title) {
  useEffect(() => {
    document.title = title;
  }, [title]);
}

// 在组件中使用
function HomePage() {
  useDocumentTitle('首页 - MyApp');
  return <div>首页内容</div>;
}

function AboutPage() {
  useDocumentTitle('关于我们 - MyApp');
  return <div>关于页面</div>;
}
```

### 4.1.2 自定义 Hook 的命名规范

```jsx
// 命名规范：必须以 use 开头
// React 通过名称检测 Hook 规则

// 正确：以 use 开头
function useWindowSize() { /* ... */ }
function useDebounce() { /* ... */ }
function useLocalStorage() { /* ... */ }

// 错误：不以 use 开头
function getWindowSize() { /* ... */ } // 不是 Hook，不能使用 Hooks
function fetchData() { /* ... */ }     // 不是 Hook，不能使用 Hooks
```

## 4.2 实用自定义 Hooks

### 4.2.1 useDebounce

防抖 Hook：延迟执行直到指定时间内没有新调用。

```jsx
import { useState, useEffect } from 'react';

// useDebounce 实现
function useDebounce(value, delay = 300) {
  const [debouncedValue, setDebouncedValue] = useState(value);
  
  useEffect(() => {
    // 设置定时器延迟更新
    const timer = setTimeout(() => {
      setDebouncedValue(value);
    }, delay);
    
    // 如果 value 或 delay 变化，清除上一个定时器
    return () => clearTimeout(timer);
  }, [value, delay]);
  
  return debouncedValue;
}

// 使用示例：搜索输入
function SearchInput() {
  const [query, setQuery] = useState('');
  const debouncedQuery = useDebounce(query, 300);
  const [results, setResults] = useState([]);
  
  useEffect(() => {
    if (debouncedQuery.trim()) {
      fetch(`/api/search?q=${debouncedQuery}`)
        .then(res => res.json())
        .then(data => setResults(data));
    } else {
      setResults([]);
    }
  }, [debouncedQuery]);
  
  return (
    <div>
      <input
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="搜索..."
      />
      <ul>
        {results.map(item => (
          <li key={item.id}>{item.name}</li>
        ))}
      </ul>
    </div>
  );
}

// 高级版本：支持回调函数
function useDebounceCallback(callback, delay = 300, deps = []) {
  const callbackRef = useRef(callback);
  
  useEffect(() => {
    callbackRef.current = callback;
  }, [callback]);
  
  useEffect(() => {
    const timer = setTimeout(() => {
      callbackRef.current();
    }, delay);
    
    return () => clearTimeout(timer);
  }, [delay, ...deps]);
}
```

### 4.2.2 useThrottle

节流 Hook：在指定时间间隔内只执行一次。

```jsx
import { useState, useEffect, useRef } from 'react';

// useThrottle 实现
function useThrottle(value, interval = 500) {
  const [throttledValue, setThrottledValue] = useState(value);
  const lastUpdated = useRef(Date.now());
  
  useEffect(() => {
    const now = Date.now();
    const timeSinceLastUpdate = now - lastUpdated.current;
    
    if (timeSinceLastUpdate >= interval) {
      // 距离上次更新超过间隔，立即更新
      setThrottledValue(value);
      lastUpdated.current = now;
    } else {
      // 距离上次更新不足间隔，设置定时器
      const timer = setTimeout(() => {
        setThrottledValue(value);
        lastUpdated.current = Date.now();
      }, interval - timeSinceLastUpdate);
      
      return () => clearTimeout(timer);
    }
  }, [value, interval]);
  
  return throttledValue;
}

// 使用示例：滚动监听
function ScrollTracker() {
  const [scrollY, setScrollY] = useState(0);
  const throttledScrollY = useThrottle(scrollY, 200);
  
  useEffect(() => {
    const handleScroll = () => {
      setScrollY(window.scrollY);
    };
    
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);
  
  return (
    <div style={{ position: 'fixed', top: 0, right: 0, padding: '10px', background: '#f0f0f0' }}>
      滚动位置: {throttledScrollY}px
    </div>
  );
}

// useThrottleCallback：节流函数调用
function useThrottleCallback(callback, interval = 500) {
  const lastCall = useRef(0);
  const timer = useRef(null);
  
  return useCallback((...args) => {
    const now = Date.now();
    const timeSinceLastCall = now - lastCall.current;
    
    if (timeSinceLastCall >= interval) {
      lastCall.current = now;
      callback(...args);
    } else if (!timer.current) {
      timer.current = setTimeout(() => {
        lastCall.current = Date.now();
        timer.current = null;
        callback(...args);
      }, interval - timeSinceLastCall);
    }
  }, [callback, interval]);
}
```

### 4.2.3 usePrevious

获取前一个值的 Hook。

```jsx
import { useRef, useEffect } from 'react';

// usePrevious 实现
function usePrevious(value) {
  const ref = useRef();
  
  useEffect(() => {
    ref.current = value;
  }, [value]);
  
  return ref.current; // 返回前一个渲染时的值
}

// 使用示例
function Counter() {
  const [count, setCount] = useState(0);
  const prevCount = usePrevious(count);
  const [direction, setDirection] = useState('无变化');
  
  useEffect(() => {
    if (prevCount !== undefined) {
      if (count > prevCount) {
        setDirection('增加');
      } else if (count < prevCount) {
        setDirection('减少');
      }
    }
  }, [count, prevCount]);
  
  return (
    <div>
      <p>当前: {count}</p>
      <p>前一个: {prevCount}</p>
      <p>方向: {direction}</p>
      <button onClick={() => setCount(c => c + 1)}>+</button>
      <button onClick={() => setCount(c => c - 1)}>-</button>
    </div>
  );
}
```

### 4.2.4 useInterval

声明式 setInterval Hook。

```jsx
import { useEffect, useRef } from 'react';

// useInterval 实现
function useInterval(callback, delay) {
  const savedCallback = useRef(callback);
  
  // 保存最新的回调
  useEffect(() => {
    savedCallback.current = callback;
  }, [callback]);
  
  // 设置定时器
  useEffect(() => {
    if (delay !== null && delay !== undefined) {
      const id = setInterval(() => {
        savedCallback.current();
      }, delay);
      
      return () => clearInterval(id);
    }
  }, [delay]);
}

// 使用示例
function Timer() {
  const [count, setCount] = useState(0);
  const [delay, setDelay] = useState(1000);
  const [isRunning, setIsRunning] = useState(true);
  
  useInterval(() => {
    setCount(c => c + 1);
  }, isRunning ? delay : null); // delay 为 null 时暂停
  
  return (
    <div>
      <p>计数: {count}</p>
      <div>
        <label>间隔 (ms): </label>
        <input
          type="number"
          value={delay}
          onChange={(e) => setDelay(Number(e.target.value))}
        />
      </div>
      <button onClick={() => setIsRunning(!isRunning)}>
        {isRunning ? '暂停' : '继续'}
      </button>
      <button onClick={() => setCount(0)}>重置</button>
    </div>
  );
}
```

### 4.2.5 useMediaQuery

响应式媒体查询 Hook。

```jsx
import { useState, useEffect } from 'react';

// useMediaQuery 实现
function useMediaQuery(query) {
  const [matches, setMatches] = useState(() => window.matchMedia(query).matches);
  
  useEffect(() => {
    const mediaQuery = window.matchMedia(query);
    
    const handleChange = (e) => {
      setMatches(e.matches);
    };
    
    mediaQuery.addEventListener('change', handleChange);
    return () => mediaQuery.removeEventListener('change', handleChange);
  }, [query]);
  
  return matches;
}

// 使用示例
function ResponsiveComponent() {
  const isMobile = useMediaQuery('(max-width: 768px)');
  const isTablet = useMediaQuery('(min-width: 769px) and (max-width: 1024px)');
  const isDesktop = useMediaQuery('(min-width: 1025px)');
  
  return (
    <div>
      <p>当前设备:</p>
      {isMobile && <p>手机 (<= 768px)</p>}
      {isTablet && <p>平板 (769px - 1024px)</p>}
      {isDesktop && <p>桌面 (>= 1025px)</p>}
    </div>
  );
}

// 使用 Hook 实现响应式布局
function ResponsiveLayout() {
  const isMobile = useMediaQuery('(max-width: 768px)');
  const isDarkMode = useMediaQuery('(prefers-color-scheme: dark)');
  const prefersReducedMotion = useMediaQuery('(prefers-reduced-motion: reduce)');
  
  return (
    <div style={{
      display: 'flex',
      flexDirection: isMobile ? 'column' : 'row',
      background: isDarkMode ? '#333' : '#fff',
      color: isDarkMode ? '#fff' : '#333',
    }}>
      {isMobile ? <MobileNav /> : <Sidebar />}
      <main>
        {prefersReducedMotion ? <StaticContent /> : <AnimatedContent />}
      </main>
    </div>
  );
}
```

### 4.2.6 useLocalStorage

本地存储 Hook，自动同步状态到 localStorage。

```jsx
import { useState, useCallback } from 'react';

// useLocalStorage 实现
function useLocalStorage(key, initialValue) {
  // 初始化状态
  const [storedValue, setStoredValue] = useState(() => {
    try {
      const item = window.localStorage.getItem(key);
      return item ? JSON.parse(item) : initialValue;
    } catch (error) {
      console.error('读取 localStorage 失败:', error);
      return initialValue;
    }
  });
  
  // 更新 localStorage 和状态
  const setValue = useCallback((value) => {
    try {
      const valueToStore = value instanceof Function ? value(storedValue) : value;
      setStoredValue(valueToStore);
      window.localStorage.setItem(key, JSON.stringify(valueToStore));
    } catch (error) {
      console.error('写入 localStorage 失败:', error);
    }
  }, [key, storedValue]);
  
  return [storedValue, setValue];
}

// 使用示例：主题设置
function ThemeSettings() {
  const [theme, setTheme] = useLocalStorage('theme', 'light');
  const [settings, setSettings] = useLocalStorage('settings', {
    fontSize: 14,
    language: 'zh',
    notifications: true,
  });
  
  const toggleTheme = () => {
    setTheme(prev => prev === 'light' ? 'dark' : 'light');
  };
  
  const updateSettings = (newSettings) => {
    setSettings(prev => ({ ...prev, ...newSettings }));
  };
  
  return (
    <div className={theme}>
      <h2>设置</h2>
      <button onClick={toggleTheme}>
        当前主题: {theme}
      </button>
      <div>
        <label>
          字体大小:
          <input
            type="range"
            min="12"
            max="24"
            value={settings.fontSize}
            onChange={(e) => updateSettings({ fontSize: Number(e.target.value) })}
          />
        </label>
      </div>
      <div>
        <label>
          语言:
          <select
            value={settings.language}
            onChange={(e) => updateSettings({ language: e.target.value })}
          >
            <option value="zh">中文</option>
            <option value="en">English</option>
          </select>
        </label>
      </div>
      <div>
        <label>
          <input
            type="checkbox"
            checked={settings.notifications}
            onChange={(e) => updateSettings({ notifications: e.target.checked })}
          />
          启用通知
        </label>
      </div>
    </div>
  );
}
```

### 4.2.7 useFetch

数据获取 Hook，支持加载状态和错误处理。

```jsx
import { useState, useEffect, useRef, useCallback } from 'react';

// useFetch 实现
function useFetch(url, options = {}) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const abortRef = useRef(null);
  
  const fetchData = useCallback(async () => {
    // 取消之前的请求
    if (abortRef.current) {
      abortRef.current.abort();
    }
    
    const abortController = new AbortController();
    abortRef.current = abortController;
    
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch(url, {
        ...options,
        signal: abortController.signal,
      });
      
      if (!response.ok) {
        throw new Error(`HTTP 错误: ${response.status}`);
      }
      
      const result = await response.json();
      
      if (!abortController.signal.aborted) {
        setData(result);
        setLoading(false);
      }
    } catch (err) {
      if (err.name !== 'AbortError') {
        setError(err.message);
        setLoading(false);
      }
    }
  }, [url]);
  
  useEffect(() => {
    fetchData();
    
    return () => {
      if (abortRef.current) {
        abortRef.current.abort();
      }
    };
  }, [fetchData]);
  
  const refetch = useCallback(() => {
    fetchData();
  }, [fetchData]);
  
  return { data, loading, error, refetch };
}

// 使用示例
function UserList() {
  const { data: users, loading, error, refetch } = useFetch('/api/users');
  
  if (loading) return <div>加载中...</div>;
  if (error) return <div>错误: {error}</div>;
  
  return (
    <div>
      <button onClick={refetch}>刷新</button>
      <ul>
        {users?.map(user => (
          <li key={user.id}>{user.name} - {user.email}</li>
        ))}
      </ul>
    </div>
  );
}

// 高级版本：支持依赖变化自动重新获取
function useFetchWithDeps(url, options = {}, deps = []) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  useEffect(() => {
    let cancelled = false;
    
    const fetchData = async () => {
      setLoading(true);
      setError(null);
      
      try {
        const response = await fetch(url, options);
        const result = await response.json();
        
        if (!cancelled) {
          setData(result);
          setLoading(false);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err.message);
          setLoading(false);
        }
      }
    };
    
    fetchData();
    
    return () => {
      cancelled = true;
    };
  }, [url, ...deps]);
  
  return { data, loading, error };
}
```

## 4.3 Hooks 组合模式

### 4.3.1 组合多个 Hooks

```jsx
// 组合 Hooks 创建更复杂的功能
function useUserPreferences() {
  const theme = useLocalStorage('theme', 'light');
  const fontSize = useLocalStorage('fontSize', 14);
  const language = useLocalStorage('language', 'zh');
  const isMobile = useMediaQuery('(max-width: 768px)');
  const prefersDark = useMediaQuery('(prefers-color-scheme: dark)');
  
  // 合并多个 Hooks 的状态
  return {
    theme: theme[0],
    setTheme: theme[1],
    fontSize: fontSize[0],
    setFontSize: fontSize[1],
    language: language[0],
    setLanguage: language[1],
    isMobile,
    isDarkMode: prefersDark,
  };
}

// 在组件中使用
function SettingsPanel() {
  const {
    theme,
    setTheme,
    fontSize,
    setFontSize,
    language,
    setLanguage,
    isMobile,
    isDarkMode,
  } = useUserPreferences();
  
  // 自动应用主题
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', isDarkMode ? 'dark' : theme);
  }, [theme, isDarkMode]);
  
  return (
    <div style={{ fontSize: `${fontSize}px` }}>
      {isMobile ? <MobileView /> : <DesktopView />}
    </div>
  );
}
```

### 4.3.2 Hook 之间的依赖关系

```jsx
// Hooks 可以相互依赖
function useSearchWithDebounce() {
  const [query, setQuery] = useState('');
  const debouncedQuery = useDebounce(query, 300);
  const { data, loading, error } = useFetch(
    debouncedQuery ? `/api/search?q=${debouncedQuery}` : null
  );
  
  return {
    query,
    setQuery,
    results: data,
    loading,
    error,
  };
}

// 分页 + 搜索组合
function usePaginatedSearch() {
  const [page, setPage] = useState(1);
  const [pageSize] = useState(20);
  const [query, setQuery] = useState('');
  const debouncedQuery = useDebounce(query, 300);
  
  // 搜索条件变化时重置页码
  useEffect(() => {
    setPage(1);
  }, [debouncedQuery]);
  
  const url = debouncedQuery
    ? `/api/search?q=${debouncedQuery}&page=${page}&pageSize=${pageSize}`
    : null;
  
  const { data, loading, error } = useFetch(url);
  
  return {
    query,
    setQuery,
    page,
    setPage,
    pageSize,
    totalPages: data?.totalPages || 0,
    results: data?.items || [],
    loading,
    error,
  };
}
```

## 4.4 测试 Hooks

### 4.4.1 使用 renderHook 测试

```jsx
import { renderHook, act } from '@testing-library/react';

// 测试 useCounter
function useCounter(initialValue = 0) {
  const [count, setCount] = useState(initialValue);
  
  const increment = useCallback(() => setCount(c => c + 1), []);
  const decrement = useCallback(() => setCount(c => c - 1), []);
  const reset = useCallback(() => setCount(initialValue), [initialValue]);
  
  return { count, increment, decrement, reset };
}

// 测试代码
describe('useCounter', () => {
  test('应该使用初始值', () => {
    const { result } = renderHook(() => useCounter(10));
    expect(result.current.count).toBe(10);
  });
  
  test('默认初始值为 0', () => {
    const { result } = renderHook(() => useCounter());
    expect(result.current.count).toBe(0);
  });
  
  test('increment 应该增加计数', () => {
    const { result } = renderHook(() => useCounter(0));
    
    act(() => {
      result.current.increment();
    });
    
    expect(result.current.count).toBe(1);
  });
  
  test('decrement 应该减少计数', () => {
    const { result } = renderHook(() => useCounter(5));
    
    act(() => {
      result.current.decrement();
    });
    
    expect(result.current.count).toBe(4);
  });
  
  test('reset 应该重置为初始值', () => {
    const { result } = renderHook(() => useCounter(100));
    
    act(() => {
      result.current.increment();
      result.current.increment();
    });
    
    expect(result.current.count).toBe(102);
    
    act(() => {
      result.current.reset();
    });
    
    expect(result.current.count).toBe(100);
  });
});

// 测试 useDebounce
describe('useDebounce', () => {
  beforeEach(() => {
    jest.useFakeTimers();
  });
  
  afterEach(() => {
    jest.useRealTimers();
  });
  
  test('应该在延迟后更新值', () => {
    const { result, rerender } = renderHook(
      ({ value, delay }) => useDebounce(value, delay),
      { initialProps: { value: 'hello', delay: 500 } }
    );
    
    // 初始值立即返回
    expect(result.current).toBe('hello');
    
    // 更新值
    rerender({ value: 'world', delay: 500 });
    
    // 延迟结束前，值不变
    expect(result.current).toBe('hello');
    
    // 快进时间
    act(() => {
      jest.advanceTimersByTime(500);
    });
    
    // 延迟结束后，值更新
    expect(result.current).toBe('world');
  });
  
  test('连续更新会重置定时器', () => {
    const { result, rerender } = renderHook(
      ({ value }) => useDebounce(value, 300),
      { initialProps: { value: 'a' } }
    );
    
    rerender({ value: 'ab' });
    act(() => { jest.advanceTimersByTime(100); });
    
    rerender({ value: 'abc' });
    act(() => { jest.advanceTimersByTime(100); });
    
    rerender({ value: 'abcd' });
    act(() => { jest.advanceTimersByTime(100); });
    
    // 定时器被重置，值还是初始值
    expect(result.current).toBe('a');
    
    act(() => { jest.advanceTimersByTime(200); });
    
    // 最后一次更新后等待了 300ms
    expect(result.current).toBe('abcd');
  });
});

// 测试 useFetch
describe('useFetch', () => {
  beforeEach(() => {
    global.fetch = jest.fn();
  });
  
  afterEach(() => {
    jest.restoreAllMocks();
  });
  
  test('应该返回加载状态和数据', async () => {
    const mockData = { id: 1, name: 'Test' };
    global.fetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockData,
    });
    
    const { result, waitForNextUpdate } = renderHook(
      () => useFetch('/api/data')
    );
    
    expect(result.current.loading).toBe(true);
    expect(result.current.data).toBeNull();
    
    await waitForNextUpdate();
    
    expect(result.current.loading).toBe(false);
    expect(result.current.data).toEqual(mockData);
  });
  
  test('应该处理错误', async () => {
    global.fetch.mockRejectedValueOnce(new Error('网络错误'));
    
    const { result, waitForNextUpdate } = renderHook(
      () => useFetch('/api/data')
    );
    
    await waitForNextUpdate();
    
    expect(result.current.loading).toBe(false);
    expect(result.current.error).toBe('网络错误');
    expect(result.current.data).toBeNull();
  });
  
  test('组件卸载时应该取消请求', () => {
    global.fetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ id: 1 }),
    });
    
    const { result, unmount } = renderHook(
      () => useFetch('/api/data')
    );
    
    unmount();
    // 不会抛出 AbortError
    expect(result.current.loading).toBe(true);
  });
});
```

## 4.5 useContext + useReducer 状态管理

### 4.5.1 迷你 Redux 模式

```jsx
import { createContext, useContext, useReducer } from 'react';

// 创建 Context
const StoreContext = createContext(null);
const DispatchContext = createContext(null);

// Provider 组件
function StoreProvider({ children, reducer, initialState }) {
  const [state, dispatch] = useReducer(reducer, initialState);
  
  return (
    <StoreContext.Provider value={state}>
      <DispatchContext.Provider value={dispatch}>
        {children}
      </DispatchContext.Provider>
    </StoreContext.Provider>
  );
}

// 自定义 Hooks
function useStore() {
  const context = useContext(StoreContext);
  if (context === undefined) {
    throw new Error('useStore 必须在 StoreProvider 内使用');
  }
  return context;
}

function useDispatch() {
  const context = useContext(DispatchContext);
  if (context === undefined) {
    throw new Error('useDispatch 必须在 StoreProvider 内使用');
  }
  return context;
}

// Action 创建器
const actions = {
  addTodo: (text) => ({ type: 'ADD_TODO', payload: { id: Date.now(), text } }),
  toggleTodo: (id) => ({ type: 'TOGGLE_TODO', payload: id }),
  deleteTodo: (id) => ({ type: 'DELETE_TODO', payload: id }),
  setFilter: (filter) => ({ type: 'SET_FILTER', payload: filter }),
};

// Reducer
function todoReducer(state, action) {
  switch (action.type) {
    case 'ADD_TODO':
      return {
        ...state,
        todos: [...state.todos, action.payload],
      };
    case 'TOGGLE_TODO':
      return {
        ...state,
        todos: state.todos.map(todo =>
          todo.id === action.payload
            ? { ...todo, completed: !todo.completed }
            : todo
        ),
      };
    case 'DELETE_TODO':
      return {
        ...state,
        todos: state.todos.filter(todo => todo.id !== action.payload),
      };
    case 'SET_FILTER':
      return { ...state, filter: action.payload };
    default:
      return state;
  }
}

// 使用示例
function App() {
  return (
    <StoreProvider
      reducer={todoReducer}
      initialState={{ todos: [], filter: 'ALL' }}
    >
      <TodoApp />
    </StoreProvider>
  );
}

function TodoApp() {
  return (
    <div>
      <AddTodo />
      <FilterBar />
      <TodoList />
    </div>
  );
}

function AddTodo() {
  const dispatch = useDispatch();
  const [text, setText] = useState('');
  
  const handleSubmit = (e) => {
    e.preventDefault();
    if (text.trim()) {
      dispatch(actions.addTodo(text));
      setText('');
    }
  };
  
  return (
    <form onSubmit={handleSubmit}>
      <input value={text} onChange={(e) => setText(e.target.value)} />
      <button type="submit">添加</button>
    </form>
  );
}

function FilterBar() {
  const { filter } = useStore();
  const dispatch = useDispatch();
  
  const filters = ['ALL', 'ACTIVE', 'COMPLETED'];
  
  return (
    <div>
      {filters.map(f => (
        <button
          key={f}
          onClick={() => dispatch(actions.setFilter(f))}
          style={{ fontWeight: filter === f ? 'bold' : 'normal' }}
        >
          {f === 'ALL' ? '全部' : f === 'ACTIVE' ? '进行中' : '已完成'}
        </button>
      ))}
    </div>
  );
}

function TodoList() {
  const { todos, filter } = useStore();
  
  const filteredTodos = todos.filter(todo => {
    if (filter === 'ACTIVE') return !todo.completed;
    if (filter === 'COMPLETED') return todo.completed;
    return true;
  });
  
  return (
    <ul>
      {filteredTodos.map(todo => (
        <TodoItem key={todo.id} todo={todo} />
      ))}
    </ul>
  );
}

function TodoItem({ todo }) {
  const dispatch = useDispatch();
  
  return (
    <li>
      <input
        type="checkbox"
        checked={todo.completed}
        onChange={() => dispatch(actions.toggleTodo(todo.id))}
      />
      <span style={{ textDecoration: todo.completed ? 'line-through' : 'none' }}>
        {todo.text}
      </span>
      <button onClick={() => dispatch(actions.deleteTodo(todo.id))}>删除</button>
    </li>
  );
}
```

### 4.5.2 Context 分割优化

```jsx
// Context 分割避免不必要的重渲染
function createStore(reducer, initialState) {
  const StateContext = createContext(initialState);
  const DispatchContext = createContext(null);
  
  function Provider({ children }) {
    const [state, dispatch] = useReducer(reducer, initialState);
    
    return (
      <StateContext.Provider value={state}>
        <DispatchContext.Provider value={dispatch}>
          {children}
        </DispatchContext.Provider>
      </StateContext.Provider>
    );
  }
  
  function useStore(selector = (state) => state) {
    const state = useContext(StateContext);
    return selector(state);
  }
  
  function useDispatch() {
    return useContext(DispatchContext);
  }
  
  return { Provider, useStore, useDispatch };
}

// 使用示例
const { Provider, useStore, useDispatch } = createStore(
  (state, action) => {
    switch (action.type) {
      case 'INCREMENT':
        return { ...state, count: state.count + 1 };
      case 'SET_USER':
        return { ...state, user: action.payload };
      default:
        return state;
    }
  },
  { count: 0, user: null, items: [] }
);

function CountDisplay() {
  // 只订阅 count，count 变化时才重渲染
  const count = useStore(state => state.count);
  console.log('CountDisplay 渲染');
  return <div>计数: {count}</div>;
}

function UserDisplay() {
  // 只订阅 user，user 变化时才重渲染
  const user = useStore(state => state.user);
  console.log('UserDisplay 渲染');
  return <div>用户: {user?.name || '未登录'}</div>;
}
```

## 4.6 forwardRef + useImperativeHandle

### 4.6.1 暴露实例方法

```jsx
import { forwardRef, useImperativeHandle, useRef } from 'react';

// 自定义输入组件
const CustomInput = forwardRef(function CustomInput(props, ref) {
  const inputRef = useRef(null);
  
  // 向父组件暴露方法
  useImperativeHandle(ref, () => ({
    focus: () => {
      inputRef.current.focus();
    },
    blur: () => {
      inputRef.current.blur();
    },
    setValue: (value) => {
      inputRef.current.value = value;
    },
    select: () => {
      inputRef.current.select();
    },
    validate: () => {
      const value = inputRef.current.value;
      return value.length >= 3;
    },
  }), []);
  
  return <input ref={inputRef} {...props} />;
});

// 父组件使用
function Form() {
  const inputRef = useRef(null);
  
  const handleFocus = () => {
    inputRef.current.focus();
  };
  
  const handleValidate = () => {
    const isValid = inputRef.current.validate();
    alert(isValid ? '验证通过' : '至少输入3个字符');
  };
  
  return (
    <div>
      <CustomInput ref={inputRef} placeholder="输入内容" />
      <button onClick={handleFocus}>聚焦</button>
      <button onClick={handleValidate}>验证</button>
    </div>
  );
}
```

### 4.6.2 React 19 中的 ref 作为 prop

```jsx
// React 19 中不再需要 forwardRef
// ref 可以直接作为 prop

// React 19 语法
function CustomInput({ ref, label, ...props }) {
  return (
    <div>
      <label>{label}</label>
      <input ref={ref} {...props} />
    </div>
  );
}

// 使用
function Form() {
  const inputRef = useRef(null);
  
  useEffect(() => {
    inputRef.current?.focus();
  }, []);
  
  return <CustomInput ref={inputRef} label="用户名" />;
}
```

## 4.7 第三方 Hooks 集成

### 4.7.1 React Router Hooks

```jsx
import {
  useParams,
  useNavigate,
  useLocation,
  useSearchParams,
  useRouteLoaderData,
} from 'react-router-dom';

// 路由 Hooks 使用示例
function ProductPage() {
  // 获取 URL 参数
  const { id } = useParams();
  
  // 编程式导航
  const navigate = useNavigate();
  
  // 获取当前位置
  const location = useLocation();
  
  // 查询参数
  const [searchParams, setSearchParams] = useSearchParams();
  const view = searchParams.get('view') || 'grid';
  
  return (
    <div>
      <p>产品 ID: {id}</p>
      <p>当前路径: {location.pathname}</p>
      <p>视图模式: {view}</p>
      <button onClick={() => navigate(-1)}>返回</button>
      <button onClick={() => setSearchParams({ view: 'list' })}>
        列表视图
      </button>
    </div>
  );
}
```

### 4.7.2 TanStack Query Hooks

```jsx
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';

// useQuery 使用
function UserProfile({ userId }) {
  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ['user', userId],
    queryFn: async () => {
      const res = await fetch(`/api/users/${userId}`);
      if (!res.ok) throw new Error('获取用户失败');
      return res.json();
    },
    staleTime: 5 * 60 * 1000, // 5 分钟内认为数据新鲜
    cacheTime: 30 * 60 * 1000, // 缓存保留 30 分钟
    retry: 3,
    refetchOnWindowFocus: true,
  });
  
  if (isLoading) return <div>加载中...</div>;
  if (error) return <div>错误: {error.message}</div>;
  
  return (
    <div>
      <h2>{data.name}</h2>
      <p>Email: {data.email}</p>
      <button onClick={refetch}>刷新</button>
    </div>
  );
}

// useMutation 使用
function CreateUserForm() {
  const queryClient = useQueryClient();
  
  const mutation = useMutation({
    mutationFn: async (newUser) => {
      const res = await fetch('/api/users', {
        method: 'POST',
        body: JSON.stringify(newUser),
        headers: { 'Content-Type': 'application/json' },
      });
      if (!res.ok) throw new Error('创建用户失败');
      return res.json();
    },
    onSuccess: (data) => {
      // 成功后刷新用户列表
      queryClient.invalidateQueries({ queryKey: ['users'] });
      alert(`用户 ${data.name} 创建成功`);
    },
    onError: (error) => {
      alert(`创建失败: ${error.message}`);
    },
  });
  
  const handleSubmit = (e) => {
    e.preventDefault();
    const formData = new FormData(e.target);
    mutation.mutate({
      name: formData.get('name'),
      email: formData.get('email'),
    });
  };
  
  return (
    <form onSubmit={handleSubmit}>
      <input name="name" placeholder="姓名" />
      <input name="email" type="email" placeholder="邮箱" />
      <button type="submit" disabled={mutation.isLoading}>
        {mutation.isLoading ? '创建中...' : '创建用户'}
      </button>
      {mutation.isError && (
        <p style={{ color: 'red' }}>{mutation.error.message}</p>
      )}
    </form>
  );
}
```

### 4.7.3 Zustand

```jsx
import { create } from 'zustand';

// 创建 store
const useStore = create((set, get) => ({
  // 状态
  count: 0,
  user: null,
  todos: [],
  
  // 计算属性
  get completedTodos() {
    return get().todos.filter(t => t.completed);
  },
  
  // Action
  increment: () => set((state) => ({ count: state.count + 1 })),
  decrement: () => set((state) => ({ count: state.count - 1 })),
  setUser: (user) => set({ user }),
  addTodo: (text) => set((state) => ({
    todos: [...state.todos, { id: Date.now(), text, completed: false }],
  })),
  toggleTodo: (id) => set((state) => ({
    todos: state.todos.map(t =>
      t.id === id ? { ...t, completed: !t.completed } : t
    ),
  })),
  
  // 异步 Action
  fetchUser: async (id) => {
    const response = await fetch(`/api/users/${id}`);
    const user = await response.json();
    set({ user });
  },
}));

// 在组件中使用
function ZustandCounter() {
  const count = useStore((state) => state.count);
  const increment = useStore((state) => state.increment);
  
  return (
    <div>
      <p>计数: {count}</p>
      <button onClick={increment}>增加</button>
    </div>
  );
}

// Zustand + Context 模式
function App() {
  return (
    <div>
      <ZustandCounter />
      <UserProfile userId={1} />
    </div>
  );
}

function UserProfile({ userId }) {
  const { user, fetchUser } = useStore();
  
  useEffect(() => {
    fetchUser(userId);
  }, [userId]);
  
  return <div>{user?.name || '加载中...'}</div>;
}
```

## 4.8 本章小结

自定义 Hooks 是 React 函数组件中复用状态逻辑的核心机制。通过合理设计和使用自定义 Hooks，可以显著提高代码的可维护性和可复用性。

**关键要点回顾：**

1. **自定义 Hook 以 use 开头**，可以调用其他 Hooks
2. **实用 Hooks**：useDebounce、useThrottle、usePrevious、useInterval、useMediaQuery、useLocalStorage、useFetch
3. **Hooks 组合**：多个 Hooks 可以组合使用，创建更高级的功能
4. **测试 Hooks**：使用 renderHook 和 act 进行测试
5. **useContext + useReducer**：实现轻量级状态管理
6. **forwardRef + useImperativeHandle**：向父组件暴露实例方法
7. **第三方 Hooks**：React Router、TanStack Query、Zustand 提供了丰富的 Hooks 生态
