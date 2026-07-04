# 第5章 React 性能优化实战

性能优化是 React 开发中至关重要的环节。本章将从组件优化、渲染优化、代码分割、工具使用等多个维度，系统性地介绍 React 性能优化的策略和实践。

## 5.1 React.memo

React.memo 是用于函数组件的高阶组件，通过浅比较 props 来决定是否跳过重渲染。

### 5.1.1 基本用法

```jsx
import { memo, useState } from 'react';

// 基础 memo 使用
const ExpensiveComponent = memo(function ExpensiveComponent({ name, count }) {
  console.log('ExpensiveComponent 渲染');
  
  // 模拟耗时渲染
  const startTime = performance.now();
  while (performance.now() - startTime < 5) {
    // 每次渲染耗时 5ms
  }
  
  return (
    <div>
      <h3>{name}</h3>
      <p>计数: {count}</p>
    </div>
  );
});

function Parent() {
  const [count, setCount] = useState(0);
  const [otherState, setOtherState] = useState(0);
  
  return (
    <div>
      <button onClick={() => setCount(c => c + 1)}>更新计数</button>
      <button onClick={() => setOtherState(c => c + 1)}>更新其他状态</button>
      
      <ExpensiveComponent name="组件" count={count} />
      {/* 当 otherState 变化时，ExpensiveComponent 不会重新渲染 */}
    </div>
  );
}
```

### 5.1.2 自定义比较函数

```jsx
import { memo } from 'react';

// 使用自定义比较函数
const UserCard = memo(
  function UserCard({ user, onSelect }) {
    console.log('UserCard 渲染');
    return (
      <div onClick={() => onSelect(user.id)}>
        <h4>{user.name}</h4>
        <p>角色: {user.role}</p>
        <p>最后登录: {user.lastLogin}</p>
      </div>
    );
  },
  (prevProps, nextProps) => {
    // 自定义比较：只有 user 的特定属性变化时才重新渲染
    return (
      prevProps.user.id === nextProps.user.id &&
      prevProps.user.name === nextProps.user.name &&
      prevProps.user.role === nextProps.user.role &&
      prevProps.onSelect === nextProps.onSelect
    );
  }
);

// 使用示例
function UserList() {
  const [users, setUsers] = useState([
    { id: 1, name: '张三', role: 'admin', lastLogin: '2024-01-01', lastUpdated: '1' },
    { id: 2, name: '李四', role: 'user', lastLogin: '2024-01-02', lastUpdated: '2' },
  ]);
  
  const handleSelect = useCallback((id) => {
    console.log('选中:', id);
  }, []);
  
  // 更新 lastUpdated 但 name 和 role 不变
  const updateTimestamp = () => {
    setUsers(prev => prev.map(u => ({
      ...u,
      lastUpdated: Date.now().toString(),
    })));
  };
  
  return (
    <div>
      <button onClick={updateTimestamp}>更新时间戳</button>
      {users.map(user => (
        <UserCard key={user.id} user={user} onSelect={handleSelect} />
      ))}
    </div>
  );
}
```

### 5.1.3 成本收益分析

```jsx
// memo 的成本收益分析

// 适合使用 memo 的场景：
// 1. 组件渲染开销大
// 2. props 经常不变
// 3. 组件在组件树中位置高
// 4. 组件包含大量子组件

// 不适合使用 memo 的场景：
// 1. 组件渲染开销小（<1ms）
// 2. props 几乎每次都会变化
// 3. 组件很简单（纯文本显示）

// memo 的成本
// 1. 内存开销：存储上一次的 props
// 2. 比较开销：浅比较 props
// 3. 维护成本：自定义比较函数可能出错

function MemoCostAnalysis() {
  return (
    <div>
      <h3>何时使用 memo</h3>
      <ul>
        <li>✅ 组件渲染包含大量计算或大量子元素</li>
        <li>✅ props 稳定，很少变化</li>
        <li>✅ 组件在组件树中频繁被父组件重渲染</li>
        <li>❌ 组件渲染极快（纯文本、简单元素）</li>
        <li>❌ props 每次都变化（如随机数、新对象）</li>
        <li>❌ 用于包裹本身就轻量的组件</li>
      </ul>
    </div>
  );
}
```

## 5.2 useMemo 和 useCallback 优化

### 5.2.1 何时使用

```jsx
import { useMemo, useCallback, useState, memo } from 'react';

// 场景 1：引用稳定性
function ReferenceStability() {
  const [count, setCount] = useState(0);
  
  // 需要稳定的引用传递给子组件或作为依赖
  const stableCallback = useCallback(() => {
    console.log('稳定的回调');
  }, []);
  
  // 需要稳定的对象引用
  const stableConfig = useMemo(() => ({
    theme: 'dark',
    fontSize: 14,
  }), []);
  
  return (
    <ChildComponent onAction={stableCallback} config={stableConfig} />
  );
}

// 场景 2：昂贵的计算
function ExpensiveCalculation({ numbers, filter }) {
  // 使用 useMemo 缓存计算结果
  const filteredAndProcessed = useMemo(() => {
    console.log('执行昂贵计算');
    
    // 过滤
    const filtered = numbers.filter(n => n > filter);
    
    // 排序
    const sorted = [...filtered].sort((a, b) => a - b);
    
    // 复杂计算
    const processed = sorted.map(n => {
      let result = 0;
      for (let i = 0; i < 100000; i++) {
        result += n * i;
      }
      return result;
    });
    
    return processed;
  }, [numbers, filter]);
  
  return (
    <div>
      <p>结果: {filteredAndProcessed.join(', ')}</p>
    </div>
  );
}
```

### 5.2.2 何时不使用

```jsx
// 场景 1：简单计算不需要 useMemo
function SimpleCalculation({ a, b }) {
  // ❌ 不必要的 useMemo
  const sum = useMemo(() => a + b, [a, b]);
  
  // ✅ 直接计算
  const sum2 = a + b;
  
  return <div>{sum2}</div>;
}

// 场景 2：原始值不需要 useCallback
function PrimitiveProps({ onClick }) {
  // ❌ 不必要的 useCallback
  const handleClick = useCallback(() => {
    onClick?.('clicked');
  }, [onClick]);
  
  // ✅ 如果 onClick 本身是稳定的，直接使用
  return <button onClick={onClick}>点击</button>;
}

// 场景 3：JSX 中的内联函数
function InlineFunctions() {
  const [count, setCount] = useState(0);
  
  return (
    <div>
      {/* ❌ useCallback 包裹内联事件处理器 */}
      <button onClick={useCallback(() => setCount(c => c + 1), [])}>
        增加
      </button>
      
      {/* ✅ 直接内联，React 的事件系统已经优化 */}
      <button onClick={() => setCount(c => c + 1)}>
        增加
      </button>
    </div>
  );
}

// 规则总结
function MemoGuidelines() {
  return (
    <div>
      <h3>使用 useMemo/useCallback 的指导原则</h3>
      <h4>应该使用:</h4>
      <ul>
        <li>计算开销大（O(n) 复杂、大量循环）</li>
        <li>作为 useEffect/useMemo 的依赖</li>
        <li>传递给 memo 包裹的子组件</li>
        <li>创建稳定引用的对象/函数</li>
      </ul>
      <h4>不应该使用:</h4>
      <ul>
        <li>简单计算（加减乘除、字符串拼接）</li>
        <li>原始值的 props</li>
        <li>仅渲染一次的组件</li>
        <li>useMemo 包裹 useState 的初始值（useState 已惰性初始化）</li>
      </ul>
    </div>
  );
}
```

## 5.3 Key 属性优化

### 5.3.1 使用稳定 Key

```jsx
import { useState } from 'react';

// ❌ 使用索引作为 key
function WrongKeyExample() {
  const [items, setItems] = useState(['A', 'B', 'C', 'D']);
  
  const removeFirst = () => {
    setItems(prev => prev.slice(1)); // 移除第一项
  };
  
  const insertFirst = () => {
    setItems(prev => [`新项-${Date.now()}`, ...prev]);
  };
  
  return (
    <div>
      <button onClick={removeFirst}>移除第一项</button>
      <button onClick={insertFirst}>插入到开头</button>
      
      {items.map((item, index) => (
        // 问题：删除第一项后，所有 item 的 key 都变了
        // 原来 key=1 的 B 变成了 key=0，React 会复用错误的 DOM
        <div key={index}>
          <input defaultValue={item} />
          <span>{item}</span>
        </div>
      ))}
      
      <p>注意：使用索引作为 key 时，删除第一项后 input 的值不会跟随</p>
    </div>
  );
}

// ✅ 使用唯一 ID 作为 key
function CorrectKeyExample() {
  const [items, setItems] = useState(
    ['A', 'B', 'C', 'D'].map(text => ({
      id: crypto.randomUUID(),
      text,
    }))
  );
  
  const removeFirst = () => {
    setItems(prev => prev.slice(1));
  };
  
  return (
    <div>
      <button onClick={removeFirst}>移除第一项</button>
      
      {items.map(item => (
        <div key={item.id}>
          <input defaultValue={item.text} />
          <span>{item.text}</span>
        </div>
      ))}
      
      <p>注意：使用唯一 ID 作为 key，删除后 input 的关联正确</p>
    </div>
  );
}
```

### 5.3.2 列表渲染最佳实践

```jsx
// 列表渲染最佳实践
function ListRenderingBestPractices() {
  const [items, setItems] = useState([]);
  const [filter, setFilter] = useState('');
  
  // 1. 在渲染前过滤/排序数据
  const filteredItems = useMemo(() => {
    return items.filter(item =>
      item.name.toLowerCase().includes(filter.toLowerCase())
    );
  }, [items, filter]);
  
  // 2. 使用稳定的 key
  // 3. 避免在渲染中创建新对象
  return (
    <div>
      <input
        value={filter}
        onChange={(e) => setFilter(e.target.value)}
        placeholder="搜索..."
      />
      
      <ul>
        {filteredItems.map(item => (
          <ListItem key={item.id} item={item} />
        ))}
      </ul>
    </div>
  );
}

// 4. 将列表项提取为独立组件
const ListItem = memo(function ListItem({ item }) {
  return (
    <li>
      <span>{item.name}</span>
      <span>{item.description}</span>
    </li>
  );
});
```

## 5.4 虚拟列表（Virtualization）

### 5.4.1 react-window

```jsx
import { FixedSizeList, VariableSizeList } from 'react-window';

// FixedSizeList：所有行高度相同
function FixedHeightList() {
  const items = Array.from({ length: 10000 }, (_, i) => ({
    id: i,
    text: `项目 #${i}`,
    description: `这是第 ${i} 个项目的详细描述`,
  }));
  
  // Row 组件，接收 style 和 index
  const Row = ({ index, style }) => {
    const item = items[index];
    return (
      <div style={style}>
        <div style={{ padding: '8px', borderBottom: '1px solid #ccc' }}>
          <strong>{item.text}</strong>
          <p>{item.description}</p>
        </div>
      </div>
    );
  };
  
  return (
    <FixedSizeList
      height={400}        // 容器高度
      width="100%"         // 容器宽度
      itemCount={items.length}  // 总项目数
      itemSize={60}        // 每行高度
    >
      {Row}
    </FixedSizeList>
  );
}

// VariableSizeList：行高不同
function VariableHeightList() {
  const items = Array.from({ length: 10000 }, (_, i) => ({
    id: i,
    text: `项目 #${i}`,
    lines: Math.floor(Math.random() * 5) + 1, // 随机行数
  }));
  
  const Row = ({ index, style }) => {
    const item = items[index];
    return (
      <div style={style}>
        <div style={{ padding: '8px', borderBottom: '1px solid #ccc' }}>
          <strong>{item.text}</strong>
          {Array.from({ length: item.lines }).map((_, i) => (
            <p key={i}>详细内容行 #{i + 1}</p>
          ))}
        </div>
      </div>
    );
  };
  
  return (
    <VariableSizeList
      height={400}
      width="100%"
      itemCount={items.length}
      itemSize={(index) => 40 + items[index].lines * 20} // 动态计算高度
    >
      {Row}
    </VariableSizeList>
  );
}
```

### 5.4.2 react-virtuoso

```jsx
import { Virtuoso, VirtuosoGrid } from 'react-virtuoso';

// Virtuoso：功能更丰富的虚拟列表
function AdvancedVirtualList() {
  const items = Array.from({ length: 100000 }, (_, i) => ({
    id: i,
    text: `项目 #${i}`,
    timestamp: Date.now() - i * 3600000,
  }));
  
  return (
    <Virtuoso
      style={{ height: '500px' }}
      totalCount={items.length}
      itemContent={(index) => {
        const item = items[index];
        return (
          <div style={{
            padding: '12px',
            borderBottom: '1px solid #e0e0e0',
            display: 'flex',
            justifyContent: 'space-between',
          }}>
            <span>{item.text}</span>
            <span style={{ color: '#888' }}>
              {new Date(item.timestamp).toLocaleString()}
            </span>
          </div>
        );
      }}
      // 高级特性
      components={{
        Header: () => <div style={{ padding: '12px', fontWeight: 'bold' }}>列表头部</div>,
        Footer: () => <div style={{ padding: '12px', textAlign: 'center' }}>列表底部</div>,
      }}
      // 滚动到顶部/底部回调
      atTopStateChange={(atTop) => console.log('是否在顶部:', atTop)}
      atBottomStateChange={(atBottom) => {
        if (atBottom) console.log('到达底部');
      }}
    />
  );
}

// 网格布局
function VirtualGrid() {
  const items = Array.from({ length: 1000 }, (_, i) => ({
    id: i,
    image: `https://picsum.photos/200/200?random=${i}`,
    title: `图片 #${i}`,
  }));
  
  return (
    <VirtuosoGrid
      style={{ height: '500px' }}
      totalCount={items.length}
      listClassName="grid-list"
      itemContent={(index) => (
        <div className="grid-item">
          <img src={items[index].image} alt={items[index].title} />
          <p>{items[index].title}</p>
        </div>
      )}
    />
  );
}
```

## 5.5 Suspense Streaming SSR

### 5.5.1 render-as-you-fetch

```jsx
import { Suspense, lazy } from 'react';

// 传统方式：fetch-then-render
function TraditionalApproach() {
  const [data, setData] = useState(null);
  
  useEffect(() => {
    fetch('/api/data')
      .then(res => res.json())
      .then(setData);
  }, []);
  
  if (!data) return <div>加载中...</div>;
  return <ExpensiveComponent data={data} />;
}

// 使用 Suspense 的 render-as-you-fetch
function SuspenseApproach() {
  return (
    <Suspense fallback={<div>加载中...</div>}>
      <DataComponent />
    </Suspense>
  );
}

// 流式 SSR 示例
function StreamSSRExample() {
  return (
    <html>
      <head>
        <title>流式 SSR</title>
      </head>
      <body>
        {/* 立即发送的 HTML */}
        <header>
          <h1>我的应用</h1>
          <nav>导航...</nav>
        </header>
        
        {/* 异步流式内容 */}
        <Suspense fallback={<div>加载主内容...</div>}>
          <MainContent />
        </Suspense>
        
        {/* 低优先级内容 */}
        <Suspense fallback={<div>加载评论...</div>}>
          <Comments />
        </Suspense>
        
        <footer>页脚</footer>
      </body>
    </html>
  );
}
```

## 5.6 代码分割

### 5.6.1 React.lazy 和 Suspense

```jsx
import { lazy, Suspense } from 'react';

// 懒加载组件
const Dashboard = lazy(() => import('./Dashboard'));
const Settings = lazy(() => import('./Settings'));
const Analytics = lazy(() => import('./Analytics'));

// 路由级别的代码分割
function App() {
  const [page, setPage] = useState('dashboard');
  
  const PageComponent = useMemo(() => {
    switch (page) {
      case 'dashboard': return Dashboard;
      case 'settings': return Settings;
      case 'analytics': return Analytics;
      default: return Dashboard;
    }
  }, [page]);
  
  return (
    <div>
      <nav>
        <button onClick={() => setPage('dashboard')}>仪表盘</button>
        <button onClick={() => setPage('settings')}>设置</button>
        <button onClick={() => setPage('analytics')}>分析</button>
      </nav>
      
      <Suspense fallback={<div>加载页面中...</div>}>
        <PageComponent />
      </Suspense>
    </div>
  );
}

// 条件加载
function ConditionalLoad({ isLoggedIn }) {
  return (
    <div>
      <Suspense fallback={<div>加载...</div>}>
        {isLoggedIn ? (
          <AdminPanel />
        ) : (
          <PublicContent />
        )}
      </Suspense>
    </div>
  );
}
```

### 5.6.2 命名导出处理

```jsx
// 对于命名导出的组件，需要中间模块
// lazy 只支持 default export

// 方式 1：中间模块
// components/Dashboard.js
// export const Dashboard = () => { ... };
// export default Dashboard;

// 方式 2：使用 .then
const Dashboard = lazy(() =>
  import('./Dashboard').then(module => ({
    default: module.Dashboard,
  }))
);

// 方式 3：重新导出
// components/index.js
// export { Dashboard } from './Dashboard';
// export { Settings } from './Settings';
// export default { Dashboard, Settings };

// 然后使用
const { Dashboard } = lazy(() =>
  import('./components').then(module => ({
    default: module.Dashboard,
  }))
);
```

## 5.7 避免不必要的重渲染

### 5.7.1 组件拆分

```jsx
// ❌ 反模式：大组件导致不必要的重渲染
function BigComponent() {
  const [count, setCount] = useState(0);
  const [text, setText] = useState('');
  
  return (
    <div>
      <div>
        <p>计数: {count}</p>
        <button onClick={() => setCount(c => c + 1)}>增加</button>
      </div>
      <div>
        <input
          value={text}
          onChange={(e) => setText(e.target.value)}
        />
      </div>
      {/* 输入 text 时，整个组件都会重渲染 */}
      <ExpensiveSection />
      <AnotherExpensiveSection />
    </div>
  );
}

// ✅ 正确做法：拆分组件，隔离状态
function GoodExample() {
  return (
    <div>
      <CounterSection />
      <InputSection />
      <ExpensiveSection />
      <AnotherExpensiveSection />
    </div>
  );
}

function CounterSection() {
  const [count, setCount] = useState(0);
  
  return (
    <div>
      <p>计数: {count}</p>
      <button onClick={() => setCount(c => c + 1)}>增加</button>
    </div>
  );
}

function InputSection() {
  const [text, setText] = useState('');
  
  return (
    <div>
      <input value={text} onChange={(e) => setText(e.target.value)} />
    </div>
  );
}
```

### 5.7.2 状态提升

```jsx
// 状态提升：将状态放在需要它的最小公共祖先

// ❌ 反模式：状态放在不必要的祖先中
function App() {
  const [count, setCount] = useState(0);
  
  return (
    <div>
      <Header /> {/* 不需要 count */}
      <Sidebar /> {/* 不需要 count */}
      <MainContent count={count} onIncrement={() => setCount(c => c + 1)} />
      <Footer /> {/* 不需要 count */}
    </div>
  );
}

// ✅ 正确做法：状态放在需要的组件附近
function App() {
  return (
    <div>
      <Header />
      <Sidebar />
      <MainContentWrapper />
      <Footer />
    </div>
  );
}

function MainContentWrapper() {
  const [count, setCount] = useState(0);
  
  return (
    <MainContent
      count={count}
      onIncrement={() => setCount(c => c + 1)}
    />
  );
}
```

### 5.7.3 Content 提升

```jsx
// Content 提升：将不变化的 JSX 提升到父组件
function ParentComponent() {
  const [count, setCount] = useState(0);
  
  // 将不变化的 JSX 作为 children 传入
  return (
    <ExpensiveLayout
      header={<Header />}
      sidebar={<Sidebar />}
    >
      <MainContent count={count} />
    </ExpensiveLayout>
  );
}

// ExpensiveLayout 不会因为 header/sidebar 变化而重渲染
const ExpensiveLayout = memo(function ExpensiveLayout({
  header,
  sidebar,
  children,
}) {
  console.log('ExpensiveLayout 渲染');
  return (
    <div>
      <header>{header}</header>
      <aside>{sidebar}</aside>
      <main>{children}</main>
    </div>
  );
});
```

## 5.8 React DevTools Profiler

### 5.8.1 Profiler 使用

```jsx
import { Profiler } from 'react';

// 使用 Profiler 测量渲染性能
function onRenderCallback(
  id,                  // 标识符
  phase,               // "mount" 或 "update"
  actualDuration,      // 本次渲染实际耗时
  baseDuration,        // 子组件渲染耗时
  startTime,           // 开始时间
  commitTime,          // 提交时间
  interactions         // 交互集合
) {
  // 将性能数据发送到分析服务
  if (actualDuration > 16) { // 超过 16ms（60fps 的一帧）
    console.warn(`组件 ${id} 渲染耗时过长: ${actualDuration.toFixed(2)}ms`);
  }
  
  // 收集统计数据
  performanceMetrics.push({
    id,
    phase,
    actualDuration,
    baseDuration,
    timestamp: Date.now(),
  });
}

function App() {
  return (
    <Profiler id="App" onRender={onRenderCallback}>
      <div>
        <Profiler id="Header" onRender={onRenderCallback}>
          <Header />
        </Profiler>
        
        <Profiler id="MainContent" onRender={onRenderCallback}>
          <MainContent />
        </Profiler>
        
        <Profiler id="Footer" onRender={onRenderCallback}>
          <Footer />
        </Profiler>
      </div>
    </Profiler>
  );
}

// 生产环境性能监控
function useRenderTiming(componentName) {
  const renderCount = useRef(0);
  const totalRenderTime = useRef(0);
  
  renderCount.current += 1;
  const startTime = performance.now();
  
  useEffect(() => {
    const renderTime = performance.now() - startTime;
    totalRenderTime.current += renderTime;
    
    // 记录到性能监控系统
    if (renderTime > 16) {
      reportSlowRender({
        component: componentName,
        renderTime,
        renderCount: renderCount.current,
        averageTime: totalRenderTime.current / renderCount.current,
      });
    }
  });
}
```

## 5.9 性能测量工具

### 5.9.1 performance.now

```jsx
// 使用 performance.now 测量渲染性能
function PerformanceMeasurement() {
  const renderStart = useRef(performance.now());
  
  // 测量渲染时间
  useEffect(() => {
    const renderDuration = performance.now() - renderStart.current;
    console.log(`渲染耗时: ${renderDuration.toFixed(2)}ms`);
    
    if (renderDuration > 16) {
      console.warn('渲染时间过长，建议优化');
    }
  });
  
  // 更新开始时间
  renderStart.current = performance.now();
  
  return <div>...</div>;
}

// 测量操作性能
function measureAsyncOperation(name, asyncFn) {
  const start = performance.now();
  
  return asyncFn().finally(() => {
    const duration = performance.now() - start;
    console.log(`${name}: ${duration.toFixed(2)}ms`);
    
    if (duration > 100) {
      reportSlowOperation(name, duration);
    }
  });
}

// 使用 Performance Observer
function usePerformanceObserver() {
  useEffect(() => {
    if (typeof PerformanceObserver === 'undefined') return;
    
    const observer = new PerformanceObserver((list) => {
      const entries = list.getEntries();
      
      entries.forEach(entry => {
        if (entry.entryType === 'longtask') {
          console.warn('长任务:', entry.duration, 'ms');
        }
        
        if (entry.entryType === 'measure') {
          console.log(`测量 [${entry.name}]: ${entry.duration}ms`);
        }
      });
    });
    
    observer.observe({ entryTypes: ['longtask', 'measure'] });
    
    return () => observer.disconnect();
  }, []);
}
```

## 5.10 本章小结

React 性能优化是一个系统工程，需要从多个维度综合考虑。

**关键要点回顾：**

1. **React.memo** 通过浅比较 props 减少不必要的重渲染，但需要评估成本收益
2. **useMemo/useCallback** 用于引用稳定性和昂贵计算，避免过早优化
3. **稳定 Key** 确保列表渲染的正确性和性能，避免使用索引作为 key
4. **虚拟列表** 处理大量数据渲染，react-window 和 react-virtuoso 是主流方案
5. **Suspense Streaming SSR** 实现流式渲染，改善首屏加载体验
6. **代码分割** 通过 React.lazy 和动态 import 减小初始包体积
7. **避免不必要的重渲染** 通过组件拆分、状态提升和 Content 提升
8. **React DevTools Profiler** 和 **performance.now** 帮助识别性能瓶颈
9. **性能优化优先级**：先测量再优化，避免过早优化
10. **核心原则**：减少渲染次数、减少渲染工作量、减少包体积
