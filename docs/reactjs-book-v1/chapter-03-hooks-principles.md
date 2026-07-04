# 第3章 React Hooks 核心原理

React Hooks 自 React 16.8 引入以来，彻底改变了 React 组件的编写方式。本章将深入探讨 Hooks 的设计动机、核心原理以及每个内置 Hook 的详细工作机制。

## 3.1 Hooks 解决的问题

在 Hooks 出现之前，React 开发者面临几个长期存在的问题。

### 3.1.1 包装器地狱（Wrapper Hell）

```jsx
// Hooks 之前的组件嵌套问题
// 使用高阶组件（HOC）和 render props 导致的深层嵌套

class UserProfile extends React.Component {
  render() {
    return (
      <WithAuth>
        {({ user }) => (
          <WithTheme>
            {({ theme }) => (
              <WithRouter>
                {({ location }) => (
                  <WithData source="/api/user">
                    {({ data }) => (
                      <div style={{ color: theme.primary }}>
                        <h1>{user.name}</h1>
                        <p>当前路径: {location.pathname}</p>
                        <pre>{JSON.stringify(data, null, 2)}</pre>
                      </div>
                    )}
                  </WithData>
                )}
              </WithRouter>
            )}
          </WithTheme>
        )}
      </WithAuth>
    );
  }
}

// 使用 Hooks 后的扁平化结构
function UserProfile() {
  const { user } = useAuth();
  const { theme } = useTheme();
  const { location } = useLocation();
  const { data } = useData('/api/user');
  
  return (
    <div style={{ color: theme.primary }}>
      <h1>{user.name}</h1>
      <p>当前路径: {location.pathname}</p>
      <pre>{JSON.stringify(data, null, 2)}</pre>
    </div>
  );
}
```

### 3.1.2 生命周期逻辑分散

```jsx
// Hooks 之前的生命周期混乱
// 相关的逻辑被分散在不同的生命周期方法中

class FriendStatus extends React.Component {
  constructor(props) {
    super(props);
    this.state = { isOnline: null };
    this.handleStatusChange = this.handleStatusChange.bind(this);
  }
  
  componentDidMount() {
    // 订阅好友状态
    ChatAPI.subscribeToFriendStatus(
      this.props.friend.id,
      this.handleStatusChange
    );
    // 日志记录
    this.logVisit();
  }
  
  componentDidUpdate(prevProps) {
    // 好友 ID 变化时重新订阅
    if (prevProps.friend.id !== this.props.friend.id) {
      ChatAPI.unsubscribeFromFriendStatus(
        prevProps.friend.id,
        this.handleStatusChange
      );
      ChatAPI.subscribeToFriendStatus(
        this.props.friend.id,
        this.handleStatusChange
      );
    }
    // 日志记录
    this.logVisit();
  }
  
  componentWillUnmount() {
    // 取消订阅
    ChatAPI.unsubscribeFromFriendStatus(
      this.props.friend.id,
      this.handleStatusChange
    );
  }
  
  handleStatusChange(status) {
    this.setState({ isOnline: status.isOnline });
  }
  
  logVisit() {
    analytics.logVisit(this.props.friend.id);
  }
  
  render() {
    return <div>{this.state.isOnline ? '在线' : '离线'}</div>;
  }
}

// 使用 Hooks 后，相关逻辑集中在一起
function FriendStatus({ friend }) {
  const [isOnline, setIsOnline] = useState(null);
  
  useEffect(() => {
    // 订阅好友状态
    ChatAPI.subscribeToFriendStatus(friend.id, setIsOnline);
    
    // 返回清理函数
    return () => {
      ChatAPI.unsubscribeFromFriendStatus(friend.id, setIsOnline);
    };
  }, [friend.id]); // 依赖 friend.id
  
  useEffect(() => {
    analytics.logVisit(friend.id);
  }, [friend.id]);
  
  return <div>{isOnline ? '在线' : '离线'}</div>;
}
```

### 3.1.3 状态逻辑难以复用

```jsx
// Hooks 之前的状态逻辑复用
// 必须通过高阶组件或 render props

// 高阶组件方式
function withWindowSize(Component) {
  return class extends React.Component {
    constructor(props) {
      super(props);
      this.state = { width: window.innerWidth, height: window.innerHeight };
    }
    
    componentDidMount() {
      window.addEventListener('resize', this.handleResize);
    }
    
    componentWillUnmount() {
      window.removeEventListener('resize', this.handleResize);
    }
    
    handleResize = () => {
      this.setState({
        width: window.innerWidth,
        height: window.innerHeight,
      });
    };
    
    render() {
      return (
        <Component
          {...this.props}
          windowSize={this.state}
        />
      );
    }
  };
}

// 使用自定义 Hooks 复用状态逻辑
function useWindowSize() {
  const [size, setSize] = useState({
    width: window.innerWidth,
    height: window.innerHeight,
  });
  
  useEffect(() => {
    const handleResize = () => {
      setSize({ width: window.innerWidth, height: window.innerHeight });
    };
    
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);
  
  return size;
}

// 任何组件都可以直接使用
function MyComponent() {
  const { width, height } = useWindowSize();
  return <div>窗口大小: {width} x {height}</div>;
}
```

## 3.2 Hooks 规则

React Hooks 有两条核心规则，违反它们会导致 bug。

### 3.2.1 只在顶层调用 Hooks

```jsx
// 规则 1：只在函数组件或自定义 Hook 的顶层调用 Hooks
// 不要在循环、条件语句或嵌套函数中调用 Hooks

// 错误示范
function BadExample({ shouldLog }) {
  if (shouldLog) {
    useEffect(() => {
      // 错误：在条件语句中使用 Hook
      console.log('logging');
    });
  }
  
  for (let i = 0; i < 10; i++) {
    useState(i);
    // 错误：在循环中使用 Hook
  }
  
  return <div>错误示例</div>;
}

// 正确做法
function GoodExample({ shouldLog }) {
  // 始终在顶层调用
  useEffect(() => {
    if (shouldLog) {
      console.log('logging');
    }
  }, [shouldLog]);
  
  const states = [];
  // 用一个 state 管理数组
  const [items] = useState(Array.from({ length: 10 }, (_, i) => i));
  
  return <div>正确示例</div>;
}
```

### 3.2.2 只在 React 函数组件和自定义 Hooks 中调用

```jsx
// 规则 2：只在 React 函数组件和自定义 Hooks 中调用 Hooks

// 错误示范：在普通函数中使用
function getWindowWidth() {
  // 错误：普通函数不能使用 Hooks
  const [width, setWidth] = useState(window.innerWidth);
  return width;
}

// 错误示范：在类组件中使用
class MyClass extends React.Component {
  render() {
    // 错误：类组件不能使用 Hooks
    const [count, setCount] = useState(0);
    return <div>{count}</div>;
  }
}

// 正确做法
function useWindowWidth() {
  // 正确：自定义 Hook 以 use 开头
  const [width, setWidth] = useState(window.innerWidth);
  return width;
}

// 正确做法：在函数组件中使用
function MyComponent() {
  const width = useWindowWidth();
  return <div>宽度: {width}</div>;
}
```

### 3.2.3 为什么需要这些规则

```jsx
// Hooks 规则的底层原因：Fiber 上的 Hooks 链表

// 在 Fiber 节点上，Hooks 以链表形式存储
// 每个 Hook 在链表中有固定的位置（索引）
// 条件/循环会破坏链表的顺序

// Hooks 链表的简化表示
function renderWithHooks(component, props, ref) {
  // 重置 Hooks 链表指针
  currentlyRenderingFiber = workInProgress;
  workInProgress.memoizedState = null; // 清空 Hooks 链表
  nextHookIndex = 0;
  
  // 执行函数组件
  const children = component(props, ref);
  
  // 完成后，Hooks 链表被构建在 memoizedState 上
  return children;
}

// useState 的内部实现（简化）
function useState(initialState) {
  // 获取当前 Hook 节点
  const hook = mountWorkInProgressHook();
  
  if (typeof initialState === 'function') {
    initialState = initialState();
  }
  
  hook.memoizedState = hook.baseState = initialState;
  hook.queue = {
    pending: null,
    dispatch: null,
    lastRenderedReducer: basicStateReducer,
    lastRenderedState: initialState,
  };
  
  const dispatch = dispatchAction.bind(
    null,
    currentlyRenderingFiber,
    hook.queue
  );
  hook.queue.dispatch = dispatch;
  
  return [hook.memoizedState, dispatch];
}

// 如果条件调用导致 Hook 跳过，链表顺序会错乱
// 第一次渲染：useState('A') → useState('B') → useState('C')
// 链表: A → B → C
// 第二次渲染（条件跳过第二个）：useState('A') → useState('C')
// 链表: A → C → ??? （第三个变成了第二个，状态错乱）
```

## 3.3 useState

useState 是最基础的 Hook，用于在函数组件中添加状态。

### 3.3.1 基本用法

```jsx
import { useState } from 'react';

// 基本用法
function Counter() {
  const [count, setCount] = useState(0);
  
  return (
    <div>
      <p>计数: {count}</p>
      <button onClick={() => setCount(count + 1)}>增加</button>
      <button onClick={() => setCount(prev => prev - 1)}>减少</button>
    </div>
  );
}

// 延迟初始化
function ExpensiveInitialization() {
  // 传入函数，只在首次渲染时执行
  const [state, setState] = useState(() => {
    const initialState = computeExpensiveValue();
    return initialState;
  });
  
  return <div>{state}</div>;
}

function computeExpensiveValue() {
  // 模拟耗时计算
  let result = 0;
  for (let i = 0; i < 1000000; i++) {
    result += i;
  }
  return result;
}
```

### 3.3.2 状态更新与批处理

```jsx
// React 18 的自动批处理
function BatchExample() {
  const [count, setCount] = useState(0);
  const [flag, setFlag] = useState(false);
  
  function handleClick() {
    // React 18 中，这些更新会被自动批处理
    // 组件只重新渲染一次
    setCount(c => c + 1);
    setFlag(f => !f);
    // 相当于：
    // setCount(c + 1);
    // setFlag(!flag);
    // React 18 会自动合并这些更新
  }
  
  // 在 React 18 中，即使在 setTimeout 或 Promise 中也会批处理
  function handleAsyncClick() {
    fetch('/api/data').then(() => {
      // React 18 中，这些也会被批处理
      setCount(c => c + 1);
      setFlag(f => !f);
    });
  }
  
  // 如果需要强制不批处理，使用 flushSync
  function handleFlushClick() {
    import('react-dom').then(({ flushSync }) => {
      flushSync(() => {
        setCount(c => c + 1);
      });
      // DOM 已更新
      flushSync(() => {
        setFlag(f => !f);
      });
      // DOM 再次更新
    });
  }
  
  console.log('渲染了'); // React 18 中只打印一次
  
  return (
    <div>
      <p>Count: {count}, Flag: {String(flag)}</p>
      <button onClick={handleClick}>批处理更新</button>
      <button onClick={handleAsyncClick}>异步批处理</button>
    </div>
  );
}
```

### 3.3.3 更新队列处理

```jsx
// useState 的更新队列处理机制（简化）

// 基本 reducer：useState 使用的默认 reducer
function basicStateReducer(state, action) {
  // 如果 action 是函数，调用它
  return typeof action === 'function' ? action(state) : action;
}

// 调度更新
function dispatchAction(fiber, queue, action) {
  // 创建更新对象
  const update = {
    lane,
    action,
    next: null,
  };
  
  // 将更新加入队列
  const pending = queue.pending;
  if (pending === null) {
    // 首次更新
    update.next = update;
  } else {
    update.next = pending.next;
    pending.next = update;
  }
  queue.pending = update;
  
  // 调度更新
  const root = markUpdateLaneFromFiberToRoot(fiber);
  ensureRootIsScheduled(root);
}

// 处理更新队列
function processUpdateQueue(workInProgress, queue, props, instance, renderLanes) {
  let newState = queue.lastRenderedState;
  let update = queue.firstBaseUpdate;
  
  // 遍历更新队列
  while (update !== null) {
    const action = update.action;
    
    // 应用更新
    newState = basicStateReducer(newState, action);
    
    update = update.next;
  }
  
  // 更新队列中的状态
  queue.lastRenderedState = newState;
  
  return newState;
}

// 函数式更新的优势
function Counter() {
  const [count, setCount] = useState(0);
  
  // 在批处理中，函数式更新可以正确累加
  function handleTripleIncrement() {
    setCount(c => c + 1); // c = 0 → 1
    setCount(c => c + 1); // c = 1 → 2
    setCount(c => c + 1); // c = 2 → 3
    // 最终结果: 3
  }
  
  // 普通更新会覆盖
  function handleWrongTriple() {
    setCount(count + 1); // count + 1 = 1
    setCount(count + 1); // count + 1 = 1
    setCount(count + 1); // count + 1 = 1
    // 最终结果: 1（所有更新都基于同一个快照）
  }
  
  return (
    <div>
      <p>{count}</p>
      <button onClick={handleTripleIncrement}>正确递增</button>
      <button onClick={handleWrongTriple}>错误递增</button>
    </div>
  );
}
```

## 3.4 useEffect

useEffect 用于在函数组件中执行副作用操作。

### 3.4.1 基本用法与执行时机

```jsx
import { useEffect, useState } from 'react';

// useEffect 的执行时机
function EffectTiming() {
  const [count, setCount] = useState(0);
  
  // 无依赖：每次渲染后执行
  useEffect(() => {
    console.log('每次渲染后执行');
  });
  
  // 空依赖：只在挂载后执行一次
  useEffect(() => {
    console.log('只在挂载时执行');
  }, []);
  
  // 有依赖：依赖变化时执行
  useEffect(() => {
    console.log('count 变化时执行');
  }, [count]);
  
  return <button onClick={() => setCount(c => c + 1)}>{count}</button>;
}
```

### 3.4.2 清理函数

```jsx
// useEffect 清理函数
function useEventListener(eventName, handler, element = window) {
  useEffect(() => {
    // 添加事件监听
    element.addEventListener(eventName, handler);
    
    // 清理函数：在组件卸载或依赖变化时执行
    return () => {
      element.removeEventListener(eventName, handler);
    };
  }, [eventName, handler, element]);
}

// 实际使用
function MouseTracker() {
  const [position, setPosition] = useState({ x: 0, y: 0 });
  
  useEffect(() => {
    const handleMouseMove = (e) => {
      setPosition({ x: e.clientX, y: e.clientY });
    };
    
    window.addEventListener('mousemove', handleMouseMove);
    
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
    };
  }, []);
  
  return (
    <div>
      鼠标位置: ({position.x}, {position.y})
    </div>
  );
}
```

### 3.4.3 依赖比较

```jsx
// React 使用 Object.is 进行依赖比较
// 注意：Object.is 是浅比较

function DependencyComparison() {
  const [config, setConfig] = useState({ theme: 'dark', lang: 'zh' });
  
  // 每次渲染都创建新对象，导致无限循环
  useEffect(() => {
    console.log('这会导致无限循环');
  }, [{ theme: 'dark', lang: 'zh' }]);
  // 每次渲染，新对象 !== 旧对象
  
  // 正确做法：使用原始值作为依赖
  useEffect(() => {
    console.log('使用原始值作为依赖');
  }, [config.theme, config.lang]);
  
  // 或者使用 useMemo 稳定引用
  const stableConfig = useMemo(
    () => ({ theme: 'dark', lang: 'zh' }),
    []
  );
  
  useEffect(() => {
    console.log('稳定的引用');
  }, [stableConfig]);
  
  return <div>依赖比较示例</div>;
}

// 常见依赖陷阱
function CommonPitfalls() {
  const [count, setCount] = useState(0);
  
  // 陷阱 1：缺少依赖
  useEffect(() => {
    const timer = setInterval(() => {
      console.log(count); // 始终打印 0
    }, 1000);
    return () => clearInterval(timer);
  }, []); // 缺少 count 依赖
  
  // 陷阱 2：不必要的依赖
  const fetchData = async () => {
    const result = await fetch('/api/data');
    return result.json();
  };
  
  useEffect(() => {
    fetchData().then(data => {
      console.log(data);
    });
  }, [fetchData]); // fetchData 每次渲染都重新创建
  
  // 解决方案：使用 useCallback 或定义在 useEffect 内部
  useEffect(() => {
    const fetchData = async () => {
      const result = await fetch('/api/data');
      return result.json();
    };
    fetchData().then(data => console.log(data));
  }, []);
  
  return <div>常见陷阱</div>;
}
```

### 3.4.4 Strict Mode 双次调用

```jsx
// React Strict Mode 在开发环境下会双次调用 effect
// 用于检测未正确清理的副作用

function StrictModeEffect() {
  const [data, setData] = useState(null);
  
  useEffect(() => {
    console.log('Effect 执行');
    
    // 在 Strict Mode 中，这个 effect 会被执行两次
    // 第一次立即被清理（清理函数执行）
    // 第二次正常执行
    
    const controller = new AbortController();
    
    fetch('/api/data', { signal: controller.signal })
      .then(res => res.json())
      .then(data => {
        if (!controller.signal.aborted) {
          setData(data);
        }
      });
    
    return () => {
      console.log('清理函数执行');
      controller.abort(); // 正确的清理
    };
  }, []);
  
  return <div>{data ? JSON.stringify(data) : '加载中...'}</div>;
}
```

## 3.5 useRef

useRef 提供了一种在渲染之间持久化值的方式。

### 3.5.1 基本用法

```jsx
import { useRef, useEffect } from 'react';

// DOM 引用
function AutoFocusInput() {
  const inputRef = useRef(null);
  
  useEffect(() => {
    // 组件挂载后自动聚焦
    inputRef.current.focus();
  }, []);
  
  return <input ref={inputRef} type="text" />;
}

// 可变值（不会触发重新渲染）
function Timer() {
  const [count, setCount] = useState(0);
  const intervalRef = useRef(null);
  
  const startTimer = () => {
    intervalRef.current = setInterval(() => {
      setCount(c => c + 1);
    }, 1000);
  };
  
  const stopTimer = () => {
    clearInterval(intervalRef.current);
  };
  
  useEffect(() => {
    return () => clearInterval(intervalRef.current);
  }, []);
  
  return (
    <div>
      <p>{count} 秒</p>
      <button onClick={startTimer}>开始</button>
      <button onClick={stopTimer}>停止</button>
    </div>
  );
}

// 保存前一个值
function usePrevious(value) {
  const ref = useRef();
  
  useEffect(() => {
    ref.current = value;
  }, [value]);
  
  return ref.current; // 返回前一个值
}

function CounterWithPrevious() {
  const [count, setCount] = useState(0);
  const prevCount = usePrevious(count);
  
  return (
    <div>
      <p>
        当前: {count}, 前一个: {prevCount}
      </p>
      <button onClick={() => setCount(c => c + 1)}>增加</button>
    </div>
  );
}
```

### 3.5.2 React 19 中的 ref 作为 prop

```jsx
// React 19 中，ref 可以直接作为 prop 传递
// 不再需要 forwardRef

// React 18 及之前
const MyInput = forwardRef((props, ref) => {
  return <input ref={ref} {...props} />;
});

// React 19
function MyInput({ ref, ...props }) {
  return <input ref={ref} {...props} />;
}

// 父组件
function Parent() {
  const inputRef = useRef(null);
  
  useEffect(() => {
    inputRef.current.focus();
  }, []);
  
  return <MyInput ref={inputRef} placeholder="React 19 ref" />;
}
```

## 3.6 useMemo 和 useCallback

用于性能优化的记忆化 Hooks。

### 3.6.1 useMemo

```jsx
import { useMemo, useState } from 'react';

// useMemo 用于缓存计算结果
function ExpensiveComputation({ numbers }) {
  // 只在 numbers 变化时重新计算
  const total = useMemo(() => {
    console.log('执行耗时计算');
    return numbers.reduce((sum, n) => {
      // 模拟耗时操作
      let result = 0;
      for (let i = 0; i < 1000000; i++) {
        result += n * i;
      }
      return sum + result;
    }, 0);
  }, [numbers]);
  
  return <div>计算结果: {total}</div>;
}

// useMemo 用于稳定引用
function StableReferences() {
  const [count, setCount] = useState(0);
  
  // 每次渲染都创建新对象
  const config = { theme: 'dark', count };
  
  // 使用 useMemo 稳定引用
  const stableConfig = useMemo(
    () => ({ theme: 'dark', count }),
    [count]
  );
  
  useEffect(() => {
    // 如果使用 config，会导致无限循环
    // 因为每次渲染 config 都是新对象
    console.log('config 变化了');
  }, [stableConfig]); // 使用稳定引用
  
  return <button onClick={() => setCount(c => c + 1)}>{count}</button>;
}
```

### 3.6.2 useCallback

```jsx
import { useCallback, useState, memo } from 'react';

// useCallback 用于稳定函数引用
function ParentComponent() {
  const [count, setCount] = useState(0);
  const [otherState, setOtherState] = useState(0);
  
  // 不使用 useCallback：每次渲染都创建新函数
  // 这会导致子组件即使被 memo 包裹也无法优化
  const handleClick = () => {
    setCount(c => c + 1);
  };
  
  // 使用 useCallback：只在 count 变化时创建新函数
  const stableHandleClick = useCallback(() => {
    setCount(c => c + 1);
  }, []); // 使用函数式更新，不需要依赖
  
  return (
    <div>
      <ExpensiveButton onClick={stableHandleClick} />
      <button onClick={() => setOtherState(s => s + 1)}>
        其他状态: {otherState}
      </button>
    </div>
  );
}

// 使用 memo 避免不必要的重渲染
const ExpensiveButton = memo(function ExpensiveButton({ onClick }) {
  console.log('ExpensiveButton 渲染');
  return <button onClick={onClick}>点击</button>;
});
```

## 3.7 useReducer

useReducer 是 useState 的替代方案，适用于复杂的状态逻辑。

### 3.7.1 基本用法

```jsx
import { useReducer } from 'react';

// 定义 reducer 函数
function todoReducer(state, action) {
  switch (action.type) {
    case 'ADD_TODO':
      return [...state, {
        id: Date.now(),
        text: action.payload,
        completed: false,
      }];
    case 'TOGGLE_TODO':
      return state.map(todo =>
        todo.id === action.payload
          ? { ...todo, completed: !todo.completed }
          : todo
      );
    case 'DELETE_TODO':
      return state.filter(todo => todo.id !== action.payload);
    case 'CLEAR_COMPLETED':
      return state.filter(todo => !todo.completed);
    default:
      return state;
  }
}

// 使用 useReducer
function TodoList() {
  const [todos, dispatch] = useReducer(todoReducer, []);
  const [text, setText] = useState('');
  
  const handleSubmit = (e) => {
    e.preventDefault();
    if (text.trim()) {
      dispatch({ type: 'ADD_TODO', payload: text });
      setText('');
    }
  };
  
  return (
    <div>
      <form onSubmit={handleSubmit}>
        <input
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="添加待办事项"
        />
        <button type="submit">添加</button>
      </form>
      
      <ul>
        {todos.map(todo => (
          <li key={todo.id}>
            <span
              style={{
                textDecoration: todo.completed ? 'line-through' : 'none',
              }}
              onClick={() => dispatch({
                type: 'TOGGLE_TODO',
                payload: todo.id,
              })}
            >
              {todo.text}
            </span>
            <button onClick={() => dispatch({
              type: 'DELETE_TODO',
              payload: todo.id,
            })}>
              删除
            </button>
          </li>
        ))}
      </ul>
      
      <button onClick={() => dispatch({ type: 'CLEAR_COMPLETED' })}>
        清除已完成
      </button>
    </div>
  );
}
```

### 3.7.2 延迟初始化

```jsx
// useReducer 的延迟初始化
function init(initialCount) {
  return { count: initialCount, lastUpdated: null };
}

function reducer(state, action) {
  switch (action.type) {
    case 'increment':
      return { ...state, count: state.count + 1, lastUpdated: Date.now() };
    case 'decrement':
      return { ...state, count: state.count - 1, lastUpdated: Date.now() };
    case 'reset':
      return init(action.payload);
    default:
      return state;
  }
}

function Counter({ initialCount = 0 }) {
  // 传入 init 函数作为第三个参数
  const [state, dispatch] = useReducer(reducer, initialCount, init);
  
  return (
    <div>
      <p>计数: {state.count}</p>
      <p>上次更新: {state.lastUpdated?.toLocaleString() || '从未'}</p>
      <button onClick={() => dispatch({ type: 'increment' })}>+</button>
      <button onClick={() => dispatch({ type: 'decrement' })}>-</button>
      <button onClick={() => dispatch({ type: 'reset', payload: initialCount })}>
        重置
      </button>
    </div>
  );
}
```

## 3.8 useLayoutEffect

useLayoutEffect 在浏览器绘制之前同步执行。

```jsx
import { useLayoutEffect, useRef, useState } from 'react';

// useLayoutEffect vs useEffect
function LayoutEffectDemo() {
  const ref = useRef(null);
  const [height, setHeight] = useState(0);
  
  // useEffect：在浏览器绘制后执行
  useEffect(() => {
    console.log('useEffect: 绘制后执行');
    // 用户会看到闪烁
  });
  
  // useLayoutEffect：在浏览器绘制前同步执行
  useLayoutEffect(() => {
    console.log('useLayoutEffect: 绘制前执行');
    // 测量 DOM 尺寸
    if (ref.current) {
      const newHeight = ref.current.getBoundingClientRect().height;
      if (newHeight !== height) {
        setHeight(newHeight);
      }
    }
  }, [height]);
  
  return (
    <div ref={ref}>
      <p>元素高度: {height}px</p>
      <Tooltip targetRef={ref} />
    </div>
  );
}

// 实际应用：Tooltip 定位
function Tooltip({ targetRef }) {
  const [position, setPosition] = useState({ top: 0, left: 0 });
  
  useLayoutEffect(() => {
    if (targetRef.current) {
      const rect = targetRef.current.getBoundingClientRect();
      // 在浏览器绘制前计算位置，避免位置跳变
      setPosition({
        top: rect.bottom + 8,
        left: rect.left + rect.width / 2,
      });
    }
  }, [targetRef]);
  
  return (
    <div style={{
      position: 'fixed',
      top: position.top,
      left: position.left,
      transform: 'translateX(-50%)',
      background: '#333',
      color: '#fff',
      padding: '4px 8px',
      borderRadius: '4px',
    }}>
      Tooltip 内容
    </div>
  );
}
```

## 3.9 useId

useId 用于生成稳定的唯一 ID，主要用于 SSR 水合。

```jsx
import { useId } from 'react';

// useId 生成唯一 ID
function FormFields() {
  // useId 生成稳定的唯一 ID
  // 在服务端和客户端生成相同的 ID，避免水合不匹配
  const id = useId();
  
  return (
    <div>
      <label htmlFor={`${id}-email`}>邮箱</label>
      <input id={`${id}-email`} type="email" />
      
      <label htmlFor={`${id}-password`}>密码</label>
      <input id={`${id}-password`} type="password" />
    </div>
  );
}

// 在列表中
function Checklist({ items }) {
  const id = useId();
  
  return (
    <ul>
      {items.map((item, index) => (
        <li key={item.id}>
          <input id={`${id}-${index}`} type="checkbox" />
          <label htmlFor={`${id}-${index}`}>{item.label}</label>
        </li>
      ))}
    </ul>
  );
}
```

## 3.10 useDeferredValue

useDeferredValue 允许延迟更新 UI 的非紧急部分。

```jsx
import { useState, useDeferredValue, memo } from 'react';

// useDeferredValue 用于延迟非紧急更新
function SearchPage() {
  const [query, setQuery] = useState('');
  const deferredQuery = useDeferredValue(query);
  const isStale = query !== deferredQuery;
  
  return (
    <div>
      <input
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="搜索..."
      />
      
      <div style={{ opacity: isStale ? 0.5 : 1 }}>
        <SearchResults query={deferredQuery} />
      </div>
      
      {isStale && <span>更新中...</span>}
    </div>
  );
}

// 慢渲染的搜索结果组件
const SearchResults = memo(function SearchResults({ query }) {
  // 模拟慢渲染
  const startTime = performance.now();
  while (performance.now() - startTime < 20) {
    // 每个结果渲染耗时 20ms
  }
  
  const items = Array.from(
    { length: 10000 },
    (_, i) => `结果 ${i}: ${query}`
  );
  
  return (
    <ul>
      {items.map((item, i) => (
        <li key={i}>{item}</li>
      ))}
    </ul>
  );
});
```

## 3.11 useTransition

useTransition 用于将状态更新标记为过渡（非紧急）。

```jsx
import { useState, useTransition } from 'react';

// useTransition 用于区分紧急和非紧急更新
function TabSwitcher() {
  const [tab, setTab] = useState('home');
  const [isPending, startTransition] = useTransition();
  
  function switchTab(nextTab) {
    // 紧急更新：更新按钮状态
    // startTransition 中的更新被视为非紧急
    
    startTransition(() => {
      setTab(nextTab);
    });
  }
  
  return (
    <div>
      <div>
        <button onClick={() => switchTab('home')}>首页</button>
        <button onClick={() => switchTab('about')}>关于</button>
        <button onClick={() => switchTab('settings')}>设置</button>
      </div>
      
      {isPending && <div>切换中...</div>}
      
      <TabContent tab={tab} />
    </div>
  );
}

// 慢渲染的 Tab 内容
function TabContent({ tab }) {
  switch (tab) {
    case 'home':
      return <SlowComponent />;
    case 'about':
      return <div>关于我们</div>;
    case 'settings':
      return <div>设置页面</div>;
    default:
      return null;
  }
}

function SlowComponent() {
  // 模拟大量内容的渲染
  const items = Array.from({ length: 5000 }, (_, i) => i);
  
  return (
    <div>
      <h2>首页内容</h2>
      {items.map(i => (
        <div key={i}>项目 #{i}</div>
      ))}
    </div>
  );
}

// startTransition 与 useTransition 的区别
function TransitionDifference() {
  // useTransition：返回 isPending 状态和 startTransition 函数
  const [isPending, startTransition] = useTransition();
  
  // startTransition：直接使用，不返回 isPending
  // import { startTransition } from 'react';
  
  const handleClick = () => {
    // 在事件处理函数外使用
    startTransition(() => {
      // 非紧急更新
    });
  };
  
  return <div>{isPending ? '加载中...' : '就绪'}</div>;
}
```

## 3.12 本章小结

Hooks 是 React 函数组件的核心机制，它们通过在 Fiber 节点上维护链表来管理状态和副作用。

**关键要点回顾：**

1. **Hooks 解决了三个核心问题**：包装器地狱、生命周期逻辑分散、状态逻辑难以复用
2. **两条核心规则**：只在顶层调用 Hooks，只在函数组件和自定义 Hooks 中调用
3. **useState** 通过更新队列和批处理机制管理状态
4. **useEffect** 在浏览器绘制后执行，支持清理函数防止内存泄漏
5. **useRef** 提供跨渲染的持久化引用，不触发重渲染
6. **useMemo/useCallback** 用于性能优化，通过引用稳定性减少不必要的子组件渲染
7. **useReducer** 适用于复杂状态逻辑，支持 dispatch 模式
8. **useLayoutEffect** 在浏览器绘制前同步执行，适合 DOM 测量
9. **useDeferredValue/useTransition** 支持并发模式下的非紧急更新

掌握这些 Hooks 的原理是深入理解 React 运行机制的关键，也为下一章的自定义 Hooks 和高级模式奠定基础。
