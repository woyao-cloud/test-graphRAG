# 第2章 React Fiber 架构深度解析

React Fiber 是 React 16 引入的全新协调引擎（reconciliation engine），它从根本上改变了 React 的渲染方式。本章将深入探讨 Fiber 架构的设计理念、数据结构、工作流程以及并发模式。

## 2.1 协调的演进：从 Stack Reconciler 到 Fiber

### 2.1.1 Stack Reconciler 的局限性

在 React 16 之前，React 使用 Stack Reconciler 进行协调。它的工作方式是递归遍历虚拟 DOM 树，同步执行整个协调过程：

```jsx
// Stack Reconciler 的递归工作方式（简化）
function reconcile(element, container) {
  const instance = createInstance(element);
  container.appendChild(instance);
  
  element.children.forEach(child => {
    reconcile(child, instance); // 递归调用
  });
}
```

**Stack Reconciler 的核心问题：**

1. **同步阻塞**：一旦开始协调，就会持续执行直到整个树遍历完毕，无法中断。
2. **无法处理优先级**：所有更新同等重要，用户输入和高优先级动画必须等待低优先级更新完成。
3. **帧丢失**：如果协调时间超过 16ms（60fps 的一帧），浏览器无法响应用户输入或渲染新帧，导致页面卡顿。

```jsx
// 问题演示：大型列表更新阻塞用户交互
function HeavyList() {
  const [items, setItems] = useState(Array(50000).fill(0));
  
  // 点击按钮触发大规模更新
  const handleClick = () => {
    setItems(prev => prev.map((_, i) => i));
    // 这次更新会阻塞主线程，用户在此期间无法与页面交互
  };
  
  return (
    <div>
      <button onClick={handleClick}>更新</button>
      {items.map((item, i) => <div key={i}>{item}</div>)}
    </div>
  );
}
```

### 2.1.2 Fiber 架构的设计目标

Fiber 架构的设计目标可以概括为以下几点：

1. **可中断的渲染**：将渲染工作分解为小单元，允许在帧之间中断和恢复。
2. **优先级调度**：为不同类型的更新分配优先级，确保高优先级更新（如用户输入）优先处理。
3. **增量渲染**：将渲染工作分散到多个帧中，避免长时间阻塞主线程。
4. **错误边界**：支持 React 错误边界特性（componentDidCatch）。
5. **并发模式**：为 Concurrent Mode 奠定基础。

```jsx
// Fiber 架构的核心思想：将工作分解为可中断的单元
const workUnit = {
  type: 'fiber',
  stateNode: domNode,
  return: parentFiber,
  child: firstChildFiber,
  sibling: nextSiblingFiber,
  // 其他属性...
};

// 调度器可以决定何时执行每个工作单元
function workLoop(deadline) {
  let shouldYield = false;
  while (nextUnitOfWork && !shouldYield) {
    nextUnitOfWork = performUnitOfWork(nextUnitOfWork);
    shouldYield = deadline.timeRemaining() < 1; // 检查剩余时间
  }
  requestIdleCallback(workLoop);
}
```

## 2.2 Fiber 节点结构

Fiber 节点是 Fiber 架构的核心数据结构。每个 Fiber 节点对应一个 React 元素，它们通过指针连接形成 Fiber 树。

### 2.2.1 核心属性

```jsx
// Fiber 节点结构（简化版本）
function FiberNode(tag, pendingProps, key, mode) {
  // 实例相关
  this.tag = tag;           // Fiber 类型：FunctionComponent, ClassComponent, HostComponent 等
  this.key = key;           // React key 属性
  this.elementType = null;  // 元素类型
  this.type = null;         // 函数组件、类组件或 DOM 标签名
  this.stateNode = null;    // 对应的 DOM 节点或组件实例
  
  // Fiber 树结构
  this.return = null;       // 父 Fiber（处理完当前节点后返回到父节点）
  this.child = null;        // 第一个子 Fiber
  this.sibling = null;      // 下一个兄弟 Fiber
  this.index = 0;           // 在兄弟节点中的位置
  
  // 属性与状态
  this.pendingProps = pendingProps;  // 新的 props
  this.memoizedProps = null;         // 上一次渲染的 props
  this.memoizedState = null;         // 上一次渲染的 state（hooks 链表挂载在此）
  
  // 副作用
  this.effectTag = NoEffect;        // 副作用类型：Placement, Update, Deletion 等
  this.nextEffect = null;           // 副作用链表中的下一个节点
  this.firstEffect = null;          // 子树中第一个有副作用的节点
  this.lastEffect = null;           // 子树中最后一个有副作用的节点
  
  // 双缓冲
  this.alternate = null;            // 指向另一个树的对应节点
  
  // 优先级
  this.lanes = NoLanes;             // 当前 Fiber 的更新优先级
  this.childLanes = NoLanes;        // 子树的更新优先级
  
  // 并发模式
  this.mode = mode;                 // ConcurrentMode, StrictMode 等
}
```

### 2.2.2 Fiber 标签类型

```jsx
// React 源码中的 Fiber 标签类型
export const FunctionComponent = 0;       // 函数组件
export const ClassComponent = 1;          // 类组件
export const IndeterminateComponent = 2;  // 未知类型（初始阶段）
export const HostRoot = 3;               // 根节点（ReactDOM.createRoot）
export const HostPortal = 4;             // Portal
export const HostComponent = 5;          // DOM 元素（div, span 等）
export const HostText = 6;               // 文本节点
export const Fragment = 7;               // Fragment
export const Mode = 8;                   // Mode 组件（ConcurrentMode）
export const ContextConsumer = 9;        // Context.Consumer
export const ContextProvider = 10;       // Context.Provider
export const ForwardRef = 11;            // forwardRef
export const Profiler = 12;              // Profiler
export const SuspenseComponent = 13;     // Suspense
export const MemoComponent = 14;         // React.memo
export const LazyComponent = 16;         // React.lazy
```

### 2.2.3 副作用标签

```jsx
// 常见的副作用标签
export const NoEffect = 0b0000000000000000000000000;       // 无副作用
export const Placement = 0b0000000000000000000000010;      // 插入 DOM
export const Update = 0b0000000000000000000000100;         // 更新 DOM
export const Deletion = 0b0000000000000000000001000;       // 删除 DOM
export const Ref = 0b0000000000000000010000000;            // ref 属性
export const Snapshot = 0b0000000000000000100000000;       // getSnapshotBeforeUpdate
export const Passive = 0b0000000000000001000000000;        // useEffect
export const Callback = 0b0000000000000010000000000;       // 回调
export const Visibility = 0b0000000000001000000000000;     // Suspense 可见性
```

### 2.2.4 Fiber 树的连接方式

Fiber 树使用三个指针连接节点，形成单向链表结构：

```jsx
// Fiber 树的三种指针关系
// return → 指向父节点
// child  → 指向第一个子节点
// sibling → 指向下一个兄弟节点

// 示例：以下 JSX 结构
<div>
  <h1>标题</h1>
  <p>段落</p>
  <span>文本</span>
</div>

// 对应的 Fiber 树结构
// div (fiber)
//  ├── return: null (root)
//  ├── child: h1 fiber
//  └── sibling: null
//
// h1 (fiber)
//  ├── return: div fiber
//  ├── child: text fiber ("标题")
//  └── sibling: p fiber
//
// p (fiber)
//  ├── return: div fiber
//  ├── child: text fiber ("段落")
//  └── sibling: span fiber
//
// span (fiber)
//  ├── return: div fiber
//  ├── child: text fiber ("文本")
//  └── sibling: null
```

## 2.3 工作循环（Work Loop）

工作循环是 Fiber 架构的核心调度机制，负责协调和分配渲染工作。

### 2.3.1 工作循环的基本结构

```jsx
// React 源码中的工作循环（简化）
let workInProgress = null;  // 当前正在处理的 Fiber 节点
let rootFiber = null;       // Fiber 树的根节点

// 并发模式的工作循环
function workLoopConcurrent() {
  while (workInProgress !== null && !shouldYield()) {
    performUnitOfWork(workInProgress);
  }
}

// 同步模式的工作循环
function workLoopSync() {
  while (workInProgress !== null) {
    performUnitOfWork(workInProgress);
  }
}

// 判断是否应该让出控制权
function shouldYield() {
  // 检查当前帧的剩余时间
  return getCurrentTime() >= deadline;
}
```

### 2.3.2 performUnitOfWork

`performUnitOfWork` 是工作循环的核心函数，它执行当前 Fiber 节点的工作，然后返回下一个要处理的节点：

```jsx
// performUnitOfWork 的工作流程（简化）
function performUnitOfWork(unitOfWork) {
  // 获取当前 Fiber 节点的 alternate（即 current 树中的对应节点）
  const current = unitOfWork.alternate;
  
  // 开始阶段：处理当前节点并返回子节点
  let next = beginWork(current, unitOfWork, renderLanes);
  
  // 如果 beginWork 返回了新的工作，说明有子节点需要处理
  if (next !== null) {
    workInProgress = next;
    return; // 继续处理子节点
  }
  
  // 如果没有子节点，进入完成阶段
  if (unitOfWork.sibling === null) {
    // 没有兄弟节点，完成当前节点并回到父节点
    completeUnitOfWork(unitOfWork);
  } else {
    // 有兄弟节点，处理兄弟节点
    workInProgress = unitOfWork.sibling;
  }
}
```

### 2.3.3 beginWork

`beginWork` 负责处理当前 Fiber 节点，根据组件类型执行不同的逻辑：

```jsx
// beginWork 的工作流程（简化）
function beginWork(current, workInProgress, renderLanes) {
  // 如果 current 为 null，说明是首次渲染（mount）
  // 如果 current 不为 null，说明是更新（update）
  
  // 检查是否需要更新
  if (current !== null) {
    // 旧的 props 和新的 props 相同
    const oldProps = current.memoizedProps;
    const newProps = workInProgress.pendingProps;
    
    if (oldProps !== newProps || hasContextChanged()) {
      // props 发生变化，需要更新
      didReceiveUpdate = true;
    } else if (!includesSomeLane(renderLanes, workInProgress.lanes)) {
      // 优先级不足，跳过此节点及其子树
      return bailoutOnAlreadyFinishedWork(current, workInProgress, renderLanes);
    }
  }
  
  // 根据 Fiber 类型执行不同的处理
  switch (workInProgress.tag) {
    case FunctionComponent: {
      const Component = workInProgress.type;
      const props = workInProgress.pendingProps;
      // 调用函数组件，获取子节点
      const children = Component(props);
      // 调和子节点
      return reconcileChildren(current, workInProgress, children);
    }
    case ClassComponent: {
      const Component = workInProgress.type;
      const instance = workInProgress.stateNode;
      // 调用 render 方法
      const children = instance.render();
      return reconcileChildren(current, workInProgress, children);
    }
    case HostComponent: {
      // 处理 DOM 元素
      const type = workInProgress.type;
      const props = workInProgress.pendingProps;
      // 创建或更新 DOM 节点
      if (current === null) {
        // 首次渲染，创建 DOM 节点
        workInProgress.stateNode = createInstance(type, props);
      }
      // 调和子节点
      return reconcileChildren(current, workInProgress, props.children);
    }
    case HostText: {
      // 文本节点没有子节点
      return null;
    }
    // ... 其他类型
  }
}
```

### 2.3.4 completeWork

`completeWork` 在子节点处理完成后执行，负责完成当前节点的收尾工作：

```jsx
// completeWork 的工作流程（简化）
function completeWork(current, workInProgress, renderLanes) {
  const newProps = workInProgress.pendingProps;
  
  switch (workInProgress.tag) {
    case HostComponent: {
      const type = workInProgress.type;
      const instance = workInProgress.stateNode;
      
      if (current !== null && workInProgress.stateNode !== null) {
        // 更新现有 DOM 元素
        updateHostComponent(current, workInProgress, type, newProps);
      } else {
        // 创建新 DOM 元素
        const instance = createInstance(type, newProps);
        // 将 DOM 元素添加到父节点
        appendAllChildren(instance, workInProgress);
        workInProgress.stateNode = instance;
      }
      
      // 收集副作用
      if (workInProgress.effectTag !== NoEffect) {
        if (returnFiber.lastEffect !== null) {
          returnFiber.lastEffect.nextEffect = workInProgress;
          returnFiber.lastEffect = workInProgress;
        } else {
          returnFiber.firstEffect = workInProgress;
          returnFiber.lastEffect = workInProgress;
        }
      }
      break;
    }
    // ... 其他类型
  }
  
  // 返回兄弟节点或父节点的兄弟节点
  if (workInProgress.sibling !== null) {
    return workInProgress.sibling;
  }
  
  // 回到父节点
  let returnFiber = workInProgress.return;
  while (returnFiber !== null) {
    if (returnFiber.sibling !== null) {
      return returnFiber.sibling;
    }
    returnFiber = returnFiber.return;
  }
  
  return null;
}
```

### 2.3.5 completeUnitOfWork

`completeUnitOfWork` 负责完成整个子树的处理，并构建副作用链表：

```jsx
// completeUnitOfWork 的流程（简化）
function completeUnitOfWork(unitOfWork) {
  let completedWork = unitOfWork;
  
  do {
    const current = completedWork.alternate;
    const returnFiber = completedWork.return;
    
    // 完成当前节点
    const next = completeWork(current, completedWork, renderLanes);
    
    if (next !== null) {
      // 有新的工作，继续处理
      workInProgress = next;
      return;
    }
    
    // 收集副作用到父节点
    if (returnFiber !== null) {
      // 将当前节点的副作用链表合并到父节点
      if (returnFiber.firstEffect === null) {
        returnFiber.firstEffect = completedWork.firstEffect;
      }
      if (completedWork.lastEffect !== null) {
        if (returnFiber.lastEffect !== null) {
          returnFiber.lastEffect.nextEffect = completedWork.firstEffect;
        }
        returnFiber.lastEffect = completedWork.lastEffect;
      }
      
      // 将当前节点本身加入父节点的副作用链表
      const effectTag = completedWork.effectTag;
      if (effectTag !== NoEffect) {
        if (returnFiber.lastEffect !== null) {
          returnFiber.lastEffect.nextEffect = completedWork;
        } else {
          returnFiber.firstEffect = completedWork;
        }
        returnFiber.lastEffect = completedWork;
      }
    }
    
    // 移动到兄弟节点或返回父节点
    completedWork = returnFiber;
    workInProgress = completedWork;
    
  } while (completedWork !== null);
}
```

## 2.4 优先级系统

Fiber 架构引入了基于优先级的调度系统，确保不同类型的更新得到适当的处理。

### 2.4.1 优先级级别

```jsx
// React 中的优先级级别
export const ImmediatePriority = 1;  // 最高优先级：同步执行，如用户输入
export const UserBlockingPriority = 2;  // 用户阻塞：点击、输入等
export const NormalPriority = 3;     // 正常优先级：普通数据更新
export const LowPriority = 4;        // 低优先级：预加载等
export const IdlePriority = 5;       // 空闲优先级：非必要更新

// 每个优先级对应不同的超时时间
const maxSigned31BitInt = 1073741823;

// ImmediatePriority 是同步的，超时时间为 -1
// UserBlockingPriority 超时时间为 250ms
// NormalPriority 超时时间为 5000ms
// LowPriority 超时时间为 10000ms
// IdlePriority 没有超时时间（永不超时）
```

### 2.4.2 Scheduler 时间切片

React 使用 `requestHostCallback` 和 `requestHostTimeout` 实现时间切片：

```jsx
// Scheduler 的时间切片实现（简化）
let deadline = 0;
const yieldInterval = 5; // 每帧最多执行 5ms

// 调度回调
function scheduleCallback(priorityLevel, callback) {
  // 根据优先级计算超时时间
  const currentTime = getCurrentTime();
  
  let timeout;
  switch (priorityLevel) {
    case ImmediatePriority:
      timeout = -1;
      break;
    case UserBlockingPriority:
      timeout = 250;
      break;
    case NormalPriority:
      timeout = 5000;
      break;
    case LowPriority:
      timeout = 10000;
      break;
    case IdlePriority:
      timeout = maxSigned31BitInt;
      break;
  }
  
  // 创建任务
  const task = {
    callback,
    priorityLevel,
    startTime: currentTime,
    expirationTime: currentTime + timeout,
    sortIndex: currentTime + timeout,
    previous: null,
    next: null,
  };
  
  // 将任务加入队列
  push(timerQueue, task);
  
  // 请求调度
  requestHostCallback(flushWork);
  
  return task;
}

// 实际执行的回调
function flushWork(hasTimeRemaining, initialTime) {
  // 设置截止时间
  deadline = initialTime + yieldInterval;
  
  try {
    return workLoop(hasTimeRemaining, initialTime);
  } finally {
    // 清理
  }
}

// 工作循环（调度器级别）
function workLoop(hasTimeRemaining, initialTime) {
  let currentTask = peek(taskQueue);
  
  while (currentTask !== null) {
    // 检查是否超时
    if (currentTask.expirationTime > initialTime && !hasTimeRemaining) {
      // 当前任务没有超时，且没有剩余时间，让出控制权
      break;
    }
    
    // 执行任务
    const callback = currentTask.callback;
    if (typeof callback === 'function') {
      currentTask.callback = null;
      const didUserCallbackTimeout = currentTask.expirationTime <= initialTime;
      const continuationCallback = callback(didUserCallbackTimeout);
      
      if (typeof continuationCallback === 'function') {
        // 任务未完成，继续执行
        currentTask.callback = continuationCallback;
        return true;
      }
    }
    
    // 移除已完成的任务
    pop(taskQueue);
    currentTask = peek(taskQueue);
  }
  
  return false; // 所有任务已完成
}
```

### 2.4.3 优先级提升与饥饿预防

React 的调度系统会防止低优先级任务被无限期延迟：

```jsx
// 饥饿预防机制
function advanceTimers(currentTime) {
  // 检查 timerQueue 中是否有超时的任务
  let timer = peek(timerQueue);
  
  while (timer !== null) {
    if (timer.callback === null) {
      // 任务已取消
      pop(timerQueue);
    } else if (timer.startTime <= currentTime) {
      // 任务已到开始时间，移入 taskQueue
      pop(timerQueue);
      timer.sortIndex = timer.expirationTime;
      push(taskQueue, timer);
    } else {
      return;
    }
    timer = peek(timerQueue);
  }
}

// 工作循环中的优先级提升
function workLoop(hasTimeRemaining, initialTime) {
  advanceTimers(currentTime); // 将超时任务提升到 taskQueue
  
  let currentTask = peek(taskQueue);
  
  while (currentTask !== null) {
    // 即使时间片已用完，超时的任务也要执行
    if (currentTask.expirationTime <= currentTime) {
      // 任务已超时，立即执行（优先级提升）
      const callback = currentTask.callback;
      currentTask.callback = null;
      callback(true); // true = didUserCallbackTimeout
    } else if (!hasTimeRemaining) {
      // 时间片用完，让出控制权
      break;
    } else {
      // 正常执行
      const callback = currentTask.callback;
      currentTask.callback = null;
      callback(false);
    }
    
    pop(taskQueue);
    currentTask = peek(taskQueue);
  }
}
```

## 2.5 双 Fiber 树（Dual Fiber Tree）

React Fiber 使用双缓冲技术维护两棵 Fiber 树：current 树和 workInProgress 树。

### 2.5.1 双缓冲原理

```jsx
// 双 Fiber 树结构
// current 树：当前已经提交到 DOM 的 Fiber 树
// workInProgress 树：正在构建的 Fiber 树（未提交）

// 初始化时
const fiberRoot = {
  current: rootFiber,  // 指向 current 树的根
};

// 更新时
function prepareFreshStack(root, renderLanes) {
  root.finishedWork = null;
  root.finishedLanes = NoLanes;
  
  // 将 workInProgress 树重置为 current 树的克隆
  workInProgress = createWorkInProgress(root.current, null);
  
  // 开始构建新的 workInProgress 树
  workInProgressRoot = root;
  workInProgressRootRenderLanes = renderLanes;
}
```

### 2.5.2 alternate 指针

每个 Fiber 节点都有一个 `alternate` 属性，指向另一棵树中的对应节点：

```jsx
// alternate 指针的工作原理
function createWorkInProgress(current, pendingProps) {
  let workInProgress = current.alternate;
  
  if (workInProgress === null) {
    // 首次创建 workInProgress 节点
    workInProgress = new FiberNode(current.tag, pendingProps, current.key, current.mode);
    workInProgress.elementType = current.elementType;
    workInProgress.type = current.type;
    workInProgress.stateNode = current.stateNode;
    
    // 建立双向连接
    workInProgress.alternate = current;
    current.alternate = workInProgress;
  } else {
    // 复用已有的 workInProgress 节点
    workInProgress.pendingProps = pendingProps;
    
    // 清除副作用
    workInProgress.effectTag = NoEffect;
    workInProgress.nextEffect = null;
    workInProgress.firstEffect = null;
    workInProgress.lastEffect = null;
  }
  
  // 复制属性
  workInProgress.childLanes = current.childLanes;
  workInProgress.lanes = current.lanes;
  workInProgress.child = current.child;
  workInProgress.memoizedProps = current.memoizedProps;
  workInProgress.memoizedState = current.memoizedState;
  
  return workInProgress;
}
```

### 2.5.3 提交阶段

当 workInProgress 树构建完成后，React 会将其提交为新的 current 树：

```jsx
// 提交阶段
function commitRoot(root) {
  const finishedWork = root.finishedWork;
  
  // 在提交前执行生命周期
  const previousActiveInstance = currentlyActiveInstance;
  
  // 1. 执行 getSnapshotBeforeUpdate
  commitBeforeMutationEffects(finishedWork);
  
  // 2. 执行 DOM 更新
  commitMutationEffects(finishedWork);
  
  // 3. 将 workInProgress 树切换为 current 树
  root.current = finishedWork;
  
  // 4. 执行 useEffect 清理和回调
  commitLayoutEffects(finishedWork);
}

// 提交后，workInProgress 树成为新的 current 树
// 旧的 current 树成为下一次渲染的 workInProgress 树
// 这就是双缓冲的交换机制
```

## 2.6 副作用链表（Effect List）

副作用链表是 Fiber 架构中收集和处理副作用的机制。

### 2.6.1 副作用的收集

```jsx
// 副作用链表的构建过程
// 在 completeUnitOfWork 阶段，将每个有副作用的 Fiber 节点加入链表

// 副作用链表的结构
// root.firstEffect → fiber1 → fiber2 → fiber3 → ... → root.lastEffect
// 其中 root.firstEffect 指向第一个有副作用的节点
// root.lastEffect 指向最后一个有副作用的节点
// 每个节点的 nextEffect 指向下一个有副作用的节点

// 副作用链表的收集示例
function completeUnitOfWork(unitOfWork) {
  let completedWork = unitOfWork;
  
  do {
    const returnFiber = completedWork.return;
    
    // 将子树的副作用链表合并到父节点
    if (returnFiber !== null) {
      if (returnFiber.firstEffect === null) {
        // 父节点还没有副作用，直接接收子树的副作用链表
        returnFiber.firstEffect = completedWork.firstEffect;
      }
      if (completedWork.lastEffect !== null) {
        // 父节点已有副作用，将子树的副作用链表追加到末尾
        if (returnFiber.lastEffect !== null) {
          returnFiber.lastEffect.nextEffect = completedWork.firstEffect;
        }
        returnFiber.lastEffect = completedWork.lastEffect;
      }
      
      // 将当前节点本身加入副作用链表
      const effectTag = completedWork.effectTag;
      if (effectTag !== NoEffect) {
        if (returnFiber.lastEffect !== null) {
          returnFiber.lastEffect.nextEffect = completedWork;
        } else {
          returnFiber.firstEffect = completedWork;
        }
        returnFiber.lastEffect = completedWork;
      }
    }
    
    completedWork = returnFiber;
  } while (completedWork !== null);
}
```

### 2.6.2 提交阶段处理副作用

```jsx
// 提交阶段对副作用链表的处理
function commitBeforeMutationEffects(firstEffect) {
  let nextEffect = firstEffect;
  
  while (nextEffect !== null) {
    const effectTag = nextEffect.effectTag;
    
    // 执行 getSnapshotBeforeUpdate
    if (effectTag & Snapshot) {
      const current = nextEffect.alternate;
      commitBeforeMutationEffectOnFiber(current, nextEffect);
    }
    
    nextEffect = nextEffect.nextEffect;
  }
}

function commitMutationEffects(firstEffect) {
  let nextEffect = firstEffect;
  
  while (nextEffect !== null) {
    const effectTag = nextEffect.effectTag;
    
    // 根据 effectTag 执行不同的 DOM 操作
    if (effectTag & Placement) {
      // 插入新节点
      commitPlacement(nextEffect);
    } else if (effectTag & Update) {
      // 更新节点
      commitWork(nextEffect);
    } else if (effectTag & Deletion) {
      // 删除节点
      commitDeletion(nextEffect);
    } else if (effectTag & Ref) {
      // 更新 ref
      commitAttachRef(nextEffect);
    }
    
    nextEffect = nextEffect.nextEffect;
  }
}

function commitLayoutEffects(firstEffect) {
  let nextEffect = firstEffect;
  
  while (nextEffect !== null) {
    const effectTag = nextEffect.effectTag;
    
    // 执行 useEffect 的清理函数和回调
    if (effectTag & Passive) {
      // 调度 useEffect 的执行（在提交后异步执行）
      schedulePassiveEffects(nextEffect);
    }
    
    // 执行 componentDidMount / componentDidUpdate
    if (effectTag & Update) {
      const current = nextEffect.alternate;
      commitLifeCycles(current, nextEffect);
    }
    
    nextEffect = nextEffect.nextEffect;
  }
}
```

## 2.7 Lane 模型

React 18 引入了 Lane 模型，使用 32 位位掩码来表示优先级，提供了更细粒度的优先级控制。

### 2.7.1 Lane 的定义

```jsx
// Lane 模型使用位掩码（bitmask）表示优先级
// 每个 lane 是 2 的幂，代表一个独立的优先级级别

// 同步 lane
export const SyncLane = 0b0000000000000000000000000000001;

// 输入连续性 lane（用户输入）
export const InputContinuousLane = 0b0000000000000000000000000000100;

// 默认 lane
export const DefaultLane = 0b0000000000000000000000000010000;

// 过渡 lane（Transition）
export const TransitionLane1 = 0b0000000000000000000000010000000;
export const TransitionLane2 = 0b0000000000000000000000100000000;
export const TransitionLane3 = 0b0000000000000000000001000000000;
export const TransitionLane4 = 0b0000000000000000000010000000000;

// 空闲 lane
export const IdleLane = 0b0000000000000000001000000000000;

// 不包含任何 lane
export const NoLane = 0b0000000000000000000000000000000;

// 包含所有 lane
export const NoLanes = 0b0000000000000000000000000000000;
export const TotalLanes = 31;
```

### 2.7.2 Lane 的操作

```jsx
// Lane 的位运算操作

// 合并 lane
function mergeLanes(a, b) {
  return a | b;
}

// 检查是否包含某个 lane
function includesSomeLane(set, subset) {
  return (set & subset) !== NoLanes;
}

// 检查是否包含所有 lane
function includesAllLanes(set, subset) {
  return (set & subset) === subset;
}

// 移除 lane
function removeLanes(set, subset) {
  return set & ~subset;
}

// 获取最高优先级的 lane
function getHighestPriorityLane(lanes) {
  return lanes & -lanes; // 利用补码特性获取最低位的 1
}

// 示例
const lanes = SyncLane | DefaultLane | IdleLane;
// lanes = 0b0000000000000000001000000010001

getHighestPriorityLane(lanes); // SyncLane (最高优先级)

// 找出所有未完成的 lane
function getNextLanes(root, wipLanes) {
  // 获取所有待处理的 lane
  const pendingLanes = root.pendingLanes;
  
  // 如果有同步 lane，优先处理
  if (includesSomeLane(pendingLanes, SyncLane)) {
    return SyncLane;
  }
  
  // 检查是否有非空闲的 lane
  const nonIdlePendingLanes = pendingLanes & ~NonIdleLanes;
  if (nonIdlePendingLanes !== NoLanes) {
    return nonIdlePendingLanes;
  }
  
  // 只有空闲 lane
  return pendingLanes & IdleLane;
}
```

### 2.7.3 Lane 在 Fiber 树中的应用

```jsx
// 每个 Fiber 节点都有 lanes 和 childLanes 属性
// lanes: 当前节点自身的更新优先级
// childLanes: 子树中所有节点的更新优先级集合

// 在 beginWork 中，通过检查 childLanes 决定是否需要深入子树
function beginWork(current, workInProgress, renderLanes) {
  // 检查当前节点的 lane 是否与渲染 lane 匹配
  const lanes = workInProgress.lanes;
  
  // 检查子树是否有待处理的更新
  const childLanes = workInProgress.childLanes;
  
  if (!includesSomeLane(renderLanes, childLanes)) {
    // 子树没有任何待处理的更新，跳过整个子树
    return bailoutOnAlreadyFinishedWork(current, workInProgress, renderLanes);
  }
  
  // 根据组件类型处理
  // ...
}

// 在调度更新时，将 lane 关联到更新对象
function scheduleUpdateOnFiber(fiber, lane) {
  const root = markUpdateLaneFromFiberToRoot(fiber, lane);
  
  if (root !== null) {
    // 从根节点开始调度
    ensureRootIsScheduled(root);
  }
}

// 从当前 Fiber 向上标记 lane
function markUpdateLaneFromFiberToRoot(sourceFiber, lane) {
  sourceFiber.lanes = mergeLanes(sourceFiber.lanes, lane);
  
  let alternate = sourceFiber.alternate;
  if (alternate !== null) {
    alternate.lanes = mergeLanes(alternate.lanes, lane);
  }
  
  // 向上遍历到根节点，标记 childLanes
  let node = sourceFiber.return;
  while (node !== null) {
    node.childLanes = mergeLanes(node.childLanes, lane);
    
    alternate = node.alternate;
    if (alternate !== null) {
      alternate.childLanes = mergeLanes(alternate.childLanes, lane);
    }
    
    node = node.return;
  }
  
  return sourceFiber.stateNode; // FiberRoot
}
```

## 2.8 并发模式（Concurrent Mode）

并发模式是 React 18 最重要的新特性，它使 React 能够同时准备多个版本的 UI。

### 2.8.1 可中断渲染

```jsx
// 并发模式的核心：可中断渲染
// 在并发模式下，React 可以在渲染过程中被中断

// 渲染过程中的中断与恢复
function renderRootConcurrent(root, lanes) {
  // 开始构建 workInProgress 树
  do {
    try {
      workLoopConcurrent();
      break;
    } catch (thrownValue) {
      // 如果抛出异常（如 Suspense 数据加载）
      // 暂停当前渲染，稍后重试
      if (thrownValue === SuspenseException) {
        // Suspense 数据未准备好，暂停
        suspendRoot(root, lanes);
        return;
      }
      
      // 其他错误，触发错误边界
      throwError(root, thrownValue);
      return;
    }
  } while (true);
  
  // 渲染完成，进入提交阶段
  if (workInProgress !== null) {
    // 渲染被中断（时间片用完）
    // 等待下一次调度
    ensureRootIsScheduled(root);
    return;
  }
  
  // 渲染完成，提交
  commitRoot(root);
}
```

### 2.8.2 Suspense

Suspense 允许组件在等待数据时显示 fallback 内容：

```jsx
// Suspense 在 Fiber 中的实现
function SuspenseComponent(props) {
  // Suspense 组件的工作原理
  // 1. 尝试渲染子组件
  // 2. 如果子组件抛出 Promise（数据未加载完成）
  // 3. 显示 fallback 内容
  // 4. Promise resolve 后重新渲染子组件
  return (
    <Suspense fallback={<Loading />}>
      <DataComponent />
    </Suspense>
  );
}

// 在 Fiber 层面，Suspense 的处理
function updateSuspenseComponent(current, workInProgress, renderLanes) {
  const nextProps = workInProgress.pendingProps;
  const showFallback = false; // 是否显示 fallback
  
  // 检查是否已经有挂起的操作
  const didSuspend = (workInProgress.effectTag & DidCapture) !== NoEffect;
  
  if (didSuspend) {
    // 已经有挂起的操作，显示 fallback
    showFallback = true;
    workInProgress.effectTag &= ~DidCapture; // 清除标记
  }
  
  // 尝试渲染主内容
  const nextPrimaryChildren = nextProps.children;
  const nextFallbackChildren = nextProps.fallback;
  
  if (showFallback) {
    // 渲染 fallback
    const fallbackFragment = mountSuspenseFallbackChildren(
      workInProgress,
      nextPrimaryChildren,
      nextFallbackChildren,
      renderLanes
    );
    return fallbackFragment;
  } else {
    // 渲染主内容
    const primaryFragment = mountSuspensePrimaryChildren(
      workInProgress,
      nextPrimaryChildren,
      renderLanes
    );
    return primaryFragment;
  }
}
```

### 2.8.3 Transitions

Transition 是 React 18 引入的概念，用于区分紧急和非紧急更新：

```jsx
import { useTransition, startTransition } from 'react';

// useTransition Hook
function SearchPage() {
  const [query, setQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [isPending, startTransition] = useTransition();
  
  const handleChange = (e) => {
    // 紧急更新：更新输入框
    setQuery(e.target.value);
    
    // 非紧急更新：搜索结果
    startTransition(() => {
      setSearchResults(filterResults(e.target.value));
    });
  };
  
  return (
    <div>
      <input value={query} onChange={handleChange} />
      {isPending && <Spinner />}
      <SearchResults results={searchResults} />
    </div>
  );
}

// startTransition 在 Fiber 层面的实现
function startTransition(scope, options) {
  const prevTransition = ReactCurrentBatchConfig.transition;
  
  // 创建一个新的 transition
  ReactCurrentBatchConfig.transition = {
    _updatedFibers: new Set(),
  };
  
  try {
    // 执行 scope 中的状态更新
    // 这些更新会被标记为 transition lane
    scope();
  } finally {
    ReactCurrentBatchConfig.transition = prevTransition;
  }
}
```

### 2.8.4 并发模式下的优先级调度

```jsx
// 并发模式下的完整调度流程
function ensureRootIsScheduled(root) {
  // 获取当前需要处理的 lanes
  const nextLanes = getNextLanes(root, root === workInProgressRoot ? workInProgressRootRenderLanes : NoLanes);
  
  if (nextLanes === NoLanes) {
    // 没有待处理的更新
    return;
  }
  
  // 确定调度优先级
  let schedulerPriorityLevel;
  
  if (includesSomeLane(nextLanes, SyncLane)) {
    // 同步 lane → ImmediatePriority
    schedulerPriorityLevel = ImmediatePriority;
  } else if (includesSomeLane(nextLanes, InputContinuousLane)) {
    // 输入连续性 lane → UserBlockingPriority
    schedulerPriorityLevel = UserBlockingPriority;
  } else {
    // 其他 lane → NormalPriority
    schedulerPriorityLevel = NormalPriority;
  }
  
  // 调度新的渲染
  scheduleCallback(schedulerPriorityLevel, () => {
    // 执行渲染
    performConcurrentWorkOnRoot(root);
  });
}
```

## 2.9 代码示例与实践

### 2.9.1 简单的 Fiber 可视化器

```jsx
// Fiber 树可视化器
import React from 'react';
import { render } from 'react-dom';

class FiberTreeViewer extends React.Component {
  componentDidMount() {
    this.visualizeFiberTree();
  }
  
  componentDidUpdate() {
    this.visualizeFiberTree();
  }
  
  visualizeFiberTree() {
    // 通过 React 内部属性访问 Fiber 树
    const fiberRoot = this._reactInternals;
    const treeData = this.buildFiberTree(fiberRoot);
    this.renderTree(treeData);
  }
  
  buildFiberTree(fiber) {
    if (!fiber) return null;
    
    const node = {
      name: fiber.type?.name || fiber.type || 'HostRoot',
      tag: fiber.tag,
      lanes: fiber.lanes,
      childLanes: fiber.childLanes,
      effectTag: fiber.effectTag,
      children: [],
    };
    
    // 遍历子节点
    let child = fiber.child;
    while (child) {
      const childNode = this.buildFiberTree(child);
      if (childNode) {
        node.children.push(childNode);
      }
      child = child.sibling;
    }
    
    return node;
  }
  
  renderTree(node) {
    const container = this.refs.tree;
    container.innerHTML = this.renderTreeNode(node, 0);
  }
  
  renderTreeNode(node, depth) {
    if (!node) return '';
    
    const indent = '  '.repeat(depth);
    const tagNames = {
      0: 'FunctionComponent',
      1: 'ClassComponent',
      3: 'HostRoot',
      5: 'HostComponent',
      7: 'Fragment',
      13: 'SuspenseComponent',
    };
    
    let html = `<div style="margin-left: ${depth * 20}px">`;
    html += `<span style="color: #888">${tagNames[node.tag] || node.tag}</span> `;
    html += `<strong>${node.name}</strong>`;
    html += ` <span style="color: #999; font-size: 12px">lanes: ${node.lanes}</span>`;
    html += '</div>';
    
    for (const child of node.children) {
      html += this.renderTreeNode(child, depth + 1);
    }
    
    return html;
  }
  
  render() {
    return (
      <div>
        <h3>Fiber Tree</h3>
        <div ref="tree" style={{ fontFamily: 'monospace' }} />
      </div>
    );
  }
}

// 使用示例
function App() {
  return (
    <div>
      <header>
        <h1>Fiber 可视化</h1>
      </header>
      <main>
        <p>查看组件背后的 Fiber 树结构</p>
      </main>
    </div>
  );
}
```

### 2.9.2 优先级演示

```jsx
// 优先级演示组件
import React, { useState, useTransition, useDeferredValue } from 'react';

function PriorityDemo() {
  const [urgentCount, setUrgentCount] = useState(0);
  const [transitionCount, setTransitionCount] = useState(0);
  const [isPending, startTransition] = useTransition();
  
  // 使用 useDeferredValue 实现类似效果
  const deferredTransitionCount = useDeferredValue(transitionCount);
  const isStale = transitionCount !== deferredTransitionCount;
  
  const handleUrgentClick = () => {
    // 紧急更新：立即生效
    setUrgentCount(c => c + 1);
  };
  
  const handleTransitionClick = () => {
    // 非紧急更新：可中断，优先级较低
    startTransition(() => {
      setTransitionCount(c => c + 1);
    });
  };
  
  return (
    <div style={{ padding: '20px' }}>
      <h2>优先级演示</h2>
      
      <div style={{ marginBottom: '20px' }}>
        <h3>紧急更新 (ImmediatePriority)</h3>
        <p>计数: {urgentCount}</p>
        <button onClick={handleUrgentClick}>
          增加紧急计数
        </button>
        <p style={{ color: '#666', fontSize: '14px' }}>
          点击后立即更新，不可中断
        </p>
      </div>
      
      <div style={{ marginBottom: '20px' }}>
        <h3>Transition 更新 (NormalPriority)</h3>
        <p>计数: {transitionCount}</p>
        <p>延迟计数: {deferredTransitionCount}</p>
        <button onClick={handleTransitionClick}>
          增加 Transition 计数
        </button>
        {isPending && <span style={{ color: 'orange' }}> 更新中...</span>}
        {isStale && <span style={{ color: 'blue' }}> 延迟值</span>}
        <p style={{ color: '#666', fontSize: '14px' }}>
          可中断，紧急更新优先
        </p>
      </div>
      
      <div style={{ marginTop: '30px' }}>
        <h3>性能对比</h3>
        <SlowList count={deferredTransitionCount} />
      </div>
    </div>
  );
}

// 模拟慢渲染的组件
function SlowList({ count }) {
  const startTime = performance.now();
  
  // 模拟耗时计算
  while (performance.now() - startTime < 3) {
    // 每个项目耗时 3ms
  }
  
  const items = Array.from({ length: count }, (_, i) => i);
  
  return (
    <div>
      <p>列表项目数: {items.length}</p>
      <div style={{ maxHeight: '200px', overflow: 'auto' }}>
        {items.map(i => (
          <div key={i} style={{ padding: '2px' }}>
            项目 #{i}
          </div>
        ))}
      </div>
    </div>
  );
}

export default PriorityDemo;
```

### 2.9.3 并发模式渲染演示

```jsx
// 并发模式渲染演示
import React, { useState, useTransition, Suspense } from 'react';

// 模拟数据获取
function fetchData(id) {
  return new Promise(resolve => {
    setTimeout(() => {
      resolve({ id, name: `用户 ${id}`, email: `user${id}@example.com` });
    }, 2000);
  });
}

// 数据包装器
function createResource(promise) {
  let status = 'pending';
  let result = promise.then(
    resolved => {
      status = 'resolved';
      result = resolved;
    },
    rejected => {
      status = 'rejected';
      result = rejected;
    }
  );
  
  return {
    read() {
      if (status === 'pending') throw result; // 抛出 Promise
      if (status === 'rejected') throw result; // 抛出错误
      return result;
    }
  };
}

// 使用 Suspense 的组件
function UserProfile({ userId }) {
  const [resource, setResource] = useState(null);
  
  // 在首次渲染时开始获取数据
  if (resource === null) {
    const newResource = createResource(fetchData(userId));
    setResource(newResource);
  }
  
  const user = resource.read(); // 可能抛出 Promise
  
  return (
    <div style={{ padding: '10px', border: '1px solid #ccc', borderRadius: '4px' }}>
      <h3>{user.name}</h3>
      <p>Email: {user.email}</p>
    </div>
  );
}

// 使用 Suspense + Transition 的并发模式示例
function ConcurrentSuspenseDemo() {
  const [userId, setUserId] = useState(1);
  const [isPending, startTransition] = useTransition();
  
  const handleNextUser = () => {
    startTransition(() => {
      setUserId(prev => prev + 1);
    });
  };
  
  return (
    <div style={{ padding: '20px' }}>
      <h2>并发模式 + Suspense 演示</h2>
      
      <div style={{ marginBottom: '20px' }}>
        <button onClick={handleNextUser}>
          下一个用户
        </button>
        <button onClick={() => setUserId(prev => Math.max(1, prev - 1))}>
          上一个用户
        </button>
        {isPending && <span style={{ color: 'orange' }}> 切换中...</span>}
      </div>
      
      <Suspense fallback={<div>加载用户数据中...</div>}>
        <UserProfile userId={userId} />
      </Suspense>
    </div>
  );
}

export default ConcurrentSuspenseDemo;
```

### 2.9.4 Fiber 更新流程总结

```jsx
// 完整的 Fiber 更新流程图解

// 阶段 1: 调度阶段 (Schedule)
// 1. 调用 setState / useState setter / dispatch
// 2. 创建 Update 对象，加入 updateQueue
// 3. 从当前 Fiber 向上标记 lane 到根节点
// 4. 调度渲染任务

// 阶段 2: 渲染阶段 (Render) - 可中断
// 1. workLoopConcurrent / workLoopSync
// 2. performUnitOfWork → beginWork → reconcileChildren
// 3. 深度优先遍历 Fiber 树
// 4. completeWork → 收集副作用到 effect list
// 5. 构建 workInProgress 树
// 6. 如果时间片用完，让出控制权，等待下一次调度

// 阶段 3: 提交阶段 (Commit) - 不可中断
// 1. before mutation: getSnapshotBeforeUpdate
// 2. mutation: 执行 DOM 操作（插入、更新、删除）
// 3. layout: 切换 current 树，执行 useEffect 回调

// 关键设计决策
// - 渲染阶段可以中断：因为 workInProgress 树还未提交到 DOM
// - 提交阶段不可中断：因为涉及实际的 DOM 操作
// - 双缓冲确保用户始终看到完整的 UI

// 时间切片的工作原理
// requestAnimationFrame → 执行 React 渲染 5ms → 让出控制权 → 浏览器处理其他任务
// → requestAnimationFrame → 执行 React 渲染 5ms → ...
// 循环直到渲染完成
```

## 2.10 本章小结

React Fiber 架构是 React 16 以来最重要的架构升级。通过引入 Fiber 节点作为基本工作单元，配合双缓冲、优先级调度和 Lane 模型，Fiber 实现了可中断的渲染过程，为 Concurrent Mode 和 React 18 的并发特性奠定了基础。

**关键要点回顾：**

1. **Fiber 节点**是 React 的最小工作单元，通过 return/child/sibling 指针形成 Fiber 树
2. **双缓冲**通过 current 树和 workInProgress 树确保用户始终看到完整的 UI
3. **工作循环**将渲染工作分解为小单元，使用时间切片避免阻塞主线程
4. **优先级系统**通过 Lane 模型实现细粒度的优先级控制
5. **副作用链表**在提交阶段高效地处理 DOM 操作
6. **并发模式**基于 Fiber 架构，实现了可中断渲染、Suspense 和 Transition

理解 Fiber 架构对于深入掌握 React 的运行机制至关重要，也是理解后续章节中 Hooks、性能优化和高级模式的基础。
