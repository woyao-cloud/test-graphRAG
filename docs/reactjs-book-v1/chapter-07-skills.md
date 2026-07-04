# 第7章 TypeScript、测试与工程化

现代 React 开发离不开 TypeScript 的类型安全、测试的质量保障以及完善的工程化工具链。本章将系统介绍这些必备技能，涵盖从类型系统到安全防护的完整工具链。

## 7.1 TypeScript 与 React

TypeScript 与 React 的结合已经成为行业标准。掌握类型系统能让你的组件更健壮、重构更安全。

### 7.1.1 泛型 Props

当组件需要处理多种数据类型时，泛型 Props 是最优雅的解决方案。

```tsx
import { ReactNode } from 'react';

// 泛型列表组件：支持任意数据类型的列表渲染
interface ListProps<T> {
  items: T[];
  renderItem: (item: T, index: number) => ReactNode;
  keyExtractor: (item: T) => string | number;
  emptyText?: string;
  loading?: boolean;
}

function List<T>({
  items,
  renderItem,
  keyExtractor,
  emptyText = '暂无数据',
  loading = false,
}: ListProps<T>) {
  if (loading) {
    return <div className="list-loading">加载中...</div>;
  }

  if (items.length === 0) {
    return <div className="list-empty">{emptyText}</div>;
  }

  return (
    <ul className="list">
      {items.map((item, index) => (
        <li key={keyExtractor(item)} className="list-item">
          {renderItem(item, index)}
        </li>
      ))}
    </ul>
  );
}

// 使用示例：不同类型的数据复用同一个 List 组件
interface User {
  id: number;
  name: string;
  email: string;
}

interface Product {
  sku: string;
  title: string;
  price: number;
}

function UserList() {
  const [users, setUsers] = useState<User[]>([]);

  return (
    <List<User>
      items={users}
      renderItem={(user) => (
        <div>
          <strong>{user.name}</strong>
          <span>{user.email}</span>
        </div>
      )}
      keyExtractor={(user) => user.id}
      emptyText="暂无用户数据"
    />
  );
}

function ProductList() {
  const [products, setProducts] = useState<Product[]>([]);

  return (
    <List<Product>
      items={products}
      renderItem={(product) => (
        <div>
          <strong>{product.title}</strong>
          <span>¥{product.price}</span>
        </div>
      )}
      keyExtractor={(product) => product.sku}
      emptyText="暂无商品数据"
    />
  );
}

// 泛型 Select 组件
interface SelectProps<T extends string | number> {
  value: T;
  options: { label: string; value: T }[];
  onChange: (value: T) => void;
  placeholder?: string;
}

function Select<T extends string | number>({
  value,
  options,
  onChange,
  placeholder = '请选择',
}: SelectProps<T>) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value as T)}
    >
      <option value="" disabled>
        {placeholder}
      </option>
      {options.map((opt) => (
        <option key={String(opt.value)} value={opt.value}>
          {opt.label}
        </option>
      ))}
    </select>
  );
}

// 使用
type SortOrder = 'asc' | 'desc' | 'default';
const [sortOrder, setSortOrder] = useState<SortOrder>('default');

<Select<SortOrder>
  value={sortOrder}
  options={[
    { label: '默认排序', value: 'default' },
    { label: '升序', value: 'asc' },
    { label: '降序', value: 'desc' },
  ]}
  onChange={setSortOrder}
/>
```

### 7.1.2 事件类型详解

React 为每种 DOM 事件提供了对应的合成事件类型。掌握事件类型是编写类型安全的事件处理器的前提。

```tsx
import {
  ChangeEvent,
  MouseEvent,
  FormEvent,
  KeyboardEvent,
  FocusEvent,
  DragEvent,
  ClipboardEvent,
} from 'react';

// ============ 表单输入事件 ============
function SearchForm() {
  const [query, setQuery] = useState('');
  const [category, setCategory] = useState('all');

  // ChangeEvent<T> — T 为具体的 DOM 元素类型
  const handleInputChange = (e: ChangeEvent<HTMLInputElement>) => {
    // e.target 自动推导为 HTMLInputElement
    setQuery(e.target.value);
    // 可以访问 input 特有的属性
    console.log(e.target.selectionStart, e.target.selectionEnd);
  };

  const handleTextareaChange = (e: ChangeEvent<HTMLTextAreaElement>) => {
    console.log(e.target.value);
  };

  const handleSelectChange = (e: ChangeEvent<HTMLSelectElement>) => {
    setCategory(e.target.value);
  };

  const handleCheckboxChange = (e: ChangeEvent<HTMLInputElement>) => {
    // e.target.checked 是 boolean
    console.log(e.target.checked);
  };

  const handleSubmit = (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const formData = new FormData(e.currentTarget);
    const data = Object.fromEntries(formData.entries());
    console.log('提交数据:', data);
  };

  return (
    <form onSubmit={handleSubmit}>
      <input
        name="query"
        value={query}
        onChange={handleInputChange}
        placeholder="搜索..."
      />
      <textarea name="description" onChange={handleTextareaChange} />
      <select value={category} onChange={handleSelectChange}>
        <option value="all">全部</option>
        <option value="tech">技术</option>
        <option value="life">生活</option>
      </select>
      <input type="checkbox" onChange={handleCheckboxChange} />
      <button type="submit">搜索</button>
    </form>
  );
}

// ============ 鼠标事件 ============
function DraggableCard() {
  const handleClick = (e: MouseEvent<HTMLButtonElement>) => {
    // 鼠标坐标
    console.log('点击位置:', e.clientX, e.clientY);
    console.log('相对于元素:', e.nativeEvent.offsetX, e.nativeEvent.offsetY);

    // 修饰键状态
    if (e.ctrlKey) console.log('Ctrl + 点击');
    if (e.shiftKey) console.log('Shift + 点击');
    if (e.altKey) console.log('Alt + 点击');
    if (e.metaKey) console.log('Cmd/Win + 点击');

    // 鼠标按键
    // e.button: 0 = 左键, 1 = 中键, 2 = 右键
    if (e.button === 0) console.log('左键点击');
    if (e.button === 2) console.log('右键点击');

    // 阻止默认行为和冒泡
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDoubleClick = (e: MouseEvent<HTMLDivElement>) => {
    console.log('双击:', e.detail); // e.detail = 2 表示双击
  };

  const handleContextMenu = (e: MouseEvent<HTMLDivElement>) => {
    e.preventDefault(); // 阻止浏览器默认右键菜单
    // 显示自定义右键菜单
    console.log('自定义右键菜单位置:', e.clientX, e.clientY);
  };

  const handleDragStart = (e: DragEvent<HTMLDivElement>) => {
    e.dataTransfer.setData('text/plain', 'dragged-item-id');
    e.dataTransfer.effectAllowed = 'move';
  };

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const itemId = e.dataTransfer.getData('text/plain');
    console.log('拖放项目:', itemId);
  };

  return (
    <div
      onDoubleClick={handleDoubleClick}
      onContextMenu={handleContextMenu}
      draggable
      onDragStart={handleDragStart}
      onDragOver={handleDragOver}
      onDrop={handleDrop}
      style={{ padding: 20, border: '1px solid #ccc' }}
    >
      <button onClick={handleClick}>操作按钮</button>
      <p>拖拽我或双击我</p>
    </div>
  );
}

// ============ 键盘事件 ============
function KeyboardShortcuts() {
  const handleKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    // e.key vs e.code
    // e.key: 用户看到的字符（受键盘布局影响）
    // e.code: 物理按键位置（不受键盘布局影响）

    switch (e.key) {
      case 'Enter':
        console.log('提交');
        break;
      case 'Escape':
        console.log('取消/关闭');
        break;
      case 'ArrowUp':
        console.log('上');
        break;
      case 'ArrowDown':
        console.log('下');
        break;
      case 'Tab':
        e.preventDefault(); // 阻止默认 Tab 行为
        console.log('自定义 Tab 处理');
        break;
    }

    // 组合键
    if (e.ctrlKey && e.key === 's') {
      e.preventDefault();
      console.log('Ctrl+S: 保存');
    }

    if (e.ctrlKey && e.key === 'z') {
      e.preventDefault();
      if (e.shiftKey) {
        console.log('Ctrl+Shift+Z: 重做');
      } else {
        console.log('Ctrl+Z: 撤销');
      }
    }
  };

  const handleKeyUp = (e: KeyboardEvent<HTMLInputElement>) => {
    // keyUp 适用于检测按键释放时间
    console.log(`按键 "${e.key}" 释放`);
  };

  return (
    <input
      onKeyDown={handleKeyDown}
      onKeyUp={handleKeyUp}
      placeholder="试试快捷键: Ctrl+S, Escape, Enter"
      style={{ width: '100%', padding: 8 }}
    />
  );
}

// ============ 焦点事件 ============
function FormField({ label, error }: { label: string; error?: string }) {
  const [isFocused, setIsFocused] = useState(false);

  const handleFocus = (e: FocusEvent<HTMLInputElement>) => {
    setIsFocused(true);
    // e.relatedTarget: 上一个聚焦的元素
    console.log('从', e.relatedTarget, '获得焦点');
  };

  const handleBlur = (e: FocusEvent<HTMLInputElement>) => {
    setIsFocused(false);
    // e.relatedTarget: 下一个获得焦点的元素
    console.log('焦点转移到', e.relatedTarget);
  };

  return (
    <div className={isFocused ? 'field-focused' : ''}>
      <label>{label}</label>
      <input onFocus={handleFocus} onBlur={handleBlur} />
      {error && <span className="field-error">{error}</span>}
    </div>
  );
}

// ============ 剪贴板事件 ============
function ClipboardDemo() {
  const handleCopy = (e: ClipboardEvent<HTMLDivElement>) => {
    e.preventDefault();
    const selection = window.getSelection()?.toString() ?? '';
    e.clipboardData.setData('text/plain', `【转载】${selection}`);
    console.log('已复制（带版权声明）');
  };

  const handlePaste = (e: ClipboardEvent<HTMLTextAreaElement>) => {
    const pastedText = e.clipboardData.getData('text/plain');
    if (pastedText.length > 1000) {
      e.preventDefault();
      alert('粘贴内容过长，限制 1000 字符');
    }
  };

  return (
    <div onCopy={handleCopy}>
      <p>选中这段文字并复制试试</p>
      <textarea onPaste={handlePaste} placeholder="粘贴内容到此处（限 1000 字符）" />
    </div>
  );
}
```

### 7.1.3 ref 类型与 forwardRef

```tsx
import { useRef, forwardRef, useImperativeHandle, RefObject, MutableRefObject, useEffect } from 'react';

// ============ RefObject vs MutableRefObject ============
// RefObject<T>:         .current 只读，用于 DOM ref（通过 ref 属性绑定）
// MutableRefObject<T>:  .current 可写，用于存储可变值（useRef 手动创建）

function RefTypeDemo() {
  // DOM ref → RefObject
  const inputRef = useRef<HTMLInputElement>(null);
  // 类型: RefObject<HTMLInputElement | null>
  // inputRef.current = otherElement; // ❌ 不能重新赋值

  // 可变 ref → MutableRefObject
  const countRef = useRef<number>(0);
  // 类型: MutableRefObject<number>
  countRef.current = 42; // ✅ 可以修改

  // useRef 的初始化值决定类型推断
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  // 类型: MutableRefObject<ReturnType<typeof setInterval> | null>

  return <input ref={inputRef} />;
}

// ============ forwardRef（React 18 风格） ============
interface TextInputProps {
  label: string;
  error?: string;
}

const TextInput = forwardRef<HTMLInputElement, TextInputProps>(
  function TextInput({ label, error, ...rest }, ref) {
    const inputId = useId();

    return (
      <div className="form-field">
        <label htmlFor={inputId}>{label}</label>
        <input
          id={inputId}
          ref={ref}
          aria-invalid={!!error}
          aria-describedby={error ? `${inputId}-error` : undefined}
          {...rest}
        />
        {error && (
          <span id={`${inputId}-error`} className="error" role="alert">
            {error}
          </span>
        )}
      </div>
    );
  }
);

// 父组件使用
function ParentForm() {
  const inputRef = useRef<HTMLInputElement>(null);

  const focusInput = () => {
    inputRef.current?.focus();
    inputRef.current?.select();
  };

  return (
    <div>
      <TextInput ref={inputRef} label="用户名" error="用户名不能为空" />
      <button onClick={focusInput}>聚焦到输入框</button>
    </div>
  );
}

// ============ useImperativeHandle ============
interface ImperativeHandle {
  focus: () => void;
  reset: () => void;
  getValue: () => string;
}

const AdvancedInput = forwardRef<ImperativeHandle, TextInputProps>(
  function AdvancedInput({ label, error, ...rest }, ref) {
    const inputRef = useRef<HTMLInputElement>(null);
    const [value, setValue] = useState('');

    useImperativeHandle(ref, () => ({
      focus: () => inputRef.current?.focus(),
      reset: () => {
        setValue('');
        inputRef.current?.focus();
      },
      getValue: () => value,
    }));

    return (
      <div className="form-field">
        <label>{label}</label>
        <input
          ref={inputRef}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          {...rest}
        />
        {error && <span className="error">{error}</span>}
      </div>
    );
  }
);

// ============ React 19: ref 作为普通 prop ============
// React 19 不再需要 forwardRef，ref 可以直接作为 prop 传递
function React19Input({
  ref,
  label,
  error,
  ...rest
}: TextInputProps & { ref?: React.Ref<HTMLInputElement> }) {
  return (
    <div className="form-field">
      <label>{label}</label>
      <input ref={ref} {...rest} />
      {error && <span className="error">{error}</span>}
    </div>
  );
}
```

### 7.1.4 FC vs JSX.Element vs ReactNode

理解这三种类型的区别是 React + TypeScript 的基本功。

```tsx
import { FC, ReactNode, ReactElement, JSX } from 'react';

// ============ ReactNode — 最宽泛的类型 ============
// ReactNode 是 React 可以渲染的所有内容的联合类型：
// ReactElement | string | number | boolean | null | undefined | ReactNode[]
// 用于 children、条件渲染内容等场景

interface CardProps {
  title: string;
  children: ReactNode; // 可以是任何可渲染内容
  footer?: ReactNode;
}

function Card({ title, children, footer }: CardProps) {
  return (
    <div className="card">
      <h2>{title}</h2>
      <div className="card-body">{children}</div>
      {footer && <div className="card-footer">{footer}</div>}
    </div>
  );
}

// 这些用法都是合法的
function CardUsage() {
  return (
    <>
      <Card title="字符串 children">一段文本</Card>
      <Card title="数字 children">{42}</Card>
      <Card title="元素 children">
        <button>操作</button>
      </Card>
      <Card title="多个 children">
        <p>段落 1</p>
        <p>段落 2</p>
      </Card>
      <Card title="条件渲染">
        {false}
        {null}
        {undefined}
        {true && <span>条件为真时显示</span>}
      </Card>
    </>
  );
}

// ============ JSX.Element / ReactElement — 较窄的类型 ============
// JSX.Element 是 ReactElement 的别名（React 19 中推荐使用 JSX.Element）
// 表示一个已创建的 JSX 元素，不能是 string/number/null 等原始值

interface LayoutProps {
  header: JSX.Element;  // 必须是 JSX 元素
  sidebar: JSX.Element; // 必须是 JSX 元素
  children: ReactNode;   // children 通常保持 ReactNode 以获得灵活性
}

function Layout({ header, sidebar, children }: LayoutProps) {
  return (
    <div className="layout">
      <header>{header}</header>
      <aside>{sidebar}</aside>
      <main>{children}</main>
    </div>
  );
}

// ============ FC / FunctionComponent — 已不推荐 ============
// FC 自动为 props 添加 children?: ReactNode
// React 19 中不再建议显式标注 FC 类型

// ❌ 旧写法（不推荐）
const OldComponent: FC<{ title: string }> = ({ title, children }) => {
  // children 自动可用，即使可能不需要
  return <div>{title}</div>;
};

// ✅ 新写法（推荐）：直接标注 props 类型
interface NewComponentProps {
  title: string;
  children?: ReactNode; // 明确声明是否需要 children
}

function NewComponent({ title, children }: NewComponentProps) {
  return <div>{title}{children}</div>;
}

// ============ 类型选择速查表 ============
// ReactNode    → children prop、条件渲染内容、Portal 内容
// JSX.Element  → 需要明确是 JSX 元素的 prop（如 header、icon）
// 直接标注 props → 组件的 props 参数（不推荐 FC）
// ReactElement<typeof MyComp> → 需要限制子组件类型时
```

### 7.1.5 自定义 Hook 类型

```tsx
import { useState, useCallback, useEffect, useRef } from 'react';

// ============ 泛型状态 Hook ============
// useState<T> — 当初始值无法推断类型时使用
function useStateDemo() {
  // 自动推断
  const [count, setCount] = useState(0);               // number
  const [name, setName] = useState('');                 // string
  const [items, setItems] = useState<string[]>([]);     // string[]

  // 显式标注（联合类型或可能为 null）
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  const [user, setUser] = useState<User | null>(null);

  // 惰性初始化保持类型
  const [data, setData] = useState<ExpensiveData>(() => {
    const saved = localStorage.getItem('data');
    return saved ? JSON.parse(saved) : createDefaultData();
  });
}

// ============ 泛型 localStorage Hook ============
function useLocalStorage<T>(
  key: string,
  initialValue: T
): [T, (value: T | ((prev: T) => T)) => void, () => void] {
  const [storedValue, setStoredValue] = useState<T>(() => {
    try {
      const item = window.localStorage.getItem(key);
      return item ? (JSON.parse(item) as T) : initialValue;
    } catch {
      return initialValue;
    }
  });

  const setValue = useCallback(
    (value: T | ((prev: T) => T)) => {
      setStoredValue((prev) => {
        const nextValue = value instanceof Function ? value(prev) : value;
        window.localStorage.setItem(key, JSON.stringify(nextValue));
        return nextValue;
      });
    },
    [key]
  );

  const removeValue = useCallback(() => {
    window.localStorage.removeItem(key);
    setStoredValue(initialValue);
  }, [key, initialValue]);

  return [storedValue, setValue, removeValue];
}

// ============ 泛型异步 Hook ============
interface UseAsyncState<T> {
  data: T | null;
  loading: boolean;
  error: Error | null;
}

interface UseAsyncReturn<T> extends UseAsyncState<T> {
  execute: (...args: any[]) => Promise<void>;
  reset: () => void;
}

function useAsync<T>(
  asyncFn: (...args: any[]) => Promise<T>
): UseAsyncReturn<T> {
  const [state, setState] = useState<UseAsyncState<T>>({
    data: null,
    loading: false,
    error: null,
  });

  const execute = useCallback(
    async (...args: any[]) => {
      setState({ data: null, loading: true, error: null });
      try {
        const data = await asyncFn(...args);
        setState({ data, loading: false, error: null });
      } catch (error) {
        setState({ data: null, loading: false, error: error as Error });
      }
    },
    [asyncFn]
  );

  const reset = useCallback(() => {
    setState({ data: null, loading: false, error: null });
  }, []);

  return { ...state, execute, reset };
}

// 使用
function UserProfile({ userId }: { userId: string }) {
  const { data: user, loading, error, execute } = useAsync<User>(
    useCallback(
      () => fetch(`/api/users/${userId}`).then((r) => r.json()),
      [userId]
    )
  );

  useEffect(() => {
    execute();
  }, [execute]);

  if (loading) return <div>加载中...</div>;
  if (error) return <div>错误: {error.message}</div>;
  if (!user) return <div>无数据</div>;

  return <div>{user.name}</div>;
}

// ============ 泛型受控/非受控 Hook ============
function useControllable<T>(
  controlledValue: T | undefined,
  defaultValue: T,
  onChange?: (value: T) => void
): [T, (value: T) => void] {
  const [internalValue, setInternalValue] = useState<T>(defaultValue);
  const isControlled = controlledValue !== undefined;
  const value = isControlled ? controlledValue : internalValue;

  const setValue = useCallback(
    (newValue: T) => {
      if (!isControlled) {
        setInternalValue(newValue);
      }
      onChange?.(newValue);
    },
    [isControlled, onChange]
  );

  return [value, setValue];
}
```

## 7.2 测试

### 7.2.1 Vitest 配置与 jsdom 环境

```tsx
// vite.config.ts — 测试相关配置
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  test: {
    // 启用全局 API（describe, it, expect 无需 import）
    globals: true,

    // jsdom 模拟浏览器环境（DOM API、localStorage 等）
    environment: 'jsdom',

    // 测试环境初始化文件
    setupFiles: ['./src/test/setup.ts'],

    // CSS 处理：导入 CSS 时返回空对象（避免 CSS 解析错误）
    css: {
      modules: {
        classNameStrategy: 'non-scoped',
      },
    },

    // 覆盖率配置
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html', 'lcov'],
      include: ['src/**/*.{ts,tsx}'],
      exclude: [
        'src/**/*.test.{ts,tsx}',
        'src/**/*.spec.{ts,tsx}',
        'src/**/*.d.ts',
        'src/test/**',
      ],
      thresholds: {
        lines: 80,
        functions: 80,
        branches: 75,
        statements: 80,
      },
    },
  },
});
```

```tsx
// src/test/setup.ts — 测试环境初始化
import '@testing-library/jest-dom/vitest';
import { cleanup } from '@testing-library/react';
import { afterEach, vi } from 'vitest';

// 每个测试后自动清理 DOM
afterEach(() => {
  cleanup();
});

// Mock 浏览器 API（jsdom 不完整支持）
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: vi.fn().mockImplementation((query: string) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: vi.fn(),
    removeListener: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  })),
});

// Mock IntersectionObserver
const mockIntersectionObserver = vi.fn();
mockIntersectionObserver.mockReturnValue({
  observe: () => null,
  unobserve: () => null,
  disconnect: () => null,
});
window.IntersectionObserver = mockIntersectionObserver;

// Mock scrollTo
window.scrollTo = vi.fn() as any;

// Mock ResizeObserver
window.ResizeObserver = vi.fn().mockImplementation(() => ({
  observe: vi.fn(),
  unobserve: vi.fn(),
  disconnect: vi.fn(),
}));
```

### 7.2.2 React Testing Library 核心 API

```tsx
import {
  render,
  screen,
  fireEvent,
  waitFor,
  waitForElementToBeRemoved,
  within,
} from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { Button, LoginForm, UserList, TodoApp } from '../components';

// ============ 查询方法速查 ============
// getBy...    → 找不到抛错（同步断言）
// queryBy...  → 找不到返回 null（验证不存在）
// findBy...   → 返回 Promise，等待出现（异步断言）
// getAllBy... → 返回数组（多个匹配）

describe('查询方法对比', () => {
  it('getByText — 同步查询，找不到抛错', () => {
    render(<Button>提交</Button>);
    // ✅ 元素存在
    expect(screen.getByText('提交')).toBeInTheDocument();
    // ❌ 元素不存在会直接抛错
    // screen.getByText('不存在的文本'); // 抛出 TestingLibraryElementError
  });

  it('queryByText — 同步查询，找不到返回 null', () => {
    render(<Button>保存</Button>);
    // ✅ 验证元素不存在
    expect(screen.queryByText('删除')).toBeNull();
    expect(screen.queryByText('删除')).not.toBeInTheDocument();
  });

  it('findByText — 异步查询，等待元素出现', async () => {
    render(<UserList />);
    // 等待异步渲染完成
    const userElement = await screen.findByText('张三');
    expect(userElement).toBeInTheDocument();
  });

  it('findAllByText — 等待多个元素出现', async () => {
    render(<UserList />);
    const items = await screen.findAllByText(/用户/);
    expect(items).toHaveLength(3);
  });
});

// ============ 组件测试模式：render + assert ============
describe('Button Component', () => {
  it('渲染默认状态', () => {
    render(<Button>点击我</Button>);

    const button = screen.getByRole('button', { name: '点击我' });
    expect(button).toBeInTheDocument();
    expect(button).not.toBeDisabled();
    expect(button).toHaveClass('btn-primary');
  });

  it('loading 状态禁用按钮并显示加载文本', () => {
    render(<Button loading>提交</Button>);

    const button = screen.getByRole('button');
    expect(button).toBeDisabled();
    expect(button).toHaveTextContent('加载中...');
  });

  it('点击触发 onClick 回调', async () => {
    const handleClick = vi.fn();
    render(<Button onClick={handleClick}>点击</Button>);

    await userEvent.click(screen.getByRole('button'));
    expect(handleClick).toHaveBeenCalledTimes(1);
  });

  it('disabled 时不触发点击', async () => {
    const handleClick = vi.fn();
    render(<Button disabled onClick={handleClick}>点击</Button>);

    await userEvent.click(screen.getByRole('button'));
    expect(handleClick).not.toHaveBeenCalled();
  });

  it('渲染不同 variant', () => {
    const { rerender } = render(<Button variant="primary">主要</Button>);
    expect(screen.getByRole('button')).toHaveClass('btn-primary');

    rerender(<Button variant="danger">危险</Button>);
    expect(screen.getByRole('button')).toHaveClass('btn-danger');
  });
});

// ============ 表单测试 ============
describe('LoginForm', () => {
  it('完整登录流程', async () => {
    const user = userEvent.setup();
    const handleSubmit = vi.fn();

    render(<LoginForm onSubmit={handleSubmit} />);

    // 填写表单
    await user.type(screen.getByLabelText(/邮箱/), 'test@example.com');
    await user.type(screen.getByLabelText(/密码/), 'secret123');

    // 提交
    await user.click(screen.getByRole('button', { name: /登录/ }));

    expect(handleSubmit).toHaveBeenCalledWith({
      email: 'test@example.com',
      password: 'secret123',
    });
  });

  it('空表单提交显示验证错误', async () => {
    render(<LoginForm onSubmit={vi.fn()} />);

    await userEvent.click(screen.getByRole('button', { name: /登录/ }));

    expect(screen.getByText(/请输入邮箱/)).toBeInTheDocument();
    expect(screen.getByText(/请输入密码/)).toBeInTheDocument();
  });

  it('邮箱格式错误显示提示', async () => {
    render(<LoginForm onSubmit={vi.fn()} />);

    await userEvent.type(screen.getByLabelText(/邮箱/), 'invalid-email');
    await userEvent.tab(); // 触发 blur 验证

    await waitFor(() => {
      expect(screen.getByText(/邮箱格式不正确/)).toBeInTheDocument();
    });
  });
});

// ============ 异步测试 ============
describe('UserList — 异步数据加载', () => {
  beforeEach(() => {
    vi.spyOn(globalThis, 'fetch');
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('加载成功显示用户列表', async () => {
    const mockUsers = [
      { id: 1, name: '张三', email: 'zhang@example.com' },
      { id: 2, name: '李四', email: 'li@example.com' },
    ];

    vi.mocked(fetch).mockResolvedValueOnce({
      ok: true,
      json: async () => mockUsers,
    } as Response);

    render(<UserList />);

    // 初始状态：显示 loading
    expect(screen.getByText(/加载中/)).toBeInTheDocument();

    // waitFor：轮询直到断言通过
    await waitFor(() => {
      expect(screen.getByText('张三')).toBeInTheDocument();
      expect(screen.getByText('李四')).toBeInTheDocument();
    });

    // loading 消失
    expect(screen.queryByText(/加载中/)).not.toBeInTheDocument();
  });

  it('加载失败显示错误信息', async () => {
    vi.mocked(fetch).mockRejectedValueOnce(new Error('网络连接失败'));

    render(<UserList />);

    await waitFor(() => {
      expect(screen.getByText(/加载失败/)).toBeInTheDocument();
    });

    // 重试按钮可用
    expect(screen.getByRole('button', { name: /重试/ })).toBeInTheDocument();
  });

  it('加载超时处理', async () => {
    vi.mocked(fetch).mockImplementationOnce(
      () =>
        new Promise((resolve) =>
          setTimeout(resolve, 10000)
        ) as Promise<Response>
    );

    render(<UserList />);

    // 等待超时提示
    await waitFor(
      () => {
        expect(screen.getByText(/请求超时/)).toBeInTheDocument();
      },
      { timeout: 6000 }
    );
  });

  it('空列表显示空状态', async () => {
    vi.mocked(fetch).mockResolvedValueOnce({
      ok: true,
      json: async () => [],
    } as Response);

    render(<UserList />);

    await waitFor(() => {
      expect(screen.getByText(/暂无用户/)).toBeInTheDocument();
    });
  });
});

// ============ user-event 高级用法 ============
describe('user-event 交互测试', () => {
  it('keyboard — 键盘快捷键', async () => {
    const user = userEvent.setup();
    const handleSave = vi.fn();

    render(<Editor onSave={handleSave} />);

    // 模拟 Ctrl+S
    await user.keyboard('{Control>}s{/Control}');
    expect(handleSave).toHaveBeenCalled();

    // 模拟 Escape 关闭
    const handleClose = vi.fn();
    render(<Modal onClose={handleClose}>内容</Modal>);
    await user.keyboard('{Escape}');
    expect(handleClose).toHaveBeenCalled();
  });

  it('type — 高级输入模拟', async () => {
    const user = userEvent.setup();

    render(<textarea placeholder="输入内容" />);
    const textarea = screen.getByPlaceholderText('输入内容');

    // 逐字输入
    await user.type(textarea, 'Hello{Enter}World');

    // 选中并删除
    await user.tripleClick(textarea);
    await user.keyboard('{Backspace}');

    // 粘贴
    await user.click(textarea);
    await user.paste('粘贴的内容');

    expect(textarea).toHaveValue('粘贴的内容');
  });

  it('tab — 焦点切换顺序', async () => {
    const user = userEvent.setup();

    render(
      <form>
        <input placeholder="第一个" />
        <input placeholder="第二个" />
        <button type="submit">提交</button>
      </form>
    );

    expect(screen.getByPlaceholderText('第一个')).toHaveFocus();

    await user.tab();
    expect(screen.getByPlaceholderText('第二个')).toHaveFocus();

    await user.tab();
    expect(screen.getByRole('button')).toHaveFocus();
  });

  it('hover — 悬停效果', async () => {
    const user = userEvent.setup();

    render(<Tooltip content="提示信息"><button>悬停我</button></Tooltip>);

    expect(screen.queryByText('提示信息')).not.toBeInTheDocument();

    await user.hover(screen.getByRole('button'));
    expect(screen.getByText('提示信息')).toBeInTheDocument();

    await user.unhover(screen.getByRole('button'));
    expect(screen.queryByText('提示信息')).not.toBeInTheDocument();
  });

  it('upload — 文件上传', async () => {
    const user = userEvent.setup();
    const handleUpload = vi.fn();

    render(<ImageUploader onUpload={handleUpload} />);

    const file = new File(['image-content'], 'photo.png', { type: 'image/png' });
    const input = screen.getByLabelText(/上传图片/);

    await user.upload(input, file);

    expect(handleUpload).toHaveBeenCalledWith(expect.any(File));
    expect(screen.getByText('photo.png')).toBeInTheDocument();
  });
});
```

### 7.2.3 自定义 Hook 测试

```tsx
import { renderHook, act } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';

describe('useCounter', () => {
  it('使用默认初始值', () => {
    const { result } = renderHook(() => useCounter());
    expect(result.current.count).toBe(0);
  });

  it('使用自定义初始值', () => {
    const { result } = renderHook(() => useCounter(10));
    expect(result.current.count).toBe(10);
  });

  it('increment 增加计数', () => {
    const { result } = renderHook(() => useCounter(0));

    act(() => {
      result.current.increment();
    });

    expect(result.current.count).toBe(1);
  });

  it('decrement 减少计数', () => {
    const { result } = renderHook(() => useCounter(5));

    act(() => {
      result.current.decrement();
      result.current.decrement();
    });

    expect(result.current.count).toBe(3);
  });

  it('reset 恢复初始值', () => {
    const { result } = renderHook(() => useCounter(100));

    act(() => {
      result.current.increment();
      result.current.increment();
      result.current.reset();
    });

    expect(result.current.count).toBe(100);
  });

  it('重新渲染时保持状态', () => {
    const { result, rerender } = renderHook(() => useCounter(0));

    act(() => {
      result.current.increment();
    });

    rerender();

    expect(result.current.count).toBe(1);
  });
});

describe('useLocalStorage', () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it('读取和写入 localStorage', () => {
    const { result } = renderHook(() => useLocalStorage('theme', 'light'));

    act(() => {
      result.current[1]('dark');
    });

    expect(result.current[0]).toBe('dark');
    expect(localStorage.getItem('theme')).toBe('"dark"');
  });

  it('使用函数式更新', () => {
    const { result } = renderHook(() => useLocalStorage<number>('count', 0));

    act(() => {
      result.current[1]((prev) => prev + 1);
    });

    expect(result.current[0]).toBe(1);
  });
});
```

## 7.3 调试

### 7.3.1 React DevTools — Components 面板

React DevTools 的 Components 面板是调试组件树的核心工具。

```tsx
// 安装：Chrome/Firefox 扩展商店搜索 "React Developer Tools"
// 或安装独立版本: npx react-devtools

// ============ displayName — 让组件在 DevTools 中可识别 ============
// React DevTools 默认使用函数名或变量名，但以下情况需要手动设置：

// 1. memo 包裹的组件
const UserCard = memo(function UserCard({ user }: { user: User }) {
  return <div>{user.name}</div>;
});
UserCard.displayName = 'UserCard';

// 2. forwardRef 包裹的组件
const TextInput = forwardRef<HTMLInputElement, TextInputProps>(
  function TextInput(props, ref) {
    return <input ref={ref} {...props} />;
  }
);
TextInput.displayName = 'TextInput';

// 3. HOC 包裹的组件
function withAuth<P extends object>(Component: React.ComponentType<P>) {
  const WithAuth = (props: P) => {
    const isAuthenticated = useAuth();
    if (!isAuthenticated) return <Navigate to="/login" />;
    return <Component {...props} />;
  };
  WithAuth.displayName = `withAuth(${Component.displayName ?? Component.name})`;
  return WithAuth;
}

// ============ DevTools Components 面板功能 ============
// - 组件树：查看完整的组件层级结构
// - Props 面板：查看/编辑当前选中组件的 props
// - State 面板：查看/编辑组件的 state（需要组件支持）
// - Hooks 面板：查看所有 hook 的当前值
// - 搜索：Ctrl+F 搜索组件名
// - 高亮更新：设置 > Highlight updates when components render
```

### 7.3.2 React DevTools — Profiler 面板

```tsx
import { Profiler } from 'react';

// ============ Profiler 编程式使用 ============
function ProfilerDemo() {
  const onRender: React.ProfilerOnRenderCallback = (
    id,              // Profiler 组件的 id
    phase,           // "mount" 或 "update"
    actualDuration,  // 本次提交中渲染该子树的实际耗时 (ms)
    baseDuration,    // 无 memo 情况下渲染子树的理论耗时 (ms)
    startTime,       // React 开始渲染的时间戳
    commitTime       // React 提交更新的时间戳
  ) => {
    // 在控制台输出详细渲染信息
    console.group(`Profiler [${id}] - ${phase}`);
    console.log('实际耗时:', `${actualDuration.toFixed(2)}ms`);
    console.log('理论耗时:', `${baseDuration.toFixed(2)}ms`);
    console.log('开始时间:', startTime);
    console.log('提交时间:', commitTime);
    console.groupEnd();

    // 标记慢渲染（帧预算 16.67ms）
    if (actualDuration > 16) {
      console.warn(
        `⚠️ 组件 "${id}" 渲染耗时 ${actualDuration.toFixed(2)}ms，超过帧预算`
      );
    }
  };

  return (
    <Profiler id="AppRoot" onRender={onRender}>
      <Profiler id="Header" onRender={onRender}>
        <Header />
      </Profiler>
      <Profiler id="Content" onRender={onRender}>
        <Content />
      </Profiler>
    </Profiler>
  );
}

// ============ DevTools Profiler 面板功能 ============
// - 录制按钮：开始/停止性能录制
// - Flamegraph（火焰图）：按层级显示组件渲染耗时，颜色越深耗时越长
// - Ranked（排行）：按渲染耗时降序排列组件
// - 提交列表：每次提交的快照，可以对比不同提交的性能差异
// - 交互追踪：追踪用户交互（点击、输入等）触发的渲染

// ============ Profiler 最佳实践 ============
// 1. 在生产构建中测试（开发模式有额外开销）
// 2. 录制 2-3 秒的操作而非整个页面加载
// 3. 关注排名靠前的组件（Pareto 原则：80% 的耗时来自 20% 的组件）
// 4. 对比优化前后的火焰图
```

### 7.3.3 浏览器断点调试

```tsx
// ============ debugger 语句 ============
// 在代码中插入 debugger 语句，浏览器会自动在此处暂停

function BuggyComponent() {
  const [count, setCount] = useState(0);
  const [data, setData] = useState<Item[]>([]);

  const handleProcess = async () => {
    // 条件断点：仅在特定条件下暂停
    if (count > 5) {
      debugger; // 浏览器在此暂停，可检查 count、data 等变量
    }

    const result = await fetchData();

    // 检查异步结果
    debugger; // 检查 result 的值
    setData(result);
  };

  return (
    <div>
      <p>Count: {count}</p>
      <button onClick={() => setCount((c) => c + 1)}>+1</button>
      <button onClick={handleProcess}>处理数据</button>
    </div>
  );
}

// ============ 浏览器 DevTools 断点类型 ============
// 1. 代码行断点：点击行号设置（蓝色标记）
// 2. 条件断点：右键行号 > "Add conditional breakpoint"
//    输入表达式如: count > 5 && data.length === 0
// 3. DOM 断点：Elements 面板 > 右键元素 > Break on >
//    - subtree modifications: 子节点变化时断点
//    - attribute modifications: 属性变化时断点
//    - node removal: 节点被移除时断点
// 4. XHR/Fetch 断点：Sources > XHR/fetch Breakpoints
//    输入 URL 片段如 "/api/users"，匹配的请求会触发断点
// 5. Event Listener 断点：Sources > Event Listener Breakpoints
//    勾选 Mouse > click，任何点击事件都会触发断点

// ============ 条件断点实用表达式 ============
// item.id === 'specific-id'              — 特定数据项
// e.target.value.length > 10             — 输入长度超限
// props.items === undefined              — prop 为 undefined
// this.state.error !== null              — 错误状态
// performance.now() - startTime > 100    — 耗时超过 100ms
```

### 7.3.4 Source Maps

Source Maps 将编译后的代码映射回原始源代码，是调试构建产物的基础。

```tsx
// vite.config.ts — Source Map 配置
export default defineConfig({
  build: {
    // 生产环境 source map 策略
    sourcemap: true, // 或 'hidden' | 'inline'

    // 各策略对比：
    // true           → 生成独立的 .js.map 文件，浏览器自动加载
    // 'hidden'       → 生成 .js.map 但不添加 sourceMappingURL 注释
    //                   适合上传到 Sentry 等监控平台，不暴露给终端用户
    // 'inline'       → source map 内联为 base64（文件大，但无额外请求）
    // false          → 不生成 source map（构建最快）
  },
});

// ============ Source Map 最佳实践 ============
// 开发环境：sourcemap: true（或默认的 eval-cheap-module-source-map）
// 生产环境：sourcemap: 'hidden'（上传到 Sentry 后删除 .map 文件）
// 调试线上问题：在 Sources 面板手动加载 .map 文件

// ============ Chrome DevTools 中使用 Source Maps ============
// 1. Sources > Filesystem > Add folder to workspace
// 2. 映射本地文件夹到线上 URL（如 app.example.com → src/）
// 3. 在 Sources 面板编辑文件，修改会保存到本地
// 4. 结合 "Local Overrides" 可以在线上环境调试本地修改
```

## 7.4 构建工具

### 7.4.1 Vite — 下一代前端构建工具

Vite 利用浏览器原生 ES Module 实现极速开发体验，生产构建使用 Rollup。

```tsx
// vite.config.ts — 完整配置示例
import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

// Vite 配置可以是一个函数，接收 mode 参数
export default defineConfig(({ mode }) => {
  // 加载环境变量（默认只加载 VITE_ 前缀的变量）
  const env = loadEnv(mode, process.cwd(), '');

  return {
    // ============ 插件 ============
    plugins: [
      react({
        // 使用 Babel（默认使用 esbuild，生产环境推荐）
        babel: {
          plugins: ['babel-plugin-macros'],
        },
        // 或使用 SWC（更快的替代方案）
        // 需要安装 @vitejs/plugin-react-swc
      }),
    ],

    // ============ 路径别名 ============
    resolve: {
      alias: {
        '@': path.resolve(__dirname, './src'),
        '@components': path.resolve(__dirname, './src/components'),
        '@hooks': path.resolve(__dirname, './src/hooks'),
        '@utils': path.resolve(__dirname, './src/utils'),
        '@types': path.resolve(__dirname, './src/types'),
        '@assets': path.resolve(__dirname, './src/assets'),
      },
    },

    // ============ 开发服务器 ============
    server: {
      port: 3000,
      open: true, // 自动打开浏览器

      // API 代理
      proxy: {
        '/api': {
          target: env.VITE_API_BASE_URL || 'http://localhost:8080',
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/api/, ''),
        },
        '/ws': {
          target: 'ws://localhost:8080',
          ws: true,
        },
      },

      // HMR 配置
      hmr: {
        overlay: true, // 错误遮罩层
      },
    },

    // ============ 构建配置 ============
    build: {
      // 目标浏览器
      target: 'es2020',

      // 输出目录
      outDir: 'dist',

      // 资源目录
      assetsDir: 'assets',

      // chunk 大小警告阈值 (KB)
      chunkSizeWarningLimit: 500,

      // Source Map
      sourcemap: mode === 'production' ? 'hidden' : true,

      // Rollup 配置（生产构建使用 Rollup）
      rollupOptions: {
        output: {
          // 手动代码分割
          manualChunks: {
            // React 核心
            'vendor-react': ['react', 'react-dom', 'react-router-dom'],

            // UI 组件库
            'vendor-ui': ['@radix-ui/react-dialog', '@radix-ui/react-dropdown-menu'],

            // 图表库（按需加载的大体积依赖）
            'vendor-charts': ['recharts'],

            // 工具库
            'vendor-utils': ['date-fns', 'zod', 'immer'],
          },
        },
      },

      // 压缩
      minify: 'terser',
      terserOptions: {
        compress: {
          drop_console: mode === 'production',
          drop_debugger: mode === 'production',
        },
      },

      // CSS 代码分割
      cssCodeSplit: true,
    },

    // ============ CSS ============
    css: {
      modules: {
        localsConvention: 'camelCaseOnly',
        scopeBehaviour: 'local',
      },
      preprocessorOptions: {
        scss: {
          additionalData: `@import "@/styles/variables.scss";`,
        },
      },
    },

    // ============ 环境变量前缀 ============
    envPrefix: ['VITE_', 'APP_'],

    // ============ 预构建依赖 ============
    // 开发模式下，Vite 使用 esbuild 预构建 node_modules 中的依赖
    // 将 CommonJS/UMD 模块转换为 ESM，并合并碎片化模块
    optimizeDeps: {
      include: [
        'react',
        'react-dom',
        'lodash-es',
        'date-fns',
      ],
      exclude: [
        // 不需要预构建的依赖（已经是 ESM 且不需要合并）
      ],
    },
  };
});
```

```tsx
// ============ Vite 环境变量 ============
// .env                  — 所有模式加载
// .env.local            — 所有模式加载（git ignore）
// .env.development      — 仅开发模式
// .env.production       — 仅生产模式

// .env
VITE_APP_TITLE=My React App
VITE_API_BASE_URL=http://localhost:8080

// .env.production
VITE_API_BASE_URL=https://api.example.com

// 使用环境变量
// import.meta.env.VITE_APP_TITLE      → "My React App"
// import.meta.env.VITE_API_BASE_URL   → 根据模式变化
// import.meta.env.MODE                → "development" | "production"
// import.meta.env.DEV                 → true (开发模式)
// import.meta.env.PROD                → true (生产模式)

// 类型声明 (src/vite-env.d.ts)
/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_APP_TITLE: string;
  readonly VITE_API_BASE_URL: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}

// ============ Vite HMR 原理 ============
// 1. Vite 启动时用 esbuild 预构建依赖（快）
// 2. 源代码通过原生 ESM 按需提供给浏览器
// 3. 修改文件时，Vite 只失效受影响的模块链
// 4. 浏览器通过 WebSocket 接收更新，执行 HMR 替换
// 5. 相比 Webpack 全量打包，Vite HMR 在大型项目中快得多
```

### 7.4.2 Webpack — 经典构建工具

```tsx
// webpack.config.js
const path = require('path');
const HtmlWebpackPlugin = require('html-webpack-plugin');
const MiniCssExtractPlugin = require('mini-css-extract-plugin');
const BundleAnalyzerPlugin = require('webpack-bundle-analyzer').BundleAnalyzerPlugin;
const TerserPlugin = require('terser-webpack-plugin');

module.exports = (env, argv) => {
  const isProduction = argv.mode === 'production';

  return {
    mode: isProduction ? 'production' : 'development',

    // ============ 入口 ============
    entry: {
      main: './src/index.tsx',
    },

    // ============ 输出 ============
    output: {
      path: path.resolve(__dirname, 'dist'),
      filename: isProduction ? '[name].[contenthash:8].js' : '[name].js',
      chunkFilename: isProduction ? '[name].[contenthash:8].chunk.js' : '[name].chunk.js',
      publicPath: '/',
      clean: true,
    },

    // ============ Source Map ============
    devtool: isProduction ? 'hidden-source-map' : 'eval-cheap-module-source-map',

    // ============ Loader 链 ============
    module: {
      rules: [
        // TypeScript / JavaScript
        {
          test: /\.(ts|tsx|js|jsx)$/,
          exclude: /node_modules/,
          use: {
            loader: 'babel-loader',
            options: {
              presets: [
                ['@babel/preset-env', { targets: '> 0.25%, not dead' }],
                ['@babel/preset-react', { runtime: 'automatic' }],
                '@babel/preset-typescript',
              ],
            },
          },
        },

        // CSS Modules
        {
          test: /\.module\.css$/,
          use: [
            isProduction ? MiniCssExtractPlugin.loader : 'style-loader',
            {
              loader: 'css-loader',
              options: {
                modules: {
                  localIdentName: isProduction
                    ? '[hash:base64:8]'
                    : '[name]__[local]--[hash:base64:5]',
                },
              },
            },
          ],
        },

        // 全局 CSS
        {
          test: /\.css$/,
          exclude: /\.module\.css$/,
          use: [
            isProduction ? MiniCssExtractPlugin.loader : 'style-loader',
            'css-loader',
          ],
        },

        // SCSS
        {
          test: /\.scss$/,
          use: [
            isProduction ? MiniCssExtractPlugin.loader : 'style-loader',
            'css-loader',
            'sass-loader',
          ],
        },

        // 图片和字体
        {
          test: /\.(png|jpg|jpeg|gif|webp|svg)$/,
          type: 'asset',
          parser: {
            dataUrlCondition: {
              maxSize: 8 * 1024, // 8KB 以下内联为 base64
            },
          },
          generator: {
            filename: 'assets/images/[name].[hash:8][ext]',
          },
        },

        {
          test: /\.(woff|woff2|eot|ttf|otf)$/,
          type: 'asset/resource',
          generator: {
            filename: 'assets/fonts/[name].[hash:8][ext]',
          },
        },
      ],
    },

    // ============ 解析 ============
    resolve: {
      extensions: ['.ts', '.tsx', '.js', '.jsx'],
      alias: {
        '@': path.resolve(__dirname, 'src'),
      },
    },

    // ============ 插件 ============
    plugins: [
      // 生成 HTML 文件
      new HtmlWebpackPlugin({
        template: './public/index.html',
        favicon: './public/favicon.ico',
      }),

      // 提取 CSS 为单独文件
      ...(isProduction
        ? [
            new MiniCssExtractPlugin({
              filename: 'css/[name].[contenthash:8].css',
            }),
          ]
        : []),

      // 打包分析（可选）
      // new BundleAnalyzerPlugin(),
    ],

    // ============ 代码分割优化 ============
    optimization: {
      // 压缩
      minimizer: [
        new TerserPlugin({
          terserOptions: {
            compress: {
              drop_console: isProduction,
            },
          },
        }),
      ],

      // SplitChunks — 自动代码分割
      splitChunks: {
        chunks: 'all',
        cacheGroups: {
          // 提取 node_modules 中的公共依赖
          vendor: {
            test: /[\\/]node_modules[\\/]/,
            name(module) {
              // 获取包名
              const packageName = module.context.match(
                /[\\/]node_modules[\\/](.*?)([\\/]|$)/
              )[1];
              // npm 包名（作用域包如 @scope/name 直接使用）
              return `vendor.${packageName.replace('@', '')}`;
            },
            priority: 10,
          },
          // 提取公共模块
          common: {
            minChunks: 2,
            priority: 5,
            reuseExistingChunk: true,
          },
        },
      },

      // 运行时代码单独打包
      runtimeChunk: 'single',
    },

    // ============ 开发服务器 ============
    devServer: {
      port: 3000,
      hot: true, // HMR
      historyApiFallback: true, // SPA 路由支持
      proxy: [
        {
          context: ['/api'],
          target: 'http://localhost:8080',
          changeOrigin: true,
        },
      ],
    },
  };
};

// ============ Vite vs Webpack 核心差异 ============
// 开发启动：Vite 秒开（esbuild 预构建 + ESM），Webpack 需要全量打包
// HMR：Vite 只更新变更模块链，Webpack 需要重新打包变更的 chunk
// 生产构建：Vite 使用 Rollup，Webpack 自带，两者性能接近
// 生态：Webpack 插件/loader 生态更成熟，Vite 基于 Rollup 插件 + 自建
// 配置复杂度：Vite 开箱即用，Webpack 需要较多配置
// 推荐：新项目用 Vite，已有 Webpack 项目不必强迁
```

## 7.5 CI/CD

### 7.5.1 GitHub Actions 完整工作流

```yaml
# .github/workflows/ci.yml
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

# 并发控制：同一分支的新提交会取消旧的运行
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

env:
  NODE_VERSION: '20'
  PNPM_VERSION: '9'

jobs:
  # ============ 代码质量检查 ============
  lint:
    name: Lint & Type Check
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: pnpm/action-setup@v4
        with:
          version: ${{ env.PNPM_VERSION }}

      - uses: actions/setup-node@v4
        with:
          node-version: ${{ env.NODE_VERSION }}
          cache: 'pnpm'

      - run: pnpm install --frozen-lockfile

      - name: ESLint
        run: pnpm lint

      - name: TypeScript 类型检查
        run: pnpm typecheck

      - name: Prettier 格式检查
        run: pnpm format:check

  # ============ 测试 ============
  test:
    name: Test
    runs-on: ubuntu-latest
    needs: lint
    steps:
      - uses: actions/checkout@v4

      - uses: pnpm/action-setup@v4
        with:
          version: ${{ env.PNPM_VERSION }}

      - uses: actions/setup-node@v4
        with:
          node-version: ${{ env.NODE_VERSION }}
          cache: 'pnpm'

      - run: pnpm install --frozen-lockfile

      - name: 运行单元测试
        run: pnpm test -- --coverage

      - name: 上传覆盖率到 Codecov
        uses: codecov/codecov-action@v4
        with:
          token: ${{ secrets.CODECOV_TOKEN }}
          files: ./coverage/lcov.info
          fail_ci_if_error: false

  # ============ 构建 ============
  build:
    name: Build
    runs-on: ubuntu-latest
    needs: test
    steps:
      - uses: actions/checkout@v4

      - uses: pnpm/action-setup@v4
        with:
          version: ${{ env.PNPM_VERSION }}

      - uses: actions/setup-node@v4
        with:
          node-version: ${{ env.NODE_VERSION }}
          cache: 'pnpm'

      - run: pnpm install --frozen-lockfile

      - name: 生产构建
        run: pnpm build

      - name: 构建产物分析
        run: |
          echo "=== 构建产物大小 ==="
          du -sh dist/
          echo "=== JS 文件 ==="
          ls -lhS dist/assets/*.js | head -10

      - name: 缓存构建产物
        uses: actions/upload-artifact@v4
        with:
          name: build-output
          path: dist/
          retention-days: 7

  # ============ 部署 ============
  deploy:
    name: Deploy
    runs-on: ubuntu-latest
    needs: build
    if: github.ref == 'refs/heads/main'
    # 需要手动审批（可选）
    # environment: production
    steps:
      - uses: actions/download-artifact@v4
        with:
          name: build-output
          path: dist/

      # Vercel 部署
      - name: Deploy to Vercel
        uses: amondnet/vercel-action@v25
        with:
          vercel-token: ${{ secrets.VERCEL_TOKEN }}
          vercel-org-id: ${{ secrets.VERCEL_ORG_ID }}
          vercel-project-id: ${{ secrets.VERCEL_PROJECT_ID }}
          vercel-args: '--prod --confirm'
          working-directory: ./

  # ============ E2E 测试（部署后） ============
  e2e:
    name: E2E Tests
    runs-on: ubuntu-latest
    needs: deploy
    steps:
      - uses: actions/checkout@v4

      - uses: pnpm/action-setup@v4
        with:
          version: ${{ env.PNPM_VERSION }}

      - uses: actions/setup-node@v4
        with:
          node-version: ${{ env.NODE_VERSION }}
          cache: 'pnpm'

      - run: pnpm install --frozen-lockfile

      - name: 安装 Playwright 浏览器
        run: npx playwright install --with-deps chromium

      - name: 运行 E2E 测试
        run: pnpm test:e2e
        env:
          BASE_URL: https://your-app.vercel.app
```

### 7.5.2 Vercel / Netlify 自动部署

```tsx
// ============ Vercel 自动部署 ============
// Vercel 连接 Git 仓库后，每次 push 自动触发部署

// 1. 连接方式：
//    - 在 vercel.com 导入 GitHub/GitLab/Bitbucket 仓库
//    - 或安装 Vercel GitHub App 自动同步

// 2. 自动部署策略：
//    - Production: main 分支的每次 push → 自动部署到生产环境
//    - Preview: 每个 PR 自动创建预览部署（带唯一 URL）
//    - 预览 URL 格式: {project}-{hash}-{scope}.vercel.app

// 3. vercel.json 配置
{
  "buildCommand": "pnpm build",
  "outputDirectory": "dist",
  "installCommand": "pnpm install --frozen-lockfile",
  "framework": "vite",
  "rewrites": [
    {
      "source": "/(.*)",
      "destination": "/index.html"
    }
  ],
  "headers": [
    {
      "source": "/assets/(.*)",
      "headers": [
        {
          "key": "Cache-Control",
          "value": "public, max-age=31536000, immutable"
        }
      ]
    }
  ]
}

// 4. 环境变量管理：
//    - Vercel Dashboard > Settings > Environment Variables
//    - 支持 Development / Preview / Production 三种环境
//    - 加密存储，构建时注入

// ============ Netlify 自动部署 ============
// Netlify 提供类似的 Git 连接和自动部署功能

// netlify.toml
[build]
  command = "pnpm build"
  publish = "dist"

[build.environment]
  NODE_VERSION = "20"

[[redirects]]
  from = "/*"
  to = "/index.html"
  status = 200

[[headers]]
  for = "/assets/*"
  [headers.values]
    Cache-Control = "public, max-age=31536000, immutable"

// Netlify 特色功能：
// - Deploy Preview: PR 自动生成预览链接
// - Branch Deploy: 非主分支自动部署（不同于主分支的子域名）
// - Split Testing: A/B 测试不同分支
// - Forms: 内置表单处理（无需后端）
// - Functions: Serverless 函数支持
// - Edge Functions: Deno 运行时，全球边缘节点执行
```

## 7.6 监控

### 7.6.1 Sentry — 错误监控与性能追踪

```tsx
// ============ Sentry 初始化 ============
// src/sentry.ts
import * as Sentry from '@sentry/react';

Sentry.init({
  dsn: import.meta.env.VITE_SENTRY_DSN,

  // 环境
  environment: import.meta.env.MODE,

  // 采样率（性能追踪）
  tracesSampleRate: import.meta.env.PROD ? 0.1 : 1.0,

  // 重放（用户操作回放）
  replaysSessionSampleRate: 0.1,  // 完整会话
  replaysOnErrorSampleRate: 1.0,  // 错误发生时的会话

  // 版本号（关联 source map）
  release: import.meta.env.VITE_APP_VERSION || '0.0.0',

  // 过滤敏感数据
  beforeSend(event) {
    // 不发送本地开发环境的错误
    if (import.meta.env.DEV) return null;
    return event;
  },

  // React 特定配置
  integrations: [
    Sentry.browserTracingIntegration(),
    Sentry.replayIntegration({
      maskAllText: true,      // 默认遮蔽文字
      blockAllMedia: true,    // 默认屏蔽媒体
    }),
    Sentry.browserProfilingIntegration(),
  ],
});

// ============ Error Boundary 集成 ============
// src/components/ErrorFallback.tsx
import type { FallbackProps } from '@sentry/react';

function ErrorFallback({ error, resetError, eventId }: FallbackProps) {
  return (
    <div role="alert" className="error-boundary">
      <h2>抱歉，页面出现了错误</h2>
      <p>错误 ID: {eventId}</p>
      <details>
        <summary>错误详情</summary>
        <pre>{error.message}</pre>
      </details>
      <div className="error-actions">
        <button onClick={resetError}>重试</button>
        <button onClick={() => window.location.reload()}>刷新页面</button>
      </div>
    </div>
  );
}

// src/main.tsx
import { ErrorBoundary, Profiler } from '@sentry/react';

function App() {
  return (
    // Sentry ErrorBoundary 自动上报错误
    <ErrorBoundary
      fallback={ErrorFallback}
      showDialog  // 显示用户反馈对话框
      beforeCapture={(scope) => {
        scope.setTag('feature', 'main-app');
        scope.setLevel('error');
      }}
    >
      {/* Sentry Profiler 追踪组件性能 */}
      <Profiler name="AppRoot">
        <RouterProvider router={router} />
      </Profiler>
    </ErrorBoundary>
  );
}

// ============ 手动上报 ============
function CheckoutPage() {
  const handlePayment = async () => {
    try {
      await processPayment();
    } catch (error) {
      // 手动上报带上下文的错误
      Sentry.captureException(error, {
        tags: {
          feature: 'payment',
          payment_method: 'credit_card',
        },
        extra: {
          order_id: 'ORD-12345',
          amount: 199.99,
        },
        user: {
          id: 'user-123',
          email: 'user@example.com',
        },
        level: 'fatal',
      });

      // 或使用 captureMessage 上报自定义事件
      Sentry.captureMessage('支付流程失败', {
        level: 'error',
        tags: { feature: 'payment' },
      });
    }
  };

  return <button onClick={handlePayment}>支付</button>;
}

// ============ Sentry Release 与 Source Map 上传 ============
// vite.config.ts
export default defineConfig({
  build: {
    sourcemap: 'hidden', // 生成但不公开引用
  },
});

// .github/workflows/release.yml 中添加步骤
// - name: Create Sentry release
//   run: |
//     npx sentry-cli releases new ${{ github.sha }}
//     npx sentry-cli releases files ${{ github.sha }} upload-sourcemaps ./dist
//     npx sentry-cli releases finalize ${{ github.sha }}
//   env:
//     SENTRY_AUTH_TOKEN: ${{ secrets.SENTRY_AUTH_TOKEN }}
//     SENTRY_ORG: your-org
//     SENTRY_PROJECT: your-project
```

### 7.6.2 Web Vitals — 性能核心指标

```tsx
// ============ Web Vitals 核心指标 ============
// LCP (Largest Contentful Paint)   — < 2.5s   最大内容绘制
// FID (First Input Delay)          — < 100ms  首次输入延迟
// CLS (Cumulative Layout Shift)    — < 0.1    累积布局偏移
// INP (Interaction to Next Paint)  — < 200ms  交互到下次绘制（替代 FID）
// FCP (First Contentful Paint)     — < 1.8s   首次内容绘制
// TTFB (Time to First Byte)        — < 800ms  首字节时间

// ============ 使用 web-vitals 库上报 ============
// npm install web-vitals

// src/vitals.ts
import { onCLS, onFID, onLCP, onINP, onFCP, onTTFB, Metric } from 'web-vitals';

type MetricReporter = (metric: Metric) => void;

// 发送到分析服务
const sendToAnalytics: MetricReporter = (metric) => {
  // 发送到 Google Analytics
  if (typeof gtag === 'function') {
    gtag('event', 'web_vitals', {
      event_category: 'Web Vitals',
      event_label: metric.id,
      value: Math.round(metric.name === 'CLS' ? metric.value * 1000 : metric.value),
      metric_name: metric.name,
      metric_value: metric.value,
      metric_rating: metric.rating, // 'good' | 'needs-improvement' | 'poor'
      non_interaction: true, // 不影响跳出率
    });
  }

  // 发送到 Sentry
  if (metric.rating !== 'good') {
    Sentry.captureMessage(`Poor Web Vital: ${metric.name}`, {
      level: 'warning',
      extra: {
        metric_name: metric.name,
        metric_value: metric.value,
        metric_rating: metric.rating,
      },
    });
  }

  // 开发环境输出到控制台
  if (import.meta.env.DEV) {
    console.log(`[Web Vital] ${metric.name}: ${metric.value} (${metric.rating})`);
  }
};

// 注册所有指标
onCLS(sendToAnalytics);
onFID(sendToAnalytics);
onLCP(sendToAnalytics);
onINP(sendToAnalytics);
onFCP(sendToAnalytics);
onTTFB(sendToAnalytics);

// ============ 自定义性能监控组件 ============
function PerformanceMonitor() {
  useEffect(() => {
    // 页面加载性能
    if (window.performance) {
      const navigation = performance.getEntriesByType(
        'navigation'
      )[0] as PerformanceNavigationTiming;

      console.table({
        DNS: `${navigation.domainLookupEnd - navigation.domainLookupStart}ms`,
        TCP: `${navigation.connectEnd - navigation.connectStart}ms`,
        Request: `${navigation.responseStart - navigation.requestStart}ms`,
        Response: `${navigation.responseEnd - navigation.responseStart}ms`,
        DOM: `${navigation.domContentLoadedEventEnd - navigation.domContentLoadedEventStart}ms`,
        Load: `${navigation.loadEventEnd - navigation.loadEventStart}ms`,
        Total: `${navigation.loadEventEnd - navigation.startTime}ms`,
      });
    }

    // 长任务监控
    if ('PerformanceObserver' in window) {
      const observer = new PerformanceObserver((list) => {
        for (const entry of list.getEntries()) {
          if (entry.duration > 50) {
            // 超过 50ms 的长任务
            console.warn(
              `长任务检测: ${entry.duration.toFixed(2)}ms`,
              entry
            );
          }
        }
      });

      observer.observe({ entryTypes: ['longtask'] });
      return () => observer.disconnect();
    }
  }, []);

  return null;
}

// ============ 延迟加载优化 LCP ============
// LCP 通常受最大图片/文本块影响
function HeroSection() {
  return (
    <section>
      {/* LCP 元素：预加载、优化尺寸 */}
      <img
        src="/hero-banner.webp"
        alt="主横幅"
        width={1200}
        height={600}
        loading="eager"      // 立即加载（不懒加载 LCP 图片）
        fetchPriority="high" // 高优先级获取
      />
    </section>
  );
}

// ============ 防止 CLS 的技巧 ============
// CLS 主要由无尺寸的图片/广告/嵌入内容导致

function CLSPrevention() {
  return (
    <>
      {/* ✅ 明确图片尺寸，预留空间 */}
      <img src="/photo.jpg" alt="" width={800} height={600} />

      {/* ✅ 使用 aspect-ratio 预留空间 */}
      <div style={{ aspectRatio: '16/9', overflow: 'hidden' }}>
        <iframe src="..." width="100%" height="100%" />
      </div>

      {/* ✅ 骨架屏占位 */}
      <Suspense fallback={<div style={{ height: 200 }} className="skeleton" />}>
        <LazyComponent />
      </Suspense>

      {/* ❌ 避免动态注入内容导致布局跳动 */}
      {/* 不要这样做：加载完成后突然插入横幅 */}
    </>
  );
}
```

## 7.7 安全

### 7.7.1 XSS 防护

```tsx
// ============ JSX 自动转义 ============
// JSX 默认对插值内容进行 HTML 转义，这是 React 最重要的 XSS 防护机制

function XSSProtection() {
  const userInput = '<img src=x onerror=alert("XSS")>';

  return (
    <div>
      {/* ✅ 安全：自动转义为纯文本 */}
      <p>{userInput}</p>
      {/* 渲染结果: &lt;img src=x onerror=alert("XSS")&gt; */}

      {/* ❌ 危险：dangerouslySetInnerHTML 绕过转义 */}
      {/* 仅在内容完全受信任时使用（如 Markdown 渲染） */}
      <div dangerouslySetInnerHTML={{ __html: userInput }} />

      {/* ❌ 危险：直接操作 DOM */}
      {/* document.getElementById('output')!.innerHTML = userInput; */}
    </div>
  );
}

// ============ 安全渲染 HTML 内容 ============
// 当必须渲染 HTML 时，使用 DOMPurify 净化
import DOMPurify from 'dompurify';

interface SafeHTMLProps {
  html: string;
  tag?: keyof JSX.IntrinsicElements;
}

function SafeHTML({ html, tag: Tag = 'div' }: SafeHTMLProps) {
  const sanitized = DOMPurify.sanitize(html, {
    ALLOWED_TAGS: ['b', 'i', 'em', 'strong', 'a', 'p', 'br', 'ul', 'ol', 'li'],
    ALLOWED_ATTR: ['href', 'title', 'target'],
    ALLOW_DATA_ATTR: false,
  });

  return <Tag dangerouslySetInnerHTML={{ __html: sanitized }} />;
}

// ============ 安全处理 URL ============
function SafeLinks() {
  const userProvidedURL = 'javascript:alert("XSS")';

  // ❌ 危险：直接使用用户提供的 URL
  // <a href={userProvidedURL}>点击</a>

  // ✅ 安全：验证并净化 URL
  const sanitizedURL = DOMPurify.sanitize(userProvidedURL);

  // ✅ 更好的做法：限制协议
  const safeURL = userProvidedURL.startsWith('https://') ||
    userProvidedURL.startsWith('http://') ||
    userProvidedURL.startsWith('/')
    ? userProvidedURL
    : '#';

  return <a href={safeURL} rel="noopener noreferrer">安全链接</a>;
}

// ============ 避免危险模式 ============
function DangerousPatterns() {
  // ❌ eval — 永远不要使用
  // eval(userInput);

  // ❌ Function 构造函数
  // new Function('return ' + userInput)();

  // ❌ setTimeout/setInterval 字符串形式
  // setTimeout('doSomething()', 1000);

  // ❌ href 中使用 javascript: 协议
  // <a href={`javascript:${userInput}`}>链接</a>

  // ❌ 不安全的 JSONP
  // <script src={`https://api.example.com/data?callback=${userInput}`} />

  return null;
}
```

### 7.7.2 CSP（Content Security Policy）头部

```tsx
// ============ CSP 响应头配置 ============
// CSP 通过限制浏览器可加载的资源来源来防御 XSS 攻击

// 示例 CSP 头部（由后端或部署平台配置）
// Content-Security-Policy:
//   default-src 'self';
//   script-src 'self' 'nonce-{random}' https://apis.google.com;
//   style-src 'self' 'unsafe-inline';
//   img-src 'self' https://images.example.com data:;
//   font-src 'self' https://fonts.gstatic.com;
//   connect-src 'self' https://api.example.com https://*.sentry.io;
//   frame-src 'self' https://www.youtube.com;
//   form-action 'self';
//   base-uri 'self';
//   object-src 'none';
//   upgrade-insecure-requests;
//   report-uri /csp-report-endpoint;

// ============ Vercel 中配置 CSP ============
// vercel.json
{
  "headers": [
    {
      "source": "/(.*)",
      "headers": [
        {
          "key": "Content-Security-Policy",
          "value": "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; font-src 'self'; connect-src 'self' https://api.example.com; frame-ancestors 'none'; base-uri 'self'; form-action 'self';"
        },
        {
          "key": "X-Content-Type-Options",
          "value": "nosniff"
        },
        {
          "key": "X-Frame-Options",
          "value": "DENY"
        },
        {
          "key": "X-XSS-Protection",
          "value": "0"
        },
        {
          "key": "Referrer-Policy",
          "value": "strict-origin-when-cross-origin"
        },
        {
          "key": "Permissions-Policy",
          "value": "camera=(), microphone=(), geolocation=()"
        }
      ]
    }
  ]
}

// ============ Netlify 中配置 CSP ============
// netlify.toml
[[headers]]
  for = "/*"
  [headers.values]
    Content-Security-Policy = "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:;"
    X-Content-Type-Options = "nosniff"
    X-Frame-Options = "DENY"
    Referrer-Policy = "strict-origin-when-cross-origin"

// ============ CSP nonce 在 React 中的使用 ============
// 服务端渲染场景：为每个请求生成随机 nonce
// 在 HTML 模板中注入：
// <script nonce="{nonce}" src="/assets/main.js"></script>

// 客户端获取 nonce 并应用到动态样式
// const nonce = document.querySelector('meta[property="csp-nonce"]')?.getAttribute('content');
// <style nonce={nonce}>{dynamicCSS}</style>
```

### 7.7.3 依赖安全审计

```bash
# ============ npm audit — 内置依赖审计 ============
# 检查已知漏洞
npm audit

# 查看详细报告
npm audit --json

# 自动修复（不破坏性更新）
npm audit fix

# 强制修复（可能包含破坏性更新）
npm audit fix --force

# ============ pnpm audit ============
pnpm audit

# 查看特定级别的漏洞
pnpm audit --audit-level high

# ============ Snyk — 高级依赖安全扫描 ============
# 安装
npm install -g snyk

# 测试项目
snyk test

# 持续监控
snyk monitor

# 自动修复
snyk wizard

# ============ GitHub Dependabot ============
# .github/dependabot.yml
version: 2
updates:
  # npm 依赖
  - package-ecosystem: "npm"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
      time: "09:00"
      timezone: "Asia/Shanghai"
    # 最多同时打开 10 个 PR
    open-pull-requests-limit: 10
    # 版本更新策略
    versioning-strategy: "auto"
    # 标签
    labels:
      - "dependencies"
      - "security"
    # 分组更新
    groups:
      react:
        patterns:
          - "react"
          - "react-dom"
          - "@types/react"
          - "@types/react-dom"
      testing:
        patterns:
          - "vitest"
          - "@testing-library/*"
      linting:
        patterns:
          - "eslint"
          - "prettier"
          - "@typescript-eslint/*"

  # GitHub Actions
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
```

```tsx
// ============ 定期审计脚本（package.json） ============
// "scripts": {
//   "audit": "pnpm audit --audit-level moderate",
//   "audit:fix": "pnpm audit --fix",
//   "outdated": "pnpm outdated",
//   "depcheck": "npx depcheck"
// }

// ============ pre-commit 钩子中检查 ============
// .husky/pre-commit
// #!/bin/sh
// # 检查是否引入了已知漏洞的依赖
// pnpm audit --audit-level high
// if [ $? -ne 0 ]; then
//   echo "⚠️ 发现高危漏洞，请运行 pnpm audit fix"
//   exit 1
// fi

// ============ CI 中集成安全扫描 ============
// .github/workflows/ci.yml 中添加 job:
// security:
//   runs-on: ubuntu-latest
//   steps:
//     - uses: actions/checkout@v4
//     - uses: pnpm/action-setup@v4
//     - run: pnpm audit --audit-level high
//     - name: Snyk Security Scan
//       uses: snyk/actions/node@master
//       env:
//         SNYK_TOKEN: ${{ secrets.SNYK_TOKEN }}
```

## 7.8 本章小结

TypeScript、测试、调试、构建、CI/CD、监控和安全构成了现代 React 开发的完整工程化体系。

**关键要点回顾：**

1. **TypeScript 类型系统**：泛型 Props（`interface Props<T>`）让组件复用更灵活；事件类型（`ChangeEvent<T>`、`MouseEvent<T>`、`KeyboardEvent<T>`）覆盖所有交互场景；`RefObject` vs `MutableRefObject` 区分只读和可变 ref；React 19 中 ref 可直接作为 prop，不再需要 `forwardRef`；`ReactNode` 用于 children，`JSX.Element` 用于需要明确 JSX 元素的场景，避免使用 `FC` 类型
2. **测试体系**：Vitest + jsdom 提供完整的浏览器环境模拟；`getByText`（同步）/ `queryByText`（验证不存在）/ `findByText`（异步等待）覆盖不同查询场景；`userEvent` 比 `fireEvent` 更接近真实用户行为（`click`、`type`、`keyboard`、`tab`、`hover`、`upload`）；`renderHook` + `act` 测试自定义 Hook；`waitFor` 轮询异步断言
3. **调试工具链**：React DevTools Components 面板查看 props/state/hooks；Profiler 火焰图定位渲染瓶颈；`debugger` 语句和条件断点精确暂停；Source Maps 映射编译代码到源码
4. **构建工具**：Vite 开发模式使用 esbuild 预构建依赖 + 原生 ESM 实现秒级 HMR，生产构建使用 Rollup 进行优化打包；Webpack 通过 loader 链处理各类文件，`SplitChunksPlugin` 实现自动代码分割；新项目优先选择 Vite
5. **CI/CD 流水线**：GitHub Actions 实现 lint -> test -> build -> deploy 自动化流水线；Vercel/Netlify 连接 Git 仓库后自动部署，PR 自动生成预览环境
6. **监控体系**：Sentry ErrorBoundary 自动捕获 React 组件错误，Profiler 追踪性能，Release 关联 source map 定位问题版本；Web Vitals 监控 LCP（< 2.5s）、FID（< 100ms）、CLS（< 0.1）、INP（< 200ms）等核心指标
7. **安全防护**：JSX 默认转义防止 XSS 注入；`dangerouslySetInnerHTML` 必须配合 DOMPurify 净化；CSP 头部限制资源加载来源；`npm audit` / Snyk / Dependabot 持续扫描依赖漏洞
