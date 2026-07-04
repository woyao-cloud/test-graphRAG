# 第八章：React 19 生态系统实战指南

React 的核心库本身只负责 UI 层。在现代前端开发中，构建一个完整的应用还需要状态管理、路由、样式方案、表单处理、动画和数据获取等诸多环节的支持。React 生态系统中涌现了大量优秀的第三方库，它们各自解决了特定领域的问题，让开发者能够快速组合出高性能、可维护的应用。

本章将系统地介绍 React 生态中七个核心领域的主流工具，通过对比表格和 TypeScript 代码示例帮助你理解每个工具的设计哲学和最佳使用场景。无论你是从零开始搭建新项目，还是在已有项目中引入新能力，都可以将本章作为选型参考和快速上手指南。

---

## 8.1 状态管理方案对比

状态管理是 React 应用架构中最核心的决策之一。不同的状态类型——服务端缓存状态、UI 交互状态、全局共享状态、表单局部状态——适合不同的工具。下面这张表覆盖了当前生态中四种主流方案的核心差异。

### 8.1.1 方案概览

| 特性 | Zustand | Jotai | TanStack Query | Redux Toolkit |
|------|---------|-------|----------------|---------------|
| **核心理念** | 基于 hook 的轻量 store | 原子化（atomic）状态 | 服务端状态缓存 | 集中式 store + 切片 |
| **学习曲线** | 极低 | 低 | 中等 | 较高 |
| **bundle 大小** | ~1 KB | ~3 KB | ~12 KB | ~11 KB（含 RTK Query） |
| **TypeScript 支持** | 优秀 | 优秀 | 一流 | 良好 |
| **服务端状态** | 需手动处理 | 需手动处理 | 一等公民 | RTK Query 支持 |
| **中间件/插件** | persist, immer, devtools | 内置集成 | 缓存策略、乐观更新 | 内置 thunk + 自定义 middleware |
| **适用场景** | 中小型全局状态、客户端状态 | 细粒度派生状态、组件级状态 | 服务端数据获取与缓存 | 大型团队、复杂状态逻辑 |

### 8.1.2 Zustand：极简全局状态

Zustand 的设计哲学是"最小 API 表面积"。它的核心只有一个 `create` 函数，返回一个可在任何组件中使用的 hook。没有 Provider 包裹、没有 action type 常量、没有 reducer 模板代码。

```typescript
// stores/useCounterStore.ts
import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';

interface CounterState {
  count: number;
  step: number;
  increment: () => void;
  decrement: () => void;
  reset: () => void;
  setStep: (step: number) => void;
}

export const useCounterStore = create<CounterState>()(
  devtools(
    persist(
      (set, get) => ({
        count: 0,
        step: 1,
        increment: () => set((state) => ({ count: state.count + state.step })),
        decrement: () => set((state) => ({ count: state.count - state.step })),
        reset: () => set({ count: 0 }),
        setStep: (step: number) => set({ step }),
      }),
      { name: 'counter-storage' }
    ),
    { name: 'CounterStore' }
  )
);
```

Zustand 的 `subscribe` 方法允许在 React 组件外部监听状态变化，这在需要将状态同步到 localStorage、WebSocket 或非 React 代码时非常有用。

```typescript
// 在 React 外部订阅状态变化
const unsubscribe = useCounterStore.subscribe(
  (state) => console.log('Count changed to:', state.count)
);

// 获取当前快照（不触发重新渲染）
const currentCount = useCounterStore.getState().count;
```

在组件中使用时，支持选择器（selector）以避免不必要的重新渲染：

```typescript
// components/Counter.tsx
import { useCounterStore } from '../stores/useCounterStore';

export function Counter() {
  // 只订阅 count，step 变化不会触发重新渲染
  const count = useCounterStore((state) => state.count);
  const increment = useCounterStore((state) => state.increment);

  return (
    <div>
      <p>Count: {count}</p>
      <button onClick={increment}>+1</button>
    </div>
  );
}
```

Zustand 的中间件系统非常灵活，常见的组合模式包括 `devtools(persist(immer(...)))`。`immer` 中间件让你直接修改状态对象而无需手动展开：

```typescript
import { create } from 'zustand';
import { immer } from 'zustand/middleware/immer';

interface TodoState {
  todos: { id: string; text: string; done: boolean }[];
  toggleTodo: (id: string) => void;
}

export const useTodoStore = create<TodoState>()(
  immer((set) => ({
    todos: [],
    toggleTodo: (id) =>
      set((state) => {
        const todo = state.todos.find((t) => t.id === id);
        if (todo) todo.done = !todo.done;
      }),
  }))
);
```

### 8.1.3 Jotai：原子化状态

Jotai 借鉴了 Recoil 的原子模型，但 API 更加简洁。状态被拆分为最小的原子（atom），组件只订阅自己需要的原子。派生状态通过组合原子自动计算，无需手动管理依赖。

```typescript
// atoms/counterAtoms.ts
import { atom } from 'jotai';

// 基础原子
export const countAtom = atom<number>(0);
export const stepAtom = atom<number>(1);

// 派生原子：读取多个原子的值并计算
export const doubledCountAtom = atom((get) => get(countAtom) * 2);

// 可写派生原子：同时支持读和写
export const countWithStepAtom = atom(
  (get) => ({ count: get(countAtom), step: get(stepAtom) }),
  (get, set, action: 'increment' | 'decrement') => {
    const step = get(stepAtom);
    set(countAtom, (prev) => (action === 'increment' ? prev + step : prev - step));
  }
);
```

Jotai 的原子在组件中使用非常简单，与 `useState` 几乎一致：

```typescript
// components/JotaiCounter.tsx
import { useAtom, useAtomValue, useSetAtom } from 'jotai';
import { countAtom, doubledCountAtom, countWithStepAtom } from '../atoms/counterAtoms';

export function JotaiCounter() {
  const [count, setCount] = useAtom(countAtom);
  const doubled = useAtomValue(doubledCountAtom);
  const dispatch = useSetAtom(countWithStepAtom);

  return (
    <div>
      <p>Count: {count}</p>
      <p>Doubled: {doubled}</p>
      <button onClick={() => setCount((c) => c + 1)}>+1</button>
      <button onClick={() => dispatch('increment')}>+Step</button>
    </div>
  );
}
```

Jotai 特别适合处理需要细粒度更新的场景，比如大量单元格的表格编辑器、画布工具、或者表单中多个字段的联动逻辑。每个原子独立触发更新，避免了 Redux 风格的单 store 全局重渲染问题。

### 8.1.4 TanStack Query：服务端状态专家

TanStack Query（前身 React Query）将"服务端状态"视为一种本质上不同于客户端状态的数据。它帮你处理缓存、后台刷新、乐观更新、分页和无限滚动等复杂场景，而你只需描述数据如何获取。

```typescript
// hooks/useUsersQuery.ts
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';

interface User {
  id: number;
  name: string;
  email: string;
}

const fetchUsers = async (): Promise<User[]> => {
  const res = await fetch('/api/users');
  if (!res.ok) throw new Error('Failed to fetch users');
  return res.json();
};

const deleteUser = async (id: number): Promise<void> => {
  const res = await fetch(`/api/users/${id}`, { method: 'DELETE' });
  if (!res.ok) throw new Error('Failed to delete user');
};

export function useUsers() {
  return useQuery({
    queryKey: ['users'],
    queryFn: fetchUsers,
    staleTime: 30_000,        // 30 秒内视为新鲜，不重新请求
    refetchInterval: 60_000,  // 每 60 秒自动后台刷新
  });
}

export function useDeleteUser() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: deleteUser,
    onSuccess: () => {
      // 删除成功后，使缓存失效并触发重新获取
      queryClient.invalidateQueries({ queryKey: ['users'] });
    },
    // 乐观更新：先假设成功，失败后回滚
    onMutate: async (deletedId) => {
      await queryClient.cancelQueries({ queryKey: ['users'] });
      const previous = queryClient.getQueryData<User[]>(['users']);
      queryClient.setQueryData<User[]>(['users'], (old) =>
        old?.filter((u) => u.id !== deletedId) ?? []
      );
      return { previous };
    },
    onError: (_err, _id, context) => {
      queryClient.setQueryData(['users'], context?.previous);
    },
  });
}
```

在组件中使用：

```typescript
// components/UserList.tsx
import { useUsers, useDeleteUser } from '../hooks/useUsersQuery';

export function UserList() {
  const { data: users, isLoading, error } = useUsers();
  const deleteMutation = useDeleteUser();

  if (isLoading) return <div>Loading...</div>;
  if (error) return <div>Error: {(error as Error).message}</div>;

  return (
    <ul>
      {users?.map((user) => (
        <li key={user.id}>
          {user.name} ({user.email})
          <button onClick={() => deleteMutation.mutate(user.id)}>Delete</button>
        </li>
      ))}
    </ul>
  );
}
```

TanStack Query 的 `queryKey` 设计是整个缓存策略的核心。相同 key 的查询共享缓存，不同 key 的查询互不干扰。key 可以是任意可序列化的值，常见模式是用数组组合资源名和参数。

### 8.1.5 Redux Toolkit：传统企业级方案

Redux Toolkit (RTK) 是 Redux 官方推荐的现代写法。它通过 `createSlice` 将 action creators 和 reducer 合并在一起，通过 `createAsyncThunk` 处理异步逻辑，通过 RTK Query 提供与 TanStack Query 类似的服务端状态管理能力。

```typescript
// store/counterSlice.ts
import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';

interface CounterState {
  value: number;
  status: 'idle' | 'loading' | 'failed';
}

const initialState: CounterState = { value: 0, status: 'idle' };

// 模拟异步操作
export const fetchInitialCount = createAsyncThunk(
  'counter/fetchInitial',
  async () => {
    const res = await fetch('/api/counter');
    const data = await res.json();
    return data.value as number;
  }
);

const counterSlice = createSlice({
  name: 'counter',
  initialState,
  reducers: {
    increment: (state) => { state.value += 1; },
    decrement: (state) => { state.value -= 1; },
    incrementByAmount: (state, action: PayloadAction<number>) => {
      state.value += action.payload;
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchInitialCount.pending, (state) => {
        state.status = 'loading';
      })
      .addCase(fetchInitialCount.fulfilled, (state, action) => {
        state.status = 'idle';
        state.value = action.payload;
      })
      .addCase(fetchInitialCount.rejected, (state) => {
        state.status = 'failed';
      });
  },
});

export const { increment, decrement, incrementByAmount } = counterSlice.actions;
export default counterSlice.reducer;
```

RTK Query 内置于 Redux Toolkit，让你用声明式 API 定义 API endpoints：

```typescript
// store/api.ts
import { createApi, fetchBaseQuery } from '@reduxjs/toolkit/query/react';

export const api = createApi({
  reducerPath: 'api',
  baseQuery: fetchBaseQuery({ baseUrl: '/api' }),
  tagTypes: ['User'],
  endpoints: (builder) => ({
    getUsers: builder.query<User[], void>({
      query: () => '/users',
      providesTags: ['User'],
    }),
    addUser: builder.mutation<User, Partial<User>>({
      query: (body) => ({ url: '/users', method: 'POST', body }),
      invalidatesTags: ['User'], // 自动触发 getUsers 重新获取
    }),
  }),
});

export const { useGetUsersQuery, useAddUserMutation } = api;
```

**选型建议**：对于新项目，如果状态以服务端数据为主，用 TanStack Query + Zustand 的组合是最轻量的选择；如果团队已有 Redux 经验且项目规模较大，Redux Toolkit 是成熟的选择；如果需要细粒度的派生状态控制（如图形编辑器），Jotai 更合适。

---

## 8.2 路由方案

### 8.2.1 React Router v7

React Router v7 引入了 `createBrowserRouter` 和 `<RouterProvider>` 的新路由模式，支持数据加载（loader）、数据变更（action）和错误边界（errorElement），使得路由定义与数据获取紧密结合。

```typescript
// router.tsx
import {
  createBrowserRouter,
  RouterProvider,
  useLoaderData,
  useActionData,
  Form,
  Link,
} from 'react-router-dom';
import type { LoaderFunctionArgs, ActionFunctionArgs } from 'react-router-dom';

interface Post {
  id: string;
  title: string;
  content: string;
  author: string;
}

// --- Loader: 在路由渲染前获取数据 ---
async function postsLoader(): Promise<Post[]> {
  const res = await fetch('/api/posts');
  if (!res.ok) throw new Response('Not Found', { status: 404 });
  return res.json();
}

async function postLoader({ params }: LoaderFunctionArgs): Promise<Post> {
  const res = await fetch(`/api/posts/${params.id}`);
  if (!res.ok) throw new Response('Post not found', { status: 404 });
  return res.json();
}

// --- Action: 处理表单提交 ---
async function createPostAction({ request }: ActionFunctionArgs) {
  const formData = await request.formData();
  const post = Object.fromEntries(formData);
  const res = await fetch('/api/posts', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(post),
  });
  return res.json();
}

// --- 错误边界组件 ---
function PostError() {
  return (
    <div style={{ padding: '2rem', textAlign: 'center' }}>
      <h2>404 - 文章未找到</h2>
      <Link to="/">返回首页</Link>
    </div>
  );
}

// --- 页面组件 ---
function PostList() {
  const posts = useLoaderData() as Post[];
  return (
    <div>
      <h1>文章列表</h1>
      <ul>
        {posts.map((post) => (
          <li key={post.id}>
            <Link to={`/posts/${post.id}`}>{post.title}</Link>
            <span> — {post.author}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

function PostDetail() {
  const post = useLoaderData() as Post;
  return (
    <article>
      <h1>{post.title}</h1>
      <p>作者：{post.author}</p>
      <div>{post.content}</div>
      <Link to="/">← 返回列表</Link>
    </article>
  );
}

function CreatePost() {
  const actionData = useActionData();
  return (
    <Form method="post">
      <label>
        标题：<input name="title" required />
      </label>
      <label>
        内容：<textarea name="content" required />
      </label>
      <label>
        作者：<input name="author" required />
      </label>
      <button type="submit">发布</button>
      {actionData && <p>发布成功！</p>}
    </Form>
  );
}

// --- 路由定义 ---
const router = createBrowserRouter([
  {
    path: '/',
    element: <PostList />,
    loader: postsLoader,
  },
  {
    path: '/posts/:id',
    element: <PostDetail />,
    loader: postLoader,
    errorElement: <PostError />,
  },
  {
    path: '/create',
    element: <CreatePost />,
    action: createPostAction,
  },
]);

export function AppRouter() {
  return <RouterProvider router={router} />;
}
```

React Router v7 的 loader 和 action 机制将数据获取从组件生命周期中解耦，这意味着即使组件还未挂载，数据就已经开始加载，减少了 loading spinner 的出现频率。同时 `errorElement` 让你可以在路由级别声明错误处理，而非在每个组件中写 try-catch。

嵌套路由通过 `<Outlet />` 实现布局复用：

```typescript
// 布局路由示例
const router = createBrowserRouter([
  {
    path: '/',
    element: <RootLayout />,  // 包含 <Outlet />
    errorElement: <GlobalError />,
    children: [
      { index: true, element: <Home /> },
      {
        path: 'dashboard',
        element: <DashboardLayout />,
        children: [
          { index: true, element: <DashboardHome /> },
          { path: 'settings', element: <Settings /> },
        ],
      },
    ],
  },
]);
```

### 8.2.2 TanStack Router

TanStack Router 是 TanStack 生态中的类型安全路由方案。它的核心卖点是 **100% TypeScript 类型安全**——从路径参数到 search params 到 loader 数据，整个链路都有完整的类型推导。

```typescript
// routes/posts.tsx
import { createFileRoute, useNavigate } from '@tanstack/react-router';

// 定义 search params 的类型（完全类型安全）
type PostSearch = {
  page?: number;
  sort?: 'newest' | 'oldest';
};

export const Route = createFileRoute('/posts')({
  validateSearch: (search: Record<string, unknown>): PostSearch => ({
    page: search.page ? Number(search.page) : 1,
    sort: search.sort === 'oldest' ? 'oldest' : 'newest',
  }),
  loaderDeps: ({ search }) => ({ page: search.page, sort: search.sort }),
  loader: async ({ deps }) => {
    const res = await fetch(`/api/posts?page=${deps.page}&sort=${deps.sort}`);
    return res.json() as Promise<Post[]>;
  },
  component: PostList,
});

function PostList() {
  const posts = Route.useLoaderData();
  const navigate = useNavigate();
  const { page } = Route.useSearch();

  return (
    <div>
      <h1>文章列表</h1>
      {/* Link 组件也享受类型安全 */}
      <Link to="/posts/$postId" params={{ postId: '1' }}>
        第一篇文章
      </Link>
      <button onClick={() => navigate({ search: { page: page + 1 } })}>
        下一页
      </button>
      <ul>
        {posts.map((post) => (
          <li key={post.id}>{post.title}</li>
        ))}
      </ul>
    </div>
  );
}
```

TanStack Router 的独特优势在于如果你写错了 `params` 或 `search` 的字段名，TypeScript 会在编译期就报错，而不是运行时才发现。这对于大型项目来说价值巨大。

---

## 8.3 样式方案对比

CSS 在 React 生态中经历了从全局 CSS 到 CSS Modules、CSS-in-JS、再到原子化 CSS 和零运行时方案的演进。下面这张表覆盖了当前最主流的四种方案。

### 8.3.1 方案概览

| 特性 | Tailwind CSS | CSS Modules | styled-components | vanilla-extract |
|------|-------------|-------------|-------------------|-----------------|
| **范式** | 原子化/utility-first | 局部作用域 CSS | CSS-in-JS（运行时） | CSS-in-JS（零运行时） |
| **类型安全** | 否（但有 IDE 插件） | 否 | 否 | 是 |
| **运行时开销** | 几乎为零（JIT） | 零 | 有（样式注入） | 零 |
| **动态样式** | 通过 className 组合 | 通过 CSS 变量 | 原生支持 props | 通过 recipe/variants |
| **学习曲线** | 低（记忆 class 名） | 极低（标准 CSS） | 低 | 中等 |
| **SSR 支持** | 优秀 | 优秀 | 需要额外配置 | 原生支持 |
| **主题系统** | 配置文件 | 需手动实现 | ThemeProvider | createTheme |

### 8.3.2 Tailwind CSS：原子化 CSS

Tailwind 的设计哲学是"直接在 HTML 中构建设计"。它提供了一组精心设计的 utility class，你通过组合它们来构建界面，而不是写自定义 CSS。

```tsx
// 组件中的 Tailwind 使用
export function DashboardCard({ title, value, trend }: {
  title: string;
  value: number;
  trend: 'up' | 'down';
}) {
  return (
    <div className="rounded-xl border border-gray-200 bg-white p-6 shadow-sm
                    hover:shadow-md transition-shadow duration-200">
      <h3 className="text-sm font-medium text-gray-500">{title}</h3>
      <p className="mt-2 text-3xl font-bold text-gray-900">{value}</p>
      <span className={`mt-1 inline-block text-sm
        ${trend === 'up' ? 'text-green-600' : 'text-red-600'}`}>
        {trend === 'up' ? '↑' : '↓'} 12.5%
      </span>
    </div>
  );
}
```

当 utility class 过于冗长时，可以使用 `@apply` 指令将常用组合提取为组件 class：

```css
/* styles/globals.css */
@tailwind base;
@tailwind components;
@tailwind utilities;

@layer components {
  .btn-primary {
    @apply rounded-lg bg-indigo-600 px-4 py-2 text-sm font-semibold
           text-white hover:bg-indigo-500 focus:outline-none
           focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2
           disabled:opacity-50 disabled:cursor-not-allowed;
  }

  .card {
    @apply rounded-xl border border-gray-200 bg-white p-6 shadow-sm;
  }
}
```

Tailwind 的 JIT（Just-In-Time）引擎会扫描你的代码，只生成实际用到的 CSS 类。这意味着即使 Tailwind 定义了数千个 utility class，最终产出的 CSS 文件通常只有几 KB。结合 `clsx` 或 `cn` 工具函数可以优雅地处理条件 class：

```typescript
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

// 使用
<button className={cn(
  'btn-primary',
  isLoading && 'opacity-50 pointer-events-none',
  variant === 'outline' && 'bg-transparent border-indigo-600 text-indigo-600'
)} />
```

### 8.3.3 CSS Modules：局部作用域

CSS Modules 是最接近原生 CSS 的方案。每个 `.module.css` 文件中的 class 名在编译时会被自动转换为唯一标识符，从而实现样式隔离，无需 BEM 等命名约定。

```css
/* components/DashboardCard.module.css */
.card {
  border-radius: 12px;
  border: 1px solid #e5e7eb;
  background: white;
  padding: 24px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
  transition: box-shadow 0.2s;
}

.card:hover {
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.title {
  font-size: 14px;
  color: #6b7280;
}

.value {
  margin-top: 8px;
  font-size: 30px;
  font-weight: 700;
  color: #111827;
}

.trendUp {
  color: #16a34a;
}

.trendDown {
  color: #dc2626;
}

/* 通过 :global 声明全局样式（不受作用域影响） */
.card :global(.ant-btn) {
  margin-top: 12px;
}
```

```tsx
// components/DashboardCard.tsx
import styles from './DashboardCard.module.css';

interface Props {
  title: string;
  value: number;
  trend: 'up' | 'down';
}

export function DashboardCard({ title, value, trend }: Props) {
  return (
    <div className={styles.card}>
      <h3 className={styles.title}>{title}</h3>
      <p className={styles.value}>{value}</p>
      <span className={trend === 'up' ? styles.trendUp : styles.trendDown}>
        {trend === 'up' ? '↑' : '↓'} 12.5%
      </span>
    </div>
  );
}
```

CSS Modules 支持组合（composes），可以从其他模块或全局样式复用样式：

```css
.base {
  padding: 12px 24px;
  border-radius: 8px;
  font-weight: 600;
}

.primary {
  composes: base;
  background: #4f46e5;
  color: white;
}

.danger {
  composes: base;
  background: #dc2626;
  color: white;
}
```

### 8.3.4 styled-components：动态 CSS-in-JS

styled-components 通过 tagged template literal 将样式直接写在组件中。它的最大优势是动态样式——组件的 props 可以直接影响样式，无需维护 className 映射表。

```tsx
import styled, { ThemeProvider, keyframes } from 'styled-components';

// 动画定义
const fadeIn = keyframes`
  from { opacity: 0; transform: translateY(8px); }
  to   { opacity: 1; transform: translateY(0); }
`;

// 带动态 props 的样式组件
const Card = styled.div<{ $highlight?: boolean }>`
  background: ${({ theme, $highlight }) =>
    $highlight ? theme.colors.primary : theme.colors.surface};
  color: ${({ $highlight }) => ($highlight ? 'white' : '#333')};
  padding: 24px;
  border-radius: 12px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
  animation: ${fadeIn} 0.3s ease-out;
  transition: transform 0.2s;

  &:hover {
    transform: translateY(-2px);
  }
`;

const Title = styled.h3`
  font-size: 14px;
  font-weight: 500;
  margin: 0;
  opacity: 0.8;
`;

const Value = styled.p`
  font-size: 32px;
  font-weight: 700;
  margin: 8px 0 4px;
`;

// 主题系统
const theme = {
  colors: {
    primary: '#4f46e5',
    surface: '#ffffff',
    success: '#16a34a',
    danger: '#dc2626',
  },
  spacing: { sm: '8px', md: '16px', lg: '24px' },
};

// 使用
<ThemeProvider theme={theme}>
  <Card $highlight>
    <Title>总用户数</Title>
    <Value>12,847</Value>
  </Card>
</ThemeProvider>
```

styled-components 的缺点是运行时开销——样式在 JavaScript 执行时动态注入到 DOM 中，这在频繁渲染大量组件时可能成为性能瓶颈。此外，SSR 场景需要额外配置 `ServerStyleSheet`。

### 8.3.5 vanilla-extract：零运行时 + 类型安全

vanilla-extract 在构建时将所有样式编译为静态 CSS 文件，运行时没有任何 JavaScript 开销。同时它提供完整的 TypeScript 类型支持，是追求性能与类型安全的大型项目的理想选择。

```typescript
// styles/dashboard.css.ts
import { style, styleVariants, createTheme } from '@vanilla-extract/css';

// 基础样式（类型安全的 CSS 属性）
export const card = style({
  padding: '24px',
  borderRadius: '12px',
  border: '1px solid #e5e7eb',
  background: '#fff',
  boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
  transition: 'box-shadow 0.2s',
  ':hover': {
    boxShadow: '0 4px 12px rgba(0,0,0,0.12)',
  },
});

export const title = style({
  fontSize: '14px',
  color: '#6b7280',
  margin: 0,
});

export const value = style({
  fontSize: '32px',
  fontWeight: 700,
  color: '#111827',
  margin: '8px 0 0',
});

// 变体（variants）
export const trend = styleVariants({
  up: { color: '#16a34a' },
  down: { color: '#dc2626' },
  flat: { color: '#6b7280' },
});

// 主题系统
const [themeClass, vars] = createTheme({
  color: { primary: '#4f46e5', background: '#f9fafb' },
  space: { small: '8px', medium: '16px', large: '24px' },
});

// 响应式
export const grid = style({
  display: 'grid',
  gap: vars.space.medium,
  '@media': {
    'screen and (min-width: 768px)': {
      gridTemplateColumns: 'repeat(2, 1fr)',
    },
    'screen and (min-width: 1024px)': {
      gridTemplateColumns: 'repeat(3, 1fr)',
    },
  },
});
```

```tsx
// components/DashboardCard.tsx
import * as styles from './dashboard.css';

export function DashboardCard({ title, value, trend }: {
  title: string; value: number; trend: 'up' | 'down' | 'flat';
}) {
  return (
    <div className={styles.card}>
      <h3 className={styles.title}>{title}</h3>
      <p className={styles.value}>{value}</p>
      <span className={styles.trend[trend]}>
        {trend === 'up' ? '↑' : trend === 'down' ? '↓' : '→'}
      </span>
    </div>
  );
}
```

---

## 8.4 表单处理：React Hook Form + Zod

表单是 Web 应用中最常见的交互模式，也是复杂度容易被低估的地方。React Hook Form 通过非受控组件（uncontrolled）策略最小化重新渲染，而 Zod 提供声明式的 schema 校验，两者结合是目前 React 生态中最主流且性能最优的表单方案。

### 8.4.1 React Hook Form 基础

```tsx
import { useForm } from 'react-hook-form';

interface LoginForm {
  email: string;
  password: string;
  rememberMe: boolean;
}

export function LoginForm() {
  const {
    register,
    handleSubmit,
    formState: { errors, isSubmitting },
    watch,
  } = useForm<LoginForm>({
    defaultValues: { email: '', password: '', rememberMe: false },
  });

  // watch 实时监听字段值
  const watchEmail = watch('email');

  const onSubmit = async (data: LoginForm) => {
    await new Promise((resolve) => setTimeout(resolve, 1000));
    console.log('Submitting:', data);
  };

  return (
    <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
      <div>
        <label htmlFor="email">Email</label>
        <input
          id="email"
          type="email"
          {...register('email', {
            required: 'Email 不能为空',
            pattern: {
              value: /^[^\s@]+@[^\s@]+\.[^\s@]+$/,
              message: 'Email 格式不正确',
            },
          })}
        />
        {errors.email && <p className="text-red-500">{errors.email.message}</p>}
      </div>

      <div>
        <label htmlFor="password">密码</label>
        <input
          id="password"
          type="password"
          {...register('password', {
            required: '密码不能为空',
            minLength: { value: 8, message: '密码至少 8 位' },
          })}
        />
        {errors.password && <p className="text-red-500">{errors.password.message}</p>}
      </div>

      <div>
        <label>
          <input type="checkbox" {...register('rememberMe')} />
          记住我
        </label>
      </div>

      {watchEmail && <p>当前输入: {watchEmail}</p>}

      <button type="submit" disabled={isSubmitting}>
        {isSubmitting ? '登录中...' : '登录'}
      </button>
    </form>
  );
}
```

### 8.4.2 Zod 校验集成

将 Zod schema 与 React Hook Form 通过 `@hookform/resolvers` 集成，可以将校验逻辑从 JSX 中完全抽离：

```typescript
// schemas/userSchema.ts
import { z } from 'zod';

export const userSchema = z
  .object({
    username: z
      .string()
      .min(3, '用户名至少 3 个字符')
      .max(20, '用户名最多 20 个字符')
      .regex(/^[a-zA-Z0-9_]+$/, '用户名只能包含字母、数字和下划线'),
    email: z.string().email('请输入有效的 Email 地址'),
    password: z
      .string()
      .min(8, '密码至少 8 个字符')
      .regex(/[A-Z]/, '密码必须包含至少一个大写字母')
      .regex(/[0-9]/, '密码必须包含至少一个数字'),
    confirmPassword: z.string(),
    age: z
      .number({ invalid_type_error: '请输入数字' })
      .int('年龄必须为整数')
      .min(18, '必须年满 18 岁')
      .max(120, '请输入有效年龄'),
    role: z.enum(['admin', 'editor', 'viewer'], {
      errorMap: () => ({ message: '请选择有效角色' }),
    }),
    bio: z.string().max(500, '个人简介最多 500 字').optional(),
  })
  .refine((data) => data.password === data.confirmPassword, {
    message: '两次密码输入不一致',
    path: ['confirmPassword'], // 错误关联到 confirmPassword 字段
  });

export type UserFormData = z.infer<typeof userSchema>;
```

```tsx
// components/UserRegistrationForm.tsx
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { userSchema, type UserFormData } from '../schemas/userSchema';

export function UserRegistrationForm() {
  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm<UserFormData>({
    resolver: zodResolver(userSchema),
    defaultValues: {
      username: '',
      email: '',
      password: '',
      confirmPassword: '',
      age: 18,
      role: 'viewer',
      bio: '',
    },
  });

  const onSubmit = (data: UserFormData) => {
    // data 的类型完全由 Zod schema 推导
    console.log('Valid data:', data);
  };

  return (
    <form onSubmit={handleSubmit(onSubmit)}>
      <div>
        <label>用户名</label>
        <input {...register('username')} />
        {errors.username && <span>{errors.username.message}</span>}
      </div>

      <div>
        <label>Email</label>
        <input type="email" {...register('email')} />
        {errors.email && <span>{errors.email.message}</span>}
      </div>

      <div>
        <label>密码</label>
        <input type="password" {...register('password')} />
        {errors.password && <span>{errors.password.message}</span>}
      </div>

      <div>
        <label>确认密码</label>
        <input type="password" {...register('confirmPassword')} />
        {errors.confirmPassword && <span>{errors.confirmPassword.message}</span>}
      </div>

      <div>
        <label>年龄</label>
        <input type="number" {...register('age', { valueAsNumber: true })} />
        {errors.age && <span>{errors.age.message}</span>}
      </div>

      <div>
        <label>角色</label>
        <select {...register('role')}>
          <option value="viewer">Viewer</option>
          <option value="editor">Editor</option>
          <option value="admin">Admin</option>
        </select>
        {errors.role && <span>{errors.role.message}</span>}
      </div>

      <div>
        <label>个人简介（可选）</label>
        <textarea {...register('bio')} rows={3} />
        {errors.bio && <span>{errors.bio.message}</span>}
      </div>

      <button type="submit">注册</button>
    </form>
  );
}
```

Zod 的 `refine` 和 `superRefine` 方法让你可以定义跨字段的校验规则（如密码确认），而 `transform` 则可以在校验通过后对数据进行预处理。整个 schema 可以作为单一事实来源被前后端共享。

---

## 8.5 动画：Framer Motion 与 React Spring

### 8.5.1 Framer Motion

Framer Motion 是目前 React 生态中最流行的声明式动画库。它的核心是 `motion` 组件——你可以像使用原生 HTML 元素一样使用 `motion.div`、`motion.button` 等，通过 props 声明动画行为。

```tsx
import { motion, AnimatePresence, LayoutGroup } from 'framer-motion';
import { useState } from 'react';

// --- 基础动画 ---
export function FadeInBox() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: 'easeOut' }}
      style={{ padding: 24, background: '#4f46e5', color: 'white', borderRadius: 12 }}
    >
      淡入并上移
    </motion.div>
  );
}

// --- 变体（variants）：管理动画状态机 ---
const listVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: { staggerChildren: 0.08, delayChildren: 0.1 },
  },
};

const itemVariants = {
  hidden: { opacity: 0, x: -20 },
  visible: { opacity: 1, x: 0, transition: { duration: 0.3 } },
};

export function StaggeredList({ items }: { items: string[] }) {
  return (
    <motion.ul variants={listVariants} initial="hidden" animate="visible">
      {items.map((item, i) => (
        <motion.li key={i} variants={itemVariants}>
          {item}
        </motion.li>
      ))}
    </motion.ul>
  );
}

// --- AnimatePresence：处理元素离开动画 ---
export function ToggleContent() {
  const [visible, setVisible] = useState(true);

  return (
    <div>
      <button onClick={() => setVisible((v) => !v)}>Toggle</button>
      <AnimatePresence>
        {visible && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.3 }}
            style={{ overflow: 'hidden', background: '#f3f4f6', padding: 16, borderRadius: 8 }}
          >
            <p>这段内容会平滑地出现和消失</p>
            <p>AnimatePresence 会在元素从 React 树中移除前播放退出动画</p>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// --- Layout 动画：自动处理布局变化 ---
export function ReorderableList() {
  const [items, setItems] = useState(['React', 'Vue', 'Angular', 'Svelte']);

  const reorder = () => {
    setItems((prev) => {
      const next = [...prev];
      const [first] = next.splice(0, 1);
      next.push(first);
      return next;
    });
  };

  return (
    <LayoutGroup>
      <button onClick={reorder}>重新排序</button>
      <motion.ul layout>
        {items.map((item) => (
          <motion.li
            key={item}
            layout
            transition={{ type: 'spring', stiffness: 400, damping: 30 }}
            style={{ padding: '12px 16px', margin: '8px 0', background: '#e0e7ff', borderRadius: 8 }}
          >
            {item}
          </motion.li>
        ))}
      </motion.ul>
    </LayoutGroup>
  );
}

// --- 手势动画 ---
export function DraggableCard() {
  return (
    <motion.div
      drag
      dragConstraints={{ left: 0, right: 300, top: 0, bottom: 300 }}
      dragElastic={0.2}
      whileHover={{ scale: 1.05 }}
      whileTap={{ scale: 0.95 }}
      style={{
        width: 120, height: 120, background: '#ec4899',
        borderRadius: 16, cursor: 'grab',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        color: 'white', fontWeight: 600,
      }}
    >
      拖拽我
    </motion.div>
  );
}
```

Framer Motion 的 `layout` prop 特别强大——当你改变元素的 CSS layout 属性（如 flex-direction、justify-content）或元素的排序时，它会自动计算差值并播放平滑过渡动画，无需手动管理起始和结束位置。

### 8.5.2 React Spring

React Spring 采用基于弹簧物理模型的动画引擎，动画效果更加自然。它的 `useSpring` 和 `useTransition` hooks 提供了与 Framer Motion 不同的 API 风格。

```tsx
import { useSpring, useTransition, animated, useTrail } from '@react-spring/web';
import { useState } from 'react';

// --- 基础弹簧动画 ---
export function SpringCounter() {
  const [count, setCount] = useState(0);
  const { number } = useSpring({
    from: { number: 0 },
    number: count,
    config: { mass: 1, tension: 280, friction: 60 },
  });

  return (
    <div>
      <animated.h1 style={{ fontSize: 48 }}>
        {number.to((n) => Math.floor(n))}
      </animated.h1>
      <button onClick={() => setCount((c) => c + 1)}>+1</button>
    </div>
  );
}

// --- Trail 动画：列表项的依次出现 ---
export function TrailList({ items }: { items: string[] }) {
  const [show, setShow] = useState(false);
  const trail = useTrail(items.length, {
    config: { mass: 5, tension: 2000, friction: 200 },
    opacity: show ? 1 : 0,
    x: show ? 0 : 20,
    from: { opacity: 0, x: 20 },
  });

  return (
    <div>
      <button onClick={() => setShow((s) => !s)}>Toggle</button>
      {trail.map((style, i) => (
        <animated.div key={i} style={style}>
          {items[i]}
        </animated.div>
      ))}
    </div>
  );
}

// --- useTransition：元素增删动画 ---
export function TransitionList() {
  const [items, setItems] = useState<string[]>([]);
  const transitions = useTransition(items, {
    from: { opacity: 0, transform: 'translateX(-100%)' },
    enter: { opacity: 1, transform: 'translateX(0%)' },
    leave: { opacity: 0, transform: 'translateX(100%)' },
    config: { tension: 220, friction: 20 },
  });

  return (
    <div>
      <button onClick={() => setItems((prev) => [...prev, `Item ${prev.length + 1}`])}>
        添加
      </button>
      <button onClick={() => setItems((prev) => prev.slice(0, -1))}>
        移除
      </button>
      {transitions((style, item) => (
        <animated.div style={style}>{item}</animated.div>
      ))}
    </div>
  );
}
```

**选型建议**：Framer Motion 的 API 更加直观和声明式，社区更大，文档更丰富，适合大多数场景。React Spring 的弹簧物理模型在需要高度自然感的动画（如拖拽惯性、物理模拟）时更有优势，但 API 更复杂。

---

## 8.6 表格：TanStack Table

TanStack Table（前身 React Table v8）是一个无头（headless）表格库。它不渲染任何 DOM 元素，只提供表格逻辑——排序、过滤、分页、列可见性、虚拟化等——而 UI 完全由你控制。这意味着你可以用任何样式方案（Tailwind、CSS Modules、styled-components）来渲染表格。

### 8.6.1 核心功能

```tsx
import {
  useReactTable,
  getCoreRowModel,
  getSortedRowModel,
  getFilteredRowModel,
  getPaginationRowModel,
  flexRender,
  createColumnHelper,
  type SortingState,
  type ColumnFiltersState,
} from '@tanstack/react-table';
import { useState } from 'react';

interface Product {
  id: number;
  name: string;
  category: string;
  price: number;
  stock: number;
  status: 'active' | 'inactive';
}

// Column Helper 提供类型安全的列定义
const columnHelper = createColumnHelper<Product>();

const columns = [
  columnHelper.accessor('id', {
    header: 'ID',
    cell: (info) => info.getValue(),
    enableColumnFilter: false,
  }),
  columnHelper.accessor('name', {
    header: '产品名称',
    cell: (info) => (
      <span className="font-medium text-gray-900">{info.getValue()}</span>
    ),
  }),
  columnHelper.accessor('category', {
    header: '分类',
    cell: (info) => info.getValue(),
    filterFn: 'equalsString',
  }),
  columnHelper.accessor('price', {
    header: '价格',
    cell: (info) => `¥${info.getValue().toLocaleString()}`,
    sortingFn: 'alphanumeric',
  }),
  columnHelper.accessor('stock', {
    header: '库存',
    cell: (info) => {
      const value = info.getValue();
      return (
        <span className={value < 10 ? 'text-red-600 font-semibold' : ''}>
          {value}
        </span>
      );
    },
  }),
  columnHelper.accessor('status', {
    header: '状态',
    cell: (info) => (
      <span
        className={`inline-block rounded-full px-2 py-0.5 text-xs font-medium ${
          info.getValue() === 'active'
            ? 'bg-green-100 text-green-800'
            : 'bg-gray-100 text-gray-800'
        }`}
      >
        {info.getValue() === 'active' ? '在售' : '下架'}
      </span>
    ),
  }),
];

export function ProductTable({ data }: { data: Product[] }) {
  const [sorting, setSorting] = useState<SortingState>([]);
  const [columnFilters, setColumnFilters] = useState<ColumnFiltersState>([]);
  const [globalFilter, setGlobalFilter] = useState('');
  const [columnVisibility, setColumnVisibility] = useState({});

  const table = useReactTable({
    data,
    columns,
    state: { sorting, columnFilters, globalFilter, columnVisibility },
    onSortingChange: setSorting,
    onColumnFiltersChange: setColumnFilters,
    onGlobalFilterChange: setGlobalFilter,
    onColumnVisibilityChange: setColumnVisibility,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
    initialState: { pagination: { pageSize: 10 } },
  });

  return (
    <div>
      {/* 全局搜索 */}
      <div className="mb-4">
        <input
          type="text"
          value={globalFilter}
          onChange={(e) => setGlobalFilter(e.target.value)}
          placeholder="搜索所有列..."
          className="rounded-lg border px-4 py-2"
        />
      </div>

      {/* 列可见性控制 */}
      <div className="mb-4 flex flex-wrap gap-2">
        {table.getAllLeafColumns().map((column) => (
          <label key={column.id} className="flex items-center gap-1 text-sm">
            <input
              type="checkbox"
              checked={column.getIsVisible()}
              onChange={column.getToggleVisibilityHandler()}
            />
            {column.id}
          </label>
        ))}
      </div>

      {/* 表格 */}
      <table className="w-full border-collapse">
        <thead>
          {table.getHeaderGroups().map((headerGroup) => (
            <tr key={headerGroup.id}>
              {headerGroup.headers.map((header) => (
                <th
                  key={header.id}
                  onClick={header.column.getToggleSortingHandler()}
                  className="cursor-pointer border-b-2 border-gray-200 px-4 py-3 text-left text-sm font-semibold text-gray-600"
                >
                  {flexRender(header.column.columnDef.header, header.getContext())}
                  {{ asc: ' ▲', desc: ' ▼' }[header.column.getIsSorted() as string] ?? ''}
                </th>
              ))}
            </tr>
          ))}
        </thead>
        <tbody>
          {table.getRowModel().rows.map((row) => (
            <tr key={row.id} className="border-b border-gray-100 hover:bg-gray-50">
              {row.getVisibleCells().map((cell) => (
                <td key={cell.id} className="px-4 py-3 text-sm">
                  {flexRender(cell.column.columnDef.cell, cell.getContext())}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>

      {/* 分页控件 */}
      <div className="mt-4 flex items-center justify-between">
        <div className="text-sm text-gray-600">
          共 {table.getFilteredRowModel().rows.length} 条记录
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => table.previousPage()}
            disabled={!table.getCanPreviousPage()}
            className="rounded border px-3 py-1 text-sm disabled:opacity-50"
          >
            上一页
          </button>
          <span className="flex items-center text-sm">
            第 {table.getState().pagination.pageIndex + 1} / {table.getPageCount()} 页
          </span>
          <button
            onClick={() => table.nextPage()}
            disabled={!table.getCanNextPage()}
            className="rounded border px-3 py-1 text-sm disabled:opacity-50"
          >
            下一页
          </button>
        </div>
      </div>
    </div>
  );
}
```

### 8.6.2 虚拟化大列表

当表格有数千甚至数万行数据时，可以使用 `@tanstack/react-virtual` 进行虚拟化渲染，只渲染可视区域内的行：

```tsx
import { useReactTable, getCoreRowModel, flexRender } from '@tanstack/react-table';
import { useVirtualizer } from '@tanstack/react-virtual';
import { useRef } from 'react';

export function VirtualizedTable({ data, columns }: { data: Product[]; columns: any[] }) {
  const table = useReactTable({
    data,
    columns,
    getCoreRowModel: getCoreRowModel(),
  });

  const tableContainerRef = useRef<HTMLDivElement>(null);

  const rowVirtualizer = useVirtualizer({
    count: table.getRowModel().rows.length,
    getScrollElement: () => tableContainerRef.current,
    estimateSize: () => 48, // 每行预估高度 48px
    overscan: 10,           // 超出可视区域的预渲染行数
  });

  const { rows } = table.getRowModel();
  const virtualRows = rowVirtualizer.getVirtualItems();
  const totalSize = rowVirtualizer.getTotalSize();
  const paddingTop = virtualRows.length > 0 ? virtualRows[0].start : 0;
  const paddingBottom =
    virtualRows.length > 0
      ? totalSize - (virtualRows[virtualRows.length - 1]?.end ?? 0)
      : 0;

  return (
    <div ref={tableContainerRef} style={{ height: '600px', overflow: 'auto' }}>
      <table style={{ width: '100%' }}>
        <thead>
          {table.getHeaderGroups().map((headerGroup) => (
            <tr key={headerGroup.id}>
              {headerGroup.headers.map((header) => (
                <th key={header.id} style={{ textAlign: 'left', padding: '8px 16px' }}>
                  {flexRender(header.column.columnDef.header, header.getContext())}
                </th>
              ))}
            </tr>
          ))}
        </thead>
        <tbody>
          {paddingTop > 0 && <tr><td style={{ height: paddingTop }} /></tr>}
          {virtualRows.map((virtualRow) => {
            const row = rows[virtualRow.index];
            return (
              <tr key={row.id}>
                {row.getVisibleCells().map((cell) => (
                  <td key={cell.id} style={{ padding: '8px 16px' }}>
                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
                  </td>
                ))}
              </tr>
            );
          })}
          {paddingBottom > 0 && <tr><td style={{ height: paddingBottom }} /></tr>}
        </tbody>
      </table>
    </div>
  );
}
```

TanStack Table 的无头架构意味着你拥有完全的 UI 控制权——同一套表格逻辑可以驱动完全不同的视觉呈现（表格、卡片列表、看板视图）。列定义中的 `cell` 渲染函数可以返回任何 React 元素，包括按钮、图片、进度条等。

---

## 8.7 数据获取：SWR 与 TanStack Query

虽然 TanStack Query 已在 8.1 节介绍过，但这里将其与 SWR 进行更细致的对比，并展示两者在高级场景中的用法。

### 8.7.1 方案对比

| 特性 | SWR | TanStack Query |
|------|-----|----------------|
| **bundle 大小** | ~5 KB | ~12 KB |
| **API 设计** | 极简（`useSWR`） | 功能全面 |
| **缓存策略** | stale-while-revalidate | 可配置（staleTime, gcTime 等） |
| **Mutation** | `useSWRMutation` | `useMutation`（更成熟） |
| **乐观更新** | 支持（手动） | 内置 onMutate 回滚 |
| **DevTools** | 无官方工具 | 官方 DevTools |
| **并行查询** | 原生支持 | `useQueries` |
| **无限滚动** | `useSWRInfinite` | `useInfiniteQuery` |

### 8.7.2 SWR 核心用法

SWR 的名字来源于缓存策略 `stale-while-revalidate`——先返回缓存中的旧数据（stale），同时后台发起请求更新（revalidate），最后返回新数据。这套流程对用户完全透明。

```typescript
import useSWR, { useSWRConfig } from 'swr';
import useSWRMutation from 'swr/mutation';

const fetcher = async <T>(url: string): Promise<T> => {
  const res = await fetch(url);
  if (!res.ok) throw new Error('Network error');
  return res.json();
};

// --- 基础数据获取 ---
export function useUserProfile(userId: string) {
  const { data, error, isLoading, isValidating } = useSWR(
    userId ? `/api/users/${userId}` : null, // null = 不发起请求
    fetcher,
    {
      revalidateOnFocus: true,        // 窗口聚焦时重新验证
      revalidateOnReconnect: true,    // 网络重连时重新验证
      refreshInterval: 30_000,        // 30 秒轮询
      dedupingInterval: 2000,         // 2 秒内相同请求去重
      onError: (err) => console.error('Failed to load user:', err),
    }
  );

  return { user: data, error, isLoading, isValidating };
}

// --- Mutation ---
async function updateUser(
  url: string,
  { arg }: { arg: { name: string; email: string } }
) {
  const res = await fetch(url, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(arg),
  });
  return res.json();
}

export function useUpdateUser(userId: string) {
  const { trigger, isMutating } = useSWRMutation(
    `/api/users/${userId}`,
    updateUser
  );

  return {
    updateUser: trigger,
    isUpdating: isMutating,
  };
}

// --- 手动重新验证 ---
export function RefreshButton() {
  const { mutate } = useSWRConfig();

  return (
    <button onClick={() => mutate('/api/users')}>
      刷新用户列表
    </button>
  );
}
```

SWR 的 `mutate` 函数支持乐观更新——在服务器响应之前，先修改本地缓存让 UI 立即反映变化：

```typescript
import { useSWRConfig } from 'swr';

export function useOptimisticUpdate() {
  const { mutate } = useSWRConfig();

  const toggleLike = async (postId: string) => {
    // 乐观更新：先修改本地缓存
    mutate(
      `/api/posts/${postId}`,
      (post: Post | undefined) =>
        post ? { ...post, liked: !post.liked } : post,
      false // false = 不重新验证
    );

    // 发送真实请求
    await fetch(`/api/posts/${postId}/like`, { method: 'POST' });

    // 请求完成后用服务器数据覆盖
    mutate(`/api/posts/${postId}`);
  };

  return { toggleLike };
}
```

### 8.7.3 TanStack Query 高级用法

TanStack Query 的 `queryClient` 提供了更细粒度的缓存控制：

```typescript
import { QueryClient, useQueryClient } from '@tanstack/react-query';

// 全局 queryClient 配置
export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 60_000,          // 默认 1 分钟视为新鲜
      gcTime: 5 * 60_000,         // 5 分钟后垃圾回收
      retry: 3,                   // 失败后重试 3 次
      refetchOnWindowFocus: true,
    },
    mutations: {
      retry: 1,
    },
  },
});

// --- 预取数据（prefetch）---
export function usePrefetchPost(postId: string) {
  const queryClient = useQueryClient();

  return () => {
    queryClient.prefetchQuery({
      queryKey: ['posts', postId],
      queryFn: () => fetch(`/api/posts/${postId}`).then((r) => r.json()),
      staleTime: 60_000,
    });
  };
}

// 在链接 hover 时预取
// <Link onMouseEnter={prefetchPost}>查看文章</Link>

// --- 依赖查询 ---
export function usePostWithAuthor(postId: string) {
  // 先获取 post
  const postQuery = useQuery({
    queryKey: ['posts', postId],
    queryFn: () => fetch(`/api/posts/${postId}`).then((r) => r.json()),
  });

  // 当 post 加载完成后，再查询 author
  const authorQuery = useQuery({
    queryKey: ['users', postQuery.data?.authorId],
    queryFn: () =>
      fetch(`/api/users/${postQuery.data.authorId}`).then((r) => r.json()),
    enabled: !!postQuery.data?.authorId, // 条件启用
  });

  return {
    post: postQuery.data,
    author: authorQuery.data,
    isLoading: postQuery.isLoading || authorQuery.isLoading,
    error: postQuery.error ?? authorQuery.error,
  };
}

// --- 并行查询 ---
export function useMultipleUsers(userIds: string[]) {
  return useQueries({
    queries: userIds.map((id) => ({
      queryKey: ['users', id],
      queryFn: () => fetch(`/api/users/${id}`).then((r) => r.json()),
      staleTime: 60_000,
    })),
  });
}
```

TanStack Query 的 `queryKey` 设计是一个精妙的缓存分层机制。当你调用 `invalidateQueries({ queryKey: ['posts'] })` 时，所有以 `['posts']` 为前缀的查询（如 `['posts', '1']`、`['posts', 'list']`）都会失效并重新获取。这种层级结构让缓存失效变得声明式和可预测。

---

## 8.8 本章小结

本章覆盖了 React 19 生态系统中七个关键领域的主流工具。总结选型建议如下：

- **状态管理**：服务端状态用 TanStack Query，客户端全局状态用 Zustand，细粒度派生状态用 Jotai，大型团队考虑 Redux Toolkit。
- **路由**：一般项目用 React Router v7（loader/action 模式），需要极致类型安全的项目用 TanStack Router。
- **样式**：追求开发速度用 Tailwind CSS，偏好标准 CSS 用 CSS Modules，需要动态样式用 styled-components，需要零运行时 + 类型安全用 vanilla-extract。
- **表单**：React Hook Form + Zod 是当前最优组合，覆盖 90% 的表单场景。
- **动画**：Framer Motion 是默认选择，React Spring 适合需要物理级自然动画的场景。
- **表格**：TanStack Table 是唯一需要考虑的选项，其无头架构适用于任何样式体系。
- **数据获取**：轻量场景用 SWR，复杂缓存需求用 TanStack Query。

记住一个原则：**不要为了用工具而用工具**。每个库都有其设计权衡，选择最匹配你当前问题的那一个，而不是试图用一个库解决所有问题。React 生态的繁荣意味着你可以在不同领域选择最佳工具，然后用它们组合出最优的解决方案。
