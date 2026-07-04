# 第10章 React 19 高级进阶与实战

React 19 引入了一系列令人振奋的新特性，从根本上改变了我们构建应用的方式。从 Actions API 到 Server Components，从 use() hook 到并发模式的完善，本章将深入探讨这些高级特性，并提供可直接用于生产环境的实战代码。

## 10.1 Actions API

React 19 的 Actions 是一套全新的服务端交互范式，它将 form 提交、数据变更和异步状态管理统一到一个声明式 API 中。Actions 的核心价值在于：自动管理 pending 状态、内置 progressive enhancement 支持、以及无需手动处理 race condition。

### 10.1.1 useActionState — 声明式表单管理

`useActionState` 是 React 19 中最重要的新 Hook 之一。它接收一个 action 函数和初始状态，返回当前状态、包装后的 action 以及 pending 状态。

```tsx
// components/LoginForm.tsx
import { useActionState } from 'react';

interface LoginState {
  error: string | null;
  success: boolean;
  fieldErrors: Record<string, string>;
}

async function loginAction(prevState: LoginState, formData: FormData): Promise<LoginState> {
  'use server';

  const email = formData.get('email') as string;
  const password = formData.get('password') as string;

  // 服务端校验
  const fieldErrors: Record<string, string> = {};
  if (!email || !email.includes('@')) {
    fieldErrors.email = '请输入有效的邮箱地址';
  }
  if (!password || password.length < 6) {
    fieldErrors.password = '密码至少需要 6 个字符';
  }
  if (Object.keys(fieldErrors).length > 0) {
    return { error: null, success: false, fieldErrors };
  }

  // 模拟数据库操作
  try {
    // const user = await db.user.findUnique({ where: { email } });
    // if (!user || !(await bcrypt.compare(password, user.password))) {
    //   return { error: '邮箱或密码错误', success: false, fieldErrors: {} };
    // }
    // await createSession(user.id);
    return { error: null, success: true, fieldErrors: {} };
  } catch {
    return { error: '登录失败，请稍后重试', success: false, fieldErrors: {} };
  }
}

export function LoginForm() {
  const [state, formAction, isPending] = useActionState(loginAction, {
    error: null,
    success: false,
    fieldErrors: {},
  });

  return (
    <form action={formAction} className="space-y-4 max-w-md mx-auto">
      <h2 className="text-2xl font-bold">用户登录</h2>

      {state.error && (
        <div role="alert" className="p-3 bg-red-50 text-red-700 rounded-md border border-red-200">
          {state.error}
        </div>
      )}

      {state.success && (
        <div className="p-3 bg-green-50 text-green-700 rounded-md border border-green-200">
          登录成功，正在跳转...
        </div>
      )}

      <div>
        <label htmlFor="email" className="block text-sm font-medium mb-1">邮箱</label>
        <input
          id="email"
          name="email"
          type="email"
          required
          className="w-full px-3 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
        {state.fieldErrors.email && (
          <p className="text-sm text-red-600 mt-1">{state.fieldErrors.email}</p>
        )}
      </div>

      <div>
        <label htmlFor="password" className="block text-sm font-medium mb-1">密码</label>
        <input
          id="password"
          name="password"
          type="password"
          required
          className="w-full px-3 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
        {state.fieldErrors.password && (
          <p className="text-sm text-red-600 mt-1">{state.fieldErrors.password}</p>
        )}
      </div>

      <button
        type="submit"
        disabled={isPending}
        className="w-full py-2 px-4 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
      >
        {isPending ? '登录中...' : '登录'}
      </button>
    </form>
  );
}
```

### 10.1.2 form action 属性

React 19 将 HTML form 的 `action` 属性提升为一等公民。任何传递给 `action` 的函数都会自动被视为 Server Action 或客户端 Action，并获得 pending 管理。

```tsx
// components/NewsletterForm.tsx
import { useActionState } from 'react';

async function subscribeAction(prevState: { message: string }, formData: FormData) {
  'use server';

  const email = formData.get('email') as string;

  // 在实际项目中，这里会调用邮件服务 API
  // await mailchimp.subscribe(email);

  return { message: `订阅成功！确认邮件已发送至 ${email}` };
}

export function NewsletterForm() {
  const [state, formAction, isPending] = useActionState(subscribeAction, {
    message: '',
  });

  return (
    <section className="bg-gray-100 p-6 rounded-lg">
      <h3 className="text-lg font-semibold mb-2">订阅我们的 Newsletter</h3>
      <form action={formAction} className="flex gap-2">
        <input
          name="email"
          type="email"
          placeholder="your@email.com"
          required
          className="flex-1 px-3 py-2 border rounded-md"
        />
        <button
          type="submit"
          disabled={isPending}
          className="px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50"
        >
          {isPending ? '订阅中...' : '订阅'}
        </button>
      </form>
      {state.message && (
        <p className="mt-2 text-green-600">{state.message}</p>
      )}
    </section>
  );
}
```

### 10.1.3 FormData API 与 Progressive Enhancement

Progressive Enhancement（渐进增强）意味着表单在没有 JavaScript 的环境下仍能正常工作。React 19 的 Actions 天然支持这一点——`action` 属性直接映射到 HTML 原生行为。

```tsx
// components/ContactForm.tsx
import { useActionState } from 'react';

interface ContactState {
  submitted: boolean;
  errors: string[];
}

async function contactAction(prevState: ContactState, formData: FormData): Promise<ContactState> {
  'use server';

  const entries = Object.fromEntries(formData.entries());
  const errors: string[] = [];

  if (!entries.name) errors.push('姓名为必填项');
  if (!entries.email) errors.push('邮箱为必填项');
  if (!entries.message) errors.push('留言内容为必填项');

  if (errors.length > 0) {
    return { submitted: false, errors };
  }

  // 存储到数据库
  // await db.contact.create({ data: entries });

  return { submitted: true, errors: [] };
}

export function ContactForm() {
  const [state, formAction, isPending] = useActionState(contactAction, {
    submitted: false,
    errors: [],
  });

  if (state.submitted) {
    return (
      <div className="p-8 text-center bg-green-50 rounded-lg">
        <h3 className="text-xl font-bold text-green-700">感谢您的留言！</h3>
        <p className="text-green-600 mt-2">我们会尽快回复您。</p>
      </div>
    );
  }

  return (
    <form action={formAction} className="space-y-4 max-w-lg">
      {state.errors.length > 0 && (
        <ul className="p-3 bg-red-50 text-red-700 rounded-md list-disc list-inside">
          {state.errors.map((err, i) => (
            <li key={i}>{err}</li>
          ))}
        </ul>
      )}

      <div>
        <label htmlFor="name" className="block text-sm font-medium">姓名</label>
        <input id="name" name="name" type="text" className="w-full mt-1 px-3 py-2 border rounded-md" />
      </div>

      <div>
        <label htmlFor="email" className="block text-sm font-medium">邮箱</label>
        <input id="email" name="email" type="email" className="w-full mt-1 px-3 py-2 border rounded-md" />
      </div>

      <div>
        <label htmlFor="message" className="block text-sm font-medium">留言</label>
        <textarea id="message" name="message" rows={5} className="w-full mt-1 px-3 py-2 border rounded-md" />
      </div>

      <button
        type="submit"
        disabled={isPending}
        className="px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50"
      >
        {isPending ? '提交中...' : '提交留言'}
      </button>
    </form>
  );
}
```

### 10.1.4 手动 FormData 操作

除了依赖 form 的 action 属性，你还可以在任何事件处理函数中手动构造和提交 FormData。

```tsx
// components/BulkUploadForm.tsx
import { useState, useRef, type FormEvent } from 'react';

export function BulkUploadForm() {
  const [files, setFiles] = useState<File[]>([]);
  const [uploading, setUploading] = useState(false);
  const formRef = useRef<HTMLFormElement>(null);

  async function handleSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    if (!formRef.current) return;

    const formData = new FormData(formRef.current);

    // 手动追加额外数据
    formData.append('timestamp', new Date().toISOString());
    formData.append('userId', 'current-user-id');

    // 添加自定义元数据
    const metadata = {
      source: 'web',
      version: '2.0',
      tags: ['bulk-upload', 'manual'],
    };
    formData.append('metadata', JSON.stringify(metadata));

    setUploading(true);
    try {
      // const result = await fetch('/api/upload', { method: 'POST', body: formData });
      await new Promise(resolve => setTimeout(resolve, 2000)); // 模拟上传
      setFiles([]);
    } finally {
      setUploading(false);
    }
  }

  return (
    <form ref={formRef} onSubmit={handleSubmit} className="space-y-4">
      <div>
        <label className="block text-sm font-medium">选择文件（支持多选）</label>
        <input
          name="files"
          type="file"
          multiple
          onChange={(e) => setFiles(Array.from(e.target.files ?? []))}
          className="mt-1 block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
        />
      </div>

      {files.length > 0 && (
        <ul className="text-sm text-gray-600">
          {files.map((f, i) => (
            <li key={i}>{f.name} ({(f.size / 1024).toFixed(1)} KB)</li>
          ))}
        </ul>
      )}

      <button
        type="submit"
        disabled={uploading || files.length === 0}
        className="px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50"
      >
        {uploading ? '上传中...' : '开始上传'}
      </button>
    </form>
  );
}
```

## 10.2 use() Hook — 在渲染中读取异步资源

`use()` 是 React 19 中新增的一个特殊 Hook（注意它没有遵循 `useXxx` 命名惯例的小写开头）。它的独特之处在于：可以在条件语句和循环中调用，可以读取 Promise 和 Context，并且与 Suspense 深度集成。

### 10.2.1 读取 Promise — 渲染时获取数据

`use()` 允许你直接在组件渲染期间 "unwrap" 一个 Promise。当 Promise 尚未 resolve 时，React 会自动暂停该组件的渲染，并触发最近的 Suspense fallback。

```tsx
// components/UserProfile.tsx
import { use, Suspense } from 'react';

interface User {
  id: number;
  name: string;
  email: string;
  avatar: string;
  bio: string;
}

async function fetchUser(id: number): Promise<User> {
  const res = await fetch(`https://jsonplaceholder.typicode.com/users/${id}`);
  if (!res.ok) throw new Error('Failed to fetch user');
  return res.json();
}

// 缓存 Promise，避免重复请求
const promiseCache = new Map<string, Promise<unknown>>();

function fetchUserCached(id: number): Promise<User> {
  const key = `user-${id}`;
  if (!promiseCache.has(key)) {
    promiseCache.set(key, fetchUser(id));
  }
  return promiseCache.get(key) as Promise<User>;
}

function UserProfileContent({ userId }: { userId: number }) {
  // use() 在渲染时读取 Promise —— 这是 React 19 的核心能力
  const user = use(fetchUserCached(userId));

  return (
    <div className="p-6 bg-white rounded-lg shadow">
      <div className="flex items-center gap-4">
        <img src={user.avatar} alt={user.name} className="w-16 h-16 rounded-full" />
        <div>
          <h2 className="text-xl font-bold">{user.name}</h2>
          <p className="text-gray-600">{user.email}</p>
        </div>
      </div>
      <p className="mt-4 text-gray-700">{user.bio}</p>
    </div>
  );
}

function UserProfileSkeleton() {
  return (
    <div className="p-6 bg-white rounded-lg shadow animate-pulse">
      <div className="flex items-center gap-4">
        <div className="w-16 h-16 bg-gray-200 rounded-full" />
        <div className="space-y-2">
          <div className="h-5 w-32 bg-gray-200 rounded" />
          <div className="h-4 w-48 bg-gray-200 rounded" />
        </div>
      </div>
      <div className="mt-4 space-y-2">
        <div className="h-4 w-full bg-gray-200 rounded" />
        <div className="h-4 w-3/4 bg-gray-200 rounded" />
      </div>
    </div>
  );
}

export function UserProfile({ userId }: { userId: number }) {
  return (
    <Suspense fallback={<UserProfileSkeleton />}>
      <UserProfileContent userId={userId} />
    </Suspense>
  );
}
```

### 10.2.2 读取 Context — 更灵活的上下文消费

`use()` 也可以替代 `useContext()` 来读取 Context。它的优势在于可以在条件语句中使用，这在需要可选 Context 的场景中非常有用。

```tsx
// contexts/ThemeContext.tsx
import { createContext, use, useState, type ReactNode } from 'react';

type Theme = 'light' | 'dark' | 'system';

interface ThemeContextValue {
  theme: Theme;
  resolvedTheme: 'light' | 'dark';
  setTheme: (t: Theme) => void;
}

const ThemeContext = createContext<ThemeContextValue | null>(null);

function getSystemTheme(): 'light' | 'dark' {
  if (typeof window === 'undefined') return 'light';
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [theme, setTheme] = useState<Theme>('system');
  const resolvedTheme = theme === 'system' ? getSystemTheme() : theme;

  return (
    <ThemeContext.Provider value={{ theme, resolvedTheme, setTheme }}>
      {children}
    </ThemeContext.Provider>
  );
}

// 使用 use() 读取 Context — 可以在条件语句中调用
export function useOptionalTheme(): ThemeContextValue | null {
  // use() 没有 "rules of hooks" 限制，可以在条件中调用
  return use(ThemeContext);
}

export function useRequiredTheme(): ThemeContextValue {
  const ctx = use(ThemeContext);
  if (!ctx) {
    throw new Error('useRequiredTheme must be used within ThemeProvider');
  }
  return ctx;
}

// components/ThemedCard.tsx
export function ThemedCard({ children }: { children: ReactNode }) {
  // 在条件语句中使用 use() — useContext 做不到这一点
  const themeCtx = useOptionalTheme();

  const isDark = themeCtx?.resolvedTheme === 'dark';

  return (
    <div
      className={`p-4 rounded-lg transition-colors ${
        isDark ? 'bg-gray-800 text-white' : 'bg-white text-gray-900'
      }`}
    >
      {children}
    </div>
  );
}
```

### 10.2.3 use() 与 Suspense 的深度集成

当多个组件都使用 `use()` 读取 Promise 时，React 会在 Suspense 边界内并行等待它们，而不是产生 waterfall。

```tsx
// components/Dashboard.tsx
import { use, Suspense } from 'react';

interface Stats { visitors: number; revenue: number; orders: number; }
interface Activity { id: number; action: string; timestamp: string; }
interface Alert { level: 'info' | 'warning' | 'error'; message: string; }

async function fetchStats(): Promise<Stats> {
  const res = await fetch('/api/stats');
  return res.json();
}

async function fetchActivity(): Promise<Activity[]> {
  const res = await fetch('/api/activity');
  return res.json();
}

async function fetchAlerts(): Promise<Alert[]> {
  const res = await fetch('/api/alerts');
  return res.json();
}

function StatsPanel() {
  const stats = use(fetchStats());
  return (
    <div className="grid grid-cols-3 gap-4">
      <div className="p-4 bg-blue-50 rounded-lg">
        <p className="text-sm text-blue-600">访客</p>
        <p className="text-2xl font-bold">{stats.visitors.toLocaleString()}</p>
      </div>
      <div className="p-4 bg-green-50 rounded-lg">
        <p className="text-sm text-green-600">收入</p>
        <p className="text-2xl font-bold">¥{stats.revenue.toLocaleString()}</p>
      </div>
      <div className="p-4 bg-purple-50 rounded-lg">
        <p className="text-sm text-purple-600">订单</p>
        <p className="text-2xl font-bold">{stats.orders.toLocaleString()}</p>
      </div>
    </div>
  );
}

function ActivityFeed() {
  const activities = use(fetchActivity());
  return (
    <div className="p-4 bg-white rounded-lg shadow">
      <h3 className="font-semibold mb-3">最近动态</h3>
      <ul className="space-y-2">
        {activities.slice(0, 5).map((a) => (
          <li key={a.id} className="text-sm text-gray-600 flex justify-between">
            <span>{a.action}</span>
            <span>{new Date(a.timestamp).toLocaleTimeString()}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

function AlertsPanel() {
  const alerts = use(fetchAlerts());
  if (alerts.length === 0) return null;

  return (
    <div className="space-y-2">
      {alerts.map((a, i) => (
        <div
          key={i}
          className={`p-3 rounded-md border text-sm ${
            a.level === 'error' ? 'bg-red-50 border-red-200 text-red-700' :
            a.level === 'warning' ? 'bg-yellow-50 border-yellow-200 text-yellow-700' :
            'bg-blue-50 border-blue-200 text-blue-700'
          }`}
        >
          {a.message}
        </div>
      ))}
    </div>
  );
}

export function Dashboard() {
  return (
    <div className="space-y-6 p-6">
      <h1 className="text-2xl font-bold">仪表盘</h1>

      {/* 三个组件并行加载，不产生 waterfall */}
      <Suspense fallback={<div className="h-24 animate-pulse bg-gray-100 rounded-lg" />}>
        <StatsPanel />
      </Suspense>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Suspense fallback={<div className="h-48 animate-pulse bg-gray-100 rounded-lg" />}>
          <ActivityFeed />
        </Suspense>
        <Suspense fallback={<div className="h-48 animate-pulse bg-gray-100 rounded-lg" />}>
          <AlertsPanel />
        </Suspense>
      </div>
    </div>
  );
}
```

## 10.3 React Server Components (RSC)

React Server Components 是 React 19 最具革命性的特性。它允许组件在服务端运行，直接访问数据库和文件系统，零 JavaScript 发送到客户端。

### 10.3.1 'use client' 指令 — 客户端边界

`'use client'` 指令标记一个组件及其所有导入为客户端组件。它是一个边界标记，告诉打包器（bundler）从这里开始，以下所有代码都在客户端运行。

```tsx
// components/ClientCounter.tsx
'use client';

import { useState } from 'react';

interface ClientCounterProps {
  initialCount: number;
  label?: string;
}

// 这是一个纯客户端组件 — 有交互，使用 useState
export function ClientCounter({ initialCount, label = '计数器' }: ClientCounterProps) {
  const [count, setCount] = useState(initialCount);
  const [history, setHistory] = useState<number[]>([initialCount]);

  function updateCount(delta: number) {
    setCount((prev) => {
      const next = prev + delta;
      setHistory((h) => [...h, next]);
      return next;
    });
  }

  return (
    <div className="p-4 border rounded-lg inline-block">
      <p className="text-sm text-gray-500 mb-2">{label}</p>
      <div className="flex items-center gap-4">
        <button
          onClick={() => updateCount(-1)}
          className="w-8 h-8 rounded-full bg-gray-100 hover:bg-gray-200 flex items-center justify-center"
        >
          -
        </button>
        <span className="text-2xl font-bold tabular-nums">{count}</span>
        <button
          onClick={() => updateCount(1)}
          className="w-8 h-8 rounded-full bg-gray-100 hover:bg-gray-200 flex items-center justify-center"
        >
          +
        </button>
      </div>
      {history.length > 1 && (
        <details className="mt-2">
          <summary className="text-xs text-gray-400 cursor-pointer">历史记录</summary>
          <div className="flex gap-1 mt-1">
            {history.map((h, i) => (
              <span key={i} className="text-xs px-1.5 py-0.5 bg-gray-100 rounded">
                {h}
              </span>
            ))}
          </div>
        </details>
      )}
    </div>
  );
}
```

### 10.3.2 'use server' 指令 — 服务端 Action

`'use server'` 可以标记整个文件或单个函数。标记文件时，该文件中的所有导出函数都是 Server Actions。标记函数时，只有该函数是 Server Action。

```tsx
// actions/posts.ts
'use server';

import { revalidatePath } from 'next/cache';
import { z } from 'zod';

// 在服务端直接访问数据库 — 无需构建 REST API
// import { db } from '@/lib/db';

const CreatePostSchema = z.object({
  title: z.string().min(3, '标题至少 3 个字符').max(200),
  content: z.string().min(10, '内容至少 10 个字符'),
  tags: z.string().optional(),
});

export interface CreatePostResult {
  success: boolean;
  postId?: string;
  error?: string;
  fieldErrors?: Record<string, string>;
}

export async function createPost(
  prevState: CreatePostResult,
  formData: FormData
): Promise<CreatePostResult> {
  const raw = {
    title: formData.get('title') as string,
    content: formData.get('content') as string,
    tags: formData.get('tags') as string,
  };

  const parsed = CreatePostSchema.safeParse(raw);
  if (!parsed.success) {
    const fieldErrors: Record<string, string> = {};
    parsed.error.issues.forEach((issue) => {
      const field = issue.path[0] as string;
      fieldErrors[field] = issue.message;
    });
    return { success: false, fieldErrors };
  }

  try {
    // 直接操作数据库 — 无 API 层
    // const post = await db.post.create({
    //   data: {
    //     title: parsed.data.title,
    //     content: parsed.data.content,
    //     tags: parsed.data.tags ? parsed.data.tags.split(',').map(t => t.trim()) : [],
    //     authorId: getCurrentUserId(),
    //   },
    // });

    // 重新验证缓存
    revalidatePath('/posts');

    return {
      success: true,
      // postId: post.id,
    };
  } catch (error) {
    return {
      success: false,
      error: error instanceof Error ? error.message : '创建失败',
    };
  }
}

export async function deletePost(postId: string): Promise<{ success: boolean; error?: string }> {
  try {
    // await db.post.delete({ where: { id: postId } });
    revalidatePath('/posts');
    return { success: true };
  } catch (error) {
    return {
      success: false,
      error: error instanceof Error ? error.message : '删除失败',
    };
  }
}
```

### 10.3.3 服务端数据获取 — 无需 API 层的直接数据库访问

这是 RSC 最强大的能力之一：服务端组件可以直接访问数据库，不需要构建和维护 REST/GraphQL API。

```tsx
// app/posts/page.tsx (App Router 中的服务端组件)
import { Suspense } from 'react';
import { ClientCounter } from '@/components/ClientCounter';

interface Post {
  id: string;
  title: string;
  excerpt: string;
  author: { name: string; avatar: string };
  publishedAt: string;
  readTime: number;
  tags: string[];
}

// 模拟数据库查询 — 实际项目中这会是真实的数据库调用
async function getPosts(): Promise<Post[]> {
  // const posts = await db.post.findMany({
  //   where: { published: true },
  //   orderBy: { publishedAt: 'desc' },
  //   include: { author: { select: { name: true, avatar: true } } },
  //   take: 20,
  // });
  // return posts;

  // 模拟延迟以展示 Suspense 效果
  await new Promise((resolve) => setTimeout(resolve, 1500));
  return [
    {
      id: '1',
      title: '深入理解 React 19 的 Actions API',
      excerpt: 'React 19 的 Actions 是一套全新的服务端交互范式...',
      author: { name: '张三', avatar: '/avatars/zhang.jpg' },
      publishedAt: '2025-06-15T08:00:00Z',
      readTime: 8,
      tags: ['React', 'Frontend'],
    },
    {
      id: '2',
      title: 'Server Components 实战指南',
      excerpt: '从零开始构建一个完整的 RSC 应用...',
      author: { name: '李四', avatar: '/avatars/li.jpg' },
      publishedAt: '2025-06-10T10:00:00Z',
      readTime: 12,
      tags: ['React', 'RSC', 'Next.js'],
    },
  ];
}

// PostCard 组件
function PostCard({ post }: { post: Post }) {
  return (
    <article className="p-6 bg-white rounded-lg shadow hover:shadow-md transition-shadow">
      <div className="flex items-center gap-3 mb-3">
        <img src={post.author.avatar} alt="" className="w-8 h-8 rounded-full" />
        <div>
          <span className="text-sm font-medium">{post.author.name}</span>
          <span className="text-xs text-gray-400 ml-2">
            {new Date(post.publishedAt).toLocaleDateString('zh-CN')}
          </span>
        </div>
      </div>
      <h2 className="text-xl font-bold mb-2">{post.title}</h2>
      <p className="text-gray-600 mb-3">{post.excerpt}</p>
      <div className="flex items-center gap-2">
        <span className="text-xs text-gray-400">{post.readTime} 分钟阅读</span>
        <div className="flex gap-1">
          {post.tags.map((tag) => (
            <span key={tag} className="text-xs px-2 py-0.5 bg-blue-50 text-blue-600 rounded-full">
              {tag}
            </span>
          ))}
        </div>
      </div>
    </article>
  );
}

// 服务端组件 — async 函数，直接在渲染时获取数据
export default async function PostsPage() {
  const posts = await getPosts();

  return (
    <main className="max-w-4xl mx-auto p-6">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold">文章列表</h1>
          <p className="text-gray-500 mt-1">共 {posts.length} 篇文章</p>
        </div>
        {/* 客户端交互组件嵌入服务端页面 */}
        <ClientCounter initialCount={posts.length} label="浏览计数" />
      </div>

      <div className="grid gap-6">
        {posts.map((post) => (
          <PostCard key={post.id} post={post} />
        ))}
      </div>
    </main>
  );
}
```

### 10.3.4 RSC 协议 — Streaming JSON

RSC 使用一种特殊的流式传输协议。服务端将组件树序列化为 JSON 流，客户端逐步解析并渲染。这不是传统的 HTML 渲染，而是一种混合策略：服务端组件输出被序列化，客户端组件占位符被保留。

```tsx
// RSC 数据流协议示意（实际由框架处理）

// 服务端输出的 RSC 流大致结构：
// M1:{"id":"./components/PostCard.tsx","chunks":["default"]}
// J0:["$","div",null,{"className":"grid gap-6","children":[
//   ["$","@1",null,{"post":{"id":"1","title":"..."}}],
//   ["$","@1",null,{"post":{"id":"2","title":"..."}}]
// ]}]

// 客户端组件占位符：
// M5:{"id":"./components/ClientCounter.tsx","name":"default","chunks":[]}
// J2:["$","@5",null,{"initialCount":2,"label":"浏览计数"}]

// 这表示 ClientCounter 的 JavaScript bundle 需要被下载，
// 然后客户端会 hydrate 这个交互组件
```

## 10.4 Server Actions — 从客户端到服务端的 Mutation

Server Actions 是 RSC 架构中处理数据变更的机制。它们让客户端组件能够调用服务端函数，就像调用本地函数一样。

### 10.4.1 基本 Server Action 调用

```tsx
// actions/comments.ts
'use server';

import { revalidatePath, revalidateTag } from 'next/cache';

export async function addComment(postId: string, content: string) {
  // 验证用户身份
  // const session = await getSession();
  // if (!session) throw new Error('请先登录');

  // 写入数据库
  // await db.comment.create({
  //   data: { postId, content, authorId: session.userId },
  // });

  // 重新验证 — 两种方式
  revalidatePath(`/posts/${postId}`);   // 按路径
  revalidateTag(`post-${postId}`);       // 按标签

  return { success: true };
}

// components/CommentForm.tsx
'use client';

import { useActionState, useRef } from 'react';
import { addComment } from '@/actions/comments';

export function CommentForm({ postId }: { postId: string }) {
  const formRef = useRef<HTMLFormElement>(null);

  async function handleAction(prevState: { success: boolean; error?: string }, formData: FormData) {
    const content = formData.get('content') as string;
    if (!content || content.trim().length < 3) {
      return { success: false, error: '评论至少需要 3 个字符' };
    }

    try {
      await addComment(postId, content);
      formRef.current?.reset();
      return { success: true };
    } catch (error) {
      return { success: false, error: error instanceof Error ? error.message : '评论失败' };
    }
  }

  const [state, formAction, isPending] = useActionState(handleAction, { success: false });

  return (
    <form ref={formRef} action={formAction} className="space-y-3">
      <textarea
        name="content"
        rows={3}
        placeholder="写下你的评论..."
        className="w-full px-3 py-2 border rounded-md resize-y"
      />
      {state.error && (
        <p className="text-sm text-red-600">{state.error}</p>
      )}
      <button
        type="submit"
        disabled={isPending}
        className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 text-sm"
      >
        {isPending ? '发布中...' : '发布评论'}
      </button>
    </form>
  );
}
```

### 10.4.2 revalidatePath 与 revalidateTag

React 的缓存策略基于路径和标签两个维度。理解它们的区别对正确使用 Server Actions 至关重要。

```tsx
// actions/revalidation-examples.ts
'use server';

import { revalidatePath, revalidateTag } from 'next/cache';

// revalidatePath — 基于 URL 路径的重新验证
export async function updateProfile(userId: string, data: FormData) {
  // await db.user.update({ where: { id: userId }, data });

  revalidatePath('/profile');             // 仅重新验证 /profile 页面
  revalidatePath('/profile', 'layout');    // 重新验证 /profile 及其所有嵌套 layout
  revalidatePath('/profile', 'page');      // 仅重新验证 /profile 的 page（默认）
  revalidatePath('/dashboard/[id]', 'page'); // 动态路由
}

// revalidateTag — 基于标签的重新验证（更细粒度）
export async function updatePost(postId: string, data: FormData) {
  // await db.post.update({ where: { id: postId }, data });

  revalidateTag(`post-${postId}`);  // 只重新验证带有此标签的缓存
  revalidateTag('posts-list');      // 重新验证文章列表
}

// 在数据获取时设置标签
// app/posts/[id]/page.tsx
// export default async function PostPage({ params }: { params: { id: string } }) {
//   const post = await fetch(`/api/posts/${params.id}`, {
//     next: { tags: [`post-${params.id}`] },  // 设置标签
//   }).then(res => res.json());
//   // ...
// }
```

### 10.4.3 无 JavaScript 环境的 Progressive Enhancement

Server Actions 的最大优势之一是它们作为原生 HTML form action 工作，这意味着即使 JavaScript 加载失败或被禁用，表单仍然可以提交。

```tsx
// app/todo/page.tsx
import { addTodo, toggleTodo, deleteTodo } from '@/actions/todos';

interface Todo {
  id: string;
  title: string;
  completed: boolean;
}

async function getTodos(): Promise<Todo[]> {
  // const todos = await db.todo.findMany({ orderBy: { createdAt: 'desc' } });
  // return todos;
  return [
    { id: '1', title: '学习 React 19 Actions API', completed: false },
    { id: '2', title: '重构旧项目到 Server Components', completed: false },
    { id: '3', title: '编写单元测试', completed: true },
  ];
}

export default async function TodoPage() {
  const todos = await getTodos();

  return (
    <main className="max-w-lg mx-auto p-6">
      <h1 className="text-2xl font-bold mb-6">待办事项</h1>

      {/* 添加待办 — 纯 form action，无需 JavaScript */}
      <form action={addTodo} className="flex gap-2 mb-6">
        <input
          name="title"
          type="text"
          placeholder="添加新的待办事项..."
          required
          className="flex-1 px-3 py-2 border rounded-md"
        />
        <button
          type="submit"
          className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
        >
          添加
        </button>
      </form>

      {/* 待办列表 */}
      <ul className="space-y-2">
        {todos.map((todo) => (
          <li
            key={todo.id}
            className={`flex items-center gap-3 p-3 rounded-md border ${
              todo.completed ? 'bg-gray-50 border-gray-200' : 'bg-white'
            }`}
          >
            {/* 切换完成状态 — form action */}
            <form action={toggleTodo} className="flex items-center">
              <input type="hidden" name="id" value={todo.id} />
              <button type="submit" className="flex items-center gap-2">
                <span
                  className={`w-5 h-5 rounded border-2 flex items-center justify-center ${
                    todo.completed
                      ? 'bg-green-500 border-green-500 text-white'
                      : 'border-gray-300'
                  }`}
                >
                  {todo.completed && '✓'}
                </span>
              </button>
            </form>

            <span className={todo.completed ? 'line-through text-gray-400' : ''}>
              {todo.title}
            </span>

            {/* 删除 — form action */}
            <form action={deleteTodo} className="ml-auto">
              <input type="hidden" name="id" value={todo.id} />
              <button
                type="submit"
                className="text-sm text-red-500 hover:text-red-700"
              >
                删除
              </button>
            </form>
          </li>
        ))}
      </ul>

      {/* 说明：即使在禁用 JavaScript 的浏览器中，以上所有操作都能正常工作 */}
      <p className="mt-6 text-xs text-gray-400 text-center">
        本页面支持 Progressive Enhancement — 无需 JavaScript 即可正常使用
      </p>
    </main>
  );
}

// actions/todos.ts
'use server';

import { revalidatePath } from 'next/cache';

export async function addTodo(formData: FormData) {
  const title = formData.get('title') as string;
  // await db.todo.create({ data: { title } });
  revalidatePath('/todo');
}

export async function toggleTodo(formData: FormData) {
  const id = formData.get('id') as string;
  // const todo = await db.todo.findUnique({ where: { id } });
  // await db.todo.update({ where: { id }, data: { completed: !todo.completed } });
  revalidatePath('/todo');
}

export async function deleteTodo(formData: FormData) {
  const id = formData.get('id') as string;
  // await db.todo.delete({ where: { id } });
  revalidatePath('/todo');
}
```

## 10.5 useOptimistic — 乐观更新

`useOptimistic` 是 React 19 新增的 Hook，用于实现乐观 UI 更新。它的工作原理是：立即更新 UI（假设操作会成功），同时在后台发送请求；如果请求失败，自动回滚到之前的状态。

### 10.5.1 基本乐观更新模式

```tsx
// components/OptimisticTodoList.tsx
'use client';

import { useOptimistic, useState, useRef, startTransition } from 'react';
import type { FormEvent } from 'react';

interface TodoItem {
  id: number;
  text: string;
  completed: boolean;
  pending?: boolean;
}

// 模拟 API — 实际项目中替换为真实请求
async function addTodoToServer(text: string): Promise<TodoItem> {
  await new Promise((resolve) => setTimeout(resolve, 1000));
  if (Math.random() < 0.2) throw new Error('服务器错误'); // 20% 失败率用于演示回滚
  return { id: Date.now(), text, completed: false };
}

export function OptimisticTodoList({ initialTodos }: { initialTodos: TodoItem[] }) {
  const [todos, setTodos] = useState(initialTodos);
  const inputRef = useRef<HTMLInputElement>(null);

  // useOptimistic 接收实际状态和一个 reducer
  // 返回乐观状态和添加乐观更新的函数
  const [optimisticTodos, addOptimisticTodo] = useOptimistic(
    todos,
    (state: TodoItem[], newTodo: TodoItem) => [...state, newTodo]
  );

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    const text = inputRef.current?.value.trim();
    if (!text) return;

    const optimisticTodo: TodoItem = {
      id: -Date.now(), // 临时 ID
      text,
      completed: false,
      pending: true,
    };

    // 在 startTransition 中包裹，标记为低优先级更新
    startTransition(async () => {
      // 1. 立即添加到乐观状态
      addOptimisticTodo(optimisticTodo);

      // 2. 发送实际请求
      try {
        const realTodo = await addTodoToServer(text);
        // 3. 用真实数据替换乐观数据
        setTodos((prev) => [...prev.filter((t) => t.id !== optimisticTodo.id), realTodo]);
      } catch {
        // 4. 失败时回滚 — 从 todos 中移除乐观条目即可
        setTodos((prev) => prev.filter((t) => t.id !== optimisticTodo.id));
      }
    });

    if (inputRef.current) inputRef.current.value = '';
  }

  return (
    <div className="max-w-md mx-auto p-4">
      <form onSubmit={handleSubmit} className="flex gap-2 mb-4">
        <input
          ref={inputRef}
          type="text"
          placeholder="添加待办..."
          className="flex-1 px-3 py-2 border rounded-md"
        />
        <button
          type="submit"
          className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
        >
          添加
        </button>
      </form>

      <ul className="space-y-2">
        {optimisticTodos.map((todo) => (
          <li
            key={todo.id}
            className={`flex items-center gap-2 p-2 rounded ${
              todo.pending
                ? 'bg-blue-50 text-blue-700 opacity-60'
                : todo.completed
                  ? 'bg-gray-50 text-gray-400 line-through'
                  : 'bg-white'
            }`}
          >
            {todo.pending && (
              <span className="inline-block w-4 h-4 border-2 border-blue-400 border-t-transparent rounded-full animate-spin" />
            )}
            {todo.text}
          </li>
        ))}
      </ul>
    </div>
  );
}
```

### 10.5.2 乐观更新与表单集成

```tsx
// components/OptimisticCommentSection.tsx
'use client';

import { useOptimistic, useState, useRef, startTransition } from 'react';

interface Comment {
  id: number;
  author: string;
  content: string;
  createdAt: string;
  likes: number;
  likedByMe: boolean;
}

async function postComment(author: string, content: string): Promise<Comment> {
  await new Promise((r) => setTimeout(r, 800));
  return {
    id: Date.now(),
    author,
    content,
    createdAt: new Date().toISOString(),
    likes: 0,
    likedByMe: false,
  };
}

async function toggleLike(commentId: number, currentlyLiked: boolean): Promise<void> {
  await new Promise((r) => setTimeout(r, 300));
  if (Math.random() < 0.1) throw new Error('点赞失败');
}

export function OptimisticCommentSection({ initialComments }: { initialComments: Comment[] }) {
  const [comments, setComments] = useState(initialComments);
  const formRef = useRef<HTMLFormElement>(null);

  // 评论的乐观更新
  const [optimisticComments, addOptimisticComment] = useOptimistic(
    comments,
    (state, newComment: Comment) => [newComment, ...state]
  );

  // 点赞的乐观更新 — 使用不同的 reducer
  const [optimisticLikes, applyOptimisticLike] = useOptimistic(
    comments,
    (state, commentId: number) =>
      state.map((c) => {
        if (c.id !== commentId) return c;
        return {
          ...c,
          likes: c.likedByMe ? c.likes - 1 : c.likes + 1,
          likedByMe: !c.likedByMe,
        };
      })
  );

  async function handlePostComment(formData: FormData) {
    const author = formData.get('author') as string;
    const content = formData.get('content') as string;

    const optimisticComment: Comment = {
      id: -Date.now(),
      author,
      content,
      createdAt: new Date().toISOString(),
      likes: 0,
      likedByMe: false,
    };

    startTransition(async () => {
      addOptimisticComment(optimisticComment);
      try {
        const real = await postComment(author, content);
        setComments((prev) => [real, ...prev.filter((c) => c.id !== optimisticComment.id)]);
        formRef.current?.reset();
      } catch {
        setComments((prev) => prev.filter((c) => c.id !== optimisticComment.id));
      }
    });
  }

  async function handleLike(commentId: number, currentlyLiked: boolean) {
    startTransition(async () => {
      applyOptimisticLike(commentId);
      try {
        await toggleLike(commentId, currentlyLiked);
        // 成功 — 乐观状态已正确，更新真实状态
        setComments((prev) =>
          prev.map((c) => {
            if (c.id !== commentId) return c;
            return {
              ...c,
              likes: currentlyLiked ? c.likes - 1 : c.likes + 1,
              likedByMe: !currentlyLiked,
            };
          })
        );
      } catch {
        // 失败 — 回滚
        setComments((prev) => [...prev]);
      }
    });
  }

  return (
    <div className="max-w-2xl mx-auto">
      <form ref={formRef} action={handlePostComment} className="space-y-3 mb-8 p-4 bg-gray-50 rounded-lg">
        <input
          name="author"
          placeholder="你的名字"
          required
          className="w-full px-3 py-2 border rounded-md"
        />
        <textarea
          name="content"
          placeholder="写下你的评论..."
          rows={3}
          required
          className="w-full px-3 py-2 border rounded-md"
        />
        <button
          type="submit"
          className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
        >
          发表评论
        </button>
      </form>

      <div className="space-y-4">
        {optimisticComments.map((comment) => (
          <div
            key={comment.id}
            className={`p-4 rounded-lg border ${
              comment.id < 0 ? 'opacity-50 bg-blue-50' : 'bg-white'
            }`}
          >
            <div className="flex items-center justify-between mb-2">
              <span className="font-semibold">{comment.author}</span>
              <span className="text-xs text-gray-400">
                {new Date(comment.createdAt).toLocaleString()}
              </span>
            </div>
            <p className="text-gray-700">{comment.content}</p>
            <button
              onClick={() => handleLike(comment.id, comment.likedByMe)}
              className={`mt-2 text-sm flex items-center gap-1 ${
                comment.likedByMe ? 'text-red-500' : 'text-gray-400'
              } hover:text-red-500 transition-colors`}
            >
              {comment.likedByMe ? '♥' : '♡'} {comment.likes}
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}
```

## 10.6 Document Metadata — 组件内声明式 SEO

React 19 将 `<title>`、`<meta>`、`<link>` 等标签提升为内置支持。你可以直接在组件中声明文档元数据，无需 `react-helmet` 等第三方库。

### 10.6.1 基本元数据声明

```tsx
// app/blog/[slug]/page.tsx
import type { Metadata } from 'next';

interface Props {
  params: { slug: string };
}

// 静态元数据
export const metadata: Metadata = {
  title: '博客文章',
  description: '阅读我们的最新博客文章',
};

// 动态元数据 — 基于路由参数
export async function generateMetadata({ params }: Props): Promise<Metadata> {
  // const post = await getPost(params.slug);

  const post = {
    title: '深入理解 React 19 Server Components',
    description: '从原理到实战，全面掌握 RSC 的核心概念和最佳实践',
    publishedAt: '2025-06-15',
    author: '张三',
    image: '/images/react-19-rsc.png',
  };

  return {
    title: post.title,
    description: post.description,
    // Open Graph 标签 — 社交分享
    openGraph: {
      title: post.title,
      description: post.description,
      type: 'article',
      publishedTime: post.publishedAt,
      authors: [post.author],
      images: [
        {
          url: post.image,
          width: 1200,
          height: 630,
          alt: post.title,
        },
      ],
    },
    // Twitter 卡片
    twitter: {
      card: 'summary_large_image',
      title: post.title,
      description: post.description,
      images: [post.image],
    },
    // 其他元数据
    robots: {
      index: true,
      follow: true,
      googleBot: {
        index: true,
        follow: true,
        'max-video-preview': -1,
        'max-image-preview': 'large',
        'max-snippet': -1,
      },
    },
    alternates: {
      canonical: `https://example.com/blog/${params.slug}`,
      languages: {
        'en-US': `https://example.com/en/blog/${params.slug}`,
        'zh-CN': `https://example.com/zh/blog/${params.slug}`,
      },
    },
  };
}

export default function BlogPostPage({ params }: Props) {
  return (
    <article className="max-w-3xl mx-auto p-6">
      <h1 className="text-3xl font-bold">深入理解 React 19 Server Components</h1>
      {/* 文章内容 */}
    </article>
  );
}
```

### 10.6.2 组件内直接使用元数据标签

React 19 支持在任何组件（包括客户端组件）中直接渲染 `<title>` 和 `<meta>` 标签，React 会自动将它们提升到 `<head>` 中。

```tsx
// components/SEOWrapper.tsx
'use client';

import { useEffect, useState } from 'react';

interface SEOWrapperProps {
  title: string;
  description: string;
  keywords?: string[];
  ogImage?: string;
  noindex?: boolean;
}

// React 19 支持在组件中直接写 <title> 和 <meta>
// 它们会自动被 hoist 到 <head> 中
export function SEOWrapper({
  title,
  description,
  keywords,
  ogImage,
  noindex,
}: SEOWrapperProps) {
  const [pageUrl, setPageUrl] = useState('');

  useEffect(() => {
    setPageUrl(window.location.href);
  }, []);

  return (
    <>
      {/* 这些标签会被 React 自动提升到 document head */}
      <title>{title}</title>
      <meta name="description" content={description} />
      {keywords && <meta name="keywords" content={keywords.join(', ')} />}
      {noindex && <meta name="robots" content="noindex, nofollow" />}

      {/* Open Graph */}
      <meta property="og:title" content={title} />
      <meta property="og:description" content={description} />
      <meta property="og:url" content={pageUrl} />
      <meta property="og:type" content="website" />
      {ogImage && <meta property="og:image" content={ogImage} />}

      {/* Twitter */}
      <meta name="twitter:card" content="summary_large_image" />
      <meta name="twitter:title" content={title} />
      <meta name="twitter:description" content={description} />
      {ogImage && <meta name="twitter:image" content={ogImage} />}

      {/* JSON-LD 结构化数据 */}
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{
          __html: JSON.stringify({
            '@context': 'https://schema.org',
            '@type': 'Article',
            headline: title,
            description: description,
            url: pageUrl,
          }),
        }}
      />
    </>
  );
}

// 使用示例
export function ProductPage() {
  return (
    <>
      <SEOWrapper
        title="React 19 精通实战 — 限时优惠"
        description="全面掌握 React 19 新特性，从入门到企业级实战"
        keywords={['React', 'React 19', '前端开发', '教程']}
        ogImage="/images/book-cover.png"
      />
      <main>{/* 页面内容 */}</main>
    </>
  );
}
```

## 10.7 Asset Loading — 资源预加载与预初始化

React 19 提供了 `preload`、`preinit`、`preconnect`、`prefetchDNS` 等 API，让你能够以声明式的方式管理资源加载优先级。

### 10.7.1 preload — 预加载关键资源

`preload()` 告诉浏览器立即开始下载某个资源，因为它很快就会需要。适用于字体、关键 CSS、首屏图片等。

```tsx
// components/AssetPreloader.tsx
import { preload, preconnect, prefetchDNS } from 'react-dom';

export function AssetPreloader() {
  // DNS 预解析 — 最早阶段
  prefetchDNS('https://api.example.com');
  prefetchDNS('https://cdn.example.com');

  // 预连接 — 建立 TCP + TLS 连接
  preconnect('https://fonts.googleapis.com');
  preconnect('https://fonts.gstatic.com', { crossOrigin: 'anonymous' });

  // 预加载关键字体
  preload('/fonts/inter-var.woff2', { as: 'font', type: 'font/woff2', crossOrigin: 'anonymous' });
  preload('/fonts/jetbrains-mono.woff2', { as: 'font', type: 'font/woff2', crossOrigin: 'anonymous' });

  // 预加载首屏关键图片
  preload('/images/hero-banner.webp', { as: 'image', fetchPriority: 'high' });

  // 预加载关键 CSS
  preload('/styles/critical.css', { as: 'style' });

  // 预加载关键 JavaScript
  preload('/scripts/analytics.js', { as: 'script' });

  return null; // 这是一个纯副作用组件
}
```

### 10.7.2 preinit — 预初始化并执行

`preinit()` 不仅下载资源，还会立即解析和执行。适用于需要尽早执行的脚本或样式。

```tsx
// components/ScriptPreinitializer.tsx
import { preinit, preload } from 'react-dom';

export function ScriptPreinitializer() {
  // preinit — 下载、解析并执行脚本
  preinit('/scripts/theme-detector.js', {
    as: 'script',
    // 此脚本需要立即执行以设置主题，防止 FOUC (Flash of Unstyled Content)
  });

  // preinit 样式表 — 下载、解析并应用
  preinit('/styles/base.css', {
    as: 'style',
    precedence: 'high', // 设置优先级
  });

  // 与 preload 的区别：
  // preload: 只下载，不执行/应用
  // preinit: 下载并立即执行/应用

  return null;
}
```

### 10.7.3 资源加载策略对比

```tsx
// components/ResourceStrategy.tsx
import { preinit, preload, preconnect, prefetchDNS } from 'react-dom';

/**
 * 资源加载优先级层级（从早到晚）：
 *
 * 1. prefetchDNS     — 仅 DNS 解析
 * 2. preconnect      — DNS + TCP + TLS
 * 3. preinit         — 下载 + 执行/应用
 * 4. preload         — 仅下载
 */

export function FullResourceStrategy() {
  // 第1层：DNS 预解析（对用户可能访问的第三方域名）
  prefetchDNS('https://analytics.google.com');

  // 第2层：预连接（对确定需要的跨域资源）
  preconnect('https://images.unsplash.com');

  // 第3层：预初始化关键脚本/样式（防止 FOUC）
  preinit('/scripts/dark-mode.js', { as: 'script' });

  // 第4层：预加载关键资源（首屏需要但不立即执行的资源）
  preload('/fonts/inter-bold.woff2', { as: 'font', type: 'font/woff2', crossOrigin: 'anonymous' });
  preload('/videos/intro.mp4', { as: 'video' });
  preload('/models/scene.glb', { as: 'fetch', crossOrigin: 'anonymous' });

  return <main>{/* 应用内容 */}</main>;
}
```

## 10.8 Web Components 集成

React 19 大大改善了对 Custom Elements（Web Components）的支持。现在你可以直接将自定义元素作为 JSX 标签使用，React 会正确处理属性映射和事件绑定。

### 10.8.1 基础 Custom Element 集成

```tsx
// components/WebComponentIntegration.tsx
import { useRef, useEffect, useState } from 'react';

// 首先，定义一个 Web Component（通常在单独的文件中定义并注册）
// class MyRating extends HTMLElement { ... }
// customElements.define('my-rating', MyRating);

// 在 React 组件中直接使用 Web Component
export function RatingDemo() {
  const [rating, setRating] = useState(3);

  // 通过 ref 监听 Web Component 的自定义事件
  const ratingRef = useRef<HTMLElement>(null);

  useEffect(() => {
    const el = ratingRef.current;
    if (!el) return;

    function handleRatingChange(e: Event) {
      const customEvent = e as CustomEvent<{ value: number }>;
      setRating(customEvent.detail.value);
    }

    el.addEventListener('rating-change', handleRatingChange);
    return () => el.removeEventListener('rating-change', handleRatingChange);
  }, []);

  return (
    <div className="p-6 space-y-4">
      <h2 className="text-xl font-bold">Web Component 集成示例</h2>

      {/* React 19 原生支持 Custom Elements 作为 JSX */}
      <my-rating
        ref={ratingRef as React.Ref<HTMLElement>}
        value={rating}
        max={5}
        label="你的评价"
      />

      {/* 只读模式 */}
      <my-rating value={4} max={5} readonly label="平均评分" />

      <p className="text-sm text-gray-600">当前评分: {rating}</p>
    </div>
  );
}
```

### 10.8.2 复杂属性映射

```tsx
// components/ComplexWebComponent.tsx
import { useRef, useEffect } from 'react';

// 声明 Web Component 的 TypeScript 类型
declare global {
  namespace JSX {
    interface IntrinsicElements {
      'my-data-grid': MyDataGridAttributes;
      'my-chart': MyChartAttributes;
    }
  }
}

interface MyDataGridAttributes {
  data?: string;
  columns?: string;
  sortable?: string;
  pageSize?: number;
  ref?: React.Ref<HTMLElement>;
  className?: string;
  children?: React.ReactNode;
}

interface MyChartAttributes {
  type?: 'bar' | 'line' | 'pie';
  'data-url'?: string;
  width?: number;
  height?: number;
  ref?: React.Ref<HTMLElement>;
  className?: string;
}

// 工具函数：将复杂对象转为 Web Component 属性
function toAttribute(value: unknown): string {
  if (typeof value === 'string') return value;
  return JSON.stringify(value);
}

interface DataGridProps {
  rows: Record<string, unknown>[];
  columns: { key: string; label: string; sortable?: boolean }[];
  pageSize?: number;
}

export function DataGrid({ rows, columns, pageSize = 20 }: DataGridProps) {
  const gridRef = useRef<HTMLElement>(null);

  // 对于复杂对象属性，通过 DOM API 直接设置 property
  // （HTML 属性只能是字符串，但 DOM property 可以是任意类型）
  useEffect(() => {
    const el = gridRef.current;
    if (!el) return;

    // 设置 DOM property — 绕过 HTML 属性字符串限制
    (el as any).data = rows;
    (el as any).columns = columns;
    (el as any).pageSize = pageSize;

    // 调用 Web Component 的方法
    if (typeof (el as any).refresh === 'function') {
      (el as any).refresh();
    }
  }, [rows, columns, pageSize]);

  return (
    <my-data-grid
      ref={gridRef as React.Ref<HTMLElement>}
      className="w-full border rounded-lg"
      data={toAttribute(rows)}
      columns={toAttribute(columns)}
      sortable="true"
    />
  );
}

// 图表组件集成
export function ChartWidget({ type, dataUrl, width = 600, height = 400 }: {
  type: 'bar' | 'line' | 'pie';
  dataUrl: string;
  width?: number;
  height?: number;
}) {
  return (
    <my-chart
      type={type}
      data-url={dataUrl}
      width={width}
      height={height}
      className="block mx-auto"
    />
  );
}
```

## 10.9 增强的 Hooks

React 19 对现有 Hooks 进行了重要改进，减少了样板代码，提升了开发体验。

### 10.9.1 ref 作为 Prop — 告别 forwardRef

在 React 19 中，`ref` 变成了一个普通的 prop。不再需要使用 `forwardRef` 来转发 ref。

```tsx
// components/ModernInput.tsx
import { useRef, useImperativeHandle, type Ref } from 'react';

// React 19: ref 是普通 prop，不需要 forwardRef 包装
interface ModernInputProps {
  ref?: Ref<HTMLInputElement>;
  label: string;
  error?: string;
  placeholder?: string;
  type?: 'text' | 'email' | 'password';
  required?: boolean;
  disabled?: boolean;
  defaultValue?: string;
}

export function ModernInput({
  ref,
  label,
  error,
  placeholder,
  type = 'text',
  ...props
}: ModernInputProps) {
  return (
    <div className="space-y-1">
      <label className="block text-sm font-medium text-gray-700">
        {label}
      </label>
      <input
        ref={ref} // 直接作为 prop 传递，不需要 forwardRef
        type={type}
        placeholder={placeholder}
        className={`w-full px-3 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 ${
          error ? 'border-red-500' : 'border-gray-300'
        }`}
        aria-invalid={!!error}
        aria-describedby={error ? `${label}-error` : undefined}
        {...props}
      />
      {error && (
        <p id={`${label}-error`} className="text-sm text-red-600">
          {error}
        </p>
      )}
    </div>
  );
}

// 使用 ref 的示例
function ParentForm() {
  const inputRef = useRef<HTMLInputElement>(null);

  function focusInput() {
    inputRef.current?.focus();
  }

  return (
    <div className="space-y-4">
      <ModernInput
        ref={inputRef} // 直接传 ref，和传其他 prop 一样
        label="用户名"
        placeholder="输入用户名"
        required
      />
      <button
        type="button"
        onClick={focusInput}
        className="px-4 py-2 bg-gray-100 rounded-md hover:bg-gray-200"
      >
        聚焦到用户名
      </button>
    </div>
  );
}
```

### 10.9.2 暴露命令式 API — useImperativeHandle 简化

```tsx
// components/AdvancedForm.tsx
import { useRef, useImperativeHandle, type Ref } from 'react';

interface FormAPI {
  focus: () => void;
  reset: () => void;
  getValues: () => Record<string, string>;
  validate: () => boolean;
}

interface AdvancedFormProps {
  ref?: Ref<FormAPI>;
  onSubmit: (data: Record<string, string>) => void;
}

export function AdvancedForm({ ref, onSubmit }: AdvancedFormProps) {
  const formRef = useRef<HTMLFormElement>(null);
  const nameRef = useRef<HTMLInputElement>(null);
  const emailRef = useRef<HTMLInputElement>(null);

  // React 19: useImperativeHandle 第一个参数直接是 ref prop
  useImperativeHandle(ref, () => ({
    focus() {
      nameRef.current?.focus();
    },
    reset() {
      formRef.current?.reset();
    },
    getValues() {
      return {
        name: nameRef.current?.value ?? '',
        email: emailRef.current?.value ?? '',
      };
    },
    validate() {
      const values = {
        name: nameRef.current?.value ?? '',
        email: emailRef.current?.value ?? '',
      };
      return values.name.length >= 2 && values.email.includes('@');
    },
  }));

  return (
    <form
      ref={formRef}
      onSubmit={(e) => {
        e.preventDefault();
        const data = {
          name: nameRef.current?.value ?? '',
          email: emailRef.current?.value ?? '',
        };
        onSubmit(data);
      }}
      className="space-y-4"
    >
      <div className="space-y-1">
        <label className="block text-sm font-medium text-gray-700">姓名</label>
        <input
          ref={nameRef}
          placeholder="至少2个字符"
          required
          className="w-full px-3 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
      </div>
      <div className="space-y-1">
        <label className="block text-sm font-medium text-gray-700">邮箱</label>
        <input
          ref={emailRef}
          type="email"
          placeholder="your@email.com"
          required
          className="w-full px-3 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
      </div>
      <button type="submit" className="px-4 py-2 bg-blue-600 text-white rounded-md">
        提交
      </button>
    </form>
  );
}

// 父组件通过 ref 控制子组件
function Controller() {
  const formRef = useRef<FormAPI>(null);

  return (
    <div className="space-y-4">
      <div className="flex gap-2">
        <button onClick={() => formRef.current?.focus()} className="px-3 py-1 bg-gray-100 rounded">
          聚焦表单
        </button>
        <button onClick={() => formRef.current?.reset()} className="px-3 py-1 bg-gray-100 rounded">
          重置表单
        </button>
        <button
          onClick={() => {
            if (formRef.current?.validate()) {
              console.log('验证通过:', formRef.current.getValues());
            } else {
              console.log('验证失败');
            }
          }}
          className="px-3 py-1 bg-gray-100 rounded"
        >
          验证
        </button>
      </div>

      <AdvancedForm
        ref={formRef}
        onSubmit={(data) => console.log('提交:', data)}
      />
    </div>
  );
}
```

### 10.9.3 useEffect cleanup 的最佳实践

React 19 中的 `useEffect` cleanup 函数语义保持不变，但配合新的 Strict Mode 行为，开发者需要更注意 cleanup 的正确性。

```tsx
// hooks/useSubscription.ts
import { useEffect, useRef } from 'react';

// 正确实现 cleanup 的订阅 Hook
interface SubscriptionOptions<T> {
  channel: string;
  onMessage: (data: T) => void;
  onError?: (error: Error) => void;
}

export function useSubscription<T = unknown>({
  channel,
  onMessage,
  onError,
}: SubscriptionOptions<T>) {
  // 使用 ref 保存最新的回调，避免 cleanup 捕获过期闭包
  const onMessageRef = useRef(onMessage);
  const onErrorRef = useRef(onError);

  useEffect(() => {
    onMessageRef.current = onMessage;
    onErrorRef.current = onError;
  });

  useEffect(() => {
    let ws: WebSocket | null = null;
    let reconnectTimer: ReturnType<typeof setTimeout>;
    let mounted = true;

    function connect() {
      if (!mounted) return;

      ws = new WebSocket(`wss://example.com/ws/${channel}`);

      ws.onmessage = (event) => {
        if (mounted) {
          const data = JSON.parse(event.data) as T;
          onMessageRef.current(data);
        }
      };

      ws.onerror = () => {
        if (mounted) {
          onErrorRef.current?.(new Error('WebSocket 连接错误'));
        }
      };

      ws.onclose = () => {
        if (mounted) {
          // 自动重连
          reconnectTimer = setTimeout(connect, 3000);
        }
      };
    }

    connect();

    // cleanup 函数 — 在组件卸载或依赖变化时执行
    return () => {
      mounted = false;
      clearTimeout(reconnectTimer);
      if (ws) {
        ws.onclose = null; // 防止重连
        ws.close();
        ws = null;
      }
    };
  }, [channel]); // 当 channel 变化时，cleanup 旧连接并创建新连接
}

// 使用示例
function LiveComments({ postId }: { postId: string }) {
  const [comments, setComments] = useState<string[]>([]);

  useSubscription<string>({
    channel: `post-${postId}-comments`,
    onMessage: (comment) => {
      setComments((prev) => [...prev, comment]);
    },
    onError: (err) => {
      console.error('订阅错误:', err);
    },
  });

  return (
    <div>
      <h3 className="font-semibold mb-2">实时评论</h3>
      <ul className="space-y-1">
        {comments.map((c, i) => (
          <li key={i} className="text-sm text-gray-600 p-2 bg-gray-50 rounded">
            {c}
          </li>
        ))}
      </ul>
    </div>
  );
}
```

## 10.10 并发模式 (Concurrent Patterns)

React 19 完善了并发渲染能力，让你可以精细控制更新的优先级。

### 10.10.1 useTransition — 非紧急更新

`useTransition` 将状态更新标记为低优先级（transition），允许 React 在渲染这些更新时保持界面响应。特别适合搜索、筛选、导航等场景。

```tsx
// components/SearchWithTransition.tsx
'use client';

import { useState, useTransition, useMemo } from 'react';

interface Product {
  id: number;
  name: string;
  category: string;
  price: number;
  stock: number;
}

// 模拟大量数据
function generateProducts(count: number): Product[] {
  const categories = ['电子产品', '图书', '服装', '食品', '家居'];
  return Array.from({ length: count }, (_, i) => ({
    id: i + 1,
    name: `商品 ${i + 1}`,
    category: categories[i % categories.length],
    price: Math.floor(Math.random() * 1000) + 10,
    stock: Math.floor(Math.random() * 100),
  }));
}

const allProducts = generateProducts(10000);

export function SearchWithTransition() {
  const [query, setQuery] = useState('');
  const [category, setCategory] = useState('全部');
  const [deferredQuery, setDeferredQuery] = useState('');

  // useTransition: isPending 表示低优先级更新正在进行中
  const [isPending, startTransition] = useTransition();

  function handleSearch(input: string) {
    // 高优先级：立即更新输入框的值
    setQuery(input);

    // 低优先级：延迟更新搜索结果
    startTransition(() => {
      setDeferredQuery(input);
    });
  }

  // 昂贵的过滤计算 — 只在 transition 中执行
  const filteredProducts = useMemo(() => {
    const q = deferredQuery.toLowerCase();
    return allProducts.filter((p) => {
      const matchQuery = p.name.includes(q) || p.category.includes(q);
      const matchCategory = category === '全部' || p.category === category;
      return matchQuery && matchCategory;
    });
  }, [deferredQuery, category]);

  const categories = ['全部', ...new Set(allProducts.map((p) => p.category))];

  return (
    <div className="max-w-4xl mx-auto p-4">
      <div className="flex gap-4 mb-4">
        <input
          type="text"
          value={query}
          onChange={(e) => handleSearch(e.target.value)}
          placeholder="搜索商品..."
          className="flex-1 px-3 py-2 border rounded-md"
        />
        <select
          value={category}
          onChange={(e) => setCategory(e.target.value)}
          className="px-3 py-2 border rounded-md"
        >
          {categories.map((c) => (
            <option key={c} value={c}>{c}</option>
          ))}
        </select>
      </div>

      {/* isPending 指示器 — 显示在 transition 进行期间 */}
      <div className="flex items-center gap-2 mb-4 text-sm text-gray-500">
        {isPending ? (
          <>
            <span className="inline-block w-4 h-4 border-2 border-blue-400 border-t-transparent rounded-full animate-spin" />
            搜索中...
          </>
        ) : (
          <span>找到 {filteredProducts.length} 个结果</span>
        )}
      </div>

      {/* 使用 opacity 过渡表示数据正在更新 */}
      <div
        className={`grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 transition-opacity duration-200 ${
          isPending ? 'opacity-70' : 'opacity-100'
        }`}
      >
        {filteredProducts.slice(0, 50).map((product) => (
          <div key={product.id} className="p-4 border rounded-lg hover:shadow-md transition-shadow">
            <h3 className="font-semibold">{product.name}</h3>
            <p className="text-sm text-gray-500">{product.category}</p>
            <div className="flex justify-between items-center mt-2">
              <span className="font-bold text-lg">¥{product.price}</span>
              <span className={`text-sm ${product.stock > 0 ? 'text-green-600' : 'text-red-500'}`}>
                {product.stock > 0 ? `库存 ${product.stock}` : '缺货'}
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
```

### 10.10.2 useDeferredValue — 延迟计算

`useDeferredValue` 与 `useTransition` 类似，但它不控制 setState 的优先级。它接收一个值并返回该值的延迟版本，适用于值来自外部（如 props、context）的场景。

```tsx
// components/DeferredList.tsx
'use client';

import { useDeferredValue, useMemo } from 'react';

interface DataItem {
  id: number;
  label: string;
  value: number;
}

interface DeferredListProps {
  items: DataItem[];
  highlight: string;
}

// 重渲染开销大的组件
function ExpensiveList({ items, highlight }: { items: DataItem[]; highlight: string }) {
  // 模拟昂贵渲染
  const start = performance.now();
  while (performance.now() - start < 5) { /* 模拟计算 */ }

  const filtered = useMemo(
    () => items.filter((item) =>
      highlight ? item.label.toLowerCase().includes(highlight.toLowerCase()) : true
    ),
    [items, highlight]
  );

  return (
    <ul className="space-y-1 max-h-96 overflow-y-auto">
      {filtered.slice(0, 100).map((item) => (
        <li key={item.id} className="flex justify-between p-2 hover:bg-gray-50 rounded">
          <span>{item.label}</span>
          <span className="text-gray-500 tabular-nums">{item.value.toLocaleString()}</span>
        </li>
      ))}
    </ul>
  );
}

export function DeferredList({ items, highlight }: DeferredListProps) {
  // highlight 来自父组件的频繁更新，使用 useDeferredValue 延迟
  const deferredHighlight = useDeferredValue(highlight);

  const isStale = highlight !== deferredHighlight;

  return (
    <div className="relative">
      <div
        className={`transition-opacity duration-200 ${
          isStale ? 'opacity-50' : 'opacity-100'
        }`}
      >
        <ExpensiveList items={items} highlight={deferredHighlight} />
      </div>
      {isStale && (
        <div className="absolute top-0 right-0">
          <span className="inline-block w-4 h-4 border-2 border-blue-400 border-t-transparent rounded-full animate-spin" />
        </div>
      )}
    </div>
  );
}
```

### 10.10.3 startTransition — 非 React 状态

`startTransition` 可以用于任何状态更新（不仅是 React state），比如 URL 参数、全局 store 等。

```tsx
// hooks/useTransitionNavigation.ts
import { startTransition } from 'react';
import { useRouter } from 'next/navigation';

export function useTransitionNavigation() {
  const router = useRouter();

  function navigate(href: string) {
    // 将路由导航标记为低优先级 transition
    startTransition(() => {
      router.push(href);
    });
  }

  function replace(href: string) {
    startTransition(() => {
      router.replace(href);
    });
  }

  // 用于非 React 状态的 transition
  function updateExternalState(updater: () => void) {
    startTransition(() => {
      updater();
    });
  }

  return { navigate, replace, updateExternalState };
}

// 使用示例 — 标签页切换
import { useState, useTransition } from 'react';

function OverviewTab() {
  return <div className="p-4">概览内容 — 显示关键指标和摘要信息</div>;
}

function AnalyticsTab() {
  return <div className="p-4">分析内容 — 详细的数据分析和图表</div>;
}

function SettingsTab() {
  return <div className="p-4">设置内容 — 应用配置和偏好设置</div>;
}

export function TabNavigation() {
  const [activeTab, setActiveTab] = useState('overview');
  const [isPending, startLocalTransition] = useTransition();

  const tabs = [
    { id: 'overview', label: '概览', content: <OverviewTab /> },
    { id: 'analytics', label: '分析', content: <AnalyticsTab /> },
    { id: 'settings', label: '设置', content: <SettingsTab /> },
  ];

  function switchTab(tabId: string) {
    startLocalTransition(() => {
      setActiveTab(tabId);
    });
  }

  const currentTab = tabs.find((t) => t.id === activeTab);

  return (
    <div>
      <nav className="flex border-b mb-4">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => switchTab(tab.id)}
            className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
              activeTab === tab.id
                ? 'border-blue-600 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </nav>
      <div className={`transition-opacity ${isPending ? 'opacity-50' : 'opacity-100'}`}>
        {currentTab?.content}
      </div>
    </div>
  );
}
```

### 10.10.4 SuspenseList — 协调多个 Suspense 边界

在 React 19 中，`SuspenseList` 虽然在某些框架中被标记为实验性，但其核心思想——控制多个 Suspense 边界的渲染顺序——在并发模式中非常重要。

```tsx
// components/SuspenseOrchestrator.tsx
import { Suspense, use } from 'react';

// React 19 中协调多个 Suspense 边界的最佳实践

function LoadingSkeleton({ height = 'h-24' }: { height?: string }) {
  return (
    <div className={`${height} bg-gray-100 rounded-lg animate-pulse`}>
      <div className="p-4 space-y-3">
        <div className="h-4 bg-gray-200 rounded w-3/4" />
        <div className="h-4 bg-gray-200 rounded w-1/2" />
        <div className="h-4 bg-gray-200 rounded w-5/6" />
      </div>
    </div>
  );
}

// 方案1：嵌套 Suspense — 外层先显示，内层后显示
function HeaderSection() {
  const data = use(new Promise<string>(resolve => setTimeout(() => resolve('Header loaded'), 400)));
  return <div className="p-4 bg-white rounded-lg"><h2>{data}</h2></div>;
}

function ContentSection() {
  const data = use(new Promise<string>(resolve => setTimeout(() => resolve('Content loaded'), 800)));
  return <div className="p-4 bg-white rounded-lg"><p>{data}</p></div>;
}

export function NestedSuspenseStrategy() {
  return (
    <div className="space-y-6">
      <Suspense fallback={<LoadingSkeleton height="h-32" />}>
        <HeaderSection />
        {/* 嵌套的 Suspense：Header 先渲染，Content 后渲染 */}
        <Suspense fallback={<LoadingSkeleton height="h-64" />}>
          <ContentSection />
        </Suspense>
      </Suspense>
    </div>
  );
}

// 方案2：独立 Suspense — 并行显示各自的 fallback
function StatsWidget() {
  const data = use(new Promise<string>(resolve => setTimeout(() => resolve('Stats loaded'), 500)));
  return <div className="p-4 bg-white rounded-lg"><p>{data}</p></div>;
}

function ChartWidget() {
  const data = use(new Promise<string>(resolve => setTimeout(() => resolve('Chart loaded'), 700)));
  return <div className="p-4 bg-white rounded-lg"><p>{data}</p></div>;
}

function ActivityWidget() {
  const data = use(new Promise<string>(resolve => setTimeout(() => resolve('Activity loaded'), 900)));
  return <div className="p-4 bg-white rounded-lg"><p>{data}</p></div>;
}

export function ParallelSuspenseStrategy() {
  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
      <Suspense fallback={<LoadingSkeleton height="h-40" />}>
        <StatsWidget />
      </Suspense>
      <Suspense fallback={<LoadingSkeleton height="h-40" />}>
        <ChartWidget />
      </Suspense>
      <Suspense fallback={<LoadingSkeleton height="h-40" />}>
        <ActivityWidget />
      </Suspense>
    </div>
  );
}

// 方案3：reveal order — 手动控制显示顺序
function RevealOrder({
  children,
  order,
}: {
  children: React.ReactNode[];
  order: 'forwards' | 'backwards' | 'together';
}) {
  if (order === 'forwards') {
    return children.reduceRight((acc, child, i) => (
      <Suspense key={i} fallback={<LoadingSkeleton height="h-20" />}>
        {child}
        {acc}
      </Suspense>
    ), null as React.ReactNode);
  }

  if (order === 'together') {
    return <>{children}</>;
  }

  return children.reduce((acc, child, i) => (
    <Suspense key={i} fallback={<LoadingSkeleton height="h-20" />}>
      {acc}
      {child}
    </Suspense>
  ), null as React.ReactNode);
}

// 辅助：模拟慢速组件
function SlowComponent({ label, delay }: { label: string; delay: number }) {
  const data = use(
    new Promise<string>((resolve) =>
      setTimeout(() => resolve(`${label} 加载完成`), delay)
    )
  );

  return (
    <div className="p-4 bg-white border rounded-lg">
      <p>{data}</p>
    </div>
  );
}

// 使用示例
export function RevealOrderDemo() {
  return (
    <div className="space-y-2">
      <h3 className="font-semibold">按顺序显示 (forwards)</h3>
      <RevealOrder order="forwards">
        <SlowComponent label="第1项" delay={300} />
        <SlowComponent label="第2项" delay={600} />
        <SlowComponent label="第3项" delay={900} />
      </RevealOrder>
    </div>
  );
}
```

## 10.11 本章小结

本章涵盖了 React 19 中最具影响力的高级特性：

- **Actions API** (`useActionState`, `form action`) 提供了声明式的表单管理和服务端交互模式，自动处理 pending 状态和 progressive enhancement。
- **`use()` Hook** 打破了传统 Hooks 的限制，支持在条件语句和循环中读取 Promise 和 Context，与 Suspense 深度集成实现了优雅的数据加载模式。
- **Server Components (RSC)** 让组件在服务端运行，直接访问数据库，零 JavaScript 发送到客户端，从根本上改变了前后端交互模式。
- **Server Actions** 提供了从客户端到服务端的 Mutation 能力，配合 `revalidatePath` 和 `revalidateTag` 实现精确的缓存失效。
- **`useOptimistic`** 实现了乐观 UI 更新，让应用在慢网络下仍然感觉即时响应，失败时自动回滚。
- **Document Metadata** 允许在组件中直接声明 SEO 标签，无需第三方库。
- **Asset Loading APIs** (`preload`, `preinit`, `preconnect`, `prefetchDNS`) 提供了声明式资源加载优先级控制。
- **Web Components 集成** 大幅改善，支持直接作为 JSX 使用。
- **增强的 Hooks** — `ref` 作为普通 prop、简化的 `useImperativeHandle`、更可靠的 cleanup 语义。
- **并发模式** — `useTransition`、`useDeferredValue`、`startTransition` 让你精细控制更新优先级，保持 UI 始终响应。

这些特性共同构成了 React 19 的核心竞争力：更好的用户体验（并发模式、乐观更新）、更简单的架构（RSC 消除 API 层）、以及更少的样板代码（ref 作为 prop、Actions API）。在实际项目中，建议逐步引入这些特性——先从 `useActionState` 和 `useTransition` 开始，再考虑 RSC 迁移，最后引入 `useOptimistic` 等高级模式。
