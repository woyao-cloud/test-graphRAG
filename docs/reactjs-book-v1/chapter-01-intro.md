# 第1章：React 19 概述与生态

## 1.1 什么是 React 19

React 19 是 Meta 开源的 JavaScript 前端库的最新主版本，于 2024 年底发布。它在前身 React 18 的基础上引入了多项革命性更新，包括 Actions、Server Components、新的 use() Hook 等。

### 1.1.1 React 19 新特性概览

| 特性 | 说明 | 影响 |
|------|------|------|
| Actions | 内置异步状态管理 | 简化表单和数据提交 |
| use() | 在 render 中读取 Promise/Context | 简化数据获取 |
| Server Components | 服务端渲染组件，零客户端 JS | 减少 bundle 体积 |
| Server Actions | 客户端调用服务端函数 | 简化数据变更 |
| useOptimistic | 乐观更新 | 提升用户体验 |
| Document Metadata | 直接管理 title/meta | 简化 SEO |
| ref as prop | 不再需要 forwardRef | 简化组件 API |

## 1.2 React 与其他框架对比

### 1.2.1 React vs Vue

| 维度 | React | Vue |
|------|-------|-----|
| 范式 | 函数式、不可变数据 | 响应式、可变数据 |
| 模板 | JSX (JavaScript) | 模板 + JSX |
| 状态管理 | useState/useReducer | ref/reactive |
| 学习曲线 | 中高 | 低 |
| 生态 | 大而灵活 | 大而统一 |

### 1.2.2 React vs Angular

| 维度 | React | Angular |
|------|-------|---------|
| 类型 | 库 | 全栈框架 |
| 架构 | 自由组合 | 约定优于配置 |
| 数据流 | 单向 | 单向 + 双向绑定 |
| 依赖注入 | 无 (Context) | 内置 DI |
| 路由 | React Router | 内置 Router |

### 1.2.3 React vs Svelte

| 维度 | React | Svelte |
|------|-------|--------|
| 机制 | 运行时 (Virtual DOM) | 编译时 (无 VDOM) |
| 状态 | useState | let 声明 + $: 响应式 |
| 包大小 | ~40KB | ~2KB |
| 生态 | 成熟丰富 | 成长中 |

## 1.3 生态系统概览

### 构建工具
- **Vite**: 默认推荐，基于 esbuild + Rollup，HMR 极快
- **Next.js**: 全栈框架，SSR/SSG/ISR
- **Remix**: 基于 Web 标准，嵌套路由

### 状态管理
- **Zustand**: 轻量、基于 hook
- **Jotai**: 原子状态、派生状态
- **TanStack Query**: 服务端状态缓存
- **Redux Toolkit**: 传统选择、RTK Query

### 路由
- **React Router v7**: loaders、actions、类型安全
- **TanStack Router**: 100% 类型安全

### 样式
- **Tailwind CSS**: 工具优先、JIT 编译
- **CSS Modules**: 作用域隔离
- **styled-components**: CSS-in-JS

## 1.4 从 React 18 迁移

### 升级步骤
```bash
npm install react@19 react-dom@19
```

### 主要变更
- `ReactDOM.render` → `createRoot`
- `PropTypes` 不再内置
- `Strict Mode` 行为改进
- `ref` 可直接作为 prop 传递
- `useEffect` 清理函数在开发模式下严格调用

## 1.5 何时选择 React

**适合使用 React:**
- 复杂交互的单页应用
- 需要跨平台 (React Native)
- 大型团队协作
- 丰富的第三方库需求

**可能不适合:**
- 简单静态页面
- SEO 优先的内容网站
- 极简 bundle 要求
