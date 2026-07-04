# 第九章：企业级 React 工程实践

企业级 React 应用面临的挑战远超个人项目：数十位开发者并行协作、多个团队维护独立子应用、设计系统需要跨产品保持一致、国际化覆盖数十种语言、主题需要适配品牌定制、无障碍访问必须满足合规要求、生产环境的监控和错误追踪不可或缺、安全漏洞可能造成严重后果。本章将逐一拆解这些工程领域的核心工具和最佳实践，为构建和维护大型 React 应用提供可操作的参考指南。

---

## 9.1 Monorepo 架构

Monorepo（单一仓库）将多个项目或包放在同一个 Git 仓库中统一管理。相比多仓库（polyrepo），它消除了跨仓库版本对齐的摩擦，让原子化提交和跨包重构成为可能。在 React 生态中，Turborepo 和 Nx 是两种主流方案。

### 9.1.1 Turborepo：基于缓存的构建编排

Turborepo 的核心设计理念是"尽量不做重复工作"。它通过分析包之间的依赖拓扑，决定哪些任务可以并行执行，并将构建产物缓存到本地或云端，使得后续构建只需重算变更的部分。

#### Pipeline 配置

`turbo.json` 是整个仓库的任务编排中枢。`pipeline` 中的每个键对应 `package.json` 中的 script 名称，`dependsOn` 定义了拓扑依赖关系。

```json
// turbo.json
{
  "$schema": "https://turbo.build/schema.json",
  "globalDependencies": ["**/.env.*local"],
  "pipeline": {
    "build": {
      "dependsOn": ["^build"],
      "outputs": [".next/**", "dist/**", ".svelte-kit/**"],
      "env": ["NODE_ENV", "API_URL", "NEXT_PUBLIC_CDN_HOST"]
    },
    "test": {
      "dependsOn": ["build"],
      "outputs": ["coverage/**"],
      "inputs": ["src/**/*.tsx", "src/**/*.ts", "test/**/*.ts", "jest.config.*"]
    },
    "lint": {
      "dependsOn": ["^build"]
    },
    "typecheck": {
      "dependsOn": ["^build"],
      "outputs": []
    },
    "dev": {
      "cache": false,
      "persistent": true
    },
    "deploy": {
      "dependsOn": ["build", "test", "lint", "typecheck"]
    }
  }
}
```

关键配置解读：
- `"^build"` 前缀 `^` 表示"先构建此包的依赖包"，再构建此包自身
- `"build"` 不带 `^` 表示"先构建此包自身"，再执行当前任务（通常用于 test 依赖 build）
- `cache: false` 禁用缓存，适用于 dev server 等持续运行的任务
- `inputs` 精确指定哪些文件变更才触发缓存失效

#### 远程缓存（Remote Caching）

本地缓存仅在单台机器上生效。当 CI 流水线和团队成员的机器共享远程缓存时，任何被他人构建过的任务都可以直接命中缓存，跳过执行。Turborepo 支持 Vercel 官方远程缓存或自建缓存服务器。

```bash
# 连接远程缓存（Vercel）
npx turbo login
npx turbo link

# 构建时自动使用远程缓存
turbo run build --cache-dir=".turbo-cache"
```

配置 `.github/workflows/ci.yml` 中的远程缓存：

```yaml
# .github/workflows/ci.yml
- name: Build
  run: turbo run build --filter="...[origin/main]" --cache-dir="node_modules/.cache/turbo"
  env:
    TURBO_TOKEN: ${{ secrets.TURBO_TOKEN }}
    TURBO_TEAM: ${{ vars.TURBO_TEAM }}
```

### 9.1.2 Nx：面向大型企业的工作区管理

Nx 提供比 Turborepo 更丰富的功能集，包括代码生成器、依赖图可视化和智能的 affected 命令。如果团队规模超过 50 人，或者需要强约束的项目结构规范，Nx 是更好的选择。

#### 项目初始化与代码生成器

Nx 的 generator（生成器）通过统一的模板生成组件、模块、库等代码骨架，保证跨团队的一致性。

```bash
# 创建 Nx 工作区
npx create-nx-workspace@latest my-org --preset=react-monorepo

# 生成 React 应用
nx g @nx/react:application admin --bundler=vite --e2eTestRunner=playwright

# 生成共享 UI 库
nx g @nx/react:library shared-ui --directory=packages/shared-ui --publishable

# 生成组件（在特定库中）
nx g @nx/react:component button --directory=packages/shared-ui/src/lib/button --export
```

自定义 generator 示例：

```typescript
// tools/generators/react-component/index.ts
import { Tree, formatFiles, generateFiles, names } from '@nx/devkit';
import * as path from 'path';

interface ComponentGeneratorOptions {
  name: string;
  directory: string;
  withTests?: boolean;
  withStories?: boolean;
}

export async function reactComponentGenerator(
  tree: Tree,
  options: ComponentGeneratorOptions
) {
  const { fileName, className } = names(options.name);
  const targetDir = path.join(options.directory, fileName);

  generateFiles(tree, path.join(__dirname, 'files'), targetDir, {
    ...options,
    fileName,
    className,
    tmpl: '',
  });

  await formatFiles(tree);
}

// tools/generators/react-component/files/__fileName__.tsx__tmpl__
import { FC } from 'react';

export interface <%= className %>Props {
  children?: React.ReactNode;
}

export const <%= className %>: FC<<%= className %>Props> = ({ children }) => {
  return <div>{children}</div>;
};
```

#### Affected 命令与依赖图

Nx 通过分析 Git 变更和项目依赖图，只对受影响的项目执行任务。依赖图由 Nx 自动从 `tsconfig.json` 的 `references` 和 `package.json` 的依赖关系中推导。

```bash
# 查看依赖图
nx graph

# 只对受当前分支变更影响的项目运行测试
nx affected --target=test --base=main

# 只构建受影响的项目及其依赖
nx affected --target=build --base=main --parallel=4

# 查看受影响的项目列表（不实际执行）
nx affected --target=build --base=main --dry-run
```

Nx 的 `affected` 命令会对比当前分支与 base 分支（如 `main`）之间的文件差异，然后遍历依赖图找出所有直接或间接受影响的项目，只对这些项目执行指定的 task。在一个有 50 个 package 的 monorepo 中，一次只改动了 `utils` 包，`nx affected --target=test` 可能只运行 5 个测试套件而非全部 50 个。

#### 项目结构示例

```
my-org/
├── apps/
│   ├── web/                   # 主站 (Next.js)
│   │   ├── src/
│   │   ├── index.html
│   │   ├── project.json       # Nx 项目配置
│   │   └── tsconfig.json
│   └── admin/                 # 管理后台 (Vite)
│       ├── src/
│       ├── project.json
│       └── tsconfig.json
├── packages/
│   ├── shared-ui/             # 共享组件库
│   │   ├── src/
│   │   │   └── lib/
│   │   │       ├── button/
│   │   │       ├── modal/
│   │   │       └── table/
│   │   ├── project.json
│   │   └── tsconfig.json
│   ├── shared-utils/          # 工具函数
│   ├── shared-types/          # TypeScript 类型定义
│   └── config-eslint/         # 共享 ESLint 配置
├── tools/
│   └── generators/            # 自定义代码生成器
├── nx.json                    # Nx 工作区配置
├── package.json
└── tsconfig.base.json
```

`nx.json` 中的任务编排配置：

```json
// nx.json
{
  "$schema": "./node_modules/nx/schemas/nx-schema.json",
  "targetDefaults": {
    "build": {
      "dependsOn": ["^build"],
      "outputs": ["{projectRoot}/dist"],
      "cache": true
    },
    "test": {
      "dependsOn": ["build"],
      "cache": true
    },
    "lint": {
      "cache": true
    },
    "typecheck": {
      "dependsOn": ["^build"],
      "cache": true
    }
  },
  "defaultBase": "main",
  "plugins": [
    { "plugin": "@nx/vite/plugin", "options": { "buildTargetName": "build" } },
    { "plugin": "@nx/next/plugin", "options": { "buildTargetName": "build" } }
  ]
}
```

---

## 9.2 微前端架构

微前端将大型单体前端拆分为多个独立开发、独立部署的子应用，每个子应用由独立团队负责。Webpack 5 的 Module Federation 和 single-spa 是实现微前端的两条主流路径。

### 9.2.1 Webpack 5 Module Federation

Module Federation 允许一个 JavaScript 应用在运行时动态加载另一个应用的模块，就像使用本地模块一样自然。

#### 远程应用（Remote）

远程应用暴露特定的模块供主应用消费。`exposes` 配置定义了暴露哪些模块，`filename` 是运行时的入口文件。

```javascript
// apps/team-a/webpack.config.js
const { ModuleFederationPlugin } = require('webpack').container;
const path = require('path');

module.exports = {
  entry: './src/index',
  mode: 'development',
  devServer: { port: 3001 },
  output: { publicPath: 'http://localhost:3001/' },
  plugins: [
    new ModuleFederationPlugin({
      name: 'teamA',
      filename: 'remoteEntry.js',
      exposes: {
        './Dashboard': './src/pages/Dashboard',
        './UserProfile': './src/pages/UserProfile',
        './store': './src/store/index',
      },
      shared: {
        react: {
          singleton: true,
          requiredVersion: '^19.0.0',
          eager: false,
        },
        'react-dom': {
          singleton: true,
          requiredVersion: '^19.0.0',
          eager: false,
        },
        'react-router-dom': {
          singleton: true,
          requiredVersion: '^7.0.0',
        },
        zustand: {
          singleton: true,
          requiredVersion: '^5.0.0',
        },
      },
    }),
  ],
};
```

#### 主应用（Host）

主应用通过 `remotes` 声明它需要消费的远程应用地址，然后使用动态 `import()` 加载远程模块。

```javascript
// apps/shell/webpack.config.js
const { ModuleFederationPlugin } = require('webpack').container;

module.exports = {
  plugins: [
    new ModuleFederationPlugin({
      name: 'shell',
      remotes: {
        teamA: 'teamA@http://localhost:3001/remoteEntry.js',
        teamB: 'teamB@http://localhost:3002/remoteEntry.js',
      },
      shared: {
        react: { singleton: true, requiredVersion: '^19.0.0', eager: true },
        'react-dom': { singleton: true, requiredVersion: '^19.0.0', eager: true },
        'react-router-dom': { singleton: true, requiredVersion: '^7.0.0' },
      },
    }),
  ],
};
```

主应用中使用远程模块：

```tsx
// apps/shell/src/App.tsx
import { Suspense, lazy, ComponentType } from 'react';
import { BrowserRouter, Routes, Route, Link } from 'react-router-dom';

// 类型安全的远程模块声明
declare module 'teamA/Dashboard' {
  const Dashboard: ComponentType;
  export default Dashboard;
}
declare module 'teamA/UserProfile' {
  const UserProfile: ComponentType;
  export default UserProfile;
}

// 按需加载远程组件
const TeamADashboard = lazy(() => import('teamA/Dashboard'));
const TeamAUserProfile = lazy(() => import('teamA/UserProfile'));

// 错误边界：远程模块加载失败时优雅降级
function RemoteErrorBoundary({ children, fallback }: {
  children: React.ReactNode;
  fallback?: React.ReactNode;
}) {
  return (
    <Suspense
      fallback={
        <div className="flex items-center justify-center h-64 text-gray-500">
          正在加载远程模块...
        </div>
      }
    >
      {children}
    </Suspense>
  );
}

export function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen">
        <nav className="flex gap-4 p-4 border-b">
          <Link to="/team-a/dashboard">Team A - Dashboard</Link>
          <Link to="/team-a/profile">Team A - Profile</Link>
        </nav>
        <main className="p-6">
          <Routes>
            <Route path="/team-a/dashboard" element={
              <RemoteErrorBoundary><TeamADashboard /></RemoteErrorBoundary>
            } />
            <Route path="/team-a/profile" element={
              <RemoteErrorBoundary><TeamAUserProfile /></RemoteErrorBoundary>
            } />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}
```

#### Shared Dependencies 策略

`shared` 配置是 Module Federation 最关键的环节。`singleton: true` 保证整个页面只存在一份 react 实例（否则 hooks 会报错）；`eager: true` 表示主应用启动时立即加载此依赖（主应用通常需要 eager 加载 react）；`requiredVersion` 声明兼容的版本范围，不匹配时控制台会发出警告。

### 9.2.2 single-spa：框架无关的微前端编排

single-spa 是一个微前端路由器，负责根据 URL 动态挂载和卸载子应用。与 Module Federation 的"模块级共享"不同，single-spa 的粒度是"应用级"——每个子应用是一个完整的 SPA，可以是 React、Vue、Angular 或任何框架。

#### 子应用注册

每个子应用需要暴露三个生命周期函数：`bootstrap`、`mount`、`unmount`。

```tsx
// apps/team-b/src/main.ts
import singleSpaReact from 'single-spa-react';
import ReactDOMClient from 'react-dom/client';
import { App } from './App';

const lifecycles = singleSpaReact({
  React: await import('react'),
  ReactDOMClient,
  rootComponent: App,
  errorBoundary(err, info, props) {
    return (
      <div className="error-boundary">
        <h2>Team B 应用加载失败</h2>
        <pre>{err.message}</pre>
      </div>
    );
  },
});

export const { bootstrap, mount, unmount } = lifecycles;
```

#### 根配置（Root Config）

根配置是 single-spa 的大脑，负责注册所有子应用并定义它们的激活条件。

```tsx
// root-config/src/index.ts
import { registerApplication, start } from 'single-spa';

registerApplication({
  name: '@my-org/team-a',
  app: () => System.import('@my-org/team-a'),
  activeWhen: (location) => location.pathname.startsWith('/team-a'),
  customProps: {
    apiBaseUrl: 'https://api.example.com/v1',
    featureFlags: { newDashboard: true },
  },
});

registerApplication({
  name: '@my-org/team-b',
  app: () => System.import('@my-org/team-b'),
  activeWhen: ['/team-b', '/shared'],
});

registerApplication({
  name: '@my-org/navbar',
  app: () => System.import('@my-org/navbar'),
  activeWhen: () => true, // 始终激活
});

start({ urlRerouteOnly: true });
```

#### Parcels：组件级微前端

single-spa 的 parcel 机制允许在子应用内部嵌入另一个微前端片段，粒度比应用更细。parcel 拥有自己的挂载/卸载生命周期，可以独立于路由存在。

```tsx
// 在 React 组件中使用 parcel
import { mountRootParcel, Parcel } from 'single-spa';
import { useRef, useEffect } from 'react';

function WidgetContainer() {
  const parcelRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let parcel: any;

    System.import('@my-org/chat-widget').then(({ bootstrap, mount, unmount }) => {
      parcel = mountRootParcel({ bootstrap, mount, unmount }, {
        domElement: parcelRef.current!,
        userId: 'current-user-id',
        theme: 'dark',
      });
    });

    return () => {
      parcel?.unmount?.();
    };
  }, []);

  return <div ref={parcelRef} />;
}
```

single-spa 的优势在于框架无关性——可以用 React 构建主应用外壳，同时在内部嵌入 Vue 或 Angular 构建的子应用。Module Federation 的优势在于更紧密的模块共享和更小的运行时开销。实际选型时，如果团队统一使用 React，Module Federation 更简洁；如果涉及多框架协作或渐进式迁移，single-spa 更灵活。

---

## 9.3 设计系统

设计系统是一套可复用的组件、样式指南和设计语言，确保跨产品的一致性。在 React 生态中，Storybook 是组件开发与文档的行业标准，Radix UI 提供无样式的无障碍原语，shadcn/ui 则将这两者结合为可直接复制使用的组件集合。

### 9.3.1 Storybook：组件的开发与文档平台

Storybook 为每个组件提供隔离的开发和展示环境。每个组件的故事（story）既是开发时的测试用例，也是团队的文档参考。

#### Story 格式与 Controls

```tsx
// packages/ui/src/Button/Button.stories.tsx
import type { Meta, StoryObj } from '@storybook/react';
import { fn } from '@storybook/test';
import { Button } from './Button';

const meta: Meta<typeof Button> = {
  title: 'Design System/Button',
  component: Button,
  tags: ['autodocs'],
  argTypes: {
    variant: {
      control: { type: 'select' },
      options: ['primary', 'secondary', 'outline', 'destructive', 'ghost', 'link'],
      description: '按钮的视觉风格变体',
    },
    size: {
      control: { type: 'radio' },
      options: ['sm', 'md', 'lg'],
      description: '按钮尺寸',
    },
    disabled: {
      control: 'boolean',
      description: '禁用状态',
    },
    loading: {
      control: 'boolean',
      description: '显示加载指示器',
    },
    children: {
      control: 'text',
      description: '按钮文本内容',
    },
  },
  args: {
    onClick: fn(),
    variant: 'primary',
    size: 'md',
    disabled: false,
    loading: false,
    children: 'Click Me',
  },
};

export default meta;
type Story = StoryObj<typeof Button>;

export const Primary: Story = { args: { variant: 'primary' } };
export const Secondary: Story = { args: { variant: 'secondary' } };
export const Outline: Story = { args: { variant: 'outline' } };
export const Destructive: Story = { args: { variant: 'destructive' } };
export const Ghost: Story = { args: { variant: 'ghost' } };

export const Small: Story = { args: { size: 'sm' } };
export const Large: Story = { args: { size: 'lg' } };

export const Loading: Story = {
  args: { loading: true, disabled: false },
};

export const Disabled: Story = {
  args: { disabled: true },
};

// 组合展示：所有变体一览
export const AllVariants: Story = {
  name: '全部变体',
  render: (args) => (
    <div className="flex flex-wrap gap-3 items-center p-4">
      {(['primary', 'secondary', 'outline', 'destructive', 'ghost', 'link'] as const).map(
        (variant) => (
          <Button key={variant} {...args} variant={variant}>
            {variant}
          </Button>
        )
      )}
    </div>
  ),
};

// 交互测试（使用 @storybook/test）
export const ClickInteraction: Story = {
  play: async ({ canvasElement, args }) => {
    const canvas = within(canvasElement);
    const button = canvas.getByRole('button');
    await userEvent.click(button);
    await expect(args.onClick).toHaveBeenCalledTimes(1);
  },
};
```

#### Addons 生态

Storybook 的 addon 生态极大扩展了其能力。常用的 addon 包括：

| Addon | 用途 |
|-------|------|
| `@storybook/addon-essentials` | 包含 controls, actions, docs, viewport 等核心功能 |
| `@storybook/addon-a11y` | 自动检测无障碍问题 |
| `@storybook/addon-links` | 在 story 之间导航 |
| `@storybook/addon-interactions` | 交互测试和 play 函数 |
| `@storybook/addon-designs` | 在组件旁嵌入 Figma 设计稿 |
| `@storybook/addon-themes` | 主题切换（亮色/暗色模式） |
| `@chromatic-com/storybook` | 视觉回归测试（截图对比） |

```typescript
// .storybook/main.ts
import type { StorybookConfig } from '@storybook/react-vite';

const config: StorybookConfig = {
  stories: ['../packages/ui/src/**/*.stories.@(ts|tsx)'],
  addons: [
    '@storybook/addon-essentials',
    '@storybook/addon-a11y',
    '@storybook/addon-interactions',
    '@storybook/addon-links',
    '@storybook/addon-themes',
    '@chromatic-com/storybook',
  ],
  framework: { name: '@storybook/react-vite', options: {} },
  docs: { autodocs: 'tag' },
};

export default config;
```

### 9.3.2 Radix UI：无障碍原语组件

Radix UI 提供一系列无样式（unstyled）的 React 组件原语，每个原语都内置了完整的 WAI-ARIA 实现、键盘导航和焦点管理。开发者完全控制样式，但无需操心无障碍细节。

```tsx
// components/Dialog.tsx —— 基于 Radix UI 的 Dialog 封装
import * as Dialog from '@radix-ui/react-dialog';
import { Cross2Icon } from '@radix-ui/react-icons';
import { forwardRef } from 'react';

interface ModalProps {
  trigger: React.ReactNode;
  title: string;
  description?: string;
  children: React.ReactNode;
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
}

export function Modal({ trigger, title, description, children, open, onOpenChange }: ModalProps) {
  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Trigger asChild>{trigger}</Dialog.Trigger>
      <Dialog.Portal>
        {/* 遮罩层 */}
        <Dialog.Overlay className="fixed inset-0 bg-black/50 data-[state=open]:animate-overlayShow" />
        {/* 对话框内容 */}
        <Dialog.Content
          className="fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 
                     w-[90vw] max-w-lg rounded-lg bg-white p-6 shadow-xl
                     data-[state=open]:animate-contentShow
                     focus:outline-none"
        >
          <Dialog.Title className="text-lg font-semibold text-gray-900">
            {title}
          </Dialog.Title>
          {description && (
            <Dialog.Description className="mt-2 text-sm text-gray-500">
              {description}
            </Dialog.Description>
          )}
          <div className="mt-4">{children}</div>
          <Dialog.Close
            className="absolute top-4 right-4 inline-flex items-center justify-center 
                       rounded-full p-1 text-gray-400 hover:bg-gray-100 hover:text-gray-600"
            aria-label="关闭"
          >
            <Cross2Icon />
          </Dialog.Close>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}

// DropdownMenu 示例
import * as DropdownMenu from '@radix-ui/react-dropdown-menu';

export function UserMenu() {
  return (
    <DropdownMenu.Root>
      <DropdownMenu.Trigger className="flex items-center gap-2 rounded-full p-1 hover:ring-2 hover:ring-primary">
        <img src="/avatar.png" alt="用户头像" className="h-8 w-8 rounded-full" />
      </DropdownMenu.Trigger>
      <DropdownMenu.Portal>
        <DropdownMenu.Content
          className="min-w-[200px] rounded-lg bg-white p-1.5 shadow-lg border"
          sideOffset={8}
        >
          <DropdownMenu.Item className="flex items-center gap-2 rounded-md px-3 py-2 text-sm outline-none cursor-pointer hover:bg-gray-100">
            个人设置
          </DropdownMenu.Item>
          <DropdownMenu.Item className="flex items-center gap-2 rounded-md px-3 py-2 text-sm outline-none cursor-pointer hover:bg-gray-100">
            团队管理
          </DropdownMenu.Item>
          <DropdownMenu.Separator className="my-1 h-px bg-gray-200" />
          <DropdownMenu.Item className="flex items-center gap-2 rounded-md px-3 py-2 text-sm text-red-600 outline-none cursor-pointer hover:bg-red-50">
            退出登录
          </DropdownMenu.Item>
        </DropdownMenu.Content>
      </DropdownMenu.Portal>
    </DropdownMenu.Root>
  );
}
```

Radix UI 的核心优势：每个组件都正确处理了 `role`、`aria-expanded`、`aria-haspopup` 等属性，键盘操作（Tab、Enter、Escape、方向键）开箱即用，焦点自动在打开/关闭时转移，符合 WCAG 2.1 AA 标准。

### 9.3.3 shadcn/ui：可复制的组件集合

shadcn/ui 不是一个 npm 包，而是一套通过 CLI 将源码直接复制到项目中的组件。它基于 Radix UI 原语和 Tailwind CSS，赋予开发者对组件源码的完全控制权。

```bash
# 初始化 shadcn/ui
npx shadcn@latest init

# 添加组件（源码直接写入项目）
npx shadcn@latest add button
npx shadcn@latest add dialog
npx shadcn@latest add dropdown-menu
npx shadcn@latest add table
npx shadcn@latest add form
```

添加后的组件位于 `src/components/ui/` 下，可以直接修改源码：

```tsx
// src/components/ui/button.tsx（shadcn/ui 生成的源码，你可以自由修改）
import * as React from "react"
import { Slot } from "@radix-ui/react-slot"
import { cva, type VariantProps } from "class-variance-authority"
import { cn } from "@/lib/utils"

const buttonVariants = cva(
  "inline-flex items-center justify-center whitespace-nowrap rounded-md text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50",
  {
    variants: {
      variant: {
        default: "bg-primary text-primary-foreground hover:bg-primary/90",
        destructive: "bg-destructive text-destructive-foreground hover:bg-destructive/90",
        outline: "border border-input bg-background hover:bg-accent hover:text-accent-foreground",
        secondary: "bg-secondary text-secondary-foreground hover:bg-secondary/80",
        ghost: "hover:bg-accent hover:text-accent-foreground",
        link: "text-primary underline-offset-4 hover:underline",
      },
      size: {
        default: "h-10 px-4 py-2",
        sm: "h-9 rounded-md px-3",
        lg: "h-11 rounded-md px-8",
        icon: "h-10 w-10",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
)

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : "button"
    return (
      <Comp
        className={cn(buttonVariants({ variant, size, className }))}
        ref={ref}
        {...props}
      />
    )
  }
)
Button.displayName = "Button"

export { Button, buttonVariants }
```

shadcn/ui 的独特哲学：不引入版本锁定的依赖，不隐藏实现细节，不强制升级策略。团队拿到的是完整的源码，可以按需修改、删减或扩展，同时保留了 Radix UI 的无障碍能力。

---

## 9.4 国际化（i18n）

国际化涉及文本翻译、日期/数字格式化、复数规则、从右到左（RTL）布局等层面。React 生态中，react-intl（FormatJS）和 i18next 是两种主流方案。

### 9.4.1 react-intl（FormatJS）

react-intl 基于 ICU 消息语法，提供了声明式的消息定义和格式化组件。

```tsx
// i18n/provider.tsx
import { IntlProvider, createIntlCache, createIntl } from 'react-intl';
import { useState, useCallback } from 'react';

// 语言包
const messagesMap: Record<string, Record<string, string>> = {
  'zh-CN': {
    'app.title': 'React 企业级应用',
    'app.greeting': '你好，{name}！',
    'app.unreadCount': '{count, plural, =0 {没有未读消息} =1 {1 条未读消息} other {# 条未读消息}}',
    'app.lastLogin': '上次登录：{date, date, long}',
    'app.balance': '账户余额：{amount, number, ::currency/CNY unit-width-narrow}',
    'app.progress': '完成度：{percent, number, ::percent scale/100}',
  },
  'en-US': {
    'app.title': 'React Enterprise App',
    'app.greeting': 'Hello, {name}!',
    'app.unreadCount': '{count, plural, =0 {No unread messages} =1 {1 unread message} other {# unread messages}}',
    'app.lastLogin': 'Last login: {date, date, long}',
    'app.balance': 'Balance: {amount, number, ::currency/USD unit-width-narrow}',
    'app.progress': 'Progress: {percent, number, ::percent scale/100}',
  },
};

export function I18nProvider({ children }: { children: React.ReactNode }) {
  const [locale, setLocale] = useState(() => navigator.language || 'zh-CN');

  const switchLocale = useCallback((nextLocale: string) => {
    setLocale(nextLocale);
    document.documentElement.lang = nextLocale;
  }, []);

  return (
    <IntlProvider locale={locale} messages={messagesMap[locale]} defaultLocale="zh-CN">
      <LocaleContext.Provider value={{ locale, switchLocale }}>
        {children}
      </LocaleContext.Provider>
    </IntlProvider>
  );
}
```

#### defineMessages 与 useIntl

```tsx
// components/Greeting.tsx
import { FormattedMessage, FormattedDate, FormattedNumber, useIntl, defineMessages } from 'react-intl';

const messages = defineMessages({
  greeting: {
    id: 'app.greeting',
    defaultMessage: '你好，{name}！',
    description: '顶部问候语，{name} 为用户显示名',
  },
  unreadCount: {
    id: 'app.unreadCount',
    defaultMessage: '{count, plural, =0 {没有未读消息} =1 {1 条未读消息} other {# 条未读消息}}',
    description: '未读消息数量提示',
  },
});

export function Greeting({ userName, unreadCount }: { userName: string; unreadCount: number }) {
  const intl = useIntl();

  return (
    <div>
      {/* 声明式：FormattedMessage 组件 */}
      <h1><FormattedMessage {...messages.greeting} values={{ name: userName }} /></h1>
      <p><FormattedMessage {...messages.unreadCount} values={{ count: unreadCount }} /></p>

      {/* 命令式：useIntl hook（适用于非 JSX 场景，如 toast 标题） */}
      <p>{intl.formatMessage(messages.greeting, { name: userName })}</p>

      {/* 日期格式化 */}
      <p><FormattedDate value={Date.now()} year="numeric" month="long" day="numeric" /></p>

      {/* 数字格式化 */}
      <p><FormattedNumber value={12345.67} style="currency" currency="CNY" /></p>
    </div>
  );
}
```

### 9.4.2 i18next + react-i18next

i18next 的生态更广泛（支持 40+ 框架/平台），其翻译文件支持嵌套结构和插值语法。

```typescript
// i18n/config.ts
import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';
import LanguageDetector from 'i18next-browser-languagedetector';

i18n
  .use(LanguageDetector)
  .use(initReactI18next)
  .init({
    resources: {
      'zh-CN': {
        translation: {
          nav: { home: '首页', about: '关于', settings: '设置' },
          user: {
            greeting: '你好，{{name}}！',
            role: '您的角色是 {{role}}',
          },
          notification: '您有 {{count}} 条新通知',
          notification_plural: '您有 {{count}} 条新通知',
          confirm_delete: '确定要删除「{{item}}」吗？此操作不可撤销。',
        },
      },
      'en-US': {
        translation: {
          nav: { home: 'Home', about: 'About', settings: 'Settings' },
          user: {
            greeting: 'Hello, {{name}}!',
            role: 'Your role is {{role}}',
          },
          notification: 'You have {{count}} new notification',
          notification_plural: 'You have {{count}} new notifications',
          confirm_delete: 'Are you sure you want to delete "{{item}}"? This cannot be undone.',
        },
      },
    },
    fallbackLng: 'en-US',
    interpolation: { escapeValue: false }, // React 已自动转义
    detection: {
      order: ['querystring', 'cookie', 'localStorage', 'navigator', 'htmlTag'],
      caches: ['localStorage', 'cookie'],
    },
  });

export default i18n;
```

#### Trans 组件与 useTranslation

```tsx
// components/UserPanel.tsx
import { useTranslation, Trans } from 'react-i18next';

export function UserPanel() {
  const { t, i18n } = useTranslation();

  return (
    <div>
      {/* 基本翻译 */}
      <h2>{t('user.greeting', { name: 'Alice' })}</h2>
      <p>{t('user.role', { role: '管理员' })}</p>

      {/* 复数规则 */}
      <p>{t('notification', { count: 5 })}</p>

      {/* Trans 组件：在翻译中嵌入 JSX（链接、加粗等） */}
      <p>
        <Trans i18nKey="confirm_delete" values={{ item: '2024年度报告' }}>
          确定要删除<b>「2024年度报告」</b>吗？<em>此操作不可撤销。</em>
        </Trans>
      </p>

      {/* 语言切换 */}
      <select
        value={i18n.language}
        onChange={(e) => i18n.changeLanguage(e.target.value)}
        className="rounded border px-2 py-1"
      >
        <option value="zh-CN">中文</option>
        <option value="en-US">English</option>
        <option value="ja-JP">日本語</option>
      </select>
    </div>
  );
}
```

### 9.4.3 ICU 消息语法详解

ICU（International Components for Unicode）消息格式是国际化消息的行业标准。react-intl 原生支持 ICU 语法，i18next 通过 `i18next-icu` 插件也可支持。

```
// 变量插值
'你好，{name}！'

// 复数规则（plural）
'{count, plural, =0 {没有项目} =1 {1 个项目} other {# 个项目}}'
// # 会被替换为 count 的值

// 选择规则（select）
'{gender, select, male {他} female {她} other {TA}}'

// 嵌套
'{gender, select, male {{count, plural, =1 {他有 1 本书} other {他有 # 本书}}} female {{count, plural, =1 {她有 1 本书} other {她有 # 本书}}} other {}}'

// 数字格式化
'{price, number, ::currency/USD}'
'{percent, number, ::percent scale/100}'

// 日期格式化
'{date, date, ::yyyyMMdd}'
'{date, date, long}'

// 持续时间
'{duration, number, ::duration-unit-display-narrow}'

// 富文本（HTML 标签）
'点击<a>这里</a>查看详情'
```

ICU 语法的关键价值在于：翻译人员只需处理字符串，复数规则由 ICU 引擎自动匹配，不同语言的复数形式差异（如阿拉伯语有 6 种复数形式）无需开发者手动处理。

---

## 9.5 主题系统

主题系统需要同时支持品牌色彩定制、亮色/暗色模式切换以及运行时动态更新。CSS 自定义属性（变量）是当前最优雅的实现方式。

### 9.5.1 CSS 变量主题架构

```css
/* styles/themes.css */
:root {
  /* 基础色彩系统 */
  --color-primary: #2563eb;
  --color-primary-hover: #1d4ed8;
  --color-primary-foreground: #ffffff;

  --color-secondary: #f3f4f6;
  --color-secondary-foreground: #1f2937;

  --color-background: #ffffff;
  --color-surface: #f9fafb;
  --color-surface-raised: #ffffff;

  --color-text-primary: #111827;
  --color-text-secondary: #6b7280;
  --color-text-disabled: #9ca3af;

  --color-border: #e5e7eb;
  --color-border-focus: var(--color-primary);

  --color-success: #16a34a;
  --color-warning: #d97706;
  --color-error: #dc2626;
  --color-info: #2563eb;

  /* 间距 */
  --spacing-xs: 4px;
  --spacing-sm: 8px;
  --spacing-md: 16px;
  --spacing-lg: 24px;
  --spacing-xl: 32px;

  /* 圆角 */
  --radius-sm: 4px;
  --radius-md: 8px;
  --radius-lg: 12px;
  --radius-full: 9999px;

  /* 阴影 */
  --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
  --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1);
  --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1);

  /* 字体 */
  --font-sans: 'Inter', system-ui, -apple-system, sans-serif;
  --font-mono: 'JetBrains Mono', 'Fira Code', monospace;
  --font-size-sm: 0.875rem;
  --font-size-base: 1rem;
  --font-size-lg: 1.125rem;
  --font-size-xl: 1.25rem;

  /* 动效 */
  --transition-fast: 150ms ease;
  --transition-normal: 250ms ease;
}
```

### 9.5.2 暗色模式实现

两种主流策略：基于 `prefers-color-scheme` 媒体查询（跟随系统）和基于 class 的手动切换（用户主动选择）。最佳实践是两者结合——默认跟随系统，但用户手动选择后以手动选择为准。

```css
/* styles/dark-mode.css */

/* 策略 A：跟随系统偏好 */
@media (prefers-color-scheme: dark) {
  :root {
    --color-background: #0f172a;
    --color-surface: #1e293b;
    --color-surface-raised: #334155;
    --color-text-primary: #f1f5f9;
    --color-text-secondary: #94a3b8;
    --color-text-disabled: #64748b;
    --color-border: #334155;
    --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.3);
    --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.4);
    --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.5);
  }
}

/* 策略 B：手动切换（class-based） */
[data-theme="dark"] {
  --color-background: #0f172a;
  --color-surface: #1e293b;
  --color-surface-raised: #334155;
  --color-text-primary: #f1f5f9;
  --color-text-secondary: #94a3b8;
  --color-text-disabled: #64748b;
  --color-border: #334155;
  --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.3);
  --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.4);
  --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.5);

  /* 暗色模式下的色彩微调 */
  --color-primary: #3b82f6;
  --color-primary-hover: #60a5fa;
  --color-success: #22c55e;
  --color-warning: #f59e0b;
  --color-error: #ef4444;
}
```

#### ThemeProvider 实现

```tsx
// providers/ThemeProvider.tsx
import { createContext, useContext, useEffect, useState, useCallback, type ReactNode } from 'react';

type Theme = 'light' | 'dark' | 'system';

interface ThemeContextValue {
  theme: Theme;
  resolvedTheme: 'light' | 'dark';
  setTheme: (theme: Theme) => void;
}

const ThemeContext = createContext<ThemeContextValue | null>(null);

function getSystemTheme(): 'light' | 'dark' {
  if (typeof window === 'undefined') return 'light';
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [theme, setThemeState] = useState<Theme>(() => {
    const stored = localStorage.getItem('app-theme') as Theme | null;
    return stored ?? 'system';
  });

  const resolvedTheme = theme === 'system' ? getSystemTheme() : theme;

  // 应用主题到 DOM
  useEffect(() => {
    const root = document.documentElement;
    root.setAttribute('data-theme', resolvedTheme);
    // 设置 color-scheme 让浏览器原生控件也跟随主题
    root.style.colorScheme = resolvedTheme;
    localStorage.setItem('app-theme', theme);
  }, [theme, resolvedTheme]);

  // 监听系统主题变化
  useEffect(() => {
    if (theme !== 'system') return;
    const mql = window.matchMedia('(prefers-color-scheme: dark)');
    const handler = () => {
      document.documentElement.setAttribute('data-theme', getSystemTheme());
    };
    mql.addEventListener('change', handler);
    return () => mql.removeEventListener('change', handler);
  }, [theme]);

  const setTheme = useCallback((t: Theme) => setThemeState(t), []);

  return (
    <ThemeContext.Provider value={{ theme, resolvedTheme, setTheme }}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  const ctx = useContext(ThemeContext);
  if (!ctx) throw new Error('useTheme must be used within a ThemeProvider');
  return ctx;
}
```

#### 主题切换组件

```tsx
// components/ThemeToggle.tsx
import { useTheme } from '@/providers/ThemeProvider';

export function ThemeToggle() {
  const { theme, resolvedTheme, setTheme } = useTheme();

  return (
    <div className="flex items-center gap-2 rounded-lg border p-1">
      <button
        onClick={() => setTheme('light')}
        className={`rounded px-3 py-1 text-sm transition-colors ${
          theme === 'light' ? 'bg-primary text-primary-foreground' : 'hover:bg-surface'
        }`}
        aria-pressed={theme === 'light'}
      >
        亮色
      </button>
      <button
        onClick={() => setTheme('dark')}
        className={`rounded px-3 py-1 text-sm transition-colors ${
          theme === 'dark' ? 'bg-primary text-primary-foreground' : 'hover:bg-surface'
        }`}
        aria-pressed={theme === 'dark'}
      >
        暗色
      </button>
      <button
        onClick={() => setTheme('system')}
        className={`rounded px-3 py-1 text-sm transition-colors ${
          theme === 'system' ? 'bg-primary text-primary-foreground' : 'hover:bg-surface'
        }`}
        aria-pressed={theme === 'system'}
      >
        跟随系统
      </button>
    </div>
  );
}
```

### 9.5.3 CSS 变量回退（Fallbacks）

CSS 变量的 `var()` 函数支持第二个参数作为回退值，在变量未定义时生效。这在组件库需要支持主题定制但不强制使用者定义所有变量时非常有用。

```css
/* 组件库的默认样式 + 回退 */
.my-button {
  /* 使用主题变量，未定义时回退到硬编码值 */
  background-color: var(--color-primary, #2563eb);
  color: var(--color-primary-foreground, #ffffff);
  border-radius: var(--radius-md, 8px);
  padding: var(--spacing-sm, 8px) var(--spacing-md, 16px);
  font-family: var(--font-sans, system-ui, sans-serif);
  /* 多层回退：先尝试 --color-primary-hover，再尝试 --color-primary，最后 #1d4ed8 */
  &:hover {
    background-color: var(--color-primary-hover, var(--color-primary, #1d4ed8));
  }
}

/* 带默认值的阴影：允许覆盖但不强制 */
.card {
  box-shadow: var(--card-shadow, 0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1));
}
```

---

## 9.6 无障碍访问（Accessibility, A11y）

Web 无障碍确保残障用户可以使用辅助技术（屏幕阅读器、键盘、语音控制等）正常访问 Web 应用。在多数国家，公共部门和大型企业的 Web 应用必须满足 WCAG 2.1 AA 标准。

### 9.6.1 ARIA 角色与标签

ARIA（Accessible Rich Internet Applications）通过 `role` 和 `aria-*` 属性为辅助技术提供语义信息，弥补 HTML 原生语义的不足。

```tsx
// components/AccessibleTabs.tsx
import { useState, useId } from 'react';

interface Tab {
  id: string;
  label: string;
  content: React.ReactNode;
}

export function AccessibleTabs({ tabs }: { tabs: Tab[] }) {
  const [activeTab, setActiveTab] = useState(tabs[0]?.id);
  const baseId = useId();

  return (
    <div>
      {/* tablist role 标识这是一个标签页组 */}
      <div role="tablist" aria-label="内容面板" className="flex border-b">
        {tabs.map((tab, index) => (
          <button
            key={tab.id}
            role="tab"
            id={`${baseId}-tab-${index}`}
            aria-selected={activeTab === tab.id}
            aria-controls={`${baseId}-panel-${index}`}
            tabIndex={activeTab === tab.id ? 0 : -1}
            onClick={() => setActiveTab(tab.id)}
            onKeyDown={(e) => {
              const currentIndex = tabs.findIndex((t) => t.id === activeTab);
              let nextIndex = currentIndex;
              if (e.key === 'ArrowRight') nextIndex = (currentIndex + 1) % tabs.length;
              if (e.key === 'ArrowLeft') nextIndex = (currentIndex - 1 + tabs.length) % tabs.length;
              if (e.key === 'Home') nextIndex = 0;
              if (e.key === 'End') nextIndex = tabs.length - 1;
              if (nextIndex !== currentIndex) {
                e.preventDefault();
                setActiveTab(tabs[nextIndex].id);
                document.getElementById(`${baseId}-tab-${nextIndex}`)?.focus();
              }
            }}
            className={`px-4 py-2 text-sm border-b-2 transition-colors ${
              activeTab === tab.id
                ? 'border-primary text-primary'
                : 'border-transparent text-text-secondary hover:text-text-primary'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* tabpanel 关联对应的 tab */}
      {tabs.map((tab, index) => (
        <div
          key={tab.id}
          role="tabpanel"
          id={`${baseId}-panel-${index}`}
          aria-labelledby={`${baseId}-tab-${index}`}
          hidden={activeTab !== tab.id}
          tabIndex={0}
          className="p-4"
        >
          {tab.content}
        </div>
      ))}
    </div>
  );
}
```

#### 常用 ARIA 属性速查

| 属性 | 用途 | 示例 |
|------|------|------|
| `aria-label` | 为无可见文本的元素提供标签 | `<button aria-label="关闭对话框">×</button>` |
| `aria-labelledby` | 引用另一个元素的文本作为标签 | `<div aria-labelledby="heading-id">` |
| `aria-describedby` | 引用描述性文本 | `<input aria-describedby="password-hint">` |
| `aria-expanded` | 指示可展开控件的状态 | `<button aria-expanded={open}>` |
| `aria-hidden` | 从辅助技术中隐藏元素 | `<span aria-hidden="true">🎉</span>` |
| `aria-live` | 动态内容更新通知（polite/assertive） | `<div aria-live="polite">` |
| `aria-current` | 指示当前项（如导航） | `<a aria-current="page">` |

### 9.6.2 键盘导航

所有交互元素必须可通过键盘操作。基本原则：可聚焦的元素（链接、按钮、表单控件）使用原生 HTML，自定义交互组件使用 `tabIndex` 和 `onKeyDown`。

```tsx
// components/KeyboardList.tsx —— 完全键盘可操作的自定义列表
import { useState, useRef, useEffect, type KeyboardEvent } from 'react';

interface ListItem {
  id: string;
  label: string;
}

export function KeyboardList({ items }: { items: ListItem[] }) {
  const [focusIndex, setFocusIndex] = useState(0);
  const listRef = useRef<HTMLUListElement>(null);

  const handleKeyDown = (e: KeyboardEvent<HTMLUListElement>) => {
    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault();
        setFocusIndex((prev) => Math.min(prev + 1, items.length - 1));
        break;
      case 'ArrowUp':
        e.preventDefault();
        setFocusIndex((prev) => Math.max(prev - 1, 0));
        break;
      case 'Home':
        e.preventDefault();
        setFocusIndex(0);
        break;
      case 'End':
        e.preventDefault();
        setFocusIndex(items.length - 1);
        break;
      case 'Enter':
      case ' ':
        e.preventDefault();
        console.log('Selected:', items[focusIndex].label);
        break;
    }
  };

  // 焦点跟随 focusIndex
  useEffect(() => {
    const item = listRef.current?.querySelector<HTMLElement>(
      `[data-index="${focusIndex}"]`
    );
    item?.focus();
  }, [focusIndex]);

  return (
    <ul
      ref={listRef}
      role="listbox"
      aria-label="可选择列表"
      tabIndex={0}
      onKeyDown={handleKeyDown}
      className="border rounded-lg overflow-hidden focus:outline-none focus:ring-2 focus:ring-primary"
    >
      {items.map((item, index) => (
        <li
          key={item.id}
          role="option"
          data-index={index}
          aria-selected={index === focusIndex}
          tabIndex={-1}
          className={`px-4 py-2 cursor-pointer outline-none ${
            index === focusIndex ? 'bg-primary/10 text-primary' : 'hover:bg-surface'
          }`}
        >
          {item.label}
        </li>
      ))}
    </ul>
  );
}
```

### 9.6.3 焦点管理

焦点管理确保键盘用户和屏幕阅读器用户在 UI 状态变化时不会被"丢失"。典型场景包括：打开 Modal 后焦点应移入 Modal、关闭 Modal 后焦点应恢复到触发按钮。

```tsx
// components/FocusManagedModal.tsx
import { useEffect, useRef } from 'react';

interface ModalProps {
  isOpen: boolean;
  onClose: () => void;
  title: string;
  children: React.ReactNode;
}

export function FocusManagedModal({ isOpen, onClose, title, children }: ModalProps) {
  const modalRef = useRef<HTMLDivElement>(null);
  const previousFocusRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    if (isOpen) {
      // 保存当前焦点，以便关闭后恢复
      previousFocusRef.current = document.activeElement as HTMLElement;

      // 将焦点移入 Modal
      requestAnimationFrame(() => {
        const firstFocusable = modalRef.current?.querySelector<HTMLElement>(
          'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
        );
        firstFocusable?.focus();
      });
    } else {
      // 恢复焦点到触发元素
      previousFocusRef.current?.focus();
    }
  }, [isOpen]);

  // 焦点陷阱：Tab 在 Modal 内循环
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Escape') {
      onClose();
      return;
    }
    if (e.key !== 'Tab') return;

    const focusableElements = modalRef.current?.querySelectorAll<HTMLElement>(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    );
    if (!focusableElements || focusableElements.length === 0) return;

    const first = focusableElements[0];
    const last = focusableElements[focusableElements.length - 1];

    if (e.shiftKey) {
      if (document.activeElement === first) {
        e.preventDefault();
        last.focus();
      }
    } else {
      if (document.activeElement === last) {
        e.preventDefault();
        first.focus();
      }
    }
  };

  if (!isOpen) return null;

  return (
    <>
      {/* 背景遮罩 */}
      <div className="fixed inset-0 bg-black/50 z-40" onClick={onClose} />
      {/* Modal */}
      <div
        ref={modalRef}
        role="dialog"
        aria-modal="true"
        aria-label={title}
        onKeyDown={handleKeyDown}
        className="fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 z-50
                   bg-white rounded-lg shadow-xl p-6 w-[90vw] max-w-lg focus:outline-none"
      >
        <h2 className="text-lg font-semibold">{title}</h2>
        <div className="mt-4">{children}</div>
        <button
          onClick={onClose}
          className="mt-4 px-4 py-2 bg-gray-200 rounded hover:bg-gray-300"
        >
          关闭
        </button>
      </div>
    </>
  );
}
```

### 9.6.4 axe-core 自动化测试

axe-core 是 Deque Labs 维护的无障碍检测引擎，可集成到单元测试、E2E 测试和 CI 流水线中。

```typescript
// tests/a11y.test.tsx
import { render } from '@testing-library/react';
import { axe, toHaveNoViolations } from 'jest-axe';
import { AccessibleTabs } from '@/components/AccessibleTabs';

expect.extend(toHaveNoViolations);

describe('AccessibleTabs 无障碍检查', () => {
  it('不应有无障碍违规', async () => {
    const tabs = [
      { id: '1', label: '个人信息', content: <p>个人信息内容</p> },
      { id: '2', label: '账户设置', content: <p>账户设置内容</p> },
      { id: '3', label: '通知偏好', content: <p>通知偏好内容</p> },
    ];

    const { container } = render(<AccessibleTabs tabs={tabs} />);
    const results = await axe(container);

    expect(results).toHaveNoViolations();
  });
});
```

在 Storybook 中集成 axe：

```typescript
// .storybook/preview.ts
import { withA11y } from '@storybook/addon-a11y';

export const decorators = [withA11y];
```

### 9.6.5 色彩对比度

WCAG 2.1 AA 要求普通文本（小于 18px）的色彩对比度至少为 4.5:1，大文本（大于等于 18px 或 14px 加粗）至少为 3:1。

```typescript
// utils/color-contrast.ts
export function getContrastRatio(hex1: string, hex2: string): number {
  const getLuminance = (hex: string): number => {
    const rgb = hex
      .replace('#', '')
      .match(/.{2}/g)!
      .map((c) => {
        const sRGB = parseInt(c, 16) / 255;
        return sRGB <= 0.03928 ? sRGB / 12.92 : Math.pow((sRGB + 0.055) / 1.055, 2.4);
      });
    return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
  };

  const l1 = getLuminance(hex1);
  const l2 = getLuminance(hex2);
  const lighter = Math.max(l1, l2);
  const darker = Math.min(l1, l2);

  return (lighter + 0.05) / (darker + 0.05);
}

export function meetsWCAGAA(foreground: string, background: string, isLargeText = false): boolean {
  const ratio = getContrastRatio(foreground, background);
  return isLargeText ? ratio >= 3 : ratio >= 4.5;
}

// 使用示例
const isAccessible = meetsWCAGAA('#6b7280', '#ffffff'); // false: 3.94 < 4.5
const isAccessibleLarge = meetsWCAGAA('#6b7280', '#ffffff', true); // true: 3.94 >= 3.0
```

在实际项目中，应使用 Figma 插件（如 Stark、Contrast）或浏览器 DevTools 的 contrast checker 在设计阶段就验证色彩对比度，而不是等到代码阶段再修复。

---

## 9.7 监控与错误追踪

生产环境的错误和性能数据是盲区。Sentry 提供错误追踪和性能监控，Datadog RUM 提供真实用户监控（Real User Monitoring）。

### 9.7.1 Sentry：错误追踪与性能监控

```typescript
// monitoring/sentry.config.ts
import * as Sentry from '@sentry/react';

export function initSentry() {
  Sentry.init({
    dsn: import.meta.env.VITE_SENTRY_DSN,
    environment: import.meta.env.MODE,
    release: `my-app@${import.meta.env.VITE_APP_VERSION}`,
    integrations: [
      // React Router v7 集成：自动捕获路由变化
      Sentry.reactRouterV7BrowserTracingIntegration(),
      // 将浏览器 console 输出作为 breadcrumb
      Sentry.browserConsoleIntegration(),
      // HTTP 请求监控
      Sentry.browserTracingIntegration(),
      // 重放会话录像（可回放错误发生前的用户操作）
      Sentry.replayIntegration({
        maskAllText: true,
        blockAllMedia: true,
      }),
    ],
    // 采样率
    tracesSampleRate: import.meta.env.PROD ? 0.1 : 1.0,
    replaysSessionSampleRate: import.meta.env.PROD ? 0.1 : 1.0,
    replaysOnErrorSampleRate: 1.0,
    // 过滤低价值错误
    beforeSend(event, hint) {
      const error = hint.originalException;
      if (error instanceof Error) {
        // 过滤浏览器扩展引起的错误
        if (error.message?.includes('ResizeObserver loop limit exceeded')) return null;
        if (error.message?.includes('chrome-extension://')) return null;
        // 过滤网络中断错误（用户离线）
        if (error.message?.includes('NetworkError') && !navigator.onLine) return null;
      }
      return event;
    },
    // 自定义去重
    beforeBreadcrumb(breadcrumb) {
      if (breadcrumb.category === 'console') {
        // 不上报 console.log 作为 breadcrumb
        if (breadcrumb.level === 'log') return null;
      }
      return breadcrumb;
    },
  });
}
```

#### ErrorBoundary 组件

```tsx
// components/SentryErrorBoundary.tsx
import * as Sentry from '@sentry/react';
import { useEffect } from 'react';

interface ErrorFallbackProps {
  error: Error;
  resetError: () => void;
}

function ErrorFallback({ error, resetError }: ErrorFallbackProps) {
  useEffect(() => {
    // 错误发生时自动上报
    Sentry.captureException(error, {
      tags: { component: 'ErrorBoundary' },
      extra: { url: window.location.href, timestamp: Date.now() },
    });
  }, [error]);

  return (
    <div role="alert" className="flex flex-col items-center justify-center min-h-[400px] p-8">
      <div className="max-w-md text-center">
        <h2 className="text-xl font-semibold text-text-primary mb-2">
          页面遇到了一些问题
        </h2>
        <p className="text-text-secondary mb-6">
          错误已自动上报给技术团队，我们会尽快修复。
        </p>
        <pre className="text-xs text-text-secondary bg-surface p-3 rounded mb-6 overflow-auto max-h-32 text-left">
          {error.message}
        </pre>
        <div className="flex gap-3 justify-center">
          <button
            onClick={resetError}
            className="px-4 py-2 bg-primary text-primary-foreground rounded-md hover:bg-primary-hover"
          >
            重试
          </button>
          <button
            onClick={() => window.location.reload()}
            className="px-4 py-2 border rounded-md hover:bg-surface"
          >
            刷新页面
          </button>
        </div>
      </div>
    </div>
  );
}

// 使用 Sentry 的 ErrorBoundary 包裹应用
export function SentryErrorBoundary({ children }: { children: React.ReactNode }) {
  return (
    <Sentry.ErrorBoundary fallback={ErrorFallback} showDialog>
      {children}
    </Sentry.ErrorBoundary>
  );
}
```

#### setUser 与 captureException

```typescript
// 用户登录后设置上下文
import * as Sentry from '@sentry/react';

export function setSentryUser(user: { id: string; email: string; role: string }) {
  Sentry.setUser({
    id: user.id,
    email: user.email,
    role: user.role,
  });
}

export function clearSentryUser() {
  Sentry.setUser(null);
}

// 在 catch 块中手动上报
async function fetchUserData(userId: string) {
  try {
    const response = await fetch(`/api/users/${userId}`);
    if (!response.ok) throw new Error(`API error: ${response.status}`);
    return await response.json();
  } catch (error) {
    Sentry.captureException(error, {
      tags: { feature: 'user-data' },
      extra: { userId },
      level: 'error',
    });
    throw error; // 继续向上抛出
  }
}
```

### 9.7.2 Datadog RUM：真实用户监控

Datadog RUM 从真实用户的浏览器中收集性能数据、错误日志和用户行为，帮助团队了解应用在生产环境中的真实表现。

```typescript
// monitoring/datadog.config.ts
import { datadogRum } from '@datadog/browser-rum';

export function initDatadogRUM() {
  datadogRum.init({
    applicationId: import.meta.env.VITE_DD_APPLICATION_ID,
    clientToken: import.meta.env.VITE_DD_CLIENT_TOKEN,
    site: 'datadoghq.com',
    service: 'my-react-app',
    env: import.meta.env.MODE,
    version: import.meta.env.VITE_APP_VERSION,
    sessionSampleRate: 100,
    sessionReplaySampleRate: import.meta.env.PROD ? 20 : 100,
    trackUserInteractions: true,
    trackResources: true,
    trackLongTasks: true,
    defaultPrivacyLevel: 'mask-user-input',
    allowedTracingUrls: [
      'https://api.example.com',
      'https://cdn.example.com',
    ],
  });

  // 全局上下文
  datadogRum.setGlobalContextProperty('app', {
    buildId: import.meta.env.VITE_BUILD_ID,
    region: 'cn-east-1',
  });
}

// 自定义用户操作追踪
export function trackUserAction(name: string, context?: Record<string, unknown>) {
  datadogRum.addAction(name, context);
}

// 自定义错误上报
export function trackError(error: Error, context?: Record<string, unknown>) {
  datadogRum.addError(error, context);
}

// 在组件中使用
import { trackUserAction } from '@/monitoring/datadog.config';

function CheckoutButton() {
  const handleClick = () => {
    trackUserAction('checkout_initiated', {
      cart_items: 3,
      cart_total: 299.99,
      currency: 'CNY',
    });
    // ... 实际的结账逻辑
  };

  return <button onClick={handleClick}>去结算</button>;
}
```

Sentry 侧重错误追踪和堆栈分析，Datadog RUM 侧重性能指标（Core Web Vitals、资源加载时间、用户操作耗时）和 Session Replay。两者可以同时使用，覆盖互补的监控维度。

---

## 9.8 安全防护

前端安全涉及多个层面：防止跨站脚本攻击（XSS）、防止跨站请求伪造（CSRF）、通过内容安全策略（CSP）限制资源加载来源、以及定期审计依赖中的已知漏洞。

### 9.8.1 Content Security Policy（CSP）

CSP 通过 HTTP 响应头或 `<meta>` 标签限制浏览器可以加载哪些资源，是防御 XSS 和数据注入攻击的最有效手段。

```nginx
# nginx.conf
add_header Content-Security-Policy "
  default-src 'self';
  script-src 'self' 'unsafe-inline' https://js.monitor-sdk.com;
  style-src 'self' 'unsafe-inline';
  img-src 'self' data: https: blob:;
  font-src 'self' data:;
  connect-src 'self' https://api.example.com https://sentry.io https://*.datadoghq.com;
  media-src 'self';
  frame-src 'self' https://www.youtube.com;
  frame-ancestors 'none';
  form-action 'self';
  base-uri 'self';
  object-src 'none';
  upgrade-insecure-requests;
" always;
```

关键指令说明：
- `default-src 'self'`：默认只允许同源资源
- `script-src 'unsafe-inline'`：允许内联脚本（React 的 `dangerouslySetInnerHTML` 如果启用了 `unsafe-inline`，CSP 无法阻止其执行——这正是为什么不应使用它的原因）
- `connect-src`：限制 fetch/XHR/WebSocket 的目标域名
- `frame-ancestors 'none'`：禁止被嵌入 iframe（防 clickjacking）
- `upgrade-insecure-requests`：自动将 HTTP 升级为 HTTPS

在开发阶段可以使用 `Content-Security-Policy-Report-Only` 头来收集违规报告而不实际阻断：

```nginx
add_header Content-Security-Policy-Report-Only "
  default-src 'self'; report-uri /csp-report-endpoint;
";
```

### 9.8.2 XSS 防护

React 默认会对 JSX 中的所有插值进行 HTML 转义，这是 React 最基础也最重要的 XSS 防线。

```tsx
// components/XSSDemo.tsx
export function XSSProtection() {
  // 恶意输入
  const userInput = '<img src=x onerror="alert(\'XSS\')">';

  return (
    <div>
      {/* 安全：React 自动转义 */}
      <p>{userInput}</p>
      {/* 渲染结果：&lt;img src=x onerror=&quot;alert(&#x27;XSS&#x27;)&quot;&gt; */}

      {/* 危险：直接注入 HTML —— 仅在 100% 信任内容来源时使用 */}
      {/* <div dangerouslySetInnerHTML={{ __html: userInput }} /> */}
    </div>
  );
}
```

当确实需要渲染用户提交的富文本 HTML 时，必须使用 HTML 净化库：

```tsx
import DOMPurify from 'dompurify';

interface SafeHtmlProps {
  html: string;
  tag?: keyof JSX.IntrinsicElements;
}

export function SafeHtml({ html, tag: Tag = 'div' }: SafeHtmlProps) {
  // DOMPurify 会移除所有危险标签和属性
  const clean = DOMPurify.sanitize(html, {
    ALLOWED_TAGS: ['b', 'i', 'em', 'strong', 'a', 'p', 'br', 'ul', 'ol', 'li'],
    ALLOWED_ATTR: ['href', 'target', 'rel'],
    ALLOW_DATA_ATTR: false,
  });

  return <Tag dangerouslySetInnerHTML={{ __html: clean }} />;
}
```

### 9.8.3 CSRF 防护

CSRF（Cross-Site Request Forgery）攻击利用用户在目标站点的已登录状态，在第三方站点上发起恶意请求。防御策略：

#### SameSite Cookie

```typescript
// server-side: 设置 session cookie
// Set-Cookie: sessionId=abc123; SameSite=Strict; Secure; HttpOnly
// SameSite=Strict: 完全禁止第三方站点携带此 cookie
// SameSite=Lax: 允许顶级导航（如点击链接）时携带，禁止子请求（如图片、iframe）携带
// Secure: 仅通过 HTTPS 传输
// HttpOnly: JavaScript 无法访问（防止通过 XSS 窃取 cookie）
```

#### CSRF Token 模式

```tsx
// hooks/useCsrf.ts
import { useState, useEffect } from 'react';

export function useCsrfToken() {
  const [token, setToken] = useState<string>('');

  useEffect(() => {
    // 首次加载时从服务器获取 CSRF token
    fetch('/api/csrf-token', { credentials: 'same-origin' })
      .then((res) => res.json())
      .then((data) => setToken(data.token))
      .catch(() => {
        // 回退：从 cookie 中读取（如果服务端使用了 double-submit cookie 模式）
        const match = document.cookie.match(/csrf-token=([^;]+)/);
        if (match) setToken(match[1]);
      });
  }, []);

  return token;
}

// 在 API 请求中携带 token
async function apiFetch(url: string, options: RequestInit = {}, csrfToken: string) {
  return fetch(url, {
    ...options,
    credentials: 'same-origin',
    headers: {
      'Content-Type': 'application/json',
      'X-CSRF-Token': csrfToken,
      ...options.headers,
    },
  });
}

// 使用示例
function SubmitForm() {
  const csrfToken = useCsrfToken();

  const handleSubmit = async (data: Record<string, unknown>) => {
    await apiFetch('/api/submit', {
      method: 'POST',
      body: JSON.stringify(data),
    }, csrfToken);
  };

  return <form onSubmit={(e) => { e.preventDefault(); handleSubmit({ name: 'test' }); }}>
    <button type="submit">提交</button>
  </form>;
}
```

### 9.8.4 依赖安全审计

npm 生态的依赖链非常深，一个间接依赖的漏洞可能影响整个应用。

```bash
# npm 内置审计
npm audit                  # 查看已知漏洞
npm audit --production     # 仅检查生产依赖
npm audit fix              # 自动修复（仅 semver-compatible 的更新）
npm audit fix --force      # 强制修复（可能包含 breaking changes）

# 在 CI 中集成审计（退出码非零会中断流水线）
npm audit --audit-level=high   # 仅在高危/严重漏洞时报错
npm audit --audit-level=moderate --production
```

#### Snyk：更深入的依赖分析

Snyk 不仅检查已知漏洞，还能发现许可证合规问题、提供修复建议和自动 PR。

```bash
# 安装 Snyk CLI
npm install -g snyk

# 认证
snyk auth

# 测试项目
snyk test                    # 终端输出漏洞列表
snyk test --json > snyk-report.json  # JSON 格式输出

# 持续监控（集成到 CI）
snyk monitor                 # 将依赖快照上传到 Snyk 仪表盘

# 自动修复
snyk wizard                  # 交互式修复引导
```

在 GitHub Actions 中集成 Snyk：

```yaml
# .github/workflows/security.yml
name: Security Audit

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 8 * * 1'  # 每周一早上 8 点

jobs:
  npm-audit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with: { node-version: '20', cache: 'npm' }
      - run: npm ci
      - run: npm audit --audit-level=high

  snyk:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: snyk/actions/setup@master
      - run: snyk test --severity-threshold=high
        env:
          SNYK_TOKEN: ${{ secrets.SNYK_TOKEN }}
```

---

## 9.9 本章小结

企业级 React 工程实践是一个多维度、跨领域的系统工程。本章覆盖的八个领域构成了大中型 React 应用的基础设施骨架：

1. **Monorepo 架构**：Turborepo 以缓存为核心，适合中等规模团队；Nx 以代码生成和依赖图为核心，适合大型组织和强约束场景。两者都通过拓扑编排和增量构建提升 CI 效率。

2. **微前端架构**：Webpack 5 Module Federation 通过模块级共享实现紧密协作，适合统一框架的团队；single-spa 通过应用级编排支持多框架共存，适合渐进式迁移和技术栈异构场景。

3. **设计系统**：Storybook 是组件的"实验室"和"产品目录"；Radix UI 提供无障碍原语，让你不必从零实现 aria 属性；shadcn/ui 将前两者融合为可直接复制和修改的组件源码。

4. **国际化**：react-intl 基于 ICU 消息语法，适合需要复杂复数/日期/数字格式化的场景；i18next 生态广泛，适合跨平台（Web、移动端、后端）统一翻译管理的场景。

5. **主题系统**：CSS 自定义属性 + `data-theme` 属性 + localStorage 持久化的组合，实现了系统偏好检测、手动切换和跨会话记忆的完整闭环。`var()` 的回退语法为组件库提供了主题定制的灵活性。

6. **无障碍访问**：ARIA 属性为辅助技术提供语义、键盘导航保证非鼠标用户的可用性、焦点管理防止用户在 UI 变化时"迷路"、axe-core 在 CI 中自动检测违规、色彩对比度校验确保视觉可读性。

7. **监控系统**：Sentry 专注错误追踪和调用链分析；Datadog RUM 专注真实用户性能指标和 Session Replay。两者覆盖"出错了怎么办"和"用户快不快"两个核心问题。

8. **安全防护**：CSP 在浏览器层面限制资源来源；React 的自动转义防止 XSS；SameSite Cookie 和 CSRF Token 防御跨站请求伪造；npm audit 和 Snyk 持续扫描依赖漏洞。

这些实践并非都需要在项目第一天就全部引入。建议按优先级逐步落地：安全 > 监控 > 无障碍 > 主题 > i18n > 设计系统 > Monorepo > 微前端。每个阶段的引入都应该解决明确的痛点，而非为了"技术先进性"而引入复杂度。
