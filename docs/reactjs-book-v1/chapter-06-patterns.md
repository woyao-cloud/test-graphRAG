# 第 6 章 React 高级模式与实践

在前面的章节中，我们已经掌握了 React 的核心概念：组件、状态、副作用、路由和状态管理。本章将深入探讨 React 开发中不可或缺的高级模式——这些模式是构建健壮、高性能、可维护的 React 应用的关键。

你将学到：

- 防抖（Debounce）与节流（Throttle）——控制高频事件的利器
- 请求限流（Rate Limiting）——保护 API 资源不被滥用
- 错误边界（Error Boundaries）——优雅地捕获并处理渲染错误
- Portal——将组件渲染到 DOM 树的任意位置
- 复合组件（Compound Components）——构建灵活的组件族
- Render Props 与 Hooks 的对比与迁移
- 高阶组件（HOC）——历史遗产与现代化替代方案
- Context 性能优化——避免不必要的重渲染
- Suspense 数据获取——声明式加载的未来

---

## 6.1 防抖（Debounce）

防抖是一种控制函数执行频率的技术：在高频触发的事件中，只有当事件停止触发一段时间后，函数才会执行。如果在等待期间事件再次触发，则重新计时。

### 6.1.1 防抖的原理

想象你在电梯里：每次有人按关门按钮，电梯都会重新开始等待。只有在一段时间内没有人再按按钮，门才会真正关闭。这就是防抖的核心思想——**等待安静期，然后执行**。

在 React 中，防抖最常见的应用场景是搜索输入框。用户每按一个键就发送一次 API 请求是不合理的——应该等用户停止输入后再发送请求。

### 6.1.2 搜索输入防抖

```typescript
import React, { useState, useEffect, useRef, useCallback } from 'react';

interface SearchResult {
  id: number;
  title: string;
  excerpt: string;
}

// 基础防抖搜索组件
const DebouncedSearch: React.FC = () => {
  const [query, setQuery] = useState<string>('');
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const fetchResults = useCallback(async (searchQuery: string) => {
    if (!searchQuery.trim()) {
      setResults([]);
      return;
    }

    setLoading(true);
    try {
      // 模拟 API 请求
      const response = await fetch(
        `https://api.example.com/search?q=${encodeURIComponent(searchQuery)}`
      );
      const data: SearchResult[] = await response.json();
      setResults(data);
    } catch (error) {
      console.error('搜索失败:', error);
      setResults([]);
    } finally {
      setLoading(false);
    }
  }, []);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    setQuery(value);

    // 清除之前的定时器
    if (timerRef.current) {
      clearTimeout(timerRef.current);
    }

    // 设置新的 300ms 防抖定时器
    timerRef.current = setTimeout(() => {
      fetchResults(value);
    }, 300);
  };

  // 组件卸载时清理定时器
  useEffect(() => {
    return () => {
      if (timerRef.current) {
        clearTimeout(timerRef.current);
      }
    };
  }, []);

  return (
    <div className="search-container">
      <input
        type="text"
        value={query}
        onChange={handleChange}
        placeholder="搜索文章..."
        className="search-input"
      />
      {loading && <div className="spinner">搜索中...</div>}
      <ul className="search-results">
        {results.map((item) => (
          <li key={item.id}>
            <h4>{item.title}</h4>
            <p>{item.excerpt}</p>
          </li>
        ))}
      </ul>
    </div>
  );
};
```

### 6.1.3 useDebounce 自定义 Hook

将防抖逻辑抽象为可复用的 Hook，是 React 中的最佳实践。

```typescript
import { useState, useEffect, useRef } from 'react';

/**
 * useDebounce - 对值进行防抖处理
 *
 * @param value - 需要防抖的原始值
 * @param delay - 防抖延迟时间（毫秒），默认 300ms
 * @returns 防抖后的值
 *
 * @example
 * const [query, setQuery] = useState('');
 * const debouncedQuery = useDebounce(query, 500);
 * // debouncedQuery 只在 query 停止变化 500ms 后才更新
 */
function useDebounce<T>(value: T, delay: number = 300): T {
  const [debouncedValue, setDebouncedValue] = useState<T>(value);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    // 清除之前的定时器
    if (timerRef.current) {
      clearTimeout(timerRef.current);
    }

    // 设置新的定时器
    timerRef.current = setTimeout(() => {
      setDebouncedValue(value);
    }, delay);

    // 清理函数：组件卸载或 value/delay 变化时清除定时器
    return () => {
      if (timerRef.current) {
        clearTimeout(timerRef.current);
      }
    };
  }, [value, delay]);

  return debouncedValue;
}

// 使用示例
const SearchWithHook: React.FC = () => {
  const [query, setQuery] = useState<string>('');
  const [results, setResults] = useState<SearchResult[]>([]);
  const debouncedQuery = useDebounce(query, 300);

  useEffect(() => {
    if (debouncedQuery.trim()) {
      fetchResults(debouncedQuery);
    } else {
      setResults([]);
    }
  }, [debouncedQuery]);

  const fetchResults = async (searchQuery: string) => {
    const response = await fetch(
      `https://api.example.com/search?q=${encodeURIComponent(searchQuery)}`
    );
    const data = await response.json();
    setResults(data);
  };

  return (
    <div>
      <input
        type="text"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="搜索..."
      />
      <ul>
        {results.map((item) => (
          <li key={item.id}>{item.title}</li>
        ))}
      </ul>
    </div>
  );
};
```

### 6.1.4 useDebouncedCallback

有时候我们需要防抖的是回调函数本身，而非某个值。`useDebouncedCallback` 直接返回一个防抖后的函数引用。

```typescript
import { useRef, useCallback, useEffect } from 'react';

/**
 * useDebouncedCallback - 返回防抖后的回调函数
 *
 * @param callback - 原始回调函数
 * @param delay - 防抖延迟时间（毫秒）
 * @returns 防抖后的函数
 *
 * @example
 * const debouncedSearch = useDebouncedCallback(
 *   (query: string) => api.search(query),
 *   300
 * );
 */
function useDebouncedCallback<T extends (...args: unknown[]) => unknown>(
  callback: T,
  delay: number = 300
): (...args: Parameters<T>) => void {
  const callbackRef = useRef<T>(callback);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // 保持 callback 引用最新
  useEffect(() => {
    callbackRef.current = callback;
  }, [callback]);

  // 组件卸载时清理
  useEffect(() => {
    return () => {
      if (timerRef.current) {
        clearTimeout(timerRef.current);
      }
    };
  }, []);

  const debouncedFn = useCallback(
    (...args: Parameters<T>) => {
      if (timerRef.current) {
        clearTimeout(timerRef.current);
      }
      timerRef.current = setTimeout(() => {
        callbackRef.current(...args);
      }, delay);
    },
    [delay]
  );

  return debouncedFn;
}
```

### 6.1.5 窗口大小调整防抖

`resize` 事件以极高频率触发（每秒可达 60 次以上），必须使用防抖来避免性能问题。

```typescript
import React, { useState, useEffect } from 'react';

const ResponsiveLayout: React.FC = () => {
  const [windowSize, setWindowSize] = useState({
    width: window.innerWidth,
    height: window.innerHeight,
  });
  const [layout, setLayout] = useState<'mobile' | 'tablet' | 'desktop'>('desktop');

  // 使用 useDebounce 处理 resize 事件
  const debouncedWidth = useDebounce(windowSize.width, 200);

  useEffect(() => {
    const handleResize = () => {
      setWindowSize({
        width: window.innerWidth,
        height: window.innerHeight,
      });
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  useEffect(() => {
    if (debouncedWidth < 768) {
      setLayout('mobile');
    } else if (debouncedWidth < 1024) {
      setLayout('tablet');
    } else {
      setLayout('desktop');
    }
  }, [debouncedWidth]);

  return (
    <div className={`layout layout--${layout}`}>
      <p>当前布局: {layout}</p>
      <p>窗口宽度: {debouncedWidth}px</p>
      <p>窗口高度: {windowSize.height}px</p>
    </div>
  );
};
```

### 6.1.6 防抖的注意事项

1. **不要在 render 中创建 debounced 函数**——每次渲染都会创建新的防抖实例，防抖将完全失效。
2. **使用 useRef 存储定时器引用**——确保跨渲染保持同一个定时器 ID。
3. **记得清理**——在 `useEffect` 的清理函数中调用 `clearTimeout`，防止内存泄漏和状态更新到已卸载的组件。
4. **延迟时间的选择**——搜索建议通常用 200-300ms，表单验证用 500ms，`resize` 事件用 150-200ms。

---

## 6.2 节流（Throttle）

节流与防抖类似，但策略不同：节流保证函数在指定时间间隔内**最多执行一次**，无论事件触发了多少次。

### 6.2.1 节流的原理

如果防抖是"等安静下来再执行"，节流就是"按固定节奏执行"。想象水龙头：防抖是等到水流停止才接水，节流是拧小水龙头让水匀速流出。

节流适合需要在持续触发期间保持**定期反馈**的场景，比如滚动事件中的无限加载。

### 6.2.2 滚动监听节流

```typescript
import React, { useState, useEffect, useRef } from 'react';

interface ScrollPosition {
  x: number;
  y: number;
}

const ScrollTracker: React.FC = () => {
  const [scrollPos, setScrollPos] = useState<ScrollPosition>({ x: 0, y: 0 });
  const [scrollPercent, setScrollPercent] = useState<number>(0);
  const rafIdRef = useRef<number | null>(null);

  // 使用 requestAnimationFrame 实现节流（推荐方式）
  useEffect(() => {
    const handleScroll = () => {
      // 如果已经有待处理的帧，跳过
      if (rafIdRef.current !== null) return;

      rafIdRef.current = requestAnimationFrame(() => {
        const pos: ScrollPosition = {
          x: window.scrollX,
          y: window.scrollY,
        };
        setScrollPos(pos);

        // 计算滚动百分比
        const scrollHeight = document.documentElement.scrollHeight - window.innerHeight;
        const percent = scrollHeight > 0 ? (pos.y / scrollHeight) * 100 : 0;
        setScrollPercent(Math.min(100, Math.max(0, percent)));

        rafIdRef.current = null;
      });
    };

    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => {
      window.removeEventListener('scroll', handleScroll);
      if (rafIdRef.current) {
        cancelAnimationFrame(rafIdRef.current);
      }
    };
  }, []);

  return (
    <div className="scroll-tracker">
      <div className="scroll-indicator">
        滚动位置: x={scrollPos.x}, y={scrollPos.y}
      </div>
      <div className="progress-bar">
        <div
          className="progress-bar__fill"
          style={{ width: `${scrollPercent}%` }}
        />
      </div>
      <p>已阅读: {scrollPercent.toFixed(1)}%</p>
    </div>
  );
};
```

### 6.2.3 使用 setTimeout 的传统节流

```typescript
import { useRef, useCallback, useEffect } from 'react';

/**
 * useThrottle - 返回节流后的回调函数
 *
 * @param callback - 原始回调函数
 * @param limit - 最小执行间隔（毫秒）
 * @param options - { leading?: boolean, trailing?: boolean }
 *   - leading: 是否在节流周期开始时立即执行（默认 true）
 *   - trailing: 是否在节流周期结束时执行最后一次（默认 true）
 * @returns 节流后的函数
 */
function useThrottle<T extends (...args: unknown[]) => unknown>(
  callback: T,
  limit: number = 200,
  options: { leading?: boolean; trailing?: boolean } = {}
): (...args: Parameters<T>) => void {
  const { leading = true, trailing = true } = options;
  const callbackRef = useRef<T>(callback);
  const lastRunRef = useRef<number>(0);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const trailingArgsRef = useRef<Parameters<T> | null>(null);

  useEffect(() => {
    callbackRef.current = callback;
  }, [callback]);

  useEffect(() => {
    return () => {
      if (timerRef.current) {
        clearTimeout(timerRef.current);
      }
    };
  }, []);

  const throttledFn = useCallback(
    (...args: Parameters<T>) => {
      const now = Date.now();
      const elapsed = now - lastRunRef.current;

      // 存储 trailing 参数
      trailingArgsRef.current = args;

      if (elapsed >= limit) {
        // 可以立即执行
        if (leading) {
          callbackRef.current(...args);
        }
        lastRunRef.current = now;

        // 设置 trailing 定时器
        if (trailing && timerRef.current === null) {
          timerRef.current = setTimeout(() => {
            lastRunRef.current = leading ? Date.now() : 0;
            timerRef.current = null;
            if (trailing && trailingArgsRef.current && !leading) {
              callbackRef.current(...trailingArgsRef.current);
            }
          }, limit);
        }
      } else if (trailing && timerRef.current === null) {
        // 节流期间，设置 trailing 定时器
        timerRef.current = setTimeout(() => {
          lastRunRef.current = leading ? Date.now() : 0;
          timerRef.current = null;
          if (trailingArgsRef.current) {
            callbackRef.current(...trailingArgsRef.current);
            trailingArgsRef.current = null;
          }
        }, limit - elapsed);
      }
    },
    [limit, leading, trailing]
  );

  return throttledFn;
}

// 使用示例
const ScrollLogger: React.FC = () => {
  const handleScroll = useThrottle(
    () => {
      console.log('节流滚动位置:', window.scrollY);
    },
    200,
    { leading: true, trailing: true }
  );

  useEffect(() => {
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, [handleScroll]);

  return <div style={{ height: '300vh' }}>向下滚动查看节流效果</div>;
};
```

### 6.2.4 无限滚动（Infinite Scroll）

无限滚动是节流的经典应用场景——在用户滚动到页面底部附近时加载更多数据。

```typescript
import React, { useState, useEffect, useRef, useCallback } from 'react';

interface Post {
  id: number;
  title: string;
  body: string;
}

const InfiniteScrollList: React.FC = () => {
  const [posts, setPosts] = useState<Post[]>([]);
  const [page, setPage] = useState<number>(1);
  const [loading, setLoading] = useState<boolean>(false);
  const [hasMore, setHasMore] = useState<boolean>(true);
  const sentinelRef = useRef<HTMLDivElement | null>(null);
  const observerRef = useRef<IntersectionObserver | null>(null);

  // 加载数据
  const loadPosts = useCallback(async (pageNum: number) => {
    setLoading(true);
    try {
      const response = await fetch(
        `https://jsonplaceholder.typicode.com/posts?_page=${pageNum}&_limit=10`
      );
      const data: Post[] = await response.json();

      if (data.length === 0) {
        setHasMore(false);
      } else {
        setPosts((prev) => [...prev, ...data]);
      }
    } catch (error) {
      console.error('加载失败:', error);
    } finally {
      setLoading(false);
    }
  }, []);

  // 初始加载
  useEffect(() => {
    loadPosts(1);
  }, [loadPosts]);

  // IntersectionObserver 检测哨兵元素
  useEffect(() => {
    if (observerRef.current) {
      observerRef.current.disconnect();
    }

    observerRef.current = new IntersectionObserver(
      (entries) => {
        const [entry] = entries;
        if (entry.isIntersecting && hasMore && !loading) {
          setPage((prev) => prev + 1);
        }
      },
      {
        rootMargin: '200px', // 提前 200px 触发，提升体验
        threshold: 0,
      }
    );

    if (sentinelRef.current) {
      observerRef.current.observe(sentinelRef.current);
    }

    return () => {
      observerRef.current?.disconnect();
    };
  }, [hasMore, loading]);

  // 页码变化时加载
  useEffect(() => {
    if (page > 1) {
      loadPosts(page);
    }
  }, [page, loadPosts]);

  return (
    <div className="infinite-scroll-container">
      <h2>文章列表</h2>
      <ul className="post-list">
        {posts.map((post) => (
          <li key={post.id} className="post-card">
            <h3>{post.title}</h3>
            <p>{post.body.substring(0, 100)}...</p>
          </li>
        ))}
      </ul>

      {/* 哨兵元素——当它进入视口时触发加载 */}
      <div ref={sentinelRef} className="scroll-sentinel" />

      {loading && (
        <div className="loading-indicator">
          <div className="spinner" />
          <span>加载中...</span>
        </div>
      )}

      {!hasMore && (
        <p className="end-message">—— 没有更多内容了 ——</p>
      )}
    </div>
  );
};
```

### 6.2.5 防抖 vs 节流：选择指南

| 场景 | 推荐技术 | 原因 |
|------|----------|------|
| 搜索输入 | 防抖 (300ms) | 等用户停止输入后才请求 |
| 窗口 resize | 防抖 (150ms) | 只在调整结束后重新计算布局 |
| 滚动事件（进度条） | 节流 (100ms) | 需要持续更新但不需要每帧都更新 |
| 按钮点击（防重复提交） | 节流 (1000ms) | 在冷却时间内禁止重复点击 |
| 无限滚动 | 节流 + IntersectionObserver | 检测到底部时加载，但限制频率 |
| 表单自动保存 | 防抖 (2000ms) | 等用户停止编辑后保存 |
| 鼠标移动跟踪 | 节流 (50ms) | 需要平滑跟踪但限制更新频率 |

---

## 6.3 请求限流（Rate Limiting）

当需要大量并发 API 请求时——比如批量上传、实时协作编辑、或者需要遵守 API 的速率限制——我们需要在前端进行请求限流。

### 6.3.1 基础限流器

```typescript
interface QueuedRequest<T> {
  execute: () => Promise<T>;
  resolve: (value: T) => void;
  reject: (error: unknown) => void;
  controller: AbortController;
}

class RateLimiter {
  private queue: QueuedRequest<unknown>[] = [];
  private activeCount: number = 0;

  constructor(
    private maxConcurrent: number = 3,
    private intervalMs: number = 1000,
    private maxPerInterval: number = 10
  ) {}

  /**
   * 添加一个请求到队列
   */
  enqueue<T>(execute: () => Promise<T>, signal?: AbortSignal): Promise<T> {
    const controller = new AbortController();

    // 支持外部 AbortSignal
    if (signal) {
      signal.addEventListener('abort', () => controller.abort());
    }

    return new Promise<T>((resolve, reject) => {
      const queued: QueuedRequest<T> = {
        execute,
        resolve,
        reject,
        controller,
      };

      controller.signal.addEventListener('abort', () => {
        const index = this.queue.indexOf(queued as QueuedRequest<unknown>);
        if (index !== -1) {
          this.queue.splice(index, 1);
          reject(new DOMException('请求已被取消', 'AbortError'));
        }
      });

      this.queue.push(queued as QueuedRequest<unknown>);
      this.processQueue();
    });
  }

  /**
   * 处理队列中的请求
   */
  private processQueue(): void {
    while (this.activeCount < this.maxConcurrent && this.queue.length > 0) {
      const request = this.queue.shift()!;
      this.activeCount++;

      request
        .execute()
        .then((result) => {
          request.resolve(result);
        })
        .catch((error) => {
          request.reject(error);
        })
        .finally(() => {
          this.activeCount--;
          this.processQueue();
        });
    }
  }

  /**
   * 清空队列
   */
  clear(): void {
    this.queue.forEach((req) => {
      req.reject(new Error('限流器已清空'));
    });
    this.queue = [];
  }

  /**
   * 获取队列状态
   */
  getStatus(): { queued: number; active: number } {
    return {
      queued: this.queue.length,
      active: this.activeCount,
    };
  }
}

// 使用示例
const apiLimiter = new RateLimiter(3, 1000, 10);

const BatchUploader: React.FC = () => {
  const [files, setFiles] = useState<File[]>([]);
  const [progress, setProgress] = useState<Record<string, 'pending' | 'uploading' | 'done' | 'error'>>({});

  const handleUpload = async () => {
    const uploadTasks = files.map((file) =>
      apiLimiter.enqueue(async () => {
        setProgress((prev) => ({ ...prev, [file.name]: 'uploading' }));

        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch('/api/upload', {
          method: 'POST',
          body: formData,
        });

        if (!response.ok) {
          throw new Error(`上传失败: ${response.statusText}`);
        }

        setProgress((prev) => ({ ...prev, [file.name]: 'done' }));
        return response.json();
      })
    );

    const results = await Promise.allSettled(uploadTasks);
    console.log('上传结果:', results);
  };

  return (
    <div>
      <input
        type="file"
        multiple
        onChange={(e) => setFiles(Array.from(e.target.files || []))}
      />
      <button onClick={handleUpload}>上传</button>
      <ul>
        {Object.entries(progress).map(([name, status]) => (
          <li key={name}>
            {name}: {status}
          </li>
        ))}
      </ul>
    </div>
  );
};
```

### 6.3.2 AbortController 取消请求

React 19 中，`fetch` 原生支持 `AbortController`。在组件卸载或新请求发起时取消旧请求，是避免竞态条件（race condition）的关键。

```typescript
import React, { useState, useEffect, useRef } from 'react';

const AbortableSearch: React.FC = () => {
  const [query, setQuery] = useState<string>('');
  const [results, setResults] = useState<string[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const abortControllerRef = useRef<AbortController | null>(null);

  const debouncedQuery = useDebounce(query, 300);

  useEffect(() => {
    if (!debouncedQuery.trim()) {
      setResults([]);
      return;
    }

    // 取消上一个请求
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    // 创建新的 AbortController
    const controller = new AbortController();
    abortControllerRef.current = controller;

    setLoading(true);

    fetch(`/api/search?q=${encodeURIComponent(debouncedQuery)}`, {
      signal: controller.signal,
    })
      .then((res) => {
        if (!res.ok) throw new Error('请求失败');
        return res.json();
      })
      .then((data) => {
        setResults(data);
        setLoading(false);
      })
      .catch((err) => {
        // AbortError 是正常行为，不需要处理
        if (err.name !== 'AbortError') {
          console.error('搜索出错:', err);
          setLoading(false);
        }
      });

    return () => {
      controller.abort();
    };
  }, [debouncedQuery]);

  return (
    <div>
      <input
        type="text"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="输入搜索关键词..."
      />
      {loading && <span>搜索中...</span>}
      <ul>
        {results.map((item, idx) => (
          <li key={idx}>{item}</li>
        ))}
      </ul>
    </div>
  );
};
```

### 6.3.3 useRateLimiter Hook

将限流逻辑封装为自定义 Hook，方便在组件中复用。

```typescript
import { useRef, useCallback } from 'react';

interface UseRateLimiterOptions {
  maxRequests: number;
  intervalMs: number;
}

interface UseRateLimiterReturn {
  /** 尝试执行请求，如果达到限制则返回 false */
  tryRequest: () => boolean;
  /** 获取剩余可用请求数 */
  remaining: () => number;
  /** 重置计数器 */
  reset: () => void;
}

function useRateLimiter(options: UseRateLimiterOptions): UseRateLimiterReturn {
  const { maxRequests, intervalMs } = options;
  const timestampsRef = useRef<number[]>([]);

  const cleanOldTimestamps = useCallback(() => {
    const now = Date.now();
    const windowStart = now - intervalMs;
    timestampsRef.current = timestampsRef.current.filter((t) => t > windowStart);
  }, [intervalMs]);

  const tryRequest = useCallback((): boolean => {
    cleanOldTimestamps();
    if (timestampsRef.current.length < maxRequests) {
      timestampsRef.current.push(Date.now());
      return true;
    }
    return false;
  }, [maxRequests, cleanOldTimestamps]);

  const remaining = useCallback((): number => {
    cleanOldTimestamps();
    return maxRequests - timestampsRef.current.length;
  }, [maxRequests, cleanOldTimestamps]);

  const reset = useCallback(() => {
    timestampsRef.current = [];
  }, []);

  return { tryRequest, remaining, reset };
}

// 使用示例：限制每分钟最多 30 次 API 调用
const RateLimitedComponent: React.FC = () => {
  const { tryRequest, remaining } = useRateLimiter({
    maxRequests: 30,
    intervalMs: 60_000,
  });

  const handleClick = async () => {
    if (!tryRequest()) {
      alert(`请求过于频繁，请稍后再试。剩余可用次数: ${remaining()}`);
      return;
    }

    try {
      const response = await fetch('/api/data');
      const data = await response.json();
      console.log('数据:', data);
    } catch (error) {
      console.error('请求失败:', error);
    }
  };

  return (
    <div>
      <p>剩余可用请求: {remaining()}</p>
      <button onClick={handleClick}>发送请求</button>
    </div>
  );
};
```

---

## 6.4 错误边界（Error Boundaries）

React 错误边界是一种特殊的组件，用于捕获其子组件树中抛出的 JavaScript 错误，显示降级 UI（fallback UI），防止整个应用崩溃。

### 6.4.1 错误边界的原理

错误边界只能捕获以下场景中的错误：

- **渲染期间**（render）
- **生命周期方法中**
- **子组件树的构造函数中**

以下场景的错误**不能**被错误边界捕获：

- 事件处理器中的错误（需要用 `try/catch`）
- 异步代码（`setTimeout`、`requestAnimationFrame`）
- 服务端渲染（SSR）
- 错误边界组件自身抛出的错误

### 6.4.2 class 组件错误边界

截至 React 19，错误边界仍然需要使用 class 组件实现，因为 React 没有为函数组件提供等价的 Hooks（`componentDidCatch` 和 `getDerivedStateFromError` 没有对应的 Hook）。

```typescript
import React, { Component, ErrorInfo, ReactNode } from 'react';

interface ErrorBoundaryProps {
  children: ReactNode;
  fallback?: ReactNode | ((error: Error, reset: () => void) => ReactNode);
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
  onReset?: () => void;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
}

class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
    };
  }

  /**
   * getDerivedStateFromError 在 render 阶段调用，
   * 用于更新 state 以显示降级 UI。
   * 这里不应该产生副作用（如日志上报）。
   */
  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return {
      hasError: true,
      error,
    };
  }

  /**
   * componentDidCatch 在 commit 阶段调用，
   * 适合用于副作用，如错误日志上报。
   */
  componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    // 上报错误到监控系统
    console.error('ErrorBoundary 捕获到错误:', error);
    console.error('组件栈:', errorInfo.componentStack);

    this.props.onError?.(error, errorInfo);
  }

  /**
   * 重置错误状态，允许子组件重新渲染
   */
  handleReset = (): void => {
    this.props.onReset?.();
    this.setState({
      hasError: false,
      error: null,
    });
  };

  render(): ReactNode {
    if (this.state.hasError) {
      // 支持自定义 fallback
      if (typeof this.props.fallback === 'function') {
        return this.props.fallback(this.state.error!, this.handleReset);
      }

      if (this.props.fallback) {
        return this.props.fallback;
      }

      // 默认降级 UI
      return (
        <DefaultErrorFallback
          error={this.state.error!}
          onReset={this.handleReset}
        />
      );
    }

    return this.props.children;
  }
}

// 默认降级 UI 组件
const DefaultErrorFallback: React.FC<{
  error: Error;
  onReset: () => void;
}> = ({ error, onReset }) => {
  return (
    <div
      role="alert"
      style={{
        padding: '2rem',
        margin: '1rem',
        border: '1px solid #f5c6cb',
        borderRadius: '8px',
        backgroundColor: '#fff3f4',
        textAlign: 'center',
      }}
    >
      <h2 style={{ color: '#d32f2f', marginBottom: '0.5rem' }}>
        页面遇到了问题
      </h2>
      <p style={{ color: '#666', marginBottom: '0.5rem' }}>
        {error.message}
      </p>
      <details style={{ marginBottom: '1rem', textAlign: 'left' }}>
        <summary>错误详情</summary>
        <pre
          style={{
            fontSize: '0.85rem',
            color: '#333',
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-word',
          }}
        >
          {error.stack}
        </pre>
      </details>
      <button
        onClick={onReset}
        style={{
          padding: '0.5rem 1.5rem',
          backgroundColor: '#d32f2f',
          color: '#fff',
          border: 'none',
          borderRadius: '4px',
          cursor: 'pointer',
          fontSize: '1rem',
        }}
      >
        重试
      </button>
    </div>
  );
};
```

### 6.4.3 错误边界的使用与组合

在实际项目中，错误边界应该被精细地放置在关键位置，而不是只在根节点包裹一层。

```typescript
import React from 'react';

// 产品卡片——每个卡片独立保护
const ProductCard: React.FC<{ productId: string }> = ({ productId }) => {
  return (
    <ErrorBoundary
      fallback={
        <div className="product-card product-card--error">
          无法加载产品信息
        </div>
      }
      onError={(error) => {
        // 上报到监控系统
        reportError('ProductCard', productId, error);
      }}
    >
      <ProductCardContent productId={productId} />
    </ErrorBoundary>
  );
};

// 侧边栏——独立保护，即使出错也不影响主内容区
const Sidebar: React.FC = () => {
  return (
    <ErrorBoundary
      fallback={<div className="sidebar sidebar--error">侧边栏加载失败</div>}
    >
      <aside className="sidebar">
        <UserWidget />
        <RecentPosts />
        <TagCloud />
      </aside>
    </ErrorBoundary>
  );
};

// 整个页面布局——多层错误边界
const AppLayout: React.FC = () => {
  return (
    <ErrorBoundary
      fallback={(error, reset) => (
        <div className="app-crash">
          <h1>应用遇到了严重错误</h1>
          <p>{error.message}</p>
          <button onClick={reset}>重新加载</button>
          <button onClick={() => window.location.reload()}>
            刷新页面
          </button>
        </div>
      )}
      onReset={() => {
        // 清理缓存
        localStorage.clear();
      }}
    >
      <div className="app-layout">
        <ErrorBoundary fallback={<div>导航加载失败</div>}>
          <Navbar />
        </ErrorBoundary>

        <div className="app-layout__body">
          <ErrorBoundary fallback={<SidebarFallback />}>
            <Sidebar />
          </ErrorBoundary>

          <main className="app-layout__main">
            <ErrorBoundary
              fallback={(error, reset) => (
                <div>
                  <p>内容区出错了</p>
                  <button onClick={reset}>重试</button>
                </div>
              )}
            >
              <Outlet />
            </ErrorBoundary>
          </main>
        </div>
      </div>
    </ErrorBoundary>
  );
};

// 错误上报辅助函数
function reportError(
  component: string,
  context: string,
  error: Error
): void {
  // 发送到 Sentry / DataDog / 自建监控
  console.error(`[${component}] 错误上下文: ${context}`, error);
  // fetch('/api/errors', {
  //   method: 'POST',
  //   body: JSON.stringify({ component, context, message: error.message, stack: error.stack }),
  // });
}
```

### 6.4.4 错误边界的局限性

对于 class 组件无法覆盖的场景，我们仍然需要手动处理：

```typescript
import React, { useState } from 'react';

const AsyncErrorHandler: React.FC = () => {
  const [error, setError] = useState<Error | null>(null);

  const handleAsyncAction = async () => {
    try {
      setError(null);
      const response = await fetch('/api/dangerous-operation');
      if (!response.ok) {
        throw new Error(`请求失败: ${response.status}`);
      }
      const data = await response.json();
      console.log('成功:', data);
    } catch (err) {
      setError(err instanceof Error ? err : new Error('未知错误'));
    }
  };

  // 事件处理器中的错误需要用 try/catch
  const handleClick = () => {
    try {
      // 可能抛出错误的同步代码
      JSON.parse('{ invalid json }');
    } catch (err) {
      setError(err instanceof Error ? err : new Error('解析失败'));
    }
  };

  if (error) {
    return (
      <div className="error-container">
        <p>操作失败: {error.message}</p>
        <button onClick={() => setError(null)}>关闭</button>
      </div>
    );
  }

  return (
    <div>
      <button onClick={handleClick}>同步操作</button>
      <button onClick={handleAsyncAction}>异步操作</button>
    </div>
  );
};
```

---

## 6.5 Portal

Portal 允许将子组件渲染到父组件 DOM 树之外的 DOM 节点中。这在处理模态框（Modal）、下拉菜单（Dropdown）、提示框（Tooltip）等需要"逃逸"父组件 CSS 限制的场景中至关重要。

### 6.5.1 createPortal 基础

```typescript
import { createPortal } from 'react-dom';

// React 19 中 createPortal 的签名
// createPortal(children, domNode, key?)
```

### 6.5.2 模态框（Modal）

```typescript
import React, { useEffect, useRef, useCallback } from 'react';
import { createPortal } from 'react-dom';

interface ModalProps {
  isOpen: boolean;
  onClose: () => void;
  title?: string;
  children: React.ReactNode;
  /** 点击遮罩层是否关闭，默认 true */
  closeOnOverlay?: boolean;
}

const Modal: React.FC<ModalProps> = ({
  isOpen,
  onClose,
  title,
  children,
  closeOnOverlay = true,
}) => {
  const overlayRef = useRef<HTMLDivElement>(null);
  const previousFocusRef = useRef<HTMLElement | null>(null);

  // ESC 键关闭
  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose();
        return;
      }

      // 焦点陷阱：Tab 键循环
      if (e.key === 'Tab' && overlayRef.current) {
        const focusableElements = overlayRef.current.querySelectorAll<HTMLElement>(
          'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
        );
        const first = focusableElements[0];
        const last = focusableElements[focusableElements.length - 1];

        if (e.shiftKey) {
          if (document.activeElement === first) {
            e.preventDefault();
            last?.focus();
          }
        } else {
          if (document.activeElement === last) {
            e.preventDefault();
            first?.focus();
          }
        }
      }
    },
    [onClose]
  );

  // 管理焦点和滚动锁定
  useEffect(() => {
    if (isOpen) {
      previousFocusRef.current = document.activeElement as HTMLElement;
      document.addEventListener('keydown', handleKeyDown);
      document.body.style.overflow = 'hidden';

      // 聚焦到模态框
      requestAnimationFrame(() => {
        overlayRef.current?.focus();
      });
    }

    return () => {
      document.removeEventListener('keydown', handleKeyDown);
      document.body.style.overflow = '';
      previousFocusRef.current?.focus();
    };
  }, [isOpen, handleKeyDown]);

  if (!isOpen) return null;

  return createPortal(
    <div
      ref={overlayRef}
      className="modal-overlay"
      onClick={(e) => {
        if (closeOnOverlay && e.target === e.currentTarget) {
          onClose();
        }
      }}
      role="dialog"
      aria-modal="true"
      aria-label={title || '对话框'}
      tabIndex={-1}
      style={{
        position: 'fixed',
        inset: 0,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        backgroundColor: 'rgba(0, 0, 0, 0.5)',
        zIndex: 1000,
      }}
    >
      <div
        className="modal-content"
        style={{
          background: '#fff',
          borderRadius: '8px',
          padding: '1.5rem',
          maxWidth: '500px',
          width: '90%',
          maxHeight: '80vh',
          overflowY: 'auto',
          position: 'relative',
          boxShadow: '0 4px 24px rgba(0, 0, 0, 0.15)',
        }}
      >
        <div
          className="modal-header"
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            marginBottom: '1rem',
          }}
        >
          <h2 style={{ margin: 0 }}>{title}</h2>
          <button
            onClick={onClose}
            aria-label="关闭"
            style={{
              background: 'none',
              border: 'none',
              fontSize: '1.5rem',
              cursor: 'pointer',
              padding: '0.25rem',
              lineHeight: 1,
            }}
          >
            &times;
          </button>
        </div>
        <div className="modal-body">{children}</div>
      </div>
    </div>,
    document.body
  );
};

// 使用示例
const ModalExample: React.FC = () => {
  const [isOpen, setIsOpen] = React.useState(false);

  return (
    <div>
      <button onClick={() => setIsOpen(true)}>打开模态框</button>
      <Modal
        isOpen={isOpen}
        onClose={() => setIsOpen(false)}
        title="确认操作"
      >
        <p>确定要执行此操作吗？</p>
        <div style={{ display: 'flex', gap: '0.5rem', marginTop: '1rem' }}>
          <button onClick={() => setIsOpen(false)}>取消</button>
          <button
            onClick={() => {
              console.log('确认执行');
              setIsOpen(false);
            }}
            style={{ backgroundColor: '#d32f2f', color: '#fff' }}
          >
            确认
          </button>
        </div>
      </Modal>
    </div>
  );
};
```

### 6.5.3 下拉菜单（Dropdown）与提示框（Tooltip）

```typescript
import React, { useState, useRef, useEffect } from 'react';
import { createPortal } from 'react-dom';

interface DropdownMenuProps {
  trigger: React.ReactNode;
  items: { label: string; onClick: () => void; disabled?: boolean }[];
}

const DropdownMenu: React.FC<DropdownMenuProps> = ({ trigger, items }) => {
  const [isOpen, setIsOpen] = useState(false);
  const triggerRef = useRef<HTMLDivElement>(null);
  const [position, setPosition] = useState({ top: 0, left: 0 });

  const updatePosition = () => {
    if (triggerRef.current) {
      const rect = triggerRef.current.getBoundingClientRect();
      setPosition({
        top: rect.bottom + 4,
        left: rect.left,
      });
    }
  };

  useEffect(() => {
    if (isOpen) {
      updatePosition();
      const handleClickOutside = (e: MouseEvent) => {
        if (
          triggerRef.current &&
          !triggerRef.current.contains(e.target as Node) &&
          !(e.target as HTMLElement).closest('.dropdown-portal')
        ) {
          setIsOpen(false);
        }
      };
      window.addEventListener('scroll', updatePosition, true);
      document.addEventListener('mousedown', handleClickOutside);
      return () => {
        window.removeEventListener('scroll', updatePosition, true);
        document.removeEventListener('mousedown', handleClickOutside);
      };
    }
  }, [isOpen]);

  return (
    <>
      <div
        ref={triggerRef}
        onClick={() => setIsOpen(!isOpen)}
        style={{ cursor: 'pointer', display: 'inline-block' }}
      >
        {trigger}
      </div>

      {isOpen &&
        createPortal(
          <div
            className="dropdown-portal"
            style={{
              position: 'fixed',
              top: position.top,
              left: position.left,
              backgroundColor: '#fff',
              border: '1px solid #ddd',
              borderRadius: '6px',
              boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)',
              minWidth: '160px',
              zIndex: 1001,
              padding: '0.25rem 0',
            }}
          >
            {items.map((item, idx) => (
              <button
                key={idx}
                onClick={() => {
                  item.onClick();
                  setIsOpen(false);
                }}
                disabled={item.disabled}
                style={{
                  display: 'block',
                  width: '100%',
                  padding: '0.5rem 1rem',
                  border: 'none',
                  background: 'none',
                  textAlign: 'left',
                  cursor: item.disabled ? 'not-allowed' : 'pointer',
                  opacity: item.disabled ? 0.5 : 1,
                  fontSize: '0.9rem',
                }}
                onMouseEnter={(e) => {
                  (e.target as HTMLButtonElement).style.backgroundColor = '#f0f0f0';
                }}
                onMouseLeave={(e) => {
                  (e.target as HTMLButtonElement).style.backgroundColor = '';
                }}
              >
                {item.label}
              </button>
            ))}
          </div>,
          document.body
        )}
    </>
  );
};

// Tooltip 组件
interface TooltipProps {
  content: React.ReactNode;
  children: React.ReactNode;
  delay?: number;
}

const Tooltip: React.FC<TooltipProps> = ({
  content,
  children,
  delay = 300,
}) => {
  const [isVisible, setIsVisible] = useState(false);
  const [position, setPosition] = useState({ top: 0, left: 0 });
  const triggerRef = useRef<HTMLDivElement>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const show = () => {
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => {
      if (triggerRef.current) {
        const rect = triggerRef.current.getBoundingClientRect();
        setPosition({
          top: rect.top - 8,
          left: rect.left + rect.width / 2,
        });
        setIsVisible(true);
      }
    }, delay);
  };

  const hide = () => {
    if (timerRef.current) clearTimeout(timerRef.current);
    setIsVisible(false);
  };

  useEffect(() => {
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, []);

  return (
    <>
      <div
        ref={triggerRef}
        onMouseEnter={show}
        onMouseLeave={hide}
        onFocus={show}
        onBlur={hide}
        style={{ display: 'inline-block' }}
      >
        {children}
      </div>

      {isVisible &&
        createPortal(
          <div
            role="tooltip"
            style={{
              position: 'fixed',
              top: position.top,
              left: position.left,
              transform: 'translate(-50%, -100%)',
              backgroundColor: '#333',
              color: '#fff',
              padding: '0.25rem 0.75rem',
              borderRadius: '4px',
              fontSize: '0.85rem',
              whiteSpace: 'nowrap',
              zIndex: 2000,
              pointerEvents: 'none',
            }}
          >
            {content}
            {/* 小三角箭头 */}
            <div
              style={{
                position: 'absolute',
                bottom: '-4px',
                left: '50%',
                transform: 'translateX(-50%)',
                width: 0,
                height: 0,
                borderLeft: '4px solid transparent',
                borderRight: '4px solid transparent',
                borderTop: '4px solid #333',
              }}
            />
          </div>,
          document.body
        )}
    </>
  );
};
```

### 6.5.4 Portal 中的事件冒泡

一个关键特性：Portal 中的 React 事件会沿着 **React 组件树**向上冒泡，而不是沿着 DOM 树。这意味着即使 Portal 将 DOM 节点渲染到 `document.body`，其事件仍然可以被父组件捕获。

```typescript
import React, { useState } from 'react';
import { createPortal } from 'react-dom';

const PortalEventBubbling: React.FC = () => {
  const [clicks, setClicks] = useState(0);

  return (
    <div
      onClick={() => setClicks((c) => c + 1)}
      style={{
        padding: '2rem',
        border: '2px solid #1976d2',
        borderRadius: '8px',
      }}
    >
      <h3>父组件（捕获 Portal 中的事件）</h3>
      <p>点击计数: {clicks}</p>

      <PortalChild />

      <p style={{ fontSize: '0.85rem', color: '#666' }}>
        即使 PortalChild 的 DOM 节点在 body 下，
        <br />
        点击它仍然会触发父组件的 onClick
      </p>
    </div>
  );
};

const PortalChild: React.FC = () => {
  return createPortal(
    <button
      style={{
        position: 'fixed',
        bottom: '2rem',
        right: '2rem',
        padding: '1rem',
        backgroundColor: '#1976d2',
        color: '#fff',
        border: 'none',
        borderRadius: '4px',
        cursor: 'pointer',
      }}
    >
      点击我——事件会冒泡到父组件
    </button>,
    document.body
  );
};
```

---

## 6.6 复合组件（Compound Components）

复合组件模式允许父组件和子组件通过隐式共享状态来协作，而不需要通过 props 逐层传递。这种模式提供了极大的灵活性——使用者可以自由排列和组合子组件。

### 6.6.1 Tabs 复合组件

```typescript
import React, {
  createContext,
  useContext,
  useState,
  useCallback,
  ReactNode,
} from 'react';

// ============ Context ============

interface TabsContextValue {
  activeTab: string;
  setActiveTab: (tab: string) => void;
}

const TabsContext = createContext<TabsContextValue | null>(null);

function useTabsContext(): TabsContextValue {
  const context = useContext(TabsContext);
  if (!context) {
    throw new Error(
      'Tabs 子组件必须在 <Tabs> 内部使用。' +
      '请确保 <TabList>, <Tab>, <TabPanel> 是 <Tabs> 的直接或间接子组件。'
    );
  }
  return context;
}

// ============ Tabs（父组件）============

interface TabsProps {
  defaultTab: string;
  onChange?: (tab: string) => void;
  children: ReactNode;
}

const Tabs: React.FC<TabsProps> = ({ defaultTab, onChange, children }) => {
  const [activeTab, setActiveTab] = useState(defaultTab);

  const handleTabChange = useCallback(
    (tab: string) => {
      setActiveTab(tab);
      onChange?.(tab);
    },
    [onChange]
  );

  return (
    <TabsContext.Provider
      value={{ activeTab, setActiveTab: handleTabChange }}
    >
      <div className="tabs">{children}</div>
    </TabsContext.Provider>
  );
};

// ============ TabList ============

interface TabListProps {
  children: ReactNode;
  className?: string;
}

const TabList: React.FC<TabListProps> = ({ children, className }) => {
  return (
    <div
      className={`tabs__list ${className || ''}`}
      role="tablist"
      style={{
        display: 'flex',
        borderBottom: '2px solid #e0e0e0',
        gap: '0',
      }}
    >
      {children}
    </div>
  );
};

// ============ Tab ============

interface TabProps {
  /** Tab 的唯一标识 */
  id: string;
  children: ReactNode;
  disabled?: boolean;
}

const Tab: React.FC<TabProps> = ({ id, children, disabled = false }) => {
  const { activeTab, setActiveTab } = useTabsContext();
  const isActive = activeTab === id;

  return (
    <button
      role="tab"
      aria-selected={isActive}
      aria-controls={`panel-${id}`}
      id={`tab-${id}`}
      disabled={disabled}
      onClick={() => !disabled && setActiveTab(id)}
      style={{
        padding: '0.75rem 1.5rem',
        border: 'none',
        borderBottom: isActive ? '2px solid #1976d2' : '2px solid transparent',
        backgroundColor: 'transparent',
        color: isActive ? '#1976d2' : disabled ? '#ccc' : '#666',
        cursor: disabled ? 'not-allowed' : 'pointer',
        fontSize: '0.95rem',
        fontWeight: isActive ? 600 : 400,
        marginBottom: '-2px',
        transition: 'all 0.2s ease',
      }}
    >
      {children}
    </button>
  );
};

// ============ TabPanel ============

interface TabPanelProps {
  /** 对应 Tab 的 id */
  tabId: string;
  children: ReactNode;
}

const TabPanel: React.FC<TabPanelProps> = ({ tabId, children }) => {
  const { activeTab } = useTabsContext();
  const isActive = activeTab === tabId;

  if (!isActive) return null;

  return (
    <div
      role="tabpanel"
      id={`panel-${tabId}`}
      aria-labelledby={`tab-${tabId}`}
      style={{ padding: '1rem 0' }}
    >
      {children}
    </div>
  );
};

// ============ 使用示例 ============

const TabsExample: React.FC = () => {
  return (
    <Tabs defaultTab="account" onChange={(tab) => console.log('切换到:', tab)}>
      <TabList>
        <Tab id="account">账号设置</Tab>
        <Tab id="profile">个人资料</Tab>
        <Tab id="notifications">通知偏好</Tab>
        <Tab id="billing" disabled>账单（即将上线）</Tab>
      </TabList>

      <TabPanel tabId="account">
        <h4>账号设置</h4>
        <p>在这里管理你的账号安全和登录方式。</p>
        <label>
          邮箱:
          <input type="email" defaultValue="user@example.com" />
        </label>
      </TabPanel>

      <TabPanel tabId="profile">
        <h4>个人资料</h4>
        <p>编辑你的公开个人信息。</p>
        <label>
          昵称:
          <input type="text" defaultValue="User" />
        </label>
      </TabPanel>

      <TabPanel tabId="notifications">
        <h4>通知偏好</h4>
        <p>选择你希望接收的通知类型。</p>
        <label>
          <input type="checkbox" defaultChecked /> 邮件通知
        </label>
      </TabPanel>
    </Tabs>
  );
};

// ============ 导出 ============

export { Tabs, TabList, Tab, TabPanel };
export type { TabsProps, TabListProps, TabProps, TabPanelProps };
```

### 6.6.2 复合组件的优势

复合组件模式的真正威力在于**灵活性**——使用者可以自由控制子组件的排列和包装方式，而无需通过 props 传递配置。

```typescript
// 灵活性演示：在 TabList 中插入额外元素
const CustomTabsLayout: React.FC = () => {
  return (
    <Tabs defaultTab="general">
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
        }}
      >
        <TabList>
          <Tab id="general">通用</Tab>
          <Tab id="advanced">高级</Tab>
        </TabList>
        {/* 在 TabList 旁边插入一个操作按钮 */}
        <button
          style={{
            padding: '0.4rem 0.8rem',
            fontSize: '0.85rem',
            cursor: 'pointer',
          }}
        >
          导出设置
        </button>
      </div>

      <div style={{ marginTop: '1rem' }}>
        <TabPanel tabId="general">
          <p>通用设置内容...</p>
        </TabPanel>
        <TabPanel tabId="advanced">
          <p>高级设置内容...</p>
        </TabPanel>
      </div>
    </Tabs>
  );
};
```

---

## 6.7 Render Props 与 Hooks

Render Props 是 React 早期的代码复用模式——通过一个返回 React 元素的函数 prop 来共享逻辑。自 React 16.8 引入 Hooks 以来，绝大多数 Render Props 的场景都可以用更简洁的 Hooks 替代。

### 6.7.1 Render Props 模式

```typescript
import React, { Component, ReactNode } from 'react';

// ============ Render Props 方式 ============

interface MousePosition {
  x: number;
  y: number;
}

interface MouseTrackerRenderProps {
  render: (position: MousePosition) => ReactNode;
}

class MouseTracker extends Component<MouseTrackerRenderProps> {
  state: MousePosition = { x: 0, y: 0 };

  handleMouseMove = (e: MouseEvent) => {
    this.setState({ x: e.clientX, y: e.clientY });
  };

  componentDidMount() {
    window.addEventListener('mousemove', this.handleMouseMove);
  }

  componentWillUnmount() {
    window.removeEventListener('mousemove', this.handleMouseMove);
  }

  render() {
    return this.props.render(this.state);
  }
}

// 使用 Render Props
const RenderPropsUsage: React.FC = () => {
  return (
    <MouseTracker
      render={({ x, y }) => (
        <div
          style={{
            position: 'fixed',
            top: y + 10,
            left: x + 10,
            padding: '0.5rem',
            backgroundColor: 'rgba(0, 0, 0, 0.7)',
            color: '#fff',
            borderRadius: '4px',
            pointerEvents: 'none',
            fontSize: '0.85rem',
          }}
        >
          鼠标位置: ({x}, {y})
        </div>
      )}
    />
  );
};
```

### 6.7.2 迁移到 Hooks

```typescript
import { useState, useEffect } from 'react';

// ============ Hooks 方式（推荐）============

interface MousePosition {
  x: number;
  y: number;
}

function useMousePosition(): MousePosition {
  const [position, setPosition] = useState<MousePosition>({ x: 0, y: 0 });

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      setPosition({ x: e.clientX, y: e.clientY });
    };

    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);

  return position;
}

// 使用 Hooks——代码更简洁，没有额外的组件层级
const HooksUsage: React.FC = () => {
  const { x, y } = useMousePosition();

  return (
    <div
      style={{
        position: 'fixed',
        top: y + 10,
        left: x + 10,
        padding: '0.5rem',
        backgroundColor: 'rgba(0, 0, 0, 0.7)',
        color: '#fff',
        borderRadius: '4px',
        pointerEvents: 'none',
        fontSize: '0.85rem',
      }}
    >
      鼠标位置: ({x}, {y})
    </div>
  );
};
```

### 6.7.3 何时仍使用 Render Props

尽管 Hooks 解决了大多数场景，Render Props 在以下情况下仍有优势：

1. **需要在 JSX 中内联渲染逻辑**——Render Props 可以直接在 JSX 中内联，而 Hooks 必须提取到单独组件。
2. **需要条件性渲染不同 UI**——当同一逻辑需要渲染多种不同的 UI 时。
3. **第三方库的 API 约定**——某些库（如 Formik v2 之前）使用 Render Props 作为主要 API。

```typescript
// 场景：同一数据需要渲染为表格和图表
interface DataFetcherProps {
  url: string;
  children: (state: {
    data: unknown[];
    loading: boolean;
    error: Error | null;
    refetch: () => void;
  }) => ReactNode;
}

const DataFetcher: React.FC<DataFetcherProps> = ({ url, children }) => {
  const [data, setData] = useState<unknown[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(url);
      const json = await res.json();
      setData(json);
    } catch (err) {
      setError(err instanceof Error ? err : new Error('请求失败'));
    } finally {
      setLoading(false);
    }
  }, [url]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  return <>{children({ data, loading, error, refetch: fetchData })}</>;
};

// 使用——同一数据渲染为不同形式
const DataDashboard: React.FC = () => {
  return (
    <DataFetcher url="/api/sales">
      {({ data, loading, error }) => {
        if (loading) return <div>加载中...</div>;
        if (error) return <div>错误: {error.message}</div>;
        return (
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
            {/* 表格视图 */}
            <div>
              <h3>数据表格</h3>
              <table>
                <tbody>
                  {data.map((row: any, i) => (
                    <tr key={i}>
                      <td>{row.name}</td>
                      <td>{row.value}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {/* 图表视图 */}
            <div>
              <h3>数据图表</h3>
              {/* 这里放置图表组件 */}
              <pre>{JSON.stringify(data, null, 2)}</pre>
            </div>
          </div>
        );
      }}
    </DataFetcher>
  );
};
```

### 6.7.4 对比总结

| 方面 | Render Props | Hooks |
|------|-------------|-------|
| 代码量 | 较多（需要包装组件） | 较少 |
| 组件层级 | 增加嵌套层级 | 无额外层级 |
| TypeScript 支持 | 复杂（泛型 props） | 良好 |
| 学习曲线 | 中等 | 较低 |
| 组合性 | 好（嵌套组合） | 极好（自由组合） |
| 调试 | 困难（嵌套地狱） | 容易（扁平调用） |
| 何时使用 | 第三方库、需要内联渲染逻辑 | **默认选择** |

---

## 6.8 高阶组件（Higher-Order Components，HOC）

高阶组件是一个函数，接收一个组件作为参数，返回一个新的增强组件。它是 React 早期代码复用的主要方式，但在 Hooks 出现后，大多数 HOC 场景都可以用 Hooks 更优雅地实现。

### 6.8.1 withLogger HOC

```typescript
import React, { ComponentType, useEffect } from 'react';

/**
 * withLogger - 为组件添加 props 变化日志
 *
 * 这是一个典型的 HOC：接收组件，返回增强后的组件。
 * 在现代 React 中，这种功能更适合用 useDebugValue 或 React DevTools 实现。
 */
function withLogger<P extends object>(
  WrappedComponent: ComponentType<P>,
  componentName: string = WrappedComponent.displayName || 'Component'
): React.FC<P> {
  const WithLogger: React.FC<P> = (props) => {
    useEffect(() => {
      console.log(`[${componentName}] mounted with props:`, props);
      return () => {
        console.log(`[${componentName}] unmounted`);
      };
    }, []); // eslint-disable-line react-hooks/exhaustive-deps

    useEffect(() => {
      console.log(`[${componentName}] props updated:`, props);
    });

    return <WrappedComponent {...props} />;
  };

  WithLogger.displayName = `withLogger(${componentName})`;
  return WithLogger;
}

// 使用示例
interface UserCardProps {
  name: string;
  email: string;
}

const UserCard: React.FC<UserCardProps> = ({ name, email }) => (
  <div className="user-card">
    <h3>{name}</h3>
    <p>{email}</p>
  </div>
);

const LoggedUserCard = withLogger(UserCard, 'UserCard');

// 等价于 Hooks 版本——更简洁
function useLogger(componentName: string, props: Record<string, unknown>): void {
  useEffect(() => {
    console.log(`[${componentName}] mounted with props:`, props);
    return () => console.log(`[${componentName}] unmounted`);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    console.log(`[${componentName}] props updated:`, props);
  });
}
```

### 6.8.2 withAuth HOC

```typescript
import React, { ComponentType } from 'react';

interface AuthContextType {
  isAuthenticated: boolean;
  user: { id: string; name: string; role: string } | null;
  login: (credentials: { username: string; password: string }) => Promise<void>;
  logout: () => void;
}

const AuthContext = React.createContext<AuthContextType | null>(null);

// HOC 版本
function withAuth<P extends object>(
  WrappedComponent: ComponentType<P & { auth: AuthContextType }>
): React.FC<P> {
  const WithAuth: React.FC<P> = (props) => {
    const auth = React.useContext(AuthContext);

    if (!auth) {
      throw new Error('withAuth 必须在 AuthProvider 内部使用');
    }

    if (!auth.isAuthenticated) {
      return (
        <div style={{ textAlign: 'center', padding: '2rem' }}>
          <p>请先登录</p>
          <button onClick={() => auth.login({ username: '', password: '' })}>
            登录
          </button>
        </div>
      );
    }

    return <WrappedComponent {...props} auth={auth} />;
  };

  WithAuth.displayName = `withAuth(${
    WrappedComponent.displayName || 'Component'
  })`;
  return WithAuth;
}

// Hooks 版本——更直接、更灵活
function useAuth(): AuthContextType {
  const auth = React.useContext(AuthContext);
  if (!auth) {
    throw new Error('useAuth 必须在 AuthProvider 内部使用');
  }
  return auth;
}

// Hooks 版本的使用
const Dashboard: React.FC = () => {
  const { user, isAuthenticated, logout } = useAuth();

  if (!isAuthenticated) {
    return <div>请登录后查看</div>;
  }

  return (
    <div>
      <h2>欢迎, {user?.name}</h2>
      <button onClick={logout}>退出</button>
    </div>
  );
};
```

### 6.8.3 withTheme HOC

```typescript
import React, { ComponentType } from 'react';

interface Theme {
  primary: string;
  secondary: string;
  background: string;
  text: string;
  border: string;
}

const lightTheme: Theme = {
  primary: '#1976d2',
  secondary: '#dc004e',
  background: '#ffffff',
  text: '#333333',
  border: '#e0e0e0',
};

const darkTheme: Theme = {
  primary: '#90caf9',
  secondary: '#f48fb1',
  background: '#121212',
  text: '#ffffff',
  border: '#333333',
};

const ThemeContext = React.createContext<{
  theme: Theme;
  toggleTheme: () => void;
}>({
  theme: lightTheme,
  toggleTheme: () => {},
});

// HOC 版本
function withTheme<P extends { theme?: Theme; toggleTheme?: () => void }>(
  WrappedComponent: ComponentType<P>
): React.FC<Omit<P, 'theme' | 'toggleTheme'>> {
  const WithTheme: React.FC<Omit<P, 'theme' | 'toggleTheme'>> = (props) => {
    const { theme, toggleTheme } = React.useContext(ThemeContext);

    return (
      <WrappedComponent
        {...(props as P)}
        theme={theme}
        toggleTheme={toggleTheme}
      />
    );
  };

  WithTheme.displayName = `withTheme(${
    WrappedComponent.displayName || 'Component'
  })`;
  return WithTheme;
}

// Hooks 版本——推荐
function useTheme() {
  return React.useContext(ThemeContext);
}

// 使用 Hooks 版本
const ThemedButton: React.FC<{ label: string; onClick: () => void }> = ({
  label,
  onClick,
}) => {
  const { theme } = useTheme();

  return (
    <button
      onClick={onClick}
      style={{
        backgroundColor: theme.primary,
        color: '#fff',
        border: `1px solid ${theme.border}`,
        padding: '0.5rem 1rem',
        borderRadius: '4px',
        cursor: 'pointer',
      }}
    >
      {label}
    </button>
  );
};
```

### 6.8.4 为什么 Hooks 更受青睐

| HOC 的问题 | Hooks 的解决方案 |
|-----------|-----------------|
| 包装地狱（Wrapper Hell）——多层 HOC 嵌套导致组件层级过深 | Hooks 是扁平调用，无额外组件层级 |
| Props 命名冲突——多个 HOC 可能注入同名的 props | 每个 Hook 返回独立的值，由调用者命名 |
| 静态方法丢失——HOC 不会自动复制被包装组件的静态属性 | 不涉及组件包装，无此问题 |
| TypeScript 类型复杂——泛型推导困难 | 类型推导简单自然 |
| 难以组合——HOC 组合顺序影响结果 | Hooks 组合顺序明确且可预测 |

---

## 6.9 Context 性能优化

Context 是 React 中共享状态的重要机制，但它有一个关键的性能陷阱：**当 Context 的 value 发生变化时，所有消费该 Context 的组件都会重渲染**，即使它们只使用了 value 中的一小部分。

### 6.9.1 Context 导致的不必要重渲染

```typescript
import React, { createContext, useContext } from 'react';

// 问题演示：一个 Context 包含多个不相关的状态
interface AppState {
  user: { name: string; avatar: string } | null;
  theme: 'light' | 'dark';
  notifications: number;
}

const AppContext = createContext<AppState | null>(null);

// 这个组件只需要 theme，但它会在 user 变化时也重渲染
const ThemeDisplay: React.FC = React.memo(() => {
  const state = useContext(AppContext);
  const rendersRef = React.useRef(0);
  rendersRef.current++;

  return (
    <div>
      当前主题: {state?.theme}
      <br />
      <small>(渲染次数: {rendersRef.current})</small>
    </div>
  );
});
```

### 6.9.2 Context 拆分（Context Splitting）

将一个大 Context 拆分为多个小 Context，每个 Context 只管理相关的状态。

```typescript
import React, {
  createContext,
  useContext,
  useState,
  useMemo,
  ReactNode,
} from 'react';

// ============ 拆分为独立的 Context ============

interface User {
  name: string;
  avatar: string;
}

// Context 1: 用户信息
const UserContext = createContext<{
  user: User | null;
  setUser: (user: User | null) => void;
} | null>(null);

// Context 2: 主题
const ThemeContext = createContext<{
  theme: 'light' | 'dark';
  toggleTheme: () => void;
} | null>(null);

// Context 3: 通知计数
const NotificationContext = createContext<{
  count: number;
  increment: () => void;
  clear: () => void;
} | null>(null);

// ============ Provider ============

const AppProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [theme, setTheme] = useState<'light' | 'dark'>('light');
  const [count, setCount] = useState(0);

  // 每个 Context value 独立 memo
  const userValue = useMemo(() => ({ user, setUser }), [user]);
  const themeValue = useMemo(
    () => ({
      theme,
      toggleTheme: () => setTheme((t) => (t === 'light' ? 'dark' : 'light')),
    }),
    [theme]
  );
  const notificationValue = useMemo(
    () => ({
      count,
      increment: () => setCount((c) => c + 1),
      clear: () => setCount(0),
    }),
    [count]
  );

  return (
    <UserContext.Provider value={userValue}>
      <ThemeContext.Provider value={themeValue}>
        <NotificationContext.Provider value={notificationValue}>
          {children}
        </NotificationContext.Provider>
      </ThemeContext.Provider>
    </UserContext.Provider>
  );
};

// ============ 专用 Hooks ============

function useUser() {
  const ctx = useContext(UserContext);
  if (!ctx) throw new Error('useUser 必须在 AppProvider 内使用');
  return ctx;
}

function useTheme() {
  const ctx = useContext(ThemeContext);
  if (!ctx) throw new Error('useTheme 必须在 AppProvider 内使用');
  return ctx;
}

function useNotifications() {
  const ctx = useContext(NotificationContext);
  if (!ctx) throw new Error('useNotifications 必须在 AppProvider 内使用');
  return ctx;
}

// ============ 使用——每个组件只订阅自己需要的 Context ============

const ThemeDisplayOptimized: React.FC = React.memo(() => {
  const { theme, toggleTheme } = useTheme(); // 只订阅 ThemeContext
  const rendersRef = React.useRef(0);
  rendersRef.current++;

  return (
    <div>
      <p>当前主题: {theme}</p>
      <button onClick={toggleTheme}>切换主题</button>
      <small>(渲染次数: {rendersRef.current})</small>
    </div>
  );
});

const UserDisplay: React.FC = React.memo(() => {
  const { user } = useUser(); // 只订阅 UserContext
  return <div>{user ? `欢迎, ${user.name}` : '未登录'}</div>;
});

const NotificationBadge: React.FC = React.memo(() => {
  const { count } = useNotifications(); // 只订阅 NotificationContext
  return <span>未读通知: {count}</span>;
});
```

### 6.9.3 分离 State 和 Dispatch Context

进一步优化：将状态和操作分离到不同的 Context，让只执行操作的组件不因状态变化而重渲染。

```typescript
import React, { createContext, useContext, useState, useMemo, useCallback } from 'react';

interface Todo {
  id: string;
  text: string;
  completed: boolean;
}

// 分离状态和操作的 Context——进一步提升性能
const TodoStateContext = createContext<Todo[]>([]);
const TodoDispatchContext = createContext<{
  addTodo: (text: string) => void;
  toggleTodo: (id: string) => void;
  removeTodo: (id: string) => void;
} | null>(null);

const TodoProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [todos, setTodos] = useState<Todo[]>([]);

  // 使用 useCallback 保持操作函数的引用稳定
  const addTodo = useCallback((text: string) => {
    setTodos((prev) => [
      ...prev,
      { id: crypto.randomUUID(), text, completed: false },
    ]);
  }, []);

  const toggleTodo = useCallback((id: string) => {
    setTodos((prev) =>
      prev.map((todo) =>
        todo.id === id ? { ...todo, completed: !todo.completed } : todo
      )
    );
  }, []);

  const removeTodo = useCallback((id: string) => {
    setTodos((prev) => prev.filter((todo) => todo.id !== id));
  }, []);

  // dispatch 对象用 useMemo 稳定引用
  const dispatchValue = useMemo(
    () => ({ addTodo, toggleTodo, removeTodo }),
    [addTodo, toggleTodo, removeTodo]
  );

  return (
    <TodoStateContext.Provider value={todos}>
      <TodoDispatchContext.Provider value={dispatchValue}>
        {children}
      </TodoDispatchContext.Provider>
    </TodoStateContext.Provider>
  );
};

// 只读数据的组件——只订阅 state Context
const TodoList: React.FC = React.memo(() => {
  const todos = useContext(TodoStateContext);
  return (
    <ul>
      {todos.map((todo) => (
        <li key={todo.id}>{todo.text}</li>
      ))}
    </ul>
  );
});

// 只执行操作的组件——只订阅 dispatch Context
const AddTodoButton: React.FC = React.memo(() => {
  const { addTodo } = useContext(TodoDispatchContext)!;
  return <button onClick={() => addTodo('New Todo')}>添加</button>;
});
```

### 6.9.4 Context 性能优化清单

1. **拆分 Context**——将一个大 Context 拆为多个小 Context，按职责分离。
2. **分离 state 和 dispatch**——只读数据的组件订阅 state Context，只执行操作的组件订阅 dispatch Context。
3. **useMemo 稳定 value 引用**——确保 Context value 对象不会在每次渲染时重新创建。
4. **useCallback 稳定函数引用**——dispatch 函数使用 useCallback 保持引用不变。
5. **React.memo 包裹消费者**——对于纯展示组件，使用 React.memo 避免因父组件渲染导致的不必要重渲染。
6. **考虑替代方案**——对于频繁更新的全局状态（如主题、认证），考虑使用 Zustand 或 Jotai 等外部状态管理库，它们天然具有更细粒度的订阅机制。

---

## 6.10 Suspense 数据获取

React Suspense 最初是为代码分割（`React.lazy`）设计的，但在 React 19 中，它已正式支持数据获取场景。Suspense 引入了一种声明式的加载状态处理方式。

### 6.10.1 三种数据获取模式

在 React 中获取数据有三种模式：

| 模式 | 描述 | 优缺点 |
|------|------|--------|
| **Fetch-on-Render** | 组件挂载后开始获取数据 | 最简单，但会导致"请求瀑布" |
| **Fetch-then-Render** | 先获取所有数据，再渲染 | 消除瀑布，但必须等所有数据就绪 |
| **Render-as-You-Fetch** | 开始获取数据的同时开始渲染 | 最优，需要 Suspense 支持 |

### 6.10.2 Fetch-on-Render（传统方式）

```typescript
import React, { useState, useEffect } from 'react';

interface User {
  id: number;
  name: string;
}

interface Post {
  id: number;
  title: string;
}

// 问题：先渲染 UserProfile，它 fetch user 数据，
// 然后才渲染 UserPosts，它再 fetch posts。
// 两个请求是串行的——这就是"请求瀑布"。
const FetchOnRender: React.FC = () => {
  const [userId, setUserId] = useState(1);

  return (
    <div>
      <select
        value={userId}
        onChange={(e) => setUserId(Number(e.target.value))}
      >
        <option value={1}>User 1</option>
        <option value={2}>User 2</option>
      </select>
      <UserProfile userId={userId} />
      <UserPosts userId={userId} />
    </div>
  );
};

const UserProfile: React.FC<{ userId: number }> = ({ userId }) => {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    fetch(`/api/users/${userId}`)
      .then((r) => r.json())
      .then((data) => {
        setUser(data);
        setLoading(false);
      });
  }, [userId]);

  if (loading) return <div>加载用户信息...</div>;
  if (!user) return <div>用户不存在</div>;
  return <h2>{user.name}</h2>;
};

const UserPosts: React.FC<{ userId: number }> = ({ userId }) => {
  const [posts, setPosts] = useState<Post[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    fetch(`/api/users/${userId}/posts`)
      .then((r) => r.json())
      .then((data) => {
        setPosts(data);
        setLoading(false);
      });
  }, [userId]);

  if (loading) return <div>加载文章...</div>;
  return (
    <ul>
      {posts.map((p) => (
        <li key={p.id}>{p.title}</li>
      ))}
    </ul>
  );
};
```

### 6.10.3 Render-as-You-Fetch（Suspense 方式）

```typescript
import React, { Suspense } from 'react';

// ============ 数据获取包装器 ============

/**
 * 创建一个 Suspense-ready 的资源包装器
 *
 * 这个包装器在调用 read() 时如果数据未就绪就"抛出"Promise，
 * 让 React Suspense 捕获并显示 fallback UI。
 * 当 Promise resolve 后，React 重新渲染组件。
 */
function createResource<T>(promise: Promise<T>): {
  read: () => T;
} {
  let status: 'pending' | 'success' | 'error' = 'pending';
  let result: T;
  let error: unknown;

  const suspender = promise.then(
    (data) => {
      status = 'success';
      result = data;
    },
    (err) => {
      status = 'error';
      error = err;
    }
  );

  return {
    read(): T {
      switch (status) {
        case 'pending':
          throw suspender; // 抛出 Promise，让 Suspense 捕获
        case 'error':
          throw error; // 抛出错误，让 ErrorBoundary 捕获
        case 'success':
          return result;
        default:
          throw new Error('未知状态');
      }
    },
  };
}

// ============ 模拟 API ============

function fetchUser(id: number): Promise<{ id: number; name: string }> {
  return new Promise((resolve) =>
    setTimeout(() => resolve({ id, name: `用户 ${id}` }), 1000)
  );
}

function fetchPosts(
  userId: number
): Promise<{ id: number; title: string }[]> {
  return new Promise((resolve) =>
    setTimeout(
      () =>
        resolve([
          { id: 1, title: `用户 ${userId} 的文章 A` },
          { id: 2, title: `用户 ${userId} 的文章 B` },
        ]),
      1500
    )
  );
}

// ============ 缓存——避免重复请求 ============

const resourceCache = new Map<string, ReturnType<typeof createResource>>();

function getOrCreateResource<T>(
  key: string,
  fetcher: () => Promise<T>
): { read: () => T } {
  if (!resourceCache.has(key)) {
    resourceCache.set(key, createResource(fetcher()));
  }
  return resourceCache.get(key)! as { read: () => T };
}

// ============ Suspense 数据组件 ============

const SuspenseUserProfile: React.FC<{ userId: number }> = ({ userId }) => {
  // 在 render 阶段就开始获取数据（render-as-you-fetch）
  const resource = getOrCreateResource(`user-${userId}`, () =>
    fetchUser(userId)
  );
  const user = resource.read(); // 如果数据未就绪，这里会 throw Promise

  return (
    <div>
      <h2>{user.name}</h2>
      <p>ID: {user.id}</p>
    </div>
  );
};

const SuspenseUserPosts: React.FC<{ userId: number }> = ({ userId }) => {
  const resource = getOrCreateResource(`posts-${userId}`, () =>
    fetchPosts(userId)
  );
  const posts = resource.read();

  return (
    <ul>
      {posts.map((post) => (
        <li key={post.id}>
          <h4>{post.title}</h4>
        </li>
      ))}
    </ul>
  );
};

// ============ 页面组装 ============

const SuspenseDataPage: React.FC = () => {
  const [userId, setUserId] = React.useState(1);

  return (
    <div>
      <select
        value={userId}
        onChange={(e) => {
          // 切换用户时清除缓存
          resourceCache.clear();
          setUserId(Number(e.target.value));
        }}
      >
        <option value={1}>User 1</option>
        <option value={2}>User 2</option>
      </select>

      <ErrorBoundary
        fallback={(error, reset) => (
          <div style={{ color: 'red' }}>
            <p>数据加载失败: {error.message}</p>
            <button onClick={reset}>重试</button>
          </div>
        )}
      >
        <Suspense
          fallback={
            <div style={{ padding: '1rem' }}>
              <div className="skeleton skeleton--text" />
              <div className="skeleton skeleton--text skeleton--short" />
            </div>
          }
        >
          <section>
            <SuspenseUserProfile userId={userId} />
          </section>

          <section>
            <Suspense
              fallback={
                <div style={{ padding: '1rem' }}>
                  <p>正在加载文章列表...</p>
                </div>
              }
            >
              <SuspenseUserPosts userId={userId} />
            </Suspense>
          </section>
        </Suspense>
      </ErrorBoundary>
    </div>
  );
};
```

### 6.10.4 React 19 中 use() Hook 的使用

React 19 引入了 `use()` Hook，它可以在渲染期间读取 Promise 和 Context。当 Promise 未 resolve 时，`use()` 会暂停组件渲染（与 Suspense 配合）。

```typescript
import React, { Suspense, use, createContext } from 'react';

// ============ 使用 use() 读取 Context（React 19）============

const LocaleContext = createContext<'zh-CN' | 'en-US'>('zh-CN');

const Greeting: React.FC = () => {
  // React 19: use() 可以在条件语句中使用 Context
  const locale = use(LocaleContext);
  const greeting = locale === 'zh-CN' ? '你好，世界！' : 'Hello, World!';
  return <h1>{greeting}</h1>;
};

// ============ 使用 use() 读取 Promise（React 19）============

async function fetchDashboardData(): Promise<{
  stats: { users: number; revenue: number };
}> {
  // 模拟 API 请求
  await new Promise((r) => setTimeout(r, 800));
  return { stats: { users: 1234, revenue: 56789 } };
}

// 在模块顶层启动请求（避免请求瀑布）
const dashboardPromise = fetchDashboardData();

const DashboardStats: React.FC = () => {
  // use() 读取 Promise——如果未 resolve，暂停渲染
  const data = use(dashboardPromise);

  return (
    <div className="dashboard-stats">
      <div className="stat-card">
        <h3>用户数</h3>
        <p>{data.stats.users.toLocaleString()}</p>
      </div>
      <div className="stat-card">
        <h3>收入</h3>
        <p>${data.stats.revenue.toLocaleString()}</p>
      </div>
    </div>
  );
};

// ============ 页面 ============

const Dashboard: React.FC = () => {
  return (
    <ErrorBoundary
      fallback={(error) => (
        <div className="error">
          <h2>仪表盘加载失败</h2>
          <p>{error.message}</p>
        </div>
      )}
    >
      <Suspense
        fallback={
          <div className="loading-skeleton">
            <div className="skeleton skeleton--card" />
            <div className="skeleton skeleton--card" />
          </div>
        }
      >
        <DashboardStats />
      </Suspense>
    </ErrorBoundary>
  );
};
```

### 6.10.5 Suspense 与 Transition 结合

使用 `useTransition` 可以在数据切换期间保持旧 UI 可见，避免显示 Suspense fallback 造成的闪烁。

```typescript
import React, { Suspense, useState, useTransition } from 'react';

const SmoothTabSwitcher: React.FC = () => {
  const [tab, setTab] = useState<string>('overview');
  const [isPending, startTransition] = useTransition();

  const handleTabChange = (newTab: string) => {
    // 使用 transition 包装状态更新
    // 在数据加载期间保持旧内容可见
    startTransition(() => {
      setTab(newTab);
    });
  };

  return (
    <div>
      <nav style={{ display: 'flex', gap: '0.5rem', marginBottom: '1rem' }}>
        {['overview', 'analytics', 'settings'].map((t) => (
          <button
            key={t}
            onClick={() => handleTabChange(t)}
            disabled={isPending}
            style={{
              padding: '0.5rem 1rem',
              backgroundColor: tab === t ? '#1976d2' : '#e0e0e0',
              color: tab === t ? '#fff' : '#333',
              border: 'none',
              borderRadius: '4px',
              cursor: isPending ? 'wait' : 'pointer',
              opacity: isPending ? 0.7 : 1,
            }}
          >
            {t === 'overview' ? '概览' : t === 'analytics' ? '分析' : '设置'}
          </button>
        ))}
      </nav>

      {isPending && (
        <div
          style={{
            height: '2px',
            backgroundColor: '#1976d2',
            animation: 'loading-bar 1s ease-in-out infinite',
            marginBottom: '1rem',
          }}
        />
      )}

      <Suspense
        fallback={
          <div style={{ padding: '2rem', textAlign: 'center' }}>
            加载中...
          </div>
        }
      >
        <TabContent tab={tab} />
      </Suspense>
    </div>
  );
};

// 模拟的 Tab 内容组件
const TabContent: React.FC<{ tab: string }> = ({ tab }) => {
  const resource = getOrCreateResource(`tab-${tab}`, () =>
    new Promise<{ content: string }>((resolve) =>
      setTimeout(
        () =>
          resolve({
            content: `这是 "${tab}" 标签页的内容。数据加载时间: ${Date.now()}`,
          }),
        1200
      )
    )
  );
  const data = resource.read();

  return (
    <div style={{ padding: '1rem', backgroundColor: '#f5f5f5', borderRadius: '4px' }}>
      {data.content}
    </div>
  );
};
```

### 6.10.6 Suspense 最佳实践

1. **在模块顶层启动数据请求**——不要等到组件渲染时才开始请求。在模块作用域或路由加载器中启动请求，让请求与代码加载并行进行。
2. **Suspense 边界要合理**——不要只在根节点放一个 Suspense。将 Suspense 放在需要独立加载状态的组件附近。
3. **ErrorBoundary 和 Suspense 配合使用**——Suspense 处理"加载中"，ErrorBoundary 处理"加载失败"。
4. **使用 useTransition 避免闪烁**——在切换数据时使用 Transition 保持旧 UI 可见。
5. **缓存请求结果**——避免在组件重新挂载时重复请求相同数据。

---

## 本章小结

本章涵盖了 React 开发中的十大高级模式与实践：

| 模式 | 核心用途 | 关键 API |
|------|----------|----------|
| **防抖（Debounce）** | 等待安静期后执行 | `setTimeout` / `clearTimeout` |
| **节流（Throttle）** | 按固定频率执行 | `requestAnimationFrame` / `setTimeout` |
| **请求限流** | 控制 API 调用频率 | `RateLimiter` / `AbortController` |
| **错误边界** | 捕获渲染错误，显示降级 UI | `componentDidCatch` / `getDerivedStateFromError` |
| **Portal** | 渲染到 DOM 树任意位置 | `createPortal` |
| **复合组件** | 隐式共享状态的灵活组件族 | `createContext` / `useContext` |
| **Render Props** | 通过函数 prop 共享逻辑 | Render Props 函数 |
| **高阶组件（HOC）** | 包装组件注入功能 | 组件包装函数 |
| **Context 优化** | 避免不必要重渲染 | Context 拆分 / `useMemo` / `React.memo` |
| **Suspense** | 声明式加载状态 | `<Suspense>` / `use()` / `useTransition` |

掌握这些模式将使你能够构建出健壮、高性能、可维护的 React 19 应用。在实际项目中，根据场景选择最合适的模式——没有银弹，只有最合适的工具。
