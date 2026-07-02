"""
第 14 章 Demo：性能与内存管理

演示 RAG 系统性能优化的关键技术：
  Embedding 缓存（避免重复计算）
  多级缓存（内存 → Redis → 磁盘）
  并发检索（并行执行多种检索策略）
  连接池管理
  性能监控与指标分析

可独立运行，无需外部依赖。

用法：
  python performance_memory.py
  python performance_memory.py --benchmark
  python performance_memory.py --mode cache-demo
"""

import argparse
import hashlib
import math
import random
import time
from collections import OrderedDict, defaultdict
from typing import Optional, Callable


# ============================================================================
# 1. Embedding Cache
# ============================================================================


class EmbeddingCache:
    """Embedding 缓存。"""

    def __init__(self, max_size: int = 1000):
        self.cache: dict[str, list[float]] = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
        self.access_order: list[str] = []  # LRU tracking

    def get_or_compute(self, text: str, compute_fn: Callable) -> list[float]:
        key = hashlib.md5(text.encode()).hexdigest()
        if key in self.cache:
            self.hits += 1
            # LRU update
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]

        self.misses += 1
        embedding = compute_fn(text)

        if len(self.cache) >= self.max_size:
            # Evict LRU (20%)
            for _ in range(self.max_size // 5):
                old_key = self.access_order.pop(0)
                del self.cache[old_key]

        self.cache[key] = embedding
        self.access_order.append(key)
        return embedding

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


# ============================================================================
# 2. Multi-Level Cache
# ============================================================================


class CacheEntry:
    def __init__(self, value: str, ttl: float = 60.0):
        self.value = value
        self.timestamp = time.time()
        self.ttl = ttl
        self.hits = 0


class MultiLevelCache:
    """多级缓存：L1(内存) + L2(磁盘模拟)。"""

    def __init__(self, l1_max: int = 100):
        self.l1: OrderedDict[str, CacheEntry] = OrderedDict()
        self.l1_max = l1_max
        self.l2_store: dict[str, CacheEntry] = {}  # 模拟 Redis/磁盘
        self.l1_hits = 0
        self.l2_hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[str]:
        # L1: 内存
        if key in self.l1:
            entry = self.l1[key]
            if time.time() - entry.timestamp < entry.ttl:
                self.l1.move_to_end(key)
                entry.hits += 1
                self.l1_hits += 1
                return entry.value
            else:
                del self.l1[key]

        # L2: 模拟磁盘/Redis
        if key in self.l2_store:
            entry = self.l2_store[key]
            if time.time() - entry.timestamp < entry.ttl * 10:
                self.l2_hits += 1
                # 回填 L1
                self._set_l1(key, entry.value, entry.ttl)
                return entry.value
            else:
                del self.l2_store[key]

        self.misses += 1
        return None

    def set(self, key: str, value: str, ttl: float = 60.0):
        self._set_l1(key, value, ttl)
        self.l2_store[key] = CacheEntry(value, ttl * 10)

    def _set_l1(self, key: str, value: str, ttl: float):
        if len(self.l1) >= self.l1_max:
            self.l1.popitem(last=False)
        self.l1[key] = CacheEntry(value, ttl)

    def stats(self) -> dict:
        total = self.l1_hits + self.l2_hits + self.misses
        return {
            "l1_hits": self.l1_hits,
            "l2_hits": self.l2_hits,
            "misses": self.misses,
            "hit_rate": (self.l1_hits + self.l2_hits) / total if total > 0 else 0,
            "l1_size": len(self.l1),
            "l2_size": len(self.l2_store),
        }


# ============================================================================
# 3. Concurrent Retriever
# ============================================================================


class MockRetriever:
    """模拟检索器（带延迟）。"""

    def __init__(self, name: str, latency_ms: float):
        self.name = name
        self.latency = latency_ms

    def search(self, query: str, top_k: int = 3) -> list[tuple[str, float]]:
        time.sleep(self.latency / 1000)  # 模拟延迟
        return [
            (f"{self.name}_result_{i}", random.uniform(0.5, 0.95))
            for i in range(top_k)
        ]


class ConcurrentRetriever:
    """并发检索器（模拟并行执行）。"""

    def __init__(self):
        self.retrievers = {
            "dense": MockRetriever("dense", 200),
            "sparse": MockRetriever("sparse", 50),
            "kg": MockRetriever("kg", 150),
        }

    def search_sequential(self, query: str) -> dict:
        """串行执行（基线）。"""
        results = {}
        for name, retriever in self.retrievers.items():
            results[name] = retriever.search(query)
        return results

    def search_parallel_simulated(self, query: str) -> dict:
        """模拟并行执行。"""
        # 串行耗时: 200 + 50 + 150 = 400ms
        # 并行耗时: max(200, 50, 150) = 200ms
        latencies = [r.latency for r in self.retrievers.values()]
        max_latency = max(latencies)
        total_sequential = sum(latencies)

        # 模拟并行：所有检索器同时开始，最慢的决定完成时间
        results = {}
        for name, retriever in self.retrievers.items():
            results[name] = retriever.search(query)

        return {
            "results": results,
            "sequential_ms": total_sequential,
            "parallel_ms": max_latency,
            "speedup": total_sequential / max_latency if max_latency > 0 else 1,
        }


# ============================================================================
# 4. Connection Pool
# ============================================================================


class Connection:
    """模拟数据库连接。"""

    def __init__(self, cid: int):
        self.id = cid
        self.created_at = time.time()
        self.in_use = False

    def query(self, sql: str) -> list:
        time.sleep(0.01)
        return [f"result_from_conn_{self.id}"]

    def close(self):
        pass


class ConnectionPool:
    """连接池。"""

    def __init__(self, min_size: int = 3, max_size: int = 10):
        self.min_size = min_size
        self.max_size = max_size
        self.pool: list[Connection] = []
        self.active_count = 0
        self.wait_count = 0
        self._init_pool()

    def _init_pool(self):
        for i in range(self.min_size):
            self.pool.append(Connection(i))

    def acquire(self) -> Connection:
        """获取连接。"""
        for conn in self.pool:
            if not conn.in_use:
                conn.in_use = True
                return conn

        # 扩容
        if self.active_count < self.max_size:
            conn = Connection(self.active_count + len(self.pool))
            conn.in_use = True
            self.pool.append(conn)
            self.active_count += 1
            return conn

        # 没有可用连接，模拟等待
        self.wait_count += 1
        time.sleep(0.05)
        return self.acquire()

    def release(self, conn: Connection):
        conn.in_use = False

    def stats(self) -> dict:
        return {
            "total": len(self.pool),
            "active": sum(1 for c in self.pool if c.in_use),
            "idle": sum(1 for c in self.pool if not c.in_use),
            "max_size": self.max_size,
            "wait_count": self.wait_count,
        }


# ============================================================================
# 5. Performance Monitor
# ============================================================================


class PerformanceMonitor:
    """性能监控器。"""

    def __init__(self):
        self.metrics: dict[str, list[float]] = defaultdict(list)

    def record(self, stage: str, latency_ms: float):
        self.metrics[stage].append(latency_ms)

    def summary(self) -> dict:
        result = {}
        for stage, latencies in self.metrics.items():
            sorted_l = sorted(latencies)
            n = len(sorted_l)
            result[stage] = {
                "p50": sorted_l[n // 2] if n > 0 else 0,
                "p95": sorted_l[int(n * 0.95)] if n > 0 else 0,
                "p99": sorted_l[int(n * 0.99)] if n > 0 else 0,
                "avg": sum(latencies) / n if n > 0 else 0,
                "min": min(latencies) if n > 0 else 0,
                "max": max(latencies) if n > 0 else 0,
                "count": n,
            }
        return result


# ============================================================================
# 6. Token Budget Manager
# ============================================================================


class TokenBudget:
    """Token 预算管理器。"""

    def __init__(self, daily_budget: int = 100000):
        self.daily_budget = daily_budget
        self.used = 0

    def can_afford(self, estimated: int) -> bool:
        return self.used + estimated <= self.daily_budget

    def spend(self, tokens: int):
        self.used += tokens

    def remaining(self) -> int:
        return max(0, self.daily_budget - self.used)

    def usage_pct(self) -> float:
        return (self.used / self.daily_budget) * 100 if self.daily_budget > 0 else 0


# ============================================================================
# Demo Functions
# ============================================================================


def demo_embedding_cache():
    """演示 Embedding 缓存效果。"""
    print("\n" + "=" * 50)
    print("[Demo] Embedding Cache")
    print("=" * 50)

    def mock_embedding(text: str) -> list[float]:
        """模拟慢速 Embedding 计算。"""
        time.sleep(0.1)  # 100ms
        features = [0.0] * 16
        for i, ch in enumerate(text[:100]):
            features[hash(ch) % 16] += 1.0
        norm = math.sqrt(sum(v * v for v in features)) or 1.0
        return [v / norm for v in features]

    cache = EmbeddingCache(max_size=100)
    texts = [
        "恒瑞医药生产哪些药品？",
        "紫杉醇的治疗机制是什么？",
        "恒瑞医药生产哪些药品？",  # 重复
        "北京协和医院使用哪些药品？",
        "紫杉醇的治疗机制是什么？",  # 重复
    ]

    print("\n  计算 5 次 Embedding（含 2 次重复）:")
    for text in texts:
        start = time.time()
        emb = cache.get_or_compute(text, mock_embedding)
        elapsed = (time.time() - start) * 1000
        source = "cache" if cache.hits > 0 and elapsed < 10 else "compute"
        print(f"    [{source:>7}] {elapsed:6.1f}ms | {text}")

    print(f"\n  命中率: {cache.hit_rate * 100:.1f}% (命中 {cache.hits}, 未命中 {cache.misses})")
    print(f"  预期: 5 次请求, 3 次计算, 2 次缓存命中 → 40% 命中率")


def demo_multi_level_cache():
    """演示多级缓存。"""
    print("\n" + "=" * 50)
    print("[Demo] Multi-Level Cache")
    print("=" * 50)

    cache = MultiLevelCache(l1_max=5)

    print("\n  第一次请求（未命中）:")
    start = time.time()
    result = cache.get("query:恒瑞医药")
    elapsed = (time.time() - start) * 1000
    print(f"    [{('miss' if result is None else 'hit'):>6}] {elapsed:.1f}ms | 结果: {result}")

    print("\n  设置缓存:")
    cache.set("query:恒瑞医药", "恒瑞医药生产注射用紫杉醇、奥沙利铂和卡培他滨")
    print("    已缓存 → L1(内存) + L2(磁盘模拟)")

    print("\n  第二次请求（L1 命中）:")
    start = time.time()
    result = cache.get("query:恒瑞医药")
    elapsed = (time.time() - start) * 1000
    print(f"    [  L1] {elapsed:.1f}ms | 结果: {result}")

    print("\n  缓存统计:")
    stats = cache.stats()
    print(f"    L1 命中: {stats['l1_hits']}, L2 命中: {stats['l2_hits']}, 未命中: {stats['misses']}")
    print(f"    综合命中率: {stats['hit_rate'] * 100:.1f}%")


def demo_connection_pool():
    """演示连接池性能。"""
    print("\n" + "=" * 50)
    print("[Demo] Connection Pool")
    print("=" * 50)

    pool = ConnectionPool(min_size=3, max_size=10)

    print(f"\n  初始池: {pool.stats()}")

    # 模拟 8 个并发请求
    print("\n  8 个并发请求（池大小: 3 → 扩展到 8）:")
    conns = []
    for i in range(8):
        conn = pool.acquire()
        conns.append(conn)
        result = conn.query("SELECT * FROM documents")
        print(f"    请求 {i+1}: 使用连接 #{conn.id} → {result}")

    print(f"\n  使用后: 活跃 {pool.stats()['active']}, 空闲 {pool.stats()['idle']}")

    # 释放连接
    for conn in conns:
        pool.release(conn)

    print(f"  释放后: 活跃 {pool.stats()['active']}, 空闲 {pool.stats()['idle']}")


def demo_benchmark():
    """基准测试：串行 vs 并发。"""
    print("\n" + "=" * 50)
    print("[Benchmark] Sequential vs Parallel")
    print("=" * 50)

    retriever = ConcurrentRetriever()

    # 串行
    print("\n  ┌─ Sequential Retrieval ──────────────")
    start = time.time()
    seq_results = retriever.search_sequential("恒瑞医药")
    seq_time = (time.time() - start) * 1000
    for name, results in seq_results.items():
        print(f"  │  {name}: {results}")
    print(f"  │  Total: {seq_time:.1f}ms")
    print(f"  └────────────────────────────────────")

    # 模拟并行
    print("\n  ┌─ Parallel Retrieval ────────────────")
    start = time.time()
    par_results = retriever.search_parallel_simulated("恒瑞医药")
    par_time = (time.time() - start) * 1000
    for name, results in par_results["results"].items():
        print(f"  │  {name}: {results}")
    print(f"  │  Total: {par_time:.1f}ms (max of all retrievers)")
    print(f"  └────────────────────────────────────")

    print(f"\n  对比:")
    print(f"    串行: {seq_time:.0f}ms")
    print(f"    并行: {par_time:.0f}ms")
    print(f"    加速比: {seq_time / par_time:.1f}x")


def demo_performance_monitor():
    """演示性能监控。"""
    print("\n" + "=" * 50)
    print("[Demo] Performance Monitor")
    print("=" * 50)

    monitor = PerformanceMonitor()

    # 模拟 100 次请求
    print("\n  模拟 100 次请求的延迟分布...")
    random.seed(42)
    for _ in range(100):
        # 各阶段延迟（ms）
        monitor.record("embedding", random.gauss(150, 30))
        monitor.record("vector_search", random.gauss(40, 10))
        monitor.record("reranker", random.gauss(250, 50))
        monitor.record("llm_generate", random.gauss(1800, 300))

    summary = monitor.summary()

    print(f"\n  {'Stage':<20} {'P50':>8} {'P95':>8} {'P99':>8} {'Avg':>8} {'Count':>8}")
    print(f"  {'─' * 62}")
    for stage, metrics in sorted(summary.items()):
        print(f"  {stage:<20} {metrics['p50']:>8.0f} {metrics['p95']:>8.0f} "
              f"{metrics['p99']:>8.0f} {metrics['avg']:>8.0f} {metrics['count']:>8}")

    print(f"\n  总响应时间 (P50): {sum(m['p50'] for m in summary.values()):.0f}ms")
    print(f"  总响应时间 (P95): {sum(m['p95'] for m in summary.values()):.0f}ms")


def demo_token_budget():
    """演示 Token 预算管理。"""
    print("\n" + "=" * 50)
    print("[Demo] Token Budget Manager")
    print("=" * 50)

    budget = TokenBudget(daily_budget=1_000_000)

    requests = [
        ("简单查询", 500),
        ("多跳推理", 2000),
        ("长篇生成", 8000),
        ("高成本查询", 15000),
    ]

    print(f"\n  日预算: {budget.daily_budget:,} tokens")
    print(f"\n  {'请求':<16} {'估计Token':>10} {'审批':>8} {'剩余':>10}")
    print(f"  {'─' * 46}")
    for name, tokens in requests:
        allowed = budget.can_afford(tokens)
        if allowed:
            budget.spend(tokens)
        print(f"  {name:<16} {tokens:>10,} {'✅' if allowed else '❌':>8} "
              f"{budget.remaining():>10,}")

    print(f"\n  日终: 已用 {budget.used:,} ({budget.usage_pct():.1f}%)")


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="性能与内存管理 Demo")
    parser.add_argument("--mode", choices=["all", "cache", "pool", "benchmark", "monitor", "budget"],
                        default="all", help="演示模式")
    args = parser.parse_args()

    print("=" * 60)
    print("性能与内存管理 Demo")
    print("=" * 60)

    modes = []
    if args.mode == "all":
        modes = ["cache", "pool", "benchmark", "monitor", "budget"]
    else:
        modes = [args.mode]

    for m in modes:
        if m == "cache":
            demo_embedding_cache()
            demo_multi_level_cache()
        elif m == "pool":
            demo_connection_pool()
        elif m == "benchmark":
            demo_benchmark()
        elif m == "monitor":
            demo_performance_monitor()
        elif m == "budget":
            demo_token_budget()

    print("\n" + "=" * 60)
    print("模式说明:")
    print("  --mode cache     Embedding 缓存 + 多级缓存")
    print("  --mode pool      连接池")
    print("  --mode benchmark 串行 vs 并发基准测试")
    print("  --mode monitor   性能监控")
    print("  --mode budget    Token 预算管理")
    print("=" * 60)


if __name__ == "__main__":
    main()
