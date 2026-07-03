#!/usr/bin/env python3
"""
ch14-performance: Query caching with TTL and performance simulation.
Demonstrates cache hit rate, latency savings, and memory usage tracking.
Self-contained, stdlib only.
"""

import time
from typing import Dict, List, Optional, Tuple


class CacheEntry:
    def __init__(self, key: str, value: str, ttl: float):
        self.key = key
        self.value = value
        self.ttl = ttl
        self.created_at = time.time()
        self.access_count = 0

    def is_expired(self) -> bool:
        return time.time() - self.created_at > self.ttl

    def access(self) -> None:
        self.access_count += 1


class QueryCache:
    def __init__(self, default_ttl: float = 300.0):
        self.default_ttl = default_ttl
        self._cache: Dict[str, CacheEntry] = {}
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[str]:
        entry = self._cache.get(key)
        if entry is None:
            self._misses += 1
            return None
        if entry.is_expired():
            del self._cache[key]
            self._misses += 1
            return None
        entry.access()
        self._hits += 1
        return entry.value

    def set(self, key: str, value: str, ttl: Optional[float] = None) -> None:
        effective_ttl = ttl if ttl is not None else self.default_ttl
        self._cache[key] = CacheEntry(key, value, effective_ttl)

    def invalidate(self, question: str) -> int:
        to_delete = [k for k in self._cache if question in k]
        for k in to_delete:
            del self._cache[k]
        return len(to_delete)

    def stats(self) -> Dict:
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        active_entries = len(self._cache)
        estimated_bytes = sum(
            len(k) + len(e.value) + 128 for k, e in self._cache.items()
        )
        return {
            "hits": self._hits,
            "misses": self._misses,
            "total_requests": total,
            "hit_rate": hit_rate,
            "active_entries": active_entries,
            "estimated_memory_kb": estimated_bytes / 1024,
        }


LATENCY_PER_QUERY = 2.5

QUERY_LOG: List[Tuple[str, bool]] = [
    ("恒瑞医药有哪些抗肿瘤药物", True),
    ("奥希替尼片的作用机制", True),
    ("非小细胞肺癌的一线治疗方案", True),
    ("恒瑞医药有哪些抗肿瘤药物", True),
    ("紫杉醇的临床应用", True),
    ("奥希替尼片的作用机制", True),
    ("北京协和医院地址", False),
    ("恒瑞医药有哪些抗肿瘤药物", True),
    ("EGFR-TKI药物对比", True),
    ("非小细胞肺癌的一线治疗方案", True),
]


def execute_query(question: str) -> str:
    """Simulate a costly RAG query execution."""
    time.sleep(0.01)
    return f"关于「{question}」的检索结果: 找到相关文档3篇，生成回答完成。"


def main():
    print("=" * 60)
    print("ch14-performance: RAG 系统性能与缓存优化")
    print("=" * 60)

    cache = QueryCache(default_ttl=600.0)
    print(f"\n缓存配置:")
    print(f"  默认 TTL: {cache.default_ttl}s (10分钟)")
    print(f"  模拟查询延迟: {LATENCY_PER_QUERY}s/次")

    print(f"\n[1/4] 缓存预热...")
    warmup_queries = set(q for q, c in QUERY_LOG if c)
    for q in warmup_queries:
        cache.set(q, execute_query(q))
    print(f"  预热条目数: {len(warmup_queries)}")

    print(f"\n[2/4] 执行查询序列 ({len(QUERY_LOG)} 次)...")
    print(f"  {'#':<4} {'查询':<40} {'来源':<8} {'耗时':<8}")
    print(f"  {'-'*60}")

    for i, (question, cacheable) in enumerate(QUERY_LOG, 1):
        t0 = time.time()
        if cacheable:
            cached = cache.get(question)
            if cached is not None:
                elapsed = time.time() - t0
                print(
                    f"  {i:<4} {question[:38]:<40} {'缓存命中':<8} {elapsed*1000:.1f}ms"
                )
                continue
        result = execute_query(question)
        elapsed = time.time() - t0
        if cacheable:
            cache.set(question, result)
            print(
                f"  {i:<4} {question[:38]:<40} {'缓存未命中':<8} {elapsed*1000:.1f}ms"
            )
        else:
            print(f"  {i:<4} {question[:38]:<40} {'不缓存':<8} {elapsed*1000:.1f}ms")

    print(f"\n[3/4] 缓存统计...")
    stats = cache.stats()

    simulated_without_cache = len(QUERY_LOG) * LATENCY_PER_QUERY
    simulated_with_cache = stats["misses"] * LATENCY_PER_QUERY + stats["hits"] * 0.05
    speedup = (
        simulated_without_cache / simulated_with_cache
        if simulated_with_cache > 0
        else float("inf")
    )

    print(f"  {'指标':<35} {'数值':<15}")
    print(f"  {'-'*50}")
    print(f"  {'总请求数':<35} {stats['total_requests']:<15}")
    print(f"  {'缓存命中数':<35} {stats['hits']:<15}")
    print(f"  {'缓存未命中数':<35} {stats['misses']:<15}")
    print(f"  {'缓存命中率':<35} {stats['hit_rate']*100:.2f}%")
    print(f"  {'活跃条目数':<35} {stats['active_entries']:<15}")
    print(f"  {'预估内存使用':<35} {stats['estimated_memory_kb']:.2f} KB")

    print(f"\n[4/4] 性能估算...")
    print(f"  {'指标':<35} {'数值':<15}")
    print(f"  {'-'*50}")
    print(f"  {'模拟无缓存总耗时':<35} {simulated_without_cache:.2f}s")
    print(f"  {'模拟有缓存总耗时':<35} {simulated_with_cache:.2f}s")
    print(f"  {'加速比':<35} {speedup:.2f}x")

    print("\n" + "=" * 60)
    print("缓存优化总结")
    print("=" * 60)
    print(f"  - 使用 TTL={cache.default_ttl}s 的缓存层")
    print(f"  - 常见问题重复率: {stats['hit_rate']*100:.1f}%")
    print(f"  - 预计加速比: {speedup:.1f}x")
    print(f"  - 额外内存开销: 仅 {stats['estimated_memory_kb']:.1f} KB")
    print("=" * 60)


if __name__ == "__main__":
    main()
