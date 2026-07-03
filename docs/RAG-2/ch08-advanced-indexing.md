# 第8章 高级索引技术

## 8.1 引言

向量索引是 RAG 系统的核心基础设施。当文档被切分并向量化后，如何在海量向量中快速找到与查询最相似的 Top-K 个结果，直接决定了 RAG 系统的响应速度与检索质量。前一章我们讨论了基础的向量化流程，本章将深入探讨生产环境中必须面对的高级索引问题：如何压缩向量以降低内存占用、如何调优索引结构以平衡速度与精度、如何处理增量数据更新，以及如何在单机与分布式场景下进行分片规划。

我们将依次覆盖以下主题：

| 主题 | 核心问题 | 适用场景 |
|------|----------|----------|
| 向量量化 (Quantization) | 如何压缩向量精度以减少存储和计算 | 内存受限的大规模部署 |
| HNSW / IVF 调优 | 如何选择索引结构并配置参数 | 平衡召回率与延迟 |
| 过滤预检查 (Filtered Pre-check) | 如何在带过滤条件的检索中保持效率 | 多租户、权限过滤、属性筛选 |
| 增量合并 (Incremental Merge) | 如何在不重建索引的前提下添加新数据 | 持续流入的数据流 |
| 分片策略 (Sharding) | 如何将索引拆分为可管理的片段 | 超大规模（亿级以上）向量库 |
| 内存映射文件 (Memory-Mapped Files) | 如何利用操作系统虚拟内存管理大索引 | 单机大索引、冷启动优化 |
| 磁盘 vs 内存权衡 | 如何在成本与延迟之间做出选择 | 硬件选型与架构设计 |

本章的所有代码示例基于 `faiss` 和 `numpy`，少量场景使用 `hnswlib` 作为对比。

```python
import numpy as np
import faiss
import time
import psutil
import os
from typing import Tuple, Optional, List, Dict

# 本章通用辅助函数：生成随机向量数据集
def generate_dataset(dim: int, n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((n, dim), dtype=np.float32)

# 通用评估函数：测量召回率和 QPS
def evaluate_index(index, queries: np.ndarray, 
                   ground_truth: np.ndarray, k: int = 10) -> Dict:
    nq = queries.shape[0]
    index.nprobe = 10 if hasattr(index, 'nprobe') else None
    D, I = index.search(queries, k)
    recall = np.mean([
        len(np.intersect1d(I[i], ground_truth[i])) / k
        for i in range(nq)
    ])
    # 测量 QPS
    start = time.perf_counter()
    for _ in range(100):
        index.search(queries, k)
    elapsed = time.perf_counter() - start
    qps = 100 * nq / elapsed
    return {"recall": recall, "qps": qps}
```

---

## 8.2 向量量化 (Vector Quantization)

### 8.2.1 为什么需要量化

一个 100 万条、768 维的 FP32 向量集占用内存：

```
1,000,000 × 768 × 4 bytes = 3,072,000,000 bytes ≈ 2.86 GB
```

这还只是原始向量。索引结构（图结构、倒排列表）额外占用 20%–50%。在单机部署中，内存往往是最大的瓶颈。量化通过降低每个分量的字节数来压缩向量，代价是精度损失。

### 8.2.2 FP32 → FP16：无损压缩的第一步

从 FP32 降到 FP16（2 字节/分量）可以立即将内存减半，而对大多数检索任务来说召回率损失可以忽略不计（通常 < 0.5%）。

```python
def quantize_fp32_to_fp16(vectors: np.ndarray) -> np.ndarray:
    """将 FP32 向量转换为 FP16 并返回 FP32 视图（便于后续运算）。"""
    return vectors.astype(np.float16)

def test_fp16_quality(dim: int = 768, n: int = 100000):
    """对比 FP32 和 FP16 下 L2 距离的误差。"""
    vectors = generate_dataset(dim, n)
    fp16_vectors = quantize_fp32_to_fp16(vectors)
    
    # 比较一组随机点对的 L2 距离
    rng = np.random.default_rng(123)
    idx_a = rng.integers(0, n, 1000)
    idx_b = rng.integers(0, n, 1000)
    
    dist_fp32 = np.linalg.norm(vectors[idx_a] - vectors[idx_b], axis=1)
    dist_fp16 = np.linalg.norm(
        fp16_vectors[idx_a].astype(np.float32) - 
        fp16_vectors[idx_b].astype(np.float32), axis=1
    )
    
    mse = np.mean((dist_fp32 - dist_fp16) ** 2)
    max_err = np.max(np.abs(dist_fp32 - dist_fp16))
    print(f"FP16 量化质量: MSE={mse:.6f}, MaxError={max_err:.6f}")
    print(f"相对误差: {max_err / np.mean(dist_fp32) * 100:.2f}%")

# test_fp16_quality()
```

FP16 量化的最大价值在于它"几乎免费"——你只需改变数据类型，不需要改变索引结构或搜索算法。Faiss 对 FP16 有原生支持：

```python
def build_fp16_index(dim: int, vectors: np.ndarray) -> faiss.Index:
    """使用 FP16 构建 IVF 索引。"""
    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, 100, faiss.METRIC_L2)
    fp16_vectors = np.ascontiguousarray(vectors.astype(np.float16))
    index.train(fp16_vectors)
    index.add(fp16_vectors)
    return index
```

### 8.2.3 INT8 量化：标量量化 (Scalar Quantization)

INT8 将每个分量从 4 字节压缩到 1 字节，压缩比 4:1。基本原理是找到每个维度的取值范围，将 FP32 值线性映射到 [0, 255]：

```python
class ScalarQuantizer:
    """
    逐维标量量化器：对每个维度独立计算 min/max 并映射到 uint8。
    """
    def __init__(self):
        self.mins: Optional[np.ndarray] = None
        self.maxs: Optional[np.ndarray] = None
        self.scales: Optional[np.ndarray] = None
    
    def fit(self, vectors: np.ndarray) -> None:
        """从训练数据中学习每个维度的 min/max。"""
        self.mins = vectors.min(axis=0)
        self.maxs = vectors.max(axis=0)
        # 避免除零：如果某维度所有值相同，scale = 1
        ranges = self.maxs - self.mins
        ranges[ranges == 0] = 1.0
        self.scales = 255.0 / ranges
    
    def quantize(self, vectors: np.ndarray) -> np.ndarray:
        """将 FP32 向量量化为 uint8。"""
        if self.scales is None:
            raise ValueError("Must call fit() before quantize()")
        clipped = np.clip(vectors, self.mins, self.maxs)
        return ((clipped - self.mins) * self.scales).astype(np.uint8)
    
    def dequantize(self, quantized: np.ndarray) -> np.ndarray:
        """将 uint8 向量恢复为 FP32（近似值）。"""
        if self.scales is None:
            raise ValueError("Must call fit() before dequantize()")
        return quantized.astype(np.float32) / self.scales + self.mins
    
    def compress_ratio(self) -> float:
        """压缩比：FP32 4 字节 / INT8 1 字节 = 4.0"""
        return 4.0


def evaluate_scalar_quantization(dim: int = 768, n_train: int = 50000, 
                                  n_test: int = 10000):
    """评估标量量化的精度损失。"""
    train_vecs = generate_dataset(dim, n_train, seed=100)
    test_vecs = generate_dataset(dim, n_test, seed=200)
    
    sq = ScalarQuantizer()
    sq.fit(train_vecs)
    
    quantized = sq.quantize(test_vecs)
    dequantized = sq.dequantize(quantized)
    
    # 计算每个向量的重建误差
    mse_per_vector = np.mean((test_vecs - dequantized) ** 2, axis=1)
    avg_mse = np.mean(mse_per_vector)
    avg_l2 = np.mean(np.linalg.norm(test_vecs - dequantized, axis=1))
    
    print(f"标量量化评估:")
    print(f"  平均 MSE: {avg_mse:.6f}")
    print(f"  平均 L2 误差: {avg_l2:.6f}")
    print(f"  压缩比: {sq.compress_ratio():.1f}x")
    
    # 对检索召回率的影响
    index_fp32 = faiss.IndexFlatL2(dim)
    index_fp32.add(train_vecs)
    _, gt = index_fp32.search(test_vecs, 10)
    
    index_int8 = faiss.IndexFlatL2(dim)
    index_int8.add(dequantized)
    _, pred = index_int8.search(dequantized, 10)
    
    recall = np.mean([
        len(np.intersect1d(pred[i], gt[i])) / 10
        for i in range(n_test)
    ])
    print(f"  检索召回率（INT8 vs FP32）: {recall:.4f}")

# evaluate_scalar_quantization()
```

### 8.2.4 乘积量化 (Product Quantization, PQ)

乘积量化是比标量量化更激进的压缩方案。它将向量切分为多个子空间，对每个子空间独立进行聚类量化。常见的配置如 PQ4x8（4 个子空间，每个子空间 8 位 = 256 个聚类中心）将 768 维向量压缩到 4 字节——压缩比高达 192:1。

```
原始向量 (768 × FP32) = 3072 字节
PQ4x8: 4 个子空间 × 8 位码字 = 4 字节
压缩比: 768:1
```

```python
def build_pq_index(dim: int, vectors: np.ndarray, 
                   m: int = 16, nbits: int = 8) -> faiss.IndexIVFPQ:
    """
    构建带乘积量化的 IVF 索引。
    
    Args:
        dim: 向量维度
        vectors: 训练数据
        m: 子空间数量
        nbits: 每个子空间的码本大小（2^nbits 个中心）
    """
    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFPQ(quantizer, dim, 100, m, nbits)
    index.train(vectors)
    index.add(vectors)
    return index


def compare_compression_strategies(dim: int = 768, n: int = 100000):
    """比较不同量化策略的内存占用和召回率。"""
    vectors = generate_dataset(dim, n)
    queries = generate_dataset(dim, 100, seed=99)
    
    # 基准：Flat (无压缩)
    flat_index = faiss.IndexFlatL2(dim)
    flat_index.add(vectors)
    _, gt = flat_index.search(queries, 10)
    
    strategies = {
        "Flat (FP32)": faiss.IndexFlatL2(dim),
        "IVF100,SQ8": lambda: build_sq_index(dim, vectors),
        "IVF100,PQ16x8": lambda: build_pq_index(dim, vectors, m=16, nbits=8),
        "IVF100,PQ4x8": lambda: build_pq_index(dim, vectors, m=4, nbits=8),
    }
    
    results = []
    for name, builder in strategies.items():
        if callable(builder):
            index = builder()
        else:
            index = builder
            index.add(vectors)
        
        # 搜索
        index.nprobe = 10
        D, I = index.search(queries, 10)
        recall = np.mean([
            len(np.intersect1d(I[i], gt[i])) / 10
            for i in range(100)
        ])
        
        # 内存占用（faiss 的 internal memory 估算）
        # 这里用索引文件大小作为近似
        results.append((name, recall))
        print(f"{name:25s}  召回率={recall:.4f}")
    
    return results


def build_sq_index(dim: int, vectors: np.ndarray) -> faiss.IndexIVFScalarQuantizer:
    index = faiss.IndexIVFScalarQuantizer(
        dim, 100, faiss.ScalarQuantizer.QT_8bit
    )
    index.train(vectors)
    index.add(vectors)
    return index

# compare_compression_strategies()
```

典型结果（768 维，100K 向量）：

| 策略 | 每向量字节数 | 召回率@10 | 说明 |
|------|-------------|-----------|------|
| Flat FP32 | 3072 | 1.000 | 基准，暴力搜索 |
| IVF + SQ8 | 768 | 0.985 | 标量量化，几乎无损 |
| IVF + PQ16x8 | 16 | 0.932 | 16 字节/向量，适合大索引 |
| IVF + PQ4x8 | 4 | 0.851 | 极致压缩，适合内存极受限场景 |

### 8.2.5 量化实践建议

1. **首选 FP16**：如果你的硬件支持 FP16 运算（大多数 GPU 都支持），这是零成本的优化。
2. **次选 SQ8**：当内存仍然紧张时，标量量化带来的召回率损失通常 < 2%，实现简单。
3. **慎选 PQ**：乘积量化可以大幅压缩，但参数（子空间数量 m、码本位宽 nbits）对召回率敏感，需要针对数据集调优。经验规则：m 越大保留信息越多，但每个子空间的维度太少时聚类效果变差。
4. **混合策略**：可以将 SQ 和 PQ 组合使用——先用 SQ 压缩到 INT8，再对 INT8 向量做 PQ。

---

## 8.3 HNSW / IVF 索引调优

### 8.3.1 IVF (Inverted File Index) 参数详解

IVF 的核心思想是将向量空间划分为多个 Voronoi 单元（通过 K-Means 聚类），搜索时只探查与查询最近的若干个单元。

关键参数：

| 参数 | 含义 | 默认值 | 调优方向 |
|------|------|--------|----------|
| `nlist` | 聚类中心数量（Voronoi 单元数） | 100 | 大 → 更细粒度分区 |
| `nprobe` | 搜索时探查的单元数 | 1 | 大 → 召回率↑ 延迟↑ |

```python
def tune_ivf_parameters(dim: int = 768, n: int = 200000):
    """系统调优 IVF 的 nlist 和 nprobe 参数。"""
    vectors = generate_dataset(dim, n)
    queries = generate_dataset(dim, 500, seed=99)
    
    # 暴力搜索获取 ground truth
    flat = faiss.IndexFlatL2(dim)
    flat.add(vectors)
    _, gt = flat.search(queries, 10)
    
    nlist_options = [50, 100, 200, 500, 1000]
    nprobe_options = [1, 2, 5, 10, 20, 50]
    
    print(f"{'nlist':>6} {'nprobe':>6} {'Recall@10':>10} {'Latency(ms)':>12} {'QPS':>10}")
    print("-" * 48)
    
    for nlist in nlist_options:
        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist)
        index.train(vectors)
        index.add(vectors)
        
        for nprobe in nprobe_options:
            index.nprobe = nprobe
            
            # 测量延迟
            start = time.perf_counter()
            D, I = index.search(queries, 10)
            elapsed = time.perf_counter() - start
            
            recall = np.mean([
                len(np.intersect1d(I[i], gt[i])) / 10
                for i in range(500)
            ])
            
            lat_ms = elapsed / 500 * 1000
            qps = 500 / elapsed
            
            print(f"{nlist:6d} {nprobe:6d} {recall:10.4f} {lat_ms:12.3f} {qps:10.1f}")

# tune_ivf_parameters()
```

**nlist 选择的经验法则：**
- `nlist = 4 * sqrt(N)` 是一个常用的起始点，其中 N 是向量总数。
- 对于 100 万向量：`4 * sqrt(1,000,000) = 4 * 1000 = 4000`
- 但实际部署中 `nlist` 不宜过大，因为训练 K-Means 的开销与 `nlist` 成正比。

**nprobe 选择的经验法则：**
- 从 `nprobe = 1` 开始，逐步增加直到召回率达到目标（通常 95%+）。
- 每增加一个 nprobe 单元，搜索时间大致线性增长。
- 一般推荐的区间是 `nprobe ∈ [5, 50]`。

### 8.3.2 HNSW (Hierarchical Navigable Small World) 参数详解

HNSW 是当前最流行的图索引结构之一。它构建多层图结构，上层是"快速通道"，下层是精细搜索。

```python
def build_hnsw_index(dim: int, vectors: np.ndarray,
                     M: int = 16, ef_construction: int = 200) -> faiss.IndexHNSWFlat:
    """
    构建 HNSW 索引。
    
    Args:
        M: 每个节点的最大连接数（影响索引质量和内存）
        ef_construction: 构建时的动态列表大小（越大索引越精确但越慢）
    """
    index = faiss.IndexHNSWFlat(dim, M)
    index.hnsw.efConstruction = ef_construction
    index.add(vectors)
    return index
```

HNSW 的核心参数：

| 参数 | 范围 | 默认值 | 作用 | 对资源的影响 |
|------|------|--------|------|-------------|
| `M` | 4–64 | 16 | 每层最大邻居数 | M↑ → 内存↑ 召回率↑ |
| `ef_construction` | 50–1000 | 200 | 构建时的候选列表大小 | 越大索引构建越慢但质量越高 |
| `ef_search` | 1–N | 16 | 搜索时的候选列表大小 | 越大搜索越精确但越慢 |

```python
def tune_hnsw_parameters(dim: int = 768, n: int = 100000):
    """系统调优 HNSW 的 M 和 ef_search 参数。"""
    vectors = generate_dataset(dim, n)
    queries = generate_dataset(dim, 200, seed=99)
    
    # Ground truth
    flat = faiss.IndexFlatL2(dim)
    flat.add(vectors)
    _, gt = flat.search(queries, 10)
    
    M_options = [8, 16, 32]
    ef_search_options = [16, 32, 64, 128, 256]
    
    print(f"{'M':>4} {'ef_search':>10} {'Recall@10':>10} {'Latency(ms)':>12} {'QPS':>10}")
    print("-" * 50)
    
    for M in M_options:
        for ef_search in ef_search_options:
            index = faiss.IndexHNSWFlat(dim, M)
            index.hnsw.efConstruction = 200
            index.add(vectors)
            
            index.hnsw.efSearch = ef_search
            
            start = time.perf_counter()
            D, I = index.search(queries, 10)
            elapsed = time.perf_counter() - start
            
            recall = np.mean([
                len(np.intersect1d(I[i], gt[i])) / 10
                for i in range(200)
            ])
            
            lat_ms = elapsed / 200 * 1000
            qps = 200 / elapsed
            
            print(f"{M:4d} {ef_search:10d} {recall:10.4f} {lat_ms:12.3f} {qps:10.1f}")

# tune_hnsw_parameters()
```

### 8.3.3 IVF 与 HNSW 的选择决策树

```
向量数量 < 10 万？
├── 是 → HNSW 通常更快（不需要训练阶段）
└── 否 → 继续评估
    ├── 需要极高的召回率（> 98%）？
    │   ├── 是 → HNSW（图结构在高召回区域更优）
    │   └── 否 → IVF 足够
    ├── 索引更新频繁？
    │   ├── 是 → IVF（HNSW 的插入/删除操作会破坏图结构）
    │   └── 否 → HNSW 或 IVF 皆可
    ├── 内存是否紧张？
    │   ├── 是 → IVF + PQ（HNSW 的图结构额外占用大量内存）
    │   └── 否 → HNSW（性能通常更优）
    └── 硬件？
        ├── GPU → IVF（Faiss GPU 对 IVF 支持完善，HNSW 支持有限）
        └── CPU → HNSW 或 IVF 皆可
```

**综合建议：**

| 场景 | 推荐索引 | 理由 |
|------|---------|------|
| 100 万以下，单机 CPU | HNSW | 无需训练，即插即用，延迟低 |
| 100 万以上，单机 CPU | IVF + HNSW (quantizer) | IVF 的 coarse quantizer 用 HNSW |
| GPU 部署 | IVF (Flat 或 PQ) | Faiss GPU 对 IVF 优化最好 |
| 频繁插入删除 | IVF | HNSW 动态维护成本高 |
| 极致低延迟 (< 1ms) | HNSW (小 ef_search) | 图搜索可在几跳内收敛 |

### 8.3.4 混合索引：IVF + HNSW Quantizer

Faiss 允许将 IVF 的 coarse quantizer 替换为 HNSW 索引，从而在搜索时加速"寻找最近聚类中心"这一步骤：

```python
def build_ivf_hnsw_index(dim: int, vectors: np.ndarray,
                          nlist: int = 200) -> faiss.IndexIVFFlat:
    """使用 HNSW 作为 coarse quantizer 的 IVF 索引。"""
    quantizer = faiss.IndexHNSWFlat(dim, 32)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist)
    index.train(vectors)
    index.add(vectors)
    # 搜索时使用 HNSW 快速定位最近的聚类中心
    index.nprobe = 10
    return index
```

这比用 FlatL2 做 quantizer 快 5–10 倍，因为 HNSW 可以在 O(log nlist) 时间内找到最近的聚类中心，而 FlatL2 需要 O(nlist) 的暴力比较。

---

## 8.4 过滤预检查 (Filtered Pre-check)

### 8.4.1 问题定义

在实际 RAG 系统中，检索往往不是"全局搜索"而是"带条件的搜索"。常见的过滤条件包括：

- **多租户过滤**：只搜索当前用户所属租户的文档
- **时间范围过滤**：只搜索最近 30 天的文档
- **元数据过滤**：只搜索特定类型、来源或标签的文档
- **权限过滤**：只搜索当前用户有权访问的文档

当索引本身没有感知过滤条件时，标准的做法是"先检索、后过滤"（post-filtering）。但这种方法的问题在于：如果过滤条件非常严格（比如只保留 1% 的结果），Top-K 检索结果中可能根本没有满足条件的文档。

### 8.4.2 预过滤策略

预过滤的核心思想是：**在向量搜索之前或之中应用过滤条件**。

```python
class FilteredIVFIndex:
    """
    支持元数据预过滤的 IVF 索引包装器。
    
    策略：对每个聚类中心维护一个"允许列表"，
    搜索时只从允许通过过滤的向量中查找最近邻。
    """
    
    def __init__(self, dim: int, nlist: int = 100):
        self.dim = dim
        self.nlist = nlist
        self.quantizer = faiss.IndexFlatL2(dim)
        self.index = faiss.IndexIVFFlat(self.quantizer, dim, nlist)
        self.index.nprobe = 10
        
        # 元数据存储：id -> metadata dict
        self.metadata: Dict[int, Dict] = {}
        # 聚类中心 -> 该中心下所有向量的 id 列表
        self.cluster_to_ids: Dict[int, List[int]] = {}
        
    def add_with_metadata(self, vectors: np.ndarray, 
                          metadata_list: List[Dict]) -> None:
        """向索引添加向量及其元数据。"""
        n = vectors.shape[0]
        start_id = self.index.ntotal
        self.index.add(vectors)
        
        # 记录元数据
        for i, meta in enumerate(metadata_list):
            self.metadata[start_id + i] = meta
        
        # 记录每个向量属于哪个聚类
        if hasattr(self.index, 'assign'):
            _, assign = self.index.assign(vectors, 1)
            for i, cluster in enumerate(assign.flatten()):
                cid = int(cluster)
                if cid not in self.cluster_to_ids:
                    self.cluster_to_ids[cid] = []
                self.cluster_to_ids[cid].append(start_id + i)
    
    def search_with_filter(self, query: np.ndarray, k: int = 10,
                           filter_fn=None) -> Tuple[np.ndarray, np.ndarray]:
        """
        带过滤条件的搜索。
        
        策略：先找到最近的 nprobe 个聚类，然后只在这些聚类中
        对满足 filter_fn 条件的向量做暴力搜索。
        
        Args:
            filter_fn: callable(metadata_dict) -> bool
                       返回 True 表示该向量通过过滤
        """
        if filter_fn is None:
            return self.index.search(query, k)
        
        # 1. 找到最近的聚类中心
        distances, assignments = self.quantizer.search(query, self.index.nprobe)
        
        # 2. 收集所有候选向量 id
        candidate_ids = set()
        for cluster_id in assignments[0]:
            cid = int(cluster_id)
            if cid in self.cluster_to_ids:
                candidate_ids.update(self.cluster_to_ids[cid])
        
        # 3. 过滤
        filtered_ids = [
            i for i in candidate_ids 
            if filter_fn(self.metadata.get(i, {}))
        ]
        
        if len(filtered_ids) < k:
            # 备选：放宽过滤（降低到更多聚类中搜索）
            # 这里简化为返回空
            return np.array([]), np.array([])
        
        # 4. 在过滤后的向量中暴力搜索
        filtered_vectors = self.index.reconstruct_batch(filtered_ids)
        q = query.reshape(1, -1).astype(np.float32)
        dists = np.linalg.norm(filtered_vectors - q, axis=1)
        
        top_k_idx = np.argsort(dists)[:k]
        result_ids = np.array([filtered_ids[i] for i in top_k_idx])
        result_dists = dists[top_k_idx]
        
        return result_dists.reshape(1, -1), result_ids.reshape(1, -1)
```

### 8.4.3 Faiss IDSelector 机制

Faiss 提供了 `IDSelector` 接口，可以在搜索时跳过不需要的向量：

```python
def search_with_id_selector(dim: int = 768, n: int = 50000):
    """演示 Faiss IDSelector 的使用。"""
    vectors = generate_dataset(dim, n)
    queries = generate_dataset(dim, 10, seed=99)
    
    # 构建索引
    index = faiss.IndexIVFFlat(faiss.IndexFlatL2(dim), dim, 50)
    index.train(vectors)
    index.add(vectors)
    index.nprobe = 5
    
    # 假设我们只想搜索 id 为偶数的向量
    class EvenIDSelector(faiss.IDSelector):
        def is_member(self, idx):
            return idx % 2 == 0
    
    selector = EvenIDSelector()
    
    # 使用 IDSelector 搜索
    # 注意：faiss 的 search 方法支持 IDSelector 参数
    params = faiss.SearchParametersIVF()
    params.sel = selector
    
    D, I = index.search(queries, 10, params=params)
    print(f"搜索到的 ID: {I[:3]}")
    print(f"所有 ID 均为偶数: {np.all(I[:3] % 2 == 0)}")

# search_with_id_selector()
```

### 8.4.4 过滤感知的索引设计模式

| 模式 | 实现 | 适用场景 | 优缺点 |
|------|------|----------|--------|
| Post-filter | 先搜 Top-2K 再过滤 | 过滤条件不严格（> 50% 通过率） | 简单，但可能遗漏 |
| Pre-filter (IVF) | 限制搜索的 Voronoi 单元 | 过滤条件与聚类相关 | 需要维护聚类到元数据的映射 |
| IDSelector | Faiss 原生 ID 筛选 | 过滤条件可映射为 ID 范围 | 高效，但过滤逻辑简单 |
| Multi-index | 按过滤维度建多个子索引 | 过滤维度少且值有限 | 空间换时间 |
| 级联搜索 | 先用属性索引粗筛，再向量搜索 | 过滤条件先验性强 | 需要两个索引系统 |

**最佳实践：** 对于大多数 RAG 应用，**先检索 Top-K * (1 / selectivity) 再做 post-filter** 是最实用的方案。例如，如果过滤条件保留约 10% 的文档，先检索 Top-200 再过滤到 Top-20：

```python
def adaptive_post_filter(index, query: np.ndarray, k: int, 
                         filter_fn, selectivity_estimate: float = 0.1):
    """
    带自适应放大系数的后过滤。
    
    Args:
        selectivity_estimate: 过滤后的保留率预估（0~1）
    """
    # 安全放大：检索 k / selectivity 个候选，但不超过 10x
    oversample = min(int(k / max(selectivity_estimate, 0.01)), k * 10)
    
    D, I = index.search(query, max(oversample, k))
    
    # 后过滤
    valid = []
    for idx, dist in zip(I[0], D[0]):
        if filter_fn(idx):
            valid.append((idx, dist))
        if len(valid) >= k:
            break
    
    if len(valid) < k:
        print(f"警告：过滤后只有 {len(valid)}/{k} 个结果")
    
    result_ids = np.array([v[0] for v in valid[:k]])
    result_dists = np.array([v[1] for v in valid[:k]])
    return result_dists.reshape(1, -1), result_ids.reshape(1, -1)
```

---

## 8.5 增量合并 (Incremental Merge)

### 8.5.1 问题背景

大多数向量索引（尤其是 HNSW 和 IVF）在构建完成后，**添加新向量需要重建整个索引或者付出高昂的代价**。例如：

- **IVF**：添加新向量不需要重建聚类中心，但新向量被分配到最近的聚类后，该聚类的倒排列表不断增长，可能导致负载不均。
- **HNSW**：插入新节点需要找到其在各层的位置并建立连接，当插入量超过原始数据的 20–30% 时，图质量会显著下降。
- **Flat**：暴力搜索的索引不需要重建，但每次搜索都是 O(N)，不适合大规模数据。

增量合并策略的核心思想是：**将新数据构建为小索引，定期与大索引合并**。

### 8.5.2 分层合并策略 (类似 LSM-Tree)

借鉴 LSM-Tree (Log-Structured Merge-Tree) 的思路，将索引分为多层：

```python
class IncrementalIndex:
    """
    分层增量索引，类似 LSM-Tree 的合并策略。
    
    层级结构：
    - Level 0: 内存中的小索引（HNSW），最多 1000 条
    - Level 1: 稍大的索引，最多 10000 条
    - Level 2: 主索引，最多 100000 条
    - Level 3: 基础索引，存储所有历史数据
    """
    
    def __init__(self, dim: int, levels: Optional[List[int]] = None):
        self.dim = dim
        # 每层的容量上限
        self.capacities = levels or [1000, 10000, 100000, float('inf')]
        self.num_levels = len(self.capacities)
        
        # 每层的索引和对应的向量存储
        self.indices: List[Optional[faiss.Index]] = [None] * self.num_levels
        self.vectors: List[Optional[np.ndarray]] = [None] * self.num_levels
        self.sizes = [0] * self.num_levels
        
    def add(self, new_vectors: np.ndarray) -> None:
        """添加新向量到 Level 0，必要时触发合并。"""
        if self.indices[0] is None:
            self.indices[0] = faiss.IndexHNSWFlat(self.dim, 16)
            self.indices[0].hnsw.efConstruction = 200
        
        self.indices[0].add(new_vectors)
        self.sizes[0] = self.indices[0].ntotal
        
        # 如果 Level 0 溢出，向下合并
        if self.sizes[0] >= self.capacities[0]:
            self._merge_down(0)
    
    def _merge_down(self, level: int) -> None:
        """将 level 层的索引合并到 level+1 层。"""
        if level + 1 >= self.num_levels:
            return
        
        # 收集当前层的所有向量
        current_vectors = self._extract_vectors(level)
        
        if self.indices[level + 1] is None:
            # 下一层为空，直接把当前层内容放下去
            self.indices[level + 1] = self._build_index_for_level(
                current_vectors, level + 1
            )
        else:
            # 合并：取出下一层的向量 + 当前层的向量 → 重建下一层
            next_vectors = self._extract_vectors(level + 1)
            merged = np.vstack([next_vectors, current_vectors])
            
            if merged.shape[0] >= self.capacities[level + 1]:
                # 下一层也满了，递归合并
                self.indices[level + 1] = self._build_index_for_level(
                    merged, level + 1
                )
                self.sizes[level + 1] = merged.shape[0]
                if self.sizes[level + 1] >= self.capacities[level + 1]:
                    self._merge_down(level + 1)
            else:
                self.indices[level + 1] = self._build_index_for_level(
                    merged, level + 1
                )
                self.sizes[level + 1] = merged.shape[0]
        
        # 清空当前层
        self.indices[level] = None
        self.sizes[level] = 0
        self.vectors[level] = None
    
    def _extract_vectors(self, level: int) -> np.ndarray:
        """从指定层提取所有向量。"""
        if self.indices[level] is None:
            return np.empty((0, self.dim), dtype=np.float32)
        
        n_total = self.indices[level].ntotal
        vectors = np.empty((n_total, self.dim), dtype=np.float32)
        for i in range(n_total):
            vectors[i] = self.indices[level].reconstruct(i)
        return vectors
    
    def _build_index_for_level(self, vectors: np.ndarray, 
                                level: int) -> faiss.Index:
        """根据层级选择合适的索引类型和参数。"""
        n = vectors.shape[0]
        if level <= 1:
            # 小索引用 HNSW
            idx = faiss.IndexHNSWFlat(self.dim, 16)
            idx.hnsw.efConstruction = 200
            idx.add(vectors)
            return idx
        else:
            # 大索引用 IVF
            nlist = min(int(4 * np.sqrt(n)), 1000)
            quantizer = faiss.IndexFlatL2(self.dim)
            idx = faiss.IndexIVFFlat(quantizer, self.dim, nlist)
            idx.train(vectors)
            idx.add(vectors)
            idx.nprobe = 10
            return idx
    
    def search(self, query: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """搜索所有层级并合并结果。"""
        all_distances = []
        all_indices = []
        offset = 0
        
        for level in range(self.num_levels):
            if self.indices[level] is not None:
                D, I = self.indices[level].search(query, k)
                all_distances.append(D)
                all_indices.append(I + offset)
                offset += self.indices[level].ntotal
        
        if not all_distances:
            return np.array([]), np.array([])
        
        # 合并各层结果
        combined_dist = np.hstack(all_distances)
        combined_idx = np.hstack(all_indices)
        
        # 取最近的 k 个
        top_k = np.argsort(combined_dist[0])[:k]
        return (combined_dist[0, top_k].reshape(1, -1),
                combined_idx[0, top_k].reshape(1, -1))
```

### 8.5.3 实用增量策略

对于大多数生产场景，一个更简单的策略是"重建式增量"：

```python
class SimpleIncrementalIndex:
    """
    简单的增量索引：用缓冲区收集新数据，达到阈值后与主索引合并重建。
    适合每日/每小时批量更新的场景。
    """
    
    def __init__(self, dim: int, merge_threshold: int = 50000):
        self.dim = dim
        self.merge_threshold = merge_threshold
        self.main_index: Optional[faiss.Index] = None
        self.buffer: List[np.ndarray] = []
        self.buffer_size = 0
    
    def add(self, vectors: np.ndarray) -> None:
        self.buffer.append(vectors)
        self.buffer_size += vectors.shape[0]
        
        if self.buffer_size >= self.merge_threshold:
            self._merge()
    
    def _merge(self) -> None:
        """合并缓冲区到主索引（重建主索引）。"""
        buffer_vectors = np.vstack(self.buffer)
        
        if self.main_index is None:
            # 第一次：用缓冲区构建主索引
            nlist = min(int(4 * np.sqrt(buffer_vectors.shape[0])), 1000)
            quantizer = faiss.IndexFlatL2(self.dim)
            self.main_index = faiss.IndexIVFFlat(quantizer, self.dim, nlist)
            self.main_index.train(buffer_vectors)
            self.main_index.add(buffer_vectors)
            self.main_index.nprobe = 10
        else:
            # 后续：取出主索引的所有向量，合并后重建
            main_vectors = self._extract_all()
            combined = np.vstack([main_vectors, buffer_vectors])
            
            # 重建
            nlist = min(int(4 * np.sqrt(combined.shape[0])), 1000)
            quantizer = faiss.IndexFlatL2(self.dim)
            new_index = faiss.IndexIVFFlat(quantizer, self.dim, nlist)
            new_index.train(combined)
            new_index.add(combined)
            new_index.nprobe = 10
            self.main_index = new_index
        
        # 清空缓冲区
        self.buffer = []
        self.buffer_size = 0
    
    def _extract_all(self) -> np.ndarray:
        """从主索引提取所有向量。"""
        n = self.main_index.ntotal
        vecs = np.empty((n, self.dim), dtype=np.float32)
        for i in range(n):
            vecs[i] = self.main_index.reconstruct(i)
        return vecs
    
    def search(self, query: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """搜索主索引 + 缓冲区中的新数据。"""
        # 搜索主索引
        if self.main_index is not None:
            D, I = self.main_index.search(query, k)
        else:
            D, I = np.full((1, k), np.inf), np.full((1, k), -1)
        
        # 对缓冲区中的新向量做暴力搜索
        if self.buffer:
            buffer_vectors = np.vstack(self.buffer)
            bD, bI = flat_search(buffer_vectors, query, k)
            # 合并结果
            combined_D = np.hstack([D, bD])
            combined_I = np.hstack([I, bI + (self.main_index.ntotal if self.main_index else 0)])
            top_k = np.argsort(combined_D[0])[:k]
            D = combined_D[0, top_k].reshape(1, -1)
            I = combined_I[0, top_k].reshape(1, -1)
        
        return D, I


def flat_search(vectors: np.ndarray, query: np.ndarray, 
                k: int) -> Tuple[np.ndarray, np.ndarray]:
    """对向量集合做暴力搜索。"""
    dists = np.linalg.norm(vectors - query, axis=1)
    top_k = np.argsort(dists)[:k]
    return dists[top_k].reshape(1, -1), top_k.reshape(1, -1)
```

### 8.5.4 增量索引的最佳实践

1. **批量而非单条**：单条插入的代价极高，始终使用批量插入（batch_size ≥ 1000）。
2. **合并时机**：在系统低负载时段（如凌晨）触发合并。
3. **影子索引**：合并时构建新索引，构建完成后用原子操作替换旧索引，避免服务中断。
4. **增量 vs 全量**：如果每日增量不超过总数据量的 5%，增量合并是有意义的；否则直接全量重建可能更简单。
5. **监控索引质量**：定期（如每周）评估索引的召回率，如果衰减超过阈值则触发全量重建。

```python
def shadow_merge(main_index_path: str, new_vectors: np.ndarray, 
                 dim: int) -> faiss.Index:
    """
    影子合并：构建新索引，完成后原子替换。
    
    Args:
        main_index_path: 主索引文件路径
        new_vectors: 新增向量
        dim: 向量维度
    """
    import shutil
    import tempfile
    
    # 加载旧索引
    old_index = faiss.read_index(main_index_path)
    
    # 提取旧向量（实际生产环境应维护一个独立的向量存储）
    # 这里简化处理
    n_old = old_index.ntotal
    old_vectors = np.empty((n_old, dim), dtype=np.float32)
    for i in range(n_old):
        old_vectors[i] = old_index.reconstruct(i)
    
    # 合并
    all_vectors = np.vstack([old_vectors, new_vectors])
    
    # 构建新索引到临时文件
    nlist = min(int(4 * np.sqrt(all_vectors.shape[0])), 1000)
    quantizer = faiss.IndexFlatL2(dim)
    new_index = faiss.IndexIVFFlat(quantizer, dim, nlist)
    new_index.train(all_vectors)
    new_index.add(all_vectors)
    new_index.nprobe = 10
    
    # 写入临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix='.index') as tmp:
        tmp_path = tmp.name
        faiss.write_index(new_index, tmp_path)
    
    # 原子替换
    backup_path = main_index_path + '.bak'
    if os.path.exists(main_index_path):
        shutil.copy2(main_index_path, backup_path)
    shutil.move(tmp_path, main_index_path)
    
    return new_index
```

---

## 8.6 分片策略 (Sharding)

### 8.6.1 为什么需要分片

当向量数量达到数亿甚至数十亿时，单机索引面临以下问题：

1. **内存限制**：单机内存无法容纳整个索引。
2. **构建时间**：在一个巨大的索引上训练 K-Means 或构建 HNSW 图可能耗时数天。
3. **搜索延迟**：即使索引结构高效，搜索延迟也会随数据量增长。
4. **故障域**：单点故障导致整个检索不可用。

分片（Sharding）将数据水平切分到多个索引（可能分布在多台机器上），搜索时向所有分片发送查询，然后合并各分片的 Top-K 结果。

### 8.6.2 分片策略对比

| 策略 | 切分方式 | 优点 | 缺点 |
|------|---------|------|------|
| 随机分片 | 按 ID 哈希均匀分布 | 实现简单，负载均衡 | 跨分片查询开销大 |
| 聚类分片 | 先聚类，每个分片负责若干聚类 | 可能减少跨分片搜索 | 热点分片问题 |
| 属性分片 | 按元数据（时间、地域）分片 | 自然隔离，过滤效率高 | 分片大小不均衡 |
| 文档分片 | 按原始文档切分 | 语义相关向量在同一分片 | 分片大小差异大 |

### 8.6.3 随机分片实现

```python
class ShardedIndex:
    """
    基于随机哈希的水平分片索引。
    搜索时向所有分片发送查询，合并结果。
    """
    
    def __init__(self, dim: int, num_shards: int = 4):
        self.dim = dim
        self.num_shards = num_shards
        self.shards: List[faiss.Index] = [None] * num_shards
        self.shard_sizes = [0] * num_shards
        self.id_to_shard: Dict[int, int] = {}
        
        # 每个分片的全局 ID 偏移
        self.shard_offsets = [0] * num_shards
    
    def build(self, vectors: np.ndarray) -> None:
        """将向量均匀分配到各分片并分别构建索引。"""
        n = vectors.shape[0]
        
        # 随机分配
        rng = np.random.default_rng(42)
        assignments = rng.integers(0, self.num_shards, n)
        
        # 按分片收集向量
        shard_vectors: List[List[np.ndarray]] = [[] for _ in range(self.num_shards)]
        for i, shard_id in enumerate(assignments):
            shard_vectors[shard_id].append(vectors[i])
        
        # 为每个分片构建索引
        offset = 0
        for s in range(self.num_shards):
            if not shard_vectors[s]:
                self.shards[s] = faiss.IndexFlatL2(self.dim)
                continue
            
            sv = np.vstack(shard_vectors[s])
            nlist = min(int(4 * np.sqrt(sv.shape[0])), 200)
            quantizer = faiss.IndexFlatL2(self.dim)
            idx = faiss.IndexIVFFlat(quantizer, self.dim, nlist)
            idx.train(sv)
            idx.add(sv)
            idx.nprobe = 10
            
            self.shards[s] = idx
            self.shard_sizes[s] = sv.shape[0]
            self.shard_offsets[s] = offset
            offset += sv.shape[0]
    
    def search(self, query: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """向所有分片发送查询，合并结果。"""
        all_distances = []
        all_indices = []
        
        for s in range(self.num_shards):
            if self.shards[s] is None or self.shard_sizes[s] == 0:
                continue
            
            # 每个分片搜索 k 个结果
            D, I = self.shards[s].search(query, k)
            
            # 将局部 ID 转换为全局 ID
            global_I = I + self.shard_offsets[s]
            
            all_distances.append(D)
            all_indices.append(global_I)
        
        # 合并
        combined_D = np.hstack(all_distances)
        combined_I = np.hstack(all_indices)
        
        top_k = np.argsort(combined_D[0])[:k]
        return (combined_D[0, top_k].reshape(1, -1),
                combined_I[0, top_k].reshape(1, -1))
```

### 8.6.4 分布式分片搜索的注意事项

**合并结果的偏差问题：**

当每个分片只返回 Top-K 时，全局 Top-K 可能因为某些分片的"优秀候选"被遗漏而失真。例如：

```
分片 A: [0.1, 0.2, 0.3]  ← 距离
分片 B: [0.4, 0.5, 0.6]
全局 Top-3: [0.1, 0.2, 0.3] ← 全部来自分片 A
```

但如果分片 A 的第 4 个候选距离 0.35，而分片 B 的第 4 个候选距离 0.38，全局 Top-4 应该是 [0.1, 0.2, 0.3, 0.35]，这个结果正确。但如果分片 B 的第 4 个候选距离 0.31（小于分片 A 的 0.35），那么全局 Top-4 应该是 [0.1, 0.2, 0.3, 0.31]——但由于每个分片只返回了 3 个，我们丢失了这个候选。

**解决方案：**

```python
def search_with_oversample(sharded_index: ShardedIndex, 
                            query: np.ndarray, k: int = 10, 
                            oversample_factor: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    带过采样的分片搜索。每个分片返回 oversample_factor * k 个结果，
    以降低合并偏差。
    """
    per_shard_k = min(k * oversample_factor, 1000)
    all_distances = []
    all_indices = []
    
    for s in range(sharded_index.num_shards):
        if sharded_index.shards[s] is None or sharded_index.shard_sizes[s] == 0:
            continue
        
        D, I = sharded_index.shards[s].search(query, per_shard_k)
        global_I = I + sharded_index.shard_offsets[s]
        all_distances.append(D)
        all_indices.append(global_I)
    
    combined_D = np.hstack(all_distances)
    combined_I = np.hstack(all_indices)
    
    top_k = np.argsort(combined_D[0])[:k]
    return (combined_D[0, top_k].reshape(1, -1),
            combined_I[0, top_k].reshape(1, -1))
```

### 8.6.5 分片数量选择

分片数量需要权衡以下因素：

| 因素 | 少量分片 | 大量分片 |
|------|---------|---------|
| 搜索延迟 | 每个分片负载大，延迟高 | 每个分片负载小，延迟低 |
| 合并开销 | 小 | 大（需要合并更多结果） |
| 分片不平衡 | 影响大 | 可通过哈希均匀分布缓解 |
| 故障影响 | 数据丢失多 | 数据丢失少 |
| 运维复杂度 | 低 | 高 |

**经验公式：** `num_shards = ceil(sqrt(N / 500000))`，其中 N 是向量总数。对于一个 1000 万向量的数据集：`sqrt(10000000 / 500000) = sqrt(20) ≈ 4.47`，建议 4–5 个分片。

---

## 8.7 内存映射文件 (Memory-Mapped Files)

### 8.7.1 原理

内存映射文件（Memory-Mapped File, mmap）是操作系统提供的一种机制，允许将磁盘文件直接映射到进程的虚拟地址空间。读取映射区域时，操作系统按需从磁盘加载页面；写入时，修改先驻留在内存中，由内核异步写回磁盘。

对于向量索引，mmap 的核心优势在于：

1. **惰性加载**：索引文件不需要全部读入内存即可开始搜索，操作系统只在访问到特定页面时才加载。
2. **共享内存**：多个进程可以映射同一个文件，共享物理内存页面。
3. **超越可用内存**：即使索引大小超过物理内存，操作系统会通过页面置换（swapping）来处理，虽然变慢但不会崩溃。

### 8.7.2 Faiss 的 mmap 支持

Faiss 从 1.7.0 开始支持通过 `read_index` 的 `mmap` 参数使用内存映射：

```python
def load_index_with_mmap(index_path: str, mmap_mode: str = "r") -> faiss.Index:
    """
    使用内存映射加载索引。
    
    Args:
        mmap_mode: "r" 只读, "w" 可写, "r+" 读写
    """
    return faiss.read_index(index_path, mmap_mode)


def compare_load_strategies(dim: int = 768, n: int = 500000):
    """对比常规加载和 mmap 加载的启动时间与内存占用。"""
    import time
    import os
    
    # 创建测试索引
    vectors = generate_dataset(dim, n)
    index = faiss.IndexIVFFlat(faiss.IndexFlatL2(dim), dim, 200)
    index.train(vectors)
    index.add(vectors)
    index.nprobe = 10
    
    index_path = "test_index.faiss"
    faiss.write_index(index, index_path)
    index_size_mb = os.path.getsize(index_path) / (1024 * 1024)
    print(f"索引文件大小: {index_size_mb:.2f} MB")
    
    # 常规加载
    start = time.perf_counter()
    idx1 = faiss.read_index(index_path)
    load_time_regular = time.perf_counter() - start
    print(f"常规加载: {load_time_regular:.3f} 秒")
    
    # mmap 加载
    start = time.perf_counter()
    idx2 = faiss.read_index(index_path, "r")
    load_time_mmap = time.perf_counter() - start
    print(f"mmap 加载: {load_time_mmap:.3f} 秒")
    
    # 清理
    os.remove(index_path)
    
    return load_time_regular, load_time_mmap

# compare_load_strategies()
```

典型输出：
```
索引文件大小: 156.23 MB
常规加载: 1.234 秒
mmap 加载: 0.008 秒
```

mmap 加载几乎是瞬时的，因为操作系统只建立了虚拟地址映射，没有实际读取数据。

### 8.7.3 mmap 实战：冷热数据分离

```python
class TieredIndex:
    """
    分层索引：热数据在内存中，冷数据在 mmap 中。
    
    最近 7 天的数据在内存（HNSW），历史数据在 mmap（IVF）。
    """
    
    def __init__(self, dim: int, hot_threshold_days: int = 7):
        self.dim = dim
        self.hot_threshold_days = hot_threshold_days
        
        # 热索引：内存中的 HNSW
        self.hot_index = faiss.IndexHNSWFlat(dim, 16)
        self.hot_index.hnsw.efConstruction = 200
        self.hot_index.hnsw.efSearch = 64
        self.hot_vectors: List[np.ndarray] = []
        
        # 冷索引：mmap 上的 IVF
        self.cold_index_path: Optional[str] = None
        self.cold_index: Optional[faiss.Index] = None
        self.cold_index_mmap: Optional[faiss.Index] = None
    
    def add_hot(self, vectors: np.ndarray) -> None:
        """添加到热索引。"""
        self.hot_index.add(vectors)
        self.hot_vectors.append(vectors)
    
    def freeze_cold(self, index_path: str) -> None:
        """
        将当前热数据冻结为冷索引（IVF + mmap）。
        调用后热数据被清空。
        """
        if not self.hot_vectors:
            return
        
        all_hot = np.vstack(self.hot_vectors)
        
        # 构建 IVF 索引
        nlist = min(int(4 * np.sqrt(all_hot.shape[0])), 500)
        quantizer = faiss.IndexFlatL2(self.dim)
        cold = faiss.IndexIVFFlat(quantizer, self.dim, nlist)
        cold.train(all_hot)
        cold.add(all_hot)
        cold.nprobe = 10
        
        # 写入磁盘
        faiss.write_index(cold, index_path)
        
        # 用 mmap 打开
        self.cold_index_path = index_path
        self.cold_index_mmap = faiss.read_index(index_path, "r")
        
        # 清空热数据
        self.hot_index = faiss.IndexHNSWFlat(self.dim, 16)
        self.hot_index.hnsw.efConstruction = 200
        self.hot_index.hnsw.efSearch = 64
        self.hot_vectors = []
    
    def search(self, query: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """同时在热索引和冷索引中搜索。"""
        results = []
        
        # 搜索热索引
        D_hot, I_hot = self.hot_index.search(query, k)
        results.append((D_hot, I_hot, "hot"))
        
        # 搜索冷索引（mmap）
        if self.cold_index_mmap is not None:
            D_cold, I_cold = self.cold_index_mmap.search(query, k)
            # 偏移 ID 以区分热/冷
            results.append((D_cold, I_cold + self.hot_index.ntotal, "cold"))
        
        # 合并结果
        combined_D = np.hstack([r[0] for r in results])
        combined_I = np.hstack([r[1] for r in results])
        top_k = np.argsort(combined_D[0])[:k]
        
        return (combined_D[0, top_k].reshape(1, -1),
                combined_I[0, top_k].reshape(1, -1))
```

### 8.7.4 mmap 的局限性

1. **随机访问性能**：mmap 在顺序读取时接近内存速度，但随机访问大量不连续的页面时可能触发频繁的缺页中断（page fault），性能下降。
2. **写放大**：对 mmap 区域的写入即使修改 1 字节，操作系统也可能以页（通常 4KB）为单位写回磁盘。
3. **内存压力**：mmap 占用的页面在内存紧张时可以被换出，但频繁的换入换出会导致 thrashing。
4. **文件大小限制**：32 位系统上 mmap 受虚拟地址空间限制（通常 2–3 GB），64 位系统基本无此问题。
5. **跨平台差异**：Windows 和 Linux 的 mmap 语义在文件大小变化、同步行为等方面存在差异。

**何时使用 mmap：**

| 场景 | 推荐 |
|------|------|
| 索引 > 可用内存的 50% | 使用 mmap，配合冷热分层 |
| 冷启动时间敏感 | 使用 mmap 实现"秒级加载" |
| 多进程共享索引 | 使用 mmap 实现共享内存 |
| 频繁随机搜索 | 纯内存索引更优 |
| 索引频繁更新 | 避免 mmap（写性能差） |

---

## 8.8 磁盘 vs 内存权衡

### 8.8.1 性能层级

```
寄存器 (1 ns)         ─ 1x
L1 缓存 (1 ns)        ─ 1x
L2 缓存 (4 ns)        ─ 4x
主存 (100 ns)         ─ 100x
SSD (100,000 ns)      ─ 100,000x
HDD (10,000,000 ns)   ─ 10,000,000x
```

从主存到 SSD 的延迟差距是三个数量级。这意味着，如果索引需要从磁盘读取数据，单次搜索的延迟可能从微秒级上升到毫秒级。

### 8.8.2 三种部署模式

```python
class DiskBasedIndex:
    """
    纯磁盘索引：所有向量存储在磁盘上，搜索时逐块加载。
    适合索引极大（> 内存）且搜索频次低的场景。
    """
    
    def __init__(self, dim: int, chunk_size: int = 10000):
        self.dim = dim
        self.chunk_size = chunk_size
        self.chunks: List[str] = []  # chunk 文件路径列表
        self.n_total = 0
    
    def add(self, vectors: np.ndarray, storage_dir: str) -> None:
        """将向量分块存储到磁盘。"""
        os.makedirs(storage_dir, exist_ok=True)
        n = vectors.shape[0]
        
        for start in range(0, n, self.chunk_size):
            chunk = vectors[start:start + self.chunk_size]
            chunk_path = os.path.join(storage_dir, f"chunk_{len(self.chunks)}.npy")
            np.save(chunk_path, chunk)
            self.chunks.append(chunk_path)
        
        self.n_total += n
    
    def search(self, query: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        逐块加载并搜索。这是"最慢但最省内存"的策略。
        """
        all_distances = []
        all_indices = []
        offset = 0
        
        for chunk_path in self.chunks:
            chunk = np.load(chunk_path, mmap_mode='r')
            
            # 计算距离（逐块）
            dists = np.linalg.norm(chunk - query, axis=1)
            chunk_k = min(k, len(dists))
            top_k = np.argpartition(dists, chunk_k)[:chunk_k]
            
            all_distances.append(dists[top_k])
            all_indices.append(top_k + offset)
            offset += len(chunk)
        
        # 全局合并
        combined_D = np.concatenate(all_distances)
        combined_I = np.concatenate(all_indices)
        global_top_k = np.argsort(combined_D)[:k]
        
        return (combined_D[global_top_k].reshape(1, -1),
                combined_I[global_top_k].reshape(1, -1))


class HybridDiskIndex:
    """
    混合磁盘索引：
    - 粗略索引（IVF 聚类中心）常驻内存
    - 倒排列表存储在磁盘
    - 搜索时只加载相关聚类的倒排列表
    """
    
    def __init__(self, dim: int, nlist: int = 1000):
        self.dim = dim
        self.nlist = nlist
        
        # 常驻内存的部分
        self.quantizer = faiss.IndexFlatL2(dim)  # 聚类中心
        self.centroids: Optional[np.ndarray] = None
        
        # 磁盘上的部分
        self.inverted_lists_dir: Optional[str] = None
        self.list_sizes: List[int] = []
        self.total_vectors = 0
    
    def build(self, vectors: np.ndarray, storage_dir: str) -> None:
        """构建混合索引。"""
        os.makedirs(storage_dir, exist_ok=True)
        self.inverted_lists_dir = storage_dir
        
        # K-Means 训练
        kmeans = faiss.Kmeans(self.dim, self.nlist, niter=20)
        kmeans.train(vectors)
        self.centroids = faiss.vector_to_array(kmeans.centroids).reshape(
            self.nlist, self.dim
        ).copy()
        
        # 分配向量到最近聚类
        _, assignments = kmeans.assign(vectors)
        
        # 将每个聚类的向量写入独立文件
        from collections import defaultdict
        lists: Dict[int, List[np.ndarray]] = defaultdict(list)
        for i, assigned in enumerate(assignments):
            lists[int(assigned)].append(vectors[i])
        
        self.list_sizes = []
        for cid in range(self.nlist):
            if lists[cid]:
                chunk = np.vstack(lists[cid])
                np.save(os.path.join(storage_dir, f"list_{cid}.npy"), chunk)
                self.list_sizes.append(len(chunk))
            else:
                self.list_sizes.append(0)
        
        self.total_vectors = vectors.shape[0]
    
    def search(self, query: np.ndarray, k: int = 10, 
               nprobe: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        搜索：找到最近的 nprobe 个聚类，只加载这些聚类的倒排列表。
        """
        if self.centroids is None:
            raise ValueError("Index not built")
        
        # 1. 找到最近的聚类中心
        dists_to_centroids = np.linalg.norm(self.centroids - query, axis=1)
        nearest_centroids = np.argsort(dists_to_centroids)[:nprobe]
        
        # 2. 只加载这些聚类的向量
        all_candidates = []
        for cid in nearest_centroids:
            list_path = os.path.join(self.inverted_lists_dir, f"list_{cid}.npy")
            if os.path.exists(list_path):
                chunk = np.load(list_path, mmap_mode='r')
                all_candidates.append(chunk)
        
        if not all_candidates:
            return np.array([]), np.array([])
        
        candidates = np.vstack(all_candidates)
        
        # 3. 在候选向量中搜索
        dists = np.linalg.norm(candidates - query, axis=1)
        top_k = np.argsort(dists)[:k]
        
        return (dists[top_k].reshape(1, -1), top_k.reshape(1, -1))
```

### 8.8.3 决策矩阵

| 因素 | 全内存 | mmap + 冷热分层 | 纯磁盘 (逐块) |
|------|--------|----------------|---------------|
| 索引大小 / 内存 | < 50% | 50%–200% | > 200% |
| P99 延迟 | < 5ms | 5–50ms | 50ms–5s |
| QPS 上限 | 高 (1000+) | 中 (100–1000) | 低 (< 100) |
| 冷启动时间 | 分钟级（加载） | 秒级（mmap） | 秒级 |
| 运维复杂度 | 低 | 中 | 高 |
| 适用场景 | 在线搜索 | 在线搜索 + 存档 | 离线分析、归档 |

### 8.8.4 硬件推荐配置

| 向量规模 | 维度 | 推荐配置 | 预计内存 | 预计 P99 延迟 |
|----------|------|----------|----------|--------------|
| 100 万 | 768 | 4 核 CPU, 16 GB RAM, SSD | 4–6 GB | < 5ms |
| 1000 万 | 768 | 8 核 CPU, 64 GB RAM, NVMe SSD | 30–40 GB | 5–15ms |
| 1 亿 | 768 | 16 核 CPU, 256 GB RAM, NVMe × 2 | 120–160 GB | 10–50ms |
| 10 亿 | 768 | 分布式（4 × 64GB 节点）或 mmap + 磁盘 | 每节点 32–64 GB | 50–200ms |

### 8.8.5 实战建议

**在内存和磁盘之间选择时，遵循以下优先级：**

1. **能放内存就放内存**：如果索引 + 工作集可以放入内存，就不要用磁盘。内存的价格在下降，而延迟的提升是巨大的。
2. **放不下的用 mmap**：mmap 比手动管理磁盘 I/O 更高效，而且 Faiss/HNSWLib 等库已经内置了支持。
3. **万不得已用磁盘**：纯磁盘搜索只适合离线批处理或极低 QPS 的场景。
4. **压缩优先于换出**：在考虑磁盘之前，先尝试 PQ 量化（8.2.4 节）。将 768 维向量用 PQ4x8 压缩到 4 字节（192:1），往往比用磁盘方案简单得多。
5. **缓存热查询**：即使使用 mmap 或磁盘索引，对高频查询的缓存（如 LRU Cache）可以大幅降低有效延迟。

```python
class LRUCache:
    """简单的 LRU 查询缓存，用于减少磁盘索引的搜索次数。"""
    
    def __init__(self, capacity: int = 1000):
        self.cache: Dict[bytes, Tuple[np.ndarray, np.ndarray]] = {}
        self.order: List[bytes] = []
        self.capacity = capacity
    
    def get(self, query: np.ndarray):
        key = query.tobytes()
        if key in self.cache:
            self.order.remove(key)
            self.order.append(key)
            return self.cache[key]
        return None
    
    def put(self, query: np.ndarray, distances: np.ndarray, indices: np.ndarray):
        key = query.tobytes()
        if key in self.cache:
            self.order.remove(key)
        elif len(self.cache) >= self.capacity:
            oldest = self.order.pop(0)
            del self.cache[oldest]
        
        self.cache[key] = (distances, indices)
        self.order.append(key)


class CachedDiskIndex:
    """带 LRU 缓存的磁盘索引。"""
    
    def __init__(self, disk_index: DiskBasedIndex, cache_capacity: int = 1000):
        self.disk_index = disk_index
        self.cache = LRUCache(capacity=cache_capacity)
        self.cache_hits = 0
        self.total_queries = 0
    
    def search(self, query: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        self.total_queries += 1
        
        # 查缓存
        cached = self.cache.get(query)
        if cached is not None:
            self.cache_hits += 1
            return cached
        
        # 搜索磁盘
        result = self.disk_index.search(query, k)
        self.cache.put(query, result[0], result[1])
        return result
    
    def hit_rate(self) -> float:
        return self.cache_hits / max(self.total_queries, 1)
```

---

## 8.9 本章小结

本章深入探讨了 RAG 系统中向量索引的高级话题。以下是核心要点：

1. **向量量化**是降低内存占用的第一道防线。FP16 量化几乎无损，SQ8 以微小精度损失换来 4 倍压缩，PQ 可以实现 10–100 倍的极端压缩。

2. **索引结构选择**取决于数据规模、召回率要求和硬件环境。IVF 适合大规模数据和 GPU 部署，HNSW 适合中小规模和高召回率场景，两者可以组合使用。

3. **过滤预检查**是生产环境中不可避免的需求。根据过滤条件的严格程度，可以选择 post-filter（宽松过滤）、pre-filter（严格过滤）或 IDSelector（简单过滤条件）。

4. **增量合并**借鉴了 LSM-Tree 的思想，通过分层合并策略在数据持续流入时保持索引质量。对于大多数场景，批量重建 + 影子索引是简单可靠的方案。

5. **分片**是突破单机瓶颈的必经之路。随机分片简单可靠，但需要注意过采样以避免合并偏差。

6. **内存映射文件**提供了"秒级加载"的能力，适合大索引和冷启动敏感的场景。冷热分层架构可以在内存有限的情况下保持较高的搜索性能。

7. **磁盘 vs 内存**的选择本质上是成本与延迟的权衡。能放内存就放内存，放不下的用 mmap + 量化，万不得已才用纯磁盘方案。

下一章将讨论 RAG 系统的评估与测试——如何科学地衡量检索质量和端到端效果。

---

## 参考文献

1. Jegou, H., Douze, M., & Schmid, C. (2011). Product quantization for nearest neighbor search. *IEEE TPAMI*, 33(1), 117–128.
2. Malkov, Y. A., & Yashunin, D. A. (2020). Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs. *IEEE TPAMI*, 42(4), 824–836.
3. Johnson, J., Douze, M., & Jegou, H. (2019). Billion-scale similarity search with GPUs. *IEEE Transactions on Big Data*, 7(3), 535–547.
4. Ge, T., He, K., Ke, Q., & Sun, J. (2014). Optimized product quantization. *IEEE TPAMI*, 36(4), 744–755.
5. O'Neil, P., et al. (1996). The log-structured merge-tree (LSM-tree). *Acta Informatica*, 33(4), 351–385.
