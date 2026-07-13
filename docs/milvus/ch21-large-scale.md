# 第21章 海量数据RAG系统落地（千万级向量）

当RAG系统的知识库规模达到千万级甚至亿级向量时，通用的检索方案将面临严峻的性能挑战。本章将系统讲解面向海量数据的Milvus架构设计、索引选型、批量入库策略和高并发检索优化方案。

## 21.1 海量数据架构设计与索引选型

### 21.1.1 千万级向量的性能瓶颈

在千万级向量规模下，RAG系统将遇到以下典型瓶颈：

- **内存瓶颈**：原始向量数据占用巨大内存。以768维FP32向量为例，1000万条向量约需 768 × 4 × 10^7 ≈ 30GB 内存。加上索引结构和标量字段，单机通常无法承载。
- **检索延迟**：全量扫描的暴力检索（FLAT）在千万级数据上毫秒级响应不可行，必须依赖高效的ANN索引。
- **写入吞吐**：千万级数据的全量入库需要数小时甚至更长时间，批量写入和索引构建必须异步进行。
- **索引构建**：在千万级数据上构建HNSW或IVF索引需要大量CPU和内存资源，构建时间可能长达数十分钟。

### 21.1.2 索引选型策略

针对不同规模的数据量，推荐的索引方案如下：

| 数据规模 | 推荐索引 | 核心参数 | 内存占用 | 检索延迟（P99） | 召回率@100 |
|---------|---------|---------|---------|---------------|-----------|
| 100万-500万 | IVF_SQ8 | nlist=4096, nprobe=128 | 约3GB | 10-30ms | 95%+ |
| 500万-2000万 | IVF_PQ | m=8, nlist=8192, nprobe=256 | 约1.5GB | 20-50ms | 90%+ |
| 2000万-1亿 | HNSW | M=24, efConstruction=500, ef=200 | 约25GB | 5-15ms | 98%+ |
| 1亿以上 | DiskANN | 基于磁盘的图索引 | 约5GB内存 | 30-100ms | 95%+ |

**IVF_SQ8** 适合中等规模场景。它对原始向量做标量量化（FP32→INT8），内存占用降低75%，检索速度提升3-5倍，召回率损失控制在1-3%以内。

**IVF_PQ** 适合大规模场景。乘积量化（Product Quantization）将向量压缩为码本索引，压缩比可达8-32倍，但召回率损失较大，适合对精度要求不那么极致的场景。

**HNSW** 是千万级场景的首选。它基于层级可导航小世界图，检索延迟与数据规模呈对数增长关系，在千万级数据上仍能保持10ms以内的检索延迟。

```python
# HNSW 索引配置（推荐用于千万级场景）
index_params = {
    "index_type": "HNSW",
    "metric_type": "IP",        # 内积，配合归一化向量等效于余弦相似度
    "params": {
        "M": 24,                # 每个节点的最大连接数（越大精度越高，内存越大）
        "efConstruction": 500,  # 构建时的动态列表大小（越大构建越慢，精度越高）
    }
}

# 检索参数
search_params = {
    "ef": 200,  # 检索时的动态列表大小（越大召回越高，延迟越大）
}
```

### 21.1.3 集群架构设计

千万级向量场景必须采用 Milvus 集群部署模式：

```
┌─────────────────────────────────────────────────────────┐
│                   负载均衡器 (Nginx/HAProxy)               │
└────────────────────┬────────────────────────────────────┘
                     │
     ┌───────────────┼───────────────┐
     ▼               ▼               ▼
┌──────────┐  ┌──────────┐  ┌──────────┐
│   Proxy   │  │   Proxy   │  │   Proxy   │  接入层（无状态，可水平扩展）
└─────┬─────┘  └─────┬─────┘  └─────┬─────┘
      │               │               │
┌─────┴───────────────────────────────┴─────┐
│         RootCoord / QueryCoord / DataCoord    │  协调层
└─────┬───────────────────────────────┬─────┘
      │                               │
┌─────┴─────┐                  ┌─────┴─────┐
│ QueryNode  │  ...  │ QueryNode  │  执行层（检索节点，水平扩展）
└─────┬─────┘                  └─────┬─────┘
      │                               │
┌─────┴─────┐                  ┌─────┴─────┐
│ DataNode   │  ...  │ DataNode   │  执行层（写入节点，水平扩展）
└─────┬─────┘                  └─────┬─────┘
      │                               │
┌─────┴───────────────────────────────┴─────┐
│     MinIO (向量数据)  Etcd (元数据)         │  存储层
└───────────────────────────────────────────┘
```

关键配置建议：

- **QueryNode 数量**：建议与 CPU 核心数成正比，每 8 核 CPU 部署 1 个 QueryNode。1000 万向量/768 维场景下，建议至少 4 个 QueryNode。
- **DataNode 数量**：建议至少 2 个，保障写入高可用。
- **Proxy 数量**：至少 2 个，前端配置负载均衡。
- **MinIO 存储**：建议使用独立集群模式，SSD 磁盘，RAID 10。

## 21.2 批量异步入库与分片存储

### 21.2.1 批量异步入库策略

千万级数据的全量入库不能使用逐条插入，必须采用批量异步写入策略：

```python
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pymilvus import MilvusClient, Collection, utility

BATCH_SIZE = 10000      # 每批 1 万条
NUM_WORKERS = 8         # 并发写入线程数

def batch_insert(client, collection_name, all_data):
    """批量异步入库"""
    total = len(all_data)
    batches = [all_data[i:i+BATCH_SIZE] for i in range(0, total, BATCH_SIZE)]
    print(f"数据总量: {total}, 批次数量: {len(batches)}, 每批: {BATCH_SIZE}")

    def insert_batch(batch_idx, batch_data):
        """单个批次的插入任务"""
        start = time.time()
        try:
            ids = client.insert(
                collection_name=collection_name,
                data=batch_data,
            )
            elapsed = time.time() - start
            print(f"  批次 {batch_idx+1}/{len(batches)}: 插入 {len(batch_data)} 条, "
                  f"耗时 {elapsed:.2f}s, 速率 {len(batch_data)/elapsed:.0f} 条/s")
            return len(ids)
        except Exception as e:
            print(f"  批次 {batch_idx+1} 插入失败: {e}")
            return 0

    # 使用线程池并发写入
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [
            executor.submit(insert_batch, i, batch)
            for i, batch in enumerate(batches)
        ]
        total_inserted = sum(f.result() for f in as_completed(futures))

    print(f"入库完成: {total_inserted}/{total}")
    return total_inserted
```

### 21.2.2 异步索引构建

数据入库后，索引构建是一个耗时的操作。在千万级场景下，应该在所有数据入库完成后再构建索引，避免边入库边构建带来的性能抖动：

```python
def build_index_async(client, collection_name, index_params):
    """异步构建索引"""
    print("开始构建索引（异步操作）...")
    start = time.time()

    client.create_index(
        collection_name=collection_name,
        index_params=index_params,
    )

    # 等待索引构建完成
    while True:
        progress = utility.index_building_progress(collection_name)
        status = progress.get("status", "")
        if status == "finished":
            elapsed = time.time() - start
            print(f"索引构建完成，耗时 {elapsed:.2f}s")
            break
        elif status == "failed":
            print("索引构建失败！")
            break
        else:
            indexed = progress.get("indexed_rows", 0)
            total = progress.get("total_rows", 0)
            print(f"  索引构建进度: {indexed}/{total} ({indexed/total*100:.1f}%)")
            time.sleep(5)

    # 加载集合到内存
    client.load_collection(collection_name)
    print("集合已加载到内存，准备就绪")
```

### 21.2.3 分片存储策略

Milvus 支持将集合数据分片存储，以充分利用多台机器的存储和计算能力：

```python
def create_sharded_collection(client, collection_name, dim, num_shards=8):
    """创建分片集合"""
    schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field("content", DataType.VARCHAR, max_length=4096)
    schema.add_field("source", DataType.VARCHAR, max_length=256)
    schema.add_field("timestamp", DataType.INT64)

    index_params = MilvusClient.prepare_index_params()
    index_params.add_index("vector", index_type="HNSW", metric_type="IP",
                           params={"M": 24, "efConstruction": 500})

    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params,
        num_shards=num_shards,  # 分片数量，建议等于 DataNode 数量
    )
    print(f"创建分片集合，分片数: {num_shards}")
```

**分片数量建议**：
- 每个分片建议承载 200万-500万 条向量
- 分片数量建议等于 DataNode 数量的整数倍
- 分片数量不要超过 64，过多的分片会带来管理开销

## 21.3 高并发检索优化与压测

### 21.3.1 连接池与客户端优化

高并发场景下，客户端连接的复用至关重要：

```python
from pymilvus import connections, Collection

class MilvusConnectionPool:
    """Milvus 连接池管理器"""

    def __init__(self, host, port, pool_size=10):
        self.host = host
        self.port = port
        self.pool_size = pool_size
        self._init_connections()

    def _init_connections(self):
        """初始化连接池"""
        for i in range(self.pool_size):
            alias = f"conn_{i}"
            connections.connect(
                alias=alias,
                host=self.host,
                port=self.port,
                secure=False,
            )
        print(f"连接池初始化完成，共 {self.pool_size} 个连接")

    def get_collection(self, name, alias=None):
        """获取集合对象"""
        conn_alias = alias or f"conn_{random.randint(0, self.pool_size - 1)}"
        return Collection(name=name, using=conn_alias)

    def close_all(self):
        """关闭所有连接"""
        for i in range(self.pool_size):
            connections.disconnect(alias=f"conn_{i}")
```

### 21.3.2 检索参数优化

高并发场景下，需要在召回率和延迟之间找到平衡点：

```python
# 低延迟模式（适合高并发在线服务）
search_params_fast = {
    "ef": 64,         # 较小的 ef 值，牺牲少量召回率换取低延迟
    "metric_type": "IP",
}

# 高精度模式（适合离线批量处理）
search_params_accurate = {
    "ef": 512,        # 较大的 ef 值，召回率更高但延迟更大
    "metric_type": "IP",
}

# 自适应模式：根据并发量动态调整
def adaptive_search(collection, query_vector, current_qps, top_k=100):
    """根据当前 QPS 自适应调整检索参数"""
    if current_qps > 1000:
        ef = 64    # 高并发时降低精度
    elif current_qps > 500:
        ef = 128
    elif current_qps > 100:
        ef = 256
    else:
        ef = 512   # 低并发时追求高精度

    results = collection.search(
        data=[query_vector],
        anns_field="vector",
        param={"ef": ef, "metric_type": "IP"},
        limit=top_k,
    )
    return results
```

### 21.3.3 缓存策略

引入缓存可以显著降低 Milvus 的查询压力：

```python
import hashlib
import json
from collections import OrderedDict

class LRUCache:
    """LRU 缓存（避免重复查询相同或相似的请求）"""

    def __init__(self, capacity=10000, ttl=300):
        self.cache = OrderedDict()
        self.capacity = capacity
        self.ttl = ttl

    def _make_key(self, query, top_k, filters):
        """生成缓存键"""
        raw = f"{query}_{top_k}_{json.dumps(filters, sort_keys=True)}"
        return hashlib.md5(raw.encode()).hexdigest()

    def get(self, query, top_k, filters=None):
        key = self._make_key(query, top_k, filters)
        if key not in self.cache:
            return None
        value, timestamp = self.cache[key]
        if time.time() - timestamp > self.ttl:
            del self.cache[key]
            return None
        # 移到末尾（LRU）
        self.cache.move_to_end(key)
        return value

    def set(self, query, top_k, filters, results):
        key = self._make_key(query, top_k, filters)
        self.cache[key] = (results, time.time())
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
```

### 21.3.4 性能压测方案

```python
import time
import threading
from statistics import mean, median, stdev

def performance_test(client, collection_name, test_queries,
                     concurrency_levels=[1, 10, 50, 100, 200]):
    """性能压测：测试不同并发级别下的 QPS 和延迟"""
    for concurrency in concurrency_levels:
        print(f"\n并发级别: {concurrency}")
        latencies = []
        errors = 0
        lock = threading.Lock()

        def worker():
            nonlocal errors
            for q in test_queries:
                start = time.time()
                try:
                    client.search(
                        collection_name=collection_name,
                        data=[q["vector"]],
                        limit=100,
                    )
                    latency = (time.time() - start) * 1000  # ms
                    with lock:
                        latencies.append(latency)
                except Exception as e:
                    with lock:
                        errors += 1

        threads = [threading.Thread(target=worker) for _ in range(concurrency)]
        start = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.time() - start

        total_queries = len(test_queries) * concurrency
        qps = total_queries / elapsed if elapsed > 0 else 0

        print(f"  总查询数: {total_queries}")
        print(f"  总耗时: {elapsed:.2f}s")
        print(f"  QPS: {qps:.0f}")
        print(f"  平均延迟: {mean(latencies):.1f}ms" if latencies else "  N/A")
        print(f"  P50延迟: {median(latencies):.1f}ms" if latencies else "  N/A")
        print(f"  P99延迟: {sorted(latencies)[int(len(latencies)*0.99)]:.1f}ms" if latencies else "  N/A")
        print(f"  错误数: {errors}")
```

### 21.3.5 千万级场景最佳实践总结

| 优化维度 | 具体措施 | 预期效果 |
|---------|---------|---------|
| 索引选型 | HNSW M=24, efConstruction=500 | 单次检索 <10ms |
| 批量写入 | 每批 1万条，8线程并发 | 写入速率 >5万条/s |
| 分片策略 | 8-16 分片，均衡分布 | 并行检索，降低单节点压力 |
| 连接池 | 10-20 个连接复用 | 避免连接建立开销 |
| 缓存策略 | LRU 缓存，TTL 5分钟 | 缓存命中率 30-50% |
| 检索参数 | ef 自适应调整 | 高并发时保障低延迟 |
| 硬件配置 | SSD 磁盘，64GB+ 内存 | 减少 IO 等待，保障内存索引 |

## 本章小结

千万级向量的RAG系统落地是一项系统工程，涉及索引选型、集群架构、批量入库、高并发检索等多个维度的优化。核心要点包括：

1. **索引选型是基础**：千万级场景推荐 HNSW 索引，亿级场景推荐 DiskANN。
2. **集群架构是保障**：必须采用分布式集群部署，Proxy、QueryNode、DataNode 均可水平扩展。
3. **批量异步入库提效率**：使用多线程并发写入，每批 1万条，入库速率可达 5万条/秒以上。
4. **缓存和连接池优化**：LRU 缓存减少重复查询，连接池复用减少连接开销。
5. **自适应参数调优**：根据实时 QPS 动态调整检索参数，在高并发和低延迟之间取得平衡。

海量数据RAG的核心理念是"分层优化、分而治之"——通过索引压缩降低数据规模、通过分片分散计算压力、通过缓存过滤重复请求、通过并发提升吞吐能力。
