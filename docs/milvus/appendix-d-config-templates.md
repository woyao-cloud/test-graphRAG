# 附录D RAG优化参数终极配置模板

本附录提供了面向不同数据规模的完整配置模板，涵盖 Milvus 索引参数、检索参数、部署配置和 Embedding 选型建议。读者可根据自己的数据量级直接复制使用，并在此基础上微调。

## D.1 小数据量配置模板（<10万向量）

### D.1.1 适用场景

- 个人知识库、小型团队文档管理
- 产品文档问答、API 文档检索
- 创业初期 MVP 验证
- 单机部署，资源有限（4GB 内存即可）

### D.1.2 Milvus 部署配置

```yaml
# docker-compose.yml — 小数据量单机部署
version: '3.5'

services:
  etcd:
    container_name: milvus-etcd
    image: quay.io/coreos/etcd:v3.5.5
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/etcd:/etcd
    command: etcd -advertise-client-urls=http://127.0.0.1:2379
             -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd

  minio:
    container_name: milvus-minio
    image: minio/minio:RELEASE.2023-03-20T20-16-18Z
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/minio:/minio_data
    command: minio server /minio_data --console-address ":9001"

  standalone:
    container_name: milvus-standalone
    image: milvusdb/milvus:v2.4.1
    command: ["milvus", "run", "standalone"]
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/milvus:/var/lib/milvus
    ports:
      - "19530:19530"
      - "9091:9091"
    depends_on:
      - etcd
      - minio

networks:
  default:
    name: milvus
```

### D.1.3 Python 配置模板

```python
"""
小数据量配置模板（<10万向量）
特点：追求最高召回率，不追求极致速度
"""

# 基础配置
MILVUS_URI = "http://localhost:19530"
COLLECTION_NAME = "kb_small"
DIM = 768          # BGE-base-zh-v1.5 的维度

# Schema 配置
def create_schema_small():
    schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=True)
    schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=DIM)
    schema.add_field("content", DataType.VARCHAR, max_length=4096)
    schema.add_field("source", DataType.VARCHAR, max_length=256)
    schema.add_field("timestamp", DataType.INT64)
    return schema

# 索引配置 — 小数据量用 FLAT（精确检索）
def create_index_small():
    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="FLAT",       # 暴力检索，100% 召回率
        metric_type="IP",        # 内积（配合归一化向量）
    )
    return index_params

# 检索参数
SEARCH_PARAMS_SMALL = {
    "metric_type": "IP",
}

# Embedding 推荐
EMBEDDING_MODEL_SMALL = "BAAI/bge-base-zh-v1.5"
# 备选：text-embedding-3-small（OpenAI，1536维，需调整 DIM）
```

## D.2 中数据量配置模板（10万-100万）

### D.2.1 适用场景

- 中型企业知识库（各部门文档汇总）
- 垂直领域知识库（医疗文献、法律法规）
- 电商商品库（数十万商品）
- 单机部署或小型集群（16GB+ 内存）

### D.2.2 Python 配置模板

```python
"""
中数据量配置模板（10万-100万向量）
特点：平衡召回率与检索速度，使用 IVF 索引
"""

MILVUS_URI = "http://localhost:19530"
COLLECTION_NAME = "kb_medium"
DIM = 768

# Schema 配置（增加更多标量字段用于过滤）
def create_schema_medium():
    schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=DIM)
    schema.add_field("title", DataType.VARCHAR, max_length=512)
    schema.add_field("content", DataType.VARCHAR, max_length=4096)
    schema.add_field("category", DataType.VARCHAR, max_length=128)
    schema.add_field("tags", DataType.VARCHAR, max_length=512)
    schema.add_field("timestamp", DataType.INT64)
    return schema

# 索引配置 — IVF_FLAT
def create_index_medium():
    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="IVF_FLAT",    # IVF 倒排索引
        metric_type="IP",
        params={"nlist": 2048},   # nlist = sqrt(n) 的量级
    )
    return index_params

# 检索参数
def get_search_params_medium(nprobe=128):
    """
    nprobe 调优建议：
    - 高召回场景：nprobe=256（召回率约 98%）
    - 平衡场景：  nprobe=128（召回率约 95%）
    - 高性能场景：nprobe=64  （召回率约 90%）
    """
    return {
        "metric_type": "IP",
        "params": {"nprobe": nprobe},
    }

# Embedding 推荐
EMBEDDING_MODEL_MEDIUM = "BAAI/bge-large-zh-v1.5"
```

### D.2.3 性能优化要点

- **批量写入**：每批 2000-5000 条，避免单条写入
- **数据分区**：按业务模块（如部门、文档类型）分区
- **定期 compact**：每周执行一次 compaction，合并小数据段
- **索引重建**：数据量增长超过 50% 后重建索引

## D.3 大数据量配置模板（100万-1000万）

### D.3.1 适用场景

- 大型企业知识库（全公司文档、邮件、聊天记录）
- 文献检索平台（百万级学术论文）
- 电商平台商品搜索（数百万商品）
- 集群部署，建议 64GB+ 内存

### D.3.2 Python 配置模板

```python
"""
大数据量配置模板（100万-1000万向量）
特点：追求高吞吐和低延迟，使用 HNSW 或 IVF_SQ8
"""

MILVUS_URI = "http://localhost:19530"
COLLECTION_NAME = "kb_large"
DIM = 768

# Schema 配置
def create_schema_large():
    schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=DIM)
    schema.add_field("title", DataType.VARCHAR, max_length=512)
    schema.add_field("content", DataType.VARCHAR, max_length=4096)
    schema.add_field("category", DataType.VARCHAR, max_length=128)
    schema.add_field("source", DataType.VARCHAR, max_length=256)
    schema.add_field("timestamp", DataType.INT64)
    return schema

# 索引配置（二选一）
def create_index_large_hnsw():
    """HNSW 索引 — 低延迟首选"""
    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="HNSW",
        metric_type="IP",
        params={
            "M": 24,                # 连接数（16-48），越大精度越高
            "efConstruction": 500,  # 构建搜索宽度（100-500）
        },
    )
    return index_params

def create_index_large_ivfsq8():
    """IVF_SQ8 索引 — 低内存占用首选"""
    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="IVF_SQ8",    # 量化压缩，内存降低 75%
        metric_type="IP",
        params={"nlist": 4096},  # nlist 随数据量增大
    )
    return index_params

# 检索参数
SEARCH_PARAMS_HNSW = {
    "metric_type": "IP",
    "params": {
        "ef": 200,    # 检索搜索宽度（64-512），越大召回越高
    },
}

SEARCH_PARAMS_IVFSQ8 = {
    "metric_type": "IP",
    "params": {
        "nprobe": 256,  # 检索的聚类数（64-512）
    },
}

# HNSW 参数速查
HNSW_PARAM_GUIDE = {
    "高精度模式":  {"M": 48, "efConstruction": 500, "ef": 400},
    "平衡模式":    {"M": 24, "efConstruction": 500, "ef": 200},
    "高性能模式":  {"M": 16, "efConstruction": 200, "ef": 100},
}

# IVF_SQ8 参数速查
IVFSQ8_PARAM_GUIDE = {
    "高精度模式":  {"nlist": 8192, "nprobe": 512},
    "平衡模式":    {"nlist": 4096, "nprobe": 256},
    "高性能模式":  {"nlist": 2048, "nprobe": 128},
}
```

### D.3.3 集群部署配置

```yaml
# docker-compose.yml — 大数据量集群部署（部分）
services:
  # ... etcd, minio 同前 ...

  querynode:
    image: milvusdb/milvus:v2.4.1
    command: ["milvus", "run", "querynode"]
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
      QUERY_NODE_CACHE_SIZE: 8     # 每个 QueryNode 缓存 8GB
    deploy:
      replicas: 3                  # 3 个 QueryNode 节点
      resources:
        limits:
          memory: 16G
          cpus: '4'

  datanode:
    image: milvusdb/milvus:v2.4.1
    command: ["milvus", "run", "datanode"]
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
    deploy:
      replicas: 2
      resources:
        limits:
          memory: 8G
          cpus: '2'
```

### D.3.4 批量入库优化

```python
# 大数据量批量入库配置
BULK_INSERT_CONFIG = {
    "batch_size": 5000,          # 每批 5000 条
    "num_workers": 8,            # 8 线程并发写入
    "flush_interval": 100000,    # 每 10 万条 flush 一次
    "index_build_after_insert": True,  # 全部入库后构建索引
}
```

## D.4 海量数据配置模板（>1000万）

### D.4.1 适用场景

- 全网级别搜索引擎
- 大型电商平台（千万级商品）
- 社交平台内容检索
- 多节点集群部署，128GB+ 内存，SSD 存储

### D.4.2 Python 配置模板

```python
"""
海量数据配置模板（>1000万向量）
特点：极致性能优化，分片存储，DiskANN 或 HNSW
"""

MILVUS_URI = "http://localhost:19530"
COLLECTION_NAME = "kb_huge"
DIM = 768
NUM_SHARDS = 16  # 分片数，建议 = DataNode 数量 * 2

# Schema 配置（精简字段，减少内存占用）
def create_schema_huge():
    schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=DIM)
    schema.add_field("content", DataType.VARCHAR, max_length=1024)  # 缩短 content 长度
    schema.add_field("category", DataType.INT64)                    # 用 INT 替代 VARCHAR
    schema.add_field("timestamp", DataType.INT64)
    return schema

# 索引配置
def create_index_huge():
    """
    亿级数据推荐 DiskANN 索引（基于磁盘的图索引）
    千万级数据推荐 HNSW 索引
    """
    if DATA_SIZE > 100_000_000:
        # 亿级：DiskANN
        index_params = MilvusClient.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="DISKANN",         # 磁盘索引，内存占用极低
            metric_type="IP",
        )
    else:
        # 千万级：HNSW
        index_params = MilvusClient.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="HNSW",
            metric_type="IP",
            params={"M": 32, "efConstruction": 500},
        )
    return index_params

# 检索参数
SEARCH_PARAMS_HUGE = {
    "metric_type": "IP",
    "params": {
        "ef": 128,          # HNSW 搜索宽度（降低以提升速度）
        "search_length": 100,  # DiskANN 搜索长度
    },
}

# 高并发连接池配置
CONNECTION_POOL_CONFIG = {
    "pool_size": 20,         # 连接池大小
    "timeout": 30,           # 超时时间（秒）
    "retry_count": 3,        # 重试次数
    "retry_interval": 0.5,  # 重试间隔（秒）
}
```

### D.4.3 分片与负载均衡

```python
# 海量数据分片创建
def create_huge_collection(client):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=create_schema_huge(),
        index_params=create_index_huge(),
        num_shards=NUM_SHARDS,
    )

# 负载均衡检索
def balanced_search(client, query_vector, top_k=100):
    """跨分片负载均衡检索"""
    # 使用多个连接轮询不同的 QueryNode
    from itertools import cycle

    connections_pool = cycle(range(CONNECTION_POOL_CONFIG["pool_size"]))
    conn_idx = next(connections_pool)
    alias = f"conn_{conn_idx}"

    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vector],
        limit=top_k,
        using=alias,  # 使用指定连接
    )
    return results
```

### D.4.4 硬件配置建议

| 资源 | 最小配置 | 推荐配置 | 说明 |
|------|---------|---------|------|
| CPU | 16 核 | 32 核 | 索引构建和检索都需要多核 |
| 内存 | 64 GB | 128 GB | HNSW 索引需加载到内存 |
| 磁盘 | 500 GB SSD | 1 TB NVMe SSD | DiskANN 需要高速磁盘 |
| 网络 | 1 Gbps | 10 Gbps | 集群节点间数据传输 |

## D.5 配置速查表

### D.5.1 索引参数速查

| 数据量 | 索引类型 | 核心参数 | 内存/向量 | 召回率 | P50延迟 |
|-------|---------|---------|----------|-------|--------|
| <10万 | FLAT | — | 4×dim bytes | 100% | 5ms |
| 10万-100万 | IVF_FLAT | nlist=2048, nprobe=128 | 4×dim bytes | 95-98% | 10ms |
| 10万-100万 | IVF_SQ8 | nlist=2048, nprobe=128 | 1×dim bytes | 93-97% | 5ms |
| 100万-1000万 | HNSW | M=24, ef=200 | ~12×dim bytes | 98-99% | 8ms |
| 100万-1000万 | IVF_SQ8 | nlist=4096, nprobe=256 | 1×dim bytes | 92-96% | 15ms |
| >1000万 | HNSW | M=32, ef=128 | ~16×dim bytes | 96-98% | 12ms |
| >1亿 | DISKANN | search_length=100 | ~0.5×dim bytes | 94-97% | 30ms |

### D.5.2 Embedding 模型速查

| 模型 | 维度 | 适用场景 | 语言 | 推荐数据量 |
|------|------|---------|------|-----------|
| BAAI/bge-base-zh-v1.5 | 768 | 通用中文 | 中文 | 全部 |
| BAAI/bge-large-zh-v1.5 | 1024 | 高精度中文 | 中文 | <500万 |
| BAAI/bge-small-zh-v1.5 | 512 | 快速检索 | 中文 | >500万 |
| text-embedding-3-small | 1536 | 通用多语言 | 多语言 | 全部 |
| text-embedding-3-large | 3072 | 高精度多语言 | 多语言 | <100万 |

### D.5.3 部署配置速查

| 数据量 | 部署模式 | 节点数量 | 内存需求 | 磁盘需求 |
|-------|---------|---------|---------|---------|
| <10万 | Standalone | 1 | 4 GB | 20 GB |
| 10万-100万 | Standalone | 1 | 16 GB | 100 GB |
| 100万-1000万 | 集群 | 3-5 | 64 GB | 500 GB |
| >1000万 | 集群 | 5-10 | 128 GB | 2 TB+ |

## D.6 快速启动脚本

```python
"""
config_selector.py — 根据数据量自动选择配置
"""

def select_config(num_vectors, dim=768):
    """根据数据量自动选择配置"""
    if num_vectors < 100_000:
        return {
            "index_type": "FLAT",
            "index_params": {},
            "search_params": {"metric_type": "IP"},
            "deploy_mode": "standalone",
            "embedding_model": "BAAI/bge-base-zh-v1.5",
            "batch_size": 1000,
            "description": "小数据量精确检索",
        }
    elif num_vectors < 1_000_000:
        return {
            "index_type": "IVF_FLAT",
            "index_params": {"nlist": 2048},
            "search_params": {"metric_type": "IP", "params": {"nprobe": 128}},
            "deploy_mode": "standalone",
            "embedding_model": "BAAI/bge-large-zh-v1.5",
            "batch_size": 5000,
            "description": "中数据量平衡模式",
        }
    elif num_vectors < 10_000_000:
        return {
            "index_type": "HNSW",
            "index_params": {"M": 24, "efConstruction": 500},
            "search_params": {"metric_type": "IP", "params": {"ef": 200}},
            "deploy_mode": "cluster",
            "embedding_model": "BAAI/bge-base-zh-v1.5",
            "batch_size": 10000,
            "num_shards": 8,
            "description": "大数据量高性能模式",
        }
    else:
        return {
            "index_type": "DISKANN" if num_vectors > 100_000_000 else "HNSW",
            "index_params": {} if num_vectors > 100_000_000 else {"M": 32, "efConstruction": 500},
            "search_params": {"metric_type": "IP", "params": {"ef": 128}},
            "deploy_mode": "cluster",
            "embedding_model": "BAAI/bge-small-zh-v1.5",
            "batch_size": 50000,
            "num_shards": 16,
            "description": "海量数据极致性能模式",
        }


# 使用示例
config = select_config(500_000)
print(f"推荐配置: {config['description']}")
print(f"索引类型: {config['index_type']}")
print(f"部署模式: {config['deploy_mode']}")
```

## 本章小结

本附录提供了面向四种数据规模的完整配置模板，核心要点包括：

1. **小数据量（<10万）**：FLAT 索引追求精确检索，单机部署，无需调参。
2. **中数据量（10万-100万）**：IVF_FLAT 索引平衡速度与精度，nlist 和 nprobe 是核心调优参数。
3. **大数据量（100万-1000万）**：HNSW 索引实现低延迟检索，集群部署保障高可用。
4. **海量数据（>1000万）**：DiskANN 或 HNSW 索引，多分片存储，连接池和缓存优化。

所有模板均可直接复制使用，只需根据实际数据量和硬件资源调整索引参数即可。建议从平衡模式开始，然后根据实际效果逐步优化。
