# 附录A：常用命令与SDK速查表

## A.1 MilvusClient 常用 API 速查

### A.1.1 客户端初始化

```python
from pymilvus import MilvusClient

# 本地 Milvus 服务
client = MilvusClient(uri="http://localhost:19530")

# Zilliz Cloud（托管服务）
client = MilvusClient(
    uri="https://your-instance.zillizcloud.com:19530",
    token="your-api-key"
)

# 使用用户名密码认证
client = MilvusClient(
    uri="http://localhost:19530",
    user="admin",
    password="your-password"
)
```

### A.1.2 集合（Collection）操作

```python
# === 创建集合 ===
# 简化方式（推荐快速原型）
client.create_collection(
    collection_name="demo_collection",
    dimension=768,
    metric_type="COSINE",
    auto_id=True,           # 自动生成主键
    id_type="int64"         # 主键类型
)

# 完整方式（生产环境推荐）
from pymilvus import DataType

schema = {
    "fields": [
        {"name": "id", "type": DataType.INT64, "is_primary": True, "auto_id": True},
        {"name": "embedding", "type": DataType.FLOAT_VECTOR, "params": {"dim": 768}},
        {"name": "text", "type": DataType.VARCHAR, "max_length": 65535},
        {"name": "category", "type": DataType.VARCHAR, "max_length": 64},
        {"name": "score", "type": DataType.FLOAT},
        {"name": "timestamp", "type": DataType.INT64}
    ]
}

client.create_collection(
    collection_name="production_collection",
    schema=schema
)

# === 查看集合 ===
# 列出所有集合
collections = client.list_collections()

# 查看集合详情
info = client.describe_collection("production_collection")

# 查看集合统计信息
stats = client.get_collection_stats("production_collection")
print(f"实体总数: {stats['row_count']}")

# === 删除集合 ===
client.drop_collection("demo_collection")
```

### A.1.3 数据操作

```python
# === 插入数据 ===
# 单条插入
client.insert(
    collection_name="production_collection",
    data={
        "embedding": [0.12, 0.34, ...],  # 768 维向量
        "text": "Milvus 是一个高性能向量数据库",
        "category": "tech",
        "score": 0.95,
        "timestamp": 1700000000
    }
)

# 批量插入
data = [
    {"embedding": vec1, "text": "文本1", "category": "A", "score": 0.9, "timestamp": ts},
    {"embedding": vec2, "text": "文本2", "category": "B", "score": 0.8, "timestamp": ts},
    # ... 更多数据
]
client.insert(collection_name="production_collection", data=data)

# === 查询数据 ===
# 按主键查询
result = client.get(
    collection_name="production_collection",
    ids=[1, 2, 3]
)

# 按条件查询
result = client.query(
    collection_name="production_collection",
    filter='category == "tech"',
    output_fields=["text", "score"],
    limit=10
)

# === 更新数据 ===
client.upsert(
    collection_name="production_collection",
    data={
        "id": 1,
        "text": "更新后的文本",
        "score": 0.99
    }
)

# === 删除数据 ===
# 按主键删除
client.delete(collection_name="production_collection", ids=[1, 2])

# 按条件删除
client.delete(
    collection_name="production_collection",
    filter='score < 0.5'
)
```

### A.1.4 向量检索

```python
# === 基础向量检索 ===
results = client.search(
    collection_name="production_collection",
    data=[query_vector],     # 查询向量，支持批量查询
    limit=10,                # 返回 Top-K 结果
    search_params={
        "metric_type": "COSINE",
        "params": {"nprobe": 64}  # IVF 索引搜索范围
    }
)

# === 带过滤条件的检索 ===
results = client.search(
    collection_name="production_collection",
    data=[query_vector],
    limit=10,
    filter='category in ["tech", "science"]',
    search_params={
        "metric_type": "COSINE",
        "params": {"nprobe": 64}
    }
)

# === 带输出字段的检索 ===
results = client.search(
    collection_name="production_collection",
    data=[query_vector],
    limit=10,
    output_fields=["text", "category", "score"],
    search_params={
        "metric_type": "COSINE",
        "params": {"nprobe": 64}
    }
)

# === 解析检索结果 ===
for hits in results:
    for hit in hits:
        print(f"ID: {hit['id']}")
        print(f"距离: {hit['distance']:.4f}")
        print(f"文本: {hit['entity']['text']}")
        print(f"类别: {hit['entity']['category']}")
        print("---")
```

### A.1.5 索引管理

```python
# === 创建索引 ===
# IVF_FLAT 索引（精度优先）
client.create_index(
    collection_name="production_collection",
    index_params={
        "metric_type": "COSINE",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 1024}
    }
)

# IVF_SQ8 索引（内存优化）
client.create_index(
    collection_name="production_collection",
    index_params={
        "metric_type": "COSINE",
        "index_type": "IVF_SQ8",
        "params": {"nlist": 1024}
    }
)

# HNSW 索引（速度优先）
client.create_index(
    collection_name="production_collection",
    index_params={
        "metric_type": "COSINE",
        "index_type": "HNSW",
        "params": {
            "M": 16,
            "efConstruction": 200
        }
    }
)

# IVF_PQ 索引（极致压缩）
client.create_index(
    collection_name="production_collection",
    index_params={
        "metric_type": "COSINE",
        "index_type": "IVF_PQ",
        "params": {
            "nlist": 1024,
            "m": 16,          # 子空间数量
            "nbits": 8         # 每个子向量的编码位数
        }
    }
)

# === 查看索引 ===
index_info = client.list_indexes("production_collection")

# === 删除索引 ===
client.drop_index(collection_name="production_collection", index_name="_default")
```

### A.1.6 分区管理

```python
# === 创建分区 ===
client.create_partition(
    collection_name="production_collection",
    partition_name="tech_docs"
)

# === 列出分区 ===
partitions = client.list_partitions("production_collection")

# === 在指定分区插入 ===
client.insert(
    collection_name="production_collection",
    partition_name="tech_docs",
    data=[...]
)

# === 在指定分区检索 ===
results = client.search(
    collection_name="production_collection",
    data=[query_vector],
    limit=10,
    partition_names=["tech_docs"]
)

# === 删除分区 ===
client.drop_partition(
    collection_name="production_collection",
    partition_name="tech_docs"
)
```

## A.2 索引参数速查表

### A.2.1 索引类型对比

| 索引类型 | 构建时间 | 搜索速度 | 内存占用 | 召回率 | 推荐数据规模 |
|---------|---------|---------|---------|-------|------------|
| IVF_FLAT | 中等 | 中等 | 高（1x） | 最高 | 百万级 |
| IVF_SQ8 | 中等 | 中等 | 低（~0.3x） | 高 | 千万级 |
| IVF_PQ | 较慢 | 中等 | 极低（~0.05x） | 中等 | 亿级 |
| HNSW | 较慢 | 极快 | 较高（~1.5x） | 最高 | 千万级 |
| DISKANN | 慢 | 较快 | 极低（硬盘） | 高 | 十亿级 |
| FLAT | 无 | 慢 | 高（1x） | 最高 | 万级以下 |

### A.2.2 索引参数详解

**IVF 系列（IVF_FLAT / IVF_SQ8 / IVF_PQ）**

| 参数 | 说明 | 推荐值 | 影响 |
|------|------|-------|------|
| nlist | 聚类中心数量 | 128-4096 | nlist 越大，构建越慢，搜索精度越高 |
| nprobe | 搜索时访问的聚类数 | 8-256 | nprobe 越大，搜索越慢，召回率越高 |

```python
# IVF 参数选择经验
# nlist 选择：根据数据量
nlist_estimate = int(4 * sqrt(total_rows))
# 示例：100 万条数据 → nlist ≈ 4000

# nprobe 选择：根据精度要求
# 高精度 (>95%): nprobe = 64-128
# 平衡 (90-95%): nprobe = 16-32
# 高吞吐 (>90%): nprobe = 4-8
```

**HNSW 索引**

| 参数 | 说明 | 推荐值 | 影响 |
|------|------|-------|------|
| M | 每个节点的最大连接数 | 8-48 | M 越大，精度越高，内存越大 |
| efConstruction | 构建时的动态搜索范围 | 100-500 | 越大构建越慢，精度越高 |
| ef | 搜索时的动态搜索范围 | 50-500 | ef 越大，搜索越慢，召回率越高 |

```python
# HNSW 参数选择经验
# M 的选择：
# M = 16  → 平衡方案（推荐）
# M = 32  → 高精度方案
# M = 8   → 低内存方案

# ef 的选择：
# ef = k * 2  → 最小推荐值（k 为 Top-K）
# ef = k * 5  → 高召回方案
```

**IVF_PQ 额外参数**

| 参数 | 说明 | 推荐值 | 影响 |
|------|------|-------|------|
| m | 子空间数量（必须能整除向量维度） | dim/2 ~ dim/8 | m 越大，压缩比越低，精度越高 |
| nbits | 每个子向量的编码位数 | 8（默认） | 通常保持默认 8 |

### A.2.3 场景化索引推荐

| 场景 | 推荐索引 | 理由 |
|------|---------|------|
| 知识库 RAG（<100 万文档） | HNSW | 毫秒级响应，精度高 |
| 知识库 RAG（>100 万文档） | IVF_SQ8 | 平衡内存和精度 |
| 日志/监控分析 | IVF_FLAT | 精度优先 |
| 亿级商品推荐 | IVF_PQ | 极致压缩，突破内存 |
| 十亿级图像检索 | DISKANN | 硬盘存储，无限扩展 |

## A.3 环境变量速查

### A.3.1 Milvus 服务端环境变量

```bash
# === 基本配置 ===
# Milvus 部署模式
export DEPLOY_MODE=standalone        # 单机模式
export DEPLOY_MODE=cluster           # 集群模式

# === 存储配置 ===
# 元数据存储（推荐 etcd）
export ETCD_ENDPOINTS=localhost:2379
export ETCD_ROOT_PATH=by-dev

# 消息存储（推荐 Pulsar 或 Kafka）
export PULSAR_ADDRESS=localhost:6650
# 或
export KAFKA_BROKER_LIST=localhost:9092

# 对象存储（推荐 MinIO 或 S3）
export MINIO_ADDRESS=localhost:9000
export MINIO_ACCESS_KEY=minioadmin
export MINIO_SECRET_KEY=minioadmin
export MINIO_USE_SSL=false

# === 资源限制 ===
# 查询节点资源
export QUERY_NODE_CPU_LIMIT=8
export QUERY_NODE_MEM_LIMIT=16GB

# 索引节点资源
export INDEX_NODE_CPU_LIMIT=16
export INDEX_NODE_MEM_LIMIT=32GB

# === 性能调优 ===
# 查询超时（毫秒）
export QUERY_TIMEOUT=30000

# 索引构建内存限制
export INDEX_BUILD_MEM_LIMIT=16GB

# 数据同步间隔（毫秒）
export DATA_SYNC_INTERVAL=100
```

### A.3.2 客户端连接配置

```python
import os
from pymilvus import MilvusClient

# 从环境变量读取配置
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
MILVUS_USER = os.getenv("MILVUS_USER", "")
MILVUS_PASSWORD = os.getenv("MILVUS_PASSWORD", "")
MILVUS_URI = os.getenv("MILVUS_URI", f"http://{MILVUS_HOST}:{MILVUS_PORT}")

# 超时和连接池配置
MILVUS_CONNECT_TIMEOUT = int(os.getenv("MILVUS_CONNECT_TIMEOUT", "10"))
MILVUS_POOL_SIZE = int(os.getenv("MILVUS_POOL_SIZE", "10"))
MILVUS_RETRY_TIMES = int(os.getenv("MILVUS_RETRY_TIMES", "3"))

client = MilvusClient(
    uri=MILVUS_URI,
    user=MILVUS_USER,
    password=MILVUS_PASSWORD,
    timeout=MILVUS_CONNECT_TIMEOUT
)
```

### A.3.3 Docker Compose 环境变量模板

```yaml
# docker-compose.yml 片段
version: '3.5'
services:
  milvus:
    image: milvusdb/milvus:latest
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
      QUERY_NODE_CPU_LIMIT: '4'
      QUERY_NODE_MEM_LIMIT: 8GB
    ports:
      - "19530:19530"
      - "9091:9091"
    volumes:
      - ./milvus/data:/var/lib/milvus
```

## A.4 Docker Compose 常用命令

### A.4.1 启动与停止

```bash
# 启动 Milvus 服务（后台运行）
docker compose -f docker-compose.yml up -d

# 启动指定服务
docker compose -f docker-compose.yml up -d milvus

# 查看启动日志
docker compose -f docker-compose.yml logs -f milvus

# 停止服务
docker compose -f docker-compose.yml stop

# 停止并删除容器
docker compose -f docker-compose.yml down

# 停止并删除容器和卷（清除数据）
docker compose -f docker-compose.yml down -v
```

### A.4.2 状态检查

```bash
# 查看所有容器状态
docker compose -f docker-compose.yml ps

# 查看资源使用情况
docker stats

# 检查 Milvus 健康状态
curl http://localhost:9091/health

# 查看 Milvus 组件状态
docker compose -f docker-compose.yml exec milvus milvusctl status
```

### A.4.3 数据管理

```bash
# 备份数据（备份 Milvus 数据目录）
tar -czf milvus_backup.tar.gz ./milvus_data/

# 恢复数据
tar -xzf milvus_backup.tar.gz -C ./milvus_data/

# 导出集合数据（通过 Python SDK）
python -c "
from pymilvus import MilvusClient
client = MilvusClient('http://localhost:19530')
results = client.query('my_collection', output_fields=['*'], limit=10000)
import json
with open('export.json', 'w') as f:
    json.dump(results, f)
"
```

### A.4.4 日志和调试

```bash
# 实时查看 Milvus 日志
docker compose -f docker-compose.yml logs -f

# 查看最近 100 行日志
docker compose -f docker-compose.yml logs --tail=100

# 查看指定服务的日志
docker compose -f docker-compose.yml logs -f milvus

# 进入容器内部调试
docker compose -f docker-compose.yml exec milvus /bin/bash

# 查看容器内的 Milvus 配置
docker compose -f docker-compose.yml exec milvus cat /milvus/configs/milvus.yaml
```

### A.4.5 升级和迁移

```bash
# 拉取最新镜像
docker compose pull milvus

# 滚动升级（先停旧版本，再启新版本）
docker compose -f docker-compose.yml up -d --no-deps --build milvus

# 导出配置
docker compose -f docker-compose.yml config > milvus_config_backup.yml

# 完整迁移（在新环境启动后导入数据）
# Step 1: 旧环境导出数据
# Step 2: 新环境启动
docker compose -f docker-compose-new.yml up -d
# Step 3: 导入数据到新环境
```

### A.4.6 性能监控

```bash
# 查看 Milvus metrics（Prometheus 格式）
curl http://localhost:9091/metrics

# 监控 CPU 和内存
docker compose -f docker-compose.yml top milvus

# 使用 Milvus Insight（GUI 管理工具）
docker run -d -p 8080:8080 \
  -e MILVUS_URL=http://localhost:19530 \
  milvusdb/milvus-insight:latest
```

### A.4.7 常用 Compose 配置模板

```yaml
# docker-compose.yml（Milvus 单机版完整模板）
version: '3.5'

services:
  etcd:
    container_name: milvus-etcd
    image: quay.io/coreos/etcd:v3.5.5
    environment:
      ETCD_AUTO_COMPACTION_MODE: revision
      ETCD_AUTO_COMPACTION_RETENTION: '1000'
      ETCD_QUOTA_BACKEND_BYTES: '4294967296'
      ETCD_SNAPSHOT_COUNT: '50000'
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/etcd:/etcd
    command: etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd

  minio:
    container_name: milvus-minio
    image: minio/minio:RELEASE.2023-03-20T20-16-18Z
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/minio:/minio_data
    command: minio server /minio_data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3

  milvus:
    container_name: milvus-standalone
    image: milvusdb/milvus:latest
    command: ["milvus", "run", "standalone"]
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
    ports:
      - "19530:19530"
      - "9091:9091"
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/milvus:/var/lib/milvus
    depends_on:
      - "etcd"
      - "minio"

networks:
  default:
    name: milvus
```

## 本章小结

本附录提供了 Milvus 日常开发与运维的常用命令速查手册，涵盖了 MilvusClient SDK 的完整 API 参考、索引参数配置指南、环境变量说明以及 Docker Compose 操作命令。无论是 RAG 系统的开发调试，还是 Milvus 服务的部署运维，本附录都可以作为随查随用的参考工具。

---

**附录说明**：本附录将持续更新，以反映 Milvus 最新版本的 API 变化和最佳实践。建议读者关注 Milvus 官方文档以获取最新信息。
