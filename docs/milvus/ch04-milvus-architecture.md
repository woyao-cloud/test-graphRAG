# 第4章：Milvus整体架构与核心组件

## 4.1 Milvus设计哲学

Milvus是一款云原生的分布式向量数据库，专为海量向量数据的存储与检索而设计。在RAG系统中，Milvus承担着"语义记忆"的核心角色——它将文档的嵌入向量持久化存储，并提供毫秒级的近似最近邻（ANN）检索能力。理解Milvus的架构设计，是构建高性能RAG系统的前提。

Milvus的核心设计理念可以概括为四个关键词：

- **计算与存储分离**：计算节点和存储节点独立扩缩容，适应不同负载模式
- **日志即数据**：采用基于日志的架构（Log as Data），所有数据变更先写入日志，再异步持久化
- **微服务化**：核心功能拆分为独立的微服务组件，每个组件可独立部署和扩展
- **云原生**：天然适配Kubernetes，支持弹性伸缩、故障恢复和滚动升级

## 4.2 分层架构详解

Milvus采用四层架构，从下到上依次为存储层、执行层、协调层和接入层。每一层都有明确的职责边界，层与层之间通过gRPC通信。

```
┌──────────────────────────────────────────────────┐
│                  接入层（Access Layer）             │
│            Proxy（无状态网关，请求路由）              │
├──────────────────────────────────────────────────┤
│                  协调层（Coordinator Layer）        │
│   RootCoord  │  QueryCoord  │  DataCoord          │
│   元数据管理   │  查询调度     │  数据管理            │
├──────────────────────────────────────────────────┤
│                  执行层（Worker Layer）             │
│   QueryNode  │  DataNode  │  IndexNode            │
│   查询执行     │  数据写入    │  索引构建             │
├──────────────────────────────────────────────────┤
│                  存储层（Storage Layer）            │
│    Etcd（元数据） │  MinIO/S3（数据） │ Pulsar（日志）  │
└──────────────────────────────────────────────────┘
```

### 4.2.1 接入层（Access Layer）

接入层由一组无状态的**Proxy**节点组成，是整个系统的流量入口。Proxy的主要职责包括：

1. **请求路由**：接收客户端的SDK请求，解析请求类型（DDL/DML/DQL），将请求转发到对应的协调服务或执行节点
2. **连接管理**：维护与客户端的长连接，支持连接池复用
3. **身份认证**：验证客户端身份，确保只有授权用户可以访问
4. **限流与监控**：对请求进行速率限制，采集请求级别的监控指标

Proxy节点是无状态的，可以水平扩展以应对高并发场景。在RAG生产环境中，建议至少部署2个Proxy节点以实现高可用。

```python
# 连接Milvus的Python示例
from pymilvus import connections

# 连接到Proxy（默认端口19530）
connections.connect(
    alias="default",
    host="localhost",  # Proxy地址
    port="19530",       # gRPC端口
    secure=False
)
```

### 4.2.2 协调层（Coordinator Layer）

协调层是Milvus的"大脑"，负责全局元数据管理、任务调度和集群状态维护。协调层包含三个核心协调器：

**RootCoord（根协调器）**

RootCoord是整个集群的"总管家"，管理所有全局元数据。它的核心职责包括：

- **集合管理**：维护所有集合（Collection）的Schema信息，包括字段定义、分片配置、别名映射等
- **全局时间戳分配**：为每个事务分配全局唯一的时间戳（TSO），保证多节点间的事务一致性
- **数据分布管理**：管理分片（Shard）与DataNode之间的映射关系
- **DDL操作处理**：处理创建/删除集合、创建/释放索引等DDL操作

RootCoord是集群中唯一持有全局锁的组件，在高并发场景下需要关注其性能。RootCoord通常部署为单节点或主备模式。

**QueryCoord（查询协调器）**

QueryCoord负责管理查询节点的生命周期和查询任务的调度：

- **查询节点管理**：监控QueryNode的健康状态，处理节点故障和动态扩缩容
- **查询计划分发**：将查询请求分解为子查询任务，分发到对应的QueryNode
- **负载均衡**：监控各QueryNode的负载情况，动态调整segment分布
- **Segment调度**：决定哪些segment需要加载到QueryNode，哪些需要释放

**DataCoord（数据协调器）**

DataCoord负责管理数据节点的生命周期和数据写入流程：

- **数据节点管理**：监控DataNode的健康状态
- **Segment分配**：为写入数据分配目标segment
- **日志检查点**：管理binlog的持久化和清理
- **索引构建调度**：触发IndexNode进行索引构建
- **数据压缩**：触发小文件合并操作（Compaction）

```python
# 创建集合的SDK操作（最终由RootCoord处理）
from pymilvus import CollectionSchema, FieldSchema, DataType, Collection

fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=256),
]
schema = CollectionSchema(fields, description="RAG知识库集合")
collection = Collection(name="rag_knowledge", schema=schema)
```

### 4.2.3 执行层（Worker Layer）

执行层是实际处理数据和执行查询的节点，包括三种工作节点：

**QueryNode（查询节点）**

QueryNode是执行向量检索和标量过滤的核心节点。每个QueryNode维护一部分segment的内存副本，并负责处理这些segment上的查询请求。

QueryNode的工作流程：

1. 接收来自Proxy的查询请求
2. 在本地加载的segment上执行向量检索（ANN搜索）
3. 应用标量过滤条件（如时间范围、类别过滤）
4. 返回Top-K结果给Proxy
5. Proxy对所有QueryNode的结果进行全局合并和排序

QueryNode的segment加载策略直接决定查询性能。热点数据应尽量常驻内存，冷数据可以按需加载。

```python
# 加载集合到QueryNode
collection.load()
```

**DataNode（数据节点）**

DataNode负责将写入数据从日志（Pulsar）持久化到对象存储（MinIO/S3）：

1. 消费Pulsar中的增量数据变更日志
2. 将数据批量写入MinIO/S3，生成binlog文件
3. 定期向DataCoord报告写入进度（checkpoint）

DataNode是典型的内存-IO平衡型节点，需要配置合理的CPU和内存资源。

**IndexNode（索引节点）**

IndexNode专门负责索引的构建任务：

- 构建向量索引（IVF、HNSW、DiskANN等）
- 构建标量索引（倒排索引、布隆过滤器等）
- 索引构建完成后，将索引文件写入对象存储

IndexNode是CPU密集型节点，在数据批量入库阶段建议配置较高的CPU资源。

### 4.2.4 存储层（Storage Layer）

Milvus采用三层存储架构，每层使用不同的存储引擎：

**Etcd——元数据存储**

Etcd是一个高可用的分布式键值存储系统，在Milvus中存储以下元数据：

- 集合Schema定义
- 分片与节点的映射关系
- 索引元数据（索引类型、参数、状态）
- 全局时间戳
- 集群节点信息

Etcd是集群的"配置中心"，其稳定性直接影响整个集群的可用性。建议Etcd集群部署3或5个节点。

```yaml
# Etcd关键配置
ETCD_AUTO_COMPACTION_MODE: revision     # 按版本号自动压缩
ETCD_AUTO_COMPACTION_RETENTION: 1000    # 保留最近1000个版本
ETCD_QUOTA_BACKEND_BYTES: 4294967296   # 后端存储限额（4GB）
```

**MinIO/S3——数据存储**

MinIO（或AWS S3、GCS等兼容的对象存储）用于持久化存储所有数据文件和索引文件：

- **binlog文件**：数据变更的日志文件，包括插入、删除、更新操作
- **索引文件**：构建完成的向量索引和标量索引文件
- **Segment文件**：数据段的物理存储文件

MinIO采用对象存储的方式，天然支持海量数据存储和地理冗余。在Docker Compose部署中，MinIO通常与Milvus部署在同一台机器上。

```yaml
# MinIO配置（docker-compose）
minio:
  image: minio/minio:RELEASE.2023-03-20T20-16-18Z
  environment:
    MINIO_ROOT_USER: minioadmin
    MINIO_ROOT_PASSWORD: minioadmin
  command: minio server /minio_data
  volumes:
    - ./volumes/minio:/minio_data
```

**Pulsar——日志存储**

Pulsar是分布式消息队列系统，在Milvus中扮演"变更数据捕获"（CDC）的角色：

- 所有数据写入操作首先被记录到Pulsar
- DataNode消费Pulsar中的日志进行持久化
- 新加入的QueryNode通过重放Pulsar日志来恢复数据
- Pulsar保证了数据的最终一致性和可回溯性

Pulsar的核心优势在于其计算与存储分离的架构和高效的日志回放能力。在Milvus 2.3+版本中，Pulsar可以被替换为RocksDB（单机模式）或Kafka（集群模式），以降低运维复杂度。

```
数据写入流程（日志即数据）：
客户端写入 → Proxy → Pulsar（日志） → DataNode（消费） → MinIO（持久化）
                                                          ↓
                                                     IndexNode（构建索引） → MinIO（索引文件）
```

## 4.3 数据写入流程

理解数据写入流程，有助于在RAG场景中优化批量入库的性能。Milvus的数据写入分为同步写入（Flush）和异步写入两种模式。

**同步写入流程**：

```
1. 客户端调用 insert() 发送数据
2. Proxy 将数据写入 Pulsar 的对应分片
3. DataNode 消费 Pulsar 消息，将数据写入 MinIO（生成 binlog）
4. 客户端调用 flush() 触发强制持久化
5. DataCoord 确认所有 binlog 写入完成
6. 数据进入"已持久化"状态，可被查询
```

**异步写入流程**（默认模式）：

```
1. 客户端调用 insert()
2. Proxy 将数据写入 Pulsar
3. 客户端立即返回（数据尚未持久化）
4. DataNode 异步消费 Pulsar 并写入 MinIO
5. 数据最终在后台完成持久化
```

在RAG的批量文档入库场景中，建议采用同步写入模式并在每批数据插入后调用flush()，确保数据立即可查。

```python
# 批量数据写入示例
import random
import numpy as np

def batch_insert(collection, texts, embeddings, batch_size=1000):
    """批量插入文档到Milvus"""
    total = len(texts)
    for i in range(0, total, batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_embs = embeddings[i:i+batch_size]
        
        entities = [
            [emb.tolist() for emb in batch_embs],  # embedding字段
            batch_texts,                             # text字段
            ["rag_doc"] * len(batch_texts),           # source字段
        ]
        
        collection.insert(entities)
        
        # 每批写入后强制持久化
        if (i // batch_size) % 5 == 0:
            collection.flush()
        
        print(f"已插入 {min(i+batch_size, total)}/{total} 条")
    
    # 最终持久化
    collection.flush()
```

## 4.4 单机版 vs 集群版

Milvus提供单机版（Standalone）和集群版（Cluster）两种部署模式，在RAG场景中需要根据数据规模和并发要求进行选择。

### 4.4.1 单机版（Standalone）

单机版将所有组件合并到单个进程中运行，适合以下场景：

- 开发测试环境
- 中小规模RAG系统（向量数量 < 1000万）
- 查询并发低（QPS < 100）
- 运维能力有限的小团队

**部署方式**：通过Docker Compose一键部署，依赖Etcd、MinIO和Milvus standalone三个容器。

```yaml
# 单机版docker-compose核心配置
standalone:
  image: milvusdb/milvus:v2.4.0
  command: ["milvus", "run", "standalone"]
  ports:
    - "19530:19530"
  depends_on:
    - "etcd"
    - "minio"
```

**优势**：部署简单、资源消耗低、运维成本低。
**劣势**：不能水平扩展、单点故障风险。

### 4.4.2 集群版（Cluster）

集群版将所有组件微服务化，每个组件可以独立部署和扩缩容。适合以下场景：

- 大规模RAG系统（向量数量 > 1000万）
- 高并发查询（QPS > 100）
- 需要高可用和故障恢复
- 企业级生产环境

**部署方式**：推荐使用Milvus Operator在Kubernetes上部署，或使用Helm Chart。

**优势**：
- 各组件独立扩缩容（Proxy水平扩展应对高并发，QueryNode水平扩展应对大数据量）
- 滚动升级不影响服务
- 故障自愈
- 资源利用率高

**劣势**：部署复杂、资源开销大、运维门槛高。

### 4.4.3 选型建议

| 场景 | 推荐模式 | 说明 |
|------|---------|------|
| 本地开发/测试 | Standalone | 一键部署，快速验证 |
| 百万级知识库（<100万向量） | Standalone | 单机足以应对 |
| 千万级知识库 | Cluster（2-4节点） | 需要分布式存储和查询 |
| 高并发在线服务 | Cluster + 水平扩展 | Proxy和QueryNode独立扩缩容 |
| 企业级生产环境 | Cluster + K8s | 高可用、弹性伸缩 |

## 4.5 数据生命周期

Milvus中数据从写入到可查询，经历完整的生命周期管理：

**数据生命周期阶段**：

```
写入 → 未持久化（Growing）→ 已持久化（Sealed）→ 已索引（Indexed）→ 已合并（Compacted）→ 已清理（Dropped）
```

**Growing Segment**：数据刚写入Pulsar但尚未持久化到MinIO，存储在DataNode内存中。此阶段数据可查询，但性能较低。

**Sealed Segment**：数据已持久化到MinIO，但尚未构建索引。查询使用暴力搜索（Flat），数据量小时尚可接受。

**Indexed Segment**：IndexNode已完成索引构建，查询性能大幅提升。这是RAG查询的目标状态。

**Compacted Segment**：多个小segment合并为大segment，减少查询时的IO开销。Compaction由DataCoord自动触发。

### 4.5.1 数据过期清理

在RAG系统中，知识库需要定期更新和清理过期数据。Milvus支持两种删除方式：

**软删除**：调用delete接口标记数据为删除状态，查询时自动过滤。

```python
# 按条件删除
collection.delete("source == 'obsolete_doc'")
```

**物理删除**：通过Compaction机制，将标记为删除的数据从物理文件中清除。

### 4.5.2 数据生命周期与RAG增量更新

RAG系统的知识库需要持续增量更新。推荐以下策略：

1. **全量重建**（低频）：每隔一定周期（如每周）重建整个索引，确保最佳性能
2. **增量更新**（高频）：新文档直接插入，使用Milvus的自动索引刷新
3. **过期清理**（定期）：删除过期文档后，触发Compaction回收存储空间

```python
def incremental_update(collection, new_texts, new_embeddings):
    """增量更新知识库"""
    # 1. 插入新数据
    mr = collection.insert([new_embeddings, new_texts])
    collection.flush()
    
    # 2. 重建索引（增量数据达到阈值时触发）
    if collection.num_entities % 10000 == 0:
        collection.create_index(
            field_name="embedding",
            index_params={
                "metric_type": "COSINE",
                "index_type": "HNSW",
                "params": {"M": 16, "efConstruction": 200}
            }
        )
        collection.load()
    
    return mr.insert_count
```

## 4.6 本章小结

本章深入剖析了Milvus的四层架构设计及其核心组件。接入层的Proxy负责请求路由和负载均衡；协调层的RootCoord、QueryCoord和DataCoord分别管理元数据、查询调度和数据写入；执行层的QueryNode、DataNode和IndexNode各自承担查询执行、数据持久化和索引构建的职责；存储层的Etcd、MinIO和Pulsar提供了元数据、数据和日志的三层存储能力。

Milvus的"日志即数据"架构设计使得数据写入先入日志再异步持久化，兼顾了写入性能和数据的最终一致性。这种设计对于RAG场景特别重要——它允许知识库在持续写入新文档的同时，查询服务不受影响。

在选择部署模式时，中小规模RAG系统推荐使用单机版，大规模和高并发场景推荐使用集群版。理解数据生命周期各阶段（Growing→Sealed→Indexed→Compacted）有助于在RAG系统中合理规划数据入库、索引构建和过期清理的策略。
