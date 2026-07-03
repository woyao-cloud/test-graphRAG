# 第5章 向量存储：RAG系统的记忆核心

## 5.1 引言

在RAG系统的技术栈中，向量存储（Vector Storage）扮演着"长期记忆"的角色。如果说嵌入模型负责将人类语言转化为机器可理解的数字表示，那么向量数据库则负责高效地存储和检索这些数字表示。本章将深入探讨向量存储的核心概念、主流方案选择、索引构建原理以及性能调优策略。

向量存储面临的核心挑战可以概括为一个简单的矛盾：**数据量越大，精确检索越慢**。在包含数百万甚至数十亿条文本片段的系统中，逐一计算查询向量与所有存储向量的相似度（即暴力搜索，brute-force search）在计算上是不可行的。这就需要借助近似最近邻搜索（Approximate Nearest Neighbor, ANN）技术，以微小的精度损失换取数个数量级的性能提升。

## 5.2 嵌入模型选型

在构建向量存储之前，首先需要选择适合业务的嵌入模型。嵌入模型的质量直接决定了向量表示的语义丰富度，进而影响检索效果的上限。

### 5.2.1 BGE-M3

BGE-M3（BAAI General Embedding-M3）是由北京智源人工智能研究院（BAAI）开发的多功能嵌入模型。"M3"代表三个核心特性：

- **Multi-Linguality（多语言）**：支持超过100种语言，尤其在中英文混合场景下表现优异
- **Multi-Granularity（多粒度）**：同时支持词级（token-level）和句级（sentence-level）表示
- **Multi-Functionality（多功能）**：支持稠密检索（dense retrieval）、稀疏检索（sparse retrieval）和多向量检索（multi-vector retrieval）

BGE-M3的核心优势在于其**混合检索能力**——同时生成稠密向量和稀疏向量（类似BM25的风格），在检索阶段可以结合两种表示的优势。

```python
# BGE-M3 使用示例
from sentence_transformers import SentenceTransformer

# 加载 BGE-M3 模型
model = SentenceTransformer("BAAI/bge-m3")

# 生成稠密向量（默认输出维度：1024）
documents = [
    "RAG系统通过检索增强大语言模型的生成能力。",
    "向量数据库使用近似最近邻搜索来加速检索。",
    "嵌入模型将文本映射到高维语义空间。"
]
doc_embeddings = model.encode(documents, normalize_embeddings=True)
print(f"文档向量形状: {doc_embeddings.shape}")  # (3, 1024)

# 对查询进行编码
query = "什么是RAG系统？"
query_embedding = model.encode(query, normalize_embeddings=True)
print(f"查询向量形状: {query_embedding.shape}")  # (1024,)
```

### 5.2.2 text-embedding-3-large

OpenAI的text-embedding-3-large是目前商业嵌入模型中的标杆产品。其主要特点包括：

- **输出维度**：最高3072维，但支持通过`dimensions`参数动态降维
- **多语言能力**：在主要语言上表现优秀，但小语种覆盖不如BGE-M3
- **API服务**：通过OpenAI API调用，无需本地部署GPU

一个值得注意的特性是**Matryoshka Representation Learning（MRL）**技术。OpenAI允许用户指定输出向量的维度（从256到3072），且无需重新训练模型。低维度向量占用更少的存储空间，检索速度更快，且精度损失可控。

```python
# text-embedding-3-large 使用示例
from openai import OpenAI
import numpy as np

client = OpenAI(api_key="your-api-key")

def get_embeddings(texts, dimensions=1024):
    """生成嵌入向量，支持动态维度调整"""
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=texts,
        dimensions=dimensions  # MRL 技术：降维而不损失过多语义
    )
    embeddings = [item.embedding for item in response.data]
    return np.array(embeddings, dtype=np.float32)

documents = [
    "Retrieval-Augmented Generation combines retrieval with generation.",
    "Vector databases enable efficient similarity search at scale.",
    "Embedding models convert text into dense vector representations."
]
embeddings = get_embeddings(documents, dimensions=1024)
print(f"向量维度: {embeddings.shape[1]}")  # 1024
print(f"向量数量: {embeddings.shape[0]}")   # 3
```

### 5.2.3 Nomic Embed Text

Nomic Embed Text是由Nomic AI开发的开源嵌入模型，以其出色的性价比和完全开源的特性受到关注。

- **模型架构**：基于BERT的改进版本，支持8192 tokens的上下文长度
- **输出维度**：768维，支持Matryoshka表示学习
- **开源许可**：Apache 2.0许可，允许商业使用和二次开发
- **本地部署**：可在消费级GPU上运行

```python
# Nomic Embed Text 使用示例
from sentence_transformers import SentenceTransformer

# 加载模型
model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

# 编码文档
documents = [
    "Nomic Embed Text supports up to 8192 tokens of context.",
    "This model is fully open-source under Apache 2.0 license.",
    "It achieves competitive performance on MTEB benchmarks."
]
embeddings = model.encode(documents, normalize_embeddings=True)
print(f"向量维度: {embeddings.shape[1]}")  # 768
```

### 5.2.4 模型选型对比

下表从多个维度对比了上述三种嵌入模型：

| 特性 | BGE-M3 | text-embedding-3-large | Nomic Embed Text |
|------|--------|----------------------|------------------|
| 默认向量维度 | 1024 | 3072 | 768 |
| 最大上下文长度 | 8192 | 8191 | 8192 |
| 开源许可 | MIT | 商业API | Apache 2.0 |
| 本地部署 | 支持 | 不支持 | 支持 |
| 多语言能力 | 极强（100+语言） | 强（主要语言） | 中（以英文为主） |
| 混合检索 | 支持 | 不支持 | 不支持 |
| MTEB平均分 | 69.2 | 64.6 | 62.8 |

**选型建议**：

- 如果需要处理**中英文混合**场景，优先选择BGE-M3
- 如果追求**最佳质量**且预算充足，选择text-embedding-3-large
- 如果需要**完全开源、本地部署**且以英文为主，选择Nomic Embed Text

## 5.3 向量数据库架构选型

向量数据库是向量存储的物理载体。不同的架构设计在性能、扩展性和易用性上各有取舍。本节分析三种主流向量数据库的架构设计。

### 5.3.1 LanceDB：磁盘优先的嵌入式数据库

LanceDB是一种**磁盘优先（disk-based）**的嵌入式向量数据库，基于Lance列式存储格式构建。

**架构特点**：

1. **零服务架构**：LanceDB作为嵌入式库运行，无需独立的数据库服务进程。应用程序直接链接LanceDB库，通过API读写本地文件。这消除了网络通信开销和运维复杂度。

2. **列式存储**：底层使用Lance格式，这是一种列式存储格式，专为AI工作负载优化。列式存储意味着读取向量数据时只需加载需要的列，而非整行数据。

3. **延迟写入（lazy write）**：数据写入时先缓存在内存中，达到阈值后才批量刷写到磁盘。这种设计优化了写入吞吐量，但写入后立即查询可能无法立即看到最新数据。

```python
# LanceDB 使用示例
import lancedb
import numpy as np

# 连接到本地数据库（如果不存在则自动创建）
db = lancedb.connect("./data/lancedb")

# 创建表
data = [
    {"vector": np.random.rand(1024).astype(np.float32), "text": "RAG系统架构设计", "id": 1},
    {"vector": np.random.rand(1024).astype(np.float32), "text": "向量索引构建方法", "id": 2},
    {"vector": np.random.rand(1024).astype(np.float32), "text": "嵌入模型选型指南", "id": 3},
]

table = db.create_table("documents", data, mode="overwrite")

# 创建 IVF-PQ 索引以加速检索
table.create_index(
    metric="cosine",      # 相似度度量方式
    num_partitions=256,   # IVF 分区数
    num_sub_vectors=96,   # PQ 子向量数量
)

# 执行相似度搜索
query_vector = np.random.rand(1024).astype(np.float32)
results = table.search(query_vector).limit(5).to_pandas()
print(results[["text", "_distance"]])
```

**适用场景**：单机应用、原型开发、对运维复杂度敏感的小型团队。

### 5.3.2 FAISS：内存优先的高性能库

FAISS（Facebook AI Similarity Search）是由Meta AI开发的高效相似度搜索库。它在内存中维护向量索引，以极致的检索性能著称。

**架构特点**：

1. **纯内存索引**：FAISS将索引完全加载到内存中，因此检索速度极快，但受限于单机内存容量。对于大规模场景，需要配合分片（sharding）策略。

2. **GPU加速**：FAISS原生支持CUDA GPU加速。在GPU上，大规模矩阵运算的并行性能使检索吞吐量提升5-10倍。

3. **丰富的索引类型**：FAISS提供了近20种索引变体，支持从精确检索到各种近似检索的灵活组合。

4. **无持久化机制**：FAISS本身不提供数据持久化功能，需要开发者自行管理索引的序列化和加载。

```python
# FAISS 使用示例
import faiss
import numpy as np

# 生成示例数据
dimension = 1024
num_vectors = 10000
data_vectors = np.random.rand(num_vectors, dimension).astype(np.float32)

# 构建 IVF + HNSW 混合索引
nlist = 100  # IVF 聚类中心数量
m = 32       # HNSW 每个节点的连接数

# 使用 IndexIVFFlat（IVF + 精确搜索）
quantizer = faiss.IndexHNSWFlat(dimension, m)
index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)

# 训练索引（IVF 需要聚类训练）
index.train(data_vectors)
index.add(data_vectors)
index.nprobe = 10  # 检索时探访的聚类数

# 执行搜索
query = np.random.rand(1, dimension).astype(np.float32)
distances, indices = index.search(query, k=5)
print(f"最近邻索引: {indices}")
print(f"相似度距离: {distances}")

# 序列化索引到磁盘
faiss.write_index(index, "./data/faiss_index.bin")

# 加载索引
loaded_index = faiss.read_index("./data/faiss_index.bin")
```

**适用场景**：对检索延迟有极致要求的在线服务、GPU资源充足的场景、需要精细控制索引参数的场景。

### 5.3.3 Milvus：分布式云原生数据库

Milvus是一款专为向量相似度搜索设计的分布式数据库，采用**计算与存储分离**的云原生架构。

**架构特点**：

1. **四层架构**：
   - **接入层（Access Layer）**：由一组无状态的Proxy节点组成，负责请求路由、认证和限流
   - **协调服务（Coordinator Service）**：管理数据分布、索引构建和查询调度
   - **工作节点（Worker Nodes）**：包括存储数据的DataNode和执行查询的QueryNode
   - **存储层（Storage Layer）**：使用对象存储（如MinIO、S3）持久化数据和日志

2. **分片与分区**：Milvus支持两级数据隔离——分片（shard）按主键哈希分布数据，分区（partition）按业务维度物理隔离数据。检索时可以指定分区，大幅缩小搜索范围。

3. **混合查询**：Milvus支持向量相似度搜索与标量字段过滤的组合查询（hybrid query）。例如，可以先按时间范围过滤文档，再在剩余文档中执行向量检索。

```python
# Milvus 使用示例
from pymilvus import (
    connections, Collection, CollectionSchema,
    FieldSchema, DataType, utility
)

# 连接到 Milvus 服务
connections.connect(host="localhost", port="19530")

# 定义集合 schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=64),
]

schema = CollectionSchema(fields, description="RAG文档集合")
collection = Collection(name="documents", schema=schema)

# 创建索引
index_params = {
    "metric_type": "COSINE",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 1024}
}
collection.create_index(field_name="embedding", index_params=index_params)

# 插入数据
import random
entities = [
    [np.random.rand(1024).tolist() for _ in range(1000)],
    ["文档内容示例" for _ in range(1000)],
    ["技术" for _ in range(1000)],
]
collection.insert(entities)
collection.flush()

# 加载集合到内存进行搜索
collection.load()

# 执行混合查询：按类别过滤 + 向量搜索
search_params = {
    "metric_type": "COSINE",
    "params": {"nprobe": 10}
}
query_vector = np.random.rand(1024).tolist()
results = collection.search(
    data=[query_vector],
    anns_field="embedding",
    param=search_params,
    limit=10,
    expr='category == "技术"',  # 标量过滤
    output_fields=["text"]
)
for hit in results[0]:
    print(f"ID: {hit.id}, Distance: {hit.distance}, Text: {hit.entity.get('text')}")
```

### 5.3.4 架构对比

| 特性 | LanceDB | FAISS | Milvus |
|------|---------|-------|--------|
| 部署模式 | 嵌入式库 | 嵌入式库 | 分布式服务 |
| 数据持久化 | 自动（Lance格式） | 需手动序列化 | 自动（对象存储） |
| 扩展方式 | 垂直扩展 | 垂直扩展+分片 | 水平扩展 |
| 检索延迟 | 毫秒级 | 微秒级 | 毫秒级 |
| 最大规模 | 千万级 | 亿级（分片） | 百亿级 |
| 运维复杂度 | 极低 | 低 | 高 |
| 混合查询 | 支持 | 不支持 | 支持 |

**选型建议**：

- **单机原型或桌面应用**：LanceDB 是最简单直接的选择
- **高吞吐在线服务**：FAISS 配合自定义服务封装
- **企业级大规模系统**：Milvus 提供最完整的功能集

## 5.4 索引类型深入解析

向量索引是向量数据库性能的核心。不同的索引类型在搜索精度、构建速度和内存占用之间做出不同的权衡。本节从原理到实践深入分析三种主流索引结构。

### 5.4.1 Flat（暴力搜索）

Flat索引不做任何近似优化，直接计算查询向量与所有存储向量的相似度。它是唯一一种保证**100%召回率**的搜索方式。

**数学原理**：

对于查询向量 `q` 和数据库向量集合 `D = {v₁, v₂, ..., vₙ}`，Flat搜索计算：

```
result = argmin_{i ∈ [1,n]} distance(q, vᵢ)
```

当使用余弦相似度时，距离计算为：

```
cosine_similarity(q, vᵢ) = q · vᵢ / (||q|| × ||vᵢ||)
```

对于1024维的向量，每次距离计算需要约1024次浮点乘法和1024次浮点加法。当数据量达到100万条时，一次搜索需要约20亿次浮点运算。

```python
# Flat 精确搜索实现
import numpy as np

def flat_search(query, database, k=10, metric="cosine"):
    """
    精确的暴力搜索

    参数:
        query: 查询向量，形状 (d,)
        database: 数据库向量，形状 (n, d)
        k: 返回的最近邻数量
        metric: 距离度量方式

    返回:
        distances: 距离数组，形状 (k,)
        indices: 对应索引数组，形状 (k,)
    """
    if metric == "cosine":
        # 余弦相似度：归一化后等价于内积
        query_norm = query / np.linalg.norm(query)
        db_norm = database / np.linalg.norm(database, axis=1, keepdims=True)
        scores = np.dot(db_norm, query_norm)
        # 取最大的k个（余弦相似度越大越相似）
        top_k_idx = np.argsort(scores)[-k:][::-1]
        return scores[top_k_idx], top_k_idx

    elif metric == "l2":
        # L2距离
        distances = np.linalg.norm(database - query, axis=1)
        top_k_idx = np.argsort(distances)[:k]
        return distances[top_k_idx], top_k_idx

    elif metric == "ip":
        # 内积（不归一化，保留原始分布信息）
        scores = np.dot(database, query)
        top_k_idx = np.argsort(scores)[-k:][::-1]
        return scores[top_k_idx], top_k_idx

# 测试
d = 1024
n = 100000
db = np.random.rand(n, d).astype(np.float32)
q = np.random.rand(d).astype(np.float32)

distances, indices = flat_search(q, db, k=5, metric="cosine")
print(f"Top-5 索引: {indices}")
print(f"相似度分数: {distances}")
```

**适用场景**：
- 数据量小于1万条的小型系统
- 需要精确结果的离线分析任务
- 作为其他近似索引的精度基准

### 5.4.2 IVF（倒排文件索引）

IVF（Inverted File Index）是向量检索中最经典的近似索引之一。其核心思想是**聚类+剪枝**——先将数据空间划分为多个聚类，搜索时只在与查询最近的几个聚类中查找。

**算法原理**：

1. **训练阶段**：使用K-means聚类算法将数据集划分为 `nlist` 个聚类，得到 `nlist` 个聚类中心
2. **索引阶段**：将每个向量分配到最近的聚类中心，构建倒排列表
3. **搜索阶段**：计算查询向量与所有聚类中心的距离，选择最近的 `nprobe` 个聚类，仅在这些聚类的向量中进行搜索

```
IVF 搜索过程:
1. 计算 q 与 nlist 个聚类中心的距离 → O(nlist × d)
2. 选择最近的 nprobe 个聚类
3. 在选中的聚类中执行暴力搜索 → O(nprobe × (n/nlist) × d)

总复杂度: O(nlist × d + nprobe × (n/nlist) × d)
当 nlist = √n, nprobe ≈ nlist 时，加速比约为 √n
```

```python
# IVF 索引实现
import numpy as np
from sklearn.cluster import MiniBatchKMeans

class IVFIndex:
    """简化的 IVF 索引实现"""

    def __init__(self, nlist=100, nprobe=10, metric="cosine"):
        self.nlist = nlist
        self.nprobe = nprobe
        self.metric = metric
        self.kmeans = None
        self.inverted_lists = {}  # 聚类ID -> 向量列表
        self.inverted_ids = {}    # 聚类ID -> 原始ID列表

    def train(self, vectors):
        """训练聚类器"""
        if self.metric == "cosine":
            vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

        self.kmeans = MiniBatchKMeans(
            n_clusters=self.nlist,
            random_state=42,
            batch_size=1024
        )
        self.kmeans.fit(vectors)
        self.centroids = self.kmeans.cluster_centers_

    def add(self, vectors, ids=None):
        """将向量添加到倒排索引"""
        if ids is None:
            ids = np.arange(len(vectors))

        # 分配每个向量到最近的聚类
        if self.metric == "cosine":
            vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

        labels = self.kmeans.predict(vectors)

        # 构建倒排列表
        for i, label in enumerate(labels):
            if label not in self.inverted_lists:
                self.inverted_lists[label] = []
                self.inverted_ids[label] = []
            self.inverted_lists[label].append(vectors[i])
            self.inverted_ids[label].append(ids[i])

        # 转换为numpy数组以提高搜索速度
        for label in self.inverted_lists:
            self.inverted_lists[label] = np.array(self.inverted_lists[label])

    def search(self, query, k=10):
        """执行IVF搜索"""
        if self.metric == "cosine":
            query = query / np.linalg.norm(query)

        # 找到最近的 nprobe 个聚类
        centroids = self.centroids
        centroid_distances = np.linalg.norm(centroids - query, axis=1)
        nearest_centroids = np.argsort(centroid_distances)[:self.nprobe]

        # 在选中的聚类中搜索
        candidates = []
        for label in nearest_centroids:
            if label in self.inverted_lists:
                cluster_vectors = self.inverted_lists[label]
                cluster_ids = self.inverted_ids[label]

                if self.metric == "cosine":
                    scores = np.dot(cluster_vectors, query)
                    for idx, score in zip(cluster_ids, scores):
                        candidates.append((score, idx))
                elif self.metric == "l2":
                    distances = np.linalg.norm(cluster_vectors - query, axis=1)
                    for idx, dist in zip(cluster_ids, distances):
                        candidates.append((-dist, idx))  # 取负值以便统一按降序排序

        # 排序并返回Top-K
        candidates.sort(key=lambda x: x[0], reverse=True)
        scores = np.array([c[0] for c in candidates[:k]])
        indices = np.array([c[1] for c in candidates[:k]])

        return scores, indices

# 使用示例
d = 1024
n = 50000
data = np.random.rand(n, d).astype(np.float32)

ivf = IVFIndex(nlist=200, nprobe=20, metric="cosine")
ivf.train(data)
ivf.add(data)

query = np.random.rand(d).astype(np.float32)
scores, indices = ivf.search(query, k=5)
print(f"IVF搜索结果索引: {indices}")
print(f"相似度分数: {scores}")
```

**参数影响**：

| 参数 | 增大时的影响 | 减小时的影响 |
|------|-------------|-------------|
| `nlist` | 更细的聚类→精度提高，训练时间增长 | 更粗的聚类→速度提高，精度下降 |
| `nprobe` | 搜索更多聚类→召回率提高，延迟增加 | 搜索更少聚类→延迟降低，召回率下降 |

**经验法则**：`nlist` 取 `4 × √n`，`nprobe` 取 `nlist / 10` 到 `nlist / 5`。

### 5.4.3 HNSW（分层可导航小世界图）

HNSW（Hierarchical Navigable Small World）是目前最流行的近似最近邻索引算法之一，以其出色的检索速度和召回率平衡著称。

**算法原理**：

HNSW的核心思想是构建一个**多层图结构**：

1. **底层（Layer 0）**：包含所有数据点，每个节点连接到最近的 `M` 个邻居
2. **上层（Layer 1, 2, ...）**：逐层稀疏，每层只包含部分数据点（通过指数衰减概率采样）
3. **搜索过程**：从顶层开始，在当前层的邻居中贪心搜索最接近查询的节点，然后逐层下探到底层

这种多层结构借鉴了跳表（skip list）的思想，使得搜索可以在对数时间内完成。

```
HNSW 搜索过程:
Layer 2:     ● ← 起始点
              ↓
Layer 1:   ● → ● → ● ← 搜索路径
              ↓    ↓
Layer 0:   ● → ● → ● → ● → ● ← 最终结果
```

```python
# HNSW 索引使用示例（使用 hnswlib 库）
import hnswlib
import numpy as np

# 参数配置
dim = 1024
num_elements = 100000

# 生成数据
data = np.float32(np.random.random((num_elements, dim)))
data = data / np.linalg.norm(data, axis=1, keepdims=True)  # L2归一化

# 初始化 HNSW 索引
index = hnswlib.Index(space='cosine', dim=dim)

# 初始化索引结构
index.init_index(
    max_elements=num_elements,
    ef_construction=200,  # 构建时的动态候选集大小
    M=32                  # 每个节点的最大连接数
)

# 设置随机种子（保证可复现性）
index.set_ef(50)  # 搜索时的动态候选集大小

# 添加数据
index.add_items(data, np.arange(num_elements))
print(f"索引中元素数量: {index.element_count}")

# 执行搜索
query = np.float32(np.random.random((1, dim)))
query = query / np.linalg.norm(query)

labels, distances = index.knn_query(query, k=10)
print(f"最近邻标签: {labels}")
print(f"距离: {distances}")

# 保存和加载索引
index.save_index("./data/hnsw_index.bin")
index.load_index("./data/hnsw_index.bin", max_elements=num_elements)
```

**搜索路径可视化**：

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_hnsw_search(index, query, num_layers=3):
    """
    可视化 HNSW 的分层搜索路径（简化版）
    注意：此函数仅用于教学演示，非实际HNSW实现
    """
    fig, axes = plt.subplots(1, num_layers, figsize=(15, 5))

    for layer in range(num_layers):
        ax = axes[layer]
        # 模拟每层的数据分布
        n_points = 100 // (2 ** layer)  # 上层点数更少
        points = np.random.randn(n_points, 2)

        # 模拟搜索路径
        path = []
        current = 0
        for _ in range(min(5, n_points)):
            path.append(current)
            # 贪心移动到最近的邻居
            dists = np.linalg.norm(points - points[current], axis=1)
            candidates = np.argsort(dists)[1:min(4, n_points)]
            current = candidates[0] if len(candidates) > 0 else current

        # 绘制
        ax.scatter(points[:, 0], points[:, 1], alpha=0.6, s=30)
        ax.scatter(0, 0, c='red', s=100, marker='x', label='Query')

        # 绘制搜索路径
        path_points = points[path]
        ax.plot(path_points[:, 0], path_points[:, 1],
                'g-', linewidth=2, alpha=0.8)
        ax.scatter(path_points[0, 0], path_points[0, 1],
                   c='green', s=50, marker='o', label='Start')

        ax.set_title(f'Layer {layer} (密度: {n_points} 个点)')
        ax.legend()

    plt.tight_layout()
    return fig
```

**HNSW参数详解**：

| 参数 | 描述 | 取值范围 | 对性能的影响 |
|------|------|---------|-------------|
| `M` | 每个节点的最大双向连接数 | 8-64 | M越大→召回率越高→内存占用越大→构建越慢 |
| `ef_construction` | 构建时的动态候选列表大小 | 100-500 | 越大→索引质量越高→构建越慢 |
| `ef_search` | 搜索时的动态候选列表大小 | 10-200 | 越大→召回率越高→搜索越慢 |

**经验法则**：
- 对于1024维的向量，`M=32` 是一个不错的起点
- `ef_construction` 通常设置为 `2 × M` 到 `8 × M`
- 在线搜索时，`ef_search` 通常设置为 `k × 2` 到 `k × 10`，其中k是返回结果数

### 5.4.4 索引类型对比总结

| 特性 | Flat | IVF | HNSW |
|------|------|-----|------|
| 搜索类型 | 精确搜索 | 近似搜索 | 近似搜索 |
| 召回率（@10） | 100% | 95-99% | 98-99.9% |
| 搜索复杂度 | O(n × d) | O(√n × d) | O(log n × d) |
| 内存占用 | O(n × d) | O(n × d + nlist × d) | O(n × M × 2) |
| 构建时间 | 无 | 中等（K-means训练） | 较慢（逐点插入） |
| 支持增量添加 | 是 | 部分（需重训练） | 是 |
| 适用于高维（>1000） | 是 | 中 | 是 |

## 5.5 相似度度量

相似度度量定义了向量之间"距离"的计算方式。选择正确的度量方式对于检索效果至关重要。

### 5.5.1 余弦相似度（Cosine Similarity）

余弦相似度衡量两个向量之间的**角度差异**，而非向量长度。计算公式为：

```
cosine_similarity(q, v) = (q · v) / (||q|| × ||v||)
                        = Σ(qᵢ × vᵢ) / (√Σqᵢ² × √Σvᵢ²)
```

**取值范围**：[-1, 1]，值越大表示越相似。

**数学特性**：
- 对向量长度不敏感，只关注方向
- 当向量经过L2归一化后，余弦相似度等价于内积

**使用场景**：文本嵌入、语义搜索——文本的"语义"主要体现在方向上，而非向量长度。

```python
def cosine_similarity_batch(query, database):
    """批量计算余弦相似度"""
    # L2 归一化
    query_norm = query / np.linalg.norm(query)
    db_norm = database / np.linalg.norm(database, axis=1, keepdims=True)
    # 归一化后内积等价于余弦相似度
    return np.dot(db_norm, query_norm)
```

### 5.5.2 L2距离（欧氏距离）

L2距离衡量向量在**欧几里得空间**中的直线距离：

```
L2_distance(q, v) = ||q - v||₂ = √Σ(qᵢ - vᵢ)²
```

**取值范围**：[0, +∞)，值越小表示越相似。

**数学特性**：
- 对向量长度敏感——长度差异大的向量天然距离远
- 在低维空间中直观性好，但在高维空间中所有点对的距离趋近于相等（维度灾难）

**使用场景**：图像嵌入、数值特征向量——当向量长度携带语义信息时。

```python
def l2_distance_batch(query, database):
    """批量计算 L2 距离"""
    diff = database - query
    return np.sqrt(np.sum(diff ** 2, axis=1))
```

### 5.5.3 内积（Inner Product / Dot Product）

内积直接计算两个向量对应位置乘积的和：

```
inner_product(q, v) = q · v = Σ(qᵢ × vᵢ)
```

**取值范围**：(-∞, +∞)，值越大表示越相似。

**数学特性**：
- 同时对向量方向和长度敏感
- 当向量未归一化时，长度大的向量天然获得更高分数

**使用场景**：推荐系统中的协同过滤、需要保留向量长度信息的场景。

### 5.5.4 度量选择指南

| 度量方式 | 是否需归一化 | 典型应用 | 备注 |
|---------|------------|---------|------|
| 余弦相似度 | 建议 | 文本语义搜索 | 最常用的文本嵌入度量 |
| L2距离 | 可选 | 图像检索、数值特征 | 在高维空间中效果可能不佳 |
| 内积 | 否 | 推荐系统、矩阵分解 | 对向量长度敏感 |

**实践建议**：当不确定选择哪种度量时，默认使用**余弦相似度**。大多数文本嵌入模型（如BGE、text-embedding-3、Nomic Embed）输出的向量已经经过归一化，此时余弦相似度与内积等价。

## 5.6 索引调优参数

索引构建不是一次性的操作，而是需要根据数据特性和业务需求进行反复调优的过程。本节详细讨论各索引的核心调优参数。

### 5.6.1 HNSW 参数调优

**M（最大连接数）**

`M` 控制HNSW图中每个节点的最大邻居数量。

- **M=8**：内存节省模式，适合内存受限场景
- **M=16**：均衡模式，适合大多数通用场景
- **M=32**：高质量模式，推荐用于高维向量（>768维）
- **M=64**：极致精度模式，适合对召回率要求极高的场景

```python
def tune_hnsw_m(data, query, ground_truth, m_values):
    """通过实验选择最优 M 值"""
    results = []
    for m in m_values:
        index = hnswlib.Index(space='cosine', dim=data.shape[1])
        index.init_index(
            max_elements=len(data),
            ef_construction=200,
            M=m
        )
        index.set_ef(50)
        index.add_items(data)

        # 评估召回率
        labels, _ = index.knn_query(query, k=10)
        recall = compute_recall(labels, ground_truth, k=10)

        # 评估内存占用
        memory = estimate_memory_hnsw(len(data), data.shape[1], m)

        results.append({
            'M': m,
            'recall@10': recall,
            'memory_mb': memory,
            'build_time_s': 0  # 实际运行时测量
        })

    return results

def compute_recall(predicted, ground_truth, k):
    """计算召回率"""
    correct = 0
    for pred, truth in zip(predicted, ground_truth):
        correct += len(set(pred[:k]) & set(truth[:k]))
    return correct / (len(predicted) * k)

def estimate_memory_hnsw(n, d, m):
    """估算HNSW索引内存占用（MB）"""
    # 向量数据: n * d * 4 bytes (float32)
    vector_memory = n * d * 4
    # 图结构: n * m * 2 * 4 bytes (int32 双向连接)
    graph_memory = n * m * 2 * 4
    # 层级信息: n * 4 bytes
    level_memory = n * 4
    total_bytes = vector_memory + graph_memory + level_memory
    return total_bytes / (1024 * 1024)
```

**efConstruction（构建质量）**

`ef_construction` 控制构建HNSW图时的搜索宽度。它决定了在插入每个节点时，考虑多少个候选节点来选择邻居。

- `ef_construction` 越大，图的质量越高，但构建时间线性增长
- 建议值：`M × 4` 到 `M × 8`
- 当 `ef_construction < M` 时，图的质量会严重下降

```python
# ef_construction 对索引质量的影响实验
def ef_construction_experiment():
    dim = 128
    n = 50000
    data = np.random.rand(n, dim).astype(np.float32)
    queries = np.random.rand(100, dim).astype(np.float32)

    ef_values = [50, 100, 200, 400]
    results = {}

    for ef in ef_values:
        index = hnswlib.Index(space='l2', dim=dim)
        index.init_index(max_elements=n, ef_construction=ef, M=16)
        index.set_ef(50)
        index.add_items(data)

        # 与暴力搜索对比召回率
        labels, _ = index.knn_query(queries, k=10)

        # 暴力搜索基准
        gt = brute_force_search(data, queries, k=10)

        recall = compute_recall(labels, gt, k=10)
        results[ef] = recall
        print(f"ef_construction={ef}: recall@10={recall:.4f}")

    return results

def brute_force_search(data, queries, k=10):
    """暴力搜索基准"""
    from scipy.spatial.distance import cdist
    results = []
    for q in queries:
        dists = cdist([q], data, metric='euclidean')[0]
        results.append(np.argsort(dists)[:k])
    return np.array(results)
```

**efSearch（搜索宽度）**

`ef_search` 是搜索时的关键调优参数。它控制搜索过程中维护的动态候选列表大小。

关键权衡：
- `ef_search` 越大，搜索越精确，但延迟越高
- `ef_search` 与召回率呈对数关系：初始增加时召回率提升显著，但存在边际效应

```python
# ef_search 延迟与召回率权衡
def ef_search_latency_tradeoff():
    """展示 ef_search 对延迟和召回率的影响"""
    index = hnswlib.Index(space='cosine', dim=1024)
    index.init_index(max_elements=100000, ef_construction=200, M=32)
    data = np.random.rand(100000, 1024).astype(np.float32)
    index.add_items(data)

    queries = np.random.rand(100, 1024).astype(np.float32)
    gt_labels = brute_force_search(data, queries, k=10)

    ef_values = [10, 20, 50, 100, 200, 400]
    for ef in ef_values:
        index.set_ef(ef)
        start = time.time()
        labels, _ = index.knn_query(queries, k=10)
        elapsed = time.time() - start

        recall = compute_recall(labels, gt_labels, k=10)
        avg_latency_ms = elapsed / len(queries) * 1000

        print(f"ef={ef:4d} | recall@10={recall:.4f} | "
              f"avg_latency={avg_latency_ms:.2f}ms")
```

### 5.6.2 IVF 参数调优

**nlist（聚类中心数量）**

`nlist` 控制K-means聚类时产生的聚类中心数量。

- 较小（nlist < 100）：聚类粗糙，搜索速度快但精度低
- 适中（100 < nlist < 1000）：均衡表现
- 较大（nlist > 1000）：聚类精细，精度高但训练和搜索都更慢

经验公式：`nlist = 4 × √n`，其中 n 是数据集大小。

```python
# nlist 选择策略
def select_nlist(dataset_size):
    """根据数据集大小选择 nlist"""
    nlist = int(4 * np.sqrt(dataset_size))
    # 确保 nlist 在合理范围内
    nlist = max(10, min(nlist, dataset_size // 10))
    return nlist

# 示例
for size in [10000, 100000, 1000000, 10000000]:
    print(f"数据集大小: {size:>10,d} → nlist={select_nlist(size)}")
```

**nprobe（搜索探访数）**

`nprobe` 控制搜索时探访的聚类数量。这是IVF中最直接的延迟-精度控制参数。

```python
# nprobe 参数影响分析
def analyze_nprobe_impact():
    """分析 nprobe 对检索性能的影响"""
    d = 1024
    n = 100000
    data = np.random.rand(n, d).astype(np.float32)

    # 构建IVF索引
    nlist = int(4 * np.sqrt(n))
    ivf = IVFIndex(nlist=nlist, nprobe=1)
    ivf.train(data)
    ivf.add(data)

    # 准备查询和基准
    queries = np.random.rand(50, d).astype(np.float32)
    gt_labels = brute_force_search(data, queries, k=10)

    nprobe_values = [1, 5, 10, 20, 50, 100]
    for nprobe in nprobe_values:
        ivf.nprobe = nprobe

        total_candidates = 0
        recalls = []

        for q in queries:
            scores, indices = ivf.search(q, k=10)
            recall = compute_recall(
                indices.reshape(1, -1),
                gt_labels[:1],
                k=10
            )
            recalls.append(recall)

        avg_recall = np.mean(recalls)
        # 搜索范围比例
        scope_ratio = nprobe / nlist * 100

        print(f"nprobe={nprobe:3d} | 搜索范围={scope_ratio:5.1f}% | "
              f"recall@10={avg_recall:.4f}")
```

### 5.6.3 系统级调优策略

除了索引参数外，系统层面的调优同样重要：

```python
# 综合调优示例
class VectorSearchTuner:
    """向量检索系统调优器"""

    def __init__(self, data, queries, ground_truth):
        self.data = data
        self.queries = queries
        self.ground_truth = ground_truth
        self.best_config = {}
        self.best_recall = 0.0

    def tune_hnsw(self, m_range=(8, 64), ef_range=(50, 400)):
        """网格搜索 HNSW 最优参数"""
        results = []

        for m in [8, 16, 32, 48, 64]:
            for ef in [50, 100, 200, 300, 400]:
                index = hnswlib.Index(space='cosine', dim=self.data.shape[1])
                index.init_index(
                    max_elements=len(self.data),
                    ef_construction=ef,
                    M=m
                )
                index.set_ef(ef)

                start = time.time()
                index.add_items(self.data)
                build_time = time.time() - start

                start = time.time()
                labels, _ = index.knn_query(self.queries, k=10)
                search_time = time.time() - start

                recall = compute_recall(
                    labels, self.ground_truth, k=10
                )

                results.append({
                    'M': m, 'ef': ef,
                    'recall': recall,
                    'build_time': build_time,
                    'search_time': search_time,
                    'search_qps': len(self.queries) / search_time
                })

        return results

    def recommend_config(self, recall_target=0.98, latency_budget_ms=50):
        """
        根据业务需求推荐最优配置

        参数:
            recall_target: 目标召回率（最低要求）
            latency_budget_ms: 延迟预算（毫秒，最高允许）
        """
        results = self.tune_hnsw()

        feasible = [
            r for r in results
            if r['recall'] >= recall_target
            and r['search_time'] * 1000 / len(self.queries) <= latency_budget_ms
        ]

        if not feasible:
            print(f"警告: 没有配置满足 recall≥{recall_target} 且 "
                  f"延迟≤{latency_budget_ms}ms")
            print("返回延迟最低的配置")
            return min(results, key=lambda r: r['search_time'])

        # 在可行配置中选择搜索速度最快的
        best = min(feasible, key=lambda r: r['search_time'])
        return best
```

## 5.7 实战：构建端到端向量存储系统

综合本章所学，下面构建一个完整的向量存储系统。

```python
"""
端到端向量存储系统示例

整合了嵌入生成、索引构建、检索和评估功能。
"""
import numpy as np
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

@dataclass
class VectorStoreConfig:
    """向量存储配置"""
    embedding_dim: int = 1024
    index_type: str = "hnsw"  # "flat" | "ivf" | "hnsw"
    metric: str = "cosine"

    # HNSW 参数
    M: int = 32
    ef_construction: int = 200
    ef_search: int = 50

    # IVF 参数
    nlist: int = 256
    nprobe: int = 20

class VectorStore:
    """统一的向量存储接口"""

    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self.index = None
        self.documents = []
        self.ids = []

    def add_documents(self, embeddings: np.ndarray,
                      documents: List[str],
                      ids: Optional[List[int]] = None):
        """添加文档到存储系统"""
        if ids is None:
            ids = list(range(len(documents)))

        self.documents.extend(documents)
        self.ids.extend(ids)

        if self.config.index_type == "flat":
            # Flat 索引：直接存储向量
            if self.index is None:
                self.index = embeddings
            else:
                self.index = np.vstack([self.index, embeddings])

        elif self.config.index_type == "hnsw":
            import hnswlib
            if self.index is None:
                self.index = hnswlib.Index(
                    space=self.config.metric,
                    dim=self.config.embedding_dim
                )
                self.index.init_index(
                    max_elements=len(embeddings) * 2,
                    ef_construction=self.config.ef_construction,
                    M=self.config.M
                )
                self.index.set_ef(self.config.ef_search)
                self.index.add_items(embeddings, np.array(ids))
            else:
                self.index.add_items(embeddings, np.array(ids))

        elif self.config.index_type == "ivf":
            from sklearn.cluster import MiniBatchKMeans
            # ... IVF 构建逻辑

    def search(self, query_embedding: np.ndarray, k: int = 10):
        """执行向量搜索"""
        if self.config.index_type == "flat":
            if self.config.metric == "cosine":
                query_norm = query_embedding / np.linalg.norm(query_embedding)
                db_norm = self.index / np.linalg.norm(
                    self.index, axis=1, keepdims=True
                )
                scores = np.dot(db_norm, query_norm)
                top_k = np.argsort(scores)[-k:][::-1]
                return {
                    "indices": top_k,
                    "scores": scores[top_k],
                    "documents": [self.documents[i] for i in top_k],
                    "ids": [self.ids[i] for i in top_k]
                }

        elif self.config.index_type == "hnsw":
            labels, distances = self.index.knn_query(
                query_embedding.reshape(1, -1), k
            )
            labels = labels[0]
            return {
                "indices": labels,
                "scores": 1 - distances[0],  # hnswlib返回L2距离，转换为相似度
                "documents": [self.documents[l] for l in labels],
                "ids": [self.ids[l] for l in labels]
            }

    def evaluate(self, queries: np.ndarray,
                 ground_truth: np.ndarray,
                 k: int = 10) -> Dict[str, float]:
        """评估检索性能"""
        total_recall = 0.0
        total_latency = 0.0

        for i, query in enumerate(queries):
            start = time.time()
            results = self.search(query, k=k)
            elapsed = time.time() - start

            predicted = set(results["indices"])
            expected = set(ground_truth[i][:k])
            recall = len(predicted & expected) / k

            total_recall += recall
            total_latency += elapsed

        return {
            "recall@k": total_recall / len(queries),
            "avg_latency_ms": total_latency / len(queries) * 1000,
            "throughput_qps": len(queries) / total_latency
        }

# 使用示例
if __name__ == "__main__":
    # 配置向量存储
    config = VectorStoreConfig(
        embedding_dim=1024,
        index_type="hnsw",
        metric="cosine",
        M=32,
        ef_construction=200,
        ef_search=100
    )

    # 创建存储实例
    store = VectorStore(config)

    # 模拟数据
    n_docs = 10000
    embeddings = np.random.rand(n_docs, 1024).astype(np.float32)
    documents = [f"文档_{i}的内容" for i in range(n_docs)]

    # 添加文档
    store.add_documents(embeddings, documents)

    # 搜索
    query = np.random.rand(1024).astype(np.float32)
    results = store.search(query, k=5)

    print("搜索结果:")
    for i, (doc, score) in enumerate(
        zip(results["documents"], results["scores"])
    ):
        print(f"  {i+1}. [{results['ids'][i]}] {doc} (score={score:.4f})")

    # 评估
    test_queries = np.random.rand(100, 1024).astype(np.float32)
    test_gt = np.random.randint(0, n_docs, size=(100, 10))
    metrics = store.evaluate(test_queries, test_gt, k=10)
    print(f"\n评估结果:")
    print(f"  recall@10: {metrics['recall@k']:.4f}")
    print(f"  平均延迟: {metrics['avg_latency_ms']:.2f}ms")
    print(f"  吞吐量: {metrics['throughput_qps']:.0f} qps")
```

## 5.8 本章小结

向量存储是RAG系统的核心基础设施，直接影响检索质量和系统性能。本章的核心要点如下：

1. **嵌入模型选型**：BGE-M3适合多语言场景，text-embedding-3-large适合追求极致质量的商业场景，Nomic Embed Text适合开源本地部署。模型选择决定了向量表示质量的上限。

2. **数据库架构选择**：LanceDB的嵌入式架构适合原型和轻量级应用，FAISS的内存索引提供极致性能，Milvus的分布式架构支撑企业级大规模系统。架构选择决定了系统的扩展边界。

3. **索引类型**：Flat提供精确但低效的检索，IVF通过聚类剪枝实现高效的近似搜索，HNSW通过多层图结构在对数时间内完成搜索。索引选择是延迟-精度的核心权衡。

4. **相似度度量**：余弦相似度是文本检索的首选度量，L2距离适合数值特征，内积适合需要保留向量长度信息的场景。

5. **参数调优**：HNSW的M、efConstruction、efSearch和IVF的nlist、nprobe是最关键的调优参数。调优的本质是在延迟预算内最大化召回率。

在实践中，向量存储的优化不是一劳永逸的。随着数据量的增长、查询分布的变化和业务需求的变化，需要持续监控检索质量并调整配置。建议在生产环境中建立完善的监控体系，定期评估recall@k和搜索延迟，确保系统始终在最优状态下运行。

在下一章中，我们将探讨如何在索引之上构建更智能的检索策略——包括查询重写、检索结果重排序和混合检索模式。
