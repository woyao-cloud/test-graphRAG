# 第6章：Milvus核心数据模型与RAG业务适配

## 6.1 核心概念

Milvus的数据模型围绕四个核心概念构建：集合（Collection）、分区（Partition）、字段（Field）和实体（Entity）。理解这些概念及其在RAG场景中的最佳实践，是设计高效知识库的基础。

### 6.1.1 集合（Collection）

集合是Milvus中最顶层的数据组织单元，类似于关系数据库中的表。一个集合包含一组字段定义（Schema）和一批数据实体。在RAG系统中，一个集合通常对应一个知识库。

```python
from pymilvus import CollectionSchema, FieldSchema, DataType, Collection

# 定义RAG知识库的集合Schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="created_at", dtype=DataType.INT64),
]
schema = CollectionSchema(fields, description="RAG知识库")
collection = Collection(name="rag_knowledge", schema=schema)
```

集合的创建是一个DDL操作，由RootCoord处理。创建完成后，集合的元数据被持久化到Etcd中。

### 6.1.2 分区（Partition）

分区是集合内部的逻辑子集，类似于关系数据库中的分区表。通过将数据划分到不同的分区，可以在检索时缩小搜索范围，提升查询性能。

```python
# 创建分区
collection.create_partition(partition_name="doc_type_report")
collection.create_partition(partition_name="doc_type_manual")
collection.create_partition(partition_name="doc_type_faq")
```

每个集合默认有一个`_default`分区。当数据写入时未指定分区，会被分配到默认分区。

### 6.1.3 字段（Field）

字段是集合中的数据列定义。Milvus支持以下几种字段类型：

| 字段类型 | 说明 | RAG中的典型用途 |
|---------|------|----------------|
| INT64 | 64位整数 | 主键ID |
| VARCHAR | 变长字符串 | 文档原文、来源 |
| FLOAT_VECTOR | 浮点向量 | 嵌入向量 |
| BINARY_VECTOR | 二进制向量 | 压缩后的嵌入向量 |
| INT8/INT16/INT32 | 整数类型 | 标签ID、状态码 |
| BOOL | 布尔值 | 启用状态 |
| FLOAT | 浮点数 | 分数、权重 |
| JSON | JSON对象 | 灵活的元数据 |

**向量字段的特殊性**：

向量字段是Milvus区别于传统数据库的核心。定义向量字段时，必须指定向量的维度（dim）。在RAG系统中，维度取决于嵌入模型：

| 嵌入模型 | 向量维度 |
|---------|---------|
| text-embedding-3-large | 3072 |
| text-embedding-3-small | 1536 |
| bge-m3 | 1024 |
| bge-large-zh-v1.5 | 1024 |
| bge-small-zh-v1.5 | 512 |

### 6.1.4 主键（Primary Key）

主键是实体在集合中的唯一标识。Milvus支持自动生成和手动指定两种主键模式：

- **自动生成**（推荐）：设置`auto_id=True`，Milvus自动生成自增ID
- **手动指定**：业务系统自行管理唯一ID，适合需要与外部系统关联的场景

```python
# 自动主键
FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True)

# 手动主键（使用文档URL或哈希值）
FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=256, is_primary=True)
```

## 6.2 字段类型设计：RAG专属方案

设计合理的字段结构是RAG系统数据模型的核心。以下是一个经过生产验证的RAG知识库Schema设计。

### 6.2.1 推荐Schema模板

```python
def create_rag_collection(collection_name, embedding_dim=1024):
    """创建标准的RAG知识库集合"""
    fields = [
        # 主键（自动生成）
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        
        # 向量字段（必选）
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim),
        
        # 文档内容（必选）
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        
        # 文档标题（可选，用于展示和排序）
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=1024),
        
        # 文档来源（可选，用于溯源）
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=256),
        
        # 文档类型（可选，用于过滤）
        FieldSchema(name="doc_type", dtype=DataType.VARCHAR, max_length=64),
        
        # 业务标签（可选，用于多维度过滤）
        FieldSchema(name="tags", dtype=DataType.VARCHAR, max_length=512),
        
        # 时间戳（可选，用于时间范围过滤）
        FieldSchema(name="created_at", dtype=DataType.INT64),
        
        # 文档长度（可选，用于质量过滤）
        FieldSchema(name="text_length", dtype=DataType.INT32),
        
        # 相似度分数（可选，用于Rerank后的持久化）
        FieldSchema(name="score", dtype=DataType.FLOAT),
    ]
    
    schema = CollectionSchema(
        fields, 
        description=f"RAG知识库: {collection_name}",
        # 启用动态字段（允许写入未在Schema中定义的字段）
        enable_dynamic_field=True
    )
    
    collection = Collection(name=collection_name, schema=schema)
    return collection
```

### 6.2.2 动态字段

Milvus支持动态字段（Dynamic Field）特性，允许在写入数据时包含Schema中未明确定义的字段。动态字段以JSON格式存储，在查询时可以通过表达式进行过滤。

```python
# 创建支持动态字段的集合
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
]
schema = CollectionSchema(fields, enable_dynamic_field=True)
collection = Collection(name="dynamic_demo", schema=schema)

# 写入动态字段数据
collection.insert([
    [np.random.rand(1024).tolist()],   # embedding
    ["文档内容"],                        # text
    # 动态字段
    {"department": "技术部", "author": "张三", "version": "2.1"}
])

# 查询动态字段
results = collection.query(
    expr='json_contains(department, "技术部")',
    output_fields=["text", "department"]
)
```

动态字段在RAG场景中的典型用途：
- **灵活的元数据**：不同来源的文档可能有不同的元数据字段
- **权限标记**：为每个文档附加访问权限信息
- **AB测试标记**：标记文档所属的实验组

### 6.2.3 稀疏向量

Milvus 2.4+版本开始支持稀疏向量（Sparse Vector）。稀疏向量是一种高维但大部分维度为0的向量表示，与传统的稠密向量（Dense Vector）形成互补。

**稀疏向量的优势**：
- 对精确关键词匹配效果好（类似BM25的机制）
- 可以在一个集合中同时使用稠密向量和稀疏向量
- 适合处理长尾词汇和专有名词

```python
# 定义包含稀疏向量的集合
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    # 稠密向量（语义检索）
    FieldSchema(name="dense_embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
    # 稀疏向量（关键词检索）
    FieldSchema(name="sparse_embedding", dtype=DataType.SPARSE_FLOAT_VECTOR),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
]
schema = CollectionSchema(fields)
collection = Collection(name="hybrid_demo", schema=schema)
```

在RAG场景中，稠密向量+稀疏向量的混合检索可以实现更好的召回效果——稠密向量负责语义匹配，稀疏向量负责精确关键词匹配。

## 6.3 分区策略

合理使用分区可以显著提升RAG系统的检索性能。分区相当于对数据做了一层预过滤，搜索时只需在相关分区内查找，减少了搜索空间。

### 6.3.1 按文档类型分区

将不同类型的文档（报告、手册、FAQ等）划分到不同的分区。

```python
# 按文档类型分区
doc_types = ["report", "manual", "faq", "policy", "news"]
for dt in doc_types:
    collection.create_partition(partition_name=f"type_{dt}")

# 写入时指定分区
collection.insert(
    data=[embeddings, texts, sources],
    partition_name="type_report"
)

# 搜索时指定分区范围
results = collection.search(
    data=[query_vector],
    anns_field="embedding",
    param=search_params,
    limit=10,
    # 只在报告和手册分区中搜索
    partition_names=["type_report", "type_manual"]
)
```

### 6.3.2 按时间分区

将数据按时间维度（年/月/日）分区，适合具有明确时效性的知识库。

```python
def time_partition_name(timestamp_ms):
    """根据时间戳生成分区名"""
    from datetime import datetime
    dt = datetime.fromtimestamp(timestamp_ms / 1000)
    return f"year_{dt.year}_month_{dt.month}"

# 写入时按时间分区
for text, emb, ts in zip(texts, embeddings, timestamps):
    partition = time_partition_name(ts)
    # 分区不存在则创建
    if not collection.has_partition(partition):
        collection.create_partition(partition_name=partition)
    
    collection.insert(
        data=[[emb], [text], [ts]],
        partition_name=partition
    )

# 搜索时限定时间范围
search_partitions = [
    "year_2026_month_1",
    "year_2026_month_2",
    "year_2026_month_3",
]
results = collection.search(
    data=[query_vector],
    anns_field="embedding",
    param=search_params,
    limit=10,
    partition_names=search_partitions
)
```

### 6.3.3 按业务模块分区

在企业级RAG系统中，不同业务模块的知识应该物理隔离。

```python
# 按业务模块分区
modules = ["hr_policy", "product_manual", "tech_doc", "legal_contract"]
for module in modules:
    collection.create_partition(partition_name=f"module_{module}")

# 搜索时只检索用户有权限的模块
def search_in_modules(collection, query_vector, allowed_modules):
    """在用户有权限的模块中搜索"""
    partitions = [f"module_{m}" for m in allowed_modules]
    return collection.search(
        data=[query_vector],
        anns_field="embedding",
        param=search_params,
        limit=10,
        partition_names=partitions
    )
```

### 6.3.4 分区策略选择建议

| 分区策略 | 适用场景 | 优势 | 劣势 |
|---------|---------|------|------|
| 按文档类型 | 多类型知识库 | 检索范围精确 | 类型间数据不均衡 |
| 按时间 | 时效性强的知识库 | 天然支持时间过滤 | 历史分区访问频率低 |
| 按业务模块 | 企业级多租户 | 天然的数据隔离 | 跨模块查询需要搜索多个分区 |
| 不分区 | 小规模知识库 | 实现简单 | 无性能优化 |

## 6.4 主键设计与重复去重

### 6.4.1 主键设计原则

在RAG场景中，文档去重是一个重要需求。合理设计主键可以避免同一文档被重复入库。

**基于内容哈希的主键**：

```python
import hashlib

def generate_doc_id(content):
    """基于文档内容生成唯一ID"""
    return hashlib.md5(content.encode()).hexdigest()

# 使用内容哈希作为主键
fields = [
    FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
]
```

**upsert操作**：

Milvus的upsert操作可以实现"存在则更新，不存在则插入"的语义，非常适合增量更新场景。

```python
def upsert_document(collection, doc_id, text, embedding):
    """插入或更新文档"""
    collection.upsert([
        [doc_id],                          # 主键
        [embedding.tolist()],              # 向量
        [text]                             # 文本
    ])
```

### 6.4.2 检索去重

即使入库时做了去重，多路检索仍然可能返回重复结果。建议在应用层进行去重：

```python
def dedup_search_results(results):
    """对检索结果进行去重"""
    seen = set()
    deduped = []
    for hit in results[0]:
        content_key = hit.entity.get("text", "")[:100]
        if content_key not in seen:
            seen.add(content_key)
            deduped.append(hit)
    return deduped
```

## 6.5 本章小结

Milvus的数据模型为RAG系统提供了灵活而强大的知识组织能力。集合对应知识库，分区提供逻辑隔离，字段定义支持丰富的元数据类型，主键设计则解决了文档去重问题。

在实际的RAG项目中，推荐遵循以下数据模型设计原则：

1. **集合粒度**：每个独立的知识域创建一个集合，如"产品知识库"、"技术文档库"、"政策法规库"
2. **分区策略**：根据访问模式选择分区策略，按业务模块或文档类型分区最为常用
3. **字段设计**：向量字段+文本字段是必选项，建议额外添加来源、类型、时间等标量字段用于过滤
4. **主键选择**：推荐使用内容哈希作为主键，天然支持去重
5. **动态字段**：在Schema不固定的场景下启用，但需要注意性能影响
6. **稀疏向量**：Milvus 2.4+的稀疏向量特性为实现混合检索提供了原生支持
