# 第9章: Milvus基础操作：RAG必备CRUD实战

## 9.1 引言

在RAG系统中，向量数据库承担着知识库存储和语义检索的双重责任。掌握Milvus的CRUD（增删改查）操作，是构建RAG应用的必备基本功。不同于关系型数据库，向量数据库的操作涉及向量字段与标量字段的协同管理，索引构建与检索参数的精细配置。本章将从Python SDK环境搭建开始，以RAG场景的真实需求为驱动，完整演示集合创建、数据写入、索引构建、检索查询、数据删除与更新等核心操作。本章内容对应项目`demos/ch09-crud-basics/main.py`中的完整代码实现。

## 9.2 Python SDK环境搭建

### 9.2.1 安装pymilvus

Milvus的Python SDK名为`pymilvus`，是操作Milvus的标准接口：

```bash
pip install pymilvus>=2.4.0
```

验证安装：

```python
import pymilvus
print(pymilvus.__version__)
# 输出示例: 2.4.0
```

### 9.2.2 连接Milvus服务

连接方式取决于部署模式：

```python
from pymilvus import MilvusClient

# 方式一：连接远程Milvus服务（Docker / K8s部署）
client = MilvusClient(uri="http://localhost:19530")

# 方式二：Embed模式（本地文件存储，无需Docker）
client = MilvusClient(uri="./milvus_rag.db")

# 方式三：连接云端Milvus（Zilliz Cloud）
client = MilvusClient(
    uri="https://your-instance.zillizcloud.com:19530",
    token="your-api-token"
)
```

在RAG开发阶段，推荐使用方式一连接本地Docker部署的Milvus；在原型验证阶段，可以使用方式二直接写入本地文件，无需任何外部依赖。

### 9.2.3 MilvusClient vs 旧版Collection接口

`MilvusClient`是pymilvus 2.4+引入的新API，相比旧版`Collection`接口，具有以下优势：

- **更简洁的API设计**：一个Client对象即可完成所有操作，无需反复创建Collection对象。
- **自动管理连接**：不再需要手动调用`connections.connect()`和`collection.load()`。
- **更好的类型提示**：IDE自动补全和类型检查支持更完善。

本书所有Demo代码均使用`MilvusClient`接口。

## 9.3 集合创建：RAG专属字段结构设计

### 9.3.1 Schema设计原则

RAG知识库的集合Schema设计直接影响检索效果和开发效率。一个典型的RAG知识库集合至少需要以下字段：

| 字段名 | 类型 | 用途 | 说明 |
|--------|------|------|------|
| id | INT64 | 主键 | 文档唯一标识，建议用自增整数或文档哈希 |
| vector | FLOAT_VECTOR | 向量字段 | 文档的Embedding向量，维度取决于Embedding模型 |
| text | VARCHAR | 文档原文 | 存储分块后的文本，用于LLM上下文拼接 |
| category | VARCHAR | 分类标签 | 可选，用于过滤检索范围 |
| timestamp | INT64 | 时间戳 | 用于时间范围过滤或增量更新 |

### 9.3.2 创建集合的完整代码

```python
from pymilvus import MilvusClient, DataType

COLLECTION_NAME = "rag_knowledge_base"
DIM = 768  # 取决于使用的Embedding模型维度

# 创建Schema
schema = MilvusClient.create_schema(
    auto_id=False,        # 手动指定ID
    enable_dynamic_field=False,  # 不启用动态字段，保持Schema严格
)

# 添加字段
schema.add_field("id", DataType.INT64, is_primary=True)
schema.add_field("vector", DataType.FLOAT_VECTOR, dim=DIM)
schema.add_field("text", DataType.VARCHAR, max_length=2048)
schema.add_field("category", DataType.VARCHAR, max_length=64)
schema.add_field("timestamp", DataType.INT64)

# 创建集合
client.create_collection(
    collection_name=COLLECTION_NAME,
    schema=schema,
)

print(f"集合 '{COLLECTION_NAME}' 创建成功，向量维度: {DIM}")
```

### 9.3.3 集合参数详解

**auto_id参数**：当设置为`True`时，Milvus自动生成唯一主键，适合不需要自定义ID的场景。但RAG场景中通常需要建立文档ID与业务ID的映射关系，建议设置为`False`，由应用层管理ID。

**enable_dynamic_field参数**：启用后，插入未在Schema中定义的字段时，Milvus会自动将其作为动态字段存储。好处是灵活，坏处是会增加存储开销和检索复杂度。RAG生产环境建议关闭，保持Schema的清晰可维护。

**max_length参数**：VARCHAR类型的字段需要指定最大长度。RAG文本分块通常在256-1024个token之间，对应约512-2048个中文字符。需要根据实际分块策略调整。

## 9.4 数据写入

### 9.4.1 单条插入

RAG系统中最基本的写入操作是单条插入，适合实时问答场景中逐条添加知识：

```python
import time

single_record = {
    "id": 1,
    "vector": [0.1] * DIM,  # 此处使用占位向量，实际应用中替换为真实Embedding
    "text": "阿司匹林是一种非甾体抗炎药，常用于缓解轻度至中度疼痛和退烧。",
    "category": "解热镇痛",
    "timestamp": int(time.time()),
}

result = client.insert(COLLECTION_NAME, single_record)
print(f"插入成功，影响行数: {result['insert_count']}")
```

### 9.4.2 批量插入

在知识库初始化或全量更新时，批量插入是最高效的方式：

```python
documents = [
    {
        "id": i,
        "vector": [0.1] * DIM,  # 替换为真实Embedding
        "text": text,
        "category": category,
        "timestamp": int(time.time()),
    }
    for i, (text, category) in enumerate(zip(texts, categories), start=2)
]

result = client.insert(COLLECTION_NAME, documents)
print(f"批量插入成功，共 {len(documents)} 条记录，影响行数: {result['insert_count']}")
```

**批量插入的性能建议**：
- 单次批量建议控制在100-1000条之间。太少则网络开销大，太多则内存占用高。
- 对于大规模初始化（百万级），建议使用多线程或异步方式分批写入。
- 如果使用Embed模式，批量过大可能导致Python进程内存溢出。

### 9.4.3 Upsert（增量更新）

RAG知识库经常需要更新已有文档的内容。Upsert操作可以一条指令完成"存在则更新，不存在则插入"的逻辑：

```python
updated_record = {
    "id": 1,
    "vector": [0.2] * DIM,  # 更新后的Embedding
    "text": "阿司匹林——经典解热镇痛药，更新版说明：增加预防心脑血管疾病的说明。",
    "category": "解热镇痛",
    "timestamp": int(time.time()),
}

result = client.upsert(COLLECTION_NAME, updated_record)
print(f"Upsert成功，影响行数: {result['upsert_count']}")

# 验证更新结果
retrieved = client.get(COLLECTION_NAME, ids=[1])[0]
print(f"更新后文本: {retrieved['text']}")
```

Upsert在RAG场景中非常实用，例如：
- 知识库文档修订后重新生成Embedding并更新。
- 增量抓取的数据中，已有文档需要覆盖更新。
- 修复错误标注的分类标签。

## 9.5 索引创建与参数配置

### 9.5.1 为什么要建索引

没有索引时，Milvus对向量执行暴力检索（FLAT），逐条计算所有向量与查询向量的距离。当数据量超过10万条时，暴力检索的延迟会变得不可接受。索引通过预组织向量结构（如聚类、图结构），大幅减少需要计算的距离次数。

### 9.5.2 创建索引

```python
# 准备索引参数
index_params = MilvusClient.prepare_index_params()
index_params.add_index(
    field_name="vector",
    index_type="IVF_FLAT",   # 索引类型
    metric_type="IP",        # 相似度度量方式
    params={"nlist": 128},   # 索引参数
)

# 创建索引
client.create_index(COLLECTION_NAME, index_params)
print("IVF_FLAT索引创建成功")

# 加载集合到内存
client.load_collection(COLLECTION_NAME)
print("集合已加载到内存，准备就绪")
```

### 9.5.3 索引类型选型

RAG场景中常见的索引类型对比如下：

| 索引类型 | 适用数据量 | 检索速度 | 内存占用 | 召回率 | RAG推荐场景 |
|---------|-----------|---------|---------|--------|-----------|
| FLAT | <10万 | 慢 | 高 | 100% | 小知识库、精确检索 |
| IVF_FLAT | 10万-100万 | 中 | 中 | 95%+ | 中型RAG知识库 |
| IVF_SQ8 | 10万-1000万 | 快 | 低（压缩75%） | 92%+ | 内存受限场景 |
| HNSW | 100万-1亿 | 极快 | 高 | 98%+ | 高召回+低延迟场景 |
| IVF_PQ | >1000万 | 快 | 极低 | 85%+ | 海量数据压缩 |

### 9.5.4 检索参数

除了索引参数，每次检索时还可以指定`search_params`来微调检索行为：

```python
search_params = {
    "metric_type": "IP",
    "params": {
        "nprobe": 32  # IVF索引的检索深度，越大召回越高但越慢
    }
}

results = client.search(
    collection_name=COLLECTION_NAME,
    data=[query_vector],
    anns_field="vector",
    search_params=search_params,
    limit=10,
)
```

`nprobe`参数是IVF系列索引的关键调优参数。默认值通常为8-16，增大到32-64可以提升召回率，但检索时间也会线性增加。

## 9.6 检索、过滤与TopK

### 9.6.1 基础向量检索

最基本的语义检索操作，根据向量相似度返回最相似的TopK条记录：

```python
query_vector = [0.15] * DIM  # 替换为查询文本的Embedding

results = client.search(
    collection_name=COLLECTION_NAME,
    data=[query_vector],
    anns_field="vector",
    limit=3,
    output_fields=["id", "text", "category"],
)

print("Top 3 检索结果:")
for i, hit in enumerate(results[0], start=1):
    entity = hit["entity"]
    print(f"  {i}. id={entity['id']}  text={entity['text']}  "
          f"category={entity['category']}  score={hit['distance']:.4f}")
```

### 9.6.2 带标量过滤的混合检索

RAG场景中经常需要限定检索范围，例如只检索特定分类的知识：

```python
results = client.search(
    collection_name=COLLECTION_NAME,
    data=[query_vector],
    anns_field="vector",
    limit=3,
    output_fields=["id", "text", "category"],
    filter='category == "抗肿瘤"',
)
```

过滤表达式支持多种语法：

```python
# 多条件AND
filter='category in ["抗生素", "解热镇痛"] and timestamp > 1700000000'

# 字符串模糊匹配（Milvus 2.3+）
filter='text like "%阿司匹林%"'

# 数值范围过滤
filter='id > 100 and id < 200'
```

### 9.6.3 TopK参数对RAG的影响

`limit`参数（即TopK）决定了每次检索返回的结果数量，直接影响RAG问答质量：

- **TopK=1~3**：适合精确问答，只取最相关的少数片段，减少LLM上下文噪音。
- **TopK=5~10**：通用RAG场景，提供足够的上下文供LLM综合判断。
- **TopK=10~20**：需要LLM自行筛选的场景，如多文档综合问答。

**经验法则**：如果LLM生成的答案出现幻觉或偏离事实，尝试减小TopK；如果答案信息不足或遗漏关键信息，尝试增大TopK。

## 9.7 数据删除、更新与过期清理

### 9.7.1 按ID删除

```python
# 删除单条记录
result = client.delete(COLLECTION_NAME, ids=[3])
print(f"删除成功，影响行数: {result['delete_count']}")

# 批量删除
result = client.delete(COLLECTION_NAME, ids=[5, 6, 7, 8])
```

### 9.7.2 按过滤条件删除（批量清理）

Milvus 2.3+支持使用过滤表达式批量删除记录：

```python
# 删除过期数据（30天前的记录）
import time
thirty_days_ago = int(time.time()) - 30 * 24 * 3600
result = client.delete(
    COLLECTION_NAME,
    filter=f'timestamp < {thirty_days_ago}',
)
print(f"已清理 {result['delete_count']} 条过期数据")
```

### 9.7.3 更新操作

Milvus没有直接的"更新"API，更新操作通过Upsert实现：使用相同的ID插入新数据即可覆盖旧数据。

```python
# 更新文档内容和向量
client.upsert(COLLECTION_NAME, {
    "id": 1,
    "vector": new_embedding,  # 重新生成Embedding
    "text": "更新后的文档内容",
    "category": "新分类",
    "timestamp": int(time.time()),
})
```

### 9.7.4 RAG知识库过期清理策略

RAG知识库中的数据会随时间推移变得陈旧，需要制定合理的过期清理策略：

**策略一：基于时间戳的定期清理**

```python
def clean_expired_data(client, collection, retention_days=90):
    """清理指定天数前的过期数据"""
    cutoff = int(time.time()) - retention_days * 24 * 3600
    result = client.delete(collection, filter=f'timestamp < {cutoff}')
    print(f"清理完成，移除 {result['delete_count']} 条过期记录")
    return result['delete_count']
```

**策略二：版本号管理**

```python
# 插入数据时携带版本号
data = {
    "id": doc_id,
    "vector": embedding,
    "text": content,
    "version": 2,  # 版本号递增
    "timestamp": int(time.time()),
}
client.upsert(COLLECTION_NAME, data)

# 清理旧版本（保留每个文档的最新版本）
# 需要应用层维护文档ID与最新版本的映射关系
```

**策略三：冷热数据分离**

将近期高频访问的数据放在热集合（高性能索引），历史数据放在冷集合（压缩索引或归档存储），通过路由逻辑决定查询范围。

## 9.8 数据备份与恢复

### 9.8.1 使用Milvus Backup工具

Milvus官方提供了`milvus-backup`工具，支持数据备份与恢复：

```bash
# 安装milvus-backup
pip install milvus-backup

# 备份整个实例
milvus-backup create -n rag_backup_20260713

# 恢复备份
milvus-backup restore -n rag_backup_20260713
```

### 9.8.2 手动备份方案

如果无法使用备份工具，可以通过导出数据的方式手动备份：

```python
def backup_collection(client, collection_name, output_file):
    """将集合数据导出为JSON文件"""
    import json
    
    # 全量查询（注意：数据量大时需分页）
    results = client.query(
        collection_name=collection_name,
        output_fields=["*"],
        limit=10000,
    )
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"备份完成，共 {len(results)} 条记录，保存至 {output_file}")
```

## 9.9 本章小结

本章以RAG场景为核心，完整演示了Milvus的Python SDK使用流程：从环境搭建、集合Schema设计、数据写入（单条/批量/Upsert）、索引创建与参数配置、向量检索与标量过滤，到数据删除、更新和过期清理。这些基础CRUD操作构成了RAG知识库管理的最小必要技能集。

在实际RAG项目中，这些操作会被封装到数据管道（Data Pipeline）中，实现从文档加载到向量入库的自动化流程。下一章将在此基础上，搭建一个最简版的Milvus-RAG问答系统，将这些CRUD操作串联到完整的问答链路中。
