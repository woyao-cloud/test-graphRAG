# 第19章 通用企业知识库问答系统

企业知识库问答系统是RAG技术最典型、最广泛的应用场景之一。大型企业内部通常积累了海量的文档资料，包括人事制度、技术规范、财务流程等，员工在日常工作中需要频繁查阅这些资料。传统的文档管理方式要么依赖人工检索（速度慢、效率低），要么依赖关键词搜索（语义理解差、召回不准）。本章将基于Milvus向量数据库，完整实现一个多部门、多角色、多权限的企业知识库问答系统。

## 19.1 需求分析与架构设计

### 19.1.1 业务需求梳理

一个典型的企业知识库问答系统需要满足以下核心需求：

- **多部门数据隔离**：不同部门（HR、技术部、财务部）的文档需要逻辑隔离，避免跨部门误检索。
- **角色权限控制**：不同角色（管理员、经理、普通员工）可访问的文档范围不同。
- **语义检索**：用户可以用自然语言提问，而非仅靠关键词匹配。
- **元数据过滤**：支持按部门、作者、时间、标签等字段过滤检索结果。
- **文档管理**：支持文档的增删改查，以及文档元数据的管理和统计。

### 19.1.2 技术架构设计

系统采用分层架构设计，从上到下依次为：

```
┌─────────────────────────────────────────┐
│              用户交互层                    │
│    Web UI / API 接口 / 命令行工具          │
├─────────────────────────────────────────┤
│              业务逻辑层                    │
│  权限校验 → 查询路由 → 结果组装 → LLM问答   │
├─────────────────────────────────────────┤
│              检索服务层                    │
│  向量检索 / 标量过滤 / 混合检索 / 重排序    │
├─────────────────────────────────────────┤
│              数据存储层                    │
│    Milvus（向量+标量）   LLM（文本生成）     │
└─────────────────────────────────────────┘
```

### 19.1.3 Milvus 数据模型设计

针对企业知识库场景，我们设计如下的集合（Collection）结构：

| 字段名 | 类型 | 说明 |
|--------|------|------|
| id | INT64 (主键) | 自动生成唯一ID |
| vector | FLOAT_VECTOR (64维) | 文档的向量表示 |
| title | VARCHAR(256) | 文档标题 |
| content | VARCHAR(1024) | 文档正文内容 |
| department | VARCHAR(32) | 所属部门 |
| author | VARCHAR(64) | 文档作者 |
| created_at | VARCHAR(32) | 创建时间 |
| tags | VARCHAR(256) | 标签（逗号分隔） |

其中`department`字段作为分区键，每个部门的数据存储在独立的分区中，检索时可以指定分区范围，实现高效的部门级数据隔离。

## 19.2 全流程代码实现

### 19.2.1 环境准备与连接

```python
from pymilvus import MilvusClient

# 连接 Milvus 服务
MILVUS_URI = "http://localhost:19530"
client = MilvusClient(uri=MILVUS_URI)
```

### 19.2.2 集合与分区创建

企业知识库需要为每个部门创建独立的分区，以实现数据隔离：

```python
COLLECTION_NAME = "enterprise_kb"
DEPARTMENTS = ["HR", "Tech", "Finance"]
DIM = 64

def ensure_collection(client: MilvusClient):
    """创建集合和部门分区"""
    # 删除已有集合（演示场景）
    if client.has_collection(COLLECTION_NAME):
        client.drop_collection(COLLECTION_NAME)

    # 定义 Schema
    schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=True)
    schema.add_field("id", datatype="INT64", is_primary=True, auto_id=True)
    schema.add_field("vector", datatype="FLOAT_VECTOR", dim=DIM)
    schema.add_field("title", datatype="VARCHAR", max_length=256)
    schema.add_field("content", datatype="VARCHAR", max_length=1024)
    schema.add_field("department", datatype="VARCHAR", max_length=32)
    schema.add_field("author", datatype="VARCHAR", max_length=64)
    schema.add_field("created_at", datatype="VARCHAR", max_length=32)
    schema.add_field("tags", datatype="VARCHAR", max_length=256)

    # 配置索引参数
    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(field_name="vector", index_type="FLAT", metric_type="L2")

    # 创建集合，指定分区数量
    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
        num_partitions=len(DEPARTMENTS),
    )

    # 为每个部门创建分区
    for dept in DEPARTMENTS:
        client.create_partition(collection_name=COLLECTION_NAME, partition_name=dept)

    print(f"集合 '{COLLECTION_NAME}' 创建成功，共 {len(DEPARTMENTS)} 个分区")
```

### 19.2.3 文档入库

```python
def insert_documents(client: MilvusClient):
    """将文档插入对应的部门分区"""
    authors_by_dept = {
        "HR": ["张经理", "王主管"],
        "Tech": ["李工", "赵工", "刘架构"],
        "Finance": ["陈总监", "周会计"],
    }

    for dept, docs in DOCUMENTS.items():
        data = []
        for doc in docs:
            # 生成向量（实际项目中应使用 Embedding 模型）
            vector = generate_embedding(doc["title"] + doc["content"])
            data.append({
                "vector": vector,
                "title": doc["title"],
                "content": doc["content"],
                "department": dept,
                "author": random.choice(authors_by_dept[dept]),
                "created_at": datetime.now().isoformat(),
                "tags": f"enterprise,{dept.lower()},kb",
            })
        # 插入到指定分区
        result = client.insert(
            collection_name=COLLECTION_NAME,
            data=data,
            partition_name=dept
        )
        print(f"  向 {dept} 分区插入 {len(result)} 条文档")
```

### 19.2.4 基于角色的权限检索

企业知识库的核心需求之一是权限控制。不同角色的用户只能看到自己有权限访问的文档：

```python
class User:
    """用户角色模拟"""
    def __init__(self, name: str, department: str, role: str):
        self.name = name
        self.department = department
        self.role = role  # "admin", "manager", "staff"

    def can_access(self, doc_department: str) -> bool:
        """权限判断：admin 看所有；manager 看本部门+HR；staff 只看本部门"""
        if self.role == "admin":
            return True
        if doc_department == "HR":
            return self.department == "HR" or self.role == "manager"
        return self.department == doc_department

def department_search(client, user, query, top_k=3):
    """基于用户权限的分区检索"""
    q_vec = generate_embedding(query)
    accessible = [d for d in DEPARTMENTS if user.can_access(d)]

    all_results = []
    for dept in accessible:
        results = client.search(
            collection_name=COLLECTION_NAME,
            data=[q_vec],
            limit=top_k,
            partition_names=[dept],
            output_fields=["title", "content", "department", "author"],
        )
        if results[0]:
            all_results.extend(results[0])

    # 按距离排序，取 TopK
    all_results.sort(key=lambda x: x["distance"])
    return all_results[:top_k]
```

这种分区级权限控制的优势在于：Milvus 的分区机制天然支持检索范围限定，无需在应用层对全量结果做二次过滤，检索性能更高、逻辑更清晰。

### 19.2.5 元数据过滤检索

除了向量相似度检索，企业知识库还经常需要按元数据字段精确过滤：

```python
def search_with_metadata_filter(
    client, query, department=None, author=None, top_k=5
):
    """带元数据过滤的检索"""
    q_vec = generate_embedding(query)

    filters = []
    if department:
        filters.append(f'department == "{department}"')
    if author:
        filters.append(f'author == "{author}"')

    filter_expr = " and ".join(filters) if filters else None

    kwargs = {
        "collection_name": COLLECTION_NAME,
        "data": [q_vec],
        "limit": top_k,
        "output_fields": ["title", "content", "department", "author"],
    }
    if filter_expr:
        kwargs["filter"] = filter_expr
        if department:
            kwargs["partition_names"] = [department]

    results = client.search(**kwargs)
    return results[0] if results else []
```

### 19.2.6 完整检索流程

```python
def main():
    client = MilvusClient(uri=MILVUS_URI)
    ensure_collection(client)
    insert_documents(client)

    # 1. 跨部门检索（管理员视图）
    query = "员工福利和培训"
    results = cross_department_search(client, query)
    for r in results:
        print(f"[{r['entity']['department']}] {r['entity']['title']}")

    # 2. 按部门检索
    query = "系统架构"
    results = search_with_metadata_filter(client, query, department="Tech")
    for r in results:
        print(f"{r['entity']['title']} (by {r['entity']['author']})")

    # 3. 文档元数据统计
    for dept in DEPARTMENTS:
        results = client.query(
            collection_name=COLLECTION_NAME,
            filter=f'department == "{dept}"',
            output_fields=["title", "author", "department"],
            limit=10,
        )
        print(f"{dept} ({len(results)} 篇文档):")
        for r in results:
            print(f"  - {r['title']} [作者: {r['author']}]")
```

## 19.3 效果评估

### 19.3.1 召回率评估

召回率（Recall）衡量的是系统在检索时能否找到所有相关的文档。在企业知识库场景中，我们使用以下测试集进行评估：

| 测试查询 | 预期结果 | 说明 |
|---------|---------|------|
| "员工福利待遇" | 薪资福利政策、考勤管理规定 | 跨部门HR文档 |
| "系统安全规范" | 安全运维手册、数据库管理规范 | 技术部文档 |
| "报销流程" | 报销管理制度、预算编制流程 | 财务部文档 |

使用 Recall@K 指标（Top-K 召回率）进行评估：

```python
def evaluate_recall(client, test_cases, top_k=3):
    """评估召回率"""
    total_recall = 0
    for case in test_cases:
        query = case["query"]
        expected = set(case["expected_titles"])
        results = cross_department_search(client, query, top_k=top_k)
        retrieved = set(r["entity"]["title"] for r in results)
        hits = len(retrieved & expected)
        recall = hits / len(expected) if expected else 0
        total_recall += recall
        print(f"查询: '{query}' → Recall@{top_k}: {recall:.2f}")
    avg_recall = total_recall / len(test_cases)
    print(f"平均 Recall@{top_k}: {avg_recall:.2f}")
    return avg_recall
```

### 19.3.2 准确率评估

准确率（Precision）衡量检索结果中相关文档的比例。在企业知识库中，准确率低意味着用户需要在大量无关结果中筛选，体验较差。

```python
def evaluate_precision(client, test_cases, top_k=3):
    """评估准确率"""
    total_precision = 0
    for case in test_cases:
        query = case["query"]
        expected = set(case["expected_titles"])
        results = cross_department_search(client, query, top_k=top_k)
        retrieved = set(r["entity"]["title"] for r in results)
        hits = len(retrieved & expected)
        precision = hits / top_k
        total_precision += precision
        print(f"查询: '{query}' → Precision@{top_k}: {precision:.2f}")
    avg_precision = total_precision / len(test_cases)
    print(f"平均 Precision@{top_k}: {avg_precision:.2f}")
    return avg_precision
```

### 19.3.3 问答质量评估

对于最终的大模型问答效果，可以从以下维度评估：

- **相关性**：答案是否直接回答了用户的问题
- **准确性**：答案中的事实信息是否正确
- **完整性**：答案是否覆盖了问题的所有方面
- **时效性**：答案是否基于最新的知识库内容

可以采用人工评分（1-5分）或使用 LLM-as-Judge 的方式自动化评估。

## 19.4 生产环境优化建议

### 19.4.1 索引选型优化

示例中使用的是 FLAT 索引（暴力检索），适用于小体量演示。生产环境中应根据数据量选择更高效的索引：

- **小于10万文档**：使用 IVF_FLAT（nlist=128）
- **10万-100万文档**：使用 IVF_SQ8（nlist=1024），压缩后内存占用降低75%
- **大于100万文档**：使用 HNSW（M=16, efConstruction=200），检索速度更快

### 19.4.2 权限控制的扩展

本文的分区权限控制适用于部门数量较少（<=64）的场景。如果部门数量超过64个，建议采用标量字段过滤而非分区，因为 Milvus 单集合的分区数量上限为4096，且大量分区会影响管理效率。

### 19.4.3 高可用部署

生产环境建议采用集群部署模式：
- 至少3个 QueryNode 节点承载检索请求
- 至少3个 DataNode 节点处理数据写入
- Etcd 集群保障元数据高可用
- MinIO 集群保障向量数据持久化

## 本章小结

本章完整实现了一个基于 Milvus 的企业知识库问答系统，涵盖了从需求分析、架构设计、数据建模、代码实现到效果评估的全流程。核心要点包括：

1. **分区机制实现数据隔离**：利用 Milvus 的分区特性，将不同部门的文档物理隔离，检索时限定分区范围，兼顾性能和安全性。
2. **角色权限控制**：在应用层实现 RBAC（基于角色的访问控制），根据用户角色动态计算可访问的分区列表。
3. **元数据过滤增强检索精准度**：结合向量检索和标量过滤，支持按部门、作者等字段精确筛选。
4. **效果评估体系**：建立召回率、准确率等量化指标，持续优化检索效果。

企业知识库是 RAG 技术最成熟、收益最明显的应用场景之一。通过 Milvus 的高效向量检索能力，可以显著提升企业内部知识的利用效率，降低员工查找信息的时间成本。
