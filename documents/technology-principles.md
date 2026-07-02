# 技术与原理文档

## 1. GraphRAG 原理

### 1.1 什么是 GraphRAG

GraphRAG (Graph-based Retrieval-Augmented Generation) 是 Microsoft 提出的一种增强型 RAG 技术。与传统 RAG 不同，GraphRAG 不仅检索相关文本片段，还构建知识图谱来理解文档间的实体关系和主题结构。

### 1.2 与传统 RAG 的对比

| 维度 | 传统 RAG | GraphRAG |
|---|---|---|
| 检索单元 | 文本块 (chunks) | 实体 + 关系 + 社区报告 |
| 上下文理解 | 局部相似度 | 全局图结构 |
| 多跳推理 | 弱 (依赖连续 chunk) | 强 (图遍历) |
| 总结能力 | 差 (局部视角) | 强 (社区报告 map-reduce) |
| 可解释性 | 文本引用 | 实体 + 关系 + 路径 |
| 索引成本 | 低 (仅 embedding) | 高 (LLM 提取 + 社区检测) |

### 1.3 GraphRAG 索引管道

```
输入文档
    │
    ▼
┌─────────────────────────────────────────────────┐
│ 1. 文本分块 (Chunking)                          │
│    将文档分割为固定大小的文本单元 (Text Units)    │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│ 2. 实体/关系提取 (Entity & Relationship Extraction)│
│    LLM 从每个文本单元中提取实体和关系             │
│    实体类型: organization, person, location, ... │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│ 3. 图最终化 (Graph Finalization)                 │
│    合并同义实体、去重关系、计算度中心性           │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│ 4. 社区检测 (Community Detection)                │
│    Leiden 层次聚类 → 多级社区结构                │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│ 5. 社区报告 (Community Summarization)             │
│    LLM 为每个社区生成摘要 (map-reduce)           │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│ 6. 向量嵌入 (Text Embedding)                     │
│    为实体描述、社区报告、文本单元生成向量         │
└─────────────────────────────────────────────────┘
```

### 1.4 GraphRAG 查询策略

**Local Search (本地搜索)**：
1. 对用户问题进行向量搜索，找到相关实体
2. 从找到的实体出发，沿关系扩展到邻居实体
3. 收集相关文本单元和社区报告
4. 将收集到的上下文注入 LLM 提示词
5. LLM 生成带引用的答案

**Global Search (全局搜索)**：
1. 对所有社区报告进行 map 操作 (LLM 评估相关性)
2. 对 map 结果进行 reduce 操作 (LLM 合并)
3. 生成全局视角的答案
4. 适合"这个数据集的主要主题是什么？"类问题

**DRIFT Search (层次化搜索)**：
1. 从用户问题出发，找到相关实体
2. 沿社区层次结构向上/下遍历
3. 在每个层次收集上下文
4. 综合多层次的社区信息生成答案
5. 适合"供应链是怎样的？"类多跳问题

**Basic Search (基础搜索)**：
1. 对用户问题进行向量搜索
2. 找到最相似的文本单元
3. 将文本单元注入 LLM 提示词
4. 生成答案
5. 最快但无图上下文

---

## 2. 图数据库技术 (Neo4j)

### 2.1 为什么选择 Neo4j

| 特性 | Neo4j | 关系型数据库 | 文档数据库 |
|---|---|---|---|
| 关系查询 | 原生 (指针跳转) | JOIN 开销大 | 不支持 |
| 多跳遍历 | O(1) 每跳 | O(log n) 每跳 | 不支持 |
| 路径查询 | 原生支持 | 递归 CTE | 不支持 |
| 图算法 | 内置 (PageRank, 社区检测) | 不支持 | 不支持 |

### 2.2 Cypher 查询语言

Cypher 是 Neo4j 的声明式图查询语言，其模式匹配语法与图结构自然对应：

```cypher
// 匹配模式: 实体 → 关系 → 实体
MATCH (恒瑞:Entity {name: '恒瑞医药'})-[:RELATES_TO]->(药品:Entity)
RETURN 药品.name, 药品.type

// 多跳遍历: 3 跳内的所有路径
MATCH path = (恒瑞:Entity {name: '恒瑞医药'})-[:RELATES_TO*1..3]-(connected)
RETURN path

// 最短路径
MATCH path = shortestPath(
  (a:Entity {name: '恒瑞医药'})-[:RELATES_TO*]-(b:Entity {name: '北京协和医院'})
)
RETURN path
```

### 2.3 图遍历算法

**广度优先搜索 (BFS)**：用于 ego network 扩展，从中心实体向外逐层遍历。

**最短路径 (Shortest Path)**：使用双向 BFS 查找两个实体间的最短连接路径。

**社区检测 (Leiden)**：在 Neo4j 中通过 APOC 插件或外部计算实现。

### 2.4 连接池管理

```
Neo4jConnection:
  ├── 最大连接数: 50 (可配置)
  ├── 连接生命周期: 3600 秒
  ├── 获取超时: 60 秒
  └── 自动重连: 失败时抛出 Neo4jConnectionError
```

---

## 3. 向量数据库技术 (LanceDB)

### 3.1 为什么选择 LanceDB

| 特性 | LanceDB | FAISS | Pinecone | Weaviate |
|---|---|---|---|---|
| 本地运行 | ✅ | ✅ | ❌ | ✅ |
| 持久化 | ✅ (磁盘) | ✅ (磁盘) | ✅ (云端) | ✅ (磁盘) |
| 增量插入 | ✅ | ❌ | ✅ | ✅ |
| 过滤 | ✅ | ❌ | ✅ | ✅ |
| 开源 | ✅ | ✅ | ❌ | ✅ |
| 零配置 | ✅ | ❌ | ❌ | ❌ |

### 3.2 向量嵌入原理

向量嵌入 (Embedding) 是将文本转换为固定维度的浮点数向量的过程。语义相似的文本在向量空间中距离更近。

```
"恒瑞医药" → [0.12, -0.34, 0.56, ..., 0.89]  (1024 维)
"齐鲁制药" → [0.11, -0.31, 0.52, ..., 0.85]  (距离近)
"北京协和" → [0.45, 0.23, -0.12, ..., -0.33]  (距离远)
```

### 3.3 向量搜索

```
VectorStore.search(query_vector, top_k=10):
  ├── 计算查询向量与所有存储向量的距离
  ├── 使用 L2 距离或余弦相似度
  ├── 返回 top_k 个最相似的记录
  └── 支持过滤表达式 (WHERE 子句)
```

### 3.4 支持的嵌入模型

| 模型 | 维度 | 用途 | 提供商 |
|---|---|---|---|
| bge-m3 | 1024 | 通用嵌入 | BAAI (本地) |
| nomic-embed-text | 768 | 通用嵌入 | Nomic (本地) |
| text-embedding-3-large | 3072 | 高质量嵌入 | OpenAI |
| text-embedding-3-small | 1536 | 快速嵌入 | OpenAI |

---

## 4. 社区检测算法 (Leiden)

### 4.1 算法原理

Leiden 算法是 Louvain 算法的改进版本，用于在图中发现社区结构。

**核心思想**：最大化模块度 (Modularity)，即社区内部连接密度高于随机期望。

**模块度公式**：
```
Q = (1/2m) * Σ[A_ij - (k_i*k_j/2m)] * δ(c_i, c_j)
```
其中：
- A_ij: 节点 i 和 j 之间的边权重
- k_i: 节点 i 的度
- m: 总边数
- δ(c_i, c_j): 如果 i 和 j 在同一社区则为 1

### 4.2 Leiden vs Louvain

| 特性 | Louvain | Leiden |
|---|---|---|
| 社区连接性 | 可能产生 disconnected 社区 | 保证连接性 |
| 收敛速度 | 慢 | 快 (2-10 倍) |
| 质量 | 好 | 更好 |
| 额外步骤 | 无 | 细化 (refinement) 步骤 |

### 4.3 层次化社区

Leiden 算法产生层次化的社区结构：

```
Level 0: 根社区 (整个图)
  ├── Level 1: 大社区
  │   ├── Level 2: 子社区
  │   │   ├── Level 3: 实体 A
  │   │   └── Level 3: 实体 B
  │   └── Level 2: 子社区
  └── Level 1: 大社区
```

这种层次结构支持不同粒度的查询 — 高层社区提供全局视角，低层社区提供局部细节。

---

## 5. LLM 集成 (LiteLLM)

### 5.1 LiteLLM 的作用

LiteLLM 是一个统一的 LLM API 适配层，提供一致的接口访问不同提供商：

```
graphrag → LiteLLM → OpenAI / DeepSeek / Ollama / Anthropic / ...
```

### 5.2 支持的提供商

| 提供商 | 模型示例 | 需要 API Key | 适用场景 |
|---|---|---|---|
| OpenAI | gpt-4.1, text-embedding-3-large | ✅ | 生产环境 |
| DeepSeek | deepseek-chat, deepseek-v4-flash | ✅ | 开发测试 (便宜) |
| Ollama | llama3.2, bge-m3 | ❌ | 本地开发 (免费) |
| Anthropic | claude-sonnet-5 | ✅ | 高质量回答 |
| Azure OpenAI | gpt-4o | ✅ | 企业部署 |

### 5.3 模型配置

```yaml
models:
  default_chat_model:
    type: litellm
    model_provider: openai
    model: deepseek-chat
    api_key: "${GRAPHRAG_API_KEY}"
    api_base: "https://api.deepseek.com/v1"
```

---

## 6. 文档解析技术

### 6.1 PDF 解析 (PyMuPDF)

PyMuPDF (fitz) 是一个高性能的 PDF 处理库：

- **文本提取**：逐页提取，保留段落结构
- **元数据提取**：标题、作者、页数
- **表格提取**：保留表格结构
- **清理**：去多余换行、修复连字符断行

### 6.2 HTML 解析 (BeautifulSoup4)

BeautifulSoup4 是一个 HTML/XML 解析库：

- **解析器**：使用 Python 内置的 html.parser
- **内容提取**：`get_text()` 提取可见文本
- **去噪**：自动移除 script、style、nav、footer 标签
- **元数据**：从 `<title>`、`<meta>` 标签提取

### 6.3 编码检测 (chardet)

chardet 自动检测文本文件的编码格式：

```
检测流程:
  1. 读取文件前 N 个字节
  2. 分析字节序列模式
  3. 返回最可能的编码 (UTF-8, GBK, Latin-1, ...)
  4. 用检测到的编码解码全文
```

---

## 7. 测试数据生成技术

### 7.1 Faker

Faker 是一个生成假数据的库，支持中文 locale：

```python
from faker import Faker
fake = Faker("zh_CN")
fake.company()    # "恒瑞医药"
fake.name()       # "张明华"
fake.city()       # "北京"
```

### 7.2 Jinja2 模板

Jinja2 是一个模板引擎，用于从实体数据渲染文档：

```jinja
# {{ by_type['pharmaceutical_company'][0].name }} 产品目录

| 药品名称 | 适应症 |
|---------|--------|
| {{ by_type['drug'][0].name }} | {{ by_type['indication'][0].name }} |
```

### 7.3 Ground Truth 验证

Ground truth 是已知正确答案的集合，用于验证系统输出：

```json
{
  "expected_entities": [
    {"name": "恒瑞医药", "type": "pharmaceutical_company"}
  ],
  "expected_relationships": [
    {"source": "恒瑞医药", "target": "注射用紫杉醇", "relation_type": "produces"}
  ],
  "test_queries": [
    {"question": "恒瑞医药生产哪些药品？", "expected_entities": ["恒瑞医药", "注射用紫杉醇"]}
  ]
}
```

评估指标：
- **Entity Recall**: ground truth 实体被提取出的比例
- **Entity Precision**: 提取的实体中正确匹配的比例
- **Relationship Recall**: ground truth 关系被提取出的比例
- **Query Pass Rate**: 预设问题中回答正确的比例
