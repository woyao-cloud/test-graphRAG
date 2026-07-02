# 数据处理流程文档

## 1. 完整数据流

```
Documents (PDF/MD/TXT/HTML)
  │
  ├─ Phase 1: 文档导入 (Ingest) ──────────────────────
  │   DocumentLoader → FormatParsers → DocumentConverter
  │   → ./input/*.txt
  │
  ├─ Phase 2: 知识图谱索引 (Index) ────────────────────
  │   ./input/*.txt
  │   → graphrag.index pipeline
  │     (load_input → chunk → extract_graph → finalize →
  │      summarize → community → embed → output)
  │   → ./output/*.parquet + ./output/lancedb/
  │
  ├─ Phase 3: Neo4j 同步 (Graph Sync) ────────────────
  │   ./output/*.parquet
  │   → Neo4jGraphSync (batch MERGE)
  │   → Neo4j Graph Database
  │
  └─ Phase 4: 查询 (Query) ──────────────────────────
      User question
      → QueryEngine
        ├─ Neo4j Cypher traversal (entity/relationship lookup)
        ├─ LanceDB vector search (semantic similarity)
        └─ Parquet fallback (community reports)
      → QueryResponse (answer + citations + graph paths)
```

---

## 2. 文档导入流程 (Ingest)

### 2.1 文件发现

```
DocumentLoader.discover()
  ├── 遍历 source_directories
  ├── 匹配 file_patterns (glob)
  ├── 按文件大小过滤 (max_file_size_mb)
  └── 返回文件路径列表
```

### 2.2 格式解析

```
ParserRegistry.parse(path)
  ├── .pdf  → PDFParser (pymupdf)
  │           ├── 逐页提取文本
  │           ├── 提取元数据 (title, author, pages)
  │           └── 清理文本 (去多余换行、修复连字符)
  ├── .md   → MarkdownParser
  │           ├── 读取文件 (编码自动检测)
  │           ├── 从 H1 提取标题
  │           └── 保留原始 markdown 文本
  ├── .txt  → TextParser
  │           ├── chardet 自动检测编码
  │           ├── 从首行提取标题
  │           └── 统计行数
  └── .html → HTMLParser (beautifulsoup4)
              ├── 解析 HTML 结构
              ├── 移除 script/style/nav/footer
              ├── 从 <title> 或 <h1> 提取标题
              └── 提取 meta description
```

### 2.3 格式转换

```
DocumentConverter.convert(doc)
  ├── 可选：剥离 markdown 格式
  ├── 可选：添加元数据头 (# Document:, # Source:, # Format:)
  ├── 清理文本 (CRLF→LF, 去控制字符)
  ├── 生成唯一文件名 (<原名>_<格式>.txt)
  └── 写入 input/ 目录
```

### 2.4 文件命名规则

```
<原文件名>_<格式>.txt
示例: manufacturer_catalog.md_md.txt
      hospital_procurement.txt_txt.txt
      distribution_contract.html_html.txt
```

---

## 3. 索引流程 (Index)

### 3.1 管道步骤

graphrag 3.x 的索引管道包含以下工作流（按执行顺序）：

```
1. load_input_documents    加载 input/*.txt 文件
2. create_base_text_units   文本分块 (chunking)
3. create_final_documents   创建最终文档表
4. extract_graph_nlp       NLP 实体提取 (fast 模式)
   或 extract_graph         LLM 实体提取 (standard 模式)
5. prune_graph             图剪枝 (去低频节点/边)
6. finalize_graph          图最终化 (合并、去重)
7. create_communities      社区检测 (Leiden 算法)
8. create_final_text_units 创建最终文本单元表
9. create_community_reports_text  社区报告生成 (LLM)
10. generate_text_embeddings      向量嵌入生成
```

### 3.2 分块策略 (Chunking)

```
配置参数:
  size: 1200 tokens (默认)
  overlap: 100 tokens
  encoding_model: cl100k_base

策略:
  - 基于 token 的分块 (非字符)
  - 相邻块有 100 token 重叠，避免信息断裂
  - 使用 cl100k_base 编码器 (与 GPT-4 兼容)
```

### 3.3 实体提取

**NLP 模式 (fast)**：
- 使用 spaCy `en_core_web_md` 模型
- 基于名词短语提取 (Noun Phrase Extraction)
- 正则表达式匹配命名实体
- 无需 LLM 调用，速度快但质量低

**LLM 模式 (standard)**：
- 使用配置的 Chat 模型 (如 DeepSeek、GPT-4)
- 从文本中提取实体名称、类型、描述
- 提取实体间关系及关系描述
- 支持多轮 gleaning (最多 2 轮)
- 质量高但成本高

### 3.4 图最终化 (Graph Finalization)

```
finalize_graph:
  ├── 合并同义实体 (基于名称相似度)
  ├── 去重关系
  ├── 计算节点度 (degree)
  ├── 计算边权重
  └── 输出最终图结构
```

### 3.5 社区检测

使用 **Leiden 算法** 进行层次化社区检测：

```
create_communities:
  ├── 输入: 最终图 (实体 + 关系)
  ├── 算法: Leiden 层次聚类
  ├── 参数: max_cluster_size = 10
  ├── 输出: 多级社区层次结构
  └── 社区报告: LLM 生成每个社区的摘要
```

Leiden 算法是 Louvain 算法的改进版，具有以下优势：
- 保证社区内部连接性
- 更快的收敛速度
- 避免产生 disconnected 社区

### 3.6 向量嵌入

```
generate_text_embeddings:
  ├── 嵌入对象: entity_description, community_full_content, text_unit_text
  ├── 模型: 配置的 Embedding 模型 (如 bge-m3, text-embedding-3-large)
  ├── 存储: LanceDB (本地向量数据库)
  └── 批量: 16 条/批, max 8192 tokens/批
```

---

## 4. Neo4j 同步流程 (Graph Sync)

### 4.1 数据映射

```
Parquet → Neo4j 映射:

entities.parquet        → (:Entity {id, name, type, description, degree, ...})
relationships.parquet   → [:RELATES_TO {id, description, weight, ...}]
communities.parquet     → (:Community {id, title, level, summary, ...})
documents.parquet       → (:Document {id, title, file_path, ...})
text_units.parquet      → (:TextUnit {id, text, document_id, ...})
```

### 4.2 同步策略

```
Neo4jGraphSync.sync_all():
  ├── 1. 连接 Neo4j (连接池)
  ├── 2. 可选: 创建索引和约束 (SchemaManager)
  ├── 3. 批量插入实体 (MERGE, 1000 条/批)
  ├── 4. 批量插入关系 (MERGE, 1000 条/批)
  ├── 5. 批量插入社区 (MERGE)
  ├── 6. 批量插入文档 (MERGE)
  ├── 7. 批量插入文本单元 (MERGE)
  └── 8. 关闭连接
```

使用 `MERGE` 语句保证幂等性 — 重复执行不会创建重复节点。

### 4.3 Neo4j 图模式

```
(:Entity {id, name, type, description, degree, community_ids, text_unit_ids})
    │
    ├──[:RELATES_TO {id, description, weight, text_unit_ids}]──→(:Entity)
    │
    ├──[:BELONGS_TO]──→(:Community {id, title, level, summary, full_content, rating})
    │
    ├──[:MENTIONED_IN]──→(:TextUnit {id, text, document_id, entity_ids, community_ids})
    │
    └──[:PART_OF]──→(:Document {id, title, file_path, text_unit_count})

(:Community)-[:PARENT_OF]→(:Community)  // 层次结构
(:TextUnit)-[:PART_OF]→(:Document)
```

---

## 5. 查询流程 (Query)

### 5.1 查询引擎架构

```
QueryEngine.ask(question, method)
  │
  ├── 1. 检查缓存 (SHA-256 key, TTL)
  │
  ├── 2. 自动路由 (method="auto")
  │     ├── 总结/趋势类 → global
  │     ├── 多跳/路径类 → drift
  │     ├── 简单事实类 → basic
  │     └── 默认 → local
  │
  ├── 3. 执行搜索
  │     ├── local:  graphrag.api.local_search
  │     ├── global: graphrag.api.global_search
  │     ├── drift:  graphrag.api.drift_search
  │     └── basic:  graphrag.api.basic_search
  │
  ├── 4. 构建响应
  │     ├── 提取引用来源
  │     ├── 格式化图上下文
  │     └── 组装 QueryResponse
  │
  └── 5. 缓存结果 + 返回
```

### 5.2 四种搜索方法对比

| 方法 | 数据源 | 适用场景 | 速度 | 图遍历 |
|---|---|---|---|---|
| **local** | 实体 + 文本 + 社区报告 | 具体事实查询 | 中 | 是 (Neo4j) |
| **global** | 社区报告 (map-reduce) | 总结/趋势 | 慢 | 否 |
| **drift** | 实体 + 文本 + 社区 (层次) | 多跳推理/供应链 | 慢 | 是 |
| **basic** | 文本单元 (向量搜索) | 简单事实 | 快 | 否 |

### 5.3 自动路由规则

```python
def _auto_route(question):
    # 全局搜索信号
    if any(s in question for s in ["overview", "summary", "trend", "theme"]):
        return "global"

    # DRIFT 搜索信号
    if any(s in question for s in ["how does", "chain", "path", "flow", "impact"]):
        return "drift"

    # 基础搜索信号
    if any(s in question for s in ["what is", "who is", "define"]):
        return "basic"

    # 默认: 本地搜索
    return "local"
```

### 5.4 图遍历操作

```
GraphTraversal:
  ├── ego_network(name, radius=2)
  │     └── 获取实体周围 N 跳内的所有节点和边
  ├── community_neighborhood(name)
  │     └── 获取实体所在社区及兄弟实体
  ├── find_all_paths_between(a, b, max_hops=5)
  │     └── 枚举两个实体间的所有路径
  ├── trace_flow(name, direction="downstream")
  │     └── 追踪供应链/信息流
  ├── find_intersection([a, b, ...])
  │     └── 查找与多个实体都相连的节点 (集合交集)
  └── impact_analysis(name)
        └── 分析实体移除的影响范围
```

---

## 6. 增量更新流程

```
IndexUpdater.update():
  ├── 1. 加载上次索引状态 (index_state.json)
  ├── 2. 检测新增/修改的文档
  ├── 3. 可选: 备份当前输出
  ├── 4. 运行增量索引管道 (standard-update)
  └── 5. 更新索引状态
```

---

## 7. REST API 端点

| 方法 | 路径 | 功能 | 请求体 |
|---|---|---|---|
| GET | `/health` | 健康检查 | — |
| GET | `/stats` | 系统统计 | — |
| POST | `/query` | 提交查询 | `{question, method, community_level}` |
| GET | `/index/status` | 索引状态 | — |
| POST | `/index` | 触发索引 | `{method}` |
| GET | `/graph/stats` | Neo4j 图统计 | — |
| POST | `/graph/sync` | 同步到 Neo4j | `{clear_first}` |
