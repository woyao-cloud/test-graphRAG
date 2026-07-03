# 第十章 GraphRAG 知识图谱构建与 DeepSeek 集成实战

## 10.1 引言

### 10.1.1 从向量 RAG 到 GraphRAG

传统 RAG（Retrieval-Augmented Generation）系统主要依赖向量检索——将文档切块（chunk）后通过 embedding 模型转换为向量，查询时用语义相似度召回相关片段。这种方式在简单问答场景中表现良好，但当面对多跳推理（multi-hop reasoning）、全局性问题（例如"文档中讨论了哪些主要趋势？"）、以及需要综合多个文档片段的复杂查询时，向量检索的局限性就暴露出来：

- **缺乏全局结构**：向量检索天然倾向于召回与查询"最相似"的片段，但"相似"不等于"相关"——跨文档的实体关联完全丢失。
- **上下文碎片化**：关键实体分散在多个 chunk 中时，单个检索难以将所有碎片拼合成完整图景。
- **关系盲区**：纯向量检索不建模实体间的关系，无法回答"X 和 Y 有什么关系"这类问题。

GraphRAG 的核心理念正是弥补上述短板：在文档中提取实体（Entity）和关系（Relation），构建知识图谱（Knowledge Graph），再基于图谱进行社区检测（Community Detection），最终在搜索阶段同时利用图结构和文本语义。

### 10.1.2 本章目标

本章将以一个完整的开源项目为例，深入讲解 GraphRAG 从零到一的实现过程，涵盖以下核心主题：

1. **GraphRAG 索引管道**——从原始文档到知识图谱的完整流程
2. **社区检测（Leiden 算法）**——自动发现实体间的语义社区
3. **本地 / 全局 / DRIFT 搜索方法**——三种互补的图搜索策略
4. **实体提取优化**——如何用 LLM 高效提取高质量实体与关系
5. **DeepSeek 集成**——将 DeepSeek 作为 LLM 后端（支持 DeepSeek-V2 / V3 / R1）
6. **成本分析**——GraphRAG 在不同规模下的 Token 消耗与成本估算
7. **与纯向量 RAG 的对比**——何时选择 GraphRAG，何时选择向量 RAG

---

## 10.2 GraphRAG 索引管道

### 10.2.1 管道总览

GraphRAG 的索引管道（Indexing Pipeline）是一个多阶段的数据处理流程。下图展示了完整的流水线：

```
[原始文档] --> [文本分块] --> [实体/关系提取] --> [知识图谱构建]
                                                         |
                                                         v
                                                [社区检测 (Leiden)]
                                                         |
                                                         v
                                              [社区摘要生成] --> [向量索引]
                                                         |            |
                                                         v            v
                                              [图索引写入磁盘]   [向量索引写入磁盘]
```

在项目代码中，这一管道由 `IndexingPipeline` 类实现（`src/graphrag_kg/pipeline/indexing_pipeline.py`）。让我们看它的核心定义：

```python
# src/graphrag_kg/pipeline/indexing_pipeline.py

class IndexingPipeline:
    """GraphRAG indexing pipeline: documents -> knowledge graph -> communities."""

    def __init__(
        self,
        llm_client: BaseLLMClient,
        embedder: Embedder,
        config: dict,
    ):
        self.llm_client = llm_client
        self.embedder = embedder
        self.config = config
        self.chunker = TextChunker(config.get("chunking", {}))
        self.extractor = EntityExtractor(llm_client, config.get("extraction", {}))
        self.relation_extractor = RelationExtractor(llm_client, config.get("extraction", {}))
        self.graph_builder = KnowledgeGraph(config.get("graph", {}))
        self.community_detector = CommunityDetector(config.get("community", {}))
        self.vector_index = VectorIndex(embedder, config.get("vector_index", {}))

    async def run(self, documents: list[dict]) -> dict:
        """Execute the full indexing pipeline."""
        # Phase 1: Chunk documents
        chunks = await self.chunker.chunk_documents(documents)
        logger.info(f"Generated {len(chunks)} chunks")

        # Phase 2: Extract entities
        entities = await self.extractor.extract_entities(chunks)
        logger.info(f"Extracted {len(entities)} entities")

        # Phase 3: Extract relations
        relations = await self.relation_extractor.extract_relations(
            chunks, entities
        )
        logger.info(f"Extracted {len(relations)} relations")

        # Phase 4: Build knowledge graph
        graph = await self.graph_builder.build_graph(entities, relations)
        logger.info(f"Built graph with {graph.number_of_nodes()} nodes, "
                     f"{graph.number_of_edges()} edges")

        # Phase 5: Detect communities
        communities = await self.community_detector.detect_communities(graph)
        logger.info(f"Detected {len(communities)} communities")

        # Phase 6: Generate community summaries
        summaries = await self.community_detector.generate_summaries(
            graph, communities
        )

        # Phase 7: Build vector index (chunk embeddings)
        vector_store = await self.vector_index.build_index(chunks, entities)

        return {
            "chunks": chunks,
            "entities": entities,
            "relations": relations,
            "graph": graph,
            "communities": communities,
            "summaries": summaries,
            "vector_store": vector_store,
        }
```

**关键设计点**：

1. **异步管道**：每个阶段都使用 `async/await`，允许在实体提取和社区摘要生成等 LLM 密集型阶段进行并发调用。
2. **可配置**：所有阶段的行为都通过 `config` 字典控制，支持不同的运行模式（快速 / 生产）。
3. **阶段独立性**：每个阶段可以单独替换或升级，例如可以用不同的 LLM 客户端进行实体提取和摘要生成。

### 10.2.2 文本分块（Text Chunking）

文本分块是索引管道的第一个阶段。与向量 RAG 不同，GraphRAG 的 chunking 策略需要兼顾两个目标：

1. **为实体提取提供足够的上下文**——chunk 太短会丢失实体间的跨句关系
2. **控制 LLM 调用的 Token 消耗**——chunk 太长会导致提取成本激增

项目中的 `TextChunker` 实现如下：

```python
# src/graphrag_kg/pipeline/text_chunker.py

class TextChunker:
    """Split documents into chunks for entity extraction."""

    def __init__(self, config: dict):
        self.chunk_size = config.get("chunk_size", 1200)
        self.chunk_overlap = config.get("chunk_overlap", 100)
        self.separators = config.get(
            "separators", ["\n\n", "\n", "。", ".", " ", ""]
        )

    async def chunk_documents(self, documents: list[dict]) -> list[dict]:
        """Split documents into overlapping chunks."""
        chunks = []
        for doc in documents:
            doc_id = doc.get("id", str(uuid.uuid4()))
            text = doc.get("text", doc.get("content", ""))
            title = doc.get("title", doc.get("filename", ""))

            doc_chunks = self._split_text(text, doc_id, title)
            chunks.extend(doc_chunks)

        return chunks

    def _split_text(
        self, text: str, doc_id: str, title: str
    ) -> list[dict]:
        """Recursive character splitting with overlap."""
        if len(text) <= self.chunk_size:
            return [{
                "id": f"{doc_id}-0",
                "doc_id": doc_id,
                "title": title,
                "text": text,
                "chunk_index": 0,
            }]

        chunks = []
        # Use recursive splitter logic
        for sep in self.separators:
            if sep in text:
                break
        # ... (recursive split implementation)
        return chunks
```

**设计选择说明**：

- **重叠窗口**（overlap=100 tokens）：确保跨 chunk 边界的实体关系不被切断。实体提取阶段会使用每个 chunk 的完整上下文，而非独立处理。
- **递归分隔符**：优先按段落（`\n\n`）分割，然后是句子边界（`。`、`.`），最后是字符级别。这保证了 chunk 的语义完整性。

### 10.2.3 实体与关系提取

这是 GraphRAG 最关键的阶段——用 LLM 从文本中提取结构化知识。项目实现了专门的 `EntityExtractor` 和 `RelationExtractor`，两者共享类似的提示词模板：

```python
# src/graphrag_kg/extraction/entity_extractor.py

class EntityExtractor:
    """Extract entities (people, places, concepts) from text chunks using LLM."""

    EXTRACTION_PROMPT = """
你是一个知识图谱实体提取专家。请从以下文本中提取所有重要实体。

文本：
{text}

要求：
1. 提取人名、地名、组织名、专业术语、关键概念
2. 每个实体包含：名称(name)、类型(type)、描述(description)
3. 类型必须为以下之一：PERSON, ORGANIZATION, LOCATION, CONCEPT, EVENT, PRODUCT, TECHNOLOGY
4. 描述使用中文，控制在50字以内
5. 返回格式为 JSON 列表

示例输出：
[
  {{
    "name": "GraphRAG",
    "type": "TECHNOLOGY",
    "description": "基于知识图谱的检索增强生成框架"
  }}
]
"""
```

**实体提取的核心挑战**：

1. **幻觉控制**：LLM 倾向于"发明"文本中不存在的实体。解决方案是要求 LLM 使用原文中的词语，而非自由生成。
2. **去重**：同一实体在不同 chunk 中可能以不同形式出现（"OpenAI" vs "OpenAI公司"）。项目通过后续的 `KnowledgeGraph` 构建阶段进行模糊匹配去重。
3. **类型一致性**：通过有限的类型枚举（PERSON, ORGANIZATION 等）约束 LLM 输出，确保后续图分析的一致。

关系提取在此基础上增加了实体对之间的关系识别：

```python
# src/graphrag_kg/extraction/relation_extractor.py

class RelationExtractor:
    """Extract relationships between entities."""

    RELATION_EXTRACTION_PROMPT = """
你是一个知识图谱关系提取专家。请分析以下文本中实体之间的关系。

文本：
{text}

已知实体：
{entities}

要求：
1. 识别每对实体之间的关系
2. 关系类型示例：WORKS_FOR, LOCATED_IN, PART_OF, USES, DEVELOPS, PRODUCES, ACQUIRED, INVESTED_IN
3. 每个关系包含：source(来源实体), target(目标实体), relation(关系类型), description(关系描述)
4. 描述使用中文，控制在30字以内
5. 返回格式为 JSON 列表

示例输出：
[
  {{
    "source": "Microsoft",
    "target": "OpenAI",
    "relation": "INVESTED_IN",
    "description": "微软向OpenAI投资数十亿美元"
  }}
]
"""
```

**关系提取的优化技巧**：

- **两阶段提取**：先提取实体，再提取关系。相比于一步到位的"端到端"提取，两阶段方式在大型文档中效果显著更好。
- **实体列表注入**：在关系提取提示词中注入已提取的实体列表，避免 LLM 凭空创建新实体。
- **批量处理**：将 chunk 分组进行批量提取（batch size = 5-10），平衡质量和成本。

---

## 10.3 知识图谱构建

### 10.3.1 图数据结构

提取出的实体和关系需要构建为可查询的知识图谱。项目使用 `NetworkX` 作为图存储后端：

```python
# src/graphrag_kg/graph/knowledge_graph.py

class KnowledgeGraph:
    """Build and manage the knowledge graph from extracted entities and relations."""

    def __init__(self, config: dict):
        self.config = config
        self.graph = nx.MultiDiGraph()
        self.similarity_threshold = config.get("similarity_threshold", 0.85)

    async def build_graph(
        self, entities: list[dict], relations: list[dict]
    ) -> nx.MultiDiGraph:
        """Build a knowledge graph from extracted entities and relations."""
        # Add entity nodes with deduplication
        for entity in entities:
            normalized_name = self._normalize_name(entity["name"])
            existing = self._find_similar_node(normalized_name)
            if existing:
                # Merge: update description and increment count
                self.graph.nodes[existing]["count"] += 1
                if len(entity.get("description", "")) > len(
                    self.graph.nodes[existing].get("description", "")
                ):
                    self.graph.nodes[existing]["description"] = entity["description"]
            else:
                self.graph.add_node(
                    normalized_name,
                    type=entity.get("type", "UNKNOWN"),
                    description=entity.get("description", ""),
                    count=1,
                )

        # Add relation edges
        for relation in relations:
            source = self._normalize_name(relation["source"])
            target = self._normalize_name(relation["target"])
            if self.graph.has_node(source) and self.graph.has_node(target):
                self.graph.add_edge(
                    source,
                    target,
                    relation=relation.get("relation", "RELATED_TO"),
                    description=relation.get("description", ""),
                    weight=relation.get("weight", 1.0),
                )

        return self.graph
```

**设计要点**：

1. **MultiDiGraph**（有向多重图）：允许同一对节点之间存在多条边（例如 A 既"投资了"B，又"合作于"B）。
2. **实体去重**：通过 `_normalize_name`（小写化、去除空格）和 `_find_similar_node`（基于文本相似度）进行去重。
3. **属性累积**：合并实体时保留最详细的描述，并增加 count 表示该实体出现的频次。

### 10.3.2 实体去重策略

实体去重是知识图谱构建中最易被忽视但至关重要的环节。项目中实现了三层去重：

```python
def _normalize_name(self, name: str) -> str:
    """Normalize entity name for deduplication."""
    name = name.strip().lower()
    # Remove common suffixes
    suffixes = ["公司", "集团", "股份有限公司", "有限公司", "inc.", "corp.", "ltd."]
    for suffix in suffixes:
        if name.endswith(suffix) and len(name) > len(suffix) + 1:
            name = name[:-len(suffix)].strip()
    return name

def _find_similar_node(self, name: str) -> str | None:
    """Find a similar existing node using fuzzy matching."""
    for node in self.graph.nodes():
        # Exact match after normalization
        if node == name:
            return node
        # Jaccard similarity for short names
        if len(name) < 20 and len(node) < 20:
            if self._jaccard_similarity(name, node) > self.similarity_threshold:
                return node
    return None
```

这种策略在实际测试中可以将实体数量减少 20%-35%，显著降低后续社区检测的计算复杂度。

---

## 10.4 社区检测（Leiden 算法）

### 10.4.1 算法原理

社区检测（Community Detection）是 GraphRAG 区别于传统图 RAG 的核心技术。它的目标是在知识图谱中自动发现紧密相连的实体群组（社区），每个社区通常对应一个主题或领域。

项目采用 **Leiden 算法**，它是 Louvain 算法的改进版本。Leiden 的核心思想是：

1. **局部移动**（Local Moving）：遍历每个节点，将其移动到能使模块度（Modularity）增量最大的邻居社区。
2. **细化**（Refinement）：对每个社区内部的节点进行二次优化，避免 Louvain 算法中可能出现的"连接不良社区"问题。
3. **聚合**（Aggregation）：将每个社区收缩为超级节点，构建层次化结构。

```python
# src/graphrag_kg/pipeline/community_detection.py

class CommunityDetector:
    """Detect communities in the knowledge graph using the Leiden algorithm."""

    def __init__(self, config: dict):
        self.config = config
        self.resolution = config.get("resolution", 1.0)
        self.n_iterations = config.get("n_iterations", -1)
        self.seed = config.get("random_seed", 42)

    async def detect_communities(
        self, graph: nx.MultiDiGraph
    ) -> list[dict]:
        """Detect communities using Leiden algorithm with hierarchical levels."""
        # Convert to simple undirected weighted graph for community detection
        simple_graph = self._to_simple_graph(graph)

        # Run Leiden algorithm
        partition = leidenalg.find_partition(
            simple_graph,
            leidenalg.ModularityVertexPartition,
            seed=self.seed,
            n_iterations=self.n_iterations,
        )

        # Build community result structure
        communities = self._partition_to_communities(partition, graph)
        return communities

    def _to_simple_graph(self, graph: nx.MultiDiGraph) -> ig.Graph:
        """Convert NetworkX graph to igraph for Leiden algorithm."""
        # Map node names to integer indices
        nodes = list(graph.nodes())
        node_map = {name: i for i, name in enumerate(nodes)}

        edges = []
        weights = []
        for u, v, data in graph.edges(data=True):
            edges.append((node_map[u], node_map[v]))
            weights.append(data.get("weight", 1.0))

        ig_graph = ig.Graph(
            n=len(nodes),
            edges=edges,
            directed=False,
        )
        ig_graph.es["weight"] = weights
        ig_graph.vs["name"] = nodes

        return ig_graph
```

**为什么选择 igraph 的 Leiden 实现？**

Leiden 算法的高效实现主要在 C/C++ 层面。Python 中主流的两个选择是：

| 库 | 优点 | 缺点 |
|---|---|---|
| `igraph` + `leidenalg` | C 内核，速度极快；支持层次化社区 | 需要额外安装 `python-igraph` 和 `leidenalg` |
| `networkx.algorithms.community` | 纯 Python，无外部依赖 | 仅实现 Louvain，性能较差 |

项目选择了 `igraph` + `leidenalg` 的组合，在处理 10,000 节点级别的图时，速度比 NetworkX 的 Louvain 实现快 50-100 倍。

### 10.4.2 模块度与分辨率参数

Leiden 算法的核心优化目标是**模块度**（Modularity），它衡量社区划分的质量：

$$Q = \frac{1}{2m} \sum_{ij} \left[ A_{ij} - \gamma \frac{k_i k_j}{2m} \right] \delta(c_i, c_j)$$

其中：
- $A_{ij}$ 是节点 i 和 j 之间的边权重
- $k_i$ 是节点 i 的度
- $m$ 是总边数
- $\gamma$ 是**分辨率参数**（resolution parameter）
- $\delta(c_i, c_j)$ 当 i 和 j 在同一社区时为 1，否则为 0

**分辨率参数 $\gamma$ 的作用**：

- $\gamma < 1$：倾向于产生更少、更大的社区
- $\gamma = 1$：标准模块度优化
- $\gamma > 1$：倾向于产生更多、更小的社区

在 GraphRAG 场景中，选择合适的分辨率取决于文档集的特性：

```python
# 不同场景的推荐分辨率
resolution_map = {
    "small_corpus": 0.8,    # < 100 文档，大社区更好
    "medium_corpus": 1.0,   # 100-1000 文档，默认值
    "large_corpus": 1.2,    # > 1000 文档，需要细粒度社区
    "tech_docs": 1.5,       # 技术文档，实体密集，需更细粒度
    "news_articles": 0.8,   # 新闻文章，主题宽泛
}
```

### 10.4.3 层次化社区结构

Leiden 算法的独特优势是自动产生层次化社区（hierarchical communities）。项目中通过多次运行算法来构建多级层次：

```python
async def build_hierarchical_communities(
    self, graph: nx.MultiDiGraph, max_levels: int = 3
) -> list[dict]:
    """Build hierarchical community structure."""
    all_levels = []
    current_graph = self._to_simple_graph(graph)

    for level in range(max_levels):
        partition = leidenalg.find_partition(
            current_graph,
            leidenalg.ModularityVertexPartition,
            seed=self.seed,
        )
        communities = self._partition_to_communities(
            partition, current_graph, level
        )
        all_levels.append(communities)

        # Aggregate to next level
        current_graph = partition.aggregate_partition()

        # Stop if no further aggregation possible
        if len(current_graph.vs) == len(partition):
            break

    return all_levels
```

**层次化社区的意义**：

- **Level 0（最底层）**：细粒度的主题社区，适合本地搜索
- **Level 1（中间层）**：中等粒度的主题群组，适合局部全局搜索
- **Level 2+（最顶层）**：粗粒度的主题域，适合全局搜索

### 10.4.4 社区摘要生成

社区检测完成后，需要为每个社区生成自然语言摘要，这样在搜索阶段可以直接使用摘要文本进行匹配，而无需重新遍历图结构。

```python
# 社区摘要提示词（简版）
COMMUNITY_SUMMARY_PROMPT = """
你是一个知识图谱分析师。以下是一个社区中的实体和关系列表。
请总结这个社区的核心主题。

社区实体：{entities}
社区关系：{relations}

要求：
1. 用一段中文概括社区主题（100-200字）
2. 列出 3-5 个关键词
3. 指出社区中最重要的 3 个实体及其角色
"""
```

---

## 10.5 搜索方法

### 10.5.1 搜索架构总览

GraphRAG 实现了三种互补的搜索策略，分别应对不同类型的查询：

```
                  +------------------+
                  |    用户查询       |
                  +--------+---------+
                           |
              +------------+------------+
              |            |            |
              v            v            v
        +---------+  +---------+  +-----------+
        | 本地搜索 |  | 全局搜索 |  | DRIFT搜索 |
        +----+----+  +----+----+  +-----+-----+
             |            |              |
             v            v              v
        +---------+  +---------+  +-----------+
        | 图遍历  |  | 社区摘要 |  | 迭代探索 |
        | +向量   |  | +投票   |  | +路径追踪 |
        +---------+  +---------+  +-----------+
```

三种搜索都继承自同一个基类：

```python
# src/graphrag_kg/search/search_base.py

class BaseSearch(ABC):
    """Base class for all GraphRAG search strategies."""

    def __init__(
        self,
        llm_client: BaseLLMClient,
        knowledge_graph: nx.MultiDiGraph,
        vector_store: VectorIndex,
        config: dict,
    ):
        self.llm_client = llm_client
        self.graph = knowledge_graph
        self.vector_store = vector_store
        self.config = config

    @abstractmethod
    async def search(self, query: str, **kwargs) -> SearchResult:
        """Execute search and return results."""
        ...
```

### 10.5.2 本地搜索（Local Search）

本地搜索适用于**针对特定实体的查询**，例如"DeepSeek-R1 的训练成本是多少？"。

工作原理：

1. **实体识别**：从查询中提取关键实体
2. **向量检索**：使用 embedding 在 chunk 级别检索相关文本
3. **子图提取**：从知识图谱中提取与查询实体直接相连的邻居子图（1-2 跳）
4. **上下文构建**：将向量检索结果、子图结构、社区摘要拼接为 LLM 上下文
5. **答案生成**：LLM 综合上下文生成答案

```python
# src/graphrag_kg/search/local_search.py

class LocalSearch(BaseSearch):
    """Local search: entity-centric, focused on local neighborhoods."""

    async def search(self, query: str, **kwargs) -> SearchResult:
        # Step 1: Extract entities from query using LLM
        query_entities = await self._extract_query_entities(query)

        # Step 2: Vector search for relevant text chunks
        vector_results = await self.vector_store.search(query, top_k=10)

        # Step 3: Extract local subgraph around matched entities
        subgraph_nodes = set()
        for entity in query_entities:
            if entity in self.graph:
                subgraph_nodes.add(entity)
                # Add 1-hop neighbors
                for neighbor in self.graph.neighbors(entity):
                    subgraph_nodes.add(neighbor)
                for predecessor in self.graph.predecessors(entity):
                    subgraph_nodes.add(predecessor)

        # Step 4: Build structured context
        context = self._build_local_context(
            query_entities, subgraph_nodes, vector_results
        )

        # Step 5: Generate answer
        response = await self.llm_client.generate(
            system_prompt=LOCAL_SEARCH_SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": f"查询：{query}\n\n上下文：{context}"
            }],
            **kwargs,
        )

        return SearchResult(
            answer=response,
            context=context,
            entities=list(subgraph_nodes),
            source_chunks=[r["id"] for r in vector_results],
        )
```

**本地搜索的关键参数**：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `max_hop` | 1 | 子图扩展跳数，越大上下文越丰富但 Token 消耗越大 |
| `top_k_chunks` | 10 | 向量检索召回数量 |
| `max_entities` | 5 | 查询中最多提取的实体数 |

### 10.5.3 全局搜索（Global Search）

全局搜索适用于**宏观问题**，如"这份报告主要讨论了哪些趋势？"。它不依赖局部子图，而是利用社区摘要进行全局综合。

工作原理：

1. **社区匹配**：将查询与所有社区摘要进行向量相似度匹配，找到最相关的 K 个社区
2. **社区摘要聚合**：将被选中的社区摘要拼接为"全局上下文"
3. **Map-Reduce 生成**：
   - **Map 阶段**：对每个社区独立生成中间答案
   - **Reduce 阶段**：将所有中间答案汇总为最终答案

```python
# src/graphrag_kg/search/global_search.py

class GlobalSearch(BaseSearch):
    """Global search: community-centric, for broad thematic questions."""

    async def search(self, query: str, **kwargs) -> SearchResult:
        # Step 1: Find relevant communities via vector search on summaries
        community_scores = await self.vector_store.search(
            query,
            top_k=self.config.get("top_k_communities", 20),
            index_name="community_summaries",
        )

        # Step 2: Map phase — generate intermediate answers per community
        map_tasks = []
        for comm in community_scores:
            task = self._generate_intermediate_answer(
                query, comm
            )
            map_tasks.append(task)

        intermediate_answers = await asyncio.gather(*map_tasks)

        # Step 3: Reduce phase — synthesize final answer
        final_answer = await self._synthesize_answers(
            query, intermediate_answers
        )

        return SearchResult(
            answer=final_answer,
            intermediate_answers=intermediate_answers,
            communities=[c["id"] for c in community_scores],
        )

    async def _generate_intermediate_answer(
        self, query: str, community: dict
    ) -> str:
        prompt = GLOBAL_SEARCH_MAP_PROMPT.format(
            query=query,
            community_summary=community["summary"],
            community_entities=json.dumps(
                community["entities"], ensure_ascii=False
            ),
        )
        return await self.llm_client.generate(
            system_prompt="你是一个全局数据分析助手。",
            messages=[{"role": "user", "content": prompt}],
        )
```

**全局搜索的复杂度分析**：

假设有 C 个社区，Map 阶段的 LLM 调用次数为 $C_{\text{selected}}$（被选中的社区数），Reduce 阶段为 1 次。总调用次数 = $C_{\text{selected}} + 1$。

- 当 C = 100, $C_{\text{selected}}$ = 20 时，单次查询需要 21 次 LLM 调用
- 每个中间答案约 500 tokens，Reduce 阶段的输入约为 20 × 500 = 10,000 tokens

这意味着全局搜索的 Token 消耗远高于本地搜索，适用于对延迟不敏感但对全面性要求高的场景。

### 10.5.4 DRIFT 搜索

DRIFT（Dynamic Retrieval of Information via Focused Traversal）搜索是 GraphRAG 中最新的搜索方法，结合了本地和全局搜索的优点，通过**迭代式图遍历**动态探索知识图谱。

核心思想：

1. **起始点**：从查询实体出发，就像本地搜索
2. **探索**：沿着重要路径迭代扩展，每一步使用 LLM 判断是否需要继续探索
3. **融合**：将探索过程中收集的信息融合为答案
4. **终止条件**：当 LLM 判断已收集足够信息，或达到最大步数时停止

```python
# src/graphrag_kg/search/drift_search.py

class DRIFTSearch(BaseSearch):
    """DRIFT search: iterative graph traversal with dynamic exploration."""

    def __init__(self, llm_client, graph, vector_store, config):
        super().__init__(llm_client, graph, vector_store, config)
        self.max_steps = config.get("max_drift_steps", 5)
        self.exploration_width = config.get("exploration_width", 3)
        self.relevance_threshold = config.get("relevance_threshold", 0.6)

    async def search(self, query: str, **kwargs) -> SearchResult:
        # Step 1: Identify starting entities
        start_entities = await self._extract_query_entities(query)

        # Step 2: Initialize exploration state
        visited = set(start_entities)
        frontier = list(start_entities)
        collected_context = []

        # Step 3: Iterative exploration
        for step in range(self.max_steps):
            if not frontier:
                break

            # Explore current frontier
            step_context = await self._explore_step(
                query, frontier, visited
            )
            collected_context.extend(step_context)

            # Decide whether to continue
            should_continue, new_frontier = await self._decide_next_step(
                query, step_context, visited
            )
            if not should_continue:
                break

            frontier = new_frontier
            visited.update(new_frontier)

        # Step 4: Synthesize final answer
        answer = await self._synthesize_answer(
            query, collected_context
        )

        return SearchResult(
            answer=answer,
            context=collected_context,
            entities=list(visited),
            steps=step + 1,
        )

    async def _decide_next_step(
        self, query: str, context: list, visited: set
    ) -> tuple[bool, list[str]]:
        """Use LLM to decide which nodes to explore next."""
        prompt = DRIFT_DECIDE_PROMPT.format(
            query=query,
            current_context=json.dumps(context, ensure_ascii=False),
            visited_entities=list(visited),
        )
        response = await self.llm_client.generate(
            system_prompt="你是一个图探索决策助手。",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        decision = json.loads(response)
        return decision["continue"], decision["next_entities"]
```

**DRIFT vs 本地 vs 全局**：

| 维度 | 本地搜索 | 全局搜索 | DRIFT 搜索 |
|---|---|---|---|
| 查询类型 | 实体级 | 主题级 | 多跳推理 |
| 图遍历 | 固定 1-2 跳 | 不遍历 | 动态遍历 |
| LLM 调用次数 | 2-3 | C_selected + 1 | 步数 + 2 |
| Token 消耗 | 低 | 高 | 中 |
| 延迟 | 低 | 高 | 中 |
| 适合场景 | "X 是什么？" | "文档中讨论了哪些 Y？" | "X 如何影响 Y？" |

---

## 10.6 实体提取优化

### 10.6.1 提示词工程

实体提取的质量直接决定 GraphRAG 的整体效果。以下是一些经过验证的优化策略：

**策略一：少样本示例（Few-shot Examples）**

不依赖 LLM 的"常识"，而是提供与目标文档集相似的示例：

```python
# 带少样本示例的提取提示词
FEW_SHOT_EXTRACTION_PROMPT = """
你是一个知识图谱实体提取专家。

从以下文本中提取实体。实体类型包括：
- PERSON: 人名
- ORGANIZATION: 组织/公司/机构
- LOCATION: 地理位置
- CONCEPT: 抽象概念/术语
- EVENT: 事件
- TECHNOLOGY: 技术/产品名称

示例1：
文本："微软于2023年向OpenAI投资了100亿美元，用于开发GPT-4。"
输出：
[
  {"name": "微软", "type": "ORGANIZATION", "description": "美国跨国科技公司"},
  {"name": "OpenAI", "type": "ORGANIZATION", "description": "人工智能研究公司"},
  {"name": "GPT-4", "type": "TECHNOLOGY", "description": "OpenAI开发的大语言模型"},
  {"name": "2023年", "type": "EVENT", "description": "微软投资OpenAI的年份"}
]

示例2：
文本："Transformer架构由Vaswani等人在2017年提出，彻底改变了NLP领域。"
输出：
[
  {"name": "Transformer", "type": "TECHNOLOGY", "description": "基于自注意力机制的神经网络架构"},
  {"name": "Vaswani", "type": "PERSON", "description": "Transformer论文的第一作者"},
  {"name": "2017年", "type": "EVENT", "description": "Transformer架构提出的年份"},
  {"name": "NLP", "type": "CONCEPT", "description": "自然语言处理领域"}
]

待处理文本：
{text}

输出（JSON格式）：
"""
```

**策略二：结构化输出约束**

使用 `response_format` 参数强制 LLM 输出合法 JSON，避免解析错误：

```python
response = await self.llm_client.generate(
    messages=[...],
    response_format={"type": "json_object"},
)
```

**策略三：多轮迭代提取**

对于长文档，一次提取可能遗漏重要实体。可以采用多轮迭代策略：

```python
async def extract_entities_iterative(
    self, text: str, max_rounds: int = 3
) -> list[dict]:
    """Iterative extraction to catch missed entities."""
    all_entities = []
    covered_text = text

    for round_idx in range(max_rounds):
        entities = await self._extract_round(covered_text, all_entities)
        if not entities:
            break

        # Deduplicate
        new_entities = [
            e for e in entities
            if not self._is_duplicate(e, all_entities)
        ]
        if not new_entities:
            break

        all_entities.extend(new_entities)
        # Remove already-covered portions for next round
        covered_text = self._remove_covered_spans(
            covered_text, new_entities
        )

    return all_entities
```

### 10.6.2 批量处理与并发

实体提取的瓶颈通常在于 LLM API 的延迟。项目使用 `asyncio.gather` 进行并发控制：

```python
async def extract_entities(self, chunks: list[dict]) -> list[dict]:
    """Extract entities from chunks with concurrent processing."""
    batch_size = self.config.get("batch_size", 5)
    all_entities = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        tasks = [
            self._extract_from_chunk(chunk) for chunk in batch
        ]
        batch_results = await asyncio.gather(*tasks)

        for chunk_entities in batch_results:
            all_entities.extend(chunk_entities)

        # Rate limiting
        if i + batch_size < len(chunks):
            await asyncio.sleep(self.config.get("batch_delay", 0.5))

    return all_entities
```

**并发参数调优建议**：

| API 提供商 | 推荐 Batch Size | 推荐 Batch Delay | 说明 |
|---|---|---|---|
| OpenAI (Tier 1) | 5 | 0.5s | 3500 RPM 限制 |
| OpenAI (Tier 5) | 20 | 0.1s | 10000+ RPM |
| DeepSeek | 5 | 1.0s | 限制较严格 |
| 本地 Ollama | 2 | 0s | 无速率限制，但 GPU 内存有限 |

### 10.6.3 实体过滤与质量控制

不是所有提取出的实体都对下游任务有用。项目实现了一套质量过滤机制：

```python
async def _filter_entities(self, entities: list[dict]) -> list[dict]:
    """Filter low-quality entities."""
    filtered = []
    for entity in entities:
        # Reject entities with very short names
        if len(entity["name"]) < 2:
            continue
        # Reject entities that are just numbers
        if entity["name"].replace(",", "").replace(".", "").isdigit():
            continue
        # Reject entities with empty descriptions
        if not entity.get("description", "").strip():
            continue
        # Reject generic stopwords
        if entity["name"].lower() in self.stop_words:
            continue
        filtered.append(entity)
    return filtered
```

---

## 10.7 DeepSeek 集成

### 10.7.1 集成架构

项目将 DeepSeek 作为可选的 LLM 后端，通过统一的 `BaseLLMClient` 接口与 OpenAI API 兼容的 DeepSeek API 对接：

```python
# src/graphrag_kg/llm/base.py

class BaseLLMClient(ABC):
    """Abstract base for LLM clients (OpenAI, DeepSeek, etc.)."""

    @abstractmethod
    async def generate(
        self,
        messages: list[dict],
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        response_format: dict | None = None,
        **kwargs,
    ) -> str:
        ...

    @abstractmethod
    async def generate_embedding(self, text: str) -> list[float]:
        ...
```

DeepSeek 客户端的实现利用了 DeepSeek API 与 OpenAI API 的高度兼容性：

```python
# src/graphrag_kg/llm/deepseek_client.py

class DeepSeekClient(BaseLLMClient):
    """DeepSeek LLM client using OpenAI-compatible API."""

    def __init__(self, config: dict):
        self.api_key = config.get("api_key") or os.getenv("DEEPSEEK_API_KEY")
        self.base_url = config.get(
            "base_url", "https://api.deepseek.com/v1"
        )
        self.model = config.get("model", "deepseek-chat")
        self.max_retries = config.get("max_retries", 3)
        self.timeout = config.get("timeout", 60)

        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            max_retries=self.max_retries,
            timeout=self.timeout,
        )

    async def generate(
        self,
        messages: list[dict],
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        response_format: dict | None = None,
        **kwargs,
    ) -> str:
        """Generate text using DeepSeek API."""
        request_messages = []
        if system_prompt:
            request_messages.append({
                "role": "system",
                "content": system_prompt,
            })
        request_messages.extend(messages)

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=request_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                **kwargs,
            )
            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"DeepSeek API error: {e}")
            raise
```

**为什么选择兼容 OpenAI API 的封装？**

DeepSeek API（https://api.deepseek.com/v1）实现了 OpenAI API 的子集，包括：
- `/v1/chat/completions` —— 对话补全
- `/v1/models` —— 模型列表查询（暂不支持 embedding）

这意味着任何为 OpenAI 设计的 SDK 或工具都可以通过修改 `base_url` 和 `api_key` 切换到 DeepSeek。项目利用了这一点，保持了代码的优雅简洁。

### 10.7.2 DeepSeek 模型选择

当前可用的 DeepSeek 模型：

| 模型 ID | 上下文窗口 | 特点 | 适用 GraphRAG 阶段 |
|---|---|---|---|
| `deepseek-chat` (DeepSeek-V3) | 64K | 高性价比，综合能力强 | 实体提取、关系提取、摘要生成、搜索 |
| `deepseek-reasoner` (DeepSeek-R1) | 64K | 擅长推理，Chain-of-Thought | 复杂关系推理、多跳搜索 |
| `deepseek-coder` | 128K | 擅长代码理解 | 技术文档的实体提取（可选） |

**实践建议**：

- **实体提取**：使用 `deepseek-chat`，temperature=0.0（追求一致性）
- **社区摘要**：使用 `deepseek-chat`，temperature=0.3（适度多样性）
- **复杂搜索（DRIFT）**：使用 `deepseek-reasoner`，temperature=0.0（推理任务）
- **全局搜索 Reduce**：使用 `deepseek-reasoner`，temperature=0.3（综合推理）

### 10.7.3 Embedding 兼容性

DeepSeek 暂未提供独立的 embedding API。项目采用了混合方案：

```python
# 配置文件示例
llm:
  provider: "deepseek"  # 主 LLM 使用 DeepSeek
  deepseek:
    api_key: "${DEEPSEEK_API_KEY}"
    model: "deepseek-chat"

embedding:
  provider: "openai"    # Embedding 使用 OpenAI
  openai:
    api_key: "${OPENAI_API_KEY}"
    model: "text-embedding-3-small"  # 1536 维，性价比高
```

这种"LLM 用 DeepSeek + Embedding 用 OpenAI"的混合模式在实际项目中被证明是性价比最优的组合。

---

## 10.8 配置系统

### 10.8.1 分层配置

项目实现了三层配置体系，通过 `ConfigLoader` 加载：

```python
# src/graphrag_kg/core/config_loader.py

class ConfigLoader:
    """Load and merge configuration from multiple sources."""

    @classmethod
    def load_config(cls, profile: str = "default") -> dict:
        """Load configuration with the given profile."""
        # Layer 1: Default config
        config = cls._load_yaml("config/default.yaml")

        # Layer 2: Profile-specific config
        profile_path = f"config/{profile}.yaml"
        if os.path.exists(profile_path):
            profile_config = cls._load_yaml(profile_path)
            config = cls._deep_merge(config, profile_config)

        # Layer 3: Environment variables / .env file
        env_overrides = cls._load_env_overrides()
        config = cls._deep_merge(config, env_overrides)

        return config
```

### 10.8.2 快速模式 vs 生产模式

**快速模式（fast.yaml）**：用于开发和测试

```yaml
# config/fast.yaml
pipeline:
  chunking:
    chunk_size: 600          # 更小的 chunk，加快处理
    chunk_overlap: 50
  extraction:
    batch_size: 3            # 更小的批次
    max_entities_per_chunk: 15

community:
  resolution: 1.0
  n_iterations: 5            # 更少的迭代次数

search:
  local:
    top_k_chunks: 5
    max_hop: 1
  global:
    top_k_communities: 10
```

**生产模式（production.yaml）**：用于正式处理

```yaml
# config/production.yaml
pipeline:
  chunking:
    chunk_size: 1200
    chunk_overlap: 100
  extraction:
    batch_size: 10
    max_entities_per_chunk: 50
    iterative_extraction: true
    iterative_rounds: 2

community:
  resolution: 1.0
  n_iterations: -1            # 自动收敛

search:
  local:
    top_k_chunks: 20
    max_hop: 2
  global:
    top_k_communities: 30
    map_batch_size: 10
  drift:
    max_steps: 8
    exploration_width: 5
```

---

## 10.9 成本分析

### 10.9.1 Token 消耗模型

GraphRAG 的 Token 消耗主要来自两个阶段：

**1. 索引阶段（一次性成本）**

索引阶段的总 Token 消耗公式：

$$C_{\text{index}} = C_{\text{entity}} + C_{\text{relation}} + C_{\text{summary}}$$

其中：
- $C_{\text{entity}} = N_{\text{chunks}} \times (T_{\text{prompt}} + T_{\text{response}})_{\text{entity}}$
- $C_{\text{relation}} = N_{\text{chunks}} \times (T_{\text{prompt}} + T_{\text{response}})_{\text{relation}}$
- $C_{\text{summary}} = N_{\text{communities}} \times (T_{\text{prompt}} + T_{\text{response}})_{\text{summary}}$

**2. 查询阶段（每次查询的成本）**

- **本地搜索**：$C_{\text{local}} = T_{\text{entity\_extract}} + T_{\text{generate}}$
- **全局搜索**：$C_{\text{global}} = K_{\text{communities}} \times T_{\text{map}} + T_{\text{reduce}}$
- **DRIFT 搜索**：$C_{\text{drift}} = (S + 1) \times T_{\text{step}} + T_{\text{synthesis}}$
  - 其中 S 为探索步数

### 10.9.2 实测数据

以下是基于项目的实际运行数据（使用 DeepSeek-chat，500 篇中等长度文档）：

| 阶段 | 输入 Token | 输出 Token | 成本 (DeepSeek $0.14/M) |
|---|---|---|---|
| **索引阶段** | | | |
| 文本分块 | 0 | 0 | $0.00 |
| 实体提取 (500 chunks) | ~600K | ~100K | ~$0.10 |
| 关系提取 (500 chunks) | ~800K | ~120K | ~$0.13 |
| 图构建 | 0 | 0 | $0.00 |
| 社区检测 | 0 | 0 | $0.00 |
| 社区摘要 (50 communities) | ~300K | ~100K | ~$0.06 |
| Embedding (500 chunks + 50 summaries) | ~500K | 0 | ~$0.02 |
| **索引总成本** | **~2.2M** | **~320K** | **~$0.31** |

| **查询阶段（单次）** | | | |
| 本地搜索 | ~2K | ~1K | ~$0.0004 |
| 全局搜索 (Map=20, Reduce=1) | ~60K | ~30K | ~$0.013 |
| DRIFT 搜索 (5 steps) | ~25K | ~15K | ~$0.006 |

### 10.9.3 成本优化策略

**策略一：使用 DeepSeek 替代 OpenAI**

以实体提取为例的成本对比（假设 500 chunks）：

| 提供商 | 模型 | 输入成本 | 输出成本 | 总成本 |
|---|---|---|---|---|
| OpenAI | GPT-4o ($2.50/$10.00) | $1.50 | $1.00 | $2.50 |
| OpenAI | GPT-4o-mini ($0.15/$0.60) | $0.09 | $0.06 | $0.15 |
| DeepSeek | deepseek-chat ($0.14/$0.28) | $0.08 | $0.03 | $0.11 |
| DeepSeek | deepseek-reasoner ($0.55/$2.19) | $0.33 | $0.22 | $0.55 |

**结论**：DeepSeek-chat 的成本约为 GPT-4o-mini 的 73%，约为 GPT-4o 的 4%。

**策略二：选择性提取**

并非所有文档都需要完整提取。可以先用分类器判断文档类型：

```python
async def should_extract(self, chunk: str) -> bool:
    """Quick check if chunk contains extractable entities."""
    prompt = f"这段文本是否包含有意义的命名实体？回答 YES 或 NO。\n{chunk}"
    response = await self.llm_client.generate(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10,
    )
    return response.strip().upper() == "YES"
```

**策略三：缓存复用**

对于重复出现的文档片段，缓存提取结果：

```python
async def extract_with_cache(self, chunk: dict) -> list[dict]:
    chunk_hash = hashlib.md5(chunk["text"].encode()).hexdigest()
    cache_key = f"entity_extract:{chunk_hash}"

    cached = await self.cache.get(cache_key)
    if cached:
        return json.loads(cached)

    entities = await self._extract_from_chunk(chunk)
    await self.cache.set(cache_key, json.dumps(entities), ttl=86400)
    return entities
```

---

## 10.10 与纯向量 RAG 的对比

### 10.10.1 对比维度

| 维度 | 纯向量 RAG | GraphRAG |
|---|---|---|
| **检索粒度** | Chunk 级别 | 实体 + 关系 + Chunk |
| **多跳推理** | 弱（需 Re-Rank 或迭代检索） | 强（图遍历天然支持） |
| **全局理解** | 差（仅语义相似度） | 好（社区摘要提供全局视图） |
| **实现复杂度** | 低 | 高 |
| **索引成本** | 低（仅 embedding） | 高（LLM 提取 + 社区检测） |
| **查询延迟** | 低（毫秒级） | 中-高（数百毫秒到数秒） |
| **可解释性** | 中等（召回 chunks） | 高（可见图路径和社区） |
| **增量更新** | 容易（直接 upsert） | 困难（需重新检测社区） |

### 10.10.2 场景决策矩阵

```python
def recommend_strategy(query: str, corpus_type: str) -> str:
    """Recommend RAG strategy based on query and corpus characteristics."""
    if _is_entity_question(query):
        return "local_graphrag"
    elif _is_global_question(query):
        return "global_graphrag"
    elif _is_multi_hop_question(query):
        return "drift_graphrag"
    elif corpus_type in ["chat_logs", "social_media"]:
        return "vector_rag"
    else:
        return "hybrid_search"

def _is_entity_question(query: str) -> bool:
    """Detect entity-centric queries."""
    patterns = [
        r"(什么是|什么是|介绍|Explain|What is)",
        r"(\w+) 和 (\w+) 的关系",
        r"(\w+) 的 (创始人|CEO|总部|产品)",
    ]
    return any(re.search(p, query) for p in patterns)

def _is_global_question(query: str) -> bool:
    """Detect global/thematic queries."""
    patterns = [
        r"(主要|主要|overall|summary|总结|概括|趋势|pattern|theme)",
        r"文档(中|里)讨论了",
        r"全部|所有|every|all",
    ]
    return any(re.search(p, query) for p in patterns)

def _is_multi_hop_question(query: str) -> bool:
    """Detect multi-hop reasoning queries."""
    patterns = [
        r"(如何影响|impact|influence|cause|导致|引发)",
        r"通过.*实现|通过.*达成",
        r"之间的 (关系|联系|关联|connection)",
    ]
    return any(re.search(p, query) for p in patterns)
```

### 10.10.3 混合搜索方案

在实际项目中，纯向量 RAG 和 GraphRAG 并非互斥。项目实现了 HybridSearch，结合两者优势：

```python
# src/graphrag_kg/search/hybrid_search.py

class HybridSearch(BaseSearch):
    """Hybrid search combining vector RAG and GraphRAG."""

    def __init__(self, llm_client, graph, vector_store, config):
        super().__init__(llm_client, graph, vector_store, config)
        self.local_search = LocalSearch(llm_client, graph, vector_store, config)
        self.vector_search = vector_store
        self.hybrid_weight = config.get("hybrid_weight", 0.5)
        # hybrid_weight: 0 = pure vector, 1 = pure GraphRAG

    async def search(self, query: str, **kwargs) -> SearchResult:
        # Run both searches in parallel
        graphrag_task = self.local_search.search(query, **kwargs)
        vector_task = self.vector_search.search(query, top_k=5)

        graphrag_result, vector_result = await asyncio.gather(
            graphrag_task, vector_task
        )

        # Combine contexts
        combined_context = self._combine_contexts(
            graphrag_result.context,
            vector_result,
            self.hybrid_weight,
        )

        # Generate answer from combined context
        answer = await self.llm_client.generate(
            system_prompt=HYBRID_SEARCH_PROMPT,
            messages=[{
                "role": "user",
                "content": f"查询：{query}\n\n上下文：{combined_context}"
            }],
        )

        return SearchResult(
            answer=answer,
            context=combined_context,
            entities=graphrag_result.entities,
            vector_sources=[r["id"] for r in vector_result],
        )
```

**混合权重调优**：

```
hybrid_weight = 0.0  →  纯向量 RAG（快速但可能不全面）
hybrid_weight = 0.3  →  向量为主，图结果为辅
hybrid_weight = 0.5  →  均衡混合（推荐默认值）
hybrid_weight = 0.7  →  图结果为主，向量为辅
hybrid_weight = 1.0  →  纯 GraphRAG（全面但较慢）
```

---

## 10.11 生产部署指南

### 10.11.1 环境配置

```bash
# .env 文件配置
DEEPSEEK_API_KEY=sk-your-deepseek-api-key
OPENAI_API_KEY=sk-your-openai-api-key  # 用于 embedding（可选）
GRAPH_PATH=./data/graph
VECTOR_DB_PATH=./data/vector_store
LOG_LEVEL=INFO

# 选择配置 profile
GRAPH_PROFILE=production  # 或 fast, default
```

### 10.11.2 管道执行

```python
# 完整运行示例
import asyncio
from graphrag_kg.core.config_loader import ConfigLoader
from graphrag_kg.llm.deepseek_client import DeepSeekClient
from graphrag_kg.pipeline.embedder import Embedder
from graphrag_kg.pipeline.indexing_pipeline import IndexingPipeline
from graphrag_kg.search.local_search import LocalSearch
from graphrag_kg.search.global_search import GlobalSearch
from graphrag_kg.search.drift_search import DRIFTSearch

async def main():
    # 1. 加载配置
    config = ConfigLoader.load_config(profile="production")

    # 2. 初始化客户端
    llm = DeepSeekClient(config["llm"]["deepseek"])
    embedder = Embedder(config["embedding"])

    # 3. 执行索引管道
    pipeline = IndexingPipeline(llm, embedder, config["pipeline"])
    documents = load_documents()  # 你的文档加载逻辑
    result = await pipeline.run(documents)

    # 4. 搜索测试
    local_search = LocalSearch(llm, result["graph"], result["vector_store"], config["search"])
    global_search = GlobalSearch(llm, result["graph"], result["vector_store"], config["search"])
    drift_search = DRIFTSearch(llm, result["graph"], result["vector_store"], config["search"])

    # 本地搜索示例
    result_local = await local_search.search("什么是 GraphRAG？")
    print(f"本地搜索答案：{result_local.answer}")

    # 全局搜索示例
    result_global = await global_search.search("文档中讨论了哪些主要技术趋势？")
    print(f"全局搜索答案：{result_global.answer}")

    # DRIFT 搜索示例
    result_drift = await drift_search.search("DeepSeek 如何影响 AI 开源生态？")
    print(f"DRIFT 搜索答案：{result_drift.answer}")

asyncio.run(main())
```

### 10.11.3 性能监控

建议在生产环境中添加以下监控指标：

```python
# 监控装饰器示例
import time
import functools

def monitor(name: str):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = await func(*args, **kwargs)
                elapsed = time.time() - start
                logger.info(f"[{name}] 成功，耗时 {elapsed:.2f}s")
                # 发送到监控系统
                metrics.histogram(f"graphrag.{name}.duration", elapsed)
                metrics.counter(f"graphrag.{name}.success").inc()
                return result
            except Exception as e:
                elapsed = time.time() - start
                logger.error(f"[{name}] 失败，耗时 {elapsed:.2f}s: {e}")
                metrics.counter(f"graphrag.{name}.failure").inc()
                raise
        return wrapper
    return decorator
```

---

## 10.12 常见问题与解决方案

### Q1: 实体提取质量不高，漏掉了很多重要实体

**可能原因**：
- Chunk 太小，上下文不足
- 提示词没有针对性
- LLM 温度过高导致输出不一致

**解决方案**：
- 增大 chunk_size 到 1200-2000
- 使用少样本提示词，包含领域相关示例
- 设置 temperature=0.0
- 启用多轮迭代提取

### Q2: 社区检测结果不理想

**可能原因**：
- 分辨率参数不合适
- 图过于稀疏（实体太少）
- 实体类型不平衡（如过多"CONCEPT"类型）

**解决方案**：
- 调整 resolution 参数（从 0.8 到 1.5 间搜索）
- 增加实体提取的覆盖范围
- 考虑过滤低频实体（count < 2）

### Q3: DeepSeek API 调用超时

**可能原因**：
- 输入太长（接近 64K 上下文限制）
- 并发过高导致限流

**解决方案**：
- 减小 batch_size
- 增加 batch_delay
- 实现指数退避重试

```python
async def _call_with_retry(self, func, max_retries=3):
    for attempt in range(max_retries):
        try:
            return await func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt  # 指数退避: 1s, 2s, 4s
            logger.warning(f"重试 {attempt + 1}/{max_retries}, 等待 {wait}s")
            await asyncio.sleep(wait)
```

### Q4: 索引成本过高

**解决方案**：
- 切换到 DeepSeek-chat 替代 GPT-4o
- 使用选择性提取（先分类再提取）
- 缓存重复文档的提取结果
- 考虑增量索引（仅处理新增文档）

---

## 10.13 总结

### 10.13.1 关键要点

1. **GraphRAG 的核心价值**在于通过知识图谱的图结构信息，弥补纯向量 RAG 在关系推理和全局理解上的不足。

2. **Leiden 算法**是社区检测的最佳选择，其层次化社区结构天然适配 GraphRAG 的多级搜索需求。

3. **三种搜索策略各有定位**：
   - 本地搜索：面向实体级精确查询，效率最高
   - 全局搜索：面向主题级宏观分析，全面但成本高
   - DRIFT 搜索：面向多跳推理，在精度和成本间取得平衡

4. **DeepSeek 是极具性价比的 LLM 后端**，特别是在索引阶段的大批量处理场景中，成本优势显著。

5. **混合搜索**（向量 + 图）是实际生产中最推荐的方式，兼顾了检索的广度（向量）和深度（图）。

### 10.13.2 未来方向

GraphRAG 仍在快速发展中，值得关注的方向包括：

- **动态图更新**：无需完全重建即可增量更新知识图谱
- **多模态 GraphRAG**：将图像、表格等多模态信息纳入图结构
- **Agentic GraphRAG**：让 LLM Agent 自主规划图遍历路径
- **更轻量的实体提取**：使用小模型或特殊微调模型替代大 LLM 进行实体提取

---

## 参考资源

- GraphRAG 论文: https://arxiv.org/abs/2404.16130
- Leiden 算法: Traag, V. A., Waltman, L., & van Eck, N. J. (2019). "From Louvain to Leiden: guaranteeing well-connected communities." *Scientific Reports*, 9(1), 5233.
- DeepSeek API: https://platform.deepseek.com/docs
- NetworkX: https://networkx.org/
- python-igraph: https://python.igraph.org/
