# 第二章 RAG 核心原理与 GraphRAG 进阶

## 2.1 本章概述

本章深入剖析传统 RAG（Retrieval-Augmented Generation）系统的核心架构与设计原理，并在此基础上引入 GraphRAG 这一知识图谱增强的进阶方案。我们将从最经典的 RAG 流水线出发，逐一拆解每个环节的设计权衡与实现策略，然后展示图结构如何从根本上改变检索范式的信息组织方式。

读者将理解以下核心问题：

- 传统 RAG 的"先检索、后生成"范式存在哪些固有局限？
- Chunk 切得越大越好还是越小越好？Embedding 维度的选择依据是什么？
- 为什么向量数据库的 ANN 检索在大规模语料上会丢失高召回率？
- GraphRAG 通过知识图谱的实体-关系建模如何弥补稠密检索的语义盲区？
- Leiden 社区检测算法如何实现从"检索片段"到"检索社区摘要"的跃迁？
- 四种搜索模式（Local / Global / DRIFT / Basic）分别解决什么场景？

## 2.2 传统 RAG 架构

### 2.2.1 标准流水线

传统 RAG 系统遵循一条高度标准化的数据流水线，通常称为 **Ingest → Chunk → Embed → Store → Retrieve → Generate** 六阶段模型：

```
                  ┌─────────────┐
                  │   Ingest    │  原始文档导入与解析
                  └──────┬──────┘
                         ▼
                  ┌─────────────┐
                  │   Chunk     │  文档分割为片段
                  └──────┬──────┘
                         ▼
                  ┌─────────────┐
                  │   Embed     │  片段向量化
                  └──────┬──────┘
                         ▼
                  ┌─────────────┐
                  │   Store     │  向量写入数据库
                  └─────────────┘

   ┌───── Query ──────────────────────────────────────┐
   │                                                  │
   ▼                                                  │
┌──────────┐    ┌───────────┐    ┌──────────────┐     │
│ Retrieve │───▶│   Rank    │───▶│  Generate    │     │
└──────────┘    └───────────┘    └──────────────┘     │
                                                    │
   ┌───────────────── Response ◀─────────────────────┘
```

**阶段详解：**

| 阶段 | 功能 | 关键决策 |
|------|------|----------|
| **Ingest（导入）** | 读取原始文档并解析为纯文本或结构化内容 | 格式支持（PDF / HTML / Markdown / DOCX / LaTeX），编码检测，元数据提取 |
| **Chunk（分割）** | 将长文档切分为语义自包含的片段 | 分块策略与参数（大小、重叠、分隔符） |
| **Embed（向量化）** | 将每个文本片段编码为稠密向量 | 模型选择与维度权衡 |
| **Store（存储）** | 向量写入数据库并构建索引 | 向量数据库选型与索引类型（IVF / HNSW / DiskANN） |
| **Retrieve（检索）** | 接收查询向量并在索引中搜索 Top-K 最近邻 | 相似度度量（cosine / dot / L2），搜索参数（ef_search / nprobe） |
| **Generate（生成）** | 将检索结果作为上下文注入 LLM 并合成答案 | Prompt 模板设计，上下文窗口管理 |

### 2.2.2 Ingest 阶段

Ingest 是流水线的起点，其质量直接影响后续所有环节。一个工业级的 Ingest 管道需要处理以下挑战：

1. **多格式解析**：不同文档格式需要不同的解析器。PDF 需要处理文本层与 OCR 的混合；HTML 需要剥离标签但保留结构语义（标题层级、列表、表格）；Markdown 相对简单但需注意代码块和数学公式的特殊处理。

2. **元数据提取**：文档标题、作者、创建日期、章节编号、URL 来源等元数据应在 Ingest 阶段提取并随向量一同存储。元数据在后续的过滤检索（Metadata Filtering）中至关重要。

3. **编码检测**：中文文档常见 UTF-8、GBK、GB2312、Big5 等编码。建议使用 `charset-normalizer` 或 `cchardet` 自动检测，避免乱码导致的语义丢失。

4. **文档拆分预判**：Ingest 阶段应输出文档的层级结构（如目录树），为后续 Chunk 阶段的"document-aware"策略提供输入。

```python
# 一个简化的 Ingest Pipeline 伪代码
class Document:
    def __init__(self, content: str, metadata: dict, structure: list):
        self.content = content
        self.metadata = metadata
        self.structure = structure  # [(heading, level, start_char, end_char), ...]

def ingest(file_path: str) -> Document:
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return parse_pdf(file_path)
    elif ext == ".html":
        return parse_html(file_path)
    elif ext == ".md":
        return parse_markdown(file_path)
    else:
        return parse_plain_text(file_path)
```

### 2.2.3 Chunk 策略

Chunk 是 RAG 系统中争议最大、影响最直接的环节。切得太小，片段缺乏上下文语境；切得太大，向量表示的语义精度下降且容易超出 LLM 上下文窗口。以下是四种主流策略：

#### 策略一：固定大小切分（Fixed-Size Chunking）

最简单直接的方法：按固定字符数或 token 数切分，通常配合一定的重叠（overlap）来保持边界语义的连续性。

```
文档: [--- 512 chars ---][--- 512 chars ---][--- 512 chars ---]
重叠:                  [128 chars overlap]
```

**参数：**

| 参数 | 典型值 | 影响 |
|------|--------|------|
| chunk_size | 256 - 1024 tokens | 越大语义越完整，但检索粒度越粗 |
| overlap | 10% - 20% of chunk_size | 越大边界信息保留越好，但存储冗余增加 |

**优点**：实现简单，计算开销极低，适合快速原型。

**缺点**：完全无视语义边界，可能在句子或段落中间切断，导致检索到的片段"前言不搭后语"。

**适用场景**：文档结构极弱或不存在的场景（如日志文件、聊天记录）。

#### 策略二：语义切分（Semantic Chunking）

基于语义边界（如句子结束、段落切换、主题转换点）进行分割。通常使用两种方法：

1. **基于分割符的切分**：以 `\n\n`（段落）、`\n`（行）、句号等为锚点，递归地合并至接近 chunk_size 的语义单元。

2. **基于嵌入相似度的切分**：计算相邻句子的嵌入余弦相似度，当相似度突降时视为语义边界。

```
句子1  句子2  句子3  句子4  句子5
  │      │      │      │      │
  └──0.92┘      │      │      │
         └──0.95┘      │      │
                └──0.89┘      │
                       └──0.45┘   ← 语义边界（相似度骤降）
                       ┌────────┐
                       │ Chunk B│
                       └────────┘
  ┌───────────────────────┐
  │       Chunk A         │
  └───────────────────────┘
```

```python
def semantic_chunk(text: str, model, threshold: float = 0.5) -> list[str]:
    sentences = split_sentences(text)
    chunks = []
    current_chunk = [sentences[0]]

    for i in range(1, len(sentences)):
        emb_prev = model.encode(sentences[i - 1])
        emb_curr = model.encode(sentences[i])
        sim = cosine_similarity(emb_prev, emb_curr)

        if sim < threshold:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
        current_chunk.append(sentences[i])

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks
```

**优点**：保持语义完整性，检索结果更连贯。

**缺点**：需要额外的嵌入计算开销；阈值需要针对不同语料调优。

#### 策略三：文档感知切分（Document-Aware Chunking）

利用文档的固有结构（标题层级、章节、列表）来指导分割。这是目前工业界最推荐的策略，因为它同时保留了语义边界和文档的层级关系。

```
文档结构：
  H1: 第一章 绪论
    H2: 1.1 研究背景
      P1: ...
      P2: ...
    H2: 1.2 相关工作
      P1: ...
  H1: 第二章 方法
    H2: 2.1 数据收集
      ...

Chunk 输出（每个 Chunk 携带上下文路径）：
  Chunk 1: 第一章 绪论 > 1.1 研究背景 > P1 ...
  Chunk 2: 第一章 绪论 > 1.1 研究背景 > P2 ...
  Chunk 3: 第一章 绪论 > 1.2 相关工作 > P1 ...
  Chunk 4: 第二章 方法 > 2.1 数据收集 > P1 ...
```

每个 Chunk 携带一个 **contextual path**（如 `第一章 绪论 > 1.1 研究背景`），使得检索到的片段即使没有原文也能理解其上下文位置。这被称为 **Contextual Retrieval**，由 Anthropic 在 2024 年推广。

**优点**：保留文档结构语义；检索结果的上下文可追溯。

**缺点**：依赖文档解析质量；对无结构文档（如纯文本日志）无效。

#### 策略四：递归切分（Recursive Chunking）

LangChain 推广的策略：用一组分隔符（优先级从高到低：`\n\n` > `\n` > `。` > `，` > 空格）递归地尝试分割，直到每个片段满足 size 约束。

```python
def recursive_chunk(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    result = []
    for sep in separators:
        if sep in text:
            parts = text.split(sep)
            # 如果分割后的每个部分仍然太大，递归
            for part in parts:
                if len(part) > chunk_size:
                    result.extend(recursive_chunk(part, chunk_size, separators[1:]))
                else:
                    result.append(part)
            break
    else:
        # 没有匹配的分隔符，直接返回
        result = [text] if len(text) <= chunk_size else [text[:chunk_size]]
    return result
```

**优点**：灵活，能自适应不同粒度的文档结构；工业实践中表现稳定。

**缺点**：参数（分隔符列表和优先级）需要针对语料调优。

#### Chunk 策略对比总结

| 策略 | 语义保持 | 实现复杂度 | 计算开销 | 推荐场景 |
|------|----------|------------|----------|----------|
| Fixed-Size | 差 | 极低 | 无 | 快速原型、无结构文本 |
| Semantic | 好 | 中 | 中 | 新闻文章、博客 |
| Document-Aware | 极好 | 高 | 低 | 技术文档、论文、书籍 |
| Recursive | 好 | 低 | 低 | 通用（工业首选） |

### 2.2.4 Embedding 模型与向量化

Embedding 模型将文本映射到稠密向量空间，使语义相似的文本在向量空间中距离更近。这是 RAG 系统的语义基础。

#### 主流 Embedding 模型对比

| 模型 | 维度 | 最大输入 tokens | 开源 | 语言支持 | 特点 |
|------|------|-----------------|------|----------|------|
| text-embedding-3-large | 3072 | 8191 | 否 | 多语言 | 最高精度，支持 Matryoshka 降维 |
| text-embedding-3-small | 1536 | 8191 | 否 | 多语言 | 性价比最优 |
| bge-m3 | 1024 | 8192 | 是 | 多语言（尤其中英） | 支持稠密/稀疏/ColBERT 多向量 |
| bge-large-zh-v1.5 | 1024 | 512 | 是 | 中文优化 | 中文检索 SOTA |
| nomic-embed-text-v1.5 | 768 | 8192 | 是 | 多语言 | 开源可用，支持 Matryoshka |
| jina-embeddings-v3 | 1024 | 8192 | 是 | 多语言 | 支持 LoRA 任务适配 |
| intfloat/multilingual-e5-large | 1024 | 512 | 是 | 多语言 | 指令前缀可调 |

**维度选择的工程考量：**

- **高维度（3072d）**：text-embedding-3-large 提供最高的语义表示精度，但存储和计算开销大。300 万条 3072 维向量以 float32 存储约需 35 GB。
- **中维度（1024d）**：bge-m3 和 e5 系列在精度和效率之间取得良好平衡，是工业应用的主流选择。
- **低维度（768d 及以下）**：nomic-embed-text 和 Matryoshka 降维后的模型适合资源受限场景（边缘设备、移动端）。

**Matryoshka Representation Learning（MRL）**：OpenAI text-embedding-3 系列支持 MRL，即模型在 256 / 512 / 1024 / 2048 / 3072 维度的子向量上均保持良好性能。这意味着可以在检索时使用低维度（如 256d）加快速度，在存储时保留高维度（3072d）保持精度。

```python
# 使用 Matryoshka 降维检索加速
import openai

# 存储时用完整维度
response = openai.embeddings.create(
    model="text-embedding-3-large",
    input=documents,
    dimensions=3072  # 完整维度
)

# 检索时截取低维度（不重新调用 API，直接切片）
query_emb = openai.embeddings.create(
    model="text-embedding-3-large",
    input=query,
    dimensions=256  # 快速检索
)
```

#### 相似度度量

向量检索的数学基础是向量相似度计算。三种主流度量：

| 度量 | 公式 | 范围 | 特点 |
|------|------|------|------|
| Cosine Similarity | cos(A,B) = A·B / (|A|×|B|) | [-1, 1] | 只关心方向，忽略模长；最常用 |
| Dot Product | A·B = Σaᵢbᵢ | (-∞, +∞) | 包含方向和模长；规范化向量后等价于 cosine |
| Euclidean (L2) | √Σ(aᵢ-bᵢ)² | [0, +∞) | 距离越小越相似；对向量模长敏感 |

**实践经验**：使用 cosine 或 normalized dot product 作为默认选择。对于 OpenAI 等已做 L2 归一化的 embedding 输出，dot product 与 cosine 等价。

### 2.2.5 向量数据库

向量数据库负责存储 embedding 向量并提供高效的近似最近邻（ANN）搜索。

#### 主流向量数据库对比

| 特性 | LanceDB | FAISS | Milvus | ChromaDB | Qdrant | Pinecone |
|------|---------|-------|--------|----------|--------|----------|
| **部署模式** | 嵌入式/无服务器 | 库 | 分布式 | 嵌入式 | 自托管/SaaS | 托管 |
| **索引类型** | IVF-PQ, DiskANN | IVF, HNSW, PQ | IVF, HNSW, DiskANN | HNSW | HNSW, PQ | 专有 |
| **持久化** | Lance 列式格式 | 内存/磁盘 | RocksDB/对象存储 | DuckDB/SQLite | 本地/云存储 | 云 |
| **标量过滤** | 好 | 有限 | 好 | 好 | 好 | 好 |
| **混合搜索** | 支持 | 需手动 | 原生支持 | 有限 | 原生支持 | 支持 |
| **分布式** | 否 | 否 | 是 | 否 | 是 | 是 |
| **中文社区** | 较少 | 广泛 | 广泛 | 一般 | 一般 | 一般 |
| **适用规模** | 百万级 | 百万-亿级 | 亿级 | 百万级 | 千万-亿级 | 亿级 |
| **学习成本** | 低 | 中 | 高 | 极低 | 中 | 低 |

**选型建议：**

- **原型开发**：ChromaDB（零配置，API 极简）或 LanceDB（嵌入式，无服务依赖）。
- **单机百万级**：FAISS + 自定义存储层（极致性能）或 LanceDB（兼顾查询与管理）。
- **生产分布式**：Milvus（功能最全，生态最完善）或 Qdrant（Rust 实现，性能优异）。
- **全托管**：Pinecone（无需运维，成本较高）。

#### ANN 索引技术简介

向量数据库的核心在于 ANN 索引，它通过牺牲少量精度来换取数量级的检索加速。

| 索引 | 原理 | 搜索复杂度 | 特点 |
|------|------|------------|------|
| IVF（倒排文件） | K-Means 聚类 + 最近簇搜索 | O(√n) | 简单有效，适合中等规模 |
| HNSW（分层可导航小世界图） | 多层跳表式图结构 | O(log n) | 高召回率，内存密集 |
| PQ（乘积量化） | 向量子空间压缩 | O(n) 压缩 | 极致内存压缩，精度损失 |
| DiskANN | SSD 上的图索引 | O(log n) | 超大规模（十亿级） |
| IVF-PQ | IVF + PQ 组合 | O(√n) | 内存与速度的平衡点 |

**HNSW 参数调优：**

```python
# HNSW 索引的核心参数
{
    "M": 16,              # 每个节点的最大连接数（默认 16，越大召回越高，内存越大）
    "ef_construction": 200,  # 构建时的搜索宽度（越大构建越慢，索引质量越高）
    "ef_search": 50,         # 搜索时的动态列表大小（越大召回越高，搜索越慢）
}
```

`M` 和 `ef_construction` 影响索引构建阶段，`ef_search` 是运行时参数，可根据精度需求动态调整。

### 2.2.6 检索方法

检索阶段决定了哪些上下文片段被送入 LLM。不同的检索方法在召回率、精度和计算开销之间做出不同的权衡。

#### 稠密检索（Dense Retrieval）

使用神经网络 embedding 将查询和文档映射到同一语义空间，通过向量相似度搜索。

```
Query: "2024年GDP增长率"
    ↓
Embedding: [0.23, -0.45, 0.78, ..., 0.12]  (1536维)
    ↓
ANN Search ───▶ Top-K 结果: [doc_42, doc_17, doc_89, ...]
```

**优点**：
- 语义匹配能力强，能捕捉同义词和概念相似性
- 零样本泛化能力好

**缺点**：
- 对罕见实体和专有名词的匹配可能弱于词法匹配
- 训练数据分布外的查询可能表现不佳

#### 稀疏检索（Sparse Retrieval）

基于词袋或 TF-IDF 的检索方法，典型代表是 BM25。

```python
# BM25 评分公式
# score(D, Q) = Σ_idf(q_i) × tf(q_i, D) × (k₁ + 1) / (tf(q_i, D) + k₁ × (1 - b + b × |D|/avgdl))
# 其中 k₁ = 1.2 ~ 2.0, b = 0.75

def bm25_score(query_terms: list[str], doc_freqs: dict, doc_len: int, avgdl: float) -> float:
    k1, b = 1.5, 0.75
    score = 0.0
    for term in query_terms:
        tf = doc_freqs.get(term, 0)
        idf = compute_idf(term)
        score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * doc_len / avgdl))
    return score
```

**优点**：
- 精确的关键词匹配，适合专有名词、代码、ID 等
- 不需要 GPU 或神经网络
- 可解释性强

**缺点**：
- 无法处理同义词和语义相似性
- "汽车"和"车辆"在词袋视角下完全不相关

#### 混合检索（Hybrid Retrieval）

结合稠密检索和稀疏检索的优点，通过加权融合或 Reranking 策略得到最终结果。

```
Query: "Python 异步编程的性能优化"
    │
    ├──▶ Dense Retrieval  ───▶ [doc_A: 0.92, doc_B: 0.88, doc_C: 0.85]
    │                          语义匹配（异步、性能、优化概念相关）
    │
    ├──▶ Sparse Retrieval ───▶ [doc_D: 8.5, doc_E: 7.2, doc_A: 6.8]
    │                          关键词匹配（Python、异步、性能、优化）
    │
    └──▶ Fusion ────▶ Reciprocal Rank Fusion (RRF) 或 加权平均
                          │
                          ▼
                  [doc_A, doc_D, doc_B, doc_E, doc_C]
```

**Reciprocal Rank Fusion (RRF)**：

```python
def rrf(results: list[list[str]], k: int = 60) -> list[str]:
    """
    Reciprocal Rank Fusion: 融合多个检索结果列表
    k 是平滑常数，通常设为 60
    """
    scores = {}
    for rank_list in results:
        for rank, doc_id in enumerate(rank_list, 1):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)
```

**加权混合**：

```python
def hybrid_score(
    dense_score: float,
    sparse_score: float,
    alpha: float = 0.5
) -> float:
    """
    alpha 控制稠密检索的权重：
    - alpha = 1.0: 纯稠密检索
    - alpha = 0.0: 纯稀疏检索
    - alpha = 0.5: 等权混合
    """
    # 需要先对两种分数做归一化
    return alpha * normalize(dense_score) + (1 - alpha) * normalize(sparse_score)
```

#### 重排序（Reranking）

在检索到 Top-K（如 100 条）候选项后，使用一个更精确（也更慢）的交叉编码器（Cross-Encoder）对候选列表重新排序，取最终的 Top-N（如 10 条）送入 LLM。

```
Query + Doc ───▶ Cross-Encoder ───▶ 相关性分数
(成对输入)         BAAI/bge-reranker-v2-m3
                   BAAI/bge-reranker-v2-gemma

检索阶段: Bi-Encoder (快速, O(n) 近似)
重排序:   Cross-Encoder (精确, O(K) 精确匹配), K << n
```

**典型流程**：

1. 稠密检索：用 Bi-Encoder（如 bge-m3）从 100 万文档中召回 Top-100
2. 稀疏检索：用 BM25 从相同语料中召回 Top-100
3. RRF 融合：合并为 Top-100 候选
4. Cross-Encoder Reranking：对 Top-100 逐对计算精确相关性，取 Top-10

## 2.3 传统 RAG 的固有局限

尽管传统 RAG 已在众多场景中展现出强大能力，但其架构存在若干根本性局限：

### 局限一：语义盲区

稠密检索依赖 embedding 向量的语义近似，但 embedding 是"有损压缩"的——一个 1024 维的向量不可能完整保留一篇 500 词文档的所有信息。当查询涉及以下场景时，传统 RAG 表现不佳：

- **多跳推理（Multi-hop Reasoning）**：需要跨多个文档片段串联信息的查询，如"A 公司收购了 B 公司，B 公司的 CEO 后来加入了哪家公司？"
- **聚合查询（Aggregate Query）**：需要综合大量片段才能回答的问题，如"报告中的五个实验有哪些共同特征？"
- **实体关系查询**：如"哪些药物与化合物 X 产生交互？"——需要理解实体间的拓扑关系而非语义相似度

### 局限二：上下文窗口的碎片化

LLM 的上下文窗口虽已扩展至 128K-200K tokens，但"能容纳"不等同于"能有效利用"：

- **中间信息丢失**：Lost-in-the-Middle 现象表明，LLM 对上下文窗口中间位置的检索结果注意力显著低于开头和结尾
- **碎片化上下文**：即使有多个相关片段，片段之间的逻辑关系和共同主题需要 LLM 自行推断，增加了推理负担

### 局限三：全局知识不可见

传统 RAG 本质上是一个**局部检索**系统——每个查询独立搜索最相似的片段。它无法回答"整个语料库呈现出什么趋势"这类需要全局视角的问题，因为没有一个单独的片段包含全局信息。

### 局限四：缺乏结构化知识

传统 RAG 将文档视为"一袋片段"（a bag of chunks），丢失了文档中实体之间的结构化关系。例如：

```
文档片段 A: "Apple 发布了新款 MacBook Pro，搭载 M3 芯片。"
文档片段 B: "M3 芯片基于 3nm 工艺，相比 M2 性能提升 50%。"

传统 RAG 检索到 A 和 B 后，LLM 需要自行推断：
  Apple ──发布──▶ MacBook Pro ──搭载──▶ M3 芯片 ──基于──▶ 3nm 工艺
```

这种关系推理对 LLM 而言并非总是可靠，尤其当涉及多步推导或大量实体时。

## 2.4 GraphRAG 架构

GraphRAG（Microsoft Research, 2024）通过引入知识图谱来解决传统 RAG 的上述局限。其核心思想是：**在文档之上构建一个实体-关系图，使检索从"片段级"提升到"知识级"**。

### 2.4.1 GraphRAG 索引流水线

GraphRAG 的索引流水线比传统 RAG 复杂得多，它包含从文本中提取结构化知识的完整流程：

```
                  ┌─────────────┐
                  │   Ingest    │  文档导入
                  └──────┬──────┘
                         ▼
                  ┌─────────────┐
                  │  Chunk +    │  文本分块
                  │  Text Units │  文本单元（可溯源的最小单位）
                  └──────┬──────┘
                         ▼
                  ┌─────────────┐
                  │  Entity/    │  LLM 驱动的实体与关系抽取
                  │  Relation   │  Entity: 人名、地名、概念、产品……
                  │  Extraction │  Relation: 实体间的语义连接
                  └──────┬──────┘
                         ▼
                  ┌─────────────┐
                  │  Entity     │  实体解析与消歧
                  │  Resolution │  合并同义实体（"Apple" vs "苹果公司"）
                  └──────┬──────┘
                         ▼
                  ┌─────────────┐
                  │  Knowledge  │  构建知识图谱
                  │  Graph      │  节点 = Entity, 边 = Relation
                  └──────┬──────┘
                         ▼
                  ┌─────────────┐
                  │  Community  │  社区检测（Leiden 算法）
                  │  Detection  │  将图划分为层次化的社区
                  └──────┬──────┘
                         ▼
                  ┌─────────────┐
                  │  Community  │  为每个社区生成自然语言摘要
                  │  Summary    │  LLM 综合社区内所有实体与关系
                  └──────┬──────┘
                         ▼
                  ┌─────────────┐
                  │  Network    │  社区摘要的向量化
                  │  Embedding  │  社区级别的向量索引
                  └─────────────┘
```

#### 流水线各步骤详解

**1. 文本单元（Text Units）**

Text Units 是 GraphRAG 中最小的可溯源文本片段，类似于传统 RAG 的 Chunk。不同的是，每个 Text Unit 与从其中抽取的实体和关系保持双向链接，形成完整的溯源链。

```json
{
  "id": "tu_0042",
  "text": "Apple 发布了新款 MacBook Pro，搭载 M3 芯片，性能提升显著。",
  "document_id": "doc_007",
  "entities": ["Apple", "MacBook Pro", "M3 芯片"],
  "relationships": [
    {"source": "Apple", "target": "MacBook Pro", "type": "发布"},
    {"source": "MacBook Pro", "target": "M3 芯片", "type": "搭载"}
  ]
}
```

**2. 实体与关系抽取（Entity & Relation Extraction）**

这是 GraphRAG 的核心步骤，通常由 LLM 驱动。给定一段文本，LLM 被提示提取其中出现的所有命名实体以及它们之间的语义关系。

```json
// LLM 抽取输出示例
{
  "entities": [
    {"name": "Apple", "type": "组织", "description": "美国科技公司"},
    {"name": "MacBook Pro", "type": "产品", "description": "Apple 的笔记本电脑系列"},
    {"name": "M3 芯片", "type": "技术", "description": "Apple 自研的第三代处理器"}
  ],
  "relationships": [
    {"source": "Apple", "target": "MacBook Pro", "type": "发布", "description": "Apple 于 2023 年 10 月发布"},
    {"source": "MacBook Pro", "target": "M3 芯片", "type": "搭载", "description": "MacBook Pro 使用 M3 芯片"}
  ]
}
```

**关键设计决策：**

- **提取粒度**：粗粒度（只提取重要实体）vs 细粒度（提取所有实体）。细粒度信息更丰富但图谱噪音更大。
- **LLM 模型选择**：GPT-4 / Claude 3.5 Sonnet 在实体抽取精度上显著优于小型模型。
- **提示词设计**：需要定义清晰的实体类型体系（Person / Organization / Location / Concept / Product / Event / Technology）和关系类型体系。

**3. 实体解析（Entity Resolution）**

不同文档可能以不同方式指代同一实体，需要做实体消歧。例如：

- "Apple" vs "苹果公司" vs "Apple Inc."
- "MacBook Pro" vs "MacBook Pro 2023"
- "M3" vs "M3 芯片" vs "Apple M3"

解析方法可以是基于规则的（名称归一化、别名映射），也可以是基于嵌入相似度的聚类。

**4. 知识图谱构建（Knowledge Graph Construction）**

解析后的实体作为图的节点（Node），关系作为边（Edge）。图结构为后续的社区检测和摘要生成提供基础。

```
      ┌──────┐    发布    ┌──────────┐
      │Apple │──────────▶│MacBook Pro│
      └──┬───┘           └─────┬────┘
         │                     │
         │ 收购                │ 搭载
         │                     │
    ┌────▼────┐          ┌────▼────┐
    │  Beats  │          │ M3 芯片 │
    └─────────┘          └────┬────┘
                              │
                        基于  │
                              │
                         ┌────▼────┐
                         │ 3nm 工艺│
                         └─────────┘
```

**5. 社区检测：Leiden 算法**

社区检测是将大规模知识图谱划分为语义相关的子图（社区）的过程。GraphRAG 使用 **Leiden 算法**，它是 Louvain 算法的改进版本。

#### Leiden 算法详解

Leiden 算法是一种基于模块度优化的图分割算法，相比 Louvain 算法有三个关键改进：

1. **保证社区连通性**：Louvain 可能产生不连通的社区（一个社区内包含分离的组件），Leiden 通过额外的细化步骤避免此问题。
2. **更快的收敛速度**：通过局部移动和细化步骤的组合，Leiden 的迭代次数更少。
3. **更高的模块度**：在同等条件下，Leiden 通常能发现比 Louvain 模块度更高的划分。

**算法步骤：**

```
Phase 1: 局部移动（Local Moving）
  对于每个节点 v，评估将其从当前社区移动到邻居社区 u 的社区是否
  能提升模块度。选择模块度增益最大的移动。

Phase 2: 细化（Refinement，Leiden 的独有步骤）
  对于 Phase 1 后的每个社区，在其内部运行一个受限的局部移动过程，
  确保最终社区是内部连通的（每个社区内的节点通过该社区内的路径
  可达）。

Phase 3: 聚合（Aggregation）
  将 Phase 2 得到的每个社区聚合成一个超级节点，构建新的图。
  然后对聚合图重复 Phase 1-3，直到模块度不再提升。
```

```
原始图:         Phase 1:            Phase 2:            Phase 3:
  ○──○           [A]──[A]           [A]──[A]           ╔══╗
  │  │    局部    │  │     细化      │  │      聚合      ║A ║──╔══╗
  ○──○    ──▶   [A]──[B]    ──▶   [A]──[A]    ──▶   ╚══╝  ║B ║
  │  │           │  │               │  │               │  ╚══╝
  ○──○           [B]──[B]           [B]──[B]           ╔══╝
                        (Louvain 可能产生此结果)       ║C ║
                        (Leiden 通过细化保证连通性)    ╚══╝
```

**模块度公式：**

```
Q = (1/2m) × Σ[ Aᵢⱼ - (kᵢ×kⱼ)/(2m) ] × δ(cᵢ, cⱼ)

其中:
  Aᵢⱼ = 节点 i 和 j 之间的边权重
  kᵢ  = 节点 i 的度数（连接的边数）
  m   = 总边数
  δ   = Kronecker delta（社区相同时为 1，否则为 0）
```

**层次化社区结构**：

Leiden 算法天然产生层次化的社区结构，这在 GraphRAG 中极为重要：

```
Level 0: [社区 0] ── [社区 1] ── [社区 2]       （最细粒度）
                 \         /
Level 1:     [超级社区 A] ── [超级社区 B]          （中等粒度）
                      │
Level 2:        [根社区]                            （最粗粒度）
```

每个层级的社区对应不同的语义抽象程度，供不同的搜索模式使用。

**6. 社区摘要生成（Community Summary Generation）**

对每个检测到的社区，GraphRAG 使用 LLM 生成自然语言摘要。摘要内容涵盖：

- 社区中的核心实体及其描述
- 实体之间的关键关系
- 社区整体的主题或共性

```json
{
  "community_id": "c_0_0042",
  "level": 0,
  "summary": "该社区涵盖 Apple 的芯片产品线，包括 M 系列芯片（M1、M2、M3）及其技术特征。核心实体包括 Apple 作为制造商，台积电作为代工厂，3nm/5nm 制程工艺。M3 芯片基于 3nm 工艺，相比 M2（5nm）性能提升 50%，能效提升 30%。",
  "entities": ["Apple", "M3 芯片", "M2 芯片", "M1 芯片", "台积电", "3nm 工艺", "5nm 工艺"],
  "relationships": [
    {"source": "M3 芯片", "target": "3nm 工艺", "type": "基于"},
    {"source": "M3 芯片", "target": "M2 芯片", "type": "取代"},
    {"source": "Apple", "target": "台积电", "type": "委托生产"}
  ]
}
```

**7. 网络嵌入（Network Embedding）**

社区摘要文本被向量化并存储，形成社区级别的向量索引。这使得检索可以在社区粒度上进行——即检索到一个社区摘要，就能获得该社区包含的所有实体和关系信息。

### 2.4.2 GraphRAG 搜索方法

GraphRAG 提供四种搜索方法，每种针对不同的查询类型设计。这是 GraphRAG 区别于传统 RAG 最显著的特性之一。

#### 方法一：Local Search（局部搜索）

**目标**：回答关于特定实体或局部知识的问题。

**工作流程**：

1. 将查询向量化，在 Text Units 的向量索引中搜索最相关的 Top-K 片段
2. 从这些片段溯源到关联的实体和关系
3. 构建一个以这些实体为中心的**局部子图**（通常 2-3 跳）
4. 将子图结构连同原始片段一起作为上下文注入 LLM

```
Query: "M3 芯片的性能指标是什么？"
    │
    ▼
Dense Retrieval (Text Units) ───▶ [tu_0042, tu_0051, tu_0067]
    │                               │
    │                        追溯关联实体
    ▼                               ▼
Local Subgraph:            M3 芯片 ──基于──▶ 3nm 工艺
                      M2 芯片 ◀──取代──┤
                                        ├──性能提升──▶ 50%
                                        └──能效提升──▶ 30%
    │
    ▼
LLM Input: [子图结构] + [原始 Text Units]
    │
    ▼
Response: "M3 芯片基于 3nm 工艺，相比 M2 芯片 CPU 性能提升 50%，能效提升 30%。"
```

**特点**：
- 聚焦性强，适合事实性查询
- 输出包含实体关系信息，比纯文本片段更结构化
- 与检索到的片段紧密关联，幻觉风险较低

#### 方法二：Global Search（全局搜索）

**目标**：回答需要综合全局知识的问题，如"整个文档集合中呈现哪些主题趋势？"

**工作流程**：

1. 在社区摘要（Community Summaries）向量索引中搜索与查询最相关的社区
2. 使用 Map-Reduce 模式：先并行获取每个社区的响应，再合并
3. 最终输出全局综合答案

```
Query: "本报告集中讨论了哪些主要技术趋势？"
    │
    ▼
Community Retrieval ───▶ [社区摘要_c_0_0012, 社区摘要_c_0_0042, ...]
    │                       (搜索社区级别的向量索引)
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Map Phase: 对每个社区摘要，LLM 生成部分答案                │
│                                                             │
│  社区 c_0_0012: "该社区讨论 AI 芯片的演进方向..."          │
│  社区 c_0_0042: "该社区讨论 Apple 芯片产品线..."           │
│  社区 c_0_0078: "该社区讨论台积电制程工艺..."              │
│  ...                                                        │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Reduce Phase: 将所有部分答案合并，LLM 综合生成全局回答     │
│                                                             │
│  Response: "报告集中讨论了三大技术趋势：                    │
│   1. AI 芯片的专用化与性能飞跃                              │
│   2. 先进制程工艺的竞争格局（3nm/5nm）                     │
│   3. 软硬件协同设计的生态构建"                             │
└─────────────────────────────────────────────────────────────┘
```

**特点**：
- 能回答传统 RAG 无法回答的全局性问题
- 计算成本高（Map-Reduce 需要多次 LLM 调用）
- 输出是综合性的，但细节精确度可能低于 Local Search

#### 方法三：DRIFT Search（动态检索推理聚焦）

**目标**：结合 Local 和 Global 的优势，通过**迭代探索**逐步构建答案。

DRIFT（Dynamic Retrieval Inference Focused Technology）是 GraphRAG 中最灵活的搜索方法。它的核心是**检索-推理循环**：

```
Query: "M3 芯片与 A17 Pro 芯片的关系是什么？"
    │
    ▼
Step 1: 初始检索
  - 在 Text Units 中搜索 "M3 芯片" → 获得 M3 芯片相关信息
  - 发现关键实体：M3 芯片, Apple, 3nm 工艺
    │
    ▼
Step 2: 推理与扩展
  - LLM 分析："M3 芯片是 Apple 的产品，与 A17 Pro 同代。"
  - 生成扩展查询："A17 Pro 芯片 规格 3nm"
  - 在图谱中搜索 A17 Pro 节点 → 获得 A17 Pro 信息
    │
    ▼
Step 3: 交叉验证
  - M3 芯片节点连接: (3nm, MacBook Pro)
  - A17 Pro 节点连接: (3nm, iPhone 15 Pro)
  - 发现共同关联：都基于 3nm 工艺
    │
    ▼
Step 4: 综合回答
  - "M3 芯片和 A17 Pro 芯片都基于台积电 3nm 工艺，分别面向 Mac 和 iPhone"
  - 提供溯源信息
```

**特点**：
- 动态迭代，能发现查询中未直接提及的相关实体
- 结合了 Local Search 的精确性和 Global Search 的广度
- 适用于需要多跳推理的复杂查询

#### 方法四：Basic Search（基础搜索）

**目标**：提供与传统 RAG 兼容的基线检索能力。

即标准的稠密检索 + LLM 生成，不使用图结构。主要用于：
- 与传统 RAG 系统的对比基准
- 简单的事实性查询
- 低延迟需求场景

#### 自动路由（Auto-Routing）

GraphRAG 还提供查询路由机制，根据查询特征自动选择最合适的搜索方法：

```python
def auto_route(query: str) -> str:
    """
    自动路由：根据查询特征选择搜索方法
    """

    # Global 关键词：需要全局视角
    global_keywords = ["趋势", "总结", "概述", "全局", "整体",
                       "主题", "共性问题", "correlation", "overview",
                       "trend", "theme", "pattern", "summary"]

    # Local 关键词：需要精确事实
    local_keywords = ["是什么", "谁", "何时", "何地", "多少",
                      "定义", "属性", "特征", "规格", "参数",
                      "what is", "who", "when", "where", "how many"]

    # DRIFT 关键词：需要多步推理
    drift_keywords = ["关系", "比较", "区别", "联系", "影响",
                      "因果关系", "如何导致", "对比",
                      "relationship", "compare", "contrast",
                      "how does", "why does"]

    if any(kw in query for kw in global_keywords):
        return "global"
    elif any(kw in query for kw in drift_keywords):
        return "drift"
    elif any(kw in query for kw in local_keywords):
        return "local"
    else:
        return "local"  # 默认 fallback
```

**实际应用建议**：自动路由可作为默认行为，同时允许用户手动指定搜索方法以覆盖自动选择。

### 2.4.3 GraphRAG vs 传统 RAG

| 维度 | 传统 RAG | GraphRAG |
|------|----------|----------|
| **信息单元** | 文本片段（Chunks） | 实体 + 关系 + 社区摘要 |
| **检索粒度** | 片段级 | 实体级 / 社区级 / 文本单元级 |
| **全局推理** | 不支持 | 支持（Global Search） |
| **多跳推理** | 依赖 LLM 自行推断 | 图谱结构天然支持 |
| **聚合查询** | 几乎无法回答 | 通过社区摘要支持 |
| **溯源能力** | 片段引用 | 实体 → 关系 → 文本单元 → 文档 |
| **索引成本** | 低（一次 Embedding） | 高（多次 LLM 调用 + 图构建） |
| **存储开销** | 向量 + 原始文本 | 向量 + 图 + 社区摘要 + 文本单元 |
| **查询延迟** | 低（毫秒级） | 中-高（Local: 秒级, Global: 数十秒） |
| **知识发现** | 被动检索已有内容 | 主动发现实体关系和主题 |
| **可解释性** | 中等（片段引用） | 高（实体关系图 + 溯源链） |
| **适用场景** | FAQ、文档问答、知识库 | 复杂推理、趋势分析、多文档综合 |
| **冷启动成本** | 低 | 高（需要 LLM 进行实体抽取和摘要生成） |

**选择指南：**

- **如果你的场景以事实性问答为主**（如"公司去年的营收是多少？"），传统 RAG 已足够，GraphRAG 的额外成本不值得。
- **如果你的场景涉及跨文档综合**（如"本季度的客户反馈主要有哪些共性问题？"），GraphRAG 的社区摘要能力能提供显著优势。
- **如果你的场景需要关系推理**（如"X 药物影响了哪些代谢通路，这些通路又与哪些疾病相关？"），GraphRAG 的图结构是天然解决方案。
- **如果你的场景需要高可解释性**（如金融风控、医疗诊断），GraphRAG 的溯源链能提供更清晰的推理路径。

## 2.5 GraphRAG 的工程挑战

### 2.5.1 计算成本

GraphRAG 索引流水线的核心步骤——实体抽取、关系抽取、社区摘要生成——都需要多次调用 LLM。对于大规模文档集，这可能带来显著的 API 成本和时间开销。

**成本估算：**

| 步骤 | LLM 调用次数 | 典型成本（GPT-4o） |
|------|-------------|-------------------|
| 实体抽取（每 Text Unit） | 1 | $0.01 - $0.05 / 千个 Text Unit |
| 实体解析 | 1 - 2 / 实体对 | $0.001 - $0.005 / 对 |
| 社区摘要（每社区） | 1 | $0.01 - $0.10 / 社区 |

对于一个包含 10,000 个 Text Unit、生成 500 个社区的文档集，预估成本约为 $50 - $500。

**优化策略：**

1. **使用小型模型进行抽取**：实体抽取任务可以用 GPT-4o-mini 或 Claude 3 Haiku 完成，摘要生成用强模型。
2. **增量索引**：只对新增文档运行实体抽取，已存在的社区只需增量更新。
3. **批处理**：将多个 Text Unit 合并为一个 LLM 调用进行批量抽取。

### 2.5.2 抽取质量

实体抽取的质量直接影响整个 GraphRAG 系统的效果。常见问题包括：

- **实体遗漏**：LLM 漏掉了重要实体
- **实体幻觉**：LLM 生成了原文中不存在的实体
- **关系类型不一致**：同一关系在不同文档中被标记为不同类型（"位于" vs "坐落于"）
- **实体粒度不一致**：有的文档抽取到"芯片"级别，有的只抽取到"处理器"级别

**缓解方法**：
- 提供详细的实体类型体系示例（Few-shot Examples）
- 对抽取结果做后处理：归一化关系类型名称
- 使用 Schema 约束输出格式（JSON Schema）

### 2.5.3 可扩展性

当文档规模从百万级增长到亿级时，GraphRAG 面临的可扩展性挑战：

- **图的规模**：亿级实体节点 + 数十亿边，需要分布式图数据库（如 Neo4j、NebulaGraph）
- **社区检测计算量**：Leiden 算法在十亿边规模上可能需要数小时
- **社区摘要存储**：每个摘要平均 500 tokens，500 万个社区约 2.5B tokens

## 2.6 本章小结

本章从传统 RAG 的六阶段流水线出发，系统性地分析了每个环节的设计原理与实现策略。我们从 Chunk 的四种策略（Fixed-Size、Semantic、Document-Aware、Recursive）切入，讨论了 Embedding 模型的维度权衡和向量数据库的选型依据，然后深入分析了稠密、稀疏和混合三种检索方法的互补关系。

在此基础上，我们引入了 GraphRAG 这一知识图谱增强范式，详细介绍了其索引流水线的七个步骤——从 Text Units 的划分到实体关系抽取、从 Leiden 社区检测到社区摘要生成。四种搜索方法（Local / Global / DRIFT / Basic）分别对应不同的查询类型，而自动路由机制则在实际应用中提供了智能化的方法选择。

**核心认知**：传统 RAG 和 GraphRAG 不是替代关系，而是互补关系。传统 RAG 在处理精确事实检索时效率极高，而 GraphRAG 在处理关系推理和全局综合时具有不可替代的优势。在实践中，一个成熟的 RAG 系统应当同时支持两种范式，并根据查询特征动态选择最优策略。

下一章将深入 GraphRAG 的工程实现，包括完整的代码示例、API 设计模式和性能调优策略。

---

## 附录：关键术语表

| 英文术语 | 中文翻译 | 简要说明 |
|----------|----------|----------|
| Chunk | 文本片段 / 分块 | 文档被分割后的最小检索单元 |
| Embedding | 向量嵌入 / 嵌入 | 文本到稠密向量的映射 |
| ANN (Approximate Nearest Neighbor) | 近似最近邻搜索 | 向量检索的核心算法 |
| HNSW (Hierarchical Navigable Small World) | 分层可导航小世界图 | 高精度 ANN 索引算法 |
| IVF (Inverted File Index) | 倒排文件索引 | 基于聚类的 ANN 索引 |
| PQ (Product Quantization) | 乘积量化 | 向量压缩技术 |
| RRF (Reciprocal Rank Fusion) | 倒数排名融合 | 多检索结果融合方法 |
| Cross-Encoder | 交叉编码器 | 对输入对进行相关性评分的模型 |
| Bi-Encoder | 双编码器 | 独立编码查询和文档的模型 |
| Community Detection | 社区检测 | 将图划分为语义子图的算法 |
| Modularity | 模块度 | 衡量图划分质量的核心指标 |
| Leiden Algorithm | Leiden 算法 | GraphRAG 使用的社区检测算法 |
| Map-Reduce | 映射-归约 | 先并行处理再聚合汇总的模式 |
| Entity Resolution | 实体解析 / 实体消歧 | 识别和合并指向同一实体的不同名称 |
| Text Unit | 文本单元 | GraphRAG 中最小的可溯源文本片段 |
| Community Summary | 社区摘要 | 社区中实体和关系的 LLM 生成摘要 |
| Multi-hop Reasoning | 多跳推理 | 需要跨越多个信息片段进行推导的推理任务 |
