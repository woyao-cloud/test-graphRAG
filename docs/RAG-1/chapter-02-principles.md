# 第2章 实现原理与传统RAG区别

## 2.1 传统RAG架构详解

传统RAG的完整数据流可以分解为以下步骤：

```
文档加载 → 文本分块 → 向量嵌入 → 索引存储 → 查询检索 → 答案生成
```

### 2.1.1 文档加载（Document Loading）

文档加载是RAG流水线的起点。企业环境中的文档格式多种多样，需要不同的解析策略：

- **PDF文档**：使用PyMuPDF（fitz）、pdfplumber或Unstructured库进行解析。对于扫描件，需要OCR支持（如paddleocr或Tesseract）。
- **Word/PPT文档**：使用python-docx和python-pptx库解析，或通过Unstructured进行统一处理。
- **HTML网页**：使用BeautifulSoup或trafilatura提取正文内容，去除导航、广告等噪声。
- **Markdown/纯文本**：直接读取，保留标题层级信息用于后续的文档感知分块。
- **数据库记录**：通过SQL查询将结构化数据导出为文本描述。
- **代码仓库**：使用tree-sitter解析代码结构，按函数、类进行组织。

```python
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from unstructured.partition.pdf import partition_pdf

# 使用Unstructured加载PDF（支持复杂布局和表格）
elements = partition_pdf(
    "document.pdf",
    strategy="hi_res",           # 高分辨率OCR模式
    extract_images_in_pdf=True,  # 提取图片中的文字
    infer_table_structure=True,  # 推断表格结构
)
```

### 2.1.2 文本分块策略

文本分块是RAG系统中最关键的预处理步骤之一。分块策略的选择直接影响检索质量和最终答案的准确性。不同的分块策略适用于不同的文档类型和查询场景。

#### 固定大小分块（Fixed-Size Chunking）

最简单的分块策略：按固定的token数量（或字符数）切割文本，通常设置一个重叠窗口来避免上下文在切割点处断裂。

```python
def fixed_size_chunk(text, chunk_size=512, overlap=50):
    """固定大小分块"""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    
    tokens = tokenizer.encode(text)
    chunks = []
    
    i = 0
    while i < len(tokens):
        chunk_tokens = tokens[i:i + chunk_size]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
        i += (chunk_size - overlap)
    
    return chunks
```

**优点**：实现简单、性能可预测、计算开销低。
**缺点**：可能切断语义完整的段落或句子，导致检索到的片段缺乏上下文。

#### 语义分块（Semantic Chunking）

基于嵌入向量的相似度检测文本中的语义边界，在主题变化处进行分割。这种方法能更好地保持每个块的主题一致性。

```python
def semantic_chunk(text, embedding_model, threshold=0.7):
    """基于嵌入相似度的语义分块"""
    sentences = split_into_sentences(text)
    if len(sentences) <= 1:
        return sentences
    
    chunks = []
    current_chunk = [sentences[0]]
    
    for i in range(1, len(sentences)):
        # 计算当前句子与上一个句子的语义相似度
        emb_current = embedding_model.encode(sentences[i])
        emb_previous = embedding_model.encode(sentences[i-1])
        similarity = cosine_similarity(emb_current, emb_previous)
        
        if similarity < threshold:
            # 语义变化，结束当前块
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]
        else:
            current_chunk.append(sentences[i])
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks
```

**优点**：语义边界更自然，块内主题一致性强。
**缺点**：计算开销较大，阈值需要针对具体文档调整。

#### 文档感知分块（Document-Aware Chunking）

尊重文档的原有结构——按Markdown标题、HTML标签层级或PDF的章节划分进行分割。

```python
from langchain.text_splitter import MarkdownHeaderTextSplitter

headers_to_split_on = [
    ("#", "H1"),
    ("##", "H2"),
    ("###", "H3"),
]

markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on
)
chunks = markdown_splitter.split_text(markdown_document)
```

**优点**：保持文档的层次结构，检索到的片段自带章节上下文。
**缺点**：对无结构文档不适用，嵌套过深可能导致块太小。

#### 递归分块（Recursive Chunking）

LangChain采用的默认策略：按优先级从高到低的分隔符列表递归分割。先尝试用段落分隔，如果块太大则降级到句子分隔，再降级到字符分隔。

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""]
)

chunks = text_splitter.split_text(document)
```

**优点**：适应性强，能处理多种文档格式，是经过广泛验证的通用策略。
**缺点**：可能在高嵌套结构下产生不均匀的块大小。

### 2.1.3 嵌入模型选择

分块后的文本需要转换为高维向量表示。选择嵌入模型时需要权衡维度大小、语言支持、推理性能和成本。

| 模型 | 维度 | 最大Token | 语言 | MTEB | 费用 |
|------|------|-----------|------|------|------|
| text-embedding-3-large | 3072 | 8191 | 多语言 | 64.6 | API付费 |
| text-embedding-3-small | 1536 | 8191 | 多语言 | 62.3 | API付费 |
| bge-m3 | 1024 | 8192 | 多语言 | 63.3 | 开源 |
| bge-large-zh-v1.5 | 1024 | 512 | 中文优先 | 62.8 | 开源 |
| nomic-embed-text-v1.5 | 768 | 8192 | 多语言 | 62.4 | 开源 |
| all-MiniLM-L6-v2 | 384 | 256 | 英文 | 58.8 | 开源 |

```python
from sentence_transformers import SentenceTransformer
import numpy as np

# 加载中文嵌入模型
model = SentenceTransformer('BAAI/bge-large-zh-v1.5')

def embed_chunks(chunks):
    """批量生成嵌入向量"""
    embeddings = model.encode(
        chunks,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True  # 归一化后可直接用点积计算相似度
    )
    return embeddings

chunks = ["什么是RAG？", "RAG是检索增强生成..."]
embeddings = embed_chunks(chunks)
print(f"嵌入维度: {embeddings.shape[1]}")  # 1024
print(f"嵌入数量: {embeddings.shape[0]}")   # 2
```

### 2.1.4 向量数据库

向量数据库负责存储嵌入向量并提供高效的相似度搜索，是RAG系统的核心基础设施组件。

#### LanceDB

LanceDB是基于列式存储格式Lance构建的向量数据库，采用零拷贝架构，无需独立的数据库服务器。

```python
import lancedb
import pyarrow as pa

db = lancedb.connect("./lancedb_data")

schema = pa.schema([
    pa.field("vector", pa.list_(pa.float32(), 1024)),
    pa.field("text", pa.string()),
    pa.field("source", pa.string()),
])

table = db.create_table("documents", schema=schema, mode="overwrite")
table.add([
    {"vector": emb, "text": text, "source": src}
    for emb, text, src in zip(embeddings, texts, sources)
])
table.create_index(metric="cosine", num_partitions=256, num_sub_vectors=96)
results = table.search(query_vector).limit(10).to_pandas()
```

#### FAISS

FAISS是Meta开发的高性能向量检索库，提供业界最先进的ANN算法实现。

```python
import faiss
import numpy as np

dimension = 1024
index = faiss.IndexFlatIP(dimension)
# 使用HNSW加速
hnsw_index = faiss.IndexHNSWFlat(dimension, 32)
hnsw_index.hnsw.efConstruction = 200
hnsw_index.hnsw.efSearch = 64
hnsw_index.add(embeddings_np)

D, I = hnsw_index.search(query_vector.reshape(1, -1), k=10)
```

#### 向量数据库对比

| 特性 | LanceDB | FAISS | Milvus | ChromaDB |
|------|---------|-------|--------|----------|
| 存储模式 | 磁盘（mmap） | 内存 | 磁盘+内存 | 磁盘+内存 |
| 分布式支持 | 否 | 否 | 是 | 否 |
| GPU加速 | 否 | 是 | 是 | 否 |
| 过滤搜索 | 是 | 有限 | 是 | 是 |
| 部署复杂度 | 低 | 低 | 高 | 低 |
| 适用场景 | 单机生产 | 高性能检索 | 大规模生产 | 原型开发 |

### 2.1.5 检索方法

#### 稠密检索（Dense Retrieval）

使用嵌入向量的余弦相似度进行语义匹配，能理解同义词和语义近似。

```python
def dense_retrieval(query, table, top_k=10):
    query_emb = model.encode(query, normalize_embeddings=True)
    results = table.search(query_emb).limit(top_k).to_pandas()
    return results
```

#### 稀疏检索（Sparse Retrieval - BM25）

基于词频和逆文档频率的关键词匹配，对精确术语匹配更有效。

```python
from rank_bm25 import BM25Okapi

def build_bm25_index(corpus):
    tokenized_corpus = [tokenize(doc) for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25

def sparse_retrieval(query, bm25, top_k=10):
    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return top_indices, scores[top_indices]
```

#### 混合检索（Hybrid Retrieval）

结合稠密和稀疏检索的优点，使用RRF（Reciprocal Rank Fusion）合并结果。

```python
def hybrid_retrieval(query, dense_fn, sparse_fn, top_k=10):
    dense_results = dense_fn(query, top_k * 2)
    sparse_results = sparse_fn(query, top_k * 2)
    
    rrf_scores = {}
    for rank, doc_id in enumerate(dense_results):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (60 + rank)
    for rank, doc_id in enumerate(sparse_results):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (60 + rank)
    
    reranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in reranked[:top_k]]
```

### 2.1.6 生成阶段

生成阶段的关键是将检索结果有效整合到LLM的上下文中。

```python
def generate_answer(query, retrieved_docs, llm):
    context = ""
    for i, doc in enumerate(retrieved_docs):
        context += f"[文档 {i+1}] {doc['source']}:\n{doc['text']}\n\n"
    
    system_prompt = """请基于参考文档回答问题。如果参考文档不足以回答，
请明确告知用户。在回答中引用相关的文档编号。"""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"参考文档：\n{context}\n\n问题：{query}"}
    ]
    
    response = llm.chat(messages)
    return response
```

## 2.2 GraphRAG核心概念

### 2.2.1 实体、关系、社区与社区报告

GraphRAG的核心创新在于从文档中提取结构化的知识图谱，而非仅保留原始的文本片段。这种知识表示方式为深度语义理解和多跳推理提供了基础。

- **实体（Entity）**：文档中提到的具体对象，如人名、组织名、地点、时间、概念等。每个实体包含名称、类型和描述。
- **关系（Relationship）**：实体之间的语义关联，如"工作于"、"位于"、"属于"、"导致"等。每条关系包含源实体、目标实体、关系类型和描述。
- **社区（Community）**：通过社区检测算法发现的紧密关联的实体群组。同一社区内的实体在语义上高度相关。
- **社区报告（Community Report）**：对每个社区生成的结构化摘要，包含社区主题、关键实体、核心关系和综合描述。

### 2.2.2 GraphRAG vs 传统RAG

| 维度 | 传统RAG | GraphRAG |
|------|---------|----------|
| 知识表示 | 文本片段（Chunks） | 实体-关系图（KG） |
| 检索粒度 | 整段文本 | 实体、关系、社区摘要 |
| 关系理解 | 弱（依赖LLM推断） | 强（图结构显式编码） |
| 多跳推理 | 困难 | 自然（沿图遍历） |
| 全局性问题 | 困难 | 强（社区摘要提供全局视角） |
| 索引复杂度 | 低（分块+嵌入） | 高（实体提取+图谱构建） |
| 查询延迟 | 低 | 中高 |
| 适用场景 | 事实型问答 | 全局理解、关系分析 |

### 2.2.3 GraphRAG索引流水线

GraphRAG的索引过程分为六个阶段：

```
加载 → 分块 → 图谱提取 → 图谱合并 → 社区检测 → 嵌入索引
```

**阶段1-2：加载与分块**

与传统RAG类似，但GraphRAG的分块通常较小（300-600 tokens），以便精确提取实体和关系。

**阶段3：图谱提取**

使用LLM从每个文本块中提取实体和关系。这是GraphRAG最核心也最昂贵的步骤。

```python
def extract_entities_and_relations(text_chunk, llm):
    prompt = f"""从以下文本中提取实体和关系。

文本：{text_chunk}

输出JSON格式：
{{
  "entities": [
    {{"name": "...", "type": "...", "description": "..."}}
  ],
  "relationships": [
    {{"source": "...", "target": "...", "relation": "...", "description": "..."}}
  ]
}}"""
    
    response = llm.generate(prompt, response_format={"type": "json_object"})
    return parse_entities_and_relations(response)
```

**阶段4：图谱合并**

将所有文本块中提取的实体和关系合并为统一的图谱，处理别名和消歧。

```python
def merge_graphs(chunk_graphs):
    merged = {"entities": {}, "relationships": []}
    
    for graph in chunk_graphs:
        for entity in graph["entities"]:
            name = entity["name"]
            if name in merged["entities"]:
                merged["entities"][name]["description"] += " | " + entity["description"]
            else:
                merged["entities"][name] = entity
        
        for rel in graph["relationships"]:
            exists = any(
                r["source"] == rel["source"] and r["target"] == rel["target"]
                for r in merged["relationships"]
            )
            if not exists:
                merged["relationships"].append(rel)
    
    return merged
```

**阶段5：社区检测**

使用Leiden算法对实体图进行社区检测。Leiden算法通过三个阶段优化模块度：局部移动（将节点移动到邻居社区）、细化（社区内部进一步分区）、聚合（合并为超级节点）。

```python
import igraph as ig

def detect_communities(graph):
    g = ig.Graph()
    g.add_vertices(list(graph["entities"].keys()))
    edges = [(r["source"], r["target"]) for r in graph["relationships"]]
    g.add_edges(edges)
    
    partition = g.community_leiden(
        objective="modularity",
        n_iterations=-1
    )
    
    communities = {}
    for idx, cid in enumerate(partition.membership):
        if cid not in communities:
            communities[cid] = []
        communities[cid].append(g.vs[idx]["name"])
    
    return communities
```

**阶段6：社区摘要与嵌入**

对每个社区生成结构化摘要，然后将实体描述、关系描述和社区报告分别向量化。

```python
def summarize_communities(graph, communities, llm):
    reports = {}
    for cid, entities in communities.items():
        prompt = f"""分析以下知识图谱社区的实体和关系。

实体：{json.dumps({e: graph["entities"][e] for e in entities}, ensure_ascii=False)}
关系：{json.dumps([r for r in graph["relationships"] if r["source"] in entities or r["target"] in entities], ensure_ascii=False)}

生成摘要包含：1）社区主题 2）核心关系模式 3）关键实体"""
        
        reports[cid] = {"entities": entities, "summary": llm.generate(prompt)}
    return reports
```

## 2.3 GraphRAG的四种搜索方法

### 2.3.1 Local Search（本地搜索）

针对具体事实型问题，通过实体扩展、文本检索和社区上下文的多路融合策略。

**搜索流程**：
1. 将用户查询与实体索引匹配，找到相关实体
2. 沿图结构扩展相关实体的邻居（1-2跳）
3. 检索扩展实体的原始文本片段
4. 获取相关社区报告作为全局上下文
5. 融合所有信息后交给LLM生成回答

**适用场景**："DeepSeek-R1的训练数据规模是多少？"——针对具体实体的事实型问题。

### 2.3.2 Global Search（全局搜索）

针对需要综合理解的全局性问题，通过社区报告的Map-Reduce实现。

**搜索流程**：
1. 将用户查询与社区报告索引匹配
2. 筛选最相关的社区报告（top-5到top-20）
3. 对每个社区报告独立生成中间回答（Map）
4. 将所有中间回答融合为最终答案（Reduce）

**适用场景**："GraphRAG系统的主要设计原则是什么？"——需要综合多个知识片段的全局性问题。

### 2.3.3 DRIFT Search（DRIFT搜索）

通过层次化社区遍历实现多跳推理。

**搜索流程**：
1. 从用户查询出发，识别初始实体
2. 在实体所在社区内部进行图遍历
3. 信息不足时跳转到相关社区继续探索
4. 动态维护探索状态，记录已访问的实体和社区
5. 收集到足够信息后生成回答

**适用场景**："DeepSeek的API定价如何影响其在中小企业中的采用？"——需要多跳推理的问题。

### 2.3.4 Basic Search（基础搜索）

最简单的检索模式，等同于传统RAG的向量相似度搜索。

**适用场景**：简单的单事实检索问题。

### 2.3.5 查询引擎自动路由

```python
def auto_route_query(query, graphrag_index, llm):
    prompt = f"""判断问题类型：
A) 具体事实型  B) 全局综合型  C) 多跳推理型  D) 简单查询型

问题：{query}
只输出类别字母。"""
    
    qtype = llm.generate(prompt).strip()
    routes = {"A": local_search, "B": global_search,
              "C": drift_search, "D": basic_search}
    return routes.get(qtype, local_search)(query, graphrag_index)
```

## 2.4 完整RAG流水线示例

以下是一个使用LanceDB和DeepSeek的完整RAG实现：

```python
import lancedb
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI

EMBEDDING_MODEL = "BAAI/bge-large-zh-v1.5"
LLM_BASE_URL = "http://localhost:11434/v1"
LLM_MODEL = "deepseek-r1:32b"

class SimpleRAG:
    def __init__(self, db_path="./rag_db"):
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        self.db = lancedb.connect(db_path)
        self.llm = OpenAI(base_url=LLM_BASE_URL, api_key="ollama")
    
    def ingest(self, documents):
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
        
        chunks, embs = [], []
        for doc in documents:
            parts = splitter.split_text(doc["text"])
            embeddings = self.embedder.encode(parts, normalize_embeddings=True)
            chunks.extend(parts)
            embs.extend(embeddings)
        
        table = self.db.create_table("kb", mode="overwrite")
        table.add([{"vector": e, "text": c} for e, c in zip(embs, chunks)])
        table.create_index(metric="cosine")
        return len(chunks)
    
    def retrieve(self, query, top_k=5):
        qemb = self.embedder.encode(query, normalize_embeddings=True)
        return self.db.open_table("kb").search(qemb).limit(top_k).to_pandas()["text"].tolist()
    
    def generate(self, query, docs):
        context = "\n\n".join(f"[{i+1}] {d}" for i, d in enumerate(docs))
        msgs = [
            {"role": "system", "content": "基于参考信息回答问题，引用来源编号。"},
            {"role": "user", "content": f"参考信息：\n{context}\n\n问题：{query}"}
        ]
        resp = self.llm.chat.completions.create(
            model=LLM_MODEL, messages=msgs, temperature=0.1, stream=True
        )
        return "".join(c.choices[0].delta.content or ""
                       for c in resp if c.choices[0].delta.content)
    
    def query(self, query):
        docs = self.retrieve(query)
        return self.generate(query, docs), docs

# 使用示例
rag = SimpleRAG()
rag.ingest([
    {"text": "GraphRAG是微软研究院提出的基于知识图谱的RAG系统。"},
    {"text": "LanceDB是一个基于列式存储的向量数据库。"},
])
answer, sources = rag.query("GraphRAG和传统RAG有什么区别？")
print(f"答案：{answer}")
print(f"参考来源：{len(sources)}个文档")
```

## 2.5 实体提取方法对比

### 2.5.1 LLM-based实体提取

使用大语言模型进行实体提取，支持复杂实体类型和关系抽取。

```python
def llm_entity_extraction(text, llm):
    prompt = f"""提取文本中的所有实体，包括：
- 人物、组织、地点、产品、概念
- 每个实体标注类型和简要描述

文本：{text}

以JSON格式输出实体列表。"""
    return llm.generate(prompt, response_format={"type": "json_object"})
```

**优点**：理解能力强，支持开放类型，准确率高。
**缺点**：成本高，速度慢，存在幻觉风险。

### 2.5.2 NLP-based实体提取

使用传统NLP工具进行实体提取。

```python
import spacy

nlp = spacy.load("zh_core_web_trf")

def nlp_entity_extraction(text):
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        entities.append({
            "name": ent.text,
            "type": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char
        })
    return entities
```

**优点**：速度快，成本低，结果稳定。
**缺点**：只支持预定义类型，对中文复杂实体识别能力有限。

在实践中，GraphRAG通常采用LLM-based提取为主、NLP-based提取为辅的混合策略，在保持质量的同时控制成本。
