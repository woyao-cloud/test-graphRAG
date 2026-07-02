# 第5章 多路召回体系

## 5.1 为什么需要多路召回

### 5.1.1 单一检索方法的局限性

在RAG系统中，检索（Retrieval）是整个流程的入口环节，其质量直接决定了后续生成的上限。然而，任何一种单一的检索方法都存在固有的局限性：

| 检索方法 | 优势 | 局限性 |
|---------|------|--------|
| 向量检索（Dense） | 语义匹配强，能处理同义词、 paraphrasing | 对精确关键词匹配弱，受 embedding 质量影响大，冷启动需要训练 |
| 关键词检索（Sparse） | 精确匹配强，BM25在 exact term 上表现稳定 | 无法处理语义相似但用词不同的查询，对同义词不敏感 |
| 结构化检索 | 精确过滤，元数据条件匹配 | 无法做语义匹配，依赖数据的结构化程度 |
| 知识图谱检索 | 关系推理，多跳关联查询 | 图谱覆盖度有限，构建成本高 |

**问题场景举例**：

- **语义相同但用词不同**：用户搜索"肺癌治疗方法"，文档中写的是"肺腺癌化疗方案"——向量检索可以匹配，BM25会漏掉
- **精确关键词必须命中**：用户搜索"API版本v3.1.5"，向量检索可能返回"v3.2.0"的内容，BM25才能精确匹配版本号
- **时效性要求**：用户问"2025年Q3财报"，需要结构化过滤时间范围，纯语义检索无法感知时间
- **实体关系查询**："阿斯利康的哪些药物被用于治疗非小细胞肺癌"——需要知识图谱的路径遍历

### 5.1.2 多路召回的互补性

多路召回（Multi-Recall / Multi-Path Retrieval）的核心思想是：**从多个独立的检索通道（Recaller）中分别获取候选文档，然后通过融合策略合并结果**。每个通道擅长不同的匹配维度，组合起来形成更全面的覆盖。

```
用户查询
    │
    ├──→ [Dense Recaller]  ──→ 向量检索结果集 A
    ├──→ [Sparse Recaller] ──→ 关键词检索结果集 B
    ├──→ [Structured Recaller] ──→ 结构化过滤结果集 C
    └──→ [KG Recaller]    ──→ 图谱检索结果集 D
    │
    └──→ [Rerank & Fusion] ──→ 最终排序结果
```

**互补性收益**：

1. **召回率提升**：多个通道覆盖不同的匹配维度，漏检的概率显著降低
2. **鲁棒性增强**：当某个通道质量下降（如 embedding 模型更新、索引迁移），其他通道仍能保障基本效果
3. **多样化结果**：不同通道返回的文档往往具有不同的信息角度，为 LLM 提供更丰富的上下文

### 5.1.3 多路召回的设计原则

设计多路召回系统时需要遵循以下原则：

- **通道独立性**：每个 Recaller 应该独立运行，一个通道的失败不应影响其他通道
- **结果可融合**：各通道返回的结果需要具有可比性（如都有评分或排序位置）
- **可扩展性**：方便新增或移除召回通道，不影响整体架构
- **延迟可控**：多路召回意味着多个串行或并行的检索调用，需要控制总延迟在可接受范围

---

## 5.2 向量检索（Dense Retrieval）

### 5.2.1 Embedding 模型选择

向量检索的核心是将文本映射到高维语义空间。选择合适的 embedding 模型是效果的关键。以下是主流的 embedding 模型对比：

| 模型 | 维度 | 最大输入长度 | 语言支持 | 适用场景 |
|------|------|-------------|---------|---------|
| BGE-M3 (BAAI) | 1024 | 8192 tokens | 多语言（中/英/其他） | 企业级多语言场景，密集+稀疏混合 |
| text-embedding-3-large (OpenAI) | 3072 / 可降维 | 8191 tokens | 多语言 | 云端 API 调用，高精度 |
| text-embedding-3-small (OpenAI) | 1536 | 8191 tokens | 多语言 | 成本敏感，效率优先 |
| nomic-embed-text-v1.5 (Nomic) | 768 | 8192 tokens | 多语言 | 本地部署，开源，轻量 |
| GTE-Qwen2 (Alibaba) | 4096 | 8192 tokens | 中英双语 | 大模型生态整合 |
| jina-embeddings-v3 (Jina AI) | 1024 | 8192 tokens | 多语言 | 任务特定 embedding |

**维度选择权衡**：

- 高维度（3072）：语义表达能力更强，但存储和计算成本更高
- 低维度（768）：检索速度快，存储空间小，但可能丢失细粒度语义
- 实践中，1024 维是多数企业场景的黄金平衡点

**代码示例：使用 BGE-M3 生成 embedding**：

```python
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List

class EmbeddingService:
    """Embedding 生成服务"""
    
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"[Embedding] 模型: {model_name}, 维度: {self.dimension}")
    
    def encode(self, texts: List[str], 
               normalize: bool = True,
               batch_size: int = 32) -> np.ndarray:
        """
        生成文本的 embedding 向量
        
        Args:
            texts: 文本列表
            normalize: 是否 L2 归一化（cosine 相似度需要）
            batch_size: 批处理大小
            
        Returns:
            shape=(n, dim) 的 numpy 数组
        """
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=normalize,
            batch_size=batch_size,
            show_progress_bar=True
        )
        return np.array(embeddings)
    
    def encode_query(self, query: str) -> np.ndarray:
        """为查询生成 embedding（BGE 需要添加 instruction）"""
        # BGE 系列模型建议为 query 添加前缀
        instruction = "为这个句子生成表示以用于检索相关文章："
        text_with_instruction = f"{instruction} {query}"
        return self.encode([text_with_instruction])[0]


# 使用示例
embedder = EmbeddingService("BAAI/bge-m3")
docs = [
    "肺癌的早期诊断方法包括低剂量CT筛查",
    "靶向治疗药物奥希替尼用于EGFR突变非小细胞肺癌",
    "免疫检查点抑制剂在肺癌治疗中的应用"
]
doc_embeddings = embedder.encode(docs)
query_embedding = embedder.encode_query("非小细胞肺癌治疗方案")
print(f"文档 embedding 形状: {doc_embeddings.shape}")  # (3, 1024)
print(f"查询 embedding 形状: {query_embedding.shape}")  # (1024,)
```

### 5.2.2 向量数据库选型

向量数据库负责存储 embedding 并支持高效的近似最近邻搜索（ANN）。以下是主流向量数据库的对比：

| 特性 | LanceDB | FAISS | Milvus | Qdrant | Chroma |
|------|---------|-------|--------|--------|--------|
| 部署模式 | 嵌入式/本地 | 库（无服务器） | 分布式服务 | 单机/分布式 | 嵌入式 |
| 索引类型 | IVF-PQ, HNSW | IVF, HNSW, PQ | IVF, HNSW, DiskANN | HNSW | HNSW |
| 持久化 | 原生列式存储 | 需自行管理 | 内置 | 内置 | SQLite |
| 分布式 | 否 | 否 | 是 | 是 | 否 |
| 过滤能力 | 强（列式过滤） | 有限 | 强 | 强 | 基础 |
| 云原生 | 否 | 否 | 是 | 是 | 否 |
| 适用场景 | 本地/单机项目 | 高性能计算 | 生产级大规模 | 生产级中小规模 | 快速原型 |

**代码示例：LanceDB 向量检索**：

```python
import lancedb
import numpy as np
from typing import List, Dict, Any

class LanceDBVectorStore:
    """基于 LanceDB 的向量存储与检索"""
    
    def __init__(self, db_path: str = "./lancedb_data"):
        self.db = lancedb.connect(db_path)
    
    def create_index(self, table_name: str, 
                     vectors: np.ndarray,
                     documents: List[str],
                     metadata: List[Dict[str, Any]] = None):
        """
        创建向量索引表
        
        Args:
            table_name: 表名
            vectors: 向量数组 shape=(n, dim)
            documents: 原始文档列表
            metadata: 可选的元数据列表
        """
        if metadata is None:
            metadata = [{}] * len(documents)
        
        data = []
        for i, (vec, doc, meta) in enumerate(zip(vectors, documents, metadata)):
            data.append({
                "id": i,
                "vector": vec.tolist(),
                "text": doc,
                **meta
            })
        
        self.db.create_table(table_name, data, mode="overwrite")
        
        # 创建 HNSW 索引
        tbl = self.db.open_table(table_name)
        tbl.create_index(
            metric="cosine",        # 距离度量
            num_partitions=256,     # IVF 分区数
            num_sub_vectors=96      # PQ 子向量数
        )
        print(f"[LanceDB] 索引创建完成: {table_name}, 文档数: {len(documents)}")
    
    def search(self, table_name: str, 
               query_vector: np.ndarray,
               top_k: int = 10,
               filter_expr: str = None) -> List[Dict]:
        """
        向量检索
        
        Args:
            table_name: 表名
            query_vector: 查询向量
            top_k: 返回 top-k 结果
            filter_expr: 过滤表达式，如 "category = 'medical'"
            
        Returns:
            检索结果列表
        """
        tbl = self.db.open_table(table_name)
        
        # 构建查询
        query = tbl.search(query_vector.tolist()) \
                    .metric("cosine") \
                    .limit(top_k)
        
        if filter_expr:
            query = query.where(filter_expr)
        
        results = query.to_list()
        return results
```

### 5.2.3 相似度度量方法

向量检索中的相似度度量决定了向量空间中的"距离"含义。三种主流度量方式：

**余弦相似度（Cosine Similarity）**：
```
cosine(q, d) = (q · d) / (||q|| × ||d||)
```

- 值域：[-1, 1]，值越大越相似
- 关注向量的方向而非长度
- 要求向量预先 L2 归一化
- 最常用于文本 embedding

**L2 距离（欧几里得距离）**：
```
L2(q, d) = sqrt(Σ(qi - di)²)
```

- 值域：[0, +∞)，值越小越相似
- 关注向量的绝对位置
- 对向量长度敏感

**内积（Inner Product / Dot Product）**：
```
IP(q, d) = Σ(qi × di)
```

- 值域：(-∞, +∞)，值越大越相似
- 当向量已归一化时，等价于余弦相似度
- 某些索引（如 HNSW）对内积优化更好

**选择指南**：

```python
import numpy as np
from typing import List

def compute_similarity(query_vector: np.ndarray,
                       document_vectors: np.ndarray,
                       metric: str = "cosine") -> np.ndarray:
    """
    计算查询与文档集合的相似度
    
    Args:
        query_vector: 查询向量 (dim,)
        document_vectors: 文档向量 (n, dim)
        metric: 度量方式: "cosine" | "l2" | "inner_product"
        
    Returns:
        相似度分数 (n,)
    """
    if metric == "cosine":
        # 确保归一化
        q_norm = query_vector / np.linalg.norm(query_vector)
        d_norm = document_vectors / np.linalg.norm(document_vectors, axis=1, keepdims=True)
        scores = np.dot(d_norm, q_norm)
        
    elif metric == "l2":
        # L2 距离（转换为相似度：值越小越相似）
        distances = np.linalg.norm(document_vectors - query_vector, axis=1)
        scores = -distances  # 负数，越大越相似
        
    elif metric == "inner_product":
        scores = np.dot(document_vectors, query_vector)
        
    else:
        raise ValueError(f"不支持的度量方式: {metric}")
    
    return scores


def normalize_vector(v: np.ndarray) -> np.ndarray:
    """L2 归一化向量"""
    norm = np.linalg.norm(v)
    if norm > 0:
        return v / norm
    return v
```

### 5.2.4 HNSW 索引参数调优

HNSW（Hierarchical Navigable Small World）是目前最流行的 ANN 索引算法。它通过构建多层图结构实现高效的近似最近邻搜索。

**核心参数**：

| 参数 | 作用 | 取值范围 | 推荐值 | 影响 |
|------|------|---------|-------|------|
| M | 每个节点的最大连接数 | 4~128 | 16~64 | M 越大，召回率越高，内存占用越大 |
| efConstruction | 构建时的动态列表大小 | 100~1000 | 200~500 | 越大，索引质量越好，构建越慢 |
| efSearch | 搜索时的动态列表大小 | 50~2000 | 100~500 | 越大，召回率越高，搜索越慢 |

**参数调优代码**：

```python
import time
import numpy as np
from typing import Dict, List

def benchmark_hnsw_params(
    dim: int = 1024,
    n_docs: int = 100000,
    n_queries: int = 100,
    m_values: List[int] = [8, 16, 32, 64],
    ef_construction_values: List[int] = [100, 200, 400],
    ef_search_values: List[int] = [50, 100, 200, 500],
    k: int = 10
) -> Dict:
    """
    HNSW 参数基准测试
    
    Args:
        dim: 向量维度
        n_docs: 文档数量
        n_queries: 查询数量
        m_values: M 参数候选值
        ef_construction_values: efConstruction 候选值
        ef_search_values: efSearch 候选值
        k: top-k
        
    Returns:
        各参数组合的性能指标
    """
    from sklearn.neighbors import NearestNeighbors
    
    # 生成随机数据
    np.random.seed(42)
    documents = np.random.randn(n_docs, dim).astype(np.float32)
    documents = documents / np.linalg.norm(documents, axis=1, keepdims=True)
    
    queries = np.random.randn(n_queries, dim).astype(np.float32)
    queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    
    # 暴力搜索作为 ground truth
    print("计算 ground truth (暴力搜索)...")
    brute_force = NearestNeighbors(n_neighbors=k, metric="cosine")
    brute_force.fit(documents)
    gt_distances, gt_indices = brute_force.kneighbors(queries)
    
    results = []
    
    for m in m_values:
        for ef_c in ef_construction_values:
            for ef_s in ef_search_values:
                import hnswlib
                
                # 构建索引
                index = hnswlib.Index(space="cosine", dim=dim)
                index.init_index(
                    max_elements=n_docs,
                    ef_construction=ef_c,
                    M=m,
                    random_seed=42
                )
                
                t_start = time.time()
                index.add_items(documents, np.arange(n_docs))
                build_time = time.time() - t_start
                
                # 设置搜索参数
                index.set_ef(ef_s)
                
                # 搜索
                t_start = time.time()
                labels, distances = index.knn_query(queries, k=k)
                search_time = time.time() - t_start
                
                # 计算召回率
                recalls = []
                for i in range(n_queries):
                    gt_set = set(gt_indices[i])
                    pred_set = set(labels[i])
                    recall = len(gt_set & pred_set) / k
                    recalls.append(recall)
                
                avg_recall = np.mean(recalls)
                avg_search_time_ms = (search_time / n_queries) * 1000
                
                results.append({
                    "M": m,
                    "efConstruction": ef_c,
                    "efSearch": ef_s,
                    "Recall@10": f"{avg_recall:.4f}",
                    "BuildTime(s)": f"{build_time:.2f}",
                    "AvgSearchTime(ms)": f"{avg_search_time_ms:.2f}",
                    "Memory(MB)": index.get_memory_usage() / (1024 * 1024)
                })
                
                print(f"M={m}, efC={ef_c}, efS={ef_s} -> "
                      f"Recall={avg_recall:.4f}, "
                      f"Search={avg_search_time_ms:.2f}ms")
    
    return results


def select_hnsw_params(recall_target: float = 0.95,
                       latency_target_ms: float = 50.0) -> Dict:
    """
    根据目标召回率和延迟自动选择 HNSW 参数
    
    经验规则：
    - 召回率要求高（>0.98）：M=64, efConstruction=400, efSearch=500
    - 平衡场景（~0.95）：M=32, efConstruction=200, efSearch=200
    - 低延迟场景（<10ms）：M=16, efConstruction=100, efSearch=100
    """
    if recall_target >= 0.98:
        return {"M": 64, "efConstruction": 400, "efSearch": 500}
    elif recall_target >= 0.95:
        return {"M": 32, "efConstruction": 200, "efSearch": 200}
    elif recall_target >= 0.90:
        return {"M": 16, "efConstruction": 100, "efSearch": 100}
    else:
        return {"M": 8, "efConstruction": 100, "efSearch": 50}
```

---

## 5.3 关键词检索（Sparse Retrieval）

### 5.3.1 BM25 算法原理

BM25（Best Matching 25）是基于概率检索框架的排序函数，是 TF-IDF 的改进版本。它的核心创新在于引入了**词频饱和**和**文档长度归一化**机制。

**BM25 公式**：

```
Score(Q, D) = Σ [IDF(qi) × TF_BM25(qi, D)]
```

其中：

**IDF 计算**：
```
IDF(qi) = log( (N - n(qi) + 0.5) / (n(qi) + 0.5) + 1 )
```

- N：文档总数
- n(qi)：包含词 qi 的文档数

**改进的 TF 计算（词频饱和）**：
```
TF_BM25(qi, D) = [tf(qi, D) × (k1 + 1)] / [tf(qi, D) + k1 × (1 - b + b × |D| / avgdl)]
```

- tf(qi, D)：词 qi 在文档 D 中的出现次数
- |D|：文档 D 的长度
- avgdl：所有文档的平均长度
- k1：控制词频饱和速度（默认 1.2~2.0）
- b：控制长度归一化强度（默认 0.75）

**BM25 vs TF-IDF 对比**：

| 特性 | TF-IDF | BM25 |
|------|--------|------|
| 词频处理 | 线性增长 | 非线性饱和（k1 控制） |
| 文档长度 | 不处理 | 长度归一化（b 控制） |
| 参数调优 | 无需参数 | 需要调优 k1、b |
| 效果稳定性 | 对长文档不友好 | 对长文档更公平 |
| 实现复杂度 | 简单 | 中等 |

**词频饱和效果**：在 TF-IDF 中，"肺癌"出现 10 次比出现 1 次得分高 10 倍。BM25 认为出现 5 次后边际收益递减，这种非线性饱和更符合直觉——一篇文章不会因为反复出现同一个词就更相关 10 倍。

### 5.3.2 纯 Python BM25 实现

```python
import math
from collections import Counter
from typing import List, Tuple

class BM25:
    """BM25 算法实现"""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Args:
            k1: 词频饱和参数，1.2~2.0，越大词频影响越大
            b: 长度归一化参数，0~1，0 为不归一化，1 为完全归一化
        """
        self.k1 = k1
        self.b = b
        self.documents: List[List[str]] = []
        self.doc_lengths: List[int] = []
        self.avgdl: float = 0.0
        self.n_docs: int = 0
        self.idf_cache: dict = {}
        self.term_freqs: List[Counter] = []
    
    def fit(self, documents: List[List[str]]):
        """
        训练 BM25 模型
        
        Args:
            documents: 分词后的文档列表，每个文档是词列表
        """
        self.documents = documents
        self.n_docs = len(documents)
        self.doc_lengths = [len(doc) for doc in documents]
        self.avgdl = sum(self.doc_lengths) / self.n_docs if self.n_docs > 0 else 0
        
        # 计算每个文档的词频
        self.term_freqs = [Counter(doc) for doc in documents]
        
        # 计算文档频率（包含每个词的文档数）
        doc_freq = Counter()
        for doc in documents:
            unique_terms = set(doc)
            doc_freq.update(unique_terms)
        
        # 计算 IDF
        self.idf_cache = {}
        for term, freq in doc_freq.items():
            # BM25 的 IDF 公式（平滑版）
            idf = math.log((self.n_docs - freq + 0.5) / (freq + 0.5) + 1.0)
            self.idf_cache[term] = idf
        
        print(f"[BM25] 训练完成: {self.n_docs} 篇文档, "
              f"词汇量: {len(self.idf_cache)}, "
              f"平均文档长度: {self.avgdl:.1f}")
    
    def score(self, query_terms: List[str], doc_index: int) -> float:
        """
        计算单个文档对查询的 BM25 得分
        
        Args:
            query_terms: 查询的分词结果
            doc_index: 文档索引
            
        Returns:
            BM25 得分
        """
        doc_length = self.doc_lengths[doc_index]
        term_freq = self.term_freqs[doc_index]
        
        score = 0.0
        for term in query_terms:
            if term not in self.idf_cache:
                continue
            
            tf = term_freq.get(term, 0)
            if tf == 0:
                continue
            
            idf = self.idf_cache[term]
            
            # BM25 TF 公式（词频饱和）
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (
                1 - self.b + self.b * doc_length / self.avgdl
            )
            tf_bm25 = numerator / denominator
            
            score += idf * tf_bm25
        
        return score
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        搜索 BM25 得分最高的文档
        
        Args:
            query: 查询文本（需提前分词）
            top_k: 返回 top-k
            
        Returns:
            [(doc_index, score), ...]
        """
        query_terms = query.split()
        scores = []
        
        for i in range(self.n_docs):
            s = self.score(query_terms, i)
            scores.append((i, s))
        
        # 按得分降序排列
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# 使用示例
bm25 = BM25(k1=1.5, b=0.75)

# 模拟分词后的文档
corpus = [
    "肺癌 早期 诊断 方法 包括 低剂量 CT 筛查".split(),
    "靶向 治疗 药物 奥希替尼 EGFR 突变 非小细胞 肺癌".split(),
    "免疫 检查点 抑制剂 肺癌 治疗 应用".split(),
    "肺腺癌 化疗 方案 铂类 药物 联合 治疗".split(),
    "肺癌 筛查 指南 推荐 高危 人群 每年 CT".split()
]

bm25.fit(corpus)

# 搜索
results = bm25.search("非小细胞肺癌 靶向 治疗", top_k=3)
for idx, score in results:
    print(f"文档 {idx}: {' '.join(corpus[idx])} -> BM25={score:.4f}")
```

### 5.3.3 Elasticsearch 全文检索集成

在生产环境中，BM25 通常通过 Elasticsearch 实现，它内置了 BM25 作为默认相似度算法。

**Elasticsearch 索引配置**：

```python
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from typing import List, Dict, Any

class ElasticsearchRetriever:
    """基于 Elasticsearch 的关键词检索"""
    
    def __init__(self, hosts: List[str] = ["http://localhost:9200"]):
        self.es = Elasticsearch(hosts)
    
    def create_index(self, index_name: str,
                     analyzer: str = "ik_max_word",
                     similarity: Dict = None):
        """
        创建带中文分词的索引
        
        Args:
            index_name: 索引名称
            analyzer: 分词器（ik_max_word 是 IK 分词器的最大粒度模式）
            similarity: BM25 参数配置
        """
        if similarity is None:
            similarity = {
                "bm25": {
                    "type": "BM25",
                    "k1": 1.2,
                    "b": 0.75
                }
            }
        
        mapping = {
            "settings": {
                "analysis": {
                    "analyzer": {
                        "ik_analyzer": {
                            "type": "custom",
                            "tokenizer": analyzer
                        }
                    }
                },
                "similarity": similarity
            },
            "mappings": {
                "properties": {
                    "content": {
                        "type": "text",
                        "analyzer": "ik_analyzer",
                        "similarity": "bm25"
                    },
                    "title": {
                        "type": "text",
                        "analyzer": "ik_analyzer",
                        "similarity": "bm25"
                    },
                    "category": {"type": "keyword"},
                    "tags": {"type": "keyword"},
                    "created_at": {"type": "date"},
                    "source": {"type": "keyword"},
                    "doc_id": {"type": "keyword"}
                }
            }
        }
        
        if self.es.indices.exists(index=index_name):
            self.es.indices.delete(index=index_name)
        
        self.es.indices.create(index=index_name, body=mapping)
        print(f"[ES] 索引创建完成: {index_name}")
    
    def bulk_index(self, index_name: str,
                   documents: List[Dict[str, Any]]):
        """
        批量索引文档
        
        Args:
            index_name: 索引名称
            documents: 文档列表，每项包含 id, content, title 等字段
        """
        actions = []
        for doc in documents:
            action = {
                "_index": index_name,
                "_id": doc.get("doc_id"),
                "_source": doc
            }
            actions.append(action)
        
        success, errors = bulk(self.es, actions)
        print(f"[ES] 批量索引: 成功 {success} 条, 失败 {len(errors)} 条")
    
    def search(self, index_name: str,
               query: str,
               fields: List[str] = None,
               top_k: int = 10,
               filter_clause: Dict = None) -> List[Dict]:
        """
        全文检索
        
        Args:
            index_name: 索引名称
            query: 查询文本
            fields: 检索字段列表，默认 ["content", "title"]
            top_k: 返回文档数
            filter_clause: 过滤条件
            
        Returns:
            检索结果列表
        """
        if fields is None:
            fields = ["content^2", "title^3"]  # title 权重更高
        
        # 构建查询
        must_clause = {
            "multi_match": {
                "query": query,
                "fields": fields,
                "type": "best_fields",
                "tie_breaker": 0.3  # 多字段匹配的融合系数
            }
        }
        
        body = {
            "query": {
                "bool": {
                    "must": [must_clause]
                }
            },
            "size": top_k,
            "highlight": {
                "fields": {
                    "content": {},
                    "title": {}
                }
            }
        }
        
        # 添加过滤条件
        if filter_clause:
            body["query"]["bool"]["filter"] = [filter_clause]
        
        response = self.es.search(index=index_name, body=body)
        
        results = []
        for hit in response["hits"]["hits"]:
            results.append({
                "doc_id": hit["_id"],
                "score": hit["_score"],
                "source": hit["_source"],
                "highlights": hit.get("highlight", {})
            })
        
        return results
```

### 5.3.4 中文分词器选择

中文没有天然的空格分隔符，分词质量直接影响 BM25 和 Elasticsearch 的检索效果。

| 分词器 | 类型 | 特点 | 适用场景 |
|--------|------|------|---------|
| jieba | Python 库 | 词典+HMM，三种模式（精确/全/搜索） | 快速原型，轻量场景 |
| IK Analyzer | ES 插件 | 词典+文法，ik_max_word / ik_smart | Elasticsearch 集成 |
| HanLP | Java/Python | 感知机+CRF+词典，支持多任务 | 需要命名实体识别 |
| THULAC | C++/Java/Python | 结构化感知机 | 高精度场景 |
| PaddleNLP | Python | 基于 ERNIE 预训练模型 | 需要上下文感知 |

**jieba 分词示例**：

```python
import jieba
import jieba.analyse

def chinese_tokenize(text: str, mode: str = "search") -> str:
    """
    中文分词
    
    Args:
        text: 中文文本
        mode: 分词模式
            - "exact": 精确模式，最常用
            - "search": 搜索引擎模式，对长词再切分
            - "tfidf": TF-IDF 关键词提取
    
    Returns:
        空格分隔的分词结果
    """
    if mode == "exact":
        return " ".join(jieba.cut(text, cut_all=False))
    elif mode == "search":
        return " ".join(jieba.cut_for_search(text))
    elif mode == "tfidf":
        keywords = jieba.analyse.extract_tags(text, topK=20)
        return " ".join(keywords)
    else:
        raise ValueError(f"不支持的分词模式: {mode}")


# 自定义词典（领域特定词）
def load_custom_dict(dict_path: str):
    """加载自定义词典，用于领域特定词汇的分词"""
    jieba.load_userdict(dict_path)


# 示例
text = "奥希替尼用于EGFR突变阳性的非小细胞肺癌患者的一线治疗"
print(f"精确模式: {chinese_tokenize(text, 'exact')}")
print(f"搜索模式: {chinese_tokenize(text, 'search')}")
print(f"TF-IDF提取: {chinese_tokenize(text, 'tfidf')}")
```

---

## 5.4 结构化检索

### 5.4.1 元数据过滤

结构化检索利用文档的元数据字段进行精确过滤，可以与向量检索或关键词检索组合使用。

```python
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime

@dataclass
class MetadataFilter:
    """元数据过滤器"""
    field: str           # 字段名
    operator: str        # 操作符: eq, neq, gt, gte, lt, lte, in, contains
    value: Any           # 值
    
    def to_dict(self) -> Dict:
        return {
            "field": self.field,
            "operator": self.operator,
            "value": self.value
        }


@dataclass
class StructuredQuery:
    """结构化查询"""
    # 组合条件
    must: List[MetadataFilter] = field(default_factory=list)    # 必须满足
    should: List[MetadataFilter] = field(default_factory=list)  # 满足其一加分
    must_not: List[MetadataFilter] = field(default_factory=list) # 必须排除
    
    def add_filter(self, field: str, operator: str, value: Any,
                   clause: str = "must"):
        """添加过滤条件"""
        filter_obj = MetadataFilter(field, operator, value)
        getattr(self, clause).append(filter_obj)
        return self
    
    def to_es_query(self) -> Dict:
        """转换为 Elasticsearch 过滤查询"""
        def filter_to_es(f: MetadataFilter) -> Dict:
            if f.operator == "eq":
                return {"term": {f.field: f.value}}
            elif f.operator == "neq":
                return {"bool": {"must_not": {"term": {f.field: f.value}}}}
            elif f.operator == "gt":
                return {"range": {f.field: {"gt": f.value}}}
            elif f.operator == "gte":
                return {"range": {f.field: {"gte": f.value}}}
            elif f.operator == "lt":
                return {"range": {f.field: {"lt": f.value}}}
            elif f.operator == "lte":
                return {"range": {f.field: {"lte": f.value}}}
            elif f.operator == "in":
                return {"terms": {f.field: f.value}}
            elif f.operator == "contains":
                return {"wildcard": {f.field: f"*{f.value}*"}}
            else:
                raise ValueError(f"不支持的运算符: {f.operator}")
        
        query = {"bool": {}}
        
        if self.must:
            query["bool"]["filter"] = [filter_to_es(f) for f in self.must]
        if self.should:
            query["bool"]["should"] = [filter_to_es(f) for f in self.should]
            query["bool"]["minimum_should_match"] = 1
        if self.must_not:
            query["bool"]["must_not"] = [filter_to_es(f) for f in self.must_not]
        
        return query
```

### 5.4.2 多维度结构化过滤

在企业 RAG 场景中，结构化过滤通常涉及多个维度：

```python
def build_enterprise_filters(
    department: Optional[str] = None,
    document_type: Optional[str] = None,
    tags: Optional[List[str]] = None,
    date_range: Optional[tuple] = None,
    author: Optional[str] = None,
    security_level: Optional[str] = None,
    project_code: Optional[str] = None
) -> StructuredQuery:
    """
    构建企业级结构化过滤条件
    
    典型场景：用户查询"2024年Q4的肺癌药物研发报告"
    - 部门: R&D
    - 文档类型: report
    - 时间范围: 2024-10~2024-12
    - 安全级别: internal
    """
    query = StructuredQuery()
    
    if department:
        query.add_filter("department", "eq", department)
    
    if document_type:
        query.add_filter("doc_type", "eq", document_type)
    
    if tags:
        query.add_filter("tags", "in", tags)
    
    if date_range:
        start_date, end_date = date_range
        if start_date:
            query.add_filter("created_at", "gte", start_date)
        if end_date:
            query.add_filter("created_at", "lte", end_date)
    
    if author:
        query.add_filter("author", "eq", author)
    
    if security_level:
        query.add_filter("security_level", "eq", security_level)
    
    if project_code:
        query.add_filter("project_code", "eq", project_code)
    
    return query
```

### 5.4.3 混合检索（向量 + 结构化过滤）

大多数向量数据库支持在 ANN 搜索的同时应用结构化过滤：

```python
class HybridSearch:
    """向量检索 + 结构化过滤的混合搜索"""
    
    def __init__(self, vector_store, metadata_store):
        self.vector_store = vector_store
        self.metadata_store = metadata_store  # 元数据索引
    
    def search_with_filter(self,
                          query_vector: np.ndarray,
                          text_query: str = None,
                          filters: StructuredQuery = None,
                          top_k: int = 20,
                          alpha: float = 0.7) -> List[Dict]:
        """
        混合搜索
        
        Args:
            query_vector: 查询向量
            text_query: 可选的文本查询（用于 BM25 融合）
            filters: 结构化过滤条件
            top_k: 返回结果数
            alpha: 向量检索权重（1-alpha 为文本检索权重）
            
        Returns:
            融合后的搜索结果
        """
        # 1. 向量检索 + 过滤
        vector_results = self.vector_store.search(
            query_vector,
            top_k=top_k * 2,  # 多取一些给融合留空间
            filter_expr=filters.to_query_string() if filters else None
        )
        
        # 2. 如果也有文本查询，做 BM25 检索 + 过滤
        text_results = []
        if text_query:
            text_results = self.bm25_search(text_query, top_k=top_k * 2)
            if filters:
                text_results = [r for r in text_results
                                if self._match_filter(r, filters)]
        
        # 3. 结果融合
        merged = self._fuse_results(
            vector_results, text_results,
            alpha=alpha,
            top_k=top_k
        )
        
        return merged
    
    def _fuse_results(self,
                      vector_results: List[Dict],
                      text_results: List[Dict],
                      alpha: float,
                      top_k: int) -> List[Dict]:
        """融合向量检索和文本检索的结果"""
        # 使用 RRF 融合（详见 5.6 节）
        from collections import defaultdict
        
        doc_scores = defaultdict(float)
        
        for rank, doc in enumerate(vector_results):
            doc_id = doc["doc_id"]
            # RRF 融合
            doc_scores[doc_id] += alpha * (1.0 / (60 + rank + 1))
        
        for rank, doc in enumerate(text_results):
            doc_id = doc["doc_id"]
            doc_scores[doc_id] += (1 - alpha) * (1.0 / (60 + rank + 1))
        
        # 按融合得分排序
        sorted_docs = sorted(doc_scores.items(),
                             key=lambda x: x[1], reverse=True)
        
        # 返回原始文档信息
        results = []
        for doc_id, score in sorted_docs[:top_k]:
            doc_info = self._get_doc_info(doc_id)
            doc_info["fusion_score"] = score
            results.append(doc_info)
        
        return results
```

---

## 5.5 知识图谱检索

### 5.5.1 实体查找与关系遍历

知识图谱检索通过查询实体的关联关系，实现深层语义的关联发现。

```python
from typing import List, Dict, Any, Optional
import neo4j

class KnowledgeGraphRetriever:
    """知识图谱检索器"""
    
    def __init__(self, uri: str, user: str, password: str):
        self.driver = neo4j.GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        self.driver.close()
    
    def find_entity(self, entity_name: str) -> List[Dict]:
        """
        查找实体
        
        MATCH (e:Entity) WHERE e.name CONTAINS $name RETURN e
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (e:Entity)
                WHERE e.name CONTAINS $name
                   OR e.synonyms CONTAINS $name
                RETURN e.name AS name,
                       e.type AS type,
                       e.description AS description,
                       labels(e) AS labels
                LIMIT 20
                """,
                name=entity_name
            )
            return [record.data() for record in result]
    
    def get_neighbors(self, entity_id: str,
                      max_hop: int = 2,
                      relation_types: List[str] = None) -> Dict:
        """
        获取实体的 N 跳邻居
        
        Args:
            entity_id: 实体 ID
            max_hop: 最大跳数（1~3 推荐）
            relation_types: 关系类型过滤
            
        Returns:
            包含节点和边的子图
        """
        with self.driver.session() as session:
            # 构建关系类型过滤
            rel_filter = ""
            if relation_types:
                rel_types = "|".join(relation_types)
                rel_filter = f"[:{rel_types}]"
            
            # N 跳邻居查询
            query = f"""
            MATCH path = (start:Entity {{id: $entity_id}})
                        -[{rel_filter}*1..{max_hop}]-
                        (neighbor:Entity)
            RETURN start.name AS source,
                   [r in relationships(path) | type(r)] AS relations,
                   neighbor.name AS target,
                   neighbor.type AS target_type,
                   length(path) AS hops
            LIMIT 200
            """
            
            result = session.run(query, entity_id=entity_id)
            
            nodes = set()
            edges = []
            
            for record in result:
                data = record.data()
                source = data["source"]
                target = data["target"]
                
                nodes.add(source)
                nodes.add(target)
                edges.append({
                    "source": source,
                    "target": target,
                    "relations": data["relations"],
                    "hops": data["hops"]
                })
            
            return {
                "query_entity": entity_id,
                "nodes": list(nodes),
                "edges": edges,
                "max_hop": max_hop
            }
    
    def find_shortest_path(self, entity_a: str,
                           entity_b: str,
                           max_depth: int = 5) -> List[Dict]:
        """
        查找两个实体间的最短路径
        
        Args:
            entity_a: 起始实体名称
            entity_b: 目标实体名称
            max_depth: 最大深度
            
        Returns:
            路径列表
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH p = shortestPath(
                    (a:Entity)-[*1..{max_depth}]-(b:Entity)
                )
                WHERE a.name = $entity_a AND b.name = $entity_b
                RETURN [node in nodes(p) | node.name] AS path_nodes,
                       [rel in relationships(p) | type(rel)] AS path_rels,
                       length(p) AS path_length
                """,
                entity_a=entity_a,
                entity_b=entity_b,
                max_depth=max_depth
            )
            return [record.data() for record in result]
```

### 5.5.2 社区上下文查询

当实体所在社区（community）中有大量相关信息时，可以提取社区级别的摘要作为上下文：

```python
def get_community_context(self, entity_name: str,
                          depth: int = 2) -> str:
    """
    获取实体的社区上下文摘要
    
    步骤：
    1. 查找实体及其 N 跳邻居
    2. 提取相关的关系描述
    3. 构建自然语言上下文
    """
    # 1. 查找实体
    entities = self.find_entity(entity_name)
    if not entities:
        return f"未找到实体: {entity_name}"
    
    entity = entities[0]
    entity_id = entity["name"]
    
    # 2. 获取邻居
    neighborhood = self.get_neighbors(entity_id, max_hop=depth)
    
    # 3. 构建上下文
    context_parts = [f"实体: {entity_id} ({entity.get('type', '未知')})"]
    
    if entity.get("description"):
        context_parts.append(f"描述: {entity['description']}")
    
    context_parts.append(f"\n关联实体 ({len(neighborhood['nodes'])} 个):")
    
    for edge in neighborhood["edges"]:
        context_parts.append(
            f"  {edge['source']} --[{edge['relations'][0]}]--> "
            f"{edge['target']} (跳数: {edge['hops']})"
        )
    
    return "\n".join(context_parts)
```

---

## 5.6 重排序与融合

### 5.6.1 RRF（Reciprocal Rank Fusion）算法

RRF 是一种无参数、无需训练的结果融合方法，通过排名位置的倒数来融合多个排序结果。

**RRF 公式**：

```
RRF(doc) = Σ [ 1 / (k + rank_i(doc)) ]
```

- rank_i(doc)：文档在第 i 个检索结果中的排名
- k：常数（通常为 60），用于平滑，防止排名过高导致的影响过大

**代码实现**：

```python
from collections import defaultdict
from typing import List, Dict, Any

def reciprocal_rank_fusion(
    result_lists: List[List[Dict[str, Any]]],
    k: int = 60,
    weights: List[float] = None,
    top_k: int = 20
) -> List[Dict]:
    """
    RRF 融合多个检索结果
    
    Args:
        result_lists: 多个检索器的结果列表
        k: RRF 常数，默认 60
        weights: 各检索器的权重，默认等权
        top_k: 最终返回结果数
        
    Returns:
        融合排序后的结果列表
    """
    if weights is None:
        weights = [1.0] * len(result_lists)
    
    # 归一化权重
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]
    
    # 累积 RRF 分数
    rrf_scores = defaultdict(float)
    doc_map = {}  # doc_id -> full document
    
    for list_idx, results in enumerate(result_lists):
        for rank, doc in enumerate(results):
            doc_id = doc.get("doc_id", doc.get("id", str(hash(str(doc)))))
            
            # RRF 分数：权重 × 1/(k + rank)
            rrf_score = weights[list_idx] * (1.0 / (k + rank + 1))
            rrf_scores[doc_id] += rrf_score
            
            # 保存文档信息（优先保存得分最高的版本）
            if doc_id not in doc_map:
                doc_map[doc_id] = {**doc}
    
    # 按 RRF 分数排序
    ranked = sorted(rrf_scores.items(),
                    key=lambda x: x[1],
                    reverse=True)[:top_k]
    
    # 构建最终结果
    results = []
    for rank, (doc_id, score) in enumerate(ranked):
        doc = doc_map[doc_id]
        doc["rrf_score"] = round(score, 6)
        doc["fusion_rank"] = rank + 1
        results.append(doc)
    
    return results
```

**RRF vs 加权融合对比**：

| 特性 | RRF | 加权线性融合 |
|------|-----|------------|
| 参数 | 只需 k（经验值 60） | 需要调优各通道权重 |
| 对分数敏感度 | 仅依赖排名，不依赖原始分数 | 依赖各通道分数分布 |
| 跨通道可比性 | 天然可比 | 需要分数归一化 |
| 鲁棒性 | 对异常高分不敏感 | 受 outlier 影响大 |
| 实现复杂度 | 极简 | 需要归一化策略 |

### 5.6.2 Cross-encoder 重排序

RRF 融合后，还可以使用 Cross-encoder 模型对候选结果进行精细重排序。Cross-encoder 将查询和文档拼接后输入 Transformer，计算相关性得分。

**Bi-encoder vs Cross-encoder**：

| 特性 | Bi-encoder（如 BGE） | Cross-encoder（如 BGE Reranker） |
|------|---------------------|-------------------------------|
| 计算方式 | 分别编码查询和文档 | 拼接后联合编码 |
| 速度 | 快（可预计算文档 embedding） | 慢（每对都要计算） |
| 精度 | 中等 | 高 |
| 适用阶段 | 第一阶段召回 | 第二阶段精排 |
| 模型大小 | 通常更大（如 326M） | 可以更小（如 278M） |

**Cross-encoder 重排序实现**：

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import List, Dict

class CrossEncoderReranker:
    """Cross-encoder 重排序器"""
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        """
        初始化 Cross-encoder 重排序模型
        
        推荐模型：
        - BAAI/bge-reranker-v2-m3: 多语言，1024 维
        - cross-encoder/ms-marco-MiniLM-L-6-v2: 英文，轻量
        - Cohere rerank: API 调用
        """
        self.device = torch.device("cuda" if torch.cuda.is_available()
                                   else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name
        ).to(self.device)
        self.model.eval()
        print(f"[Reranker] 模型: {model_name}, 设备: {self.device}")
    
    def rerank(self,
               query: str,
               documents: List[Dict],
               top_k: int = 10,
               batch_size: int = 32) -> List[Dict]:
        """
        对候选文档进行重排序
        
        Args:
            query: 查询文本
            documents: 候选文档列表，每项包含 text 字段
            top_k: 返回 top-k
            batch_size: 批处理大小
            
        Returns:
            重排序后的文档列表
        """
        pairs = [(query, doc["text"]) for doc in documents]
        scores = []
        
        # 分批推理
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i + batch_size]
            
            inputs = self.tokenizer(
                batch_pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_scores = outputs.logits.squeeze(-1).cpu().tolist()
                
                if isinstance(batch_scores, float):
                    batch_scores = [batch_scores]
                
                scores.extend(batch_scores)
        
        # 为每个文档添加重排序分数
        for doc, score in zip(documents, scores):
            doc["rerank_score"] = round(score, 4)
        
        # 按重排序分数排序
        sorted_docs = sorted(documents,
                             key=lambda x: x["rerank_score"],
                             reverse=True)
        
        # 更新排名
        for rank, doc in enumerate(sorted_docs[:top_k]):
            doc["rerank_rank"] = rank + 1
        
        return sorted_docs[:top_k]


class CohereReranker:
    """Cohere API 重排序（云端调用）"""
    
    def __init__(self, api_key: str, model: str = "rerank-v3.5"):
        import cohere
        self.client = cohere.Client(api_key)
        self.model = model
    
    def rerank(self, query: str, documents: List[Dict],
               top_k: int = 10) -> List[Dict]:
        """
        使用 Cohere rerank API
        
        Args:
            query: 查询文本
            documents: 候选文档列表
            top_k: 返回 top-k
            
        Returns:
            重排序后的文档
        """
        texts = [doc["text"] for doc in documents]
        
        response = self.client.rerank(
            model=self.model,
            query=query,
            documents=texts,
            top_n=top_k,
            return_documents=True
        )
        
        results = []
        for result in response.results:
            doc = documents[result.index].copy()
            doc["rerank_score"] = result.relevance_score
            doc["rerank_rank"] = result.index + 1
            results.append(doc)
        
        return results
```

### 5.6.3 去重与合并

多路召回必然带来结果重复。去重是融合前的必要步骤：

```python
from typing import List, Dict, Set

def dedup_results(results: List[Dict],
                  strategy: str = "content_hash",
                  similarity_threshold: float = 0.85) -> List[Dict]:
    """
    对检索结果进行去重
    
    Args:
        results: 检索结果列表
        strategy: 去重策略
            - "content_hash": 内容哈希精确去重
            - "cosine": 语义相似度去重（需 embedding）
            - "title": 标题精确去重
        similarity_threshold: 语义去重的相似度阈值
        
    Returns:
        去重后的结果列表
    """
    if strategy == "content_hash":
        return _dedup_by_hash(results)
    elif strategy == "title":
        return _dedup_by_title(results)
    elif strategy == "cosine":
        return _dedup_by_similarity(results, similarity_threshold)
    else:
        raise ValueError(f"不支持的策略: {strategy}")


def _dedup_by_hash(results: List[Dict]) -> List[Dict]:
    """基于内容哈希的精确去重"""
    import hashlib
    seen: Set[str] = set()
    deduped = []
    
    for doc in results:
        content = doc.get("text", doc.get("content", ""))
        doc_hash = hashlib.md5(content.encode()).hexdigest()
        
        if doc_hash not in seen:
            seen.add(doc_hash)
            deduped.append(doc)
    
    print(f"[Dedup] 去重前: {len(results)}, 去重后: {len(deduped)}")
    return deduped


def _dedup_by_title(results: List[Dict]) -> List[Dict]:
    """基于标题去重"""
    seen: Set[str] = set()
    deduped = []
    
    for doc in results:
        title = doc.get("title", "").strip()
        if title and title not in seen:
            seen.add(title)
            deduped.append(doc)
        elif not title:
            deduped.append(doc)  # 无标题的文档保留
    
    return deduped


def _dedup_by_similarity(results: List[Dict],
                         threshold: float) -> List[Dict]:
    """基于语义相似度去重（保留第一个出现的版本）"""
    from sentence_transformers import SentenceTransformer
    
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [doc.get("text", doc.get("content", ""))
             for doc in results]
    
    embeddings = model.encode(texts, normalize_embeddings=True)
    
    keep = [True] * len(results)
    for i in range(len(results)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(results)):
            similarity = np.dot(embeddings[i], embeddings[j])
            if similarity > threshold:
                keep[j] = False
    
    deduped = [doc for doc, kept in zip(results, keep) if kept]
    print(f"[Dedup] 去重前: {len(results)}, 去重后: {len(deduped)}")
    return deduped
```

---

## 5.7 多路召回流水线完整实现

以下是一个完整的端到端多路召回流水线：

```python
import time
import logging
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RecallResult:
    """单个召回通道的结果"""
    channel_name: str
    documents: List[Dict]
    latency_ms: float
    status: str  # "success" | "error"


@dataclass
class RecallConfig:
    """召回配置"""
    dense_top_k: int = 20
    sparse_top_k: int = 20
    structured_top_k: int = 10
    kg_top_k: int = 10
    final_top_k: int = 10
    rrf_k: int = 60
    enable_rerank: bool = True
    rerank_top_k: int = 10


class MultiRecallPipeline:
    """多路召回流水线"""
    
    def __init__(self, config: RecallConfig = None):
        self.config = config or RecallConfig()
        self.recallers: Dict[str, Callable] = {}
        logger.info(f"[MultiRecall] 初始化完成, 配置: {config}")
    
    def register_recaller(self, name: str,
                          recall_fn: Callable,
                          top_k: int = None):
        """
        注册召回通道
        
        Args:
            name: 通道名称
            recall_fn: 召回函数，接受 query 参数返回文档列表
            top_k: 该通道的 top_k（可选）
        """
        self.recallers[name] = {
            "fn": recall_fn,
            "top_k": top_k
        }
        logger.info(f"[MultiRecall] 注册通道: {name}")
    
    def recall(self, query: str, **kwargs) -> Dict:
        """
        执行多路召回
        
        Args:
            query: 查询文本
            **kwargs: 传递给各个召回通道的额外参数
            
        Returns:
            {
                "results": 最终结果列表,
                "channels": 各通道结果,
                "latency": 总延迟,
                "metrics": 各通道延迟统计
            }
        """
        all_results = []
        channel_results = {}
        total_latency = 0
        
        # 阶段1: 并行执行各召回通道
        for name, recaller in self.recallers.items():
            try:
                t_start = time.time()
                
                # 执行召回
                docs = recaller["fn"](
                    query,
                    top_k=recaller["top_k"] or self.config.dense_top_k,
                    **kwargs
                )
                
                latency = (time.time() - t_start) * 1000
                total_latency += latency
                
                result = RecallResult(
                    channel_name=name,
                    documents=docs,
                    latency_ms=round(latency, 2),
                    status="success"
                )
                
                logger.info(f"[Recall] {name}: {len(docs)} 条, "
                           f"{latency:.0f}ms")
                
            except Exception as e:
                result = RecallResult(
                    channel_name=name,
                    documents=[],
                    latency_ms=0,
                    status=f"error: {str(e)}"
                )
                logger.error(f"[Recall] {name} 失败: {e}")
            
            channel_results[name] = result
            all_results.append(result.documents)
        
        # 阶段2: 去重
        flat_results = []
        seen_ids = set()
        for docs in all_results:
            for doc in docs:
                doc_id = doc.get("doc_id", doc.get("id", ""))
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    flat_results.append(doc)
        
        logger.info(f"[Recall] 去重后: {len(flat_results)} 条")
        
        # 阶段3: RRF 融合
        fused_results = reciprocal_rank_fusion(
            all_results,
            k=self.config.rrf_k,
            top_k=self.config.final_top_k * 2  # 为 rerank 留空间
        )
        
        # 阶段4: Cross-encoder 重排序（可选）
        if self.config.enable_rerank and fused_results:
            t_start = time.time()
            reranker = CrossEncoderReranker()
            final_results = reranker.rerank(
                query,
                fused_results,
                top_k=self.config.rerank_top_k
            )
            rerank_latency = (time.time() - t_start) * 1000
            logger.info(f"[Recall] 重排序: {rerank_latency:.0f}ms")
        else:
            final_results = fused_results[:self.config.final_top_k]
        
        return {
            "results": final_results,
            "channels": channel_results,
            "total_latency_ms": round(total_latency, 2),
            "total_results": len(final_results)
        }


# ============================================================
# 使用示例
# ============================================================

def demo_multi_recall():
    """多路召回流水线演示"""
    
    # 1. 初始化流水线
    config = RecallConfig(
        dense_top_k=20,
        sparse_top_k=20,
        final_top_k=10,
        enable_rerank=True
    )
    pipeline = MultiRecallPipeline(config)
    
    # 2. 注册各召回通道
    
    # 2.1 向量检索通道
    def dense_recaller(query: str, top_k: int = 20) -> List[Dict]:
        """模拟向量检索"""
        # 实际项目中调用 embedding + 向量数据库
        return [
            {"doc_id": "d1", "text": "非小细胞肺癌靶向治疗指南",
             "score": 0.92, "channel": "dense"},
            {"doc_id": "d2", "text": "奥希替尼用药指南",
             "score": 0.88, "channel": "dense"},
        ]
    
    # 2.2 关键词检索通道
    def sparse_recaller(query: str, top_k: int = 20) -> List[Dict]:
        """模拟 BM25 检索"""
        return [
            {"doc_id": "d3", "text": "肺癌诊疗指南2024版",
             "score": 25.3, "channel": "sparse"},
            {"doc_id": "d1", "text": "非小细胞肺癌靶向治疗指南",
             "score": 22.1, "channel": "sparse"},
        ]
    
    # 2.3 结构化检索通道
    def structured_recaller(query: str, top_k: int = 10) -> List[Dict]:
        """模拟结构化检索"""
        return [
            {"doc_id": "d4", "text": "2024年肺癌研究进展报告",
             "score": 1.0, "channel": "structured"},
        ]
    
    # 2.4 知识图谱检索通道
    def kg_recaller(query: str, top_k: int = 10) -> List[Dict]:
        """模拟知识图谱检索"""
        return [
            {"doc_id": "d5", "text": "奥希替尼→EGFR突变→非小细胞肺癌",
             "score": 0.95, "channel": "kg"},
        ]
    
    pipeline.register_recaller("dense", dense_recaller)
    pipeline.register_recaller("sparse", sparse_recaller)
    pipeline.register_recaller("structured", structured_recaller)
    pipeline.register_recaller("kg", kg_recaller)
    
    # 3. 执行多路召回
    result = pipeline.recall("非小细胞肺癌的靶向药物有哪些")
    
    # 4. 输出结果
    print(f"\n总延迟: {result['total_latency_ms']}ms")
    print(f"最终结果数: {result['total_results']}")
    print("\n各通道结果:")
    for name, channel in result["channels"].items():
        print(f"  [{name}] {channel.status}, "
              f"{len(channel.documents)} 条, {channel.latency_ms}ms")
    
    print("\n最终排序结果:")
    for i, doc in enumerate(result["results"]):
        print(f"  #{i+1}: {doc['text'][:50]}... "
              f"(RRF: {doc.get('rrf_score', 'N/A')}, "
              f"通道: {doc.get('channel', 'N/A')})")


if __name__ == "__main__":
    demo_multi_recall()
```

---

## 5.8 检索效果评估

### 5.8.1 核心评估指标

| 指标 | 定义 | 值域 | 适用场景 |
|------|------|------|---------|
| Recall@K | Top-K 中相关文档占比 | [0, 1] | 召回率敏感场景 |
| Precision@K | Top-K 中相关文档占比（同 Recall） | [0, 1] | 精确度敏感场景 |
| MRR | 第一个相关文档排名的倒数均值 | (0, 1] | 只需一个正确答案的场景 |
| NDCG | 归一化折损累积增益 | [0, 1] | 多级相关性评估 |
| MAP | 平均精确率均值 | [0, 1] | 综合排序质量 |

**完整评估代码**：

```python
import numpy as np
from typing import List, Set

class RetrievalEvaluator:
    """检索效果评估器"""
    
    def __init__(self):
        self.metrics = {}
    
    def recall_at_k(self, relevant: Set[str],
                    retrieved: List[str], k: int) -> float:
        """
        Recall@K: Top-K 中相关文档占比
        
        Recall@K = |relevant ∩ retrieved[:K]| / |relevant|
        """
        if len(relevant) == 0:
            return 0.0
        retrieved_k = set(retrieved[:k])
        return len(relevant & retrieved_k) / len(relevant)
    
    def precision_at_k(self, relevant: Set[str],
                       retrieved: List[str], k: int) -> float:
        """
        Precision@K: Top-K 中相关文档占比
        
        Precision@K = |relevant ∩ retrieved[:K]| / K
        """
        if k == 0:
            return 0.0
        retrieved_k = set(retrieved[:k])
        return len(relevant & retrieved_k) / k
    
    def mrr(self, relevant: Set[str],
            retrieved: List[str]) -> float:
        """
        MRR: 第一个相关文档排名的倒数
        
        MRR = 1 / rank_of_first_relevant
        """
        for i, doc_id in enumerate(retrieved):
            if doc_id in relevant:
                return 1.0 / (i + 1)
        return 0.0
    
    def ndcg(self, relevance_scores: List[float],
             retrieved: List[str], k: int) -> float:
        """
        NDCG@K: 归一化折损累积增益
        
        DCG = Σ (2^rel_i - 1) / log2(i + 2)
        IDCG = Σ (2^best_rel_i - 1) / log2(i + 2)
        NDCG = DCG / IDCG
        """
        # 将 doc_id 映射到相关性分数
        score_map = {doc_id: score
                     for doc_id, score in zip(retrieved, relevance_scores)}
        
        # 计算 DCG
        dcg = 0.0
        for i in range(min(k, len(retrieved))):
            doc_id = retrieved[i]
            rel = score_map.get(doc_id, 0)
            dcg += (2 ** rel - 1) / np.log2(i + 2)
        
        # 计算 IDCG（理想排序）
        ideal_scores = sorted(relevance_scores, reverse=True)
        idcg = 0.0
        for i in range(min(k, len(ideal_scores))):
            idcg += (2 ** ideal_scores[i] - 1) / np.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def evaluate(self, query_results: List[Dict],
                 ground_truth: Dict[str, Set[str]],
                 k_values: List[int] = None) -> Dict:
        """
        对一组查询结果进行评估
        
        Args:
            query_results: [{query, retrieved: [doc_ids], relevance_scores}]
            ground_truth: {query: set(relevant_doc_ids)}
            k_values: 要评估的 K 值列表
            
        Returns:
            各指标的平均值
        """
        if k_values is None:
            k_values = [1, 3, 5, 10, 20, 50]
        
        metrics = {
            f"Recall@{k}": [] for k in k_values
        }
        metrics.update({
            f"Precision@{k}": [] for k in k_values
        })
        metrics["MRR"] = []
        metrics["NDCG@10"] = []
        metrics["NDCG@20"] = []
        
        for item in query_results:
            query = item["query"]
            retrieved = item["retrieved"]
            rel_scores = item.get("relevance_scores", None)
            relevant = ground_truth.get(query, set())
            
            for k in k_values:
                metrics[f"Recall@{k}"].append(
                    self.recall_at_k(relevant, retrieved, k)
                )
                metrics[f"Precision@{k}"].append(
                    self.precision_at_k(relevant, retrieved, k)
                )
            
            metrics["MRR"].append(self.mrr(relevant, retrieved))
            
            if rel_scores:
                metrics["NDCG@10"].append(
                    self.ndcg(rel_scores, retrieved, 10)
                )
                metrics["NDCG@20"].append(
                    self.ndcg(rel_scores, retrieved, 20)
                )
        
        # 计算平均值
        avg_metrics = {}
        for name, values in metrics.items():
            if values:
                avg_metrics[name] = round(np.mean(values), 4)
            else:
                avg_metrics[name] = "N/A"
        
        return avg_metrics
```

### 5.8.2 评估实验示例

```python
def run_evaluation_demo():
    """运行检索评估演示"""
    
    # 模拟 ground truth
    ground_truth = {
        "非小细胞肺癌靶向治疗": {"d1", "d2", "d5"},
        "免疫检查点抑制剂": {"d3", "d6"},
        "肺癌早期筛查": {"d4", "d7", "d8"},
    }
    
    # 模拟不同检索策略的结果
    strategies = {
        "Dense Only": [
            {"query": "非小细胞肺癌靶向治疗",
             "retrieved": ["d1", "d3", "d5", "d2", "d4"],
             "relevance_scores": [2, 1, 2, 2, 0]},
            {"query": "免疫检查点抑制剂",
             "retrieved": ["d6", "d1", "d3", "d2", "d8"],
             "relevance_scores": [2, 0, 2, 0, 1]},
        ],
        "Sparse Only": [
            {"query": "非小细胞肺癌靶向治疗",
             "retrieved": ["d2", "d4", "d1", "d6", "d3"],
             "relevance_scores": [2, 0, 2, 0, 1]},
            {"query": "免疫检查点抑制剂",
             "retrieved": ["d3", "d1", "d8", "d4", "d6"],
             "relevance_scores": [2, 0, 1, 0, 2]},
        ],
        "Multi-Recall (Dense+Sparse)": [
            {"query": "非小细胞肺癌靶向治疗",
             "retrieved": ["d1", "d2", "d5", "d3", "d4"],
             "relevance_scores": [2, 2, 2, 1, 0]},
            {"query": "免疫检查点抑制剂",
             "retrieved": ["d6", "d3", "d8", "d1", "d4"],
             "relevance_scores": [2, 2, 1, 0, 0]},
        ],
    }
    
    evaluator = RetrievalEvaluator()
    
    print("=" * 80)
    print(f"{'策略':<30} {'Recall@5':<12} {'Precision@5':<12} {'MRR':<12} {'NDCG@10':<12}")
    print("=" * 80)
    
    for strategy, results in strategies.items():
        metrics = evaluator.evaluate(results, ground_truth)
        print(f"{strategy:<30} "
              f"{metrics['Recall@5']:<12.4f} "
              f"{metrics['Precision@5']:<12.4f} "
              f"{metrics['MRR']:<12.4f} "
              f"{metrics['NDCG@10']:<12.4f}")
    
    print("=" * 80)
    print("结论: 多路召回在所有指标上均优于单一检索方法")
```

### 5.8.3 A/B 测试评估框架

```python
class ABTestFramework:
    """A/B 测试评估框架"""
    
    def __init__(self, query_log_path: str):
        self.queries = self._load_query_log(query_log_path)
        self.judgments = {}  # 人工标注的相关性判断
    
    def _load_query_log(self, path: str) -> List[Dict]:
        """加载查询日志"""
        import json
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]
    
    def run_ab_test(self,
                    strategy_a: Callable,
                    strategy_b: Callable,
                    sample_size: int = 100) -> Dict:
        """
        运行 A/B 测试
        
        比较两种检索策略的效果差异
        """
        import random
        queries = random.sample(self.queries, min(sample_size, len(self.queries)))
        
        results = {"A": [], "B": []}
        
        for query_data in queries:
            query = query_data["query"]
            ground_truth = set(query_data.get("relevant_docs", []))
            
            # 执行策略 A
            docs_a = strategy_a(query)
            retrieved_a = [d["doc_id"] for d in docs_a]
            
            # 执行策略 B
            docs_b = strategy_b(query)
            retrieved_b = [d["doc_id"] for d in docs_b]
            
            evaluator = RetrievalEvaluator()
            
            results["A"].append({
                "recall@10": evaluator.recall_at_k(ground_truth, retrieved_a, 10),
                "mrr": evaluator.mrr(ground_truth, retrieved_a),
                "latency_ms": docs_a.get("latency_ms", 0)
            })
            
            results["B"].append({
                "recall@10": evaluator.recall_at_k(ground_truth, retrieved_b, 10),
                "mrr": evaluator.mrr(ground_truth, retrieved_b),
                "latency_ms": docs_b.get("latency_ms", 0)
            })
        
        # 汇总统计
        summary = {}
        for strategy in ["A", "B"]:
            metrics = results[strategy]
            summary[strategy] = {
                "avg_recall@10": np.mean([m["recall@10"] for m in metrics]),
                "avg_mrr": np.mean([m["mrr"] for m in metrics]),
                "avg_latency_ms": np.mean([m["latency_ms"] for m in metrics]),
                "p95_latency_ms": np.percentile(
                    [m["latency_ms"] for m in metrics], 95
                )
            }
        
        return summary
```

---

## 5.9 实践指南与经验总结

### 5.9.1 多路召回通道选择矩阵

| 场景 | 推荐通道 | 说明 |
|------|---------|------|
| 语义匹配为主 | Dense (权重 0.6) + Sparse (权重 0.4) | 通用知识问答 |
| 精确匹配需求高 | Sparse (权重 0.7) + Dense (权重 0.3) | 代码、版本号、编号 |
| 多模态知识 | Dense + KG | 实体关系密集型场景 |
| 时间敏感 | Dense + Structured (时间过滤) | 新闻、财报 |
| 专业领域（医学/法律） | Dense + KG + Structured | 需要精确分类 |
| 代码搜索 | Sparse (代码文本) + Dense (语义) | 双通道代码检索 |

### 5.9.2 常见问题与解决方案

| 问题 | 现象 | 解决方案 |
|------|------|---------|
| 向量检索冷启动 | 新文档 embedding 质量差 | 使用成熟的预训练模型，预热期用 BM25 兜底 |
| 多路召回延迟高 | 每个通道串行执行 | 异步并行调用各 Recaller |
| 融合后结果变差 | RRF 被低质量通道拖累 | 加权 RRF，或先过滤低质量通道 |
| 召回结果重复率高 | 各通道返回大量相同文档 | 融合前去重，保留得分最高的版本 |
| 长尾查询效果差 | 罕见词/新词匹配不上 | 混合使用 n-gram 和子词 tokenizer |
| 评估指标与用户体验不一致 | 指标高但用户不满意 | 增加人工评估维度，关注 NDCG 而非仅 Recall |

### 5.9.3 多路召回参数速查表

| 参数 | 推荐范围 | 默认值 | 说明 |
|------|---------|-------|------|
| dense_top_k | 20~100 | 50 | 向量检索候选数 |
| sparse_top_k | 20~100 | 50 | BM25 候选数 |
| rrf_k | 30~100 | 60 | RRF 平滑常数 |
| efSearch | 100~500 | 200 | HNSW 搜索精度 |
| BM25 k1 | 1.2~2.0 | 1.5 | 词频饱和参数 |
| BM25 b | 0.6~0.85 | 0.75 | 长度归一化参数 |
| Cross-encoder batch_size | 16~64 | 32 | 重排序批处理大小 |
| chunk_overlap | 50~200 | 100 | 文档块重叠大小 |

---

## 本章小结

多路召回是 RAG 系统从"能用"到"好用"的关键升级。单一检索方法各有其盲区——向量检索语义强大但漏精确匹配，BM25 精确匹配好但不懂语义，结构化检索精准但灵活不足。多路召回通过组合多个互补的检索通道，配合 RRF 融合和 Cross-encoder 重排序，显著提升召回质量和鲁棒性。

实践中，建议从"向量 + BM25"双通道起步，逐步根据业务需求加入结构化过滤和知识图谱通道。评估指标上，不要只看 Recall@K，要结合 MRR（关注首个结果）和 NDCG（关注排序质量）来全面衡量。参数调优时，RRF 的 k 值和 BM25 的 k1/b 是最先需要调整的杠杆点。
