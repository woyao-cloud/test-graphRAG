# 第七章 多路召回：检索精度与召回率的终极博弈

## 7.1 为什么需要多路召回

### 7.1.1 单一检索方法的局限

在构建 RAG 系统时，最朴素的想法是：用向量检索（Vector Retrieval）把用户 query 转为 embedding，然后在向量数据库中找最相似的 Top-K 段落。这在很多场景下确实能工作，但单一检索方法存在以下本质局限：

| 检索方法 | 优势 | 致命缺陷 |
|---|---|---|
| Dense Retrieval（密集检索） | 语义匹配强，能处理同义词、 paraphrase | 对高频词、稀有实体不敏感；需要大量训练数据 |
| Sparse Retrieval（稀疏检索/BM25） | 精确词匹配，零样本可用，可解释性强 | 无法处理语义鸿沟（"车" vs "汽车"） |
| 结构化检索（Metadata Filter） | 精确过滤时间、类别、标签 | 无法处理模糊语义查询 |
| KG Retrieval（图检索） | 关系推理，多跳关联 | 图谱覆盖率有限，构建成本高 |

一个现实的例子：用户问 "2024年诺贝尔物理学奖得主的主要贡献是什么？"

- 纯向量检索可能把 "2023年诺贝尔奖" 排在前面（语义相似度高但时间错误）
- 纯 BM25 可能命中 "诺贝尔物理学奖" 但错过用 "honoree" 而非 "winner" 的文档
- 纯结构化检索需要用户精确知道过滤条件
- 纯 KG 检索可能图里根本没有 2024 年的节点

**多路召回（Multi-Recall / Multi-Path Retrieval）** 的核心思想是：**用多条检索路径同时从不同角度搜索，然后将结果融合排序**，让每种方法的优势互补、劣势对冲。

### 7.1.2 多路召回的总体架构

```
User Query
    |
    +---> Dense Retriever (Embedding + ANN) ──┐
    +---> Sparse Retriever (BM25 / SPLADE) ──┤
    +---> Structured Retriever (Metadata) ────┤
    +---> KG Retriever (Entity / Path) ──────┤
    |                                        |
    +----------> Fusion & Rerank <───────────+
                      |
               Dedup & Merge
                      |
              Final Top-K Results
```

### 7.1.3 多路召回的核心挑战

1. **异构结果融合**：各路的分数尺度不同（余弦相似度 0~1 vs BM25 0~几十），无法直接比较
2. **重复结果去重**：同一段落可能被多条路同时召回
3. **排序一致性**：融合后的排序需要优于任一路单独的结果
4. **延迟与成本**：多路并行增加了检索延迟和 API 成本

本章将逐一解决这些问题，并提供完整的生产级 Python 实现。

---

## 7.2 密集向量检索（Dense Retrieval）

### 7.2.1 原理概述

密集检索将文本映射到高维稠密向量空间，通过计算向量之间的余弦相似度（Cosine Similarity）来衡量语义相关性。

```
query ──→ Encoder ──→ q_vector (768d)
doc   ──→ Encoder ──→ d_vector (768d)
sim = cos(q_vector, d_vector) = q·d / (|q| * |d|)
```

常用的编码器包括：
- **Sentence-BERT** (all-MiniLM-L6-v2, all-mpnet-base-v2)
- **OpenAI Embeddings** (text-embedding-3-small, text-embedding-3-large)
- **Cohere Embeddings** (embed-english-v3.0)
- **BGE Embeddings** (BAAI/bge-large-zh-v1.5) — 中文场景推荐

### 7.2.2 使用 Sentence-Transformers 实现密集检索

```python
# ch07_dense_retriever.py
"""
密集向量检索实现 —— 基于 Sentence-Transformers
支持批量编码、ANN 检索、缓存机制
"""

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class Document:
    """文档数据结构"""
    id: str
    text: str
    metadata: dict = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None


class DenseRetriever:
    """
    密集向量检索器

    Parameters
    ----------
    model_name : str
        Sentence-Transformers 模型名称
    device : str
        计算设备 ('cpu', 'cuda', 'mps')
    batch_size : int
        编码时的批处理大小
    normalize : bool
        是否对向量做 L2 归一化（默认 True，余弦相似度等价于内积）
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-large-zh-v1.5",
        device: str = "cpu",
        batch_size: int = 32,
        normalize: bool = True,
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.normalize = normalize
        self._model = None
        self.documents: List[Document] = []
        self.embedding_matrix: Optional[np.ndarray] = None
        self.doc_id_to_idx: dict = {}

    def _load_model(self):
        """懒加载编码模型"""
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(
                self.model_name,
                device=self.device
            )
            print(f"[DenseRetriever] 模型 {self.model_name} 加载完成")
        except ImportError:
            raise ImportError(
                "请安装 sentence-transformers: pip install sentence-transformers"
            )

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        批量编码文本为稠密向量

        Parameters
        ----------
        texts : List[str]
            待编码的文本列表

        Returns
        -------
        np.ndarray
            形状为 (len(texts), embedding_dim) 的向量矩阵
        """
        self._load_model()
        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=self.normalize,
        )
        return np.array(embeddings, dtype=np.float32)

    def index_documents(self, documents: List[Document]):
        """
        构建文档索引（全量编码 + 内存存储）

        Parameters
        ----------
        documents : List[Document]
            待索引的文档列表
        """
        self.documents = documents
        texts = [doc.text for doc in documents]

        print(f"[DenseRetriever] 正在编码 {len(texts)} 篇文档...")
        embeddings = self.encode(texts)

        for i, doc in enumerate(documents):
            doc.embedding = embeddings[i]
            self.doc_id_to_idx[doc.id] = i

        self.embedding_matrix = embeddings
        print(f"[DenseRetriever] 索引完成，向量维度: {embeddings.shape[1]}")

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        return_scores: bool = True,
    ) -> List[Tuple[Document, float]]:
        """
        检索与 query 最相似的 Top-K 文档

        使用暴力线性扫描（适用于小规模索引）。
        大规模场景请换用 FAISS (见 7.2.3 节)。

        Parameters
        ----------
        query : str
            用户查询
        top_k : int
            返回的最相似文档数量
        return_scores : bool
            是否返回相似度分数

        Returns
        -------
        List[Tuple[Document, float]]
            排序后的 (文档, 分数) 列表
        """
        if self.embedding_matrix is None:
            raise RuntimeError("请先调用 index_documents() 构建索引")

        query_vec = self.encode([query])[0]

        # 余弦相似度 = 点积（向量已归一化）
        scores = np.dot(self.embedding_matrix, query_vec)

        # 取 Top-K 索引
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            doc = self.documents[idx]
            score = float(scores[idx])
            results.append((doc, score))

        return results


# ============================================================
# 使用示例
# ============================================================
if __name__ == "__main__":
    # 1. 准备文档
    docs = [
        Document(id="1", text="Transformer 架构由编码器和解码器组成"),
        Document(id="2", text="BERT 使用双向注意力机制进行预训练"),
        Document(id="3", text="GPT 系列模型采用自回归生成方式"),
        Document(id="4", text="注意力机制允许模型关注输入序列的不同位置"),
        Document(id="5", text="残差连接解决了深层网络的梯度消失问题"),
    ]

    # 2. 初始化检索器
    retriever = DenseRetriever(
        model_name="BAAI/bge-small-zh-v1.5",  # 轻量中文模型
        device="cpu",
    )

    # 3. 构建索引
    retriever.index_documents(docs)

    # 4. 检索
    query = "什么是自回归语言模型"
    results = retriever.retrieve(query, top_k=3)

    print(f"\n查询: {query}")
    print("检索结果:")
    for doc, score in results:
        print(f"  [{doc.id}] (score={score:.4f}) {doc.text}")
```

### 7.2.3 使用 FAISS 加速大规模检索

当文档数量超过十万级别时，暴力线性扫描不可接受。FAISS（Facebook AI Similarity Search）提供了高效的近似最近邻（ANN）搜索。

```python
# ch07_faiss_retriever.py
"""
基于 FAISS 的大规模密集向量检索
支持 IVF 索引和 GPU 加速
"""

import numpy as np
import pickle
import os
from typing import List, Optional, Tuple


class FAISSRetriever:
    """
    FAISS 加速的密集检索器

    支持两种索引模式：
    - "flat": 精确搜索（暴力扫描），适合 <100K 文档
    - "ivf": 近似搜索（倒排文件索引），适合 >100K 文档

    Parameters
    ----------
    dim : int
        向量维度
    index_type : str
        索引类型 ("flat" 或 "ivf")
    nlist : int
        IVF 聚类中心数（仅 index_type="ivf" 时有效）
    use_gpu : bool
        是否使用 GPU
    """

    def __init__(
        self,
        dim: int = 768,
        index_type: str = "flat",
        nlist: int = 100,
        use_gpu: bool = False,
    ):
        self.dim = dim
        self.index_type = index_type
        self.nlist = nlist
        self.use_gpu = use_gpu
        self.index = None
        self.documents: List = []
        self.is_trained = False

    def _create_index(self, vectors: np.ndarray):
        """创建 FAISS 索引"""
        import faiss

        if self.index_type == "flat":
            # 精确索引: IndexFlatIP (内积，等价于归一化后的余弦)
            self.index = faiss.IndexFlatIP(self.dim)
            self.index.add(vectors)
            self.is_trained = True

        elif self.index_type == "ivf":
            # 近似索引: IVF + Flat
            quantizer = faiss.IndexFlatIP(self.dim)
            self.index = faiss.IndexIVFFlat(
                quantizer, self.dim, self.nlist, faiss.METRIC_INNER_PRODUCT
            )
            # IVF 需要训练
            self.index.train(vectors)
            self.index.add(vectors)
            self.is_trained = True

        else:
            raise ValueError(f"不支持的索引类型: {self.index_type}")

        # GPU 加速
        if self.use_gpu:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

        print(
            f"[FAISSRetriever] 索引创建完成，类型={self.index_type}"
            f"，文档数={self.index.ntotal}"
        )

    def index_documents(
        self,
        documents: List,
        embeddings: np.ndarray,
    ):
        """
        索引文档

        Parameters
        ----------
        documents : List
            文档列表（任意类型，仅保持引用）
        embeddings : np.ndarray
            形状为 (N, dim) 的向量矩阵
        """
        assert embeddings.shape[1] == self.dim, \
            f"向量维度不匹配: 期望 {self.dim}, 实际 {embeddings.shape[1]}"
        assert len(documents) == embeddings.shape[0], \
            "文档数与向量数不匹配"

        self.documents = documents
        self._create_index(embeddings)

    def retrieve(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        nprobe: int = 10,
    ) -> List[Tuple[int, float]]:
        """
        检索 Top-K 结果

        Parameters
        ----------
        query_vector : np.ndarray
            查询向量，形状为 (dim,) 或 (1, dim)
        top_k : int
            返回数量
        nprobe : int
            IVF 探针数（仅 IVF 索引有效，越大越精确但越慢）

        Returns
        -------
        List[Tuple[int, float]]
            (文档索引, 相似度分数) 列表
        """
        if self.index is None:
            raise RuntimeError("请先调用 index_documents()")

        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1)

        # IVF 索引设置探针数
        if self.index_type == "ivf":
            faiss_params = faiss.ParameterSpace()
            faiss_params.set_index_parameter(self.index, "nprobe", nprobe)

        # 搜索
        scores, indices = self.index.search(query_vector, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS 返回 -1 表示无效结果
                continue
            results.append((int(idx), float(score)))

        return results

    def save(self, path: str):
        """
        保存索引到磁盘
        """
        import faiss
        os.makedirs(os.path.dirname(path), exist_ok=True)

        faiss.write_index(self.index, f"{path}.index")
        with open(f"{path}.docs.pkl", "wb") as f:
            pickle.dump({
                "documents": self.documents,
                "dim": self.dim,
                "index_type": self.index_type,
            }, f)
        print(f"[FAISSRetriever] 索引已保存到 {path}")

    @classmethod
    def load(cls, path: str) -> "FAISSRetriever":
        """从磁盘加载索引"""
        import faiss

        with open(f"{path}.docs.pkl", "rb") as f:
            meta = pickle.load(f)

        retriever = cls(
            dim=meta["dim"],
            index_type=meta["index_type"],
        )
        retriever.index = faiss.read_index(f"{path}.index")
        retriever.documents = meta["documents"]
        retriever.is_trained = True
        print(f"[FAISSRetriever] 索引已加载，文档数={retriever.index.ntotal}")
        return retriever


# ============================================================
# 使用示例
# ============================================================
if __name__ == "__main__":
    import numpy as np

    # 模拟 10000 篇文档的向量（768 维）
    np.random.seed(42)
    num_docs = 10000
    dim = 768
    embeddings = np.random.randn(num_docs, dim).astype(np.float32)
    # L2 归一化
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    docs = [f"doc_{i}" for i in range(num_docs)]

    retriever = FAISSRetriever(dim=dim, index_type="flat")
    retriever.index_documents(docs, embeddings)

    query_vec = np.random.randn(dim).astype(np.float32)
    query_vec = query_vec / np.linalg.norm(query_vec)

    results = retriever.retrieve(query_vec, top_k=5)
    print("\n检索结果:")
    for idx, score in results:
        print(f"  [{idx}] {docs[idx]} (score={score:.4f})")
```

### 7.2.4 使用 OpenAI Embeddings 的云端密集检索

```python
# ch07_openai_retriever.py
"""
基于 OpenAI Embeddings API 的云端密集检索
适合不想本地部署模型的场景
"""

import numpy as np
from typing import List, Optional, Tuple
import time


class OpenAIEmbeddingRetriever:
    """
    基于 OpenAI Embeddings 的检索器

    Parameters
    ----------
    model : str
        OpenAI Embedding 模型名
    api_key : str, optional
        API Key，默认从环境变量 OPENAI_API_KEY 读取
    max_retries : int
        API 调用失败时的最大重试次数
    batch_size : int
        API 批处理大小
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        max_retries: int = 3,
        batch_size: int = 20,
    ):
        self.model = model
        self.api_key = api_key
        self.max_retries = max_retries
        self.batch_size = batch_size
        self._client = None

    def _get_client(self):
        """懒加载 OpenAI 客户端"""
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("请安装 openai: pip install openai")
        return self._client

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        调用 OpenAI API 编码文本

        支持退避重试（exponential backoff）
        """
        client = self._get_client()
        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]

            for attempt in range(self.max_retries):
                try:
                    response = client.embeddings.create(
                        model=self.model,
                        input=batch,
                    )
                    # 按输入顺序排列
                    batch_embeddings = [
                        item.embedding
                        for item in sorted(response.data, key=lambda x: x.index)
                    ]
                    all_embeddings.extend(batch_embeddings)
                    break
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        raise RuntimeError(
                            f"OpenAI API 调用失败: {e}"
                        )
                    wait_time = 2 ** attempt
                    print(f"API 调用失败，{wait_time}s 后重试...")
                    time.sleep(wait_time)

        return np.array(all_embeddings, dtype=np.float32)

    def cosine_similarity(
        self, query_vec: np.ndarray, doc_matrix: np.ndarray
    ) -> np.ndarray:
        """计算余弦相似度矩阵"""
        # 归一化
        query_norm = query_vec / np.linalg.norm(query_vec)
        doc_norm = doc_matrix / np.linalg.norm(
            doc_matrix, axis=1, keepdims=True
        )
        return np.dot(doc_norm, query_norm)


# ============================================================
# 使用示例
# ============================================================
if __name__ == "__main__":
    # 注意：运行前请设置 OPENAI_API_KEY 环境变量
    retriever = OpenAIEmbeddingRetriever(
        model="text-embedding-3-small",
    )

    texts = [
        "RAG 系统通过检索增强生成能力",
        "向量数据库是实现语义搜索的核心组件",
        "Embedding 模型将文本映射到向量空间",
    ]

    print("正在编码文档...")
    embeddings = retriever.encode(texts)
    print(f"向量维度: {embeddings.shape[1]}")
    print(f"向量形状: {embeddings.shape}")
```

### 7.2.5 密集检索的优缺点总结

| 维度 | 说明 |
|---|---|
| 语义理解 | 强：能处理同义词、paraphrase、上下文歧义 |
| 零样本能力 | 取决于模型：BGE/SFR 等开源模型零样本表现好 |
| 多语言 | 取决于模型：mBERT/XLM-R 支持多语言 |
| 长文本 | 有长度限制（通常 512 tokens），超长需切片 |
| 计算成本 | 高：需要 GPU 编码 + 向量存储 + ANN 索引 |
| 可解释性 | 低：难以解释"为什么这两个向量相似" |
| 高频词 | 可能被罕见语义维度淹没 |

---

## 7.3 BM25 关键词稀疏检索（Sparse Retrieval）

### 7.3.1 BM25 算法原理

BM25（Best Matching 25）是概率检索模型的代表作，基于 TF-IDF 改进而来。其核心公式为：

```
BM25(q, d) = Σ_{t in q} IDF(t) * TF_BM25(t, d)

其中：
TF_BM25(t, d) = (k1 + 1) * f(t, d) / (k1 * (1 - b + b * |d|/avgdl) + f(t, d))
IDF(t) = log((N - n(t) + 0.5) / (n(t) + 0.5) + 1)
```

各参数含义：
- `f(t, d)`：词 t 在文档 d 中的词频
- `|d|`：文档长度
- `avgdl`：所有文档的平均长度
- `N`：文档总数
- `n(t)`：包含词 t 的文档数
- `k1`：词频饱和度（通常 1.2~1.5）
- `b`：长度归一化系数（通常 0.75）

BM25 的关键洞察：
1. **词频饱和**：TF 不是线性增长，达到阈值后边际收益递减（由 k1 控制）
2. **文档长度归一化**：长文档包含更多词是正常的，需要折扣（由 b 控制）
3. **逆文档频率**：稀有词的 IDF 更高，更有区分度

### 7.3.2 纯 Python 实现 BM25（教育版）

```python
# ch07_bm25_pure.py
"""
BM25 算法纯 Python 实现 —— 教育用途
展示 BM25 的每个计算细节
"""

import math
from collections import Counter
from typing import List, Dict, Tuple


class BM25Pure:
    """
    BM25 纯 Python 实现

    不依赖任何外部库，适合教学和调试。
    生产环境请使用 rank_bm25 或 Elasticsearch。

    Parameters
    ----------
    k1 : float
        词频饱和度参数 (1.2 ~ 2.0)
    b : float
        文档长度归一化参数 (0.0 ~ 1.0)
    epsilon : float
        防止除零的小常数
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 1e-10,
    ):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon

        # 以下在 fit() 时填充
        self.corpus: List[List[str]] = []          # 分词后的语料
        self.doc_lengths: List[int] = []           # 每篇文档的长度
        self.avgdl: float = 0.0                    # 平均文档长度
        self.N: int = 0                            # 文档总数
        self.df: Dict[str, int] = {}               # 文档频率 (term -> 包含该词的文档数)
        self.idf: Dict[str, float] = {}            # 逆文档频率缓存

    def _tokenize(self, text: str) -> List[str]:
        """
        分词器

        生产环境请用 jieba（中文）或 spaCy（英文）。
        这里做简单实现。
        """
        # 简单分词：按空格和标点分割，转小写
        import re
        tokens = re.findall(r"\w+", text.lower())
        return tokens

    def fit(self, documents: List[str]):
        """
        训练 BM25 模型

        统计文档频率、文档长度等统计量。
        """
        self.corpus = [self._tokenize(doc) for doc in documents]
        self.N = len(documents)
        self.doc_lengths = [len(tokens) for tokens in self.corpus]
        self.avgdl = sum(self.doc_lengths) / max(self.N, 1)

        # 计算文档频率
        self.df = {}
        for tokens in self.corpus:
            # 每篇文档中的词只计数一次
            unique_terms = set(tokens)
            for term in unique_terms:
                self.df[term] = self.df.get(term, 0) + 1

        # 预计算 IDF
        self.idf = {}
        for term, doc_freq in self.df.items():
            idf = math.log(
                (self.N - doc_freq + 0.5) / (doc_freq + 0.5) + 1
            )
            self.idf[term] = idf

        print(
            f"[BM25Pure] 训练完成，语料规模: {self.N} 篇文档, "
            f"词汇量: {len(self.df)}, 平均长度: {self.avgdl:.2f}"
        )

    def _score_document(self, query_tokens: List[str], doc_idx: int) -> float:
        """
        计算单篇文档相对于 query 的 BM25 分数

        这是 BM25 的核心计算逻辑。
        """
        doc_tokens = self.corpus[doc_idx]
        doc_len = self.doc_lengths[doc_idx]

        # 文档词频统计
        tf_counter = Counter(doc_tokens)

        score = 0.0
        for term in query_tokens:
            if term not in self.idf:
                # query 中的词在语料中未出现，跳过
                continue

            tf = tf_counter.get(term, 0)
            if tf == 0:
                continue

            idf = self.idf[term]

            # BM25 TF 变换（带长度归一化）
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (
                1 - self.b + self.b * doc_len / self.avgdl
            )
            tf_bm25 = numerator / max(denominator, self.epsilon)

            score += idf * tf_bm25

        return score

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Tuple[int, float]]:
        """
        检索 Top-K 文档

        Parameters
        ----------
        query : str
            查询文本
        top_k : int
            返回结果数

        Returns
        -------
        List[Tuple[int, float]]
            (文档索引, BM25分数) 列表
        """
        if not self.corpus:
            raise RuntimeError("请先调用 fit()")

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        # 计算所有文档的分数
        scores = []
        for i in range(self.N):
            score = self._score_document(query_tokens, i)
            scores.append((i, score))

        # 按分数降序排列
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores[:top_k]


# ============================================================
# 使用示例
# ============================================================
if __name__ == "__main__":
    documents = [
        "Transformer 架构由编码器和解码器组成",
        "BERT 使用双向注意力机制进行预训练",
        "GPT 系列模型采用自回归生成方式",
        "注意力机制允许模型关注输入序列的不同位置",
        "残差连接解决了深层网络的梯度消失问题",
    ]

    bm25 = BM25Pure(k1=1.5, b=0.75)
    bm25.fit(documents)

    query = "自回归生成"
    results = bm25.retrieve(query, top_k=3)

    print(f"\n查询: {query}")
    print("BM25 检索结果:")
    for idx, score in results:
        print(f"  [doc_{idx}] (score={score:.4f}) {documents[idx]}")

    # 展示每个词的 IDF
    print(f"\n词汇 IDF 统计:")
    for term, idf in sorted(
        bm25.idf.items(), key=lambda x: x[1], reverse=True
    )[:10]:
        df = bm25.df[term]
        print(f"  {term}: IDF={idf:.4f}, DF={df}/{bm25.N}")
```

### 7.3.3 使用 rank_bm25 库的高效实现

```python
# ch07_bm25_rank.py
"""
使用 rank_bm25 库的高效 BM25 实现
支持 BM25Okapi, BM25L, BM25Plus 多种变体
"""

from rank_bm25 import BM25Okapi, BM25L, BM25Plus
import jieba
from typing import List, Tuple


class ChineseBM25:
    """
    中文 BM25 检索器

    使用 jieba 分词 + rank_bm25 库。
    支持三种 BM25 变体：

    - BM25Okapi: 经典 BM25
    - BM25L: 对长文档更友好的变体
    - BM25Plus: 处理词频为 0 的场景

    Parameters
    ----------
    variant : str
        BM25 变体 ("okapi", "l", "plus")
    k1 : float
        词频饱和度参数
    b : float
        长度归一化参数
    delta : float
        BM25Plus 专用参数
    """

    def __init__(
        self,
        variant: str = "okapi",
        k1: float = 1.5,
        b: float = 0.75,
        delta: float = 0.5,
    ):
        self.variant = variant
        self.k1 = k1
        self.b = b
        self.delta = delta
        self._bm25 = None
        self._tokenized_corpus: List[List[str]] = []
        self.documents: List[str] = []

    def _tokenize(self, text: str) -> List[str]:
        """中文分词"""
        return list(jieba.cut(text))

    def fit(self, documents: List[str]):
        """
        训练 BM25 模型
        """
        self.documents = documents
        self._tokenized_corpus = [self._tokenize(doc) for doc in documents]

        if self.variant == "okapi":
            self._bm25 = BM25Okapi(
                self._tokenized_corpus,
                k1=self.k1,
                b=self.b,
            )
        elif self.variant == "l":
            self._bm25 = BM25L(
                self._tokenized_corpus,
                k1=self.k1,
                b=self.b,
            )
        elif self.variant == "plus":
            self._bm25 = BM25Plus(
                self._tokenized_corpus,
                k1=self.k1,
                b=self.b,
                delta=self.delta,
            )
        else:
            raise ValueError(f"不支持的变体: {self.variant}")

        print(
            f"[ChineseBM25] 训练完成，变体={self.variant}, "
            f"语料规模={len(documents)}"
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        检索 Top-K 文档

        Returns
        -------
        List[Tuple[str, float]]
            (文档文本, BM25分数) 列表
        """
        if self._bm25 is None:
            raise RuntimeError("请先调用 fit()")

        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        # 取 Top-K
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        return [
            (self.documents[i], float(scores[i]))
            for i in top_indices
        ]


# ============================================================
# 使用示例
# ============================================================
if __name__ == "__main__":
    documents = [
        "Transformer 架构由编码器和解码器组成，是 NLP 领域的里程碑",
        "BERT 使用双向注意力机制进行预训练，在 11 项任务上刷新纪录",
        "GPT 系列模型采用自回归生成方式，引领了 LLM 的发展",
        "注意力机制允许模型关注输入序列的不同位置，是核心创新",
        "残差连接解决了深层网络的梯度消失问题，使训练更稳定",
        "多头注意力从不同表示子空间学习信息，增强了模型表达能力",
        "位置编码为 Transformer 提供序列位置信息，弥补自注意力的不足",
        "Layer Normalization 稳定了训练过程，加速了模型收敛",
    ]

    bm25 = ChineseBM25(variant="okapi", k1=1.5, b=0.75)
    bm25.fit(documents)

    queries = [
        "自回归生成模型",
        "注意力机制",
        "训练稳定性",
    ]

    for query in queries:
        results = bm25.retrieve(query, top_k=2)
        print(f"\n查询: {query}")
        for doc_text, score in results:
            print(f"  (score={score:.4f}) {doc_text[:50]}...")
```

### 7.3.4 BM25 各变体对比

| 变体 | 核心改进 | 适用场景 |
|---|---|---|
| BM25Okapi | 经典实现，参数 k1=1.2~2.0, b=0.75 | 通用场景，默认选择 |
| BM25L | 对长文档中词频衰减更慢 | 文档长度差异大的场景 |
| BM25Plus | 避免词频为 0 时分数为 0 | query 中有生僻词时更鲁棒 |

### 7.3.5 密集检索 vs 稀疏检索对比

| 维度 | Dense (密集) | Sparse (稀疏/BM25) |
|---|---|---|
| 匹配方式 | 语义匹配（向量空间） | 精确词匹配（倒排索引） |
| 同义词处理 | 自动（"车" ≈ "汽车"） | 不处理 |
| 稀有实体 | 容易被忽略 | 高 IDF，精确命中 |
| 训练需求 | 需要训练数据或预训练模型 | 零训练，仅需统计 |
| 推理速度 | O(N*dim) 或 ANN 近似 | O(N * avg_query_len) |
| 存储 | 768/1024 维 float 向量 | 倒排索引（稀疏） |
| 可解释性 | 低 | 高（知道是哪个词匹配的） |

---

## 7.4 结构化元数据检索（Structured Retrieval）

### 7.4.1 为什么需要结构化检索

向量检索和 BM25 都基于文本内容，但很多场景需要精确的字段匹配：

- **时间范围**："2024年发表的论文"
- **类别过滤**："只看技术类文章"
- **来源限制**："来自 arxiv.org"
- **作者过滤**："作者是 Geoffrey Hinton"
- **质量门槛**："引用数 > 100"

结构化检索与内容检索结合，形成 **混合检索（Hybrid Search）** 策略。

### 7.4.2 基于 SQLite 的结构化 + 向量混合检索

```python
# ch07_structured_retriever.py
"""
结构化元数据检索 + 混合检索实现

支持：
1. 纯结构化过滤（SQL WHERE 子句）
2. 结构化过滤 + 向量排序
3. 结构化过滤 + BM25 排序
"""

import sqlite3
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class StructuredDocument:
    """带结构化字段的文档"""
    id: str
    title: str
    content: str
    source: str = ""
    category: str = ""
    author: str = ""
    published_at: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    score: float = 0.0
    citation_count: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)


class StructuredRetriever:
    """
    结构化检索器

    使用 SQLite 作为存储后端，支持：
    - 等值过滤 (category = 'tech')
    - 范围过滤 (published_at > '2024-01-01')
    - IN 过滤 (source IN ('arxiv', 'openreview'))
    - 标签包含 (tag IN ('transformer', 'attention'))
    - 排序与分页

    Parameters
    ----------
    db_path : str
        SQLite 数据库路径
    """

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        """创建文档表和标签表"""
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                source TEXT DEFAULT '',
                category TEXT DEFAULT '',
                author TEXT DEFAULT '',
                published_at TEXT,
                citation_count INTEGER DEFAULT 0,
                extra TEXT DEFAULT '{}',
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS doc_tags (
                doc_id TEXT NOT NULL,
                tag TEXT NOT NULL,
                PRIMARY KEY (doc_id, tag),
                FOREIGN KEY (doc_id) REFERENCES documents(id)
            )
        """)
        # 索引
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_category ON documents(category)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_source ON documents(source)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_published ON documents(published_at)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_citation ON documents(citation_count)"
        )
        self.conn.commit()

    def add_documents(self, docs: List[StructuredDocument]):
        """
        批量添加文档

        Parameters
        ----------
        docs : List[StructuredDocument]
            待添加的文档列表
        """
        cursor = self.conn.cursor()
        for doc in docs:
            cursor.execute("""
                INSERT OR REPLACE INTO documents
                (id, title, content, source, category, author,
                 published_at, citation_count, extra)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                doc.id, doc.title, doc.content, doc.source,
                doc.category, doc.author, doc.published_at,
                doc.citation_count, json.dumps(doc.extra),
            ))
            # 插入标签
            for tag in doc.tags:
                cursor.execute(
                    "INSERT OR REPLACE INTO doc_tags (doc_id, tag) VALUES (?, ?)",
                    (doc.id, tag),
                )
        self.conn.commit()
        print(f"[StructuredRetriever] 已添加 {len(docs)} 篇文档")

    def _build_filter_query(
        self,
        filters: Dict[str, Any],
    ) -> Tuple[str, List]:
        """
        构建 WHERE 子句

        filters 支持以下键：
        - category: str — 等值匹配
        - source: str — 等值匹配
        - author: str — 等值匹配
        - tags: List[str] — 标签包含（AND 逻辑）
        - published_after: str — 发布日期 >=
        - published_before: str — 发布日期 <=
        - min_citations: int — 引用数 >=
        - max_citations: int — 引用数 <=
        - keyword: str — 内容关键词模糊匹配
        """
        conditions = []
        params = []

        if "category" in filters:
            conditions.append("d.category = ?")
            params.append(filters["category"])

        if "source" in filters:
            conditions.append("d.source = ?")
            params.append(filters["source"])

        if "author" in filters:
            conditions.append("d.author = ?")
            params.append(filters["author"])

        if "tags" in filters and filters["tags"]:
            # 文档必须包含所有指定标签
            for tag in filters["tags"]:
                conditions.append(
                    "d.id IN (SELECT doc_id FROM doc_tags WHERE tag = ?)"
                )
                params.append(tag)

        if "published_after" in filters:
            conditions.append("d.published_at >= ?")
            params.append(filters["published_after"])

        if "published_before" in filters:
            conditions.append("d.published_at <= ?")
            params.append(filters["published_before"])

        if "min_citations" in filters:
            conditions.append("d.citation_count >= ?")
            params.append(filters["min_citations"])

        if "max_citations" in filters:
            conditions.append("d.citation_count <= ?")
            params.append(filters["max_citations"])

        if "keyword" in filters:
            conditions.append("d.content LIKE ?")
            params.append(f"%{filters['keyword']}%")

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        return where_clause, params

    def retrieve_by_filters(
        self,
        filters: Dict[str, Any],
        top_k: int = 10,
        order_by: str = "citation_count DESC",
    ) -> List[StructuredDocument]:
        """
        纯结构化过滤检索

        Parameters
        ----------
        filters : Dict[str, Any]
            过滤条件
        top_k : int
            返回数量
        order_by : str
            排序字段

        Returns
        -------
        List[StructuredDocument]
            匹配的文档列表
        """
        where_clause, params = self._build_filter_query(filters)

        query = f"""
            SELECT d.*, GROUP_CONCAT(t.tag, ',') as tags_str
            FROM documents d
            LEFT JOIN doc_tags t ON d.id = t.doc_id
            WHERE {where_clause}
            GROUP BY d.id
            ORDER BY {order_by}
            LIMIT ?
        """
        params.append(top_k)

        cursor = self.conn.execute(query, params)
        rows = cursor.fetchall()

        results = []
        for row in rows:
            tags = row["tags_str"].split(",") if row["tags_str"] else []
            results.append(StructuredDocument(
                id=row["id"],
                title=row["title"],
                content=row["content"],
                source=row["source"],
                category=row["category"],
                author=row["author"],
                published_at=row["published_at"],
                tags=tags,
                citation_count=row["citation_count"],
                extra=json.loads(row["extra"]),
            ))

        return results

    def count_by_filters(self, filters: Dict[str, Any]) -> int:
        """返回匹配条件的文档数"""
        where_clause, params = self._build_filter_query(filters)
        query = f"SELECT COUNT(*) FROM documents d WHERE {where_clause}"
        cursor = self.conn.execute(query, params)
        return cursor.fetchone()[0]


# ============================================================
# 使用示例
# ============================================================
if __name__ == "__main__":
    retriever = StructuredRetriever()

    # 添加示例文档
    docs = [
        StructuredDocument(
            id="1",
            title="Transformer: Attention Is All You Need",
            content="Transformer 架构由编码器和解码器组成...",
            source="arxiv",
            category="deep_learning",
            author="Vaswani et al.",
            published_at="2017-06-12",
            tags=["transformer", "attention", "nlp"],
            citation_count=95000,
        ),
        StructuredDocument(
            id="2",
            title="BERT: Pre-training of Deep Bidirectional Transformers",
            content="BERT 使用双向注意力机制进行预训练...",
            source="arxiv",
            category="deep_learning",
            author="Devlin et al.",
            published_at="2018-10-11",
            tags=["bert", "pretraining", "nlp"],
            citation_count=65000,
        ),
        StructuredDocument(
            id="3",
            title="GPT-3: Language Models are Few-Shot Learners",
            content="GPT-3 展示了大规模语言模型的少样本学习能力...",
            source="arxiv",
            category="deep_learning",
            author="Brown et al.",
            published_at="2020-05-28",
            tags=["gpt", "few-shot", "llm"],
            citation_count=35000,
        ),
        StructuredDocument(
            id="4",
            title="LoRA: Low-Rank Adaptation of Large Language Models",
            content="LoRA 通过低秩矩阵实现高效微调...",
            source="openreview",
            category="efficiency",
            author="Hu et al.",
            published_at="2021-06-15",
            tags=["lora", "fine-tuning", "efficiency"],
            citation_count=12000,
        ),
    ]
    retriever.add_documents(docs)

    # 示例 1: 按类别和来源过滤
    print("\n=== 示例 1: arxiv 上的 deep_learning 论文 ===")
    results = retriever.retrieve_by_filters(
        filters={"category": "deep_learning", "source": "arxiv"},
        order_by="citation_count DESC",
    )
    for doc in results:
        print(f"  [{doc.id}] {doc.title} (引用: {doc.citation_count})")

    # 示例 2: 时间范围 + 最低引用数
    print("\n=== 示例 2: 2020年后且引用 > 10000 ===")
    results = retriever.retrieve_by_filters(
        filters={
            "published_after": "2020-01-01",
            "min_citations": 10000,
        },
    )
    for doc in results:
        print(f"  [{doc.id}] {doc.title} ({doc.published_at})")

    # 示例 3: 标签过滤
    print("\n=== 示例 3: 包含 'attention' 标签 ===")
    results = retriever.retrieve_by_filters(
        filters={"tags": ["attention"]},
    )
    for doc in results:
        print(f"  [{doc.id}] {doc.title}, tags: {doc.tags}")

    # 示例 4: 组合查询
    print("\n=== 示例 4: 复杂组合查询 ===")
    count = retriever.count_by_filters({
        "category": "deep_learning",
        "min_citations": 50000,
    })
    print(f"  引用数 >= 50000 的 deep_learning 论文数: {count}")
```

### 7.4.3 结构化过滤 + 向量检索的混合检索

```python
# ch07_hybrid_structured_dense.py
"""
结构化过滤 + 向量检索的混合检索

策略：
1. 先用结构化条件过滤出候选集合
2. 再对候选集合做向量相似度排序

优势：减少向量检索范围，提高精度和效率
"""

from typing import List, Optional, Tuple
import numpy as np


class HybridDenseRetriever:
    """
    混合检索器（结构化过滤 + 密集向量排序）

    支持三种模式：
    - "post_filter": 先向量检索，再应用结构化过滤
    - "pre_filter": 先用结构化过滤缩小范围，再向量检索
    - "two_stage": 先用结构化粗略过滤，再向量排序，最后重排

    Parameters
    ----------
    dense_retriever : FAISSRetriever
        向量检索器
    structured_retriever : StructuredRetriever
        结构化检索器
    mode : str
        混合模式
    """

    def __init__(
        self,
        dense_retriever,
        structured_retriever,
        mode: str = "pre_filter",
    ):
        self.dense = dense_retriever
        self.structured = structured_retriever
        self.mode = mode

    def retrieve(
        self,
        query_vector: np.ndarray,
        query_text: str = "",
        filters: Optional[dict] = None,
        top_k: int = 10,
        dense_candidate_k: int = 100,
        struct_candidate_k: int = 200,
    ) -> List[Tuple[str, float]]:
        """
        混合检索

        Parameters
        ----------
        query_vector : np.ndarray
            查询向量
        query_text : str
            查询文本（用于 fallback）
        filters : dict, optional
            结构化过滤条件
        top_k : int
            最终返回数量
        dense_candidate_k : int
            向量检索候选数量（post_filter 模式）
        struct_candidate_k : int
            结构化检索候选数量（pre_filter 模式）

        Returns
        -------
        List[Tuple[str, float]]
            (文档ID, 分数) 列表
        """
        if self.mode == "post_filter":
            return self._post_filter(
                query_vector, filters, top_k, dense_candidate_k
            )
        elif self.mode == "pre_filter":
            return self._pre_filter(
                query_vector, filters, top_k, struct_candidate_k
            )
        elif self.mode == "two_stage":
            return self._two_stage(
                query_vector, filters, top_k,
                dense_candidate_k, struct_candidate_k,
            )
        else:
            raise ValueError(f"不支持的混合模式: {self.mode}")

    def _post_filter(
        self,
        query_vec: np.ndarray,
        filters: Optional[dict],
        top_k: int,
        candidate_k: int,
    ) -> List[Tuple[str, float]]:
        """
        后过滤模式

        先向量检索出较多候选，再用结构化条件过滤。
        适合结构化过滤条件宽松、不想遗漏的场景。
        """
        # 1. 向量检索（取更多候选）
        dense_results = self.dense.retrieve(query_vec, top_k=candidate_k)

        # 2. 结构化过滤
        filtered = []
        if filters:
            for doc_idx, score in dense_results:
                doc_id = self.dense.documents[doc_idx]
                # 检查文档是否匹配过滤条件
                doc = self.structured.retrieve_by_filters(
                    {"id": doc_id}, top_k=1
                )
                if doc:
                    filtered.append((doc_id, score))
        else:
            filtered = [
                (self.dense.documents[idx], score)
                for idx, score in dense_results
            ]

        return filtered[:top_k]

    def _pre_filter(
        self,
        query_vec: np.ndarray,
        filters: Optional[dict],
        top_k: int,
        candidate_k: int,
    ) -> List[Tuple[str, float]]:
        """
        预过滤模式

        先用结构化条件缩小范围，再向量检索。
        适合结构化过滤条件严格、候选集大幅缩小的场景。
        """
        # 1. 结构化过滤
        filtered_docs = self.structured.retrieve_by_filters(
            filters or {}, top_k=candidate_k
        )
        filtered_ids = {doc.id for doc in filtered_docs}

        # 2. 在过滤后的子集上做向量检索
        # （简化实现：先全量检索再过滤）
        all_dense = self.dense.retrieve(query_vec, top_k=candidate_k)
        filtered = [
            (self.dense.documents[idx], score)
            for idx, score in all_dense
            if self.dense.documents[idx] in filtered_ids
        ]

        return filtered[:top_k]

    def _two_stage(
        self,
        query_vec: np.ndarray,
        filters: Optional[dict],
        top_k: int,
        dense_k: int,
        struct_k: int,
    ) -> List[Tuple[str, float]]:
        """
        两阶段模式

        第一轮：向量检索 + 结构化检索 各自取候选
        第二轮：合并候选集，重新排序
        """
        # 第一轮：多路检索
        dense_results = self.dense.retrieve(query_vec, top_k=dense_k)
        struct_results = self.structured.retrieve_by_filters(
            filters or {}, top_k=struct_k
        )

        # 合并候选集
        candidate_ids = set()
        for idx, _ in dense_results:
            candidate_ids.add(self.dense.documents[idx])
        for doc in struct_results:
            candidate_ids.add(doc.id)

        # 第二轮：在合并集中做向量排序
        candidate_list = list(candidate_ids)
        if not candidate_list:
            return []

        # 找到这些候选在向量索引中的位置并计算分数
        scored = []
        for doc_id in candidate_list:
            if doc_id in self.dense.doc_id_to_idx:
                idx = self.dense.doc_id_to_idx[doc_id]
                score = float(np.dot(
                    self.dense.embedding_matrix[idx], query_vec
                ))
                scored.append((doc_id, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
```

### 7.4.4 基于时间衰减的检索

时间敏感性是结构化检索中的重要场景。最新的信息通常比旧信息更有价值。

```python
# ch07_time_decay.py
"""
时间衰减评分

在检索结果排序中引入时间维度，
让较新的结果获得分数加成。
"""

import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple


class TimeDecayScorer:
    """
    时间衰减评分器

    支持多种衰减策略：
    - "linear": 线性衰减
    - "exponential": 指数衰减
    - "step": 阶梯衰减（按时间窗口分档）

    Parameters
    ----------
    strategy : str
        衰减策略
    half_life_days : int
        半衰期（天数），exponential 策略专用
    max_boost_days : int
        最大提升天数，linear 策略专用
    """

    def __init__(
        self,
        strategy: str = "exponential",
        half_life_days: int = 30,
        max_boost_days: int = 365,
    ):
        self.strategy = strategy
        self.half_life_days = half_life_days
        self.max_boost_days = max_boost_days

    def time_decay_factor(
        self,
        published_date: str,
        reference_date: Optional[str] = None,
    ) -> float:
        """
        计算时间衰减因子（0 ~ 1+）

        返回 > 1 表示加分（近期），< 1 表示减分（远期）
        """
        pub = datetime.strptime(published_date, "%Y-%m-%d")
        ref = (
            datetime.strptime(reference_date, "%Y-%m-%d")
            if reference_date
            else datetime.now()
        )

        days_diff = (ref - pub).days

        if days_diff < 0:
            return 1.0  # 未来的文档不做衰减

        if self.strategy == "linear":
            # 线性衰减：max_boost_days 内从 1.5 线性降到 1.0
            if days_diff <= self.max_boost_days:
                return 1.5 - 0.5 * (days_diff / self.max_boost_days)
            else:
                return 0.5

        elif self.strategy == "exponential":
            # 指数衰减：半衰期后分数减半
            # factor = 2^(-days / half_life)
            # 第 0 天: 1.0, 第 half_life 天: 0.5, 第 2*half_life 天: 0.25
            return float(np.power(2.0, -days_diff / self.half_life_days))

        elif self.strategy == "step":
            # 阶梯衰减
            if days_diff <= 7:
                return 1.2  # 一周内
            elif days_diff <= 30:
                return 1.0  # 一个月内
            elif days_diff <= 90:
                return 0.8  # 三个月内
            elif days_diff <= 365:
                return 0.6  # 一年内
            else:
                return 0.4  # 一年以上

        else:
            raise ValueError(f"不支持的策略: {self.strategy}")

    def apply_time_decay(
        self,
        results: List[Tuple[str, float, str]],
        alpha: float = 0.3,
    ) -> List[Tuple[str, float]]:
        """
        对检索结果应用时间衰减

        Parameters
        ----------
        results : List[Tuple[str, float, str]]
            (文档ID, 原始分数, 发布日期) 列表
        alpha : float
            时间因素的权重（0 ~ 1），越大时间影响越强

        Returns
        -------
        List[Tuple[str, float]]
            (文档ID, 调整后分数) 列表
        """
        adjusted = []
        for doc_id, original_score, pub_date in results:
            decay = self.time_decay_factor(pub_date)
            # 加权融合：保留 alpha 权重给时间，1-alpha 给原始分数
            adjusted_score = (1 - alpha) * original_score + alpha * decay
            adjusted.append((doc_id, adjusted_score))

        adjusted.sort(key=lambda x: x[1], reverse=True)
        return adjusted


# ============================================================
# 使用示例
# ============================================================
if __name__ == "__main__":
    scorer = TimeDecayScorer(
        strategy="exponential",
        half_life_days=30,
    )

    # 模拟检索结果
    results = [
        ("doc_1", 0.85, "2024-12-01"),  # 7 个月前
        ("doc_2", 0.82, "2025-06-15"),  # ~0.5 个月前
        ("doc_3", 0.78, "2025-01-01"),  # 6 个月前
        ("doc_4", 0.75, "2025-05-01"),  # 2 个月前
    ]

    print("原始排序:")
    for doc_id, score, date in results:
        print(f"  {doc_id}: score={score:.4f}, date={date}")

    for alpha in [0.0, 0.3, 0.5, 1.0]:
        adjusted = scorer.apply_time_decay(results, alpha=alpha)
        print(f"\n时间衰减后 (alpha={alpha}):")
        for doc_id, score in adjusted:
            print(f"  {doc_id}: adjusted_score={score:.4f}")
```

---

## 7.5 知识图谱检索（KG Retrieval）

### 7.5.1 为什么需要知识图谱检索

知识图谱（Knowledge Graph, KG）以三元组 `(头实体, 关系, 尾实体)` 的形式组织结构化知识。在 RAG 中引入 KG 检索可以：

1. **精确实体匹配**：`(Transformer, 发明年份, 2017)` — 比纯文本更精确
2. **多跳推理**：`A 是 B 的作者` + `B 获得了 C 奖` → `A 可能知道 C 奖的信息`
3. **关系导向检索**：按关系类型筛选（因果关系、属性关系等）
4. **社区检测**：从图谱中挖掘相关实体群组

### 7.5.2 知识图谱构建

```python
# ch07_kg_build.py
"""
轻量级知识图谱构建
支持三元组存储、实体检索、路径查询
"""

from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict
import json


class KnowledgeGraph:
    """
    轻量级知识图谱

    支持：
    - 三元组存储（头实体, 关系, 尾实体）
    - 实体属性存储
    - 邻接查询
    - 路径检索
    - 社区检测

    存储结构：
    - forward_edges: (head) -> [(rel, tail)]
    - backward_edges: (tail) -> [(rel, head)]
    - entity_properties: (entity) -> {prop: value}
    """

    def __init__(self):
        # 正向边索引
        self.forward_edges: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        # 反向边索引
        self.backward_edges: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        # 实体属性
        self.entity_properties: Dict[str, Dict] = defaultdict(dict)
        # 所有实体集合
        self.entities: Set[str] = set()
        # 所有关系集合
        self.relations: Set[str] = set()

    def add_triple(
        self,
        head: str,
        relation: str,
        tail: str,
        properties: Optional[dict] = None,
    ):
        """
        添加三元组

        Parameters
        ----------
        head : str
            头实体
        relation : str
            关系
        tail : str
            尾实体
        properties : dict, optional
            三元组属性（如置信度、来源等）
        """
        self.entities.add(head)
        self.entities.add(tail)
        self.relations.add(relation)

        triple_data = (relation, tail, properties or {})
        self.forward_edges[head].append(triple_data)

        reverse_data = (relation, head, properties or {})
        self.backward_edges[tail].append(reverse_data)

    def add_entity_property(self, entity: str, key: str, value):
        """添加实体属性"""
        self.entities.add(entity)
        self.entity_properties[entity][key] = value

    def get_neighbors(
        self,
        entity: str,
        relation: Optional[str] = None,
        direction: str = "both",
    ) -> List[Tuple[str, str, str, dict]]:
        """
        获取实体的邻接实体

        Parameters
        ----------
        entity : str
            查询实体
        relation : str, optional
            按关系过滤
        direction : str
            "forward": 只向前（head -> tail）
            "backward": 只向后（tail -> head）
            "both": 双向

        Returns
        -------
        List[Tuple[str, str, str, dict]]
            [(关系, 邻接实体, 方向, 属性)]
        """
        results = []

        if direction in ("forward", "both"):
            for rel, tail, props in self.forward_edges.get(entity, []):
                if relation is None or rel == relation:
                    results.append((rel, tail, "forward", props))

        if direction in ("backward", "both"):
            for rel, head, props in self.backward_edges.get(entity, []):
                if relation is None or rel == relation:
                    results.append((rel, head, "backward", props))

        return results

    def find_paths(
        self,
        start: str,
        end: str,
        max_depth: int = 3,
    ) -> List[List[Tuple[str, str, str]]]:
        """
        BFS 查找最短路径

        Returns
        -------
        List[List[Tuple[str, str, str]]]
            路径列表，每条路径由 (实体, 关系, 方向) 组成
        """
        if start not in self.entities or end not in self.entities:
            return []

        # BFS
        visited = {start}
        queue = [[(start, "", "")]]  # (entity, relation, direction)

        while queue:
            path = queue.pop(0)
            current = path[-1][0]

            if current == end:
                return [path]  # 找到一条最短路径

            if len(path) - 1 >= max_depth:
                continue

            for rel, neighbor, direction, _ in self.get_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    new_path = path + [(neighbor, rel, direction)]
                    queue.append(new_path)

        return []  # 未找到路径

    def get_entity_context(
        self,
        entity: str,
        depth: int = 1,
        max_neighbors: int = 20,
    ) -> Dict:
        """
        获取实体的局部上下文子图

        Parameters
        ----------
        entity : str
            中心实体
        depth : int
            扩展深度
        max_neighbors : int
            每层最大邻居数

        Returns
        -------
        Dict
            {"triples": [...], "properties": {...}, "entities": [...]}
        """
        context = {
            "triples": [],
            "properties": dict(self.entity_properties.get(entity, {})),
            "entities": {entity},
        }

        current_level = {entity}
        for d in range(depth):
            next_level = set()
            for e in current_level:
                neighbors = self.get_neighbors(e)
                for rel, neighbor, direction, props in neighbors[:max_neighbors]:
                    context["triples"].append({
                        "head": e if direction == "forward" else neighbor,
                        "relation": rel,
                        "tail": neighbor if direction == "forward" else e,
                    })
                    context["entities"].add(neighbor)
                    next_level.add(neighbor)
            current_level = next_level

        context["entities"] = list(context["entities"])
        return context

    def community_detection(
        self,
        min_community_size: int = 2,
    ) -> List[Set[str]]:
        """
        简单社区检测（基于连通分量）

        对于大型图谱，请使用 Louvain / Leiden 算法。
        """
        visited = set()
        communities = []

        for entity in self.entities:
            if entity in visited:
                continue

            # BFS 找连通分量
            community = set()
            queue = [entity]
            visited.add(entity)

            while queue:
                current = queue.pop(0)
                community.add(current)

                for _, neighbor, _, _ in self.get_neighbors(current):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

            if len(community) >= min_community_size:
                communities.append(community)

        return communities

    def to_dict(self) -> dict:
        """导出为字典"""
        triples = []
        for head, edges in self.forward_edges.items():
            for rel, tail, props in edges:
                triples.append({
                    "head": head,
                    "relation": rel,
                    "tail": tail,
                    "properties": props,
                })
        return {
            "entities": list(self.entities),
            "relations": list(self.relations),
            "triples": triples,
            "entity_properties": dict(self.entity_properties),
        }

    def save(self, path: str):
        """保存到 JSON 文件"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        print(f"[KG] 已保存到 {path}, 实体数={len(self.entities)}")

    @classmethod
    def load(cls, path: str) -> "KnowledgeGraph":
        """从 JSON 文件加载"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        kg = cls()
        for triple in data.get("triples", []):
            kg.add_triple(
                triple["head"],
                triple["relation"],
                triple["tail"],
                triple.get("properties"),
            )
        for entity, props in data.get("entity_properties", {}).items():
            for key, value in props.items():
                kg.add_entity_property(entity, key, value)

        return kg


# ============================================================
# 使用示例
# ============================================================
if __name__ == "__main__":
    kg = KnowledgeGraph()

    # 添加 NLP 领域知识
    kg.add_triple("Transformer", "提出", "Attention机制")
    kg.add_triple("Transformer", "应用", "机器翻译")
    kg.add_triple("Transformer", "作者", "Vaswani等人")
    kg.add_triple("BERT", "基于", "Transformer")
    kg.add_triple("BERT", "提出", "Masked Language Model")
    kg.add_triple("BERT", "作者", "Devlin等人")
    kg.add_triple("GPT", "基于", "Transformer")
    kg.add_triple("GPT", "提出", "自回归生成")
    kg.add_triple("GPT-3", "属于", "GPT系列")
    kg.add_triple("GPT-3", "提出", "Few-shot Learning")
    kg.add_triple("GPT-3", "作者", "Brown等人")

    # 实体属性
    kg.add_entity_property("Transformer", "年份", 2017)
    kg.add_entity_property("BERT", "年份", 2018)
    kg.add_entity_property("GPT-3", "年份", 2020)

    # 查询示例
    print("=== 邻居查询: Transformer ===")
    for rel, neighbor, direction, props in kg.get_neighbors("Transformer"):
        print(f"  {direction}: --[{rel}]--> {neighbor}")

    print("\n=== 路径查找: GPT-3 -> Attention机制 ===")
    paths = kg.find_paths("GPT-3", "Attention机制", max_depth=4)
    for path in paths:
        print("  " + " -> ".join(
            f"{node}({rel})" for node, rel, _ in path
        ))

    print("\n=== 上下文子图: BERT (depth=1) ===")
    ctx = kg.get_entity_context("BERT", depth=1)
    for t in ctx["triples"]:
        print(f"  ({t['head']}) --[{t['relation']}]--> ({t['tail']})")

    print(f"\n=== 社区检测 ===")
    communities = kg.community_detection()
    for i, comm in enumerate(communities):
        print(f"  社区 {i+1}: {comm}")
```

### 7.5.3 基于知识图谱的检索

```python
# ch07_kg_retriever.py
"""
基于知识图谱的 RAG 检索器

支持三种检索模式：
1. Entity Retrieval: 从 query 中提取实体，检索实体属性
2. Path Retrieval: 检索实体之间的路径关系
3. Community Retrieval: 检索相关社区的所有三元组
"""

from typing import List, Dict, Tuple, Optional
import re
import numpy as np


class KGRetriever:
    """
    知识图谱检索器

    将用户 query 映射到图谱中的实体，然后检索相关信息。

    Parameters
    ----------
    kg : KnowledgeGraph
        已构建的知识图谱
    entity_extractor : callable, optional
        从 query 中提取实体的函数
    max_path_depth : int
        路径检索最大深度
    max_community_size : int
        社区检索最大实体数
    """

    def __init__(
        self,
        kg: "KnowledgeGraph",
        entity_extractor=None,
        max_path_depth: int = 3,
        max_community_size: int = 50,
    ):
        self.kg = kg
        self.max_path_depth = max_path_depth
        self.max_community_size = max_community_size

        if entity_extractor is None:
            self.entity_extractor = self._default_entity_extractor
        else:
            self.entity_extractor = entity_extractor

    def _default_entity_extractor(self, query: str) -> List[str]:
        """
        默认实体提取器

        在 KG 已知实体中匹配 query 中的子串。
        生产环境请使用 NER 模型（spaCy / HanLP）。
        """
        matched = []
        for entity in self.kg.entities:
            if entity.lower() in query.lower():
                matched.append(entity)
        return matched

    def retrieve_by_entity(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Dict]:
        """
        实体检索模式

        1. 从 query 中提取实体
        2. 获取每个实体的邻接三元组
        3. 按相关性排序

        Returns
        -------
        List[Dict]
            [{"head", "relation", "tail", "entity", "type"}, ...]
        """
        entities = self.entity_extractor(query)
        if not entities:
            return []

        results = []
        for entity in entities:
            # 实体属性
            props = self.kg.entity_properties.get(entity, {})
            for key, value in props.items():
                results.append({
                    "head": entity,
                    "relation": f"属性:{key}",
                    "tail": str(value),
                    "entity": entity,
                    "type": "property",
                })

            # 前向边
            for rel, tail, _ in self.kg.forward_edges.get(entity, []):
                results.append({
                    "head": entity,
                    "relation": rel,
                    "tail": tail,
                    "entity": entity,
                    "type": "forward",
                })

            # 后向边
            for rel, head, _ in self.kg.backward_edges.get(entity, []):
                results.append({
                    "head": head,
                    "relation": rel,
                    "tail": entity,
                    "entity": entity,
                    "type": "backward",
                })

        return results[:top_k]

    def retrieve_by_path(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Dict]:
        """
        路径检索模式

        当 query 中包含多个实体时，检索它们之间的路径。
        """
        entities = self.entity_extractor(query)
        if len(entities) < 2:
            # 如果只有一个实体，回退到实体检索
            return self.retrieve_by_entity(query, top_k)

        results = []

        # 对所有实体对找路径
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                paths = self.kg.find_paths(
                    entities[i], entities[j],
                    max_depth=self.max_path_depth,
                )
                for path in paths:
                    results.append({
                        "path": [
                            {"entity": node, "relation": rel}
                            for node, rel, _ in path
                        ],
                        "source": entities[i],
                        "target": entities[j],
                        "type": "path",
                        "path_length": len(path) - 1,
                    })

        return results[:top_k]

    def retrieve_by_community(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Dict]:
        """
        社区检索模式

        1. 提取 query 中的实体
        2. 找到实体所在的社区
        3. 返回社区内所有三元组
        """
        entities = self.entity_extractor(query)
        if not entities:
            return []

        communities = self.kg.community_detection()

        # 找到包含查询实体的社区
        relevant_communities = []
        for comm in communities:
            if any(e in comm for e in entities):
                relevant_communities.append(comm)

        results = []
        for comm in relevant_communities:
            for entity in list(comm)[:self.max_community_size]:
                for rel, tail, _ in self.kg.forward_edges.get(entity, []):
                    if tail in comm:
                        results.append({
                            "head": entity,
                            "relation": rel,
                            "tail": tail,
                            "community_size": len(comm),
                            "type": "community",
                        })

        return results[:top_k]

    def retrieve(
        self,
        query: str,
        mode: str = "entity",
        top_k: int = 10,
    ) -> List[Dict]:
        """
        统一检索接口

        Parameters
        ----------
        query : str
            用户查询
        mode : str
            检索模式: "entity" | "path" | "community"
        top_k : int
            返回数量

        Returns
        -------
        List[Dict]
            检索结果
        """
        if mode == "entity":
            return self.retrieve_by_entity(query, top_k)
        elif mode == "path":
            return self.retrieve_by_path(query, top_k)
        elif mode == "community":
            return self.retrieve_by_community(query, top_k)
        else:
            raise ValueError(f"不支持的检索模式: {mode}")


# ============================================================
# 使用示例
# ============================================================
if __name__ == "__main__":
    # 使用上一节构建的 KG
    from ch07_kg_build import KnowledgeGraph

    kg = KnowledgeGraph()

    # 构建一个小型 NLP 知识图谱
    kg.add_triple("Transformer", "提出", "Attention机制")
    kg.add_triple("BERT", "基于", "Transformer")
    kg.add_triple("GPT", "基于", "Transformer")
    kg.add_triple("GPT-3", "属于", "GPT系列")
    kg.add_triple("GPT-3", "提出", "Few-shot Learning")
    kg.add_entity_property("Transformer", "年份", 2017)
    kg.add_entity_property("BERT", "年份", 2018)
    kg.add_entity_property("GPT-3", "年份", 2020)

    retriever = KGRetriever(kg)

    # 实体检索
    print("=== 实体检索: Transformer ===")
    results = retriever.retrieve_by_entity("Transformer")
    for r in results:
        print(f"  ({r['head']}) --[{r['relation']}]--> ({r['tail']})")

    # 路径检索
    print("\n=== 路径检索: GPT-3 和 Attention机制 ===")
    results = retriever.retrieve_by_path("GPT-3 和 Attention机制 的关系")
    for r in results:
        path_str = " -> ".join(
            f"{p['entity']}({p['relation']})"
            for p in r["path"]
        )
        print(f"  路径 (长度={r['path_length']}): {path_str}")

    # 社区检索
    print("\n=== 社区检索: BERT ===")
    results = retriever.retrieve_by_community("BERT")
    for r in results[:5]:
        print(f"  ({r['head']}) --[{r['relation']}]--> ({r['tail']})")
```

### 7.5.4 GraphRAG：微软的高级图检索方案

微软的 GraphRAG 是 KG 检索的进阶方案，其核心流程为：

```
1. Entity Extraction: 从文档中提取实体和关系 → 构建图谱
2. Community Detection: 使用 Leiden 算法发现社区
3. Community Summarization: 对每个社区生成自然语言摘要
4. Retrieval: 根据 query 找到相关社区，返回社区摘要 + 关键三元组
```

GraphRAG 的核心优势在于：
- **全局理解**：不只是检索单个实体，而是理解整个主题社区
- **多层次摘要**：从局部（实体级）到全局（社区级）的信息
- **关系推理**：发现文档中没有直接写出的隐式关系

---

## 7.6 结果融合：RRF 算法

### 7.6.1 RRF 的原理

**RRF（Reciprocal Rank Fusion，倒数排名融合）** 是目前最主流的多路召回融合算法。其核心思想是：

> **每路检索结果的排名（而非分数）决定了融合后的权重**

RRF 公式：

```
RRF_score(d) = Σ_{r in R} 1 / (k + rank_r(d))

其中：
- R: 所有检索路线的集合
- rank_r(d): 文档 d 在路线 r 中的排名（从 1 开始）
- k: 平滑参数（通常 60）
```

**RRF 的优势：**
1. **分数无关**：各路分数尺度不同也没关系（BM25 的几十 vs 余弦相似度的 0~1）
2. **鲁棒性**：单路排错的影响有限
3. **无需训练**：零参数（除了 k）
4. **可解释**：每篇文档的得分可以追溯到各路的排名

### 7.6.2 RRF 的纯 Python 实现

```python
# ch07_rrf.py
"""
RRF (Reciprocal Rank Fusion) 算法实现

支持：
- 标准 RRF 融合
- 加权 RRF（各路有权重）
- RRF 变体（不同 k 值策略）
- 结果去重
"""

from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import math


class RRFusion:
    """
    倒数排名融合（Reciprocal Rank Fusion）

    Parameters
    ----------
    k : int
        平滑参数，默认 60。
        值越小，排名靠前的文档权重越大；
        值越大，排名靠后的文档也有机会被选上。

        经验值：k=60 在多个 benchmark 上表现稳定。
        如果需要强调 Top-1 的重要性，可以降低到 10~30；
        如果希望更多多样性，可以提高到 60~100。
    """

    def __init__(self, k: int = 60):
        self.k = k

    def fuse(
        self,
        results_list: List[List[Tuple[str, float]]],
        weights: Optional[List[float]] = None,
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        融合多路检索结果

        Parameters
        ----------
        results_list : List[List[Tuple[str, float]]]
            多路检索结果，每路是 [(doc_id, score), ...] 列表。
            score 在此算法中不被使用（只用排名），
            保留是为了与检索器接口兼容。
        weights : List[float], optional
            各路权重。默认均等。
            例如 [1.0, 2.0, 0.5] 表示第二路权重是第一路的 2 倍。
        top_k : int
            融合后返回的 Top-K 结果数

        Returns
        -------
        List[Tuple[str, float]]
            融合后的 (doc_id, rrf_score) 列表
        """
        if not results_list:
            return []

        n_routes = len(results_list)
        if weights is None:
            weights = [1.0] * n_routes
        else:
            assert len(weights) == n_routes, \
                "weights 长度必须等于 results_list 长度"

        # 归一化权重
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        # 聚合 RRF 分数
        rrf_scores: Dict[str, float] = defaultdict(float)

        for route_idx, results in enumerate(results_list):
            w = weights[route_idx]
            for rank, (doc_id, _) in enumerate(results):
                # rank 从 1 开始
                rrf_score = w / (self.k + rank + 1)
                rrf_scores[doc_id] += rrf_score

        # 按 RRF 分数降序排列
        sorted_results = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        return sorted_results[:top_k]

    def fuse_with_details(
        self,
        results_list: List[List[Tuple[str, float]]],
        route_names: Optional[List[str]] = None,
        weights: Optional[List[float]] = None,
        top_k: int = 10,
    ) -> Dict:
        """
        融合并返回详细信息（用于分析和调试）

        Returns
        -------
        Dict
            {
                "results": [(doc_id, score), ...],
                "details": {
                    doc_id: {
                        "rrf_score": float,
                        "per_route": {route_name: (rank, contribution)},
                    }
                }
            }
        """
        n_routes = len(results_list)
        if route_names is None:
            route_names = [f"route_{i}" for i in range(n_routes)]
        if weights is None:
            weights = [1.0] * n_routes

        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        rrf_scores: Dict[str, float] = defaultdict(float)
        details: Dict[str, Dict] = defaultdict(
            lambda: {"rrf_score": 0.0, "per_route": {}}
        )

        for route_idx, results in enumerate(results_list):
            w = weights[route_idx]
            name = route_names[route_idx]
            for rank, (doc_id, original_score) in enumerate(results):
                contribution = w / (self.k + rank + 1)
                rrf_scores[doc_id] += contribution
                details[doc_id]["rrf_score"] += contribution
                details[doc_id]["per_route"][name] = {
                    "rank": rank + 1,
                    "original_score": original_score,
                    "contribution": contribution,
                }

        sorted_results = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]

        return {
            "results": sorted_results,
            "details": {
                doc_id: details[doc_id]
                for doc_id, _ in sorted_results
            },
        }


class WeightedRRFusion(RRFusion):
    """
    加权 RRF 融合

    支持为每路分配不同权重。
    继承自 RRFusion，使用 fuse() 的 weights 参数。
    """

    def __init__(
        self,
        k: int = 60,
        default_weights: Optional[List[float]] = None,
    ):
        super().__init__(k)
        self.default_weights = default_weights

    def fuse(
        self,
        results_list: List[List[Tuple[str, float]]],
        weights: Optional[List[float]] = None,
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        weights = weights or self.default_weights
        return super().fuse(results_list, weights, top_k)


class AdaptiveRRFusion(RRFusion):
    """
    自适应 RRF 融合

    根据各路结果集的多样性动态调整权重。
    多样性高的路线获得更高权重。

    多样性度量：各路结果集合之间的 Jaccard 距离。
    """

    def __init__(self, k: int = 60, diversity_weight: float = 0.3):
        super().__init__(k)
        self.diversity_weight = diversity_weight

    def _compute_diversity_weights(
        self,
        results_list: List[List[Tuple[str, float]]],
    ) -> List[float]:
        """根据每路结果的独特性计算权重"""
        n = len(results_list)
        if n <= 1:
            return [1.0] * n

        # 收集每路的文档 ID 集合
        doc_sets = [
            {doc_id for doc_id, _ in results}
            for results in results_list
        ]

        # 计算每路的平均 Jaccard 距离（与其它路的差异度）
        diversity_scores = []
        for i in range(n):
            total_jaccard = 0.0
            for j in range(n):
                if i == j:
                    continue
                intersection = len(doc_sets[i] & doc_sets[j])
                union = len(doc_sets[i] | doc_sets[j])
                jaccard = intersection / max(union, 1)
                total_jaccard += jaccard
            avg_jaccard = total_jaccard / max(n - 1, 1)
            # 多样性 = 1 - 平均 Jaccard 系数
            diversity = 1 - avg_jaccard
            diversity_scores.append(diversity)

        # 归一化
        total = sum(diversity_scores)
        if total == 0:
            return [1.0 / n] * n

        return [d / total for d in diversity_scores]

    def fuse(
        self,
        results_list: List[List[Tuple[str, float]]],
        weights: Optional[List[float]] = None,
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        if weights is None:
            weights = self._compute_diversity_weights(results_list)

        # 融合基础权重和多样性权重
        diversity_weights = self._compute_diversity_weights(results_list)
        combined_weights = [
            (1 - self.diversity_weight) * w
            + self.diversity_weight * dw
            for w, dw in zip(weights, diversity_weights)
        ]

        return super().fuse(results_list, combined_weights, top_k)


# ============================================================
# 使用示例
# ============================================================
if __name__ == "__main__":
    # 模拟三路检索结果
    dense_results = [
        ("doc_3", 0.92), ("doc_1", 0.88), ("doc_5", 0.85),
        ("doc_2", 0.82), ("doc_4", 0.79),
    ]
    bm25_results = [
        ("doc_1", 12.5), ("doc_2", 10.2), ("doc_4", 9.8),
        ("doc_3", 8.5), ("doc_6", 7.2),
    ]
    kg_results = [
        ("doc_3", 0.95), ("doc_7", 0.90), ("doc_1", 0.85),
        ("doc_5", 0.80),
    ]

    results_list = [dense_results, bm25_results, kg_results]
    route_names = ["Dense", "BM25", "KG"]

    rrf = RRFusion(k=60)

    print("=== 标准 RRF 融合 ===")
    results = rrf.fuse(results_list, top_k=5)
    for doc_id, score in results:
        print(f"  {doc_id}: RRF_score={score:.4f}")

    print("\n=== 带权重的 RRF 融合（BM25 权重加倍）===")
    results = rrf.fuse(results_list, weights=[1.0, 2.0, 1.0], top_k=5)
    for doc_id, score in results:
        print(f"  {doc_id}: RRF_score={score:.4f}")

    print("\n=== RRF 详细分析 ===")
    detail_result = rrf.fuse_with_details(
        results_list,
        route_names=route_names,
        top_k=5,
    )
    for doc_id, score in detail_result["results"]:
        print(f"\n  {doc_id} (total: {score:.4f}):")
        for route_name, info in detail_result["details"][doc_id]["per_route"].items():
            print(
                f"    {route_name}: rank={info['rank']}, "
                f"contrib={info['contribution']:.4f}"
            )

    print("\n=== 自适应 RRF 融合 ===")
    adaptive_rrf = AdaptiveRRFusion(k=60, diversity_weight=0.3)
    results = adaptive_rrf.fuse(results_list, top_k=5)
    for doc_id, score in results:
        print(f"  {doc_id}: RRF_score={score:.4f}")
```

### 7.6.3 RRF 参数 k 的影响

| k 值 | 排名 1 的贡献 | 排名 10 的贡献 | 排名 100 的贡献 | 特性 |
|---|---|---|---|---|
| 1 | 0.500 | 0.091 | 0.010 | 极度强调 Top-1 |
| 10 | 0.091 | 0.050 | 0.009 | 平衡 |
| 60 | 0.016 | 0.014 | 0.006 | 推荐值，稳健 |
| 100 | 0.010 | 0.009 | 0.005 | 各排名差异小 |

### 7.6.4 RRF vs 其它融合方法

| 方法 | 公式 | 优点 | 缺点 |
|---|---|---|---|
| **RRF** | Σ 1/(k+rank) | 分数无关，鲁棒 | 忽略分数信息 |
| **Score Normalization + Average** | Σ score_i / max_score | 利用分数信息 | 对异常值敏感 |
| **Min-Max Normalization + Weighted** | Σ w_i * (s_i - min)/(max-min) | 保留分布 | 需要全局统计 |
| **Learning to Rank** | 学习排序函数 | 理论上最优 | 需要标注数据 |

---

## 7.7 Cross-Encoder 重排序

### 7.7.1 为什么需要重排序

向量检索和 BM25 都属于 **Bi-Encoder** 范式——query 和 doc 各自独立编码，然后计算相似度。这种方式高效（可以预计算 doc embedding），但**无法捕捉 query 和 doc 之间的细粒度交互**。

**Cross-Encoder** 将 query 和 doc 拼接后一起输入 Transformer，能够建模两者之间的深度交互。

```
Bi-Encoder:    query → Encoder → q_vec   doc → Encoder → d_vec   → cos(q_vec, d_vec)
Cross-Encoder: [CLS] query [SEP] doc [SEP] → Encoder → [CLS] → classifier → relevance_score
```

**为什么只在重排序阶段使用 Cross-Encoder？**

Cross-Encoder 需要为每个 (query, doc) 对做一次前向传播，计算量是 O(N) 而不是 O(1)。因此实践中将其作为**重排序（Reranker）**阶段，只对多路召回的前 N 个候选（如 100 个）做重新排序。

### 7.7.2 使用 Sentence-Transformers 的 Cross-Encoder

```python
# ch07_cross_encoder.py
"""
Cross-Encoder 重排序实现

支持：
- 基于 Sentence-Transformers 的 Cross-Encoder
- 批量重排序
- 分数归一化
- 与多路召回管线集成
"""

from typing import List, Tuple, Optional
import numpy as np


class CrossEncoderReranker:
    """
    Cross-Encoder 重排序器

    将 query 和 doc pair 输入 Cross-Encoder 模型，
    输出相关性分数（通常 0~1）。

    Parameters
    ----------
    model_name : str
        Cross-Encoder 模型名称

        推荐模型：
        - "cross-encoder/ms-marco-MiniLM-L-6-v2": 英文通用
        - "BAAI/bge-reranker-v2-m3": 多语言（含中文）
        - "BAAI/bge-reranker-large": 中文专用
        - "Cohere/rerank-english-v3.0": Cohere 云端 API

    device : str
        计算设备
    batch_size : int
        批处理大小
    max_length : int
        最大输入长度（token）
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str = "cpu",
        batch_size: int = 32,
        max_length: int = 512,
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self._model = None

    def _load_model(self):
        """懒加载 Cross-Encoder 模型"""
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(
                self.model_name,
                device=self.device,
                max_length=self.max_length,
            )
            print(f"[CrossEncoder] 模型 {self.model_name} 加载完成")
        except ImportError:
            raise ImportError(
                "请安装 sentence-transformers: pip install sentence-transformers"
            )

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[str, str]],
        top_k: Optional[int] = None,
        return_scores: bool = True,
    ) -> List[Tuple[str, float]]:
        """
        重排序

        Parameters
        ----------
        query : str
            用户查询
        candidates : List[Tuple[str, str]]
            候选列表 [(doc_id, doc_text), ...]
        top_k : int, optional
            返回结果数，默认返回全部
        return_scores : bool
            是否返回分数

        Returns
        -------
        List[Tuple[str, float]]
            重排序后的 (doc_id, score) 列表
        """
        self._load_model()

        if not candidates:
            return []

        # 准备 (query, doc) 对
        pairs = [(query, doc_text) for _, doc_text in candidates]

        # 批量预测
        scores = self._model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        # 组合结果
        results = [
            (candidates[i][0], float(scores[i]))
            for i in range(len(candidates))
        ]

        # 按分数降序排列
        results.sort(key=lambda x: x[1], reverse=True)

        if top_k is not None:
            results = results[:top_k]

        return results

    def rerank_with_normalized_scores(
        self,
        query: str,
        candidates: List[Tuple[str, str]],
        top_k: Optional[int] = None,
        normalize_method: str = "minmax",
    ) -> List[Tuple[str, float]]:
        """
        重排序并归一化分数

        Parameters
        ----------
        normalize_method : str
            "minmax": 归一化到 [0, 1]
            "softmax": 转换为概率分布
            "zscore": 标准化（均值为 0，方差为 1）
        """
        results = self.rerank(query, candidates, return_scores=True)

        if not results:
            return results

        scores = np.array([s for _, s in results])

        if normalize_method == "minmax":
            min_s, max_s = scores.min(), scores.max()
            if max_s > min_s:
                normalized = (scores - min_s) / (max_s - min_s)
            else:
                normalized = np.ones_like(scores)

        elif normalize_method == "softmax":
            exp_s = np.exp(scores - scores.max())  # 数值稳定
            normalized = exp_s / exp_s.sum()

        elif normalize_method == "zscore":
            mean, std = scores.mean(), scores.std()
            if std > 0:
                normalized = (scores - mean) / std
            else:
                normalized = np.zeros_like(scores)

        else:
            raise ValueError(f"不支持的归一化方法: {normalize_method}")

        return [
            (doc_id, float(norm_score))
            for (doc_id, _), norm_score in zip(results, normalized)
        ]


# ============================================================
# 使用示例
# ============================================================
if __name__ == "__main__":
    reranker = CrossEncoderReranker(
        model_name="BAAI/bge-reranker-v2-m3",
        device="cpu",
    )

    query = "什么是自回归语言模型？"
    candidates = [
        ("doc_1", "Transformer 架构由编码器和解码器组成"),
        ("doc_2", "BERT 使用双向注意力机制进行预训练"),
        ("doc_3", "GPT 系列模型采用自回归生成方式，每次预测下一个 token"),
        ("doc_4", "注意力机制允许模型关注输入序列的不同位置"),
        ("doc_5", "残差连接解决了深层网络的梯度消失问题"),
    ]

    # 重排序
    results = reranker.rerank(query, candidates, top_k=3)

    print(f"查询: {query}")
    print("Cross-Encoder 重排序结果:")
    for doc_id, score in results:
        print(f"  {doc_id}: score={score:.4f}")

    # 归一化分数
    print("\n归一化分数 (minmax):")
    norm_results = reranker.rerank_with_normalized_scores(
        query, candidates, normalize_method="minmax"
    )
    for doc_id, score in norm_results:
        print(f"  {doc_id}: normalized_score={score:.4f}")
```

### 7.7.3 使用 Cohere Rerank API

```python
# ch07_cohere_reranker.py
"""
Cohere Rerank API 重排序实现

适合不想本地部署模型的场景。
Cohere 的 rerank 模型专门针对重排序优化，支持多语言。
"""

from typing import List, Tuple, Optional


class CohereReranker:
    """
    Cohere Rerank API 重排序器

    Parameters
    ----------
    model : str
        Cohere rerank 模型
        - "rerank-english-v3.0": 英文
        - "rerank-multilingual-v3.0": 多语言
    api_key : str, optional
        Cohere API Key，默认从环境变量 COHERE_API_KEY 读取
    max_tokens : int
        每篇文档的最大 token 数
    """

    def __init__(
        self,
        model: str = "rerank-multilingual-v3.0",
        api_key: Optional[str] = None,
        max_tokens: int = 512,
    ):
        self.model = model
        self.api_key = api_key
        self.max_tokens = max_tokens
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            import cohere
            self._client = cohere.Client(api_key=self.api_key)
        except ImportError:
            raise ImportError("请安装 cohere: pip install cohere")
        return self._client

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 10,
        doc_ids: Optional[List[str]] = None,
    ) -> List[Tuple[str, float]]:
        """
        使用 Cohere API 重排序

        Parameters
        ----------
        query : str
            查询
        documents : List[str]
            待重排序的文档列表
        top_k : int
            返回结果数
        doc_ids : List[str], optional
            文档 ID（默认使用索引作为 ID）

        Returns
        -------
        List[Tuple[str, float]]
            重排序后的 (doc_id, relevance_score) 列表
        """
        client = self._get_client()

        if doc_ids is None:
            doc_ids = [str(i) for i in range(len(documents))]

        response = client.rerank(
            model=self.model,
            query=query,
            documents=documents,
            top_n=top_k,
            max_tokens_per_doc=self.max_tokens,
        )

        results = []
        for result in response.results:
            doc_idx = result.index
            doc_id = doc_ids[doc_idx]
            score = result.relevance_score
            results.append((doc_id, float(score)))

        return results


# ============================================================
# 使用示例
# ============================================================
if __name__ == "__main__":
    # 注意：运行前请设置 COHERE_API_KEY 环境变量
    reranker = CohereReranker(
        model="rerank-multilingual-v3.0",
    )

    query = "自回归语言模型的训练方式"
    documents = [
        "GPT 系列模型采用自回归生成方式，每次预测下一个 token",
        "BERT 使用双向注意力机制进行预训练",
        "Transformer 架构由编码器和解码器组成",
        "自回归模型通过最大化序列概率进行训练",
    ]

    results = reranker.rerank(query, documents, top_k=3)
    print(f"查询: {query}")
    for doc_id, score in results:
        idx = int(doc_id)
        print(f"  [{idx}] (score={score:.4f}) {documents[idx][:40]}...")
```

### 7.7.4 Bi-Encoder vs Cross-Encoder 对比

| 维度 | Bi-Encoder | Cross-Encoder |
|---|---|---|
| 编码方式 | query 和 doc 分别编码 | query 和 doc 拼接后编码 |
| 交互粒度 | 只有最后向量交互 | 每层 Transformer 都在交互 |
| 精度 | 较低（信息瓶颈在最终向量） | 较高（完整交互） |
| 速度 | 快（doc embedding 可预计算） | 慢（每对需重新计算） |
| 可扩展性 | 支持百万级文档索引 | 仅适用于重排序阶段 |
| 典型场景 | 第一阶段检索 | 第二阶段重排序 |

### 7.7.5 完整的多路召回 + 重排序管线

```python
# ch07_full_pipeline.py
"""
完整的多路召回 + 重排序管线

将前几节的内容整合为一个端到端的检索系统。

流程：
1. 多路并行检索（Dense + BM25 + Structured + KG）
2. RRF 融合
3. Cross-Encoder 重排序
4. 去重与合并
5. 最终结果输出
"""

from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import OrderedDict
import time


@dataclass
class RetrievedChunk:
    """检索结果块"""
    chunk_id: str
    text: str
    source: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    scores: Dict[str, float] = field(default_factory=dict)
    rrf_score: float = 0.0
    rerank_score: float = 0.0


class MultiRecallPipeline:
    """
    多路召回 + 重排序完整管线

    Parameters
    ----------
    dense_retriever : optional
        密集检索器
    bm25_retriever : optional
        BM25 检索器
    structured_retriever : optional
        结构化检索器
    kg_retriever : optional
        KG 检索器
    reranker : optional
        Cross-Encoder 重排序器
    fusion : RRFusion
        RRF 融合器
    """

    def __init__(
        self,
        dense_retriever=None,
        bm25_retriever=None,
        structured_retriever=None,
        kg_retriever=None,
        reranker=None,
        fusion=None,
    ):
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.structured_retriever = structured_retriever
        self.kg_retriever = kg_retriever
        self.reranker = reranker

        from ch07_rrf import RRFusion
        self.fusion = fusion or RRFusion(k=60)

    def retrieve(
        self,
        query: str,
        query_vector: Optional[np.ndarray] = None,
        filters: Optional[dict] = None,
        top_k: int = 10,
        candidate_k: int = 50,
        use_reranker: bool = True,
        verbose: bool = True,
    ) -> Tuple[List[RetrievedChunk], Dict]:
        """
        执行多路召回

        Parameters
        ----------
        query : str
            用户查询
        query_vector : np.ndarray, optional
            预计算的查询向量（避免重复编码）
        filters : dict, optional
            结构化过滤条件
        top_k : int
            最终返回结果数
        candidate_k : int
            每路检索的候选数
        use_reranker : bool
            是否使用 Cross-Encoder 重排序
        verbose : bool
            是否打印详细信息

        Returns
        -------
        Tuple[List[RetrievedChunk], Dict]
            (最终结果, 统计信息)
        """
        stats = {
            "routes": {},
            "total_candidates": 0,
            "unique_candidates": 0,
            "timings": {},
        }

        # ========== 第 1 阶段：多路并行检索 ==========
        all_results = []  # List of List[(doc_id, score)]
        route_names = []
        doc_pool: Dict[str, RetrievedChunk] = {}

        # --- Dense 检索 ---
        if self.dense_retriever:
            t0 = time.time()
            dense_results = self.dense_retriever.retrieve(
                query, top_k=candidate_k
            )
            t1 = time.time()
            all_results.append(dense_results)
            route_names.append("dense")
            stats["routes"]["dense"] = {
                "candidates": len(dense_results),
                "time_ms": (t1 - t0) * 1000,
            }
            for doc_id, score in dense_results:
                if doc_id not in doc_pool:
                    doc_pool[doc_id] = RetrievedChunk(
                        chunk_id=doc_id,
                        text=self.dense_retriever.documents[int(doc_id)]
                        if isinstance(doc_id, str) else "",
                    )
                doc_pool[doc_id].scores["dense"] = score

        # --- BM25 检索 ---
        if self.bm25_retriever:
            t0 = time.time()
            bm25_results = self.bm25_retriever.retrieve(
                query, top_k=candidate_k
            )
            t1 = time.time()
            # 统一格式为 [(doc_id, score)]
            formatted = []
            for item in bm25_results:
                if isinstance(item, tuple) and len(item) == 2:
                    formatted.append(item)
                else:
                    formatted.append((str(item[0]), float(item[1])))
            all_results.append(formatted)
            route_names.append("bm25")
            stats["routes"]["bm25"] = {
                "candidates": len(formatted),
                "time_ms": (t1 - t0) * 1000,
            }
            for doc_id, score in formatted:
                if doc_id not in doc_pool:
                    doc_pool[doc_id] = RetrievedChunk(chunk_id=doc_id)
                doc_pool[doc_id].scores["bm25"] = score

        # --- 结构化检索 ---
        if self.structured_retriever and filters:
            t0 = time.time()
            struct_results = self.structured_retriever.retrieve_by_filters(
                filters, top_k=candidate_k
            )
            t1 = time.time()
            formatted = [(doc.id, 1.0) for doc in struct_results]
            all_results.append(formatted)
            route_names.append("structured")
            stats["routes"]["structured"] = {
                "candidates": len(formatted),
                "time_ms": (t1 - t0) * 1000,
            }
            for doc in struct_results:
                if doc.id not in doc_pool:
                    doc_pool[doc.id] = RetrievedChunk(
                        chunk_id=doc.id, text=doc.content
                    )
                doc_pool[doc.id].scores["structured"] = 1.0

        # --- KG 检索 ---
        if self.kg_retriever:
            t0 = time.time()
            kg_results = self.kg_retriever.retrieve_by_entity(
                query, top_k=candidate_k
            )
            t1 = time.time()
            formatted = []
            for r in kg_results:
                key = f"kg_{r['head']}_{r['relation']}_{r['tail']}"
                formatted.append((key, 1.0))
            all_results.append(formatted)
            route_names.append("kg")
            stats["routes"]["kg"] = {
                "candidates": len(formatted),
                "time_ms": (t1 - t0) * 1000,
            }

        stats["total_candidates"] = sum(
            len(r) for r in all_results
        )

        # ========== 第 2 阶段：RRF 融合 ==========
        t0 = time.time()
        fused = self.fusion.fuse(
            all_results,
            route_names=route_names,
            top_k=candidate_k,  # 保留更多候选给 reranker
        )
        t1 = time.time()
        stats["timings"]["fusion_ms"] = (t1 - t0) * 1000
        stats["unique_candidates"] = len(fused)

        # 更新 RRF 分数
        for doc_id, rrf_score in fused:
            if doc_id in doc_pool:
                doc_pool[doc_id].rrf_score = rrf_score

        if verbose:
            print(f"\n多路召回统计:")
            for name, route_stats in stats["routes"].items():
                print(
                    f"  {name}: {route_stats['candidates']} 篇候选, "
                    f"{route_stats['time_ms']:.1f}ms"
                )
            print(
                f"  RRF 融合: {stats['unique_candidates']} 篇唯一候选, "
                f"{stats['timings']['fusion_ms']:.1f}ms"
            )

        # ========== 第 3 阶段：Cross-Encoder 重排序 ==========
        if use_reranker and self.reranker and fused:
            t0 = time.time()

            # 构建待重排序的候选
            rerank_candidates = []
            for doc_id, _ in fused:
                chunk = doc_pool.get(doc_id)
                if chunk and chunk.text:
                    rerank_candidates.append((doc_id, chunk.text))
                else:
                    # 如果没有文本，尝试从文档源获取
                    rerank_candidates.append((doc_id, f"Document {doc_id}"))

            # 重排序
            reranked = self.reranker.rerank(query, rerank_candidates)

            # 更新重排序分数
            for doc_id, rerank_score in reranked:
                if doc_id in doc_pool:
                    doc_pool[doc_id].rerank_score = rerank_score

            # 使用重排序结果
            final_results = reranked[:top_k]

            t1 = time.time()
            stats["timings"]["rerank_ms"] = (t1 - t0) * 1000

            if verbose:
                print(
                    f"  Cross-Encoder 重排序: "
                    f"{stats['timings']['rerank_ms']:.1f}ms"
                )
        else:
            # 不使用重排序，直接使用 RRF 结果
            final_results = fused[:top_k]

        # ========== 第 4 阶段：组装结果 ==========
        final_chunks = []
        for doc_id, score in final_results:
            chunk = doc_pool.get(doc_id)
            if chunk:
                chunk.rerank_score = score
                final_chunks.append(chunk)
            else:
                final_chunks.append(RetrievedChunk(
                    chunk_id=doc_id,
                    text=f"Document {doc_id}",
                    rerank_score=score,
                ))

        stats["total_time_ms"] = sum(
            v if isinstance(v, float) else v.get("time_ms", 0)
            for v in stats["timings"].values()
        )
        for route_stats in stats["routes"].values():
            stats["total_time_ms"] += route_stats.get("time_ms", 0)

        return final_chunks, stats
```

---

## 7.8 去重与合并

### 7.8.1 为什么需要去重

多路召回不可避免地会产生重复结果：

- 同一段文本被 Dense 和 BM25 同时召回
- 同一内容的多个版本（如不同格式的引用）
- 语义上高度相似的片段（如摘要和原文）

### 7.8.2 精确去重 vs 语义去重

```python
# ch07_dedup.py
"""
结果去重与合并

支持两种去重策略：
1. 精确去重：基于文本哈希
2. 语义去重：基于向量相似度
"""

from typing import List, Dict, Set, Tuple, Optional
import hashlib
import numpy as np


class Deduplicator:
    """
    检索结果去重器

    Parameters
    ----------
    strategy : str
        去重策略
        - "exact": 精确去重（基于文本哈希）
        - "semantic": 语义去重（基于 embedding 相似度）
    similarity_threshold : float
        语义去重的相似度阈值（默认 0.85）
    """

    def __init__(
        self,
        strategy: str = "exact",
        similarity_threshold: float = 0.85,
    ):
        self.strategy = strategy
        self.similarity_threshold = similarity_threshold

    def _text_hash(self, text: str) -> str:
        """计算文本的 MD5 哈希"""
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def dedup_exact(
        self,
        results: List[Tuple[str, str, float]],
    ) -> List[Tuple[str, str, float]]:
        """
        精确去重

        基于文本的 MD5 哈希，完全相同的文本只保留一份。
        保留第一次出现的版本。

        Parameters
        ----------
        results : List[Tuple[str, str, float]]
            (doc_id, text, score) 列表

        Returns
        -------
        List[Tuple[str, str, float]]
            去重后的列表
        """
        seen_hashes: Set[str] = set()
        deduped = []

        for doc_id, text, score in results:
            h = self._text_hash(text)
            if h not in seen_hashes:
                seen_hashes.add(h)
                deduped.append((doc_id, text, score))

        return deduped

    def dedup_semantic(
        self,
        results: List[Tuple[str, str, float]],
        embeddings: Optional[np.ndarray] = None,
    ) -> List[Tuple[str, str, float]]:
        """
        语义去重

        基于 embedding 余弦相似度，语义高度相似的文本只保留一份。
        保留分数最高的版本。

        Parameters
        ----------
        results : List[Tuple[str, str, float]]
            (doc_id, text, score) 列表
        embeddings : np.ndarray, optional
            预计算的文本 embeddings，形状 (N, dim)

        Returns
        -------
        List[Tuple[str, str, float]]
            去重后的列表
        """
        if not results:
            return []

        if embeddings is None:
            # 没有预计算 embedding 则回退到精确去重
            return self.dedup_exact(results)

        # 按分数降序排列（优先保留高分）
        sorted_results = sorted(
            results, key=lambda x: x[2], reverse=True
        )

        kept_indices = []
        for i in range(len(sorted_results)):
            is_duplicate = False
            for j in kept_indices:
                # 计算余弦相似度
                sim = float(np.dot(embeddings[i], embeddings[j]))
                if sim > self.similarity_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                kept_indices.append(i)

        return [sorted_results[i] for i in kept_indices]

    def dedup(
        self,
        results: List[Tuple[str, str, float]],
        embeddings: Optional[np.ndarray] = None,
    ) -> List[Tuple[str, str, float]]:
        """统一去重接口"""
        if self.strategy == "exact":
            return self.dedup_exact(results)
        elif self.strategy == "semantic":
            return self.dedup_semantic(results, embeddings)
        else:
            raise ValueError(f"不支持的策略: {self.strategy}")

    def merge_and_dedup(
        self,
        result_lists: List[List[Tuple[str, str, float]]],
        strategy: str = "max_score",
    ) -> List[Tuple[str, str, float]]:
        """
        合并多路结果并去重

        Parameters
        ----------
        result_lists : List[List[Tuple[str, str, float]]]
            多路结果
        strategy : str
            "max_score": 重复项保留最高分
            "first": 保留第一次出现的
            "average": 取平均分

        Returns
        -------
        List[Tuple[str, str, float]]
            合并去重后的结果
        """
        merged: Dict[str, Tuple[str, List[float]]] = {}

        for results in result_lists:
            for doc_id, text, score in results:
                if doc_id not in merged:
                    merged[doc_id] = (text, [score])
                else:
                    merged[doc_id][1].append(score)

        final_results = []
        for doc_id, (text, scores) in merged.items():
            if strategy == "max_score":
                final_score = max(scores)
            elif strategy == "first":
                final_score = scores[0]
            elif strategy == "average":
                final_score = sum(scores) / len(scores)
            else:
                raise ValueError(f"不支持的去重策略: {strategy}")

            final_results.append((doc_id, text, final_score))

        # 按分数降序排列
        final_results.sort(key=lambda x: x[2], reverse=True)

        # 去重
        return self.dedup(final_results)


# ============================================================
# 使用示例
# ============================================================
if __name__ == "__main__":
    dedup = Deduplicator(strategy="exact")

    # 模拟有重复的结果
    results = [
        ("doc_1", "Transformer 架构由编码器和解码器组成", 0.95),
        ("doc_2", "BERT 使用双向注意力机制进行预训练", 0.88),
        ("doc_3", "Transformer 架构由编码器和解码器组成", 0.85),  # 重复
        ("doc_4", "GPT 系列模型采用自回归生成方式", 0.82),
        ("doc_1", "Transformer 架构由编码器和解码器组成", 0.80),  # 重复
    ]

    print("去重前:", len(results))
    deduped = dedup.merge_and_dedup([results], strategy="max_score")
    print("去重后:", len(deduped))
    for doc_id, text, score in deduped:
        print(f"  {doc_id}: (score={score:.4f}) {text}")
```

---

## 7.9 评估指标

### 7.9.1 核心指标定义

| 指标 | 公式 | 含义 |
|---|---|---|
| **Recall@K** | 检索到的相关文档数 / 总相关文档数 | 召回率，越大越好 |
| **Precision@K** | 检索到的相关文档数 / K | 精确率，越大越好 |
| **MRR** (Mean Reciprocal Rank) | 1/N * Σ 1/rank_i | 第一个相关结果的排名倒数均值 |
| **NDCG@K** (Normalized Discounted Cumulative Gain) | DCG@K / IDCG@K | 排序质量，考虑多级相关性 |

### 7.9.2 评估指标实现

```python
# ch07_evaluation.py
"""
检索评估指标实现

支持：
- Recall@K
- Precision@K
- F1@K
- MRR (Mean Reciprocal Rank)
- MAP (Mean Average Precision)
- NDCG@K (Normalized Discounted Cumulative Gain)
- Hit Rate@K
"""

from typing import List, Set, Dict, Optional, Union
import numpy as np
import math


class RetrievalMetrics:
    """
    检索评估指标计算器

    Parameters
    ----------
    relevance_level : int
        多级相关性判定的阈值。
        如果 relevance >= 该值，视为"相关"。
    """

    def __init__(self, relevance_level: int = 1):
        self.relevance_level = relevance_level

    # ========== 二元相关性指标 ==========

    def recall_at_k(
        self,
        retrieved: List[str],
        relevant: Set[str],
        k: int = 10,
    ) -> float:
        """
        Recall@K

        在前 K 个检索结果中，相关文档的比例。
        关注"有没有漏掉"。

        Parameters
        ----------
        retrieved : List[str]
            检索结果的文档 ID 列表
        relevant : Set[str]
            所有相关文档的 ID 集合
        k : int
            Top-K 截断

        Returns
        -------
        float
            Recall@K 值 (0 ~ 1)
        """
        if not relevant:
            return 0.0

        retrieved_at_k = set(retrieved[:k])
        hits = len(retrieved_at_k & relevant)

        return hits / len(relevant)

    def precision_at_k(
        self,
        retrieved: List[str],
        relevant: Set[str],
        k: int = 10,
    ) -> float:
        """
        Precision@K

        前 K 个检索结果中有多少是相关的。
        关注"有没有找错"。

        Parameters
        ----------
        retrieved : List[str]
            检索结果的文档 ID 列表
        relevant : Set[str]
            所有相关文档的 ID 集合
        k : int
            Top-K 截断

        Returns
        -------
        float
            Precision@K 值 (0 ~ 1)
        """
        if k == 0:
            return 0.0

        retrieved_at_k = retrieved[:k]
        hits = sum(1 for doc_id in retrieved_at_k if doc_id in relevant)

        return hits / k

    def f1_at_k(
        self,
        retrieved: List[str],
        relevant: Set[str],
        k: int = 10,
    ) -> float:
        """
        F1@K

        Precision@K 和 Recall@K 的调和平均。
        """
        p = self.precision_at_k(retrieved, relevant, k)
        r = self.recall_at_k(retrieved, relevant, k)

        if p + r == 0:
            return 0.0

        return 2 * p * r / (p + r)

    def hit_rate_at_k(
        self,
        retrieved: List[str],
        relevant: Set[str],
        k: int = 10,
    ) -> float:
        """
        Hit Rate@K

        前 K 个结果中是否包含至少一个相关文档。
        常用于评估"用户能否在第一页看到相关内容"。

        Returns
        -------
        float
            0.0 或 1.0
        """
        retrieved_at_k = set(retrieved[:k])
        return 1.0 if retrieved_at_k & relevant else 0.0

    def average_precision(
        self,
        retrieved: List[str],
        relevant: Set[str],
    ) -> float:
        """
        Average Precision (AP)

        在排序列表的每个相关位置计算 Precision，然后取平均。

        AP = Σ_{k=1}^{n} P@k * rel_k / total_relevant

        其中 rel_k 表示第 k 个结果是否相关 (0/1)。
        """
        if not relevant:
            return 0.0

        total_relevant = len(relevant)
        hits = 0
        sum_precision = 0.0

        for k, doc_id in enumerate(retrieved, start=1):
            if doc_id in relevant:
                hits += 1
                sum_precision += hits / k

        return sum_precision / total_relevant

    def mean_reciprocal_rank(
        self,
        queries_results: List[List[str]],
        queries_relevant: List[Set[str]],
    ) -> float:
        """
        MRR (Mean Reciprocal Rank)

        第一个相关结果的排名倒数，对所有 query 取平均。

        MRR = (1/Q) * Σ_{q=1}^{Q} 1 / rank_q

        其中 rank_q 是第一个相关结果在排序中的位置。

        Parameters
        ----------
        queries_results : List[List[str]]
            每个 query 的检索结果列表
        queries_relevant : List[Set[str]]
            每个 query 的相关文档集合

        Returns
        -------
        float
            MRR 值 (0 ~ 1)
        """
        if not queries_results:
            return 0.0

        reciprocal_ranks = []

        for retrieved, relevant in zip(queries_results, queries_relevant):
            rr = 0.0
            for rank, doc_id in enumerate(retrieved, start=1):
                if doc_id in relevant:
                    rr = 1.0 / rank
                    break
            reciprocal_ranks.append(rr)

        return float(np.mean(reciprocal_ranks))

    # ========== 多级相关性指标 ==========

    def dcg_at_k(
        self,
        relevance_scores: List[float],
        k: int = 10,
    ) -> float:
        """
        DCG@K (Discounted Cumulative Gain)

        DCG@K = Σ_{i=1}^{K} (2^{rel_i} - 1) / log2(i + 1)

        其中 rel_i 是第 i 个结果的相关性分数（多级）。
        """
        relevance_scores = relevance_scores[:k]

        dcg = 0.0
        for i, rel in enumerate(relevance_scores, start=1):
            if i == 1:
                dcg += rel  # 第一项不折扣
            else:
                dcg += rel / math.log2(i + 1)

        return dcg

    def ndcg_at_k(
        self,
        retrieved: List[str],
        relevance_dict: Dict[str, float],
        k: int = 10,
    ) -> float:
        """
        NDCG@K (Normalized DCG)

        NDCG@K = DCG@K / IDCG@K

        其中 IDCG@K 是理想排序下的 DCG@K。

        Parameters
        ----------
        retrieved : List[str]
            检索结果列表
        relevance_dict : Dict[str, float]
            每个文档的多级相关性分数
        k : int
            Top-K 截断

        Returns
        -------
        float
            NDCG@K 值 (0 ~ 1)
        """
        # 实际 DCG
        relevance_scores = [
            relevance_dict.get(doc_id, 0.0)
            for doc_id in retrieved[:k]
        ]
        dcg = self.dcg_at_k(relevance_scores, k)

        # 理想 DCG (IDCG) — 按相关性分数降序排列
        ideal_scores = sorted(
            relevance_dict.values(), reverse=True
        )[:k]
        idcg = self.dcg_at_k(ideal_scores, k)

        if idcg == 0:
            return 0.0

        return dcg / idcg

    # ========== 批量评估 ==========

    def evaluate_all(
        self,
        queries_results: List[List[str]],
        queries_relevant: List[Set[str]],
        queries_relevance_dict: Optional[List[Dict[str, float]]] = None,
        k_list: List[int] = [1, 3, 5, 10, 20],
    ) -> Dict:
        """
        批量计算所有指标

        Parameters
        ----------
        queries_results : List[List[str]]
            每个 query 的检索结果
        queries_relevant : List[Set[str]]
            每个 query 的相关文档集合（二元）
        queries_relevance_dict : List[Dict[str, float]], optional
            每个 query 的多级相关性分数（用于 NDCG）
        k_list : List[int]
            要计算的 K 值列表

        Returns
        -------
        Dict
            {
                "recall@K": {k: mean_recall},
                "precision@K": {k: mean_precision},
                "mrr": float,
                "map": float,
                "ndcg@K": {k: mean_ndcg},
                "hit_rate@K": {k: mean_hit_rate},
                "per_query": [...]
            }
        """
        n_queries = len(queries_results)
        assert n_queries == len(queries_relevant), \
            "queries_results 和 queries_relevant 长度必须一致"

        results = {
            "recall": {k: [] for k in k_list},
            "precision": {k: [] for k in k_list},
            "f1": {k: [] for k in k_list},
            "hit_rate": {k: [] for k in k_list},
            "ndcg": {k: [] for k in k_list},
            "ap": [],
            "per_query": [],
        }

        for q_idx in range(n_queries):
            retrieved = queries_results[q_idx]
            relevant = queries_relevant[q_idx]
            rel_dict = (
                queries_relevance_dict[q_idx]
                if queries_relevance_dict
                else {doc_id: 1.0 for doc_id in relevant}
            )

            query_metrics = {"query_idx": q_idx}

            for k in k_list:
                results["recall"][k].append(
                    self.recall_at_k(retrieved, relevant, k)
                )
                results["precision"][k].append(
                    self.precision_at_k(retrieved, relevant, k)
                )
                results["f1"][k].append(
                    self.f1_at_k(retrieved, relevant, k)
                )
                results["hit_rate"][k].append(
                    self.hit_rate_at_k(retrieved, relevant, k)
                )
                results["ndcg"][k].append(
                    self.ndcg_at_k(retrieved, rel_dict, k)
                )

                query_metrics[f"recall@{k}"] = results["recall"][k][-1]
                query_metrics[f"precision@{k}"] = results["precision"][k][-1]
                query_metrics[f"ndcg@{k}"] = results["ndcg"][k][-1]

            ap = self.average_precision(retrieved, relevant)
            results["ap"].append(ap)
            query_metrics["ap"] = ap

            results["per_query"].append(query_metrics)

        # 聚合为均值
        aggregated = {
            "num_queries": n_queries,
            "recall": {
                k: float(np.mean(v)) for k, v in results["recall"].items()
            },
            "precision": {
                k: float(np.mean(v)) for k, v in results["precision"].items()
            },
            "f1": {
                k: float(np.mean(v)) for k, v in results["f1"].items()
            },
            "hit_rate": {
                k: float(np.mean(v)) for k, v in results["hit_rate"].items()
            },
            "ndcg": {
                k: float(np.mean(v)) for k, v in results["ndcg"].items()
            },
            "map": float(np.mean(results["ap"])),
            "mrr": self.mean_reciprocal_rank(
                queries_results, queries_relevant
            ),
            "per_query": results["per_query"],
        }

        return aggregated

    def print_report(
        self,
        metrics: Dict,
        title: str = "检索评估报告",
    ):
        """打印评估报告"""
        print(f"\n{'=' * 50}")
        print(f"  {title}")
        print(f"{'=' * 50}")

        print(f"\n查询数量: {metrics['num_queries']}")
        print(f"MAP: {metrics['map']:.4f}")
        print(f"MRR: {metrics['mrr']:.4f}")

        print(f"\n{'K':>5} | {'Recall':>8} | {'Precision':>10} | "
              f"{'F1':>8} | {'NDCG':>8} | {'HitRate':>8}")
        print("-" * 60)

        for k in sorted(metrics["recall"].keys()):
            print(
                f"{k:>5} | {metrics['recall'][k]:>8.4f} | "
                f"{metrics['precision'][k]:>10.4f} | "
                f"{metrics['f1'][k]:>8.4f} | "
                f"{metrics['ndcg'][k]:>8.4f} | "
                f"{metrics['hit_rate'][k]:>8.4f}"
            )

        print(f"{'=' * 50}")


# ============================================================
# 使用示例
# ============================================================
if __name__ == "__main__":
    evaluator = RetrievalMetrics()

    # 模拟 3 个 query 的检索结果
    queries_results = [
        ["doc_3", "doc_1", "doc_5", "doc_2", "doc_4"],
        ["doc_2", "doc_4", "doc_1", "doc_3", "doc_5"],
        ["doc_1", "doc_3", "doc_2", "doc_5", "doc_4"],
    ]
    queries_relevant = [
        {"doc_1", "doc_3", "doc_6"},
        {"doc_2", "doc_4"},
        {"doc_1", "doc_5", "doc_7", "doc_8"},
    ]
    queries_relevance_dict = [
        {"doc_3": 2.0, "doc_1": 1.0, "doc_5": 0.5, "doc_2": 0.0, "doc_4": 0.0},
        {"doc_2": 2.0, "doc_4": 1.0, "doc_1": 0.0, "doc_3": 0.0, "doc_5": 0.0},
        {"doc_1": 2.0, "doc_3": 1.0, "doc_2": 0.5, "doc_5": 0.0, "doc_4": 0.0},
    ]

    metrics = evaluator.evaluate_all(
        queries_results,
        queries_relevant,
        queries_relevance_dict,
        k_list=[1, 3, 5],
    )

    evaluator.print_report(metrics)

    # 单指标示例
    print("\n=== 单指标示例 ===")
    query = "doc_3" in queries_results[0][:3]
    recall_3 = evaluator.recall_at_k(
        queries_results[0], queries_relevant[0], k=3
    )
    print(f"Query 0 Recall@3: {recall_3:.4f}")

    precision_3 = evaluator.precision_at_k(
        queries_results[0], queries_relevant[0], k=3
    )
    print(f"Query 0 Precision@3: {precision_3:.4f}")

    ndcg_5 = evaluator.ndcg_at_k(
        queries_results[0], queries_relevance_dict[0], k=5
    )
    print(f"Query 0 NDCG@5: {ndcg_5:.4f}")
```

### 7.9.3 不同 K 值下的指标解读

```
K=1:   用户第一眼看到的结果是否相关（最严格）
K=3:   移动端一屏能否看到相关内容
K=5:   桌面端首屏能否看到相关内容
K=10:  用户是否愿意翻到第二页
K=20:  全面召回能力
```

### 7.9.4 多路召回 vs 单路召回评估对比

```python
# ch07_comparison.py
"""
多路召回与单路召回的评估对比

展示 RRF 融合后各项指标的提升。
"""

import numpy as np
from ch07_evaluation import RetrievalMetrics


def simulate_evaluation():
    """模拟对比实验"""
    evaluator = RetrievalMetrics()

    # 模拟 10 个 query
    np.random.seed(42)
    n_queries = 10
    n_docs = 100

    # 模拟相关文档（每个 query 3~8 个相关文档）
    all_relevant = []
    for _ in range(n_queries):
        n_rel = np.random.randint(3, 9)
        relevant = set(
            np.random.choice(n_docs, n_rel, replace=False).astype(str)
        )
        all_relevant.append(relevant)

    # 模拟 Dense 检索结果
    dense_results = []
    for relevant in all_relevant:
        # Dense 在语义上较好，随机放一些相关文档在前面
        result = list(relevant) + [
            str(i) for i in range(n_docs)
            if str(i) not in relevant
        ]
        np.random.shuffle(result)
        # 让相关文档尽量靠前
        for i, doc_id in enumerate(result):
            if doc_id in relevant:
                result.remove(doc_id)
                result.insert(
                    max(0, i % 10), doc_id  # 放在前 10 位
                )
        dense_results.append(result[:20])

    # 模拟 BM25 检索结果（与 Dense 不同，更擅长精确匹配）
    bm25_results = []
    for relevant in all_relevant:
        result = list(relevant) + [
            str(i) for i in range(n_docs)
            if str(i) not in relevant
        ]
        np.random.shuffle(result)
        # BM25 的排序与 Dense 不同
        for i, doc_id in enumerate(result):
            if doc_id in relevant:
                result.remove(doc_id)
                result.insert(
                    max(0, 5 + i % 15), doc_id  # 位置与 Dense 不同
                )
        bm25_results.append(result[:20])

    # 模拟 RRF 融合结果
    from ch07_rrf import RRFusion
    rrf = RRFusion(k=60)
    rrf_results = []

    for q_idx in range(n_queries):
        # 构造 RRF 输入
        dense_list = [(doc_id, 1.0) for doc_id in dense_results[q_idx]]
        bm25_list = [(doc_id, 1.0) for doc_id in bm25_results[q_idx]]

        fused = rrf.fuse([dense_list, bm25_list], top_k=20)
        rrf_results.append([doc_id for doc_id, _ in fused])

    # 评估
    dense_metrics = evaluator.evaluate_all(
        dense_results, all_relevant, k_list=[1, 3, 5, 10]
    )
    bm25_metrics = evaluator.evaluate_all(
        bm25_results, all_relevant, k_list=[1, 3, 5, 10]
    )
    rrf_metrics = evaluator.evaluate_all(
        rrf_results, all_relevant, k_list=[1, 3, 5, 10]
    )

    print("=" * 65)
    print(f"{'指标':>15} | {'Dense':>8} | {'BM25':>8} | {'RRF':>8} | {'提升':>10}")
    print("=" * 65)

    for k in [1, 3, 5, 10]:
        d_recall = dense_metrics["recall"][k]
        b_recall = bm25_metrics["recall"][k]
        r_recall = rrf_metrics["recall"][k]
        improvement = (r_recall - max(d_recall, b_recall)) / max(d_recall, b_recall, 0.001) * 100
        print(
            f"{f'Recall@{k}':>15} | {d_recall:>8.4f} | "
            f"{b_recall:>8.4f} | {r_recall:>8.4f} | "
            f"{improvement:>+8.2f}%"
        )

    print("-" * 65)

    print(
        f"{'MRR':>15} | "
        f"{dense_metrics['mrr']:>8.4f} | "
        f"{bm25_metrics['mrr']:>8.4f} | "
        f"{rrf_metrics['mrr']:>8.4f} | "
        f"{(rrf_metrics['mrr'] - max(dense_metrics['mrr'], bm25_metrics['mrr'])) / max(dense_metrics['mrr'], bm25_metrics['mrr'], 0.001) * 100:>+8.2f}%"
    )

    print("=" * 65)


if __name__ == "__main__":
    simulate_evaluation()
```

---

## 7.10 完整案例：生产级多路召回系统

```python
# ch07_production_pipeline.py
"""
生产级多路召回系统

整合本章所有技术，形成一个可直接使用的检索系统。

特性：
- 四路召回：Dense + BM25 + Structured + KG
- RRF 融合
- Cross-Encoder 重排序
- 精确 + 语义去重
- 评估指标监控
- 详细日志
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import time
import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """搜索结果"""
    chunk_id: str
    text: str
    score: float = 0.0
    source_route: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class ProductionMultiRecall:
    """
    生产级多路召回系统

    使用示例：

    >>> pipeline = ProductionMultiRecall(config={...})
    >>> pipeline.build_index(documents)
    >>> results = pipeline.search("用户查询")
    >>> print(results)
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self._init_components()

    def _default_config(self) -> Dict:
        return {
            "dense": {
                "model": "BAAI/bge-small-zh-v1.5",
                "device": "cpu",
                "top_k": 100,
                "weight": 1.0,
            },
            "bm25": {
                "k1": 1.5,
                "b": 0.75,
                "top_k": 100,
                "weight": 1.0,
            },
            "reranker": {
                "model": "BAAI/bge-reranker-v2-m3",
                "device": "cpu",
                "top_k": 30,
            },
            "fusion": {
                "k": 60,
            },
            "dedup": {
                "strategy": "exact",
            },
        }

    def _init_components(self):
        """初始化各组件"""
        # Dense Retriever
        dense_config = self.config.get("dense", {})
        self.dense_retriever = DenseRetriever(
            model_name=dense_config.get("model", "BAAI/bge-small-zh-v1.5"),
            device=dense_config.get("device", "cpu"),
        )

        # BM25 Retriever
        bm25_config = self.config.get("bm25", {})
        self.bm25_retriever = ChineseBM25(
            k1=bm25_config.get("k1", 1.5),
            b=bm25_config.get("b", 0.75),
        )

        # RRF Fusion
        fusion_config = self.config.get("fusion", {})
        self.fusion = RRFusion(k=fusion_config.get("k", 60))

        # Cross-Encoder Reranker
        reranker_config = self.config.get("reranker", {})
        self.reranker = CrossEncoderReranker(
            model_name=reranker_config.get(
                "model", "BAAI/bge-reranker-v2-m3"
            ),
            device=reranker_config.get("device", "cpu"),
        )

        # Deduplicator
        dedup_config = self.config.get("dedup", {})
        self.dedup = Deduplicator(
            strategy=dedup_config.get("strategy", "exact"),
        )

        self._is_indexed = False

    def build_index(self, documents: List[Dict]):
        """
        构建索引

        Parameters
        ----------
        documents : List[Dict]
            文档列表，每篇包含：
            - "id": str
            - "text": str
            - "metadata": dict (optional)
        """
        logger.info(f"开始构建索引，文档数: {len(documents)}")

        # 准备 Dense 检索
        doc_objects = [
            Document(id=doc["id"], text=doc["text"])
            for doc in documents
        ]
        self.dense_retriever.index_documents(doc_objects)

        # 准备 BM25 检索
        texts = [doc["text"] for doc in documents]
        self.bm25_retriever.fit(texts)

        # 存储原文
        self.doc_texts = {
            doc["id"]: doc["text"] for doc in documents
        }

        self._is_indexed = True
        logger.info("索引构建完成")

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict] = None,
        return_scores: bool = True,
    ) -> List[SearchResult]:
        """
        搜索入口

        Parameters
        ----------
        query : str
            用户查询
        top_k : int
            返回结果数
        filters : dict, optional
            结构化过滤条件
        return_scores : bool
            是否返回分数

        Returns
        -------
        List[SearchResult]
            搜索结果
        """
        if not self._is_indexed:
            raise RuntimeError("请先调用 build_index()")

        logger.info(f"查询: {query}")
        t_start = time.time()

        # Step 1: 多路召回
        dense_config = self.config.get("dense", {})
        bm25_config = self.config.get("bm25", {})

        dense_results = self.dense_retriever.retrieve(
            query, top_k=dense_config.get("top_k", 100)
        )
        bm25_results = self.bm25_retriever.retrieve(
            query, top_k=bm25_config.get("top_k", 100)
        )

        # Step 2: RRF 融合
        weights = [
            dense_config.get("weight", 1.0),
            bm25_config.get("weight", 1.0),
        ]
        fused = self.fusion.fuse(
            [dense_results, bm25_results],
            weights=weights,
            top_k=self.config.get("reranker", {}).get("top_k", 30),
        )

        # Step 3: Cross-Encoder 重排序
        reranker_config = self.config.get("reranker", {})
        rerank_top_k = reranker_config.get("top_k", 30)

        candidates = []
        for doc_id, _ in fused[:rerank_top_k]:
            text = self.doc_texts.get(doc_id, "")
            candidates.append((doc_id, text))

        reranked = self.reranker.rerank(query, candidates)

        # Step 4: 去重
        dedup_results = []
        for doc_id, score in reranked[:top_k]:
            text = self.doc_texts.get(doc_id, "")
            dedup_results.append((doc_id, text, score))

        # Step 5: 组装结果
        final_results = []
        for doc_id, text, score in dedup_results:
            final_results.append(SearchResult(
                chunk_id=doc_id,
                text=text,
                score=score,
            ))

        t_end = time.time()
        logger.info(
            f"检索完成，耗时 {(t_end - t_start) * 1000:.1f}ms, "
            f"返回 {len(final_results)} 条结果"
        )

        return final_results

    def evaluate(
        self,
        test_queries: List[Tuple[str, List[str]]],
        k_list: List[int] = [1, 3, 5, 10],
    ) -> Dict:
        """
        在测试集上评估

        Parameters
        ----------
        test_queries : List[Tuple[str, List[str]]]
            [(query, [相关文档 ID]), ...]
        k_list : List[int]
            要评估的 K 值

        Returns
        -------
        Dict
            评估结果
        """
        queries_results = []
        queries_relevant = []

        for query, relevant_ids in test_queries:
            results = self.search(query, top_k=max(k_list))
            queries_results.append([r.chunk_id for r in results])
            queries_relevant.append(set(relevant_ids))

        evaluator = RetrievalMetrics()
        metrics = evaluator.evaluate_all(
            queries_results, queries_relevant, k_list=k_list
        )

        return metrics


# ============================================================
# 使用示例
# ============================================================
if __name__ == "__main__":
    # 1. 准备文档
    documents = [
        {"id": "1", "text": "Transformer 架构由编码器和解码器组成，是 NLP 领域的里程碑"},
        {"id": "2", "text": "BERT 使用双向注意力机制进行预训练，在 11 项任务上刷新纪录"},
        {"id": "3", "text": "GPT 系列模型采用自回归生成方式，引领了 LLM 的发展"},
        {"id": "4", "text": "注意力机制允许模型关注输入序列的不同位置，是核心创新"},
        {"id": "5", "text": "残差连接解决了深层网络的梯度消失问题，使训练更稳定"},
        {"id": "6", "text": "多头注意力从不同表示子空间学习信息，增强了模型表达能力"},
        {"id": "7", "text": "位置编码为 Transformer 提供序列位置信息"},
        {"id": "8", "text": "Layer Normalization 稳定了训练过程，加速了模型收敛"},
        {"id": "9", "text": "自回归模型通过预测下一个 token 进行训练"},
        {"id": "10", "text": "Masked Language Model 通过预测被遮盖的词进行训练"},
    ]

    # 2. 初始化系统
    pipeline = ProductionMultiRecall()

    # 3. 构建索引
    pipeline.build_index(documents)

    # 4. 搜索
    queries = [
        "自回归生成模型是如何工作的",
        "注意力机制为什么重要",
        "BERT 和 GPT 有什么区别",
    ]

    for query in queries:
        print(f"\n{'=' * 50}")
        print(f"查询: {query}")
        print(f"{'=' * 50}")
        results = pipeline.search(query, top_k=3)
        for i, result in enumerate(results, 1):
            print(f"  {i}. [{result.chunk_id}] (score={result.score:.4f}) {result.text}")
```

---

## 7.11 本章小结

### 7.11.1 多路召回的黄金法则

1. **多样性优于精度**：每路检索方法的"盲区"不同，多路互补效果远好于单路极致优化
2. **RRF 是默认融合方案**：分数无关、无需训练、鲁棒性强
3. **Cross-Encoder 是必选项**：重排序阶段投入少量计算带来显著精度提升
4. **结构化过滤不要忘**：时间和类别过滤是防止语义漂移的简单有效手段
5. **KG 检索是差异化武器**：在需要精确关系推理的场景中不可替代

### 7.11.2 各组件性能参考

| 组件 | 100 篇文档 | 10K 篇文档 | 1M 篇文档 |
|---|---|---|---|
| Dense (暴力) | 1ms | 100ms | 不可行 |
| Dense (FAISS IVF) | 1ms | 2ms | 50ms |
| BM25 | 0.5ms | 5ms | 50ms |
| Structured (SQLite) | 0.5ms | 1ms | 5ms |
| Cross-Encoder (100 候选) | 50ms | 50ms | 50ms |

### 7.11.3 典型配置推荐

| 场景 | Dense | BM25 | Structured | KG | Reranker |
|---|---|---|---|---|---|
| 通用问答 | 必需 | 必需 | 可选 | 可选 | 推荐 |
| 学术论文搜索 | 必需 | 必需 | 必需 | 可选 | 推荐 |
| 电商搜索 | 必需 | 必需 | 必需 | 推荐 | 推荐 |
| 医疗知识库 | 必需 | 必需 | 必需 | 必需 | 必需 |
| 法律文书检索 | 可选 | 必需 | 必需 | 推荐 | 必需 |

### 7.11.4 练习题

1. **基础题**：实现一个简化版 RRF，只用 2 路结果融合，比较 k=1 和 k=60 时的结果差异
2. **进阶题**：在 BM25 中添加字段加权（标题权重 > 正文权重），观察对检索结果的影响
3. **挑战题**：实现一个 Learning-to-Rank 排序器，使用 LambdaRank 替换 RRF
4. **实践题**：用本章的多路召回管线在 MS MARCO 数据集上评估，对比 Dense Only / BM25 Only / Dense+BM25+RRF / Full Pipeline 的 MRR@10

### 7.11.5 参考资源

- [RRF 原始论文](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
- [Sentence-Transformers Cross-Encoder](https://www.sbert.net/examples/applications/cross-encoder/README.html)
- [FAISS 官方文档](https://github.com/facebookresearch/faiss)
- [BM25 算法详解](https://en.wikipedia.org/wiki/Okapi_BM25)
- [NDCG 指标详解](https://en.wikipedia.org/wiki/Discounted_cumulative_gain)
- [GraphRAG: 微软图检索增强生成](https://microsoft.github.io/graphrag/)
- [BGE Reranker 模型](https://huggingface.co/BAAI/bge-reranker-v2-m3)
