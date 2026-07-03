# 第11章 混合检索策略

## 11.1 引言

在前面的章节中，我们分别讨论了**密集检索**（Dense Retrieval）和**稀疏检索**（Sparse Retrieval）两种范式。密集检索使用嵌入向量（Embeddings）在语义空间中寻找相似内容，而稀疏检索则依赖关键词匹配（如 BM25）来实现精确匹配。两种方法各有优劣：

| 特性 | 密集检索（Dense） | 稀疏检索（Sparse） |
|------|-------------------|-------------------|
| 语义理解 | 强，可捕捉同义词和意译 | 弱，依赖字面匹配 |
| 精确匹配 | 弱，可能漏掉精确关键词 | 强，确保关键词命中 |
| 罕见词处理 | 较差，罕见词嵌入不充分 | 很好，直接匹配词项 |
| 领域适应性 | 需要微调 | 无需训练 |
| 计算开销 | 高（向量计算） | 低（倒排索引） |

混合检索（Hybrid Search）的核心思想是将这两种方法的优势结合起来，通过融合策略（Fusion Strategy）对多路检索结果进行排序和整合，从而在召回率和精确率之间取得更好的平衡。

本章将系统地介绍混合检索的各种核心技术，包括：

1. **稠密-稀疏融合**：RRF、加权求和等融合算法
2. **自适应权重调整**：根据查询特征动态调整融合权重
3. **Small-to-Big 检索**：从细粒度块到粗粒度上下文的渐进式检索
4. **Step-back Prompting**：通过抽象化查询提升检索质量
5. **多阶段检索**：粗筛 + 精排的级联架构
6. **查询重写技术**：多种查询变换策略

## 11.2 稠密-稀疏融合

稠密-稀疏融合是混合检索最基础也最核心的技术。其基本流程如下：

```
用户查询
    │
    ├──→ 稠密检索（嵌入向量相似度）──→ 得分列表 A
    │
    └──→ 稀疏检索（BM25/TF-IDF） ──→ 得分列表 B
                                        │
                                        ▼
                              融合排序（RRF/加权和）
                                        │
                                        ▼
                                  最终排序结果
```

### 11.2.1 倒数排名融合（RRF）

**倒数排名融合**（Reciprocal Rank Fusion, RRF）是一种不依赖得分绝对值、仅使用排名信息的融合方法。其基本公式为：

```
RRF_score(d) = Σ [ 1 / (k + rank_i(d)) ]
```

其中 `rank_i(d)` 是文档 `d` 在第 `i` 路检索中的排名，`k` 是一个平滑常数（通常取 60）。

RRF 的核心优势在于它不需要各路检索的得分具有可比性——只需要排名即可。这使得它非常适合融合使用不同评分机制的检索器。

```python
import numpy as np
from typing import List, Dict, Tuple, Any


def reciprocal_rank_fusion(
    rankings: List[List[str]],
    k: int = 60,
    doc_scores: Dict[str, float] = None
) -> List[Tuple[str, float]]:
    """
    倒数排名融合（RRF）算法。

    参数:
        rankings: 多路检索结果列表，每路是一个文档ID列表（按排名降序）
        k: 平滑常数，默认 60
        doc_scores: 可选的文档最终得分字典（用于记录）

    返回:
        融合后的 (文档ID, RRF得分) 列表，按得分降序排列
    """
    rrf_scores = {}

    for rank_list in rankings:
        for rank, doc_id in enumerate(rank_list):
            # rank 从 0 开始，所以实际排名 = rank + 1
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0.0
            rrf_scores[doc_id] += 1.0 / (k + rank + 1)

    # 按 RRF 得分降序排序
    sorted_results = sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return sorted_results


def dense_retrieval(query: str, documents: List[str], top_k: int = 10) -> List[str]:
    """
    模拟密集检索（使用简单的 TF-IDF 风格向量作为示例）。
    实际应用中应使用 SentenceTransformer 等模型。
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    # 将查询和文档一起向量化
    all_texts = [query] + documents
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        max_features=5000
    )
    vectors = vectorizer.fit_transform(all_texts)

    query_vec = vectors[0:1]
    doc_vecs = vectors[1:]

    # 计算余弦相似度
    similarities = cosine_similarity(query_vec, doc_vecs).flatten()

    # 按相似度排序并返回文档索引
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [str(idx) for idx in top_indices]


def sparse_retrieval(query: str, documents: List[str], top_k: int = 10) -> List[str]:
    """
    模拟稀疏检索（BM25 风格）。
    使用字符 n-gram 模拟 BM25 的关键词匹配特性。
    """
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    # 使用 unigram 和 bigram 模拟词袋模型
    all_texts = [query] + documents
    vectorizer = CountVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=10000
    )
    vectors = vectorizer.fit_transform(all_texts)

    query_vec = vectors[0:1]
    doc_vecs = vectors[1:]

    # 使用更"稀疏"的匹配方式——只计算词项重叠
    # 这里用点积代替余弦，更接近 BM25 的累加特性
    scores = (query_vec @ doc_vecs.T).toarray().flatten()

    top_indices = np.argsort(scores)[::-1][:top_k]
    return [str(idx) for idx in top_indices]


def hybrid_search_rrf(
    query: str,
    documents: List[str],
    top_k: int = 10,
    k_rrf: int = 60
) -> List[Tuple[str, float]]:
    """
    使用 RRF 的混合检索。

    参数:
        query: 用户查询
        documents: 文档列表
        top_k: 每路检索返回的文档数
        k_rrf: RRF 平滑常数

    返回:
        融合后的 (文档ID, 得分) 列表
    """
    # 执行两路检索
    dense_results = dense_retrieval(query, documents, top_k)
    sparse_results = sparse_retrieval(query, documents, top_k)

    print(f"密集检索结果: {dense_results}")
    print(f"稀疏检索结果: {sparse_results}")

    # RRF 融合
    fused = reciprocal_rank_fusion(
        [dense_results, sparse_results],
        k=k_rrf
    )

    return fused


# ============================================================
# 示例运行
# ============================================================
if __name__ == "__main__":
    documents = [
        "Python 是一种高级编程语言，以其简洁的语法和强大的库生态闻名。",
        "JavaScript 是 Web 开发的核心语言，运行在浏览器和 Node.js 环境中。",
        "深度学习是机器学习的一个分支，使用多层神经网络进行学习。",
        "自然语言处理（NLP）让计算机能够理解和生成人类语言。",
        "Python 在数据科学和机器学习领域有着广泛的应用。",
        "Java 是一种面向对象的编程语言，广泛应用于企业级开发。",
        "Transformer 架构彻底改变了 NLP 领域的面貌。",
        "BERT 和 GPT 都是基于 Transformer 的预训练语言模型。",
        "数据库管理系统（DBMS）用于高效地存储和检索数据。",
        "向量数据库专门用于存储和查询高维向量数据。",
    ]

    query = "Python 机器学习的应用"

    print(f"查询: {query}\n")
    results = hybrid_search_rrf(query, documents, top_k=5)

    print("\nRRF 融合结果:")
    for doc_id, score in results:
        print(f"  文档 {doc_id}: {documents[int(doc_id)]} (得分: {score:.4f})")
```

### 11.2.2 加权求和融合

与 RRF 不同，加权求和（Weighted Sum）直接使用各路检索的原始得分进行融合。这种方法要求各路得分能够被归一化到可比较的范围。

```python
def normalize_scores(
    scores: Dict[str, float],
    method: str = "minmax"
) -> Dict[str, float]:
    """
    对得分进行归一化处理。

    参数:
        scores: 原始得分字典
        method: 归一化方法 ("minmax" 或 "zscore")

    返回:
        归一化后的得分字典
    """
    values = np.array(list(scores.values()))

    if method == "minmax":
        # Min-Max 归一化到 [0, 1]
        vmin, vmax = values.min(), values.max()
        if vmax - vmin < 1e-10:
            normalized = np.zeros_like(values)
        else:
            normalized = (values - vmin) / (vmax - vmin)

    elif method == "zscore":
        # Z-Score 归一化
        mean, std = values.mean(), values.std()
        if std < 1e-10:
            normalized = np.zeros_like(values)
        else:
            normalized = (values - mean) / std

    else:
        raise ValueError(f"未知的归一化方法: {method}")

    return dict(zip(scores.keys(), normalized))


def weighted_sum_fusion(
    score_lists: List[Dict[str, float]],
    weights: List[float] = None,
    normalization: str = "minmax"
) -> List[Tuple[str, float]]:
    """
    加权求和融合。

    参数:
        score_lists: 多路得分字典列表，每个字典是 {文档ID: 得分}
        weights: 每路检索的权重，默认等权
        normalization: 得分归一化方法

    返回:
        融合后的 (文档ID, 总分) 列表
    """
    n_rankers = len(score_lists)

    if weights is None:
        weights = [1.0 / n_rankers] * n_rankers

    assert len(weights) == n_rankers, "权重数量必须与检索器数量一致"
    assert abs(sum(weights) - 1.0) < 1e-6, "权重之和必须为 1"

    # 归一化各路得分
    normalized_lists = []
    for scores in score_lists:
        normalized_lists.append(
            normalize_scores(scores, method=normalization)
        )

    # 加权求和
    fused_scores = {}
    all_doc_ids = set()
    for scores in normalized_lists:
        all_doc_ids.update(scores.keys())

    for doc_id in all_doc_ids:
        total = 0.0
        for weight, scores in zip(weights, normalized_lists):
            total += weight * scores.get(doc_id, 0.0)
        fused_scores[doc_id] = total

    # 按总分降序排列
    sorted_results = sorted(
        fused_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return sorted_results


def hybrid_search_weighted(
    query: str,
    documents: List[str],
    top_k: int = 10,
    dense_weight: float = 0.5,
    sparse_weight: float = 0.5
) -> List[Tuple[str, float]]:
    """
    使用加权求和的混合检索。

    参数:
        query: 用户查询
        documents: 文档列表
        top_k: 每路返回的文档数
        dense_weight: 密集检索权重
        sparse_weight: 稀疏检索权重

    返回:
        融合后的 (文档ID, 得分) 列表
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    all_texts = [query] + documents
    top_k = min(top_k, len(documents))

    # ---- 密集检索（语义） ----
    dense_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        max_features=5000
    )
    dense_vectors = dense_vectorizer.fit_transform(all_texts)
    dense_sim = cosine_similarity(dense_vectors[0:1], dense_vectors[1:]).flatten()

    dense_scores = {}
    for i, score in enumerate(dense_sim):
        dense_scores[str(i)] = float(score)

    # ---- 稀疏检索（关键词） ----
    sparse_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 1),
        max_features=10000,
        sublinear_tf=True  # 使用 log(tf) 平滑
    )
    sparse_vectors = sparse_vectorizer.fit_transform(all_texts)

    # 使用点积（更接近 BM25 的累加特性）
    sparse_scores_val = (sparse_vectors[0:1] @ sparse_vectors[1:].T).toarray().flatten()

    sparse_scores = {}
    for i, score in enumerate(sparse_scores_val):
        sparse_scores[str(i)] = float(score)

    # ---- 融合 ----
    fused = weighted_sum_fusion(
        [dense_scores, sparse_scores],
        weights=[dense_weight, sparse_weight],
        normalization="minmax"
    )

    return fused[:top_k]


# ============================================================
# 带参数搜索的加权融合
# ============================================================
def tune_hybrid_weights(
    query: str,
    documents: List[str],
    relevant_docs: List[str],
    weight_grid: List[float] = None
) -> Tuple[float, float, float]:
    """
    通过网格搜索找到最优的密集/稀疏权重组合。

    参数:
        query: 查询
        documents: 文档列表
        relevant_docs: 相关文档ID列表
        weight_grid: 要搜索的密集权重列表

    返回:
        (最优密集权重, 最优稀疏权重, 最优召回率)
    """
    if weight_grid is None:
        weight_grid = np.arange(0.0, 1.05, 0.05)

    best_recall = 0.0
    best_weights = (0.5, 0.5)

    for dw in weight_grid:
        sw = 1.0 - dw
        results = hybrid_search_weighted(
            query, documents, top_k=len(documents),
            dense_weight=dw, sparse_weight=sw
        )

        retrieved = set(doc_id for doc_id, _ in results)
        relevant_set = set(relevant_docs)

        if len(relevant_set) == 0:
            continue

        recall = len(retrieved & relevant_set) / len(relevant_set)

        if recall > best_recall:
            best_recall = recall
            best_weights = (dw, sw)

    return (*best_weights, best_recall)


if __name__ == "__main__":
    documents = [
        "Python 是一种高级编程语言，以其简洁的语法和强大的库生态闻名。",
        "JavaScript 是 Web 开发的核心语言，运行在浏览器和 Node.js 环境中。",
        "深度学习是机器学习的一个分支，使用多层神经网络进行学习。",
        "自然语言处理（NLP）让计算机能够理解和生成人类语言。",
        "Python 在数据科学和机器学习领域有着广泛的应用。",
        "Java 是一种面向对象的编程语言，广泛应用于企业级开发。",
        "Transformer 架构彻底改变了 NLP 领域的面貌。",
        "BERT 和 GPT 都是基于 Transformer 的预训练语言模型。",
        "数据库管理系统（DBMS）用于高效地存储和检索数据。",
        "向量数据库专门用于存储和查询高维向量数据。",
    ]

    query = "Python 机器学习的应用"

    print(f"查询: {query}")
    print(f"密集权重=0.7, 稀疏权重=0.3:")
    results = hybrid_search_weighted(query, documents, top_k=5,
                                      dense_weight=0.7, sparse_weight=0.3)
    for doc_id, score in results:
        print(f"  文档 {doc_id}: {documents[int(doc_id)]} (得分: {score:.4f})")

    print(f"\n密集权重=0.3, 稀疏权重=0.7:")
    results = hybrid_search_weighted(query, documents, top_k=5,
                                      dense_weight=0.3, sparse_weight=0.7)
    for doc_id, score in results:
        print(f"  文档 {doc_id}: {documents[int(doc_id)]} (得分: {score:.4f})")
```

### 11.2.3 RRF 与加权求和的对比

```python
def compare_fusion_methods(
    query: str,
    documents: List[str],
    ground_truth: List[str],
    k_values: List[int] = None
) -> Dict:
    """
    对比 RRF 和加权求和的召回率表现。

    参数:
        query: 查询
        documents: 文档列表
        ground_truth: 人工标注的相关文档ID列表
        k_values: 要评估的 top-k 值列表

    返回:
        包含两种方法在不同 k 值下召回率的字典
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]

    results = {
        "rrf": {},
        "weighted_d0.5": {},
        "weighted_d0.7": {},
        "weighted_d0.3": {}
    }

    dense_rank = dense_retrieval(query, documents, top_k=len(documents))
    sparse_rank = sparse_retrieval(query, documents, top_k=len(documents))

    for k in k_values:
        # RRF
        rrf_results = reciprocal_rank_fusion(
            [dense_rank[:k], sparse_rank[:k]], k=60
        )
        rrf_topk = set(doc_id for doc_id, _ in rrf_results[:k])

        # 加权和 (dense=0.5)
        ws05 = hybrid_search_weighted(query, documents, top_k=k,
                                       dense_weight=0.5, sparse_weight=0.5)
        ws05_topk = set(doc_id for doc_id, _ in ws05)

        # 加权和 (dense=0.7)
        ws07 = hybrid_search_weighted(query, documents, top_k=k,
                                       dense_weight=0.7, sparse_weight=0.3)
        ws07_topk = set(doc_id for doc_id, _ in ws07)

        # 加权和 (dense=0.3)
        ws03 = hybrid_search_weighted(query, documents, top_k=k,
                                       dense_weight=0.3, sparse_weight=0.7)
        ws03_topk = set(doc_id for doc_id, _ in ws03)

        gt = set(ground_truth)

        def recall(retrieved):
            return len(retrieved & gt) / len(gt) if gt else 0.0

        results["rrf"][k] = recall(rrf_topk)
        results["weighted_d0.5"][k] = recall(ws05_topk)
        results["weighted_d0.7"][k] = recall(ws07_topk)
        results["weighted_d0.3"][k] = recall(ws03_topk)

    return results


# ============================================================
# 实际使用示例：使用 Sentence Transformers 进行真实密集检索
# ============================================================
class RealHybridRetriever:
    """
    使用真实嵌入模型的混合检索器。
    需要安装: pip install sentence-transformers rank_bm25
    """

    def __init__(
        self,
        dense_model_name: str = "BAAI/bge-small-zh-v1.5",
        sparse_weight: float = 0.3,
        dense_weight: float = 0.7,
        fusion_method: str = "weighted_sum"
    ):
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.fusion_method = fusion_method
        self.documents = []
        self.embeddings = None
        self.bm25 = None
        self._dense_model = None

    def _lazy_load_dense_model(self):
        """延迟加载密集检索模型"""
        if self._dense_model is None:
            from sentence_transformers import SentenceTransformer
            self._dense_model = SentenceTransformer(
                self.dense_model_name,
                device="cpu"
            )
        return self._dense_model

    def _build_bm25(self, documents: List[str]):
        """构建 BM25 索引"""
        from rank_bm25 import BM25Okapi
        import jieba

        tokenized_docs = []
        for doc in documents:
            # 使用 jieba 进行中文分词
            tokens = list(jieba.cut(doc))
            tokenized_docs.append(tokens)

        self.bm25 = BM25Okapi(tokenized_docs)

    def index(self, documents: List[str]):
        """
        索引文档。

        参数:
            documents: 文档列表
        """
        self.documents = documents

        # 构建密集检索索引
        model = self._lazy_load_dense_model()
        self.embeddings = model.encode(documents, normalize_embeddings=True)

        # 构建稀疏检索索引
        self._build_bm25(documents)

        print(f"索引完成: {len(documents)} 篇文档")
        print(f"  嵌入维度: {self.embeddings.shape[1]}")

    def search(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Tuple[int, str, float]]:
        """
        执行混合检索。

        参数:
            query: 查询字符串
            top_k: 返回结果数

        返回:
            (索引, 文档内容, 融合得分) 列表
        """
        # 密集检索
        model = self._lazy_load_dense_model()
        query_emb = model.encode(query, normalize_embeddings=True)

        dense_scores = {}
        for i, doc_emb in enumerate(self.embeddings):
            score = np.dot(query_emb, doc_emb)
            dense_scores[i] = float(score)

        # 稀疏检索 (BM25)
        import jieba
        query_tokens = list(jieba.cut(query))
        bm25_scores = self.bm25.get_scores(query_tokens)

        sparse_scores = {}
        for i, score in enumerate(bm25_scores):
            sparse_scores[i] = float(score)

        # 融合
        if self.fusion_method == "weighted_sum":
            fused = weighted_sum_fusion(
                [dense_scores, sparse_scores],
                weights=[self.dense_weight, self.sparse_weight],
                normalization="minmax"
            )
        elif self.fusion_method == "rrf":
            # 先获取排名
            dense_rank = sorted(dense_scores.keys(),
                                key=lambda x: dense_scores[x], reverse=True)
            sparse_rank = sorted(sparse_scores.keys(),
                                 key=lambda x: sparse_scores[x], reverse=True)
            fused = reciprocal_rank_fusion(
                [[str(d) for d in dense_rank], [str(d) for d in sparse_rank]]
            )
            # 转回整数索引
            fused = [(int(doc_id), score) for doc_id, score in fused]
        else:
            raise ValueError(f"未知融合方法: {self.fusion_method}")

        # 返回 top-k 结果
        results = []
        for idx, score in fused[:top_k]:
            results.append((idx, self.documents[idx], score))

        return results


# ============================================================
# 使用示例
# ============================================================
if __name__ == "__main__":
    # 注意: 运行以下代码需要安装 sentence-transformers 和 rank_bm25
    # pip install sentence-transformers rank_bm25 jieba

    sample_docs = [
        "Python 在数据科学领域有着广泛应用，包括NumPy、Pandas等库。",
        "深度学习模型需要大量GPU资源进行训练。",
        "自然语言处理技术用于文本分类和情感分析。",
        "向量数据库如Milvus和Qdrant支持高效的相似度搜索。",
        "BERT模型通过掩码语言模型进行预训练。",
        "RAG系统结合了检索和生成两个阶段。",
        "BM25是一种基于词频和逆文档频率的排序函数。",
        "Sentence Transformer将句子映射到固定维度的向量空间。",
        "倒排索引是全文搜索的核心数据结构。",
        "混合检索结合了稀疏检索和密集检索的优势。",
    ]

    # 实例化检索器
    retriever = RealHybridRetriever(
        dense_model_name="BAAI/bge-small-zh-v1.5",
        dense_weight=0.7,
        sparse_weight=0.3,
        fusion_method="weighted_sum"
    )

    retriever.index(sample_docs)

    # 执行搜索
    query = "深度学习的训练资源需求"
    results = retriever.search(query, top_k=3)

    print(f"\n查询: {query}")
    print("混合检索结果:")
    for idx, doc, score in results:
        print(f"  [{idx}] (得分: {score:.4f}) {doc}")
```

## 11.3 自适应权重调整

固定的融合权重无法适应所有查询类型。有些查询更适合语义匹配（如"机器学习的核心概念"），而有些则更需要精确关键词匹配（如"Python 3.12 新特性"）。**自适应权重调整**（Adaptive Weight Adjustment）根据查询的特征动态决定各路检索的权重。

### 11.3.1 基于查询特征的权重预测

```python
from typing import List, Dict, Tuple, Optional
import re
import math


class QueryFeatureExtractor:
    """
    查询特征提取器，用于分析查询的各种统计特征。
    """

    @staticmethod
    def extract_features(query: str) -> Dict[str, float]:
        """
        提取查询的各类特征。

        参数:
            query: 查询字符串

        返回:
            特征字典
        """
        features = {}

        # 1. 查询长度特征
        features["length_chars"] = len(query)
        features["length_words"] = len(query.split())

        # 2. 词项统计
        words = query.split()
        features["avg_word_length"] = (
            sum(len(w) for w in words) / len(words) if words else 0.0
        )

        # 3. 大写/特殊字符比例
        upper_count = sum(1 for c in query if c.isupper())
        features["upper_ratio"] = upper_count / len(query) if query else 0.0

        # 4. 数字比例
        digit_count = sum(1 for c in query if c.isdigit())
        features["digit_ratio"] = digit_count / len(query) if query else 0.0

        # 5. 标点符号比例
        punct_count = sum(1 for c in query if not c.isalnum() and not c.isspace())
        features["punct_ratio"] = punct_count / len(query) if query else 0.0

        # 6. 专有名词/技术术语检测
        # 以大写字母开头的词（英文语境）
        proper_nouns = sum(1 for w in words if w and w[0].isupper())
        features["proper_noun_ratio"] = (
            proper_nouns / len(words) if words else 0.0
        )

        # 7. 信息熵（衡量查询的信息量）
        char_freq = {}
        for c in query.lower():
            char_freq[c] = char_freq.get(c, 0) + 1
        entropy = 0.0
        for c, freq in char_freq.items():
            p = freq / len(query)
            if p > 0:
                entropy -= p * math.log2(p)
        features["entropy"] = entropy

        # 8. 中文字符比例
        chinese_count = sum(1 for c in query if '一' <= c <= '鿿')
        features["chinese_ratio"] = (
            chinese_count / len(query) if query else 0.0
        )

        # 9. 英文单词比例
        english_words = sum(
            1 for w in words if w and all('a' <= c.lower() <= 'z' for c in w)
        )
        features["english_word_ratio"] = (
            english_words / len(words) if words else 0.0
        )

        return features

    @staticmethod
    def is_semantic_query(features: Dict[str, float]) -> float:
        """
        判断查询是否更倾向于语义匹配（返回 0~1 的分数）。

        语义查询的特征：
        - 长度较长（说明是完整问题而非关键词）
        - 信息熵高（词汇多样性高）
        - 中文字符比例高（中文查询更依赖语义）
        - 专有名词比例低

        返回:
            语义倾向性分数，越高越适合密集检索
        """
        score = 0.0

        # 长查询更可能是语义查询
        if features["length_chars"] > 20:
            score += 0.3
        elif features["length_chars"] > 10:
            score += 0.15

        # 高信息熵说明词汇丰富
        if features["entropy"] > 3.5:
            score += 0.2

        # 中文比例高
        if features["chinese_ratio"] > 0.5:
            score += 0.25

        # 专有名词少（专有名词多说明是精确查找）
        if features["proper_noun_ratio"] < 0.1:
            score += 0.15

        # 英文单词多可能包含技术术语，需要精确匹配
        if features["english_word_ratio"] > 0.3:
            score -= 0.2

        return max(0.0, min(1.0, score))

    @staticmethod
    def is_keyword_query(features: Dict[str, float]) -> float:
        """
        判断查询是否更倾向于关键词匹配（返回 0~1 的分数）。
        """
        score = 0.0

        # 短查询更可能是关键词查询
        if features["length_chars"] < 8:
            score += 0.3

        # 包含数字（版本号、年份等）
        if features["digit_ratio"] > 0.05:
            score += 0.2

        # 包含大写字母（专有名词、缩写）
        if features["upper_ratio"] > 0.1:
            score += 0.2

        # 技术术语（英文单词多）
        if features["english_word_ratio"] > 0.3:
            score += 0.2

        # 低熵（词汇重复度高，像是关键词堆叠）
        if features["entropy"] < 2.5:
            score += 0.15

        return max(0.0, min(1.0, score))


class AdaptiveWeightHybridRetriever:
    """
    自适应权重混合检索器。

    根据查询特征动态调整密集检索和稀疏检索的权重。
    """

    def __init__(
        self,
        dense_model_name: str = "BAAI/bge-small-zh-v1.5",
        base_dense_weight: float = 0.5,
        base_sparse_weight: float = 0.5,
        weight_adjust_range: float = 0.4
    ):
        self.base_dense_weight = base_dense_weight
        self.base_sparse_weight = base_sparse_weight
        self.weight_adjust_range = weight_adjust_range
        self.feature_extractor = QueryFeatureExtractor()
        self.documents = []
        self.embeddings = None
        self.bm25 = None

    def _compute_adaptive_weights(
        self,
        query: str
    ) -> Tuple[float, float]:
        """
        计算自适应权重。

        参数:
            query: 查询字符串

        返回:
            (密集检索权重, 稀疏检索权重)
        """
        features = self.feature_extractor.extract_features(query)

        semantic_score = self.feature_extractor.is_semantic_query(features)
        keyword_score = self.feature_extractor.is_keyword_query(features)

        print(f"\n查询特征分析:")
        for k, v in features.items():
            print(f"  {k}: {v:.4f}")
        print(f"  语义倾向: {semantic_score:.4f}")
        print(f"  关键词倾向: {keyword_score:.4f}")

        # 根据语义倾向调整密集检索权重
        # semantic_score 越高，密集检索权重越大
        dense_adjustment = (semantic_score - 0.5) * 2 * self.weight_adjust_range

        dense_weight = self.base_dense_weight + dense_adjustment
        sparse_weight = 1.0 - dense_weight

        # 约束在 [0.1, 0.9] 范围内
        dense_weight = max(0.1, min(0.9, dense_weight))
        sparse_weight = 1.0 - dense_weight

        print(f"  调整后密集权重: {dense_weight:.4f}")
        print(f"  调整后稀疏权重: {sparse_weight:.4f}")

        return dense_weight, sparse_weight

    def index(self, documents: List[str]):
        """索引文档（同前）"""
        from sentence_transformers import SentenceTransformer

        self.documents = documents
        model = SentenceTransformer(
            self.dense_model_name,
            device="cpu"
        )
        self.embeddings = model.encode(documents, normalize_embeddings=True)

        from rank_bm25 import BM25Okapi
        import jieba
        tokenized_docs = []
        for doc in documents:
            tokens = list(jieba.cut(doc))
            tokenized_docs.append(tokens)
        self.bm25 = BM25Okapi(tokenized_docs)

    def search(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Tuple[int, str, float]]:
        """
        使用自适应权重执行混合检索。

        参数:
            query: 查询字符串
            top_k: 返回结果数

        返回:
            (索引, 文档内容, 融合得分) 列表
        """
        dense_weight, sparse_weight = self._compute_adaptive_weights(query)

        # 密集检索
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(
            self.dense_model_name,
            device="cpu"
        )
        query_emb = model.encode(query, normalize_embeddings=True)

        dense_scores = {}
        for i, doc_emb in enumerate(self.embeddings):
            dense_scores[i] = float(np.dot(query_emb, doc_emb))

        # 稀疏检索
        import jieba
        query_tokens = list(jieba.cut(query))
        bm25_scores = self.bm25.get_scores(query_tokens)

        sparse_scores = {}
        for i, score in enumerate(bm25_scores):
            sparse_scores[i] = float(score)

        # 加权融合
        fused = weighted_sum_fusion(
            [dense_scores, sparse_scores],
            weights=[dense_weight, sparse_weight],
            normalization="minmax"
        )

        results = []
        for idx, score in fused[:top_k]:
            results.append((idx, self.documents[idx], score))

        return results


# ============================================================
# 使用示例
# ============================================================
if __name__ == "__main__":
    sample_docs = [
        "Python 3.12 引入了新的语法特性和性能改进。",
        "深度学习需要大量的 GPU 计算资源进行模型训练。",
        "RAG 技术通过检索外部知识来增强大语言模型的生成能力。",
        "BM25 算法基于词频和文档频率计算相关性得分。",
        "向量嵌入将文本映射到高维语义空间。",
        "Transformer 架构使用自注意力机制处理序列数据。",
        "Milvus 是一个开源的向量数据库，支持 GPU 加速。",
        "自然语言处理任务包括文本分类、命名实体识别等。",
        "Apache Lucene 是一个高性能的全文搜索引擎库。",
        "混合检索融合了关键词匹配和语义搜索的优势。",
    ]

    retriever = AdaptiveWeightHybridRetriever(
        dense_model_name="BAAI/bge-small-zh-v1.5",
        base_dense_weight=0.5,
        weight_adjust_range=0.4
    )

    retriever.index(sample_docs)

    # 测试不同查询
    queries = [
        "Python 3.12 新特性",           # 关键词密集型
        "深度学习需要哪些计算资源",     # 语义密集型
        "RAG 技术如何增强 LLM",         # 混合型
    ]

    for query in queries:
        print(f"\n{'='*60}")
        print(f"查询: {query}")
        results = retriever.search(query, top_k=3)
        print("检索结果:")
        for idx, doc, score in results:
            print(f"  [{idx}] (得分: {score:.4f}) {doc}")
```

### 11.3.2 基于历史反馈的在线权重学习

```python
class OnlineWeightLearner:
    """
    基于用户点击反馈的在线权重学习器。

    使用简单的 Bandit 算法来动态调整检索权重。
    """

    def __init__(
        self,
        weight_grid: List[float] = None,
        alpha: float = 0.1
    ):
        """
        参数:
            weight_grid: 要尝试的密集权重候选列表
            alpha: 学习率，控制权重更新的速度
        """
        if weight_grid is None:
            self.weight_grid = np.arange(0.1, 1.0, 0.1)
        else:
            self.weight_grid = weight_grid

        self.alpha = alpha
        self.n_arms = len(self.weight_grid)

        # 每个臂的累积奖励和尝试次数
        self.cumulative_rewards = np.zeros(self.n_arms)
        self.n_pulls = np.ones(self.n_arms)  # 拉普拉斯平滑

    def select_weight(self, strategy: str = "ucb") -> float:
        """
        选择当前最优的密集权重。

        参数:
            strategy: 选择策略
                - "greedy": 贪心选择当前最优
                - "ucb": 上置信界（Upper Confidence Bound）
                - "epsilon_greedy": epsilon-贪心

        返回:
            选择的密集权重
        """
        if strategy == "greedy":
            best_arm = np.argmax(
                self.cumulative_rewards / self.n_pulls
            )

        elif strategy == "ucb":
            # UCB: 选择均值 + 置信上界最大的臂
            total_pulls = self.n_pulls.sum()
            mean_rewards = self.cumulative_rewards / self.n_pulls
            confidence = np.sqrt(
                2 * np.log(total_pulls) / self.n_pulls
            )
            ucb_scores = mean_rewards + confidence
            best_arm = np.argmax(ucb_scores)

        elif strategy == "epsilon_greedy":
            epsilon = 0.1
            if np.random.random() < epsilon:
                # 探索：随机选择
                best_arm = np.random.randint(self.n_arms)
            else:
                # 利用：选择当前最优
                best_arm = np.argmax(
                    self.cumulative_rewards / self.n_pulls
                )

        else:
            raise ValueError(f"未知策略: {strategy}")

        return self.weight_grid[best_arm], best_arm

    def update(
        self,
        arm_idx: int,
        reward: float
    ):
        """
        根据反馈更新权重。

        参数:
            arm_idx: 选择的权重臂索引
            reward: 反馈奖励（如 NDCG、点击率等）
        """
        self.n_pulls[arm_idx] += 1
        self.cumulative_rewards[arm_idx] += reward

        # 指数移动平均更新
        old_mean = (
            self.cumulative_rewards[arm_idx] / self.n_pulls[arm_idx]
        )
        new_value = old_mean + self.alpha * (
            reward - old_mean
        )

        print(f"  臂 {arm_idx} (密集权重={self.weight_grid[arm_idx]:.1f}): "
              f"奖励={reward:.4f}, 新均值={new_value:.4f}")

    def get_best_weight(self) -> float:
        """获取当前最优的密集权重。"""
        best_arm = np.argmax(
            self.cumulative_rewards / self.n_pulls
        )
        return self.weight_grid[best_arm]

    def get_weight_summary(self) -> Dict:
        """获取各权重的统计摘要。"""
        summary = {}
        for i, w in enumerate(self.weight_grid):
            summary[w] = {
                "pulls": int(self.n_pulls[i]),
                "mean_reward": self.cumulative_rewards[i] / self.n_pulls[i],
                "cumulative_reward": self.cumulative_rewards[i]
            }
        return summary


def simulate_user_feedback(
    query: str,
    results: List[Tuple[int, str, float]],
    ground_truth_relevant: List[int]
) -> float:
    """
    模拟用户反馈（NDCG@3）。

    参数:
        query: 查询
        results: 检索结果列表
        ground_truth_relevant: 人工标注的相关文档索引

    返回:
        NDCG@3 得分
    """
    # 简化版 NDCG 计算
    k = min(3, len(results))
    dcg = 0.0
    idcg = 0.0

    for i in range(k):
        doc_idx = results[i][0]
        rel = 1.0 if doc_idx in ground_truth_relevant else 0.0
        dcg += (2 ** rel - 1) / np.log2(i + 2)

    # 理想排序
    for i in range(min(k, len(ground_truth_relevant))):
        idcg += 1.0 / np.log2(i + 2)

    return dcg / idcg if idcg > 0 else 0.0


# ============================================================
# 在线权重学习示例
# ============================================================
if __name__ == "__main__":
    # 模拟多轮查询
    queries_with_relevance = [
        ("Python 3.12 新特性", [0]),
        ("深度学习计算资源需求", [1]),
        ("RAG 技术架构", [2]),
        ("BM25 算法原理", [3]),
        ("向量嵌入维度", [4]),
        ("Transformer 注意力机制", [5]),
        ("向量数据库 Milvus", [6]),
        ("NLP 文本分类任务", [7]),
        ("全文搜索引擎 Lucene", [8]),
        ("混合检索优势", [9]),
    ]

    learner = OnlineWeightLearner(
        weight_grid=np.arange(0.1, 1.0, 0.1),
        alpha=0.1
    )

    print("=== 在线权重学习 ===")
    for round_num, (query, relevant) in enumerate(queries_with_relevance):
        print(f"\n第 {round_num + 1} 轮:")
        print(f"  查询: {query}")

        # 选择权重
        dense_weight, arm_idx = learner.select_weight(strategy="ucb")
        print(f"  选择密集权重: {dense_weight:.1f}")

        # 模拟检索（此处简化，直接返回相关文档）
        # 实际中应调用检索器
        mock_results = [(idx, "", 0.0) for idx in range(10)]

        # 获取用户反馈
        feedback = simulate_user_feedback(query, mock_results, relevant)
        print(f"  用户反馈 (NDCG): {feedback:.4f}")

        # 更新权重
        learner.update(arm_idx, feedback)

    print(f"\n=== 最终学习结果 ===")
    print(f"最优密集权重: {learner.get_best_weight():.1f}")
    print("\n各权重统计:")
    summary = learner.get_weight_summary()
    for w, stats in sorted(summary.items()):
        print(f"  密集权重 {w:.1f}: 尝试 {stats['pulls']} 次, "
              f"平均奖励 {stats['mean_reward']:.4f}")
```

## 11.4 Small-to-Big 检索

**Small-to-Big 检索**（从小到大检索）是一种分层检索策略。其核心思想是：在小块（Chunk）级别进行精确检索，然后将命中的小块扩展为其所属的大块（如段落、章节）作为上下文提供给 LLM。

```
文档层级:
┌─────────────────────────────────────┐
│            整篇文档                  │  ← 大块（提供给 LLM）
├──────────────────┬──────────────────┤
│   第1段          │   第2段          │  ← 中块
├────┬────┬────┬───┴──┬────┬────┬────┤
│ S1 │ S2 │ S3 │ S4  │ S5 │ S6 │ S7 │  ← 小块（检索用）
└────┴────┴────┴─────┴────┴────┴────┘
          │
     检索命中 S3、S5
          │
          ▼
扩展为包含 S3 的段落 1 + 包含 S5 的段落 2
```

```python
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass


@dataclass
class DocumentChunk:
    """文档块的数据结构。"""
    chunk_id: str
    text: str
    parent_id: str        # 父级块ID（如段落ID）
    doc_id: str           # 文档ID
    start_char: int       # 在父级中的起始位置
    end_char: int         # 在父级中的结束位置
    metadata: Dict = None


@dataclass
class DocumentSection:
    """文档段落（中块）的数据结构。"""
    section_id: str
    text: str
    doc_id: str
    child_chunk_ids: List[str] = None
    metadata: Dict = None


class SmallToBigRetriever:
    """
    Small-to-Big 检索器。

    在小块级别进行精确检索，然后扩展到中块/大块作为上下文。
    """

    def __init__(
        self,
        chunk_size: int = 200,         # 小块字符数
        chunk_overlap: int = 50,       # 小块重叠字符数
        context_window: str = "parent", # 上下文窗口: "parent" 或 "doc"
        max_context_chunks: int = 5     # 最大返回的上下文块数
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.context_window = context_window
        self.max_context_chunks = max_context_chunks

        self.chunks: Dict[str, DocumentChunk] = {}
        self.sections: Dict[str, DocumentSection] = {}
        self.documents: Dict[str, str] = {}

    def _chunk_text(
        self,
        text: str,
        chunk_size: int,
        overlap: int
    ) -> List[Tuple[str, int, int]]:
        """
        将文本切分为小块。

        参数:
            text: 原始文本
            chunk_size: 块大小（字符数）
            overlap: 重叠字符数

        返回:
            (块文本, 起始位置, 结束位置) 列表
        """
        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + chunk_size, text_len)

            # 尽量在句子边界处切分
            if end < text_len:
                # 从后往前找句子结束符
                for sep in ['。', '！', '？', '\n', '.', '!', '?']:
                    last_sep = text.rfind(sep, start, end)
                    if last_sep > start + chunk_size // 2:
                        end = last_sep + 1
                        break

            chunk_text = text[start:end]
            chunks.append((chunk_text, start, end))

            # 移动起始位置（考虑重叠）
            next_start = end - overlap if end < text_len else text_len
            if next_start <= start:
                next_start = end
            start = next_start

        return chunks

    def _split_into_sections(
        self,
        text: str,
        doc_id: str
    ) -> List[DocumentSection]:
        """
        将文档分割为段落（中块）。

        按空行或段落标记分割。
        """
        sections = []

        # 按双换行或章节标题分割
        import re
        # 匹配章节标题或空行分隔
        section_boundaries = list(re.finditer(
            r'\n\s*\n|^#+\s+.*$|^【.*】$',
            text,
            re.MULTILINE
        ))

        if not section_boundaries:
            # 如果没有明显的段落边界，将整个文档作为一个段落
            section = DocumentSection(
                section_id=f"{doc_id}_sec_0",
                text=text,
                doc_id=doc_id,
                child_chunk_ids=[]
            )
            sections.append(section)
            return sections

        prev_end = 0
        for i, match in enumerate(section_boundaries):
            start = match.start()
            if start > prev_end:
                section_text = text[prev_end:start].strip()
                if section_text:
                    section = DocumentSection(
                        section_id=f"{doc_id}_sec_{i}",
                        text=section_text,
                        doc_id=doc_id,
                        child_chunk_ids=[]
                    )
                    sections.append(section)
            prev_end = match.end()

        # 最后一段
        if prev_end < len(text):
            section_text = text[prev_end:].strip()
            if section_text:
                section = DocumentSection(
                    section_id=f"{doc_id}_sec_{len(sections)}",
                    text=section_text,
                    doc_id=doc_id,
                    child_chunk_ids=[]
                )
                sections.append(section)

        return sections

    def index_document(
        self,
        doc_id: str,
        text: str
    ):
        """
        索引一篇文档，创建分层结构。

        参数:
            doc_id: 文档ID
            text: 文档全文
        """
        self.documents[doc_id] = text

        # 1. 分割为段落（中块）
        sections = self._split_into_sections(text, doc_id)
        section_map = {}

        for section in sections:
            self.sections[section.section_id] = section
            section_map[section.section_id] = section

            # 2. 将每个段落进一步切分为小块
            small_chunks = self._chunk_text(
                section.text,
                self.chunk_size,
                self.chunk_overlap
            )

            for j, (chunk_text, start, end) in enumerate(small_chunks):
                chunk_id = f"{section.section_id}_chunk_{j}"
                chunk = DocumentChunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    parent_id=section.section_id,
                    doc_id=doc_id,
                    start_char=start,
                    end_char=end
                )
                self.chunks[chunk_id] = chunk
                section.child_chunk_ids.append(chunk_id)

        print(f"文档 {doc_id} 索引完成:")
        print(f"  段落数: {len(sections)}")
        print(f"  小块数: {len(self.chunks)}")

    def _retrieve_small_chunks(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        在小块级别执行检索（使用 BM25）。

        参数:
            query: 查询
            top_k: 返回的小块数

        返回:
            (chunk_id, 得分) 列表
        """
        from rank_bm25 import BM25Okapi
        import jieba

        if not self.chunks:
            return []

        chunk_ids = list(self.chunks.keys())
        chunk_texts = [self.chunks[cid].text for cid in chunk_ids]

        # 构建 BM25 索引
        tokenized_chunks = [
            list(jieba.cut(text)) for text in chunk_texts
        ]
        bm25 = BM25Okapi(tokenized_chunks)

        query_tokens = list(jieba.cut(query))
        scores = bm25.get_scores(query_tokens)

        # 排序并返回 top-k
        scored_chunks = list(zip(chunk_ids, scores))
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        return scored_chunks[:top_k]

    def retrieve(
        self,
        query: str,
        top_k_small: int = 10
    ) -> List[Tuple[str, str, float]]:
        """
        执行 Small-to-Big 检索。

        参数:
            query: 查询
            top_k_small: 检索的小块数量

        返回:
            (上下文文本, 来源信息, 得分) 列表
        """
        # Step 1: 在小块级别检索
        small_results = self._retrieve_small_chunks(query, top_k_small)

        print(f"小块检索结果:")
        for chunk_id, score in small_results[:5]:
            print(f"  {chunk_id}: {self.chunks[chunk_id].text[:50]}... (得分: {score:.4f})")

        # Step 2: 扩展到上下文窗口
        context_map = {}  # context_id -> (context_text, total_score, chunk_count)

        for chunk_id, score in small_results:
            chunk = self.chunks[chunk_id]

            if self.context_window == "parent":
                # 扩展为父段落
                context_id = chunk.parent_id
                section = self.sections.get(context_id)
                if section is None:
                    continue
                context_text = section.text

            elif self.context_window == "doc":
                # 扩展为整篇文档
                context_id = chunk.doc_id
                context_text = self.documents.get(context_id, "")

            else:
                # 直接使用小块本身
                context_id = chunk_id
                context_text = chunk.text

            if context_id not in context_map:
                context_map[context_id] = [context_text, 0.0, 0]

            # 聚合得分（取最高分或平均分）
            context_map[context_id][1] = max(
                context_map[context_id][1], score
            )
            context_map[context_id][2] += 1

        # Step 3: 排序并返回 top-k
        sorted_contexts = sorted(
            context_map.items(),
            key=lambda x: x[1][1],
            reverse=True
        )[:self.max_context_chunks]

        results = []
        for context_id, (context_text, score, chunk_count) in sorted_contexts:
            source = f"上下文: {context_id} (包含 {chunk_count} 个命中块)"
            results.append((context_text, source, score))

        return results


# ============================================================
# 使用示例
# ============================================================
if __name__ == "__main__":
    # 模拟长文档
    document_text = """
# 深度学习概述

深度学习是机器学习的一个重要分支，它使用多层神经网络来学习数据的层次化表示。

## 神经网络基础

神经网络由输入层、隐藏层和输出层组成。每一层包含多个神经元，神经元之间通过权重连接。

激活函数是神经网络中的关键组件。常见的激活函数包括 ReLU、Sigmoid 和 Tanh。

反向传播算法通过链式法则计算梯度，并使用梯度下降法更新网络权重。

## 卷积神经网络

卷积神经网络（CNN）特别适合处理图像数据。它使用卷积核来提取局部特征。

池化层用于降低特征图的空间维度，减少参数数量。

## 循环神经网络

循环神经网络（RNN）擅长处理序列数据，如文本和时间序列。

LSTM 和 GRU 是 RNN 的变体，它们通过门控机制解决了长期依赖问题。

## Transformer 架构

Transformer 架构完全基于注意力机制，摒弃了循环结构。

自注意力机制允许模型在处理每个位置时关注序列中的所有位置。

多头注意力通过多个注意力头捕捉不同子空间的特征表示。

BERT 和 GPT 是基于 Transformer 的著名预训练模型。

## 训练技巧

学习率调度对训练深度网络至关重要。常见策略包括阶梯式下降和余弦退火。

正则化技术如 Dropout 和 L2 正则化可以防止过拟合。

批量归一化通过标准化层输入来加速训练。
    """

    retriever = SmallToBigRetriever(
        chunk_size=100,
        chunk_overlap=20,
        context_window="parent",
        max_context_chunks=3
    )

    retriever.index_document("doc_1", document_text)

    query = "Transformer 自注意力机制"
    print(f"\n查询: {query}")
    print("=" * 60)

    results = retriever.retrieve(query, top_k_small=5)

    print("\n最终结果:")
    for i, (context_text, source, score) in enumerate(results):
        print(f"\n--- 结果 {i+1} (得分: {score:.4f}) ---")
        print(f"来源: {source}")
        print(f"内容: {context_text[:200]}...")
```

## 11.5 Step-back Prompting

**Step-back Prompting**（后退提示）是一种通过生成更抽象的问题来提升检索质量的技术。当用户的原始问题过于具体时，直接检索可能找不到匹配的文档。Step-back Prompting 首先生成一个更通用、更抽象的问题，用这个抽象问题检索出相关背景知识，再结合原始问题进行精细化检索或直接回答。

```
原始问题: "GPT-4 在律师资格考试中的 percentile 是多少？"
              │
              ▼
后退问题: "大规模语言模型在法律领域的评估方法有哪些？"
              │
              ▼
    检索到: "LLM 评估基准概述"、"法律 AI 评测方法"
              │
              ▼
结合原始问题和检索结果，生成最终回答
```

```python
from typing import List, Dict, Tuple, Optional
import re


class StepBackPromptingRetriever:
    """
    基于 Step-back Prompting 的检索器。

    首先生成抽象问题，然后结合抽象问题和原始问题进行检索。
    """

    def __init__(
        self,
        llm_generate_backward_query: callable = None,
        retriever: object = None,
        top_k_original: int = 5,
        top_k_abstract: int = 5
    ):
        """
        参数:
            llm_generate_backward_query: 生成后退问题的函数
            retriever: 底层的检索器对象
            top_k_original: 原始问题检索数量
            top_k_abstract: 后退问题检索数量
        """
        self.llm_generate_backward_query = llm_generate_backward_query
        self.retriever = retriever
        self.top_k_original = top_k_original
        self.top_k_abstract = top_k_abstract

    @staticmethod
    def default_step_back_prompt(query: str) -> str:
        """
        默认的后退问题生成逻辑（基于模板）。

        在实际应用中，应使用 LLM 来生成更准确的后退问题。
        """
        # 规则模板匹配
        patterns = [
            # "X 在 Y 中的表现" -> "Y 相关的概念和评估方法"
            (r'(.+?)在(.+?)中的(.+?)(表现|效果|应用|性能)',
             lambda m: f"{m.group(2)}相关的{m.group(3)}方法和概念"),

            # "X 是什么原理" -> "X 的基本原理和机制"
            (r'(.+?)是什么(原理|机制|工作方式|流程)',
             lambda m: f"{m.group(1)}的基本原理和核心概念"),

            # "为什么 X" -> "X 的原因和影响因素"
            (r'为什么(.+?)(会|能|可以|需要|出现)(.+)',
             lambda m: f"{m.group(1)}{m.group(2)}{m.group(3)}的原因和影响因素"),

            # "如何实现 X" -> "X 的实现方法和步骤"
            (r'如何(实现|完成|做|进行|使用)(.+?)',
             lambda m: f"{m.group(2)}的实现方法和技术方案"),

            # "X 和 Y 的区别" -> "X 和 Y 的对比分析"
            (r'(.+?)和(.+?)的(区别|差异|不同|对比)',
             lambda m: f"{m.group(1)}和{m.group(2)}的对比分析和各自特点"),

            # "X 的最新进展" -> "X 的发展现状和趋势"
            (r'(.+?)的(最新|最近|新)(进展|发展|突破|趋势|研究)',
             lambda m: f"{m.group(1)}的发展现状和未来趋势"),
        ]

        for pattern, template_fn in patterns:
            match = re.search(pattern, query)
            if match:
                return template_fn(match)

        # 默认策略：提取核心名词短语
        # 去掉疑问词和修饰语
        query = re.sub(r'^(如何|为什么|什么是|怎样|哪个|哪些|有没有)\s*', '', query)
        # 如果查询较短，直接作为后退问题
        if len(query) < 20:
            return f"{query}的基本概念和原理"

        # 尝试提取主语和核心内容
        core_match = re.search(r'(.{2,30}?)(的|是|在|对于|关于)', query)
        if core_match:
            core = core_match.group(1)
            return f"{core}的概念、原理和应用"

        return f"{query[:30]}的概念和基本原理"

    def generate_abstract_query(self, query: str) -> str:
        """
        生成后退问题（抽象化的查询）。

        参数:
            query: 原始用户查询

        返回:
            后退问题
        """
        if self.llm_generate_backward_query:
            return self.llm_generate_backward_query(query)
        else:
            return self.default_step_back_prompt(query)

    def retrieve(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Tuple[str, str, float]]:
        """
        执行 Step-back 检索。

        参数:
            query: 原始查询
            top_k: 返回结果数

        返回:
            (文档内容, 来源标签, 得分) 列表
        """
        # 生成后退问题
        abstract_query = self.generate_abstract_query(query)
        print(f"原始查询: {query}")
        print(f"后退查询: {abstract_query}")

        # 使用后退问题检索（获取背景知识）
        abstract_results = self.retriever.search(
            abstract_query, top_k=self.top_k_abstract
        ) if hasattr(self.retriever, 'search') else []

        # 使用原始问题检索
        original_results = self.retriever.search(
            query, top_k=self.top_k_original
        ) if hasattr(self.retriever, 'search') else []

        # 合并结果（去重）
        seen_docs = set()
        merged_results = []

        # 优先保留原始问题的结果，但用后退问题的结果补充
        for doc_id, text, score in original_results:
            key = (doc_id, text[:100])
            if key not in seen_docs:
                seen_docs.add(key)
                merged_results.append((text, f"原始查询匹配", score))

        for doc_id, text, score in abstract_results:
            key = (doc_id, text[:100])
            if key not in seen_docs:
                seen_docs.add(key)
                # 给后退问题的结果一个较小的权重
                merged_results.append((text, f"后退查询匹配", score * 0.8))

        return merged_results[:top_k]


# ============================================================
# LLM 版 Step-back Prompting（使用 OpenAI API）
# ============================================================
class LLMStepBackRetriever(StepBackPromptingRetriever):
    """
    使用 LLM 生成后退问题的检索器。

    需要安装: pip install openai
    """

    def __init__(
        self,
        api_key: str = None,
        model: str = "gpt-4o-mini",
        retriever: object = None,
        **kwargs
    ):
        self.api_key = api_key
        self.model = model

        # 创建 LLM 后退问题生成函数
        llm_generator = self._create_llm_generator()

        super().__init__(
            llm_generate_backward_query=llm_generator,
            retriever=retriever,
            **kwargs
        )

    def _create_llm_generator(self) -> callable:
        """创建 LLM 后退问题生成闭包。"""
        def generate_with_llm(query: str) -> str:
            """
            使用 LLM 生成后退问题。

            提示词设计：
            你是一个查询分析助手。用户提出了一个具体的检索查询。
            请生成一个更通用、更抽象的后退问题（Step-back Question），
            该问题可以帮助检索到回答原始问题所需的背景知识。

            要求：
            - 后退问题应该更宽泛、更具概括性
            - 后退问题应该覆盖原始问题的核心主题
            - 不要包含原始问题中的具体细节
            - 只返回后退问题本身，不要任何解释
            """
            # 这里展示调用 API 的代码框架
            # 实际使用时取消注释并填入正确的 API key

            # from openai import OpenAI
            # client = OpenAI(api_key=self.api_key)
            # response = client.chat.completions.create(
            #     model=self.model,
            #     messages=[
            #         {"role": "system", "content": SYSTEM_PROMPT},
            #         {"role": "user", "content": f"原始查询: {query}"}
            #     ],
            #     temperature=0.3,
            #     max_tokens=100
            # )
            # return response.choices[0].message.content.strip()

            # 回退到默认逻辑
            return self.default_step_back_prompt(query)

        return generate_with_llm


# ============================================================
# 完整示例：Step-back Prompting 在 RAG 中的应用
# ============================================================
class RAGWithStepBack:
    """
    集成 Step-back Prompting 的完整 RAG 系统。
    """

    def __init__(
        self,
        documents: List[str],
        retriever=None,
        llm_answer_fn: callable = None
    ):
        self.documents = documents
        self.retriever = retriever or self._build_default_retriever()
        self.llm_answer_fn = llm_answer_fn or self._default_answer_fn

    def _build_default_retriever(self):
        """构建默认的检索器（简化版 BM25）。"""
        from rank_bm25 import BM25Okapi
        import jieba

        tokenized_docs = [
            list(jieba.cut(doc)) for doc in self.documents
        ]
        bm25 = BM25Okapi(tokenized_docs)

        class SimpleRetriever:
            def __init__(self, bm25, docs):
                self.bm25 = bm25
                self.docs = docs

            def search(self, query, top_k=5):
                tokens = list(jieba.cut(query))
                scores = self.bm25.get_scores(tokens)
                scored = list(enumerate(scores))
                scored.sort(key=lambda x: x[1], reverse=True)
                return [
                    (idx, self.docs[idx], score)
                    for idx, score in scored[:top_k]
                ]

        return SimpleRetriever(bm25, self.documents)

    def _default_answer_fn(self, query: str, context: str) -> str:
        """默认的答案生成函数（模拟）。"""
        return f"基于以下上下文回答 '{query}':\n{context[:200]}..."

    def generate_abstract_query(self, query: str) -> str:
        """生成后退问题。"""
        return StepBackPromptingRetriever.default_step_back_prompt(query)

    def answer(self, query: str) -> Tuple[str, List[str]]:
        """
        使用 Step-back Prompting 生成答案。

        参数:
            query: 用户查询

        返回:
            (答案, 检索到的文档列表)
        """
        # Step 1: 生成后退问题
        abstract_query = self.generate_abstract_query(query)

        # Step 2: 用后退问题检索背景知识
        abstract_docs = self.retriever.search(abstract_query, top_k=3)

        # Step 3: 用原始问题检索具体信息
        specific_docs = self.retriever.search(query, top_k=3)

        # Step 4: 合并上下文
        all_docs = []
        seen = set()
        for idx, doc, score in abstract_docs + specific_docs:
            if doc not in seen:
                seen.add(doc)
                all_docs.append(doc)

        context = "\n\n".join([
            f"[文档 {i+1}] {doc}" for i, doc in enumerate(all_docs)
        ])

        # Step 5: 生成答案
        answer = self.llm_answer_fn(query, context)

        return answer, all_docs


# ============================================================
# 运行示例
# ============================================================
if __name__ == "__main__":
    documents = [
        "Transformer 架构使用自注意力机制来处理序列数据，摒弃了传统的循环结构。",
        "多头注意力（Multi-Head Attention）允许模型从不同的表示子空间学习信息。",
        "BERT 使用掩码语言模型（MLM）进行预训练，GPT 使用自回归语言建模。",
        "自注意力机制计算序列中每个位置对其他位置的注意力权重。",
        "预训练语言模型通过在海量文本上训练来学习通用的语言表示。",
        "Fine-tuning 是在预训练模型基础上使用任务特定数据进一步训练的过程。",
        "模型评估指标包括准确率、精确率、召回率和 F1 分数。",
        "自然语言处理（NLP）是人工智能的一个重要分支，致力于让计算机理解语言。",
    ]

    rag = RAGWithStepBack(documents)

    test_queries = [
        "BERT 的预训练方式是什么原理",
        "自注意力机制的计算流程",
        "如何评估 NLP 模型的性能",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"查询: {query}")
        print(f"后退问题: {rag.generate_abstract_query(query)}")
        answer, docs = rag.answer(query)
        print(f"检索文档数: {len(docs)}")
        for i, doc in enumerate(docs):
            print(f"  文档 {i+1}: {doc}")
        print(f"答案: {answer[:100]}...")
```

## 11.6 多阶段检索（粗筛 + 精排）

多阶段检索（Multi-stage Retrieval）采用级联架构：第一级使用高效的粗筛方法快速选出候选文档，第二级使用更精确但更慢的精排方法对候选文档进行重排序。

```
                      ┌─────────────┐
                      │   用户查询   │
                      └──────┬──────┘
                             │
                      ┌──────▼──────┐
         Stage 1:     │   粗筛阶段   │  BM25 / 近似最近邻搜索
         (高效)       │  (Recall)   │  从海量文档中筛选 Top-N
                      └──────┬──────┘
                             │
                             │  Top-N 候选文档 (N=100~1000)
                             │
                      ┌──────▼──────┐
         Stage 2:     │   精排阶段   │  Cross-encoder / LLM
         (精确)       │  (Precision) │  对候选文档进行精确重排序
                      └──────┬──────┘
                             │
                             │  Top-K 最终结果 (K=5~20)
                             │
                      ┌──────▼──────┐
                      │   生成答案   │
                      └─────────────┘
```

```python
from typing import List, Dict, Tuple, Optional, Callable
import numpy as np


class MultiStageRetriever:
    """
    多阶段检索器。

    第一阶段：粗筛（使用 BM25 或向量检索，快速召回）
    第二阶段：精排（使用 Cross-encoder 或更精确的模型）
    """

    def __init__(
        self,
        stage1_retriever: Callable = None,
        stage2_ranker: Callable = None,
        stage1_top_k: int = 100,
        stage2_top_k: int = 10
    ):
        self.stage1_retriever = stage1_retriever
        self.stage2_ranker = stage2_ranker
        self.stage1_top_k = stage1_top_k
        self.stage2_top_k = stage2_top_k

    def stage1_retrieve(
        self,
        query: str,
        documents: List[str],
        top_k: int = None
    ) -> List[Tuple[int, str, float]]:
        """
        第一阶段：粗筛。

        使用 BM25 快速检索，从全部文档中选出候选文档。
        """
        if top_k is None:
            top_k = self.stage1_top_k

        top_k = min(top_k, len(documents))

        if self.stage1_retriever:
            return self.stage1_retriever(query, documents, top_k)

        # 默认使用 BM25
        from rank_bm25 import BM25Okapi
        import jieba

        tokenized_docs = [
            list(jieba.cut(doc)) for doc in documents
        ]
        bm25 = BM25Okapi(tokenized_docs)
        query_tokens = list(jieba.cut(query))
        scores = bm25.get_scores(query_tokens)

        scored = [(i, documents[i], float(scores[i]))
                  for i in range(len(documents))]
        scored.sort(key=lambda x: x[2], reverse=True)

        return scored[:top_k]

    def stage2_rerank(
        self,
        query: str,
        candidates: List[Tuple[int, str, float]],
        top_k: int = None
    ) -> List[Tuple[int, str, float]]:
        """
        第二阶段：精排。

        使用更精确的方法对候选文档进行重排序。
        """
        if top_k is None:
            top_k = self.stage2_top_k

        if self.stage2_ranker:
            return self.stage2_ranker(query, candidates, top_k)

        # 默认使用更精细的 TF-IDF（字符级 n-gram）
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        candidate_texts = [doc for _, doc, _ in candidates]
        all_texts = [query] + candidate_texts

        # 使用字符级 3-5 gram 进行更精确的语义匹配
        vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            max_features=20000,
            sublinear_tf=True
        )
        vectors = vectorizer.fit_transform(all_texts)

        query_vec = vectors[0:1]
        doc_vecs = vectors[1:]

        similarities = cosine_similarity(query_vec, doc_vecs).flatten()

        reranked = []
        for i, (orig_idx, doc, score) in enumerate(candidates):
            reranked.append((orig_idx, doc, float(similarities[i])))

        reranked.sort(key=lambda x: x[2], reverse=True)

        return reranked[:top_k]

    def retrieve(
        self,
        query: str,
        documents: List[str]
    ) -> List[Tuple[int, str, float]]:
        """
        执行多阶段检索。

        参数:
            query: 查询
            documents: 全部文档列表

        返回:
            精排后的 (索引, 文档内容, 得分) 列表
        """
        print(f"第一阶段（粗筛）: 从 {len(documents)} 篇文档中选出 top-{self.stage1_top_k}...")
        candidates = self.stage1_retrieve(query, documents, self.stage1_top_k)
        print(f"  粗筛完成: 选出 {len(candidates)} 篇候选文档")

        print(f"第二阶段（精排）: 对 {len(candidates)} 篇候选文档进行重排序...")
        results = self.stage2_rerank(query, candidates, self.stage2_top_k)
        print(f"  精排完成: 返回 top-{len(results)}")

        return results


# ============================================================
# Cross-encoder 精排器
# ============================================================
class CrossEncoderReranker:
    """
    使用 Cross-encoder 模型进行精排。

    Cross-encoder 将查询和文档拼接后输入 Transformer，
    直接输出相关性得分，比 Bi-encoder（双编码器）更精确但更慢。

    需要安装: pip install sentence-transformers
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size: int = 32,
        device: str = "cpu"
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.model = None

    def _load_model(self):
        """延迟加载 Cross-encoder 模型。"""
        if self.model is None:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(
                self.model_name,
                device=self.device
            )

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[int, str, float]],
        top_k: int = 10
    ) -> List[Tuple[int, str, float]]:
        """
        使用 Cross-encoder 对候选文档重排序。

        参数:
            query: 查询
            candidates: 候选文档列表 (index, text, score)
            top_k: 返回的 top-k 数量

        返回:
            重排序后的 (index, text, score) 列表
        """
        self._load_model()

        # 准备 Cross-encoder 的输入对
        pairs = [[query, doc] for _, doc, _ in candidates]

        # 批量预测
        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False
        )

        # 如果 scores 是二维的，取第二列（[not_relevant, relevant] 的概率）
        if scores.ndim == 2 and scores.shape[1] == 2:
            scores = scores[:, 1]

        # 按 Cross-encoder 得分重排序
        reranked = []
        for i, (orig_idx, doc, _) in enumerate(candidates):
            reranked.append((orig_idx, doc, float(scores[i])))

        reranked.sort(key=lambda x: x[2], reverse=True)

        return reranked[:top_k]


# ============================================================
# LLM 精排器
# ============================================================
class LLMReranker:
    """
    使用 LLM 进行精排。

    将查询和候选文档一起输入 LLM，让 LLM 评估文档的相关性。
    虽然最慢，但可以处理需要深度理解的复杂查询。
    """

    def __init__(
        self,
        api_key: str = None,
        model: str = "gpt-4o-mini",
        max_candidates: int = 20
    ):
        self.api_key = api_key
        self.model = model
        self.max_candidates = max_candidates

    def _create_rerank_prompt(
        self,
        query: str,
        documents: List[Tuple[int, str]]
    ) -> str:
        """创建精排提示词。"""
        prompt = f"""你是一个文档相关性评估专家。请评估以下文档与用户查询的相关性。

用户查询: {query}

请对以下每个文档的相关性进行评分（0-10分），并给出简要理由。
评分标准：
- 10分：完全相关，直接回答了查询
- 7-9分：高度相关，提供了重要信息
- 4-6分：部分相关，提供了间接信息
- 1-3分：弱相关，仅有轻微关联
- 0分：完全不相关

文档列表：
"""
        for idx, doc in documents:
            prompt += f"\n[文档 {idx}] {doc[:200]}"

        prompt += """
请按以下格式输出（每行一个文档）：
文档ID | 评分 | 理由
"""
        return prompt

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[int, str, float]],
        top_k: int = 10
    ) -> List[Tuple[int, str, float]]:
        """
        使用 LLM 对候选文档重排序。

        参数:
            query: 查询
            candidates: 候选文档列表
            top_k: 返回数量

        返回:
            重排序后的结果列表
        """
        # 限制候选数量（LLM 上下文限制）
        limited_candidates = candidates[:self.max_candidates]

        # 创建提示词（实际调用 LLM API）
        prompt = self._create_rerank_prompt(query, [
            (idx, doc) for idx, doc, _ in limited_candidates
        ])

        # 模拟 LLM 评分（实际使用时替换为 API 调用）
        llm_scores = []
        for idx, doc, _ in limited_candidates:
            # 模拟 LLM 判断：计算查询和文档的重叠词数
            query_words = set(query.lower().split())
            doc_words = set(doc.lower().split())
            overlap = len(query_words & doc_words)
            score = min(10.0, overlap * 2.5 + 2.0)
            llm_scores.append((idx, doc, score))

        llm_scores.sort(key=lambda x: x[2], reverse=True)

        return llm_scores[:top_k]


# ============================================================
# 多阶段检索评估
# ============================================================
class MultiStageEvaluator:
    """
    多阶段检索的评估器。
    对比单阶段和两阶段检索的效果。
    """

    def __init__(self, documents: List[str], ground_truth: Dict[str, List[int]]):
        """
        参数:
            documents: 文档列表
            ground_truth: {查询: [相关文档索引列表]}
        """
        self.documents = documents
        self.ground_truth = ground_truth

    def evaluate(
        self,
        retriever: MultiStageRetriever,
        queries: List[str]
    ) -> Dict:
        """
        评估检索器的性能。

        返回:
            包含各指标的字典
        """
        total_recall_1 = 0.0
        total_recall_3 = 0.0
        total_recall_5 = 0.0
        total_mrr = 0.0

        num_queries = len(queries)

        for query in queries:
            relevant = set(self.ground_truth.get(query, []))

            if not relevant:
                continue

            results = retriever.retrieve(query, self.documents)
            retrieved_indices = [idx for idx, _, _ in results]

            # Recall@k
            retrieved_1 = set(retrieved_indices[:1])
            retrieved_3 = set(retrieved_indices[:3])
            retrieved_5 = set(retrieved_indices[:5])

            total_recall_1 += len(retrieved_1 & relevant) / len(relevant)
            total_recall_3 += len(retrieved_3 & relevant) / len(relevant)
            total_recall_5 += len(retrieved_5 & relevant) / len(relevant)

            # MRR (Mean Reciprocal Rank)
            for rank, idx in enumerate(retrieved_indices):
                if idx in relevant:
                    total_mrr += 1.0 / (rank + 1)
                    break

        return {
            "recall@1": total_recall_1 / num_queries,
            "recall@3": total_recall_3 / num_queries,
            "recall@5": total_recall_5 / num_queries,
            "mrr": total_mrr / num_queries,
            "num_queries": num_queries
        }


# ============================================================
# 运行示例
# ============================================================
if __name__ == "__main__":
    documents = [
        "混合检索结合了稀疏检索和密集检索的优点，在多种场景下表现优异。",
        "BM25 是一种基于概率检索框架的排序函数，广泛用于信息检索。",
        "向量嵌入将文本映射到高维空间，语义相近的文本在空间中距离更近。",
        "RRF（倒数排名融合）通过排名而非得分来融合多路检索结果。",
        "Cross-encoder 将查询和文档拼接后输入 Transformer，直接预测相关性。",
        "Bi-encoder 分别编码查询和文档，使用余弦相似度计算相关性。",
        "近似最近邻搜索（ANN）通过牺牲少量精度来大幅提升检索速度。",
        "倒排索引是全文搜索引擎的核心数据结构，支持高效的词项查找。",
        "HNSW 是一种基于分层图的 ANN 算法，在速度和精度之间取得良好平衡。",
        "乘积量化（PQ）通过将向量分解为子空间并量化来压缩向量。",
        "查询重写通过改写原始查询来提升检索质量。",
        "HyDE（假设文档嵌入）先生成假设文档，再用其嵌入进行检索。",
    ]

    ground_truth = {
        "混合检索的融合方法": [0, 3],
        "BM25 排序原理": [1],
        "向量嵌入和语义搜索": [2],
        "Cross-encoder 和 Bi-encoder 的区别": [4, 5],
    }

    print("=== 多阶段检索示例 ===")
    print(f"文档库大小: {len(documents)} 篇")

    retriever = MultiStageRetriever(
        stage1_top_k=5,   # 第一段粗筛取 top-5
        stage2_top_k=3    # 第二段精排取 top-3
    )

    test_query = "向量嵌入的语义搜索原理"
    print(f"\n查询: {test_query}")

    results = retriever.retrieve(test_query, documents)

    print(f"\n最终结果:")
    for i, (idx, doc, score) in enumerate(results):
        print(f"  #{i+1} [文档 {idx}] (得分: {score:.4f}) {doc}")

    # 对比评估
    print(f"\n{'='*60}")
    print("评估对比: 单阶段 vs 两阶段")
    print(f"{'='*60}")

    # 单阶段检索
    single_stage = MultiStageRetriever(
        stage1_top_k=10,
        stage2_top_k=3
    )

    evaluator = MultiStageEvaluator(documents, ground_truth)
    metrics = evaluator.evaluate(single_stage, list(ground_truth.keys()))

    print(f"单阶段检索结果:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
```

## 11.7 查询重写技术

查询重写（Query Rewriting）是提升检索质量的重要手段。用户提出的原始查询往往不够精确或完整，通过重写可以将其转化为更适合检索的形式。

### 11.7.1 查询重写方法概览

```python
from typing import List, Dict, Tuple, Optional, Callable
import re


class QueryRewriter:
    """
    查询重写器，支持多种重写策略。
    """

    # 同义词映射（示例）
    SYNONYM_MAP = {
        "AI": ["人工智能", "artificial intelligence"],
        "ML": ["机器学习", "machine learning"],
        "NLP": ["自然语言处理", "natural language processing"],
        "DL": ["深度学习", "deep learning"],
        "LLM": ["大语言模型", "large language model"],
        "CNN": ["卷积神经网络", "convolutional neural network"],
        "RNN": ["循环神经网络", "recurrent neural network"],
        "GNN": ["图神经网络", "graph neural network"],
        "GAN": ["生成对抗网络", "generative adversarial network"],
        "API": ["应用程序接口", "application programming interface"],
        "DB": ["数据库", "database"],
        "GPU": ["图形处理器", "graphics processing unit"],
    }

    def __init__(self, llm_rewriter: Callable = None):
        self.llm_rewriter = llm_rewriter

    # ----- 策略1: 同义词扩展 -----
    def synonym_expansion(self, query: str) -> List[str]:
        """
        同义词扩展：将查询中的缩写和术语替换为同义词。

        返回多个查询变体。
        """
        expanded_queries = [query]

        # 对每个缩写生成一个扩展版本
        for abbrev, expansions in self.SYNONYM_MAP.items():
            if abbrev in query:
                for expansion in expansions:
                    new_query = query.replace(abbrev, expansion)
                    if new_query != query:
                        expanded_queries.append(new_query)

        return expanded_queries

    # ----- 策略2: 查询分解 -----
    def query_decomposition(self, query: str) -> List[str]:
        """
        查询分解：将复杂查询分解为多个子查询。

        例如："深度学习和自然语言处理的最新进展"
        -> ["深度学习最新进展", "自然语言处理最新进展"]
        """
        # 检测连接词
        conjunctions = ['和', '与', '及', '以及', '、', ',', '，']

        sub_queries = [query]

        for conj in conjunctions:
            if conj in query:
                parts = [p.strip() for p in query.split(conj)]

                # 尝试将每个部分作为独立查询
                if len(parts) >= 2:
                    # 提取公共后缀
                    common_suffix = self._extract_common_suffix(parts)

                    sub_queries = []
                    for part in parts:
                        if common_suffix:
                            sub_q = part + common_suffix
                        else:
                            sub_q = part
                        sub_queries.append(sub_q)

                    break

        return sub_queries

    @staticmethod
    def _extract_common_suffix(parts: List[str]) -> str:
        """提取多个部分的公共后缀。"""
        if not parts:
            return ""

        # 取最后一部分的后半段作为可能的公共后缀
        suffixes = []
        for part in parts:
            words = part.split()
            if len(words) >= 2:
                suffixes.append(' '.join(words[-2:]))

        if not suffixes:
            return ""

        # 检查是否所有部分的最后几个词相同
        common = suffixes[0]
        for s in suffixes[1:]:
            while common and not s.endswith(common):
                common = common[1:]

        return common if common else ""

    # ----- 策略3: 查询扩展 -----
    def query_expansion(
        self,
        query: str,
        expansion_terms: List[str] = None
    ) -> List[str]:
        """
        查询扩展：在原始查询基础上添加相关术语。

        参数:
            query: 原始查询
            expansion_terms: 要添加的扩展词列表

        返回:
            扩展后的查询列表
        """
        if expansion_terms is None:
            # 默认扩展：添加"概念"、"原理"、"应用"等通用词
            expansion_terms = ["概念", "原理", "方法", "应用", "技术"]

        expanded = [query]

        for term in expansion_terms:
            expanded.append(f"{query} {term}")

        return expanded

    # ----- 策略4: 查询压缩 -----
    def query_compression(self, query: str, max_words: int = 5) -> str:
        """
        查询压缩：移除停用词和非核心词，保留关键词。

        例如："我想了解一下深度学习的基本原理是什么"
        -> "深度学习 基本原理"
        """
        # 停用词列表
        stopwords = {
            '我', '你', '他', '她', '它', '我们', '你们', '他们',
            '的', '了', '在', '是', '有', '和', '与', '就', '也',
            '还', '都', '要', '可以', '能', '会', '想', '让', '被',
            '把', '被', '从', '到', '对', '为', '为了', '因为', '所以',
            '但是', '然而', '如果', '虽然', '这个', '那个', '什么', '怎么',
            '如何', '为什么', '哪些', '哪个', '有没有', '是否', '吗', '呢',
            '吧', '啊', '呀', '哦', '嗯', '哈', '嘿',
            '请问', '请教', '求教', '求助', '问一下', '了解一下',
            'about', 'what', 'how', 'why', 'where', 'when', 'which',
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'can', 'could', 'should', 'may', 'might', 'shall',
            'i', 'you', 'he', 'she', 'it', 'we', 'they',
            'this', 'that', 'these', 'those',
        }

        # 分词并过滤
        words = query.split()
        keywords = [w for w in words if w not in stopwords and len(w) > 1]

        # 如果过滤后太少，保留原始内容
        if len(keywords) < 2:
            return query

        return ' '.join(keywords[:max_words])

    # ----- 策略5: 查询改写（LLM） -----
    def llm_rewrite(self, query: str, instruction: str = None) -> str:
        """
        使用 LLM 改写查询。

        参数:
            query: 原始查询
            instruction: 改写指令

        返回:
            改写后的查询
        """
        if self.llm_rewriter:
            return self.llm_rewriter(query, instruction)

        # 默认改写：如果查询是疑问句，转换为陈述句
        question_words = ['什么', '怎么', '如何', '为什么', '哪些', '哪个']

        for qw in question_words:
            if qw in query:
                # 移除疑问词
                rewritten = query.replace(f"是{qw}", "")
                rewritten = rewritten.replace(qw, "")
                rewritten = rewritten.strip().rstrip('？?')
                return rewritten

        return query

    # ----- 综合重写 -----
    def rewrite(
        self,
        query: str,
        strategies: List[str] = None
    ) -> List[str]:
        """
        综合使用多种重写策略，生成查询变体。

        参数:
            query: 原始查询
            strategies: 要使用的策略列表

        返回:
            所有生成的查询变体
        """
        if strategies is None:
            strategies = ["original", "compressed", "expanded", "decomposed"]

        all_queries = set()
        all_queries.add(query)

        for strategy in strategies:
            if strategy == "original":
                continue

            elif strategy == "compressed":
                compressed = self.query_compression(query)
                all_queries.add(compressed)

            elif strategy == "expanded":
                expanded_list = self.query_expansion(query)
                all_queries.update(expanded_list)

            elif strategy == "decomposed":
                decomposed_list = self.query_decomposition(query)
                all_queries.update(decomposed_list)

            elif strategy == "synonym":
                synonym_list = self.synonym_expansion(query)
                all_queries.update(synonym_list)

            elif strategy == "llm":
                rewritten = self.llm_rewrite(query)
                all_queries.add(rewritten)

        return list(all_queries)


# ============================================================
# 多查询检索（Multi-Query Retrieval）
# ============================================================
class MultiQueryRetriever:
    """
    多查询检索器。

    生成多个查询变体，分别检索后合并结果。
    """

    def __init__(
        self,
        query_rewriter: QueryRewriter = None,
        retriever: object = None,
        top_k_per_query: int = 5,
        final_top_k: int = 10
    ):
        self.query_rewriter = query_rewriter or QueryRewriter()
        self.retriever = retriever
        self.top_k_per_query = top_k_per_query
        self.final_top_k = final_top_k

    def retrieve(
        self,
        query: str,
        documents: List[str]
    ) -> List[Tuple[int, str, float]]:
        """
        执行多查询检索。

        参数:
            query: 原始查询
            documents: 文档列表

        返回:
            合并后的 (索引, 文档内容, 聚合得分) 列表
        """
        # Step 1: 生成查询变体
        query_variants = self.query_rewriter.rewrite(query)
        print(f"原始查询: {query}")
        print(f"查询变体 ({len(query_variants)} 个):")
        for i, qv in enumerate(query_variants):
            print(f"  [{i+1}] {qv}")

        # Step 2: 对每个查询变体进行检索
        all_results = []  # (doc_idx, doc_text, query_idx, score)

        for q_idx, qv in enumerate(query_variants):
            if self.retriever:
                results = self.retriever.search(qv, top_k=self.top_k_per_query)
            else:
                # 使用简单的 BM25
                from rank_bm25 import BM25Okapi
                import jieba

                tokenized_docs = [
                    list(jieba.cut(doc)) for doc in documents
                ]
                bm25 = BM25Okapi(tokenized_docs)
                tokens = list(jieba.cut(qv))
                scores = bm25.get_scores(tokens)

                scored = [(i, documents[i], float(scores[i]))
                          for i in range(len(documents))]
                scored.sort(key=lambda x: x[2], reverse=True)
                results = scored[:self.top_k_per_query]

            for idx, doc, score in results:
                all_results.append((idx, doc, q_idx, score))

        # Step 3: 合并结果
        # 使用 RRF 融合多查询结果
        doc_votes = {}  # doc_idx -> [(query_idx, rank)]
        for rank, (idx, doc, q_idx, _) in enumerate(all_results):
            if idx not in doc_votes:
                doc_votes[idx] = []
            doc_votes[idx].append((q_idx, rank))

        # 计算 RRF 得分
        rrf_scores = {}
        k_rrf = 60

        for idx, votes in doc_votes.items():
            score = 0.0
            for q_idx, rank in votes:
                score += 1.0 / (k_rrf + rank + 1)
            rrf_scores[idx] = score

        # 排序
        sorted_results = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:self.final_top_k]

        return [
            (idx, documents[idx], score)
            for idx, score in sorted_results
        ]


# ============================================================
# 运行示例
# ============================================================
if __name__ == "__main__":
    documents = [
        "深度学习使用多层神经网络来学习数据的层次化表示。",
        "卷积神经网络（CNN）在图像识别任务中表现优异。",
        "循环神经网络（RNN）擅长处理文本和时间序列数据。",
        "Transformer 架构使用自注意力机制，在 NLP 任务中取得了突破。",
        "BERT 通过掩码语言模型进行预训练，在下游任务上微调。",
        "GPT 使用自回归方式生成文本，在零样本学习上表现出色。",
        "图神经网络（GNN）用于处理图结构数据，如社交网络和分子结构。",
        "生成对抗网络（GAN）由生成器和判别器组成，用于生成真实数据。",
        "强化学习通过与环境交互来学习最优策略。",
        "迁移学习将预训练模型的知识迁移到目标任务上。",
    ]

    rewriter = QueryRewriter()
    retriever = MultiQueryRetriever(
        query_rewriter=rewriter,
        top_k_per_query=3,
        final_top_k=5
    )

    test_query = "CNN 和 RNN 的应用区别"

    print("=" * 60)
    print("多查询检索示例")
    print("=" * 60)

    results = retriever.retrieve(test_query, documents)

    print(f"\n最终结果:")
    for i, (idx, doc, score) in enumerate(results):
        print(f"  #{i+1} [文档 {idx}] (得分: {score:.4f}) {doc}")
```

### 11.7.2 查询重写的综合应用

```python
class QueryRewritePipeline:
    """
    查询重写流水线，按顺序应用多种重写策略。
    """

    def __init__(self):
        self.rewriter = QueryRewriter()

    def process(
        self,
        query: str,
        mode: str = "auto"
    ) -> Dict[str, object]:
        """
        对查询进行全面分析并应用重写策略。

        参数:
            query: 原始查询
            mode: 处理模式
                - "auto": 自动选择策略
                - "aggressive": 应用所有策略
                - "conservative": 仅应用安全的策略

        返回:
            包含所有重写结果的字典
        """
        result = {
            "original": query,
            "analysis": self._analyze_query(query),
            "variants": {},
            "recommended": None
        }

        # 同义词扩展
        synonyms = self.rewriter.synonym_expansion(query)
        result["variants"]["synonym"] = synonyms

        # 查询分解
        decomposed = self.rewriter.query_decomposition(query)
        result["variants"]["decomposed"] = decomposed

        # 查询压缩
        compressed = self.rewriter.query_compression(query)
        result["variants"]["compressed"] = compressed

        # 查询扩展
        expanded = self.rewriter.query_expansion(query)
        result["variants"]["expanded"] = expanded

        # 推荐策略
        result["recommended"] = self._recommend_strategy(
            result["analysis"], mode
        )

        return result

    def _analyze_query(self, query: str) -> Dict:
        """
        分析查询特征。
        """
        analysis = {
            "length": len(query),
            "word_count": len(query.split()),
            "has_abbreviation": bool(re.search(
                r'\b[A-Z]{2,}\b', query
            )),
            "has_question_word": bool(re.search(
                r'什么|怎么|如何|为什么|哪些|哪个', query
            )),
            "has_conjunction": bool(re.search(
                r'和|与|及|以及|、', query
            )),
            "has_specific_term": bool(re.search(
                r'实现|代码|API|版本|配置|安装|部署', query
            )),
            "is_complex": len(query.split()) >= 8,
            "is_short": len(query) < 10,
        }

        # 查询类型分类
        if analysis["has_question_word"]:
            analysis["type"] = "question"
        elif analysis["has_conjunction"]:
            analysis["type"] = "comparison"
        elif analysis["has_specific_term"]:
            analysis["type"] = "specific"
        elif analysis["is_short"]:
            analysis["type"] = "keyword"
        else:
            analysis["type"] = "statement"

        return analysis

    def _recommend_strategy(
        self,
        analysis: Dict,
        mode: str
    ) -> List[str]:
        """
        根据查询分析结果推荐重写策略。

        参数:
            analysis: 查询分析结果
            mode: 处理模式

        返回:
            推荐使用的策略列表
        """
        if mode == "aggressive":
            return ["original", "compressed", "expanded",
                    "decomposed", "synonym"]

        if mode == "conservative":
            return ["original", "compressed"]

        # auto 模式
        strategies = ["original"]

        # 长查询需要压缩
        if analysis["is_complex"]:
            strategies.append("compressed")

        # 包含缩写需要同义词扩展
        if analysis["has_abbreviation"]:
            strategies.append("synonym")

        # 复合查询需要分解
        if analysis["has_conjunction"]:
            strategies.append("decomposed")

        # 短查询需要扩展
        if analysis["is_short"]:
            strategies.append("expanded")

        return strategies


# ============================================================
# 演示
# ============================================================
def demonstrate_query_rewriting():
    """演示查询重写的各种技术和效果。"""
    pipeline = QueryRewritePipeline()

    test_queries = [
        "CNN 和 RNN 的区别是什么",
        "深度学习",
        "如何使用 Python 实现 BERT 模型的 Fine-tuning",
        "GPU 在深度学习训练中的作用",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"原始查询: {query}")
        print(f"{'='*60}")

        result = pipeline.process(query, mode="auto")

        print(f"\n查询分析:")
        analysis = result["analysis"]
        print(f"  类型: {analysis['type']}")
        print(f"  长度: {analysis['length']} 字符, {analysis['word_count']} 词")
        print(f"  包含缩写: {analysis['has_abbreviation']}")
        print(f"  包含疑问词: {analysis['has_question_word']}")
        print(f"  包含连接词: {analysis['has_conjunction']}")

        print(f"\n推荐策略: {result['recommended']}")

        print(f"\n查询变体:")
        for strategy, variants in result["variants"].items():
            print(f"  [{strategy}]")
            for v in variants:
                print(f"    - {v}")


if __name__ == "__main__":
    demonstrate_query_rewriting()
```

## 11.8 综合案例：完整的混合检索系统

下面我们将所有技术整合为一个完整的混合检索系统：

```python
class CompleteHybridSearchSystem:
    """
    完整的混合检索系统，整合了本章介绍的所有技术：

    1. 稠密 + 稀疏融合（RRF + 加权和）
    2. 自适应权重调整
    3. Small-to-Big 检索
    4. Step-back Prompting
    5. 多阶段检索（粗筛 + 精排）
    6. 查询重写
    """

    def __init__(
        self,
        documents: List[str] = None,
        # 检索配置
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        adaptive_weight: bool = True,
        fusion_method: str = "rrf",
        # 多阶段配置
        use_multi_stage: bool = True,
        stage1_top_k: int = 50,
        stage2_top_k: int = 10,
        # Small-to-Big 配置
        use_small_to_big: bool = False,
        # Step-back 配置
        use_step_back: bool = True,
        # 查询重写配置
        use_query_rewrite: bool = True,
        rewrite_strategies: List[str] = None
    ):
        self.documents = documents or []
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.adaptive_weight = adaptive_weight
        self.fusion_method = fusion_method
        self.use_multi_stage = use_multi_stage
        self.stage1_top_k = stage1_top_k
        self.stage2_top_k = stage2_top_k
        self.use_small_to_big = use_small_to_big
        self.use_step_back = use_step_back
        self.use_query_rewrite = use_query_rewrite
        self.rewrite_strategies = rewrite_strategies or [
            "original", "compressed", "expanded"
        ]

        # 子组件
        self.query_rewriter = QueryRewriter()
        self.feature_extractor = QueryFeatureExtractor()
        self.step_back = StepBackPromptingRetriever()

        # 检索缓存
        self._dense_model = None
        self._bm25 = None
        self._embeddings = None

    def _build_index(self):
        """构建检索索引。"""
        if self._bm25 is not None:
            return

        from rank_bm25 import BM25Okapi
        import jieba

        # BM25 索引
        tokenized_docs = [
            list(jieba.cut(doc)) for doc in self.documents
        ]
        self._bm25 = BM25Okapi(tokenized_docs)

        # 向量索引（简化版，使用 TF-IDF 字符 n-gram 模拟）
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            max_features=10000,
            sublinear_tf=True
        )
        self._embeddings = self._vectorizer.fit_transform(self.documents)

    def _dense_search(
        self,
        query: str,
        top_k: int
    ) -> List[Tuple[int, float]]:
        """密集检索。"""
        self._build_index()
        query_vec = self._vectorizer.transform([query])
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(
            query_vec, self._embeddings
        ).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [(int(i), float(similarities[i])) for i in top_indices]

    def _sparse_search(
        self,
        query: str,
        top_k: int
    ) -> List[Tuple[int, float]]:
        """稀疏检索（BM25）。"""
        self._build_index()
        import jieba
        query_tokens = list(jieba.cut(query))
        scores = self._bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_indices]

    def _compute_weights(self, query: str) -> Tuple[float, float]:
        """计算自适应权重。"""
        if not self.adaptive_weight:
            return self.dense_weight, self.sparse_weight

        features = self.feature_extractor.extract_features(query)
        semantic_score = self.feature_extractor.is_semantic_query(features)

        # 语义倾向越高，密集检索权重越大
        dense_w = self.dense_weight + (semantic_score - 0.5) * 0.6
        dense_w = max(0.2, min(0.8, dense_w))
        sparse_w = 1.0 - dense_w

        return dense_w, sparse_w

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        执行完整的多策略混合检索。

        参数:
            query: 用户查询
            top_k: 返回结果数

        返回:
            检索结果列表
        """
        print(f"\n{'='*60}")
        print(f"混合检索系统")
        print(f"{'='*60}")
        print(f"查询: {query}")

        # ===== Step 1: 查询重写 =====
        query_variants = [query]
        if self.use_query_rewrite:
            variants = self.query_rewriter.rewrite(
                query, strategies=self.rewrite_strategies
            )
            query_variants = variants
            print(f"\n[查询重写] 生成 {len(variants)} 个查询变体")

        # ===== Step 2: Step-back 生成 =====
        step_back_query = None
        if self.use_step_back:
            step_back_query = self.step_back.generate_abstract_query(query)
            query_variants.append(step_back_query)
            print(f"[Step-back] 后退查询: {step_back_query}")

        # ===== Step 3: 对每个查询变体执行检索 =====
        all_results = []  # [(doc_idx, score, source_query_idx)]

        for q_idx, qv in enumerate(query_variants):
            # 计算自适应权重
            dense_w, sparse_w = self._compute_weights(qv)

            # 密集检索
            dense_results = self._dense_search(qv, self.stage1_top_k)

            # 稀疏检索
            sparse_results = self._sparse_search(qv, self.stage1_top_k)

            # 融合
            if self.fusion_method == "rrf":
                dense_rank = [str(idx) for idx, _ in dense_results]
                sparse_rank = [str(idx) for idx, _ in sparse_results]
                fused = reciprocal_rank_fusion(
                    [dense_rank, sparse_rank], k=60
                )
                for doc_id_str, score in fused:
                    all_results.append((int(doc_id_str), score, q_idx))
            else:
                # 加权和
                dense_dict = dict(dense_results)
                sparse_dict = dict(sparse_results)
                fused = weighted_sum_fusion(
                    [dense_dict, sparse_dict],
                    weights=[dense_w, sparse_w],
                    normalization="minmax"
                )
                for doc_id, score in fused:
                    all_results.append((doc_id, score, q_idx))

        # ===== Step 4: 多查询结果融合 =====
        doc_agg = {}
        for doc_idx, score, q_idx in all_results:
            if doc_idx not in doc_agg:
                doc_agg[doc_idx] = {"score": 0.0, "count": 0, "sources": set()}
            doc_agg[doc_idx]["score"] += score
            doc_agg[doc_idx]["count"] += 1
            doc_agg[doc_idx]["sources"].add(q_idx)

        # 归一化并排序
        for doc_idx in doc_agg:
            doc_agg[doc_idx]["score"] /= doc_agg[doc_idx]["count"]

        sorted_docs = sorted(
            doc_agg.items(),
            key=lambda x: x[1]["score"],
            reverse=True
        )[:top_k]

        # ===== Step 5: 构建结果 =====
        results = []
        for doc_idx, info in sorted_docs:
            results.append({
                "index": doc_idx,
                "text": self.documents[doc_idx],
                "score": info["score"],
                "match_count": info["count"],
                "source_variants": info["sources"]
            })

        return results


# ============================================================
# 系统演示
# ============================================================
def demonstrate_complete_system():
    """演示完整混合检索系统。"""
    documents = [
        "混合检索结合了稀疏检索和密集检索的优点，能够提升检索的召回率和精确率。",
        "RRF（倒数排名融合）通过排名信息融合多路检索结果，不需要得分可比较。",
        "加权求和融合需要对各路得分进行归一化处理，使其在相同尺度上可比。",
        "自适应权重根据查询特征动态调整密集和稀疏检索的权重比例。",
        "Small-to-Big 检索在小块级别检索，扩展到段落级别作为上下文。",
        "Step-back Prompting 通过生成更抽象的后退问题来获取背景知识。",
        "多阶段检索使用粗筛加精排的级联架构，兼顾效率和精度。",
        "查询重写通过同义词扩展、分解、压缩等策略改进原始查询。",
        "Cross-encoder 比 Bi-encoder 更精确但更慢，适合精排阶段。",
        "BM25 是一种基于词频和逆文档频率的经典排序算法。",
        "向量嵌入通过将文本映射到高维空间来捕捉语义相似度。",
        "HNSW 算法通过分层图结构实现高效的近似最近邻搜索。",
    ]

    system = CompleteHybridSearchSystem(
        documents=documents,
        adaptive_weight=True,
        fusion_method="rrf",
        use_multi_stage=True,
        use_step_back=True,
        use_query_rewrite=True,
        stage1_top_k=10,
        stage2_top_k=5
    )

    test_queries = [
        "如何融合密集和稀疏检索的结果",
        "提升检索质量的策略有哪些",
    ]

    for query in test_queries:
        results = system.search(query, top_k=3)
        print(f"\n检索结果:")
        for i, r in enumerate(results):
            print(f"  #{i+1} [文档 {r['index']}] "
                  f"(得分: {r['score']:.4f}, "
                  f"匹配: {r['match_count']}个查询变体)")
            print(f"     {r['text']}")


if __name__ == "__main__":
    demonstrate_complete_system()
```

## 11.9 本章小结

本章深入介绍了混合检索的各种核心技术，这些技术在实际 RAG 系统中扮演着至关重要的角色。以下是关键要点的总结：

| 技术 | 核心思想 | 适用场景 | 优势 |
|------|---------|---------|------|
| **稠密-稀疏融合** | 融合语义匹配和关键词匹配 | 通用场景 | 兼顾语义理解与精确匹配 |
| **RRF** | 基于排名而非得分融合 | 多路异构检索器 | 无需得分归一化，鲁棒性强 |
| **加权求和** | 归一化后按权重融合得分 | 同构检索器 | 可精细控制各路贡献 |
| **自适应权重** | 根据查询特征动态调权 | 查询类型多样 | 查询自适应，效果更优 |
| **Small-to-Big** | 小块检索 + 大块上下文 | 长文档检索 | 精确定位 + 充分上下文 |
| **Step-back** | 抽象问题获取背景知识 | 具体/复杂问题 | 提升知识覆盖面 |
| **多阶段检索** | 粗筛 + 精排级联 | 大规模文档库 | 效率与精度的平衡 |
| **查询重写** | 多种策略优化原始查询 | 查询质量参差不齐 | 提升检索上限 |

### 实践建议

1. **优先使用 RRF 融合**：RRF 简单、鲁棒、无需训练，适合大多数场景的基线方案。
2. **根据查询类型调优权重**：精确查询（如版本号、产品名）应加大稀疏检索权重；语义查询（如概念理解、对比分析）应加大密集检索权重。
3. **多阶段检索是工程标配**：在规模化部署中，第一段使用高效的 ANN 或 BM25，第二段使用 Cross-encoder 或小模型精排。
4. **查询重写成本低收益高**：查询重写不需要额外模型部署，通过规则即可显著提升检索质量。
5. **Step-back 适合复杂推理**：当问题涉及深层推理时，Step-back 提供关键背景知识补充。
6. **Small-to-Big 解决块大小矛盾**：小块提高检索精度，大块提供完整上下文，二者兼得。

混合检索不是单一技术，而是一套技术组合。实际系统中应根据数据特点、查询分布和性能要求，灵活选择和组合上述技术，才能在召回率、精确率和推理质量之间取得最佳平衡。
