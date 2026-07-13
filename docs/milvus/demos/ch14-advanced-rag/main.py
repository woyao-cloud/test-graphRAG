"""
ch14-advanced-rag: Advanced RAG Architecture (stdlib-only)
Demonstrates multi-stage retrieval, query decomposition, multi-turn context, and hybrid fusion.
All pure Python, no external dependencies.
"""

import hashlib
import math
import re
from collections import Counter

# ── Sample Document Corpus ─────────────────────────────────────────────────────

DOCUMENTS = [
    {"id": "doc1", "title": "机器学习基础", "content": "机器学习是人工智能的核心领域。监督学习使用标注数据进行训练。无监督学习发现数据中的隐藏模式。强化学习通过与环境交互来学习。"},
    {"id": "doc2", "title": "深度学习简介", "content": "深度学习使用多层神经网络。卷积神经网络（CNN）擅长图像处理。循环神经网络（RNN）适合序列数据。Transformer架构推动了NLP的发展。"},
    {"id": "doc3", "title": "自然语言处理", "content": "NLP技术包括分词、词性标注和命名实体识别。词嵌入将词语映射为向量。BERT和GPT是重要的预训练语言模型。机器翻译是NLP的典型应用。"},
    {"id": "doc4", "title": "计算机视觉", "content": "计算机视觉使机器能够理解图像和视频。目标检测识别图像中的物体。图像分割将图像划分为语义区域。人脸识别是广泛应用的视觉技术。"},
    {"id": "doc5", "title": "强化学习", "content": "强化学习通过奖励信号训练智能体。Q学习是一种经典的无模型算法。深度Q网络（DQN）结合了深度学习和Q学习。策略梯度方法直接优化策略。"},
    {"id": "doc6", "title": "监督学习算法", "content": "线性回归预测连续值。逻辑回归用于二分类。决策树基于特征进行分裂。支持向量机寻找最优超平面。随机森林集成多个决策树。"},
    {"id": "doc7", "title": "神经网络架构", "content": "前馈神经网络是最基本的架构。残差网络（ResNet）解决了梯度消失问题。注意力机制让模型关注重要部分。生成对抗网络（GAN）用于生成新数据。"},
    {"id": "doc8", "title": "模型评估方法", "content": "交叉验证评估模型泛化能力。混淆矩阵展示分类结果。精确率和召回率衡量分类性能。F1分数是精确率和召回率的调和平均。ROC曲线评估二分类器。"},
]

# ── Embedding Simulation ────────────────────────────────────────────────────────

DIM = 16


def _hash_vector(text: str, dim: int = DIM) -> list[float]:
    """Generate a deterministic pseudo-embedding from text."""
    h = hashlib.md5(text.encode()).hexdigest()
    return [((int(h[i : i + 2], 16) / 255.0) * 2 - 1) for i in range(0, dim * 2, 2)]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# Precompute document embeddings
DOC_EMBEDDINGS = {doc["id"]: _hash_vector(doc["title"] + " " + doc["content"]) for doc in DOCUMENTS}

# ── 1. Multi-Stage Retrieval ────────────────────────────────────────────────────


def stage1_coarse_search(query: str, top_k: int = 20) -> list[dict]:
    """Coarse retrieval: return top-k documents by cosine similarity."""
    q_vec = _hash_vector(query)
    scored = []
    for doc in DOCUMENTS:
        sim = cosine_similarity(q_vec, DOC_EMBEDDINGS[doc["id"]])
        scored.append({"id": doc["id"], "title": doc["title"], "content": doc["content"], "score": sim})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def stage2_fine_rerank(candidates: list[dict], query: str, top_k: int = 5) -> list[dict]:
    """
    Fine reranking: re-score candidates using keyword overlap + position bonus.
    """
    query_tokens = set(query.lower())

    def rerank_score(item: dict) -> float:
        vec_score = item["score"]
        text = (item["title"] + " " + item["content"]).lower()
        token_overlap = sum(1 for t in query_tokens if t in text)
        keyword_score = token_overlap / max(len(query_tokens), 1)
        # Combine: 60% vector + 40% keyword
        return 0.6 * vec_score + 0.4 * keyword_score

    for item in candidates:
        item["rerank_score"] = rerank_score(item)

    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    return candidates[:top_k]


# ── 2. Query Decomposition ──────────────────────────────────────────────────────


def decompose_query(query: str) -> list[str]:
    """Break a complex question into simpler sub-questions."""
    sub_questions = []

    # Split on conjunctions and question markers
    parts = re.split(r"(?:和|与|以及|、|,|，|what|how|why|explain|compare)", query)
    parts = [p.strip() for p in parts if p.strip()]

    if len(parts) > 1:
        for p in parts:
            sub_questions.append(f"Explain {p}" if len(p) < 50 else p)
    else:
        # Single question: break by topic hints
        topics = re.findall(r"(CNN|RNN|Transformer|BERT|GPT|ResNet|GAN|DQN)", query)
        if topics:
            sub_questions = [f"What is {t}" for t in topics]
        else:
            sub_questions = [query]

    return sub_questions


# ── 3. Multi-Turn Context Management ────────────────────────────────────────────


class ConversationManager:
    """Maintains conversation history for multi-turn RAG."""

    def __init__(self, max_turns: int = 5):
        self.history: list[dict] = []
        self.max_turns = max_turns

    def add_turn(self, query: str, response: str, retrieved_docs: list[str]):
        self.history.append({"query": query, "response": response, "retrieved_docs": retrieved_docs})
        if len(self.history) > self.max_turns:
            self.history.pop(0)

    def get_context(self) -> str:
        """Format history as context string."""
        if not self.history:
            return ""
        lines = ["Previous conversation:"]
        for i, turn in enumerate(self.history, 1):
            lines.append(f"  Q{i}: {turn['query']}")
            lines.append(f"  A{i}: {turn['response'][:100]}...")
        return "\n".join(lines)

    def get_relevant_history(self, current_query: str, top_k: int = 2) -> list[dict]:
        """Find most relevant past turns for current query."""
        q_vec = _hash_vector(current_query)
        scored = []
        for turn in self.history:
            sim = cosine_similarity(q_vec, _hash_vector(turn["query"]))
            scored.append((sim, turn))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in scored[:top_k]]


# ── 4. Hybrid Fusion ────────────────────────────────────────────────────────────


def hybrid_search(
    query: str,
    docs: list[dict],
    vector_weight: float = 0.6,
    keyword_weight: float = 0.4,
    top_k: int = 5,
) -> list[dict]:
    """
    Hybrid fusion: combine vector similarity score with keyword BM25-like score.
    """
    q_vec = _hash_vector(query)
    query_terms = Counter(re.findall(r"\w+", query.lower()))

    def bm25_like(doc_text: str, avg_len: float, k1: float = 1.5, b: float = 0.75) -> float:
        """Simplified BM25 scoring."""
        doc_terms = Counter(re.findall(r"\w+", doc_text.lower()))
        doc_len = sum(doc_terms.values())
        if doc_len == 0:
            return 0.0
        score = 0.0
        for term, qf in query_terms.items():
            if term in doc_terms:
                tf = doc_terms[term] / doc_len
                score += qf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avg_len))
        return score

    texts = [doc["title"] + " " + doc["content"] for doc in docs]
    avg_len = sum(len(t.split()) for t in texts) / max(len(texts), 1)

    results = []
    for doc in docs:
        text = doc["title"] + " " + doc["content"]
        vec_score = cosine_similarity(q_vec, _hash_vector(text))
        kw_score = bm25_like(text, avg_len)
        fused = vector_weight * vec_score + keyword_weight * kw_score
        results.append({**doc, "vector_score": vec_score, "keyword_score": kw_score, "fusion_score": fused})

    results.sort(key=lambda x: x["fusion_score"], reverse=True)
    return results[:top_k]


# ── Main Demo ───────────────────────────────────────────────────────────────────


def main():
    print("=" * 60)
    print("ch14: Advanced RAG Architecture Demo (stdlib-only)")
    print("=" * 60)

    # 1. Multi-stage retrieval
    print("\n[1] Multi-Stage Retrieval")
    print("-" * 40)
    query = "神经网络和深度学习的关系"
    print(f"Query: {query}")

    coarse_results = stage1_coarse_search(query, top_k=20)
    print(f"\n  Coarse search: {len(coarse_results)} candidates")
    for r in coarse_results[:5]:
        print(f"    [{r['id']}] {r['title']} (score: {r['score']:.4f})")

    fine_results = stage2_fine_rerank(coarse_results, query, top_k=5)
    print(f"\n  Fine rerank: top-5 results")
    for r in fine_results:
        print(f"    [{r['id']}] {r['title']} (rerank: {r['rerank_score']:.4f})")

    # 2. Query decomposition
    print("\n[2] Query Decomposition")
    print("-" * 40)
    complex_query = "CNN和RNN有什么区别，Transformer又是什么"
    print(f"Complex query: {complex_query}")
    sub_questions = decompose_query(complex_query)
    print(f"Decomposed into {len(sub_questions)} sub-questions:")
    for i, sq in enumerate(sub_questions, 1):
        print(f"  Sub-query {i}: {sq}")

    # 3. Multi-turn context
    print("\n[3] Multi-Turn Context Management")
    print("-" * 40)
    cm = ConversationManager(max_turns=3)
    cm.add_turn("什么是机器学习", "机器学习是让计算机从数据中学习的领域...", ["doc1"])
    cm.add_turn("CNN在图像处理中的应用", "CNN通过卷积层提取图像特征...", ["doc2", "doc4"])
    print("Conversation history:")
    print(cm.get_context())

    new_query = "Transformer比RNN好在哪里"
    relevant = cm.get_relevant_history(new_query)
    print(f"\nRelevant history for '{new_query}':")
    for turn in relevant:
        print(f"  Q: {turn['query']}")

    cm.add_turn(new_query, "Transformer通过自注意力机制解决了RNN的序列依赖问题...", ["doc2", "doc3"])
    print(f"\nAfter adding new turn, total turns: {len(cm.history)}")

    # 4. Hybrid fusion
    print("\n[4] Hybrid Fusion (Vector + Keyword)")
    print("-" * 40)
    fusion_query = "分类算法评估指标"
    print(f"Query: {fusion_query}")
    hybrid_results = hybrid_search(fusion_query, DOCUMENTS, top_k=4)
    for r in hybrid_results:
        print(f"    [{r['id']}] {r['title']} | vec={r['vector_score']:.4f} kw={r['keyword_score']:.4f} fused={r['fusion_score']:.4f}")

    print("\n" + "=" * 60)
    print("Advanced RAG Demo Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
