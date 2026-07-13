"""
ch22-evaluation: RAG Evaluation Demo (stdlib-only)
Defines test queries with ground truth, computes recall@k, precision@k, MRR,
simulates answer generation with BLEU-like score and F1.
All pure Python, no external dependencies.
"""

import math
import re
from collections import Counter

# ── Sample Document Corpus ──────────────────────────────────────────────────────

DOCUMENTS = {
    "doc1": {"title": "机器学习基础", "content": "机器学习是人工智能的核心领域。监督学习使用标注数据训练模型。"},
    "doc2": {"title": "深度学习简介", "content": "深度学习使用多层神经网络。CNN擅长图像处理，RNN适合序列数据。"},
    "doc3": {"title": "自然语言处理", "content": "NLP技术包括分词和词性标注。BERT和GPT是重要的预训练语言模型。"},
    "doc4": {"title": "计算机视觉基础", "content": "计算机视觉使机器理解图像。目标检测识别物体，图像分割划分区域。"},
    "doc5": {"title": "强化学习概述", "content": "强化学习通过奖励信号训练智能体。Q学习和DQN是经典算法。"},
    "doc6": {"title": "监督学习算法", "content": "线性回归、逻辑回归、决策树、SVM和随机森林是常见监督学习算法。"},
    "doc7": {"title": "神经网络架构", "content": "前馈网络、CNN、RNN、ResNet、注意力机制和GAN是常见神经网络架构。"},
    "doc8": {"title": "模型评估方法", "content": "交叉验证、混淆矩阵、精确率、召回率、F1分数和ROC曲线是常用评估方法。"},
    "doc9": {"title": "聚类算法", "content": "K-means、层次聚类和DBSCAN是常见无监督聚类算法。"},
    "doc10": {"title": "特征工程", "content": "特征工程包括特征选择、特征提取和特征构造。PCA是常用降维方法。"},
}

# ── Test Queries with Ground Truth ──────────────────────────────────────────────

TEST_QUERIES = [
    {
        "query": "什么是监督学习？",
        "expected_docs": ["doc1", "doc6"],
        "expected_answer": "监督学习是一种使用标注数据进行训练的机器学习方法。",
    },
    {
        "query": "CNN和RNN的区别",
        "expected_docs": ["doc2", "doc7"],
        "expected_answer": "CNN擅长图像处理，RNN适合序列数据处理。",
    },
    {
        "query": "常用的模型评估指标",
        "expected_docs": ["doc8"],
        "expected_answer": "常用评估指标包括精确率、召回率、F1分数和ROC曲线。",
    },
    {
        "query": "BERT和GPT是什么",
        "expected_docs": ["doc3"],
        "expected_answer": "BERT和GPT是重要的预训练语言模型，用于自然语言处理任务。",
    },
    {
        "query": "强化学习经典算法",
        "expected_docs": ["doc5"],
        "expected_answer": "Q学习和深度Q网络（DQN）是强化学习的经典算法。",
    },
]


# ── Retrieval Simulation ────────────────────────────────────────────────────────


def simple_retrieve(query: str, top_k: int = 5) -> list[str]:
    """
    Simple keyword-based retrieval simulation.
    Returns ranked document IDs based on term overlap.
    """
    query_terms = set(re.findall(r"[一-龥\w]+", query.lower()))
    scored = []

    for doc_id, doc in DOCUMENTS.items():
        text = (doc["title"] + " " + doc["content"]).lower()
        doc_terms = set(re.findall(r"[一-龥\w]+", text))
        overlap = len(query_terms & doc_terms)
        scored.append((doc_id, overlap))

    scored.sort(key=lambda x: (-x[1], x[0]))
    return [doc_id for doc_id, _ in scored[:top_k]]


# ── 1. Recall@k ────────────────────────────────────────────────────────────────


def recall_at_k(retrieved: list[str], expected: list[str], k: int) -> float:
    """Recall@k = |retrieved[:k] ∩ expected| / |expected|."""
    if not expected:
        return 0.0
    retrieved_k = set(retrieved[:k])
    expected_set = set(expected)
    hits = len(retrieved_k & expected_set)
    return hits / len(expected_set)


# ── 2. Precision@k ─────────────────────────────────────────────────────────────


def precision_at_k(retrieved: list[str], expected: list[str], k: int) -> float:
    """Precision@k = |retrieved[:k] ∩ expected| / k."""
    if k == 0:
        return 0.0
    retrieved_k = set(retrieved[:k])
    expected_set = set(expected)
    hits = len(retrieved_k & expected_set)
    return hits / k


# ── 3. Mean Reciprocal Rank (MRR) ──────────────────────────────────────────────


def reciprocal_rank(retrieved: list[str], expected: list[str]) -> float:
    """RR = 1 / rank of first relevant document (0 if none found)."""
    expected_set = set(expected)
    for i, doc_id in enumerate(retrieved, 1):
        if doc_id in expected_set:
            return 1.0 / i
    return 0.0


def mean_reciprocal_rank(all_retrieved: list[list[str]], all_expected: list[list[str]]) -> float:
    """MRR = mean of reciprocal ranks across all queries."""
    if not all_retrieved:
        return 0.0
    rrs = [reciprocal_rank(ret, exp) for ret, exp in zip(all_retrieved, all_expected)]
    return sum(rrs) / len(rrs)


# ── 4. BLEU-like Score ─────────────────────────────────────────────────────────


def bleu_like(candidate: str, reference: str, max_n: int = 2) -> float:
    """
    Simplified BLEU score: geometric mean of n-gram precisions (n=1,2)
    with brevity penalty.
    """
    cand_tokens = re.findall(r"[一-龥\w]+", candidate.lower())
    ref_tokens = re.findall(r"[一-龥\w]+", reference.lower())

    if not cand_tokens or not ref_tokens:
        return 0.0

    precisions = []
    for n in range(1, max_n + 1):
        cand_ngrams = Counter(
            tuple(cand_tokens[i : i + n]) for i in range(len(cand_tokens) - n + 1)
        )
        ref_ngrams = Counter(
            tuple(ref_tokens[i : i + n]) for i in range(len(ref_tokens) - n + 1)
        )

        if not cand_ngrams:
            precisions.append(0.0)
            continue

        matches = sum(min(cand_ngrams[ng], ref_ngrams.get(ng, 0)) for ng in cand_ngrams)
        total = sum(cand_ngrams.values())
        precisions.append(matches / total if total > 0 else 0.0)

    # Geometric mean
    if any(p == 0 for p in precisions):
        return 0.0
    geo_mean = math.exp(sum(math.log(p) for p in precisions) / len(precisions))

    # Brevity penalty
    c = len(cand_tokens)
    r = len(ref_tokens)
    bp = math.exp(1 - r / c) if c < r else 1.0

    return bp * geo_mean


# ── 5. F1 Score ────────────────────────────────────────────────────────────────


def f1_score(candidate: str, reference: str) -> float:
    """F1 = 2 * P * R / (P + R) based on token overlap."""
    cand_tokens = set(re.findall(r"[一-龥\w]+", candidate.lower()))
    ref_tokens = set(re.findall(r"[一-龥\w]+", reference.lower()))

    if not cand_tokens or not ref_tokens:
        return 0.0

    common = cand_tokens & ref_tokens
    if not common:
        return 0.0

    precision = len(common) / len(cand_tokens)
    recall = len(common) / len(ref_tokens)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


# ── Answer Generation Simulation ────────────────────────────────────────────────


def simulate_answer(query: str, retrieved_docs: list[str]) -> str:
    """
    Simulate answer generation by extracting relevant content from retrieved docs.
    """
    parts = []
    for doc_id in retrieved_docs[:3]:  # use top-3 docs
        if doc_id in DOCUMENTS:
            parts.append(DOCUMENTS[doc_id]["content"][:60])
    return " ".join(parts) if parts else "No relevant information found."


# ── Main Demo ───────────────────────────────────────────────────────────────────


def main():
    print("=" * 60)
    print("ch22: RAG Evaluation Demo (stdlib-only)")
    print("=" * 60)

    all_retrieved = []
    all_expected = []
    all_candidates = []
    all_references = []

    # Evaluate each query
    print(f"\n[1] Retrieval Evaluation (per query)")
    print("-" * 40)

    for i, tq in enumerate(TEST_QUERIES, 1):
        query = tq["query"]
        expected = tq["expected_docs"]
        retrieved = simple_retrieve(query, top_k=5)

        all_retrieved.append(retrieved)
        all_expected.append(expected)

        r1 = recall_at_k(retrieved, expected, k=1)
        r3 = recall_at_k(retrieved, expected, k=3)
        r5 = recall_at_k(retrieved, expected, k=5)
        p1 = precision_at_k(retrieved, expected, k=1)
        p3 = precision_at_k(retrieved, expected, k=3)
        rr = reciprocal_rank(retrieved, expected)

        print(f"\n  Query {i}: \"{query}\"")
        print(f"  Expected: {expected}")
        print(f"  Retrieved: {retrieved}")
        print(f"  Recall@1={r1:.3f}  Recall@3={r3:.3f}  Recall@5={r5:.3f}")
        print(f"  Precision@1={p1:.3f}  Precision@3={p3:.3f}")
        print(f"  Reciprocal Rank={rr:.3f}")

    # Aggregate retrieval metrics
    print(f"\n[2] Aggregate Retrieval Metrics")
    print("-" * 40)

    avg_r1 = sum(recall_at_k(r, e, 1) for r, e in zip(all_retrieved, all_expected)) / len(TEST_QUERIES)
    avg_r3 = sum(recall_at_k(r, e, 3) for r, e in zip(all_retrieved, all_expected)) / len(TEST_QUERIES)
    avg_r5 = sum(recall_at_k(r, e, 5) for r, e in zip(all_retrieved, all_expected)) / len(TEST_QUERIES)
    avg_p1 = sum(precision_at_k(r, e, 1) for r, e in zip(all_retrieved, all_expected)) / len(TEST_QUERIES)
    avg_p3 = sum(precision_at_k(r, e, 3) for r, e in zip(all_retrieved, all_expected)) / len(TEST_QUERIES)
    mrr = mean_reciprocal_rank(all_retrieved, all_expected)

    print(f"  Mean Recall@1: {avg_r1:.3f}")
    print(f"  Mean Recall@3: {avg_r3:.3f}")
    print(f"  Mean Recall@5: {avg_r5:.3f}")
    print(f"  Mean Precision@1: {avg_p1:.3f}")
    print(f"  Mean Precision@3: {avg_p3:.3f}")
    print(f"  MRR: {mrr:.3f}")

    # Answer generation evaluation
    print(f"\n[3] Answer Generation Evaluation")
    print("-" * 40)

    for i, tq in enumerate(TEST_QUERIES, 1):
        retrieved = simple_retrieve(tq["query"], top_k=3)
        candidate = simulate_answer(tq["query"], retrieved)
        reference = tq["expected_answer"]

        all_candidates.append(candidate)
        all_references.append(reference)

        bleu = bleu_like(candidate, reference)
        f1 = f1_score(candidate, reference)

        print(f"\n  Query {i}: \"{tq['query']}\"")
        print(f"  Reference: \"{reference}\"")
        print(f"  Generated: \"{candidate[:80]}...\"")
        print(f"  BLEU-like: {bleu:.3f}")
        print(f"  F1 Score:  {f1:.3f}")

    # Final report
    print("\n" + "=" * 60)
    print("Evaluation Report")
    print("=" * 60)
    print(f"\n  Test queries: {len(TEST_QUERIES)}")
    print(f"\n  [Retrieval]")
    print(f"    Mean Recall@1:   {avg_r1:.3f}")
    print(f"    Mean Recall@3:   {avg_r3:.3f}")
    print(f"    Mean Recall@5:   {avg_r5:.3f}")
    print(f"    Mean Precision@1: {avg_p1:.3f}")
    print(f"    Mean Precision@3: {avg_p3:.3f}")
    print(f"    MRR:              {mrr:.3f}")
    print(f"\n  [Generation]")
    avg_bleu = sum(bleu_like(c, r) for c, r in zip(all_candidates, all_references)) / len(TEST_QUERIES)
    avg_f1 = sum(f1_score(c, r) for c, r in zip(all_candidates, all_references)) / len(TEST_QUERIES)
    print(f"    Mean BLEU-like:  {avg_bleu:.3f}")
    print(f"    Mean F1:         {avg_f1:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
