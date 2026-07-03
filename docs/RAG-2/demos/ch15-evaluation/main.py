#!/usr/bin/env python3
"""
ch15-evaluation: RAG evaluation metrics demo — retrieval and generation evaluation.
Implements recall@k, precision@k, MRR, NDCG, entity coverage, and completeness.
Uses only stdlib.
"""

import math
from dataclasses import dataclass, field
from typing import List, Set, Dict, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class EvalCase:
    question: str
    ground_truth: str
    relevant_docs: List[int]   # IDs of relevant documents
    expected_entities: List[str]  # Entities expected in the answer


# ---------------------------------------------------------------------------
# Sample documents
# ---------------------------------------------------------------------------
SAMPLE_DOCS = [
    "恒瑞医药是中国领先的创新药研发企业，专注于抗肿瘤药物领域。",
    "注射用紫杉醇是微管抑制剂，用于乳腺癌和卵巢癌的化疗。",
    "奥希替尼片是第三代EGFR-TKI，用于非小细胞肺癌的靶向治疗。",
    "PD-1抑制剂通过阻断PD-1/PD-L1通路激活免疫系统抗肿瘤。",
    "国药控股是中国最大的医药分销企业，拥有全国性物流网络。",
    "北京协和医院是三级甲等综合医院，在肿瘤诊疗方面经验丰富。",
    "卡瑞利珠单抗是恒瑞医药自主研发的PD-1抑制剂。",
    "非小细胞肺癌约占肺癌总数的85%，是最常见的肺癌类型。",
]


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
TEST_CASES = [
    EvalCase(
        question="恒瑞医药的主要业务是什么？",
        ground_truth="恒瑞医药专注于抗肿瘤创新药物的研发和生产。",
        relevant_docs=[0, 6],
        expected_entities=["恒瑞医药", "抗肿瘤药物", "创新药"],
    ),
    EvalCase(
        question="肺癌的治疗方法有哪些？",
        ground_truth="非小细胞肺癌可用奥希替尼进行靶向治疗，也可使用PD-1抑制剂进行免疫治疗。",
        relevant_docs=[2, 3, 7],
        expected_entities=["非小细胞肺癌", "奥希替尼", "PD-1抑制剂", "靶向治疗"],
    ),
    EvalCase(
        question="PD-1抑制剂的作用机制是什么？",
        ground_truth="PD-1抑制剂通过阻断PD-1/PD-L1信号通路，重新激活T细胞对肿瘤细胞的免疫杀伤作用。",
        relevant_docs=[3, 6],
        expected_entities=["PD-1抑制剂", "PD-1", "PD-L1", "免疫治疗", "T细胞"],
    ),
]


# ---------------------------------------------------------------------------
# RetrievalEvaluator
# ---------------------------------------------------------------------------
class RetrievalEvaluator:
    """Evaluate retrieval quality."""

    @staticmethod
    def recall_at_k(retrieved: List[int], relevant: Set[int], k: int) -> float:
        """Recall@K: fraction of relevant docs retrieved in top-K."""
        if not relevant:
            return 0.0
        retrieved_k = set(retrieved[:k])
        return len(retrieved_k & relevant) / len(relevant)

    @staticmethod
    def precision_at_k(retrieved: List[int], relevant: Set[int], k: int) -> float:
        """Precision@K: fraction of top-K retrieved docs that are relevant."""
        if k == 0:
            return 0.0
        retrieved_k = set(retrieved[:k])
        return len(retrieved_k & relevant) / k

    @staticmethod
    def mrr(retrieved: List[int], relevant: Set[int]) -> float:
        """Mean Reciprocal Rank: 1/rank of first relevant document."""
        for i, doc_id in enumerate(retrieved, start=1):
            if doc_id in relevant:
                return 1.0 / i
        return 0.0

    @staticmethod
    def ndcg(retrieved: List[int], relevant: Set[int], k: int) -> float:
        """NDCG@K: Normalized Discounted Cumulative Gain."""
        dcg = 0.0
        for i, doc_id in enumerate(retrieved[:k], start=1):
            if doc_id in relevant:
                dcg += 1.0 / math.log2(i + 1)

        # Ideal DCG
        ideal_count = min(len(relevant), k)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_count))

        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def evaluate_all(retrieved: List[int], relevant: Set[int], ks: List[int] = None) -> dict:
        """Run all retrieval metrics."""
        if ks is None:
            ks = [1, 3, 5]
        results = {}
        for k in ks:
            results[f"recall@{k}"] = RetrievalEvaluator.recall_at_k(retrieved, relevant, k)
            results[f"precision@{k}"] = RetrievalEvaluator.precision_at_k(retrieved, relevant, k)
            results[f"ndcg@{k}"] = RetrievalEvaluator.ndcg(retrieved, relevant, k)
        results["mrr"] = RetrievalEvaluator.mrr(retrieved, relevant)
        return results


# ---------------------------------------------------------------------------
# GenerationEvaluator
# ---------------------------------------------------------------------------
class GenerationEvaluator:
    """Evaluate generated answer quality."""

    @staticmethod
    def entity_coverage(answer: str, expected_entities: List[str]) -> float:
        """Fraction of expected entities found in the generated answer."""
        if not expected_entities:
            return 1.0
        found = sum(1 for ent in expected_entities if ent in answer)
        return found / len(expected_entities)

    @staticmethod
    def completeness(answer: str, ground_truth: str) -> float:
        """Heuristic completeness: fraction of ground truth bigrams in answer."""
        if not ground_truth or not answer:
            return 0.0
        # Use character bigrams for simplicity
        def bigrams(s: str) -> Set[str]:
            return {s[i:i+2] for i in range(len(s) - 1)}
        gt_bigrams = bigrams(ground_truth)
        ans_bigrams = bigrams(answer)
        if not gt_bigrams:
            return 1.0
        return len(gt_bigrams & ans_bigrams) / len(gt_bigrams)


# ---------------------------------------------------------------------------
# Simulated retrieval
# ---------------------------------------------------------------------------
def simulate_retrieval(query: str) -> Tuple[List[int], List[float]]:
    """Simulate retrieval results based on query keywords."""
    query_lower = query.lower()
    scores = []

    for i, doc in enumerate(SAMPLE_DOCS):
        doc_lower = doc.lower()
        # Count keyword matches
        keywords = set(query)  # character-level for Chinese
        match_count = sum(1 for k in keywords if k in doc_lower)
        # Also check for specific terms
        if "恒瑞" in query and "恒瑞" in doc:
            match_count += 3
        if "肺癌" in query and "肺癌" in doc:
            match_count += 3
        if "PD-1" in query and "PD-1" in doc:
            match_count += 3
        if "靶向" in query and "靶向" in doc:
            match_count += 2
        if "免疫" in query and "免疫" in doc:
            match_count += 2

        scores.append((i, match_count))

    scores.sort(key=lambda x: x[1], reverse=True)
    retrieved_ids = [s[0] for s in scores]
    retrieved_scores = [float(s[1]) for s in scores]
    return retrieved_ids, retrieved_scores


# ---------------------------------------------------------------------------
# Simulated generation
# ---------------------------------------------------------------------------
def simulate_generation(query: str, retrieved_ids: List[int]) -> str:
    """Simulate answer generation from retrieved docs."""
    parts = [f"关于'{query}'的检索结果:"]
    for i, doc_id in enumerate(retrieved_ids[:3]):
        parts.append(f"  [{i+1}] {SAMPLE_DOCS[doc_id][:50]}...")
    parts.append("基于以上信息生成的回答。")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("第15章 评估指标演示 (Evaluation Metrics Demo)")
    print("=" * 60)

    ret_eval = RetrievalEvaluator()
    gen_eval = GenerationEvaluator()

    print(f"\n测试用例数: {len(TEST_CASES)}")
    print(f"文档总数: {len(SAMPLE_DOCS)}")

    all_results = []

    for case_idx, case in enumerate(TEST_CASES):
        print(f"\n{'=' * 60}")
        print(f"测试用例 #{case_idx + 1}: {case.question}")
        print(f"{'=' * 60}")

        # Simulate retrieval
        retrieved_ids, retrieved_scores = simulate_retrieval(case.question)
        relevant = set(case.relevant_docs)

        print(f"\n相关文档ID: {sorted(relevant)}")
        print(f"检索结果(前5): {retrieved_ids[:5]}")

        # Retrieval metrics
        print(f"\n--- 检索评估 ---")
        metrics = ret_eval.evaluate_all(retrieved_ids, relevant, ks=[1, 3, 5])
        for metric, value in metrics.items():
            print(f"   {metric}: {value:.4f}")

        # Simulate generation
        generated = simulate_generation(case.question, retrieved_ids)

        # Generation metrics
        print(f"\n--- 生成评估 ---")
        coverage = gen_eval.entity_coverage(generated, case.expected_entities)
        completeness = gen_eval.completeness(generated, case.ground_truth)
        print(f"   实体覆盖率: {coverage:.4f} (预期实体: {case.expected_entities})")
        print(f"   完整性:     {completeness:.4f}")
        print(f"   生成答案预览: {generated[:80]}...")

        all_results.append({
            "case": case_idx + 1,
            "question": case.question,
            **metrics,
            "entity_coverage": coverage,
            "completeness": completeness,
        })

    # Summary table
    print(f"\n{'=' * 60}")
    print("评估汇总报告")
    print(f"{'=' * 60}")
    header = f"{'用例':<6} {'R@1':<8} {'R@3':<8} {'P@3':<8} {'MRR':<8} {'NDCG@3':<10} {'实体覆盖':<10} {'完整性':<8}"
    print(header)
    print("-" * len(header))

    avg_recall1 = 0
    avg_mrr = 0
    avg_coverage = 0
    for r in all_results:
        print(f"{r['case']:<6} {r['recall@1']:<8.4f} {r['recall@3']:<8.4f} "
              f"{r['precision@3']:<8.4f} {r['mrr']:<8.4f} {r['ndcg@3']:<10.4f} "
              f"{r['entity_coverage']:<10.4f} {r['completeness']:<8.4f}")
        avg_recall1 += r['recall@1']
        avg_mrr += r['mrr']
        avg_coverage += r['entity_coverage']

    n = len(all_results)
    print("-" * len(header))
    print(f"{'平均':<6} {avg_recall1/n:<8.4f} {'':8} {'':8} {avg_mrr/n:<8.4f} {'':10} {avg_coverage/n:<10.4f} {'':8}")

    print(f"\n{'=' * 60}")
    print("演示完成。")
    print("=" * 60)


if __name__ == "__main__":
    main()
