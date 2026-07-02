"""
Demo 8: 评估体系 — Retrieval & Generation Evaluation
=====================================================
Implements RetrievalEvaluator (Recall@K, Precision@K, MRR, NDCG)
and GenerationEvaluator (entity_coverage, completeness).
Runs evaluation on sample test cases and prints a report.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class TestCase:
    query: str
    relevant_doc_ids: Set[int]
    expected_entities: Set[str] = field(default_factory=set)
    expected_keywords: Set[str] = field(default_factory=set)


@dataclass
class RetrievalResult:
    doc_id: int
    score: float
    is_relevant: bool = False


# ---------------------------------------------------------------------------
# Sample Test Cases
# ---------------------------------------------------------------------------

TEST_CASES = [
    TestCase(
        query="RAG retrieval methods",
        relevant_doc_ids={1, 2, 3},
        expected_entities={"RAG", "retrieval", "BM25", "Dense"},
        expected_keywords={"retrieval", "method", "search"},
    ),
    TestCase(
        query="GraphRAG knowledge graph",
        relevant_doc_ids={4, 5},
        expected_entities={"GraphRAG", "entity", "relationship", "graph"},
        expected_keywords={"graph", "knowledge", "entity"},
    ),
    TestCase(
        query="chunking strategies",
        relevant_doc_ids={6, 7},
        expected_entities={"chunking", "overlap", "sentence"},
        expected_keywords={"chunk", "split", "segment"},
    ),
]


# ---------------------------------------------------------------------------
# Simulated Retrievers
# ---------------------------------------------------------------------------

def simulate_retrieval(query: str, top_k: int = 5) -> List[RetrievalResult]:
    """Simulate a retriever returning ranked results."""
    # Keyword-based simulation
    query_words = set(query.lower().split())
    all_docs = {
        1: {"tokens": {"rag", "retrieval", "generation", "method"}, "score": 0.9},
        2: {"tokens": {"bm25", "retrieval", "ranking", "sparse"}, "score": 0.8},
        3: {"tokens": {"dense", "embedding", "retrieval", "vector"}, "score": 0.7},
        4: {"tokens": {"graphrag", "graph", "entity", "relationship"}, "score": 0.6},
        5: {"tokens": {"knowledge", "graph", "triple", "fact"}, "score": 0.5},
        6: {"tokens": {"chunking", "split", "segment", "overlap"}, "score": 0.4},
        7: {"tokens": {"sentence", "chunk", "tokenizer", "strategy"}, "score": 0.3},
        8: {"tokens": {"evaluation", "recall", "precision", "mrr"}, "score": 0.2},
    }

    results = []
    for did, info in all_docs.items():
        overlap = len(query_words & info["tokens"])
        if overlap > 0:
            score = info["score"] * (1 + 0.1 * overlap)
        else:
            score = 0.0
        results.append(RetrievalResult(did, round(score, 4)))

    results.sort(key=lambda r: r.score, reverse=True)
    return results[:top_k]


# ---------------------------------------------------------------------------
# RetrievalEvaluator
# ---------------------------------------------------------------------------

class RetrievalEvaluator:
    """Evaluates retrieval quality with standard IR metrics."""

    @staticmethod
    def recall_at_k(results: List[RetrievalResult], relevant: Set[int], k: int) -> float:
        if not relevant:
            return 0.0
        retrieved = set(r.doc_id for r in results[:k])
        hit = len(retrieved & relevant)
        return hit / len(relevant)

    @staticmethod
    def precision_at_k(results: List[RetrievalResult], relevant: Set[int], k: int) -> float:
        if k == 0:
            return 0.0
        retrieved = set(r.doc_id for r in results[:k])
        hit = len(retrieved & relevant)
        return hit / k

    @staticmethod
    def mrr(results: List[RetrievalResult], relevant: Set[int]) -> float:
        """Mean Reciprocal Rank: 1/rank of first relevant document."""
        for rank, r in enumerate(results, start=1):
            if r.doc_id in relevant:
                return 1.0 / rank
        return 0.0

    @staticmethod
    def ndcg(results: List[RetrievalResult], relevant: Set[int], k: int) -> float:
        """Normalized Discounted Cumulative Gain at K."""
        def dcg(scores: List[float]) -> float:
            return sum(s / math.log2(i + 2) for i, s in enumerate(scores) if s > 0)

        # Relevance scores: 1 if relevant, 0 otherwise
        rel = [1.0 if r.doc_id in relevant else 0.0 for r in results[:k]]
        # Ideal ranking: all relevant first
        ideal = sorted(rel, reverse=True)

        actual_dcg = dcg(rel)
        ideal_dcg = dcg(ideal)
        return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0

    def evaluate(self, results: List[RetrievalResult], relevant: Set[int]) -> Dict[str, float]:
        """Compute all metrics at once."""
        return {
            "Recall@1": round(self.recall_at_k(results, relevant, 1), 4),
            "Recall@3": round(self.recall_at_k(results, relevant, 3), 4),
            "Recall@5": round(self.recall_at_k(results, relevant, 5), 4),
            "Precision@1": round(self.precision_at_k(results, relevant, 1), 4),
            "Precision@3": round(self.precision_at_k(results, relevant, 3), 4),
            "Precision@5": round(self.precision_at_k(results, relevant, 5), 4),
            "MRR": round(self.mrr(results, relevant), 4),
            "NDCG@3": round(self.ndcg(results, relevant, 3), 4),
            "NDCG@5": round(self.ndcg(results, relevant, 5), 4),
        }


# ---------------------------------------------------------------------------
# GenerationEvaluator
# ---------------------------------------------------------------------------

class GenerationEvaluator:
    """Evaluates generation quality (requires ground truth)."""

    @staticmethod
    def entity_coverage(generated_text: str, expected_entities: Set[str]) -> float:
        """Fraction of expected entities appearing in the generated text."""
        if not expected_entities:
            return 1.0
        text_lower = generated_text.lower()
        covered = sum(1 for e in expected_entities if e.lower() in text_lower)
        return covered / len(expected_entities)

    @staticmethod
    def completeness(generated_text: str, expected_keywords: Set[str]) -> float:
        """Fraction of expected keywords covered in the generated text."""
        if not expected_keywords:
            return 1.0
        text_lower = generated_text.lower()
        covered = sum(1 for kw in expected_keywords if kw.lower() in text_lower)
        return covered / len(expected_keywords)

    def evaluate(self, generated_text: str, expected_entities: Set[str],
                 expected_keywords: Set[str]) -> Dict[str, float]:
        return {
            "entity_coverage": round(self.entity_coverage(generated_text, expected_entities), 4),
            "completeness": round(self.completeness(generated_text, expected_keywords), 4),
        }


# ---------------------------------------------------------------------------
# Simulated Generation
# ---------------------------------------------------------------------------

def simulate_generate(query: str, relevant_ids: Set[int]) -> str:
    """Simulate LLM generation given relevant documents."""
    doc_info = {
        1: "RAG combines retrieval with generation.",
        2: "BM25 is a popular sparse retrieval method.",
        3: "Dense retrieval uses embeddings for semantic search.",
        4: "GraphRAG builds a knowledge graph from documents.",
        5: "Knowledge graphs store entities and their relationships.",
        6: "Chunking splits documents into smaller pieces.",
        7: "Sentence-based chunking preserves semantic boundaries.",
    }
    context = " ".join(doc_info.get(did, "") for did in relevant_ids)
    return (
        f"In response to '{query}': "
        f"Based on the retrieved documents, {context} "
        f"This approach improves both retrieval quality and generation accuracy."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Demo 8: 评估体系 — Retrieval & Generation Evaluation")
    print("=" * 60)

    ret_eval = RetrievalEvaluator()
    gen_eval = GenerationEvaluator()

    print(f"\nRunning evaluation on {len(TEST_CASES)} test cases...")
    print()

    all_ret_metrics: Dict[str, List[float]] = {}

    for tc in TEST_CASES:
        print("-" * 50)
        print(f"Test Case: \"{tc.query}\"")
        print(f"  Relevant docs: {sorted(tc.relevant_doc_ids)}")

        # Retrieval
        results = simulate_retrieval(tc.query, top_k=5)
        print(f"  Retrieved: {[r.doc_id for r in results]}")

        metrics = ret_eval.evaluate(results, tc.relevant_doc_ids)
        print("  Retrieval Metrics:")
        for key, val in metrics.items():
            print(f"    {key:<15s} = {val:.4f}")
            all_ret_metrics.setdefault(key, []).append(val)

        # Generation
        generated = simulate_generate(tc.query, tc.relevant_doc_ids)
        print(f"  Generated: \"{generated[:80]}...\"")
        gen_metrics = gen_eval.evaluate(generated, tc.expected_entities, tc.expected_keywords)
        print("  Generation Metrics:")
        for key, val in gen_metrics.items():
            print(f"    {key:<20s} = {val:.4f}")

    # Summary report
    print("\n" + "=" * 60)
    print("SUMMARY REPORT")
    print("=" * 60)
    print(f"\nAverage Retrieval Metrics (across {len(TEST_CASES)} test cases):")
    for key in sorted(all_ret_metrics.keys()):
        values = all_ret_metrics[key]
        avg = sum(values) / len(values)
        print(f"  {key:<15s} = {avg:.4f}")

    print("\nInterpretation:")
    print("  - Recall@K: How many relevant docs are retrieved in top-K")
    print("  - Precision@K: How many retrieved docs are relevant")
    print("  - MRR: Rank of the first relevant result (higher = better)")
    print("  - NDCG: Ranking quality vs. ideal ranking")
    print("  - Entity Coverage: Does generation use expected entities?")
    print("  - Completeness: Does generation cover expected keywords?")

    print("\n" + "=" * 60)
    print("Evaluation provides quantitative quality metrics for RAG.")
    print("=" * 60)


if __name__ == "__main__":
    main()
