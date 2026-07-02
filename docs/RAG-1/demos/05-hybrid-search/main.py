"""
Demo 5: 混合检索 — Hybrid Search with Adaptive Alpha
=====================================================
Implements a HybridRetriever combining dense (TF-IDF) and sparse (BM25) scores
with a configurable alpha parameter, plus adaptive alpha selection based on
query characteristics.
"""

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class Document:
    id: int
    title: str
    content: str


SAMPLE_DOCS = [
    Document(1, "Python Basics", "Python is a high-level programming language."),
    Document(2, "Machine Learning", "ML uses algorithms to learn from data."),
    Document(3, "RAG Architecture", "RAG combines retrieval with generation."),
    Document(4, "Vector Search", "Vector search uses embeddings for similarity."),
    Document(5, "BM25 Scoring", "BM25 is a bag-of-words retrieval function."),
    Document(6, "Data Structures", "Arrays, trees, and hash maps are data structures."),
    Document(7, "Neural Networks", "Neural networks are inspired by the brain."),
    Document(8, "Information Retrieval", "IR is about finding relevant documents."),
]


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z]+", text.lower())


# ---------------------------------------------------------------------------
# Dense Retriever (TF-IDF)
# ---------------------------------------------------------------------------

class DenseRetriever:
    def __init__(self, docs: List[Document]):
        self.docs = docs
        self._build()

    def _build(self):
        self._tokens = [tokenize(d.content + " " + d.title) for d in self.docs]
        N = len(self.docs)
        df: Counter = Counter()
        for toks in self._tokens:
            for t in set(toks):
                df[t] += 1
        self._idf = {t: math.log((N + 1) / (df[t] + 1)) + 1 for t in df}

    def score(self, query: str) -> List[float]:
        qt = tokenize(query)
        scores = []
        for toks in self._tokens:
            tf = Counter(toks)
            s = sum(tf.get(t, 0) * self._idf.get(t, 0) for t in qt)
            scores.append(s)
        return scores


# ---------------------------------------------------------------------------
# Sparse Retriever (BM25)
# ---------------------------------------------------------------------------

class BM25Retriever:
    def __init__(self, docs: List[Document], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1, self.b = k1, b
        self._build()

    def _build(self):
        self._tokens = [tokenize(d.content + " " + d.title) for d in self.docs]
        self._avgdl = sum(len(t) for t in self._tokens) / max(len(self.docs), 1)
        N = len(self.docs)
        df: Counter = Counter()
        for toks in self._tokens:
            for t in set(toks):
                df[t] += 1
        self._idf = {t: math.log((N - df[t] + 0.5) / (df[t] + 0.5) + 1) for t in df}

    def score(self, query: str) -> List[float]:
        qt = tokenize(query)
        scores = []
        for i, toks in enumerate(self._tokens):
            tf = Counter(toks)
            dl = len(toks)
            s = 0.0
            for t in qt:
                if t in self._idf:
                    tf_val = tf.get(t, 0)
                    s += (self._idf[t] * tf_val * (self.k1 + 1)) / (
                        tf_val + self.k1 * (1 - self.b + self.b * dl / self._avgdl)
                    )
            scores.append(s)
        return scores


# ---------------------------------------------------------------------------
# Hybrid Retriever
# ---------------------------------------------------------------------------

def min_max_normalize(scores: List[float]) -> List[float]:
    """Normalize scores to [0, 1] range."""
    mn, mx = min(scores), max(scores)
    if mx == mn:
        return [0.0] * len(scores)
    return [(s - mn) / (mx - mn) for s in scores]


class HybridRetriever:
    """Combines dense and sparse scores with configurable alpha."""

    def __init__(self, docs: List[Document]):
        self.docs = docs
        self._dense = DenseRetriever(docs)
        self._sparse = BM25Retriever(docs)

    def retrieve(self, query: str, alpha: float = 0.5, top_k: int = 5) -> List[Tuple[Document, float]]:
        """
        alpha=1.0 -> pure dense (TF-IDF)
        alpha=0.0 -> pure sparse (BM25)
        0 < alpha < 1 -> hybrid
        """
        dense_scores = min_max_normalize(self._dense.score(query))
        sparse_scores = min_max_normalize(self._sparse.score(query))

        hybrid = [
            alpha * d + (1 - alpha) * s
            for d, s in zip(dense_scores, sparse_scores)
        ]

        indexed = list(enumerate(hybrid))
        indexed.sort(key=lambda x: x[1], reverse=True)
        return [(self.docs[i], round(s, 4)) for i, s in indexed[:top_k] if s > 0]


# ---------------------------------------------------------------------------
# Adaptive Alpha Selection
# ---------------------------------------------------------------------------

def estimate_query_type(query: str) -> Dict[str, float]:
    """Analyze query characteristics to suggest an alpha value."""
    qt = tokenize(query)
    qlen = len(qt)

    # Unique term ratio (high -> specific terms, favors sparse)
    unique_ratio = len(set(qt)) / max(qlen, 1)

    # Avg word length (longer words are more specific)
    avg_word_len = sum(len(w) for w in qt) / max(qlen, 1)

    # Contains named entities (capitalized words in original query)
    named_entities = sum(1 for w in query.split() if w[0].isupper() if len(w) > 1)
    entity_ratio = named_entities / max(len(query.split()), 1)

    # Heuristic alpha
    alpha = 0.5  # default

    # Short queries -> favor dense (semantic matching)
    if qlen <= 2:
        alpha = 0.7
    # High unique ratio + long words -> exact match, favor sparse
    if unique_ratio > 0.8 and avg_word_len > 6:
        alpha = 0.3
    # Many named entities -> favor dense (semantic)
    if entity_ratio > 0.4:
        alpha = 0.6

    return {
        "query_length": qlen,
        "unique_ratio": round(unique_ratio, 2),
        "avg_word_length": round(avg_word_len, 2),
        "entity_ratio": round(entity_ratio, 2),
        "suggested_alpha": round(alpha, 2),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Demo 5: 混合检索 — Hybrid Search with Adaptive Alpha")
    print("=" * 60)

    docs = SAMPLE_DOCS
    hybrid = HybridRetriever(docs)

    queries = [
        "What is retrieval?",
        "BM25 scoring function",
        "Python programming language",
        "Neural network learning",
    ]

    print(f"\n{'='*60}")
    print(f"{'Query Type Analysis':^60}")
    print(f"{'='*60}")

    for query in queries:
        print(f"\nQuery: \"{query}\"")
        info = estimate_query_type(query)
        print(f"  Length={info['query_length']}, UniqueRatio={info['unique_ratio']}, "
              f"AvgWordLen={info['avg_word_length']}, EntityRatio={info['entity_ratio']}")
        print(f"  Suggested alpha = {info['suggested_alpha']}")

        # Compare all modes
        for alpha, label in [(1.0, "Dense only (alpha=1.0)"),
                             (0.0, "Sparse only (alpha=0.0)"),
                             (0.5, "Hybrid     (alpha=0.5)"),
                             (info['suggested_alpha'], "Adaptive   (alpha={:.1f})".format(info['suggested_alpha']))]:
            results = hybrid.retrieve(query, alpha=alpha, top_k=3)
            doc_ids = [str(d.id) for d, _ in results]
            scores = [s for _, s in results]
            print(f"  {label:<32s} docs=[{','.join(doc_ids):>12s}] scores={scores}")

    print("\n" + "=" * 60)
    print("Adaptive alpha selects the best balance per query type.")
    print("=" * 60)


if __name__ == "__main__":
    main()
