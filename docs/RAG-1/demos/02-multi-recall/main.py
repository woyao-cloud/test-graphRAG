"""
Demo 2: 多路召回 — Multi-Recall with RRF Fusion
=================================================
Implements three retrievers (TF-IDF dense, BM25, Metadata) and fuses results
using Reciprocal Rank Fusion (RRF) to improve coverage.
"""

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data Model
# ---------------------------------------------------------------------------

@dataclass
class Document:
    id: int
    title: str
    content: str
    tags: List[str] = field(default_factory=list)
    category: str = ""


SAMPLE_DOCS = [
    Document(1, "RAG Introduction",
             "Retrieval-Augmented Generation combines retrieval with generation.",
             tags=["rag", "basics"], category="concept"),
    Document(2, "TF-IDF Explained",
             "TF-IDF weighs terms by frequency and inverse document frequency.",
             tags=["retrieval", "tfidf"], category="technique"),
    Document(3, "BM25 Algorithm",
             "BM25 is a probabilistic retrieval function that extends TF-IDF.",
             tags=["retrieval", "bm25"], category="technique"),
    Document(4, "Vector Embeddings",
             "Embeddings convert text into dense vector representations.",
             tags=["embedding", "vector"], category="technique"),
    Document(5, "Hybrid Search",
             "Hybrid search combines sparse and dense retrieval methods.",
             tags=["retrieval", "hybrid"], category="technique"),
    Document(6, "Chunking Strategies",
             "Chunking splits documents for effective retrieval in RAG.",
             tags=["rag", "chunking"], category="pipeline"),
    Document(7, "Evaluation Metrics",
             "Recall, Precision, MRR, and NDCG measure retrieval quality.",
             tags=["evaluation", "metrics"], category="eval"),
    Document(8, "Prompt Engineering",
             "Prompt design is critical for LLM generation quality.",
             tags=["llm", "prompt"], category="generation"),
]


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z]+", text.lower())


# ---------------------------------------------------------------------------
# Retriever 1: DenseRetriever (TF-IDF-based)
# ---------------------------------------------------------------------------

class DenseRetriever:
    """TF-IDF vector-space retriever."""

    def __init__(self, docs: List[Document]):
        self.docs = docs
        self._build_index()

    def _build_index(self):
        self.doc_tokens = [tokenize(d.content + " " + d.title) for d in self.docs]
        N = len(self.docs)
        df: Counter = Counter()
        for tokens in self.doc_tokens:
            for t in set(tokens):
                df[t] += 1
        self.idf = {t: math.log((N + 1) / (df[t] + 1)) + 1 for t in df}

    def _score(self, query_tokens: List[str], doc_idx: int) -> float:
        tf = Counter(self.doc_tokens[doc_idx])
        score = 0.0
        for t in query_tokens:
            if t in self.idf:
                score += tf.get(t, 0) * self.idf[t]
        return score

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[Document, float]]:
        q_tokens = tokenize(query)
        scores = [(i, self._score(q_tokens, i)) for i in range(len(self.docs))]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [(self.docs[i], round(s, 4)) for i, s in scores[:top_k] if s > 0]


# ---------------------------------------------------------------------------
# Retriever 2: BM25Retriever
# ---------------------------------------------------------------------------

class BM25Retriever:
    """BM25 probabilistic retrieval."""

    def __init__(self, docs: List[Document], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1 = k1
        self.b = b
        self._build_index()

    def _build_index(self):
        self.doc_tokens = [tokenize(d.content + " " + d.title) for d in self.docs]
        self.avgdl = sum(len(t) for t in self.doc_tokens) / max(len(self.docs), 1)
        N = len(self.docs)
        df: Counter = Counter()
        for tokens in self.doc_tokens:
            for t in set(tokens):
                df[t] += 1
        self.idf = {t: math.log((N - df[t] + 0.5) / (df[t] + 0.5) + 1) for t in df}

    def _score(self, query_tokens: List[str], doc_idx: int) -> float:
        tokens = self.doc_tokens[doc_idx]
        tf = Counter(tokens)
        dl = len(tokens)
        score = 0.0
        for t in query_tokens:
            if t in self.idf:
                tf_val = tf.get(t, 0)
                score += (self.idf[t] * tf_val * (self.k1 + 1)) / (
                    tf_val + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                )
        return score

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[Document, float]]:
        q_tokens = tokenize(query)
        scores = [(i, self._score(q_tokens, i)) for i in range(len(self.docs))]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [(self.docs[i], round(s, 4)) for i, s in scores[:top_k] if s > 0]


# ---------------------------------------------------------------------------
# Retriever 3: MetadataRetriever (tag/category filter)
# ---------------------------------------------------------------------------

class MetadataRetriever:
    """Retrieve by matching tags and category keywords."""

    def __init__(self, docs: List[Document]):
        self.docs = docs

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[Document, float]]:
        q_tokens = set(tokenize(query))
        scores = []
        for doc in self.docs:
            score = 0.0
            for tag in doc.tags:
                if tag in q_tokens:
                    score += 2.0
            for word in tokenize(doc.category):
                if word in q_tokens:
                    score += 1.0
            for word in tokenize(doc.title):
                if word in q_tokens:
                    score += 1.5
            scores.append((doc, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [(doc, s) for doc, s in scores[:top_k] if s > 0]


# ---------------------------------------------------------------------------
# RRF Fusion
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    result_lists: List[List[Tuple[Document, float]]],
    k: int = 60,
    top_k: int = 5,
) -> List[Tuple[Document, float]]:
    """
    Reciprocal Rank Fusion: combine ranked lists from multiple retrievers.
    RRF score = sum(1 / (k + rank( doc in list ))).
    """
    fused: Dict[int, float] = {}
    doc_map: Dict[int, Document] = {}

    for ranked in result_lists:
        for rank, (doc, _) in enumerate(ranked, start=1):
            fused[doc.id] = fused.get(doc.id, 0.0) + 1.0 / (k + rank)
            doc_map[doc.id] = doc

    sorted_ids = sorted(fused, key=lambda did: fused[did], reverse=True)
    return [(doc_map[did], round(fused[did], 4)) for did in sorted_ids[:top_k]]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Demo 2: 多路召回 — Multi-Recall with RRF Fusion")
    print("=" * 60)

    docs = SAMPLE_DOCS
    print(f"\nLoaded {len(docs)} sample documents.")

    # Init retrievers
    dense = DenseRetriever(docs)
    bm25 = BM25Retriever(docs)
    meta = MetadataRetriever(docs)

    # Test query
    query = "retrieval techniques for RAG"
    print(f"\nQuery: \"{query}\"")

    # Retrieve from each
    dense_results = dense.retrieve(query, top_k=4)
    bm25_results = bm25.retrieve(query, top_k=4)
    meta_results = meta.retrieve(query, top_k=4)

    print("\n--- DenseRetriever (TF-IDF) Results ---")
    for doc, score in dense_results:
        print(f"  [{doc.id}] {doc.title:<25s} score={score:.4f}")

    print("\n--- BM25Retriever Results ---")
    for doc, score in bm25_results:
        print(f"  [{doc.id}] {doc.title:<25s} score={score:.4f}")

    print("\n--- MetadataRetriever Results ---")
    for doc, score in meta_results:
        print(f"  [{doc.id}] {doc.title:<25s} score={score:.4f}")

    # Fusion
    print("\n--- RRF Fusion (combined results) ---")
    fused = reciprocal_rank_fusion([dense_results, bm25_results, meta_results], top_k=5)
    for doc, score in fused:
        print(f"  [{doc.id}] {doc.title:<25s} rrf_score={score:.4f}")

    # Coverage comparison
    all_dense_ids = {d.id for d, _ in dense_results}
    all_bm25_ids = {d.id for d, _ in bm25_results}
    all_meta_ids = {d.id for d, _ in meta_results}
    all_fused_ids = {d.id for d, _ in fused}

    print("\n--- Coverage Comparison ---")
    print(f"  DenseRetriever docs:  {sorted(all_dense_ids)}")
    print(f"  BM25Retriever docs:   {sorted(all_bm25_ids)}")
    print(f"  MetadataRetriever docs:{sorted(all_meta_ids)}")
    print(f"  Fused docs:           {sorted(all_fused_ids)}")
    print(f"  Unique from fusion:   {sorted(all_fused_ids - all_dense_ids - all_bm25_ids) or 'none'}")

    print("\n" + "=" * 60)
    print("Multi-recall fusion improves coverage and robustness.")
    print("=" * 60)


if __name__ == "__main__":
    main()
