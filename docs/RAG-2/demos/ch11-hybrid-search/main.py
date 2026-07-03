#!/usr/bin/env python3
"""
ch11-hybrid-search: Hybrid search demo combining dense (TF-IDF) and sparse (BM25)
retrieval with adaptive alpha weighting. Uses only stdlib.
"""

import math
import collections
from dataclasses import dataclass, field
from typing import List, Dict, Tuple


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------
@dataclass(unsafe_hash=True)
class Document:
    id: int
    title: str
    content: str


# ---------------------------------------------------------------------------
# Sample corpus
# ---------------------------------------------------------------------------
SAMPLE_DOCS = [
    Document(1, "恒瑞医药研发", "恒瑞医药持续投入大量资金进行创新药物研发，拥有多个在研项目。"),
    Document(2, "紫杉醇化疗", "紫杉醇是广泛应用于乳腺癌和卵巢癌的化疗药物。"),
    Document(3, "奥希替尼靶向", "奥希替尼是EGFR突变阳性非小细胞肺癌的靶向治疗药物。"),
    Document(4, "PD-1免疫治疗", "PD-1抑制剂通过激活免疫系统攻击肿瘤细胞。"),
    Document(5, "国药控股分销", "国药控股覆盖全国的药械分销网络为医院提供供应保障。"),
    Document(6, "协和医院诊疗", "北京协和医院在肿瘤诊疗方面具有丰富的临床经验。"),
    Document(7, "卡瑞利珠单抗", "卡瑞利珠单抗是恒瑞医药自主研发的PD-1抑制剂。"),
    Document(8, "阿帕替尼", "阿帕替尼是一种抗血管生成药物，用于胃癌治疗。"),
]


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
def tokenize(text: str) -> List[str]:
    try:
        import jieba
        return list(jieba.cut(text))
    except ImportError:
        text = text.strip()
        if len(text) <= 1:
            return [text] if text else []
        return [text[i:i+2] for i in range(len(text) - 1)] + list(text)


# ---------------------------------------------------------------------------
# Normalizer
# ---------------------------------------------------------------------------
def normalize_scores(scores: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
    """Min-max normalize scores to [0, 1]."""
    if not scores:
        return scores
    values = [s for _, s in scores]
    min_s = min(values)
    max_s = max(values)
    if max_s == min_s:
        return [(d, 1.0) for d, _ in scores]
    return [(d, (s - min_s) / (max_s - min_s)) for d, s in scores]


# ---------------------------------------------------------------------------
# HybridRetriever
# ---------------------------------------------------------------------------
class HybridRetriever:
    """Combines dense (TF-IDF) and sparse (BM25) retrieval."""

    def __init__(self):
        self.docs: List[Document] = []
        # Dense (TF-IDF) data
        self.idf: Dict[str, float] = {}
        self.doc_vectors: List[Dict[str, float]] = []
        # Sparse (BM25) data
        self.avgdl: float = 0.0
        self.doc_lens: List[int] = []
        self.df: Dict[str, int] = {}
        self.doc_tokens: List[List[str]] = []
        self.k1 = 1.5
        self.b = 0.75

    def fit(self, docs: List[Document]):
        self.docs = docs
        tokenized = [tokenize(d.content) for d in docs]
        self.doc_tokens = tokenized
        self.doc_lens = [len(toks) for toks in tokenized]
        self.avgdl = sum(self.doc_lens) / len(self.doc_lens) if self.doc_lens else 0

        n = len(docs)
        for toks in tokenized:
            for t in set(toks):
                self.df[t] = self.df.get(t, 0) + 1
        self.idf = {t: math.log((n + 1) / (c + 1)) + 1 for t, c in self.df.items()}

        # Build TF-IDF vectors
        for toks in tokenized:
            tf = collections.Counter(toks)
            total = len(toks)
            vec = {}
            for t, c in tf.items():
                if t in self.idf:
                    vec[t] = (c / total) * self.idf[t]
            self.doc_vectors.append(vec)
        return self

    # ---- Dense search (TF-IDF) ----
    def _vectorize(self, text: str) -> Dict[str, float]:
        toks = tokenize(text)
        tf = collections.Counter(toks)
        total = len(toks)
        return {t: (c / total) * self.idf.get(t, 0) for t, c in tf.items() if t in self.idf}

    def _cosine(self, a: dict, b: dict) -> float:
        keys = set(a) & set(b)
        dot = sum(a[k] * b[k] for k in keys)
        na = math.sqrt(sum(v*v for v in a.values()))
        nb = math.sqrt(sum(v*v for v in b.values()))
        return dot / (na * nb) if na and nb else 0.0

    def dense_search(self, query: str) -> List[Tuple[Document, float]]:
        qv = self._vectorize(query)
        scores = [(doc, self._cosine(qv, dv)) for doc, dv in zip(self.docs, self.doc_vectors)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return normalize_scores(scores)

    # ---- Sparse search (BM25) ----
    def sparse_search(self, query: str) -> List[Tuple[Document, float]]:
        qtokens = tokenize(query)
        scores = []
        for i, doc in enumerate(self.docs):
            score = 0.0
            dl = self.doc_lens[i]
            tf = collections.Counter(self.doc_tokens[i])
            for q in qtokens:
                if q not in self.idf:
                    continue
                f = tf.get(q, 0)
                bm25_idf = math.log((len(self.docs) - self.df.get(q, 0) + 0.5) / (self.df.get(q, 0) + 0.5) + 1)
                score += bm25_idf * (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            scores.append((doc, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return normalize_scores(scores)

    # ---- Hybrid search ----
    def hybrid_search(self, query: str, alpha: float = 0.5) -> List[Tuple[Document, float]]:
        """alpha: weight for dense (TF-IDF). Sparse weight = 1-alpha."""
        dense_map = dict(self.dense_search(query))
        sparse_map = dict(self.sparse_search(query))

        combined = []
        for doc in self.docs:
            ds = dense_map.get(doc, 0.0)
            ss = sparse_map.get(doc, 0.0)
            hybrid = alpha * ds + (1 - alpha) * ss
            combined.append((doc, hybrid))

        combined.sort(key=lambda x: x[1], reverse=True)
        return combined

    # ---- Adaptive alpha ----
    def adaptive_search(self, query: str) -> Tuple[List[Tuple[Document, float]], float]:
        """Choose alpha based on query characteristics."""
        q_len = len(query)

        # Short query (<=3 chars): favor dense (semantic)
        # Medium query (4-8 chars): balanced
        # Long query (>8 chars): favor sparse (exact match)
        if q_len <= 3:
            alpha = 0.7
        elif q_len <= 8:
            alpha = 0.5
        else:
            alpha = 0.3

        return self.hybrid_search(query, alpha), alpha


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("第11章 混合搜索演示 (Hybrid Search Demo)")
    print("=" * 60)

    # Build index
    print("\n[索引] 构建混合检索索引...")
    retriever = HybridRetriever().fit(SAMPLE_DOCS)
    print(f"   文档数: {len(SAMPLE_DOCS)}")
    print(f"   词汇量: {len(retriever.idf)}")

    # Test queries
    test_queries = [
        ("短查询", "肺癌"),
        ("中查询", "肿瘤靶向药物"),
        ("长查询", "恒瑞医药研发的抗肿瘤靶向药物有哪些"),
    ]

    print(f"\n{'=' * 60}")
    print("混合搜索对比")
    print(f"{'=' * 60}")

    for qtype, query in test_queries:
        print(f"\n--- {qtype}: \"{query}\" (长度={len(query)}) ---")

        # Dense only
        dense = retriever.dense_search(query)
        # Sparse only
        sparse = retriever.sparse_search(query)
        # Hybrid with adaptive alpha
        hybrid, alpha = retriever.adaptive_search(query)

        print(f"   自适应alpha = {alpha:.2f} (短→密集, 长→稀疏)")

        # Build comparison table
        print(f"\n   {'文档':<20} {'密集':<10} {'稀疏':<10} {'混合':<10}")
        print(f"   {'-'*50}")
        dense_map = dict(dense)
        sparse_map = dict(sparse)
        hybrid_map = dict(hybrid)
        for doc in SAMPLE_DOCS:
            ds = dense_map.get(doc, 0.0)
            ss = sparse_map.get(doc, 0.0)
            hs = hybrid_map.get(doc, 0.0)
            if ds > 0 or ss > 0 or hs > 0:
                print(f"   {doc.title:<20} {ds:<10.4f} {ss:<10.4f} {hs:<10.4f}")

    # Summary
    print(f"\n{'=' * 60}")
    print("自适应Alpha策略")
    print(f"{'=' * 60}")
    print(f"   短查询 (<=3字符): alpha=0.7 (密集70% + 稀疏30%)")
    print(f"   中查询 (4-8字符): alpha=0.5 (密集50% + 稀疏50%)")
    print(f"   长查询 (>8字符): alpha=0.3 (密集30% + 稀疏70%)")
    print(f"\n   密集搜索 (TF-IDF): 适合语义匹配，对同义词效果好")
    print(f"   稀疏搜索 (BM25): 适合关键词精确匹配，对长查询效果好")
    print(f"   混合搜索: 结合两者优势，通过alpha动态调整权重")

    print("\n" + "=" * 60)
    print("演示完成。")
    print("=" * 60)


if __name__ == "__main__":
    main()
