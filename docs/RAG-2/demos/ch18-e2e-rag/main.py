"""
ch18-e2e-rag — Complete end-to-end RAG pipeline demo.

Pipeline: load_docs() → chunk_documents() → SimpleTfidfVectorizer.fit() →
search() → generate_answer().  Includes 6 sample Chinese pharma documents
and an evaluation harness with recall, precision, MRR, and latency metrics.
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any


# ===================================================================
# 1.  Documents (Chinese pharma)
# ===================================================================

SAMPLE_DOCS: dict[str, str] = {
    "阿司匹林": (
        "阿司匹林（乙酰水杨酸）是一种非甾体抗炎药（NSAID），广泛用于镇痛、退热和心血管保护。"
        "常见剂量：81毫克（低剂量/婴儿阿司匹林）和325毫克（常规强度）。"
        "成人每日最大剂量为4000毫克。半衰期约为3小时（低剂量）至15小时（高剂量）。"
        "阿司匹林通过不可逆地抑制环氧合酶（COX）发挥抗血小板作用。"
    ),
    "布洛芬": (
        "布洛芬是一种非甾体抗炎药，用于缓解疼痛、发热和炎症。"
        "成人常规剂量为200-400毫克，每4-6小时一次。"
        "非处方每日最大剂量为1200毫克，处方剂量可达3200毫克。"
        "半衰期为2-4小时，起效时间为30-60分钟。"
        "布洛芬通过抑制COX-1和COX-2酶减少前列腺素合成。"
    ),
    "对乙酰氨基酚": (
        "对乙酰氨基酚（扑热息痛）是一种镇痛和解热药，不属于NSAID。"
        "成人剂量为325-650毫克，每4-6小时一次。"
        "每日最大剂量为3000毫克（部分来源建议4000毫克）。"
        "半衰期为2-3小时。过量使用可导致严重肝损伤。"
        "与酒精同时使用会增加肝毒性风险。"
    ),
    "药物相互作用": (
        "NSAIDs和对乙酰氨基酚可以交替用于疼痛管理。"
        "同时服用阿司匹林和布洛芬会降低阿司匹林的心脏保护作用。"
        "酒精与对乙酰氨基酚合用会增加肝毒性风险。"
        "NSAIDs在肾病患者或有胃肠道出血史的患者中应谨慎使用。"
        "华法林与NSAIDs合用增加出血风险。"
    ),
    "他汀类药物": (
        "他汀类药物是HMG-CoA还原酶抑制剂，用于降低胆固醇。"
        "阿托伐他汀半衰期约14小时，瑞舒伐他汀约19小时。"
        "辛伐他汀半衰期约3小时，普伐他汀约1.5小时。"
        "常见副作用包括肌肉疼痛、肝酶升高。"
        "他汀类药物应在晚上服用以最大化降脂效果。"
    ),
    "抗生素使用原则": (
        "抗生素用于治疗细菌感染，对病毒感染无效。"
        "常见抗生素包括青霉素类、头孢菌素类、大环内酯类和氟喹诺酮类。"
        "抗生素耐药性是一个全球性健康威胁。"
        "应完成全程治疗，即使症状改善也不应提前停药。"
        "阿莫西林是一种广谱青霉素类抗生素。"
    ),
}


# ===================================================================
# 2.  Chunking
# ===================================================================

@dataclass
class Chunk:
    doc_id: str
    text: str
    index: int = 0


def chunk_documents(
    docs: dict[str, str],
    chunk_size: int = 100,
    overlap: int = 20,
) -> list[Chunk]:
    """Fixed-size character chunking with overlap."""
    chunks: list[Chunk] = []
    for doc_id, text in docs.items():
        start = 0
        idx = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(Chunk(doc_id=doc_id, text=text[start:end], index=idx))
            idx += 1
            start += chunk_size - overlap
    return chunks


# ===================================================================
# 3.  Simple TF-IDF Vectorizer
# ===================================================================

@dataclass
class SimpleTfidfVectorizer:
    """Minimal TF-IDF implementation with only stdlib."""

    _vocab: dict[str, int] = field(default_factory=dict, init=False)
    _idf: dict[str, float] = field(default_factory=dict, init=False)
    _chunks: list[Chunk] = field(default_factory=list, init=False)
    _tfidf_matrix: list[list[float]] = field(default_factory=list, init=False)

    # ── tokenisation ────────────────────────────────────────────────

    def _tokenize(self, text: str) -> list[str]:
        """Simple Chinese-aware tokenisation by character bigrams + English words."""
        tokens: list[str] = []

        # Extract English words
        for word in re.findall(r"[a-zA-Z]+", text):
            if len(word) >= 2:
                tokens.append(word.lower())

        # Extract Chinese character bigrams (simple approximation)
        chars = re.findall(r"[一-鿿]", text)
        for i in range(len(chars) - 1):
            bigram = chars[i] + chars[i + 1]
            tokens.append(bigram)

        # Also add single chars (for short docs)
        for ch in chars:
            tokens.append(ch)

        return tokens

    # ── fit ─────────────────────────────────────────────────────────

    def fit(self, chunks: list[Chunk]) -> SimpleTfidfVectorizer:
        """Build vocabulary and IDF from a list of Chunks."""
        self._chunks = chunks

        # Build vocabulary and DF
        df: Counter[str] = Counter()
        for chunk in chunks:
            tokens = set(self._tokenize(chunk.text))
            for tok in tokens:
                df[tok] += 1

        self._vocab = {tok: idx for idx, (tok, _) in enumerate(df.most_common())}

        # Compute IDF
        n_docs = len(chunks)
        self._idf = {
            tok: math.log((n_docs + 1) / (count + 1)) + 1
            for tok, count in df.items()
        }

        # Build TF-IDF matrix
        self._tfidf_matrix = [self._tf(chunk) for chunk in chunks]
        return self

    def _tf(self, chunk: Chunk) -> list[float]:
        """Compute TF-IDF vector for a single chunk."""
        tokens = self._tokenize(chunk.text)
        tf_raw = Counter(tokens)
        max_tf = max(tf_raw.values()) if tf_raw else 1

        vec = [0.0] * len(self._vocab)
        for tok, count in tf_raw.items():
            if tok in self._vocab:
                idx = self._vocab[tok]
                tf_norm = count / max_tf
                vec[idx] = tf_norm * self._idf.get(tok, 1.0)
        return vec

    # ── search ──────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 3) -> list[tuple[Chunk, float]]:
        """Return top-k chunks with cosine similarity scores."""
        if not self._tfidf_matrix:
            return []

        q_vec = self._tf(Chunk(doc_id="query", text=query))
        scores: list[tuple[int, float]] = []

        for idx, d_vec in enumerate(self._tfidf_matrix):
            sim = self._cosine_similarity(q_vec, d_vec)
            scores.append((idx, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [(self._chunks[idx], sim) for idx, sim in scores[:top_k]]

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        dot = sum(av * bv for av, bv in zip(a, b))
        na = math.sqrt(sum(av * av for av in a))
        nb = math.sqrt(sum(bv * bv for bv in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)


# ===================================================================
# 4.  Answer generation (simulated)
# ===================================================================

def generate_answer(query: str, top_chunks: list[tuple[Chunk, float]]) -> str:
    """Simulate answer generation from retrieved chunks."""
    if not top_chunks:
        return "无法检索到相关信息。"

    parts: list[str] = []
    seen = set()
    for chunk, score in top_chunks:
        if chunk.doc_id not in seen:
            parts.append(f"根据《{chunk.doc_id}》中的信息：{chunk.text}")
            seen.add(chunk.doc_id)

    return "\n\n".join(parts)


# ===================================================================
# 5.  Evaluation
# ===================================================================

@dataclass
class EvalCase:
    question: str
    ground_truth: str
    relevant_docs: list[str]


@dataclass
class SimpleEvaluator:
    """Minimal retrieval evaluator."""

    def recall(self, retrieved: list[str], relevant: set[str]) -> float:
        if not relevant:
            return 0.0
        hits = sum(1 for d in retrieved if d in relevant)
        return hits / len(relevant)

    def precision(self, retrieved: list[str], relevant: set[str]) -> float:
        if not retrieved:
            return 0.0
        hits = sum(1 for d in retrieved if d in relevant)
        return hits / len(retrieved)

    def mrr(self, results: list[tuple[list[str], set[str]]]) -> float:
        if not results:
            return 0.0
        rr_sum = 0.0
        for retrieved, relevant in results:
            for i, d in enumerate(retrieved, 1):
                if d in relevant:
                    rr_sum += 1.0 / i
                    break
        return rr_sum / len(results)


# ===================================================================
# 6.  main
# ===================================================================

def main() -> None:
    print("=" * 60)
    print("  End-to-End RAG Pipeline Demo")
    print("=" * 60)
    print()

    # ── load & chunk ─────────────────────────────────────────────────
    print("  [1/5] Loading documents ...")
    docs = SAMPLE_DOCS
    print(f"        {len(docs)} documents loaded.")
    print()

    print("  [2/5] Chunking documents ...")
    chunks = chunk_documents(docs, chunk_size=100, overlap=20)
    print(f"        {len(chunks)} chunks created (chunk_size=100, overlap=20).")
    print()

    # ── index ────────────────────────────────────────────────────────
    print("  [3/5] Indexing with TF-IDF ...")
    start = time.perf_counter()
    vectorizer = SimpleTfidfVectorizer()
    vectorizer.fit(chunks)
    elapsed = time.perf_counter() - start
    print(f"        Indexed in {elapsed * 1000:.1f} ms.")
    print(f"        Vocabulary size: {len(vectorizer._vocab)} tokens.")
    print()

    # ── queries ──────────────────────────────────────────────────────
    print("  [4/5] Running test queries ...")
    print()

    eval_cases: list[EvalCase] = [
        EvalCase(
            question="阿司匹林的半衰期是多少？",
            ground_truth="阿司匹林半衰期约3-15小时。",
            relevant_docs=["阿司匹林"],
        ),
        EvalCase(
            question="布洛芬和对乙酰氨基酚的区别",
            ground_truth="布洛芬是NSAID，对乙酰氨基酚不是NSAID。",
            relevant_docs=["布洛芬", "对乙酰氨基酚"],
        ),
        EvalCase(
            question="NSAIDs的副作用有哪些？",
            ground_truth="NSAIDs可导致胃肠道出血和肾损伤。",
            relevant_docs=["药物相互作用", "布洛芬"],
        ),
    ]

    all_retrieved_doc_ids: list[list[str]] = []
    all_relevant_sets: list[set[str]] = []
    all_latencies_ms: list[float] = []

    for i, ec in enumerate(eval_cases, 1):
        t0 = time.perf_counter()
        top_chunks = vectorizer.search(ec.question, top_k=3)
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000

        retrieved_ids = list(dict.fromkeys(c.doc_id for c, _ in top_chunks))
        all_retrieved_doc_ids.append(retrieved_ids)
        all_relevant_sets.append(set(ec.relevant_docs))
        all_latencies_ms.append(latency_ms)

        print(f"  Query {i}: {ec.question}")
        print(f"    Retrieved docs : {retrieved_ids}")
        print(f"    Latency        : {latency_ms:.2f} ms")

        answer = generate_answer(ec.question, top_chunks)
        print(f"    Generated answer:")
        for line in answer.split("\n"):
            print(f"      {line}")
        print()

    # ── evaluation ───────────────────────────────────────────────────
    print("  [5/5] Evaluation results")
    print()

    evaluator = SimpleEvaluator()

    all_recall: list[float] = []
    all_precision: list[float] = []

    for i, (retrieved, relevant) in enumerate(
        zip(all_retrieved_doc_ids, all_relevant_sets), 1
    ):
        r = evaluator.recall(retrieved, relevant)
        p = evaluator.precision(retrieved, relevant)
        all_recall.append(r)
        all_precision.append(p)
        print(f"    Query {i}:  Recall={r:.2f}  Precision={p:.2f}  "
              f"Retrieved={retrieved}  Relevant={relevant}")

    mrr_val = evaluator.mrr(list(zip(all_retrieved_doc_ids, all_relevant_sets)))

    avg_recall = sum(all_recall) / len(all_recall)
    avg_precision = sum(all_precision) / len(all_precision)

    print()
    print("-" * 60)
    print("  Summary")
    print("-" * 60)
    print(f"  Average Recall    : {avg_recall:.4f}")
    print(f"  Average Precision : {avg_precision:.4f}")
    print(f"  MRR               : {mrr_val:.4f}")
    print(f"  Number of queries : {len(eval_cases)}")
    print(f"  Avg latency/query : {sum(all_latencies_ms) / len(all_latencies_ms):.2f} ms")
    print()


if __name__ == "__main__":
    main()
