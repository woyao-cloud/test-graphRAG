"""
第 11 章 Demo：层级检索与混合检索

演示多种检索策略的混合使用：
  稠密检索 + 稀疏检索 + RRF 融合
  Small-to-Big 检索
  Step-back Prompting

可独立运行，无需外部依赖（内置模拟 Embedding 和 LLM）。

用法：
  python hybrid_search.py
  python hybrid_search.py --query "恒瑞医药生产哪些药品？"
  python hybrid_search.py --query "抗肿瘤药物有哪些？" --method small-to-big
"""

import argparse
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class Chunk:
    id: str
    text: str
    token_count: int = 0
    metadata: dict = field(default_factory=dict)
    parent_id: Optional[str] = None
    children: list[str] = field(default_factory=list)


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float
    source: str = ""  # dense / sparse / hybrid


@dataclass
class Document:
    id: str
    title: str
    text: str


# ============================================================================
# Mock Embedding
# ============================================================================


class MockEmbedding:
    """模拟文本嵌入（基于字符频率的伪向量）。"""

    def embed(self, text: str) -> list[float]:
        features = [0.0] * 32
        for i, ch in enumerate(text[:200]):
            features[hash(ch) % 32] += 1.0
        norm = math.sqrt(sum(v * v for v in features)) or 1.0
        return [v / norm for v in features]

    def cosine_similarity(self, a: list[float], b: list[float]) -> float:
        return sum(av * bv for av, bv in zip(a, b))


# ============================================================================
# Dense Retriever
# ============================================================================


class DenseRetriever:
    """稠密检索（向量语义搜索）。"""

    def __init__(self, embedding: MockEmbedding):
        self.embedding = embedding
        self.chunks: list[Chunk] = []
        self.embeddings: list[list[float]] = []

    def index(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.embeddings = [self.embedding.embed(c.text) for c in chunks]

    def search(self, query: str, top_k: int = 5) -> list[ScoredChunk]:
        q_emb = self.embedding.embed(query)
        scored = []
        for i, chunk in enumerate(self.chunks):
            score = self.embedding.cosine_similarity(q_emb, self.embeddings[i])
            scored.append(ScoredChunk(chunk=chunk, score=score, source="dense"))
        scored.sort(key=lambda x: -x.score)
        return scored[:top_k]


# ============================================================================
# Sparse Retriever (BM25-like)
# ============================================================================


class SparseRetriever:
    """稀疏检索（基于词频的 BM25 模拟）。"""

    def __init__(self):
        self.chunks: list[Chunk] = []
        self.doc_freq: dict[str, int] = defaultdict(int)
        self.term_freqs: list[dict[str, int]] = []
        self.avg_dl: float = 0.0
        self.k1 = 1.5
        self.b = 0.75

    def index(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.term_freqs = []
        total_len = 0

        for chunk in chunks:
            tokens = self._tokenize(chunk.text)
            tf = defaultdict(int)
            for t in tokens:
                tf[t] += 1
            self.term_freqs.append(dict(tf))
            total_len += len(tokens)

            for t in set(tokens):
                self.doc_freq[t] += 1

        self.avg_dl = total_len / len(chunks) if chunks else 1.0

    def search(self, query: str, top_k: int = 5) -> list[ScoredChunk]:
        query_tokens = self._tokenize(query)
        if not query_tokens or not self.chunks:
            return []

        n_docs = len(self.chunks)
        scored = []

        for i, chunk in enumerate(self.chunks):
            tf = self.term_freqs[i]
            dl = sum(tf.values())
            score = 0.0

            for qt in query_tokens:
                if qt not in self.doc_freq:
                    continue
                df = self.doc_freq[qt]
                idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
                term_freq = tf.get(qt, 0)
                tf_norm = (
                    term_freq * (self.k1 + 1)
                ) / (term_freq + self.k1 * (1 - self.b + self.b * dl / self.avg_dl))
                score += idf * tf_norm

            scored.append(ScoredChunk(chunk=chunk, score=score, source="sparse"))

        scored.sort(key=lambda x: -x.score)
        return scored[:top_k]

    def _tokenize(self, text: str) -> list[str]:
        """简单分词（中文逐字 + 英文按词）。"""
        tokens = []
        # 英文单词
        for word in re.findall(r"\w+", text):
            tokens.append(word.lower())
        # 中文字符（bi-gram）
        chars = re.findall(r"[\u4e00-\u9fff]", text)
        for i in range(len(chars) - 1):
            tokens.append(chars[i] + chars[i + 1])
        return tokens


# ============================================================================
# Rank Fusion
# ============================================================================


def reciprocal_rank_fusion(
    result_lists: list[list[ScoredChunk]],
    k: int = 60,
    top_k: int = 10,
) -> list[ScoredChunk]:
    """RRF 融合。"""
    rrf_scores: dict[str, tuple[float, ScoredChunk]] = {}

    for rank_list in result_lists:
        for rank, scored in enumerate(rank_list, start=1):
            cid = scored.chunk.id
            if cid not in rrf_scores:
                rrf_scores[cid] = (0.0, scored)
            current, _ = rrf_scores[cid]
            rrf_scores[cid] = (current + 1.0 / (k + rank), scored)

    sorted_results = sorted(rrf_scores.values(), key=lambda x: -x[0])
    return [ScoredChunk(chunk=s.chunk, score=sc, source="hybrid")
            for sc, s in sorted_results[:top_k]]


# ============================================================================
# Hybrid Retriever
# ============================================================================


class HybridRetriever:
    """混合检索器（Dense + Sparse + RRF）。"""

    def __init__(self, embedding: MockEmbedding):
        self.dense = DenseRetriever(embedding)
        self.sparse = SparseRetriever()

    def index(self, chunks: list[Chunk]):
        self.dense.index(chunks)
        self.sparse.index(chunks)

    def search(
        self,
        query: str,
        top_k: int = 5,
        alpha: float = 0.5,
    ) -> list[ScoredChunk]:
        """加权融合检索。"""
        dense_results = self.dense.search(query, top_k=top_k * 2)
        sparse_results = self.sparse.search(query, top_k=top_k * 2)

        # 分数归一化
        dense_results = self._normalize(dense_results)
        sparse_results = self._normalize(sparse_results)

        # 加权融合
        fused: dict[str, ScoredChunk] = {}
        for s in dense_results:
            fused[s.chunk.id] = ScoredChunk(
                chunk=s.chunk,
                score=alpha * s.score,
                source="hybrid",
            )
        for s in sparse_results:
            if s.chunk.id in fused:
                fused[s.chunk.id].score += (1 - alpha) * s.score
            else:
                fused[s.chunk.id] = ScoredChunk(
                    chunk=s.chunk,
                    score=(1 - alpha) * s.score,
                    source="hybrid",
                )

        results = sorted(fused.values(), key=lambda x: -x.score)
        return results[:top_k]

    def search_rrf(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[ScoredChunk]:
        """RRF 融合检索。"""
        dense_results = self.dense.search(query, top_k=top_k * 2)
        sparse_results = self.sparse.search(query, top_k=top_k * 2)
        return reciprocal_rank_fusion([dense_results, sparse_results], top_k=top_k)

    def _normalize(self, results: list[ScoredChunk]) -> list[ScoredChunk]:
        if not results:
            return results
        scores = [s.score for s in results]
        min_s, max_s = min(scores), max(scores)
        if max_s == min_s:
            return results
        for s in results:
            s.score = (s.score - min_s) / (max_s - min_s)
        return results


# ============================================================================
# Small-to-Big Retriever
# ============================================================================


class SmallToBigRetriever:
    """Small-to-Big 检索器。"""

    def __init__(self, embedding: MockEmbedding):
        self.embedding = embedding
        self.small_chunks: list[Chunk] = []
        self.big_chunks: dict[str, Chunk] = {}
        self.parent_map: dict[str, str] = {}  # small_id -> big_id

    def index(self, documents: list[str]):
        """构建 Small + Big 双层索引。"""
        for doc_id, text in enumerate(documents):
            # Big chunks (parent)
            big_parts = self._split_by_chars(text, 600)
            for bp in big_parts:
                big_chunk = Chunk(
                    id=f"big_{doc_id}_{len(self.big_chunks)}",
                    text=bp,
                    token_count=len(bp) // 2,
                )
                self.big_chunks[big_chunk.id] = big_chunk

                # Small chunks (children)
                small_parts = self._split_by_chars(bp, 120)
                for sp in small_parts:
                    small_chunk = Chunk(
                        id=f"sml_{doc_id}_{len(self.small_chunks)}",
                        text=sp,
                        token_count=len(sp) // 2,
                        parent_id=big_chunk.id,
                    )
                    self.small_chunks.append(small_chunk)
                    self.parent_map[small_chunk.id] = big_chunk.id

    def search(self, query: str, top_k: int = 3) -> list[Chunk]:
        """检索：从小块搜索，映射到大块返回。"""
        q_emb = self.embedding.embed(query)

        # 1. 在小块中搜索
        scored_small = []
        for chunk in self.small_chunks:
            c_emb = self.embedding.embed(chunk.text)
            score = sum(av * bv for av, bv in zip(q_emb, c_emb))
            scored_small.append((score, chunk))

        scored_small.sort(key=lambda x: -x[0])
        top_small = scored_small[:top_k * 2]

        # 2. 映射到大块（去重）
        seen_big = set()
        result = []
        for _, small_chunk in top_small:
            big_id = self.parent_map.get(small_chunk.id)
            if big_id and big_id not in seen_big:
                seen_big.add(big_id)
                big_chunk = self.big_chunks.get(big_id)
                if big_chunk:
                    result.append(big_chunk)

        return result[:top_k]

    def _split_by_chars(self, text: str, max_chars: int) -> list[str]:
        """按字符数拆分。"""
        chunks = []
        for i in range(0, len(text), max_chars):
            chunks.append(text[i:i + max_chars])
        return chunks


# ============================================================================
# Step-back Retriever
# ============================================================================


class StepBackRetriever:
    """Step-back Prompting 增强检索。"""

    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever

    def _generate_stepback_query(self, question: str) -> str:
        """模拟生成 step-back 问题（实际中调用 LLM）。"""
        # 基于规则的 step-back 映射
        stepback_map = {
            "恒瑞医药": "制药企业的产品线和业务布局",
            "紫杉醇": "抗肿瘤药物的分类和作用机制",
            "抗肿瘤": "肿瘤治疗药物的分类和临床应用",
            "制药": "中国制药行业的产业链结构",
        }

        for keyword, general in stepback_map.items():
            if keyword in question:
                return general

        # 默认 step-back
        return f"{question} 的一般原理和背景知识"

    def search(self, question: str, top_k: int = 5) -> list[ScoredChunk]:
        """Step-back 增强检索。"""
        stepback_q = self._generate_stepback_query(question)

        # Step-back 结果（通用知识）
        stepback_results = self.retriever.search(stepback_q, top_k=top_k // 2 + 1)

        # 原始结果（精确知识）
        direct_results = self.retriever.search(question, top_k=top_k // 2 + 1)

        # 融合
        seen = set()
        results = []
        for r in stepback_results + direct_results:
            if r.chunk.id not in seen:
                seen.add(r.chunk.id)
                r.source = "stepback" if r in stepback_results[:len(stepback_results)] else r.source
                results.append(r)

        return results[:top_k]


# ============================================================================
# Sample Data
# ============================================================================


SAMPLE_DOCUMENTS = [
    # doc_0
    """恒瑞医药是中国领先的制药企业，主要专注于抗肿瘤药物的研发和生产。
公司成立于1997年，总部位于江苏省连云港市。恒瑞医药的主要产品包括
注射用紫杉醇、奥沙利铂和卡培他滨，这些药物广泛应用于非小细胞肺癌、
乳腺癌和结直肠癌等疾病的治疗。公司每年投入大量资金用于新药研发，
拥有多项国家发明专利。恒瑞医药的产品已出口至美国、欧洲、日本等
多个国家和地区。""",

    # doc_1
    """紫杉醇是一种重要的抗肿瘤化疗药物，属于微管抑制剂类药物。
它通过促进微管蛋白聚合、抑制微管解聚而发挥抗肿瘤作用。紫杉醇
主要用于治疗非小细胞肺癌、乳腺癌、卵巢癌等多种实体瘤。注射用
紫杉醇是恒瑞医药的主要产品之一，采用先进的脂质体制剂技术，
提高了药物的靶向性和疗效。""",

    # doc_2
    """齐鲁制药是中国另一家大型制药企业，总部位于山东省济南市。
齐鲁制药主要生产顺铂和卡培他滨等抗肿瘤药物。顺铂是应用最广泛
的铂类抗肿瘤药物之一，用于治疗肺癌、卵巢癌、膀胱癌等多种实体瘤。
卡培他滨是一种口服氟尿嘧啶类抗肿瘤药物，用于结直肠癌和乳腺癌的
治疗。""",
]


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="层级检索与混合检索 Demo")
    parser.add_argument("--query", default="恒瑞医药生产哪些药品？", help="查询问题")
    parser.add_argument(
        "--method",
        choices=["hybrid", "weighted", "rrf", "small-to-big", "stepback", "all"],
        default="all",
        help="检索方法",
    )
    args = parser.parse_args()

    embedding = MockEmbedding()
    hybrid = HybridRetriever(embedding)

    # ============================================================
    # Build index
    # ============================================================
    # Flat chunks for hybrid search
    all_chunks = []
    for doc_id, text in enumerate(SAMPLE_DOCUMENTS):
        # Split each doc into chunks
        chunk_size = 200
        for i in range(0, len(text), chunk_size):
            chunk_text = text[i:i + chunk_size].strip()
            if chunk_text:
                all_chunks.append(Chunk(
                    id=f"doc{doc_id}_chunk{i // chunk_size}",
                    text=chunk_text,
                    metadata={"doc_id": doc_id},
                ))
    hybrid.index(all_chunks)

    # Small-to-Big index
    stb = SmallToBigRetriever(embedding)
    stb.index(SAMPLE_DOCUMENTS)

    # Step-back retriever
    stepback = StepBackRetriever(hybrid)

    query = args.query
    method = args.method

    print("=" * 60)
    print("层级检索与混合检索 Demo")
    print("=" * 60)
    print(f"Query: {query}\n")

    methods_to_run = (
        ["weighted", "rrf", "small-to-big", "stepback"]
        if method == "all"
        else [method]
    )

    for m in methods_to_run:
        print(f"\n{'─' * 50}")
        print(f"[Method: {m}]")
        print(f"{'─' * 50}")

        if m == "weighted":
            # 加权融合（α 从 0 到 1 变化）
            for alpha in [0.0, 0.3, 0.5, 0.7, 1.0]:
                results = hybrid.search(query, top_k=3, alpha=alpha)
                print(f"\n  alpha={alpha:.1f} (0=纯稀疏, 1=纯稠密):")
                for r in results:
                    print(f"    [{r.score:.4f}] {r.chunk.text[:80]}...")

        elif m == "rrf":
            results = hybrid.search_rrf(query, top_k=3)
            print("\n  RRF 融合结果 (Dense + Sparse):")
            for r in results:
                print(f"    [{r.score:.4f}] {r.chunk.text[:80]}...")

        elif m == "small-to-big":
            results = stb.search(query, top_k=3)
            print("\n  Small-to-Big 检索结果:")
            for i, chunk in enumerate(results):
                print(f"\n  Big Chunk #{i + 1} ({len(chunk.text)} chars):")
                print(f"    {chunk.text[:150]}...")

        elif m == "stepback":
            results = stepback.search(query, top_k=4)
            print(f"\n  Step-back query: {stepback._generate_stepback_query(query)}")
            print("  检索结果:")
            for r in results:
                print(f"    [{r.source}] [{r.score:.4f}] {r.chunk.text[:80]}...")

    # 对比 Dense vs Sparse vs Hybrid
    if method == "all":
        print(f"\n{'=' * 50}")
        print("召回结果对比 (Top-3)")
        print(f"{'=' * 50}")

        dense = embedding  # reuse
        for doc_id, text in enumerate(SAMPLE_DOCUMENTS):
            q_emb = dense.embed(query)
            d_emb = dense.embed(text[:200])
            score = sum(av * bv for av, bv in zip(q_emb, d_emb))
            print(f"\n  Doc {doc_id} (向量相似度: {score:.4f}):")
            print(f"    {text[:120]}...")


if __name__ == "__main__":
    main()
