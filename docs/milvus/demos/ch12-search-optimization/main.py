"""
ch12-search-optimization: RAG Search Optimization Techniques

Demonstrates four search optimization techniques:
  1. Adaptive TopK  - return more results for short queries, fewer for specific ones
  2. Score threshold filtering - filter results below confidence threshold
  3. Reranking simulation - reorder results based on keyword overlap with query
  4. Multi-vector search - search with multiple query variations and merge
"""
import math
import random
from collections import Counter
from pymilvus import MilvusClient

# ---------------------------------------------------------------------------
# 30 sample documents (medical + tech mix)
# ---------------------------------------------------------------------------
DOCUMENTS = [
    "阿司匹林用于缓解轻度至中度疼痛和退烧",
    "布洛芬具有镇痛、抗炎和解热作用",
    "青霉素通过破坏细菌细胞壁杀灭革兰氏阳性菌",
    "高血压患者应控制盐摄入并坚持运动",
    "2型糖尿病需要生活方式干预和药物治疗",
    "对乙酰氨基酚是常用的解热镇痛药物",
    "他汀类药物可降低胆固醇预防心血管疾病",
    "奥美拉唑用于治疗胃酸过多和胃食管反流",
    "二甲双胍是2型糖尿病的一线口服降糖药",
    "氯吡格雷用于预防动脉粥样硬化血栓事件",
    "氨氯地平属于钙通道阻滞剂类降压药",
    "左甲状腺素用于治疗甲状腺功能减退症",
    "阿莫西林是广谱青霉素类抗生素",
    "头孢克洛属于第二代头孢菌素类抗生素",
    "甲硝唑用于治疗厌氧菌感染和原虫感染",
    "泼尼松是一种糖皮质激素具有抗炎免疫抑制作用",
    "地西泮属于苯二氮卓类镇静催眠药物",
    "氟西汀是选择性5-羟色胺再摄取抑制剂抗抑郁药",
    "奥氮平用于治疗精神分裂症和双相情感障碍",
    "多奈哌齐用于治疗阿尔茨海默病改善认知功能",
    "机器学习是人工智能的核心分支之一",
    "深度学习使用多层神经网络进行特征学习",
    "自然语言处理让计算机理解人类语言",
    "计算机视觉使机器能够识别和处理图像",
    "向量数据库用于高效存储和检索嵌入向量",
    "Milvus支持多种索引类型加速向量搜索",
    "RAG系统结合检索和生成提高回答准确性",
    "Transformer模型是NLP领域的基础架构",
    "强化学习通过与环境交互学习最优策略",
    "知识图谱以图结构存储实体及其关系信息",
]

# ---------------------------------------------------------------------------
# Simple embedding: TF-IDF-like character frequency vector
# ---------------------------------------------------------------------------
def build_vocab(docs: list[str]) -> list[str]:
    chars: set[str] = set()
    for d in docs:
        chars.update(d)
    return sorted(chars)


def char_freq_vector(text: str, vocab: list[str]) -> list[float]:
    counter = Counter(text)
    text_len = max(len(text), 1)
    vec = [counter.get(ch, 0) / text_len for ch in vocab]
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm > 0 else vec


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na > 0 and nb > 0 else 0.0


# ---------------------------------------------------------------------------
# Optimization 1: Adaptive TopK
# ---------------------------------------------------------------------------
def adaptive_topk(query: str, base_k: int = 5, min_k: int = 2, max_k: int = 10) -> int:
    """Return more results for short/vague queries, fewer for long/specific ones.

    Rationale: short queries tend to be broader and benefit from more candidates;
    long/specific queries usually have fewer relevant documents.
    """
    length = len(query)
    if length <= 4:
        return max_k
    elif length <= 10:
        return base_k
    else:
        return min_k


# ---------------------------------------------------------------------------
# Optimization 2: Score threshold filtering
# ---------------------------------------------------------------------------
def filter_by_threshold(
    results: list[dict], threshold: float = 0.15
) -> list[dict]:
    """Filter out results below the similarity score threshold."""
    return [r for r in results if r["distance"] >= threshold]


# ---------------------------------------------------------------------------
# Optimization 3: Reranking by keyword overlap
# ---------------------------------------------------------------------------
def rerank_by_keyword_overlap(
    query: str, results: list[dict]
) -> list[dict]:
    """Rerank results based on keyword/character overlap with the query.

    This simulates a cross-encoder or LLM-based reranker by computing
    character-level Jaccard similarity between query and each result text.
    """
    query_chars = set(query)
    for r in results:
        text = r["entity"]["text"]
        text_chars = set(text)
        intersection = len(query_chars & text_chars)
        union = len(query_chars | text_chars)
        jaccard = intersection / union if union > 0 else 0.0
        # Blend original distance (0.6) with keyword overlap (0.4)
        r["rerank_score"] = 0.6 * r["distance"] + 0.4 * jaccard

    results.sort(key=lambda r: -r["rerank_score"])
    return results


# ---------------------------------------------------------------------------
# Optimization 4: Multi-vector search
# ---------------------------------------------------------------------------
def multi_vector_search(
    client: MilvusClient,
    collection_name: str,
    query: str,
    vocab: list[str],
    k: int = 3,
) -> list[dict]:
    """Search with multiple query variations and merge results.

    Generates 3 query variations:
      1. Original query
      2. Query with repeated keywords (emphasizing key terms)
      3. Query truncated to first half (focusing on leading terms)

    Results from each variation are merged by document ID, with scores
    averaged across variations that retrieved them.
    """
    # Generate query variations
    words = list(query)
    mid = len(words) // 2
    variations = [
        query,
        query + query[:mid],  # repeat first half
        "".join(words[:mid]),  # first half only
    ]

    merged: dict[int, dict] = {}

    for var in variations:
        qvec = char_freq_vector(var, vocab)
        results = client.search(
            collection_name=collection_name,
            data=[qvec],
            limit=k * 2,  # fetch extra for merging
            output_fields=["text"],
        )[0]

        for hit in results:
            doc_id = hit["id"]
            if doc_id not in merged:
                merged[doc_id] = {
                    "id": doc_id,
                    "entity": hit["entity"],
                    "scores": [],
                }
            merged[doc_id]["scores"].append(hit["distance"])

    # Average scores across variations and sort
    final = []
    for doc_id, data in merged.items():
        avg_score = sum(data["scores"]) / len(data["scores"])
        final.append({
            "id": doc_id,
            "distance": avg_score,
            "entity": data["entity"],
        })

    final.sort(key=lambda r: -r["distance"])
    return final[:k]


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
def setup_collection(client: MilvusClient, name: str, dim: int):
    if client.has_collection(name):
        client.drop_collection(name)
    client.create_collection(
        collection_name=name,
        dimension=dim,
        auto_id=False,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("ch12-search-optimization: RAG Search Optimization Techniques")
    print("=" * 70)

    # Build vocab and embed all documents
    vocab = build_vocab(DOCUMENTS)
    dim = len(vocab)
    doc_vectors = [char_freq_vector(d, vocab) for d in DOCUMENTS]

    # Connect to Milvus
    client = MilvusClient(uri="http://localhost:19530")
    collection_name = "ch12_search_opt"
    setup_collection(client, collection_name, dim)

    # Insert documents
    data = [
        {"id": i, "vector": doc_vectors[i], "text": DOCUMENTS[i]}
        for i in range(len(DOCUMENTS))
    ]
    client.insert(collection_name=collection_name, data=data)
    print(f"\nCollection '{collection_name}' created with {len(DOCUMENTS)} documents (dim={dim})")

    # Queries to test
    queries = [
        ("Short", "疼痛"),           # broad, short
        ("Medium", "抗生素治疗感染"),   # medium
        ("Long", "治疗高血压和糖尿病的常用药物有哪些"),  # long, specific
    ]

    for label, query in queries:
        print(f"\n{'=' * 70}")
        print(f"Query [{label}]: \"{query}\"")
        print(f"{'=' * 70}")

        query_vec = char_freq_vector(query, vocab)

        # --- Baseline ---
        baseline = client.search(
            collection_name=collection_name,
            data=[query_vec],
            limit=5,
            output_fields=["text"],
        )[0]

        print(f"\n--- Baseline (top-5) ---")
        for i, h in enumerate(baseline):
            print(f"  #{i+1} (score={h['distance']:.4f}): {h['entity']['text']}")

        # --- 1. Adaptive TopK ---
        k = adaptive_topk(query)
        adaptive = client.search(
            collection_name=collection_name,
            data=[query_vec],
            limit=k,
            output_fields=["text"],
        )[0]

        print(f"\n--- [Opt1] Adaptive TopK (k={k}) ---")
        for i, h in enumerate(adaptive):
            print(f"  #{i+1} (score={h['distance']:.4f}): {h['entity']['text']}")

        # --- 2. Score threshold filtering ---
        threshold = 0.12
        filtered = filter_by_threshold(baseline, threshold=threshold)

        print(f"\n--- [Opt2] Score Threshold (>{threshold}) ---")
        if filtered:
            for i, h in enumerate(filtered):
                print(f"  #{i+1} (score={h['distance']:.4f}): {h['entity']['text']}")
        else:
            print(f"  (no results above threshold {threshold})")

        # --- 3. Reranking ---
        reranked = rerank_by_keyword_overlap(query, baseline.copy())

        print(f"\n--- [Opt3] Reranking (keyword overlap) ---")
        for i, h in enumerate(reranked):
            print(f"  #{i+1} (rerank={h['rerank_score']:.4f}): {h['entity']['text']}")

        # --- 4. Multi-vector search ---
        multi = multi_vector_search(client, collection_name, query, vocab, k=3)

        print(f"\n--- [Opt4] Multi-Vector Search (3 variations) ---")
        for i, h in enumerate(multi):
            print(f"  #{i+1} (avg_score={h['distance']:.4f}): {h['entity']['text']}")

    # --- Comparison summary ---
    print(f"\n{'=' * 70}")
    print("Comparison Summary")
    print(f"{'=' * 70}")
    print(f"""
  Technique               | Benefit
  ------------------------|-------------------------------------------
  Adaptive TopK           | Broad queries get more candidates;
                          | specific queries get fewer but more relevant
  Score Threshold         | Filters out low-confidence noise;
                          | improves precision at cost of recall
  Reranking               | Reorders results using finer-grained
                          | similarity (simulated keyword overlap)
  Multi-Vector Search     | Multiple query perspectives reduce the risk
                          | of missing relevant documents
""")

    # Cleanup
    client.drop_collection(collection_name)
    print(f"\nCollection '{collection_name}' dropped. Demo complete.")


if __name__ == "__main__":
    main()
