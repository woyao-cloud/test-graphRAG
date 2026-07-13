"""
ch10-basic-rag: Minimal RAG system using Milvus
- Pure Python TF-IDF-like embedding (character frequency)
- Milvus vector storage and similarity search
- Simple answer generation by extracting key sentences
"""
import math
from collections import Counter
from pymilvus import MilvusClient

# ---------------------------------------------------------------------------
# 1. Sample Chinese documents (5 medical texts about drugs/diseases)
# ---------------------------------------------------------------------------
DOCUMENTS = [
    "阿司匹林是一种非甾体抗炎药，常用于缓解轻度至中度疼痛、退烧和抗炎。长期低剂量使用可预防心脑血管疾病。",
    "布洛芬通过抑制环氧化酶活性，减少前列腺素合成，从而发挥镇痛、抗炎和解热作用。常见的副作用包括胃肠道不适。",
    "青霉素是最早发现的抗生素之一，通过破坏细菌细胞壁合成来杀灭革兰氏阳性菌。对青霉素过敏者禁用。",
    "高血压是一种常见的慢性病，指动脉血压持续升高。主要危险因素包括高盐饮食、肥胖、缺乏运动和遗传因素。",
    "2型糖尿病以胰岛素抵抗和胰岛素分泌不足为特征。治疗包括生活方式干预、口服降糖药物和胰岛素注射。",
]

QUERY = "哪些药物可以治疗炎症和疼痛？"

# ---------------------------------------------------------------------------
# 2. Simple TF-IDF-like embedding using character frequency
# ---------------------------------------------------------------------------
def char_freq_vector(text: str, vocab: list[str]) -> list[float]:
    """Compute a TF-IDF-like vector using character frequencies.

    For each character in the vocabulary, compute:
      tf = count_in_text / len(text)
      idf = log(total_docs / docs_containing_char)  (smoothed)
    Returns a normalized vector.
    """
    text_len = len(text) if len(text) > 0 else 1
    counter = Counter(text)

    tfidf = []
    for ch in vocab:
        tf = counter.get(ch, 0) / text_len
        idf = 1.0  # simplified: uniform idf weight
        tfidf.append(tf * idf)

    # L2 normalize
    norm = math.sqrt(sum(v * v for v in tfidf))
    if norm > 0:
        tfidf = [v / norm for v in tfidf]
    return tfidf


def build_vocab(docs: list[str]) -> list[str]:
    """Build character vocabulary from documents."""
    chars: set[str] = set()
    for doc in docs:
        chars.update(doc)
    return sorted(chars)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# 3. Store vectors + text in Milvus collection
# ---------------------------------------------------------------------------
def setup_milvus(client: MilvusClient, collection_name: str, dim: int):
    """Create Milvus collection and insert documents."""
    # Drop if exists
    if client.has_collection(collection_name):
        client.drop_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        dimension=dim,
        auto_id=False,
    )


# ---------------------------------------------------------------------------
# 4. Generate answer by extracting key sentences
# ---------------------------------------------------------------------------
def generate_answer(query: str, contexts: list[str]) -> str:
    """Simple answer generation: extract sentences that overlap with query words."""
    query_chars = set(query)
    scored_sentences: list[tuple[float, str]] = []

    for text in contexts:
        for sentence in text.replace("。", "。|").split("|"):
            sentence = sentence.strip()
            if not sentence:
                continue
            sentence_chars = set(sentence)
            overlap = len(query_chars & sentence_chars)
            if len(query_chars) > 0:
                score = overlap / len(query_chars)
            else:
                score = 0
            scored_sentences.append((score, sentence))

    scored_sentences.sort(key=lambda x: -x[0])
    top = [s for _, s in scored_sentences[:3] if _ > 0]
    if not top:
        return "无法根据已有信息生成回答。"

    return "根据检索到的信息：" + "；".join(top) + "。"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("ch10-basic-rag: Minimal RAG System with Milvus")
    print("=" * 60)

    # Build vocabulary
    vocab = build_vocab(DOCUMENTS)
    dim = len(vocab)
    print(f"\n[1] Vocabulary built: {dim} unique characters")

    # Embed documents
    doc_vectors = [char_freq_vector(d, vocab) for d in DOCUMENTS]
    print(f"[2] Document vectors computed ({len(doc_vectors)} docs)")

    # Connect to Milvus
    client = MilvusClient(uri="http://localhost:19530")
    collection_name = "ch10_basic_rag"

    setup_milvus(client, collection_name, dim)
    print(f"[3] Milvus collection '{collection_name}' created (dim={dim})")

    # Insert documents
    data = [
        {"id": i, "vector": doc_vectors[i], "text": DOCUMENTS[i]}
        for i in range(len(DOCUMENTS))
    ]
    insert_result = client.insert(collection_name=collection_name, data=data)
    print(f"[4] Inserted {len(data)} documents into Milvus (insert count: {insert_result})")

    # Query
    print(f"\n[5] User Query: \"{QUERY}\"")
    query_vec = char_freq_vector(QUERY, vocab)

    # Search Milvus
    search_result = client.search(
        collection_name=collection_name,
        data=[query_vec],
        limit=3,
        output_fields=["text"],
    )

    print(f"\n[6] Top-3 Retrieved Contexts:")
    retrieved_texts: list[str] = []
    for i, hit in enumerate(search_result[0]):
        text = hit["entity"]["text"]
        score = hit["distance"]
        retrieved_texts.append(text)
        print(f"    #{i+1} (score={score:.4f}): {text[:60]}...")

    # Generate answer
    print(f"\n[7] Generated Answer:")
    answer = generate_answer(QUERY, retrieved_texts)
    print(f"    {answer}")

    # Cleanup
    client.drop_collection(collection_name)
    print(f"\n[8] Collection '{collection_name}' dropped. Done.")


if __name__ == "__main__":
    main()
