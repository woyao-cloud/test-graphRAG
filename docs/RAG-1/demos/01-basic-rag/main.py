"""
Demo 1: 基础 RAG 实现 — Basic RAG Pipeline
=============================================
Shows the full RAG flow: load documents → chunk → build TF-IDF index → search → generate answer.
Uses only Python stdlib + math (no external dependencies).
"""

import math
import re
from collections import Counter
from typing import List, Tuple


# ---------------------------------------------------------------------------
# 1. Document Loading
# ---------------------------------------------------------------------------

def load_sample_documents() -> List[str]:
    """Return a small corpus of sample documents about RAG and AI topics."""
    return [
        "Retrieval-Augmented Generation (RAG) is a technique that combines "
        "information retrieval with text generation. It allows LLMs to access "
        "external knowledge during inference.",

        "The RAG pipeline consists of two main stages: retrieval and generation. "
        "First, relevant documents are retrieved from a knowledge base. "
        "Then, a language model generates an answer conditioned on those documents.",

        "TF-IDF (Term Frequency-Inverse Document Frequency) is a classic "
        "information retrieval method. It weighs terms by how often they appear "
        "in a document versus how rare they are across the corpus.",

        "Chunking strategies are critical in RAG systems. Common approaches "
        "include fixed-size chunking, sentence splitting, and semantic chunking. "
        "The choice affects both retrieval quality and generation quality.",

        "Vector databases like FAISS, Pinecone, and Weaviate store embeddings "
        "for fast similarity search. They are the backbone of modern RAG retrieval.",
    ]


# ---------------------------------------------------------------------------
# 2. Chunking
# ---------------------------------------------------------------------------

def chunk_document(text: str, chunk_size: int = 60, overlap: int = 10) -> List[str]:
    """
    Split a document into fixed-size chunks with overlap.
    Uses word boundaries to avoid cutting words.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
        if start >= len(words):
            break
    return chunks


def chunk_all_documents(docs: List[str], chunk_size: int = 60, overlap: int = 10) -> List[str]:
    """Chunk all documents and collect them into a flat list."""
    all_chunks = []
    for i, doc in enumerate(docs):
        chunks = chunk_document(doc, chunk_size, overlap)
        all_chunks.extend(chunks)
        print(f"  Doc {i+1}: {len(chunks)} chunk(s)")
    return all_chunks


# ---------------------------------------------------------------------------
# 3. TF-IDF Index
# ---------------------------------------------------------------------------

def tokenize(text: str) -> List[str]:
    """Lowercase, split on non-alphanumeric, remove short tokens."""
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    return [t for t in tokens if len(t) >= 2]


def compute_tf(tokens: List[str]) -> Counter:
    """Term frequency for a single document."""
    return Counter(tokens)


def compute_idf(corpus_tokens: List[List[str]]) -> dict:
    """Inverse document frequency: log(N / df(t))."""
    N = len(corpus_tokens)
    doc_freq: Counter = Counter()
    for tokens in corpus_tokens:
        for t in set(tokens):
            doc_freq[t] += 1
    idf = {t: math.log((N + 1) / (df + 1)) + 1 for t, df in doc_freq.items()}
    return idf


def build_tfidf_index(chunks: List[str]) -> Tuple[List[Counter], dict, List[str]]:
    """Build TF-IDF index from chunks. Returns (tf_list, idf_dict, vocab)."""
    print("\n--- Building TF-IDF Index ---")
    corpus_tokens = [tokenize(ch) for ch in chunks]
    tf_list = [compute_tf(tokens) for tokens in corpus_tokens]
    idf = compute_idf(corpus_tokens)
    vocab = sorted(idf.keys())
    print(f"  Vocabulary size: {len(vocab)}")
    print(f"  Number of chunks: {len(chunks)}")
    return tf_list, idf, vocab


def compute_tfidf_score(query_tokens: List[str], tf: Counter, idf: dict) -> float:
    """Compute TF-IDF cosine similarity score (simplified dot product)."""
    score = 0.0
    for term in query_tokens:
        if term in idf:
            tf_val = tf.get(term, 0)
            score += tf_val * idf[term]
    return score


# ---------------------------------------------------------------------------
# 4. Search
# ---------------------------------------------------------------------------

def search(query: str, chunks: List[str], tf_list: List[Counter], idf: dict, top_k: int = 3) -> List[Tuple[int, float, str]]:
    """Search chunks by TF-IDF relevance."""
    print(f"\n--- Searching: \"{query}\" ---")
    query_tokens = tokenize(query)
    scores = []
    for idx, tf in enumerate(tf_list):
        score = compute_tfidf_score(query_tokens, tf, idf)
        scores.append((idx, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    results = [(idx, round(score, 4), chunks[idx]) for idx, score in scores[:top_k] if score > 0]
    return results


# ---------------------------------------------------------------------------
# 5. Generation (simulated)
# ---------------------------------------------------------------------------

def generate_answer(query: str, retrieved_chunks: List[str]) -> str:
    """Simulate answer generation based on retrieved context."""
    print("\n--- Generating Answer ---")
    context = "\n".join(f"- {chunk[:100]}..." for chunk in retrieved_chunks)
    answer = (
        f"Based on the retrieved documents:\n{context}\n\n"
        f"Answer: The query \"{query}\" relates to the following key points "
        f"found in {len(retrieved_chunks)} relevant document passages. "
        f"RAG systems retrieve external knowledge and feed it to an LLM "
        f"to produce grounded, factually-informed responses."
    )
    return answer


# ---------------------------------------------------------------------------
# Main: Full Pipeline
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Demo 1: 基础 RAG 实现 — Basic RAG Pipeline")
    print("=" * 60)

    # Step 1: Load
    print("\n>>> Step 1: Load Sample Documents")
    docs = load_sample_documents()
    print(f"  Loaded {len(docs)} documents")

    # Step 2: Chunk
    print("\n>>> Step 2: Chunk Documents")
    chunks = chunk_all_documents(docs)

    # Step 3: Build TF-IDF Index
    tf_list, idf, vocab = build_tfidf_index(chunks)

    # Step 4: Search
    query = "How does RAG combine retrieval with generation?"
    results = search(query, chunks, tf_list, idf, top_k=2)

    if not results:
        print("  No relevant results found.")
        return

    retrieved_chunks = []
    for idx, score, text in results:
        retrieved_chunks.append(text)
        print(f"  [{idx}] score={score:.4f}: {text[:80]}...")

    # Step 5: Generate
    answer = generate_answer(query, retrieved_chunks)
    print(answer)

    print("\n" + "=" * 60)
    print("Pipeline complete: load → chunk → index → search → generate")
    print("=" * 60)


if __name__ == "__main__":
    main()
