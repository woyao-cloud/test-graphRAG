"""
ch11-langchain-rag: LangChain + Milvus integration pattern

Demonstrates how LangChain's Milvus wrapper connects to Milvus as a vector store.
Since we cannot call a real LLM here, we simulate embeddings with numpy and
show the full setup pipeline:
  1. Create simulated embeddings
  2. Connect Milvus via LangChain's Milvus wrapper
  3. Add documents to the store
  4. Perform similarity search
  5. Use the retriever interface
"""
import numpy as np
from langchain_community.vectorstores import Milvus
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

# ---------------------------------------------------------------------------
# 1. Simulated embeddings
# ---------------------------------------------------------------------------
EMBEDDING_DIM = 128

class SimulatedEmbeddings(Embeddings):
    """A fake embedding model that returns deterministic numpy vectors.

    In a real application you would use e.g. OpenAIEmbeddings or
    HuggingFaceEmbeddings. Here we generate a stable random vector per
    document/query so we can demonstrate the Milvus integration without
    depending on any external model.
    """

    def __init__(self, dim: int = EMBEDDING_DIM):
        self.dim = dim
        # Deterministic seed so results are repeatable
        self._rng = np.random.RandomState(42)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return a fixed random vector for each text (same seed = same vec)."""
        # Use hash of text to seed so same text always gets same vector
        vectors = []
        for t in texts:
            seed = hash(t) % (2**31)
            rng = np.random.RandomState(seed)
            vec = rng.randn(self.dim).astype(np.float64)
            vec = vec / np.linalg.norm(vec)  # L2 normalize
            vectors.append(vec.tolist())
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """Return embedding for a single query text."""
        return self.embed_documents([text])[0]


# ---------------------------------------------------------------------------
# 2. Documents
# ---------------------------------------------------------------------------
DOCUMENTS = [
    "LangChain is a framework for developing applications powered by language models.",
    "Milvus is a high-performance vector database built for scalable similarity search.",
    "RAG (Retrieval-Augmented Generation) combines retrieval with LLM generation.",
    "Vector embeddings convert text into numerical representations for semantic search.",
    "LangChain provides a Milvus wrapper that integrates Milvus as a vector store backend.",
]

QUERIES = [
    "What is Milvus used for?",
    "How does RAG work?",
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("ch11-langchain-rag: LangChain + Milvus Integration Pattern")
    print("=" * 70)

    # Step 1: Create embeddings
    print("\n[Step 1] Creating simulated embeddings (dim={})".format(EMBEDDING_DIM))
    embeddings = SimulatedEmbeddings()
    sample_embedding = embeddings.embed_query("test")
    print(f"         Sample embedding vector (first 5 values): {sample_embedding[:5]}")
    print(f"         Embedding dimension: {len(sample_embedding)}")

    # Step 2: Create Document objects
    print("\n[Step 2] Creating Document objects")
    docs = [Document(page_content=t) for t in DOCUMENTS]
    for i, d in enumerate(docs):
        print(f"         Doc {i+1}: {d.page_content}")

    # Step 3: Connect Milvus via LangChain wrapper
    print("\n[Step 3] Connecting to Milvus via LangChain's Milvus wrapper")
    print("         from langchain_community.vectorstores import Milvus")
    vector_store = Milvus.from_documents(
        documents=docs,
        embedding=embeddings,
        connection_args={"uri": "http://localhost:19530"},
        collection_name="ch11_langchain_demo",
        drop_old=True,
    )
    print("         Collection 'ch11_langchain_demo' created with {} documents".format(
        vector_store.collection_name
    ))

    # Step 4: Similarity search
    print("\n[Step 4] Performing similarity search")
    for q in QUERIES:
        print(f"\n         Query: \"{q}\"")
        results = vector_store.similarity_search(q, k=2)
        for i, doc in enumerate(results):
            print(f"         Result #{i+1}: {doc.page_content}")

    # Step 5: Similarity search with score
    print("\n[Step 5] Similarity search with relevance scores")
    results_with_score = vector_store.similarity_search_with_score(QUERIES[0], k=3)
    for i, (doc, score) in enumerate(results_with_score):
        print(f"         #{i+1} (score={score:.4f}): {doc.page_content}")

    # Step 6: Retriever interface
    print("\n[Step 6] Using the retriever interface")
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    print("         retriever = vector_store.as_retriever(search_kwargs={'k': 2})")
    retrieved_docs = retriever.invoke(QUERIES[1])
    print(f'         Query: "{QUERIES[1]}"')
    for i, doc in enumerate(retrieved_docs):
        print(f"         Retrieved #{i+1}: {doc.page_content}")

    # Step 7: Add more documents
    print("\n[Step 7] Adding more documents after initial setup")
    new_docs = [
        Document(page_content="LangChain Milvus wrapper supports both sync and async operations."),
        Document(page_content="Vector search enables semantic similarity matching across large datasets."),
    ]
    vector_store.add_documents(new_docs)
    print(f"         Added {len(new_docs)} new documents. Total in store: {len(DOCUMENTS) + len(new_docs)}")

    # Verify by searching again
    results = vector_store.similarity_search("async vector search", k=2)
    print(f'         Verify search: "{results[0].page_content}"')

    # Cleanup
    vector_store.collection.drop()
    print("\n[Step 8] Collection dropped. Demo complete.")


if __name__ == "__main__":
    main()
