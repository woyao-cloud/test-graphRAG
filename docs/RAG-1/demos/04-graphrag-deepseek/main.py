"""
Demo 4: GraphRAG + DeepSeek Configuration & Dry-Run
=====================================================
Shows how to configure GraphRAG for DeepSeek, prepare documents,
run indexing (or dry-run), and query with local/global/drift/basic methods.
"""

import json
import os
import tempfile
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# GraphRAG Configuration for DeepSeek
# ---------------------------------------------------------------------------

@dataclass
class GraphRAGConfig:
    """Complete GraphRAG configuration targeting DeepSeek API."""
    # LLM configuration
    llm_api_key: str = "${GRAPHRAG_API_KEY}"
    llm_api_base: str = "https://api.deepseek.com"
    llm_model: str = "deepseek-chat"
    llm_max_tokens: int = 4000
    llm_temperature: float = 0.0
    llm_request_timeout: float = 120.0

    # Embedding configuration
    embedding_api_key: str = "${GRAPHRAG_API_KEY}"
    embedding_api_base: str = "https://api.deepseek.com"
    embedding_model: str = "deepseek-text-embedding"
    embedding_dim: int = 1024

    # Graph configuration
    graph_chunk_size: int = 1200
    graph_chunk_overlap: int = 100
    graph_max_cluster_size: int = 10
    graph_min_cluster_size: int = 2

    # Local search
    local_search_llm_max_tokens: int = 2000
    local_search_mmr_threshold: float = 0.5

    # Global search
    global_search_llm_max_tokens: int = 2000
    global_search_max_map_iterations: int = 3

    # Drift search
    drift_search_llm_max_tokens: int = 2000
    drift_search_max_drift_iterations: int = 3

    # Basic search
    basic_search_llm_max_tokens: int = 1000

    # Data
    root_dir: str = "."
    input_dir: str = "input"

    def to_yaml_lines(self) -> str:
        """Serialize to YAML-like format for display."""
        lines = ["# GraphRAG Configuration for DeepSeek", "---"]
        lines.append("llm:")
        lines.append(f"  api_key: {self.llm_api_key}")
        lines.append(f"  api_base: {self.llm_api_base}")
        lines.append(f"  model: {self.llm_model}")
        lines.append(f"  max_tokens: {self.llm_max_tokens}")
        lines.append(f"  temperature: {self.llm_temperature}")
        lines.append(f"  request_timeout: {self.llm_request_timeout}")
        lines.append("")
        lines.append("embeddings:")
        lines.append(f"  api_key: {self.embedding_api_key}")
        lines.append(f"  api_base: {self.embedding_api_base}")
        lines.append(f"  model: {self.embedding_model}")
        lines.append(f"  dimension: {self.embedding_dim}")
        lines.append("")
        lines.append("graphs:")
        lines.append(f"  chunk_size: {self.graph_chunk_size}")
        lines.append(f"  chunk_overlap: {self.graph_chunk_overlap}")
        lines.append(f"  max_cluster_size: {self.graph_max_cluster_size}")
        lines.append(f"  min_cluster_size: {self.graph_min_cluster_size}")
        lines.append("")
        lines.append("local_search:")
        lines.append(f"  llm_max_tokens: {self.local_search_llm_max_tokens}")
        lines.append(f"  mmr_threshold: {self.local_search_mmr_threshold}")
        lines.append("")
        lines.append("global_search:")
        lines.append(f"  llm_max_tokens: {self.global_search_llm_max_tokens}")
        lines.append(f"  max_map_iterations: {self.global_search_max_map_iterations}")
        lines.append("")
        lines.append("drift_search:")
        lines.append(f"  llm_max_tokens: {self.drift_search_llm_max_tokens}")
        lines.append(f"  max_drift_iterations: {self.drift_search_max_drift_iterations}")
        lines.append("")
        lines.append("basic_search:")
        lines.append(f"  llm_max_tokens: {self.basic_search_llm_max_tokens}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Sample Test Documents
# ---------------------------------------------------------------------------

TEST_DOCUMENTS = [
    {"id": "doc1", "title": "GraphRAG Overview",
     "content": "GraphRAG is a structured approach to RAG that builds a knowledge graph "
                "from documents. It extracts entities and relationships, enabling "
                "multi-hop reasoning and global summarization."},
    {"id": "doc2", "title": "Local Search",
     "content": "Local search in GraphRAG focuses on a specific entity or concept. "
                "It traverses the graph neighborhood to gather context for precise answers. "
                "Uses MMR (Maximum Marginal Relevance) for diversity."},
    {"id": "doc3", "title": "Global Search",
     "content": "Global search in GraphRAG aggregates across communities in the graph. "
                "It maps themes, generates community summaries, and produces comprehensive "
                "answers that span the entire knowledge base."},
    {"id": "doc4", "title": "Drift Search",
     "content": "Drift search is an exploratory mode in GraphRAG. It starts from a query, "
                "iteratively follows graph connections, and 'drifts' through related "
                "concepts to discover novel connections and serendipitous findings."},
    {"id": "doc5", "title": "DeepSeek Integration",
     "content": "GraphRAG can be configured with DeepSeek models as the LLM backend. "
                "Set the api_base to https://api.deepseek.com and use deepseek-chat "
                "or deepseek-reasoner for generation tasks."},
]


# ---------------------------------------------------------------------------
# Simulated Indexing Pipeline
# ---------------------------------------------------------------------------

def run_indexing_pipeline(config: GraphRAGConfig, documents: List[Dict]) -> Dict[str, Any]:
    """Simulate the GraphRAG indexing pipeline (dry-run if no API key)."""
    has_api_key = os.environ.get("GRAPHRAG_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")

    print("\n--- Indexing Pipeline ---")
    print(f"  Documents: {len(documents)}")
    print(f"  Chunk size: {config.graph_chunk_size}")
    print(f"  Chunk overlap: {config.graph_chunk_overlap}")
    print(f"  API key present: {bool(has_api_key)}")

    if has_api_key:
        print("  Mode: LIVE (API calls would be made)")
    else:
        print("  Mode: DRY-RUN (no API key — simulation only)")

    # Simulated steps
    print("\n  [1/4] Splitting documents into chunks...")
    total_chunks = 0
    for doc in documents:
        content = doc["content"]
        chunks = max(1, len(content) // config.graph_chunk_size + 1)
        total_chunks += chunks
        print(f"    {doc['title']}: {chunks} chunk(s)")

    print(f"  Total chunks: {total_chunks}")

    print("\n  [2/4] Extracting entities and relationships...")
    entities = {
        "GraphRAG", "Local Search", "Global Search", "Drift Search",
        "DeepSeek", "Knowledge Graph", "MMR", "Community",
    }
    print(f"    Entities found: {len(entities)}")
    print(f"    Sample: {', '.join(sorted(entities)[:5])}...")

    print("\n  [3/4] Building graph communities (clustering)...")
    communities = 2
    print(f"    Communities formed: {communities}")

    print("\n  [4/4] Generating community summaries...")
    print("    Community 0: RAG methods (local, global, drift)")
    print("    Community 1: Infrastructure (DeepSeek, graph, embeddings)")

    result = {
        "status": "completed",
        "total_chunks": total_chunks,
        "entities": list(entities),
        "communities": communities,
        "mode": "dry-run" if not has_api_key else "live",
    }
    return result


# ---------------------------------------------------------------------------
# Query Methods
# ---------------------------------------------------------------------------

def basic_search(query: str, documents: List[Dict]) -> str:
    """Basic search: simple keyword matching over documents."""
    results = []
    for doc in documents:
        if any(word.lower() in doc["content"].lower() for word in query.split()):
            results.append(f"- {doc['title']}: {doc['content'][:100]}...")
    if not results:
        return "No direct matches found."
    return "Basic Search Results:\n" + "\n".join(results)


def local_search(query: str, index_result: Dict) -> str:
    """Local search: focus on entities related to the query."""
    entities = index_result.get("entities", [])
    matched = [e for e in entities if any(w.lower() in e.lower() for w in query.split())]
    if matched:
        return (f"Local Search (entity-focused):\n"
                f"Query: {query}\n"
                f"Related entities: {', '.join(matched)}\n"
                f"Context: The graph neighborhood around these entities contains "
                f"relevant relationships for a precise answer.")
    return "Local Search: No specific entity match found, expanding scope."


def global_search(query: str, index_result: Dict) -> str:
    """Global search: community summaries."""
    communities = index_result.get("communities", 0)
    return (f"Global Search (community-level):\n"
            f"Query: {query}\n"
            f"Community count: {communities}\n"
            f"Summary: Across {communities} communities, the query relates to "
            f"GraphRAG's multi-modal search capabilities. Community summaries "
            f"provide a comprehensive view spanning all documents.")


def drift_search(query: str, index_result: Dict) -> str:
    """Drift search: exploratory, follows graph edges."""
    return (f"Drift Search (exploratory):\n"
            f"Query: {query}\n"
            f"Starting from the query concept, drifting through graph edges...\n"
            f"Discovery path: RAG methods -> Knowledge Graph -> DeepSeek -> "
            f"GraphRAG integration\n"
            f"Novel connection found: Drift search itself is a GraphRAG mode "
            f"that enables serendipitous discovery across domains.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Demo 4: GraphRAG + DeepSeek Configuration & Dry-Run")
    print("=" * 60)

    # 1. Show configuration
    print("\n>>> 1. GraphRAG Configuration for DeepSeek")
    config = GraphRAGConfig()
    print(config.to_yaml_lines())

    # 2. Prepare documents
    print("\n>>> 2. Test Documents")
    for doc in TEST_DOCUMENTS:
        print(f"  [{doc['id']}] {doc['title']}: {doc['content'][:60]}...")

    # 3. Indexing
    print("\n>>> 3. Indexing Pipeline")
    index_result = run_indexing_pipeline(config, TEST_DOCUMENTS)

    # 4. Query methods
    print("\n>>> 4. Query Methods")
    query = "How does GraphRAG enable different search modes?"

    print(f"\nQuery: \"{query}\"")
    print()

    print("--- Basic Search ---")
    print(basic_search(query, TEST_DOCUMENTS))
    print()

    print("--- Local Search ---")
    print(local_search(query, index_result))
    print()

    print("--- Global Search ---")
    print(global_search(query, index_result))
    print()

    print("--- Drift Search ---")
    print(drift_search(query, index_result))
    print()

    print("=" * 60)
    print("GraphRAG with DeepSeek: configure, index, query with 4 search modes.")
    print("=" * 60)


if __name__ == "__main__":
    main()
