"""GraphRAG Knowledge Graph Q&A System.

A Python-based GraphRAG system for Knowledge Graph Q&A, using Microsoft's
graphrag library as the foundation with Neo4j + Milvus hybrid storage.
"""

from .core import milvus_store  # Register custom Milvus vector store

__version__ = "0.1.0"
