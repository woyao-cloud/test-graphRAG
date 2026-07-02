"""Configuration model for GraphRAG-KG.

Extends Microsoft GraphRAG's native settings with Neo4j, ingestion,
pipeline orchestration, and query engine configuration.

All settings are Pydantic models with validation and sensible defaults.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ============================================================================
# Neo4j Configuration
# ============================================================================


class Neo4jConfig(BaseModel):
    """Neo4j connection and sync settings."""

    uri: str = Field(
        default="bolt://localhost:7687",
        description="Neo4j Bolt URI",
    )
    username: str = Field(
        default="neo4j",
        description="Neo4j username",
    )
    password: str = Field(
        default="password",
        description="Neo4j password",
    )
    database: str = Field(
        default="neo4j",
        description="Neo4j database name",
    )

    # Connection pool
    max_connection_lifetime: int = Field(
        default=3600,
        ge=60,
        description="Max connection lifetime in seconds",
    )
    max_connection_pool_size: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Max connection pool size",
    )
    connection_acquisition_timeout: int = Field(
        default=60,
        ge=5,
        description="Connection acquisition timeout in seconds",
    )

    # Sync settings
    sync_batch_size: int = Field(
        default=1000,
        ge=100,
        le=100000,
        description="Batch size for Neo4j data sync",
    )
    sync_create_indexes: bool = Field(
        default=True,
        description="Auto-create indexes and constraints on sync",
    )
    store_embeddings: bool = Field(
        default=False,
        description="Store embeddings in Neo4j (use LanceDB for vectors)",
    )

    @property
    def auth(self) -> tuple[str, str]:
        """Return (username, password) tuple for driver auth."""
        return (self.username, self.password)


# ============================================================================
# Ingestion Configuration
# ============================================================================


class IngestionConfig(BaseModel):
    """Document ingestion settings."""

    source_directories: list[str] = Field(
        default=["./documents"],
        description="Directories to scan for documents",
    )
    file_patterns: list[str] = Field(
        default=["**/*.pdf", "**/*.md", "**/*.txt", "**/*.html"],
        description="Glob patterns for document discovery",
    )
    recursive: bool = Field(
        default=True,
        description="Scan directories recursively",
    )
    encoding: str = Field(
        default="utf-8",
        description="Default text encoding",
    )
    clean_html: bool = Field(
        default=True,
        description="Strip HTML tags during parsing",
    )
    extract_metadata: bool = Field(
        default=True,
        description="Extract document metadata",
    )
    max_file_size_mb: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Maximum file size in MB",
    )


# ============================================================================
# Pipeline Configuration
# ============================================================================


class PipelineConfig(BaseModel):
    """Pipeline orchestration settings."""

    auto_index_on_ingest: bool = Field(
        default=False,
        description="Automatically run indexing after ingestion",
    )
    incremental: bool = Field(
        default=False,
        description="Use incremental indexing",
    )
    backup_previous_output: bool = Field(
        default=True,
        description="Backup previous output before re-indexing",
    )
    max_workers: int = Field(
        default=4,
        ge=1,
        le=32,
        description="Max parallel workers for pipeline steps",
    )


# ============================================================================
# Query Configuration
# ============================================================================


class QueryConfig(BaseModel):
    """Query engine settings."""

    default_method: Literal["local", "global", "drift", "basic", "auto"] = Field(
        default="local",
        description="Default search method",
    )
    response_type: str = Field(
        default="Multiple Paragraphs",
        description="LLM response style",
    )
    max_context_tokens: int = Field(
        default=12000,
        ge=1000,
        le=128000,
        description="Max tokens in query context window",
    )
    include_sources: bool = Field(
        default=True,
        description="Include source citations in answers",
    )
    streaming: bool = Field(
        default=False,
        description="Stream responses token-by-token",
    )
    cache_ttl_seconds: int = Field(
        default=3600,
        ge=0,
        description="Query result cache TTL (0 = disabled)",
    )


# ============================================================================
# GraphRAG-KG Top-Level Config
# ============================================================================


class KGConfig(BaseModel):
    """Top-level configuration for GraphRAG-KG.

    Extends Microsoft GraphRAG's settings with project-level configuration
    for Neo4j storage, document ingestion, pipeline orchestration, and querying.
    """

    # Project identity
    project_name: str = Field(
        default="my-knowledge-graph",
        min_length=1,
        max_length=128,
        description="Project name",
    )
    description: str = Field(
        default="",
        max_length=1024,
        description="Project description",
    )

    # Sub-configs
    neo4j: Neo4jConfig = Field(
        default_factory=Neo4jConfig,
        description="Neo4j connection and sync settings",
    )
    ingestion: IngestionConfig = Field(
        default_factory=IngestionConfig,
        description="Document ingestion settings",
    )
    pipeline: PipelineConfig = Field(
        default_factory=PipelineConfig,
        description="Pipeline orchestration settings",
    )
    query: QueryConfig = Field(
        default_factory=QueryConfig,
        description="Query engine settings",
    )

    # GraphRAG native settings (passthrough)
    graphrag_settings: dict[str, Any] = Field(
        default_factory=dict,
        description="Raw GraphRAG settings (passed through to graphrag library)",
    )

    # Model configuration
    chat_model: str = Field(
        default="gpt-4.1",
        description="Default chat/completion model",
    )
    chat_model_provider: str = Field(
        default="openai",
        description="Model provider (openai, anthropic, azure, ollama, deepseek)",
    )
    embedding_model: str = Field(
        default="text-embedding-3-large",
        description="Default embedding model",
    )
    embedding_model_provider: str = Field(
        default="openai",
        description="Embedding model provider",
    )
    api_key: str = Field(
        default="",
        description="API key for LLM provider",
    )
    api_base: str = Field(
        default="",
        description="API base URL (for custom endpoints)",
    )

    # Project paths
    root_dir: Path = Field(
        default=Path("."),
        description="Project root directory",
    )
    input_dir: Path = Field(
        default=Path("input"),
        description="Directory for input text files",
    )
    output_dir: Path = Field(
        default=Path("output"),
        description="Directory for graphrag output",
    )
    prompts_dir: Path = Field(
        default=Path("prompts"),
        description="Directory for prompt templates",
    )
    cache_dir: Path = Field(
        default=Path("cache"),
        description="Directory for cache data",
    )
    logs_dir: Path = Field(
        default=Path("logs"),
        description="Directory for log files",
    )

    @field_validator("root_dir", mode="before")
    @classmethod
    def resolve_root_dir(cls, v: Any) -> Path:
        """Resolve root_dir to absolute path."""
        return Path(v).resolve()

    @model_validator(mode="after")
    def resolve_relative_paths(self) -> "KGConfig":
        """Resolve relative paths against root_dir."""
        root = self.root_dir
        for field_name in ["input_dir", "output_dir", "prompts_dir", "cache_dir", "logs_dir"]:
            value = getattr(self, field_name)
            if not value.is_absolute():
                object.__setattr__(self, field_name, (root / value).resolve())
        return self

    def to_graphrag_config(self) -> Any:
        """Generate a graphrag 3.x GraphRagConfig object.

        Uses the installed graphrag library's config model directly.
        """
        import os
        from graphrag.config.models.graph_rag_config import GraphRagConfig
        from graphrag_llm.config.model_config import ModelConfig
        from graphrag_input.input_config import InputConfig
        from graphrag_storage.storage_config import StorageConfig, StorageType
        from graphrag_vectors.vector_store_config import VectorStoreConfig, VectorStoreType
        from graphrag.config.models.reporting_config import ReportingConfig
        from graphrag_cache.cache_config import CacheConfig

        api_key = self.api_key or os.environ.get("GRAPHRAG_API_KEY", "") or None
        api_base = self.api_base or os.environ.get("GRAPHRAG_API_BASE", "") or None

        # Separate config for embedding model (may use different provider)
        embed_api_key = os.environ.get("GRAPHRAG_API_KEY_EMBED") or api_key
        embed_api_base = os.environ.get("GRAPHRAG_API_BASE_EMBED") or api_base or None

        # Determine model string for LiteLLM
        # When api_base is set (non-OpenAI endpoint), use "openai/model" format
        chat_model_str = self.chat_model
        if api_base and "api.openai.com" not in api_base and not chat_model_str.startswith("openai/"):
            chat_model_str = f"openai/{self.chat_model}"

        embed_model_str = self.embedding_model
        if embed_api_base and "api.openai.com" not in embed_api_base and "/" not in embed_model_str:
            # Detect provider prefix: ollama for localhost, openai for others
            prefix = "ollama" if ("localhost" in embed_api_base or "127.0.0.1" in embed_api_base) else "openai"
            embed_model_str = f"{prefix}/{self.embedding_model}"

        # Strip /v1 suffix from API bases (LiteLLM appends it automatically)
        chat_api_base = api_base.rstrip("/v1").rstrip("/") if api_base else None
        embed_api_base_clean = embed_api_base.rstrip("/v1").rstrip("/") if embed_api_base else None

        # Build model configs
        completion_models = {
            "default_completion_model": ModelConfig(
                type="litellm",
                model_provider="openai",
                model=chat_model_str,
                api_key=api_key,
                api_base=chat_api_base,
            ),
        }
        embedding_models = {
            "default_embedding_model": ModelConfig(
                type="litellm",
                model_provider="openai",
                model=embed_model_str,
                api_key=embed_api_key,
                api_base=embed_api_base_clean,
            ),
        }

        # Build GraphRagConfig
        config = GraphRagConfig(
            completion_models=completion_models,
            embedding_models=embedding_models,
            input=InputConfig(
                type="text",
                encoding=self.ingestion.encoding,
                file_pattern=".*\\.txt$",
            ),
            input_storage=StorageConfig(
                type=StorageType.File,
                base_dir=str(self.input_dir),
            ),
            output_storage=StorageConfig(
                type=StorageType.File,
                base_dir=str(self.output_dir),
            ),
            cache=CacheConfig(
                type="json",
                storage=StorageConfig(
                    type=StorageType.File,
                    base_dir=str(self.cache_dir),
                ),
            ),
            reporting=ReportingConfig(
                type="file",
                base_dir=str(self.logs_dir),
            ),
            vector_store=VectorStoreConfig(
                type=VectorStoreType.LanceDB,
                db_uri=str(self.output_dir / "lancedb"),
            ),
        )

        return config

    @classmethod
    def defaults(cls) -> "KGConfig":
        """Return a config with all defaults."""
        return cls()


# ============================================================================
# Config Profile Definitions
# ============================================================================


def get_fast_profile() -> KGConfig:
    """Fast indexing profile — fewer gleanings, smaller chunks, lower cost."""
    return KGConfig(
        project_name="graphrag-fast",
        description="Fast indexing profile for quick iterations",
        chat_model="gpt-4.1-mini",
        embedding_model="text-embedding-3-small",
        pipeline= PipelineConfig(max_workers=8),
        query=QueryConfig(default_method="basic", include_sources=False, max_context_tokens=4000),
        graphrag_settings={
            "chunking": {
                "type": "tokens",
                "encoding_model": "cl100k_base",
                "size": 600,
                "overlap": 50,
            },
            "extract_graph": {
                "max_gleanings": 0,
                "entity_types": ["organization", "person", "location"],
            },
            "cluster_graph": {"max_cluster_size": 5},
        },
    )


def get_production_profile() -> KGConfig:
    """Production profile — full extraction, large context, high quality."""
    return KGConfig(
        project_name="graphrag-production",
        description="Production profile for comprehensive indexing",
        chat_model="gpt-4.1",
        embedding_model="text-embedding-3-large",
        pipeline=PipelineConfig(
            max_workers=4,
            backup_previous_output=True,
        ),
        query=QueryConfig(
            default_method="drift",
            max_context_tokens=24000,
        ),
        graphrag_settings={
            "chunking": {
                "type": "tokens",
                "encoding_model": "cl100k_base",
                "size": 1200,
                "overlap": 100,
            },
            "extract_graph": {
                "max_gleanings": 2,
                "entity_types": [
                    "organization", "person", "location", "event",
                    "concept", "technology", "product",
                ],
            },
            "cluster_graph": {"max_cluster_size": 10},
            "snapshots": {"embeddings": True, "graphml": True},
        },
    )


PROFILES = {
    "default": KGConfig.defaults,
    "fast": get_fast_profile,
    "production": get_production_profile,
}
