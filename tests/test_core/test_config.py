"""Tests for the KGConfig model and sub-configs."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from graphrag_kg.core.config import (
    KGConfig,
    Neo4jConfig,
    IngestionConfig,
    PipelineConfig,
    QueryConfig,
    get_fast_profile,
    get_production_profile,
)


class TestNeo4jConfig:
    """Tests for Neo4jConfig."""

    def test_defaults(self):
        config = Neo4jConfig()
        assert config.uri == "bolt://localhost:7687"
        assert config.username == "neo4j"
        assert config.password == "password"
        assert config.database == "neo4j"
        assert config.sync_batch_size == 1000

    def test_auth_tuple(self):
        config = Neo4jConfig(username="admin", password="secret")
        assert config.auth == ("admin", "secret")

    def test_custom_values(self):
        config = Neo4jConfig(
            uri="bolt://neo4j.example.com:7687",
            database="mydb",
            max_connection_pool_size=100,
        )
        assert config.uri == "bolt://neo4j.example.com:7687"
        assert config.database == "mydb"
        assert config.max_connection_pool_size == 100

    def test_invalid_pool_size(self):
        with pytest.raises(ValidationError):
            Neo4jConfig(max_connection_pool_size=0)

    def test_invalid_sync_batch(self):
        with pytest.raises(ValidationError):
            Neo4jConfig(sync_batch_size=50)  # below min of 100


class TestIngestionConfig:
    """Tests for IngestionConfig."""

    def test_defaults(self):
        config = IngestionConfig()
        assert "./documents" in config.source_directories
        assert "**/*.pdf" in config.file_patterns
        assert config.recursive is True

    def test_invalid_file_size(self):
        with pytest.raises(ValidationError):
            IngestionConfig(max_file_size_mb=0)


class TestQueryConfig:
    """Tests for QueryConfig."""

    def test_defaults(self):
        config = QueryConfig()
        assert config.default_method == "local"
        assert config.include_sources is True

    def test_invalid_method(self):
        with pytest.raises(ValidationError):
            QueryConfig(default_method="invalid_method")

    def test_valid_methods(self):
        for method in ["local", "global", "drift", "basic", "auto"]:
            config = QueryConfig(default_method=method)
            assert config.default_method == method


class TestKGConfig:
    """Tests for the top-level KGConfig."""

    def test_defaults(self, default_config):
        assert default_config.project_name == "my-knowledge-graph"
        assert default_config.chat_model == "gpt-4.1"
        assert default_config.embedding_model == "text-embedding-3-large"
        assert isinstance(default_config.neo4j, Neo4jConfig)
        assert isinstance(default_config.ingestion, IngestionConfig)
        assert isinstance(default_config.pipeline, PipelineConfig)
        assert isinstance(default_config.query, QueryConfig)

    def test_path_resolution(self):
        config = KGConfig(root_dir=Path("/tmp/test_project"))
        assert config.root_dir == Path("/tmp/test_project").resolve()
        assert config.input_dir == Path("/tmp/test_project/input").resolve()
        assert config.output_dir == Path("/tmp/test_project/output").resolve()

    def test_to_graphrag_config(self, default_config):
        graphrag_config = default_config.to_graphrag_config()
        assert "models" in graphrag_config
        assert "input" in graphrag_config
        assert "output" in graphrag_config
        assert "vector_store" in graphrag_config
        assert graphrag_config["vector_store"]["type"] == "lancedb"
        assert "default_chat_model" in graphrag_config["models"]
        assert graphrag_config["models"]["default_chat_model"]["model"] == "gpt-4.1"

    def test_project_name_validation(self):
        with pytest.raises(ValidationError):
            KGConfig(project_name="")  # too short

    def test_config_immutability_after_creation(self, default_config):
        """Sub-configs should be modifiable after creation."""
        default_config.neo4j.uri = "bolt://custom:7687"
        assert default_config.neo4j.uri == "bolt://custom:7687"


class TestProfiles:
    """Tests for predefined configuration profiles."""

    def test_fast_profile(self):
        config = get_fast_profile()
        assert config.chat_model == "gpt-4.1-mini"
        assert config.embedding_model == "text-embedding-3-small"
        assert config.pipeline.max_workers == 8
        assert config.query.default_method == "basic"

    def test_production_profile(self):
        config = get_production_profile()
        assert config.chat_model == "gpt-4.1"
        assert config.embedding_model == "text-embedding-3-large"
        assert config.query.default_method == "drift"
        assert config.query.max_context_tokens == 24000

    def test_profiles_are_valid(self):
        """All predefined profiles should be valid KGConfig instances."""
        assert isinstance(get_fast_profile(), KGConfig)
        assert isinstance(get_production_profile(), KGConfig)
        assert isinstance(KGConfig.defaults(), KGConfig)
