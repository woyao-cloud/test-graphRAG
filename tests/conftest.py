"""Shared test fixtures for GraphRAG-KG tests."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Generator

import pytest

from graphrag_kg.core.config import KGConfig, Neo4jConfig, IngestionConfig
from graphrag_kg.core.config_loader import ConfigLoader


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test projects."""
    with tempfile.TemporaryDirectory(prefix="graphrag_test_") as tmp:
        yield Path(tmp)


@pytest.fixture
def default_config() -> KGConfig:
    """Return a default KGConfig."""
    return KGConfig()


@pytest.fixture
def fast_config() -> KGConfig:
    """Return a fast-profile KGConfig."""
    from graphrag_kg.core.config import get_fast_profile
    return get_fast_profile()


@pytest.fixture
def production_config() -> KGConfig:
    """Return a production-profile KGConfig."""
    from graphrag_kg.core.config import get_production_profile
    return get_production_profile()


@pytest.fixture
def config_loader() -> ConfigLoader:
    """Return a fresh ConfigLoader."""
    return ConfigLoader()


@pytest.fixture
def sample_yaml_config() -> str:
    """Return a minimal valid YAML config string."""
    return """\
kg:
  project_name: "test-project"
  description: "Test project for unit tests"

  neo4j:
    uri: "bolt://localhost:7687"
    username: "neo4j"
    password: "testpass"

  ingestion:
    source_directories:
      - "./test_docs"

  chat_model: "gpt-4.1-mini"
  api_key: "test-key"

models:
  default_chat_model:
    type: litellm
    model_provider: openai
    model: gpt-4.1-mini
    api_key: "test-key"
  default_embedding_model:
    type: litellm
    model_provider: openai
    model: text-embedding-3-small
    api_key: "test-key"
"""


@pytest.fixture
def temp_yaml_config(temp_dir: Path, sample_yaml_config: str) -> Path:
    """Write a sample YAML config to a temp file."""
    config_path = temp_dir / "settings.yaml"
    config_path.write_text(sample_yaml_config, encoding="utf-8")
    return config_path


@pytest.fixture
def pharma_ground_truth_path() -> Path:
    """Return path to generated pharma_supply_chain ground truth."""
    path = Path("tests/fixtures/generated/pharma_supply_chain/ground_truth.json")
    if not path.exists():
        pytest.skip("Pharma supply chain test data not generated. Run graphrag-kg data generate first.")
    return path


@pytest.fixture
def tech_ground_truth_path() -> Path:
    """Return path to generated tech_company ground truth."""
    path = Path("tests/fixtures/generated/tech_company/ground_truth.json")
    if not path.exists():
        pytest.skip("Tech company test data not generated. Run graphrag-kg data generate --scenario tech_company first.")
    return path
