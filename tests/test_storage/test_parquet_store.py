"""Tests for ParquetStore."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from graphrag_kg.storage.parquet_store import ParquetStore
from graphrag_kg.core.errors import StorageError


class TestParquetStore:
    """Tests for ParquetStore."""

    @pytest.fixture
    def store(self, temp_dir):
        output_dir = temp_dir / "output"
        output_dir.mkdir()
        return ParquetStore(output_dir)

    @pytest.fixture
    def populated_store(self, store):
        """Store with sample data written."""
        df_entities = pd.DataFrame([
            {"id": "e1", "name": "Test Entity", "type": "organization", "description": "A test entity"},
            {"id": "e2", "name": "Another Entity", "type": "person", "description": "Another one"},
        ])
        store.write(df_entities, "entities")

        df_rels = pd.DataFrame([
            {"id": "r1", "source": "Test Entity", "target": "Another Entity",
             "description": "employs", "weight": 1.0},
        ])
        store.write(df_rels, "relationships")

        store.write(pd.DataFrame([{"id": 1, "title": "Community 1"}]), "communities")
        store.write(pd.DataFrame([{"id": "t1", "text": "Sample text"}]), "text_units")

        return store

    def test_read_entities(self, populated_store):
        df = populated_store.read_entities()
        assert len(df) == 2
        assert "Test Entity" in df["name"].values

    def test_read_relationships(self, populated_store):
        df = populated_store.read_relationships()
        assert len(df) == 1
        assert df.iloc[0]["source"] == "Test Entity"

    def test_read_nonexistent(self, store):
        with pytest.raises(StorageError):
            store.read("entities")

    def test_is_indexed(self, populated_store):
        assert populated_store.is_indexed() is True

    def test_is_indexed_empty(self, store):
        assert store.is_indexed() is False

    def test_get_index_stats(self, populated_store):
        stats = populated_store.get_index_stats()
        assert stats["entities"] == 2
        assert stats["relationships"] == 1

    def test_get_entity_by_name(self, populated_store):
        entity = populated_store.get_entity_by_name("Test Entity")
        assert entity is not None
        assert entity["type"] == "organization"

    def test_get_entity_by_name_not_found(self, populated_store):
        entity = populated_store.get_entity_by_name("Nonexistent")
        assert entity is None

    def test_get_relationships_for_entity(self, populated_store):
        rels = populated_store.get_relationships_for_entity("Test Entity")
        assert len(rels) == 1

    def test_list_available(self, populated_store):
        available = populated_store.list_available()
        assert "entities" in available
        assert "relationships" in available

    def test_write(self, store):
        df = pd.DataFrame([{"id": "x1", "name": "X"}])
        path = store.write(df, "test_table")
        assert path.exists()
        assert path.suffix == ".parquet"
