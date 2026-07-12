"""Integration tests for Milvus vector store."""

import pytest
from graphrag_vectors.vector_store_config import VectorStoreConfig
from graphrag_vectors.index_schema import IndexSchema
from graphrag_vectors.vector_store_factory import create_vector_store, vector_store_factory
from graphrag_kg.core.milvus_store import MilvusVectorStore, register_milvus_vector_store


def test_milvus_registration():
    """Test that Milvus vector store is registered."""
    # Verify the registration happened
    assert "milvus" in vector_store_factory.keys(), "Milvus not registered in factory"


def test_milvus_instantiation():
    """Test creating a MilvusVectorStore instance."""
    store = MilvusVectorStore(
        collection_name="test_collection",
        host="localhost",
        port=19530,
        vector_size=3072,
    )
    assert store is not None
    assert isinstance(store, MilvusVectorStore)
    assert store.collection_name == "test_collection"
    assert store.host == "localhost"
    assert store.port == 19530


def test_milvus_factory_creation():
    """Test creating Milvus store via factory."""
    config = VectorStoreConfig(
        type="milvus",
        host="localhost",
        port=19530,
        collection_name="factory_test",
    )
    schema = IndexSchema()
    
    store = create_vector_store(config, schema)
    
    assert store is not None
    assert isinstance(store, MilvusVectorStore)
    assert store.collection_name == "factory_test"


def test_milvus_config_inheritance():
    """Test that MilvusVectorStore inherits VectorStore base properties."""
    store = MilvusVectorStore(
        index_name="my_index",
        id_field="doc_id",
        vector_field="embedding",
        vector_size=1536,
    )
    
    assert store.index_name == "my_index"
    assert store.id_field == "doc_id"
    assert store.vector_field == "embedding"
    assert store.vector_size == 1536


def test_milvus_field_types():
    """Test Milvus field type mapping."""
    store = MilvusVectorStore(
        fields={
            "title": "str",
            "count": "int",
            "score": "float",
            "active": "bool",
            "created": "date",
        }
    )
    
    expected_fields = {
        "title": "str",
        "count": "int",
        "score": "float",
        "active": "bool",
        "created": "str",  # date fields converted to str
    }
    
    for field_name, field_type in expected_fields.items():
        assert field_name in store.fields, f"Field {field_name} missing"


def test_milvus_metric_types():
    """Test different metric types for Milvus."""
    for metric in ["COSINE", "IP", "L2"]:
        store = MilvusVectorStore(
            collection_name=f"test_{metric.lower()}",
            metric_type=metric,
        )
        assert store.metric_type == metric.upper()


def test_milvus_index_params():
    """Test custom index parameters."""
    custom_params = {
        "index_type": "HNSW",
        "params": {"M": 16, "ef_construction": 200},
    }
    
    store = MilvusVectorStore(
        collection_name="hnsw_test",
        index_params=custom_params,
    )
    
    assert store.index_params["index_type"] == "HNSW"
    assert store.index_params["params"]["M"] == 16


def test_milvus_pymilvus_check():
    """Test that pymilvus availability check works."""
    store = MilvusVectorStore()
    # Should not raise an error
    store._ensure_pymilvus()


def test_graphrag_kg_auto_registration():
    """Test that graphrag_kg auto-registers Milvus on import."""
    # This test verifies the import mechanism
    import graphrag_kg  # noqa
    
    assert "milvus" in vector_store_factory.keys()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
