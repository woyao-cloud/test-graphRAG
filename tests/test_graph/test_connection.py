"""Tests for Neo4jConnection (connection only, no database required)."""

from __future__ import annotations

import pytest

from graphrag_kg.core.config import Neo4jConfig
from graphrag_kg.graph.connection import Neo4jConnection
from graphrag_kg.core.errors import Neo4jConnectionError


class TestNeo4jConnection:
    """Tests for Neo4jConnection (unit tests, no live DB)."""

    def test_config_stored(self):
        config = Neo4jConfig(uri="bolt://test:7687", username="testuser", password="testpass")
        conn = Neo4jConnection(config)
        assert conn.config.uri == "bolt://test:7687"
        assert conn.config.username == "testuser"

    def test_not_connected_initially(self):
        config = Neo4jConfig()
        conn = Neo4jConnection(config)
        assert conn.is_connected is False

    def test_connection_refused(self):
        """Should raise Neo4jConnectionError when server not available."""
        config = Neo4jConfig(uri="bolt://localhost:17687")  # Non-standard port
        conn = Neo4jConnection(config)

        with pytest.raises(Neo4jConnectionError):
            conn.connect()

    def test_close_when_not_connected(self):
        """Should not raise when closing unconnected driver."""
        config = Neo4jConfig()
        conn = Neo4jConnection(config)
        conn.close()  # Should not raise

    def test_health_check_when_disconnected(self):
        config = Neo4jConfig(uri="bolt://localhost:17687")
        conn = Neo4jConnection(config)
        health = conn.health_check()
        assert health["connected"] is False

    def test_context_manager(self):
        config = Neo4jConfig(uri="bolt://localhost:17687")
        conn = Neo4jConnection(config)

        with pytest.raises(Neo4jConnectionError):
            with conn:
                pass  # Should fail to connect
