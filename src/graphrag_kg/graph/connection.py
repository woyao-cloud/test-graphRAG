"""Neo4j driver and connection pool management.

Provides a context-managed Neo4j driver with connection pooling,
health checks, and automatic reconnection.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Optional

from neo4j import Driver, GraphDatabase, Session, Transaction
from neo4j.exceptions import ServiceUnavailable, AuthError

from graphrag_kg.core.config import Neo4jConfig
from graphrag_kg.core.errors import Neo4jConnectionError

logger = logging.getLogger("graphrag_kg.graph.connection")


class Neo4jConnection:
    """Manages a Neo4j driver with connection pooling.

    Usage:
        conn = Neo4jConnection(config.neo4j)
        conn.connect()
        with conn.session() as session:
            result = session.run("MATCH (n) RETURN count(n)")
        conn.close()
    """

    def __init__(self, config: Neo4jConfig):
        self.config = config
        self._driver: Optional[Driver] = None

    # ------------------------------------------------------------------
    # Connection Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> Driver:
        """Establish connection to Neo4j and return the driver.

        Raises:
            Neo4jConnectionError: If connection fails.
        """
        if self._driver is not None:
            return self._driver

        try:
            self._driver = GraphDatabase.driver(
                self.config.uri,
                auth=(self.config.username, self.config.password),
                max_connection_lifetime=self.config.max_connection_lifetime,
                max_connection_pool_size=self.config.max_connection_pool_size,
                connection_acquisition_timeout=self.config.connection_acquisition_timeout,
            )
            # Verify connectivity
            self._driver.verify_connectivity()
            logger.info(f"Connected to Neo4j at {self.config.uri}")
            return self._driver
        except AuthError as e:
            raise Neo4jConnectionError(
                f"Authentication failed for Neo4j at {self.config.uri}: {e}"
            ) from e
        except ServiceUnavailable as e:
            raise Neo4jConnectionError(
                f"Neo4j service unavailable at {self.config.uri}. "
                f"Is the database running? {e}"
            ) from e
        except Exception as e:
            raise Neo4jConnectionError(
                f"Failed to connect to Neo4j at {self.config.uri}: {e}"
            ) from e

    def close(self) -> None:
        """Close the driver and release all connections."""
        if self._driver is not None:
            self._driver.close()
            self._driver = None
            logger.info("Neo4j connection closed")

    @property
    def is_connected(self) -> bool:
        """Check if the driver is connected."""
        if self._driver is None:
            return False
        try:
            self._driver.verify_connectivity()
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Session Management
    # ------------------------------------------------------------------

    @contextmanager
    def session(self, database: Optional[str] = None) -> Session:
        """Get a Neo4j session context manager.

        Usage:
            with conn.session() as session:
                result = session.run("MATCH (n) RETURN n")
        """
        if self._driver is None:
            self.connect()

        db = database or self.config.database
        session = self._driver.session(database=db)
        try:
            yield session
        finally:
            session.close()

    @contextmanager
    def transaction(self, database: Optional[str] = None) -> Transaction:
        """Get a write transaction context manager.

        Usage:
            with conn.transaction() as tx:
                tx.run("CREATE (n:Entity {name: $name})", name="Test")
        """
        with self.session(database) as session:
            with session.begin_transaction() as tx:
                yield tx
                tx.commit()

    # ------------------------------------------------------------------
    # Health Check
    # ------------------------------------------------------------------

    def health_check(self) -> dict[str, Any]:
        """Check Neo4j connectivity and return status."""
        try:
            with self.session() as session:
                result = session.run("RETURN 1 as ok")
                record = result.single()
                version_result = session.run("CALL dbms.components()").single()

                return {
                    "connected": True,
                    "uri": self.config.uri,
                    "database": self.config.database,
                    "neo4j_version": version_result["versions"][0] if version_result else "unknown",
                }
        except Exception as e:
            return {
                "connected": False,
                "uri": self.config.uri,
                "error": str(e),
            }

    # ------------------------------------------------------------------
    # Database Management
    # ------------------------------------------------------------------

    def clear_database(self, confirm: bool = False) -> dict[str, Any]:
        """Delete all nodes and relationships. Requires confirmation.

        Args:
            confirm: Must be True to actually execute.

        Returns:
            Dict with counts of deleted nodes and relationships.
        """
        if not confirm:
            return {"deleted": False, "reason": "confirmation required"}

        with self.session() as session:
            # Count before
            node_count = session.run("MATCH (n) RETURN count(n) as c").single()["c"]
            rel_count = session.run("MATCH ()-[r]->() RETURN count(r) as c").single()["c"]

            # Delete all
            session.run("MATCH (n) DETACH DELETE n")

            logger.info(f"Cleared Neo4j database: {node_count} nodes, {rel_count} relationships")

            return {
                "deleted": True,
                "nodes_deleted": node_count,
                "relationships_deleted": rel_count,
            }

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.close()
