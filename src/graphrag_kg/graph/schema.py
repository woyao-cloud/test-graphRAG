"""Neo4j schema management for GraphRAG-KG.

Creates indexes, constraints, and node/relationship type definitions
for the knowledge graph stored in Neo4j.
"""

from __future__ import annotations

import logging

from neo4j import Session

logger = logging.getLogger("graphrag_kg.graph.schema")

# Indexes for fast lookup
INDEXES = [
    # Entity indexes
    "CREATE INDEX entity_id IF NOT EXISTS FOR (n:Entity) ON (n.id)",
    "CREATE INDEX entity_name IF NOT EXISTS FOR (n:Entity) ON (n.name)",
    "CREATE INDEX entity_type IF NOT EXISTS FOR (n:Entity) ON (n.type)",
    "CREATE INDEX entity_human_id IF NOT EXISTS FOR (n:Entity) ON (n.human_readable_id)",

    # Community indexes
    "CREATE INDEX community_id IF NOT EXISTS FOR (n:Community) ON (n.id)",
    "CREATE INDEX community_level IF NOT EXISTS FOR (n:Community) ON (n.level)",

    # Document indexes
    "CREATE INDEX document_id IF NOT EXISTS FOR (n:Document) ON (n.id)",

    # Text unit indexes
    "CREATE INDEX textunit_id IF NOT EXISTS FOR (n:TextUnit) ON (n.id)",

    # Relationship indexes
    "CREATE INDEX relates_to_id IF NOT EXISTS FOR ()-[r:RELATES_TO]-() ON (r.id)",
]

# Constraints for uniqueness
CONSTRAINTS = [
    "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT community_id_unique IF NOT EXISTS FOR (n:Community) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT document_id_unique IF NOT EXISTS FOR (n:Document) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT textunit_id_unique IF NOT EXISTS FOR (n:TextUnit) REQUIRE n.id IS UNIQUE",
]

# Full-text indexes for search (Neo4j 5.x)
FULLTEXT_INDEXES = [
    # Entity name and description search
    """CREATE FULLTEXT INDEX entity_search IF NOT EXISTS
       FOR (n:Entity) ON EACH [n.name, n.description]""",
    # Community title and content search
    """CREATE FULLTEXT INDEX community_search IF NOT EXISTS
       FOR (n:Community) ON EACH [n.title, n.summary]""",
]


class SchemaManager:
    """Manages Neo4j schema setup for GraphRAG-KG.

    Creates indexes, constraints, and optionally full-text search indexes
    for the knowledge graph stored in Neo4j.
    """

    def __init__(self, session: Session):
        self.session = session

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup_all(self, include_fulltext: bool = True) -> dict[str, int]:
        """Create all indexes and constraints.

        Args:
            include_fulltext: Whether to create full-text search indexes.

        Returns:
            Dict with counts of created/dropped items.
        """
        results = {
            "indexes_created": 0,
            "constraints_created": 0,
            "fulltext_indexes_created": 0,
            "errors": 0,
        }

        # Create constraints first (required for some indexes)
        for cypher in CONSTRAINTS:
            try:
                self.session.run(cypher)
                results["constraints_created"] += 1
            except Exception as e:
                logger.warning(f"Constraint creation warning: {e}")
                results["errors"] += 1

        # Create indexes
        for cypher in INDEXES:
            try:
                self.session.run(cypher)
                results["indexes_created"] += 1
            except Exception as e:
                logger.warning(f"Index creation warning: {e}")
                results["errors"] += 1

        # Create full-text indexes
        if include_fulltext:
            for cypher in FULLTEXT_INDEXES:
                try:
                    self.session.run(cypher)
                    results["fulltext_indexes_created"] += 1
                except Exception as e:
                    logger.warning(f"Full-text index creation warning: {e}")
                    results["errors"] += 1

        logger.info(
            f"Schema setup complete: "
            f"{results['constraints_created']} constraints, "
            f"{results['indexes_created']} indexes, "
            f"{results['fulltext_indexes_created']} full-text indexes"
        )

        return results

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    def drop_all(self) -> dict[str, int]:
        """Drop all indexes and constraints."""
        results = {"indexes_dropped": 0, "constraints_dropped": 0}

        # Drop indexes
        index_list = self.session.run("SHOW INDEXES").data()
        for idx in index_list:
            name = idx.get("name", "")
            if name and "graphrag" not in name.lower():
                # Only drop indexes that aren't system indexes
                try:
                    self.session.run(f"DROP INDEX {name}")
                    results["indexes_dropped"] += 1
                except Exception:
                    pass

        # Drop constraints
        constraint_list = self.session.run("SHOW CONSTRAINTS").data()
        for con in constraint_list:
            name = con.get("name", "")
            if name:
                try:
                    self.session.run(f"DROP CONSTRAINT {name}")
                    results["constraints_dropped"] += 1
                except Exception:
                    pass

        return results

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        """Get current schema status."""
        try:
            indexes = self.session.run("SHOW INDEXES").data()
            constraints = self.session.run("SHOW CONSTRAINTS").data()

            return {
                "index_count": len(indexes),
                "constraint_count": len(constraints),
                "indexes": [
                    {"name": i.get("name", ""), "type": i.get("type", "")}
                    for i in indexes
                ],
                "constraints": [
                    {"name": c.get("name", ""), "type": c.get("type", "")}
                    for c in constraints
                ],
            }
        except Exception as e:
            return {"error": str(e)}
