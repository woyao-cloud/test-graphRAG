"""Cypher query library for GraphRAG knowledge graph traversal.

Provides pre-built Cypher queries for common graph operations:
entity lookup, neighborhood traversal, path finding, community
analysis, and hybrid search support.
"""

from __future__ import annotations

from typing import Any, Optional

from neo4j import Session


class CypherQueries:
    """Collection of Cypher queries for knowledge graph operations.

    Designed to work with the Neo4j schema created by SchemaManager
    and populated by Neo4jGraphSync.
    """

    def __init__(self, session: Session):
        self.session = session

    # ------------------------------------------------------------------
    # Entity Lookup
    # ------------------------------------------------------------------

    def get_entity(self, name: str) -> Optional[dict[str, Any]]:
        """Look up an entity by name."""
        result = self.session.run(
            "MATCH (e:Entity {name: $name}) RETURN e",
            name=name,
        )
        record = result.single()
        return dict(record["e"]) if record else None

    def get_entity_by_id(self, entity_id: str) -> Optional[dict[str, Any]]:
        """Look up an entity by ID."""
        result = self.session.run(
            "MATCH (e:Entity {id: $id}) RETURN e",
            id=str(entity_id),
        )
        record = result.single()
        return dict(record["e"]) if record else None

    def search_entities(self, search_term: str, limit: int = 20) -> list[dict[str, Any]]:
        """Full-text search for entities."""
        try:
            result = self.session.run(
                """CALL db.index.fulltext.queryNodes('entity_search', $term)
                   YIELD node, score
                   RETURN node, score
                   ORDER BY score DESC
                   LIMIT $limit""",
                term=search_term,
                limit=limit,
            )
            return [{"entity": dict(r["node"]), "score": r["score"]} for r in result]
        except Exception:
            # Fallback to CONTAINS search
            result = self.session.run(
                """MATCH (e:Entity)
                   WHERE e.name CONTAINS $term OR e.description CONTAINS $term
                   RETURN e
                   LIMIT $limit""",
                term=search_term,
                limit=limit,
            )
            return [{"entity": dict(r["e"])} for r in result]

    # ------------------------------------------------------------------
    # Neighborhood Traversal
    # ------------------------------------------------------------------

    def get_neighbors(
        self, entity_name: str, hops: int = 1, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Get neighboring entities within N hops."""
        result = self.session.run(
            f"""MATCH (e:Entity {{name: $name}})-[:RELATES_TO*1..{hops}]-(neighbor:Entity)
                RETURN DISTINCT neighbor, e
                LIMIT $limit""",
            name=entity_name,
            limit=limit,
        )
        return [
            {"source": dict(r["e"]), "neighbor": dict(r["neighbor"])}
            for r in result
        ]

    def get_entity_with_relationships(
        self, entity_name: str, limit: int = 50
    ) -> dict[str, Any]:
        """Get an entity with all its direct relationships."""
        result = self.session.run(
            """MATCH (e:Entity {name: $name})-[r:RELATES_TO]-(other:Entity)
               RETURN e, collect(DISTINCT {relationship: r, entity: other}) as connections
               LIMIT $limit""",
            name=entity_name,
            limit=limit,
        )
        record = result.single()
        if record:
            return {
                "entity": dict(record["e"]),
                "connections": [
                    {
                        "relationship": dict(c["relationship"]),
                        "entity": dict(c["entity"]),
                    }
                    for c in record["connections"]
                ],
            }
        return {"entity": None, "connections": []}

    # ------------------------------------------------------------------
    # Path Finding
    # ------------------------------------------------------------------

    def find_shortest_path(
        self, source_name: str, target_name: str, max_hops: int = 5
    ) -> list[dict[str, Any]]:
        """Find the shortest path between two entities."""
        result = self.session.run(
            f"""MATCH path = shortestPath(
                    (a:Entity {{name: $source}})-[:RELATES_TO*1..{max_hops}]-(b:Entity {{name: $target}})
                )
                RETURN path, length(path) as hops""",
            source=source_name,
            target=target_name,
        )
        record = result.single()
        if record:
            return [{"path": str(record["path"]), "hops": record["hops"]}]
        return []

    def find_all_paths(
        self, source_name: str, target_name: str, max_hops: int = 4, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Find all paths between two entities."""
        result = self.session.run(
            f"""MATCH path = (a:Entity {{name: $source}})-[:RELATES_TO*1..{max_hops}]-(b:Entity {{name: $target}})
                RETURN path, length(path) as hops
                ORDER BY hops
                LIMIT $limit""",
            source=source_name,
            target=target_name,
            limit=limit,
        )
        return [
            {"path": str(r["path"]), "hops": r["hops"]}
            for r in result
        ]

    # ------------------------------------------------------------------
    # Community Analysis
    # ------------------------------------------------------------------

    def get_entity_communities(self, entity_name: str) -> list[dict[str, Any]]:
        """Get communities an entity belongs to."""
        result = self.session.run(
            """MATCH (e:Entity {name: $name})
               OPTIONAL MATCH (e)-[:BELONGS_TO]->(c:Community)
               RETURN c
               ORDER BY c.level""",
            name=entity_name,
        )
        return [dict(r["c"]) for r in result if r["c"]]

    def get_community_hierarchy(self, community_id: int) -> list[dict[str, Any]]:
        """Get parent-child hierarchy for a community."""
        result = self.session.run(
            """MATCH path = (c:Community {id: $id})-[:PARENT_OF*0..5]->(descendant:Community)
               RETURN path, length(path) as depth""",
            id=community_id,
        )
        return [
            {"path": str(r["path"]), "depth": r["depth"]}
            for r in result
        ]

    def get_community_entities(
        self, community_id: int, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Get entities in a community."""
        result = self.session.run(
            """MATCH (c:Community {id: $id})
               OPTIONAL MATCH (e:Entity)-[:BELONGS_TO]->(c)
               RETURN e, c.title as community_title
               LIMIT $limit""",
            id=community_id,
            limit=limit,
        )
        return [
            {"entity": dict(r["e"]), "community": r["community_title"]}
            for r in result if r["e"]
        ]

    # ------------------------------------------------------------------
    # Set Operations
    # ------------------------------------------------------------------

    def find_common_connections(
        self, entity_a: str, entity_b: str, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Find entities connected to both A and B (intersection)."""
        result = self.session.run(
            """MATCH (common:Entity)-[:RELATES_TO]-(a:Entity {name: $name_a})
               MATCH (common)-[:RELATES_TO]-(b:Entity {name: $name_b})
               RETURN DISTINCT common
               LIMIT $limit""",
            name_a=entity_a,
            name_b=entity_b,
            limit=limit,
        )
        return [dict(r["common"]) for r in result]

    # ------------------------------------------------------------------
    # Graph Statistics
    # ------------------------------------------------------------------

    def get_graph_stats(self) -> dict[str, int]:
        """Get overall graph statistics."""
        node_count = self.session.run(
            "MATCH (n) RETURN count(n) as count"
        ).single()["count"]
        entity_count = self.session.run(
            "MATCH (n:Entity) RETURN count(n) as count"
        ).single()["count"]
        rel_count = self.session.run(
            "MATCH ()-[r]->() RETURN count(r) as count"
        ).single()["count"]
        community_count = self.session.run(
            "MATCH (n:Community) RETURN count(n) as count"
        ).single()["count"]
        doc_count = self.session.run(
            "MATCH (n:Document) RETURN count(n) as count"
        ).single()["count"]

        return {
            "total_nodes": node_count,
            "entities": entity_count,
            "relationships": rel_count,
            "communities": community_count,
            "documents": doc_count,
        }

    def get_top_entities(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get entities with the highest degree (most connections)."""
        result = self.session.run(
            """MATCH (e:Entity)-[r:RELATES_TO]-()
               RETURN e.name as name, e.type as type, count(r) as degree
               ORDER BY degree DESC
               LIMIT $limit""",
            limit=limit,
        )
        return [dict(r) for r in result]
