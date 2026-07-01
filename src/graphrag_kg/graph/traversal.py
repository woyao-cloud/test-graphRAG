"""Advanced graph traversal utilities for query-time graph exploration.

Provides ego networks, multi-hop path enumeration, community neighborhoods,
and hybrid traversal strategies that combine vector search with graph walks.
"""

from __future__ import annotations

from typing import Any, Optional

from neo4j import Session


class GraphTraversal:
    """Advanced graph traversal strategies for knowledge graph exploration.

    Complements CypherQueries with higher-level traversal patterns used
    by the query engine to expand search context.
    """

    def __init__(self, session: Session):
        self.session = session

    # ------------------------------------------------------------------
    # Ego Network
    # ------------------------------------------------------------------

    def ego_network(
        self, entity_name: str, radius: int = 2, limit: int = 200
    ) -> dict[str, Any]:
        """Get the ego network around an entity.

        Returns all entities and relationships within `radius` hops
        of the target entity, suitable for local search context.
        """
        result = self.session.run(
            f"""MATCH (center:Entity {{name: $name}})
                OPTIONAL MATCH path = (center)-[:RELATES_TO*1..{radius}]-(neighbor:Entity)
                WITH center, collect(DISTINCT neighbor) as neighbors
                OPTIONAL MATCH (n1:Entity)-[r:RELATES_TO]->(n2:Entity)
                WHERE n1 IN neighbors OR n1 = center
                  AND n2 IN neighbors OR n2 = center
                RETURN center,
                       neighbors,
                       collect(DISTINCT {{source: n1.name, target: n2.name,
                                         relationship: type(r), description: r.description,
                                         weight: r.weight}}) as edges
                LIMIT $limit""",
            name=entity_name,
            limit=limit,
        )
        record = result.single()
        if not record:
            return {"center": None, "nodes": [], "edges": []}

        return {
            "center": dict(record["center"]),
            "nodes": [dict(n) for n in record["neighbors"]],
            "edges": record["edges"],
        }

    # ------------------------------------------------------------------
    # Community Neighborhood
    # ------------------------------------------------------------------

    def community_neighborhood(
        self, entity_name: str, depth: int = 1
    ) -> dict[str, Any]:
        """Get the community context for an entity.

        Returns the entity's community and all sibling entities
        within the same community, plus parent/child communities.
        """
        result = self.session.run(
            """MATCH (e:Entity {name: $name})
               OPTIONAL MATCH (e)-[:BELONGS_TO]->(c:Community)
               OPTIONAL MATCH (sibling:Entity)-[:BELONGS_TO]->(c)
               WHERE sibling <> e
               OPTIONAL MATCH (c)<-[:PARENT_OF*0..2]-(ancestor:Community)
               OPTIONAL MATCH (c)-[:PARENT_OF*0..2]->(descendant:Community)
               RETURN e, c,
                      collect(DISTINCT sibling) as siblings,
                      collect(DISTINCT ancestor) as ancestors,
                      collect(DISTINCT descendant) as descendants""",
            name=entity_name,
        )
        record = result.single()
        if not record:
            return {}

        return {
            "entity": dict(record["e"]),
            "community": dict(record["c"]) if record["c"] else None,
            "siblings": [dict(s) for s in record["siblings"]],
            "ancestors": [dict(a) for a in record["ancestors"] if a],
            "descendants": [dict(d) for d in record["descendants"] if d],
        }

    # ------------------------------------------------------------------
    # Multi-hop Path Enumeration
    # ------------------------------------------------------------------

    def find_all_paths_between(
        self, source: str, target: str, max_hops: int = 5, limit: int = 20
    ) -> list[dict[str, Any]]:
        """Enumerate all paths between two entities up to max_hops."""
        result = self.session.run(
            f"""MATCH path = (a:Entity {{name: $source}})-[:RELATES_TO*1..{max_hops}]->(b:Entity {{name: $target}})
                RETURN
                    [node in nodes(path) | node.name] as node_names,
                    [rel in relationships(path) | rel.description] as edge_descriptions,
                    length(path) as hops
                ORDER BY hops
                LIMIT $limit""",
            source=source,
            target=target,
            limit=limit,
        )
        return [
            {
                "path": r["node_names"],
                "edges": r["edge_descriptions"],
                "hops": r["hops"],
            }
            for r in result
        ]

    # ------------------------------------------------------------------
    # Supply Chain / Flow Tracing
    # ------------------------------------------------------------------

    def trace_flow(
        self, start_name: str, direction: str = "downstream", max_hops: int = 6
    ) -> list[dict[str, Any]]:
        """Trace a supply chain or information flow from an entity.

        Args:
            start_name: Starting entity name.
            direction: 'downstream' (outgoing), 'upstream' (incoming), 'both'.
            max_hops: Maximum traversal depth.
        """
        if direction == "downstream":
            pattern = f"(start:Entity {{name: $name}})-[:RELATES_TO*1..{max_hops}]->(end:Entity)"
        elif direction == "upstream":
            pattern = f"(start:Entity {{name: $name}})<-[:RELATES_TO*1..{max_hops}]-(end:Entity)"
        else:
            pattern = f"(start:Entity {{name: $name}})-[:RELATES_TO*1..{max_hops}]-(end:Entity)"

        result = self.session.run(
            f"""MATCH path = {pattern}
                WHERE start <> end
                RETURN
                    [node in nodes(path) | node.name] as path_names,
                    [node in nodes(path) | labels(node)[0]] as path_types,
                    length(path) as depth
                ORDER BY depth
                LIMIT 50""",
            name=start_name,
        )
        return [
            {
                "path": r["path_names"],
                "types": r["path_types"],
                "depth": r["depth"],
            }
            for r in result
        ]

    # ------------------------------------------------------------------
    # Set Operations
    # ------------------------------------------------------------------

    def find_intersection(
        self, entity_names: list[str], limit: int = 50
    ) -> list[dict[str, Any]]:
        """Find entities connected to ALL entities in the list (intersection).

        Useful for "which hospitals use both drug A and drug B?" queries.
        """
        if len(entity_names) < 2:
            return []

        # Build match for each entity
        matches = []
        for i, name in enumerate(entity_names):
            alias = f"e{i}"
            matches.append(
                f"MATCH (common:Entity)-[:RELATES_TO]-(e{i}:Entity {{name: $name_{i}}})"
            )

        cypher = (
            "\n".join(matches) +
            "\nRETURN DISTINCT common, common.name as name, labels(common) as labels\n"
            f"LIMIT {limit}"
        )

        params = {f"name_{i}": name for i, name in enumerate(entity_names)}
        result = self.session.run(cypher, **params)

        return [
            {"entity": dict(r["common"]), "name": r["name"], "labels": r["labels"]}
            for r in result
        ]

    # ------------------------------------------------------------------
    # Impact Analysis
    # ------------------------------------------------------------------

    def impact_analysis(
        self, entity_name: str, max_hops: int = 4
    ) -> dict[str, Any]:
        """Analyze the impact radius of an entity.

        Returns all downstream entities that would be affected if this
        entity were removed from the graph.
        """
        result = self.session.run(
            f"""MATCH (start:Entity {{name: $name}})-[:RELATES_TO*1..{max_hops}]->(affected:Entity)
                WHERE start <> affected
                WITH affected, count(*) as path_count
                RETURN
                    affected.name as name,
                    labels(affected) as types,
                    path_count
                ORDER BY path_count DESC
                LIMIT 100""",
            name=entity_name,
        )
        return {
            "source": entity_name,
            "affected_entities": [
                {"name": r["name"], "types": r["types"], "path_count": r["path_count"]}
                for r in result
            ],
        }
