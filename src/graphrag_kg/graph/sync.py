"""Neo4j graph synchronization from Parquet data.

Reads graphrag's parquet output files and batch-inserts entities,
relationships, and communities into Neo4j with proper node/edge types.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd

from graphrag_kg.core.config import KGConfig
from graphrag_kg.core.errors import Neo4jSyncError
from graphrag_kg.graph.connection import Neo4jConnection
from graphrag_kg.graph.schema import SchemaManager
from graphrag_kg.storage.parquet_store import ParquetStore

logger = logging.getLogger("graphrag_kg.graph.sync")


class Neo4jGraphSync:
    """Synchronizes graphrag parquet output to Neo4j.

    Creates typed nodes for entities, communities, documents, and text units,
    with labeled relationships between them. Supports batch insertion for
    large graphs.
    """

    def __init__(self, config: KGConfig):
        self.config = config
        self.connection = Neo4jConnection(config.neo4j)
        self.parquet = ParquetStore(config.output_dir)
        self.batch_size = config.neo4j.sync_batch_size

    # ------------------------------------------------------------------
    # Full Sync
    # ------------------------------------------------------------------

    def sync_all(self, clear_first: bool = False) -> dict[str, int]:
        """Synchronize all data from Parquet to Neo4j.

        Args:
            clear_first: If True, clear the Neo4j database before syncing.

        Returns:
            Dict with counts of synced nodes and relationships.

        Raises:
            Neo4jSyncError: If sync fails.
        """
        if not self.parquet.is_indexed():
            raise Neo4jSyncError(
                "No index data found. Run 'graphrag-kg index' first."
            )

        try:
            self.connection.connect()

            # Optionally clear existing data
            if clear_first:
                self.connection.clear_database(confirm=True)

            # Setup schema (indexes, constraints)
            with self.connection.session() as session:
                schema = SchemaManager(session)
                if self.config.neo4j.sync_create_indexes:
                    schema.setup_all()

            # Sync each data type
            results: dict[str, int] = {}

            entity_count = self.sync_entities()
            results["entities"] = entity_count

            relationship_count = self.sync_relationships()
            results["relationships"] = relationship_count

            community_count = self.sync_communities()
            results["communities"] = community_count

            doc_count = self.sync_documents()
            results["documents"] = doc_count

            tu_count = self.sync_text_units()
            results["text_units"] = tu_count

            cr_count = self.sync_community_reports()
            results["community_reports_links"] = cr_count

            logger.info(
                f"Sync complete: {entity_count} entities, "
                f"{relationship_count} relationships, "
                f"{community_count} communities"
            )

            return results

        except Exception as e:
            raise Neo4jSyncError(f"Neo4j sync failed: {e}") from e
        finally:
            self.connection.close()

    # ------------------------------------------------------------------
    # Entity Sync
    # ------------------------------------------------------------------

    def sync_entities(self) -> int:
        """Sync entities from entities.parquet to Neo4j (:Entity) nodes."""
        try:
            df = self.parquet.read_entities()
        except Exception:
            logger.warning("No entities.parquet found")
            return 0

        count = 0
        with self.connection.session() as session:
            for batch in self._batch(df):
                records = []
                for _, row in batch.iterrows():
                    props = self._row_to_props(row)
                    records.append({
                        "id": str(props.pop("id", props.pop("short_id", ""))),
                        "name": str(props.pop("name", props.pop("entity", props.pop("title", "")))),
                        "type": str(props.pop("type", "unknown")),
                        "description": str(props.pop("description", "")),
                        "degree": int(props.pop("degree", props.pop("frequency", 0))),
                        "human_readable_id": int(props.pop("human_readable_id", 0)),
                        "community_ids": self._to_list(props.pop("community_ids", [])),
                        "text_unit_ids": self._to_list(props.pop("text_unit_ids", [])),
                        "extra": props,  # Remaining properties
                    })

                result = session.run(
                    """UNWIND $records AS rec
                    MERGE (n:Entity {id: rec.id})
                    SET n.name = rec.name,
                        n.type = rec.type,
                        n.description = rec.description,
                        n.degree = rec.degree,
                        n.human_readable_id = rec.human_readable_id,
                        n.community_ids = rec.community_ids,
                        n.text_unit_ids = rec.text_unit_ids
                    RETURN count(n) as created""",
                    records=records,
                )
                count += result.single()["created"]

        logger.info(f"Synced {count} entities to Neo4j")
        return count

    # ------------------------------------------------------------------
    # Relationship Sync
    # ------------------------------------------------------------------

    def sync_relationships(self) -> int:
        """Sync relationships from relationships.parquet to Neo4j (:RELATES_TO) edges."""
        try:
            df = self.parquet.read_relationships()
        except Exception:
            logger.warning("No relationships.parquet found")
            return 0

        count = 0
        with self.connection.session() as session:
            for batch in self._batch(df):
                records = []
                for _, row in batch.iterrows():
                    props = self._row_to_props(row)
                    source = str(props.pop("source", props.pop("source_name", "")))
                    target = str(props.pop("target", props.pop("target_name", "")))
                    if not source or not target:
                        continue

                    records.append({
                        "source": source,
                        "target": target,
                        "rel_id": str(props.pop("id", props.pop("short_id", f"{source}-{target}"))),
                        "description": str(props.pop("description", "")),
                        "weight": float(props.pop("weight", props.pop("rank", 1.0))),
                        "text_unit_ids": self._to_list(props.pop("text_unit_ids", [])),
                        "human_readable_id": int(props.pop("human_readable_id", 0)),
                    })

                result = session.run(
                    """UNWIND $records AS rec
                    MATCH (source:Entity {name: rec.source})
                    MATCH (target:Entity {name: rec.target})
                    MERGE (source)-[r:RELATES_TO {id: rec.rel_id}]->(target)
                    SET r.description = rec.description,
                        r.weight = rec.weight,
                        r.text_unit_ids = rec.text_unit_ids,
                        r.human_readable_id = rec.human_readable_id
                    RETURN count(r) as created""",
                    records=records,
                )
                count += result.single()["created"]

        logger.info(f"Synced {count} relationships to Neo4j")
        return count

    # ------------------------------------------------------------------
    # Community Sync
    # ------------------------------------------------------------------

    def sync_communities(self) -> int:
        """Sync communities from communities.parquet to Neo4j (:Community) nodes."""
        try:
            df = self.parquet.read_communities()
        except Exception:
            logger.warning("No communities.parquet found")
            return 0

        count = 0
        with self.connection.session() as session:
            for batch in self._batch(df):
                records = []
                for _, row in batch.iterrows():
                    props = self._row_to_props(row)
                    records.append({
                        "id": int(props.pop("id", props.pop("community", 0))),
                        "title": str(props.pop("title", f"Community {props.get('id', '?')}")),
                        "level": int(props.pop("level", 0)),
                        "summary": str(props.pop("summary", "")),
                        "full_content": str(props.pop("full_content", "")),
                        "rating": float(props.pop("rating", props.pop("rank", 0.0))),
                        "entity_count": int(props.pop("entity_count", props.pop("size", 0))),
                        "parent_community_id": int(props.pop("parent_community_id", -1)),
                    })

                result = session.run(
                    """UNWIND $records AS rec
                    MERGE (c:Community {id: rec.id})
                    SET c.title = rec.title,
                        c.level = rec.level,
                        c.summary = rec.summary,
                        c.full_content = rec.full_content,
                        c.rating = rec.rating,
                        c.entity_count = rec.entity_count
                    RETURN count(c) as created""",
                    records=records,
                )
                count += result.single()["created"]

        logger.info(f"Synced {count} communities to Neo4j")
        return count

    # ------------------------------------------------------------------
    # Document Sync
    # ------------------------------------------------------------------

    def sync_documents(self) -> int:
        """Sync documents from documents.parquet to Neo4j (:Document) nodes."""
        try:
            df = self.parquet.read_documents()
        except Exception:
            return 0

        count = 0
        with self.connection.session() as session:
            for batch in self._batch(df):
                records = []
                for _, row in batch.iterrows():
                    props = self._row_to_props(row)
                    records.append({
                        "id": str(props.pop("id", props.pop("short_id", ""))),
                        "title": str(props.pop("title", props.pop("name", ""))),
                        "file_path": str(props.pop("file_path", props.pop("source", ""))),
                        "text_unit_count": int(props.pop("text_unit_count", props.pop("n_units", 0))),
                    })

                result = session.run(
                    """UNWIND $records AS rec
                    MERGE (d:Document {id: rec.id})
                    SET d.title = rec.title,
                        d.file_path = rec.file_path,
                        d.text_unit_count = rec.text_unit_count
                    RETURN count(d) as created""",
                    records=records,
                )
                count += result.single()["created"]

        logger.info(f"Synced {count} documents to Neo4j")
        return count

    # ------------------------------------------------------------------
    # Text Unit Sync
    # ------------------------------------------------------------------

    def sync_text_units(self) -> int:
        """Sync text units from text_units.parquet to Neo4j (:TextUnit) nodes."""
        try:
            df = self.parquet.read_text_units()
        except Exception:
            return 0

        count = 0
        with self.connection.session() as session:
            for batch in self._batch(df):
                records = []
                for _, row in batch.iterrows():
                    props = self._row_to_props(row)
                    records.append({
                        "id": str(props.pop("id", props.pop("short_id", ""))),
                        "text": str(props.pop("text", props.pop("chunk", ""))),
                        "document_id": str(props.pop("document_id", props.pop("doc_id", ""))),
                        "entity_ids": self._to_list(props.pop("entity_ids", [])),
                        "community_ids": self._to_list(props.pop("community_ids", [])),
                    })

                result = session.run(
                    """UNWIND $records AS rec
                    MERGE (tu:TextUnit {id: rec.id})
                    SET tu.text = rec.text,
                        tu.document_id = rec.document_id,
                        tu.entity_ids = rec.entity_ids,
                        tu.community_ids = rec.community_ids
                    RETURN count(tu) as created""",
                    records=records,
                )
                count += result.single()["created"]

        logger.info(f"Synced {count} text units to Neo4j")
        return count

    # ------------------------------------------------------------------
    # Community Report Links
    # ------------------------------------------------------------------

    def sync_community_reports(self) -> int:
        """Link community reports to communities and create entity-community edges."""
        try:
            reports_df = self.parquet.read_community_reports()
        except Exception:
            return 0

        count = 0
        with self.connection.session() as session:
            for batch in self._batch(reports_df):
                records = []
                for _, row in batch.iterrows():
                    props = self._row_to_props(row)
                    community_id = int(props.pop("community", props.pop("id", -1)))
                    records.append({
                        "community_id": community_id,
                        "title": str(props.pop("title", "")),
                        "summary": str(props.pop("summary", "")),
                        "full_content": str(props.pop("full_content", props.pop("content", ""))),
                        "rating": float(props.pop("rating", props.pop("rank", 0.0))),
                    })

                result = session.run(
                    """UNWIND $records AS rec
                    MATCH (c:Community {id: rec.community_id})
                    SET c.summary = coalesce(c.summary, rec.summary),
                        c.full_content = coalesce(c.full_content, rec.full_content),
                        c.rating = rec.rating
                    RETURN count(c) as updated""",
                    records=records,
                )
                count += result.single()["updated"]

        logger.info(f"Updated {count} communities with reports")
        return count

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _batch(self, df: pd.DataFrame) -> list[pd.DataFrame]:
        """Split a DataFrame into batches."""
        for i in range(0, len(df), self.batch_size):
            yield df.iloc[i:i + self.batch_size]

    def _row_to_props(self, row: pd.Series) -> dict[str, Any]:
        """Convert a DataFrame row to a dict of properties."""
        props = row.to_dict()
        # Remove NaN values
        return {
            k: v for k, v in props.items()
            if v is not None and not (isinstance(v, float) and pd.isna(v))
        }

    def _to_list(self, value: Any) -> list:
        """Convert a value to a list safely."""
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            try:
                import json
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return [value] if value else []
        return [value]
