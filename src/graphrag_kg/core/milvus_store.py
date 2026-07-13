"""Milvus vector store using MilvusClient API (no deprecation warnings).

Implements the graphrag_vectors VectorStore ABC with the modern
MilvusClient interface (pymilvus >= 2.4).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from graphrag_vectors.vector_store import VectorStore, VectorStoreDocument, VectorStoreSearchResult
from graphrag_vectors.vector_store_factory import register_vector_store

logger = logging.getLogger("graphrag_kg.core.milvus_store")

try:
    from pymilvus import DataType, MilvusClient
except ImportError:  # pragma: no cover
    MilvusClient = None
    DataType = None
    _PYMILVUS_AVAILABLE = False
else:
    _PYMILVUS_AVAILABLE = True


def _field_type_to_milvus_dtype(field_type: str) -> Any:
    """Map graphrag field type strings to Milvus DataType."""
    ft = field_type.lower()
    if ft in ("str", "string"):
        return DataType.VARCHAR
    if ft in ("int", "integer"):
        return DataType.INT64
    if ft in ("float", "double"):
        return DataType.DOUBLE
    if ft in ("bool", "boolean"):
        return DataType.BOOL
    return DataType.VARCHAR


class MilvusVectorStore(VectorStore):
    """Milvus-backed vector store using the modern MilvusClient API."""

    def __init__(
        self,
        index_name: str = "vector_index",
        id_field: str = "id",
        vector_field: str = "vector",
        create_date_field: str = "create_date",
        update_date_field: str = "update_date",
        vector_size: int = 3072,
        fields: dict[str, str] | None = None,
        timestamp_exploder=None,
        host: str = "localhost",
        port: int | str = 19530,
        collection_name: str | None = None,
        metric_type: str = "IP",
        index_params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_name=index_name,
            id_field=id_field,
            vector_field=vector_field,
            create_date_field=create_date_field,
            update_date_field=update_date_field,
            vector_size=vector_size,
            fields=fields,
            timestamp_exploder=timestamp_exploder,
            **kwargs,
        )
        self.host = host
        self.port = int(port)
        # Prefer index_name (per-table from index_schema) over collection_name
        self.collection_name = index_name if index_name != "vector_index" else (collection_name or index_name)
        self.metric_type = metric_type.upper()
        self.index_params = index_params or {
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024},
        }
        self._client: MilvusClient | None = None

    # ------------------------------------------------------------------
    # Client lifecycle
    # ------------------------------------------------------------------

    def _ensure_pymilvus(self) -> None:
        if not _PYMILVUS_AVAILABLE:
            raise RuntimeError(
                "Milvus vector store requires pymilvus. "
                "Install it with 'pip install pymilvus'."
            )

    @property
    def client(self) -> MilvusClient:
        """Get or create the MilvusClient singleton."""
        if self._client is None:
            self._ensure_pymilvus()
            self._client = MilvusClient(host=self.host, port=self.port)
        return self._client

    # ------------------------------------------------------------------
    # VectorStore ABC implementation
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Connect to Milvus — lazy via client property."""
        _ = self.client  # triggers init

    def create_index(self) -> None:
        """Ensure the collection exists with schema and index."""
        self._ensure_pymilvus()
        if self.client.has_collection(self.collection_name):
            return

        schema = self._build_milvus_schema()
        ip = self._build_milvus_index_params()
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=ip,
        )
        self.client.load_collection(self.collection_name)
        logger.info(
            "Created collection '%s' (dim=%d, metric=%s)",
            self.collection_name,
            self.vector_size,
            self.metric_type,
        )

    def load_documents(self, documents: list[VectorStoreDocument]) -> None:
        """Batch-insert documents."""
        self._ensure_pymilvus()
        self.create_index()

        rows: list[dict[str, Any]] = []
        for doc in documents:
            self._prepare_document(doc)
            row: dict[str, Any] = {
                self.id_field: str(doc.id),
                self.vector_field: doc.vector or [],
                "json_data": json.dumps(doc.data or {}, ensure_ascii=False),
                self.create_date_field: doc.create_date or "",
                self.update_date_field: doc.update_date or "",
            }
            for name in (self.fields or {}):
                if name in {self.id_field, self.vector_field}:
                    continue
                row[name] = doc.data.get(name) if doc.data else None
            rows.append(row)

        if not rows:
            return

        self.client.insert(self.collection_name, rows)
        try:
            self.client.flush(self.collection_name)
        except Exception:
            logger.debug("Milvus flush failed", exc_info=True)

    def similarity_search_by_vector(
        self,
        query_embedding: list[float],
        k: int = 10,
        select: list[str] | None = None,
        filters: Any = None,
        include_vectors: bool = True,
    ) -> list[VectorStoreSearchResult]:
        """Search by vector similarity using MilvusClient."""
        self._ensure_pymilvus()
        self.create_index()

        # Determine output fields
        output_fields = []
        if select:
            output_fields = list(select)
        else:
            output_fields = [
                self.id_field,
                "json_data",
                self.create_date_field,
                self.update_date_field,
            ]
            # Add content fields that exist in the collection schema
            try:
                desc = self.client.describe_collection(self.collection_name)
                schema_fields = {f["name"] for f in desc.get("schema", {}).get("fields", [])}
                for cf in ["text", "title", "description", "summary", "full_content"]:
                    if cf in schema_fields and cf not in output_fields:
                        output_fields.append(cf)
            except Exception:
                pass

        # MilvusClient returns: [[{"id":..., "distance":..., "entity":{...}}]]
        raw = self.client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            anns_field=self.vector_field,
            search_params={
                "metric_type": self.metric_type,
                "params": self.index_params.get("search_params", {"nprobe": 10}),
            },
            limit=k,
            output_fields=output_fields,
        )

        hits = raw[0] if raw else []
        results: list[VectorStoreSearchResult] = []
        for hit in hits:
            entity = hit.get("entity", {})
            distance = hit.get("distance", 0.0)
            score = self._score_from_distance(distance)
            result = self._parse_search_hit(hit["id"], entity, score)
            if include_vectors:
                try:
                    qr = self.client.get(self.collection_name, ids=[hit["id"]])
                    if qr:
                        vec = qr[0].get(self.vector_field)
                        result.document.vector = vec
                except Exception:
                    pass
            results.append(result)

        return results

    def search_by_id(
        self,
        id: str,
        select: list[str] | None = None,
        include_vectors: bool = True,
    ) -> VectorStoreDocument:
        """Look up a document by its ID."""
        self._ensure_pymilvus()
        self.create_index()

        output_fields = select or [
            self.id_field,
            "json_data",
            self.create_date_field,
            self.update_date_field,
        ]
        if include_vectors and self.vector_field not in output_fields:
            output_fields.append(self.vector_field)

        raw = self.client.get(
            self.collection_name,
            ids=[id],
            output_fields=output_fields,
        )
        if not raw:
            raise ValueError(f"Document not found: {id}")

        result = self._parse_search_hit(raw[0].get(self.id_field), raw[0], None)
        if include_vectors:
            result.document.vector = raw[0].get(self.vector_field)
        return result.document

    def count(self) -> int:
        """Return number of entities in the collection."""
        self._ensure_pymilvus()
        if not self.client.has_collection(self.collection_name):
            return 0
        desc = self.client.describe_collection(self.collection_name)
        return desc.get("num_entities", 0)

    def remove(self, ids: list[str]) -> None:
        """Delete documents by their IDs."""
        self._ensure_pymilvus()
        self.create_index()
        self.client.delete(self.collection_name, ids=ids)
        try:
            self.client.flush(self.collection_name)
        except Exception:
            logger.debug("Milvus flush failed after delete", exc_info=True)

    def update(self, document: VectorStoreDocument) -> None:
        """Replace a document (delete + re-insert)."""
        self.remove([str(document.id)])
        self.load_documents([document])

    # ------------------------------------------------------------------
    # Schema and index helpers
    # ------------------------------------------------------------------

    def _build_milvus_schema(self) -> Any:
        """Build a MilvusClient schema with id, vector, json_data, date fields."""
        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field(
            self.id_field,
            datatype=DataType.VARCHAR,
            max_length=1024,
            is_primary=True,
        )
        schema.add_field(
            self.vector_field,
            datatype=DataType.FLOAT_VECTOR,
            dim=self.vector_size,
        )
        for name, field_type in (self.fields or {}).items():
            if name in {self.id_field, self.vector_field}:
                continue
            dtype = _field_type_to_milvus_dtype(field_type)
            if dtype == DataType.VARCHAR:
                schema.add_field(name, datatype=dtype, max_length=1024)
            else:
                schema.add_field(name, datatype=dtype)
        schema.add_field("json_data", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(self.create_date_field, datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(self.update_date_field, datatype=DataType.VARCHAR, max_length=64)
        return schema

    def _build_milvus_index_params(self) -> Any:
        """Build index params for MilvusClient."""
        ip = MilvusClient.prepare_index_params()
        ip.add_index(
            field_name=self.vector_field,
            index_type=self.index_params.get("index_type", "IVF_FLAT"),
            metric_type=self.metric_type,
            params=self.index_params.get("params", {"nlist": 1024}),
        )
        return ip

    # ------------------------------------------------------------------
    # Result parsing
    # ------------------------------------------------------------------

    def _parse_search_hit(
        self,
        raw_id: Any,
        entity: dict[str, Any],
        score: float | None,
    ) -> VectorStoreSearchResult:
        """Parse a MilvusClient search hit into a VectorStoreSearchResult."""
        # Extract data from json_data or entity fields
        data: dict = {}
        data_value = entity.get("json_data") or ""
        if isinstance(data_value, bytes):
            data_value = data_value.decode("utf-8", errors="ignore")
        if data_value.strip():
            try:
                data = json.loads(data_value)
            except Exception:
                pass

        if not data:
            for field_name in (self.fields or {}):
                val = entity.get(field_name)
                if val is not None:
                    data[field_name] = val
            for cf in ["text", "title", "description", "summary", "full_content"]:
                val = entity.get(cf)
                if val is not None:
                    data[cf] = val

        document = VectorStoreDocument(
            id=str(raw_id) if raw_id is not None else "",
            vector=None,
            data=data,
            create_date=entity.get(self.create_date_field),
            update_date=entity.get(self.update_date_field),
        )
        return VectorStoreSearchResult(document=document, score=score or 0.0)

    def _score_from_distance(self, distance: float) -> float:
        """Convert Milvus distance to similarity score."""
        distance = float(distance)
        if self.metric_type == "COSINE":
            return max(-1.0, min(1.0, 1.0 - distance))
        if self.metric_type in {"IP", "INNER_PRODUCT"}:
            return distance
        return 1.0 / (1.0 + distance)


def register_milvus_vector_store() -> None:
    register_vector_store("milvus", MilvusVectorStore)


register_milvus_vector_store()
