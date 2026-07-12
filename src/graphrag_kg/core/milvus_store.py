from __future__ import annotations

import json
import logging
from typing import Any

from graphrag_vectors.vector_store import VectorStore, VectorStoreDocument, VectorStoreSearchResult
from graphrag_vectors.vector_store_factory import register_vector_store

logger = logging.getLogger("graphrag_kg.core.milvus_store")

try:
    from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility
except ImportError:  # pragma: no cover
    Collection = None
    CollectionSchema = None
    DataType = None
    FieldSchema = None
    connections = None
    utility = None
    _PYMILVUS_AVAILABLE = False
else:
    _PYMILVUS_AVAILABLE = True


def _field_type_to_milvus_dtype(field_type: str) -> Any:
    field_type = field_type.lower()
    if field_type in ("str", "string"):
        return DataType.VARCHAR
    if field_type in ("int", "integer"):
        return DataType.INT64
    if field_type in ("float", "double"):
        return DataType.DOUBLE
    if field_type in ("bool", "boolean"):
        return DataType.BOOL
    if field_type == "date":
        return DataType.VARCHAR
    return DataType.VARCHAR


class MilvusVectorStore(VectorStore):
    """A Milvus-backed vector store for GraphRAG custom vector storage."""

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
        metric_type: str = "COSINE",
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
        self.collection_name = collection_name or index_name
        self.metric_type = metric_type.upper()
        self.index_params = index_params or {
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024},
        }
        self._connection_alias = f"milvus_{self.collection_name}"
        self._collection = None

    def _ensure_pymilvus(self) -> None:
        if not _PYMILVUS_AVAILABLE:
            raise RuntimeError(
                "Milvus vector store requires pymilvus. Install it with 'pip install pymilvus'."
            )

    def _connect(self) -> None:
        self._ensure_pymilvus()

        if not connections.has_connection(self._connection_alias):
            connections.connect(
                alias=self._connection_alias,
                host=self.host,
                port=self.port,
            )

        if utility.has_collection(self.collection_name, using=self._connection_alias):
            self._collection = Collection(
                self.collection_name, using=self._connection_alias
            )
        else:
            self._collection = None

    def _build_collection_fields(self) -> list[Any]:
        if not _PYMILVUS_AVAILABLE:
            raise RuntimeError("pymilvus is not installed")

        fields: list[Any] = [
            FieldSchema(
                name=self.id_field,
                dtype=DataType.VARCHAR,
                is_primary=True,
                max_length=1024,
            ),
            FieldSchema(
                name=self.vector_field,
                dtype=DataType.FLOAT_VECTOR,
                dim=self.vector_size,
            ),
        ]

        for name, field_type in (self.fields or {}).items():
            if name in {self.id_field, self.vector_field}:
                continue
            dtype = _field_type_to_milvus_dtype(field_type)
            if dtype == DataType.VARCHAR:
                fields.append(
                    FieldSchema(name=name, dtype=dtype, max_length=1024)
                )
            else:
                fields.append(FieldSchema(name=name, dtype=dtype))

        fields.extend(
            [
                FieldSchema(
                    name="json_data",
                    dtype=DataType.VARCHAR,
                    max_length=65535,
                ),
                FieldSchema(
                    name=self.create_date_field,
                    dtype=DataType.VARCHAR,
                    max_length=64,
                ),
                FieldSchema(
                    name=self.update_date_field,
                    dtype=DataType.VARCHAR,
                    max_length=64,
                ),
            ]
        )

        return fields

    def connect(self) -> None:
        self._connect()

    def create_index(self) -> None:
        self._ensure_pymilvus()
        self._connect()

        if self._collection is None:
            schema = CollectionSchema(
                fields=self._build_collection_fields(),
                description="GraphRAG Milvus vector store schema",
            )
            self._collection = Collection(
                self.collection_name,
                schema,
                using=self._connection_alias,
            )

        try:
            self._collection.create_index(
                field_name=self.vector_field,
                index_params={
                    "index_type": self.index_params.get(
                        "index_type", "IVF_FLAT"
                    ),
                    "metric_type": self.metric_type,
                    "params": self.index_params.get(
                        "params", {"nlist": 1024}
                    ),
                },
            )
        except Exception:
            logger.debug("Milvus index already exists or create_index failed", exc_info=True)

        self._collection.load()

    def _serialize_document(self, document: VectorStoreDocument) -> dict[str, list[Any]]:
        self._prepare_document(document)

        row: dict[str, list[Any]] = {
            self.id_field: [str(document.id)],
            self.vector_field: [document.vector or []],
            "json_data": [json.dumps(document.data or {}, ensure_ascii=False)],
            self.create_date_field: [document.create_date or ""],
            self.update_date_field: [document.update_date or ""],
        }

        for name in (self.fields or {}).keys():
            if name in {self.id_field, self.vector_field}:
                continue
            row[name] = [document.data.get(name) if document.data else None]

        return row

    def load_documents(self, documents: list[VectorStoreDocument]) -> None:
        self._ensure_pymilvus()
        self.create_index()

        rows = [self._serialize_document(document) for document in documents]
        if not rows:
            return

        columns: dict[str, list[Any]] = {}
        for row in rows:
            for key, values in row.items():
                columns.setdefault(key, []).extend(values)

        self._collection.insert(columns)
        try:
            self._collection.flush()
        except Exception:
            logger.debug("Milvus flush failed or is unsupported", exc_info=True)

    def _extract_field(self, entity: Any, name: str) -> Any:
        if entity is None:
            return None
        if isinstance(entity, dict):
            return entity.get(name)
        if hasattr(entity, "get"):
            return entity.get(name)
        if hasattr(entity, name):
            return getattr(entity, name)
        try:
            return entity[name]
        except Exception:
            return None

    def _parse_entity(self, entity: Any, score: float | None) -> VectorStoreSearchResult:
        data_value = self._extract_field(entity, "json_data") or "{}"
        if isinstance(data_value, bytes):
            data_value = data_value.decode("utf-8", errors="ignore")
        try:
            data = json.loads(data_value)
        except Exception:
            data = {}

        raw_id = self._extract_field(entity, self.id_field)
        document = VectorStoreDocument(
            id=str(raw_id) if raw_id is not None else "",
            vector=None,
            data=data,
            create_date=self._extract_field(entity, self.create_date_field),
            update_date=self._extract_field(entity, self.update_date_field),
        )

        if score is None:
            score = 0.0

        return VectorStoreSearchResult(document=document, score=score)

    def _score_from_distance(self, distance: float) -> float:
        distance = float(distance)
        if self.metric_type == "COSINE":
            return max(-1.0, min(1.0, 1.0 - distance))
        if self.metric_type in {"IP", "INNER_PRODUCT"}:
            return distance
        return 1.0 / (1.0 + distance)

    def similarity_search_by_vector(
        self,
        query_embedding: list[float],
        k: int = 10,
        select: list[str] | None = None,
        filters: Any = None,
        include_vectors: bool = True,
    ) -> list[VectorStoreSearchResult]:
        self._ensure_pymilvus()
        self.create_index()

        output_fields = []
        if select:
            output_fields.extend(select)
        else:
            output_fields = [
                self.id_field,
                "json_data",
                self.create_date_field,
                self.update_date_field,
            ]

        if include_vectors:
            output_fields.append(self.vector_field)

        search_params = {
            "metric_type": self.metric_type,
            "params": self.index_params.get("search_params", {"nprobe": 10}),
        }

        search_result = self._collection.search(
            data=[query_embedding],
            anns_field=self.vector_field,
            param=search_params,
            limit=k,
            output_fields=output_fields,
        )

        hits = search_result[0] if search_result else []
        results: list[VectorStoreSearchResult] = []

        for hit in hits:
            entity = getattr(hit, "entity", None) or hit
            distance = getattr(hit, "distance", None) or getattr(hit, "score", None)
            score = self._score_from_distance(distance) if distance is not None else 0.0
            result = self._parse_entity(entity, score)
            if include_vectors and hasattr(entity, self.vector_field):
                result.document.vector = self._extract_field(entity, self.vector_field)
            results.append(result)

        return results

    def search_by_id(
        self,
        id: str,
        select: list[str] | None = None,
        include_vectors: bool = True,
    ) -> VectorStoreDocument:
        self._ensure_pymilvus()
        self.create_index()

        entity = self._collection.query(
            expr=f"{self.id_field} == '{id}'",
            output_fields=select or [
                self.id_field,
                "json_data",
                self.create_date_field,
                self.update_date_field,
            ],
        )

        if not entity:
            raise ValueError(f"Document not found: {id}")

        result = self._parse_entity(entity[0], None)
        if include_vectors:
            result.document.vector = self._extract_field(
                entity[0], self.vector_field
            )
        return result.document

    def count(self) -> int:
        self._ensure_pymilvus()
        self._connect()
        if self._collection is None:
            return 0
        return self._collection.num_entities

    def remove(self, ids: list[str]) -> None:
        self._ensure_pymilvus()
        self.create_index()
        expr = " or ".join(
            f"{self.id_field} == '{item}'" for item in ids
        )
        self._collection.delete(expr)
        try:
            self._collection.flush()
        except Exception:
            logger.debug("Milvus flush failed or unsupported after delete", exc_info=True)

    def update(self, document: VectorStoreDocument) -> None:
        self._ensure_pymilvus()
        self.remove([str(document.id)])
        self.load_documents([document])


def register_milvus_vector_store() -> None:
    register_vector_store("milvus", MilvusVectorStore)


register_milvus_vector_store()
