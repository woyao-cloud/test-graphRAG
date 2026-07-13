"""Generate embeddings and populate Milvus for querying.
Run this after graphrag index completes when the index pipeline's
embedding step fails (e.g., DeepSeek doesn't support response_format).

Uses the modern MilvusClient API (no deprecation warnings).
"""
import logging
from pathlib import Path

import pandas as pd
from pymilvus import DataType, MilvusClient

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("embed_gen")

OUTPUT_DIR = Path("D:/claude-code-project/graphRAG/output")

# Milvus connection
MILVUS_HOST = "localhost"
MILVUS_PORT = 19530

# Ollama BGE-M3 embedding endpoint
EMBED_URL = "http://localhost:11434/api/embed"
EMBED_MODEL = "bge-m3"
VECTOR_DIM = 1024


def _get_client() -> MilvusClient:
    """Get a MilvusClient instance."""
    return MilvusClient(host=MILVUS_HOST, port=MILVUS_PORT)


def _ensure_collection(name: str, client: MilvusClient) -> None:
    """Create a Milvus collection with proper schema if it doesn't exist."""
    if client.has_collection(name):
        return

    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
    schema.add_field("id", datatype=DataType.VARCHAR, max_length=1024, is_primary=True)
    schema.add_field("vector", datatype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM)
    schema.add_field("json_data", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field("create_date", datatype=DataType.VARCHAR, max_length=64)
    schema.add_field("update_date", datatype=DataType.VARCHAR, max_length=64)

    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="IVF_FLAT",
        metric_type="IP",
        params={"nlist": 1024},
    )

    client.create_collection(collection_name=name, schema=schema, index_params=index_params)
    client.load_collection(name)
    log.info("Created collection '%s' with %d-dim vectors", name, VECTOR_DIM)


def get_embedding(text: str) -> list[float]:
    """Get embedding vector from Ollama BGE-M3."""
    import requests

    resp = requests.post(
        EMBED_URL,
        json={"model": EMBED_MODEL, "input": text},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["embeddings"][0]


def generate_text_unit_embeddings(client: MilvusClient) -> None:
    """Generate text unit embeddings and store in Milvus."""
    path = OUTPUT_DIR / "text_units.parquet"
    if not path.exists():
        log.warning("No text_units.parquet found")
        return

    df = pd.read_parquet(path)
    if len(df) == 0:
        log.warning("text_units.parquet is empty")
        return

    log.info("Generating embeddings for %d text units...", len(df))
    _ensure_collection("text_unit_text", client)

    records = []
    for i, row in df.iterrows():
        text = str(row.get("text", ""))
        tid = str(row.get("id", f"tu_{i}"))
        if not text:
            continue

        try:
            vec = get_embedding(text)
            records.append({
                "id": tid,
                "vector": vec,
                "text": text,
                "n_tokens": int(row.get("n_tokens", 0)),
                "document_id": str(row.get("document_id", "")),
                "json_data": "{}",
                "create_date": "",
                "update_date": "",
            })
            if (i + 1) % 5 == 0:
                log.info("  %d/%d text units done", i + 1, len(df))
        except Exception as e:
            log.warning("Failed to embed text unit %s: %s", tid, e)

    if records:
        client.insert("text_unit_text", records)
        client.flush("text_unit_text")
        log.info("Stored %d text unit embeddings in Milvus", len(records))


def generate_entity_embeddings(client: MilvusClient) -> None:
    """Generate entity description embeddings."""
    path = OUTPUT_DIR / "entities.parquet"
    if not path.exists():
        log.warning("No entities.parquet found")
        return

    df = pd.read_parquet(path)
    if len(df) == 0:
        return

    log.info("Generating embeddings for %d entities...", len(df))
    _ensure_collection("entity_description", client)

    records = []
    for i, row in df.iterrows():
        desc = str(row.get("description", ""))
        eid = str(row.get("id", f"ent_{i}"))
        name = str(row.get("title", ""))
        text = f"{name}: {desc}" if desc else name
        if not text.strip():
            continue

        try:
            vec = get_embedding(text)
            records.append({
                "id": eid,
                "vector": vec,
                "title": name,
                "type": str(row.get("type", "")),
                "description": desc,
                "json_data": "{}",
                "create_date": "",
                "update_date": "",
            })
            if (i + 1) % 10 == 0:
                log.info("  %d/%d entities done", i + 1, len(df))
        except Exception as e:
            log.warning("Failed to embed entity %s: %s", eid, e)

    if records:
        client.insert("entity_description", records)
        client.flush("entity_description")
        log.info("Stored %d entity embeddings in Milvus", len(records))


def generate_community_embeddings(client: MilvusClient) -> None:
    """Generate community report embeddings."""
    path = OUTPUT_DIR / "community_reports.parquet"
    if not path.exists():
        log.info("No community_reports.parquet (expected in fast mode)")
        return

    df = pd.read_parquet(path)
    if len(df) == 0:
        return

    log.info("Generating embeddings for %d community reports...", len(df))
    _ensure_collection("community_full_content", client)

    records = []
    for i, row in df.iterrows():
        content = str(row.get("full_content", row.get("content", "")))
        cid = str(row.get("id", f"comm_{i}"))
        title = str(row.get("title", ""))
        text = f"{title}: {content}" if content else title
        if not text.strip():
            continue

        try:
            vec = get_embedding(text)
            records.append({
                "id": cid,
                "vector": vec,
                "title": title,
                "summary": str(row.get("summary", "")),
                "full_content": content,
                "level": int(row.get("level", 0)),
                "json_data": "{}",
                "create_date": "",
                "update_date": "",
            })
            if (i + 1) % 5 == 0:
                log.info("  %d/%d communities done", i + 1, len(df))
        except Exception as e:
            log.warning("Failed to embed community %s: %s", cid, e)

    if records:
        client.insert("community_full_content", records)
        client.flush("community_full_content")
        log.info("Stored %d community embeddings in Milvus", len(records))


if __name__ == "__main__":
    log.info("=" * 50)
    log.info("Generating missing embeddings for Milvus")
    log.info("Using %s via Ollama at %s", EMBED_MODEL, EMBED_URL)
    log.info("Vector dimension: %d", VECTOR_DIM)
    log.info("=" * 50)

    client = _get_client()

    generate_text_unit_embeddings(client)
    generate_entity_embeddings(client)
    generate_community_embeddings(client)

    log.info("=" * 50)
    log.info("Done! Verifying Milvus collections...")
    for name in client.list_collections():
        desc = client.describe_collection(name)
        log.info("  %s: %s entities", name, desc.get("num_entities", "?"))
    log.info("=" * 50)
