"""Generate embeddings and populate LanceDB for querying.
Run this after graphrag index completes to fix missing LanceDB data.
"""
import json
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("embed_gen")

OUTPUT_DIR = Path("D:/claude-code-project/graphRAG/output")
LANCEDB_DIR = OUTPUT_DIR / "lancedb"

# Ollama BGE-M3 embedding endpoint
EMBED_URL = "http://localhost:11434/api/embed"
EMBED_MODEL = "bge-m3"
VECTOR_DIM = 1024


def get_embedding(text: str) -> list[float]:
    """Get embedding vector from Ollama BGE-M3."""
    import requests
    resp = requests.post(
        EMBED_URL,
        json={"model": EMBED_MODEL, "input": text},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["embeddings"][0]


def generate_text_unit_embeddings():
    """Generate text unit embeddings and store in LanceDB."""
    path = OUTPUT_DIR / "text_units.parquet"
    if not path.exists():
        log.warning("No text_units.parquet found")
        return

    df = pd.read_parquet(path)
    if len(df) == 0:
        log.warning("text_units.parquet is empty")
        return

    log.info(f"Generating embeddings for {len(df)} text units...")
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
            })
            if (i + 1) % 5 == 0:
                log.info(f"  {i+1}/{len(df)} text units done")
        except Exception as e:
            log.warning(f"  Failed to embed text unit {tid}: {e}")

    if records:
        import lancedb
        db = lancedb.connect(str(LANCEDB_DIR))
        tbl = db.create_table("text_unit_text", data=records, mode="overwrite")
        log.info(f"Stored {len(records)} text unit embeddings in LanceDB")


def generate_entity_embeddings():
    """Generate entity description embeddings."""
    path = OUTPUT_DIR / "entities.parquet"
    if not path.exists():
        log.warning("No entities.parquet found")
        return

    df = pd.read_parquet(path)
    if len(df) == 0:
        return

    log.info(f"Generating embeddings for {len(df)} entities...")
    records = []
    for i, row in df.iterrows():
        desc = str(row.get("description", ""))
        eid = str(row.get("id", f"ent_{i}"))
        name = str(row.get("title", ""))
        # Use name + description for better embedding
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
            })
            if (i + 1) % 10 == 0:
                log.info(f"  {i+1}/{len(df)} entities done")
        except Exception as e:
            log.warning(f"  Failed to embed entity {eid}: {e}")

    if records:
        import lancedb
        db = lancedb.connect(str(LANCEDB_DIR))
        tbl = db.create_table("entity_description", data=records, mode="overwrite")
        log.info(f"Stored {len(records)} entity embeddings in LanceDB")


def generate_community_embeddings():
    """Generate community report embeddings."""
    path = OUTPUT_DIR / "community_reports.parquet"
    if not path.exists():
        log.info("No community_reports.parquet (expected in fast mode)")
        return

    df = pd.read_parquet(path)
    if len(df) == 0:
        return

    log.info(f"Generating embeddings for {len(df)} community reports...")
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
            })
            if (i + 1) % 5 == 0:
                log.info(f"  {i+1}/{len(df)} communities done")
        except Exception as e:
            log.warning(f"  Failed to embed community {cid}: {e}")

    if records:
        import lancedb
        db = lancedb.connect(str(LANCEDB_DIR))
        tbl = db.create_table("community_full_content", data=records, mode="overwrite")
        log.info(f"Stored {len(records)} community embeddings in LanceDB")


if __name__ == "__main__":
    log.info("=" * 50)
    log.info("Generating missing embeddings for LanceDB")
    log.info(f"Using {EMBED_MODEL} via Ollama at {EMBED_URL}")
    log.info(f"Vector dimension: {VECTOR_DIM}")
    log.info("=" * 50)

    generate_text_unit_embeddings()
    generate_entity_embeddings()
    generate_community_embeddings()

    log.info("=" * 50)
    log.info("Done! Verifying LanceDB...")
    import lancedb
    db = lancedb.connect(str(LANCEDB_DIR))
    for t in db.table_names():
        tbl = db.open_table(t)
        log.info(f"  {t}: {tbl.count_rows()} rows")
    log.info("=" * 50)
