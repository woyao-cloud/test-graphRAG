"""
ch13-data-pipeline: Data Processing Pipeline for RAG
Demonstrates smart chunking, batch insert, incremental update, and data cleanup.
Uses sample Chinese medical text about 肿瘤治疗 (cancer treatment).
"""

import hashlib
import re
import time
from datetime import datetime

from pymilvus import MilvusClient

# ── Configuration ──────────────────────────────────────────────────────────────
MILVUS_URI = "http://localhost:19530"
COLLECTION_NAME = "data_pipeline_demo"
DIM = 8  # small dimension for demo

# ── 1. Smart Chunking ──────────────────────────────────────────────────────────

SAMPLE_TEXT = """
肿瘤治疗是现代医学的重要领域。随着医学技术的不断进步，肿瘤治疗的方法也在不断更新。

手术切除是早期肿瘤的主要治疗手段。对于局限性实体肿瘤，手术可以直接切除肿瘤组织。
然而，手术并非适用于所有患者。肿瘤的位置、大小以及患者的身体状况都会影响手术的可行性。

放射治疗利用高能量射线杀死癌细胞。放疗可以单独使用，也可以与手术或化疗联合使用。
现代放疗技术包括IMRT（调强放疗）和SBRT（立体定向体部放疗），能够精准定位肿瘤区域。

化学治疗使用药物杀死快速分裂的细胞。化疗药物通过血液循环到达全身，对已经扩散的肿瘤有效。
但化疗也会影响正常细胞，导致脱发、恶心等副作用。

靶向治疗是针对特定基因突变设计的药物。与化疗不同，靶向药物只攻击携带特定突变的癌细胞。
常见的靶向治疗包括针对EGFR突变、ALK融合等靶点的药物。

免疫治疗通过激活人体自身免疫系统来对抗肿瘤。PD-1/PD-L1抑制剂是免疫治疗的代表药物。
免疫治疗在某些肿瘤类型中取得了显著效果，如黑色素瘤和非小细胞肺癌。

中医治疗在肿瘤综合治疗中发挥辅助作用。中药可以减轻放化疗的副作用，提高患者生活质量。
常用的中医方法包括中药汤剂、针灸和推拿。

肿瘤预防同样重要。健康的生活方式、定期体检和早期筛查可以降低肿瘤发生率和死亡率。
"""


def split_by_paragraphs(text: str) -> list[str]:
    """Split text by paragraphs (double newlines)."""
    paragraphs = [p.strip() for p in text.strip().split("\n\n") if p.strip()]
    return paragraphs


def split_by_sentences(text: str) -> list[str]:
    """Split text by Chinese/English sentence endings."""
    sentences = re.split(r"(?<=[。！？.!?])\s*", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(
    text: str,
    method: str = "paragraph",
    max_chars: int = 200,
    overlap_chars: int = 30,
) -> list[dict]:
    """
    Smart chunking: split text into chunks with metadata.
    Returns list of dicts: {chunk_index, text, char_count}.
    """
    chunks = []

    if method == "paragraph":
        segments = split_by_paragraphs(text)
    elif method == "sentence":
        segments = split_by_sentences(text)
    else:
        segments = [text]

    buffer = ""
    chunk_idx = 0

    for seg in segments:
        if len(buffer) + len(seg) <= max_chars:
            buffer += seg + (" " if method == "sentence" else "\n")
        else:
            if buffer:
                chunks.append(
                    {
                        "chunk_index": chunk_idx,
                        "text": buffer.strip(),
                        "char_count": len(buffer.strip()),
                    }
                )
                chunk_idx += 1
            # overlap: keep last overlap_chars worth of text
            overlap = buffer[-overlap_chars:] if len(buffer) >= overlap_chars else buffer
            buffer = overlap + seg + (" " if method == "sentence" else "\n")

    if buffer:
        chunks.append(
            {
                "chunk_index": chunk_idx,
                "text": buffer.strip(),
                "char_count": len(buffer.strip()),
            }
        )

    return chunks


def generate_embedding(text: str, dim: int = DIM) -> list[float]:
    """Simulate embedding generation (hash-based deterministic vector)."""
    h = hashlib.md5(text.encode()).hexdigest()
    return [((int(h[i : i + 2], 16) / 255.0) * 2 - 1) for i in range(0, dim * 2, 2)]


# ── 2. Batch Insert ────────────────────────────────────────────────────────────


def batch_insert(
    client: MilvusClient,
    chunks: list[dict],
    source_id: str,
    batch_size: int = 5,
) -> list[str]:
    """Insert chunks into Milvus in batches. Returns list of inserted IDs."""
    entities = []
    now = datetime.now().isoformat()

    for chunk in chunks:
        text = chunk["text"]
        entities.append(
            {
                "id": hashlib.md5(f"{source_id}:{chunk['chunk_index']}".encode()).hexdigest(),
                "vector": generate_embedding(text),
                "text": text,
                "source_id": source_id,
                "chunk_index": chunk["chunk_index"],
                "char_count": chunk["char_count"],
                "created_at": now,
            }
        )

    ids = []
    for i in range(0, len(entities), batch_size):
        batch = entities[i : i + batch_size]
        result = client.insert(collection_name=COLLECTION_NAME, data=batch)
        ids.extend(result)
        print(f"  Inserted batch of {len(batch)} chunks")

    return ids


# ── 3. Incremental Update ──────────────────────────────────────────────────────


def incremental_update(
    client: MilvusClient,
    new_chunks: list[dict],
    source_id: str,
) -> tuple[list[str], list[str]]:
    """
    Incremental update: insert new chunks, identify duplicates.
    Returns (inserted_ids, duplicate_ids).
    """
    inserted_ids = []
    duplicate_ids = []

    for chunk in new_chunks:
        chunk_id = hashlib.md5(f"{source_id}:{chunk['chunk_index']}".encode()).hexdigest()
        # Check if this chunk already exists
        existing = client.get(collection_name=COLLECTION_NAME, ids=[chunk_id])
        if existing:
            duplicate_ids.append(chunk_id)
            print(f"  Duplicate found: chunk {chunk['chunk_index']}")
        else:
            entity = {
                "id": chunk_id,
                "vector": generate_embedding(chunk["text"]),
                "text": chunk["text"],
                "source_id": source_id,
                "chunk_index": chunk["chunk_index"],
                "char_count": chunk["char_count"],
                "created_at": datetime.now().isoformat(),
            }
            client.insert(collection_name=COLLECTION_NAME, data=[entity])
            inserted_ids.append(chunk_id)
            print(f"  Inserted new chunk {chunk['chunk_index']}")

    return inserted_ids, duplicate_ids


# ── 4. Data Cleanup ────────────────────────────────────────────────────────────


def cleanup_by_timestamp(client: MilvusClient, before_timestamp: str) -> int:
    """
    Remove chunks created before a given timestamp.
    Uses query + delete pattern since MilvusClient does not support range delete.
    """
    # Query for old entities
    results = client.query(
        collection_name=COLLECTION_NAME,
        filter=f'created_at < "{before_timestamp}"',
        output_fields=["id"],
    )

    if not results:
        print("  No outdated chunks found")
        return 0

    ids_to_delete = [r["id"] for r in results]
    client.delete(collection_name=COLLECTION_NAME, ids=ids_to_delete)
    print(f"  Deleted {len(ids_to_delete)} outdated chunks")
    return len(ids_to_delete)


# ── Main Demo ──────────────────────────────────────────────────────────────────


def ensure_collection(client: MilvusClient):
    """Create collection if not exists."""
    if client.has_collection(COLLECTION_NAME):
        client.drop_collection(COLLECTION_NAME)

    schema = MilvusClient.create_schema(
        auto_id=False,
        enable_dynamic_field=False,
    )
    schema.add_field("id", datatype="VARCHAR", max_length=64, is_primary=True)
    schema.add_field("vector", datatype="FLOAT_VECTOR", dim=DIM)
    schema.add_field("text", datatype="VARCHAR", max_length=1024)
    schema.add_field("source_id", datatype="VARCHAR", max_length=64)
    schema.add_field("chunk_index", datatype="INT32")
    schema.add_field("char_count", datatype="INT32")
    schema.add_field("created_at", datatype="VARCHAR", max_length=32)

    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(field_name="vector", index_type="FLAT", metric_type="L2")

    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
    )
    print(f"Created collection '{COLLECTION_NAME}'")


def main():
    print("=" * 60)
    print("ch13: Data Pipeline Demo")
    print("=" * 60)

    # Step 0: Connect and create collection
    print("\n[0] Connecting to Milvus...")
    client = MilvusClient(uri=MILVUS_URI)
    ensure_collection(client)

    # Step 1: Smart chunking
    print("\n[1] Smart Chunking")
    print("-" * 40)
    print(f"Input text length: {len(SAMPLE_TEXT)} chars")

    # Method 1: by paragraph
    para_chunks = chunk_text(SAMPLE_TEXT, method="paragraph", max_chars=300)
    print(f"\n  Paragraph chunking: {len(para_chunks)} chunks")
    for c in para_chunks:
        print(f"    Chunk {c['chunk_index']}: {c['char_count']} chars - {c['text'][:60]}...")

    # Method 2: by sentence
    sent_chunks = chunk_text(SAMPLE_TEXT, method="sentence", max_chars=150, overlap_chars=20)
    print(f"\n  Sentence chunking: {len(sent_chunks)} chunks")
    for c in sent_chunks:
        print(f"    Chunk {c['chunk_index']}: {c['char_count']} chars - {c['text'][:60]}...")

    # Step 2: Batch insert
    print("\n[2] Batch Insert")
    print("-" * 40)
    source_id = "doc_cancer_treatment_v1"
    ids = batch_insert(client, para_chunks, source_id=source_id, batch_size=3)
    print(f"  Total inserted: {len(ids)} chunks")

    # Step 3: Incremental update
    print("\n[3] Incremental Update")
    print("-" * 40)
    new_text = "\n\n细胞治疗是肿瘤治疗的新兴领域。CAR-T细胞疗法在血液系统肿瘤中取得了突破性进展。"
    new_chunks = chunk_text(new_text, method="paragraph", max_chars=300)
    source_id_v2 = "doc_cancer_treatment_v2"
    inserted, duplicates = incremental_update(client, new_chunks, source_id=source_id_v2)
    print(f"  Inserted: {len(inserted)}, Duplicates: {len(duplicates)}")

    # Also try inserting same chunks again to show duplicate detection
    print("\n  (Re-inserting same chunks to show duplicate detection)")
    _, dupes = incremental_update(client, new_chunks, source_id=source_id_v2)
    print(f"  Duplicates found this round: {len(dupes)}")

    # Step 4: Data cleanup
    print("\n[4] Data Cleanup")
    print("-" * 40)
    # Delete chunks from old version
    old_ids = client.query(
        collection_name=COLLECTION_NAME,
        filter=f'source_id == "{source_id}"',
        output_fields=["id"],
    )
    if old_ids:
        ids_to_delete = [r["id"] for r in old_ids]
        client.delete(collection_name=COLLECTION_NAME, ids=ids_to_delete)
        print(f"  Cleaned up {len(ids_to_delete)} chunks from '{source_id}'")
    else:
        print("  No chunks to clean up")

    # Final collection stats
    stats = client.query(
        collection_name=COLLECTION_NAME,
        output_fields=["id", "source_id", "chunk_index", "created_at"],
        limit=100,
    )
    print(f"\n  Remaining entities: {len(stats)}")
    for entity in stats:
        print(f"    id={entity['id'][:16]}... source={entity['source_id']} chunk={entity['chunk_index']}")

    print("\n" + "=" * 60)
    print("Data Pipeline Demo Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
