"""
ch09-crud-basics: Milvus CRUD operations with MilvusClient API.

Demonstrates:
  1. Create collection with schema
  2. Insert single record
  3. Batch insert 20 records
  4. Create IVF_FLAT index
  5. Search by vector similarity
  6. Search with filter
  7. Get by ID
  8. Upsert (update existing)
  9. Delete by ID
  10. Drop collection
"""

import os
import random
import time

import numpy as np
from pymilvus import MilvusClient, DataType

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
COLLECTION_NAME = "ch09_crud_demo"
DIM = 128


def print_step(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def make_random_vector(dim: int = DIM) -> list[float]:
    return [random.random() for _ in range(dim)]


def main() -> None:
    # ------------------------------------------------------------------ #
    # 0. Connect
    # ------------------------------------------------------------------ #
    print_step("0. Connecting to Milvus")
    client = MilvusClient(uri=f"http://{MILVUS_HOST}:19530")
    print(f"   Connected to {MILVUS_HOST}:19530")

    # Drop if exists from a previous run
    client.drop_collection(COLLECTION_NAME)

    # ------------------------------------------------------------------ #
    # 1. Create collection with schema
    # ------------------------------------------------------------------ #
    print_step("1. Create collection with schema")

    schema = MilvusClient.create_schema(
        auto_id=False,
        enable_dynamic_field=False,
    )
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=DIM)
    schema.add_field("text", DataType.VARCHAR, max_length=512)
    schema.add_field("category", DataType.VARCHAR, max_length=64)
    schema.add_field("timestamp", DataType.INT64)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
    )
    print(f"   Collection '{COLLECTION_NAME}' created.")

    # ------------------------------------------------------------------ #
    # 2. Insert single record
    # ------------------------------------------------------------------ #
    print_step("2. Insert single record")

    single = {
        "id": 1,
        "vector": make_random_vector(),
        "text": "阿司匹林用于解热镇痛",
        "category": "解热镇痛",
        "timestamp": int(time.time()),
    }
    res = client.insert(COLLECTION_NAME, single)
    print(f"   Inserted 1 record. Insert count: {res['insert_count']}")

    # ------------------------------------------------------------------ #
    # 3. Batch insert 20 records (Chinese medical texts)
    # ------------------------------------------------------------------ #
    print_step("3. Batch insert 20 records")

    drugs = [
        "阿司匹林", "布洛芬", "对乙酰氨基酚", "头孢克肟", "阿莫西林",
        "红霉素", "奥美拉唑", "二甲双胍", "硝苯地平", "卡托普利",
        "氯沙坦", "阿托伐他汀", "辛伐他汀", "氨氯地平", "美托洛尔",
        "地高辛", "华法林", "胰岛素", "格列本脲", "泼尼松",
    ]
    categories = [
        "解热镇痛", "解热镇痛", "解热镇痛", "抗生素", "抗生素",
        "抗生素", "消化系统", "降糖药", "降压药", "降压药",
        "降压药", "降脂药", "降脂药", "降压药", "降压药",
        "强心药", "抗凝血", "降糖药", "降糖药", "抗肿瘤",
    ]

    batch = []
    for i, (drug, cat) in enumerate(zip(drugs, categories), start=2):
        batch.append({
            "id": i,
            "vector": make_random_vector(),
            "text": f"{drug}是常用的{categories[i-2]}药物",
            "category": cat,
            "timestamp": int(time.time()),
        })

    res = client.insert(COLLECTION_NAME, batch)
    print(f"   Inserted {len(batch)} records. Insert count: {res['insert_count']}")

    # ------------------------------------------------------------------ #
    # 4. Create IVF_FLAT index
    # ------------------------------------------------------------------ #
    print_step("4. Create IVF_FLAT index")

    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="IVF_FLAT",
        metric_type="IP",
        params={"nlist": 128},
    )
    client.create_index(COLLECTION_NAME, index_params)
    print("   IVF_FLAT index created on 'vector' field.")

    client.load_collection(COLLECTION_NAME)
    print("   Collection loaded into memory.")

    # ------------------------------------------------------------------ #
    # 5. Search by vector similarity
    # ------------------------------------------------------------------ #
    print_step("5. Search by vector similarity")

    query_vec = make_random_vector()
    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vec],
        anns_field="vector",
        limit=3,
        output_fields=["id", "text", "category"],
    )
    print("   Top 3 results:")
    for i, hit in enumerate(results[0], start=1):
        entity = hit["entity"]
        print(f"     {i}. id={entity['id']}  text={entity['text']}  "
              f"category={entity['category']}  score={hit['distance']:.4f}")

    # ------------------------------------------------------------------ #
    # 6. Search with filter (category == "抗肿瘤")
    # ------------------------------------------------------------------ #
    print_step('6. Search with filter (category == "抗肿瘤")')

    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vec],
        anns_field="vector",
        limit=3,
        output_fields=["id", "text", "category"],
        filter='category == "抗肿瘤"',
    )
    print("   Top 3 results filtered by 抗肿瘤:")
    if results[0]:
        for i, hit in enumerate(results[0], start=1):
            entity = hit["entity"]
            print(f"     {i}. id={entity['id']}  text={entity['text']}  "
                  f"score={hit['distance']:.4f}")
    else:
        print("     (no matching results)")

    # ------------------------------------------------------------------ #
    # 7. Get by ID
    # ------------------------------------------------------------------ #
    print_step("7. Get by ID")

    res = client.get(COLLECTION_NAME, ids=[1, 5, 10])
    print("   Retrieved records:")
    for row in res:
        print(f"     id={row['id']}  text={row['text']}  category={row['category']}")

    # ------------------------------------------------------------------ #
    # 8. Upsert (update existing)
    # ------------------------------------------------------------------ #
    print_step("8. Upsert (update existing record)")

    updated = {
        "id": 1,
        "vector": make_random_vector(),
        "text": "阿司匹林——经典解热镇痛药，更新版说明",
        "category": "解热镇痛",
        "timestamp": int(time.time()),
    }
    res = client.upsert(COLLECTION_NAME, updated)
    print(f"   Upsert count: {res['upsert_count']}")

    # Verify
    row = client.get(COLLECTION_NAME, ids=[1])[0]
    print(f"   After upsert, id=1 text: {row['text']}")

    # ------------------------------------------------------------------ #
    # 9. Delete by ID
    # ------------------------------------------------------------------ #
    print_step("9. Delete by ID")

    res = client.delete(COLLECTION_NAME, ids=[3])
    print(f"   Deleted id=3. Delete count: {res['delete_count']}")

    # Verify
    remaining = client.get(COLLECTION_NAME, ids=[3])
    print(f"   Get id=3 after delete: {remaining}")

    # ------------------------------------------------------------------ #
    # 10. Drop collection
    # ------------------------------------------------------------------ #
    print_step("10. Drop collection")

    client.drop_collection(COLLECTION_NAME)
    print(f"   Collection '{COLLECTION_NAME}' dropped.")

    print(f"\n{'=' * 60}")
    print("  CRUD demo completed successfully!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        raise
