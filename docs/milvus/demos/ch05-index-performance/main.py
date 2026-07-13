"""
ch05-index-performance: Compare Milvus index types — FLAT, IVF_FLAT, HNSW.

Creates 3 collections with different index types, inserts 1000 random
128-dim vectors into each, times queries, and prints a comparison table.
"""

import os
import random
import time

import numpy as np
from pymilvus import MilvusClient, DataType

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
DIM = 128
NUM_VECTORS = 1000
NUM_QUERIES = 10
TOP_K = 10

INDEX_CONFIGS = [
    {
        "name": "FLAT",
        "index_type": "FLAT",
        "metric_type": "L2",
        "params": {},
    },
    {
        "name": "IVF_FLAT",
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 128},
    },
    {
        "name": "HNSW",
        "index_type": "HNSW",
        "metric_type": "L2",
        "params": {"M": 16, "efConstruction": 200},
    },
]


def make_vectors(n: int, dim: int = DIM) -> list[list[float]]:
    return [[random.random() for _ in range(dim)] for _ in range(n)]


def create_collection(client: MilvusClient, name: str) -> None:
    client.drop_collection(name)
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=DIM)
    client.create_collection(collection_name=name, schema=schema)


def build_index(client: MilvusClient, name: str, idx_cfg: dict) -> float:
    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type=idx_cfg["index_type"],
        metric_type=idx_cfg["metric_type"],
        params=idx_cfg["params"],
    )
    t0 = time.perf_counter()
    client.create_index(name, index_params)
    client.load_collection(name)
    # Force index build by running a dummy search
    dummy = [[random.random() for _ in range(DIM)]]
    client.search(
        collection_name=name,
        data=dummy,
        anns_field="vector",
        limit=1,
    )
    elapsed = time.perf_counter() - t0
    return elapsed


def benchmark_queries(
    client: MilvusClient,
    name: str,
    queries: list[list[float]],
    top_k: int = TOP_K,
) -> float:
    times = []
    for q in queries:
        t0 = time.perf_counter()
        client.search(
            collection_name=name,
            data=[q],
            anns_field="vector",
            limit=top_k,
        )
        times.append(time.perf_counter() - t0)
    return sum(times) / len(times)


def recall_vs_reference(
    client: MilvusClient,
    ref_name: str,
    test_name: str,
    queries: list[list[float]],
    top_k: int = TOP_K,
) -> float:
    """Compute recall@top_k using the reference collection as ground truth."""
    ref_results = client.search(
        collection_name=ref_name,
        data=queries,
        anns_field="vector",
        limit=top_k,
    )
    test_results = client.search(
        collection_name=test_name,
        data=queries,
        anns_field="vector",
        limit=top_k,
    )

    hits = 0
    total = 0
    for ref_hits, test_hits in zip(ref_results, test_results):
        ref_ids = {h["id"] for h in ref_hits}
        for h in test_hits:
            if h["id"] in ref_ids:
                hits += 1
            total += 1
    return hits / total if total > 0 else 0.0


def main() -> None:
    client = MilvusClient(uri=f"http://{MILVUS_HOST}:19530")
    print(f"Connected to {MILVUS_HOST}:19530\n")

    # Pre-generate vectors (same data for all collections)
    vectors = make_vectors(NUM_VECTORS)
    query_vectors = make_vectors(NUM_QUERIES)

    results = []

    for cfg in INDEX_CONFIGS:
        coll_name = f"idx_{cfg['name'].lower()}"
        print(f"--- {cfg['name']} ---")

        create_collection(client, coll_name)

        # Insert data
        data = [
            {"id": i, "vector": v}
            for i, v in enumerate(vectors)
        ]
        client.insert(coll_name, data)
        print(f"  Inserted {NUM_VECTORS} vectors")

        # Build index
        build_time = build_index(client, coll_name, cfg)
        print(f"  Index build time: {build_time:.4f}s")

        # Query
        avg_time = benchmark_queries(client, coll_name, query_vectors)
        print(f"  Avg query time ({NUM_QUERIES} queries): {avg_time * 1000:.3f}ms")

        results.append({
            "name": cfg["name"],
            "build_time": build_time,
            "avg_query_ms": avg_time * 1000,
        })

    # Compute recall vs FLAT (exact search) for IVF_FLAT and HNSW
    ref_name = "idx_flat"
    for r in results:
        if r["name"] == "FLAT":
            r["recall"] = 1.0
        else:
            coll_name = f"idx_{r['name'].lower()}"
            rec = recall_vs_reference(client, ref_name, coll_name, query_vectors)
            r["recall"] = rec

    # ------------------------------------------------------------------ #
    # Print comparison table
    # ------------------------------------------------------------------ #
    print()
    print("=" * 68)
    print("  Index Comparison Summary")
    print("=" * 68)
    print(f"  {'Index':<12} {'Build Time':<14} {'Avg Query':<14} {'Recall vs FLAT'}")
    print(f"  {'-' * 12} {'-' * 14} {'-' * 14} {'-' * 14}")
    for r in results:
        print(f"  {r['name']:<12} {r['build_time']:<10.4f}s  "
              f"{r['avg_query_ms']:<10.3f}ms  {r['recall']:.4f}")
    print("=" * 68)

    # ------------------------------------------------------------------ #
    # Cleanup
    # ------------------------------------------------------------------ #
    for cfg in INDEX_CONFIGS:
        coll_name = f"idx_{cfg['name'].lower()}"
        client.drop_collection(coll_name)
        print(f"  Dropped collection '{coll_name}'")

    print("\nIndex performance demo completed successfully!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        raise
