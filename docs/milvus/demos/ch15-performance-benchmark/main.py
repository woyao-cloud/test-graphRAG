"""
ch15-performance-benchmark: Performance Benchmarking for Milvus
Measures insert throughput, search QPS/latency, and tests different batch sizes.
Uses MilvusClient API.
"""

import random
import time

import numpy as np
from pymilvus import MilvusClient

# ── Configuration ──────────────────────────────────────────────────────────────
MILVUS_URI = "http://localhost:19530"
COLLECTION_NAME = "perf_benchmark"
DIM = 128
NUM_VECTORS = 500  # configurable
BATCH_SIZES = [10, 50, 100]
SEARCH_N = 50  # number of search queries for QPS test
TOP_K = 10


def generate_vectors(n: int, dim: int) -> list[list[float]]:
    """Generate random float vectors."""
    return np.random.randn(n, dim).astype(np.float32).tolist()


def ensure_collection(client: MilvusClient):
    """Create collection with IVF_FLAT index for benchmark."""
    if client.has_collection(COLLECTION_NAME):
        client.drop_collection(COLLECTION_NAME)

    schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field("id", datatype="INT64", is_primary=True, auto_id=True)
    schema.add_field("vector", datatype="FLOAT_VECTOR", dim=DIM)

    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(field_name="vector", index_type="IVF_FLAT", metric_type="L2", params={"nlist": 128})

    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
    )
    print(f"Created collection '{COLLECTION_NAME}' with dim={DIM}")


def benchmark_insert(
    client: MilvusClient,
    vectors: list[list[float]],
    batch_size: int,
) -> tuple[int, float]:
    """
    Insert vectors in given batch size.
    Returns (total_inserted, elapsed_seconds).
    """
    start = time.perf_counter()
    total = 0
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i : i + batch_size]
        data = [{"vector": v} for v in batch]
        client.insert(collection_name=COLLECTION_NAME, data=data)
        total += len(batch)
    elapsed = time.perf_counter() - start
    return total, elapsed


def benchmark_search(
    client: MilvusClient,
    queries: list[list[float]],
    top_k: int = TOP_K,
) -> tuple[int, float]:
    """
    Run search queries sequentially.
    Returns (num_queries, elapsed_seconds).
    """
    start = time.perf_counter()
    for q in queries:
        client.search(
            collection_name=COLLECTION_NAME,
            data=[q],
            limit=top_k,
        )
    elapsed = time.perf_counter() - start
    return len(queries), elapsed


def main():
    print("=" * 60)
    print("ch15: Performance Benchmark Demo")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Vectors: {NUM_VECTORS}")
    print(f"  Dimension: {DIM}")
    print(f"  Batch sizes: {BATCH_SIZES}")
    print(f"  Search queries: {SEARCH_N}")
    print(f"  Top-K: {TOP_K}")

    # Connect
    print("\n[0] Connecting to Milvus...")
    client = MilvusClient(uri=MILVUS_URI)
    ensure_collection(client)

    # Generate test data
    print("\n[1] Generating test data...")
    all_vectors = generate_vectors(NUM_VECTORS, DIM)
    print(f"  Generated {len(all_vectors)} vectors of dim {DIM}")

    # ── Insert Benchmark ──────────────────────────────────────────────────────
    print("\n[2] Insert Throughput Benchmark")
    print("-" * 40)

    results = []
    for bs in BATCH_SIZES:
        # Recreate collection for each test to have clean state
        ensure_collection(client)
        total, elapsed = benchmark_insert(client, all_vectors, bs)
        throughput = total / elapsed if elapsed > 0 else 0
        results.append((bs, total, elapsed, throughput))
        print(f"  Batch size {bs:>4}: {total} vectors in {elapsed:.2f}s = {throughput:.0f} vectors/s")

    # Best batch size
    best = max(results, key=lambda r: r[3])
    print(f"\n  Best batch size: {best[0]} ({best[3]:.0f} vectors/s)")

    # ── Search Benchmark ──────────────────────────────────────────────────────
    print("\n[3] Search Latency & QPS Benchmark")
    print("-" * 40)

    # Ensure data exists with best batch size
    ensure_collection(client)
    benchmark_insert(client, all_vectors, best[0])

    # Load index (Milvus loads index automatically for search)
    client.load_collection(COLLECTION_NAME)

    # Generate query vectors
    search_queries = generate_vectors(SEARCH_N, DIM)

    n_queries, elapsed = benchmark_search(client, search_queries)
    qps = n_queries / elapsed if elapsed > 0 else 0
    avg_latency = (elapsed / n_queries) * 1000  # ms

    print(f"  Queries: {n_queries}")
    print(f"  Total time: {elapsed:.2f}s")
    print(f"  QPS: {qps:.1f}")
    print(f"  Avg latency: {avg_latency:.2f} ms")

    # ── Latency Distribution ──────────────────────────────────────────────────
    print("\n[4] Latency Distribution (individual queries)")
    print("-" * 40)

    latencies = []
    for q in search_queries[:20]:  # sample 20 queries
        t0 = time.perf_counter()
        client.search(collection_name=COLLECTION_NAME, data=[q], limit=TOP_K)
        latencies.append((time.perf_counter() - t0) * 1000)

    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]
    p99 = latencies[int(len(latencies) * 0.99)]
    print(f"  Samples: {len(latencies)}")
    print(f"  P50 latency: {p50:.2f} ms")
    print(f"  P95 latency: {p95:.2f} ms")
    print(f"  P99 latency: {p99:.2f} ms")

    # ── Final Report ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Benchmark Report")
    print("=" * 60)
    print(f"  Collection: {COLLECTION_NAME}")
    print(f"  Dimension:  {DIM}")
    print(f"  Index:      IVF_FLAT (nlist=128)")
    print(f"  Metric:     L2")
    print()
    print(f"  Insert Results:")
    for bs, total, elapsed, throughput in results:
        print(f"    Batch {bs:>4}: {throughput:>8.0f} vectors/s")
    print(f"  Best batch: {best[0]} ({best[3]:.0f} vectors/s)")
    print()
    print(f"  Search Results:")
    print(f"    QPS:       {qps:>8.1f}")
    print(f"    Avg Lat:   {avg_latency:>8.2f} ms")
    print(f"    P50:       {p50:>8.2f} ms")
    print(f"    P95:       {p95:>8.2f} ms")
    print(f"    P99:       {p99:>8.2f} ms")
    print("=" * 60)


if __name__ == "__main__":
    main()
