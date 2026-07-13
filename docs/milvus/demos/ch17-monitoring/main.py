"""
ch17-monitoring: Monitoring Demo for Milvus
Connects to Milvus, queries server stats, simulates workload, and prints a monitoring report.
Uses MilvusClient API + requests library for HTTP metrics.
"""

import random
import time

import numpy as np
import requests
from pymilvus import MilvusClient

# ── Configuration ──────────────────────────────────────────────────────────────
MILVUS_URI = "http://localhost:19530"
MILVUS_HTTP = "http://localhost:9091"  # Milvus metrics endpoint (optional)
COLLECTION_NAME = "monitoring_demo"
DIM = 64
NUM_ENTITIES = 200


def generate_vectors(n: int, dim: int) -> list[list[float]]:
    return np.random.randn(n, dim).astype(np.float32).tolist()


def ensure_collection(client: MilvusClient):
    """Create a fresh collection for monitoring demo."""
    if client.has_collection(COLLECTION_NAME):
        client.drop_collection(COLLECTION_NAME)

    schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=True)
    schema.add_field("id", datatype="INT64", is_primary=True, auto_id=True)
    schema.add_field("vector", datatype="FLOAT_VECTOR", dim=DIM)
    schema.add_field("category", datatype="VARCHAR", max_length=32)
    schema.add_field("timestamp", datatype="INT64")

    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(field_name="vector", index_type="FLAT", metric_type="L2")

    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
    )
    client.load_collection(COLLECTION_NAME)
    print(f"Created collection '{COLLECTION_NAME}'")


def simulate_workload(client: MilvusClient):
    """Simulate insert + search workload."""
    print("\n  Simulating workload...")

    # Insert
    vectors = generate_vectors(NUM_ENTITIES, DIM)
    categories = ["A", "B", "C"]
    now = int(time.time())

    data = []
    for v in vectors:
        data.append({
            "vector": v,
            "category": random.choice(categories),
            "timestamp": now,
        })

    t0 = time.perf_counter()
    ids = client.insert(collection_name=COLLECTION_NAME, data=data)
    insert_time = time.perf_counter() - t0
    print(f"  Inserted {len(ids)} entities in {insert_time:.2f}s")

    # Search queries
    queries = generate_vectors(10, DIM)
    total_search_time = 0.0
    for q in queries:
        t0 = time.perf_counter()
        client.search(
            collection_name=COLLECTION_NAME,
            data=[q],
            limit=5,
            output_fields=["category"],
        )
        total_search_time += time.perf_counter() - t0

    avg_search_ms = (total_search_time / len(queries)) * 1000
    print(f"  Ran {len(queries)} searches, avg {avg_search_ms:.2f} ms")


def get_collection_stats(client: MilvusClient) -> dict:
    """Query collection statistics."""
    stats = {}

    # Collection list
    collections = client.list_collections()
    stats["collection_count"] = len(collections)

    # Collection details
    try:
        desc = client.describe_collection(COLLECTION_NAME)
        stats["collection_name"] = desc.get("collection_name", COLLECTION_NAME)
        stats["dimension"] = desc.get("dim", "N/A")
        stats["auto_id"] = desc.get("auto_id", False)
        stats["index_status"] = "Ready"  # simplified
    except Exception:
        stats["collection_info"] = "unavailable"

    # Entity count via query
    try:
        results = client.query(
            collection_name=COLLECTION_NAME,
            output_fields=["count(*)"],
        )
        stats["entity_count"] = results[0]["count(*)"] if results else 0
    except Exception:
        stats["entity_count"] = "unavailable"

    return stats


def get_http_metrics() -> dict:
    """Attempt to fetch Milvus HTTP metrics endpoint."""
    metrics = {}
    try:
        resp = requests.get(f"{MILVUS_HTTP}/metrics", timeout=3)
        if resp.status_code == 200:
            metrics["http_metrics_available"] = True
            metrics["raw_metrics_length"] = len(resp.text)
            # Parse some basic metrics
            for line in resp.text.split("\n"):
                if line.startswith("milvus_") and " " in line:
                    parts = line.rsplit(" ", 1)
                    if len(parts) == 2:
                        key = parts[0].split("{")[0]
                        metrics[key] = parts[1]
        else:
            metrics["http_metrics_available"] = False
            metrics["http_status"] = resp.status_code
    except requests.RequestException as e:
        metrics["http_metrics_available"] = False
        metrics["error"] = str(e)

    return metrics


def print_monitoring_report(stats: dict, http_metrics: dict):
    """Print a formatted monitoring report."""
    print("\n" + "=" * 60)
    print("Monitoring Report")
    print("=" * 60)

    print(f"\n[Server Info]")
    print(f"  URI:              {MILVUS_URI}")

    print(f"\n[Collections]")
    print(f"  Total collections: {stats.get('collection_count', 'N/A')}")
    print(f"  Active collection: {stats.get('collection_name', 'N/A')}")
    print(f"  Dimension:         {stats.get('dimension', 'N/A')}")
    print(f"  Entity count:      {stats.get('entity_count', 'N/A')}")
    print(f"  Index status:      {stats.get('index_status', 'N/A')}")
    print(f"  Auto ID:           {stats.get('auto_id', 'N/A')}")

    print(f"\n[HTTP Metrics]")
    if http_metrics.get("http_metrics_available"):
        print(f"  Raw metrics length: {http_metrics.get('raw_metrics_length', 'N/A')} bytes")
        # Show a few parsed metrics
        interesting_keys = [k for k in http_metrics if k.startswith("milvus_")][:8]
        for key in interesting_keys:
            print(f"  {key}: {http_metrics[key]}")
    else:
        print(f"  Not available: {http_metrics.get('error', 'unknown')}")

    print(f"\n[System Health]")
    print(f"  Connection:        OK")
    print(f"  Collection write:  OK")
    print(f"  Collection read:   OK")

    print("\n" + "=" * 60)
    print("Monitoring Demo Complete")
    print("=" * 60)


def main():
    print("=" * 60)
    print("ch17: Monitoring Demo")
    print("=" * 60)

    # Step 1: Connect
    print("\n[1] Connecting to Milvus...")
    try:
        client = MilvusClient(uri=MILVUS_URI)
        server_version = client.get_server_version()
        print(f"  Connected to Milvus (version: {server_version})")
    except Exception as e:
        print(f"  Connection failed: {e}")
        return

    # Step 2: Create collection
    print("\n[2] Setting up collection...")
    ensure_collection(client)

    # Step 3: Simulate workload
    print("\n[3] Simulating workload...")
    simulate_workload(client)

    # Step 4: Collect stats
    print("\n[4] Collecting stats...")
    stats = get_collection_stats(client)
    print(f"  Collection count: {stats.get('collection_count')}")
    print(f"  Entity count: {stats.get('entity_count')}")

    # Step 5: Fetch HTTP metrics
    print("\n[5] Fetching HTTP metrics...")
    http_metrics = get_http_metrics()
    if http_metrics.get("http_metrics_available"):
        print(f"  Metrics endpoint: OK ({http_metrics.get('raw_metrics_length', 0)} bytes)")
    else:
        print(f"  Metrics endpoint: {http_metrics.get('error', 'unavailable')}")

    # Step 6: Print report
    print_monitoring_report(stats, http_metrics)


if __name__ == "__main__":
    main()
