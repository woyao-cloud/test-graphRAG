# Milvus Docker Setup Guide

## Quick Fix for Current Issue

The `milvusdb/milvus:latest` image has compatibility issues on Windows Docker Desktop. Choose one of these solutions:

### Solution 1: Use Standalone Milvus Image (RECOMMENDED)

Replace the Milvus service in `docker-compose.yml`:

```yaml
milvus:
  image: milvusdb/milvus:v0.4.8-standalone
  container_name: graphrag-kg-milvus
  ports:
    - "19530:19530"
    - "9091:9091"
  environment:
    - TZ=UTC
  volumes:
    - milvus_data:/var/lib/milvus
  healthcheck:
    test: ["CMD-SHELL", "curl -s http://localhost:9091/healthz || exit 1"]
    interval: 10s
    timeout: 5s
    retries: 30
    start_period: 60s
  restart: unless-stopped
```

Then restart:
```bash
docker compose down
docker compose up -d milvus neo4j
```

### Solution 2: Add Embedded etcd

Update the Milvus service environment to include proper etcd configuration:

```yaml
milvus:
  image: milvusdb/milvus:latest
  environment:
    - TZ=UTC
    - ETCD_ENDPOINTS=localhost:2379
    - ETCD_USE_EMBED=true
    - ETCD_DATA_DIR=/var/lib/milvus/etcd
    - MILVUS_COMMON_LOG_LEVEL=info
  # ... rest of config
```

### Solution 3: Use Lite Version

```yaml
milvus:
  image: milvusdb/milvus:latest-lite
  # ... rest of config
```

## Verification After Fix

Once Docker is running, test connectivity:

```bash
# Check services are running
docker compose ps

# Test Milvus connection
python -c "
from pymilvus import MilvusClient
import time
time.sleep(5)  # Wait for startup
client = MilvusClient(uri='http://localhost:19530')
print('Collections:', client.list_collections())
"

# Test Neo4j connection  
python -c "
from neo4j import GraphDatabase
driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'password'))
with driver.session() as session:
    print(session.run('RETURN 1').single()[0])
driver.close()
"
```

## Development Without Docker

For development/testing without Docker, you can use the Python integration directly:

```python
import graphrag_kg
from graphrag_vectors.vector_store_factory import create_vector_store
from graphrag_vectors.vector_store_config import VectorStoreConfig
from graphrag_vectors.index_schema import IndexSchema

# Configure for local/test Milvus
config = VectorStoreConfig(
    type="milvus",
    host="localhost",
    port=19530,
    collection_name="test_collection"
)

schema = IndexSchema()
store = create_vector_store(config, schema)

# Note: Requires a running Milvus service to actually connect
# For unit testing, see tests/test_milvus_integration.py
```

## Run Tests Without Milvus Service

```bash
# Unit tests (no service required)
pytest tests/test_milvus_integration.py -v

# Should show 9/9 PASSED
```

## Production Deployment Notes

For production, consider:
1. Use explicit Milvus version (not `latest`)
2. Configure persistent volumes properly
3. Set resource limits and requests
4. Use separate etcd cluster for HA
5. Configure backup and disaster recovery

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `exit code 134` | Use standalone image or older version |
| `connection refused` | Wait for startup, check firewall |
| `etcd connection failed` | Use embedded etcd config or add etcd service |
| `grpc timeout` | Increase healthcheck timeout, check Docker network |
