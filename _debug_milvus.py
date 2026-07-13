"""Debug Milvus search results."""
from pymilvus import connections, Collection, utility

connections.connect(host="localhost", port=19530)

# Test search on text_unit_text
collection = Collection("text_unit_text")
collection.load()

import random
query = [random.random() for _ in range(1024)]
results = collection.search(
    data=[query],
    anns_field="vector",
    param={"metric_type": "IP", "params": {"nprobe": 10}},
    limit=3,
    output_fields=["id", "text"],
)
for hit in results[0]:
    print(f"Score: {hit.score}")
    print(f"ID: {hit.id}")
    entity = hit.entity
    print(f"Entity type: {type(entity)}")
    if hasattr(entity, "get"):
        text = entity.get("text")
        print(f"Text: {str(text)[:100] if text else 'NONE'}")
    # Also check fields
    print("---")
