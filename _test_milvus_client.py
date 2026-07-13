"""Test MilvusClient search and insert patterns."""
from pymilvus import MilvusClient, DataType

client = MilvusClient(host="localhost", port=19530)

# Test insert with row-oriented data
schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
schema.add_field("id", datatype=DataType.VARCHAR, max_length=1024, is_primary=True)
schema.add_field("vector", datatype=DataType.FLOAT_VECTOR, dim=1024)

index_params = MilvusClient.prepare_index_params()
index_params.add_index(field_name="vector", index_type="IVF_FLAT", metric_type="IP", params={"nlist": 128})

# Create temp collection
client.create_collection("_test_mc", schema=schema, index_params=index_params)
print("Collection created")

# Insert row-oriented
data = [
    {"id": "test1", "vector": [0.1]*1024, "text": "hello world", "json_data": "{}", "create_date": "", "update_date": ""},
    {"id": "test2", "vector": [0.2]*1024, "text": "foo bar", "json_data": "{}", "create_date": "", "update_date": ""},
]
res = client.insert("_test_mc", data)
print("Insert result:", res)

# Flush
client.flush("_test_mc")
client.load_collection("_test_mc")

# Search
import random
query = [random.random() for _ in range(1024)]
results = client.search(
    collection_name="_test_mc",
    data=[query],
    anns_field="vector",
    search_params={"metric_type": "IP", "params": {"nprobe": 10}},
    limit=2,
    output_fields=["id", "text"],
)
print("Search result count:", len(results[0]))
for hit in results[0]:
    print(f"  ID: {hit['id']}, Score: {hit['distance']:.4f}, Text: {hit.get('entity',{}).get('text','')[:50]}")

# Query by ID
q = client.get("_test_mc", ids=["test1"])
print("Get by id:", q)

# Count
cnt = client.query("_test_mc", output_fields=["count(*)"])
print("Count:", cnt)

# Drop
client.drop_collection("_test_mc")
client.close()
print("Done")
