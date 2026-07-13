"""
ch07-hybrid-search: Hybrid search strategies with MilvusClient API.

Demonstrates 4 search strategies on a product dataset:
  1. Pure vector search
  2. Vector search + category filter (expr filtering)
  3. Vector search + range filter (price between X and Y)
  4. Multi-field filter (category + price + stock)
"""

import os
import random
import time

from pymilvus import MilvusClient, DataType

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
COLLECTION_NAME = "ch07_hybrid_demo"
DIM = 128


def print_step(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def make_vector(dim: int = DIM) -> list[float]:
    return [random.random() for _ in range(dim)]


def build_products() -> list[dict]:
    """Return 50 sample Chinese pharmaceutical products."""
    products = [
        # (id, text, category, price, stock)
        (1,  "阿司匹林肠溶片 100mg*30片", "解热镇痛", 12.50, 500),
        (2,  "布洛芬缓释胶囊 300mg*20粒", "解热镇痛", 18.00, 300),
        (3,  "对乙酰氨基酚片 500mg*10片", "解热镇痛", 8.50, 800),
        (4,  "双氯芬酸钠缓释片 75mg*10片", "解热镇痛", 22.00, 200),
        (5,  "吲哚美辛栓 100mg*6枚", "解热镇痛", 15.00, 150),
        (6,  "头孢克肟胶囊 100mg*6粒", "抗生素", 35.00, 400),
        (7,  "阿莫西林胶囊 250mg*24粒", "抗生素", 16.00, 600),
        (8,  "红霉素肠溶片 125mg*24片", "抗生素", 12.00, 350),
        (9,  "罗红霉素分散片 150mg*6片", "抗生素", 28.00, 250),
        (10, "左氧氟沙星片 200mg*10片", "抗生素", 42.00, 180),
        (11, "奥美拉唑肠溶胶囊 20mg*14粒", "消化系统", 25.00, 450),
        (12, "雷贝拉唑钠肠溶片 10mg*7片", "消化系统", 38.00, 300),
        (13, "多潘立酮片 10mg*30片", "消化系统", 15.00, 500),
        (14, "铝碳酸镁咀嚼片 500mg*20片", "消化系统", 18.50, 400),
        (15, "蒙脱石散 3g*10袋", "消化系统", 9.00, 700),
        (16, "二甲双胍缓释片 500mg*30片", "降糖药", 32.00, 350),
        (17, "格列美脲片 2mg*15片", "降糖药", 45.00, 200),
        (18, "阿卡波糖片 50mg*30片", "降糖药", 55.00, 180),
        (19, "胰岛素注射液 300U/3ml", "降糖药", 68.00, 100),
        (20, "西格列汀片 100mg*7片", "降糖药", 88.00, 120),
        (21, "硝苯地平控释片 30mg*7片", "降压药", 42.00, 300),
        (22, "卡托普利片 25mg*100片", "降压药", 15.00, 500),
        (23, "氯沙坦钾片 50mg*7片", "降压药", 48.00, 250),
        (24, "氨氯地平片 5mg*7片", "降压药", 35.00, 400),
        (25, "美托洛尔缓释片 47.5mg*7片", "降压药", 30.00, 350),
        (26, "阿托伐他汀钙片 20mg*7片", "降脂药", 55.00, 300),
        (27, "辛伐他汀片 20mg*7片", "降脂药", 38.00, 280),
        (28, "瑞舒伐他汀钙片 10mg*7片", "降脂药", 65.00, 220),
        (29, "非诺贝特胶囊 200mg*10粒", "降脂药", 42.00, 150),
        (30, "依折麦布片 10mg*10片", "降脂药", 72.00, 100),
        (31, "地高辛片 0.25mg*30片", "强心药", 25.00, 200),
        (32, "硝酸甘油片 0.5mg*25片", "强心药", 18.00, 400),
        (33, "华法林钠片 2.5mg*60片", "抗凝血", 30.00, 250),
        (34, "氯吡格雷片 75mg*7片", "抗凝血", 95.00, 180),
        (35, "达比加群酯胶囊 110mg*10粒", "抗凝血", 120.00, 80),
        (36, "环磷酰胺片 50mg*50片", "抗肿瘤", 150.00, 60),
        (37, "甲氨蝶呤片 2.5mg*100片", "抗肿瘤", 85.00, 90),
        (38, "氟尿嘧啶注射液 250mg/10ml", "抗肿瘤", 45.00, 120),
        (39, "紫杉醇注射液 30mg/5ml", "抗肿瘤", 320.00, 40),
        (40, "卡培他滨片 500mg*60片", "抗肿瘤", 280.00, 50),
        (41, "泼尼松片 5mg*100片", "激素类", 12.00, 600),
        (42, "地塞米松片 0.75mg*100片", "激素类", 10.00, 700),
        (43, "甲泼尼龙片 4mg*30片", "激素类", 35.00, 300),
        (44, "维生素C片 100mg*100片", "维生素", 5.00, 1000),
        (45, "维生素D滴剂 400U*30粒", "维生素", 28.00, 500),
        (46, "钙尔奇D片 600mg*60片", "维生素", 65.00, 400),
        (47, "氯雷他定片 10mg*6片", "抗过敏", 18.00, 500),
        (48, "西替利嗪滴剂 10mg/ml*20ml", "抗过敏", 32.00, 300),
        (49, "孟鲁司特钠片 10mg*5片", "抗过敏", 38.00, 250),
        (50, "酮替芬片 1mg*60片", "抗过敏", 15.00, 350),
    ]

    batch = []
    for pid, text, cat, price, stock in products:
        batch.append({
            "id": pid,
            "vector": make_vector(),
            "text": text,
            "category": cat,
            "price": price,
            "stock": stock,
        })
    return batch


def setup_collection(client: MilvusClient) -> None:
    client.drop_collection(COLLECTION_NAME)

    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=DIM)
    schema.add_field("text", DataType.VARCHAR, max_length=256)
    schema.add_field("category", DataType.VARCHAR, max_length=32)
    schema.add_field("price", DataType.FLOAT)
    schema.add_field("stock", DataType.INT64)

    client.create_collection(collection_name=COLLECTION_NAME, schema=schema)
    print(f"Collection '{COLLECTION_NAME}' created.")

    # Insert data
    products = build_products()
    client.insert(COLLECTION_NAME, products)
    print(f"Inserted {len(products)} products.")

    # Create IVF_FLAT index
    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="IVF_FLAT",
        metric_type="IP",
        params={"nlist": 128},
    )
    client.create_index(COLLECTION_NAME, index_params)
    client.load_collection(COLLECTION_NAME)
    print("Index created and collection loaded.\n")


def main() -> None:
    client = MilvusClient(uri=f"http://{MILVUS_HOST}:19530")
    print(f"Connected to {MILVUS_HOST}:19530\n")

    setup_collection(client)

    # Use a random query vector for all strategies
    query_vec = make_vector()
    top_k = 5

    # ------------------------------------------------------------------ #
    # Strategy 1: Pure vector search
    # ------------------------------------------------------------------ #
    print_step("Strategy 1: Pure vector search")
    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vec],
        anns_field="vector",
        limit=top_k,
        output_fields=["id", "text", "category", "price", "stock"],
    )
    print(f"  Top {top_k} results (no filter):")
    for i, hit in enumerate(results[0], start=1):
        e = hit["entity"]
        print(f"    {i}. [{e['category']}] {e['text']}  "
              f"price={e['price']}  stock={e['stock']}  score={hit['distance']:.4f}")

    # ------------------------------------------------------------------ #
    # Strategy 2: Vector search + category filter
    # ------------------------------------------------------------------ #
    print_step('Strategy 2: Vector search + category filter (category == "抗肿瘤")')
    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vec],
        anns_field="vector",
        limit=top_k,
        output_fields=["id", "text", "category", "price", "stock"],
        filter='category == "抗肿瘤"',
    )
    print(f"  Top {top_k} results (filter: 抗肿瘤):")
    for i, hit in enumerate(results[0], start=1):
        e = hit["entity"]
        print(f"    {i}. [{e['category']}] {e['text']}  "
              f"price={e['price']}  stock={e['stock']}  score={hit['distance']:.4f}")

    # ------------------------------------------------------------------ #
    # Strategy 3: Vector search + range filter (price between 20 and 50)
    # ------------------------------------------------------------------ #
    print_step("Strategy 3: Vector search + range filter (20 <= price <= 50)")
    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vec],
        anns_field="vector",
        limit=top_k,
        output_fields=["id", "text", "category", "price", "stock"],
        filter="price >= 20 and price <= 50",
    )
    print(f"  Top {top_k} results (filter: 20 <= price <= 50):")
    for i, hit in enumerate(results[0], start=1):
        e = hit["entity"]
        print(f"    {i}. [{e['category']}] {e['text']}  "
              f"price={e['price']}  stock={e['stock']}  score={hit['distance']:.4f}")

    # ------------------------------------------------------------------ #
    # Strategy 4: Multi-field filter (category + price + stock)
    # ------------------------------------------------------------------ #
    print_step("Strategy 4: Multi-field filter (降压药 + price <= 40 + stock >= 200)")
    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vec],
        anns_field="vector",
        limit=top_k,
        output_fields=["id", "text", "category", "price", "stock"],
        filter='category == "降压药" and price <= 40 and stock >= 200',
    )
    print(f"  Top {top_k} results (filter: 降压药, price<=40, stock>=200):")
    for i, hit in enumerate(results[0], start=1):
        e = hit["entity"]
        print(f"    {i}. [{e['category']}] {e['text']}  "
              f"price={e['price']}  stock={e['stock']}  score={hit['distance']:.4f}")

    # ------------------------------------------------------------------ #
    # Cleanup
    # ------------------------------------------------------------------ #
    print()
    print_step("Cleanup")
    client.drop_collection(COLLECTION_NAME)
    print(f"Collection '{COLLECTION_NAME}' dropped.")

    print(f"\n{'=' * 60}")
    print("  Hybrid search demo completed successfully!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        raise
