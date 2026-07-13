"""
ch20-vertical-rag: Medical Domain RAG Demo
Medical literature collection with structured schema (disease, drug, category).
Search by disease name filter + vector similarity, drug name filter + vector similarity.
Prints treatment recommendations based on retrieved documents.
Uses MilvusClient API with sample Chinese medical data.
"""

import hashlib

import numpy as np
from pymilvus import MilvusClient

# ── Configuration ──────────────────────────────────────────────────────────────
MILVUS_URI = "http://localhost:19530"
COLLECTION_NAME = "medical_kb"
DIM = 64

# ── Sample Chinese Medical Literature ──────────────────────────────────────────

MEDICAL_DOCS = [
    {
        "title": "非小细胞肺癌靶向治疗指南",
        "content": "非小细胞肺癌（NSCLC）是肺癌最常见的类型。EGFR突变患者推荐使用奥希替尼作为一线治疗。ALK融合阳性患者推荐使用阿来替尼。ROS1重排患者可使用克唑替尼。",
        "disease": "非小细胞肺癌",
        "drug": "奥希替尼",
        "category": "治疗指南",
    },
    {
        "title": "奥希替尼临床应用专家共识",
        "content": "奥希替尼是第三代EGFR-TKI，用于EGFR T790M突变阳性的晚期NSCLC患者。常见不良反应包括皮疹、腹泻和甲沟炎。肝功能异常需定期监测。",
        "disease": "非小细胞肺癌",
        "drug": "奥希替尼",
        "category": "专家共识",
    },
    {
        "title": "肺癌免疫治疗新进展",
        "content": "PD-1/PD-L1抑制剂在肺癌治疗中取得突破。帕博利珠单抗联合化疗已成为晚期NSCLC一线标准。度伐利尤单抗用于III期不可切除NSCLC的维持治疗。",
        "disease": "非小细胞肺癌",
        "drug": "帕博利珠单抗",
        "category": "研究进展",
    },
    {
        "title": "乳腺癌内分泌治疗规范",
        "content": "激素受体阳性乳腺癌患者推荐内分泌治疗。他莫昔芬用于绝经前患者。芳香化酶抑制剂用于绝经后患者。CDK4/6抑制剂联合内分泌治疗改善生存。",
        "disease": "乳腺癌",
        "drug": "他莫昔芬",
        "category": "治疗规范",
    },
    {
        "title": "HER2阳性乳腺癌靶向治疗",
        "content": "曲妥珠单抗是HER2阳性乳腺癌的基础靶向药物。帕妥珠单抗联合曲妥珠单抗双靶方案效果更优。T-DM1用于HER2阳性晚期乳腺癌的二线治疗。",
        "disease": "乳腺癌",
        "drug": "曲妥珠单抗",
        "category": "靶向治疗",
    },
    {
        "title": "胃癌综合治疗策略",
        "content": "胃癌治疗包括手术、化疗和靶向治疗。HER2阳性胃癌推荐曲妥珠单抗联合化疗。PD-1抑制剂在晚期胃癌中显示疗效。雷莫西尤单抗用于二线治疗。",
        "disease": "胃癌",
        "drug": "雷莫西尤单抗",
        "category": "综合治疗",
    },
    {
        "title": "结直肠癌诊疗指南",
        "content": "结直肠癌早期以手术为主。晚期患者使用FOLFOX或FOLFIRI化疗方案。西妥昔单抗用于RAS野生型患者。贝伐珠单抗可用于一线治疗。",
        "disease": "结直肠癌",
        "drug": "西妥昔单抗",
        "category": "诊疗指南",
    },
    {
        "title": "肝细胞癌系统治疗进展",
        "content": "索拉非尼是晚期肝癌的一线靶向药物。仑伐替尼非劣效于索拉非尼。阿替利珠单抗联合贝伐珠单抗免疫联合方案显示优越疗效。",
        "disease": "肝细胞癌",
        "drug": "索拉非尼",
        "category": "系统治疗",
    },
    {
        "title": "慢性髓系白血病治疗",
        "content": "伊马替尼是CML一线标准治疗。尼洛替尼和达沙替尼用于二线治疗。治疗目标包括血液学缓解、细胞遗传学缓解和分子学缓解。",
        "disease": "慢性髓系白血病",
        "drug": "伊马替尼",
        "category": "治疗标准",
    },
    {
        "title": "类风湿关节炎生物制剂治疗",
        "content": "甲氨蝶呤是RA的基础用药。TNF-α抑制剂如阿达木单抗用于中重度RA。托珠单抗用于TNF-α抑制剂无效患者。JAK抑制剂是新型口服靶向药物。",
        "disease": "类风湿关节炎",
        "drug": "阿达木单抗",
        "category": "生物制剂",
    },
    {
        "title": "糖尿病治疗药物选择",
        "content": "二甲双胍是2型糖尿病的一线用药。SGLT2抑制剂有心血管获益。GLP-1受体激动剂用于肥胖患者。胰岛素用于口服药控制不佳的患者。",
        "disease": "糖尿病",
        "drug": "二甲双胍",
        "category": "药物治疗",
    },
    {
        "title": "高血压合理用药指南",
        "content": "钙通道阻滞剂和ACEI/ARB是常用降压药物。联合用药提高降压效果。根据患者合并症选择个体化方案。氨氯地平是常用CCB类药物。",
        "disease": "高血压",
        "drug": "氨氯地平",
        "category": "合理用药",
    },
]


def generate_embedding(text: str, dim: int = DIM) -> list[float]:
    """Deterministic pseudo-embedding."""
    h = hashlib.md5(text.encode()).hexdigest()
    return [((int(h[i : i + 2], 16) / 255.0) * 2 - 1) for i in range(0, dim * 2, 2)]


# ── Collection Setup ────────────────────────────────────────────────────────────


def ensure_collection(client: MilvusClient):
    """Create collection with structured medical schema."""
    if client.has_collection(COLLECTION_NAME):
        client.drop_collection(COLLECTION_NAME)

    schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field("id", datatype="INT64", is_primary=True, auto_id=True)
    schema.add_field("vector", datatype="FLOAT_VECTOR", dim=DIM)
    schema.add_field("title", datatype="VARCHAR", max_length=256)
    schema.add_field("content", datatype="VARCHAR", max_length=1024)
    schema.add_field("disease", datatype="VARCHAR", max_length=128)
    schema.add_field("drug", datatype="VARCHAR", max_length=128)
    schema.add_field("category", datatype="VARCHAR", max_length=64)

    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(field_name="vector", index_type="FLAT", metric_type="L2")

    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
    )
    print(f"Created collection '{COLLECTION_NAME}'")


def insert_medical_docs(client: MilvusClient):
    """Insert all medical documents."""
    data = []
    for doc in MEDICAL_DOCS:
        text_for_vec = f"{doc['title']} {doc['content']} {doc['disease']} {doc['drug']}"
        data.append({
            "vector": generate_embedding(text_for_vec),
            "title": doc["title"],
            "content": doc["content"],
            "disease": doc["disease"],
            "drug": doc["drug"],
            "category": doc["category"],
        })

    ids = client.insert(collection_name=COLLECTION_NAME, data=data)
    print(f"Inserted {len(ids)} medical documents")
    return ids


# ── Search Functions ────────────────────────────────────────────────────────────


def search_by_disease(
    client: MilvusClient,
    disease: str,
    query: str = "",
    top_k: int = 5,
) -> list[dict]:
    """Search by disease name filter + optional vector similarity."""
    q_vec = generate_embedding(query) if query else generate_embedding(disease)
    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[q_vec],
        limit=top_k,
        filter=f'disease == "{disease}"',
        output_fields=["title", "content", "disease", "drug", "category"],
    )
    return results[0] if results else []


def search_by_drug(
    client: MilvusClient,
    drug: str,
    query: str = "",
    top_k: int = 5,
) -> list[dict]:
    """Search by drug name filter + optional vector similarity."""
    q_vec = generate_embedding(query) if query else generate_embedding(drug)
    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[q_vec],
        limit=top_k,
        filter=f'drug == "{drug}"',
        output_fields=["title", "content", "disease", "drug", "category"],
    )
    return results[0] if results else []


def search_by_category(
    client: MilvusClient,
    category: str,
    top_k: int = 10,
) -> list[dict]:
    """Search by category (e.g., 治疗指南, 专家共识)."""
    results = client.query(
        collection_name=COLLECTION_NAME,
        filter=f'category == "{category}"',
        output_fields=["title", "disease", "drug", "category"],
        limit=top_k,
    )
    return results


def search_open(
    client: MilvusClient,
    query: str,
    top_k: int = 5,
) -> list[dict]:
    """Open search without filters."""
    q_vec = generate_embedding(query)
    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[q_vec],
        limit=top_k,
        output_fields=["title", "content", "disease", "drug", "category"],
    )
    return results[0] if results else []


# ── Treatment Recommendation ────────────────────────────────────────────────────


def print_treatment_recommendation(disease: str, results: list[dict]):
    """Format and print treatment recommendations based on retrieved docs."""
    print(f"\n  Treatment Recommendations for '{disease}':")
    print(f"  {'-' * 50}")

    drugs = set()
    categories = set()

    for r in results:
        entity = r["entity"]
        drugs.add(entity["drug"])
        categories.add(entity["category"])
        print(f"    [{entity['category']}] {entity['title']}")
        print(f"      Drug: {entity['drug']}")
        print(f"      Summary: {entity['content'][:100]}...")
        print()

    print(f"  Related drugs: {', '.join(sorted(drugs))}")
    print(f"  Document categories: {', '.join(sorted(categories))}")


# ── Main Demo ───────────────────────────────────────────────────────────────────


def main():
    print("=" * 60)
    print("ch20: Medical Domain RAG Demo")
    print("=" * 60)

    print("\n[0] Connecting to Milvus...")
    client = MilvusClient(uri=MILVUS_URI)
    ensure_collection(client)
    insert_medical_docs(client)

    # 1. Search by disease name
    print("\n[1] Search by Disease: 非小细胞肺癌")
    print("-" * 40)
    results = search_by_disease(client, "非小细胞肺癌", "靶向治疗")
    print(f"  Found {len(results)} results")
    for r in results:
        e = r["entity"]
        print(f"    {e['title']} (drug: {e['drug']}, dist: {r['distance']:.4f})")

    # 2. Search by drug name
    print("\n[2] Search by Drug: 曲妥珠单抗")
    print("-" * 40)
    results = search_by_drug(client, "曲妥珠单抗")
    for r in results:
        e = r["entity"]
        print(f"    {e['title']} (disease: {e['disease']}, dist: {r['distance']:.4f})")

    # 3. Open search (no filters)
    print("\n[3] Open Search: '糖尿病治疗'")
    print("-" * 40)
    results = search_open(client, "糖尿病治疗")
    for r in results:
        e = r["entity"]
        print(f"    {e['title']} (disease: {e['disease']}, drug: {e['drug']}, dist: {r['distance']:.4f})")

    # 4. Treatment recommendation (non-small cell lung cancer)
    print("\n[4] Treatment Recommendation: 非小细胞肺癌")
    print("-" * 40)
    results = search_by_disease(client, "非小细胞肺癌", "治疗")
    print_treatment_recommendation("非小细胞肺癌", results)

    # 5. Treatment recommendation (breast cancer)
    print("\n[5] Treatment Recommendation: 乳腺癌")
    print("-" * 40)
    results = search_by_disease(client, "乳腺癌", "治疗")
    print_treatment_recommendation("乳腺癌", results)

    # 6. Browse by category
    print("\n[6] Browse by Category: 治疗指南")
    print("-" * 40)
    results = search_by_category(client, "治疗指南")
    for r in results:
        print(f"    {r['title']} (disease: {r['disease']}, drug: {r['drug']})")

    print("\n" + "=" * 60)
    print("Medical RAG Demo Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
