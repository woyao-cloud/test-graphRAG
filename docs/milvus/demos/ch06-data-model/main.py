"""
ch06-data-model: Milvus 数据模型设计演示
=========================================
演示集合、分区、字段设计，以及不同场景下的 Schema 设计模式。
需要连接 Milvus 服务（MILVUS_HOST 环境变量，默认 localhost）。
"""

import os
import time

try:
    from pymilvus import DataType, MilvusClient
except ImportError:
    print("需要安装 pymilvus: pip install pymilvus")
    raise

MILVUS_HOST = os.environ.get("MILVUS_HOST", "localhost")
MILVUS_PORT = int(os.environ.get("MILVUS_PORT", "19530"))


def connect() -> MilvusClient:
    """连接 Milvus 服务。"""
    print(f"  连接 Milvus 服务: {MILVUS_HOST}:{MILVUS_PORT}")
    client = MilvusClient(host=MILVUS_HOST, port=MILVUS_PORT)
    print("  ✅ 连接成功！")
    return client


# ======================================================================
# 1. 基础集合设计
# ======================================================================

def demo_simple_collection(client: MilvusClient):
    """演示最简单的集合设计：id + vector。"""
    print("=" * 60)
    print("【1. 基础集合设计：id + vector】")
    print("=" * 60)

    collection_name = "demo_simple"

    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("id", datatype=DataType.VARCHAR, max_length=64, is_primary=True)
    schema.add_field("vector", datatype=DataType.FLOAT_VECTOR, dim=128)

    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(field_name="vector", index_type="IVF_FLAT", metric_type="IP", params={"nlist": 128})

    if client.has_collection(collection_name):
        client.drop_collection(collection_name)
    client.create_collection(collection_name, schema=schema, index_params=index_params)

    print(f"  创建集合: {collection_name}")
    desc = client.describe_collection(collection_name)
    print(f"  字段数: {len(desc['schema']['fields'])}")
    for f in desc["schema"]["fields"]:
        print(f"    字段: {f['name']:10} 类型: {str(f['type']):20}")
    print()

    client.drop_collection(collection_name)
    print("  已清理\n")


# ======================================================================
# 2. RAG 集合设计
# ======================================================================

def demo_rag_collection(client: MilvusClient):
    """演示 RAG 场景的集合设计：id + vector + 文本 + 元数据。"""
    print("=" * 60)
    print("【2. RAG 集合设计：完整字段结构】")
    print("=" * 60)

    collection_name = "demo_rag_schema"

    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
    schema.add_field("doc_id", datatype=DataType.VARCHAR, max_length=64, is_primary=True)
    schema.add_field("vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
    schema.add_field("title", datatype=DataType.VARCHAR, max_length=256)
    schema.add_field("content", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field("author", datatype=DataType.VARCHAR, max_length=128)
    schema.add_field("department", datatype=DataType.VARCHAR, max_length=64)
    schema.add_field("category", datatype=DataType.VARCHAR, max_length=64)
    schema.add_field("publish_date", datatype=DataType.VARCHAR, max_length=32)
    schema.add_field("file_type", datatype=DataType.VARCHAR, max_length=16)
    schema.add_field("page_count", datatype=DataType.INT64)
    schema.add_field("version", datatype=DataType.INT64)

    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(field_name="vector", index_type="HNSW", metric_type="IP",
                           params={"M": 16, "efConstruction": 200})

    if client.has_collection(collection_name):
        client.drop_collection(collection_name)
    client.create_collection(collection_name, schema=schema, index_params=index_params)

    print(f"  创建集合: {collection_name}")
    desc = client.describe_collection(collection_name)
    print(f"  字段数: {len(desc['schema']['fields'])}")
    print()
    print("  字段设计说明：")
    print("  - doc_id:    文档唯一标识（主键）")
    print("  - vector:    文档嵌入向量（1024 维，用于语义检索）")
    print("  - title:     文档标题（用于展示和关键词过滤）")
    print("  - content:   文档原文（用于 LLM 上下文拼接）")
    print("  - author:    文档作者（用于权限过滤）")
    print("  - department:所属部门（用于分区过滤）")
    print("  - category:  文档分类（用于标签过滤）")
    print("  - publish_date: 发布日期（用于时间范围过滤）")
    print("  - file_type: 文件类型（pdf/docx/txt）")
    print("  - page_count: 页数（用于排序筛选）")
    print("  - version:   版本号（用于版本管理）")
    print()

    client.drop_collection(collection_name)
    print("  已清理\n")


# ======================================================================
# 3. 分区设计演示
# ======================================================================

def demo_partition_design(client: MilvusClient):
    """演示分区策略：按部门分区。"""
    print("=" * 60)
    print("【3. 分区设计：按部门分区】")
    print("=" * 60)

    collection_name = "demo_partitions"
    partitions = ["HR", "Engineering", "Finance", "Marketing"]

    # 创建集合
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
    schema.add_field("id", datatype=DataType.VARCHAR, max_length=64, is_primary=True)
    schema.add_field("vector", datatype=DataType.FLOAT_VECTOR, dim=128)
    schema.add_field("content", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field("department", datatype=DataType.VARCHAR, max_length=32)

    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(field_name="vector", index_type="FLAT", metric_type="IP")

    if client.has_collection(collection_name):
        client.drop_collection(collection_name)
    client.create_collection(collection_name, schema=schema, index_params=index_params)

    # 创建分区
    print(f"  创建集合: {collection_name}")
    print(f"  创建分区:")
    for p in partitions:
        client.create_partition(collection_name, partition_name=p)
        print(f"    - {p}")

    # 向不同分区插入数据
    import random
    for dept in partitions:
        records = [
            {
                "id": f"{dept.lower()}_{i}",
                "vector": [random.random() for _ in range(128)],
                "content": f"这是{dept}部门的第{i}份文档",
                "department": dept,
            }
            for i in range(5)
        ]
        client.insert(collection_name, records, partition_name=dept)
        print(f"  向 {dept} 分区插入 5 条记录")

    client.flush(collection_name)

    # 演示分区检索
    print()
    print("  分区检索演示：")
    query = [random.random() for _ in range(128)]

    # 全集合搜索
    results_all = client.search(
        collection_name=collection_name, data=[query], anns_field="vector",
        search_params={"metric_type": "IP", "params": {"nprobe": 10}},
        limit=3, output_fields=["id", "content", "department"],
    )
    print(f"  全集合搜索: 返回 {len(results_all[0])} 条结果")
    for hit in results_all[0]:
        print(f"    id={hit['id']} dept={hit['entity']['department']}")

    # 仅 HR 分区搜索
    results_hr = client.search(
        collection_name=collection_name, data=[query], anns_field="vector",
        search_params={"metric_type": "IP", "params": {"nprobe": 10}},
        limit=3, output_fields=["id", "content", "department"],
        partition_names=["HR"],
    )
    print(f"  HR分区搜索: 返回 {len(results_hr[0])} 条结果（全部来自 HR）")
    for hit in results_hr[0]:
        print(f"    id={hit['id']} dept={hit['entity']['department']}")

    # 清理
    client.drop_collection(collection_name)
    print("\n  已清理\n")


# ======================================================================
# 4. 动态字段演示
# ======================================================================

def demo_dynamic_fields(client: MilvusClient):
    """演示动态字段功能（enable_dynamic_field=True）。"""
    print("=" * 60)
    print("【4. 动态字段演示】")
    print("=" * 60)

    collection_name = "demo_dynamic"

    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
    schema.add_field("id", datatype=DataType.VARCHAR, max_length=64, is_primary=True)
    schema.add_field("vector", datatype=DataType.FLOAT_VECTOR, dim=128)

    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(field_name="vector", index_type="FLAT", metric_type="IP")

    if client.has_collection(collection_name):
        client.drop_collection(collection_name)
    client.create_collection(collection_name, schema=schema, index_params=index_params)

    # 插入带不同字段的记录
    import random
    records = [
        {"id": "doc_1", "vector": [random.random() for _ in range(128)],
         "title": "文档一", "author": "张三", "tags": ["AI", "ML"]},
        {"id": "doc_2", "vector": [random.random() for _ in range(128)],
         "title": "文档二", "category": "技术", "priority": "high"},
        {"id": "doc_3", "vector": [random.random() for _ in range(128)],
         "title": "文档三", "author": "李四", "language": "en", "pages": 42},
    ]
    client.insert(collection_name, records)
    print(f"  插入 3 条记录，每条记录有不同的动态字段")
    for r in records:
        dynamic_keys = [k for k in r if k not in ("id", "vector")]
        print(f"    {r['id']}: 动态字段 = {dynamic_keys}")

    # 动态字段也可以用于过滤
    query = [random.random() for _ in range(128)]
    results = client.search(
        collection_name=collection_name, data=[query], anns_field="vector",
        search_params={"metric_type": "IP", "params": {"nprobe": 10}},
        limit=5, output_fields=["id", "title", "author", "category"],
        filter='author == "张三"',
    )
    print(f"\n  动态字段过滤检索 (author=张三): {len(results[0])} 条")

    client.drop_collection(collection_name)
    print("\n  已清理\n")


# ======================================================================
# Main
# ======================================================================

def main():
    print("=" * 60)
    print("  Milvus 实战指南 — Ch6: Milvus 核心数据模型")
    print("  演示集合、分区、字段设计与动态字段功能")
    print("=" * 60)
    print()

    try:
        client = connect()
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        print("请确保 Milvus 服务已启动: docker compose up -d milvus")
        return

    demo_simple_collection(client)
    demo_rag_collection(client)
    demo_partition_design(client)
    demo_dynamic_fields(client)

    client.close()

    print("=" * 60)
    print("  演示完成！关键概念：")
    print("  - 集合 (Collection): 等同于关系数据库中的表")
    print("  - 分区 (Partition): 集合内的数据子集，加速过滤")
    print("  - 字段 (Field): 向量字段 + 标量字段")
    print("  - 主键 (Primary Key): 唯一标识每条记录")
    print("  - 动态字段: 无需预定义 Schema，灵活扩展")
    print("  - RAG 场景推荐字段: doc_id, vector, title, content, tags, date")
    print("=" * 60)


if __name__ == "__main__":
    main()
