"""
ch04-milvus-architecture: Milvus 架构组件演示
=============================================
纯 Python 标准库实现，模拟 Milvus 各组件之间的交互流程。
可直接运行: python main.py
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Any


# ======================================================================
# 数据模型
# ======================================================================

@dataclass
class Collection:
    """模拟 Milvus 集合。"""
    name: str
    dim: int
    data: list[dict] = field(default_factory=list)


# ======================================================================
# Milvus 架构组件模拟
# ======================================================================

class Proxy:
    """接入层 — 接收客户端请求，路由到协调节点。"""

    def __init__(self):
        self.name = "Proxy"
        self.stats = {"requests_handled": 0}

    def handle_request(self, request: dict) -> dict:
        """接收客户端请求并路由。"""
        self.stats["requests_handled"] += 1
        print(f"  [Proxy] 收到请求: {request['type']}")
        print(f"  [Proxy] 验证请求合法性... 通过")
        print(f"  [Proxy] 路由到协调节点...")
        return {"status": "routed", "request": request}


class RootCoord:
    """根协调节点 — 管理集群元数据、时间戳分配。"""

    def __init__(self):
        self.name = "RootCoord"
        self.ts = 0

    def allocate_timestamp(self) -> int:
        """分配全局时间戳。"""
        self.ts += 1
        return self.ts

    def describe_collection(self, name: str) -> dict:
        """获取集合元数据。"""
        print(f"  [RootCoord] 查询集合 '{name}' 的元数据...")
        return {"name": name, "status": "active"}


class DataCoord:
    """数据协调节点 — 管理数据写入、索引构建。"""

    def __init__(self):
        self.name = "DataCoord"
        self.segments: list[dict] = []

    def assign_segment(self, collection: str, partition: str) -> str:
        """分配数据段。"""
        seg_id = f"seg_{len(self.segments) + 1}"
        self.segments.append({
            "id": seg_id,
            "collection": collection,
            "partition": partition,
            "rows": 0,
        })
        print(f"  [DataCoord] 分配数据段: {seg_id}")
        return seg_id

    def flush_segment(self, seg_id: str) -> None:
        """持久化数据段到存储。"""
        print(f"  [DataCoord] 持久化数据段 {seg_id} 到 MinIO...")


class QueryCoord:
    """查询协调节点 — 管理查询调度、结果聚合。"""

    def __init__(self):
        self.name = "QueryCoord"

    def plan_query(self, collection: str, topk: int) -> list[str]:
        """规划查询执行计划。"""
        print(f"  [QueryCoord] 规划查询: collection={collection}, topk={topk}")
        nodes = ["QueryNode-1", "QueryNode-2"]
        print(f"  [QueryCoord] 选择查询节点: {nodes}")
        return nodes


class QueryNode:
    """查询节点 — 执行实际向量检索。"""

    def __init__(self, node_id: str):
        self.node_id = node_id
        self.name = f"QueryNode-{node_id}"

    def search(self, collection: Collection, query: list[float], topk: int) -> list[dict]:
        """在本地数据上执行向量检索。"""
        print(f"  [{self.name}] 在集合 '{collection.name}' 上执行检索...")
        # 模拟检索耗时
        time.sleep(0.05)
        results: list[dict] = []
        for i, vec in enumerate(collection.data):
            score = sum(a * b for a, b in zip(query, vec["vector"]))
            results.append({"id": vec["id"], "score": score, "data": vec["data"]})
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:topk]


class DataNode:
    """数据节点 — 管理数据持久化。"""

    def __init__(self, node_id: str):
        self.node_id = node_id
        self.name = f"DataNode-{node_id}"

    def insert(self, collection: Collection, records: list[dict]) -> int:
        """写入数据到本地存储。"""
        print(f"  [{self.name}] 写入 {len(records)} 条记录到集合 '{collection.name}'")
        collection.data.extend(records)
        return len(records)


class MinIOStorage:
    """模拟 MinIO 对象存储。"""

    def __init__(self):
        self.buckets: dict[str, list] = {"vectors": [], "index": [], "logs": []}

    def save(self, bucket: str, key: str, data: Any) -> None:
        print(f"  [MinIO] 存储: bucket={bucket}, key={key}")


class EtcdMetaStore:
    """模拟 Etcd 元数据存储。"""

    def __init__(self):
        self.meta: dict[str, Any] = {}

    def put(self, key: str, value: Any) -> None:
        self.meta[key] = value
        print(f"  [Etcd] 写入元数据: {key}")

    def get(self, key: str) -> Any:
        return self.meta.get(key)


# ======================================================================
# 完整流程模拟
# ======================================================================

def demo_write_flow():
    """模拟数据写入流程。"""
    print("=" * 60)
    print("【Milvus 数据写入流程模拟】")
    print("=" * 60)
    print()
    print("  Client → Proxy → RootCoord → DataCoord → DataNode → MinIO")
    print()

    # 初始化组件
    proxy = Proxy()
    root_coord = RootCoord()
    data_coord = DataCoord()
    data_node = DataNode("1")
    minio = MinIOStorage()
    etcd = EtcdMetaStore()

    # 创建集合
    collection = Collection(name="medical_knowledge", dim=128)
    etcd.put(f"collection/{collection.name}", {"dim": collection.dim})

    # 写入流程
    print("--- Step 1: 客户端发起写入请求 ---")
    request = {"type": "insert", "collection": collection.name, "count": 10}
    proxy.handle_request(request)

    print()
    print("--- Step 2: RootCoord 分配时间戳 ---")
    ts = root_coord.allocate_timestamp()
    print(f"  [RootCoord] 分配全局时间戳: {ts}")

    print()
    print("--- Step 3: DataCoord 分配数据段 ---")
    seg_id = data_coord.assign_segment(collection.name, "_default")

    print()
    print("--- Step 4: DataNode 写入数据 ---")
    import random
    records = [
        {
            "id": f"doc_{i}",
            "vector": [random.random() for _ in range(128)],
            "data": {"text": f"文档 {i} 的内容"},
        }
        for i in range(10)
    ]
    data_node.insert(collection, records)

    print()
    print("--- Step 5: DataCoord 触发持久化 ---")
    data_coord.flush_segment(seg_id)
    minio.save("vectors", f"{collection.name}/{seg_id}.parquet", records)

    print()
    print("--- Step 6: 元数据更新 ---")
    etcd.put(f"segment/{seg_id}", {"collection": collection.name, "rows": len(records)})

    print()
    print("✅ 写入流程完成！")
    print()


def demo_search_flow():
    """模拟查询检索流程。"""
    print("=" * 60)
    print("【Milvus 查询检索流程模拟】")
    print("=" * 60)
    print()
    print("  Client → Proxy → QueryCoord → QueryNode → 结果聚合 → Client")
    print()

    # 初始化组件
    proxy = Proxy()
    query_coord = QueryCoord()
    query_node_1 = QueryNode("1")
    query_node_2 = QueryNode("2")

    # 创建带数据的集合
    import random
    collection = Collection(name="medical_knowledge", dim=128)
    for i in range(50):
        collection.data.append({
            "id": f"doc_{i}",
            "vector": [random.random() for _ in range(128)],
            "data": {"text": f"文档 {i} 的内容"},
        })

    # 查询流程
    query = [random.random() for _ in range(128)]

    print("--- Step 1: 客户端发起查询请求 ---")
    request = {"type": "search", "collection": collection.name, "topk": 5}
    proxy.handle_request(request)

    print()
    print("--- Step 2: QueryCoord 规划查询 ---")
    nodes = query_coord.plan_query(collection.name, 5)

    print()
    print("--- Step 3: 分发查询到 QueryNode ---")
    all_results = []
    for node_id in nodes:
        node = QueryNode(node_id)
        results = node.search(collection, query, 5)
        print(f"  [{node.name}] 返回 {len(results)} 条结果")
        all_results.extend(results)

    print()
    print("--- Step 4: QueryCoord 合并结果 ---")
    all_results.sort(key=lambda x: x["score"], reverse=True)
    merged = all_results[:5]
    print(f"  [QueryCoord] 合并后共 {len(merged)} 条结果")

    print()
    print("--- Step 5: Proxy 返回结果 ---")
    print(f"  [Proxy] 返回 Top-{len(merged)} 结果给客户端")
    for i, r in enumerate(merged, 1):
        print(f"    #{i}: id={r['id']}, score={r['score']:.4f}")

    print()
    print("✅ 查询流程完成！")
    print()


def demo_architecture_diagram():
    """打印 Milvus 分层架构图。"""
    print("=" * 60)
    print("【Milvus 分层架构总览】")
    print("=" * 60)
    print()
    print("  ┌─────────────────────────────────────────────────┐")
    print("  │              接入层 (Access Layer)               │")
    print("  │               Proxy / Load Balancer             │")
    print("  ├─────────────────────────────────────────────────┤")
    print("  │              协调层 (Coordinator Layer)          │")
    print("  │   RootCoord  QueryCoord  DataCoord  IndexCoord  │")
    print("  ├─────────────────────────────────────────────────┤")
    print("  │              执行层 (Worker Layer)               │")
    print("  │   QueryNode  DataNode  IndexNode                 │")
    print("  ├─────────────────────────────────────────────────┤")
    print("  │              存储层 (Storage Layer)              │")
    print("  │   MetaStore(Etcd)  ObjectStore(MinIO)  LogStore │")
    print("  └─────────────────────────────────────────────────┘")
    print()
    print("  数据流路径：")
    print("  写入: Client → Proxy → RootCoord → DataCoord → DataNode → MinIO")
    print("  查询: Client → Proxy → QueryCoord → QueryNode → 聚合 → Client")
    print("  索引: DataNode → IndexCoord → IndexNode → MinIO")
    print()


# ======================================================================
# Main
# ======================================================================

def main():
    print("=" * 60)
    print("  Milvus 实战指南 — Ch4: Milvus 整体架构与核心组件")
    print("  本演示模拟 Milvus 各组件的交互流程")
    print("=" * 60)
    print()

    demo_architecture_diagram()
    demo_write_flow()
    demo_search_flow()

    print("=" * 60)
    print("  演示完成！关键要点：")
    print("  - Proxy: 统一入口，请求路由与负载均衡")
    print("  - RootCoord: 全局元数据管理与时间戳分配")
    print("  - QueryCoord: 查询计划与结果聚合")
    print("  - DataCoord: 数据段管理与持久化调度")
    print("  - QueryNode: 执行向量检索")
    print("  - DataNode: 数据写入与本地存储")
    print("  - Etcd: 元数据存储")
    print("  - MinIO: 向量数据与索引文件存储")
    print("=" * 60)


if __name__ == "__main__":
    main()
