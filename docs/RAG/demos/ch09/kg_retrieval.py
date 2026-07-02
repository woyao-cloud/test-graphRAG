"""
第 9 章 Demo：知识图谱检索

演示在内存图结构上的知识图谱检索：
  实体链接 → 图遍历 → 混合检索
可独立运行，无需外部依赖。

用法：
  python kg_retrieval.py
  python kg_retrieval.py --query "恒瑞医药的供应商是谁？"
"""

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class Entity:
    id: str
    name: str
    type: str  # company/drug/chemical/hospital/regulator
    description: str = ""
    aliases: list[str] = field(default_factory=list)


@dataclass
class Relation:
    source_id: str
    target_id: str
    type: str  # PRODUCES/SUPPLIES/DISTRIBUTES/PURCHASES/REGULATES
    properties: dict = field(default_factory=dict)


class KnowledgeGraph:
    """内存知识图谱。"""

    def __init__(self):
        self.entities: dict[str, Entity] = {}
        self.relations: list[Relation] = []
        self.adjacency: dict[str, list[tuple[str, str, str]]] = defaultdict(list)  # node_id -> [(rel_type, target_id, target_name)]

    def add_entity(self, entity: Entity):
        self.entities[entity.id] = entity

    def add_relation(self, relation: Relation):
        self.relations.append(relation)
        target_name = self.entities[relation.target_id].name if relation.target_id in self.entities else ""
        self.adjacency[relation.source_id].append((relation.type, relation.target_id, target_name))
        self.adjacency[relation.target_id].append((relation.type, relation.source_id, self.entities.get(relation.source_id, Entity("","","")).name))


# ============================================================================
# Build Sample Graph
# ============================================================================


def build_pharma_graph() -> KnowledgeGraph:
    """构建医药供应链知识图谱。"""
    kg = KnowledgeGraph()

    # 实体
    entities = [
        Entity("e1", "恒瑞医药", "company", "中国领先的制药企业，专注于抗肿瘤药物", ["恒瑞", "恒瑞医药有限公司"]),
        Entity("e2", "齐鲁制药", "company", "中国主要制药企业，生产抗肿瘤和抗感染药物", ["齐鲁"]),
        Entity("e3", "注射用紫杉醇", "drug", "抗肿瘤药物，用于非小细胞肺癌和乳腺癌治疗", ["紫杉醇", "紫杉醇注射剂"]),
        Entity("e4", "卡瑞利珠单抗", "drug", "PD-1抑制剂，用于霍奇金淋巴瘤治疗", []),
        Entity("e5", "吉非替尼片", "drug", "EGFR-TKI靶向药，用于非小细胞肺癌治疗", ["吉非替尼"]),
        Entity("e6", "华海药业", "company", "紫杉醇API原料药供应商，年产能5000kg", ["华海"]),
        Entity("e7", "国药控股", "company", "中国最大药品分销商，华东区覆盖37家三甲医院", ["国药"]),
        Entity("e8", "北京协和医院", "hospital", "三级甲等综合医院", ["协和医院"]),
        Entity("e9", "华东医院", "hospital", "三级甲等综合医院，位于上海", []),
        Entity("e10", "紫杉醇API", "chemical", "紫杉醇原料药，用于制备注射用紫杉醇", ["原料药"]),
        Entity("e11", "国家药监局", "regulator", "药品监督管理机构", ["NMPA", "药监局"]),
        Entity("e12", "正大天晴", "company", "中国制药企业", []),
    ]
    for e in entities:
        kg.add_entity(e)

    # 关系
    relations = [
        # 生产关系
        Relation("e1", "e3", "PRODUCES", {"since": 2010, "share": "100%"}),
        Relation("e1", "e4", "PRODUCES", {"since": 2019}),
        Relation("e2", "e5", "PRODUCES", {"since": 2015}),
        # 供应链关系
        Relation("e6", "e10", "SUPPLIES", {"annual_volume": "5000kg", "customers": ["恒瑞医药"]}),
        Relation("e10", "e3", "IS_RAW_MATERIAL_OF", {}),
        Relation("e3", "e1", "PRODUCED_BY", {}),
        # 分销关系
        Relation("e7", "e1", "DISTRIBUTES", {"region": "华东区", "contract_value": "5亿元/年"}),
        Relation("e7", "e2", "DISTRIBUTES", {"region": "华东区"}),
        # 采购关系
        Relation("e8", "e3", "PURCHASES", {"annual_volume": "50000支", "supplier": "恒瑞医药"}),
        Relation("e9", "e3", "PURCHASES", {"annual_volume": "30000支"}),
        Relation("e8", "e5", "PURCHASES", {"annual_volume": "20000盒"}),
        # 监管关系
        Relation("e11", "e3", "REGULATES", {"approval_number": "国药准字H20000001"}),
        Relation("e11", "e4", "REGULATES", {}),
        # 合作关系
        Relation("e1", "e12", "COOPERATES_WITH", {}),
    ]
    for r in relations:
        kg.add_relation(r)

    return kg


# ============================================================================
# Entity Linker
# ============================================================================


class EntityLinker:
    """实体链接器。"""

    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg
        self._build_index()

    def _build_index(self):
        """构建实体名→ID 索引。"""
        self.name_to_id = {}
        for eid, entity in self.kg.entities.items():
            self.name_to_id[entity.name] = eid
            for alias in entity.aliases:
                self.name_to_id[alias] = eid

    def link(self, text: str) -> list[tuple[str, str]]:
        """识别文本中的实体，返回 (实体ID, 实体名) 列表。"""
        found = []
        for name, eid in sorted(self.name_to_id.items(), key=lambda x: -len(x[0])):
            if name in text:
                found.append((eid, self.kg.entities[eid].name))
        # 去重（保留最长匹配）
        seen = set()
        unique = []
        for eid, name in found:
            if eid not in seen:
                seen.add(eid)
                unique.append((eid, name))
        return unique


# ============================================================================
# Graph Traverser
# ============================================================================


class GraphTraverser:
    """知识图谱遍历器。"""

    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg

    def ego_network(self, entity_id: str, max_depth: int = 2) -> dict:
        """获取实体的自我网络。"""
        visited = {entity_id: 0}
        queue = [(entity_id, 0)]
        network = {"nodes": {}, "edges": []}

        while queue:
            current, depth = queue.pop(0)
            if depth >= max_depth:
                continue

            entity = self.kg.entities.get(current)
            if not entity:
                continue
            network["nodes"][current] = {"id": current, "name": entity.name, "type": entity.type}

            for rel_type, neighbor_id, neighbor_name in self.kg.adjacency.get(current, []):
                if neighbor_id not in visited:
                    visited[neighbor_id] = depth + 1
                    queue.append((neighbor_id, depth + 1))

                network["edges"].append({
                    "source": current, "target": neighbor_id,
                    "type": rel_type,
                })

                if neighbor_id not in network["nodes"]:
                    neighbor = self.kg.entities.get(neighbor_id)
                    if neighbor:
                        network["nodes"][neighbor_id] = {
                            "id": neighbor_id, "name": neighbor.name, "type": neighbor.type,
                        }

        return network

    def find_path(self, source_id: str, target_id: str) -> Optional[list[str]]:
        """BFS 最短路径。"""
        visited = {source_id}
        queue = [[source_id]]

        while queue:
            path = queue.pop(0)
            current = path[-1]
            if current == target_id:
                return path

            for rel_type, neighbor_id, _ in self.kg.adjacency.get(current, []):
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append(path + [neighbor_id])
        return None

    def format_path(self, path: list[str]) -> str:
        """将路径格式化为可读字符串。"""
        if not path:
            return "未找到路径"
        parts = []
        for i, node_id in enumerate(path):
            entity = self.kg.entities.get(node_id)
            if not entity:
                continue
            if i > 0:
                # 找到连接的关系
                prev = path[i - 1]
                for rel_type, neighbor_id, _ in self.kg.adjacency.get(prev, []):
                    if neighbor_id == node_id:
                        parts.append(f"-[{rel_type}]->")
                        break
            parts.append(entity.name)
        return " ".join(parts)


# ============================================================================
# KG-enhanced Retriever
# ============================================================================


class KGRetriever:
    """知识图谱增强的检索器。"""

    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg
        self.linker = EntityLinker(kg)
        self.traverser = GraphTraverser(kg)

    def retrieve(self, query: str) -> dict:
        """检索知识图谱，返回结构化上下文。"""
        # 1. 实体链接
        entities = self.linker.link(query)
        if not entities:
            return {"query": query, "entities_found": [], "contexts": []}

        # 2. 图遍历
        contexts = []
        all_nodes = set()
        for eid, name in entities:
            network = self.traverser.ego_network(eid, max_depth=2)

            # 格式化上下文
            entity = self.kg.entities[eid]
            context_text = f"[实体] {entity.name} ({entity.type}): {entity.description}"

            # 添加关系描述
            for edge in network["edges"]:
                src_name = network["nodes"].get(edge["source"], {}).get("name", "")
                tgt_name = network["nodes"].get(edge["target"], {}).get("name", "")
                if edge["type"] in ["PRODUCES", "SUPPLIES", "DISTRIBUTES", "PURCHASES"]:
                    context_text += f"\n  - {edge['type']}: {src_name} → {tgt_name}"

            contexts.append({
                "entity_id": eid,
                "entity_name": name,
                "context": context_text,
                "network": network,
            })
            all_nodes.update(network["nodes"].keys())

        # 3. 路径分析（如果查询涉及多个实体，计算路径）
        paths = []
        if len(entities) >= 2:
            for i in range(len(entities)):
                for j in range(i + 1, len(entities)):
                    path = self.traverser.find_path(entities[i][0], entities[j][0])
                    if path:
                        paths.append({
                            "from": entities[i][1],
                            "to": entities[j][1],
                            "path": self.traverser.format_path(path),
                        })

        return {
            "query": query,
            "entities_found": [{"id": e[0], "name": e[1]} for e in entities],
            "contexts": contexts,
            "paths_between_entities": paths,
        }


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="知识图谱检索演示")
    parser.add_argument("--query", default="恒瑞医药的紫杉醇供应商是谁？", help="查询")
    args = parser.parse_args()

    # 构建知识图谱
    kg = build_pharma_graph()
    print(f"[KG] 实体数: {len(kg.entities)}, 关系数: {len(kg.relations)}")

    retriever = KGRetriever(kg)

    # 实体链接测试
    print(f"\n{'=' * 60}")
    print(f"[Query] {args.query}")
    print(f"{'=' * 60}")

    result = retriever.retrieve(args.query)

    print(f"\n[Entities Found]")
    for e in result["entities_found"]:
        print(f"  - {e['name']} ({e['id']})")

    print(f"\n[KG Contexts]")
    for ctx in result["contexts"]:
        print(f"\n  --- {ctx['entity_name']} ---")
        for line in ctx["context"].split("\n"):
            print(f"  {line}")

    if result["paths_between_entities"]:
        print(f"\n[Paths]")
        for p in result["paths_between_entities"]:
            print(f"  {p['from']} → {p['to']}: {p['path']}")

    # 额外演示：最短路径查询
    print(f"\n{'=' * 60}")
    print("[Demo] 最短路径: 恒瑞医药 → 北京协和医院")
    print(f"{'=' * 60}")

    traverser = GraphTraverser(kg)
    path = traverser.find_path("e1", "e8")
    print(f"  {traverser.format_path(path)}")

    print(f"\n{'=' * 60}")
    print("[Demo] 最短路径: 华海药业 → 华东医院")
    print(f"{'=' * 60}")
    path = traverser.find_path("e6", "e9")
    print(f"  {traverser.format_path(path)}")


if __name__ == "__main__":
    main()
