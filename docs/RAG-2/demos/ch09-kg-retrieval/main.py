#!/usr/bin/env python3
"""
ch09-kg-retrieval: Knowledge Graph retrieval demo with a pharmaceutical KG.
Builds a KG of entities and relationships, then demonstrates graph traversal,
path finding, and RAG context formatting. Uses only stdlib.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple
from collections import deque


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class Entity:
    name: str
    type: str  # e.g. "公司", "药物", "疾病", "医院", "靶点"
    description: str = ""

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return f"[{self.type}] {self.name}"


@dataclass
class Relationship:
    source: str
    target: str
    rel_type: str  # e.g. "研发", "治疗", "分销", "合作"

    def __repr__(self):
        return f"{self.source} --[{self.rel_type}]--> {self.target}"


# ---------------------------------------------------------------------------
# Knowledge Graph
# ---------------------------------------------------------------------------
class KnowledgeGraph:
    """Simple in-memory knowledge graph."""

    def __init__(self):
        self.entities: Dict[str, Entity] = {}
        self.relationships: List[Relationship] = []
        # adjacency: source -> list of (target, rel_type)
        self._adj: Dict[str, List[Tuple[str, str]]] = {}
        # reverse adjacency: target -> list of (source, rel_type)
        self._radj: Dict[str, List[Tuple[str, str]]] = {}

    def add_entity(self, entity: Entity):
        self.entities[entity.name] = entity

    def add_relationship(self, rel: Relationship):
        self.relationships.append(rel)
        if rel.source not in self._adj:
            self._adj[rel.source] = []
        self._adj[rel.source].append((rel.target, rel.rel_type))
        if rel.target not in self._radj:
            self._radj[rel.target] = []
        self._radj[rel.target].append((rel.source, rel.rel_type))

    def get_entity(self, name: str) -> Optional[Entity]:
        return self.entities.get(name)

    def get_relationships(self, source: Optional[str] = None) -> List[Relationship]:
        if source is None:
            return self.relationships
        return [r for r in self.relationships if r.source == source]


# ---------------------------------------------------------------------------
# Build a pharmaceutical knowledge graph
# ---------------------------------------------------------------------------
def build_pharma_kg() -> KnowledgeGraph:
    kg = KnowledgeGraph()

    # Entities
    entities = [
        Entity("恒瑞医药", "公司", "中国领先的创新药研发企业"),
        Entity("注射用紫杉醇", "药物", "微管抑制剂类抗肿瘤化疗药物"),
        Entity("奥希替尼片", "药物", "第三代EGFR-TKI靶向药物"),
        Entity("国药控股", "公司", "中国最大的医药分销企业"),
        Entity("北京协和医院", "医院", "国家级三级甲等综合医院"),
        Entity("非小细胞肺癌", "疾病", "最常见的肺癌类型"),
        Entity("乳腺癌", "疾病", "女性最常见的恶性肿瘤之一"),
        Entity("EGFR", "靶点", "表皮生长因子受体，肺癌重要治疗靶点"),
        Entity("PD-1", "靶点", "免疫检查点蛋白，肿瘤免疫治疗靶点"),
        Entity("卡瑞利珠单抗", "药物", "恒瑞医药研发的PD-1抑制剂"),
    ]
    for e in entities:
        kg.add_entity(e)

    # Relationships
    rels = [
        Relationship("恒瑞医药", "卡瑞利珠单抗", "研发"),
        Relationship("恒瑞医药", "注射用紫杉醇", "生产"),
        Relationship("卡瑞利珠单抗", "PD-1", "靶向"),
        Relationship("奥希替尼片", "EGFR", "靶向"),
        Relationship("奥希替尼片", "非小细胞肺癌", "治疗"),
        Relationship("注射用紫杉醇", "乳腺癌", "治疗"),
        Relationship("注射用紫杉醇", "非小细胞肺癌", "治疗"),
        Relationship("国药控股", "恒瑞医药", "分销"),
        Relationship("国药控股", "北京协和医院", "供应"),
        Relationship("北京协和医院", "非小细胞肺癌", "诊疗"),
        Relationship("北京协和医院", "乳腺癌", "诊疗"),
        Relationship("恒瑞医药", "北京协和医院", "临床合作"),
    ]
    for r in rels:
        kg.add_relationship(r)

    return kg


# ---------------------------------------------------------------------------
# GraphQueryEngine
# ---------------------------------------------------------------------------
class GraphQueryEngine:
    """Query engine for knowledge graph traversal and path finding."""

    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg

    def get_neighbors(self, name: str, hops: int = 1) -> List[Tuple[str, str, int]]:
        """Get entities reachable within `hops` steps, returning (entity_name, rel_type, hop)."""
        if hops < 1:
            return []

        visited: Set[str] = {name}
        results: List[Tuple[str, str, int]] = []
        # BFS queue: (entity_name, rel_type_used_to_get_here, hop)
        queue = deque()
        # Initialize with direct neighbors
        for target, rel in self.kg._adj.get(name, []):
            queue.append((target, rel, 1))
        for source, rel in self.kg._radj.get(name, []):
            queue.append((source, f"{rel}(反向)", 1))

        while queue:
            current, rel, hop = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            results.append((current, rel, hop))

            if hop < hops:
                for target, r in self.kg._adj.get(current, []):
                    if target not in visited:
                        queue.append((target, r, hop + 1))
                for source, r in self.kg._radj.get(current, []):
                    if source not in visited:
                        queue.append((source, f"{r}(反向)", hop + 1))

        return results

    def find_path(self, source: str, target: str) -> Optional[List[Tuple[str, str]]]:
        """BFS shortest path from source to target. Returns list of (entity, rel_type) pairs."""
        if source not in self.kg.entities or target not in self.kg.entities:
            return None

        visited: Set[str] = {source}
        # queue: (current_entity, path_so_far)
        # path_so_far: list of (entity, rel_type)
        queue = deque()
        for t, r in self.kg._adj.get(source, []):
            queue.append((t, [(source, ""), (t, r)]))
        for s, r in self.kg._radj.get(source, []):
            queue.append((s, [(source, ""), (s, f"{r}(反向)")]))

        while queue:
            current, path = queue.popleft()
            if current == target:
                return path
            if current in visited:
                continue
            visited.add(current)
            for t, r in self.kg._adj.get(current, []):
                if t not in visited:
                    queue.append((t, path + [(t, r)]))
            for s, r in self.kg._radj.get(current, []):
                if s not in visited:
                    queue.append((s, path + [(s, f"{r}(反向)")]))

        return None

    def format_for_rag(self, entity_names: List[str]) -> str:
        """Format entities and their relationships as RAG context."""
        lines: List[str] = ["知识图谱上下文:", ""]
        for name in entity_names:
            entity = self.kg.get_entity(name)
            if not entity:
                continue
            lines.append(f"## {entity.name} ({entity.type})")
            if entity.description:
                lines.append(f"   描述: {entity.description}")
            # Outgoing relationships
            out_rels = self.kg.get_relationships(name)
            if out_rels:
                for r in out_rels:
                    target_ent = self.kg.get_entity(r.target)
                    target_desc = f" ({target_ent.type})" if target_ent else ""
                    lines.append(f"   -> [{r.rel_type}] {r.target}{target_desc}")
            # Incoming relationships
            in_rels = [(r.source, r.rel_type) for r in self.kg.relationships if r.target == name]
            if in_rels:
                for src, rel in in_rels:
                    src_ent = self.kg.get_entity(src)
                    src_desc = f" ({src_ent.type})" if src_ent else ""
                    lines.append(f"   <- [{rel}] {src}{src_desc}")
            lines.append("")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("第9章 知识图谱检索演示 (KG Retrieval Demo)")
    print("=" * 60)

    # Build KG
    print("\n[构建知识图谱]")
    kg = build_pharma_kg()
    print(f"   实体数: {len(kg.entities)}")
    print(f"   关系数: {len(kg.relationships)}")
    print("\n实体列表:")
    for name, ent in kg.entities.items():
        print(f"   {ent}")
    print("\n关系列表:")
    for rel in kg.relationships:
        print(f"   {rel}")

    engine = GraphQueryEngine(kg)

    # --- Graph traversal ---
    print(f"\n{'=' * 60}")
    print("[图谱遍历] 从 恒瑞医药 出发")
    print(f"{'=' * 60}")

    print("\n--- 1跳邻居 ---")
    neighbors_1 = engine.get_neighbors("恒瑞医药", hops=1)
    for name, rel, hop in neighbors_1:
        ent = kg.get_entity(name)
        print(f"   {ent}  via [{rel}]")

    print("\n--- 2跳邻居 ---")
    neighbors_2 = engine.get_neighbors("恒瑞医药", hops=2)
    for name, rel, hop in neighbors_2:
        ent = kg.get_entity(name)
        print(f"   [跳数={hop}] {ent}  via [{rel}]")

    # --- Path finding ---
    print(f"\n{'=' * 60}")
    print("[路径查找]")
    print(f"{'=' * 60}")

    paths_to_find = [
        ("恒瑞医药", "非小细胞肺癌"),
        ("国药控股", "奥希替尼片"),
        ("卡瑞利珠单抗", "北京协和医院"),
    ]
    for src, tgt in paths_to_find:
        print(f"\n--- 路径: {src} -> {tgt} ---")
        path = engine.find_path(src, tgt)
        if path:
            for entity, rel in path:
                if rel:
                    print(f"   --[{rel}]--> {entity}")
                else:
                    print(f"   {entity}")
        else:
            print(f"   未找到路径")

    # --- RAG context ---
    print(f"\n{'=' * 60}")
    print("[RAG上下文格式化]")
    print(f"{'=' * 60}")
    context = engine.format_for_rag(["恒瑞医药", "奥希替尼片", "非小细胞肺癌", "EGFR"])
    print(f"\n{context}")

    print("\n" + "=" * 60)
    print("演示完成。")
    print("=" * 60)


if __name__ == "__main__":
    main()
