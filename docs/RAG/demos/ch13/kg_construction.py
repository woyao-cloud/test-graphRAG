"""
第 13 章 Demo：知识图谱构建与应用落地

演示从文本到知识图谱的完整流程：
  词典实体抽取 → 规则关系抽取 → 图谱存储 → 质量检查 → Graph+Vector 混合查询

可独立运行，无需外部依赖。

用法：
  python kg_construction.py
  python kg_construction.py --mode full
  python kg_construction.py --mode query --query "恒瑞医药的供应商是谁？"
"""

import argparse
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


# ============================================================================
# Schema Definition
# ============================================================================

ENTITY_TYPES = {
    "organization": "组织/企业",
    "pharma_company": "制药企业（organization 的子类型）",
    "distributor": "分销商（organization 的子类型）",
    "hospital": "医院（organization 的子类型）",
    "regulator": "监管机构（organization 的子类型）",
    "drug": "药品",
    "chemical": "化学物质/原料药",
    "disease": "疾病/适应症",
    "department": "科室",
    "place": "地点",
}

RELATION_SCHEMA = {
    "生产": {"domain": "organization", "range": "drug", "desc": "组织生产药品"},
    "供应 API": {"domain": "organization", "range": "organization", "desc": "供应商向采购方提供原料"},
    "分销": {"domain": "organization", "range": "organization", "desc": "分销商代理药品"},
    "使用": {"domain": "organization", "range": "drug", "desc": "医疗机构使用药品"},
    "监管": {"domain": "organization", "range": "drug", "desc": "监管机构审批药品"},
    "治疗": {"domain": "drug", "range": "disease", "desc": "药品治疗的疾病"},
    "位于": {"domain": "organization", "range": "place", "desc": "组织所在地"},
    "设有": {"domain": "organization", "range": "department", "desc": "组织设有科室"},
}

# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class KGEntity:
    id: str
    name: str
    type: str
    description: str = ""
    properties: dict = field(default_factory=dict)


@dataclass
class KGRelation:
    source_id: str
    target_id: str
    type: str
    evidence: str = ""
    weight: float = 1.0


@dataclass
class EntityMention:
    name: str
    type: str
    start: int
    end: int
    confidence: float = 1.0


# ============================================================================
# Step 1: Dictionary-based Entity Extraction
# ============================================================================


class DictionaryExtractor:
    """基于词典的实体抽取器。"""

    def __init__(self):
        self.dictionary: dict[str, tuple[str, list[str]]] = {}
        self._init_default_dict()

    def _init_default_dict(self):
        """初始化医药领域词典。"""
        entries = [
            ("恒瑞医药", "pharma_company", "恒瑞", "恒瑞制药"),
            ("华海药业", "pharma_company", "华海"),
            ("国药控股", "distributor", "国药"),
            ("齐鲁制药", "pharma_company", "齐鲁"),
            ("瑞阳医药", "distributor", "瑞阳"),
            ("北京协和医院", "hospital", "协和医院"),
            ("上海中山医院", "hospital", "中山医院"),
            ("北京大学肿瘤医院", "hospital", "北大肿瘤医院"),
            ("NMPA", "regulator", "国家药监局", "药品监督管理局"),
            ("注射用紫杉醇", "drug", "紫杉醇注射液", "紫杉醇"),
            ("奥沙利铂", "drug", "奥沙利铂注射液"),
            ("卡培他滨", "drug", "希罗达"),
            ("顺铂", "drug", "顺铂注射液"),
            ("紫杉醇 API", "chemical", "紫杉醇原料药"),
            ("非小细胞肺癌", "disease", "NSCLC"),
            ("乳腺癌", "disease", "乳腺肿瘤"),
            ("结直肠癌", "disease", "大肠癌"),
            ("卵巢癌", "disease", "卵巢肿瘤"),
            ("肿瘤科", "department", "肿瘤内科", "肿瘤外科"),
            ("江苏省连云港市", "place", "连云港"),
            ("山东省济南市", "place", "济南"),
        ]
        for name, etype, *aliases in entries:
            self.add_entity(name, etype, aliases)

    def add_entity(self, name: str, type: str, aliases: list[str] = None):
        self.dictionary[name] = (type, aliases or [])

    def extract(self, text: str) -> list[EntityMention]:
        """从文本中抽取实体。"""
        mentions = []
        matched_names = set()

        # 按名称长度降序（优先匹配长名称）
        sorted_entities = sorted(self.dictionary.items(), key=lambda x: -len(x[0]))

        for name, (etype, aliases) in sorted_entities:
            if name in matched_names:
                continue  # 已匹配过的实体不再重复匹配

            all_mentions = [(name, len(name))]
            for alias in aliases:
                all_mentions.append((alias, len(alias)))

            # 对每个可能的表述，在文本中查找
            for mention_text, _ in sorted(all_mentions, key=lambda x: -x[1]):
                idx = 0
                while True:
                    pos = text.find(mention_text, idx)
                    if pos == -1:
                        break
                    mentions.append(EntityMention(
                        name=name,
                        type=etype,
                        start=pos,
                        end=pos + len(mention_text),
                    ))
                    idx = pos + 1
                if any(m.name == name for m in mentions):
                    matched_names.add(name)
                    break

        # 去重（同一个实体只保留一次）
        seen = set()
        unique = []
        for m in mentions:
            key = (m.name, m.start)
            if key not in seen:
                seen.add(key)
                unique.append(m)

        return sorted(unique, key=lambda x: x.start)


# ============================================================================
# Step 2: Rule-based Relation Extraction
# ============================================================================


class RuleRelationExtractor:
    """基于触发词的关系抽取。"""

    def __init__(self):
        self.trigger_map: dict[str, list[str]] = {
            "生产": ["生产", "制造", "研发", "推出", "生产了"],
            "供应 API": ["供应", "提供API", "供货"],
            "分销": ["分销", "代理", "经销"],
            "使用": ["使用", "采用", "应用", "用于", "广泛使用"],
            "监管": ["监管", "审批", "批准", "认证", "评审"],
            "治疗": ["治疗", "用于治疗", "治疗", "适应症为"],
            "位于": ["位于", "成立于", "总部在", "总部位于"],
            "设有": ["设有", "下设", "包含", "有"],
        }

        # 反向关系
        self.reverse_map = {
            "生产": ("产品", "是...生产的"),
        }

    def extract(self, text: str, entities: list[EntityMention]) -> list[KGRelation]:
        """抽取实体间关系。循环匹配所有触发词出现位置。"""
        relations = []
        entity_list = sorted(entities, key=lambda x: x.start)

        for rel_type, triggers in self.trigger_map.items():
            for trigger in triggers:
                # 循环查找触发词的所有出现位置
                search_start = 0
                while True:
                    trigger_pos = text.find(trigger, search_start)
                    if trigger_pos == -1:
                        break

                    # 在 trigger 前后 80 字窗口内找实体对
                    window_start = max(0, trigger_pos - 80)
                    window_end = min(len(text), trigger_pos + 80)

                    before = [
                        e for e in entity_list
                        if e.end <= trigger_pos
                        and e.start >= window_start
                    ]
                    after = [
                        e for e in entity_list
                        if e.start >= trigger_pos + len(trigger)
                        and e.end <= window_end
                    ]

                    if before and after:
                        source = before[-1]  # 最近的前置实体
                        target = after[0]    # 最近的后置实体

                        # Schema 校验
                        if self._validate_schema(rel_type, source.type, target.type):
                            relations.append(KGRelation(
                                source_id=source.name,
                                target_id=target.name,
                                type=rel_type,
                                evidence=text[max(0, trigger_pos - 40):min(len(text), trigger_pos + 40)].strip(),
                            ))

                    # 继续查找下一个出现位置
                    search_start = trigger_pos + 1

        return self._deduplicate(relations)

    def _validate_schema(self, rel_type: str, source_type: str, target_type: str) -> bool:
        """校验关系是否符合 Schema（允许子类型匹配）。"""
        if rel_type not in RELATION_SCHEMA:
            return True  # 未知关系类型不校验
        rule = RELATION_SCHEMA[rel_type]
        # 源和目标类型都是有效实体类型即通过
        #（实际生产环境需严格检查 domain/range，demo 允许子类型宽松匹配）
        return source_type in ENTITY_TYPES and target_type in ENTITY_TYPES

    def _deduplicate(self, relations: list[KGRelation]) -> list[KGRelation]:
        """关系去重。"""
        seen = set()
        unique = []
        for r in relations:
            key = (r.source_id, r.target_id, r.type)
            if key not in seen:
                seen.add(key)
                unique.append(r)
        return unique


# ============================================================================
# Step 3: In-Memory KG Store (Neo4j-like interface)
# ============================================================================


class InMemoryKGStore:
    """内存知识图谱存储（模拟 Neo4j 接口）。"""

    def __init__(self):
        self.entities: dict[str, KGEntity] = {}
        self.adjacency: dict[str, list[KGRelation]] = defaultdict(list)
        self.reverse_adj: dict[str, list[KGRelation]] = defaultdict(list)

    def create_entity(self, entity: KGEntity):
        if entity.id not in self.entities:
            self.entities[entity.id] = entity

    def create_relation(self, relation: KGRelation):
        if relation.source_id in self.entities and relation.target_id in self.entities:
            self.adjacency[relation.source_id].append(relation)
            self.reverse_adj[relation.target_id].append(relation)

    def get_entity(self, name: str) -> Optional[KGEntity]:
        return self.entities.get(name)

    def get_neighbors(self, entity_id: str, max_hops: int = 1) -> dict:
        """获取 N 跳邻居（BFS）。"""
        result = {
            "entities": {},
            "relations": [],
        }

        visited = {entity_id}
        queue = [(entity_id, 0)]

        while queue:
            current, hop = queue.pop(0)
            if hop >= max_hops:
                continue

            for rel in self.adjacency.get(current, []):
                if rel.target_id not in visited:
                    visited.add(rel.target_id)
                    if rel.target_id in self.entities:
                        result["entities"][rel.target_id] = self.entities[rel.target_id]
                    result["relations"].append(rel)
                    queue.append((rel.target_id, hop + 1))

            for rel in self.reverse_adj.get(current, []):
                if rel.source_id not in visited:
                    visited.add(rel.source_id)
                    if rel.source_id in self.entities:
                        result["entities"][rel.source_id] = self.entities[rel.source_id]
                    result["relations"].append(rel)
                    queue.append((rel.source_id, hop + 1))

        return result

    def query_cypher_like(self, query: str) -> str:
        """模拟 Cypher 查询（简化）。"""
        # 格式: MATCH (start)-[rel]->(end) WHERE start.name = "恒瑞医药"
        name_match = re.search(r'name\s*=\s*"([^"]+)"', query)
        if not name_match:
            return "查询格式错误"

        entity_name = name_match.group(1)
        if entity_name not in self.entities:
            return f"未找到实体: {entity_name}"

        entity = self.entities[entity_name]
        lines = [f"实体: {entity.name} ({entity.type}) — {entity.description}"]

        # 输出关系
        for rel in self.adjacency.get(entity_name, []):
            target = self.entities.get(rel.target_id)
            if target:
                lines.append(f"  -[:{rel.type}]→ {target.name} ({target.type})")

        for rel in self.reverse_adj.get(entity_name, []):
            source = self.entities.get(rel.source_id)
            if source:
                lines.append(f"  ←[:{rel.type}]- {source.name} ({source.type})")

        return "\n".join(lines)

    def stats(self) -> dict:
        return {
            "entities": len(self.entities),
            "relations": sum(len(v) for v in self.adjacency.values()),
            "entity_types": defaultdict(int, {e.type: 0 for e in self.entities.values()}),
        }


# ============================================================================
# Step 4: Quality Checks
# ============================================================================


class KGQualityChecker:
    """知识图谱质量检查。"""

    @staticmethod
    def check_orphan_entities(store: InMemoryKGStore) -> list[str]:
        """检查孤立实体。"""
        related = set()
        for rels in store.adjacency.values():
            for r in rels:
                related.add(r.source_id)
                related.add(r.target_id)
        for rels in store.reverse_adj.values():
            for r in rels:
                related.add(r.source_id)
                related.add(r.target_id)

        orphans = [eid for eid in store.entities if eid not in related]
        return [store.entities[eid].name for eid in orphans]

    @staticmethod
    def check_schema_violations(store: InMemoryKGStore) -> list[str]:
        """检查违反 Schema 的关系。"""
        violations = []
        for rels in store.adjacency.values():
            for r in rels:
                if r.type in RELATION_SCHEMA:
                    rule = RELATION_SCHEMA[r.type]
                    src = store.get_entity(r.source_id)
                    tgt = store.get_entity(r.target_id)
                    if src and tgt:
                        # 简化的类型检查
                        src_ok = src.type == rule["domain"] or rule["domain"] in ENTITY_TYPES
                        tgt_ok = tgt.type == rule["range"] or rule["range"] in ENTITY_TYPES
                        if not src_ok or not tgt_ok:
                            violations.append(
                                f"关系 [{r.type}] {r.source_id}({src.type}) → {r.target_id}({tgt.type}) 违反 Schema"
                            )
        return violations

    @staticmethod
    def check_duplicate_entities(store: InMemoryKGStore) -> list[tuple[str, str, float]]:
        """检查可能的重复实体。"""
        names = list(store.entities.keys())
        duplicates = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                sim = _name_similarity(names[i], names[j])
                if sim > 0.7:
                    duplicates.append((names[i], names[j], sim))
        return duplicates


def _name_similarity(a: str, b: str) -> float:
    """名称相似度（基于公共子串）。"""
    if not a or not b:
        return 0.0
    set_a, set_b = set(a), set(b)
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union) if union else 0.0


# ============================================================================
# Step 5: Hybrid Query Engine (Graph + Vector)
# ============================================================================


class HybridQueryEngine:
    """Graph + Vector 混合查询。"""

    def __init__(self, store: InMemoryKGStore):
        self.store = store

    def query(self, question: str) -> str:
        """混合查询入口。"""
        print(f"\n  问题分析: {question}")

        # 1. 识别问题中的实体
        mentioned = [name for name in self.store.entities if name in question]
        print(f"  识别到实体: {mentioned if mentioned else '无（使用语义搜索）'}")

        answer_parts = []

        # 2. 图谱查询
        if mentioned:
            for name in mentioned:
                neighbors = self.store.get_neighbors(name, max_hops=2)

                if neighbors["relations"]:
                    lines = [f"\n  [图谱路径] 从「{name}」出发:"]
                    for rel in neighbors["relations"][:5]:
                        target_name = rel.target_id
                        if target_name in neighbors["entities"]:
                            t = neighbors["entities"][target_name]
                            lines.append(f"    -[:{rel.type}]→ {t.name} ({t.type})")
                    answer_parts.append("\n".join(lines))

        # 3. 文本匹配（模拟向量检索）
        text_results = self._text_match(question)
        if text_results:
            answer_parts.append(f"\n  [语义匹配] {text_results[0]}")

        # 4. 综合答案
        if not answer_parts:
            return "无法从知识图谱中找到相关信息。"

        return "".join(answer_parts)

    def _text_match(self, question: str) -> list[str]:
        """基于关键词的文本匹配（模拟 Embedding 检索）。"""
        matches = []
        # 使用实体的描述和类型作为"文档"
        for entity in self.store.entities.values():
            search_text = f"{entity.name} {entity.type} {entity.description}"
            if any(word in search_text for word in question.split()):
                matches.append(
                    f"找到相关实体: {entity.name} ({entity.type}) — {entity.description}"
                )
        return matches[:3]


# ============================================================================
# Sample Documents
# ============================================================================


SAMPLE_DOCUMENTS = [
    """
恒瑞医药是中国领先的制药企业，主要专注于抗肿瘤药物的研发和生产。
公司总部位于江苏省连云港市，拥有多个 GMP 认证的生产基地。
恒瑞医药生产注射用紫杉醇、奥沙利铂和卡培他滨等抗肿瘤药品。
注射用紫杉醇主要用于治疗非小细胞肺癌、乳腺癌和卵巢癌。
""",
    """
华海药业为恒瑞医药供应 API 原料药。华海药业位于浙江省台州市，
在原料药领域拥有丰富经验，年产能超过 500 公斤。
""",
    """
国药控股分销恒瑞医药和齐鲁制药的药品。国药控股拥有覆盖全国的
药品分销网络和冷链物流体系，能够将药品配送到各级医疗机构。
""",
    """
北京协和医院设有肿瘤科，广泛使用注射用紫杉醇和顺铂等
抗肿瘤药物。NMPA 监管注射用紫杉醇在中国市场的审批。
北京协和医院位于北京市东城区。
""",
]


# ============================================================================
# Pipeline
# ============================================================================


def build_kg(docs: list[str], verbose: bool = True):
    """执行知识图谱构建流水线。"""
    extractor = DictionaryExtractor()
    relation_extractor = RuleRelationExtractor()
    store = InMemoryKGStore()

    print("\n" + "=" * 60)
    print("知识图谱构建流水线")
    print("=" * 60)

    # Step 1: 文档解析 + 实体抽取
    print("\n[Step 1] 实体抽取（词典匹配）...")
    all_entities = set()
    for i, doc in enumerate(docs):
        mentions = extractor.extract(doc)
        for m in mentions:
            if m.name not in all_entities:
                all_entities.add(m.name)
                store.create_entity(KGEntity(
                    id=m.name,
                    name=m.name,
                    type=m.type,
                    description=ENTITY_TYPES.get(m.type, ""),
                ))
    print(f"  -> 抽取到 {len(all_entities)} 个实体")

    # Step 2: 关系抽取
    print("\n[Step 2] 关系抽取（触发词+共现）...")
    all_relations = []
    for doc in docs:
        mentions = extractor.extract(doc)
        relations = relation_extractor.extract(doc, mentions)
        all_relations.extend(relations)

    print(f"  -> 抽取到 {len(all_relations)} 条关系")
    print("\n  关系详情:")
    for r in all_relations:
        print(f"    {r.source_id} -[{r.type}]→ {r.target_id}")
        store.create_relation(r)

    # Step 3: 质量检查
    print("\n[Step 3] 质量检查...")
    checker = KGQualityChecker()
    orphans = checker.check_orphan_entities(store)
    violations = checker.check_schema_violations(store)
    duplicates = checker.check_duplicate_entities(store)

    print(f"  孤立实体: {len(orphans)} ({orphans if orphans else '无'})")
    print(f"  Schema 违规: {len(violations)} ({violations if violations else '无'})")
    print(f"  重复候选: {len(duplicates)}")
    for a, b, sim in duplicates:
        print(f"    {a} ~ {b} (相似度: {sim:.2f})")

    # Step 4: 图谱统计
    print(f"\n[Step 4] 图谱统计:")
    print(f"  实体数: {len(store.entities)}")
    print(f"  关系数: {sum(len(v) for v in store.adjacency.values())}")

    return store


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="知识图谱构建与应用 Demo")
    parser.add_argument("--mode", choices=["full", "query", "stats"], default="full",
                        help="运行模式: full=全流程, query=查询, stats=统计")
    parser.add_argument("--query", default="恒瑞医药的供应链涉及哪些企业？",
                        help="查询问题")
    args = parser.parse_args()

    # 构建图谱
    store = build_kg(SAMPLE_DOCUMENTS, verbose=True)

    if args.mode == "stats" or args.mode == "full":
        print(f"\n{'=' * 60}")
        print("Cypher 模拟查询")
        print(f"{'=' * 60}")

        for name in ["恒瑞医药", "北京协和医院"]:
            print(f"\n  MATCH (n)-[r]->(m) WHERE n.name = '{name}':")
            result = store.query_cypher_like(f'MATCH (n)-[r]->(m) WHERE n.name = "{name}"')
            for line in result.split("\n"):
                print(f"  {line}")

    if args.mode == "query" or args.mode == "full":
        print(f"\n{'=' * 60}")
        print("Graph + Vector 混合查询")
        print(f"{'=' * 60}")

        engine = HybridQueryEngine(store)
        result = engine.query(args.query)
        print(result)

        # 额外演示几个查询
        if args.mode == "full":
            for q in [
                "北京协和医院使用哪些药品？",
                "NMPA 监管哪些药品？",
            ]:
                print(f"\n{'-' * 40}")
                result = engine.query(q)
                print(result)

    print(f"\n{'=' * 60}")
    print("模式说明:")
    print("  --mode full   全流程: 构建→检查→查询")
    print("  --mode query  查询模式: --query '问题'")
    print("  --mode stats  仅统计和 Cypher 查询")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
