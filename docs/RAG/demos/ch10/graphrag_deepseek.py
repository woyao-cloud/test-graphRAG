"""
第 10 章 Demo：GraphRAG 深度实践

演示简化版 GraphRAG 流水线：
  文档分块 → 实体/关系抽取 → 知识图谱构建 → 社区检测 → 社区摘要 → 查询

可独立运行，无需外部依赖（内置模拟 LLM 和 Embedding）。

用法：
  python graphrag_deepseek.py
  python graphrag_deepseek.py --query "恒瑞医药生产哪些药品？"
  python graphrag_deepseek.py --query "紫杉醇的供应链是怎样的？" --mode drift
  python graphrag_deepseek.py --query "这个数据集主要讨论什么？" --mode global
"""

import argparse
import json
import math
import re
import time
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
    type: str  # organization / drug / chemical / hospital / regulator / person / geo / event
    description: str = ""
    text_unit_ids: list[str] = field(default_factory=list)


@dataclass
class Relation:
    source_id: str
    target_id: str
    type: str
    description: str = ""
    weight: float = 1.0


@dataclass
class TextUnit:
    id: str
    text: str
    document_id: str = ""
    token_count: int = 0


@dataclass
class Community:
    id: str
    level: int  # 0=fine, 1=medium, 2=coarse
    entity_ids: list[str] = field(default_factory=list)
    relation_ids: list[int] = field(default_factory=list)


@dataclass
class CommunityReport:
    community_id: str
    level: int
    title: str
    summary: str
    findings: list[str] = field(default_factory=list)


# ============================================================================
# Mock LLM (simulates LLM extraction and summarization)
# ============================================================================


class MockLLM:
    """模拟 LLM，实际运行时替换为真实的 DeepSeek/OpenAI 调用。"""

    def extract_entities_and_relations(self, text: str) -> tuple[list[dict], list[dict]]:
        """从文本中提取实体和关系（基于规则的模拟）。"""
        entities = []
        relations = []

        # 医药领域实体词典
        entity_patterns = {
            "恒瑞医药": ("organization", "制药企业，抗肿瘤药物主要生产商"),
            "华海药业": ("organization", "原料药和 API 供应商"),
            "国药控股": ("organization", "药品分销商"),
            "齐鲁制药": ("organization", "制药企业"),
            "瑞阳医药": ("organization", "药品分销商"),
            "北京协和医院": ("hospital", "三级甲等综合医院"),
            "上海中山医院": ("hospital", "三级甲等综合医院"),
            "北京大学肿瘤医院": ("hospital", "肿瘤专科医院"),
            "注射用紫杉醇": ("drug", "紫杉醇类抗肿瘤化疗药物"),
            "紫杉醇": ("chemical", "紫杉醇 API 原料药"),
            "奥沙利铂": ("drug", "铂类抗肿瘤化疗药物"),
            "卡培他滨": ("drug", "口服抗肿瘤化疗药物"),
            "顺铂": ("drug", "铂类抗肿瘤化疗药物"),
            "NMPA": ("regulator", "国家药品监督管理局"),
            "FDA": ("regulator", "美国食品药品监督管理局"),
            "肿瘤科": ("department", "肿瘤治疗科室"),
            "华海API": ("chemical", "华海药业生产的紫杉醇 API"),
        }

        # 关系模式
        relation_patterns = [
            ("恒瑞医药", "注射用紫杉醇", "生产", "恒瑞医药生产注射用紫杉醇"),
            ("恒瑞医药", "奥沙利铂", "生产", "恒瑞医药生产奥沙利铂"),
            ("恒瑞医药", "卡培他滨", "生产", "恒瑞医药生产卡培他滨"),
            ("齐鲁制药", "顺铂", "生产", "齐鲁制药生产顺铂"),
            ("齐鲁制药", "卡培他滨", "生产", "齐鲁制药生产卡培他滨"),
            ("华海药业", "华海API", "供应", "华海药业供应紫杉醇 API"),
            ("华海药业", "恒瑞医药", "供应API", "华海药业向恒瑞医药供应紫杉醇 API"),
            ("国药控股", "恒瑞医药", "分销", "国药控股分销恒瑞医药的药品"),
            ("国药控股", "齐鲁制药", "分销", "国药控股分销齐鲁制药的药品"),
            ("瑞阳医药", "齐鲁制药", "分销", "瑞阳医药分销齐鲁制药的药品"),
            ("北京协和医院", "注射用紫杉醇", "使用", "北京协和医院使用注射用紫杉醇"),
            ("北京协和医院", "奥沙利铂", "使用", "北京协和医院使用奥沙利铂"),
            ("北京协和医院", "顺铂", "使用", "北京协和医院使用顺铂"),
            ("上海中山医院", "注射用紫杉醇", "使用", "上海中山医院使用注射用紫杉醇"),
            ("北京大学肿瘤医院", "注射用紫杉醇", "使用", "北京大学肿瘤医院使用注射用紫杉醇"),
            ("北京大学肿瘤医院", "卡培他滨", "使用", "北京大学肿瘤医院使用卡培他滨"),
            ("北京协和医院", "肿瘤科", "设有", "北京协和医院设有肿瘤科"),
            ("NMPA", "注射用紫杉醇", "监管", "NMPA 监管注射用紫杉醇的审批"),
            ("FDA", "注射用紫杉醇", "监管", "FDA 监管注射用紫杉醇的美国市场"),
            ("紫杉醇", "华海API", "提纯", "紫杉醇经提纯制成华海API"),
        ]

        found_entities = set()
        for name, (etype, desc) in entity_patterns.items():
            if name in text:
                eid = name.lower().replace(" ", "_")
                entities.append({
                    "id": eid,
                    "name": name,
                    "type": etype,
                    "description": desc,
                })
                found_entities.add(name)

        for src, tgt, rtype, desc in relation_patterns:
            if src in text and tgt in text:
                relations.append({
                    "source_id": src.lower().replace(" ", "_"),
                    "target_id": tgt.lower().replace(" ", "_"),
                    "type": rtype,
                    "description": desc,
                })

        return entities, relations

    def summarize_community(
        self, community_id: str, entities: list[Entity], relations: list[Relation]
    ) -> CommunityReport:
        """模拟社区摘要生成。"""
        entity_names = [e.name for e in entities]
        entity_types = set(e.type for e in entities)
        type_label = "、".join(sorted(entity_types))

        # 根据实体类型生成摘要
        if "drug" in entity_types and "organization" in entity_types:
            theme = self._detect_theme(entities, relations)
            title = f"{theme}相关实体"
            findings = self._generate_findings(entities, relations, theme)
        elif "hospital" in entity_types:
            title = f"医疗机构生态"
            findings = [
                f"{e.name}使用多种抗肿瘤药物"
                for e in entities if e.type == "hospital"
            ]
        elif "regulator" in entity_types:
            title = f"监管与合规"
            findings = [
                f"{e.name}负责药品审批和市场监管"
                for e in entities if e.type == "regulator"
            ]
        else:
            title = f"实体集群 ({type_label})"
            findings = [f"包含 {len(entities)} 个相关实体"]

        summary = (
            f"该社区包含 {len(entities)} 个{type_label}实体和 "
            f"{len(relations)} 条关系。"
            f"涉及领域：{theme if 'theme' in dir() else title}。"
        )

        return CommunityReport(
            community_id=community_id,
            level=0,
            title=title,
            summary=summary,
            findings=findings,
        )

    def _detect_theme(self, entities: list[Entity], relations: list[Relation]) -> str:
        """检测社区主题。"""
        drugs = [e for e in entities if e.type == "drug" or e.type == "chemical"]
        orgs = [e for e in entities if e.type == "organization"]

        drug_names = [d.name for d in drugs]
        org_names = [o.name for o in orgs]

        if "恒瑞医药" in org_names:
            if "注射用紫杉醇" in drug_names or "紫杉醇" in drug_names:
                return "恒瑞医药-紫杉醇产业链"
            return "恒瑞医药产品线"
        if "齐鲁制药" in org_names:
            return "齐鲁制药产品线"
        if "国药控股" in org_names:
            return "药品分销网络"
        if "北京协和医院" in org_names:
            return "医疗机构用药"
        return "医药生态"

    def _generate_findings(
        self, entities: list[Entity], relations: list[Relation], theme: str
    ) -> list[str]:
        """生成关键发现。"""
        findings = []
        for r in relations:
            src_name = next(
                (e.name for e in entities if e.id == r.source_id),
                r.source_id,
            )
            tgt_name = next(
                (e.name for e in entities if e.id == r.target_id),
                r.target_id,
            )
            findings.append(f"{src_name} {r.type} {tgt_name}")
        return findings[:5]  # 最多 5 条


# ============================================================================
# Embedding (simple mock using character overlap)
# ============================================================================


class MockEmbedding:
    """模拟 Embedding 模型，基于字符重叠计算相似度。"""

    def embed(self, text: str) -> list[float]:
        """生成伪嵌入向量。"""
        # 使用文本的字符频率作为简单的特征向量
        features = [0.0] * 64
        for i, ch in enumerate(text):
            features[hash(ch) % 64] += 1.0
        # 归一化
        norm = math.sqrt(sum(v * v for v in features)) or 1.0
        return [v / norm for v in features]

    def similarity(self, a: list[float], b: list[float]) -> float:
        """余弦相似度。"""
        dot = sum(av * bv for av, bv in zip(a, b))
        return dot


# ============================================================================
# GraphRAG Indexing Pipeline
# ============================================================================


class GraphRAGIndexer:
    """简化版 GraphRAG 索引流水线。"""

    def __init__(self, llm: MockLLM, embedding: MockEmbedding):
        self.llm = llm
        self.embedding = embedding

        self.text_units: list[TextUnit] = []
        self.entities: dict[str, Entity] = {}
        self.relations: list[Relation] = []
        self.communities: list[Community] = []
        self.reports: list[CommunityReport] = []

        # 邻接表
        self.adjacency: dict[str, list[tuple[str, str]]] = defaultdict(list)

    def run(self, documents: list[str], verbose: bool = True):
        """执行完整索引流水线。"""
        if verbose:
            print("=" * 60)
            print("GraphRAG Indexing Pipeline")
            print("=" * 60)

        self._chunk_documents(documents, verbose)
        self._extract_entities_and_relations(verbose)
        self._build_graph(verbose)
        self._detect_communities(verbose)
        self._summarize_communities(verbose)

        if verbose:
            print("\n[OK] Indexing complete!")
            print(f"  Text Units: {len(self.text_units)}")
            print(f"  Entities: {len(self.entities)}")
            print(f"  Relations: {len(self.relations)}")
            print(f"  Communities: {len(self.communities)}")

    def _chunk_documents(self, documents: list[str], verbose: bool):
        """文档分块（简化版）。"""
        if verbose:
            print("\n[1/5] Chunking documents...")

        for doc_id, text in enumerate(documents):
            # 简单按句子分块
            sentences = re.split(r"[。！？\n]", text)
            chunk = ""
            chunk_id = 0
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                if len(chunk) + len(sentence) > 500:  # 每块约 500 字
                    tu = TextUnit(
                        id=f"tu_{doc_id}_{chunk_id}",
                        text=chunk.strip(),
                        document_id=f"doc_{doc_id}",
                        token_count=len(chunk) // 2,
                    )
                    self.text_units.append(tu)
                    chunk_id += 1
                    chunk = sentence + "。"
                else:
                    chunk += sentence + "。"
            if chunk.strip():
                self.text_units.append(
                    TextUnit(
                        id=f"tu_{doc_id}_{chunk_id}",
                        text=chunk.strip(),
                        document_id=f"doc_{doc_id}",
                        token_count=len(chunk) // 2,
                    )
                )
        if verbose:
            print(f"  -> {len(self.text_units)} text units created")

    def _extract_entities_and_relations(self, verbose: bool):
        """实体和关系抽取。"""
        if verbose:
            print("\n[2/5] Extracting entities and relations (with gleaning)...")

        all_entities: dict[str, Entity] = {}
        all_relations: list[Relation] = []

        for tu in self.text_units:
            entities_data, relations_data = self.llm.extract_entities_and_relations(
                tu.text
            )

            for ed in entities_data:
                if ed["id"] not in all_entities:
                    all_entities[ed["id"]] = Entity(
                        id=ed["id"],
                        name=ed["name"],
                        type=ed["type"],
                        description=ed["description"],
                        text_unit_ids=[],
                    )
                all_entities[ed["id"]].text_unit_ids.append(tu.id)

            for rd in relations_data:
                all_relations.append(
                    Relation(
                        source_id=rd["source_id"],
                        target_id=rd["target_id"],
                        type=rd["type"],
                        description=rd["description"],
                    )
                )

        self.entities = all_entities
        self.relations = all_relations

        if verbose:
            print(f"  -> {len(self.entities)} entities extracted")
            print(f"  -> {len(self.relations)} relations extracted")

    def _build_graph(self, verbose: bool):
        """构建知识图谱（邻接表）。"""
        if verbose:
            print("\n[3/5] Building knowledge graph...")

        for r in self.relations:
            self.adjacency[r.source_id].append((r.type, r.target_id))
            self.adjacency[r.target_id].append((r.type, r.source_id))

    def _detect_communities(self, verbose: bool):
        """社区检测（基于关系的连通分量 + 简单聚类）。"""
        if verbose:
            print("\n[4/5] Detecting communities (Leiden approximation)...")

        # 使用并查集找连通分量
        parent = {}

        def find(x):
            if x not in parent:
                parent[x] = x
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        for eid in self.entities:
            find(eid)

        for r in self.relations:
            union(r.source_id, r.target_id)

        # 按连通分量分组
        components: dict[str, list[str]] = defaultdict(list)
        for eid, _ in self.entities.items():
            root = find(eid)
            components[root].append(eid)

        # 每个连通分量作为一个基础社区
        for i, (root, entity_ids) in enumerate(components.items()):
            # Level 0: 细粒度社区
            community = Community(
                id=f"c_l0_{i}",
                level=0,
                entity_ids=entity_ids,
            )
            self.communities.append(community)

        # Level 1: 中等粒度（将小社区合并）
        if len(self.communities) > 3:
            mid_size = len(self.communities) // 3
            for i in range(3):
                start = i * mid_size
                end = start + mid_size if i < 2 else len(self.communities)
                merged_entities = []
                for c in self.communities[start:end]:
                    merged_entities.extend(c.entity_ids)
                self.communities.append(
                    Community(
                        id=f"c_l1_{i}",
                        level=1,
                        entity_ids=merged_entities,
                    )
                )

        # Level 2: 粗粒度（所有实体）
        all_entity_ids = list(self.entities.keys())
        self.communities.append(
            Community(
                id="c_l2_0",
                level=2,
                entity_ids=all_entity_ids,
            )
        )

        if verbose:
            levels = defaultdict(int)
            for c in self.communities:
                levels[c.level] += 1
            for level, count in sorted(levels.items()):
                print(f"  -> Level {level}: {count} communities")

    def _summarize_communities(self, verbose: bool):
        """为每个社区生成摘要。"""
        if verbose:
            print("\n[5/5] Summarizing communities...")

        for community in self.communities:
            if community.level > 0:
                continue  # 仅为 Level 0 生成报告

            entities = [
                self.entities[eid]
                for eid in community.entity_ids
                if eid in self.entities
            ]
            relations = [
                r
                for r in self.relations
                if r.source_id in community.entity_ids
                and r.target_id in community.entity_ids
            ]

            report = self.llm.summarize_community(
                community.id, entities, relations
            )
            self.reports.append(report)

        if verbose:
            print(f"  -> {len(self.reports)} community reports generated")


# ============================================================================
# Query Engine
# ============================================================================


class GraphRAGQueryEngine:
    """GraphRAG 查询引擎，支持 Local / Global / DRIFT 三种模式。"""

    def __init__(
        self,
        indexer: GraphRAGIndexer,
        llm: MockLLM,
        embedding: MockEmbedding,
    ):
        self.indexer = indexer
        self.llm = llm
        self.embedding = embedding

    def query(self, question: str, mode: str = "auto") -> str:
        """执行查询。"""
        mode = self._resolve_mode(question, mode)

        print(f"\n[Query Mode: {mode.upper()}]")
        print(f"Question: {question}\n")

        if mode == "local":
            return self._local_search(question)
        elif mode == "global":
            return self._global_search(question)
        elif mode == "drift":
            return self._drift_search(question)
        else:
            return self._basic_search(question)

    def _resolve_mode(self, question: str, preferred: str) -> str:
        """自动选择查询模式。"""
        if preferred != "auto":
            return preferred

        # 关键词启发式判断
        global_keywords = ["主题", "总结", "概述", "趋势", "主要", "整体", "总体"]
        drift_keywords = ["供应链", "路径", "流程", "链路", "影响", "关系", "怎么", "如何"]
        local_keywords = ["什么", "哪些", "谁", "哪个", "多少"]

        question_lower = question
        if any(kw in question_lower for kw in global_keywords):
            return "global"
        if any(kw in question_lower for kw in drift_keywords):
            return "drift"
        if any(kw in question_lower for kw in local_keywords):
            return "local"
        return "local"

    def _local_search(self, question: str) -> str:
        """Local Search：实体匹配 → 图扩展 → 生成答案。"""
        q_emb = self.embedding.embed(question)

        # 1. 语义匹配实体
        scored_entities = []
        for eid, entity in self.indexer.entities.items():
            e_emb = self.embedding.embed(entity.name + entity.description)
            score = self.embedding.similarity(q_emb, e_emb)
            scored_entities.append((score, eid, entity))

        scored_entities.sort(key=lambda x: -x[0])
        top_entities = scored_entities[:3]

        if not top_entities:
            return "未找到相关实体。"

        # 2. 图扩展：从匹配实体出发，收集 1-2 跳邻居信息
        context_lines = []
        visited = set()

        for score, eid, entity in top_entities:
            visited.add(eid)
            context_lines.append(
                f"实体: {entity.name} ({entity.type}) — {entity.description}"
            )

            # 1 跳邻居
            if eid in self.indexer.adjacency:
                for rel_type, neighbor_id in self.indexer.adjacency[eid]:
                    if neighbor_id in self.indexer.entities:
                        neighbor = self.indexer.entities[neighbor_id]
                        context_lines.append(
                            f"  -> [{rel_type}] {neighbor.name}"
                        )
                        visited.add(neighbor_id)

        # 3. 收集涉及的 TextUnit
        text_unit_texts = []
        for eid in visited:
            if eid in self.indexer.entities:
                for tu_id in self.indexer.entities[eid].text_unit_ids:
                    for tu in self.indexer.text_units:
                        if tu.id == tu_id:
                            text_unit_texts.append(tu.text[:200])

        # 4. 模拟 LLM 生成答案
        return self._generate_local_answer(
            question, top_entities, context_lines, text_unit_texts
        )

    def _generate_local_answer(
        self,
        question: str,
        top_entities: list,
        context_lines: list[str],
        text_units: list[str],
    ) -> str:
        """基于 Local Search 的上下文生成答案。"""
        # 提取关键实体名
        entity_names = [e.name for _, _, e in top_entities]

        # 检查关系
        relations_found = []
        for line in context_lines:
            if line.startswith("  ->"):
                relations_found.append(line.strip())

        answer_parts = []

        if "生产" in question or "哪些药品" in question or "什么药品" in question:
            drugs = []
            for _, _, e in top_entities:
                for r in self.indexer.relations:
                    if r.source_id == e.id and r.type == "生产":
                        target = self.indexer.entities.get(r.target_id)
                        if target:
                            drugs.append(target.name)
            if drugs:
                answer_parts.append(
                    f"根据知识图谱，{entity_names[0]}生产的药品包括：{'、'.join(drugs)}。"
                )

        elif "供应商" in question or "供应链" in question or "来源" in question:
            suppliers = []
            for _, _, e in top_entities:
                for r in self.indexer.relations:
                    if r.target_id == e.id and "供应" in r.type:
                        source = self.indexer.entities.get(r.source_id)
                        if source:
                            suppliers.append(f"{source.name}（{r.description}）")
            if suppliers:
                answer_parts.append(
                    f"{entity_names[0]}的供应商包括：{'；'.join(suppliers)}。"
                )

        if not answer_parts:
            # 通用回答
            answer_parts.append(
                f"在知识图谱中找到以下相关实体：{'、'.join(entity_names)}。"
            )
            if relations_found:
                answer_parts.append(
                    f"实体间关系：{'；'.join(relations_found[:5])}。"
                )

        return "\n".join(answer_parts)

    def _global_search(self, question: str) -> str:
        """Global Search：搜索社区报告 → Map-Reduce 生成答案。"""
        reports = self.indexer.reports
        if not reports:
            return "没有社区报告可用。"

        q_emb = self.embedding.embed(question)

        # 1. 语义匹配社区报告
        scored_reports = []
        for report in reports:
            r_emb = self.embedding.embed(report.summary + " " + " ".join(report.findings))
            score = self.embedding.similarity(q_emb, r_emb)
            scored_reports.append((score, report))

        scored_reports.sort(key=lambda x: -x[0])
        top_reports = scored_reports[:3]

        # 2. Map 阶段：每个社区报告生成部分答案
        partial_answers = []
        for score, report in top_reports:
            partial = (
                f"[社区: {report.title}]\n"
                f"{report.summary}\n"
                f"关键发现:\n"
                + "\n".join(f"  - {f}" for f in report.findings[:3])
            )
            partial_answers.append(partial)

        # 3. Reduce 阶段：汇总
        all_findings = []
        all_titles = []
        for _, report in top_reports:
            all_findings.extend(report.findings)
            all_titles.append(report.title)

        # 去重
        unique_findings = list(dict.fromkeys(all_findings))

        answer = (
            f"## 全局分析结果\n\n"
            f"该数据集共包含 {len(self.indexer.entities)} 个实体和 "
            f"{len(self.indexer.relations)} 条关系，"
            f"划分为 {len(self.indexer.communities)} 个社区。\n\n"
            f"### 主要主题\n\n"
        )

        for title in all_titles:
            answer += f"- **{title}**\n"

        answer += "\n### 关键发现\n\n"
        for f in unique_findings[:8]:
            answer += f"- {f}\n"

        return answer

    def _drift_search(self, question: str) -> str:
        """DRIFT Search：层次化社区遍历，追踪多跳关系路径。"""
        # 1. 找到最佳起点社区
        q_emb = self.embedding.embed(question)
        best_report = None
        best_score = -1
        for report in self.indexer.reports:
            r_emb = self.embedding.embed(report.summary)
            score = self.embedding.similarity(q_emb, r_emb)
            if score > best_score:
                best_score = score
                best_report = report

        if not best_report:
            return "未找到相关社区。"

        # 2. 从起点社区出发进行路径遍历
        # 找出该社区的所有 Level 0 社区
        community = None
        for c in self.indexer.communities:
            if c.id == best_report.community_id:
                community = c
                break

        if not community:
            return "社区信息不可用。"

        # 3. 在社区内执行 BFS 路径遍历
        entity_ids = community.entity_ids
        entity_names = {
            eid: self.indexer.entities[eid].name
            for eid in entity_ids
            if eid in self.indexer.entities
        }

        # 找到与问题相关的起点实体
        q_emb = self.embedding.embed(question)
        start_entity = None
        start_score = -1
        for eid, name in entity_names.items():
            e_emb = self.embedding.embed(name)
            score = self.embedding.similarity(q_emb, e_emb)
            if score > start_score:
                start_score = score
                start_entity = eid

        if not start_entity:
            return "未在社区中找到相关起点实体。"

        # 4. BFS 遍历路径
        paths = []
        visited = {start_entity}
        queue = [(start_entity, [start_entity])]

        while queue and len(paths) < 3:
            current, path = queue.pop(0)
            if len(path) > 5:  # 最大深度 5
                continue

            if current in self.indexer.adjacency:
                for rel_type, neighbor_id in self.indexer.adjacency[current]:
                    if neighbor_id not in visited and neighbor_id in entity_names:
                        visited.add(neighbor_id)
                        new_path = path + [neighbor_id]
                        if len(new_path) >= 2:
                            paths.append((new_path, rel_type))
                        queue.append((neighbor_id, new_path))

        # 5. 构建 DRIFT 答案
        if not paths:
            return f"在社区中发现实体「{entity_names.get(start_entity, start_entity)}」，但未找到完整路径。"

        answer_lines = [
            f"## DRIFT 层次化遍历结果\n",
            f"起点社区: {best_report.title}",
            f"起点实体: {entity_names.get(start_entity, start_entity)}\n",
            f"### 供应链/路径追踪\n",
        ]

        for path, last_rel in paths:
            path_str = " → ".join(
                entity_names.get(eid, eid) for eid in path
            )
            answer_lines.append(f"- {path_str}")

        # 补充说明
        all_path_entities = set()
        for path, _ in paths:
            all_path_entities.update(path)

        answer_lines.extend([
            "\n### 路径分析",
            f"共发现 {len(paths)} 条关联路径，涉及 {len(all_path_entities)} 个实体。",
        ])

        return "\n".join(answer_lines)

    def _basic_search(self, question: str) -> str:
        """Basic Search：纯文本单元搜索。"""
        q_emb = self.embedding.embed(question)

        scored_units = []
        for tu in self.indexer.text_units:
            t_emb = self.embedding.embed(tu.text[:200])
            score = self.embedding.similarity(q_emb, t_emb)
            scored_units.append((score, tu))

        scored_units.sort(key=lambda x: -x[0])
        top_units = scored_units[:2]

        if not top_units:
            return "未找到相关文本。"

        answer = "### Basic Search 结果\n\n"
        for score, tu in top_units:
            answer += f"[相关性: {score:.2f}] {tu.text[:300]}\n\n"

        return answer


# ============================================================================
# Sample Documents
# ============================================================================


SAMPLE_DOCUMENTS = [
    # doc_0
    """
恒瑞医药是中国领先的制药企业，主要专注于抗肿瘤药物的研发和生产。
公司总部位于江苏省连云港市，拥有多个 GMP 认证的生产基地。
恒瑞医药的主要产品包括注射用紫杉醇、奥沙利铂和卡培他滨，
这些药物广泛应用于各类癌症的化疗治疗。

华海药业是恒瑞医药的重要供应商，为其提供紫杉醇 API 原料药。
华海药业在原料药领域拥有丰富经验，其生产的紫杉醇 API
纯度达到 USP 标准，年产能超过 500 公斤。

国药控股是恒瑞医药在华东地区的独家分销商。
国药控股拥有完善的冷链物流体系，确保药品在运输过程中的质量。
通过国药控股的分销网络，恒瑞医药的药品能够覆盖华东地区
超过 200 家医院和医疗机构。

北京协和医院是恒瑞医药的重要客户，其肿瘤科广泛使用
恒瑞医药生产的注射用紫杉醇和奥沙利铂。
上海中山医院也在其肿瘤治疗方案中使用注射用紫杉醇。
北京大学肿瘤医院作为专科肿瘤医院，使用卡培他滨和注射用紫杉醇。
""",
    # doc_1
    """
齐鲁制药是中国另一家大型制药企业，总部位于山东省济南市。
齐鲁制药主要生产顺铂和卡培他滨等抗肿瘤药物。
顺铂是应用最广泛的铂类抗肿瘤药物之一，用于治疗
肺癌、卵巢癌、膀胱癌等多种实体瘤。

国药控股和瑞阳医药是齐鲁制药的主要分销合作伙伴。
瑞阳医药在华北地区拥有广泛的医院网络，
能够将齐鲁制药的药品快速配送到各级医疗机构。

北京协和医院同时使用齐鲁制药生产的顺铂和恒瑞医药生产的
注射用紫杉醇，这两类药物在其肿瘤综合治疗方案中
扮演着互补角色。
""",
    # doc_2
    """
注射用紫杉醇（Paclitaxel Injection）是一种微管抑制剂，
通过促进微管蛋白聚合、抑制微管解聚而发挥抗肿瘤作用。
它主要用于治疗非小细胞肺癌、乳腺癌和卵巢癌等多种实体瘤。

注射用紫杉醇的供应链涉及多个环节。首先，华海药业从天然
紫杉树皮中提取并纯化紫杉醇 API，然后供应给恒瑞医药等
制药企业进行制剂生产。恒瑞医药完成制剂生产后，
通过国药控股等分销商配送到医院终端。

NMPA（国家药品监督管理局）负责注射用紫杉醇在中国市场的
审批和监管。同时，该药物也获得了 FDA 的上市批准，
在美国市场销售。
""",
]


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="GraphRAG 深度实践 Demo"
    )
    parser.add_argument(
        "--query",
        default="恒瑞医药生产哪些药品？",
        help="查询问题",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "local", "global", "drift", "basic"],
        default="auto",
        help="查询模式（默认 auto 自动选择）",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("GraphRAG 深度实践 Demo")
    print("=" * 60)

    # 初始化组件
    llm = MockLLM()
    embedding = MockEmbedding()

    # 执行索引
    indexer = GraphRAGIndexer(llm, embedding)
    indexer.run(SAMPLE_DOCUMENTS, verbose=True)

    # 执行查询
    engine = GraphRAGQueryEngine(indexer, llm, embedding)
    answer = engine.query(args.query, mode=args.mode)

    print("\n" + "=" * 60)
    print("Answer:")
    print("=" * 60)
    print(answer)

    # 显示查询模式建议
    print("\n" + "-" * 40)
    print("查询模式推荐:")
    print("  --mode local  具体事实查询（如「哪些药品」「谁」）")
    print("  --mode global 全局总结查询（如「主要主题」「趋势」）")
    print("  --mode drift  多跳推理查询（如「供应链」「路径」）")
    print("  --mode basic  纯文本搜索（无图上下文）")
    print("  --mode auto   根据问题自动选择（默认）")


if __name__ == "__main__":
    main()
