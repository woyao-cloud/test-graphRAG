"""Query factory for generating test questions from ground truth.

Given a set of known entities and relationships, generates natural-language
questions that require single-hop or multi-hop graph traversal to answer.
"""

from __future__ import annotations

import random
from typing import Any

from graphrag_kg.data.entities import EntityRelationshipBuilder
from graphrag_kg.data.ground_truth import Entity, Relationship, TestQuery


class QueryFactory:
    """Generates test queries based on known entity/relationship structure.

    Produces questions at varying difficulty levels:
    - Level 1: Direct fact lookup (single entity)
    - Level 2: Single-hop relationship (A -> B)
    - Level 3: Multi-hop traversal (A -> B -> C)
    - Level 4: Set intersection (find entities satisfying multiple criteria)
    - Level 5: Path tracing (describe the full chain from A to Z)
    """

    def __init__(self, builder: EntityRelationshipBuilder, seed: int = 42):
        self.builder = builder
        self.rng = random.Random(seed)

    def generate_pharma_queries(self) -> list[TestQuery]:
        """Generate pharma supply chain test queries."""
        queries: list[TestQuery] = []

        # Build lookup maps
        pharmas = self.builder.get_by_type("pharmaceutical_company")
        drugs = self.builder.get_by_type("drug")
        hospitals = self.builder.get_by_type("hospital")
        depts = self.builder.get_by_type("clinical_department")
        distributors = self.builder.get_by_type("distributor")
        indications = self.builder.get_by_type("indication")
        apis = self.builder.get_by_type("api_raw_material")
        regions = self.builder.get_by_type("region")

        if not all([pharmas, drugs, hospitals, distributors, indications]):
            return queries

        # --- Level 1: Direct fact lookup ---
        if drugs:
            drug = drugs[0]
            queries.append(TestQuery(
                question=f"{drug.name}主要用于治疗哪些疾病？",
                search_method="local",
                expected_answer_contains=drug.description_contains[:2],
                expected_entities_in_response=[drug.name],
                relevant_documents=["manufacturer_catalog.md", "clinical_usage_report.md"],
                min_hop_count=1,
            ))

        # --- Level 2: Single-hop ---
        if pharmas and drugs:
            pharma = pharmas[0]
            queries.append(TestQuery(
                question=f"{pharma.name}生产哪些主要药品？",
                search_method="local",
                expected_entities_in_response=[pharma.name],
                expected_relationship_path="制药公司-[生产]->药品",
                relevant_documents=["manufacturer_catalog.md"],
                min_hop_count=1,
            ))

        # --- Level 3: Multi-hop ---
        if hospitals and drugs and pharmas and indications:
            hospital = hospitals[0]
            queries.append(TestQuery(
                question=f"{hospital.name}使用哪些制药公司生产的抗肿瘤药物？",
                search_method="local",
                hops_description=f"{hospital.name} → 采购 → 药品 → 生产商 → 适应症(肿瘤)",
                expected_entities_in_response=[hospital.name],
                expected_relationship_path="医院-[采购]-药品-[生产]-制药公司-[治疗]-适应症",
                relevant_documents=["hospital_procurement.txt", "manufacturer_catalog.md",
                                   "clinical_usage_report.md"],
                min_hop_count=3,
            ))

        # --- Level 4: Supply chain trace ---
        if apis and drugs and pharmas and distributors and hospitals:
            api = apis[0]
            queries.append(TestQuery(
                question=f"{api.name}从原料药到临床使用的完整供应链是怎样的？",
                search_method="drift",
                hops_description="原料药 → 生产商 → 药品 → 分销商 → 医院 → 科室 → 适应症",
                expected_entities_in_response=[api.name],
                expected_relationship_path="原料-[供应]-药企-[生产]-药品-[配送]-分销商-[配发]-医院-[开具]-科室",
                relevant_documents=["supply_chain_overview.txt"],
                min_hop_count=4,
            ))

        # --- Level 5: Distributor portfolio ---
        if distributors and regions and drugs and pharmas:
            dist = distributors[0]
            region = regions[0] if regions else None
            queries.append(TestQuery(
                question=f"{dist.name}在{region.name if region else '华东区域'}分销哪些制药公司的哪些药品？"
                         f"这些药品分别用于治疗什么疾病？",
                search_method="global",
                hops_description="分销商 → 区域 → 分销合同 → 药品 → 生产商 → 适应症",
                expected_entities_in_response=[dist.name],
                expected_relationship_path="分销商-[覆盖]-区域-[签订]-合同-[涉及]-药品-[治疗]-适应症",
                relevant_documents=["distribution_contract.html", "manufacturer_catalog.md",
                                   "supply_chain_overview.txt"],
                min_hop_count=4,
            ))

        # --- Level 6: Set intersection ---
        if len(pharmas) >= 2:
            p1, p2 = pharmas[0], pharmas[1]
            queries.append(TestQuery(
                question=f"哪些医院同时使用{p1.name}和{p2.name}的药品？",
                search_method="local",
                hops_description=f"制药公司 → 药品 → 采购订单 → 医院（交集）",
                expected_entities_in_response=[p1.name, p2.name],
                expected_relationship_path="药企-[生产]-药品-[采购]-医院 ← 需要查交集",
                relevant_documents=["hospital_procurement.txt", "manufacturer_catalog.md"],
                min_hop_count=2,
            ))

        # --- Level 7: Impact analysis ---
        if drugs and hospitals and depts and indications:
            drug = drugs[0]
            queries.append(TestQuery(
                question=f"如果{drug.name}供应中断，会影响哪些医院的哪些科室的治疗？",
                search_method="drift",
                hops_description="药品 → 库存 → 医院 → 科室 → 适应症",
                expected_entities_in_response=[drug.name],
                expected_relationship_path="药品-[库存]-医院-[配发]-科室-[治疗]-适应症",
                relevant_documents=["hospital_procurement.txt", "clinical_usage_report.md",
                                   "quality_inspection.html"],
                min_hop_count=3,
            ))

        # --- Level 8: Regulatory compliance ---
        if pharmas and drugs:
            pharma = pharmas[0]
            drug = drugs[0]
            queries.append(TestQuery(
                question=f"{pharma.name}的{drug.name}获得了哪些监管机构的审批？审批编号是什么？",
                search_method="local",
                hops_description="制药公司 → 药品批文 → 监管机构",
                expected_entities_in_response=[pharma.name, drug.name],
                expected_relationship_path="药企-[持有]-批文-[批准]-监管机构",
                relevant_documents=["regulatory_approval.pdf", "manufacturer_catalog.md"],
                min_hop_count=2,
            ))

        return queries

    def generate_tech_queries(self) -> list[TestQuery]:
        """Generate tech company test queries."""
        queries: list[TestQuery] = []

        orgs = self.builder.get_by_type("organization")
        people = self.builder.get_by_type("person")
        techs = self.builder.get_by_type("technology")

        if not all([orgs, people]):
            return queries

        # Level 1: Direct fact
        if orgs and people:
            org = orgs[0]
            person = people[0]
            queries.append(TestQuery(
                question=f"Who is the CEO of {org.name} and what is their background?",
                search_method="local",
                expected_answer_contains=org.description_contains[:2],
                expected_entities_in_response=[org.name],
                relevant_documents=["company_profiles.md", "executive_bios.md"],
                min_hop_count=1,
            ))

        # Level 2: Relationship
        if orgs and techs:
            org = orgs[0]
            tech = techs[0]
            queries.append(TestQuery(
                question=f"What technology does {org.name} develop?",
                search_method="local",
                expected_entities_in_response=[org.name],
                expected_relationship_path="公司-[开发]->技术",
                relevant_documents=["company_profiles.md", "product_launch.txt"],
                min_hop_count=1,
            ))

        # Level 3: Competition
        if len(orgs) >= 2:
            o1, o2 = orgs[0], orgs[1]
            queries.append(TestQuery(
                question=f"How does {o1.name} compete with {o2.name}?",
                search_method="local",
                expected_entities_in_response=[o1.name, o2.name],
                expected_relationship_path="公司-[竞争]->公司",
                relevant_documents=["industry_report.pdf"],
                min_hop_count=1,
            ))

        # Level 4: Multi-hop
        if orgs and people and techs:
            org = orgs[0]
            queries.append(TestQuery(
                question=f"What projects are {org.name}'s key people leading?",
                search_method="drift",
                expected_entities_in_response=[org.name],
                expected_relationship_path="公司-[雇佣]->人-[领导开发]->技术",
                relevant_documents=["company_profiles.md", "executive_bios.md"],
                min_hop_count=2,
            ))

        # Level 5: Event impact
        if orgs:
            org = orgs[0]
            queries.append(TestQuery(
                question=f"What major events has {org.name} been involved in?",
                search_method="global",
                expected_entities_in_response=[org.name],
                expected_relationship_path="公司-[参与]->事件",
                relevant_documents=["industry_report.pdf", "partnership_news.html"],
                min_hop_count=1,
            ))

        return queries
