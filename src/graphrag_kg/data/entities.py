"""Entity factory for generating consistent synthetic entities.

Uses Faker with fixed seeds to produce deterministic entity pools across
multiple entity types (organizations, persons, locations, drugs, etc.).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Optional

from faker import Faker

from graphrag_kg.data.ground_truth import Entity


@dataclass
class EntitySpec:
    """Specification for a type of entity to generate."""

    entity_type: str
    count: int
    prefix: str = ""  # e.g. "Pharma" for "Pharma Corp"
    suffix: str = ""  # e.g. " Hospital" for "X Hospital"
    name_pool: list[str] = field(default_factory=list)  # Explicit names (override faker)
    properties_template: dict[str, Any] = field(default_factory=dict)


class EntityFactory:
    """Generates deterministic synthetic entities for a scenario.

    Uses a fixed seed for reproducibility. Supports both Faker-generated
    names and explicit name pools for domain-specific entities.
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.faker = Faker("zh_CN")  # Chinese locale for realistic names
        self.faker.seed_instance(seed)
        self._name_counter: dict[str, int] = {}

    def generate_pool(self, specs: list[EntitySpec]) -> list[Entity]:
        """Generate a pool of entities from specifications.

        Args:
            specs: List of EntitySpec defining what entity types to generate.

        Returns:
            List of Entity objects with deterministic names and types.
        """
        entities: list[Entity] = []
        for spec in specs:
            for i in range(spec.count):
                entity = self._generate_one(spec, i)
                entities.append(entity)
        return entities

    def _generate_one(self, spec: EntitySpec, index: int) -> Entity:
        """Generate a single entity from a spec."""
        entity_type = spec.entity_type

        # Use explicit name pool if provided
        if index < len(spec.name_pool):
            name = spec.name_pool[index]
        else:
            name = self._generate_name(entity_type, spec)

        # Apply prefix/suffix
        if spec.prefix and not any(name.startswith(p) for p in [spec.prefix]):
            name = f"{spec.prefix}{name}"
        if spec.suffix and not name.endswith(spec.suffix):
            name = f"{name}{spec.suffix}"

        description = self._generate_description(entity_type, name)

        return Entity(
            name=name,
            type=entity_type,
            description_contains=description,
            properties=spec.properties_template.copy(),
        )

    def _generate_name(self, entity_type: str, spec: EntitySpec) -> str:
        """Generate a name based on entity type."""
        generators = {
            "organization": self.faker.company,
            "pharmaceutical_company": self._generate_pharma_name,
            "person": self.faker.name,
            "location": self.faker.city,
            "region": self._generate_region_name,
            "drug": self._generate_drug_name,
            "api_raw_material": self._generate_api_name,
            "distributor": self._generate_distributor_name,
            "hospital": self._generate_hospital_name,
            "clinical_department": self._generate_department_name,
            "indication": self._generate_indication_name,
            "regulatory_body": self._generate_regulatory_name,
            "drug_approval": self._generate_approval_number,
            "technology": self._generate_tech_name,
            "event": self._generate_event_name,
            "supply_contract": self._generate_contract_name,
        }

        generator = generators.get(entity_type, self.faker.word)
        return generator().title() if hasattr(generator, "title") else str(generator())

    def _generate_pharma_name(self) -> str:
        """Generate a Chinese pharmaceutical company name."""
        prefixes = ["恒瑞", "齐鲁", "石药", "正大天晴", "复星", "华润", "扬子江", "科伦"]
        suffixes = ["医药", "制药", "药业", "生物"]
        if self.rng.random() < 0.7:
            return self.rng.choice(prefixes) + self.rng.choice(suffixes)
        return self.faker.company()

    def _generate_drug_name(self) -> str:
        """Generate a realistic drug name."""
        brand_names = [
            "注射用紫杉醇", "奥希替尼片", "贝伐珠单抗注射液",
            "重组人促红素注射液", "吉非替尼片", "厄洛替尼片",
            "克唑替尼胶囊", "来那度胺胶囊", "硼替佐米粉针",
            "伊马替尼片", "舒尼替尼胶囊", "帕博利珠单抗注射液",
            "纳武利尤单抗注射液", "曲妥珠单抗粉针", "利妥昔单抗注射液",
            "甲磺酸阿帕替尼片", "卡瑞利珠单抗粉针", "安罗替尼胶囊",
            "培美曲塞二钠粉针", "多西他赛注射液",
        ]
        generic_prefixes = ["注射用", "片", "胶囊", "注射液", "粉针"]
        if self.rng.random() < 0.8:
            return self.rng.choice(brand_names)
        return f"Test-Drug-{self.rng.randint(1000, 9999)}"

    def _generate_api_name(self) -> str:
        """Generate an API (Active Pharmaceutical Ingredient) name."""
        apis = [
            "紫杉醇API", "吉非替尼中间体", "奥希替尼游离碱",
            "贝伐珠单抗原液", "来那度胺原料药", "硼替佐米粗品",
            "伊马替尼碱基", "培美曲塞二钠", "多西他赛半合成品",
            "卡瑞利珠单抗原液",
        ]
        return self.rng.choice(apis)

    def _generate_distributor_name(self) -> str:
        """Generate a pharmaceutical distributor name."""
        names = ["国药控股", "华润医药", "上药控股", "九州通医药", "广州医药", "南京医药"]
        return self.rng.choice(names)

    def _generate_hospital_name(self) -> str:
        """Generate a Chinese hospital name."""
        cities = ["北京", "上海", "广州", "成都", "武汉", "杭州", "南京", "天津"]
        names = ["协和医院", "瑞金医院", "中山医院", "华山医院", "华西医院", "同济医院"]
        if self.rng.random() < 0.7:
            suffix = self.rng.choice(
                ["大学附属第一医院", "大学肿瘤医院", "大学人民医院", "市中心医院"]
            )
            return self.rng.choice(cities) + suffix
        return self.rng.choice(cities) + self.rng.choice(names)

    def _generate_department_name(self) -> str:
        """Generate a clinical department name."""
        depts = [
            "肿瘤内科", "血液科", "呼吸与危重症医学科",
            "心血管内科", "神经外科", "消化内科",
            "妇瘤科", "胸外科", "放疗科",
            "泌尿外科", "乳腺外科", "骨科",
        ]
        return self.rng.choice(depts)

    def _generate_indication_name(self) -> str:
        """Generate a disease indication name."""
        indications = [
            "非小细胞肺癌", "转移性乳腺癌", "复发胶质母细胞瘤",
            "结直肠癌", "胃腺癌", "肝细胞癌",
            "肾细胞癌", "卵巢癌", "胰腺癌",
            "弥漫大B细胞淋巴瘤", "多发性骨髓瘤",
            "慢性粒细胞白血病", "前列腺癌",
            "黑色素瘤", "甲状腺癌",
        ]
        return self.rng.choice(indications)

    def _generate_region_name(self) -> str:
        """Generate a Chinese region name."""
        regions = ["华东区", "华南区", "华北区", "华中区", "西南区", "西北区", "东北区"]
        return self.rng.choice(regions)

    def _generate_regulatory_name(self) -> str:
        """Generate a regulatory body name."""
        return self.rng.choice(["NMPA国家药品监督管理局", "FDA美国食品药品监督管理局"])

    def _generate_approval_number(self) -> str:
        """Generate a drug approval number."""
        year = self.rng.randint(2018, 2025)
        seq = self.rng.randint(1, 99999)
        return f"NMPA-{year}-{seq:05d}"

    def _generate_tech_name(self) -> str:
        """Generate a technology name."""
        prefixes = ["Project", "Platform", "Engine", "System"]
        names = ["Titanium", "Quantum", "Fusion", "Apex", "Nova", "Phoenix", "Horizon"]
        return f"{self.rng.choice(prefixes)} {self.rng.choice(names)}"

    def _generate_event_name(self) -> str:
        """Generate a business event name."""
        events = [
            "Merger", "Acquisition", "IPO", "Partnership", "Spin-off",
            "Clinical Trial Result", "FDA Approval", "Product Launch",
            "Patent Expiration", "Market Entry",
        ]
        year = self.rng.randint(2020, 2026)
        return f"{year} {self.rng.choice(events)}"

    def _generate_contract_name(self) -> str:
        """Generate a supply/distribution contract name."""
        years = f"{self.rng.randint(2023, 2026)}-{self.rng.randint(2027, 2030)}"
        return f"供应合同-{years}"

    def _generate_description(self, entity_type: str, name: str) -> list[str]:
        """Generate description keywords for an entity."""
        templates: dict[str, list[list[str]]] = {
            "pharmaceutical_company": [
                ["创新药", "研发", "生产", "销售"],
                ["仿制药", "生物药", "化学药"],
                ["肿瘤", "自身免疫", "抗感染"],
            ],
            "drug": [
                ["处方药", "抗肿瘤", "靶向治疗"],
                ["注射剂", "口服制剂"],
                ["医保目录", "国家集采"],
            ],
            "api_raw_material": [
                ["原料药", "中间体", "GMP认证"],
                ["化学合成", "发酵工艺"],
            ],
            "distributor": [
                ["医药流通", "供应链", "冷链物流"],
                ["医院配送", "药房配送"],
            ],
            "hospital": [
                ["三甲医院", "教学医院", "医保定点"],
                ["肿瘤中心", "临床研究基地"],
            ],
            "clinical_department": [
                ["临床科室", "住院病房", "门诊"],
                ["化疗", "靶向治疗", "免疫治疗"],
            ],
            "person": [
                ["主任医师", "教授", "博士生导师"],
                ["采购总监", "供应链管理"],
            ],
            "indication": [
                ["恶性肿瘤", "靶向治疗适应症"],
                ["发病率", "五年生存率"],
            ],
            "regulatory_body": [
                ["药品审评", "注册审批", "GMP检查"],
            ],
            "drug_approval": [
                ["药品注册批件", "上市许可"],
                ["化药", "生物制品"],
            ],
            "region": [
                ["区域", "分销网络", "市场覆盖"],
            ],
            "organization": [
                ["企业", "公司", "集团"],
            ],
            "technology": [
                ["技术平台", "研发"],
            ],
            "event": [
                ["商业事件", "行业动态"],
            ],
            "supply_contract": [
                ["供应协议", "采购框架"],
            ],
        }

        options = templates.get(entity_type, [["实体", "数据"]])
        chosen = self.rng.choice(options)
        return chosen.copy()


class EntityRelationshipBuilder:
    """Builds consistent relationships between entities in a pool."""

    def __init__(self, entities: list[Entity], seed: int = 42):
        self.entities = entities
        self.by_type: dict[str, list[Entity]] = {}
        self.by_name: dict[str, Entity] = {}
        for e in entities:
            self.by_type.setdefault(e.type, []).append(e)
            self.by_name[e.name] = e
        self.rng = random.Random(seed)

    def get_by_type(self, entity_type: str) -> list[Entity]:
        """Get all entities of a given type."""
        return self.by_type.get(entity_type, [])

    def get_by_name(self, name: str) -> Optional[Entity]:
        """Get entity by name."""
        return self.by_name.get(name)

    def sample(self, entity_type: str, k: int = 1) -> list[Entity]:
        """Sample k entities of a given type."""
        pool = self.get_by_type(entity_type)
        if len(pool) <= k:
            return pool.copy()
        return self.rng.sample(pool, k)
