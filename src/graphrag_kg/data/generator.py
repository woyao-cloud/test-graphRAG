"""Main test data generator orchestrating entity, relationship, document,
and query generation for a complete scenario.

Usage:
    generator = TestDataGenerator(scenario="pharma_supply_chain", seed=42)
    gt = generator.generate(output_dir)
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Optional

from jinja2 import Environment, FileSystemLoader, Template

from graphrag_kg.core.errors import DataGenerationError
from graphrag_kg.data.entities import (
    EntityFactory,
    EntityRelationshipBuilder,
    EntitySpec,
)
from graphrag_kg.data.ground_truth import (
    Community,
    Entity,
    GroundTruth,
    Relationship,
    TestQuery,
)
from graphrag_kg.data.queries import QueryFactory
from graphrag_kg.data.relationships import RelationshipFactory, RelationshipRule


# ============================================================================
# Scenario Definitions
# ============================================================================

PHARMA_SUPPLY_CHAIN_SPECS: list[EntitySpec] = [
    EntitySpec(
        entity_type="pharmaceutical_company",
        count=5,
        name_pool=["恒瑞医药", "齐鲁制药", "石药集团", "正大天晴", "复星医药"],
        properties_template={"sector": "pharma", "region": "China"},
    ),
    EntitySpec(
        entity_type="drug",
        count=8,
        name_pool=[
            "注射用紫杉醇", "奥希替尼片", "贝伐珠单抗注射液",
            "重组人促红素注射液", "吉非替尼片", "来那度胺胶囊",
            "卡瑞利珠单抗粉针", "安罗替尼胶囊",
        ],
        properties_template={"category": "prescription", "drug_type": "antineoplastic"},
    ),
    EntitySpec(
        entity_type="api_raw_material",
        count=5,
        name_pool=["紫杉醇API", "吉非替尼中间体", "奥希替尼游离碱",
                   "来那度胺原料药", "卡瑞利珠单抗原液"],
        properties_template={"grade": "pharmaceutical"},
    ),
    EntitySpec(
        entity_type="distributor",
        count=4,
        name_pool=["国药控股", "华润医药", "上药控股", "九州通医药"],
        properties_template={"coverage": "national"},
    ),
    EntitySpec(
        entity_type="hospital",
        count=6,
        name_pool=[
            "北京协和医院", "上海瑞金医院", "中山大学肿瘤医院",
            "四川大学华西医院", "复旦大学附属肿瘤医院", "浙江大学附属第一医院",
        ],
        properties_template={"level": "三甲", "type": "teaching"},
    ),
    EntitySpec(
        entity_type="clinical_department",
        count=6,
        name_pool=["肿瘤内科", "血液科", "呼吸与危重症医学科",
                   "心血管内科", "神经外科", "消化内科"],
        properties_template={"setting": "inpatient"},
    ),
    EntitySpec(
        entity_type="region",
        count=4,
        name_pool=["华东区", "华南区", "华北区", "西南区"],
        properties_template={},
    ),
    EntitySpec(
        entity_type="regulatory_body",
        count=2,
        name_pool=["NMPA国家药品监督管理局", "FDA美国食品药品监督管理局"],
        properties_template={},
    ),
    EntitySpec(
        entity_type="person",
        count=5,
        name_pool=["张明华", "李建国", "王丽萍", "陈志强", "刘芳"],
        properties_template={"role": "professional"},
    ),
    EntitySpec(
        entity_type="drug_approval",
        count=4,
        name_pool=["NMPA-2024-00892", "NMPA-2023-01567", "NMPA-2022-03142", "NMPA-2024-00123"],
        properties_template={},
    ),
    EntitySpec(
        entity_type="indication",
        count=6,
        name_pool=["非小细胞肺癌", "转移性乳腺癌", "结直肠癌",
                   "复发胶质母细胞瘤", "胃腺癌", "肝细胞癌"],
        properties_template={"disease_area": "oncology"},
    ),
    EntitySpec(
        entity_type="supply_contract",
        count=4,
        name_pool=["供应合同-2024-2027", "供应合同-2023-2026",
                   "供应合同-2025-2028", "供应合同-2024-2029"],
        properties_template={},
    ),
]

TECH_COMPANY_SPECS: list[EntitySpec] = [
    EntitySpec(
        entity_type="organization",
        count=5,
        name_pool=["Acme Corp", "Nexus Dynamics", "QuantumLeap AI",
                   "CloudPeak Systems", "DataForge Inc"],
        properties_template={"sector": "technology"},
    ),
    EntitySpec(
        entity_type="person",
        count=5,
        name_pool=["Satya Nadella", "Dr. Elena Voss", "James Chen",
                   "Maria Rodriguez", "Alex Park"],
        properties_template={"role": "executive"},
    ),
    EntitySpec(
        entity_type="technology",
        count=4,
        name_pool=["Project Titanium", "Quantum Engine", "Fusion Platform", "Apex AI"],
        properties_template={"stage": "development"},
    ),
    EntitySpec(
        entity_type="location",
        count=4,
        name_pool=["Seattle", "Silicon Valley", "Austin", "Boston"],
        properties_template={"country": "USA"},
    ),
    EntitySpec(
        entity_type="event",
        count=3,
        name_pool=["2025 Acquisition", "2024 IPO", "2026 Partnership"],
        properties_template={},
    ),
]


# ============================================================================
# Main Generator
# ============================================================================


class TestDataGenerator:
    """Generates complete test data scenarios with ground truth.

    Attributes:
        scenario: The scenario name (e.g. "pharma_supply_chain", "tech_company").
        seed: Random seed for deterministic generation.
        entity_count: Target number of entities (may be overridden by scenario specs).
        doc_count: Number of documents to generate.
        output_formats: List of output formats (pdf, md, txt, html).
    """

    SCENARIO_SPECS = {
        "pharma_supply_chain": PHARMA_SUPPLY_CHAIN_SPECS,
        "tech_company": TECH_COMPANY_SPECS,
    }

    SCENARIO_RELATIONSHIP_RULES = {
        "pharma_supply_chain": "generate_pharma_supply_chain_rules",
        "tech_company": "generate_tech_company_rules",
    }

    SCENARIO_QUERY_GENERATORS = {
        "pharma_supply_chain": "generate_pharma_queries",
        "tech_company": "generate_tech_queries",
    }

    TEMPLATE_DIR = Path(__file__).parent / "templates"

    def __init__(
        self,
        scenario: str = "pharma_supply_chain",
        seed: int = 42,
        entity_count: int = 0,
        doc_count: int = 0,
        output_formats: Optional[list[str]] = None,
    ):
        if scenario not in self.SCENARIO_SPECS:
            raise DataGenerationError(
                f"Unknown scenario '{scenario}'. "
                f"Available: {list(self.SCENARIO_SPECS.keys())}"
            )

        self.scenario = scenario
        self.seed = seed
        self.rng = random.Random(seed)
        self.output_formats = output_formats or ["md", "txt", "html", "pdf"]

        # Entity factory
        self.entity_factory = EntityFactory(seed=seed)

        # Entity specs (use scenario defaults, optionally override count)
        self.specs = self.SCENARIO_SPECS[scenario]
        self.doc_count = doc_count or len(self._get_template_names())

        # Jinja2 environment
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(self.TEMPLATE_DIR / scenario)),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def _get_template_names(self) -> list[str]:
        """Get the list of Jinja2 template files for this scenario."""
        template_dir = self.TEMPLATE_DIR / self.scenario
        if not template_dir.exists():
            return [f"doc_{i}.md.j2" for i in range(1, 6)]
        return sorted(
            f.name for f in template_dir.glob("*.j2")
        )

    def generate(self, output_dir: Path) -> GroundTruth:
        """Run the complete test data generation pipeline.

        Args:
            output_dir: Directory to write generated documents and ground truth.

        Returns:
            GroundTruth object containing all expected entities, relationships,
            communities, and test queries.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Generate entity pool
        entities = self.entity_factory.generate_pool(self.specs)

        # Step 2: Build entity relationship index
        builder = EntityRelationshipBuilder(entities, seed=self.seed)

        # Step 3: Generate relationships
        rel_factory = RelationshipFactory(builder, seed=self.seed)
        rules_method = getattr(rel_factory, self.SCENARIO_RELATIONSHIP_RULES[self.scenario])
        relationships = rel_factory.generate(rules_method())

        # Step 4: Assign communities
        communities = self._assign_communities(entities, relationships)

        # Step 5: Generate test queries
        query_factory = QueryFactory(builder, seed=self.seed)
        query_method = getattr(query_factory, self.SCENARIO_QUERY_GENERATORS[self.scenario])
        test_queries = query_method()

        # Step 6: Render documents from templates
        doc_dir = output_dir / "documents"
        doc_dir.mkdir(parents=True, exist_ok=True)
        document_files = self._render_documents(entities, relationships, communities, doc_dir)

        # Step 7: Count entity types
        entity_type_counts: dict[str, int] = {}
        for e in entities:
            entity_type_counts[e.type] = entity_type_counts.get(e.type, 0) + 1

        # Step 8: Build ground truth
        ground_truth = GroundTruth(
            scenario=self.scenario,
            seed=self.seed,
            document_count=len(document_files),
            entities=entities,
            relationships=relationships,
            communities=communities,
            test_queries=test_queries,
            document_files=document_files,
            entity_type_counts=entity_type_counts,
        )

        # Step 9: Write ground truth JSON
        ground_truth.to_json(output_dir / "ground_truth.json")

        # Step 10: Write queries JSON
        self._write_queries_json(test_queries, output_dir / "queries.json")

        # Step 11: Write scenario README
        self._write_readme(ground_truth, output_dir / "README.md")

        return ground_truth

    def _assign_communities(
        self, entities: list[Entity], relationships: list[Relationship]
    ) -> list[Community]:
        """Assign entities to communities based on relationship clusters.

        Uses a simple clustering approach: entities that are connected via
        relationships are grouped into communities.
        """
        # Build adjacency list
        adj: dict[str, set[str]] = {e.name: set() for e in entities}
        for r in relationships:
            adj.setdefault(r.source, set()).add(r.target)
            adj.setdefault(r.target, set()).add(r.source)

        # Simple BFS-based community detection
        visited: set[str] = set()
        communities: list[Community] = []
        community_id = 0

        for entity in entities:
            if entity.name in visited:
                continue

            # BFS to find connected component
            component: list[str] = []
            queue = [entity.name]
            while queue:
                node = queue.pop(0)
                if node in visited:
                    continue
                visited.add(node)
                component.append(node)
                for neighbor in adj.get(node, set()):
                    if neighbor not in visited:
                        queue.append(neighbor)

            if component:
                # Generate a community title based on entity types in the component
                comp_entities = [e for e in entities if e.name in component]
                types_in_comp = list(set(e.type for e in comp_entities))
                title = self._generate_community_title(comp_entities, types_in_comp)
                themes = self._extract_themes(comp_entities)

                communities.append(Community(
                    id=community_id,
                    title=title,
                    level=0,
                    entity_names=component,
                    themes=themes,
                ))
                community_id += 1

        return communities

    def _generate_community_title(
        self, entities: list[Entity], types: list[str]
    ) -> str:
        """Generate a descriptive title for a community."""
        scenario_titles = {
            "pharma_supply_chain": [
                "药品生产与供应链", "肿瘤治疗与临床用药",
                "分销网络与区域覆盖", "药品监管与审批",
                "医院采购与临床应用",
            ],
            "tech_company": [
                "Cloud Computing Ecosystem", "AI/ML Innovation",
                "Strategic Partnerships", "Product Development",
                "Market Competition",
            ],
        }
        titles = scenario_titles.get(self.scenario, [f"Community {len(types)}"])
        idx = len(types) - 1 if len(types) <= len(titles) else self.rng.randint(0, len(titles) - 1)
        return titles[min(idx, len(titles) - 1)]

    def _extract_themes(self, entities: list[Entity]) -> list[str]:
        """Extract thematic keywords from entity descriptions."""
        all_keywords: list[str] = []
        for e in entities:
            all_keywords.extend(e.description_contains)
        # Return top unique keywords
        seen: set[str] = set()
        unique = []
        for kw in all_keywords:
            if kw not in seen:
                seen.add(kw)
                unique.append(kw)
        return unique[:5]

    def _render_documents(
        self,
        entities: list[Entity],
        relationships: list[Relationship],
        communities: list[Community],
        output_dir: Path,
    ) -> list[str]:
        """Render all Jinja2 templates into document files.

        Returns list of relative file paths for generated documents.
        """
        template_names = self._get_template_names()
        document_files: list[str] = []

        # Build context for templates
        context = self._build_template_context(entities, relationships, communities)

        for i, template_name in enumerate(template_names):
            if not template_name.endswith(".j2"):
                continue

            # Get base name without .j2 extension
            base_name = template_name[:-3]  # remove .j2

            try:
                template = self.jinja_env.get_template(template_name)
                content = template.render(**context, doc_index=i)

                # Write in all requested formats
                for fmt in self.output_formats:
                    ext = self._format_to_extension(fmt)
                    file_name = f"{base_name}.{ext}"
                    file_path = output_dir / file_name

                    if fmt == "html" and not base_name.endswith(".html"):
                        file_path = output_dir / f"{base_name}.html"

                    self._write_document(content, file_path, fmt)
                    if file_path.exists():
                        document_files.append(f"documents/{file_path.name}")

            except Exception as e:
                raise DataGenerationError(
                    f"Failed to render template '{template_name}': {e}"
                ) from e

        return document_files

    def _build_template_context(
        self,
        entities: list[Entity],
        relationships: list[Relationship],
        communities: list[Community],
    ) -> dict[str, Any]:
        """Build the Jinja2 template context with entity/relationship data."""
        # Group entities by type
        by_type: dict[str, list[Entity]] = {}
        for e in entities:
            by_type.setdefault(e.type, []).append(e)

        # Build relationship index by source
        rels_by_source: dict[str, list[Relationship]] = {}
        for r in relationships:
            rels_by_source.setdefault(r.source, []).append(r)

        # Generate some fake dates and numbers for realism
        dates = [
            f"{self.rng.randint(2020, 2025)}年{self.rng.randint(1,12)}月{self.rng.randint(1,28)}日"
            for _ in range(20)
        ]
        amounts = [
            f"¥{self.rng.randint(100, 50000)}万" for _ in range(10)
        ]

        return {
            "entities": entities,
            "relationships": relationships,
            "communities": communities,
            "by_type": by_type,
            "rels_by_source": rels_by_source,
            "dates": dates,
            "amounts": amounts,
            "scenario": self.scenario,
            "seed": self.seed,
            "rng": self.rng,
        }

    def _write_document(self, content: str, path: Path, fmt: str) -> None:
        """Write content in the specified format."""
        if fmt == "md" or path.suffix == ".md":
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
        elif fmt == "txt":
            # Strip markdown formatting for plain text
            text = self._strip_markdown(content)
            txt_path = path.with_suffix(".txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)
        elif fmt == "html":
            html = self._md_to_html(content)
            html_path = path.with_suffix(".html")
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html)
        elif fmt == "pdf":
            # Write markdown first, then convert
            md_path = path.with_suffix(".md")
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(content)
            try:
                self._md_to_pdf(content, path.with_suffix(".pdf"))
            except Exception:
                pass  # PDF conversion is best-effort

    def _strip_markdown(self, content: str) -> str:
        """Basic markdown-to-plain-text conversion."""
        import re
        # Remove headers
        content = re.sub(r'^#{1,6}\s+', '', content, flags=re.MULTILINE)
        # Remove bold/italic
        content = re.sub(r'\*{1,3}([^*]+)\*{1,3}', r'\1', content)
        # Remove links
        content = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', content)
        # Remove code blocks
        content = re.sub(r'```[^`]*```', '', content, flags=re.DOTALL)
        # Remove inline code
        content = re.sub(r'`([^`]+)`', r'\1', content)
        return content

    def _md_to_html(self, content: str) -> str:
        """Convert markdown to HTML using the markdown library."""
        try:
            import markdown
            return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="utf-8"><title>Document</title></head>
<body>
{markdown.markdown(content, extensions=['tables', 'fenced_code'])}
</body>
</html>"""
        except ImportError:
            return f"<pre>{content}</pre>"

    def _md_to_pdf(self, content: str, path: Path) -> None:
        """Convert markdown to PDF (best-effort)."""
        try:
            import markdown
            html = self._md_to_html(content)

            # Try weasyprint first, then fall back
            try:
                from weasyprint import HTML
                HTML(string=html).write_pdf(str(path))
            except ImportError:
                # Try pdfkit
                try:
                    import pdfkit
                    pdfkit.from_string(html, str(path))
                except ImportError:
                    pass  # PDF not available
        except Exception:
            pass

    def _write_queries_json(self, queries: list[TestQuery], path: Path) -> None:
        """Write test queries to a JSON file."""
        data = [
            {
                "question": q.question,
                "search_method": q.search_method,
                "expected_answer_contains": q.expected_answer_contains,
                "expected_entities_in_response": q.expected_entities_in_response,
                "expected_relationship_path": q.expected_relationship_path,
                "relevant_documents": q.relevant_documents,
                "hops_description": q.hops_description,
                "min_hop_count": q.min_hop_count,
            }
            for q in queries
        ]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _write_readme(self, gt: GroundTruth, path: Path) -> None:
        """Write a scenario README."""
        content = f"""# {gt.scenario} - Test Scenario

- **Seed**: {gt.seed}
- **Documents**: {gt.document_count}
- **Entities**: {len(gt.entities)}
- **Relationships**: {len(gt.relationships)}
- **Communities**: {len(gt.communities)}
- **Test Queries**: {len(gt.test_queries)}

## Entity Types
"""
        for etype, count in sorted(gt.entity_type_counts.items()):
            content += f"- {etype}: {count}\n"

        content += "\n## Test Queries\n\n"
        for i, q in enumerate(gt.test_queries, 1):
            content += f"{i}. **{q.question}**\n"
            if q.hops_description:
                content += f"   - Hops: {q.hops_description}\n"
            content += f"   - Method: {q.search_method}\n"

        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    @staticmethod
    def list_scenarios() -> list[str]:
        """List available scenario names."""
        return list(TestDataGenerator.SCENARIO_SPECS.keys())

    def _format_to_extension(self, fmt: str) -> str:
        """Map format name to file extension."""
        mapping = {"md": "md", "txt": "txt", "html": "html", "pdf": "pdf"}
        return mapping.get(fmt, fmt)
