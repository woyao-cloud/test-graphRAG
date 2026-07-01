"""Relationship factory for generating cross-document entity relationships.

Builds structured, multi-hop relationship networks between entities in
a test scenario. Supports supply chain, organizational, and competitive
relationship types.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable

from graphrag_kg.data.entities import EntityRelationshipBuilder
from graphrag_kg.data.ground_truth import Entity, Relationship


@dataclass
class RelationshipRule:
    """A rule defining how to generate relationships between entity types."""

    source_type: str
    target_type: str
    relation_type: str
    description_template: str  # e.g. "{source} employs {target} as CEO since {year}"
    min_per_source: int = 1
    max_per_source: int = 3
    bidirectional: bool = False
    weight_range: tuple[float, float] = (0.5, 1.0)
    # Optional filter: only create relationship if condition is met
    condition: Callable[[Entity, Entity], bool] | None = None


class RelationshipFactory:
    """Generates deterministic relationships from a set of rules.

    Uses EntityRelationshipBuilder to locate entities by type and applies
    relationship rules to build a consistent graph.
    """

    def __init__(self, builder: EntityRelationshipBuilder, seed: int = 42):
        self.builder = builder
        self.rng = random.Random(seed)
        self._relationship_counter: dict[str, int] = {}

    def generate(self, rules: list[RelationshipRule]) -> list[Relationship]:
        """Generate all relationships based on rules.

        Args:
            rules: List of RelationshipRule defining relationship patterns.

        Returns:
            List of Relationship objects.
        """
        relationships: list[Relationship] = []
        for rule in rules:
            generated = self._apply_rule(rule)
            relationships.extend(generated)
        return relationships

    def _apply_rule(self, rule: RelationshipRule) -> list[Relationship]:
        """Apply a single relationship rule."""
        sources = self.builder.get_by_type(rule.source_type)
        targets = self.builder.get_by_type(rule.target_type)

        if not sources or not targets:
            return []

        results: list[Relationship] = []
        for source in sources:
            count = self.rng.randint(rule.min_per_source, rule.max_per_source)
            # Select targets (may include same target for 1-to-many relations)
            selected = self.rng.choices(
                targets,
                k=min(count, len(targets)),
            )

            for target in selected:
                # Skip self-relationships
                if source.name == target.name:
                    continue
                # Apply condition filter if present
                if rule.condition and not rule.condition(source, target):
                    continue

                desc = rule.description_template.format(
                    source=source.name, target=target.name
                )
                weight = round(
                    self.rng.uniform(*rule.weight_range), 2
                )

                relationship = Relationship(
                    source=source.name,
                    target=target.name,
                    relation_type=rule.relation_type,
                    description_contains=[desc] + source.description_contains[:1],
                    weight=weight,
                )
                results.append(relationship)

                # Add bidirectional if specified
                if rule.bidirectional:
                    rev_desc = rule.description_template.format(
                        source=target.name, target=source.name
                    )
                    results.append(
                        Relationship(
                            source=target.name,
                            target=source.name,
                            relation_type=rule.relation_type,
                            description_contains=[rev_desc],
                            weight=weight,
                        )
                    )

        return results

    def generate_pharma_supply_chain_rules(self) -> list[RelationshipRule]:
        """Return the standard rules for the pharma_supply_chain scenario."""
        return [
            # === Regulatory Layer ===
            RelationshipRule(
                source_type="regulatory_body",
                target_type="drug_approval",
                relation_type="approves",
                description_template="{source} 批准了 {target}",
                min_per_source=2,
                max_per_source=4,
            ),
            RelationshipRule(
                source_type="pharmaceutical_company",
                target_type="drug_approval",
                relation_type="holds",
                description_template="{source} 持有 {target}",
                min_per_source=1,
                max_per_source=2,
            ),
            # === Production Layer ===
            RelationshipRule(
                source_type="api_raw_material",
                target_type="pharmaceutical_company",
                relation_type="supplies",
                description_template="{source} 供应给 {target} 用于药品生产",
                min_per_source=1,
                max_per_source=3,
            ),
            RelationshipRule(
                source_type="pharmaceutical_company",
                target_type="drug",
                relation_type="produces",
                description_template="{source} 生产 {target}",
                min_per_source=1,
                max_per_source=4,
            ),
            RelationshipRule(
                source_type="pharmaceutical_company",
                target_type="supply_contract",
                relation_type="signs",
                description_template="{source} 签署了 {target}",
                min_per_source=1,
                max_per_source=2,
            ),
            RelationshipRule(
                source_type="api_raw_material",
                target_type="supply_contract",
                relation_type="signs",
                description_template="{source} 签署了 {target}",
                min_per_source=0,
                max_per_source=2,
            ),
            # === Distribution Layer ===
            RelationshipRule(
                source_type="pharmaceutical_company",
                target_type="distributor",
                relation_type="authorizes",
                description_template="{source} 授权 {target} 为其药品分销商",
                min_per_source=1,
                max_per_source=3,
            ),
            RelationshipRule(
                source_type="distributor",
                target_type="hospital",
                relation_type="delivers_to",
                description_template="{source} 向 {target} 配送药品",
                min_per_source=1,
                max_per_source=4,
            ),
            RelationshipRule(
                source_type="distributor",
                target_type="region",
                relation_type="covers",
                description_template="{source} 覆盖 {target} 的药品分销网络",
                min_per_source=1,
                max_per_source=3,
            ),
            RelationshipRule(
                source_type="pharmaceutical_company",
                target_type="distributor",
                relation_type="signs_distribution_contract",
                description_template="{source} 与 {target} 签订了分销合同",
                min_per_source=0,
                max_per_source=2,
            ),
            # === Clinical Usage Layer ===
            RelationshipRule(
                source_type="hospital",
                target_type="drug",
                relation_type="stocks",
                description_template="{source} 药房库存 {target}",
                min_per_source=1,
                max_per_source=5,
            ),
            RelationshipRule(
                source_type="clinical_department",
                target_type="drug",
                relation_type="prescribes",
                description_template="{source} 为患者开具 {target} 处方",
                min_per_source=1,
                max_per_source=4,
            ),
            RelationshipRule(
                source_type="drug",
                target_type="indication",
                relation_type="treats",
                description_template="{source} 用于治疗 {target}",
                min_per_source=1,
                max_per_source=3,
            ),
            RelationshipRule(
                source_type="hospital",
                target_type="clinical_department",
                relation_type="has_department",
                description_template="{source} 设有 {target}",
                min_per_source=1,
                max_per_source=3,
            ),
            # === Personnel ===
            RelationshipRule(
                source_type="person",
                target_type="clinical_department",
                relation_type="works_at",
                description_template="{source} 在 {target} 任职",
                min_per_source=1,
                max_per_source=1,
            ),
            RelationshipRule(
                source_type="person",
                target_type="hospital",
                relation_type="works_at",
                description_template="{source} 任职于 {target}",
                min_per_source=0,
                max_per_source=1,
            ),
            RelationshipRule(
                source_type="person",
                target_type="distributor",
                relation_type="works_at",
                description_template="{source} 任职于 {target} 采购部",
                min_per_source=0,
                max_per_source=1,
            ),
        ]

    def generate_tech_company_rules(self) -> list[RelationshipRule]:
        """Return standard rules for the tech_company scenario."""
        return [
            RelationshipRule(
                source_type="organization",
                target_type="person",
                relation_type="employs",
                description_template="{source} employs {target}",
                min_per_source=1,
                max_per_source=3,
            ),
            RelationshipRule(
                source_type="organization",
                target_type="technology",
                relation_type="develops",
                description_template="{source} develops {target}",
                min_per_source=1,
                max_per_source=2,
            ),
            RelationshipRule(
                source_type="organization",
                target_type="location",
                relation_type="headquartered_in",
                description_template="{source} is headquartered in {target}",
                min_per_source=1,
                max_per_source=1,
            ),
            RelationshipRule(
                source_type="organization",
                target_type="organization",
                relation_type="competes_with",
                description_template="{source} competes with {target}",
                min_per_source=1,
                max_per_source=2,
            ),
            RelationshipRule(
                source_type="organization",
                target_type="event",
                relation_type="involved_in",
                description_template="{source} was involved in {target}",
                min_per_source=0,
                max_per_source=1,
            ),
            RelationshipRule(
                source_type="person",
                target_type="technology",
                relation_type="leads_development",
                description_template="{source} leads the development of {target}",
                min_per_source=0,
                max_per_source=1,
            ),
        ]
