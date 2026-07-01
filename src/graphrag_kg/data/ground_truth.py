"""Ground truth data models, serialization, and evaluation.

Defines the canonical data structures for expected entities, relationships,
communities, and test queries. Also provides evaluation utilities to compare
extracted results against ground truth.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class Entity:
    """An expected entity in the ground truth."""

    name: str
    type: str  # organization, person, location, technology, event, etc.
    description_contains: list[str] = field(default_factory=list)
    mentioned_in_docs: list[str] = field(default_factory=list)
    related_to: list[str] = field(default_factory=list)
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class Relationship:
    """An expected relationship between two entities."""

    source: str
    target: str
    relation_type: str  # employs, acquires, supplies, partners_with, etc.
    description_contains: list[str] = field(default_factory=list)
    mentioned_in_docs: list[str] = field(default_factory=list)
    weight: float = 1.0
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class Community:
    """An expected community / cluster of entities."""

    id: int
    title: str
    level: int = 0
    entity_names: list[str] = field(default_factory=list)
    themes: list[str] = field(default_factory=list)
    parent_community_id: Optional[int] = None


@dataclass
class TestQuery:
    """A test query with expected answer constraints."""

    question: str
    search_method: str = "local"  # local, global, drift, basic, auto
    expected_answer_contains: list[str] = field(default_factory=list)
    expected_entities_in_response: list[str] = field(default_factory=list)
    expected_relationship_path: str = ""
    relevant_documents: list[str] = field(default_factory=list)
    hops_description: str = ""
    min_hop_count: int = 1


@dataclass
class GroundTruth:
    """Complete ground truth for a test scenario."""

    scenario: str
    seed: int
    document_count: int
    entities: list[Entity] = field(default_factory=list)
    relationships: list[Relationship] = field(default_factory=list)
    communities: list[Community] = field(default_factory=list)
    test_queries: list[TestQuery] = field(default_factory=list)
    document_files: list[str] = field(default_factory=list)
    entity_type_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "scenario": self.scenario,
            "seed": self.seed,
            "document_count": self.document_count,
            "expected_entities": [
                {
                    "name": e.name,
                    "type": e.type,
                    "description_contains": e.description_contains,
                    "mentioned_in_docs": e.mentioned_in_docs,
                    "related_to": e.related_to,
                    **e.properties,
                }
                for e in self.entities
            ],
            "expected_relationships": [
                {
                    "source": r.source,
                    "target": r.target,
                    "relation_type": r.relation_type,
                    "description_contains": r.description_contains,
                    "mentioned_in_docs": r.mentioned_in_docs,
                    "weight": r.weight,
                    **r.properties,
                }
                for r in self.relationships
            ],
            "expected_communities": [
                {
                    "id": c.id,
                    "title": c.title,
                    "level": c.level,
                    "entity_names": c.entity_names,
                    "themes": c.themes,
                    "parent_community_id": c.parent_community_id,
                }
                for c in self.communities
            ],
            "test_queries": [
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
                for q in self.test_queries
            ],
            "document_files": self.document_files,
            "entity_type_counts": self.entity_type_counts,
        }

    def to_json(self, path: Path) -> None:
        """Write ground truth to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, path: Path) -> "GroundTruth":
        """Load ground truth from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return cls(
            scenario=data["scenario"],
            seed=data["seed"],
            document_count=data["document_count"],
            entities=[
                Entity(
                    name=e["name"],
                    type=e["type"],
                    description_contains=e.get("description_contains", []),
                    mentioned_in_docs=e.get("mentioned_in_docs", []),
                    related_to=e.get("related_to", []),
                )
                for e in data["expected_entities"]
            ],
            relationships=[
                Relationship(
                    source=r["source"],
                    target=r["target"],
                    relation_type=r["relation_type"],
                    description_contains=r.get("description_contains", []),
                    mentioned_in_docs=r.get("mentioned_in_docs", []),
                    weight=r.get("weight", 1.0),
                )
                for r in data["expected_relationships"]
            ],
            communities=[
                Community(
                    id=c["id"],
                    title=c["title"],
                    level=c.get("level", 0),
                    entity_names=c.get("entity_names", []),
                    themes=c.get("themes", []),
                    parent_community_id=c.get("parent_community_id"),
                )
                for c in data["expected_communities"]
            ],
            test_queries=[
                TestQuery(
                    question=q["question"],
                    search_method=q.get("search_method", "local"),
                    expected_answer_contains=q.get("expected_answer_contains", []),
                    expected_entities_in_response=q.get("expected_entities_in_response", []),
                    expected_relationship_path=q.get("expected_relationship_path", ""),
                    relevant_documents=q.get("relevant_documents", []),
                    hops_description=q.get("hops_description", ""),
                    min_hop_count=q.get("min_hop_count", 1),
                )
                for q in data["test_queries"]
            ],
            document_files=data.get("document_files", []),
            entity_type_counts=data.get("entity_type_counts", {}),
        )

    def summary(self) -> str:
        """Return a human-readable summary of the ground truth."""
        lines = [
            f"Scenario: {self.scenario}",
            f"Seed: {self.seed}",
            f"Documents: {self.document_count}",
            f"Entities: {len(self.entities)}",
            f"Relationships: {len(self.relationships)}",
            f"Communities: {len(self.communities)}",
            f"Test Queries: {len(self.test_queries)}",
            "",
            "Entity types:",
        ]
        for etype, count in sorted(self.entity_type_counts.items()):
            lines.append(f"  {etype}: {count}")
        return "\n".join(lines)
