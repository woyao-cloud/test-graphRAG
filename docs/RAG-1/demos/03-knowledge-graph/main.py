"""
Demo 3: 知识图谱 RAG — Knowledge Graph Enhanced RAG
====================================================
Builds a sample knowledge graph in the pharma supply chain domain,
implements BFS traversal, and shows how graph context enriches LLM prompts.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Knowledge Graph Data Model
# ---------------------------------------------------------------------------

@dataclass
class Entity:
    id: str
    name: str
    type: str  # e.g., "drug", "company", "hospital", "regulator"
    properties: Dict[str, str] = field(default_factory=dict)

    def __hash__(self):
        return hash(self.id)

    def __repr__(self):
        return f"{self.type}:{self.name}"


@dataclass
class Relationship:
    source_id: str
    target_id: str
    relation: str  # e.g., "manufactures", "supplies", "regulates"
    properties: Dict[str, str] = field(default_factory=dict)


class KnowledgeGraph:
    """Simple in-memory knowledge graph."""

    def __init__(self):
        self.entities: Dict[str, Entity] = {}
        self.relationships: List[Relationship] = []
        # adjacency: entity_id -> list of (target_id, relation)
        self._adj: Dict[str, List[Tuple[str, str]]] = {}

    def add_entity(self, entity: Entity):
        self.entities[entity.id] = entity
        if entity.id not in self._adj:
            self._adj[entity.id] = []

    def add_relationship(self, rel: Relationship):
        self.relationships.append(rel)
        self._adj.setdefault(rel.source_id, []).append((rel.target_id, rel.relation))
        # add reverse for undirected traversal
        self._adj.setdefault(rel.target_id, []).append((rel.source_id, f"inverse_{rel.relation}"))

    def get_entity(self, eid: str) -> Optional[Entity]:
        return self.entities.get(eid)

    def get_neighbors(self, eid: str) -> List[Tuple[Entity, str]]:
        """Get all neighboring entities and relations."""
        result = []
        for neighbor_id, rel in self._adj.get(eid, []):
            entity = self.entities.get(neighbor_id)
            if entity:
                result.append((entity, rel))
        return result

    def find_shortest_path(self, source_id: str, target_id: str) -> Optional[List[str]]:
        """BFS shortest path between two entities."""
        if source_id not in self.entities or target_id not in self.entities:
            return None
        visited: Set[str] = set()
        queue: deque = deque()
        queue.append((source_id, [source_id]))
        visited.add(source_id)
        while queue:
            current, path = queue.popleft()
            if current == target_id:
                return path
            for neighbor_id, _ in self._adj.get(current, []):
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, path + [neighbor_id]))
        return None

    def format_graph_context(self, seed_id: str, depth: int = 2) -> str:
        """Format subgraph around an entity as a readable context block."""
        lines = []
        visited: Set[str] = set()
        queue: deque = deque()
        queue.append((seed_id, 0))
        visited.add(seed_id)

        while queue:
            eid, d = queue.popleft()
            if d > depth:
                continue
            entity = self.entities.get(eid)
            if not entity:
                continue
            indent = "  " * d
            lines.append(f"{indent}- {entity.name} ({entity.type})")
            for neighbor, rel in self.get_neighbors(eid):
                if neighbor.id not in visited:
                    visited.add(neighbor.id)
                    queue.append((neighbor.id, d + 1))
                    lines.append(f"{indent}  --[{rel}]--> {neighbor.name}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Build Pharma Supply Chain Knowledge Graph
# ---------------------------------------------------------------------------

def build_pharma_kg() -> KnowledgeGraph:
    kg = KnowledgeGraph()

    # Drugs
    kg.add_entity(Entity("d1", "Paracetamol", "drug", {"class": "analgesic"}))
    kg.add_entity(Entity("d2", "Ibuprofen", "drug", {"class": "NSAID"}))
    kg.add_entity(Entity("d3", "Amoxicillin", "drug", {"class": "antibiotic"}))
    kg.add_entity(Entity("d4", "Insulin", "drug", {"class": "hormone"}))

    # Companies
    kg.add_entity(Entity("c1", "PharmaCorp", "company", {"country": "USA"}))
    kg.add_entity(Entity("c2", "MediGen Labs", "company", {"country": "India"}))
    kg.add_entity(Entity("c3", "BioHealth Inc", "company", {"country": "Germany"}))

    # Hospitals
    kg.add_entity(Entity("h1", "City General Hospital", "hospital", {"beds": "500"}))
    kg.add_entity(Entity("h2", "St. Mary's Medical Center", "hospital", {"beds": "350"}))
    kg.add_entity(Entity("h3", "Rural Health Clinic", "hospital", {"beds": "50"}))

    # Regulators
    kg.add_entity(Entity("r1", "FDA", "regulator", {"region": "USA"}))
    kg.add_entity(Entity("r2", "EMA", "regulator", {"region": "Europe"}))

    # Relationships
    kg.add_relationship(Relationship("c1", "d1", "manufactures"))
    kg.add_relationship(Relationship("c1", "d2", "manufactures"))
    kg.add_relationship(Relationship("c2", "d3", "manufactures"))
    kg.add_relationship(Relationship("c3", "d4", "manufactures"))
    kg.add_relationship(Relationship("c1", "h1", "supplies"))
    kg.add_relationship(Relationship("c1", "h2", "supplies"))
    kg.add_relationship(Relationship("c2", "h3", "supplies"))
    kg.add_relationship(Relationship("c3", "h1", "supplies"))
    kg.add_relationship(Relationship("r1", "c1", "regulates"))
    kg.add_relationship(Relationship("r1", "d1", "approves"))
    kg.add_relationship(Relationship("r2", "c3", "regulates"))
    kg.add_relationship(Relationship("d1", "d2", "competes_with"))
    kg.add_relationship(Relationship("d3", "d4", "competes_with"))
    kg.add_relationship(Relationship("h1", "h2", "refers_to"))

    return kg


# ---------------------------------------------------------------------------
# Graph-Enhanced RAG
# ---------------------------------------------------------------------------

def graph_enhanced_rag(kg: KnowledgeGraph, query: str) -> str:
    """
    Given a natural language query, find the most relevant entity,
    extract its graph context, and format a prompt-ready context block.
    """
    # Simple keyword-to-entity mapping
    keyword_map = {
        "paracetamol": "d1", "ibuprofen": "d2", "amoxicillin": "d3", "insulin": "d4",
        "pharmacorp": "c1", "medigen": "c2", "biohealth": "c3",
        "city general": "h1", "st. mary": "h2", "rural health": "h3",
        "fda": "r1", "ema": "r2",
    }

    query_lower = query.lower()
    matched_id = None
    for keyword, eid in keyword_map.items():
        if keyword in query_lower:
            matched_id = eid
            break

    if not matched_id:
        return f"Could not find a relevant entity for query: {query}"

    entity = kg.get_entity(matched_id)
    context = kg.format_graph_context(matched_id, depth=2)

    prompt = f"""You are a supply chain analyst. Use the knowledge graph context below to answer the question.

=== Knowledge Graph Context (centered on {entity.name}) ===
{context}

=== Question ===
{query}

=== Instructions ===
Answer concisely, citing the entities and relationships from the graph.
"""
    return prompt


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Demo 3: 知识图谱 RAG — Knowledge Graph Enhanced RAG")
    print("=" * 60)

    kg = build_pharma_kg()
    print(f"\nBuilt knowledge graph: {len(kg.entities)} entities, {len(kg.relationships)} relationships")

    # --- Demo 1: Neighbors ---
    print("\n--- Get Neighbors: PharmaCorp (c1) ---")
    for neighbor, rel in kg.get_neighbors("c1"):
        print(f"  --[{rel}]--> {neighbor}")

    # --- Demo 2: Shortest Path ---
    print("\n--- Shortest Path: FDA (r1) -> City General Hospital (h1) ---")
    path = kg.find_shortest_path("r1", "h1")
    if path:
        path_names = " -> ".join(kg.get_entity(e).name for e in path)
        print(f"  Path: {path_names}")
    else:
        print("  No path found.")

    # --- Demo 3: Graph-Enhanced RAG ---
    print("\n--- Graph-Enhanced RAG Query ---")
    query = "How does FDA regulate PharmaCorp's supply chain?"
    prompt = graph_enhanced_rag(kg, query)
    print(prompt)

    print("\n" + "=" * 60)
    print("Graph context enriches RAG with structured entity relationships.")
    print("=" * 60)


if __name__ == "__main__":
    main()
