"""Response context builder for grounded Q&A answers.

Extracts source citations, formats graph context, and assembles
structured QueryResponse objects with answer + citations + graph data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class SourceCitation:
    """A source citation for a claim in the answer."""

    text_unit_id: str = ""
    text: str = ""
    document_name: str = ""
    entity_name: str = ""
    relationship: str = ""

    def to_markdown(self) -> str:
        """Format citation as markdown."""
        parts = []
        if self.document_name:
            parts.append(f"📄 {self.document_name}")
        if self.text:
            excerpt = self.text[:150] + "..." if len(self.text) > 150 else self.text
            parts.append(f"> {excerpt}")
        return "\n".join(parts)


@dataclass
class GraphContext:
    """Graph traversal context for a query answer."""

    source_entity: Optional[dict[str, Any]] = None
    related_entities: list[dict[str, Any]] = field(default_factory=list)
    communities: list[dict[str, Any]] = field(default_factory=list)
    paths: list[dict[str, Any]] = field(default_factory=list)
    ego_network: Optional[dict[str, Any]] = None

    def to_context_text(self) -> str:
        """Format graph context as text for LLM prompt."""
        lines = ["--- Graph Context ---"]

        if self.source_entity:
            lines.append(f"\nEntity: {self.source_entity.get('name', '')}")
            lines.append(f"Type: {self.source_entity.get('type', '')}")
            lines.append(f"Description: {self.source_entity.get('description', '')}")

        if self.related_entities:
            lines.append(f"\nRelated Entities ({len(self.related_entities)}):")
            for e in self.related_entities[:10]:
                lines.append(f"  - {e.get('name', '?')} ({e.get('type', '?')})")

        if self.communities:
            lines.append(f"\nCommunities ({len(self.communities)}):")
            for c in self.communities[:5]:
                title = c.get("title", "?")
                summary = c.get("summary", "")
                lines.append(f"  - {title}: {summary[:200]}")

        if self.paths:
            lines.append(f"\nRelevant Paths ({len(self.paths)}):")
            for p in self.paths[:5]:
                path_str = " → ".join(str(n) for n in p.get("path", []))
                lines.append(f"  - {path_str}")

        return "\n".join(lines)


@dataclass
class QueryResponse:
    """Complete response from a GraphRAG query.

    Attributes:
        answer: The natural language answer text.
        search_method: Method used (local, global, drift, basic).
        citations: Source citations supporting the answer.
        graph_context: Graph traversal data used.
        processing_time_ms: Total processing time in milliseconds.
        llm_usage: Token usage info if available.
    """

    answer: str
    search_method: str = "local"
    citations: list[SourceCitation] = field(default_factory=list)
    graph_context: Optional[GraphContext] = None
    processing_time_ms: float = 0.0
    llm_usage: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-compatible dict."""
        return {
            "answer": self.answer,
            "search_method": self.search_method,
            "citations": [
                {
                    "text_unit_id": c.text_unit_id,
                    "text": c.text,
                    "document_name": c.document_name,
                    "entity_name": c.entity_name,
                    "relationship": c.relationship,
                }
                for c in self.citations
            ],
            "graph_context": self.graph_context.to_context_text() if self.graph_context else None,
            "processing_time_ms": self.processing_time_ms,
            "llm_usage": self.llm_usage,
        }

    def to_markdown(self) -> str:
        """Format response as markdown with citations."""
        lines = [self.answer, ""]

        if self.citations:
            lines.append("---")
            lines.append(f"*Search method: {self.search_method}*")
            lines.append(f"*Sources: {len(self.citations)}*")
            lines.append("")
            for i, c in enumerate(self.citations[:5], 1):
                lines.append(f"**Source {i}:** {c.to_markdown()}")
                lines.append("")

        if self.processing_time_ms > 0:
            lines.append(f"*Processing time: {self.processing_time_ms:.0f}ms*")

        return "\n".join(lines)


class ContextBuilder:
    """Builds QueryResponse objects from search results.

    Extracts source citations, formats graph context, and assembles
    grounded answers with evidence chains.
    """

    def build_response(
        self,
        answer: str,
        search_method: str,
        sources: Optional[list[dict[str, Any]]] = None,
        graph_context: Optional[GraphContext] = None,
        processing_time_ms: float = 0.0,
        llm_usage: Optional[dict[str, int]] = None,
    ) -> QueryResponse:
        """Build a complete QueryResponse from search results.

        Args:
            answer: The LLM-generated answer text.
            search_method: Which search method produced this.
            sources: Raw source data from the search engine.
            graph_context: Optional graph traversal context.
            processing_time_ms: Time spent processing.
            llm_usage: Token usage data.

        Returns:
            A QueryResponse with extracted citations and context.
        """
        citations = self._extract_citations(sources or [])

        return QueryResponse(
            answer=answer,
            search_method=search_method,
            citations=citations,
            graph_context=graph_context,
            processing_time_ms=processing_time_ms,
            llm_usage=llm_usage or {},
        )

    def _extract_citations(
        self, sources: list[dict[str, Any]]
    ) -> list[SourceCitation]:
        """Extract SourceCitation objects from raw source data.

        Handles various source formats from different search methods.
        """
        citations: list[SourceCitation] = []
        seen_ids: set[str] = set()

        for src in sources:
            text_unit_id = str(src.get("id", src.get("text_unit_id", "")))
            if text_unit_id in seen_ids:
                continue
            seen_ids.add(text_unit_id)

            citation = SourceCitation(
                text_unit_id=text_unit_id,
                text=str(src.get("text", src.get("chunk", ""))),
                document_name=str(src.get("document_name", src.get("source", ""))),
                entity_name=str(src.get("entity", src.get("entity_name", ""))),
                relationship=str(src.get("relationship", "")),
            )
            citations.append(citation)

        return citations

    def build_graph_context(
        self,
        source_entity: Optional[dict[str, Any]] = None,
        related_entities: Optional[list[dict[str, Any]]] = None,
        communities: Optional[list[dict[str, Any]]] = None,
        paths: Optional[list[dict[str, Any]]] = None,
    ) -> GraphContext:
        """Build a GraphContext from search traversal results."""
        return GraphContext(
            source_entity=source_entity,
            related_entities=related_entities or [],
            communities=communities or [],
            paths=paths or [],
        )
