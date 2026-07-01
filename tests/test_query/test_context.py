"""Tests for query context, citations, and QueryResponse."""

from __future__ import annotations

from graphrag_kg.query.context import (
    ContextBuilder,
    GraphContext,
    QueryResponse,
    SourceCitation,
)


class TestSourceCitation:
    def test_create(self):
        c = SourceCitation(
            text_unit_id="tu1",
            text="Sample text content",
            document_name="doc.md",
            entity_name="Test Entity",
        )
        assert c.text_unit_id == "tu1"
        assert c.document_name == "doc.md"

    def test_to_markdown(self):
        c = SourceCitation(
            text_unit_id="tu1",
            text="Some content",
            document_name="test.md",
        )
        md = c.to_markdown()
        assert "test.md" in md
        assert "Some content" in md


class TestGraphContext:
    def test_empty(self):
        ctx = GraphContext()
        assert ctx.source_entity is None
        assert ctx.related_entities == []
        assert ctx.communities == []

    def test_to_context_text(self):
        ctx = GraphContext(
            source_entity={"name": "Test Corp", "type": "organization", "description": "A company"},
            related_entities=[{"name": "CEO", "type": "person"}],
            communities=[{"title": "Tech", "summary": "Technology sector"}],
        )
        text = ctx.to_context_text()
        assert "Test Corp" in text
        assert "Tech" in text
        assert "Technology sector" in text


class TestQueryResponse:
    def test_create(self):
        r = QueryResponse(
            answer="This is the answer.",
            search_method="local",
            processing_time_ms=123.4,
        )
        assert r.answer == "This is the answer."
        assert r.search_method == "local"
        assert r.processing_time_ms == 123.4

    def test_to_dict(self):
        r = QueryResponse(
            answer="Test answer",
            search_method="global",
            citations=[SourceCitation(text="source text", document_name="doc.txt")],
        )
        d = r.to_dict()
        assert d["answer"] == "Test answer"
        assert d["search_method"] == "global"
        assert len(d["citations"]) == 1

    def test_to_markdown(self):
        r = QueryResponse(
            answer="The answer.",
            search_method="local",
            citations=[SourceCitation(document_name="doc.md", text="some text")],
        )
        md = r.to_markdown()
        assert "The answer." in md
        assert "doc.md" in md
        assert "local" in md


class TestContextBuilder:
    def test_build_response(self):
        builder = ContextBuilder()
        response = builder.build_response(
            answer="Test answer",
            search_method="basic",
            sources=[
                {"id": "s1", "text": "Source text", "document_name": "doc.txt"},
            ],
            processing_time_ms=100.0,
        )
        assert response.answer == "Test answer"
        assert len(response.citations) == 1
        assert response.citations[0].text == "Source text"

    def test_build_response_dedup(self):
        """Should deduplicate sources by ID."""
        builder = ContextBuilder()
        response = builder.build_response(
            answer="Test",
            search_method="local",
            sources=[
                {"id": "s1", "text": "Text 1"},
                {"id": "s1", "text": "Text 1"},  # Duplicate
                {"id": "s2", "text": "Text 2"},
            ],
        )
        assert len(response.citations) == 2

    def test_build_graph_context(self):
        builder = ContextBuilder()
        ctx = builder.build_graph_context(
            source_entity={"name": "X"},
            related_entities=[{"name": "Y"}],
            communities=[{"title": "Community A"}],
            paths=[{"path": ["X", "Y"], "hops": 1}],
        )
        assert ctx.source_entity["name"] == "X"
        assert len(ctx.related_entities) == 1
        assert len(ctx.communities) == 1
        assert len(ctx.paths) == 1
