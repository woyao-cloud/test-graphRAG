"""Tests for DocumentConverter."""

from __future__ import annotations

from pathlib import Path

import pytest

from graphrag_kg.ingest.converter import DocumentConverter
from graphrag_kg.ingest.parsers import Document


class TestDocumentConverter:
    """Tests for DocumentConverter."""

    @pytest.fixture
    def converter(self, temp_dir):
        input_dir = temp_dir / "input"
        input_dir.mkdir()
        return DocumentConverter(input_dir=input_dir)

    def test_convert_single(self, converter, temp_dir):
        """Should convert a Document to a graphrag input file."""
        path = temp_dir / "source.md"
        path.write_text("# Test Doc\n\nSome content.", encoding="utf-8")

        doc = Document(
            source_path=path,
            text="# Test Doc\n\nSome content.",
            title="Test Doc",
            format="md",
        )

        output_path = converter.convert(doc)
        assert output_path.exists()
        assert output_path.suffix == ".txt"
        assert output_path.parent == converter.input_dir

        content = output_path.read_text(encoding="utf-8")
        assert "Test Doc" in content
        assert "Some content" in content

    def test_convert_strips_markdown(self, converter, temp_dir):
        """Should strip markdown when strip_markdown=True."""
        converter.strip_markdown = True
        path = temp_dir / "md_source.md"
        path.write_text("# Title\n\n**bold** text with [link](http://x.com)", encoding="utf-8")

        doc = Document(
            source_path=path,
            text="# Title\n\n**bold** text with [link](http://x.com)",
            title="Title",
            format="md",
        )

        output_path = converter.convert(doc)
        content = output_path.read_text(encoding="utf-8")
        assert "**bold**" not in content
        assert "bold" in content
        assert "[link]" not in content

    def test_convert_prepends_metadata(self, converter, temp_dir):
        """Should prepend metadata header when prepend_metadata=True."""
        converter.prepend_metadata = True
        path = temp_dir / "meta_source.md"
        path.write_text("# Doc\n\nBody.", encoding="utf-8")

        doc = Document(
            source_path=path,
            text="# Doc\n\nBody.",
            title="Doc",
            format="md",
            metadata={"author": "Test Author", "pages": 5},
        )

        output_path = converter.convert(doc)
        content = output_path.read_text(encoding="utf-8")
        assert "# Document: Doc" in content
        assert "# Source: meta_source.md" in content
        assert "# Author: Test Author" in content
        assert "# Pages: 5" in content

    def test_convert_no_metadata(self, converter, temp_dir):
        """Should not prepend metadata when disabled."""
        converter.prepend_metadata = False
        path = temp_dir / "no_meta.md"
        path.write_text("# Doc\n\nBody.", encoding="utf-8")

        doc = Document(
            source_path=path,
            text="# Doc\n\nBody.",
            title="Doc",
            format="md",
        )

        output_path = converter.convert(doc)
        content = output_path.read_text(encoding="utf-8")
        assert "# Document:" not in content

    def test_convert_all(self, converter, temp_dir):
        """Should convert multiple documents."""
        docs = []
        for i in range(3):
            path = temp_dir / f"source_{i}.md"
            path.write_text(f"# Doc {i}\n\nContent {i}.", encoding="utf-8")
            docs.append(Document(
                source_path=path,
                text=f"# Doc {i}\n\nContent {i}.",
                title=f"Doc {i}",
                format="md",
            ))

        output_paths = converter.convert_all(docs)
        assert len(output_paths) == 3
        for p in output_paths:
            assert p.exists()
            assert p.suffix == ".txt"

    def test_convert_all_clears_existing(self, converter, temp_dir):
        """Should clear existing .txt files before converting."""
        # Create an existing file
        existing = converter.input_dir / "old_file.txt"
        existing.write_text("old content", encoding="utf-8")

        # Convert a new document
        path = temp_dir / "new.md"
        path.write_text("# New\n\nContent.", encoding="utf-8")
        doc = Document(
            source_path=path,
            text="# New\n\nContent.",
            title="New",
            format="md",
        )

        output_paths = converter.convert_all([doc], clear_existing=True)
        assert not existing.exists()
        assert len(output_paths) == 1

    def test_unique_output_names(self, converter, temp_dir):
        """Should generate unique names for same-name files in different formats."""
        path1 = temp_dir / "report.md"
        path2 = temp_dir / "report.html"
        path1.write_text("# MD", encoding="utf-8")
        path2.write_text("<html>HTML</html>", encoding="utf-8")

        doc1 = Document(source_path=path1, text="# MD", title="Report", format="md")
        doc2 = Document(source_path=path2, text="HTML", title="Report", format="html")

        out1 = converter.convert(doc1)
        out2 = converter.convert(doc2)

        assert out1.name != out2.name
        assert out1.name.endswith("_md.txt")
        assert out2.name.endswith("_html.txt")

    def test_conversion_report(self, converter, temp_dir):
        """Should generate a summary report."""
        path = temp_dir / "report.md"
        path.write_text("# R\n\nContent.", encoding="utf-8")
        doc = Document(source_path=path, text="# R\n\nContent.", title="R", format="md")

        output_paths = converter.convert_all([doc])
        report = converter.get_conversion_report([doc], output_paths)

        assert report["documents_converted"] == 1
        assert report["strip_markdown"] == converter.strip_markdown
        assert "input_dir" in report

    def test_clean_text_normalizes_line_endings(self, converter):
        """Should normalize CRLF to LF."""
        text = "line1\r\nline2\r\nline3"
        cleaned = converter._clean_text(text)
        assert "\r\n" not in cleaned
        assert "line1\nline2\nline3" in cleaned

    def test_clean_text_collapses_blank_lines(self, converter):
        """Should collapse excessive blank lines."""
        text = "para1\n\n\n\n\n\npara2"
        cleaned = converter._clean_text(text)
        assert "\n\n\n\n\n\n" not in cleaned
        assert "para1" in cleaned
        assert "para2" in cleaned
