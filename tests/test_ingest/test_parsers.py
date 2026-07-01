"""Tests for document parsers (PDF, Markdown, Text, HTML)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from graphrag_kg.ingest.parsers import (
    Document,
    MarkdownParser,
    TextParser,
    HTMLParser,
    ParserRegistry,
)


class TestMarkdownParser:
    """Tests for MarkdownParser."""

    @pytest.fixture
    def parser(self):
        return MarkdownParser()

    @pytest.fixture
    def sample_md(self):
        return """# Test Document

This is a **bold** paragraph with *italic* text.

## Section Two

- List item 1
- List item 2
- List item 3

[Link text](https://example.com)

```python
print("hello")
```
"""

    def test_parse_basic(self, parser, sample_md, temp_dir):
        path = temp_dir / "test.md"
        path.write_text(sample_md, encoding="utf-8")

        doc = parser.parse(path)
        assert isinstance(doc, Document)
        assert doc.format == "md"
        assert doc.title == "Test Document"
        assert "bold" in doc.text
        assert "Section Two" in doc.text

    def test_supported_extensions(self, parser):
        exts = parser.supported_extensions()
        assert ".md" in exts
        assert ".markdown" in exts

    def test_supports(self, parser):
        assert parser.supports(Path("doc.md"))
        assert parser.supports(Path("doc.markdown"))
        assert not parser.supports(Path("doc.txt"))

    def test_strip_markdown(self, parser, sample_md):
        stripped = parser.strip_markdown(sample_md)
        assert "**bold**" not in stripped
        assert "bold" in stripped
        # Code blocks should be removed entirely
        assert "```" not in stripped
        assert "print" not in stripped
        assert "[Link text]" not in stripped
        assert "Link text" in stripped

    def test_extract_title_from_h1(self, parser, temp_dir):
        path = temp_dir / "titled.md"
        path.write_text("# My Custom Title\n\nContent here.", encoding="utf-8")

        doc = parser.parse(path)
        assert doc.title == "My Custom Title"

    def test_fallback_title(self, parser, temp_dir):
        path = temp_dir / "no_title.md"
        path.write_text("Just some content without a heading.", encoding="utf-8")

        doc = parser.parse(path)
        assert doc.title == "no_title"

    def test_encoding_fallback(self, parser, temp_dir):
        path = temp_dir / "gbk.md"
        path.write_bytes("# GBK Document\n\nContent".encode("gbk"))

        doc = parser.parse(path, encoding="utf-8")
        assert doc.text  # Should not crash


class TestTextParser:
    """Tests for TextParser."""

    @pytest.fixture
    def parser(self):
        return TextParser()

    def test_parse_utf8(self, parser, temp_dir):
        path = temp_dir / "sample.txt"
        path.write_text("Line 1\nLine 2\nLine 3", encoding="utf-8")

        doc = parser.parse(path)
        assert doc.format == "txt"
        assert doc.text == "Line 1\nLine 2\nLine 3"
        assert doc.metadata["line_count"] == 3

    def test_parse_with_encoding_detection(self, parser, temp_dir):
        path = temp_dir / "gbk.txt"
        path.write_bytes("GBK编码文件内容\n第二行\n第三行".encode("gbk"))

        doc = parser.parse(path, encoding="utf-8")
        assert doc.text  # Should parse with fallback encoding
        assert len(doc.text) > 0

    def test_title_extraction(self, parser, temp_dir):
        path = temp_dir / "titled.txt"
        path.write_text("Annual Report 2024\n\nContent goes here.", encoding="utf-8")

        doc = parser.parse(path)
        assert doc.title == "Annual Report 2024"

    def test_supported_extensions(self, parser):
        exts = parser.supported_extensions()
        assert ".txt" in exts
        assert ".log" in exts


class TestHTMLParser:
    """Tests for HTMLParser."""

    @pytest.fixture
    def parser(self):
        return HTMLParser()

    def test_parse_basic(self, parser, temp_dir):
        html = """<!DOCTYPE html>
<html>
<head><title>Test Page</title></head>
<body>
<h1>Welcome</h1>
<p>This is a <strong>test</strong> paragraph.</p>
<ul>
<li>Item 1</li>
<li>Item 2</li>
</ul>
<script>console.log('hidden');</script>
</body>
</html>"""
        path = temp_dir / "test.html"
        path.write_text(html, encoding="utf-8")

        doc = parser.parse(path)
        assert doc.format == "html"
        assert doc.title == "Test Page"
        assert "Welcome" in doc.text
        assert "test" in doc.text
        assert "Item 1" in doc.text
        # Script content should be stripped
        assert "console.log" not in doc.text

    def test_script_removal(self, parser, temp_dir):
        html = "<html><body><p>Visible</p><script>hidden()</script><p>Also visible</p></body></html>"
        path = temp_dir / "script.html"
        path.write_text(html, encoding="utf-8")

        doc = parser.parse(path)
        assert "Visible" in doc.text
        assert "hidden" not in doc.text
        assert "Also visible" in doc.text

    def test_style_removal(self, parser, temp_dir):
        html = "<html><head><style>body { color: red; }</style></head><body><p>Text</p></body></html>"
        path = temp_dir / "style.html"
        path.write_text(html, encoding="utf-8")

        doc = parser.parse(path)
        assert "Text" in doc.text
        assert "color: red" not in doc.text


class TestParserRegistry:
    """Tests for ParserRegistry."""

    def test_auto_detect(self, temp_dir):
        registry = ParserRegistry()

        # MD file
        md_path = temp_dir / "test.md"
        md_path.write_text("# Hello", encoding="utf-8")
        doc = registry.parse(md_path)
        assert doc.format == "md"

        # TXT file
        txt_path = temp_dir / "test.txt"
        txt_path.write_text("Hello", encoding="utf-8")
        doc = registry.parse(txt_path)
        assert doc.format == "txt"

        # HTML file
        html_path = temp_dir / "test.html"
        html_path.write_text("<html><body>Hello</body></html>", encoding="utf-8")
        doc = registry.parse(html_path)
        assert doc.format == "html"

    def test_unsupported_format(self, temp_dir):
        registry = ParserRegistry()
        path = temp_dir / "test.xyz"
        path.write_text("data")

        with pytest.raises(ValueError, match="No parser available"):
            registry.parse(path)

    def test_supported_formats(self):
        registry = ParserRegistry()
        formats = registry.supported_formats
        assert "pdf" in formats
        assert "md" in formats
        assert "txt" in formats
        assert "html" in formats

    def test_get_parser(self):
        registry = ParserRegistry()
        parser = registry.get_parser(Path("doc.md"))
        assert isinstance(parser, MarkdownParser)

        parser = registry.get_parser(Path("doc.txt"))
        assert isinstance(parser, TextParser)

        parser = registry.get_parser(Path("doc.xyz"))
        assert parser is None


class TestDocument:
    """Tests for Document dataclass."""

    def test_basic_properties(self, temp_dir):
        path = temp_dir / "test.txt"
        path.write_text("Hello World", encoding="utf-8")

        doc = Document(
            source_path=path,
            text="Hello World",
            title="Test",
            format="txt",
        )
        assert doc.size_bytes > 0
        assert doc.text_length == 11
        assert not doc.is_empty

    def test_empty_document(self, temp_dir):
        path = temp_dir / "empty.txt"
        path.write_text("", encoding="utf-8")

        doc = Document(
            source_path=path,
            text="   \n  ",
            format="txt",
        )
        assert doc.is_empty
