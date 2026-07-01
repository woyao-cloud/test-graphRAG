"""Tests for DocumentLoader."""

from __future__ import annotations

from pathlib import Path

import pytest

from graphrag_kg.ingest.loader import DocumentLoader
from graphrag_kg.ingest.parsers import Document


class TestDocumentLoader:
    """Tests for DocumentLoader."""

    @pytest.fixture
    def loader(self):
        return DocumentLoader()

    def test_discover_single_dir(self, loader, temp_dir):
        """Should discover files matching supported extensions."""
        (temp_dir / "doc1.md").write_text("# Doc 1")
        (temp_dir / "doc2.txt").write_text("Doc 2")
        (temp_dir / "doc3.html").write_text("<html>Doc 3</html>")
        (temp_dir / "image.png").write_text("not a document")

        files = loader.discover([temp_dir])
        assert len(files) == 3
        names = [f.name for f in files]
        assert "doc1.md" in names
        assert "doc2.txt" in names
        assert "doc3.html" in names
        assert "image.png" not in names

    def test_discover_with_patterns(self, loader, temp_dir):
        """Should filter by custom patterns."""
        (temp_dir / "doc1.md").write_text("# Doc 1")
        (temp_dir / "doc2.txt").write_text("Doc 2")

        files = loader.discover([temp_dir], file_patterns=["**/*.md"])
        assert len(files) == 1
        assert files[0].name == "doc1.md"

    def test_discover_non_recursive(self, loader, temp_dir):
        """Non-recursive should only find top-level files."""
        subdir = temp_dir / "sub"
        subdir.mkdir()
        (temp_dir / "top.md").write_text("top")
        (subdir / "nested.md").write_text("nested")

        files = loader.discover([temp_dir], recursive=False)
        assert len(files) == 1
        assert files[0].name == "top.md"

    def test_discover_multiple_dirs(self, loader, temp_dir):
        """Should merge results from multiple directories."""
        dir1 = temp_dir / "dir1"
        dir2 = temp_dir / "dir2"
        dir1.mkdir()
        dir2.mkdir()
        (dir1 / "a.md").write_text("a")
        (dir2 / "b.txt").write_text("b")

        files = loader.discover([dir1, dir2])
        assert len(files) == 2

    def test_discover_nonexistent_dir(self, loader):
        """Should not crash on nonexistent directories."""
        files = loader.discover([Path("/nonexistent/path")])
        assert files == []

    def test_load_single(self, loader, temp_dir):
        """Should parse a single file successfully."""
        path = temp_dir / "single.md"
        path.write_text("# Single\n\nContent here.", encoding="utf-8")

        doc = loader.load_single(path)
        assert isinstance(doc, Document)
        assert doc.format == "md"
        assert doc.title == "Single"

    def test_load_single_nonexistent(self, loader):
        """Should raise on nonexistent file."""
        with pytest.raises(FileNotFoundError):
            loader.load_single(Path("/nonexistent/file.md"))

    def test_size_filter(self, loader, temp_dir):
        """Should skip files exceeding max size."""
        loader.max_file_size_bytes = 10  # 10 bytes max
        path = temp_dir / "large.md"
        path.write_text("This is way more than ten bytes of text content")

        files = loader.discover([temp_dir])
        valid = loader._filter_by_size(files)
        assert len(valid) == 0

    def test_load_multiple(self, loader, temp_dir):
        """Should parse multiple files."""
        (temp_dir / "a.md").write_text("# A\nContent A")
        (temp_dir / "b.txt").write_text("Content B")

        docs = loader.load([temp_dir])
        assert len(docs) == 2
        formats = {d.format for d in docs}
        assert "md" in formats
        assert "txt" in formats

    def test_load_report(self, loader, temp_dir):
        """Should generate a summary report."""
        (temp_dir / "a.md").write_text("# A\nContent")

        docs = loader.load([temp_dir])
        report = loader.get_load_report(docs)

        assert report["total_documents"] == 1
        assert report["total_characters"] > 0
        assert report["by_format"]["md"] == 1


class TestLoaderWithGeneratedData:
    """Integration tests using Phase 0 generated data."""

    @pytest.fixture
    def pharma_docs_dir(self):
        """Path to generated pharma_supply_chain documents."""
        path = Path("tests/fixtures/generated/pharma_supply_chain/documents")
        if not path.exists():
            pytest.skip("Pharma test data not generated")
        return path

    def test_discover_generated_docs(self, pharma_docs_dir):
        """Should discover all generated pharma documents."""
        loader = DocumentLoader()
        files = loader.discover([pharma_docs_dir], recursive=False)
        # At least the 7 original md files should be present
        assert len(files) >= 7, f"Expected >=7 files, got {len(files)}"

    def test_parse_generated_md(self, pharma_docs_dir):
        """Should parse a generated markdown file correctly."""
        md_files = list(pharma_docs_dir.glob("*.md"))
        if not md_files:
            pytest.skip("No .md files in generated data")

        loader = DocumentLoader()
        doc = loader.load_single(md_files[0])
        assert doc.format == "md"
        assert not doc.is_empty
        assert doc.text_length > 100  # Should have substantial content

    def test_parse_generated_html(self, pharma_docs_dir):
        """Should parse a generated HTML file correctly."""
        html_files = list(pharma_docs_dir.glob("*.html"))
        if not html_files:
            pytest.skip("No .html files in generated data")

        loader = DocumentLoader()
        doc = loader.load_single(html_files[0])
        assert doc.format == "html"
        assert not doc.is_empty

    def test_parse_generated_txt(self, pharma_docs_dir):
        """Should parse a generated text file correctly."""
        txt_files = list(pharma_docs_dir.glob("*.txt"))
        if not txt_files:
            pytest.skip("No .txt files in generated data")

        loader = DocumentLoader()
        doc = loader.load_single(txt_files[0])
        assert doc.format == "txt"
        assert not doc.is_empty
