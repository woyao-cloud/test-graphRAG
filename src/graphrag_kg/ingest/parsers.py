"""Multi-format document parsers for GraphRAG-KG ingestion.

Supports PDF (pymupdf), Markdown, plain text (with encoding detection),
and HTML (beautifulsoup4) parsing. Each parser returns a Document object
with extracted text content and metadata.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


@dataclass
class Document:
    """A parsed document ready for graphrag ingestion.

    Attributes:
        source_path: Original file path.
        text: Extracted plain text content.
        title: Document title (derived from filename or content).
        format: Original format (pdf, md, txt, html).
        encoding: Detected or specified text encoding.
        metadata: Additional metadata (author, date, pages, etc.).
        parse_errors: Non-fatal warnings encountered during parsing.
    """

    source_path: Path
    text: str
    title: str = ""
    format: str = ""
    encoding: str = "utf-8"
    metadata: dict = field(default_factory=dict)
    parse_errors: list[str] = field(default_factory=list)

    @property
    def size_bytes(self) -> int:
        """File size in bytes."""
        try:
            return self.source_path.stat().st_size
        except OSError:
            return 0

    @property
    def text_length(self) -> int:
        """Character count of extracted text."""
        return len(self.text)

    @property
    def is_empty(self) -> bool:
        """Whether the document has no extracted text."""
        return len(self.text.strip()) == 0


# ============================================================================
# Parser Interface
# ============================================================================


class BaseParser:
    """Base class for format-specific parsers."""

    format_name: str = "unknown"

    def parse(self, path: Path, encoding: str = "utf-8") -> Document:
        """Parse a file into a Document.

        Args:
            path: Path to the source file.
            encoding: Text encoding hint.

        Returns:
            A Document with extracted text and metadata.

        Raises:
            FileNotFoundError: If file doesn't exist.
            ValueError: If file format doesn't match parser.
        """
        raise NotImplementedError

    def supports(self, path: Path) -> bool:
        """Check if this parser supports the given file."""
        return path.suffix.lower() in self.supported_extensions()

    def supported_extensions(self) -> set[str]:
        """Return set of supported file extensions."""
        raise NotImplementedError


# ============================================================================
# PDF Parser
# ============================================================================


class PDFParser(BaseParser):
    """Parse PDF files using pymupdf (fitz).

    Extracts text page by page, preserving paragraph structure.
    Falls back to pdfplumber if pymupdf is unavailable.
    """

    format_name = "pdf"

    def supported_extensions(self) -> set[str]:
        return {".pdf"}

    def parse(self, path: Path, encoding: str = "utf-8") -> Document:
        doc = Document(
            source_path=path,
            text="",
            title=path.stem,
            format="pdf",
            encoding=encoding,
            metadata={
                "filename": path.name,
                "parsed_at": datetime.now(timezone.utc).isoformat(),
            },
        )

        try:
            import fitz  # pymupdf

            pdf = fitz.open(str(path))
            doc.metadata["pages"] = len(pdf)

            # Extract PDF metadata
            pdf_meta = pdf.metadata
            if pdf_meta:
                if pdf_meta.get("title"):
                    doc.title = pdf_meta["title"]
                if pdf_meta.get("author"):
                    doc.metadata["author"] = pdf_meta["author"]

            # Extract text page by page
            pages_text: list[str] = []
            for page_num, page in enumerate(pdf):
                try:
                    text = page.get_text("text")
                    if text.strip():
                        pages_text.append(text)
                except Exception as e:
                    doc.parse_errors.append(f"Page {page_num + 1}: {e}")

            pdf.close()
            doc.text = "\n\n".join(pages_text)

        except ImportError:
            # Fallback to pdfplumber
            try:
                import pdfplumber

                with pdfplumber.open(str(path)) as pdf:
                    doc.metadata["pages"] = len(pdf.pages)
                    pages_text = []
                    for page in pdf.pages:
                        text = page.extract_text()
                        if text:
                            pages_text.append(text)
                    doc.text = "\n\n".join(pages_text)

            except ImportError:
                raise ImportError(
                    "No PDF parser available. Install pymupdf or pdfplumber: "
                    "pip install pymupdf"
                )

        # Clean up extracted text
        doc.text = self._clean_text(doc.text)

        return doc

    def _clean_text(self, text: str) -> str:
        """Clean up PDF-extracted text."""
        # Remove excessive newlines
        text = re.sub(r"\n{4,}", "\n\n\n", text)
        # Remove form feed characters
        text = text.replace("\f", "\n")
        # Normalize whitespace
        text = re.sub(r"[ \t]+", " ", text)
        # Fix hyphenated line breaks
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
        return text.strip()


# ============================================================================
# Markdown Parser
# ============================================================================


class MarkdownParser(BaseParser):
    """Parse Markdown files, optionally stripping syntax for plain text output."""

    format_name = "md"

    def supported_extensions(self) -> set[str]:
        return {".md", ".markdown", ".mdown", ".mkd"}

    def parse(self, path: Path, encoding: str = "utf-8") -> Document:
        text = self._read_file(path, encoding)

        # Extract title from first heading
        title = path.stem
        match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
        if match:
            title = match.group(1).strip()

        return Document(
            source_path=path,
            text=text,
            title=title,
            format="md",
            encoding=encoding,
            metadata={
                "filename": path.name,
                "size_bytes": path.stat().st_size,
                "parsed_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    def strip_markdown(self, text: str) -> str:
        """Convert markdown to plain text by removing formatting."""
        # Remove code blocks
        text = re.sub(r"```[^`]*```", "", text, flags=re.DOTALL)
        # Remove inline code
        text = re.sub(r"`([^`]+)`", r"\1", text)
        # Remove images
        text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)
        # Convert links to text
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
        # Remove bold/italic markers
        text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
        text = re.sub(r"_{1,3}([^_]+)_{1,3}", r"\1", text)
        # Remove heading markers
        text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
        # Remove horizontal rules
        text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
        # Remove blockquote markers
        text = re.sub(r"^>\s+", "", text, flags=re.MULTILINE)
        # Remove list markers
        text = re.sub(r"^[\s]*[-*+]\s+", "", text, flags=re.MULTILINE)
        text = re.sub(r"^[\s]*\d+\.\s+", "", text, flags=re.MULTILINE)
        # Remove table formatting (keep content)
        text = re.sub(r"\|", " ", text)
        text = re.sub(r"^[\s]*[-:| ]+$", "", text, flags=re.MULTILINE)
        return text.strip()

    def _read_file(self, path: Path, encoding: str) -> str:
        """Read file with encoding fallback."""
        try:
            return path.read_text(encoding=encoding)
        except (UnicodeDecodeError, LookupError):
            # Try common encodings
            for enc in ["utf-8-sig", "latin-1", "gbk", "gb2312"]:
                try:
                    return path.read_text(encoding=enc)
                except UnicodeDecodeError:
                    continue
            # Last resort
            return path.read_text(encoding="utf-8", errors="replace")


# ============================================================================
# Plain Text Parser
# ============================================================================


class TextParser(BaseParser):
    """Parse plain text files with automatic encoding detection."""

    format_name = "txt"

    def supported_extensions(self) -> set[str]:
        return {".txt", ".text", ".log", ".csv", ".tsv"}

    def parse(self, path: Path, encoding: str = "utf-8") -> Document:
        detected_encoding = encoding
        text = ""

        # Try specified encoding first
        try:
            text = path.read_text(encoding=encoding)
        except (UnicodeDecodeError, LookupError):
            # Use chardet for detection
            try:
                import chardet
                raw = path.read_bytes()
                result = chardet.detect(raw)
                detected_encoding = result.get("encoding", "utf-8") or "utf-8"
                text = raw.decode(detected_encoding, errors="replace")
            except ImportError:
                # Fallback: try common encodings
                for enc in ["utf-8", "utf-8-sig", "gbk", "gb2312", "latin-1"]:
                    try:
                        text = path.read_text(encoding=enc)
                        detected_encoding = enc
                        break
                    except UnicodeDecodeError:
                        continue
                if not text:
                    text = path.read_text(encoding="utf-8", errors="replace")

        # Derive title from first non-empty line or filename
        title = path.stem
        lines = text.strip().split("\n")
        for line in lines[:5]:
            stripped = line.strip()
            if stripped and len(stripped) > 3:
                # Use first substantial line as title
                if len(stripped) < 100:
                    title = stripped
                break

        return Document(
            source_path=path,
            text=text.strip(),
            title=title,
            format="txt",
            encoding=detected_encoding,
            metadata={
                "filename": path.name,
                "size_bytes": path.stat().st_size,
                "line_count": len(lines),
                "parsed_at": datetime.now(timezone.utc).isoformat(),
            },
        )


# ============================================================================
# HTML Parser
# ============================================================================


class HTMLParser(BaseParser):
    """Parse HTML files using beautifulsoup4.

    Extracts visible text, stripping scripts, styles, and navigation elements.
    """

    format_name = "html"

    def supported_extensions(self) -> set[str]:
        return {".html", ".htm", ".xhtml"}

    def parse(self, path: Path, encoding: str = "utf-8") -> Document:
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError(
                "HTML parsing requires beautifulsoup4: pip install beautifulsoup4"
            )

        # Read raw HTML
        raw_html = self._read_file(path, encoding)

        soup = BeautifulSoup(raw_html, "html.parser")

        # Extract title
        title = path.stem
        title_tag = soup.find("title")
        if title_tag and title_tag.string:
            title = title_tag.string.strip()
        else:
            h1 = soup.find("h1")
            if h1 and h1.get_text(strip=True):
                title = h1.get_text(strip=True)

        # Remove non-content elements
        for tag in soup.find_all(["script", "style", "nav", "footer", "noscript"]):
            tag.decompose()

        # Also remove common non-content classes
        for tag in soup.find_all(class_=re.compile(
            r"(nav|menu|sidebar|footer|advertisement|comment|meta)",
            re.IGNORECASE,
        )):
            tag.decompose()

        # Extract text from body
        body = soup.find("body")
        if body:
            text = body.get_text(separator="\n", strip=True)
        else:
            text = soup.get_text(separator="\n", strip=True)

        # Clean up whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()

        # Extract metadata
        metadata: dict = {
            "filename": path.name,
            "size_bytes": path.stat().st_size,
            "parsed_at": datetime.now(timezone.utc).isoformat(),
        }

        # Try to extract meta description
        meta_desc = soup.find("meta", attrs={"name": "description"})
        if meta_desc and meta_desc.get("content"):
            metadata["description"] = meta_desc["content"]

        # Count links
        links = soup.find_all("a", href=True)
        metadata["link_count"] = len(links)

        return Document(
            source_path=path,
            text=text,
            title=title,
            format="html",
            encoding=encoding,
            metadata=metadata,
        )

    def _read_file(self, path: Path, encoding: str) -> str:
        """Read HTML file with encoding detection."""
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            raw = path.read_bytes()

            # Check for encoding in meta tags
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(raw[:4096], "html.parser")
                meta_charset = soup.find("meta", attrs={"charset": True})
                if meta_charset:
                    return raw.decode(meta_charset["charset"], errors="replace")
                meta_content = soup.find("meta", attrs={"http-equiv": lambda x: x and x.lower() == "content-type"})
                if meta_content and meta_content.get("content"):
                    import re as re_mod
                    match = re_mod.search(r"charset=([\w-]+)", meta_content["content"])
                    if match:
                        return raw.decode(match.group(1), errors="replace")
            except Exception:
                pass

            return raw.decode("utf-8", errors="replace")


# ============================================================================
# Parser Registry
# ============================================================================


class ParserRegistry:
    """Registry of format parsers with auto-detection."""

    def __init__(self):
        self._parsers: dict[str, BaseParser] = {}
        self._ext_map: dict[str, BaseParser] = {}

        # Register default parsers
        self.register(PDFParser())
        self.register(MarkdownParser())
        self.register(TextParser())
        self.register(HTMLParser())

    def register(self, parser: BaseParser) -> None:
        """Register a parser."""
        self._parsers[parser.format_name] = parser
        for ext in parser.supported_extensions():
            self._ext_map[ext.lower()] = parser

    def get_parser(self, path: Path) -> Optional[BaseParser]:
        """Get the appropriate parser for a file path."""
        ext = path.suffix.lower()
        return self._ext_map.get(ext)

    def parse(self, path: Path, encoding: str = "utf-8") -> Document:
        """Auto-detect format and parse a file.

        Raises:
            ValueError: If no parser is available for the file format.
        """
        parser = self.get_parser(path)
        if parser is None:
            ext = path.suffix.lower()
            raise ValueError(
                f"No parser available for format '{ext}'. "
                f"Supported formats: {sorted(self._ext_map.keys())}"
            )
        return parser.parse(path, encoding=encoding)

    @property
    def supported_formats(self) -> list[str]:
        """List of supported format names."""
        return sorted(self._parsers.keys())

    @property
    def supported_extensions(self) -> list[str]:
        """List of supported file extensions."""
        return sorted(self._ext_map.keys())
