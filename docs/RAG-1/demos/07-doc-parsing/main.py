"""
Demo 7: 文档结构化解析 — Structured Document Parsing
=====================================================
Parses Markdown and HTML content into a structured Document model,
then applies a StructureAwareChunker that splits by heading boundaries.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Structured Document Model
# ---------------------------------------------------------------------------

@dataclass
class Heading:
    level: int  # 1-6
    text: str


@dataclass
class Paragraph:
    text: str


@dataclass
class Table:
    headers: List[str]
    rows: List[List[str]]


@dataclass
class CodeBlock:
    language: str
    code: str


@dataclass
class Section:
    heading: Heading
    content: List  # mix of Paragraph | Table | CodeBlock


@dataclass
class Document:
    title: str
    sections: List[Section] = field(default_factory=list)

    def __repr__(self):
        parts = [f"Document(title='{self.title}', sections={len(self.sections)})"]
        for s in self.sections:
            parts.append(f"  H{s.heading.level}: {s.heading.text} ({len(s.content)} blocks)")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Markdown Parser
# ---------------------------------------------------------------------------

def parse_markdown(text: str) -> Document:
    """Parse Markdown text into a structured Document."""
    lines = text.split("\n")
    title = "Untitled"
    current_section: Optional[Section] = None
    sections: List[Section] = []

    for line in lines:
        stripped = line.strip()

        # Heading detection
        heading_match = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if heading_match:
            level = len(heading_match.group(1))
            text = heading_match.group(2)
            heading = Heading(level, text)
            current_section = Section(heading, [])
            sections.append(current_section)
            if level == 1:
                title = text
            continue

        # Code block detection
        code_match = re.match(r"^```(\w*)", stripped)
        if code_match:
            lang = code_match.group(1)
            code_lines = []
            for cl in lines[lines.index(line) + 1:]:
                if cl.strip().startswith("```"):
                    break
                code_lines.append(cl)
            if current_section is not None:
                current_section.content.append(CodeBlock(lang, "\n".join(code_lines)))
            continue

        # Table detection (simple pipe-based)
        if "|" in stripped and re.search(r"\|.*\|", stripped):
            cols = [c.strip() for c in stripped.split("|") if c.strip()]
            if current_section is not None:
                # Check if previous content is a Table
                if current_section.content and isinstance(current_section.content[-1], Table):
                    current_section.content[-1].rows.append(cols)
                else:
                    current_section.content.append(Table([], [cols]))
            continue

        # Skip separator rows
        if re.match(r"^[\|\s\-:]+$", stripped):
            continue

        # Regular paragraph
        if stripped and current_section is not None:
            current_section.content.append(Paragraph(stripped))

    return Document(title, sections)


# ---------------------------------------------------------------------------
# HTML Parser (simplified)
# ---------------------------------------------------------------------------

def parse_html(text: str) -> Document:
    """Parse a simplified HTML subset into a structured Document."""
    title = "Untitled"
    current_section: Optional[Section] = None
    sections: List[Section] = []

    # Extract title
    t_match = re.search(r"<title>(.*?)</title>", text, re.IGNORECASE)
    if t_match:
        title = t_match.group(1).strip()

    # Process block-level tags
    # First, extract body content
    body_match = re.search(r"<body>(.*?)</body>", text, re.IGNORECASE | re.DOTALL)
    body = body_match.group(1) if body_match else text

    # Split by headings
    # We'll use a simple approach: iterate through heading tags
    pattern = re.compile(r"<(h[1-6])>(.*?)</\1>", re.IGNORECASE)
    pos = 0
    for match in pattern.finditer(body):
        # Content before this heading becomes part of the previous section
        between = body[pos:match.start()].strip()
        if between and current_section is not None:
            # Extract paragraphs
            for p in re.findall(r">([^<]+)<", between):
                cleaned = re.sub(r"<[^>]+>", "", p).strip()
                if cleaned:
                    current_section.content.append(Paragraph(cleaned))

        level = int(match.group(1)[1])
        text = re.sub(r"<[^>]+>", "", match.group(2)).strip()
        heading = Heading(level, text)
        current_section = Section(heading, [])
        sections.append(current_section)
        if level == 1:
            title = text
        pos = match.end()

    # Remaining content after last heading
    remaining = body[pos:].strip()
    if remaining and current_section is not None:
        for p in re.findall(r">([^<]+)<", remaining):
            cleaned = re.sub(r"<[^>]+>", "", p).strip()
            if cleaned:
                current_section.content.append(Paragraph(cleaned))

    return Document(title, sections)


# ---------------------------------------------------------------------------
# StructureAwareChunker
# ---------------------------------------------------------------------------

class StructureAwareChunker:
    """Splits by heading boundaries, preserving document structure."""

    def __init__(self, max_chunk_size: int = 500):
        self.max_chunk_size = max_chunk_size

    def chunk(self, doc: Document) -> List[str]:
        """Split document into chunks at heading boundaries."""
        chunks = []
        for section in doc.sections:
            heading_str = f"{'#' * section.heading.level} {section.heading.text}"
            content_parts = []

            for block in section.content:
                if isinstance(block, Paragraph):
                    content_parts.append(block.text)
                elif isinstance(block, CodeBlock):
                    content_parts.append(f"```{block.language}\n{block.code}\n```")
                elif isinstance(block, Table):
                    if block.headers:
                        header = "| " + " | ".join(block.headers) + " |"
                        sep = "| " + " | ".join("---" for _ in block.headers) + " |"
                        rows = "\n".join(
                            "| " + " | ".join(r) + " |" for r in block.rows
                        )
                        content_parts.append(f"{header}\n{sep}\n{rows}")
                    else:
                        for r in block.rows:
                            content_parts.append("| " + " | ".join(r) + " |")

            content = "\n".join(content_parts)
            chunk_text = f"{heading_str}\n{content}"

            # Split long sections into sub-chunks
            if len(chunk_text) > self.max_chunk_size:
                # Simple split by paragraph
                paras = chunk_text.split("\n")
                current = heading_str
                for para in paras[1:]:
                    if len(current) + len(para) < self.max_chunk_size:
                        current += "\n" + para
                    else:
                        chunks.append(current)
                        current = heading_str + "\n" + para
                if current:
                    chunks.append(current)
            else:
                chunks.append(chunk_text)

        return chunks


# ---------------------------------------------------------------------------
# Sample Content
# ---------------------------------------------------------------------------

SAMPLE_MARKDOWN = """# RAG System Architecture

## Overview

RAG (Retrieval-Augmented Generation) combines information retrieval with language model generation.
This architecture enables LLMs to access external knowledge during inference.

## Components

The RAG pipeline consists of several key components:

| Component | Description | Example |
|-----------|-------------|---------|
| Retriever | Finds relevant documents | BM25, Dense |
| Generator | Produces final answer | GPT, Claude |
| Index | Stores document representations | Vector DB |

## Implementation Details

### Chunking Strategy

Documents are split into chunks before indexing. Common strategies include:

```python
def chunk_document(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = words[i:i + chunk_size]
        chunks.append(' '.join(chunk))
    return chunks
```

### Retrieval Methods

Two main retrieval approaches are used in modern RAG systems:
1. Sparse retrieval (BM25) - exact keyword matching
2. Dense retrieval (Embeddings) - semantic similarity

## Conclusion

RAG architecture continues to evolve with new techniques like GraphRAG and Agentic RAG.
"""

SAMPLE_HTML = """<html>
<head><title>RAG System Architecture</title></head>
<body>
<h1>RAG System Architecture</h1>
<p>RAG combines information retrieval with language model generation.</p>
<h2>Components</h2>
<p>The pipeline includes a retriever, generator, and index.</p>
<h2>Chunking</h2>
<p>Documents are split into chunks before indexing.</p>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Demo 7: 文档结构化解析 — Structured Document Parsing")
    print("=" * 60)

    # --- Markdown Parsing ---
    print("\n>>> 1. Parsing Markdown")
    md_doc = parse_markdown(SAMPLE_MARKDOWN)
    print(md_doc)
    print()

    # Show section details
    for i, section in enumerate(md_doc.sections, 1):
        print(f"\n  Section {i}: H{section.heading.level} \"{section.heading.text}\"")
        for j, block in enumerate(section.content, 1):
            if isinstance(block, Paragraph):
                print(f"    [{j}] Paragraph: {block.text[:60]}...")
            elif isinstance(block, CodeBlock):
                print(f"    [{j}] CodeBlock ({block.language}): {len(block.code)} chars")
            elif isinstance(block, Table):
                print(f"    [{j}] Table: {len(block.headers)} cols x {len(block.rows)} rows")

    # --- HTML Parsing ---
    print("\n>>> 2. Parsing HTML")
    html_doc = parse_html(SAMPLE_HTML)
    print(html_doc)

    # --- Structure-Aware Chunking ---
    print("\n>>> 3. Structure-Aware Chunking (Markdown)")
    chunker = StructureAwareChunker(max_chunk_size=300)
    chunks = chunker.chunk(md_doc)
    print(f"  Produced {len(chunks)} chunk(s):")
    for i, chunk in enumerate(chunks, 1):
        print(f"\n  --- Chunk {i} ({len(chunk)} chars) ---")
        print(f"  {chunk[:120]}...")

    print("\n" + "=" * 60)
    print("Structure-aware chunking preserves document hierarchy.")
    print("=" * 60)


if __name__ == "__main__":
    main()
