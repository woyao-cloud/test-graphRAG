"""
第 4 章 Demo：多源文档处理管线

演示完整的文档处理流程：多格式解析 → 清洗 → 切分 → 入库。
可独立运行，无需外部依赖。

用法：
  python document_processing_pipeline.py
  python document_processing_pipeline.py --input-dir ./my_docs
"""

import argparse
import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional


# ============================================================================
# Part 1: 数据结构定义
# ============================================================================


@dataclass
class Document:
    """统一文档模型。"""
    id: str
    title: str
    content: str
    source_path: str
    source_format: str
    metadata: dict = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class Chunk:
    """文档块。"""
    id: str
    doc_id: str
    content: str
    metadata: dict = field(default_factory=dict)
    embedding: Optional[list[float]] = None


# ============================================================================
# Part 2: 多格式文档解析器
# ============================================================================


class DocumentParser:
    """多格式文档解析器（演示版，支持模拟数据）。"""

    def __init__(self):
        self.parsers: dict[str, Callable] = {
            ".txt": self._parse_text,
            ".md": self._parse_markdown,
            ".html": self._parse_html,
            ".json": self._parse_json,
        }

    def parse(self, content: str, source_path: str, fmt: str) -> Document:
        """解析文档内容。"""
        parser = self.parsers.get(fmt)
        if not parser:
            # Fallback: 按纯文本处理
            return self._parse_text(content, source_path, fmt)

        return parser(content, source_path, fmt)

    def _parse_text(self, content: str, source_path: str, fmt: str) -> Document:
        """纯文本解析。"""
        lines = [line.strip() for line in content.split("\n")]
        title = lines[0] if lines else Path(source_path).stem
        # 编码检测
        return Document(
            id=self._make_id(source_path),
            title=title,
            content="\n".join(lines),
            source_path=source_path,
            source_format=fmt,
            metadata={"line_count": len(lines)},
        )

    def _parse_markdown(self, content: str, source_path: str, fmt: str) -> Document:
        """Markdown 解析（保留标题层级）。"""
        # 提取标题
        title_match = re.search(r"^# (.+)$", content, re.MULTILINE)
        title = title_match.group(1) if title_match else Path(source_path).stem

        # 提取章节结构
        sections = re.findall(r"^(#{1,3}) (.+)$", content, re.MULTILINE)

        return Document(
            id=self._make_id(source_path),
            title=title,
            content=content,
            source_path=source_path,
            source_format=fmt,
            metadata={
                "sections": [{"level": len(s[0]), "title": s[1]} for s in sections],
                "has_code_blocks": "```" in content,
            },
        )

    def _parse_html(self, content: str, source_path: str, fmt: str) -> Document:
        """HTML 解析（去标签）。"""
        # 简单去标签
        text = re.sub(r"<[^>]+>", "", content)
        text = re.sub(r"\s+", " ", text).strip()

        # 提取 title
        title_match = re.search(r"<title>(.+?)</title>", content, re.IGNORECASE)
        title = title_match.group(1) if title_match else Path(source_path).stem

        return Document(
            id=self._make_id(source_path),
            title=title,
            content=text,
            source_path=source_path,
            source_format=fmt,
            metadata={"original_length": len(content)},
        )

    def _parse_json(self, content: str, source_path: str, fmt: str) -> Document:
        """JSON 文档解析。"""
        data = json.loads(content)
        title = data.get("title", data.get("name", Path(source_path).stem))
        text = data.get("content", data.get("text", json.dumps(data, ensure_ascii=False)))
        return Document(
            id=self._make_id(source_path),
            title=title,
            content=text,
            source_path=source_path,
            source_format=fmt,
            metadata={"keys": list(data.keys())},
        )

    def _make_id(self, path: str) -> str:
        return hashlib.md5(path.encode()).hexdigest()[:12]


# ============================================================================
# Part 3: 数据清洗管线
# ============================================================================


class DataCleaner:
    """数据清洗管线，规则可配置。"""

    def __init__(self):
        self.rules = [
            ("remove_null_bytes", self._remove_null_bytes),
            ("normalize_unicode", self._normalize_unicode),
            ("collapse_whitespace", self._collapse_whitespace),
            ("remove_empty_lines", self._remove_empty_lines),
        ]

    def clean(self, doc: Document) -> Document:
        """执行所有清洗规则。"""
        content = doc.content
        applied = []
        for name, rule in self.rules:
            content = rule(content)
            applied.append(name)
        doc.content = content
        doc.metadata["cleaning_rules"] = applied
        return doc

    def _remove_null_bytes(self, text: str) -> str:
        return text.replace("\x00", "")

    def _normalize_unicode(self, text: str) -> str:
        import unicodedata
        return unicodedata.normalize("NFKC", text)

    def _collapse_whitespace(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def _remove_empty_lines(self, text: str) -> str:
        lines = [l for l in text.split("\n") if l.strip()]
        return "\n".join(lines)

    def add_rule(self, name: str, rule: Callable[[str], str]):
        """添加自定义清洗规则。"""
        self.rules.append((name, rule))


# ============================================================================
# Part 4: 文档切分器
# ============================================================================


class DocumentChunker:
    """多种切分策略。"""

    @staticmethod
    def fixed_size_chunk(doc: Document, chunk_size: int = 300, overlap: int = 30) -> list[Chunk]:
        """固定大小切分 + 重叠。"""
        text = doc.content
        chunks = []
        start = 0
        idx = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end]
            chunks.append(Chunk(
                id=f"{doc.id}_chunk_{idx:04d}",
                doc_id=doc.id,
                content=chunk_text,
                metadata={
                    **doc.metadata,
                    "chunk_type": "fixed",
                    "char_range": [start, end],
                },
            ))
            idx += 1
            start += chunk_size - overlap
        return chunks

    @staticmethod
    def recursive_chunk(doc: Document, max_chars: int = 500) -> list[Chunk]:
        """递归切分（按段落 → 句子 → 固定长度降级）。"""
        text = doc.content
        chunks = []

        # Level 1: 按段落切分
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

        current_chunks = []
        current_len = 0
        for para in paragraphs:
            if current_len + len(para) > max_chars and current_chunks:
                # 合并当前批次
                chunks.append(Chunk(
                    id=f"{doc.id}_rec_{len(chunks):04d}",
                    doc_id=doc.id,
                    content="\n".join(current_chunks),
                    metadata={**doc.metadata, "chunk_type": "recursive"},
                ))
                current_chunks = []
                current_len = 0
            current_chunks.append(para)
            current_len += len(para)

        if current_chunks:
            chunks.append(Chunk(
                id=f"{doc.id}_rec_{len(chunks):04d}",
                doc_id=doc.id,
                content="\n".join(current_chunks),
                metadata={**doc.metadata, "chunk_type": "recursive"},
            ))

        return chunks if chunks else DocumentChunker.fixed_size_chunk(doc, max_chars)


# ============================================================================
# Part 5: 完整处理管线
# ============================================================================


class ProcessingPipeline:
    """端到端处理管线。"""

    def __init__(self):
        self.parser = DocumentParser()
        self.cleaner = DataCleaner()
        self.chunker = DocumentChunker()
        self.documents: list[Document] = []
        self.chunks: list[Chunk] = []

    def process_file(self, file_path: Path) -> tuple[Document, list[Chunk]]:
        """处理单个文件。"""
        ext = file_path.suffix.lower()
        fmt = ext.lstrip(".")

        # 1. 读取
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        # 2. 解析
        doc = self.parser.parse(content, str(file_path), fmt)
        print(f"  [解析] {file_path.name} → {fmt} 格式, {len(content)} 字符")

        # 3. 清洗
        doc = self.cleaner.clean(doc)
        print(f"  [清洗] 规则: {', '.join(doc.metadata['cleaning_rules'])}")

        # 4. 切分
        chunks = DocumentChunker.recursive_chunk(doc)
        print(f"  [切分] {len(chunks)} 个文档块")

        # 5. 统计
        doc.metadata["chunk_count"] = len(chunks)
        doc.metadata["total_chars"] = len(doc.content)

        return doc, chunks

    def process_batch(self, input_dir: Path, file_pattern: str = "*") -> list[Document]:
        """批量处理目录下所有文件。"""
        self.documents = []
        self.chunks = []

        files = list(input_dir.rglob(file_pattern))
        print(f"\n发现 {len(files)} 个文件\n")

        for file_path in files:
            if file_path.suffix.lower() not in {".txt", ".md", ".html", ".json"}:
                print(f"  [跳过] {file_path.name} (不支持格式)")
                continue

            try:
                doc, chunks = self.process_file(file_path)
                self.documents.append(doc)
                self.chunks.extend(chunks)
            except Exception as e:
                print(f"  [错误] {file_path.name}: {e}")

        self._print_summary()
        return self.documents

    def _print_summary(self):
        """输出处理汇总。"""
        print("\n" + "=" * 50)
        print("处理汇总")
        print("=" * 50)
        print(f"文档总数: {len(self.documents)}")
        print(f"文档块总数: {len(self.chunks)}")
        total_chars = sum(len(d.content) for d in self.documents)
        print(f"总字符数: {total_chars:,}")
        print(f"平均每文档块字符: {total_chars // max(len(self.chunks), 1):,}")

        formats = {}
        for d in self.documents:
            formats[d.source_format] = formats.get(d.source_format, 0) + 1
        if formats:
            print("格式分布: " + ", ".join(f"{k}: {v}" for k, v in formats.items()))


# ============================================================================
# Main
# ============================================================================


def create_sample_files(output_dir: Path):
    """生成示例文档供演示。"""
    # 示例 Markdown
    md_content = """# 恒瑞医药产品手册

## 1. 抗肿瘤药物

恒瑞医药是国内领先的抗肿瘤药物生产企业。

### 1.1 注射用紫杉醇（白蛋白结合型）
- 适应症：非小细胞肺癌、乳腺癌
- 规格：100mg/瓶

### 1.2 卡瑞利珠单抗
- 适应症：霍奇金淋巴瘤
- 规格：200mg/瓶

## 2. 供应链合作伙伴

- 原料药供应商：华海药业
- 区域分销商：国药控股
"""

    # 示例纯文本
    txt_content = """国药控股2023年财报概要

总营业收入：2000亿元人民币
净利润：85亿元人民币
华东区分销额占比：35%

主要合作药企：
  1. 恒瑞医药
  2. 齐鲁制药
  3. 正大天晴
"""

    # 示例 HTML
    html_content = """<html>
<head><title>北京协和医院 - 药品采购指南</title></head>
<body>
<h1>2024年抗肿瘤药物采购目录</h1>
<table>
<tr><th>药品名称</th><th>供应商</th><th>采购量</th></tr>
<tr><td>注射用紫杉醇</td><td>恒瑞医药</td><td>50000支</td></tr>
<tr><td>吉非替尼片</td><td>齐鲁制药</td><td>30000盒</td></tr>
</table>
</body></html>"""

    files = {
        "product_manual.md": md_content,
        "financial_report.txt": txt_content,
        "procurement_guide.html": html_content,
    }

    for name, content in files.items():
        (output_dir / name).write_text(content, encoding="utf-8")
        print(f"  创建: {name}")


def main():
    parser = argparse.ArgumentParser(description="文档处理管线演示")
    parser.add_argument("--input-dir", default=None, help="输入文档目录")
    parser.add_argument("--chunk-size", type=int, default=300, help="切分块大小")
    args = parser.parse_args()

    pipeline = ProcessingPipeline()

    if args.input_dir:
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            print(f"目录不存在: {input_dir}")
            return
    else:
        # 使用临时目录生成示例文件
        import tempfile
        input_dir = Path(tempfile.mkdtemp(prefix="rag_demo_"))
        print("生成示例文档...")
        create_sample_files(input_dir)

    # 执行处理管线
    t0 = time.time()
    pipeline.process_batch(input_dir)
    elapsed = time.time() - t0

    print(f"\n总耗时: {elapsed:.2f}s")
    print("处理完成！")


if __name__ == "__main__":
    main()
