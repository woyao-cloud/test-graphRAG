"""
ch04-doc-processing: 文档处理演示 - RAG管道的文档解析与结构化分块
================================================================
使用stdlib的HTML解析器和Markdown解析功能。
可直接运行: python main.py
"""

import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import List


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class Document:
    """解析后的文档结构。"""
    title: str = ""
    paragraphs: List[str] = field(default_factory=list)
    tables: List[List[List[str]]] = field(default_factory=list)  # list of (rows of cells)
    headings: List[tuple] = field(default_factory=list)  # (level, text)


# ---------------------------------------------------------------------------
# 生成样例文件
# ---------------------------------------------------------------------------

SAMPLE_MD = """# 恒瑞医药2024年年度报告摘要

## 公司概况

恒瑞医药是中国领先的创新药研发企业，成立于1970年，总部位于江苏省连云港市。

公司主要从事抗肿瘤药物、麻醉药物、造影剂等领域的研发、生产和销售。

## 财务摘要

| 指标 | 2024年 | 2023年 | 同比增长 |
|------|--------|--------|----------|
| 营业收入 | 280亿元 | 250亿元 | +12% |
| 研发投入 | 60亿元 | 55亿元 | +9% |
| 净利润 | 70亿元 | 62亿元 | +13% |

## 研发管线

截至2024年底，公司共有30余个创新药处于临床研究阶段。

重点管线包括：
- 抗肿瘤领域：PD-1抑制剂、PARP抑制剂、CDK4/6抑制剂
- 代谢疾病领域：GLP-1受体激动剂、SGLT2抑制剂
- 自身免疫领域：JAK抑制剂、IL-17抑制剂

## 市场表现

恒瑞医药的产品已覆盖全国超过3000家医院。
"""

SAMPLE_HTML = """<!DOCTYPE html>
<html>
<head><title>国家药品监督管理局 - 药品审批公告</title></head>
<body>
<h1>2024年药品审批公告</h1>

<h2>新药获批情况</h2>
<p>2024年，国家药品监督管理局共批准了48个新药上市，其中包括26个国产创新药。</p>
<p>获批药物涵盖抗肿瘤、心血管、代谢性疾病等多个治疗领域。</p>

<h2>重点获批药物列表</h2>
<table border="1">
<tr><th>药品名称</th><th>企业名称</th><th>适应症</th></tr>
<tr><td>甲磺酸奥希替尼片</td><td>阿斯利康</td><td>非小细胞肺癌</td></tr>
<tr><td>注射用紫杉醇(白蛋白结合型)</td><td>恒瑞医药</td><td>乳腺癌</td></tr>
<tr><td>司美格鲁肽注射液</td><td>诺和诺德</td><td>2型糖尿病</td></tr>
</table>

<h2>政策动态</h2>
<p>国家药监局持续推进药品审评审批制度改革，加速创新药上市进程。</p>
<p>优先审评审批通道已覆盖突破性治疗药物、罕见病用药等类别。</p>
</body>
</html>
"""


def create_sample_files(dir_path: str):
    """在指定目录下创建样例 .md 和 .html 测试文件。"""
    md_path = os.path.join(dir_path, "sample_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(SAMPLE_MD)
    print(f"  [创建] {md_path} ({len(SAMPLE_MD)} bytes)")

    html_path = os.path.join(dir_path, "sample_announcement.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(SAMPLE_HTML)
    print(f"  [创建] {html_path} ({len(SAMPLE_HTML)} bytes)")

    return md_path, html_path


# ---------------------------------------------------------------------------
# Markdown 解析器
# ---------------------------------------------------------------------------

RE_HEADING = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
RE_TABLE_ROW = re.compile(r"^\|(.+)\|$")
RE_TABLE_SEP = re.compile(r"^\|[-:\s|]+\|$")


def parse_markdown(filepath: str) -> Document:
    """解析Markdown文件，提取标题、段落、表格和标题层级。"""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    doc = Document()
    lines = content.split("\n")

    # 从第一个h1提取标题
    heading_match = RE_HEADING.search(content)
    if heading_match:
        doc.title = heading_match.group(2).strip()

    i = 0
    while i < len(lines):
        line = lines[i]

        # 检测标题
        hm = re.match(r"^(#{1,6})\s+(.+)$", line)
        if hm:
            level = len(hm.group(1))
            text = hm.group(2).strip()
            doc.headings.append((level, text))
            i += 1
            continue

        # 检测表格
        if RE_TABLE_ROW.match(line):
            table_rows = []
            # 收集所有表格行
            while i < len(lines) and RE_TABLE_ROW.match(lines[i]):
                row_line = lines[i]
                # 跳过分隔行（|---|）
                if RE_TABLE_SEP.match(row_line):
                    i += 1
                    continue
                cells = [
                    cell.strip()
                    for cell in row_line.strip("|").split("|")
                ]
                table_rows.append(cells)
                i += 1
            if table_rows:
                doc.tables.append(table_rows)
            continue

        # 普通段落
        if line.strip():
            doc.paragraphs.append(line.strip())

        i += 1

    return doc


# ---------------------------------------------------------------------------
# HTML 解析器
# ---------------------------------------------------------------------------


class MyHTMLParser(HTMLParser):
    """自定义HTML解析器，提取标题、段落和表格内容。"""

    def __init__(self):
        super().__init__()
        self.doc = Document()
        self._in_tag = ""
        self._current_text = []
        self._current_table: List[List[str]] = []
        self._current_row: List[str] = []
        self._current_cell: List[str] = []
        self._skip_tag = 0
        self._heading_level = 0

    def _flush_text(self, target: str):
        text = "".join(self._current_text).strip()
        if text:
            if target == "paragraph":
                self.doc.paragraphs.append(text)
            elif target == "heading":
                self.doc.headings.append((self._heading_level, text))
                if not self.doc.title:
                    self.doc.title = text
            elif target == "cell":
                self._current_cell.append(text)
        self._current_text = []

    def handle_starttag(self, tag, attrs):
        tag_lower = tag.lower()
        if tag_lower in ("script", "style"):
            self._skip_tag += 1
            return
        if tag_lower == "h1":
            self._heading_level = 1
            self._flush_text("paragraph")
        elif tag_lower == "h2":
            self._heading_level = 2
            self._flush_text("paragraph")
        elif tag_lower == "h3":
            self._heading_level = 3
            self._flush_text("paragraph")
        elif tag_lower == "p":
            self._flush_text("paragraph")
        elif tag_lower == "tr":
            if self._current_row:
                self._current_table.append(self._current_row)
                self._current_row = []
        elif tag_lower in ("td", "th"):
            self._flush_text("cell")

    def handle_endtag(self, tag):
        tag_lower = tag.lower()
        if tag_lower in ("script", "style"):
            if self._skip_tag > 0:
                self._skip_tag -= 1
            return
        if self._skip_tag > 0:
            return
        if tag_lower in ("h1", "h2", "h3"):
            self._flush_text("heading")
            self._heading_level = 0
        elif tag_lower == "p":
            self._flush_text("paragraph")
        elif tag_lower in ("td", "th"):
            self._current_row.append("".join(self._current_cell).strip())
            self._current_cell = []
        elif tag_lower == "tr":
            if self._current_row:
                self._current_table.append(self._current_row)
                self._current_row = []
        elif tag_lower == "table":
            if self._current_table:
                self.doc.tables.append(self._current_table)
                self._current_table = []

    def handle_data(self, data):
        if self._skip_tag > 0:
            return
        self._current_text.append(data)


def parse_html(filepath: str) -> Document:
    """解析HTML文件，提取文档结构。"""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    parser = MyHTMLParser()
    parser.feed(content)
    return parser.doc


# ---------------------------------------------------------------------------
# 结构感知分块器
# ---------------------------------------------------------------------------


@dataclass
class StructureAwareChunker:
    """根据标题边界对文档进行分块的结构感知分块器。"""

    min_chunk_size: int = 20

    def chunk(self, doc: Document) -> List[dict]:
        """将文档按标题边界切块。

        Returns:
            块字典列表，每项包含 'heading'、'content' 和 'size'。
        """
        if not doc.headings:
            return [
                {
                    "heading": doc.title or "(无标题)",
                    "content": "\n".join(doc.paragraphs),
                    "size": sum(len(p) for p in doc.paragraphs),
                }
            ]

        chunks = []
        # 构建 (start_idx, heading_text) 的映射
        heading_map: List[tuple] = []  # (paragraph_start_index, heading)
        para_count = len(doc.paragraphs)

        # 将标题映射到段落区间
        for level, htext in doc.headings:
            heading_map.append((htext, level))

        if not heading_map:
            return chunks

        # 简化：按标题在文本流中的出现顺序分块
        # 将文档重新组合为带标注的文本流
        flow = []  # list of (type, text) where type is 'heading' or 'para'
        for level, htext in doc.headings:
            flow.append(("heading", htext, level))
        for p in doc.paragraphs:
            flow.append(("para", p, 0))

        # 按标题分组
        current_chunk_heading = doc.title or "文档开头"
        current_chunk_parts = []
        tables_consumed = 0

        for item in flow:
            if item[0] == "heading":
                # 如果当前块有内容，保存它
                if current_chunk_parts:
                    content = "\n".join(current_chunk_parts)
                    if len(content) >= self.min_chunk_size:
                        chunks.append(
                            {
                                "heading": current_chunk_heading,
                                "content": content,
                                "size": len(content),
                            }
                        )
                    else:
                        # 合并到下一个块
                        pass
                current_chunk_heading = f"{'#' * item[2]} {item[1]}"
                current_chunk_parts = []
                # 尝试包含表格
                while tables_consumed < len(doc.tables):
                    tbl = doc.tables[tables_consumed]
                    tbl_text = _format_table(tbl)
                    # 检查表格是否属于此标题（基于位置启发式）
                    # 简单策略：将未分配的表格附加到当前块
                    current_chunk_parts.append(tbl_text)
                    tables_consumed += 1
            else:
                current_chunk_parts.append(item[1])

        # 最后一个块
        if current_chunk_parts:
            content = "\n".join(current_chunk_parts)
            if len(content) >= self.min_chunk_size:
                chunks.append(
                    {
                        "heading": current_chunk_heading,
                        "content": content,
                        "size": len(content),
                    }
                )

        # 如果没有产生任何块，创建一个兜底块
        if not chunks:
            all_text = "\n".join(doc.paragraphs)
            if all_text:
                chunks.append(
                    {
                        "heading": doc.title or "(全文)",
                        "content": all_text,
                        "size": len(all_text),
                    }
                )

        return chunks


def _format_table(table: List[List[str]]) -> str:
    """将解析的表格格式化为可读的文本表示。"""
    if not table:
        return ""
    lines = []
    for row in table:
        lines.append(" | ".join(row))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def main():
    print("=" * 60)
    print("  Document Processing Demo (RAG Pipeline)")
    print("=" * 60)

    # 1. 创建临时目录和样例文件
    tmpdir = tempfile.mkdtemp(prefix="rag_doc_demo_")
    print(f"\n[1] 创建样例文件到: {tmpdir}")
    md_path, html_path = create_sample_files(tmpdir)

    # 2. 解析Markdown
    print(f"\n[2] 解析Markdown文件: {os.path.basename(md_path)}")
    md_doc = parse_markdown(md_path)
    _print_doc_stats(md_doc)

    # 3. 解析HTML
    print(f"\n[3] 解析HTML文件: {os.path.basename(html_path)}")
    html_doc = parse_html(html_path)
    _print_doc_stats(html_doc)

    # 4. 结构感知分块 - Markdown
    print(f"\n[4] 结构感知分块 (Markdown)")
    chunker = StructureAwareChunker(min_chunk_size=20)
    md_chunks = chunker.chunk(md_doc)
    _print_chunks(md_chunks)

    # 5. 结构感知分块 - HTML
    print(f"\n[5] 结构感知分块 (HTML)")
    html_chunks = chunker.chunk(html_doc)
    _print_chunks(html_chunks)

    # 6. 清理
    shutil.rmtree(tmpdir)
    print(f"\n[6] 清理临时目录: {tmpdir}")

    print("\n" + "=" * 60)
    print("  Demo 完成")
    print("=" * 60)


def _print_doc_stats(doc: Document):
    """打印文档统计信息。"""
    print(f"  标题: {doc.title}")
    print(f"  标题层级: {', '.join(f'H{l}' for l, t in doc.headings)}")
    for level, text in doc.headings:
        print(f"    {'  ' * (level - 1)}H{level}: {text}")
    print(f"  段落数: {len(doc.paragraphs)}")
    print(f"  表格数: {len(doc.tables)}")
    for i, tbl in enumerate(doc.tables):
        print(f"    表格 #{i + 1}: {len(tbl)} 行 x {len(tbl[0]) if tbl else 0} 列")


def _print_chunks(chunks: List[dict]):
    """打印分块结果。"""
    print(f"  总块数: {len(chunks)}")
    for i, chk in enumerate(chunks):
        print(f"\n  --- 块 #{i + 1} ---")
        print(f"  标题: {chk['heading']}")
        print(f"  大小: {chk['size']} 字符")
        content_preview = chk["content"][:120].replace("\n", " ")
        if len(chk["content"]) > 120:
            content_preview += "..."
        print(f"  内容: {content_preview}")


if __name__ == "__main__":
    main()
