# 第4章 文档处理：从原始文件到可检索文本

## 4.1 引言

在构建 RAG（检索增强生成）系统的过程中，**文档处理** 是最基础也最容易被低估的环节。原始文档格式各异 —— PDF、Word、Excel、HTML、Markdown、扫描件 —— 它们必须先经过一系列解析、清洗、分块（chunking）步骤，才能进入向量数据库并支持语义检索。本章将完整覆盖从原始文件摄入到高质量文本块产出的全流程，提供大量可直接运行的 Python 代码示例，并深入讨论每种策略的适用场景与权衡。

本章涵盖以下主题：

| 主题 | 说明 |
|------|------|
| 多格式解析 | PDF、Word、Excel、HTML、Markdown 的解析库选型与代码实现 |
| OCR 与布局分析 | PaddleOCR、Tesseract 的集成，版面还原与阅读顺序恢复 |
| 数据清洗 | 去重（MinHash / SimHash）、格式标准化、PII 检测与过滤、噪声剔除 |
| 编码检测 | 使用 chardet 自动识别文件编码 |
| 分块策略 | Token 级、语义级、结构感知、递归分块的原理与代码 |
| 参数调优 | 块大小、重叠、分块策略选择的实验方法与评估指标 |

---

## 4.2 多格式解析 (Multi-Format Parsing)

现实世界的文档很少以纯文本形式存在。企业的知识库中混杂着 PDF 合同、Word 报告、Excel 数据表、HTML 网页和 Markdown 技术文档。每种格式都有其独特的解析挑战。

### 4.2.1 PDF 解析 (PyMuPDF / pdfplumber)

PDF（Portable Document Format）是最常见的文档格式，也是解析难度最高的格式之一。PDF 的本质是页面描述语言 —— 它记录的是"在坐标 (x, y) 处绘制文本'Hello'"这样的指令，而非段落、标题等逻辑结构。

**PyMuPDF（fitz）** 是目前 Python 生态中最推荐的 PDF 解析库，原因如下：

- 解析速度快（底层用 C 实现）
- 文本提取质量高
- 支持 PDF 注释、元数据、目录（TOC）
- 内置图片提取功能

```python
# ch04_pdf_parsing.py
"""PDF 解析示例：使用 PyMuPDF (fitz) 提取文本、表格与元数据"""

import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    使用 PyMuPDF 提取 PDF 全部文本。
    
    Args:
        pdf_path: PDF 文件路径
    
    Returns:
        合并后的纯文本字符串
    """
    doc = fitz.open(pdf_path)
    full_text = []
    
    for page_num, page in enumerate(doc):
        # 按页提取文本，保留大致布局
        text = page.get_text("text")
        full_text.append(f"--- 第 {page_num + 1} 页 ---\n{text}")
    
    doc.close()
    return "\n\n".join(full_text)


def extract_text_with_layout(pdf_path: str) -> str:
    """
    按布局块（block）提取文本，保留更多结构信息。
    PyMuPDF 的 block 模式会按段落/图片分组。
    """
    doc = fitz.open(pdf_path)
    blocks_info = []
    
    for page in doc:
        # blocks 是包含文本块、图片块的列表
        blocks = page.get_text("blocks")
        for b in blocks:
            # block 结构: (x0, y0, x1, y1, text, block_type, block_no)
            x0, y0, x1, y1, text, block_type, block_no = b
            if block_type == 0:  # 文本块
                blocks_info.append({
                    "bbox": (x0, y0, x1, y1),
                    "text": text.strip(),
                    "block_no": block_no,
                    "page": page.number + 1
                })
            # block_type == 1 表示图片块
    
    doc.close()
    return blocks_info


def extract_table_from_pdf(pdf_path: str, page_number: int = 0) -> list:
    """
    使用 PyMuPDF 查找并提取表格数据。
    注意：PyMuPDF 不内置表格检测，这里用启发式方法识别。
    
    更专业的表格提取推荐使用 pdfplumber 或 Camelot。
    """
    doc = fitz.open(pdf_path)
    page = doc[page_number]
    
    # 查找表格：寻找排列整齐的文本行
    tabs = page.find_tables()
    tables = []
    
    for tab in tabs:
        # tab.extract() 返回二维列表
        table_data = tab.extract()
        tables.append(table_data)
    
    doc.close()
    return tables


def extract_metadata(pdf_path: str) -> dict:
    """提取 PDF 元数据"""
    doc = fitz.open(pdf_path)
    metadata = doc.metadata  # 包含 title, author, subject, keywords 等
    toc = doc.get_toc()      # 目录 (Table of Contents)
    doc.close()
    
    return {
        "metadata": metadata,
        "toc": toc,
        "page_count": doc.page_count if hasattr(doc, 'page_count') else None
    }


# ========== 使用示例 ==========
if __name__ == "__main__":
    # 基本文本提取
    text = extract_text_from_pdf("sample.pdf")
    print(f"提取文本长度: {len(text)} 字符")
    print(text[:500])  # 预览前 500 字符
    
    # 布局感知提取
    blocks = extract_text_with_layout("sample.pdf")
    print(f"共提取 {len(blocks)} 个文本块")
    
    # 表格提取
    tables = extract_table_from_pdf("sample.pdf", 0)
    for t in tables:
        for row in t[:3]:  # 前 3 行
            print(row)
    
    # 元数据
    meta = extract_metadata("sample.pdf")
    print(f"标题: {meta['metadata'].get('title', 'N/A')}")
```

**pdfplumber 的补充用法**：当需要更精确的表格提取或字符级定位时，pdfplumber 是更好的选择。

```python
# ch04_pdfplumber.py
"""pdfplumber 示例：精确的表格与文本提取"""

import pdfplumber

def extract_tables_with_pdfplumber(pdf_path: str) -> list[list[list]]:
    """
    使用 pdfplumber 提取所有页面中的表格。
    pdfplumber 对表格有更好的检测算法。
    
    Returns:
        列表的列表的列表：页面 → 表格 → 行 → 单元格
    """
    all_tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            tables = page.extract_tables()
            for table in tables:
                all_tables.append({
                    "page": page_num + 1,
                    "data": table
                })
    return all_tables


def extract_text_with_positions(pdf_path: str) -> list[dict]:
    """
    提取每个字符的精确位置信息。
    对于需要精确还原布局的场景非常有用。
    """
    chars = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            for char in page.chars:
                chars.append({
                    "text": char["text"],
                    "x0": char["x0"],
                    "y0": char["y0"],
                    "x1": char["x1"],
                    "y1": char["y1"],
                    "fontname": char.get("fontname", ""),
                    "size": char.get("size", 0),
                    "page": page.page_number
                })
    return chars
```

**PDF 解析的常见问题与对策**：

| 问题 | 原因 | 对策 |
|------|------|------|
| 文本乱码 | 字体编码不规范 | 尝试多种提取模式 (`text`/`raw`/`dict`) |
| 提取顺序错乱 | PDF 内部文本排序与阅读顺序不一致 | 按坐标排序，或使用布局分析 |
| 表格丢失 | 表格以绝对坐标绘制 | 使用 pdfplumber / Camelot |
| 中文显示异常 | 缺少中文字体映射 | 检查 `fitz.Tools.set_unicode(true)` |
| 扫描件无文本 | 本质是图片 | 先 OCR（见 4.3 节） |

### 4.2.2 Word 文档解析 (python-docx)

Microsoft Word 格式（.docx）实际上是一个 ZIP 包，内部包含 XML 格式的文档内容。python-docx 库可以读取并操作 .docx 文件。

```python
# ch04_word_parsing.py
"""Word 文档解析示例：使用 python-docx"""

from docx import Document
from docx.oxml.ns import qn
import re

def extract_text_from_docx(docx_path: str) -> str:
    """
    提取 Word 文档的全部文本。
    python-docx 按段落组织文本。
    """
    doc = Document(docx_path)
    paragraphs = []
    
    for para in doc.paragraphs:
        text = para.text.strip()
        if text:  # 跳过空段落
            paragraphs.append(text)
    
    return "\n".join(paragraphs)


def extract_structured_docx(docx_path: str) -> dict:
    """
    提取 Word 文档的结构化内容，包括标题层级。
    python-docx 通过 Style 名称识别标题级别。
    """
    doc = Document(docx_path)
    content = {
        "paragraphs": [],
        "tables": [],
        "headers_footers": {},
        "sections": []
    }
    
    # 提取段落及其样式
    for para in doc.paragraphs:
        style_name = para.style.name if para.style else "Normal"
        text = para.text.strip()
        
        if not text:
            continue
        
        # 判断是否为标题
        heading_level = 0
        if style_name.startswith("Heading"):
            try:
                # "Heading 1" → 1, "Heading 2" → 2
                heading_level = int(style_name.split()[-1])
            except (ValueError, IndexError):
                heading_level = 1
        
        content["paragraphs"].append({
            "text": text,
            "style": style_name,
            "heading_level": heading_level,
            "runs": [
                {
                    "text": run.text,
                    "bold": run.bold,
                    "italic": run.italic,
                    "font": run.font.name if run.font.name else None
                }
                for run in para.runs
            ]
        })
    
    # 提取表格
    for table in doc.tables:
        table_data = []
        for row in table.rows:
            row_data = [cell.text.strip() for cell in row.cells]
            table_data.append(row_data)
        content["tables"].append(table_data)
    
    # 提取页眉页脚
    for section in doc.sections:
        header = section.header
        footer = section.footer
        content["headers_footers"][f"section_{section.start_type}"] = {
            "header": "".join(p.text for p in header.paragraphs if p.text),
            "footer": "".join(p.text for p in footer.paragraphs if p.text)
        }
    
    return content


def extract_images_from_docx(docx_path: str, output_dir: str) -> list[str]:
    """
    从 Word 文档中提取嵌入的图片。
    .docx 本质是 ZIP 包，图片存储在 word/media/ 目录下。
    """
    import zipfile
    from pathlib import Path
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    extracted_images = []
    with zipfile.ZipFile(docx_path, 'r') as z:
        for name in z.namelist():
            if name.startswith("word/media/") and not name.endswith("/"):
                # 保留原始文件名
                image_data = z.read(name)
                image_name = Path(name).name
                image_path = output_path / image_name
                with open(image_path, "wb") as f:
                    f.write(image_data)
                extracted_images.append(str(image_path))
    
    return extracted_images


# ========== 使用示例 ==========
if __name__ == "__main__":
    text = extract_text_from_docx("report.docx")
    print(f"Word 文档文本长度: {len(text)} 字符")
    
    structured = extract_structured_docx("report.docx")
    headings = [p for p in structured["paragraphs"] if p["heading_level"] > 0]
    print(f"共发现 {len(headings)} 个标题")
    for h in headings:
        print(f"  H{h['heading_level']}: {h['text'][:60]}")
    
    images = extract_images_from_docx("report.docx", "./extracted_images")
    print(f"提取了 {len(images)} 张图片")
```

**python-docx 的限制**：
- 不支持 .doc（旧版 Word 格式），需先用 LibreOffice 或 Win32 COM 转换
- 对复杂格式（文本框、数学公式、图表）支持有限
- 无法直接渲染页面布局

### 4.2.3 Excel 解析 (openpyxl)

Excel 文件在 RAG 场景中经常被忽略，但企业数据中有大量信息存储于电子表格中。openpyxl 是处理 .xlsx 文件的首选库。

```python
# ch04_excel_parsing.py
"""Excel 解析示例：使用 openpyxl"""

from openpyxl import load_workbook
from openpyxl.utils import get_column_letter
from typing import Any

def extract_excel_text(excel_path: str) -> dict[str, str]:
    """
    提取 Excel 中所有工作表的内容，合并为文本。
    
    策略：每个工作表作为一个独立文档，
    每行格式化为 "列名: 值" 的形式。
    """
    wb = load_workbook(excel_path, data_only=True)  # data_only=True 获取计算后的值
    result = {}
    
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        lines = []
        headers = []
        
        # 读取表头（第一行）
        first_row = True
        for row in ws.iter_rows(min_row=1, values_only=False):
            if first_row:
                headers = [cell.value for cell in row]
                first_row = False
                continue
            
            # 将每行格式化为 "列名: 值"
            row_values = []
            for header, cell in zip(headers, row):
                if cell.value is not None and header is not None:
                    row_values.append(f"{header}: {cell.value}")
            
            if row_values:
                lines.append(" | ".join(row_values))
        
        result[sheet_name] = "\n".join(lines)
    
    wb.close()
    return result


def extract_excel_structured(excel_path: str) -> dict[str, list[dict[str, Any]]]:
    """
    以结构化方式提取 Excel 数据，保留行列关系。
    适合需要精确引用单元格数据的场景。
    """
    wb = load_workbook(excel_path, data_only=True)
    result = {}
    
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        rows_data = []
        headers = []
        
        for row_idx, row in enumerate(ws.iter_rows(values_only=True), start=1):
            if row_idx == 1:
                # 第一行为表头
                headers = [str(cell) if cell is not None else f"Column_{idx}"
                          for idx, cell in enumerate(row)]
                continue
            
            row_dict = {}
            for col_idx, (header, cell_value) in enumerate(zip(headers, row)):
                if cell_value is not None:
                    row_dict[header] = cell_value
            
            if row_dict:  # 跳过全空行
                rows_data.append(row_dict)
        
        result[sheet_name] = {
            "headers": headers,
            "rows": rows_data,
            "row_count": len(rows_data)
        }
    
    wb.close()
    return result


def extract_excel_metadata(excel_path: str) -> dict:
    """提取 Excel 文件元数据"""
    wb = load_workbook(excel_path)
    props = wb.properties
    
    return {
        "title": props.title,
        "creator": props.creator,
        "description": props.description,
        "created": props.created.isoformat() if props.created else None,
        "modified": props.modified.isoformat() if props.modified else None,
        "sheets": wb.sheetnames,
        "sheet_count": len(wb.sheetnames)
    }


def merge_excel_to_documents(excel_path: str) -> list[dict]:
    """
    将 Excel 转换为 RAG 就绪的文档列表。
    每个工作表作为独立文档，保留上下文。
    """
    wb = load_workbook(excel_path, data_only=True)
    documents = []
    
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        
        # 构建该工作表的文本表示
        lines = [f"# 工作表: {sheet_name}"]
        lines.append(f"# 行数: {ws.max_row}, 列数: {ws.max_column}")
        lines.append("")
        
        for row in ws.iter_rows(values_only=True):
            # 过滤 None 值，格式化为 CSV 风格
            row_str = ", ".join(
                str(cell) if cell is not None else ""
                for cell in row
            )
            if row_str.strip():
                lines.append(row_str)
        
        documents.append({
            "source": excel_path,
            "sheet": sheet_name,
            "content": "\n".join(lines),
            "metadata": {
                "type": "excel",
                "sheet": sheet_name,
                "rows": ws.max_row,
                "columns": ws.max_column
            }
        })
    
    wb.close()
    return documents


# ========== 使用示例 ==========
if __name__ == "__main__":
    docs = merge_excel_to_documents("sales_data.xlsx")
    print(f"共生成 {len(docs)} 个文档")
    for doc in docs:
        print(f"  工作表: {doc['sheet']}, 内容长度: {len(doc['content'])}")
```

### 4.2.4 HTML 解析 (BeautifulSoup4)

HTML 是 Web 上最丰富的信息载体。与 PDF 不同，HTML 有明确的标签结构（`<h1>`、`<p>`、`<table>`），解析的关键在于"结构化提取"而非"文本还原"。

```python
# ch04_html_parsing.py
"""HTML 解析示例：使用 BeautifulSoup4"""

from bs4 import BeautifulSoup, Tag
import re
from urllib.parse import urljoin

def extract_text_from_html(html_content: str, base_url: str = "") -> str:
    """
    从 HTML 中提取纯文本。
    使用 .get_text() 方法并清理多余空白。
    """
    soup = BeautifulSoup(html_content, "html.parser")
    
    # 移除脚本和样式内容
    for element in soup(["script", "style", "meta", "link", "noscript"]):
        element.decompose()
    
    text = soup.get_text(separator="\n", strip=True)
    # 清理多余空行
    text = re.sub(r"\n{3,}", "\n\n", text)
    
    return text


def extract_structured_html(html_content: str, base_url: str = "") -> dict:
    """
    结构化提取 HTML 内容。
    保留标题层级、段落、列表、表格等语义信息。
    """
    soup = BeautifulSoup(html_content, "html.parser")
    
    # 移除干扰元素
    for tag in soup(["script", "style", "nav", "footer", "aside"]):
        tag.decompose()
    
    result = {
        "title": "",
        "headings": [],
        "paragraphs": [],
        "lists": [],
        "tables": [],
        "links": [],
        "metadata": {}
    }
    
    # 提取标题
    title_tag = soup.find("title")
    if title_tag:
        result["title"] = title_tag.get_text(strip=True)
    
    # 提取元数据 (Open Graph / Meta)
    for meta in soup.find_all("meta"):
        name = meta.get("name") or meta.get("property", "")
        content = meta.get("content", "")
        if name and content:
            result["metadata"][name] = content
    
    # 提取标题结构
    for tag_name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
        for heading in soup.find_all(tag_name):
            level = int(tag_name[1])
            text = heading.get_text(strip=True)
            if text:
                result["headings"].append({
                    "level": level,
                    "text": text,
                    "tag": tag_name
                })
    
    # 提取段落
    for p in soup.find_all("p"):
        text = p.get_text(strip=True)
        if text and len(text) > 10:  # 过滤过短内容
            result["paragraphs"].append(text)
    
    # 提取列表
    for list_tag in soup.find_all(["ul", "ol"]):
        items = []
        for li in list_tag.find_all("li"):
            items.append(li.get_text(strip=True))
        if items:
            result["lists"].append({
                "type": list_tag.name,  # "ul" 或 "ol"
                "items": items
            })
    
    # 提取表格
    for table in soup.find_all("table"):
        table_data = []
        for row in table.find_all("tr"):
            cells = []
            for cell in row.find_all(["td", "th"]):
                cells.append(cell.get_text(strip=True))
            if cells:
                table_data.append(cells)
        if table_data:
            result["tables"].append(table_data)
    
    # 提取链接
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        text = a_tag.get_text(strip=True)
        if text and href and not href.startswith(("#", "javascript:")):
            full_url = urljoin(base_url, href)
            result["links"].append({
                "text": text,
                "url": full_url
            })
    
    return result


def html_to_markdown(html_content: str) -> str:
    """
    将 HTML 转换为 Markdown 格式。
    保留标题、链接、列表等结构。
    """
    soup = BeautifulSoup(html_content, "html.parser")
    
    # 移除无用元素
    for tag in soup(["script", "style", "nav", "footer", "aside"]):
        tag.decompose()
    
    lines = []
    
    def process_element(el, indent_level=0):
        """递归处理 HTML 元素"""
        if isinstance(el, str):
            text = el.strip()
            if text:
                lines.append(("  " * indent_level) + text)
            return
        
        tag_name = el.name if hasattr(el, 'name') else None
        
        if tag_name in ("h1", "h2", "h3", "h4", "h5", "h6"):
            level = int(tag_name[1])
            text = el.get_text(strip=True)
            if text:
                lines.append(f"{'#' * level} {text}")
                lines.append("")
        
        elif tag_name == "p":
            text = el.get_text(strip=True)
            if text:
                lines.append(text)
                lines.append("")
        
        elif tag_name == "a":
            href = el.get("href", "")
            text = el.get_text(strip=True)
            if text and href:
                lines[-1] = lines[-1] + f"[{text}]({href})"
            elif text:
                lines[-1] = lines[-1] + text
        
        elif tag_name in ("ul", "ol"):
            for li in el.find_all("li", recursive=False):
                prefix = "- " if tag_name == "ul" else "1. "
                text = li.get_text(strip=True)
                if text:
                    lines.append(("  " * indent_level) + prefix + text)
                    # 处理嵌套列表
                    nested = li.find(["ul", "ol"])
                    if nested:
                        process_element(nested, indent_level + 1)
            lines.append("")
        
        elif tag_name in ("b", "strong"):
            text = el.get_text(strip=True)
            if text:
                lines[-1] = lines[-1] + f"**{text}**"
        
        elif tag_name in ("i", "em"):
            text = el.get_text(strip=True)
            if text:
                lines[-1] = lines[-1] + f"*{text}*"
        
        elif tag_name == "br":
            lines.append("")
        
        elif tag_name == "img":
            src = el.get("src", "")
            alt = el.get("alt", "")
            if src:
                lines.append(f"![{alt}]({src})")
        
        elif tag_name == "pre":
            code = el.get_text()
            lines.append("```")
            lines.append(code)
            lines.append("```")
            lines.append("")
        
        elif tag_name == "hr":
            lines.append("---")
            lines.append("")
        
        else:
            # 递归处理子元素
            for child in el.children:
                process_element(child, indent_level)
    
    process_element(soup.body if soup.body else soup)
    return "\n".join(lines)


# ========== 使用示例 ==========
if __name__ == "__main__":
    with open("webpage.html", "r", encoding="utf-8") as f:
        html = f.read()
    
    # 纯文本提取
    text = extract_text_from_html(html)
    print(f"提取文本长度: {len(text)}")
    
    # 结构化提取
    structured = extract_structured_html(html)
    print(f"标题: {structured['title']}")
    print(f"段落数: {len(structured['paragraphs'])}")
    print(f"链接数: {len(structured['links'])}")
    
    # 转为 Markdown
    md = html_to_markdown(html)
    print(f"Markdown 长度: {len(md)}")
```

### 4.2.5 Markdown 解析

Markdown 本身已经接近理想的分块粒度 —— 它有明确的标题层级，天然适合按标题分割。但在 RAG 中，我们通常需要自定义分块逻辑而非简单按文件分割。

```python
# ch04_markdown_parsing.py
"""Markdown 解析示例"""

import re
from typing import Optional

def parse_markdown_sections(md_content: str) -> list[dict]:
    """
    按标题层级解析 Markdown 文档。
    返回包含标题、内容和层级的段落列表。
    """
    lines = md_content.split("\n")
    sections = []
    current_section = {
        "heading": "root",
        "level": 0,
        "content": []
    }
    
    heading_pattern = re.compile(r"^(#{1,6})\s+(.+)$")
    
    for line in lines:
        match = heading_pattern.match(line)
        if match:
            # 保存上一节
            if current_section["content"]:
                sections.append(current_section)
            
            level = len(match.group(1))
            heading_text = match.group(2).strip()
            current_section = {
                "heading": heading_text,
                "level": level,
                "content": []
            }
        else:
            current_section["content"].append(line)
    
    # 保存最后一节
    if current_section["content"]:
        sections.append(current_section)
    
    # 合并内容
    for section in sections:
        section["text"] = "\n".join(section["content"]).strip()
        del section["content"]
    
    return sections


def markdown_to_chunks(md_content: str, base_level: int = 1) -> list[dict]:
    """
    将 Markdown 按标题分割为 RAG 块。
    每个块包含其标题上下文。
    """
    sections = parse_markdown_sections(md_content)
    chunks = []
    heading_context = []  # 标题上下文栈
    
    for section in sections:
        level = section["level"]
        
        # 更新上下文栈
        while heading_context and heading_context[-1]["level"] >= level:
            heading_context.pop()
        
        if level >= base_level:
            heading_context.append({
                "level": level,
                "heading": section["heading"]
            })
        
        # 构建完整上下文标题
        context_heading = " > ".join(
            h["heading"] for h in heading_context
        )
        
        # 生成块内容（包含上下文标题）
        chunk_text = f"# {context_heading}\n\n{section['text']}" if context_heading else section["text"]
        
        chunks.append({
            "text": chunk_text.strip(),
            "heading": section["heading"],
            "level": section["level"],
            "context": context_heading
        })
    
    return chunks


# ========== 使用示例 ==========
if __name__ == "__main__":
    with open("document.md", "r", encoding="utf-8") as f:
        md = f.read()
    
    chunks = markdown_to_chunks(md)
    print(f"共生成 {len(chunks)} 个块")
    for chunk in chunks[:5]:
        print(f"  [{'#' * chunk['level']}] {chunk['heading']}")
        print(f"  上下文: {chunk['context']}")
        print(f"  文本长度: {len(chunk['text'])}")
        print()
```

### 4.2.6 统一解析器接口

在实际项目中，我们需要一个统一的解析器接口来屏蔽不同格式的差异：

```python
# ch04_unified_parser.py
"""统一文档解析器接口"""

from pathlib import Path
from typing import Optional, Protocol
from dataclasses import dataclass, field

@dataclass
class ParsedDocument:
    """统一解析结果"""
    content: str
    metadata: dict = field(default_factory=dict)
    source: str = ""
    format: str = ""


class DocumentParser(Protocol):
    """解析器协议"""
    def parse(self, file_path: str) -> ParsedDocument: ...


class PDFParser:
    def parse(self, file_path: str) -> ParsedDocument:
        import fitz
        doc = fitz.open(file_path)
        text = "\n\n".join(page.get_text("text") for page in doc)
        meta = {
            "pages": len(doc),
            "title": doc.metadata.get("title", ""),
            "author": doc.metadata.get("author", "")
        }
        doc.close()
        return ParsedDocument(
            content=text,
            metadata=meta,
            source=file_path,
            format="pdf"
        )


class DocxParser:
    def parse(self, file_path: str) -> ParsedDocument:
        from docx import Document
        doc = Document(file_path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        text = "\n".join(paragraphs)
        return ParsedDocument(
            content=text,
            metadata={"paragraphs": len(paragraphs)},
            source=file_path,
            format="docx"
        )


class HTMLParser:
    def parse(self, file_path: str) -> ParsedDocument:
        from bs4 import BeautifulSoup
        with open(file_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        title = soup.title.string if soup.title else ""
        return ParsedDocument(
            content=text,
            metadata={"title": title},
            source=file_path,
            format="html"
        )


class MarkdownParser:
    def parse(self, file_path: str) -> ParsedDocument:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        return ParsedDocument(
            content=text,
            metadata={"format": "markdown"},
            source=file_path,
            format="markdown"
        )


class ExcelParser:
    def parse(self, file_path: str) -> ParsedDocument:
        from openpyxl import load_workbook
        wb = load_workbook(file_path, data_only=True)
        texts = []
        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            texts.append(f"## Sheet: {sheet_name}")
            for row in ws.iter_rows(values_only=True):
                row_str = ", ".join(str(c) for c in row if c is not None)
                if row_str.strip():
                    texts.append(row_str)
        wb.close()
        return ParsedDocument(
            content="\n".join(texts),
            metadata={"sheets": wb.sheetnames},
            source=file_path,
            format="excel"
        )


# 解析器注册表
PARSER_REGISTRY: dict[str, DocumentParser] = {
    ".pdf": PDFParser(),
    ".docx": DocxParser(),
    ".html": HTMLParser(),
    ".htm": HTMLParser(),
    ".md": MarkdownParser(),
    ".markdown": MarkdownParser(),
    ".xlsx": ExcelParser(),
    ".xls": ExcelParser(),
}

def parse_document(file_path: str) -> ParsedDocument:
    """统一文档解析入口"""
    ext = Path(file_path).suffix.lower()
    parser = PARSER_REGISTRY.get(ext)
    if parser is None:
        raise ValueError(f"不支持的文件格式: {ext}")
    return parser.parse(file_path)


# ========== 使用示例 ==========
if __name__ == "__main__":
    for file_path in ["doc.pdf", "report.docx", "page.html", "notes.md", "data.xlsx"]:
        try:
            doc = parse_document(file_path)
            print(f"[{doc.format}] {Path(file_path).name}: {len(doc.content)} 字符")
        except Exception as e:
            print(f"解析失败 {file_path}: {e}")
```

---

## 4.3 OCR 与布局分析 (OCR and Layout Analysis)

当文档是扫描件或图片时，无法直接提取文本。此时需要 OCR（光学字符识别，Optical Character Recognition）技术。更进一步，**布局分析**可以识别文档中的段落、标题、表格、图片区域，并恢复正确的阅读顺序。

### 4.3.1 PaddleOCR

PaddleOCR 是百度开源的 OCR 工具包，对中文支持极好，支持 80+ 语言的文本识别。

```python
# ch04_paddleocr_example.py
"""PaddleOCR 集成示例"""

from paddleocr import PaddleOCR
import numpy as np
from PIL import Image

def init_ocr(lang: str = "ch") -> PaddleOCR:
    """
    初始化 PaddleOCR 引擎。
    
    Args:
        lang: 语言代码，'ch' 表示中文，'en' 表示英文
    
    Returns:
        PaddleOCR 实例
    """
    ocr = PaddleOCR(
        use_angle_cls=True,  # 启用文字方向分类
        lang=lang,
        use_gpu=False,       # CPU 推理；如果有 GPU 可设为 True
        show_log=False       # 关闭详细日志
    )
    return ocr


def ocr_image(ocr: PaddleOCR, image_path: str) -> list[dict]:
    """
    对单张图片执行 OCR。
    
    Returns:
        包含识别结果的列表，每个元素:
        {
            "text": 识别文本,
            "confidence": 置信度,
            "bbox": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]  # 四个角坐标
        }
    """
    result = ocr.ocr(image_path, cls=True)
    
    parsed = []
    for line in result[0]:  # result[0] 是第一个（也是唯一）图片的结果
        bbox, (text, confidence) = line
        parsed.append({
            "text": text,
            "confidence": confidence,
            "bbox": bbox
        })
    
    return parsed


def ocr_with_layout(ocr: PaddleOCR, image_path: str) -> str:
    """
    执行 OCR 并按阅读顺序合并文本。
    根据垂直位置（y 坐标）对文本行排序。
    """
    results = ocr_image(ocr, image_path)
    
    # 按 y 坐标（大致行）和 x 坐标排序
    # 先按 y 中心点聚类为"行"，再按 x 排序
    if not results:
        return ""
    
    # 按 y 坐标均值分组
    y_centers = [np.mean([p[1] for p in r["bbox"]]) for r in results]
    
    # 简单的排序：按 y 排序，相同 y 范围按 x 排序
    sorted_results = sorted(
        results,
        key=lambda r: (np.mean([p[1] for p in r["bbox"]]),
                       np.mean([p[0] for p in r["bbox"]]))
    )
    
    lines = []
    prev_y = None
    current_line = []
    
    for r in sorted_results:
        y_center = np.mean([p[1] for p in r["bbox"]])
        
        if prev_y is None or abs(y_center - prev_y) < 20:  # 同一行阈值
            current_line.append(r["text"])
        else:
            lines.append(" ".join(current_line))
            current_line = [r["text"]]
        
        prev_y = y_center
    
    if current_line:
        lines.append(" ".join(current_line))
    
    return "\n".join(lines)


# ========== 使用示例 ==========
if __name__ == "__main__":
    ocr = init_ocr("ch")
    text = ocr_with_layout(ocr, "scanned_document.png")
    print(text[:1000])
```

### 4.3.2 Tesseract OCR

Tesseract 是 Google 维护的老牌 OCR 引擎，通过 `pytesseract` 在 Python 中使用。

```python
# ch04_tesseract_example.py
"""Tesseract OCR 集成示例"""

import pytesseract
from PIL import Image
import numpy as np

def tesseract_ocr(image_path: str, lang: str = "chi_sim+eng") -> str:
    """
    使用 Tesseract 执行 OCR。
    
    Args:
        image_path: 图片路径
        lang: 语言，'chi_sim' 简体中文，'eng' 英文，'+' 连接多语言
    
    Returns:
        识别文本
    """
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image, lang=lang)
    return text.strip()


def tesseract_with_details(image_path: str, lang: str = "chi_sim+eng") -> dict:
    """
    获取包含详细位置信息的 OCR 结果。
    """
    image = Image.open(image_path)
    
    # 获取单词级数据
    data = pytesseract.image_to_data(image, lang=lang, output_type=pytesseract.Output.DICT)
    
    words = []
    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        if text and int(data["conf"][i]) > 30:  # 过滤低置信度结果
            words.append({
                "text": text,
                "confidence": int(data["conf"][i]),
                "bbox": {
                    "x": data["left"][i],
                    "y": data["top"][i],
                    "w": data["width"][i],
                    "h": data["height"][i]
                },
                "block_num": data["block_num"][i],
                "line_num": data["line_num"][i]
            })
    
    # 获取段落级数据
    paragraphs = pytesseract.image_to_string(image, lang=lang).strip().split("\n\n")
    
    return {
        "full_text": pytesseract.image_to_string(image, lang=lang).strip(),
        "words": words,
        "paragraphs": [p.strip() for p in paragraphs if p.strip()]
    }


def preprocess_image_for_ocr(image_path: str, output_path: str = None) -> np.ndarray:
    """
    OCR 前图像预处理：提高识别准确率。
    
    步骤:
    1. 灰度化
    2. 二值化 (阈值处理)
    3. 去噪
    4. 倾斜校正
    """
    import cv2
    
    img = cv2.imread(image_path)
    
    # 灰度化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 二值化 (Otsu 自动阈值)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 去噪 (中值滤波)
    denoised = cv2.medianBlur(binary, 3)
    
    if output_path:
        cv2.imwrite(output_path, denoised)
    
    return denoised


# ========== 使用示例 ==========
if __name__ == "__main__":
    # 预处理
    preprocessed = preprocess_image_for_ocr("scan.jpg", "scan_processed.png")
    
    # OCR 识别
    text = tesseract_ocr("scan_processed.png")
    print(text[:1000])
    
    # 获取详细数据
    details = tesseract_with_details("scan_processed.png")
    print(f"识别单词数: {len(details['words'])}")
    print(f"段落数: {len(details['paragraphs'])}")
```

### 4.3.3 布局分析 (Layout Analysis)

对于复杂的文档（多栏排版、混合图文），仅 OCR 是不够的。我们需要**布局分析**来识别不同区域的功能（标题、正文、表格、图片），并恢复正确的阅读顺序。

```python
# ch04_layout_analysis.py
"""文档布局分析示例"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class LayoutRegion:
    """布局区域"""
    label: str         # 区域类型: title, text, table, figure, header, footer
    bbox: tuple        # (x1, y1, x2, y2)
    text: str = ""
    confidence: float = 0.0


class SimpleLayoutAnalyzer:
    """
    基于规则的简易布局分析器。
    适用于 PDF 或 OCR 结果的后处理。
    """
    
    def __init__(self, page_width: float, page_height: float):
        self.page_width = page_width
        self.page_height = page_height
    
    def classify_block(self, x0: float, y0: float, x1: float, y1: float,
                       text: str, font_size: float = None) -> str:
        """
        根据位置和内容判断块类型。
        """
        # 页眉/页脚检测（靠近页面顶部/底部）
        if y0 < self.page_height * 0.05:
            return "header"
        if y1 > self.page_height * 0.95:
            return "footer"
        
        # 标题检测（大字或位于页面顶部）
        if font_size and font_size > 14:
            return "title"
        if y0 < self.page_height * 0.15 and len(text) < 100:
            return "title"
        
        # 表格检测（存在大量对齐的短文本）
        lines = text.split("\n")
        if len(lines) > 2:
            # 检查是否每行都有相似的模式（表格特征）
            line_lengths = [len(l.split()) for l in lines if l.strip()]
            if line_lengths and max(line_lengths) - min(line_lengths) < 3:
                return "table"
        
        # 默认视为正文
        return "text"
    
    def analyze(self, blocks: list[dict]) -> list[LayoutRegion]:
        """
        分析文本块列表，返回带标签的区域。
        
        Args:
            blocks: 每个块包含 bbox、text、font_size
        
        Returns:
            带标签的区域列表
        """
        regions = []
        for block in blocks:
            label = self.classify_block(
                block["bbox"][0], block["bbox"][1],
                block["bbox"][2], block["bbox"][3],
                block["text"],
                block.get("font_size")
            )
            regions.append(LayoutRegion(
                label=label,
                bbox=block["bbox"],
                text=block["text"]
            ))
        return regions
    
    def restore_reading_order(self, regions: list[LayoutRegion]) -> list[LayoutRegion]:
        """
        恢复正确的阅读顺序。
        多栏布局需要先从左到右、再从上到下排序。
        
        策略：
        1. 检测是否为多栏布局
        2. 如果是，按"先列后行"排序
        3. 否则按"先行后列"排序
        """
        if not regions:
            return regions
        
        # 检测多栏布局：检查 x 坐标分布
        x_centers = [(r.bbox[0] + r.bbox[2]) / 2 for r in regions]
        
        # 如果块分布在两个明显的 x 区域，则视为双栏
        x_threshold = self.page_width / 3
        left_blocks = [r for r in regions if (r.bbox[0] + r.bbox[2]) / 2 < x_threshold * 2]
        right_blocks = [r for r in regions if (r.bbox[0] + r.bbox[2]) / 2 >= x_threshold * 2]
        
        if left_blocks and right_blocks:
            # 双栏排序：先左栏后右栏，每栏内按 y 排序
            left_sorted = sorted(left_blocks, key=lambda r: (r.bbox[1] + r.bbox[3]) / 2)
            right_sorted = sorted(right_blocks, key=lambda r: (r.bbox[1] + r.bbox[3]) / 2)
            return left_sorted + right_sorted
        else:
            # 单栏：直接按 y 排序
            return sorted(regions, key=lambda r: (r.bbox[1] + r.bbox[3]) / 2)


# ========== 使用示例 ==========
if __name__ == "__main__":
    # 模拟从 PDF 提取的块
    blocks = [
        {"bbox": (50, 30, 550, 60), "text": "文档标题", "font_size": 18},
        {"bbox": (50, 80, 300, 200), "text": "左栏正文...\n多行内容", "font_size": 11},
        {"bbox": (320, 80, 550, 200), "text": "右栏正文...\n多行内容", "font_size": 11},
        {"bbox": (50, 220, 550, 230), "text": "页脚信息", "font_size": 8},
    ]
    
    analyzer = SimpleLayoutAnalyzer(page_width=600, page_height=800)
    regions = analyzer.analyze(blocks)
    
    for r in regions:
        print(f"[{r.label}] {r.text[:50]}")
    
    ordered = analyzer.restore_reading_order(regions)
    print("\n恢复阅读顺序后:")
    for r in ordered:
        print(f"  [{r.label}] {r.text[:50]}")
```

**布局分析的进阶方向**：

对于更复杂的布局分析需求，推荐使用以下工具：

| 工具 | 适用场景 | 说明 |
|------|----------|------|
| **LayoutParser** | 通用文档布局 | 基于深度学习的版面分析工具包 |
| **Unstructured.io** | 企业文档处理 | 集成了布局分析、OCR、分块的 Pipeline |
| **Tesseract + page segmentation** | 简单版面 | Tesseract 内置的页面分割模式 |
| **DocTR** | 端到端文档解析 | 基于 Transformer 的文档理解 |
| **pdfplumber + 规则** | PDF 表格提取 | 精确的表格检测与提取 |

---

## 4.4 数据清洗 (Data Cleaning)

解析后的文本通常是"脏"的 —— 包含重复内容、无关噪声、格式不统一，甚至可能包含敏感信息（PII，Personally Identifiable Information）。数据清洗是文档处理中不可跳过的一步。

### 4.4.1 去重 (Deduplication)

在文档集中，重复内容会严重降低检索质量。MinHash 和 SimHash 是两种广泛使用的近似去重算法。

#### MinHash 去重

MinHash 通过 Jaccard 相似度判断两篇文档的相似性。核心思想：对文档的 shingle（n-gram）集合进行多次哈希，保留每个哈希的最小值作为"签名"，签名相同的文档视为近似重复。

```python
# ch04_minhash_dedup.py
"""MinHash 去重实现"""

import hashlib
from typing import Set, List
import mmh3  # 需要 pip install mmh3
from dataclasses import dataclass

@dataclass
class Document:
    """待处理的文档"""
    id: str
    text: str
    metadata: dict = None


class MinHashDeduplicator:
    """
    MinHash 近似去重器。
    
    原理：
    1. 将文档分词为 shingle（n-gram）集合
    2. 对每个 shingle 应用 k 个不同的哈希函数
    3. 每个哈希函数取最小值，构成 k 维签名
    4. 签名相同的文档视为近似重复
    """
    
    def __init__(self, num_hashes: int = 128, shingle_size: int = 5):
        """
        Args:
            num_hashes: 哈希函数数量（签名维度），越大越精确但越慢
            shingle_size: n-gram 的大小（以词为单位）
        """
        self.num_hashes = num_hashes
        self.shingle_size = shingle_size
        # 预生成哈希函数的种子
        self.hash_seeds = list(range(num_hashes))
    
    def _tokenize(self, text: str) -> Set[str]:
        """分词并生成 shingle"""
        # 简单分词：按空白分割
        words = text.lower().split()
        shingles = set()
        for i in range(len(words) - self.shingle_size + 1):
            shingle = " ".join(words[i:i + self.shingle_size])
            shingles.add(shingle)
        return shingles
    
    def _minhash_signature(self, shingles: Set[str]) -> List[int]:
        """
        计算文档的 MinHash 签名。
        对每个 shingle 应用 num_hashes 个哈希函数，取每维的最小值。
        """
        signature = [float('inf')] * self.num_hashes
        
        for shingle in shingles:
            for i, seed in enumerate(self.hash_seeds):
                # 使用 mmh3 作为哈希函数
                hash_val = mmh3.hash(shingle, seed=seed, signed=False)
                if hash_val < signature[i]:
                    signature[i] = hash_val
        
        return signature
    
    def _jaccard_from_signatures(self, sig1: List[int], sig2: List[int]) -> float:
        """
        从签名估计 Jaccard 相似度。
        签名中相同位置值相等的比例 ≈ Jaccard 相似度。
        """
        if len(sig1) != len(sig2):
            raise ValueError("签名维度必须相同")
        equal_count = sum(1 for a, b in zip(sig1, sig2) if a == b)
        return equal_count / len(sig1)
    
    def deduplicate(self, documents: List[Document], threshold: float = 0.8) -> List[Document]:
        """
        去重：保留与已有文档 Jaccard 相似度低于 threshold 的文档。
        
        Args:
            documents: 文档列表
            threshold: 相似度阈值，高于此值视为重复
        
        Returns:
            去重后的文档列表
        """
        if not documents:
            return []
        
        # 计算所有文档的签名
        signatures = []
        for doc in documents:
            shingles = self._tokenize(doc.text)
            sig = self._minhash_signature(shingles)
            signatures.append(sig)
        
        # LSH (Locality-Sensitive Hashing) 加速
        # 简化实现：逐个比较
        kept = []
        kept_signatures = []
        
        for i, doc in enumerate(documents):
            is_duplicate = False
            for kept_sig in kept_signatures:
                similarity = self._jaccard_from_signatures(signatures[i], kept_sig)
                if similarity >= threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                kept.append(doc)
                kept_signatures.append(signatures[i])
        
        return kept


# ========== 使用示例 ==========
if __name__ == "__main__":
    docs = [
        Document("1", "RAG 系统将检索与生成结合，提升问答质量"),
        Document("2", "RAG 系统将检索与生成结合，提升问答质量。这是第二句。"),
        Document("3", "完全不同的内容：机器学习基础理论"),
    ]
    
    dedup = MinHashDeduplicator(num_hashes=64, shingle_size=2)
    filtered = dedup.deduplicate(docs, threshold=0.7)
    
    print(f"去重前: {len(docs)} 篇")
    print(f"去重后: {len(filtered)} 篇")
    for doc in filtered:
        print(f"  保留: {doc.text[:60]}")
```

#### SimHash 去重

SimHash 是 Google 提出的另一种近似去重算法，特别适合海量文本的去重。它将文本映射为一个固定长度的"指纹"，通过海明距离（Hamming distance）衡量相似度。

```python
# ch04_simhash_dedup.py
"""SimHash 去重实现"""

import hashlib
from typing import List


class SimHash:
    """
    SimHash 实现。
    
    原理：
    1. 对每个特征（词）计算 hash，得到 f 位二进制值
    2. 对 hash 的每一位：为 1 则 +weight，为 0 则 -weight
    3. 累加所有特征后，正数为 1，负数为 0，得到 f 位指纹
    4. 通过海明距离衡量相似度
    """
    
    def __init__(self, f: int = 64):
        """
        Args:
            f: 指纹位数，常用 64 或 128
        """
        self.f = f
    
    def _hash_string(self, text: str) -> int:
        """计算字符串的 MD5 hash 并返回整数"""
        return int(hashlib.md5(text.encode('utf-8')).hexdigest(), 16)
    
    def _get_features(self, text: str) -> List[tuple[str, int]]:
        """
        提取文本特征及其权重。
        这里用词频作为权重。
        """
        words = text.lower().split()
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        return list(word_counts.items())
    
    def compute_fingerprint(self, text: str) -> int:
        """
        计算文本的 SimHash 指纹。
        
        Returns:
            整数表示的 f 位指纹
        """
        # 初始化累加向量
        v = [0] * self.f
        features = self._get_features(text)
        
        for word, weight in features:
            word_hash = self._hash_string(word)
            
            # 对每一位进行操作
            for i in range(self.f):
                bit = (word_hash >> i) & 1
                if bit == 1:
                    v[i] += weight
                else:
                    v[i] -= weight
        
        # 生成指纹
        fingerprint = 0
        for i in range(self.f):
            if v[i] > 0:
                fingerprint |= (1 << i)
        
        return fingerprint
    
    def hamming_distance(self, fp1: int, fp2: int) -> int:
        """计算两个指纹的海明距离"""
        xor = fp1 ^ fp2
        distance = 0
        while xor:
            distance += 1
            xor &= xor - 1  # 清除最低位的 1
        return distance
    
    def similarity(self, fp1: int, fp2: int) -> float:
        """计算相似度（0~1）"""
        distance = self.hamming_distance(fp1, fp2)
        return 1.0 - distance / self.f
    
    def deduplicate(self, texts: List[str], threshold: float = 0.85) -> List[str]:
        """
        使用 SimHash 去重。
        
        Args:
            texts: 文本列表
            threshold: 相似度阈值
        
        Returns:
            去重后的文本列表
        """
        if not texts:
            return []
        
        fingerprints = [self.compute_fingerprint(t) for t in texts]
        kept = []
        
        for i, text in enumerate(texts):
            is_duplicate = False
            for kept_fp in [self.compute_fingerprint(k) for k in kept]:
                sim = self.similarity(fingerprints[i], kept_fp)
                if sim >= threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                kept.append(text)
        
        return kept


# ========== 使用示例 ==========
if __name__ == "__main__":
    simhash = SimHash(f=64)
    
    texts = [
        "RAG 系统将检索与生成结合提升问答质量",
        "RAG 系统将检索与生成结合并提升问答质量",  # 非常相似
        "深度学习是机器学习的一个子领域"  # 完全不同
    ]
    
    for t in texts:
        fp = simhash.compute_fingerprint(t)
        print(f"文本: {t[:40]}...")
        print(f"  指纹: {bin(fp)}")
    
    # 比较相似度
    sim = simhash.similarity(
        simhash.compute_fingerprint(texts[0]),
        simhash.compute_fingerprint(texts[1])
    )
    print(f"\n文本 0 和 1 的相似度: {sim:.3f}")
    
    sim2 = simhash.similarity(
        simhash.compute_fingerprint(texts[0]),
        simhash.compute_fingerprint(texts[2])
    )
    print(f"文本 0 和 2 的相似度: {sim2:.3f}")
    
    filtered = simhash.deduplicate(texts, threshold=0.8)
    print(f"\n去重后保留 {len(filtered)} 篇")
```

### 4.4.2 格式标准化 (Format Normalization)

不同来源的文本可能有不同的格式约定 —— 全角/半角混用、多余空白、不统一的换行等。

```python
# ch04_format_normalization.py
"""格式标准化工具"""

import re
import unicodedata

def normalize_text(text: str) -> str:
    """
    全面的文本标准化。
    
    步骤：
    1. Unicode 标准化 (NFKC)
    2. 全角符号转半角
    3. 多余空白清理
    4. 换行符统一
    5. 多余空行合并
    """
    # Unicode 标准化
    text = unicodedata.normalize("NFKC", text)
    
    # 全角英文字母、数字、符号转半角
    # 全角范围：FF01-FF5E，对应半角 21-7E
    result = []
    for char in text:
        code = ord(char)
        if 0xFF01 <= code <= 0xFF5E:
            result.append(chr(code - 0xFEE0))
        elif code == 0x3000:  # 全角空格
            result.append(" ")
        else:
            result.append(char)
    text = "".join(result)
    
    # 统一换行符：\r\n → \n, \r → \n
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    
    # 合并连续空白为单空格（保留换行）
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        line = re.sub(r"[ \t]+", " ", line).strip()
        if line:
            cleaned_lines.append(line)
        else:
            cleaned_lines.append("")
    
    # 合并多余空行（最多连续两个换行）
    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    
    # 去除首尾空白
    text = text.strip()
    
    return text


def normalize_url(text: str) -> str:
    """清理文本中的 URL"""
    url_pattern = re.compile(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|'
        r'(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    )
    # 也可以选择保留 URL
    return url_pattern.sub("<URL>", text)


def normalize_whitespace_for_chinese(text: str) -> str:
    """
    针对中文的空白规范化。
    中文和英文之间加空格，中文与标点之间不加。
    """
    # 中文与英文之间加空格
    text = re.sub(r'([一-鿿])([a-zA-Z])', r'\1 \2', text)
    text = re.sub(r'([a-zA-Z])([一-鿿])', r'\1 \2', text)
    
    # 中文与数字之间加空格
    text = re.sub(r'([一-鿿])(\d)', r'\1 \2', text)
    text = re.sub(r'(\d)([一-鿿])', r'\1 \2', text)
    
    return text


# ========== 使用示例 ==========
if __name__ == "__main__":
    dirty = """　　Hello, World!这是　一段　测试文本。
    
    有全角ＡＢＣ和半角ABC混在一起。
    URL: https://example.com/doc
    """
    
    clean = normalize_text(dirty)
    print(f"标准化前:\n{dirty!r}\n")
    print(f"标准化后:\n{clean!r}")
    
    # 中文空格规范化
    chinese_text = "RAG系统与LLM结合使用效果很好"
    normalized = normalize_whitespace_for_chinese(chinese_text)
    print(f"\n中文空格规范:\n  前: {chinese_text}\n  后: {normalized}")
```

### 4.4.3 PII 检测与过滤

在将文档送入 RAG 系统前，必须检测并过滤个人身份信息（PII），以避免隐私泄露。

```python
# ch04_pii_detection.py
"""PII (个人身份信息) 检测与过滤"""

import re
from typing import List, Tuple

class PIIDetector:
    """
    PII 检测器：识别并过滤常见个人信息。
    
    支持的 PII 类型：
    - 手机号码
    - 身份证号
    - 邮箱地址
    - IP 地址
    - 银行卡号
    - 家庭住址（简单启发式）
    """
    
    def __init__(self):
        # 手机号码（中国大陆）
        self.phone_pattern = re.compile(r'1[3-9]\d{9}')
        
        # 身份证号（18 位）
        self.id_card_pattern = re.compile(
            r'[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])'
            r'(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx]'
        )
        
        # 邮箱地址
        self.email_pattern = re.compile(
            r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        )
        
        # IP 地址 (IPv4)
        self.ip_pattern = re.compile(
            r'(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}'
            r'(?:25[0-5]|2[0-4]\d|[01]?\d\d?)'
        )
        
        # 银行卡号（16-19 位数字）
        self.bank_card_pattern = re.compile(r'\b\d{16,19}\b')
        
        # 地址模式（简单：包含"路""号""街道""小区"等）
        self.address_pattern = re.compile(
            r'[一-鿿]*(?:路|街|道|巷|弄|号|号楼|小区|村|'
            r'组|室|栋|单元|层)[一-鿿\d]*'
        )
    
    def detect_all(self, text: str) -> List[Tuple[str, str, int, int]]:
        """
        检测文本中所有 PII。
        
        Returns:
            [(type, value, start, end), ...]
        """
        results = []
        
        patterns = [
            ("phone", self.phone_pattern),
            ("id_card", self.id_card_pattern),
            ("email", self.email_pattern),
            ("ip", self.ip_pattern),
            ("bank_card", self.bank_card_pattern),
            ("address", self.address_pattern),
        ]
        
        for pii_type, pattern in patterns:
            for match in pattern.finditer(text):
                results.append((
                    pii_type,
                    match.group(),
                    match.start(),
                    match.end()
                ))
        
        # 按位置排序
        results.sort(key=lambda x: x[2])
        return results
    
    def filter_pii(self, text: str, replacement: str = "[REDACTED]") -> str:
        """
        过滤文本中的 PII，替换为占位符。
        """
        detections = self.detect_all(text)
        
        # 从后往前替换，避免位置偏移
        result = text
        for pii_type, value, start, end in reversed(detections):
            result = result[:start] + f"[{pii_type}:{replacement}]" + result[end:]
        
        return result
    
    def mask_pii_partial(self, text: str) -> str:
        """
        部分遮盖 PII（保留部分字符用于上下文理解）。
        
        手机: 138****1234
        邮箱: u***@example.com
        身份证: 110***********1234
        """
        # 手机号遮盖
        text = self.phone_pattern.sub(
            lambda m: m.group()[:3] + "****" + m.group()[-4:],
            text
        )
        
        # 邮箱遮盖
        def mask_email(match):
            email = match.group()
            name, domain = email.split("@")
            if len(name) <= 2:
                masked_name = name[0] + "***"
            else:
                masked_name = name[0] + "***" + name[-1]
            return masked_name + "@" + domain
        
        text = self.email_pattern.sub(mask_email, text)
        
        # 身份证遮盖
        text = self.id_card_pattern.sub(
            lambda m: m.group()[:6] + "********" + m.group()[-4:],
            text
        )
        
        return text


# ========== 使用示例 ==========
if __name__ == "__main__":
    detector = PIIDetector()
    
    sample = """
    客户信息：
    姓名：张三
    电话：13812345678
    邮箱：zhangsan@example.com
    身份证：110101199001011234
    地址：北京市海淀区中关村大街1号院
    """
    
    print("检测到的 PII:")
    for pii_type, value, start, end in detector.detect_all(sample):
        print(f"  [{pii_type}] {value}")
    
    print("\n过滤后:")
    print(detector.filter_pii(sample))
    
    print("\n部分遮盖:")
    print(detector.mask_pii_partial(sample))
```

### 4.4.4 噪声过滤 (Noise Filtering)

来自网页、爬虫或其他来源的文本通常包含大量噪声 —— 导航栏、广告、版权声明、重复的页眉页脚等。

```python
# ch04_noise_filtering.py
"""噪声过滤"""

import re
from typing import List

class NoiseFilter:
    """
    文本噪声过滤器。
    
    过滤类型：
    1. 短行噪声（少于 N 个字符的无意义行）
    2. 重复的页眉/页脚
    3. 广告/导航模式
    4. HTML 残留
    5. 特殊字符序列
    """
    
    def __init__(self):
        # 常见的噪声模式
        self.noise_patterns = [
            # 版权信息
            r'copyright\s*(?:©|\(c\))\s*\d{4}',
            r'版权所有',
            r'all rights reserved',
            # 导航文本
            r'^(?:首页|上一页|下一页|末页|返回|搜索|登录|注册)$',
            # 常见广告
            r'(?:广告|推广|赞助|推荐|热门)',
            # 页码
            r'^\s*[-–—]?\s*\d+\s*[-–—]?\s*$',
            # 水平分割线
            r'^[-=*_]{3,}$',
            # HTML 标签残留
            r'<[^>]+>',
            # URL（可选的过滤）
            r'https?://\S+',
        ]
        self.compiled_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.noise_patterns
        ]
    
    def is_noise_line(self, line: str, min_length: int = 5) -> bool:
        """
        判断一行是否为噪声。
        """
        line = line.strip()
        if not line:
            return True  # 空行视为噪声
        
        if len(line) < min_length:
            # 短行且不含中文字符
            if not re.search(r'[一-鿿]', line):
                return True
        
        for pattern in self.compiled_patterns:
            if pattern.search(line):
                return True
        
        # 特殊字符占比过高
        special_ratio = len(re.findall(r'[^a-zA-Z0-9一-鿿\s，。！？、]', line))
        if len(line) > 0 and special_ratio / len(line) > 0.3:
            return True
        
        return False
    
    def filter_noise(self, text: str, min_line_length: int = 5) -> str:
        """
        过滤文本中的噪声行。
        """
        lines = text.split("\n")
        filtered_lines = [line for line in lines
                         if not self.is_noise_line(line, min_line_length)]
        return "\n".join(filtered_lines)
    
    def remove_duplicate_lines(self, text: str) -> str:
        """
        移除重复行（保留首次出现）。
        """
        seen = set()
        lines = text.split("\n")
        unique_lines = []
        
        for line in lines:
            stripped = line.strip()
            if stripped and stripped not in seen:
                seen.add(stripped)
                unique_lines.append(line)
            elif not stripped:
                unique_lines.append(line)  # 保留空行
        
        return "\n".join(unique_lines)
    
    def remove_header_footer(self, text: str, 
                              header_lines: int = 3,
                              footer_lines: int = 3) -> str:
        """
        移除每页的页眉页脚。
        假设多页文档有相同的页眉页脚模式。
        """
        pages = text.split("--- 第")  # 假设有分页标记
        if len(pages) <= 1:
            return text
        
        cleaned_pages = []
        for page in pages:
            lines = page.split("\n")
            if len(lines) > header_lines + footer_lines:
                cleaned = lines[header_lines:-footer_lines]
            else:
                cleaned = lines
            cleaned_pages.append("\n".join(cleaned))
        
        return "\n".join(cleaned_pages)


# ========== 使用示例 ==========
if __name__ == "__main__":
    filter_noise = NoiseFilter()
    
    noisy_text = """
    首页
    ============
    
    RAG 系统将检索与生成结合。
    
    版权所有 © 2024 All Rights Reserved.
    
    广告：点击这里获取更多信息。
    
    下一页
    
    这是正文内容，包含有用信息。
    深度学习是机器学习的子领域。
    
    <script>alert('hello')</script>
    
    页脚 | 关于我们 | 联系方式
    """
    
    print("过滤前:")
    print(repr(noisy_text))
    print("\n过滤后:")
    filtered = filter_noise.filter_noise(noisy_text)
    print(repr(filtered))
```

---

## 4.5 编码检测 (Encoding Detection)

在处理来自不同来源的文件时，编码问题是最常见的"隐形杀手"。一个 UTF-8 编码的文件如果用 GBK 打开，就会出现乱码。chardet 库可以自动检测文本编码。

```python
# ch04_encoding_detection.py
"""编码检测与处理"""

import chardet
from pathlib import Path

def detect_encoding(file_path: str, sample_size: int = 10000) -> dict:
    """
    检测文件的编码方式。
    
    Args:
        file_path: 文件路径
        sample_size: 用于检测的字节数
    
    Returns:
        包含编码信息和置信度的字典
    """
    with open(file_path, "rb") as f:
        raw_data = f.read(sample_size)
    
    result = chardet.detect(raw_data)
    return result  # {"encoding": "utf-8", "confidence": 0.99, "language": ""}


def read_with_auto_encoding(file_path: str) -> str:
    """
    自动检测编码并读取文件。
    如果检测失败，尝试常见编码。
    """
    # 第一步：检测编码
    detected = detect_encoding(file_path)
    encoding = detected.get("encoding")
    confidence = detected.get("confidence", 0)
    
    if encoding and confidence > 0.7:
        try:
            with open(file_path, "r", encoding=encoding) as f:
                return f.read()
        except (UnicodeDecodeError, LookupError):
            pass
    
    # 第二步：回退尝试常见编码
    common_encodings = [
        "utf-8", "gbk", "gb2312", "gb18030",
        "big5", "shift-jis", "euc-kr",
        "latin-1", "cp1252", "iso-8859-1"
    ]
    
    for enc in common_encodings:
        try:
            with open(file_path, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    
    # 最后的回退：latin-1 不会报错但可能产生乱码
    with open(file_path, "r", encoding="latin-1") as f:
        return f.read()


def batch_detect_encodings(file_paths: list[str]) -> dict[str, dict]:
    """
    批量检测多个文件的编码。
    
    Returns:
        {file_path: {"encoding": str, "confidence": float}}
    """
    results = {}
    for fp in file_paths:
        results[fp] = detect_encoding(fp)
    return results


def convert_encoding(input_path: str, output_path: str, 
                     target_encoding: str = "utf-8") -> bool:
    """
    将文件转换为目标编码。
    """
    try:
        content = read_with_auto_encoding(input_path)
        with open(output_path, "w", encoding=target_encoding) as f:
            f.write(content)
        return True
    except Exception as e:
        print(f"转换失败: {e}")
        return False


# ========== 使用示例 ==========
if __name__ == "__main__":
    # 检测编码
    result = detect_encoding("unknown_encoding.txt")
    print(f"检测结果: {result}")
    print(f"编码: {result['encoding']}, 置信度: {result['confidence']:.2%}")
    
    # 自动读取
    content = read_with_auto_encoding("unknown_encoding.txt")
    print(f"读取成功: {len(content)} 字符")
    
    # 批量检测
    files = ["doc1.txt", "doc2.txt", "doc3.txt"]
    results = batch_detect_encodings(files)
    for file_path, info in results.items():
        print(f"{file_path}: {info['encoding']} (置信度: {info['confidence']:.0%})")
```

**常见编码速查表**：

| 编码 | 适用语言 | 特点 |
|------|----------|------|
| UTF-8 | 通用 | 国际标准，兼容 ASCII，推荐使用 |
| GBK | 简体中文 | 兼容 GB2312，Windows 中文版默认 |
| GB2312 | 简体中文 | 早期标准，字符集较小 |
| GB18030 | 简体中文 | 最新国标，兼容 GBK |
| Big5 | 繁体中文 | 港澳台地区使用 |
| Shift-JIS | 日文 | 日本工业标准 |
| EUC-KR | 韩文 | 韩文编码 |
| Latin-1 | 西欧语言 | 单字节，不会报解码错 |

---

## 4.6 分块策略 (Chunking Strategies)

分块（chunking）是 RAG 系统中最关键的设计决策之一。块的大小和分割方式直接影响检索效果和生成质量。分块的目标是在以下三个维度间取得平衡：

1. **语义完整性**：每个块应该包含完整、自洽的信息单元
2. **检索精度**：块太小则上下文不足，块太大则噪声过多
3. **嵌入质量**：嵌入模型通常有最大 token 限制（如 512 或 8192）

### 4.6.1 基于 Token 的分块 (Token-based Chunking)

最简单的分块方式，按 token 数量固定分割。

```python
# ch04_token_chunking.py
"""基于 Token 的分块"""

from typing import List, Optional
import tiktoken  # OpenAI 的 tokenizer

class TokenChunker:
    """
    基于 Token 的分块器。
    使用 tiktoken 计算 token 数，支持不同的编码器。
    """
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50,
                 encoding_name: str = "cl100k_base"):
        """
        Args:
            chunk_size: 每块的目标 token 数
            chunk_overlap: 相邻块的重叠 token 数
            encoding_name: tiktoken 编码器名称
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding(encoding_name)
    
    def tokenize(self, text: str) -> List[int]:
        """将文本转为 token ID 列表"""
        return self.encoding.encode(text)
    
    def detokenize(self, tokens: List[int]) -> str:
        """将 token ID 列表转回文本"""
        return self.encoding.decode(tokens)
    
    def split_text(self, text: str) -> List[str]:
        """
        按 token 数分割文本。
        
        Returns:
            文本块列表
        """
        tokens = self.tokenize(text)
        chunks = []
        
        start = 0
        while start < len(tokens):
            # 确定当前块的结束位置
            end = start + self.chunk_size
            
            # 获取当前块
            chunk_tokens = tokens[start:end]
            chunk_text = self.detokenize(chunk_tokens)
            chunks.append(chunk_text)
            
            # 移动起始位置（考虑重叠）
            start += self.chunk_size - self.chunk_overlap
        
        return chunks
    
    def count_tokens(self, text: str) -> int:
        """计算文本的 token 数"""
        return len(self.tokenize(text))


class SentenceAwareTokenChunker(TokenChunker):
    """
    句感知的 Token 分块器。
    在分块边界处尽量保持句子完整。
    """
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50,
                 encoding_name: str = "cl100k_base"):
        super().__init__(chunk_size, chunk_overlap, encoding_name)
        # 中英文句子边界
        self.sentence_delimiters = {".", "!", "?", "。", "！", "？", "\n"}
    
    def _find_sentence_boundary(self, tokens: List[int], 
                                 preferred_end: int) -> int:
        """
        在 preferred_end 附近寻找合适的句子边界。
        回退范围：preferred_end 的前后各 20% chunk_size。
        """
        if not tokens:
            return 0
        
        text = self.detokenize(tokens)
        
        # 在 preferred_end 附近搜索
        search_range = int(self.chunk_size * 0.2)
        search_start = max(0, preferred_end - search_range)
        search_end = min(len(tokens), preferred_end + search_range)
        
        # 在搜索范围内找句子边界
        for pos in range(search_end - 1, search_start - 1, -1):
            # 检查该位置的字符是否为句子分隔符
            token_text = self.detokenize(tokens[pos:pos+1])
            if token_text and token_text[-1] in self.sentence_delimiters:
                return pos + 1  # 边界在分隔符之后
        
        # 没有找到合适的句子边界，返回 preferred_end
        return preferred_end
    
    def split_text(self, text: str) -> List[str]:
        """
        句感知分割：优先在句子边界处切分。
        """
        tokens = self.tokenize(text)
        chunks = []
        
        start = 0
        while start < len(tokens):
            preferred_end = start + self.chunk_size
            if preferred_end >= len(tokens):
                # 最后一块
                chunk_text = self.detokenize(tokens[start:])
                chunks.append(chunk_text)
                break
            
            # 寻找句子边界
            actual_end = self._find_sentence_boundary(tokens, preferred_end)
            
            chunk_text = self.detokenize(tokens[start:actual_end])
            if chunk_text.strip():
                chunks.append(chunk_text)
            
            # 移动起始位置
            start = actual_end - self.chunk_overlap
        
        return chunks


# ========== 使用示例 ==========
if __name__ == "__main__":
    chunker = SentenceAwareTokenChunker(chunk_size=200, chunk_overlap=30)
    
    text = """
    检索增强生成（Retrieval-Augmented Generation, RAG）是一种将信息检索与文本生成相结合的技术。
    它通过从知识库中检索相关文档来增强大语言模型的生成能力。
    这种方法可以有效缓解大语言模型的知识截止问题和幻觉现象。
    近年来，RAG 系统在企业应用中得到了广泛关注。
    本章将详细介绍 RAG 系统的文档处理流程。
    """
    
    chunks = chunker.split_text(text)
    print(f"共生成 {len(chunks)} 个块:\n")
    for i, chunk in enumerate(chunks):
        token_count = chunker.count_tokens(chunk)
        print(f"--- 块 {i+1} ({token_count} tokens) ---")
        print(chunk.strip())
        print()
```

### 4.6.2 语义分块 (Semantic Chunking)

语义分块的目标是让每个块包含语义完整的信息单元。常见的方法包括基于主题变化检测（如文本的 embedding 突变点）或利用 LLM 进行分块。

```python
# ch04_semantic_chunking.py
"""语义分块实现"""

import numpy as np
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class SemanticChunker:
    """
    基于语义相似度的分块器。
    
    原理：检测连续句子之间的语义变化，
    当相似度低于阈值时，认为话题发生转变，在此处切分。
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2",
                 similarity_threshold: float = 0.6,
                 min_chunk_size: int = 3,  # 最少句子数
                 max_chunk_size: int = 20):  # 最多句子数
        """
        Args:
            model_name: SentenceTransformer 模型
            similarity_threshold: 相似度阈值，低于此值视为话题转变
            min_chunk_size: 块的最少句子数
            max_chunk_size: 块的最多句子数
        """
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
    
    def _split_sentences(self, text: str) -> List[str]:
        """
        将文本分割为句子。
        支持中英文句子分割。
        """
        import re
        # 中英文句子边界
        sentence_end = re.compile(r'([。！？\.!?\n])\s*')
        sentences = sentence_end.split(text)
        
        # 合并分隔符到前一个句子
        result = []
        for i in range(0, len(sentences) - 1, 2):
            sentence = sentences[i] + sentences[i + 1]
            sentence = sentence.strip()
            if sentence:
                result.append(sentence)
        
        # 处理可能的剩余文本
        if len(sentences) % 2 == 1 and sentences[-1].strip():
            result.append(sentences[-1].strip())
        
        return result
    
    def split_text(self, text: str) -> List[str]:
        """
        基于语义相似度对文本进行分块。
        """
        sentences = self._split_sentences(text)
        if len(sentences) <= self.min_chunk_size:
            return [text]
        
        # 计算每个句子的 embedding
        sentence_embeddings = self.model.encode(sentences)
        
        # 计算相邻句子的余弦相似度
        similarities = []
        for i in range(len(sentences) - 1):
            sim = cosine_similarity(
                [sentence_embeddings[i]],
                [sentence_embeddings[i + 1]]
            )[0][0]
            similarities.append(sim)
        
        # 找到话题转变点（相似度骤降的位置）
        split_points = []
        for i, sim in enumerate(similarities):
            if sim < self.similarity_threshold:
                split_points.append(i + 1)  # +1 因为是在句子 i 和 i+1 之间切
        
        # 应用最小/最大块大小约束
        chunks = []
        start = 0
        for split_point in split_points:
            if split_point - start >= self.min_chunk_size:
                if split_point - start <= self.max_chunk_size:
                    chunk_text = " ".join(sentences[start:split_point])
                    chunks.append(chunk_text)
                    start = split_point
        
        # 剩余部分
        remaining = sentences[start:]
        if remaining:
            if chunks and len(remaining) < self.min_chunk_size:
                # 剩余过少，合并到上一个块
                chunks[-1] = chunks[-1] + " " + " ".join(remaining)
            else:
                chunks.append(" ".join(remaining))
        
        return chunks


class LLMBasedChunker:
    """
    基于 LLM 的语义分块器。
    使用语言模型来判断语义边界。
    
    注意：此方法开销较大，适合对分块质量要求极高的场景。
    """
    
    def __init__(self, llm_client, model: str = "gpt-4o-mini"):
        self.client = llm_client
        self.model = model
    
    def split_text(self, text: str, max_chunk_size: int = 2000) -> List[str]:
        """
        使用 LLM 分析文本结构并进行语义分块。
        
        提示词（prompt）设计：
        让 LLM 识别文档中的主要语义单元，
        并在每个单元的结尾插入 <CHUNK_BOUNDARY> 标记。
        """
        prompt = f"""请分析以下文本的语义结构，找出自然的分段边界。
        在每个语义完整的段落末尾插入 <CHUNK_BOUNDARY>。
        要求：
        1. 保持每个块的语义完整性
        2. 每个块应该是自包含的信息单元
        3. 不要分割紧密相关的句子
        
        文本：
        {text}
        
        请在每个语义边界处插入 <CHUNK_BOUNDARY> 标记，输出完整文本。"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        marked_text = response.choices[0].message.content
        chunks = [c.strip() for c in marked_text.split("<CHUNK_BOUNDARY>")
                 if c.strip()]
        
        return chunks


# ========== 使用示例 ==========
if __name__ == "__main__":
    chunker = SemanticChunker(similarity_threshold=0.65)
    
    text = """
    检索增强生成（RAG）是当前 NLP 领域的热门技术。
    它将信息检索与文本生成有机结合。
    
    深度学习的快速发展推动了 RAG 技术的进步。
    Transformer 架构为这一领域奠定了基础。
    
    在实际应用中，RAG 系统需要处理多种文档格式。
    包括 PDF、Word、HTML 等。
    每种格式都有其独特的解析挑战。
    """
    
    chunks = chunker.split_text(text)
    print(f"语义分块结果 ({len(chunks)} 个块):\n")
    for i, chunk in enumerate(chunks):
        print(f"--- 块 {i+1} ---")
        print(chunk)
        print()
```

### 4.6.3 结构感知分块 (Structure-Aware Chunking)

结构感知分块利用文档本身的层次结构（标题、段落、列表）来确定分块边界。这是最推荐的分块策略。

```python
# ch04_structure_chunking.py
"""结构感知分块"""

from typing import List, Optional
import re

class StructureAwareChunker:
    """
    结构感知分块器。
    
    利用文档的标题层级和段落结构进行分块。
    支持 Markdown 标题、HTML 标题、PDF 大纲等结构。
    """
    
    def __init__(self, max_chunk_size: int = 1000,
                 min_chunk_size: int = 100,
                 heading_level: int = 2):
        """
        Args:
            max_chunk_size: 块的最大字符数
            min_chunk_size: 块的最小字符数
            heading_level: 分块参考的标题级别（1=最高级标题）
        """
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.heading_level = heading_level
    
    def parse_headings(self, text: str) -> List[dict]:
        """
        解析文本中的标题结构。
        支持 Markdown 和纯文本标题。
        """
        sections = []
        lines = text.split("\n")
        current_section = {
            "heading": "root",
            "level": 0,
            "start_line": 0,
            "content_lines": []
        }
        
        # Markdown 标题模式
        md_heading = re.compile(r'^(#{1,6})\s+(.+)$')
        # 纯文本标题模式（全大写或加粗行）
        text_heading = re.compile(r'^[A-Z一-鿿][^。！？\n]*[：:]?$')
        
        for i, line in enumerate(lines):
            md_match = md_heading.match(line)
            if md_match:
                # 保存上一节
                if current_section["content_lines"]:
                    sections.append(current_section)
                
                level = len(md_match.group(1))
                heading = md_match.group(2).strip()
                current_section = {
                    "heading": heading,
                    "level": level,
                    "start_line": i,
                    "content_lines": []
                }
            elif (text_heading.match(line.strip()) and 
                  len(line.strip()) < 80 and
                  current_section["level"] == 0):
                # 可能是纯文本标题
                if current_section["content_lines"]:
                    sections.append(current_section)
                current_section = {
                    "heading": line.strip(),
                    "level": 1,
                    "start_line": i,
                    "content_lines": []
                }
            else:
                current_section["content_lines"].append(line)
        
        # 最后一节
        if current_section["content_lines"]:
            sections.append(current_section)
        
        # 合并内容
        for section in sections:
            section["text"] = "\n".join(section["content_lines"]).strip()
            del section["content_lines"]
        
        return sections
    
    def split_text(self, text: str) -> List[dict]:
        """
        结构感知分块。
        
        Returns:
            包含 heading、text、level 的块列表
        """
        sections = self.parse_headings(text)
        chunks = []
        
        # 构建标题上下文栈
        heading_stack = []
        
        for section in sections:
            level = section["level"]
            heading = section["heading"]
            content = section["text"]
            
            # 更新标题栈
            while heading_stack and heading_stack[-1]["level"] >= level:
                heading_stack.pop()
            
            if level > 0 and level <= self.heading_level:
                heading_stack.append({
                    "level": level,
                    "heading": heading
                })
            
            # 如果内容为空，跳过
            if not content:
                continue
            
            # 构建上下文标题
            context = " > ".join(h["heading"] for h in heading_stack)
            
            # 如果内容超过 max_chunk_size，进一步分割
            if len(content) > self.max_chunk_size:
                sub_chunks = self._split_large_section(content, context)
                chunks.extend(sub_chunks)
            else:
                chunks.append({
                    "heading": heading,
                    "level": level,
                    "context": context,
                    "text": content
                })
        
        return chunks
    
    def _split_large_section(self, text: str, context: str) -> List[dict]:
        """
        对过长的章节进行二次分割。
        按段落或句子分割。
        """
        paragraphs = re.split(r'\n\s*\n', text)
        chunks = []
        current_chunk = []
        current_size = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            if current_size + len(para) > self.max_chunk_size and current_chunk:
                chunks.append({
                    "heading": context,
                    "level": 0,
                    "context": context,
                    "text": "\n\n".join(current_chunk)
                })
                current_chunk = []
                current_size = 0
            
            current_chunk.append(para)
            current_size += len(para)
        
        if current_chunk:
            chunks.append({
                "heading": context,
                "level": 0,
                "context": context,
                "text": "\n\n".join(current_chunk)
            })
        
        return chunks


# ========== 使用示例 ==========
if __name__ == "__main__":
    chunker = StructureAwareChunker(max_chunk_size=500, heading_level=2)
    
    md_text = """
    # 第一章：RAG 系统概述
    
    ## 1.1 什么是 RAG
    
    检索增强生成（RAG）是一种将信息检索与文本生成相结合的技术。
    它通过从知识库中检索相关文档来增强大语言模型的生成能力。
    
    ## 1.2 RAG 的优势
    
    RAG 系统具有以下优势：
    1. 知识更新方便
    2. 减少幻觉
    3. 可解释性强
    
    ### 1.2.1 知识更新
    
    只需更新知识库即可让模型获取新知识。
    
    ## 1.3 应用场景
    
    RAG 可应用于问答系统、客服系统、知识管理等多个领域。
    """
    
    chunks = chunker.split_text(md_text)
    print(f"结构感知分块结果 ({len(chunks)} 个块):\n")
    for i, chunk in enumerate(chunks):
        print(f"--- 块 {i+1} ---")
        print(f"  标题: {chunk['heading']}")
        print(f"  上下文: {chunk['context']}")
        print(f"  内容长度: {len(chunk['text'])}")
        print(f"  内容预览: {chunk['text'][:100]}...")
        print()
```

### 4.6.4 递归分块 (Recursive Chunking)

递归分块是 LangChain 等框架中广泛采用的策略。它先用较大的分隔符（如段落边界）分割，如果块太大则递归地用更小的分隔符（句子、token）进一步分割。

```python
# ch04_recursive_chunking.py
"""递归分块实现"""

from typing import List, Optional, Callable
import re

class RecursiveChunker:
    """
    递归分块器。
    
    策略：
    1. 从最大粒度的分隔符开始（段落）
    2. 如果生成的块仍然超过目标大小，用更小粒度的分隔符（句子）
    3. 递归直到所有块都小于目标大小
    """
    
    def __init__(self, chunk_size: int = 500,
                 chunk_overlap: int = 50,
                 separators: Optional[List[str]] = None,
                 length_function: Callable = len):
        """
        Args:
            chunk_size: 目标块大小（以 length_function 为单位）
            chunk_overlap: 重叠大小
            separators: 分隔符列表，按优先级从高到低
            length_function: 计算块大小的函数（默认为字符数）
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function
        
        # 默认分隔符（按优先级）
        self.separators = separators or [
            "\n\n",    # 段落
            "\n",      # 行
            "。",      # 中文句号
            ". ",      # 英文句点
            "；",      # 中文分号
            "，",      # 中文逗号
            ", ",      # 英文逗号
            " ",       # 空格
            ""         # 字符级（最后回退）
        ]
    
    def split_text(self, text: str) -> List[str]:
        """递归分块入口"""
        return self._recursive_split(text, self.separators)
    
    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """
        递归分割文本。
        """
        final_chunks = []
        
        # 如果文本已经小于目标大小，直接返回
        if self.length_function(text) <= self.chunk_size:
            return [text]
        
        # 获取当前使用的分隔符
        separator = separators[0]
        remaining_separators = separators[1:]
        
        # 使用当前分隔符分割
        if separator:
            parts = text.split(separator)
        else:
            # 字符级分割
            parts = list(text)
        
        # 合并小片段
        merged = self._merge_parts(parts, separator)
        
        # 对每个合并后的片段递归处理
        for part in merged:
            if self.length_function(part) <= self.chunk_size:
                final_chunks.append(part)
            elif remaining_separators:
                # 递归使用更小粒度的分隔符
                sub_chunks = self._recursive_split(part, remaining_separators)
                final_chunks.extend(sub_chunks)
            else:
                # 没有更多分隔符了，强制分割
                final_chunks.extend(self._force_split(part))
        
        # 应用重叠
        return self._apply_overlap(final_chunks)
    
    def _merge_parts(self, parts: List[str], separator: str) -> List[str]:
        """
        合并小片段，使每个片段尽可能接近 chunk_size。
        """
        merged = []
        current = ""
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            if not current:
                current = part
            elif self.length_function(current) + self.length_function(separator) + \
                 self.length_function(part) <= self.chunk_size:
                current += separator + part
            else:
                merged.append(current)
                current = part
        
        if current:
            merged.append(current)
        
        return merged
    
    def _force_split(self, text: str) -> List[str]:
        """
        强制分割：当没有合适分隔符时，按 chunk_size 硬切。
        """
        chunks = []
        start = 0
        while start < self.length_function(text):
            end = min(start + self.chunk_size, self.length_function(text))
            chunks.append(text[start:end])
            start = end
        return chunks
    
    def _apply_overlap(self, chunks: List[str]) -> List[str]:
        """
        在相邻块之间应用重叠。
        """
        if self.chunk_overlap <= 0 or len(chunks) <= 1:
            return chunks
        
        result = []
        for i, chunk in enumerate(chunks):
            if i > 0:
                # 从前一个块末尾取 overlap 长度
                prev_chunk = chunks[i - 1]
                if self.length_function(prev_chunk) > self.chunk_overlap:
                    overlap_text = prev_chunk[-self.chunk_overlap:]
                    chunk = overlap_text + chunk
            
            result.append(chunk)
        
        return result


# ========== 使用示例 ==========
if __name__ == "__main__":
    chunker = RecursiveChunker(
        chunk_size=200,
        chunk_overlap=30,
        length_function=len
    )
    
    text = """
    第一章：RAG 系统概述
    
    检索增强生成（Retrieval-Augmented Generation，简称 RAG）是一种将信息检索与文本生成相结合的人工智能技术。
    它的核心思想是在生成回答之前，先从外部知识库中检索与问题相关的文档片段，
    然后基于这些检索到的信息来生成更准确、更可靠的回答。
    
    这一方法可以有效缓解大语言模型的两个主要问题：知识截止日期和幻觉现象。
    知识截止日期指的是模型训练数据只覆盖到某个时间点之前的信息；
    而幻觉现象则是指模型可能生成看似合理但实际错误的内容。
    
    RAG 系统的优势包括：知识可以随时更新、结果可追溯、减少幻觉、提高准确性。
    这些优势使得 RAG 在企业级应用中获得了广泛的关注和采用。
    """
    
    chunks = chunker.split_text(text)
    print(f"递归分块结果 ({len(chunks)} 个块):\n")
    for i, chunk in enumerate(chunks):
        print(f"--- 块 {i+1} (长度: {len(chunk)}) ---")
        print(chunk.strip())
        print()
```

### 4.6.5 分块策略对比

| 策略 | 语义完整性 | 实现复杂度 | 适用场景 |
|------|-----------|-----------|----------|
| **Token 分块** | 低 | 极低 | 快速原型、简单文本 |
| **语义分块** | 高 | 中 | 主题清晰的文档 |
| **结构感知** | 高 | 中 | 有层次结构的文档（推荐） |
| **递归分块** | 中 | 低 | 通用场景 |
| **LLM 分块** | 最高 | 高 | 质量优先、不计成本 |

---

## 4.7 分块参数调优 (Chunking Parameter Tuning)

分块参数（块大小、重叠大小、分割策略）对 RAG 系统的最终效果有显著影响。参数调优应该基于实验数据而非直觉。

### 4.7.1 评估指标

```python
# ch04_chunk_evaluation.py
"""分块质量评估"""

from typing import List, Callable
import numpy as np
from sentence_transformers import SentenceTransformer

class ChunkEvaluator:
    """
    分块质量评估器。
    
    评估维度：
    1. 语义一致性：块内句子的语义相似度
    2. 信息密度：有效信息占比
    3. 大小均匀性：块大小的分布
    4. 检索命中率（需要测试集）
    """
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(embedding_model)
    
    def evaluate_semantic_coherence(self, chunks: List[str]) -> float:
        """
        评估语义一致性。
        计算每个块内句子对的平均余弦相似度。
        """
        if not chunks:
            return 0.0
        
        coherence_scores = []
        for chunk in chunks:
            # 将块分割为句子
            sentences = [s.strip() for s in chunk.replace("。", ". ").split(".")
                        if s.strip()]
            
            if len(sentences) < 2:
                continue
            
            # 计算句子 embedding
            embeddings = self.model.encode(sentences)
            
            # 计算所有句子对之间的平均相似度
            from sklearn.metrics.pairwise import cosine_similarity
            sim_matrix = cosine_similarity(embeddings)
            
            # 取上三角的平均值（排除对角线）
            n = len(sentences)
            upper_tri = [sim_matrix[i][j] 
                        for i in range(n) for j in range(i+1, n)]
            
            if upper_tri:
                coherence_scores.append(np.mean(upper_tri))
        
        return np.mean(coherence_scores) if coherence_scores else 0.0
    
    def evaluate_size_uniformity(self, chunks: List[str],
                                   length_fn: Callable = len) -> float:
        """
        评估块大小的均匀性。
        使用变异系数（CV）衡量，CV 越小越均匀。
        """
        if len(chunks) < 2:
            return 1.0
        
        sizes = [length_fn(c) for c in chunks]
        cv = np.std(sizes) / np.mean(sizes) if np.mean(sizes) > 0 else 1.0
        
        # 将 CV 转换为 0~1 分数（CV=0 为 1.0，CV=1 为 0.0）
        return max(0.0, 1.0 - cv)
    
    def evaluate_information_density(self, chunks: List[str]) -> float:
        """
        评估信息密度。
        基于"有效字符"占比的简单估计。
        """
        total_chars = sum(len(c) for c in chunks)
        if total_chars == 0:
            return 0.0
        
        # "有效"字符：非空白、非标点重复的字符
        import re
        effective = 0
        for chunk in chunks:
            # 移除多余空白和重复标点
            cleaned = re.sub(r'\s+', ' ', chunk)
            cleaned = re.sub(r'([，。！？、；：])\1+', r'\1', cleaned)
            effective += len(cleaned.strip())
        
        return effective / total_chars
    
    def evaluate_retrieval_accuracy(self, chunks: List[str],
                                     queries: List[str],
                                     relevant_chunks: List[List[int]]) -> float:
        """
        评估检索准确率。
        需要人工标注的测试集。
        
        Args:
            chunks: 所有块
            queries: 查询列表
            relevant_chunks: 每个查询对应的相关块索引列表
        
        Returns:
            Recall@K 的平均值
        """
        if not chunks or not queries:
            return 0.0
        
        # 计算所有块的 embedding
        chunk_embeddings = self.model.encode(chunks)
        
        recalls = []
        for query, relevant in zip(queries, relevant_chunks):
            query_embedding = self.model.encode([query])
            
            # 计算相似度并排序
            similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
            top_k_indices = np.argsort(similarities)[-len(relevant):][::-1]
            
            # 计算 Recall
            hits = len(set(top_k_indices) & set(relevant))
            recall = hits / len(relevant) if relevant else 0
            recalls.append(recall)
        
        return np.mean(recalls)
    
    def full_report(self, chunks: List[str],
                     queries: List[str] = None,
                     relevant_chunks: List[List[int]] = None) -> dict:
        """
        生成完整的评估报告。
        """
        report = {
            "num_chunks": len(chunks),
            "avg_chunk_size": np.mean([len(c) for c in chunks]),
            "std_chunk_size": np.std([len(c) for c in chunks]),
            "min_chunk_size": min(len(c) for c in chunks),
            "max_chunk_size": max(len(c) for c in chunks),
            "semantic_coherence": self.evaluate_semantic_coherence(chunks),
            "size_uniformity": self.evaluate_size_uniformity(chunks),
            "information_density": self.evaluate_information_density(chunks),
        }
        
        if queries and relevant_chunks:
            report["retrieval_accuracy"] = self.evaluate_retrieval_accuracy(
                chunks, queries, relevant_chunks
            )
        
        return report


# ========== 使用示例 ==========
if __name__ == "__main__":
    evaluator = ChunkEvaluator()
    
    # 比较不同分块策略的效果
    strategies = {
        "token": ["这是第一个块的内容。", "这是第二个块的内容。"],
        "semantic": ["语义完整的段落一。包含多个相关句子。",
                     "语义完整的段落二。同样是相关的内容。"],
    }
    
    for name, chunks in strategies.items():
        report = evaluator.full_report(chunks)
        print(f"\n{name} 分块评估:")
        for metric, value in report.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.3f}")
            else:
                print(f"  {metric}: {value}")
```

### 4.7.2 参数搜索实验

```python
# ch04_parameter_search.py
"""分块参数搜索"""

import itertools
from typing import List, Callable, Dict, Any

def grid_search_chunking(
    text: str,
    chunker_class,
    param_grid: Dict[str, List[Any]],
    evaluator: ChunkEvaluator,
    queries: List[str] = None,
    relevant_chunks: List[List[int]] = None
) -> List[Dict]:
    """
    分块参数的网格搜索。
    
    Args:
        text: 待分块的文本
        chunker_class: 分块器类
        param_grid: 参数字典 {参数名: [候选值]}
        evaluator: 评估器实例
        queries: 测试查询
        relevant_chunks: 相关块索引
    
    Returns:
        按综合评分排序的参数-结果列表
    """
    results = []
    
    # 生成所有参数组合
    keys = param_grid.keys()
    values = param_grid.values()
    
    for combination in itertools.product(*values):
        params = dict(zip(keys, combination))
        
        # 创建分块器并分块
        chunker = chunker_class(**params)
        chunks = chunker.split_text(text)
        
        # 评估
        report = evaluator.full_report(chunks, queries, relevant_chunks)
        
        # 综合评分（加权）
        score = (
            report.get("semantic_coherence", 0) * 0.3 +
            report.get("size_uniformity", 0) * 0.2 +
            report.get("information_density", 0) * 0.1 +
            report.get("retrieval_accuracy", 0) * 0.4
        )
        
        results.append({
            "params": params,
            "score": score,
            "report": report
        })
    
    # 按评分排序
    results.sort(key=lambda r: r["score"], reverse=True)
    return results


# ========== 使用示例 ==========
if __name__ == "__main__":
    from ch04_recursive_chunking import RecursiveChunker
    from ch04_chunk_evaluation import ChunkEvaluator
    
    evaluator = ChunkEvaluator()
    
    param_grid = {
        "chunk_size": [200, 500, 1000],
        "chunk_overlap": [0, 50, 100],
    }
    
    # 示例文本
    text = "检索增强生成（RAG）是一种..." * 100
    
    results = grid_search_chunking(
        text=text,
        chunker_class=RecursiveChunker,
        param_grid=param_grid,
        evaluator=evaluator
    )
    
    print("参数搜索排名:")
    for i, result in enumerate(results[:5]):
        print(f"\n第 {i+1} 名 (评分: {result['score']:.3f}):")
        for param, value in result["params"].items():
            print(f"  {param}: {value}")
        print(f"  语义一致性: {result['report']['semantic_coherence']:.3f}")
        print(f"  大小均匀性: {result['report']['size_uniformity']:.3f}")
```

### 4.7.3 调优建议

**块大小（Chunk Size）的选择**：

| 块大小 | 适用场景 | 优缺点 |
|--------|----------|--------|
| 128-256 tokens | 短文本、FAQ、精准匹配 | 精度高但上下文不足 |
| 512 tokens | 通用场景、中等长度文档 | 平衡较好，常用默认值 |
| 1024-2048 tokens | 长文档、需要大量上下文 | 检索噪声增大 |
| > 2048 tokens | 综述类、长文档分析 | 适合高上下文嵌入模型 |

**重叠大小（Overlap）的选择**：

- 通常设为块大小的 10%-20%
- 过小的重叠（<5%）可能丢失边界信息
- 过大的重叠（>30%）增加存储成本且收益递减

**分块策略选择指南**：

1. **Markdown/HTML 文档**：优先使用结构感知分块
2. **学术论文/报告**：结构感知 + 语义分块混合
3. **网页抓取内容**：递归分块
4. **聊天记录/日志**：Token 分块
5. **法律/金融文档**：结构感知 + 最小块大小约束

---

## 4.8 完整文档处理 Pipeline

将本章所有内容整合为一个完整的文档处理流程：

```python
# ch04_pipeline.py
"""完整文档处理 Pipeline"""

from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass, field
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ProcessingResult:
    """处理结果"""
    source: str
    raw_text: str
    cleaned_text: str
    chunks: List[dict]
    metadata: dict = field(default_factory=dict)
    pii_found: List[str] = field(default_factory=list)
    encoding: str = ""


class DocumentProcessingPipeline:
    """
    完整的文档处理 Pipeline。
    
    流程：
    1. 编码检测 → 2. 格式解析 → 3. 数据清洗
    4. PII 检测 → 5. 格式标准化 → 6. 分块
    """
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50,
                 enable_pii_detection: bool = True):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.enable_pii_detection = enable_pii_detection
    
    def process_file(self, file_path: str) -> Optional[ProcessingResult]:
        """
        处理单个文件。
        """
        from pathlib import Path
        path = Path(file_path)
        
        if not path.exists():
            logger.error(f"文件不存在: {file_path}")
            return None
        
        try:
            # 1. 编码检测
            from ch04_encoding_detection import detect_encoding, read_with_auto_encoding
            enc_info = detect_encoding(file_path)
            encoding = enc_info.get("encoding", "unknown")
            logger.info(f"检测到编码: {encoding}")
            
            # 2. 格式解析
            from ch04_unified_parser import parse_document
            parsed = parse_document(file_path)
            raw_text = parsed.content
            logger.info(f"解析完成: {len(raw_text)} 字符")
            
            # 3. 数据清洗 - 噪声过滤
            from ch04_noise_filtering import NoiseFilter
            noise_filter = NoiseFilter()
            cleaned_text = noise_filter.filter_noise(raw_text)
            cleaned_text = noise_filter.remove_duplicate_lines(cleaned_text)
            logger.info(f"清洗后: {len(cleaned_text)} 字符")
            
            # 4. PII 检测
            pii_found = []
            if self.enable_pii_detection:
                from ch04_pii_detection import PIIDetector
                pii_detector = PIIDetector()
                detections = pii_detector.detect_all(cleaned_text)
                pii_found = [f"[{t}] {v}" for t, v, _, _ in detections]
                
                if pii_found:
                    logger.warning(f"发现 {len(pii_found)} 处 PII")
                    # 自动遮盖
                    cleaned_text = pii_detector.mask_pii_partial(cleaned_text)
            
            # 5. 格式标准化
            from ch04_format_normalization import normalize_text
            cleaned_text = normalize_text(cleaned_text)
            
            # 6. 分块
            from ch04_recursive_chunking import RecursiveChunker
            chunker = RecursiveChunker(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            chunk_texts = chunker.split_text(cleaned_text)
            
            chunks = []
            for i, text in enumerate(chunk_texts):
                chunks.append({
                    "index": i,
                    "text": text,
                    "token_count": len(text.split()),  # 近似
                    "source": file_path
                })
            
            logger.info(f"分块完成: {len(chunks)} 个块")
            
            return ProcessingResult(
                source=file_path,
                raw_text=raw_text,
                cleaned_text=cleaned_text,
                chunks=chunks,
                metadata={
                    "format": parsed.format,
                    "original_size": len(raw_text),
                    "cleaned_size": len(cleaned_text),
                    "num_chunks": len(chunks)
                },
                pii_found=pii_found,
                encoding=encoding
            )
            
        except Exception as e:
            logger.error(f"处理失败 {file_path}: {e}")
            return None
    
    def process_batch(self, file_paths: List[str]) -> List[ProcessingResult]:
        """
        批量处理多个文件。
        """
        results = []
        for fp in file_paths:
            result = self.process_file(fp)
            if result:
                results.append(result)
        return results


# ========== 使用示例 ==========
if __name__ == "__main__":
    pipeline = DocumentProcessingPipeline(
        chunk_size=512,
        chunk_overlap=50,
        enable_pii_detection=True
    )
    
    # 处理单个文件
    result = pipeline.process_file("document.pdf")
    if result:
        print(f"处理完成: {result.source}")
        print(f"  原始大小: {result.metadata['original_size']} 字符")
        print(f"  清洗后: {result.metadata['cleaned_size']} 字符")
        print(f"  分块数: {result.metadata['num_chunks']}")
        print(f"  编码: {result.encoding}")
        
        if result.pii_found:
            print(f"  PII 告警: {len(result.pii_found)} 处")
    
    # 批量处理
    files = ["doc1.pdf", "doc2.docx", "doc3.html", "doc4.md"]
    results = pipeline.process_batch(files)
    print(f"\n批量处理完成: {len(results)}/{len(files)} 成功")
```

---

## 4.9 本章小结

文档处理是 RAG 系统的第一个也是最重要的环节。本章涵盖了从原始文件到高质量文本块的全流程：

1. **多格式解析**：PDF（PyMuPDF/pdfplumber）、Word（python-docx）、Excel（openpyxl）、HTML（BeautifulSoup4）、Markdown 各有其解析策略和注意事项。

2. **OCR 与布局分析**：PaddleOCR 和 Tesseract 是中文 OCR 的主流选择；布局分析可以恢复文档的阅读顺序和结构层次。

3. **数据清洗**：MinHash 和 SimHash 提供高效的近似去重；格式标准化、PII 检测和噪声过滤确保文本质量。

4. **编码检测**：chardet 是自动识别文件编码的标准工具，配合回退策略可以处理绝大多数编码问题。

5. **分块策略**：Token 分块、语义分块、结构感知分块、递归分块各有适用场景。推荐结构感知分块作为首选。

6. **参数调优**：块大小、重叠、分割策略需要通过实验来确定最优值，而非凭经验猜测。

**关键建议**：
- 在处理文档前，始终先检测编码
- 优先使用结构感知分块
- 对敏感数据必须做 PII 检测和过滤
- 分块参数应通过实验确定，而非使用固定值
- 建立完整的处理 Pipeline，统一管理各阶段

下一章将深入讨论 embedding 模型的选择与优化，以及向量数据库的构建策略。
