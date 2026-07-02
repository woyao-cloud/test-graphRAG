# 第9章 文档结构化 RAG

## 9.1 企业文档的多样性与挑战

### 9.1.1 企业文档的复杂性

企业 RAG 系统面对的不是干净的纯文本文档，而是格式各异、结构复杂的各类文件。处理这些文档的挑战远超简单的"读文件-切分-索引"流程。

**企业常见文档类型**：

| 格式 | 复杂性 | 挑战 | 占比（典型企业） |
|------|--------|------|----------------|
| PDF | 高 | 文本/表格/图片混合，无结构化标记 | 40% |
| Word (docx) | 中 | 样式复杂，页眉页脚干扰 | 25% |
| Excel (xlsx) | 中 | 表格结构识别，行列映射 | 15% |
| PPT (pptx) | 高 | 幻灯片内布局复杂 | 5% |
| HTML/网页 | 中 | 导航/广告干扰 | 10% |
| Markdown/纯文本 | 低 | 最易处理 | 5% |

### 9.1.2 文档解析流水线

```
原始文件
    │
    ├── [PDF] → PyMuPDF / Camelot / PaddleOCR → 文本+表格+图片
    ├── [DOCX] → python-docx → 段落+表格+样式
    ├── [XLSX] → openpyxl → 行+列+值
    ├── [HTML] → BeautifulSoup / readability → 正文+链接
    └── [MD/TXT] → 直接读取
    │
    ▼
[统一文档模型] → [结构树构建] → [清洗过滤] → [分块索引]
```

核心目标是将各种格式的文档转化为**统一的、结构化的文档表示**，保留文档的层级关系和语义单元。

---

## 9.2 PDF 解析

### 9.2.1 PyMuPDF 文本提取

PyMuPDF（fitz）是 Python 中最成熟的 PDF 文本提取库之一，支持精确的文本定位和布局分析。

```python
import fitz  # PyMuPDF
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class TextBlock:
    """PDF 文本块"""
    text: str
    page_num: int
    bbox: tuple          # (x0, y0, x1, y1) 边界框
    font_size: float     # 字体大小
    font_name: str       # 字体名称
    is_bold: bool = False
    is_italic: bool = False
    block_type: str = "text"  # text, image, table


class PyMuPDFExtractor:
    """基于 PyMuPDF 的 PDF 文本提取"""
    
    def __init__(self, file_path: str):
        """
        Args:
            file_path: PDF 文件路径
        """
        self.doc = fitz.open(file_path)
        self.file_path = file_path
    
    def extract_text(self) -> str:
        """
        提取全部文本
        
        Returns:
            纯文本
        """
        text = ""
        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            text += page.get_text()
        
        return text
    
    def extract_structured_blocks(self) -> List[TextBlock]:
        """
        提取结构化的文本块
        
        Returns:
            包含位置和格式信息的文本块列表
        """
        blocks = []
        
        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            
            # 获取页面的文本块（按读取顺序）
            page_dict = page.get_text("dict")
            
            for block in page_dict["blocks"]:
                if block["type"] == 0:  # 文本块
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text = span["text"].strip()
                            if not text:
                                continue
                            
                            blocks.append(TextBlock(
                                text=text,
                                page_num=page_num + 1,
                                bbox=span["bbox"],
                                font_size=round(span["size"], 1),
                                font_name=span["font"],
                                is_bold="Bold" in span["font"],
                                is_italic="Italic" in span["font"],
                                block_type="text"
                            ))
        
        return blocks
    
    def detect_headings(self) -> List[TextBlock]:
        """
        根据字体大小和样式检测标题
        
        Returns:
            标题块列表
        """
        blocks = self.extract_structured_blocks()
        
        if not blocks:
            return []
        
        # 计算字体大小分布，确定标题阈值
        font_sizes = [b.font_size for b in blocks]
        avg_font = sum(font_sizes) / len(font_sizes)
        
        # 字体大小超过平均值 1.2 倍视为标题
        headings = [
            b for b in blocks
            if b.font_size > avg_font * 1.2 or b.is_bold
        ]
        
        return headings
    
    def extract_toc(self) -> List[Dict]:
        """
        提取目录结构
        
        Returns:
            [{"title": 标题, "page": 页码, "level": 层级}, ...]
        """
        toc = self.doc.get_toc()
        
        result = []
        for item in toc:
            result.append({
                "level": item[0],
                "title": item[1],
                "page": item[2]
            })
        
        return result
    
    def extract_images(self, page_num: int) -> List[bytes]:
        """
        提取页面中的图片
        
        Args:
            page_num: 页码（从 0 开始）
            
        Returns:
            图片二进制数据列表
        """
        page = self.doc[page_num]
        images = []
        
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = self.doc.extract_image(xref)
            images.append(base_image["image"])
        
        return images
    
    def close(self):
        self.doc.close()


# 使用示例
extractor = PyMuPDFExtractor("document.pdf")

# 提取标题
headings = extractor.detect_headings()
print("检测到的标题：")
for h in headings[:10]:
    print(f"  [第{h.page_num}页] {h.text} (字号: {h.font_size})")

# 提取目录
toc = extractor.extract_toc()
if toc:
    print("\n目录结构：")
    for item in toc:
        indent = "  " * (item["level"] - 1)
        print(f"  {indent}{item['title']} ... {item['page']}")

extractor.close()
```

### 9.2.2 表格提取（Camelot / Tabula / pdfplumber）

PDF 表格提取是文档解析中最具挑战性的任务之一。

```python
import camelot
import pdfplumber
from typing import List, Dict, Optional


class PDFTableExtractor:
    """PDF 表格提取器"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
    
    def extract_with_camelot(self, pages: str = "all") -> List[Dict]:
        """
        使用 Camelot 提取表格
        
        Camelot 特点：
        - Lattice 模式：检测线条分隔的表格
        - Stream 模式：检测空白分隔的表格
        - 适合有明确边框的表格
        
        Args:
            pages: 页码，如 "1,2,3" 或 "all"
            
        Returns:
            表格数据列表
        """
        tables = []
        
        # Lattice 模式（有线表格）
        try:
            lattice_tables = camelot.read_pdf(
                self.file_path,
                pages=pages,
                flavor="lattice"
            )
            
            for i, table in enumerate(lattice_tables):
                tables.append({
                    "method": "camelot_lattice",
                    "page": table.parsing_report.get("page", 0),
                    "accuracy": table.parsing_report.get("accuracy", 0),
                    "headers": table.df.iloc[0].tolist(),
                    "rows": table.df.iloc[1:].values.tolist(),
                    "dataframe": table.df
                })
        except Exception as e:
            print(f"[Camelot] Lattice 模式失败: {e}")
        
        # Stream 模式（无线表格）
        try:
            stream_tables = camelot.read_pdf(
                self.file_path,
                pages=pages,
                flavor="stream"
            )
            
            for i, table in enumerate(stream_tables):
                tables.append({
                    "method": "camelot_stream",
                    "page": table.parsing_report.get("page", 0),
                    "accuracy": table.parsing_report.get("accuracy", 0),
                    "headers": table.df.iloc[0].tolist(),
                    "rows": table.df.iloc[1:].values.tolist(),
                })
        except Exception as e:
            print(f"[Camelot] Stream 模式失败: {e}")
        
        return tables
    
    def extract_with_pdfplumber(self) -> List[Dict]:
        """
        使用 pdfplumber 提取表格
        
        pdfplumber 特点：
        - 更轻量
        - 支持精确的单元格定位
        - 适合复杂布局的表格
        """
        tables = []
        
        with pdfplumber.open(self.file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page_tables = page.extract_tables()
                
                for table_data in page_tables:
                    if not table_data:
                        continue
                    
                    # 提取表头
                    headers = table_data[0] if table_data else []
                    
                    # 提取数据行
                    rows = table_data[1:] if len(table_data) > 1 else []
                    
                    # 清理空值和空白
                    headers = [h.strip() if h else "" for h in headers]
                    rows = [
                        [cell.strip() if cell else "" for cell in row]
                        for row in rows
                    ]
                    
                    tables.append({
                        "method": "pdfplumber",
                        "page": page_num + 1,
                        "headers": headers,
                        "rows": rows,
                        "num_rows": len(rows),
                        "num_cols": len(headers)
                    })
        
        return tables
    
    def table_to_markdown(self, table: Dict) -> str:
        """
        将表格转换为 Markdown 格式
        
        Args:
            table: 表格数据
            
        Returns:
            Markdown 表格字符串
        """
        headers = table.get("headers", [])
        rows = table.get("rows", [])
        
        if not headers and not rows:
            return ""
        
        # 表头
        md = "| " + " | ".join(headers) + " |\n"
        # 分隔行
        md += "| " + " | ".join(["---"] * len(headers)) + " |\n"
        # 数据行
        for row in rows:
            md += "| " + " | ".join(row) + " |\n"
        
        return md


def extract_table_with_fallback(file_path: str) -> List[Dict]:
    """
    使用多种方法提取表格（自动降级）
    
    策略：Camelot → pdfplumber → 正则手动提取
    """
    extractor = PDFTableExtractor(file_path)
    
    # 策略1: Camelot
    tables = extractor.extract_with_camelot()
    if tables:
        return tables
    
    # 策略2: pdfplumber
    tables = extractor.extract_with_pdfplumber()
    if tables:
        return tables
    
    return []
```

### 9.2.3 OCR 识别（PaddleOCR / Tesseract）

对于扫描版 PDF 或图片型 PDF，需要 OCR 进行文字识别：

```python
class PDFOCRExtractor:
    """PDF OCR 识别"""
    
    def __init__(self, use_paddle: bool = True):
        """
        Args:
            use_paddle: True 使用 PaddleOCR，False 使用 Tesseract
        """
        self.use_paddle = use_paddle
        
        if use_paddle:
            from paddleocr import PaddleOCR
            self.ocr = PaddleOCR(
                use_angle_cls=True,
                lang="ch",
                show_log=False
            )
        else:
            import pytesseract
            self.ocr = pytesseract
    
    def extract_from_image(self, image_path: str) -> str:
        """
        从图片中提取文字
        
        Args:
            image_path: 图片路径
            
        Returns:
            识别的文本
        """
        if self.use_paddle:
            result = self.ocr.ocr(image_path, cls=True)
            text = ""
            for line in result:
                for word_info in line:
                    text += word_info[1][0]
            return text
        else:
            from PIL import Image
            img = Image.open(image_path)
            text = self.ocr.image_to_string(img, lang="chi_sim")
            return text
    
    def extract_from_pdf(self, pdf_path: str,
                         dpi: int = 300) -> List[str]:
        """
        对 PDF 逐页进行 OCR
        
        Args:
            pdf_path: PDF 文件路径
            dpi: 图片分辨率
            
        Returns:
            每页的识别文本
        """
        import fitz
        from PIL import Image
        import io
        
        doc = fitz.open(pdf_path)
        page_texts = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # 将 PDF 页渲染为图片
            pix = page.get_pixmap(dpi=dpi)
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            
            # OCR 识别
            if self.use_paddle:
                # PaddleOCR 接受 numpy 数组
                import numpy as np
                img_array = np.array(img)
                result = self.ocr.ocr(img_array, cls=True)
                
                text = ""
                if result and result[0]:
                    for line in result:
                        for word_info in line:
                            text += word_info[1][0]
            else:
                text = self.ocr.image_to_string(img, lang="chi_sim")
            
            page_texts.append(text)
            print(f"[OCR] 第 {page_num + 1}/{len(doc)} 页完成")
        
        doc.close()
        return page_texts
```

### 9.2.4 PDF 布局分析

理解 PDF 的布局结构（单栏/双栏、标题层级、段落关系）对高质量提取至关重要：

```python
class PDFLayoutAnalyzer:
    """PDF 布局分析器"""
    
    def __init__(self, file_path: str):
        self.doc = fitz.open(file_path)
    
    def analyze_layout(self, page_num: int) -> Dict:
        """
        分析页面布局
        
        Args:
            page_num: 页码
            
        Returns:
            布局信息
        """
        page = self.doc[page_num]
        blocks = page.get_text("dict")["blocks"]
        
        # 收集所有文本块的位置
        text_blocks = []
        for block in blocks:
            if block["type"] == 0:  # 文本
                bbox = block["bbox"]
                text_blocks.append({
                    "x0": bbox[0],
                    "y0": bbox[1],
                    "x1": bbox[2],
                    "y1": bbox[3],
                    "width": bbox[2] - bbox[0],
                    "height": bbox[3] - bbox[1],
                })
        
        if not text_blocks:
            return {"columns": 1, "type": "unknown"}
        
        # 检测列数
        x_centers = [(b["x0"] + b["x1"]) / 2 for b in text_blocks]
        
        # 通过 x 坐标聚类判断列数
        from collections import Counter
        x_bins = [round(x / 50) * 50 for x in x_centers]  # 分桶
        column_clusters = Counter(x_bins)
        
        # 主要的列位置
        main_columns = [
            pos for pos, count in column_clusters.most_common(4)
            if count > len(text_blocks) * 0.1  # 至少占 10% 的块
        ]
        
        num_columns = len(main_columns)
        
        # 检测页眉页脚
        page_height = page.rect.height
        page_width = page.rect.width
        
        headers = [
            b for b in text_blocks
            if b["y0"] < page_height * 0.1  # 顶部 10%
        ]
        footers = [
            b for b in text_blocks
            if b["y1"] > page_height * 0.9  # 底部 10%
        ]
        
        return {
            "page_num": page_num + 1,
            "columns": max(num_columns, 1),
            "has_header": len(headers) > 0,
            "has_footer": len(footers) > 0,
            "page_width": page_width,
            "page_height": page_height,
            "text_block_count": len(text_blocks)
        }
    
    def detect_multi_column(self, page_num: int) -> bool:
        """检测是否为多栏布局"""
        layout = self.analyze_layout(page_num)
        return layout["columns"] >= 2
    
    def extract_by_reading_order(self) -> str:
        """
        按阅读顺序提取文本（处理多栏布局）
        
        多栏 PDF 的阅读顺序是：先读完左栏，再读右栏。
        如果按 y 坐标顺序提取，会将左右栏交错提取。
        """
        full_text = ""
        
        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            layout = self.analyze_layout(page_num)
            
            if layout["columns"] <= 1:
                # 单栏：直接提取
                full_text += page.get_text()
            else:
                # 多栏：按栏提取
                blocks = page.get_text("dict")["blocks"]
                page_width = layout["page_width"]
                
                # 按 x 坐标分栏
                left_blocks = []
                right_blocks = []
                
                for block in blocks:
                    if block["type"] != 0:
                        continue
                    x_center = (block["bbox"][0] + block["bbox"][2]) / 2
                    
                    if x_center < page_width / 2:
                        left_blocks.append(block)
                    else:
                        right_blocks.append(block)
                
                # 按 y 坐标排序
                left_blocks.sort(key=lambda b: b["bbox"][1])
                right_blocks.sort(key=lambda b: b["bbox"][1])
                
                # 先读左栏，再读右栏
                for block in left_blocks + right_blocks:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            full_text += span["text"]
                        full_text += "\n"
                    full_text += "\n"
        
        return full_text
    
    def close(self):
        self.doc.close()
```

---

## 9.3 Word 文档解析

### 9.3.1 python-docx 基础解析

```python
from docx import Document
from docx.table import Table
from docx.text import Paragraph
from docx.oxml.ns import qn
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class DocElement:
    """Word 文档元素"""
    type: str  # "paragraph", "table", "heading", "list_item"
    text: str
    style: Optional[str] = None
    level: int = 0  # 标题层级或列表层级
    table_data: Optional[List[List[str]]] = None


class DocxParser:
    """Word 文档解析器"""
    
    def __init__(self, file_path: str):
        """
        Args:
            file_path: .docx 文件路径
        """
        self.doc = Document(file_path)
        self.file_path = file_path
    
    def parse_all(self) -> List[DocElement]:
        """
        解析文档的所有元素
        
        保留文档的原始结构顺序（段落、表格、标题交替出现）
        """
        elements = []
        
        for element in self.doc.element.body:
            tag_name = element.tag.split("}")[-1] if "}" in element.tag else element.tag
            
            if tag_name == "p":
                # 段落
                para = Paragraph(element, self.doc)
                para_elem = self._parse_paragraph(para)
                if para_elem:
                    elements.append(para_elem)
                    
            elif tag_name == "tbl":
                # 表格
                table = Table(element, self.doc)
                table_elem = self._parse_table(table)
                if table_elem:
                    elements.append(table_elem)
        
        return elements
    
    def _parse_paragraph(self, para: Paragraph) -> Optional[DocElement]:
        """解析段落"""
        text = para.text.strip()
        if not text:
            return None
        
        style_name = para.style.name if para.style else ""
        
        # 检测标题
        if style_name.startswith("Heading"):
            try:
                level = int(style_name.replace("Heading ", ""))
            except ValueError:
                level = 1
            
            return DocElement(
                type="heading",
                text=text,
                style=style_name,
                level=level
            )
        
        # 检测列表
        if para._element.find(qn("w:numPr")) is not None:
            return DocElement(
                type="list_item",
                text=text,
                style=style_name,
                level=self._get_list_level(para)
            )
        
        # 普通段落
        return DocElement(
            type="paragraph",
            text=text,
            style=style_name
        )
    
    def _parse_table(self, table: Table) -> DocElement:
        """解析表格"""
        rows = []
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            rows.append(cells)
        
        return DocElement(
            type="table",
            text=f"[表格: {len(rows)}行 x {len(rows[0]) if rows else 0}列]",
            table_data=rows
        )
    
    def _get_list_level(self, para: Paragraph) -> int:
        """获取列表层级"""
        num_pr = para._element.find(qn("w:numPr"))
        if num_pr is not None:
            ilvl = num_pr.find(qn("w:ilvl"))
            if ilvl is not None:
                return int(ilvl.get(qn("w:val"), 0))
        return 0
    
    def extract_styles(self) -> Dict[str, List[str]]:
        """
        提取文档样式信息
        
        Returns:
            {style_name: [使用该样式的文本列表]}
        """
        styles = {}
        
        for para in self.doc.paragraphs:
            if para.style and para.text.strip():
                style_name = para.style.name
                if style_name not in styles:
                    styles[style_name] = []
                styles[style_name].append(para.text.strip())
        
        return styles
    
    def get_heading_hierarchy(self) -> List[Dict]:
        """
        获取文档的标题层级结构
        
        Returns:
            [{"level": 1, "title": "第一章", "children": [...]}, ...]
        """
        hierarchy = []
        path = []  # 当前路径栈
        
        for para in self.doc.paragraphs:
            if para.style and para.style.name.startswith("Heading"):
                text = para.text.strip()
                if not text:
                    continue
                
                try:
                    level = int(para.style.name.replace("Heading ", ""))
                except ValueError:
                    continue
                
                node = {
                    "level": level,
                    "title": text,
                    "children": []
                }
                
                # 维护路径栈
                while path and path[-1]["level"] >= level:
                    path.pop()
                
                if path:
                    path[-1]["children"].append(node)
                else:
                    hierarchy.append(node)
                
                path.append(node)
        
        return hierarchy
    
    def to_markdown(self) -> str:
        """将文档转换为 Markdown"""
        md_lines = []
        
        for element in self.parse_all():
            if element.type == "heading":
                prefix = "#" * element.level
                md_lines.append(f"{prefix} {element.text}")
                md_lines.append("")
                
            elif element.type == "paragraph":
                md_lines.append(element.text)
                md_lines.append("")
                
            elif element.type == "list_item":
                indent = "  " * element.level
                md_lines.append(f"{indent}- {element.text}")
                
            elif element.type == "table" and element.table_data:
                # Markdown 表格
                rows = element.table_data
                if rows:
                    headers = rows[0]
                    md_lines.append("| " + " | ".join(headers) + " |")
                    md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
                    for row in rows[1:]:
                        md_lines.append("| " + " | ".join(row) + " |")
                    md_lines.append("")
        
        return "\n".join(md_lines)
```

---

## 9.4 Excel 解析

### 9.4.1 openpyxl 基础解析

```python
from openpyxl import load_workbook
from openpyxl.utils import get_column_letter
from typing import List, Dict, Optional


class ExcelParser:
    """Excel 文件解析器"""
    
    def __init__(self, file_path: str, data_only: bool = True):
        """
        Args:
            file_path: .xlsx 文件路径
            data_only: 是否只读取计算后的值（而非公式）
        """
        self.workbook = load_workbook(file_path, data_only=data_only)
        self.file_path = file_path
    
    def get_sheet_names(self) -> List[str]:
        """获取所有工作表名称"""
        return self.workbook.sheetnames
    
    def parse_sheet(self, sheet_name: str) -> Dict:
        """
        解析单个工作表
        
        Args:
            sheet_name: 工作表名称
            
        Returns:
            {
                "name": 工作表名,
                "rows": 行数,
                "cols": 列数,
                "headers": 表头行,
                "data": 数据行列表,
                "merged_cells": 合并单元格信息
            }
        """
        ws = self.workbook[sheet_name]
        
        # 获取所有行的数据
        all_rows = []
        for row in ws.iter_rows(values_only=True):
            all_rows.append(list(row))
        
        if not all_rows:
            return {"name": sheet_name, "rows": 0, "cols": 0, 
                    "headers": [], "data": [], "merged_cells": []}
        
        # 检测表头
        headers = self._detect_headers(all_rows)
        
        # 获取合并单元格
        merged_cells = []
        for merged_range in ws.merged_cells.ranges:
            merged_cells.append(str(merged_range))
        
        # 提取数据行（表头之后的行）
        header_row_count = len(headers) if isinstance(headers[0], list) else 1
        data_rows = all_rows[header_row_count:]
        
        # 清理空行
        data_rows = [row for row in data_rows if any(cell is not None for cell in row)]
        
        return {
            "name": sheet_name,
            "rows": len(all_rows),
            "cols": len(all_rows[0]) if all_rows else 0,
            "headers": headers if isinstance(headers, list) else [headers],
            "data": data_rows,
            "merged_cells": merged_cells
        }
    
    def _detect_headers(self, rows: List[List]) -> List[str]:
        """
        自动检测表头行
        
        策略：
        1. 第一行通常是表头
        2. 如果第一行全为 None 或空，跳过
        3. 如果第一行是数字而第二行是字符串，第二行可能是表头
        """
        if not rows:
            return []
        
        first_row = rows[0]
        
        # 检查第一行是否全部是字符串
        if all(isinstance(cell, str) and cell.strip() for cell in first_row if cell is not None):
            return first_row
        
        # 检查第二行
        if len(rows) > 1:
            second_row = rows[1]
            if all(isinstance(cell, str) for cell in second_row if cell is not None):
                return second_row
        
        # 默认返回第一行
        return first_row
    
    def parse_all_sheets(self) -> Dict[str, Dict]:
        """解析所有工作表"""
        return {
            name: self.parse_sheet(name)
            for name in self.get_sheet_names()
        }
    
    def sheet_to_markdown(self, sheet_name: str) -> str:
        """将工作表转换为 Markdown 表格"""
        data = self.parse_sheet(sheet_name)
        
        if not data["headers"] and not data["data"]:
            return f"## 工作表: {sheet_name}\n\n(空表)\n"
        
        md = f"## 工作表: {sheet_name}\n\n"
        
        # 表头
        md += "| " + " | ".join(str(h) for h in data["headers"]) + " |\n"
        md += "| " + " | ".join(["---"] * len(data["headers"])) + " |\n"
        
        # 数据
        for row in data["data"][:100]:  # 最多显示 100 行
            md += "| " + " | ".join(str(cell or "") for cell in row) + " |\n"
        
        if len(data["data"]) > 100:
            md += f"\n*（仅显示前 100 行，共 {len(data['data'])} 行）*\n"
        
        return md
    
    def sheet_to_structured_rows(self, sheet_name: str) -> List[Dict]:
        """
        将工作表转换为结构化字典列表
        
        Returns:
            [{header1: value1, header2: value2, ...}, ...]
        """
        data = self.parse_sheet(sheet_name)
        headers = data["headers"]
        
        structured = []
        for row in data["data"]:
            # 补齐长度
            padded_row = row + [None] * (len(headers) - len(row))
            record = {}
            for i, header in enumerate(headers):
                if i < len(padded_row):
                    record[str(header)] = padded_row[i]
            structured.append(record)
        
        return structured
    
    def close(self):
        self.workbook.close()
```

---

## 9.5 Web 页面解析

### 9.5.1 BeautifulSoup 解析

```python
from bs4 import BeautifulSoup, Tag
import requests
from typing import List, Dict, Optional


class WebPageParser:
    """网页解析器"""
    
    def __init__(self, url: str = None, html: str = None):
        """
        Args:
            url: 网页 URL
            html: 原始 HTML（优先使用）
        """
        self.url = url
        if html:
            self.soup = BeautifulSoup(html, "html.parser")
        elif url:
            self.soup = self._fetch(url)
        else:
            self.soup = None
    
    def _fetch(self, url: str) -> BeautifulSoup:
        """获取网页"""
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0.0.0 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        # 自动检测编码
        response.encoding = response.apparent_encoding
        
        return BeautifulSoup(response.text, "html.parser")
    
    def extract_title(self) -> str:
        """提取页面标题"""
        if not self.soup:
            return ""
        
        title = self.soup.find("title")
        return title.get_text(strip=True) if title else ""
    
    def extract_main_content(self) -> str:
        """
        提取页面主要内容
        
        使用多种启发式方法：
        1. 查找 <article> 标签
        2. 查找 <main> 标签
        3. 查找 content 类/id 的 div
        4. 回退到 body
        """
        if not self.soup:
            return ""
        
        # 策略1: article 标签
        article = self.soup.find("article")
        if article:
            return self._clean_element(article)
        
        # 策略2: main 标签
        main_tag = self.soup.find("main")
        if main_tag:
            return self._clean_element(main_tag)
        
        # 策略3: 常见 content 选择器
        content_selectors = [
            {"class": "content"},
            {"class": "post-content"},
            {"class": "article-content"},
            {"class": "entry-content"},
            {"id": "content"},
            {"id": "main-content"},
            {"class": "document-content"},
        ]
        
        for selector in content_selectors:
            element = self.soup.find(selector)
            if element:
                return self._clean_element(element)
        
        # 策略4: body
        body = self.soup.find("body")
        if body:
            return self._clean_element(body)
        
        return self.soup.get_text(strip=True)
    
    def _clean_element(self, element: Tag) -> str:
        """清理元素，移除不需要的部分"""
        # 移除 script, style, nav, footer, header, aside
        for tag in element.find_all(["script", "style", "nav", "footer",
                                      "header", "aside", "noscript"]):
            tag.decompose()
        
        return element.get_text(separator="\n", strip=True)
    
    def extract_links(self) -> List[Dict]:
        """
        提取页面中的所有链接
        
        Returns:
            [{"text": 链接文本, "url": URL, "is_internal": 是否内部链接}, ...]
        """
        if not self.soup:
            return []
        
        links = []
        for a_tag in self.soup.find_all("a", href=True):
            href = a_tag["href"]
            text = a_tag.get_text(strip=True)
            
            if not text:
                continue
            
            # 判断是否为内部链接
            is_internal = href.startswith("/") or (
                self.url and href.startswith(self.url)
            )
            
            links.append({
                "text": text,
                "url": href,
                "is_internal": is_internal
            })
        
        return links
    
    def extract_metadata(self) -> Dict:
        """
        提取页面元数据
        
        Returns:
            {"title": ..., "description": ..., "keywords": ..., "author": ...}
        """
        if not self.soup:
            return {}
        
        metadata = {}
        
        # 从 meta 标签提取
        meta_tags = {
            "description": "description",
            "keywords": "keywords",
            "author": "author",
        }
        
        for key, name in meta_tags.items():
            meta = self.soup.find("meta", attrs={"name": name})
            if meta and meta.get("content"):
                metadata[key] = meta["content"]
        
        # Open Graph 标签
        og_tags = {
            "og:title": "og_title",
            "og:description": "og_description",
            "og:type": "og_type",
        }
        
        for prop, key in og_tags.items():
            meta = self.soup.find("meta", attrs={"property": prop})
            if meta and meta.get("content"):
                metadata[key] = meta["content"]
        
        return metadata
    
    def to_markdown(self) -> str:
        """将网页转换为 Markdown"""
        title = self.extract_title()
        content = self.extract_main_content()
        
        md = f"# {title}\n\n" if title else ""
        md += content
        
        return md
```

### 9.5.2 Readability 算法

Readability 算法是 Mozilla 开发的正文提取算法，能有效过滤导航、广告等干扰内容：

```python
from readability import Document as ReadabilityDoc
import re

class ReadabilityExtractor:
    """基于 Readability 算法的正文提取"""
    
    def __init__(self, html: str, url: str = None):
        """
        Args:
            html: 原始 HTML
            url: 页面 URL（用于处理相对路径）
        """
        self.doc = ReadabilityDoc(html)
        if url:
            self.doc = ReadabilityDoc(html, url=url)
    
    def extract(self) -> Dict:
        """
        提取正文
        
        Returns:
            {
                "title": 标题,
                "content": HTML 格式的正文,
                "text_content": 纯文本格式的正文,
                "excerpt": 摘要,
                "byline": 作者,
                "length": 字符数
            }
        """
        summary = self.doc.summary()
        
        # 提取纯文本
        soup = BeautifulSoup(summary, "html.parser")
        text_content = soup.get_text(separator="\n", strip=True)
        
        return {
            "title": self.doc.title(),
            "content": summary,
            "text_content": text_content,
            "length": len(text_content)
        }
    
    def extract_clean_text(self) -> str:
        """提取干净的纯文本"""
        result = self.extract()
        # 进一步清理多余空白
        text = result["text_content"]
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)
        return text
```

### 9.5.3 html2text 转换

```python
import html2text

class HTMLToMarkdown:
    """HTML 转 Markdown"""
    
    def __init__(self):
        self.converter = html2text.HTML2Text()
        
        # 配置转换器
        self.converter.body_width = 0           # 不自动换行
        self.converter.ignore_links = False      # 保留链接
        self.converter.ignore_images = False     # 保留图片
        self.converter.ignore_emphasis = False   # 保留强调
        self.converter.protect_links = True      # 保护链接不被截断
        self.converter.unicode_snob = True       # 使用 Unicode
        self.converter.skip_internal_links = False
        self.converter.mark_code = True          # 标记代码块
    
    def convert(self, html: str) -> str:
        """
        将 HTML 转换为 Markdown
        
        Args:
            html: 原始 HTML
            
        Returns:
            Markdown 文本
        """
        return self.converter.handle(html)
    
    def convert_url(self, url: str) -> str:
        """抓取并转换网页"""
        import requests
        response = requests.get(url, timeout=30)
        response.encoding = response.apparent_encoding
        return self.convert(response.text)
```

---

## 9.6 复杂文档处理

### 9.6.1 多栏布局处理

```python
class MultiColumnProcessor:
    """多栏布局处理器"""
    
    def __init__(self, page_width: float, page_height: float,
                 column_gap_threshold: float = 50):
        """
        Args:
            page_width: 页面宽度
            page_height: 页面高度
            column_gap_threshold: 列间距阈值（像素）
        """
        self.page_width = page_width
        self.page_height = page_height
        self.gap_threshold = column_gap_threshold
    
    def detect_columns(self, text_blocks: List[Dict]) -> int:
        """
        检测文档的列数
        
        Args:
            text_blocks: 文本块列表，每项含 x0, x1, y0, y1
            
        Returns:
            列数
        """
        if not text_blocks:
            return 1
        
        # 获取所有文本块的 x 范围
        x_positions = []
        for block in text_blocks:
            x_positions.append(block["x0"])
            x_positions.append(block["x1"])
        
        # 检测空白区域（列间隙）
        x_positions.sort()
        gaps = []
        for i in range(1, len(x_positions)):
            gap = x_positions[i] - x_positions[i - 1]
            if gap > self.gap_threshold:
                gaps.append((x_positions[i - 1], x_positions[i]))
        
        return len(gaps) + 1 if gaps else 1
    
    def sort_by_reading_order(self, blocks: List[Dict]) -> List[Dict]:
        """
        按阅读顺序排序
        
        先按列分组，再按 y 坐标排序
        """
        num_columns = self.detect_columns(blocks)
        
        if num_columns <= 1:
            # 单栏：按 y 排序
            return sorted(blocks, key=lambda b: (b["y0"], b["x0"]))
        
        # 多栏：分栏排序
        column_width = self.page_width / num_columns
        columns = [[] for _ in range(num_columns)]
        
        for block in blocks:
            # 计算块的中心 x 坐标
            center_x = (block["x0"] + block["x1"]) / 2
            col_index = min(int(center_x / column_width), num_columns - 1)
            columns[col_index].append(block)
        
        # 每栏内按 y 排序
        for col in columns:
            col.sort(key=lambda b: b["y0"])
        
        # 按阅读顺序合并（先左后右）
        result = []
        for col in columns:
            result.extend(col)
        
        return result
```

### 9.6.2 页眉页脚过滤

```python
class HeaderFooterFilter:
    """页眉页脚过滤器"""
    
    def __init__(self, page_height: float,
                 header_ratio: float = 0.08,
                 footer_ratio: float = 0.08):
        """
        Args:
            page_height: 页面高度
            header_ratio: 页眉区域占比（顶部）
            footer_ratio: 页脚区域占比（底部）
        """
        self.header_threshold = page_height * header_ratio
        self.footer_threshold = page_height * (1 - footer_ratio)
    
    def is_header(self, y0: float, y1: float) -> bool:
        """判断是否为页眉"""
        return y1 < self.header_threshold
    
    def is_footer(self, y0: float, y1: float) -> bool:
        """判断是否为页脚"""
        return y0 > self.footer_threshold
    
    def filter_blocks(self, blocks: List[Dict]) -> List[Dict]:
        """
        过滤页眉页脚
        
        Args:
            blocks: 文本块列表，每项含 y0, y1
            
        Returns:
            过滤后的文本块
        """
        # 统计出现在页眉/页脚区域的文本
        header_texts = set()
        footer_texts = set()
        
        for block in blocks:
            if self.is_header(block["y0"], block["y1"]):
                header_texts.add(block.get("text", "").strip())
            elif self.is_footer(block["y0"], block["y1"]):
                footer_texts.add(block.get("text", "").strip())
        
        # 过滤跨页重复的页眉页脚
        filtered = []
        for block in blocks:
            text = block.get("text", "").strip()
            
            if self.is_header(block["y0"], block["y1"]):
                # 只保留第一页的页眉
                if text in header_texts and block.get("page_num", 1) > 1:
                    continue
            
            if self.is_footer(block["y0"], block["y1"]):
                # 跳过页脚
                if text in footer_texts:
                    continue
            
            filtered.append(block)
        
        return filtered
    
    @staticmethod
    def detect_page_numbers(text_blocks: List[Dict]) -> List[str]:
        """检测页码"""
        page_numbers = []
        for block in text_blocks:
            text = block.get("text", "").strip()
            # 匹配页码模式：纯数字，或 "— N —", "第N页"
            import re
            if re.match(r"^\d+$", text) and len(text) <= 4:
                page_numbers.append(text)
            elif re.match(r"^第\d+页$", text):
                page_numbers.append(text)
        return page_numbers
```

### 9.6.3 文档结构树

将解析后的文档构建为树形结构，保留层级关系：

```python
from typing import List, Optional

class DocumentNode:
    """文档树节点"""
    
    def __init__(self, element: Dict):
        self.element = element
        self.children: List[DocumentNode] = []
        self.parent: Optional[DocumentNode] = None
    
    def add_child(self, child: "DocumentNode"):
        child.parent = self
        self.children.append(child)
    
    def get_text(self) -> str:
        """获取节点及其子节点的完整文本"""
        texts = [self.element.get("text", "")]
        for child in self.children:
            child_text = child.get_text()
            if child_text:
                texts.append(child_text)
        return "\n".join(texts)
    
    def find_by_type(self, elem_type: str) -> List["DocumentNode"]:
        """按类型查找子节点"""
        results = []
        if self.element.get("type") == elem_type:
            results.append(self)
        for child in self.children:
            results.extend(child.find_by_type(elem_type))
        return results


class DocumentTreeBuilder:
    """文档结构树构建器"""
    
    def build_tree(self, elements: List[Dict]) -> DocumentNode:
        """
        从扁平的元素列表构建文档树
        
        策略：
        1. 标题元素作为分支节点
        2. 段落和表格作为叶子节点
        3. 标题层级决定嵌套深度
        """
        root = DocumentNode({
            "type": "root",
            "text": "",
            "level": 0
        })
        
        current_path = [root]  # 当前路径栈
        
        for elem in elements:
            elem_type = elem.get("type", "paragraph")
            node = DocumentNode(elem)
            
            if elem_type == "heading":
                level = elem.get("level", 1)
                
                # 找到合适的父节点
                while (len(current_path) > 1 and 
                       current_path[-1].element.get("level", 0) >= level):
                    current_path.pop()
                
                # 添加到父节点
                current_path[-1].add_child(node)
                current_path.append(node)
                
            else:
                # 段落、表格等作为当前标题的子节点
                current_path[-1].add_child(node)
        
        return root
    
    def tree_to_markdown(self, node: DocumentNode, indent: int = 0) -> str:
        """将文档树转换为 Markdown"""
        lines = []
        prefix = "  " * indent
        
        elem_type = node.element.get("type", "")
        text = node.element.get("text", "")
        
        if elem_type == "root":
            pass
        elif elem_type == "heading":
            level = node.element.get("level", 1)
            lines.append(f"{'#' * level} {text}")
        elif elem_type == "table":
            lines.append(f"\n{prefix}[表格] {text}\n")
        elif elem_type == "list_item":
            lines.append(f"{prefix}- {text}")
        else:
            lines.append(f"{prefix}{text}")
        
        for child in node.children:
            child_text = self.tree_to_markdown(child, indent)
            if child_text:
                lines.append(child_text)
        
        return "\n".join(lines)
```

---

## 9.7 统一文档解析流水线

```python
from typing import Dict, List, Optional
import os

class UnifiedDocumentParser:
    """统一文档解析器"""
    
    def __init__(self):
        self.parsers = {
            ".pdf": self._parse_pdf,
            ".docx": self._parse_docx,
            ".xlsx": self._parse_xlsx,
            ".html": self._parse_html,
            ".htm": self._parse_html,
            ".md": self._parse_text,
            ".txt": self._parse_text,
        }
    
    def parse(self, file_path: str) -> Dict:
        """
        解析文档（自动识别格式）
        
        Args:
            file_path: 文件路径
            
        Returns:
            统一格式的文档结构
        """
        ext = os.path.splitext(file_path)[1].lower()
        
        parser = self.parsers.get(ext)
        if not parser:
            raise ValueError(f"不支持的文件格式: {ext}")
        
        return parser(file_path)
    
    def _parse_pdf(self, file_path: str) -> Dict:
        """解析 PDF"""
        extractor = PyMuPDFExtractor(file_path)
        
        # 提取文本块
        blocks = extractor.extract_structured_blocks()
        
        # 检测布局
        if blocks:
            first_page_layout = PDFLayoutAnalyzer(file_path).analyze_layout(0)
            is_multi_column = first_page_layout.get("columns", 1) > 1
        else:
            is_multi_column = False
        
        # 提取表格
        table_extractor = PDFTableExtractor(file_path)
        tables = table_extractor.extract_with_camelot()
        if not tables:
            tables = table_extractor.extract_with_pdfplumber()
        
        return {
            "format": "pdf",
            "file_path": file_path,
            "blocks": blocks,
            "tables": tables,
            "text": extractor.extract_text(),
            "headings": extractor.detect_headings(),
            "toc": extractor.extract_toc(),
            "page_count": len(extractor.doc),
            "is_multi_column": is_multi_column
        }
    
    def _parse_docx(self, file_path: str) -> Dict:
        """解析 Word 文档"""
        parser = DocxParser(file_path)
        elements = parser.parse_all()
        
        return {
            "format": "docx",
            "file_path": file_path,
            "elements": [{
                "type": e.type,
                "text": e.text,
                "level": e.level,
                "style": e.style
            } for e in elements],
            "headings": parser.get_heading_hierarchy(),
            "markdown": parser.to_markdown()
        }
    
    def _parse_xlsx(self, file_path: str) -> Dict:
        """解析 Excel"""
        parser = ExcelParser(file_path)
        sheets = parser.parse_all_sheets()
        
        return {
            "format": "xlsx",
            "file_path": file_path,
            "sheets": sheets,
            "sheet_names": parser.get_sheet_names(),
            "total_sheets": len(sheets)
        }
    
    def _parse_html(self, file_path: str) -> Dict:
        """解析 HTML"""
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            html = f.read()
        
        # 使用 Readability
        readability = ReadabilityExtractor(html)
        clean_text = readability.extract_clean_text()
        
        # 使用 html2text
        converter = HTMLToMarkdown()
        markdown = converter.convert(html)
        
        return {
            "format": "html",
            "file_path": file_path,
            "text": clean_text,
            "markdown": markdown,
            "title": readability.extract()["title"]
        }
    
    def _parse_text(self, file_path: str) -> Dict:
        """解析纯文本/Markdown"""
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        
        return {
            "format": os.path.splitext(file_path)[1].lower(),
            "file_path": file_path,
            "text": text
        }


# 使用示例
parser = UnifiedDocumentParser()

# 自动识别格式并解析
result = parser.parse("report.pdf")
print(f"格式: {result['format']}")
print(f"文本长度: {len(result.get('text', ''))}")
print(f"表格数量: {len(result.get('tables', []))}")
```

---

## 9.8 最佳实践与常见问题

### 9.8.1 各格式处理策略总结

| 格式 | 主方案 | 备选方案 | 特别注意 |
|------|--------|---------|---------|
| PDF（文本型） | PyMuPDF | pdfminer.six | 多栏布局、页眉页脚 |
| PDF（扫描型） | PaddleOCR | Tesseract | 识别率、版面分析 |
| PDF（表格型） | Camelot | pdfplumber | 合并单元格、跨页表格 |
| Word | python-docx | mammoth | 样式继承、表格嵌套 |
| Excel | openpyxl | pandas | 合并单元格、多级表头 |
| HTML | Readability | BeautifulSoup | 动态内容、编码检测 |
| 纯文本 | 直接读取 | - | 编码检测（UTF-8/GBK） |

### 9.8.2 常见问题与解决方案

| 问题 | 现象 | 解决方案 |
|------|------|---------|
| PDF 提取乱码 | 文字变成乱码或方块 | 检查字体嵌入，使用 OCR 作为降级方案 |
| 表格提取不完整 | 合并单元格丢失 | 使用 pdfplumber 的单元格级定位 |
| OCR 精度不足 | 中英文混排识别差 | 使用 PaddleOCR 而非 Tesseract |
| 页眉页脚干扰 | 每页末尾出现重复内容 | 基于位置和文本重复度过滤 |
| 多栏错乱 | 左右栏内容混排 | 布局分析 + 按栏排序 |
| Excel 空行/空列 | 大量 None 值 | 去除全空行，限制最大列宽 |
| HTML 编码错误 | 中文显示乱码 | 使用 apparent_encoding 自动检测 |

---

## 本章小结

文档结构化解析是 RAG 系统处理企业文档的第一步，也是最容易被低估的一步。PDF、Word、Excel、HTML 等格式各有其独特的解析挑战——PDF 的布局多样性、扫描件的 OCR 需求、Excel 的表格结构识别、HTML 的噪声过滤——都需要针对性的处理策略。

本章的核心理念是**统一文档模型**：无论输入格式如何，最终都应转化为结构化的、保留层级关系的文档表示。这个统一表示可以是一个文档树，也可以是一组带元数据的文本块。在此基础上，后续的切分、索引、检索环节才能高效工作。

最佳实践总结：
- PDF 优先使用 PyMuPDF 提取，Camelot 处理表格，PaddleOCR 处理扫描件
- Word 文档的样式信息是宝贵的结构线索，不要丢弃
- Excel 的表头自动检测需要启发式算法，合并单元格需要特殊处理
- HTML 使用 Readability 算法提取正文效果最好
- 页眉页脚过滤和列布局分析是 PDF 处理的两大关键环节
- 始终保留原始文件路径和位置信息，方便追溯和调试
