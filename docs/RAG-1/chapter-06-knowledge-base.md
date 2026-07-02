# 第6章 企业知识数据治理

## 6.1 数据治理在 RAG 中的核心地位

### 6.1.1 为什么数据治理如此重要

在 RAG 系统中，"Garbage In, Garbage Out"（垃圾进，垃圾出）这个原则比任何其他环节都更加突出。检索系统的输出质量直接受制于知识库的数据质量。一个经过良好治理的知识库，即使在检索策略不太复杂的情况下，也能产出令人满意的结果；相反，一个充满噪声、重复、过时数据的知识库，即使使用最先进的检索模型，也难以获得好的效果。

**数据质量问题对 RAG 的影响**：

| 问题类型 | 表现 | 对 RAG 的影响 |
|---------|------|-------------|
| 数据重复 | 同一内容出现在多个文档块 | 检索结果冗余，LLM 上下文被重复信息占据 |
| 数据噪声 | 无关内容、广告、页眉页脚 | 检索到不相关内容，降低 LLM 回答准确率 |
| 格式不统一 | 同一概念用不同表达方式 | embedding 无法对齐语义，召回率下降 |
| 信息过时 | 旧版本文档未被更新 | LLM 基于过时信息作答，产生事实错误 |
| 敏感信息 | PII、机密数据未过滤 | 合规风险，安全漏洞 |
| 数据碎片化 | 上下文被不合理切分 | LLM 无法获取完整的语义单元 |

### 6.1.2 数据治理的核心流程

企业知识数据治理可以分为以下核心阶段：

```
数据源 → [采集] → 原始数据 → [清洗] → 干净数据 → [切分] → 文档块 → [索引] → 检索索引
                 ↑              ↑              ↑               ↑
             文件扫描       去重/格式化     分块策略       增量更新
             数据库同步      PII过滤       语义切分       版本管理
             API集成         噪声过滤       结构感知       一致性检查
```

每个阶段都有其特定的技术挑战和最佳实践。本章将逐一深入探讨。

---

## 6.2 数据采集

### 6.2.1 文件系统扫描

企业知识库最常见的起点是文件系统——网络共享目录、内部 Wiki 导出文件、项目文档等。

```python
import os
import hashlib
from pathlib import Path
from typing import List, Dict, Generator
from datetime import datetime
import json

class FileScanner:
    """文件系统扫描器"""
    
    SUPPORTED_EXTENSIONS = {
        ".txt", ".md", ".pdf", ".docx", ".xlsx",
        ".pptx", ".csv", ".json", ".yaml", ".yml",
        ".html", ".htm", ".xml"
    }
    
    def __init__(self, base_paths: List[str],
                 exclude_dirs: List[str] = None):
        """
        Args:
            base_paths: 扫描的根目录列表
            exclude_dirs: 排除的目录名列表
        """
        self.base_paths = [Path(p) for p in base_paths]
        self.exclude_dirs = set(exclude_dirs or [
            "__pycache__", ".git", "node_modules",
            "venv", ".venv", ".idea", ".vscode",
            "build", "dist", "target"
        ])
    
    def scan(self) -> Generator[Dict, None, None]:
        """
        扫描文件系统，生成文档元数据
        
        Yields:
            {
                "path": 文件绝对路径,
                "relative_path": 相对路径,
                "size": 文件大小(字节),
                "modified_at": 最后修改时间,
                "created_at": 创建时间,
                "extension": 文件扩展名,
                "file_hash": 文件内容哈希,
                "status": "new" | "modified" | "unchanged"
            }
        """
        for base_path in self.base_paths:
            if not base_path.exists():
                print(f"[Scanner] 目录不存在: {base_path}")
                continue
            
            for file_path in base_path.rglob("*"):
                # 跳过目录
                if not file_path.is_file():
                    continue
                
                # 检查扩展名
                if file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                    continue
                
                # 检查排除目录
                if self._is_excluded(file_path):
                    continue
                
                # 获取文件信息
                stat = file_path.stat()
                relative_path = file_path.relative_to(base_path)
                
                yield {
                    "path": str(file_path.absolute()),
                    "relative_path": str(relative_path),
                    "size": stat.st_size,
                    "modified_at": datetime.fromtimestamp(stat.st_mtime),
                    "created_at": datetime.fromtimestamp(stat.st_ctime),
                    "extension": file_path.suffix.lower(),
                    "file_hash": self._compute_hash(file_path),
                    "base_dir": str(base_path)
                }
    
    def _is_excluded(self, file_path: Path) -> bool:
        """检查文件是否在排除目录中"""
        for part in file_path.parts:
            if part in self.exclude_dirs:
                return True
        return False
    
    @staticmethod
    def _compute_hash(file_path: Path, algorithm: str = "sha256") -> str:
        """计算文件哈希值，用于变更检测"""
        hasher = hashlib.new(algorithm)
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()


# 使用示例
scanner = FileScanner(
    base_paths=["D:\\enterprise_docs", "D:\\wiki_export"],
    exclude_dirs=["__pycache__", ".git", "temp"]
)

for file_info in scanner.scan():
    print(f"{file_info['relative_path']} - "
          f"{file_info['size'] / 1024:.1f}KB - "
          f"{file_info['modified_at']}")
```

### 6.2.2 数据库同步（CDC）

对于存储在关系型数据库中的业务数据，需要使用 CDC（Change Data Capture）机制进行增量同步。

```python
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime
from typing import List, Dict, Optional

class DatabaseCDCSync:
    """基于 CDC 的数据库增量同步"""
    
    def __init__(self, conn_params: Dict):
        """
        Args:
            conn_params: 数据库连接参数
        """
        self.conn_params = conn_params
        self.connection = None
    
    def connect(self):
        """建立数据库连接"""
        self.connection = psycopg2.connect(**self.conn_params)
    
    def close(self):
        if self.connection:
            self.connection.close()
    
    def full_sync(self, table: str,
                  batch_size: int = 1000) -> Generator[List[Dict], None, None]:
        """
        全量同步
        
        Args:
            table: 表名
            batch_size: 每批处理行数
        """
        with self.connection.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            total = cur.fetchone()["count"]
            
            cur.execute(f"""
                SELECT * FROM {table}
                ORDER BY id
            """)
            
            batch = []
            for row in cur:
                batch.append(dict(row))
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
            
            if batch:
                yield batch
    
    def incremental_sync(self, table: str,
                         timestamp_column: str,
                         last_sync: datetime,
                         batch_size: int = 1000) -> Generator[List[Dict], None, None]:
        """
        增量同步（基于时间戳）
        
        Args:
            table: 表名
            timestamp_column: 时间戳列名（updated_at 等）
            last_sync: 上次同步时间
            batch_size: 每批处理行数
        """
        with self.connection.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(f"""
                SELECT * FROM {table}
                WHERE {timestamp_column} > %s
                ORDER BY {timestamp_column}
            """, (last_sync,))
            
            batch = []
            for row in cur:
                batch.append(dict(row))
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
            
            if batch:
                yield batch


class DebeziumCDC:
    """基于 Debezium + Kafka 的实时 CDC"""
    
    def __init__(self, kafka_bootstrap_servers: str,
                 topic_prefix: str):
        """
        Debezium 通过 Kafka Connect 捕获数据库变更事件
        
        Args:
            kafka_bootstrap_servers: Kafka 地址
            topic_prefix: 主题前缀（通常为 Debezium connector 名称）
        """
        from kafka import KafkaConsumer
        self.consumer = KafkaConsumer(
            bootstrap_servers=kafka_bootstrap_servers,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            auto_offset_reset="earliest",
            enable_auto_commit=True
        )
        self.topic_prefix = topic_prefix
    
    def subscribe(self, tables: List[str]):
        """
        订阅表的变更事件
        
        Args:
            tables: 要监听的表名列表
        """
        topics = [f"{self.topic_prefix}.public.{table}" for table in tables]
        self.consumer.subscribe(topics)
        print(f"[CDC] 订阅主题: {topics}")
    
    def listen(self):
        """
        监听变更事件
        
        事件格式:
        {
            "op": "c" | "u" | "d" | "r",  # create/update/delete/read
            "before": {...} | None,
            "after": {...} | None,
            "ts_ms": 时间戳
        }
        """
        for message in self.consumer:
            event = message.value
            if event.get("payload"):
                payload = event["payload"]
                yield {
                    "operation": payload["op"],
                    "table": message.topic.split(".")[-1],
                    "before": payload.get("before"),
                    "after": payload.get("after"),
                    "timestamp": payload.get("ts_ms")
                }
```

### 6.2.3 API 集成（Confluence / Notion / SharePoint）

企业知识通常分散在多个协作平台中，需要通过各自的 API 进行集成。

```python
import requests
from typing import List, Dict, Optional
from datetime import datetime
import time

class ConfluenceCollector:
    """Confluence 知识库采集"""
    
    def __init__(self, base_url: str, api_token: str, username: str):
        """
        Args:
            base_url: Confluence 基础 URL
            api_token: API Token
            username: 用户名
        """
        self.base_url = base_url.rstrip("/")
        self.auth = (username, api_token)
        self.session = requests.Session()
        self.session.auth = self.auth
    
    def get_spaces(self) -> List[Dict]:
        """获取所有空间"""
        url = f"{self.base_url}/rest/api/space"
        response = self.session.get(url, params={"limit": 100})
        response.raise_for_status()
        return response.json()["results"]
    
    def get_pages(self, space_key: str,
                  start: int = 0,
                  limit: int = 50) -> List[Dict]:
        """
        获取空间下的所有页面
        
        Args:
            space_key: 空间标识
            start: 分页起始
            limit: 每页数量
            
        Returns:
            页面列表
        """
        url = f"{self.base_url}/rest/api/content"
        params = {
            "spaceKey": space_key,
            "type": "page",
            "start": start,
            "limit": limit,
            "expand": "body.storage,version,ancestors"
        }
        response = self.session.get(url, params=params)
        response.raise_for_status()
        return response.json()["results"]
    
    def get_page_content(self, page_id: str) -> Dict:
        """
        获取页面内容
        
        Args:
            page_id: 页面 ID
            
        Returns:
            包含标题、内容、元数据的字典
        """
        url = f"{self.base_url}/rest/api/content/{page_id}"
        params = {
            "expand": "body.storage,version,metadata.labels,space"
        }
        response = self.session.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        # 提取 HTML 内容并转换为纯文本
        html_content = data["body"]["storage"]["value"]
        text_content = self._html_to_text(html_content)
        
        return {
            "id": data["id"],
            "title": data["title"],
            "content": text_content,
            "html_content": html_content,
            "version": data["version"]["number"],
            "space": data["space"]["key"],
            "created_at": data["version"]["when"],
            "author": data["version"]["by"]["displayName"],
            "url": f"{self.base_url}/spaces/{data['space']['key']}/pages/{data['id']}",
            "source": "confluence"
        }
    
    def sync_all_pages(self, space_keys: List[str] = None) -> List[Dict]:
        """
        同步所有空间的所有页面
        
        Args:
            space_keys: 空间标识列表，None 表示同步所有空间
            
        Returns:
            所有页面内容
        """
        if space_keys is None:
            spaces = self.get_spaces()
            space_keys = [s["key"] for s in spaces]
        
        all_pages = []
        for space_key in space_keys:
            print(f"[Confluence] 同步空间: {space_key}")
            
            start = 0
            while True:
                pages = self.get_pages(space_key, start=start)
                if not pages:
                    break
                
                for page in pages:
                    try:
                        content = self.get_page_content(page["id"])
                        all_pages.append(content)
                        print(f"  -> {content['title']} (v{content['version']})")
                        time.sleep(0.5)  # 限速
                    except Exception as e:
                        print(f"  !! 获取页面 {page['id']} 失败: {e}")
                
                start += len(pages)
        
        print(f"[Confluence] 同步完成: {len(all_pages)} 个页面")
        return all_pages
    
    @staticmethod
    def _html_to_text(html: str) -> str:
        """HTML 转纯文本"""
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        
        # 移除 script 和 style
        for tag in soup(["script", "style"]):
            tag.decompose()
        
        text = soup.get_text(separator="\n")
        # 清理多余空白
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        return "\n".join(lines)


class NotionCollector:
    """Notion 知识库采集"""
    
    def __init__(self, api_key: str):
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Notion-Version": "2022-06-28",
            "Content-Type": "application/json"
        }
        self.base_url = "https://api.notion.com/v1"
    
    def search_pages(self, query: str = None) -> List[Dict]:
        """
        搜索页面
        
        Args:
            query: 搜索关键词，None 则返回所有页面
        """
        url = f"{self.base_url}/search"
        body = {}
        if query:
            body["query"] = query
        
        response = requests.post(url, headers=self.headers, json=body)
        response.raise_for_status()
        return response.json()["results"]
    
    def get_page_content(self, page_id: str) -> Dict:
        """
        获取页面内容（包含所有块）
        
        Args:
            page_id: 页面 ID
        """
        url = f"{self.base_url}/blocks/{page_id}/children"
        blocks = []
        start_cursor = None
        
        while True:
            params = {"page_size": 100}
            if start_cursor:
                params["start_cursor"] = start_cursor
            
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            blocks.extend(data["results"])
            
            if not data.get("has_more"):
                break
            start_cursor = data["next_cursor"]
        
        # 将块转换为纯文本
        text = self._blocks_to_text(blocks)
        
        return {
            "id": page_id,
            "content": text,
            "blocks": blocks,
            "source": "notion"
        }
    
    def _blocks_to_text(self, blocks: List[Dict]) -> str:
        """将 Notion 块转换为纯文本"""
        text_parts = []
        for block in blocks:
            block_type = block.get("type", "")
            rich_text = block.get(block_type, {}).get("rich_text", [])
            
            for rt in rich_text:
                text_parts.append(rt.get("plain_text", ""))
            
            text_parts.append("\n")
        
        return "".join(text_parts)
```

### 6.2.4 网络爬虫（Scrapy）

对于公开的文档网站或内部 Wiki，可以使用网络爬虫进行采集。

```python
import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.linkextractors import LinkExtractor
from typing import Set

class DocumentationSpider(scrapy.Spider):
    """文档网站爬虫"""
    
    name = "doc_spider"
    
    def __init__(self, start_urls: list,
                 allowed_domains: list,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_urls = start_urls
        self.allowed_domains = allowed_domains
        self.visited_urls: Set[str] = set()
        
        # 链接提取器：只提取同域名下的文档链接
        self.link_extractor = LinkExtractor(
            allow_domains=allowed_domains,
            deny_extensions=[
                "pdf", "zip", "rar", "gz", "exe",
                "jpg", "png", "gif", "mp4", "avi"
            ]
        )
    
    def parse(self, response):
        """解析页面"""
        url = response.url
        
        if url in self.visited_urls:
            return
        self.visited_urls.add(url)
        
        # 提取标题
        title = response.css("title::text").get(default="").strip()
        
        # 提取正文内容
        content = self._extract_main_content(response)
        
        yield {
            "url": url,
            "title": title,
            "content": content,
            "crawled_at": scrapy.utils.log.get_scrapy_root_handler()
        }
        
        # 提取并跟进链接
        for link in self.link_extractor.extract_links(response):
            yield scrapy.Request(link.url, callback=self.parse)
    
    def _extract_main_content(self, response) -> str:
        """
        提取页面主要内容
        
        使用 readability 算法或简单的正文提取
        """
        # 策略1: 尝试提取 article 标签
        article = response.css("article")
        if article:
            return article.get()
        
        # 策略2: 尝试提取 main 标签
        main = response.css("main")
        if main:
            return main.get()
        
        # 策略3: 提取 body 并移除导航/页脚
        body = response.css("body").get()
        return body or ""


def run_spider(start_urls: List[str],
               allowed_domains: List[str],
               output_file: str = "crawled_docs.json"):
    """
    运行爬虫
    
    Args:
        start_urls: 起始 URL 列表
        allowed_domains: 允许的域名列表
        output_file: 输出文件路径
    """
    process = CrawlerProcess(settings={
        "FEEDS": {
            output_file: {"format": "jsonlines"}
        },
        "USER_AGENT": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36",
        "ROBOTSTXT_OBEY": True,
        "CONCURRENT_REQUESTS": 8,
        "DOWNLOAD_DELAY": 1,
        "DEPTH_LIMIT": 5
    })
    
    process.crawl(DocumentationSpider,
                  start_urls=start_urls,
                  allowed_domains=allowed_domains)
    process.start()
```

### 6.2.5 消息队列集成（Kafka）

对于流式数据源，通过 Kafka 实现实时数据接入：

```python
from kafka import KafkaProducer, KafkaConsumer
import json
from typing import Callable

class DataIngestionPipeline:
    """基于 Kafka 的数据接入管道"""
    
    def __init__(self, bootstrap_servers: str = "localhost:9092"):
        self.bootstrap_servers = bootstrap_servers
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            acks="all",  # 等待所有副本确认
            retries=3
        )
    
    def ingest(self, topic: str, data: Dict):
        """
        将数据发送到 Kafka 主题
        
        Args:
            topic: Kafka 主题名
            data: 数据字典
        """
        future = self.producer.send(topic, value=data)
        result = future.get(timeout=10)
        print(f"[Ingestion] 发送到 {topic}: "
              f"partition={result.partition}, offset={result.offset}")
    
    def start_consumer(self, topic: str,
                       processor: Callable[[Dict], None],
                       group_id: str = "rag-ingestion"):
        """
        启动消费者处理数据
        
        Args:
            topic: 主题名
            processor: 数据处理函数
            group_id: 消费者组 ID
        """
        consumer = KafkaConsumer(
            topic,
            bootstrap_servers=self.bootstrap_servers,
            group_id=group_id,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            auto_offset_reset="earliest",
            enable_auto_commit=True
        )
        
        print(f"[Consumer] 开始消费主题: {topic}")
        for message in consumer:
            try:
                processor(message.value)
                print(f"[Consumer] 处理完成: offset={message.offset}")
            except Exception as e:
                print(f"[Consumer] 处理失败: {e}")
```

---

## 6.3 数据清洗

### 6.3.1 去重（MinHash 与 SimHash）

文档去重是数据清洗中最关键的一步。重复内容会导致检索结果冗余，浪费 LLM 上下文窗口。

**MinHash 算法**：适用于大规模文本集合的近似去重，通过 Jaccard 相似度估计来判断文档是否重复。

**SimHash 算法**：Google 提出的指纹去重算法，将文本映射为固定长度的指纹（通常 64 位或 128 位），通过汉明距离判断相似度。

```python
import hashlib
import numpy as np
from typing import List, Set, Tuple
from datasketch import MinHash, MinHashLSH

class DocumentDeduplicator:
    """文档去重器"""
    
    def __init__(self, threshold: float = 0.8,
                 num_perm: int = 128):
        """
        Args:
            threshold: Jaccard 相似度阈值，超过则视为重复
            num_perm: MinHash 的排列数，越大精度越高
        """
        self.threshold = threshold
        self.num_perm = num_perm
        self.lsh = MinHashLSH(
            threshold=threshold,
            num_perm=num_perm
        )
        self.documents: Dict[str, str] = {}
    
    def _tokenize(self, text: str) -> Set[str]:
        """将文本转换为 shingle 集合"""
        # 使用 3-gram（字符级 shingle）
        shingles = set()
        for i in range(len(text) - 2):
            shingles.add(text[i:i+3])
        return shingles
    
    def _compute_minhash(self, text: str) -> MinHash:
        """计算文本的 MinHash 签名"""
        m = MinHash(num_perm=self.num_perm)
        for shingle in self._tokenize(text):
            m.update(shingle.encode("utf-8"))
        return m
    
    def add_document(self, doc_id: str, text: str):
        """添加文档到去重索引"""
        m = self._compute_minhash(text)
        self.lsh.insert(doc_id, m)
        self.documents[doc_id] = text
    
    def find_duplicates(self, text: str,
                        return_similarities: bool = False) -> List:
        """
        查找与给定文本相似的已存在文档
        
        Args:
            text: 查询文本
            return_similarities: 是否返回相似度
            
        Returns:
            重复文档 ID 列表，或 (doc_id, similarity) 元组列表
        """
        m = self._compute_minhash(text)
        results = self.lsh.query(m)
        
        if return_similarities:
            similarities = []
            query_shingles = self._tokenize(text)
            for doc_id in results:
                doc_shingles = self._tokenize(self.documents[doc_id])
                jaccard = len(query_shingles & doc_shingles) / \
                         len(query_shingles | doc_shingles)
                similarities.append((doc_id, jaccard))
            return sorted(similarities, key=lambda x: x[1], reverse=True)
        
        return results
    
    def dedup_collection(self, documents: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """
        对文档集合进行去重
        
        Args:
            documents: [(doc_id, text), ...]
            
        Returns:
            去重后的文档列表
        """
        # 按文本长度降序排列（保留最长的版本）
        sorted_docs = sorted(documents, key=lambda x: len(x[1]), reverse=True)
        
        deduped = []
        duplicate_count = 0
        
        for doc_id, text in sorted_docs:
            duplicates = self.find_duplicates(text)
            if not duplicates:
                self.add_document(doc_id, text)
                deduped.append((doc_id, text))
            else:
                duplicate_count += 1
        
        print(f"[Dedup] 总计: {len(documents)}, "
              f"去重后: {len(deduped)}, "
              f"重复: {duplicate_count}")
        
        return deduped


class SimHashDeduplicator:
    """基于 SimHash 的指纹去重"""
    
    def __init__(self, fingerprint_bits: int = 64,
                 hamming_threshold: int = 3):
        """
        Args:
            fingerprint_bits: 指纹位数
            hamming_threshold: 汉明距离阈值
        """
        self.fingerprint_bits = fingerprint_bits
        self.hamming_threshold = hamming_threshold
        self.fingerprints: List[Tuple[str, int]] = []  # [(doc_id, fingerprint)]
    
    def compute_fingerprint(self, text: str) -> int:
        """
        计算文本的 SimHash 指纹
        
        步骤：
        1. 将文本分词并计算每个词的哈希值
        2. 对每个哈希位的每一位，根据词频加权累加
        3. 最终每一位取符号（正为1，负为0）
        """
        import jieba
        words = jieba.cut(text)
        
        # 初始化 V 向量
        v = np.zeros(self.fingerprint_bits, dtype=np.int64)
        
        for word in words:
            if len(word.strip()) == 0:
                continue
            
            # 计算词的哈希值
            word_hash = hashlib.md5(word.encode("utf-8")).digest()
            hash_int = int.from_bytes(word_hash[:8], byteorder="big")
            
            # 对每一位进行加权
            for i in range(self.fingerprint_bits):
                bit = (hash_int >> i) & 1
                if bit == 1:
                    v[i] += 1
                else:
                    v[i] -= 1
        
        # 生成指纹
        fingerprint = 0
        for i in range(self.fingerprint_bits):
            if v[i] > 0:
                fingerprint |= (1 << i)
        
        return fingerprint
    
    def hamming_distance(self, fp1: int, fp2: int) -> int:
        """计算两个指纹的汉明距离"""
        xor = fp1 ^ fp2
        return bin(xor).count("1")
    
    def is_duplicate(self, fingerprint: int) -> bool:
        """检查指纹是否与已有的相似"""
        for _, existing_fp in self.fingerprints:
            if self.hamming_distance(fingerprint, existing_fp) <= self.hamming_threshold:
                return True
        return False
    
    def add_document(self, doc_id: str, text: str) -> bool:
        """
        添加文档（如果是重复则跳过）
        
        Returns:
            True 表示新文档，False 表示重复
        """
        fp = self.compute_fingerprint(text)
        if self.is_duplicate(fp):
            return False
        
        self.fingerprints.append((doc_id, fp))
        return True
```

### 6.3.2 格式归一化

不同来源的文档格式各异，需要进行归一化处理：

```python
import re
from typing import Optional

class TextNormalizer:
    """文本归一化工具"""
    
    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """归一化空白字符"""
        # 将各种空白字符替换为普通空格
        text = re.sub(r"[\t\n\r\f\v]+", " ", text)
        # 合并多个连续空格
        text = re.sub(r" +", " ", text)
        return text.strip()
    
    @staticmethod
    def normalize_unicode(text: str) -> str:
        """归一化 Unicode 字符"""
        import unicodedata
        # NFC 归一化（将组合字符转换为标准形式）
        return unicodedata.normalize("NFKC", text)
    
    @staticmethod
    def normalize_quotes(text: str) -> str:
        """归一化引号"""
        text = text.replace("“", '"').replace("”", '"')
        text = text.replace("‘", "'").replace("’", "'")
        text = text.replace("「", "[").replace("」", "]")
        return text
    
    @staticmethod
    def normalize_dashes(text: str) -> str:
        """归一化破折号"""
        text = re.sub(r"[–—]", "-", text)
        return text
    
    @staticmethod
    def normalize_fullwidth(text: str) -> str:
        """全角字符转半角"""
        result = []
        for char in text:
            code = ord(char)
            if 0xFF01 <= code <= 0xFF5E:
                # 全角字母数字转半角
                result.append(chr(code - 0xFEE0))
            elif code == 0x3000:
                # 全角空格转半角
                result.append(" ")
            else:
                result.append(char)
        return "".join(result)
    
    def normalize_all(self, text: str) -> str:
        """执行所有归一化"""
        text = self.normalize_unicode(text)
        text = self.normalize_fullwidth(text)
        text = self.normalize_quotes(text)
        text = self.normalize_dashes(text)
        text = self.normalize_whitespace(text)
        return text
```

### 6.3.3 PII 检测与脱敏

对于企业知识库，PII（个人身份信息）的检测和脱敏是合规的基本要求：

```python
import re
from typing import List, Tuple

class PIIDetector:
    """PII 检测器"""
    
    # 中文 PII 模式
    PATTERNS = {
        "phone": r"1[3-9]\d{9}",
        "id_card": r"[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx]",
        "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "ip_address": r"(?:\d{1,3}\.){3}\d{1,3}",
        "bank_card": r"\d{16}|\d{19}",
        "passport": r"[A-Za-z]\d{8}",
        "license_plate": r"[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤川青藏琼宁][A-Z][A-HJ-NP-Z0-9]{4,5}",
    }
    
    def __init__(self, patterns: Dict[str, str] = None):
        self.patterns = patterns or self.PATTERNS
        self.compiled = {
            name: re.compile(pattern)
            for name, pattern in self.patterns.items()
        }
    
    def detect(self, text: str) -> List[Tuple[str, str, int, int]]:
        """
        检测文本中的 PII
        
        Returns:
            [(type, value, start_pos, end_pos), ...]
        """
        findings = []
        for pii_type, pattern in self.compiled.items():
            for match in pattern.finditer(text):
                findings.append((
                    pii_type,
                    match.group(),
                    match.start(),
                    match.end()
                ))
        return findings
    
    def redact(self, text: str,
               replacement: str = "[REDACTED]",
               mask_char: str = "*") -> str:
        """
        对 PII 进行脱敏处理
        
        Args:
            text: 原始文本
            replacement: 替换字符串
            mask_char: 掩码字符
            
        Returns:
            脱敏后的文本
        """
        findings = self.detect(text)
        
        # 按位置降序排列（从后往前替换，避免位置偏移）
        findings.sort(key=lambda x: x[3], reverse=True)
        
        result = text
        for pii_type, value, start, end in findings:
            if pii_type in ("phone", "id_card", "bank_card"):
                # 保留前后各 4 位，中间掩码
                if len(value) > 8:
                    masked = value[:4] + mask_char * (len(value) - 8) + value[-4:]
                else:
                    masked = mask_char * len(value)
            elif pii_type == "email":
                # 保留域名，掩码用户名
                at_pos = value.index("@")
                masked = mask_char * min(at_pos, 4) + value[at_pos:]
            else:
                masked = replacement
            
            result = result[:start] + masked + result[end:]
        
        return result
    
    def summary(self, text: str) -> Dict[str, int]:
        """返回 PII 检测摘要"""
        findings = self.detect(text)
        summary = {}
        for pii_type, _, _, _ in findings:
            summary[pii_type] = summary.get(pii_type, 0) + 1
        return summary
```

### 6.3.4 HTML 清洗

从网页或 Wiki 采集的内容包含大量 HTML 标签，需要清洗以提取纯文本：

```python
from bs4 import BeautifulSoup, Tag
from typing import List

class HTMLCleaner:
    """HTML 清洗器"""
    
    # 需要移除的标签
    REMOVE_TAGS = {
        "script", "style", "nav", "footer", "header",
        "aside", "iframe", "noscript", "form", "input",
        "button", "select", "textarea", "svg", "canvas",
        "img", "video", "audio", "embed", "object",
        "applet", "frameset", "frame", "noframes"
    }
    
    # 需要替换为文本的标签
    REPLACE_TAGS = {
        "br": "\n",
        "hr": "\n---\n",
        "p": "\n\n",
        "div": "\n",
        "tr": "\n",
        "th": "\t",
        "td": "\t",
        "li": "\n  - ",
        "h1": "\n\n# ",
        "h2": "\n\n## ",
        "h3": "\n\n### ",
        "h4": "\n\n#### ",
    }
    
    def clean(self, html: str) -> str:
        """
        清洗 HTML 提取纯文本
        
        Args:
            html: 原始 HTML
            
        Returns:
            清洗后的纯文本
        """
        soup = BeautifulSoup(html, "html.parser")
        
        # 移除不需要的标签
        for tag in soup.find_all(self.REMOVE_TAGS):
            tag.decompose()
        
        # 处理需要替换的标签
        for tag_name, replacement in self.REPLACE_TAGS.items():
            for tag in soup.find_all(tag_name):
                # 在标签后插入替换文本
                if tag.string:
                    tag.insert_after(replacement)
        
        # 获取纯文本
        text = soup.get_text(separator=" ")
        
        # 清理多余空白
        text = re.sub(r" +", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        
        return text.strip()
    
    def extract_links(self, html: str) -> List[Dict[str, str]]:
        """提取页面中的所有链接"""
        soup = BeautifulSoup(html, "html.parser")
        links = []
        
        for a_tag in soup.find_all("a", href=True):
            links.append({
                "text": a_tag.get_text(strip=True),
                "url": a_tag["href"],
                "title": a_tag.get("title", "")
            })
        
        return links
    
    def extract_tables(self, html: str) -> List[List[List[str]]]:
        """提取 HTML 表格"""
        soup = BeautifulSoup(html, "html.parser")
        tables = []
        
        for table in soup.find_all("table"):
            rows = []
            for tr in table.find_all("tr"):
                cells = []
                for cell in tr.find_all(["td", "th"]):
                    cells.append(cell.get_text(strip=True))
                if cells:
                    rows.append(cells)
            if rows:
                tables.append(rows)
        
        return tables
```

---

## 6.4 文本切分

### 6.4.1 切分策略对比

文本切分（Chunking）是将长文档分割成适合检索和 LLM 处理的片段。切分策略直接影响检索效果。

| 策略 | 原理 | 优点 | 缺点 | 适用场景 |
|------|------|------|------|---------|
| 固定大小 + 重叠 | 按 token 数切分，相邻块有重叠 | 简单可控，实现方便 | 可能切断语义单元 | 通用场景 |
| 语义切分 | 按句子/段落自然边界切分 | 保持语义完整性 | 块大小不均匀 | 散文、报告 |
| 结构感知 | 按标题/章节切分 | 保持文档结构 | 依赖文档结构 | 结构化文档 |
| 递归切分 | 先大块再细分，直到满足大小 | 灵活自适应 | 实现复杂 | 多层级文档 |
| 基于模型切分 | 用模型判断语义边界 | 最精确 | 速度慢，成本高 | 高质量场景 |

### 6.4.2 Token 级固定大小切分

```python
from typing import List, Iterator
import tiktoken

class TokenChunker:
    """基于 Token 计数的固定大小切分"""
    
    def __init__(self, chunk_size: int = 512,
                 chunk_overlap: int = 128,
                 encoding_name: str = "cl100k_base"):
        """
        Args:
            chunk_size: 每个块的 token 数
            chunk_overlap: 相邻块的重叠 token 数
            encoding_name: tokenizer 名称
                - cl100k_base: GPT-4, GPT-3.5
                - p50k_base: Codex
                - r50k_base: GPT-3
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding(encoding_name)
    
    def split_text(self, text: str) -> List[str]:
        """
        按 token 数切分文本
        
        Args:
            text: 输入文本
            
        Returns:
            切分后的文本块列表
        """
        tokens = self.encoding.encode(text)
        chunks = []
        
        if len(tokens) <= self.chunk_size:
            return [text]
        
        start = 0
        while start < len(tokens):
            end = start + self.chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoding.decode(chunk_tokens)
            chunks.append(chunk_text)
            
            # 移动起始位置（减去重叠部分）
            start += self.chunk_size - self.chunk_overlap
        
        return chunks
    
    def count_tokens(self, text: str) -> int:
        """计算文本的 token 数"""
        return len(self.encoding.encode(text))


class RecursiveCharacterChunker:
    """递归字符级切分（LangChain 风格）"""
    
    def __init__(self, chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 separators: List[str] = None):
        """
        Args:
            chunk_size: 块大小（字符数）
            chunk_overlap: 重叠字符数
            separators: 分隔符优先级列表
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or [
            "\n\n",  # 段落
            "\n",    # 行
            "。",    # 句号
            "！",    # 感叹号
            "？",    # 问号
            "；",    # 分号
            "，",    # 逗号
            " ",     # 空格
            ""       # 字符级
        ]
    
    def split_text(self, text: str) -> List[str]:
        """
        递归切分文本
        
        策略：从最大的分隔符开始尝试，如果切分后块仍然太大，
        使用更小的分隔符继续切分。
        """
        return self._split(text, self.separators)
    
    def _split(self, text: str, separators: List[str]) -> List[str]:
        """递归切分的核心逻辑"""
        chunks = []
        final_chunks = []
        
        if not separators:
            # 没有可用分隔符，按字符切分
            return self._split_by_chars(text)
        
        separator = separators[0]
        remaining_separators = separators[1:]
        
        # 使用当前分隔符切分
        splits = text.split(separator) if separator else list(text)
        
        # 合并小块
        current_chunk = ""
        for split in splits:
            if not split:
                continue
            
            candidate = current_chunk + (separator if current_chunk else "") + split
            
            if len(candidate) <= self.chunk_size:
                current_chunk = candidate
            else:
                if current_chunk:
                    if len(current_chunk) > self.chunk_size and remaining_separators:
                        # 当前块仍然太大，递归切分
                        final_chunks.extend(
                            self._split(current_chunk, remaining_separators)
                        )
                    else:
                        final_chunks.append(current_chunk)
                
                if len(split) <= self.chunk_size:
                    current_chunk = split
                elif remaining_separators:
                    final_chunks.extend(self._split(split, remaining_separators))
                    current_chunk = ""
                else:
                    current_chunk = split
        
        # 处理最后一个块
        if current_chunk:
            if len(current_chunk) > self.chunk_size and remaining_separators:
                final_chunks.extend(
                    self._split(current_chunk, remaining_separators)
                )
            else:
                final_chunks.append(current_chunk)
        
        # 添加重叠
        result = []
        for i, chunk in enumerate(final_chunks):
            if i > 0 and self.chunk_overlap > 0:
                # 从前一个块末尾取重叠部分
                prev_chunk = final_chunks[i - 1]
                overlap_text = prev_chunk[-self.chunk_overlap:] if len(prev_chunk) > self.chunk_overlap else prev_chunk
                chunk = overlap_text + chunk
            result.append(chunk)
        
        return result
    
    def _split_by_chars(self, text: str) -> List[str]:
        """按字符切分"""
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk = text[i:i + self.chunk_size]
            if chunk:
                chunks.append(chunk)
        return chunks
```

### 6.4.3 语义切分

语义切分利用 NLP 技术识别句子和段落的自然边界：

```python
import re
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import numpy as np

class SemanticChunker:
    """基于语义的文档切分"""
    
    def __init__(self, model_name: str = "BAAI/bge-m3",
                 similarity_threshold: float = 0.6,
                 max_chunk_size: int = 500):
        """
        语义切分：当相邻句子/段落的语义相似度低于阈值时切分
        
        Args:
            model_name: embedding 模型
            similarity_threshold: 切分阈值（低于此值则切分）
            max_chunk_size: 块最大字符数
        """
        self.model = SentenceTransformer(model_name)
        self.threshold = similarity_threshold
        self.max_chunk_size = max_chunk_size
    
    def split_by_sentences(self, text: str) -> List[str]:
        """按句子切分"""
        # 中文句子分割
        sentences = re.split(r"(?<=[。！？\n])\s*", text)
        return [s.strip() for s in sentences if s.strip()]
    
    def split_by_paragraphs(self, text: str) -> List[str]:
        """按段落切分"""
        paragraphs = text.split("\n\n")
        return [p.strip() for p in paragraphs if p.strip()]
    
    def semantic_split(self, text: str) -> List[Dict]:
        """
        基于语义相似度的切分
        
        Returns:
            [{"text": 块文本, "embedding": 块 embedding}, ...]
        """
        sentences = self.split_by_sentences(text)
        if len(sentences) <= 1:
            return [{"text": text, "embedding": None}]
        
        # 计算每个句子的 embedding
        embeddings = self.model.encode(sentences, normalize_embeddings=True)
        
        # 识别切分点
        chunks = []
        current_chunk = [sentences[0]]
        current_embedding = embeddings[0]
        
        for i in range(1, len(sentences)):
            sentence = sentences[i]
            sentence_embedding = embeddings[i]
            
            # 计算当前块整体与下一句的相似度
            similarity = np.dot(current_embedding, sentence_embedding)
            
            # 检查是否超出最大长度
            candidate_len = len("".join(current_chunk + [sentence]))
            
            if similarity < self.threshold or candidate_len > self.max_chunk_size:
                # 保存当前块
                chunk_text = "".join(current_chunk)
                chunks.append({
                    "text": chunk_text,
                    "embedding": current_embedding
                })
                # 开始新块
                current_chunk = [sentence]
                current_embedding = sentence_embedding
            else:
                current_chunk.append(sentence)
                # 更新块 embedding（简单平均）
                current_embedding = (
                    current_embedding * (len(current_chunk) - 1) + sentence_embedding
                ) / len(current_chunk)
        
        # 最后一个块
        if current_chunk:
            chunk_text = "".join(current_chunk)
            chunks.append({
                "text": chunk_text,
                "embedding": current_embedding
            })
        
        return chunks
```

### 6.4.4 结构感知切分

对于有明确层级结构的文档（Markdown、HTML、LaTeX），按结构切分可以保持语义完整性：

```python
from typing import List, Dict, Optional
import re

class StructureAwareChunker:
    """结构感知的文档切分"""
    
    def __init__(self, max_section_size: int = 1500):
        """
        Args:
            max_section_size: 每个 section 的最大字符数
        """
        self.max_section_size = max_section_size
    
    def split_markdown(self, markdown_text: str) -> List[Dict]:
        """
        按 Markdown 标题结构切分
        
        Returns:
            [{"title": 标题, "content": 内容, "level": 标题层级, "hierarchy": 完整路径}, ...]
        """
        lines = markdown_text.split("\n")
        chunks = []
        
        # 解析标题结构
        current_section = {
            "title": "前言",
            "content": [],
            "level": 0,
            "hierarchy": ["前言"]
        }
        
        # 标题层级栈
        hierarchy_stack = [("前言", 0)]
        
        for line in lines:
            header_match = re.match(r"^(#{1,6})\s+(.+)$", line)
            
            if header_match:
                # 保存当前 section
                if current_section["content"]:
                    chunks.append(self._finalize_chunk(current_section))
                
                # 开始新 section
                level = len(header_match.group(1))
                title = header_match.group(2)
                
                # 更新层级栈
                while hierarchy_stack and hierarchy_stack[-1][1] >= level:
                    hierarchy_stack.pop()
                hierarchy_stack.append((title, level))
                
                current_section = {
                    "title": title,
                    "content": [],
                    "level": level,
                    "hierarchy": [h[0] for h in hierarchy_stack]
                }
            else:
                current_section["content"].append(line)
        
        # 最后一个 section
        if current_section["content"]:
            chunks.append(self._finalize_chunk(current_section))
        
        # 处理超大的 section（递归切分）
        final_chunks = []
        for chunk in chunks:
            if len(chunk["content"]) > self.max_section_size:
                sub_chunks = self._split_large_section(chunk)
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(chunk)
        
        return final_chunks
    
    def _finalize_chunk(self, section: Dict) -> Dict:
        """将 section 格式化为最终块"""
        content = "\n".join(section["content"]).strip()
        return {
            "title": section["title"],
            "content": content,
            "level": section["level"],
            "hierarchy": " > ".join(section["hierarchy"]),
            "size": len(content)
        }
    
    def _split_large_section(self, section: Dict) -> List[Dict]:
        """切分过大的 section"""
        content = section["content"]
        sub_chunks = []
        
        # 按段落切分
        paragraphs = content.split("\n\n")
        current = []
        current_size = 0
        
        for para in paragraphs:
            if current_size + len(para) > self.max_section_size and current:
                sub_chunks.append({
                    "title": f"{section['title']} (续 {len(sub_chunks) + 1})",
                    "content": "\n\n".join(current),
                    "level": section["level"] + 1,
                    "hierarchy": section["hierarchy"] + [f"续 {len(sub_chunks) + 1}"]
                })
                current = [para]
                current_size = len(para)
            else:
                current.append(para)
                current_size += len(para)
        
        if current:
            sub_chunks.append({
                "title": f"{section['title']} (续 {len(sub_chunks) + 1})",
                "content": "\n\n".join(current),
                "level": section["level"] + 1,
                "hierarchy": section["hierarchy"] + [f"续 {len(sub_chunks) + 1}"]
            })
        
        return sub_chunks
```

### 6.4.5 切分参数调优

切分参数（chunk_size, overlap）对检索效果有显著影响：

```python
from typing import List, Dict, Callable
import numpy as np

class ChunkingOptimizer:
    """切分参数优化器"""
    
    def __init__(self, test_queries: List[str],
                 test_documents: List[str],
                 evaluate_fn: Callable):
        """
        Args:
            test_queries: 测试查询列表
            test_documents: 测试文档列表
            evaluate_fn: 评估函数，接受 (chunks, queries) 返回指标
        """
        self.test_queries = test_queries
        self.test_documents = test_documents
        self.evaluate_fn = evaluate_fn
    
    def grid_search(self, chunk_sizes: List[int],
                    overlaps: List[int]) -> List[Dict]:
        """
        网格搜索最佳切分参数
        
        Args:
            chunk_sizes: chunk_size 候选值
            overlaps: overlap 候选值
            
        Returns:
            各参数组合的评估结果
        """
        results = []
        
        for chunk_size in chunk_sizes:
            for overlap in overlaps:
                if overlap >= chunk_size:
                    continue  # 重叠不能大于块大小
                
                # 使用当前参数切分
                chunker = RecursiveCharacterChunker(
                    chunk_size=chunk_size,
                    chunk_overlap=overlap
                )
                
                all_chunks = []
                for doc in self.test_documents:
                    chunks = chunker.split_text(doc)
                    all_chunks.extend(chunks)
                
                # 评估
                metrics = self.evaluate_fn(all_chunks, self.test_queries)
                
                results.append({
                    "chunk_size": chunk_size,
                    "overlap": overlap,
                    "num_chunks": len(all_chunks),
                    **metrics
                })
                
                print(f"[Optimize] chunk_size={chunk_size}, "
                      f"overlap={overlap} -> "
                      f"Recall@10={metrics.get('recall@10', 'N/A')}")
        
        # 按 Recall@10 排序
        results.sort(key=lambda r: r.get("recall@10", 0), reverse=True)
        return results
    
    @staticmethod
    def recommend_params(num_docs: int, avg_doc_len: int,
                         use_case: str = "general") -> Dict:
        """
        根据文档特征推荐切分参数
        
        Args:
            num_docs: 文档数量
            avg_doc_len: 平均文档长度（字符数）
            use_case: 使用场景
                - "general": 通用
                - "qa": 问答（需要较短的精确块）
                - "summary": 摘要（需要较长的上下文块）
                - "code": 代码（按函数/类切分）
        """
        recommendations = {
            "general": {
                "chunk_size": 500,
                "overlap": 100,
                "strategy": "recursive"
            },
            "qa": {
                "chunk_size": 300,
                "overlap": 50,
                "strategy": "semantic"
            },
            "summary": {
                "chunk_size": 1000,
                "overlap": 200,
                "strategy": "structure_aware"
            },
            "code": {
                "chunk_size": 800,
                "overlap": 0,
                "strategy": "structure_aware"
            }
        }
        
        base = recommendations.get(use_case, recommendations["general"])
        
        # 根据文档长度调整
        if avg_doc_len < 200:
            # 短文档：不需要切分
            base["chunk_size"] = avg_doc_len
            base["overlap"] = 0
        elif avg_doc_len > 10000:
            # 长文档：增大块大小
            base["chunk_size"] = min(base["chunk_size"] * 2, 2000)
        
        return base
```

---

## 6.5 存储与更新

### 6.5.1 全量索引与增量更新

```python
from datetime import datetime
from typing import List, Dict, Optional, Set
import json
import os

class IndexManager:
    """索引管理器"""
    
    def __init__(self, index_path: str,
                 metadata_path: str = None):
        """
        Args:
            index_path: 索引存储路径
            metadata_path: 元数据存储路径
        """
        self.index_path = index_path
        self.metadata_path = metadata_path or f"{index_path}_metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """加载索引元数据"""
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "last_full_index": None,
            "last_incremental": None,
            "indexed_files": {},
            "stats": {
                "total_documents": 0,
                "total_chunks": 0,
                "index_size_bytes": 0
            }
        }
    
    def _save_metadata(self):
        """保存索引元数据"""
        os.makedirs(os.path.dirname(self.metadata_path), exist_ok=True)
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
    
    def full_index(self, documents: List[Dict]):
        """
        全量索引
        
        Args:
            documents: 文档列表
        """
        print(f"[Index] 开始全量索引: {len(documents)} 个文档")
        
        # 清空旧索引
        self.metadata["indexed_files"] = {}
        
        # 构建新索引
        for doc in documents:
            file_path = doc.get("path", doc.get("id", ""))
            file_hash = doc.get("file_hash", "")
            
            self.metadata["indexed_files"][file_path] = {
                "hash": file_hash,
                "indexed_at": datetime.now().isoformat(),
                "size": doc.get("size", 0)
            }
        
        self.metadata["last_full_index"] = datetime.now().isoformat()
        self.metadata["stats"]["total_documents"] = len(documents)
        self._save_metadata()
        
        print(f"[Index] 全量索引完成: {len(documents)} 个文档")
    
    def incremental_update(self, new_files: List[Dict],
                           modified_files: List[Dict],
                           deleted_files: List[str]):
        """
        增量更新索引
        
        Args:
            new_files: 新文件列表
            modified_files: 修改过的文件列表
            deleted_files: 删除的文件路径列表
        """
        print(f"[Index] 增量更新: "
              f"+{len(new_files)} 新增, "
              f"~{len(modified_files)} 修改, "
              f"-{len(deleted_files)} 删除")
        
        # 新增文件
        for doc in new_files:
            file_path = doc.get("path", doc.get("id", ""))
            self.metadata["indexed_files"][file_path] = {
                "hash": doc.get("file_hash", ""),
                "indexed_at": datetime.now().isoformat(),
                "size": doc.get("size", 0)
            }
        
        # 修改文件
        for doc in modified_files:
            file_path = doc.get("path", doc.get("id", ""))
            self.metadata["indexed_files"][file_path] = {
                "hash": doc.get("file_hash", ""),
                "indexed_at": datetime.now().isoformat(),
                "size": doc.get("size", 0),
                "updated": True
            }
        
        # 删除文件
        for file_path in deleted_files:
            if file_path in self.metadata["indexed_files"]:
                del self.metadata["indexed_files"][file_path]
        
        self.metadata["last_incremental"] = datetime.now().isoformat()
        self.metadata["stats"]["total_documents"] = len(
            self.metadata["indexed_files"]
        )
        self._save_metadata()
        
        print(f"[Index] 增量更新完成")
    
    def detect_changes(self, current_files: Dict[str, str]) -> Dict:
        """
        检测文件变更
        
        Args:
            current_files: {file_path: file_hash}
            
        Returns:
            {
                "new": [文件列表],
                "modified": [文件列表],
                "deleted": [文件列表],
                "unchanged": [文件列表]
            }
        """
        indexed_files = self.metadata["indexed_files"]
        
        new_files = []
        modified_files = []
        deleted_files = []
        unchanged_files = []
        
        current_paths = set(current_files.keys())
        indexed_paths = set(indexed_files.keys())
        
        # 新增文件
        for path in current_paths - indexed_paths:
            new_files.append(path)
        
        # 删除文件
        for path in indexed_paths - current_paths:
            deleted_files.append(path)
        
        # 修改/未变文件
        for path in current_paths & indexed_paths:
            if current_files[path] != indexed_files[path].get("hash", ""):
                modified_files.append(path)
            else:
                unchanged_files.append(path)
        
        return {
            "new": new_files,
            "modified": modified_files,
            "deleted": deleted_files,
            "unchanged": unchanged_files
        }
    
    def get_index_stats(self) -> Dict:
        """获取索引统计信息"""
        return {
            **self.metadata["stats"],
            "last_full_index": self.metadata["last_full_index"],
            "last_incremental": self.metadata["last_incremental"],
            "indexed_file_count": len(self.metadata["indexed_files"])
        }
```

### 6.5.2 版本管理与一致性

```python
import shutil
from typing import Optional

class KnowledgeBaseVersioning:
    """知识库版本管理"""
    
    def __init__(self, base_path: str,
                 version_dir: str = "versions"):
        """
        Args:
            base_path: 知识库路径
            version_dir: 版本存储目录
        """
        self.base_path = base_path
        self.version_dir = os.path.join(base_path, version_dir)
        os.makedirs(self.version_dir, exist_ok=True)
    
    def create_snapshot(self, version_name: str = None) -> str:
        """
        创建知识库快照
        
        Args:
            version_name: 版本名称（默认使用时间戳）
            
        Returns:
            版本路径
        """
        if version_name is None:
            version_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        snapshot_path = os.path.join(self.version_dir, version_name)
        
        # 复制当前索引到快照目录
        shutil.copytree(
            self.base_path,
            snapshot_path,
            ignore=shutil.ignore_patterns("versions")
        )
        
        print(f"[Version] 创建快照: {version_name}")
        return snapshot_path
    
    def list_versions(self) -> List[Dict]:
        """列出所有版本"""
        versions = []
        for name in sorted(os.listdir(self.version_dir), reverse=True):
            version_path = os.path.join(self.version_dir, name)
            if os.path.isdir(version_path):
                mtime = os.path.getmtime(version_path)
                versions.append({
                    "name": name,
                    "created_at": datetime.fromtimestamp(mtime).isoformat(),
                    "path": version_path
                })
        return versions
    
    def rollback(self, version_name: str):
        """
        回滚到指定版本
        
        Args:
            version_name: 版本名称
        """
        snapshot_path = os.path.join(self.version_dir, version_name)
        if not os.path.exists(snapshot_path):
            raise ValueError(f"版本不存在: {version_name}")
        
        # 备份当前版本
        current_backup = self.create_snapshot(
            f"_rollback_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        print(f"[Version] 当前版本已备份: {current_backup}")
        
        # 清理当前索引
        for item in os.listdir(self.base_path):
            item_path = os.path.join(self.base_path, item)
            if item != "versions":
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)
        
        # 恢复快照
        for item in os.listdir(snapshot_path):
            src = os.path.join(snapshot_path, item)
            dst = os.path.join(self.base_path, item)
            if os.path.isdir(src):
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
        
        print(f"[Version] 回滚到版本: {version_name}")
    
    def cleanup_old_versions(self, keep_count: int = 10):
        """清理旧版本"""
        versions = self.list_versions()
        if len(versions) <= keep_count:
            return
        
        to_delete = versions[keep_count:]
        for version in to_delete:
            shutil.rmtree(version["path"])
            print(f"[Version] 删除旧版本: {version['name']}")
```

---

## 6.6 企业数据治理最佳实践

### 6.6.1 治理流程总览

```
                      ┌─────────────────────────────┐
                      │    数据源发现与注册          │
                      │  (文件系统/DB/API/Kafka)     │
                      └──────────┬──────────────────┘
                                 │
                      ┌──────────▼──────────────────┐
                      │    数据采集与同步             │
                      │  (全量/增量/实时/定时)       │
                      └──────────┬──────────────────┘
                                 │
                      ┌──────────▼──────────────────┐
                      │    数据清洗                  │
                      │  ┌─────┬──────┬──────┬───┐  │
                      │  │去重 │归一化│ PII  │噪声│  │
                      │  └─────┴──────┴──────┴───┘  │
                      └──────────┬──────────────────┘
                                 │
                      ┌──────────▼──────────────────┐
                      │    文本切分                  │
                      │  (固定/语义/结构/递归)      │
                      └──────────┬──────────────────┘
                                 │
                      ┌──────────▼──────────────────┐
                      │    索引构建与存储            │
                      │  (全量/增量/版本管理)       │
                      └──────────┬──────────────────┘
                                 │
                      ┌──────────▼──────────────────┐
                      │    质量监控与治理            │
                      │  (新鲜度/一致性/覆盖率)     │
                      └─────────────────────────────┘
```

### 6.6.2 常见问题与解决方案

| 问题 | 现象 | 解决方案 |
|------|------|---------|
| 数据源过多 | 数百个目录/表需要管理 | 使用数据目录工具（Apache Atlas），统一元数据管理 |
| 重复文档严重 | 检索结果 30%+ 是重复的 | MinHash + LSH 去重，定期全量去重扫描 |
| PII 泄露风险 | 知识库中包含用户隐私 | PII 检测 + 脱敏流水线，定期合规审计 |
| 数据不一致 | 不同来源同一概念描述不同 | 建立统一本体（Ontology），entity resolution |
| 索引漂移 | 源文档更新但索引未更新 | CDC 监控 + 增量索引，设置 TTL 刷新策略 |
| 切分质量差 | 语义被切碎 | 结构感知切分 + 语义边界检测 |
| 存储膨胀 | 版本管理和快照占用大量空间 | 差异存储（只存变更），定期清理旧版本 |

---

## 本章小结

企业知识数据治理是 RAG 系统落地的基石。本章从数据采集（文件扫描、数据库 CDC、API 集成、网络爬虫、Kafka）、数据清洗（去重、归一化、PII 检测、HTML 清洗）、文本切分（固定大小、语义、结构感知、递归）到存储与更新（全量/增量索引、版本管理），覆盖了知识库构建的完整生命周期。

核心要点：
- 数据质量是第一优先级：花 70% 的时间在数据清洗上，花 30% 的时间在检索策略上
- 去重是性价比最高的优化：一次 MinHash 去重可以提升 10-20% 的检索质量
- 切分策略因场景而异：问答场景用小块（300 tokens），摘要场景用大块（1000 tokens）
- 增量更新是生产系统的必备能力：全量索引在文档量超过 10 万后就不再实用
- 版本管理是安全的最后一道防线：回滚能力让你敢于尝试新的切分策略和索引参数
