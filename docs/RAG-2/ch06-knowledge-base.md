# 第6章 知识库构建与管理

知识库（Knowledge Base）是 RAG 系统的核心资产——它直接决定了检索阶段能够获取到什么信息，进而影响最终生成答案的质量。如果说 Embedding 模型和检索算法决定了"怎么找"，那么知识库就决定了"能找到什么"。一个高质量的知识库需要解决数据从哪里来、如何清洗、如何更新、如何保证安全等一系列问题。

本章将系统性地介绍知识库的完整生命周期：从多源数据集成、文档质量评估、版本管理与更新机制，到运维监控和访问控制。每部分都会给出可落地的 Python 代码示例和生产环境的最佳实践。

## 6.1 数据源集成

企业级 RAG 系统通常需要从多种异构数据源中摄取知识。数据源的类型决定了接入方式、数据格式和同步策略。

### 6.1.1 数据源分类

从数据特征角度，常见的数据源可以分为以下几类：

| 类型 | 典型系统 | 数据特征 | 接入难度 |
|------|---------|---------|---------|
| 文件系统 | NAS, S3, MinIO, 共享目录 | 结构化/半结构化文档 | 低 |
| 数据库 | MySQL, PostgreSQL, MongoDB | 结构化表数据 | 中 |
| Wiki 系统 | Confluence, Notion, Wiki.js | 富文本页面 | 中 |
| 协同办公 | SharePoint, Google Docs, 飞书 | 在线文档 | 中 |
| 代码仓库 | GitHub, GitLab | Markdown, 代码注释 | 低 |
| 消息队列 | Kafka, RabbitMQ | 流式事件数据 | 高 |
| 外部站点 | 公司官网, 第三方文档 | HTML 页面 | 中 |

下面我们对最核心的几种数据源分别展开讨论。

### 6.1.2 文件系统集成

文件系统是最基础也是最通用的数据源。企业内部的共享目录、NAS 存储、或对象存储（S3/MinIO）中通常积累了大量的 Word、PDF、Markdown 等格式的文档。

**设计要点：**

- **增量扫描**：首次全量扫描后，后续只扫描变更的文件（基于 mtime 或文件 hash）
- **格式自适应**：根据文件扩展名自动选择对应的解析器
- **软删除处理**：源文件被删除后，知识库中对应的文档应当被标记为过期而非立即删除

以下是一个文件系统扫描器的实现示例：

```python
"""
文件系统数据源扫描器
支持增量扫描、格式自适应、事件回调
"""

import os
import hashlib
import mimetypes
from pathlib import Path
from datetime import datetime
from typing import Callable, Optional
from dataclasses import dataclass, field


@dataclass
class FileEvent:
    """文件变更事件"""
    path: str
    event_type: str  # "created", "modified", "deleted"
    file_hash: str
    size: int
    mtime: float
    mime_type: str


class FileSystemScanner:
    """
    文件系统扫描器
    
    Usage:
        scanner = FileSystemScanner(
            base_path="/data/documents",
            file_extensions={".md", ".pdf", ".docx", ".txt"},
            exclude_dirs={"node_modules", ".git", "__pycache__"},
            on_file_created=handle_created,
            on_file_modified=handle_modified,
            on_file_deleted=handle_deleted,
        )
        scanner.run_full_scan()
        # 增量扫描
        scanner.run_incremental_scan()
    """
    
    def __init__(
        self,
        base_path: str,
        file_extensions: set = None,
        exclude_dirs: set = None,
        exclude_patterns: list = None,
        follow_symlinks: bool = False,
        on_file_created: Optional[Callable] = None,
        on_file_modified: Optional[Callable] = None,
        on_file_deleted: Optional[Callable] = None,
    ):
        self.base_path = Path(base_path)
        if not self.base_path.exists():
            raise FileNotFoundError(f"路径不存在: {base_path}")
        
        self.file_extensions = file_extensions or {".md", ".txt", ".pdf", ".docx"}
        self.exclude_dirs = exclude_dirs or {"node_modules", ".git", "__pycache__"}
        self.exclude_patterns = exclude_patterns or []
        self.follow_symlinks = follow_symlinks
        self.on_file_created = on_file_created
        self.on_file_modified = on_file_modified
        self.on_file_deleted = on_file_deleted
        
        # 内部状态：记录上次扫描的文件快照
        # key: 相对路径, value: (file_hash, mtime)
        self._file_snapshot: dict[str, tuple[str, float]] = {}
        self._manifest_path = self.base_path / ".scanner_manifest.json"
    
    def _compute_file_hash(self, file_path: Path) -> str:
        """计算文件的 SHA256 哈希，用于检测内容变更"""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            # 对大文件使用分块读取
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def _should_include(self, file_path: Path) -> bool:
        """判断文件是否应该被纳入扫描"""
        # 检查扩展名
        if file_path.suffix.lower() not in self.file_extensions:
            return False
        
        # 检查父目录是否在排除列表中
        for part in file_path.relative_to(self.base_path).parts:
            if part in self.exclude_dirs:
                return False
        
        # 检查排除模式
        rel_path = str(file_path.relative_to(self.base_path))
        for pattern in self.exclude_patterns:
            if pattern in rel_path:
                return False
        
        return True
    
    def _scan_files(self) -> dict[str, tuple[str, float, int, str]]:
        """
        扫描目录，返回 {相对路径: (hash, mtime, size, mime_type)}
        """
        result = {}
        for root, dirs, files in os.walk(self.base_path, followlinks=self.follow_symlinks):
            # 原地修剪排除目录（修改 dirs 会影响 os.walk 的遍历行为）
            dirs[:] = [d for d in dirs if d not in self.exclude_dirs]
            
            root_path = Path(root)
            for file_name in files:
                file_path = root_path / file_name
                
                if not self._should_include(file_path):
                    continue
                
                rel_path = str(file_path.relative_to(self.base_path))
                stat = file_path.stat()
                mtime = stat.st_mtime
                size = stat.st_size
                file_hash = self._compute_file_hash(file_path)
                mime_type, _ = mimetypes.guess_type(str(file_path))
                mime_type = mime_type or "application/octet-stream"
                
                result[rel_path] = (file_hash, mtime, size, mime_type)
        
        return result
    
    def run_full_scan(self):
        """执行全量扫描，触发所有文件的 created 事件"""
        print(f"[Scanner] 开始全量扫描: {self.base_path}")
        current_files = self._scan_files()
        
        for rel_path, (file_hash, mtime, size, mime_type) in current_files.items():
            event = FileEvent(
                path=rel_path,
                event_type="created",
                file_hash=file_hash,
                size=size,
                mtime=mtime,
                mime_type=mime_type,
            )
            if self.on_file_created:
                self.on_file_created(event)
        
        self._file_snapshot = {
            k: (v[0], v[1]) for k, v in current_files.items()
        }
        self._save_manifest()
        print(f"[Scanner] 全量扫描完成，发现 {len(current_files)} 个文件")
    
    def run_incremental_scan(self):
        """
        执行增量扫描，对比文件快照检测变更
        只触发发生了变化的文件事件
        """
        print(f"[Scanner] 开始增量扫描: {self.base_path}")
        self._load_manifest()
        
        current_files = self._scan_files()
        current_keys = set(current_files.keys())
        previous_keys = set(self._file_snapshot.keys())
        
        # 检测新增文件
        new_keys = current_keys - previous_keys
        for rel_path in new_keys:
            file_hash, mtime, size, mime_type = current_files[rel_path]
            event = FileEvent(
                path=rel_path,
                event_type="created",
                file_hash=file_hash,
                size=size,
                mtime=mtime,
                mime_type=mime_type,
            )
            if self.on_file_created:
                self.on_file_created(event)
        
        # 检测修改文件（hash 变化）
        common_keys = current_keys & previous_keys
        for rel_path in common_keys:
            prev_hash, prev_mtime = self._file_snapshot[rel_path]
            file_hash, mtime, size, mime_type = current_files[rel_path]
            
            if file_hash != prev_hash:
                event = FileEvent(
                    path=rel_path,
                    event_type="modified",
                    file_hash=file_hash,
                    size=size,
                    mtime=mtime,
                    mime_type=mime_type,
                )
                if self.on_file_modified:
                    self.on_file_modified(event)
        
        # 检测删除文件
        deleted_keys = previous_keys - current_keys
        for rel_path in deleted_keys:
            event = FileEvent(
                path=rel_path,
                event_type="deleted",
                file_hash="",
                size=0,
                mtime=0,
                mime_type="",
            )
            if self.on_file_deleted:
                self.on_file_deleted(event)
        
        # 更新快照
        self._file_snapshot = {
            k: (v[0], v[1]) for k, v in current_files.items()
        }
        self._save_manifest()
        
        print(
            f"[Scanner] 增量扫描完成: "
            f"{len(new_keys)} 新增, {len(common_keys)} 未变, "
            f"{len(deleted_keys)} 删除"
        )
    
    def _save_manifest(self):
        """保存文件快照到磁盘"""
        import json
        manifest = {
            "version": 1,
            "scanned_at": datetime.now().isoformat(),
            "files": {
                path: {"hash": h, "mtime": m}
                for path, (h, m) in self._file_snapshot.items()
            },
        }
        with open(self._manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
    
    def _load_manifest(self):
        """从磁盘加载文件快照"""
        import json
        if not self._manifest_path.exists():
            self._file_snapshot = {}
            return
        
        try:
            with open(self._manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            self._file_snapshot = {
                path: (data["hash"], data["mtime"])
                for path, data in manifest.get("files", {}).items()
            }
        except (json.JSONDecodeError, KeyError) as e:
            print(f"[Scanner] 警告: manifest 解析失败，将重新全量扫描: {e}")
            self._file_snapshot = {}
```

**关键设计思路：**

- 使用 `_file_snapshot` 记录上次扫描状态，支持增量扫描
- 通过 SHA256 哈希而非简单的 mtime 来判断文件是否修改——mtime 可能在文件复制/同步时被改变但内容未变
- 扫描清单（manifest）持久化到 JSON 文件，重启后仍能进行增量扫描
- 使用 os.walk 的 dirs 原地裁剪技巧避免遍历排除目录

### 6.1.3 数据库 CDC 集成

对于存储在关系型数据库或 MongoDB 中的业务数据，传统的定时全量导出方式效率低下。CDC（Change Data Capture，变更数据捕获）技术可以实时捕获数据库的变更事件，是实现数据近实时同步的首选方案。

CDC 的主流实现方式有：

| 方式 | 原理 | 优点 | 缺点 |
|------|------|------|------|
| 基于触发器 | 在源表上创建触发器记录变更 | 实现简单 | 对源库有性能影响 |
| 基于日志解析 | 解析数据库事务日志（如 MySQL binlog） | 对源库无侵入 | 实现复杂 |
| 基于时间戳轮询 | 查询带有时间戳列的表 | 无需额外组件 | 有延迟，无法捕获删除 |
| 基于版本号 | 每行记录 version 字段 | 简单可靠 | 需要改造表结构 |

在企业实践中，**基于日志解析的 CDC** 是最推荐的方式，它不对源库产生额外的查询压力，也不会漏掉任何变更。Debezium 是目前最成熟的开源 CDC 平台，它支持 MySQL、PostgreSQL、MongoDB 等多种数据库，并将变更事件输出到 Kafka。

下面是一个模拟 Debezium 变更事件的消费和处理示例：

```python
"""
基于 Debezium + Kafka 的数据库 CDC 消费者
将数据库表的变更事件同步到知识库
"""

import json
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class CDCEvent:
    """CDC 变更事件的数据结构"""
    op: str           # "c"=create, "u"=update, "d"=delete, "r"=read(snapshot)
    table: str        # 表名
    db: str           # 数据库名
    before: Optional[dict]  # 变更前的数据 (delete/update 时有值)
    after: Optional[dict]   # 变更后的数据 (create/update 时有值)
    ts_ms: int        # 事件时间戳 (毫秒)
    source_offset: str       # binlog 位置信息，用于断点续传


class CDCConsumer:
    """
    CDC 事件消费者
    
    从 Kafka 读取 Debezium 格式的变更事件，
    将其转换为知识库的文档变更操作。
    
    Usage:
        consumer = CDCConsumer(
            kafka_bootstrap_servers="localhost:9092",
            kafka_topic="dbserver1.inventory.products",
            group_id="knowledge-base-cdc",
        )
        consumer.start()
    """
    
    def __init__(
        self,
        kafka_bootstrap_servers: str,
        kafka_topic: str,
        group_id: str,
        table_to_doc_mapping: dict = None,
    ):
        self.kafka_bootstrap_servers = kafka_bootstrap_servers
        self.kafka_topic = kafka_topic
        self.group_id = group_id
        self.table_to_doc_mapping = table_to_doc_mapping or {}
        self._consumer = None
        self._offset_file = "cdc_offset.json"
    
    def _parse_debezium_event(self, raw_message: str) -> Optional[CDCEvent]:
        """
        解析 Debezium 格式的变更消息
        
        Debezium 的消息格式：
        {
            "payload": {
                "op": "u",
                "before": {...},
                "after": {...},
                "source": {
                    "db": "inventory",
                    "table": "products",
                    "ts_ms": 1700000000000,
                    "file": "mysql-bin.000003",
                    "pos": 545
                },
                "ts_ms": 1700000001000
            }
        }
        """
        try:
            message = json.loads(raw_message)
            payload = message.get("payload", {})
            
            op = payload.get("op", "")
            if op not in ("c", "u", "d", "r"):
                return None
            
            source = payload.get("source", {})
            offset = f"{source.get('file', '')}:{source.get('pos', 0)}"
            
            return CDCEvent(
                op=op,
                table=source.get("table", ""),
                db=source.get("db", ""),
                before=payload.get("before"),
                after=payload.get("after"),
                ts_ms=payload.get("ts_ms", 0),
                source_offset=offset,
            )
        except (json.JSONDecodeError, KeyError) as e:
            print(f"[CDC] 解析消息失败: {e}")
            return None
    
    def _process_event(self, event: CDCEvent):
        """处理单个 CDC 事件"""
        if event.table not in self.table_to_doc_mapping:
            print(f"[CDC] 跳过未映射的表: {event.table}")
            return
        
        mapper = self.table_to_doc_mapping[event.table]
        
        if event.op in ("c", "r"):
            # 新增记录 → 创建文档
            if event.after:
                doc = mapper.row_to_document(event.after)
                self._upsert_document(doc)
                print(
                    f"[CDC] CREATE: {event.table}.{event.after.get('id', '?')} "
                    f"-> offset={event.source_offset}"
                )
        
        elif event.op == "u":
            # 更新记录 → 更新文档
            if event.after:
                doc = mapper.row_to_document(event.after)
                self._upsert_document(doc)
                print(
                    f"[CDC] UPDATE: {event.table}.{event.after.get('id', '?')} "
                    f"-> offset={event.source_offset}"
                )
        
        elif event.op == "d":
            # 删除记录 → 删除文档
            if event.before:
                doc_id = event.before.get("id")
                self._delete_document(doc_id)
                print(
                    f"[CDC] DELETE: {event.table}.{doc_id} "
                    f"-> offset={event.source_offset}"
                )
    
    def _upsert_document(self, doc: dict):
        """
        将数据库记录转换为知识库文档并写入
        实际实现中会调用 Embedding 和向量存储接口
        """
        # 这里仅做示意，实际需要集成 chunking + embedding + index
        print(f"[CDC] Upsert 文档: {doc.get('id')} - {doc.get('title', '')[:50]}")
        # 1. 对文档内容进行分块
        # 2. 对每个块生成 embedding
        # 3. 写入向量数据库（先删除旧向量再插入新向量）
    
    def _delete_document(self, doc_id: str):
        """从知识库中删除文档"""
        print(f"[CDC] 删除文档: {doc_id}")
        # 1. 从向量数据库中删除该文档对应的所有向量
        # 2. 从文档元数据存储中删除记录
    
    def start(self):
        """
        启动 CDC 消费者
        持续监听 Kafka 主题，处理变更事件
        """
        print(
            f"[CDC] 启动消费者: topic={self.kafka_topic}, "
            f"group={self.group_id}"
        )
        # 实际实现中使用 kafka-python 或 confluent-kafka 库
        # self._consumer = KafkaConsumer(
        #     self.kafka_topic,
        #     bootstrap_servers=self.kafka_bootstrap_servers,
        #     group_id=self.group_id,
        #     auto_offset_reset="earliest",
        #     enable_auto_commit=False,  # 手动提交 offset 实现精确一次语义
        # )
        # 
        # for raw_msg in self._consumer:
        #     event = self._parse_debezium_event(raw_msg.value.decode())
        #     if event:
        #         self._process_event(event)
        #         # 保存 offset 用于断点续传
        #         self._save_offset(event.source_offset)
        #         self._consumer.commit()
    
    def _save_offset(self, offset: str):
        """持久化消费进度，用于断点续传"""
        import json
        data = {"topic": self.kafka_topic, "offset": offset, "saved_at": datetime.now().isoformat()}
        with open(self._offset_file, "w") as f:
            json.dump(data, f)


# 使用示例
class ProductDocumentMapper:
    """将 products 表记录映射为知识库文档"""
    
    @staticmethod
    def row_to_document(row: dict) -> dict:
        return {
            "id": f"product:{row['id']}",
            "title": row["name"],
            "content": f"{row['name']}: {row['description']}",
            "metadata": {
                "source": "mysql:products",
                "category": row.get("category"),
                "price": row.get("price"),
                "updated_at": row.get("updated_at"),
            },
        }
```

**CDC 集成的关键注意事项：**

1. **断点续传**：记录 binlog 位置或 Kafka offset，崩溃后能从断点恢复而不丢失数据
2. **幂等性**：同一条记录的多次变更重复消费不应导致数据不一致
3. **Schema 变更**：源表结构变更（如新增列）时，CDC 消费者需要优雅处理
4. **初始快照**：首次启动 CDC 时需要先做一次全量快照（Debezium 的 snapshot 模式），然后再增量同步
5. **删除处理**：数据库中的物理删除在知识库中应做软删除（标记过期），以便检索时能感知到文档已被移除

### 6.1.4 API 集成（Confluence / Notion / SharePoint）

企业内部 Wiki 和协同办公平台是现代 RAG 系统的重要知识来源。这类平台通常提供 REST API，我们可以通过定期轮询或 Webhook 来获取文档变更。

以 Confluence 为例，其 API 集成需要处理的关键问题包括：

1. **分页遍历**：API 返回的数据通常是分页的，需要处理分页逻辑
2. **增量同步**：通过 `version` 字段或 `lastModified` 时间戳判断是否需要更新
3. **富文本转换**：Confluence 的存储格式（Storage Format）或 CFM（Confluence Flavored Markup）需要转换为纯文本或 Markdown
4. **附件处理**：页面中嵌入的图片、表格等附件需要单独下载和处理
5. **空间/页面树结构**：需要遍历整个页面树，确定哪些空间需要纳入知识库

以下是一个 Confluence 集成示例：

```python
"""
Confluence 数据源集成
支持空间遍历、页面增量同步、富文本解析
"""

import hashlib
import json
import time
from datetime import datetime
from typing import Optional
from dataclasses import dataclass


@dataclass
class ConfluencePage:
    id: str
    title: str
    version: int
    body_text: str
    space_key: str
    parent_id: Optional[str]
    last_modified: str
    author: str
    labels: list[str]


class ConfluenceDataSource:
    """
    Confluence 知识库数据源
    
    通过 Confluence REST API 获取空间和页面内容，
    支持增量同步（基于版本号）。
    
    API 文档: https://developer.atlassian.com/cloud/confluence/rest/
    """
    
    def __init__(
        self,
        base_url: str,
        username: str,
        api_token: str,
        space_keys: list[str] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.auth = (username, api_token)
        self.space_keys = space_keys or []
        self._session = None
    
    def _request(self, path: str, params: dict = None) -> dict:
        """发送 HTTP 请求到 Confluence API"""
        import requests
        url = f"{self.base_url}/wiki/api/v2/{path.lstrip('/')}"
        headers = {"Accept": "application/json"}
        
        response = requests.get(
            url, auth=self.auth, headers=headers, params=params
        )
        response.raise_for_status()
        return response.json()
    
    def _parse_storage_to_text(self, storage: str) -> str:
        """
        将 Confluence Storage Format (XHTML) 解析为纯文本
        
        简单实现：剥离所有 HTML 标签。
        生产环境中建议使用 html2text 或 BeautifulSoup 做更精细的转换，
        保留标题层级和列表结构以利于后续分块。
        """
        import re
        # 移除 HTML 标签
        text = re.sub(r"<[^>]+>", "", storage)
        # 解码 HTML 实体
        import html
        text = html.unescape(text)
        # 合并多余空白行
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
    
    def get_space_pages(self, space_key: str, since_version: int = 0) -> list[ConfluencePage]:
        """
        获取指定空间的所有页面
        
        Args:
            space_key: Confluence 空间标识
            since_version: 只返回版本号大于此值的页面（增量同步）
        
        处理分页逻辑，遍历所有页面。
        """
        pages = []
        cursor = None
        
        while True:
            params = {
                "space-key": space_key,
                "limit": 100,
                "body-format": "storage",
                "status": "current",
            }
            if cursor:
                params["cursor"] = cursor
            
            data = self._request("/pages", params)
            
            for result in data.get("results", []):
                version = result["version"]["number"]
                if version <= since_version:
                    continue
                
                body = result.get("body", {}).get("storage", {}).get("value", "")
                body_text = self._parse_storage_to_text(body)
                
                labels = [
                    label["name"]
                    for label in result.get("labels", {}).get("results", [])
                ]
                
                page = ConfluencePage(
                    id=result["id"],
                    title=result["title"],
                    version=version,
                    body_text=body_text,
                    space_key=space_key,
                    parent_id=result.get("parentId"),
                    last_modified=result["version"]["when"],
                    author=result["version"]["by"].get("displayName", ""),
                    labels=labels,
                )
                pages.append(page)
            
            # 处理分页
            links = data.get("_links", {})
            if "next" in links:
                # 从 next 链接中提取 cursor
                # /wiki/api/v2/pages?cursor=xxx
                cursor = links["next"].split("cursor=")[-1]
            else:
                break
        
        return pages
    
    def sync_all_spaces(self, version_tracker: dict[str, int] = None):
        """
        同步所有配置的空间
        
        Args:
            version_tracker: {page_id: last_version} 字典，记录每个页面最后同步的版本
                             None 表示全量同步
        
        Returns:
            需要更新/新增的页面列表
        """
        version_tracker = version_tracker or {}
        updated_pages = []
        
        for space_key in self.space_keys:
            print(f"[Confluence] 同步空间: {space_key}")
            since_version = max(version_tracker.values()) if version_tracker else 0
            
            # 按页面粒度追踪版本
            pages = self.get_space_pages(space_key, since_version=0)
            
            for page in pages:
                last_version = version_tracker.get(page.id, 0)
                if page.version > last_version:
                    updated_pages.append(page)
                    version_tracker[page.id] = page.version
            
            print(
                f"[Confluence] 空间 {space_key}: "
                f"共 {len(pages)} 页, 其中 {len(updated_pages)} 页需要更新"
            )
            
            # 避免 API 限流
            time.sleep(0.5)
        
        return updated_pages


# Notion 集成示例（使用 Notion API）
class NotionDataSource:
    """
    Notion 数据源集成
    
    通过 Notion API 读取数据库和页面内容。
    Notion 的数据模型是 Block 树结构，需要递归遍历所有 Block。
    """
    
    def __init__(self, api_key: str, database_ids: list[str] = None):
        self.api_key = api_key
        self.database_ids = database_ids or []
        self.base_url = "https://api.notion.com/v1"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Notion-Version": "2022-06-28",
            "Content-Type": "application/json",
        }
    
    def _request(self, path: str, method: str = "GET", **kwargs) -> dict:
        import requests
        url = f"{self.base_url}/{path.lstrip('/')}"
        response = requests.request(method, url, headers=self.headers, **kwargs)
        response.raise_for_status()
        return response.json()
    
    def _extract_rich_text(self, rich_text_list: list) -> str:
        """提取 Notion rich_text 数组中的纯文本"""
        return "".join(
            item.get("plain_text", "")
            for item in rich_text_list
        )
    
    def _parse_block(self, block: dict) -> str:
        """解析 Notion Block 为文本"""
        block_type = block.get("type", "unsupported")
        block_data = block.get(block_type, {})
        
        if block_type == "paragraph":
            return self._extract_rich_text(block_data.get("rich_text", [])) + "\n"
        
        elif block_type in ("heading_1", "heading_2", "heading_3"):
            prefix = "#" * int(block_type.split("_")[1])
            text = self._extract_rich_text(block_data.get("rich_text", []))
            return f"{prefix} {text}\n"
        
        elif block_type == "bulleted_list_item":
            text = self._extract_rich_text(block_data.get("rich_text", []))
            return f"- {text}\n"
        
        elif block_type == "numbered_list_item":
            text = self._extract_rich_text(block_data.get("rich_text", []))
            return f"1. {text}\n"
        
        elif block_type == "code":
            text = self._extract_rich_text(block_data.get("rich_text", []))
            lang = block_data.get("language", "")
            return f"```{lang}\n{text}\n```\n"
        
        elif block_type == "table":
            # 表格解析较复杂，此处省略
            return "[table]\n"
        
        elif block_type == "child_page":
            # 子页面需要单独处理
            return ""
        
        else:
            return ""
    
    def _get_block_children(self, block_id: str) -> list[dict]:
        """递归获取 Block 的所有子 Block"""
        blocks = []
        cursor = None
        
        while True:
            params = {"page_size": 100}
            if cursor:
                params["start_cursor"] = cursor
            
            data = self._request(f"blocks/{block_id}/children", params=params)
            blocks.extend(data.get("results", []))
            
            if not data.get("has_more"):
                break
            cursor = data.get("next_cursor")
        
        return blocks
    
    def page_to_text(self, page_id: str) -> str:
        """将 Notion 页面递归展开为纯文本"""
        texts = []
        blocks = self._get_block_children(page_id)
        
        for block in blocks:
            text = self._parse_block(block)
            texts.append(text)
            
            # 某些 Block 类型有子 Block（如 toggle、quote、child_page）
            if block.get("has_children"):
                child_blocks = self._get_block_children(block["id"])
                for child in child_blocks:
                    texts.append(self._parse_block(child))
        
        return "".join(texts)
    
    def sync_database(self, database_id: str) -> list[dict]:
        """
        同步一个 Notion Database 中的所有页面
        
        Notion Database 中的每一行是一个 Page，
        需要读取每个 Page 的内容。
        """
        pages = []
        cursor = None
        
        while True:
            params = {"page_size": 100}
            if cursor:
                params["start_cursor"] = cursor
            
            data = self._request(f"databases/{database_id}/query", method="POST", json=params)
            
            for result in data.get("results", []):
                properties = result.get("properties", {})
                page_id = result["id"]
                
                # 提取页面标题
                title_prop = None
                for prop_name, prop_value in properties.items():
                    if prop_value.get("type") == "title":
                        title_prop = prop_value
                        break
                
                title = ""
                if title_prop:
                    title = self._extract_rich_text(title_prop.get("title", []))
                
                # 获取页面正文内容
                content = self.page_to_text(page_id)
                
                # 提取其他属性（如标签、日期等）
                metadata = {}
                for prop_name, prop_value in properties.items():
                    prop_type = prop_value.get("type")
                    if prop_type == "select":
                        select = prop_value.get("select")
                        if select:
                            metadata[prop_name] = select["name"]
                    elif prop_type == "multi_select":
                        options = prop_value.get("multi_select", [])
                        metadata[prop_name] = [opt["name"] for opt in options]
                    elif prop_type == "date":
                        date = prop_value.get("date")
                        if date:
                            metadata[prop_name] = date.get("start")
                
                pages.append({
                    "id": page_id,
                    "title": title,
                    "content": content,
                    "metadata": metadata,
                    "last_edited": result.get("last_edited_time"),
                })
            
            if not data.get("has_more"):
                break
            cursor = data.get("next_cursor")
        
        return pages
```

**API 集成的通用最佳实践：**

- **限流处理**：企业 API 通常有速率限制（Rate Limit），需要在代码中实现退避重试
- **Webhook 支持**：如果平台支持 Webhook（如 Confluence 的 Webhook 或 Notion 的 API 变更通知），优先使用 Webhook 而非轮询
- **增量标识**：利用平台的版本号或更新时间戳字段做增量同步，避免每次都全量拉取
- **内容降级**：某些文档可能包含复杂格式（如流程图、嵌入式表格），需要定义降级策略——至少保留文本信息和结构

### 6.1.5 网络爬虫集成（Scrapy）

对于外部网站（如公司官网、产品文档站点、第三方知识库），需要通过网络爬虫来采集数据。Scrapy 是 Python 生态中最成熟的开源爬虫框架。

```python
"""
基于 Scrapy 的文档站点爬虫
爬取技术文档网站并结构化存储
"""

import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
from bs4 import BeautifulSoup
import re


class DocSiteSpider(CrawlSpider):
    """
    文档站点爬虫
    
    从指定的起始 URL 开始，遵循同域名下的链接进行爬取，
    将每个页面的标题、正文、元数据提取并结构化输出。
    """
    
    name = "doc_site_spider"
    
    # 爬取规则
    rules = (
        Rule(
            LinkExtractor(
                allow_domains=[],  # 在 __init__ 中设置
                deny_extensions=[
                    "pdf", "zip", "png", "jpg", "jpeg",
                    "gif", "mp4", "avi", "exe", "dmg",
                ],
                deny=r"/(tag|category|author)/",  # 跳过标签/分类页
            ),
            callback="parse_page",
            follow=True,
        ),
    )
    
    def __init__(self, start_urls=None, allowed_domains=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_urls = start_urls or []
        if allowed_domains:
            self.rules[0].link_extractor.allow_domains = allowed_domains
    
    def parse_page(self, response):
        """
        解析页面内容
        
        提取标题、正文、最后修改时间、Meta 描述等信息，
        输出结构化数据供后续处理。
        """
        soup = BeautifulSoup(response.text, "html.parser")
        
        # 移除导航、页脚、广告等非内容元素
        for selector in [
            "nav", "footer", ".sidebar", ".toc",
            "#nav", "#footer", ".advertisement",
            "script", "style", "noscript",
        ]:
            for element in soup.select(selector):
                element.decompose()
        
        # 提取标题
        title = ""
        h1 = soup.find("h1")
        if h1:
            title = h1.get_text(strip=True)
        
        # 尝试从 <title> 标签获取
        if not title and soup.title:
            title = soup.title.get_text(strip=True)
        
        # 提取正文
        main_content = soup.find("main") or soup.find("article") or soup.find("body")
        text = ""
        if main_content:
            text = main_content.get_text(separator="\n", strip=True)
        
        # 清理多余空白
        text = re.sub(r"\n{3,}", "\n\n", text)
        
        # 提取最后修改时间
        last_modified = response.headers.get("Last-Modified", b"").decode()
        
        # 提取 Meta description
        meta_desc = ""
        meta_tag = soup.find("meta", attrs={"name": "description"})
        if meta_tag:
            meta_desc = meta_tag.get("content", "")
        
        yield {
            "url": response.url,
            "title": title,
            "content": text,
            "meta_description": meta_desc,
            "last_modified": last_modified,
            "content_length": len(text),
            "crawled_at": scrapy.utils.response.response_time(response).isoformat(),
        }


# 运行爬虫
def run_spider(start_urls: list[str], output_file: str = "crawled_docs.json"):
    """
    运行文档爬虫并保存结果
    
    Args:
        start_urls: 爬取起始 URL 列表
        output_file: 输出 JSON 文件路径
    """
    from scrapy.crawler import CrawlerProcess
    from scrapy.utils.project import get_project_settings
    
    process = CrawlerProcess(settings={
        "USER_AGENT": "KnowledgeBaseCrawler/1.0",
        "ROBOTSTXT_OBEY": True,
        "CONCURRENT_REQUESTS": 16,
        "DOWNLOAD_DELAY": 0.5,
        "COOKIES_ENABLED": False,
        "FEEDS": {
            output_file: {
                "format": "jsonlines",
                "encoding": "utf-8",
                "store_empty": False,
            },
        },
        # 反爬策略：使用随机 User-Agent 和代理
        "DOWNLOADER_MIDDLEWARES": {
            "scrapy.downloadermiddlewares.useragent.UserAgentMiddleware": None,
            # 生产环境建议使用 scrapy-user-agents 和 scrapy-rotating-proxies
        },
    })
    
    spider = DocSiteSpider(
        start_urls=start_urls,
        allowed_domains=[urlparse(url).netloc for url in start_urls],
    )
    
    process.crawl(spider)
    process.start()  # 阻塞直到爬取完成


# 使用示例
if __name__ == "__main__":
    from urllib.parse import urlparse
    
    docs_sites = [
        "https://docs.example.com/en/latest/",
        "https://help.example.com/",
    ]
    
    run_spider(docs_sites, output_file="company_docs.jsonl")
```

**爬虫集成的关键考量：**

1. **robots.txt 遵守**：除非有明确的业务授权，否则应遵守目标站点的 robots.txt
2. **爬取礼貌**：设置合理的 DOWNLOAD_DELAY，避免对目标站点造成压力
3. **内容去重**：同一个 URL 在不同时间爬取的内容可能有变化，需要基于内容 hash 去重
4. **动态页面**：对于 JavaScript 渲染的页面（SPA），需要使用 Splash 或 Selenium 中间件
5. **增量爬取**：通过 Last-Modified 或 ETag 实现增量爬取，减少重复传输

### 6.1.6 消息队列集成（Kafka）

在实时性要求高的场景下，知识库需要以流式方式消费数据变更事件。Kafka 是业界最广泛使用的消息队列平台，适合构建知识库的实时数据管道。

Kafka 在知识库场景中的典型应用模式：

```python
"""
基于 Kafka 的流式知识库数据管道

从多个 Kafka 主题消费数据变更事件，
经过清洗、转换后写入知识库。
"""

import json
import threading
from datetime import datetime
from typing import Callable, Optional
from dataclasses import dataclass, field


@dataclass
class KnowledgeDocument:
    """知识库文档的通用数据模型"""
    doc_id: str
    title: str
    content: str
    source: str       # 数据源标识，如 "kafka:orders"
    source_ts: int     # 源数据的时间戳
    metadata: dict = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class KnowledgePipeline:
    """
    Kafka 流式知识库管道
    
    从 Kafka 消费数据 → 解析 → 分块 → Embedding → 写入向量库
    
    支持多个输入主题和动态路由。
    """
    
    def __init__(
        self,
        bootstrap_servers: str,
        input_topics: list[str],
        group_id: str = "knowledge-pipeline",
        processors: dict[str, Callable] = None,
    ):
        """
        Args:
            bootstrap_servers: Kafka 集群地址
            input_topics: 消费的主题列表
            group_id: 消费者组 ID
            processors: {topic: processor_function} 映射
                        每个 processor 接收原始消息 bytes，返回 KnowledgeDocument
        """
        self.bootstrap_servers = bootstrap_servers
        self.input_topics = input_topics
        self.group_id = group_id
        self.processors = processors or {}
        self._running = False
        self._consumer_threads: list[threading.Thread] = []
    
    def _default_processor(self, raw_value: bytes) -> Optional[KnowledgeDocument]:
        """默认消息处理器：将 JSON 消息转换为文档"""
        try:
            data = json.loads(raw_value.decode("utf-8"))
            return KnowledgeDocument(
                doc_id=data.get("id", ""),
                title=data.get("title", ""),
                content=data.get("content", ""),
                source=data.get("source", "unknown"),
                source_ts=data.get("timestamp", 0),
                metadata=data.get("metadata", {}),
            )
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"[Pipeline] 消息解析失败: {e}")
            return None
    
    def _process_message(self, topic: str, raw_value: bytes):
        """处理单条 Kafka 消息"""
        processor = self.processors.get(topic, self._default_processor)
        doc = processor(raw_value)
        
        if doc is None:
            return
        
        # 文档质量检查（见 6.2 节）
        if not self._quality_check(doc):
            print(f"[Pipeline] 文档未通过质量检查，跳过: {doc.doc_id}")
            return
        
        # 分块（Chunking）
        chunks = self._chunk_document(doc)
        
        # 生成 Embedding 并写入向量库
        self._index_chunks(chunks, doc.source)
    
    def _quality_check(self, doc: KnowledgeDocument) -> bool:
        """基本质量检查"""
        if not doc.content or len(doc.content.strip()) < 10:
            return False
        if not doc.doc_id:
            return False
        return True
    
    def _chunk_document(self, doc: KnowledgeDocument) -> list[dict]:
        """
        文档分块
        
        简单的固定大小分块，生产环境建议使用语义分块
        （详见第 5 章文档分块策略）
        """
        chunk_size = 512
        overlap = 64
        chunks = []
        
        text = doc.content
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end]
            
            chunks.append({
                "doc_id": f"{doc.doc_id}#chunk-{len(chunks)}",
                "parent_id": doc.doc_id,
                "title": doc.title,
                "content": chunk_text,
                "source": doc.source,
                "metadata": doc.metadata,
            })
            
            if end >= len(text):
                break
            start = end - overlap
        
        return chunks
    
    def _index_chunks(self, chunks: list[dict], source: str):
        """
        为分块生成 Embedding 并写入向量数据库
        
        实际实现中会调用 Embedding API 和向量存储接口
        """
        print(f"[Pipeline] 索引 {len(chunks)} 个分块 (source={source})")
        # 1. 批量调用 Embedding API
        # 2. 写入向量数据库（先删除旧分块再插入）
    
    def start(self):
        """启动 Kafka 消费者"""
        print(
            f"[Pipeline] 启动知识库管道: topics={self.input_topics}, "
            f"group={self.group_id}"
        )
        self._running = True
        
        # 实际使用 kafka-python 或 confluent-kafka
        # consumer = KafkaConsumer(
        #     *self.input_topics,
        #     bootstrap_servers=self.bootstrap_servers,
        #     group_id=self.group_id,
        #     auto_offset_reset="latest",
        #     enable_auto_commit=False,
        #     max_poll_records=500,  # 批量消费，提高吞吐
        # )
        #
        # while self._running:
        #     raw_msgs = consumer.poll(timeout_ms=1000)
        #     for topic_partition, msgs in raw_msgs.items():
        #         for msg in msgs:
        #             self._process_message(msg.topic, msg.value)
        #     consumer.commit()
    
    def stop(self):
        """优雅停止"""
        self._running = False
        print("[Pipeline] 知识库管道已停止")


# 多主题路由示例
processors = {
    "source.orders": lambda v: KnowledgeDocument(
        doc_id=f"order:{data['order_id']}",
        title=f"订单 {data['order_id']}",
        content=f"客户: {data['customer']}\n商品: {data['product']}\n金额: {data['amount']}",
        source="kafka:orders",
        source_ts=data["created_at"],
        metadata={"status": data["status"], "amount": data["amount"]},
    ),
    "source.wiki": lambda v: KnowledgeDocument(
        doc_id=f"wiki:{data['page_id']}",
        title=data["title"],
        content=data["body"],
        source="kafka:wiki",
        source_ts=data["updated_at"],
        metadata={"author": data.get("author"), "space": data.get("space")},
    ),
}

pipeline = KnowledgePipeline(
    bootstrap_servers="kafka:9092",
    input_topics=["source.orders", "source.wiki"],
    processors=processors,
)
```

**Kafka 管道的设计要点：**

- **批量处理**：使用 `poll()` 批量拉取消息而非逐条处理，可以批量调用 Embedding API 提高吞吐
- **死信队列**：处理失败的消息应写入死信主题（DLQ），避免阻塞主流程
- **消息顺序**：如果需要保证同一文档的更新有序，可以使用分区键（Partition Key）确保同一 doc_id 的消息进入同一分区
- **幂等写入**：向量数据库的写入操作应支持 upsert，避免重复消费导致数据错乱

---

## 6.2 文档质量评估

不是所有文档都适合纳入知识库。低质量的文档不仅浪费存储和计算资源，还会在检索阶段引入噪声，降低 RAG 系统的整体表现。因此，在文档进入知识库之前进行质量评估是必不可少的环节。

### 6.2.1 质量评估维度

文档质量可以从以下多个维度进行评估：

| 维度 | 说明 | 评估方法 |
|------|------|---------|
| 内容完整性 | 文档是否有足够的信息量 | 字数、段落数、涵盖的标题层级 |
| 可读性 | 文本是否易于理解 | 句子长度、词汇复杂度、Flesch 可读性分数 |
| 信息密度 | 单位文本中包含的有用信息量 | 停用词比例、关键术语密度 |
| 时效性 | 文档信息是否过时 | 最后修改时间、引用的外部链接是否有效 |
| 权威性 | 信息来源是否可信 | 数据源等级、作者级别、引用次数 |
| 独特性 | 是否与已有文档高度重复 | 与其他文档的余弦相似度 |
| 规范性 | 格式是否符合要求 | 编码是否规范、是否有乱码 |

### 6.2.2 质量评估器实现

以下是一个综合文档质量评估器的实现：

```python
"""
文档质量评估器

从多个维度评估文档质量，输出质量分数和具体的质量问题。
支持自定义阈值和评估规则。
"""

import re
import math
from datetime import datetime, timezone
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class QualityReport:
    """文档质量评估报告"""
    doc_id: str
    overall_score: float  # 0.0 - 1.0
    dimensions: dict[str, float] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    passed: bool = True


class DocumentQualityAssessor:
    """
    文档质量评估器
    
    从内容完整性、可读性、信息密度、时效性等维度评估文档质量。
    
    Usage:
        assessor = DocumentQualityAssessor()
        report = assessor.evaluate(doc_id="doc_001", content=text, metadata={...})
        if report.passed:
            # 纳入知识库
            pass
        else:
            # 记录质量问题，标记人工审核
            pass
    """
    
    def __init__(
        self,
        min_content_length: int = 50,
        min_paragraphs: int = 2,
        max_title_ratio: float = 0.3,
        max_stopword_ratio: float = 0.7,
        max_content_age_days: int = 730,  # 2年
    ):
        self.min_content_length = min_content_length
        self.min_paragraphs = min_paragraphs
        self.max_title_ratio = max_title_ratio
        self.max_stopword_ratio = max_stopword_ratio
        self.max_content_age_days = max_content_age_days
        
        # 常用中文停用词
        self.stopwords = set(
            "的了在是在我有和就不都一个到说很会也要去"
            "你他没对为能又这那被看把但还而所以如果因为"
            "虽然然后而且或者但是不过可以应该可能已经正在"
            "通过根据关于按照除了从对于与及以及等"
        )
    
    def _evaluate_completeness(self, content: str) -> tuple[float, list[str]]:
        """
        评估内容完整性
        
        考虑因素：
        - 总字符数
        - 段落数
        - 句子数
        - 是否包含标题
        """
        issues = []
        
        # 字符数检查
        text_length = len(content.strip())
        if text_length < self.min_content_length:
            issues.append(f"内容过短: {text_length} 字符 (最低要求: {self.min_content_length})")
            return 0.0, issues
        
        # 段落数检查（以连续换行分隔）
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        para_count = len(paragraphs)
        if para_count < self.min_paragraphs:
            issues.append(f"段落数不足: {para_count} (最低要求: {self.min_paragraphs})")
        
        # 句子数检查（简单以句号、问号、感叹号分隔）
        sentences = re.split(r"[。！？\n]", content)
        sentences = [s.strip() for s in sentences if s.strip()]
        sentence_count = len(sentences)
        if sentence_count < 3:
            issues.append(f"句子数不足: {sentence_count}")
        
        # 综合分数
        score = min(1.0, (
            0.3 * min(1.0, text_length / 500) +
            0.3 * min(1.0, para_count / 5) +
            0.2 * min(1.0, sentence_count / 10) +
            0.2 * min(1.0, text_length / 2000)
        ))
        
        return score, issues
    
    def _evaluate_readability(self, content: str) -> tuple[float, list[str]]:
        """
        评估可读性
        
        基于中文文本特征：
        - 平均句子长度
        - 长句比例
        - 词汇重复度
        """
        issues = []
        
        sentences = re.split(r"[。！？\n]", content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return 0.0, ["无法计算可读性：无有效句子"]
        
        # 平均句子长度（字符数）
        avg_sentence_len = sum(len(s) for s in sentences) / len(sentences)
        
        # 长句比例（超过 100 字符的句子）
        long_sentence_ratio = sum(1 for s in sentences if len(s) > 100) / len(sentences)
        
        # 打分：句子长度适中为佳
        if avg_sentence_len < 10:
            length_score = 0.5  # 句子过短，可能过于碎片化
        elif avg_sentence_len <= 40:
            length_score = 1.0  # 理想范围
        elif avg_sentence_len <= 80:
            length_score = 0.7  # 偏长
        else:
            length_score = 0.3  # 句子过长
            issues.append(f"平均句子长度过长: {avg_sentence_len:.0f} 字符")
        
        # 长句比例惩罚
        long_penalty = max(0, long_sentence_ratio - 0.3) * 0.5
        score = max(0.0, length_score - long_penalty)
        
        return score, issues
    
    def _evaluate_information_density(self, content: str) -> tuple[float, list[str]]:
        """
        评估信息密度
        
        通过停用词比例来衡量：停用词比例越低，信息密度越高
        """
        issues = []
        
        if not content.strip():
            return 0.0, ["内容为空"]
        
        # 计算停用词占比
        total_chars = len(content.strip())
        stopword_count = sum(1 for c in content if c in self.stopwords)
        stopword_ratio = stopword_count / total_chars if total_chars > 0 else 1.0
        
        if stopword_ratio > self.max_stopword_ratio:
            issues.append(f"停用词比例过高: {stopword_ratio:.1%}")
            score = max(0, 1.0 - (stopword_ratio - self.max_stopword_ratio) * 2)
        else:
            score = 1.0
        
        return score, issues
    
    def _evaluate_timeliness(
        self, last_modified: Optional[str]
    ) -> tuple[float, list[str]]:
        """
        评估时效性
        
        根据文档的最后修改时间判断信息是否过时。
        """
        issues = []
        
        if not last_modified:
            # 无时间信息，保守给中等分数
            return 0.5, ["缺少最后修改时间，无法评估时效性"]
        
        try:
            if isinstance(last_modified, str):
                modified_dt = datetime.fromisoformat(last_modified)
            else:
                modified_dt = last_modified
            
            now = datetime.now(timezone.utc)
            age_days = (now - modified_dt).days if modified_dt.tzinfo else \
                       (datetime.now() - modified_dt).days
            
            if age_days < 0:
                return 1.0, []  # 未来的时间戳，视为最新
            
            if age_days > self.max_content_age_days:
                issues.append(f"内容已过期: {age_days} 天 (最大允许: {self.max_content_age_days} 天)")
                return 0.2, issues
            
            # 线性衰减：越新的文档分数越高
            score = 1.0 - (age_days / self.max_content_age_days) * 0.5
            return max(0.2, score), issues
            
        except (ValueError, TypeError):
            return 0.5, ["无法解析最后修改时间"]
    
    def evaluate(
        self,
        doc_id: str,
        content: str,
        metadata: dict = None,
    ) -> QualityReport:
        """
        对文档进行综合质量评估
        
        Args:
            doc_id: 文档标识
            content: 文档正文内容
            metadata: 文档元数据，可包含 last_modified, author, source 等
        
        Returns:
            QualityReport: 质量评估报告
        """
        metadata = metadata or {}
        all_issues = []
        
        # 各维度评估
        completeness_score, completeness_issues = self._evaluate_completeness(content)
        readability_score, readability_issues = self._evaluate_readability(content)
        density_score, density_issues = self._evaluate_information_density(content)
        timeliness_score, timeliness_issues = self._evaluate_timeliness(
            metadata.get("last_modified")
        )
        
        # 汇总所有问题
        all_issues.extend(completeness_issues)
        all_issues.extend(readability_issues)
        all_issues.extend(density_issues)
        all_issues.extend(timeliness_issues)
        
        # 权重配置（可根据业务需求调整）
        weights = {
            "completeness": 0.35,
            "readability": 0.25,
            "density": 0.20,
            "timeliness": 0.20,
        }
        
        # 加权综合分数
        overall = (
            weights["completeness"] * completeness_score +
            weights["readability"] * readability_score +
            weights["density"] * density_score +
            weights["timeliness"] * timeliness_score
        )
        
        # 判定是否通过
        passed = overall >= 0.5 and len(all_issues) <= 3
        
        return QualityReport(
            doc_id=doc_id,
            overall_score=round(overall, 4),
            dimensions={
                "completeness": round(completeness_score, 4),
                "readability": round(readability_score, 4),
                "density": round(density_score, 4),
                "timeliness": round(timeliness_score, 4),
            },
            issues=all_issues[:5],  # 最多报告 5 个问题
            passed=passed,
        )


# 使用示例
assessor = DocumentQualityAssessor()

test_docs = [
    {
        "doc_id": "doc_001",
        "content": "这是一篇很短的内容。",
        "metadata": {"last_modified": "2020-01-01"},
    },
    {
        "doc_id": "doc_002",
        "content": "\n\n".join([
            "# 产品使用指南\n\n本文档详细介绍产品的安装和配置步骤。",
            "## 系统要求\n\n需要 Python 3.9 或更高版本，建议使用 8GB 以上内存。",
            "## 安装步骤\n\n首先使用 pip 安装依赖包。然后配置环境变量。最后启动服务。",
            "## 配置说明\n\n配置文件位于 config 目录下。支持 YAML 和 JSON 两种格式。",
            "## 常见问题\n\n如果遇到连接超时，请检查防火墙设置和网络配置。",
        ]),
        "metadata": {"last_modified": "2026-06-15"},
    },
]

for doc in test_docs:
    report = assessor.evaluate(doc["doc_id"], doc["content"], doc["metadata"])
    print(f"\n文档: {report.doc_id}")
    print(f"  综合评分: {report.overall_score:.2f}")
    print(f"  各维度: {report.dimensions}")
    print(f"  通过: {report.passed}")
    if report.issues:
        print(f"  问题: {report.issues}")
```

### 6.2.3 质量阈值与处理策略

不同场景对文档质量的要求不同。以下是一组参考阈值：

| 应用场景 | 最低质量分 | 特殊要求 |
|---------|-----------|---------|
| 智能客服知识库 | 0.6 | 强调时效性和权威性 |
| 研发文档检索 | 0.4 | 可接受技术草稿，但要求代码示例完整 |
| 法律合规文档 | 0.8 | 强调完整性和权威性，必须有明确的来源和日期 |
| 产品手册 | 0.5 | 强调结构规范性 |

对于未通过质量评估的文档，可以选择以下策略：

1. **拒绝入库**：质量过低的文档直接丢弃
2. **标记人工审核**：边缘质量的文档标记后等待人工确认
3. **降级处理**：低质量文档仍可入库，但在检索时降低其权重
4. **自动修复**：对于格式问题（如编码错误），尝试自动修复

---

## 6.3 版本管理

知识库中的文档会随时间不断变化，合理的版本管理机制是保证数据一致性和可追溯性的基础。

### 6.3.1 全量更新 vs 增量更新

全量更新和增量更新各有适用场景：

| 对比维度 | 全量更新 | 增量更新 |
|---------|---------|---------|
| 实现复杂度 | 低 | 高 |
| 数据一致性 | 强（整体替换） | 最终一致性 |
| 资源消耗 | 高（全部重新处理） | 低（只处理变更） |
| 执行频率 | 低（每日/每周） | 高（实时/分钟级） |
| 适用场景 | 首次构建、数据修复、定期一致性校验 | 日常同步、实时管道 |
| 风险 | 更新过程中服务不可用 | 可能存在漏同步 |

在实际系统中，全量更新和增量更新通常是组合使用的：

- **增量更新**作为日常同步的主力机制，保证数据近实时性
- **全量更新**作为周期性的一致性校验（例如每周一次），确保增量同步没有遗漏或偏差

```python
"""
版本管理器
管理知识库的全量更新和增量更新策略
"""

import json
import time
from datetime import datetime, timedelta
from typing import Optional


class VersionManager:
    """
    知识库版本管理器
    
    管理全量更新和增量更新的执行策略，
    记录每次更新的版本信息，支持回滚。
    
    Usage:
        mgr = VersionManager(storage_path="./kb_versions")
        
        # 全量更新
        mgr.perform_full_update(update_func=my_full_update)
        
        # 增量更新
        mgr.perform_incremental_update(update_func=my_incremental_update)
    """
    
    def __init__(self, storage_path: str = "./kb_versions"):
        self.storage_path = storage_path
        self._ensure_storage()
    
    def _ensure_storage(self):
        """确保版本存储目录存在"""
        import os
        os.makedirs(self.storage_path, exist_ok=True)
    
    def _get_manifest_path(self) -> str:
        return f"{self.storage_path}/manifest.json"
    
    def _read_manifest(self) -> dict:
        """读取版本清单"""
        manifest_path = self._get_manifest_path()
        try:
            with open(manifest_path, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {
                "version": 0,
                "last_full_update": None,
                "last_incremental_update": None,
                "full_update_count": 0,
                "incremental_update_count": 0,
            }
    
    def _write_manifest(self, manifest: dict):
        with open(self._get_manifest_path(), "w") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
    
    def perform_full_update(
        self,
        update_func: callable,
        force: bool = False,
    ) -> bool:
        """
        执行全量更新
        
        全量更新会重建整个知识库索引。
        建议在系统低负载时段执行。
        
        Args:
            update_func: 执行全量更新的函数
                         签名: () -> bool (成功返回 True)
            force: 是否强制更新（跳过更新间隔检查）
        
        Returns:
            bool: 更新是否成功
        """
        manifest = self._read_manifest()
        
        # 检查更新间隔（至少 24 小时）
        if not force and manifest["last_full_update"]:
            last_update = datetime.fromisoformat(manifest["last_full_update"])
            if datetime.now() - last_update < timedelta(hours=1):
                print("[Version] 全量更新间隔过短，跳过")
                return False
        
        version = manifest["version"] + 1
        start_time = time.time()
        
        print(f"[Version] 开始全量更新 (version={version})...")
        
        try:
            success = update_func()
            if not success:
                print("[Version] 全量更新失败")
                return False
            
            elapsed = time.time() - start_time
            
            manifest["version"] = version
            manifest["last_full_update"] = datetime.now().isoformat()
            manifest["full_update_count"] = manifest.get("full_update_count", 0) + 1
            manifest["last_full_update_duration"] = round(elapsed, 2)
            
            self._write_manifest(manifest)
            
            print(
                f"[Version] 全量更新完成: version={version}, "
                f"耗时={elapsed:.1f}s"
            )
            return True
            
        except Exception as e:
            print(f"[Version] 全量更新异常: {e}")
            return False
    
    def perform_incremental_update(
        self,
        update_func: callable,
    ) -> bool:
        """
        执行增量更新
        
        增量更新只处理变更的文档。
        可以高频执行（分钟级）。
        
        Args:
            update_func: 执行增量更新的函数
                         签名: (last_version: int) -> bool
        
        Returns:
            bool: 更新是否成功
        """
        manifest = self._read_manifest()
        last_version = manifest.get("last_incremental_version", 0)
        current_version = manifest["version"]
        
        start_time = time.time()
        
        print(
            f"[Version] 开始增量更新 "
            f"(last_version={last_version}, current_version={current_version})..."
        )
        
        try:
            success = update_func(last_version)
            if not success:
                print("[Version] 增量更新失败")
                return False
            
            elapsed = time.time() - start_time
            
            manifest["last_incremental_version"] = current_version
            manifest["last_incremental_update"] = datetime.now().isoformat()
            manifest["incremental_update_count"] = manifest.get("incremental_update_count", 0) + 1
            
            self._write_manifest(manifest)
            
            print(f"[Version] 增量更新完成: 耗时={elapsed:.1f}s")
            return True
            
        except Exception as e:
            print(f"[Version] 增量更新异常: {e}")
            return False
    
    def get_current_version(self) -> int:
        """获取当前知识库版本号"""
        manifest = self._read_manifest()
        return manifest["version"]
    
    def get_update_status(self) -> dict:
        """获取更新状态摘要"""
        manifest = self._read_manifest()
        return {
            "version": manifest["version"],
            "last_full_update": manifest.get("last_full_update"),
            "last_incremental_update": manifest.get("last_incremental_update"),
            "full_update_count": manifest.get("full_update_count", 0),
            "incremental_update_count": manifest.get("incremental_update_count", 0),
            "last_full_update_duration": manifest.get("last_full_update_duration"),
        }
```

### 6.3.2 版本回滚策略

当知识库更新出现异常时，需要能够回滚到上一个稳定版本。回滚策略包括：

1. **快照回滚**：全量更新前保存向量数据库的快照，失败时直接恢复
2. **版本标记回滚**：每个文档维护多个版本，通过版本标记切换
3. **时间点恢复**：基于 CDC 事件日志重建到任意时间点的状态

对于向量数据库，回滚操作较为复杂。一个实用策略是：

- 全量更新时**不原地覆盖**，而是构建新的索引集合
- 新索引构建完成并通过验证后，**原子切换**查询指向
- 如果新索引有问题，立即**切回**旧索引

```python
class IndexSwitcher:
    """
    向量索引切换器
    
    支持原子切换和回滚的索引管理策略。
    """
    
    def __init__(self, vector_store, active_index_alias: str = "active"):
        self.vector_store = vector_store
        self.active_alias = active_index_alias
    
    def build_new_index(self, version: int) -> str:
        """构建新版本索引"""
        index_name = f"kb_v{version}"
        print(f"[IndexSwitcher] 构建新索引: {index_name}")
        
        # 1. 创建新的索引集合
        # 2. 执行全量文档写入
        # 3. 返回新索引名称
        
        return index_name
    
    def atomic_switch(self, new_index_name: str) -> bool:
        """
        原子切换索引
        
        将 active 别名从旧索引指向新索引。
        如果失败，active 别名仍指向旧索引。
        """
        old_index = self._get_active_index()
        print(
            f"[IndexSwitcher] 切换索引: "
            f"{old_index} -> {new_index_name}"
        )
        
        try:
            # 1. 将新索引的别名设为 active
            # 2. 从旧索引移除 active 别名
            # 3. 删除旧索引（可选）
            return True
        except Exception as e:
            print(f"[IndexSwitcher] 切换失败，保留旧索引: {e}")
            return False
    
    def rollback(self, target_version: int) -> bool:
        """回滚到指定版本"""
        target_index = f"kb_v{target_version}"
        return self.atomic_switch(target_index)
    
    def _get_active_index(self) -> Optional[str]:
        """获取当前 active 索引名称"""
        # 查询向量数据库的别名映射
        return None
```

---

## 6.4 更新机制

知识库的更新机制决定了数据的新鲜度和一致性。本节介绍变更检测和数据同步的核心机制。

### 6.4.1 变更检测策略

变更检测（Change Detection）是增量更新的前提。不同的数据源需要不同的检测策略：

| 策略 | 原理 | 适用场景 | 延迟 |
|------|------|---------|------|
| 轮询检测 | 定期检查文件 mtime 或数据版本 | 文件系统、数据库轮询 | 分钟级 |
| 事件驱动 | 通过 Webhook 或 CDC 接收变更通知 | Kafka、Debezium、GitHub Webhook | 秒级 |
| 对比检测 | 对比当前数据和快照的差异 | 全量一致性校验 | 取决于数据量 |
| 日志解析 | 解析数据库 WAL 或应用日志 | MySQL binlog、PostgreSQL WAL | 近实时 |

以下是混合使用轮询和事件驱动的变更检测器：

```python
"""
变更检测器
支持轮询和事件驱动两种模式
"""

import hashlib
import json
import time
from datetime import datetime
from typing import Callable, Optional
from dataclasses import dataclass


@dataclass
class DocumentChange:
    doc_id: str
    change_type: str  # "created", "updated", "deleted"
    source: str
    detected_at: str
    payload: dict = None


class ChangeDetector:
    """
    变更检测器
    
    支持轮询检测（Polling）和事件驱动（Event-driven）两种模式。
    可以同时注册多个检测源。
    
    Usage:
        detector = ChangeDetector()
        
        # 注册轮询检测器
        detector.register_polling_detector(
            name="file_system",
            detect_func=my_file_check,
            interval_seconds=300,  # 5 分钟
        )
        
        # 注册事件处理器
        detector.register_event_handler(
            source="kafka_cdc",
            handler=my_cdc_handler,
        )
        
        detector.start()
    """
    
    def __init__(self, change_callback: Callable[[DocumentChange], None] = None):
        self.change_callback = change_callback
        self.polling_detectors: list[dict] = []
        self.event_handlers: list[dict] = []
        self._running = False
    
    def register_polling_detector(
        self,
        name: str,
        detect_func: Callable[[], list[DocumentChange]],
        interval_seconds: int = 300,
    ):
        """
        注册轮询检测器
        
        Args:
            name: 检测器名称
            detect_func: 检测函数，返回变更列表
            interval_seconds: 轮询间隔（秒）
        """
        self.polling_detectors.append({
            "name": name,
            "detect_func": detect_func,
            "interval": interval_seconds,
            "last_run": 0,
        })
        print(f"[ChangeDetector] 注册轮询检测器: {name} (间隔={interval_seconds}s)")
    
    def register_event_handler(
        self,
        source: str,
        handler: Callable,
    ):
        """
        注册事件驱动处理器
        
        Args:
            source: 事件源标识
            handler: 事件处理函数，接收原始事件并返回 DocumentChange 列表
        """
        self.event_handlers.append({
            "source": source,
            "handler": handler,
        })
        print(f"[ChangeDetector] 注册事件处理器: {source}")
    
    def _run_polling_cycle(self, detector: dict):
        """执行一次轮询检测"""
        try:
            changes = detector["detect_func"]()
            for change in changes:
                if self.change_callback:
                    self.change_callback(change)
        except Exception as e:
            print(
                f"[ChangeDetector] 轮询检测失败: "
                f"{detector['name']}: {e}"
            )
    
    def _polling_loop(self):
        """轮询主循环"""
        while self._running:
            now = time.time()
            
            for detector in self.polling_detectors:
                elapsed = now - detector["last_run"]
                if elapsed >= detector["interval"]:
                    self._run_polling_cycle(detector)
                    detector["last_run"] = now
            
            time.sleep(1)  # 每秒检查一次
    
    def start(self):
        """启动变更检测"""
        print("[ChangeDetector] 启动变更检测...")
        self._running = True
        self._polling_loop()
    
    def stop(self):
        """停止变更检测"""
        self._running = False
        print("[ChangeDetector] 变更检测已停止")


# 示例：文件系统变更检测函数
def file_system_change_detector() -> list[DocumentChange]:
    """轮询检测文件系统变更（实际使用 FileSystemScanner）"""
    # 这里是简化的示例
    # 实际实现中调用 FileSystemScanner.run_incremental_scan()
    return []


# 示例：数据库变更检测函数
def database_change_detector() -> list[DocumentChange]:
    """轮询检测数据库表变更（基于时间戳列）"""
    # 查询 last_updated > last_check_time 的记录
    # 转换为 DocumentChange 列表
    return []
```

### 6.4.2 CDC 实现模式

基于 CDC 的实时同步是目前最先进的更新机制。以下是 CDC 的三种常见实现模式：

**模式一：基于应用日志的 CDC**

应用程序在写入数据库的同时，将变更事件写入一个专门的日志表或消息队列。这种方式的优点是简单，缺点是侵入性强——需要修改应用程序代码。

**模式二：基于数据库触发器的 CDC**

在数据库层面创建触发器，自动将变更记录到审计表或日志表。这种方式的优点是对应用透明，缺点是可能影响数据库性能。

**模式三：基于事务日志解析的 CDC**（推荐）

直接解析数据库的事务日志（如 MySQL 的 binlog、PostgreSQL 的 WAL），捕获所有的数据变更。这种方式的优点是零侵入、零延迟，缺点是配置复杂。

```python
"""
MySQL binlog CDC 示例（模拟）
使用 python-mysql-replication 库
"""

# 实际使用 pip install mysql-replication
# 以下是核心流程示意

def setup_mysql_cdc():
    """
    配置 MySQL binlog 监听
    
    需要 MySQL 开启 binlog:
        server-id = 1
        log_bin = /var/log/mysql/mysql-bin.log
        binlog_format = ROW  # 必须使用 ROW 格式
        binlog_row_image = FULL
        expire_logs_days = 7
    """
    # from pymysqlreplication import BinLogStreamReader
    # from pymysqlreplication.row_event import (
    #     WriteRowsEvent,
    #     UpdateRowsEvent,
    #     DeleteRowsEvent,
    # )
    #
    # mysql_settings = {
    #     "host": "127.0.0.1",
    #     "port": 3306,
    #     "user": "replicator",
    #     "passwd": "password",
    # }
    #
    # stream = BinLogStreamReader(
    #     connection_settings=mysql_settings,
    #     server_id=100,
    #     only_events=[WriteRowsEvent, UpdateRowsEvent, DeleteRowsEvent],
    #     only_schemas=["knowledge_base"],
    #     only_tables=["documents"],
    #     # resume_stream=True,      # 断点续传
    #     # blocking=True,           # 阻塞等待新事件
    # )
    #
    # for binlog_event in stream:
    #     for row in binlog_event.rows:
    #         event_type = {
    #             WriteRowsEvent: "c",
    #             UpdateRowsEvent: "u",
    #             DeleteRowsEvent: "d",
    #         }[type(binlog_event)]
    #
    #         process_cdc_event(event_type, row)
    #
    pass
```

---

## 6.5 维护与监控

知识库上线后不是一劳永逸的，持续的维护和监控才能保证系统的健康运行。

### 6.5.1 核心监控指标

知识库的健康状态需要从以下几个维度进行监控：

**数据层面指标：**

| 指标 | 说明 | 告警阈值 | 严重程度 |
|------|------|---------|---------|
| 文档总数 | 知识库中的文档数量 | 异常增长/下降 | 中等 |
| 新增文档速率 | 单位时间新增文档数 | 持续为零（可能同步中断） | 高 |
| 文档平均质量分 | 所有文档质量评分的均值 | 低于 0.4 | 高 |
| 文档时效性分布 | 各时间段文档占比 | 过时文档超过 30% | 中 |
| 数据源同步延迟 | 各数据源距上次同步的时间 | 超过配置间隔的 2 倍 | 高 |

**性能层面指标：**

| 指标 | 说明 | 告警阈值 |
|------|------|---------|
| 全量更新耗时 | 全量同步的执行时间 | 超过基准值的 2 倍 |
| 增量更新耗时 | 增量同步的执行时间 | 持续超过 5 分钟 |
| 文档处理吞吐量 | 每秒处理的文档数 | 低于基准值的 50% |
| Embedding 调用延迟 | Embedding API 的响应时间 | P99 超过 5 秒 |
| 向量写入延迟 | 向量数据库写入时间 | P99 超过 2 秒 |

### 6.5.2 健康检查与告警

```python
"""
知识库健康检查与告警系统
"""

import json
import time
from datetime import datetime, timedelta
from typing import Optional


class KnowledgeBaseHealthMonitor:
    """
    知识库健康监控器
    
    定期执行健康检查，记录指标，发送告警。
    
    Usage:
        monitor = KnowledgeBaseHealthMonitor(
            vector_store=my_vector_store,
            alert_channels=["slack", "email"],
        )
        
        # 注册检查项
        monitor.register_check("document_count", check_doc_count, interval=300)
        monitor.register_check("sync_lag", check_sync_lag, interval=60)
        
        monitor.start()
    """
    
    def __init__(
        self,
        vector_store=None,
        alert_channels: list[str] = None,
        metrics_storage_path: str = "./kb_metrics",
    ):
        self.vector_store = vector_store
        self.alert_channels = alert_channels or []
        self.metrics_storage_path = metrics_storage_path
        self.checks: list[dict] = []
        self.alert_rules: list[dict] = []
        self._running = False
        
        # 确保存储目录存在
        import os
        os.makedirs(metrics_storage_path, exist_ok=True)
    
    def register_check(
        self,
        name: str,
        check_func: callable,
        interval: int = 300,
    ):
        """
        注册健康检查项
        
        Args:
            name: 检查项名称
            check_func: 检查函数，返回 (passed: bool, details: dict)
            interval: 检查间隔（秒）
        """
        self.checks.append({
            "name": name,
            "func": check_func,
            "interval": interval,
            "last_run": 0,
            "last_result": None,
        })
    
    def add_alert_rule(
        self,
        check_name: str,
        condition: str,  # "below", "above", "equals"
        threshold: float,
        message: str,
        severity: str = "warning",  # "info", "warning", "critical"
    ):
        """添加告警规则"""
        self.alert_rules.append({
            "check_name": check_name,
            "condition": condition,
            "threshold": threshold,
            "message": message,
            "severity": severity,
        })
    
    def _send_alert(self, rule: dict, current_value: float):
        """发送告警"""
        message = (
            f"[{rule['severity'].upper()}] {rule['message']} "
            f"(当前值: {current_value}, 阈值: {rule['threshold']})"
        )
        print(f"[Monitor] 告警: {message}")
        
        for channel in self.alert_channels:
            if channel == "slack":
                self._send_slack_alert(message)
            elif channel == "email":
                self._send_email_alert(message)
    
    def _send_slack_alert(self, message: str):
        """发送 Slack 告警"""
        # 实际实现中使用 Slack Webhook
        pass
    
    def _send_email_alert(self, message: str):
        """发送邮件告警"""
        # 实际实现中使用 SMTP
        pass
    
    def _save_metric(self, name: str, value: float, timestamp: str):
        """保存指标到本地存储"""
        metric_file = f"{self.metrics_storage_path}/{name}.jsonl"
        record = json.dumps({"timestamp": timestamp, "value": value})
        with open(metric_file, "a") as f:
            f.write(record + "\n")
    
    def _run_checks(self):
        """执行所有健康检查"""
        now = time.time()
        timestamp = datetime.now().isoformat()
        
        for check in self.checks:
            elapsed = now - check["last_run"]
            if elapsed < check["interval"]:
                continue
            
            try:
                passed, details = check["func"]()
                
                # 保存指标
                if "value" in details:
                    self._save_metric(check["name"], details["value"], timestamp)
                
                check["last_result"] = {"passed": passed, "details": details, "timestamp": timestamp}
                check["last_run"] = now
                
                # 检查告警规则
                if "value" in details:
                    for rule in self.alert_rules:
                        if rule["check_name"] != check["name"]:
                            continue
                        
                        value = details["value"]
                        triggered = False
                        
                        if rule["condition"] == "below" and value < rule["threshold"]:
                            triggered = True
                        elif rule["condition"] == "above" and value > rule["threshold"]:
                            triggered = True
                        elif rule["condition"] == "equals" and value == rule["threshold"]:
                            triggered = True
                        
                        if triggered:
                            self._send_alert(rule, value)
                
                if not passed:
                    print(f"[Monitor] 检查失败: {check['name']} - {details}")
                
            except Exception as e:
                print(f"[Monitor] 检查异常: {check['name']}: {e}")
    
    def start(self):
        """启动监控"""
        print("[Monitor] 启动知识库健康监控...")
        self._running = True
        
        while self._running:
            self._run_checks()
            time.sleep(5)
    
    def stop(self):
        self._running = False
    
    def get_status_report(self) -> dict:
        """生成当前状态报告"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "checks": {},
            "overall_health": "healthy",
        }
        
        failure_count = 0
        for check in self.checks:
            result = check.get("last_result")
            report["checks"][check["name"]] = {
                "passed": result["passed"] if result else "unknown",
                "last_run": result["timestamp"] if result else None,
            }
            if result and not result["passed"]:
                failure_count += 1
        
        if failure_count > len(self.checks) / 2:
            report["overall_health"] = "critical"
        elif failure_count > 0:
            report["overall_health"] = "degraded"
        
        return report


# 健康检查函数示例
def check_document_count(vector_store) -> tuple[bool, dict]:
    """检查文档数量是否在正常范围"""
    try:
        count = vector_store.count_documents()
        # 假设正常范围是 1000 - 100000
        if count < 1000:
            return False, {"value": count, "message": f"文档数量异常偏低: {count}"}
        elif count > 100000:
            return False, {"value": count, "message": f"文档数量异常偏高: {count}"}
        else:
            return True, {"value": count, "message": f"文档数量正常: {count}"}
    except Exception as e:
        return False, {"value": -1, "message": f"查询失败: {e}"}


def check_sync_lag(sources_status: dict) -> tuple[bool, dict]:
    """检查各数据源的同步延迟"""
    max_lag = 0
    all_healthy = True
    
    for source_name, status in sources_status.items():
        last_sync = status.get("last_sync")
        if last_sync is None:
            all_healthy = False
            continue
        
        lag = (datetime.now() - datetime.fromisoformat(last_sync)).total_seconds()
        max_lag = max(max_lag, lag)
        
        if lag > status.get("max_allowed_lag_seconds", 3600):
            all_healthy = False
    
    return all_healthy, {
        "value": max_lag,
        "max_lag_seconds": max_lag,
        "message": f"最大同步延迟: {max_lag:.0f} 秒",
    }
```

### 6.5.3 日志与审计

知识库的所有变更操作都应记录审计日志，以便追踪问题来源和满足合规要求：

```python
class AuditLogger:
    """
    知识库操作审计日志
    
    记录所有知识库变更操作，支持查询和回溯。
    """
    
    def __init__(self, storage_backend=None):
        # 可以写入文件、数据库或 ELK
        self.backend = storage_backend or []
    
    def log(
        self,
        action: str,         # "create", "update", "delete", "sync", "rollback"
        target_id: str,      # 文档 ID 或索引版本
        source: str,         # 操作来源（数据源）
        operator: str,       # 操作者（系统或用户）
        details: dict = None,
    ):
        """记录一条审计日志"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "target_id": target_id,
            "source": source,
            "operator": operator,
            "details": details or {},
        }
        
        print(f"[Audit] {entry}")
        
        # 持久化存储
        self._persist(entry)
    
    def _persist(self, entry: dict):
        """持久化审计日志"""
        import json
        log_file = f"./audit_logs/{datetime.now().strftime('%Y-%m-%d')}.jsonl"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    def query(
        self,
        start_time: str = None,
        end_time: str = None,
        action: str = None,
        target_id: str = None,
        limit: int = 100,
    ) -> list[dict]:
        """查询审计日志"""
        # 实际实现中读取日志文件并过滤
        return []
```

---

## 6.6 访问控制集成

企业级知识库中，不同角色和团队只能访问与其权限匹配的文档。访问控制（Access Control）是知识库安全的基础。

### 6.6.1 权限模型设计

知识库的权限模型通常包含三个要素：

- **主体（Subject）**：用户、用户组、角色
- **资源（Resource）**：文档、文档集合、知识库空间
- **操作（Permission）**：读、写、删除、管理

推荐的权限模型是 **RBAC（Role-Based Access Control）+ 资源级权限**的组合：

```python
"""
知识库访问控制系统

基于 RBAC + 资源级权限的访问控制模型。
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional


class Permission(Enum):
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"


@dataclass
class User:
    id: str
    name: str
    roles: list[str] = field(default_factory=list)
    groups: list[str] = field(default_factory=list)


@dataclass
class DocumentACL:
    """
    文档级访问控制列表
    
    每个文档可以绑定一个 ACL，控制哪些用户/角色/组可以访问。
    """
    doc_id: str
    allowed_users: list[str] = field(default_factory=list)
    allowed_roles: list[str] = field(default_factory=list)
    allowed_groups: list[str] = field(default_factory=list)
    public: bool = False  # 是否公开访问


class AccessController:
    """
    知识库访问控制器
    
    提供权限检查和文档过滤功能。
    集成到检索流程中，确保用户只能看到其有权访问的文档。
    
    Usage:
        acl = AccessController()
        
        # 注册用户和角色
        acl.register_role("editor", [Permission.READ, Permission.WRITE])
        acl.register_user(User(id="u001", name="Alice", roles=["editor"]))
        
        # 设置文档权限
        acl.set_document_acl(DocumentACL(
            doc_id="doc_001",
            allowed_roles=["editor", "admin"],
        ))
        
        # 检查权限
        if acl.check_permission("u001", "doc_001", Permission.READ):
            # 允许访问
            pass
    """
    
    def __init__(self):
        self.users: dict[str, User] = {}
        self.roles: dict[str, set[Permission]] = {}
        self.doc_acls: dict[str, DocumentACL] = {}
        self.group_membership: dict[str, set[str]] = {}  # group_id -> user_ids
    
    def register_role(self, role_name: str, permissions: list[Permission]):
        """注册角色及其权限集合"""
        self.roles[role_name] = set(permissions)
    
    def register_user(self, user: User):
        """注册用户"""
        self.users[user.id] = user
    
    def add_user_to_group(self, user_id: str, group_id: str):
        """将用户添加到用户组"""
        if group_id not in self.group_membership:
            self.group_membership[group_id] = set()
        self.group_membership[group_id].add(user_id)
    
    def set_document_acl(self, acl: DocumentACL):
        """设置文档的 ACL"""
        self.doc_acls[acl.doc_id] = acl
    
    def _get_user_effective_permissions(self, user_id: str) -> set[Permission]:
        """获取用户所有有效的权限（从角色和组继承）"""
        user = self.users.get(user_id)
        if not user:
            return set()
        
        effective = set()
        
        # 从角色继承
        for role_name in user.roles:
            role_perms = self.roles.get(role_name, set())
            effective.update(role_perms)
        
        return effective
    
    def check_permission(
        self, user_id: str, doc_id: str, required_perm: Permission
    ) -> bool:
        """
        检查用户是否对文档拥有指定权限
        
        检查顺序：
        1. 文档是否公开
        2. 用户是否在文档的白名单中
        3. 用户角色是否有权限
        4. 用户组是否允许
        """
        acl = self.doc_acls.get(doc_id)
        if not acl:
            # 无 ACL 的文档默认禁止访问（安全优先）
            return False
        
        if acl.public:
            return True
        
        user = self.users.get(user_id)
        if not user:
            return False
        
        # 用户级别白名单
        if user_id in acl.allowed_users:
            return True
        
        # 角色级别检查
        for role_name in acl.allowed_roles:
            if role_name in user.roles:
                role_perms = self.roles.get(role_name, set())
                if required_perm in role_perms:
                    return True
        
        # 用户组级别检查
        for group_id in acl.allowed_groups:
            group_users = self.group_membership.get(group_id, set())
            if user_id in group_users:
                # 检查组是否有该权限
                group_role = f"group:{group_id}"
                role_perms = self.roles.get(group_role, set())
                if required_perm in role_perms:
                    return True
        
        return False
    
    def filter_accessible_docs(
        self, user_id: str, doc_ids: list[str], required_perm: Permission = Permission.READ
    ) -> list[str]:
        """
        从文档列表中过滤出用户有权访问的文档
        
        在检索阶段调用，确保用户只能看到其有权访问的文档。
        
        Args:
            user_id: 用户 ID
            doc_ids: 候选文档 ID 列表
            required_perm: 所需权限
        
        Returns:
            用户有权访问的文档 ID 列表
        """
        accessible = []
        for doc_id in doc_ids:
            if self.check_permission(user_id, doc_id, required_perm):
                accessible.append(doc_id)
        
        return accessible


# 集成到检索流程的示例
class SecureRetriever:
    """
    带访问控制的检索器
    
    在向量检索后，对结果进行权限过滤。
    """
    
    def __init__(self, vector_store, access_controller: AccessController):
        self.vector_store = vector_store
        self.acl = access_controller
    
    def retrieve(
        self,
        query: str,
        user_id: str,
        top_k: int = 10,
    ) -> list[dict]:
        """
        执行带权限过滤的检索
        
        流程：
        1. 向量检索获取候选文档（多取一些，因为后续要过滤）
        2. 权限过滤
        3. 返回用户有权访问的文档
        """
        # 多取一些以补偿权限过滤的损耗
        fetch_k = top_k * 3
        
        candidates = self.vector_store.search(query, top_k=fetch_k)
        candidate_ids = [doc["id"] for doc in candidates]
        
        # 权限过滤
        accessible_ids = set(
            self.acl.filter_accessible_docs(user_id, candidate_ids)
        )
        
        # 返回过滤后的结果
        results = [
            doc for doc in candidates
            if doc["id"] in accessible_ids
        ]
        
        return results[:top_k]
```

### 6.6.2 与 SSO/LDAP 集成

企业环境中，用户信息和组织结构通常存储在 SSO（Single Sign-On）或 LDAP（Lightweight Directory Access Protocol）系统中。知识库的访问控制需要与这些系统集成：

```python
"""
LDAP/SSO 集成示例
"""

import ldap3


class LDAPUserSync:
    """
    LDAP 用户同步
    
    从 LDAP/AD 同步用户和组信息到知识库的访问控制系统。
    """
    
    def __init__(
        self,
        server: str,
        base_dn: str,
        bind_user: str,
        bind_password: str,
        access_controller: AccessController,
    ):
        self.server = server
        self.base_dn = base_dn
        self.bind_user = bind_user
        self.bind_password = bind_password
        self.acl = access_controller
    
    def sync_users(self):
        """从 LDAP 同步所有用户"""
        server = ldap3.Server(self.server, get_info=ldap3.ALL)
        
        with ldap3.Connection(
            server, self.bind_user, self.bind_password, auto_bind=True
        ) as conn:
            # 查询所有用户
            conn.search(
                search_base=self.base_dn,
                search_filter="(objectClass=user)",
                attributes=["cn", "mail", "department", "memberOf"],
            )
            
            for entry in conn.entries:
                user_id = entry.mail.value if entry.mail.value else entry.cn.value
                groups = [
                    group.split(",")[0].split("=")[1]
                    for group in entry.memberOf.values
                ] if entry.memberOf.values else []
                
                user = User(
                    id=user_id,
                    name=entry.cn.value,
                    roles=self._map_ldap_groups_to_roles(groups),
                    groups=groups,
                )
                self.acl.register_user(user)
                
                # 同步用户组
                for group in groups:
                    self.acl.add_user_to_group(user_id, group)
    
    def _map_ldap_groups_to_roles(self, ldap_groups: list[str]) -> list[str]:
        """将 LDAP 组映射为知识库角色"""
        mapping = {
            "CN=KnowledgeBase-Admins": ["admin"],
            "CN=KnowledgeBase-Editors": ["editor"],
            "CN=KnowledgeBase-Readers": ["reader"],
        }
        
        roles = []
        for group in ldap_groups:
            if group in mapping:
                roles.extend(mapping[group])
        
        return roles
```

### 6.6.3 文档级权限继承

企业文档通常有层级结构（如 Confluence 的页面树、文件系统的目录树），权限应该能够从父节点继承到子节点：

```python
class HierarchicalACLManager:
    """
    层级权限管理器
    
    支持权限继承：子文档继承父文档/空间的权限设置。
    """
    
    def __init__(self, access_controller: AccessController):
        self.acl = access_controller
        self.parent_map: dict[str, Optional[str]] = {}  # child_id -> parent_id
    
    def set_hierarchy(self, parent_id: str, child_ids: list[str]):
        """设置文档间的层级关系"""
        for child_id in child_ids:
            self.parent_map[child_id] = parent_id
    
    def get_effective_acl(self, doc_id: str) -> Optional[DocumentACL]:
        """获取文档的有效 ACL（考虑继承）"""
        # 先看文档自身是否有 ACL
        acl = self.acl.doc_acls.get(doc_id)
        if acl:
            return acl
        
        # 向上查找父节点的 ACL
        parent_id = self.parent_map.get(doc_id)
        while parent_id:
            parent_acl = self.acl.doc_acls.get(parent_id)
            if parent_acl:
                return parent_acl
            parent_id = self.parent_map.get(parent_id)
        
        # 都没有找到，返回默认（禁止访问）
        return None
    
    def check_permission(
        self, user_id: str, doc_id: str, required_perm: Permission
    ) -> bool:
        """检查权限（含继承）"""
        effective_acl = self.get_effective_acl(doc_id)
        if effective_acl is None:
            return False
        
        # 使用 ACL 检查权限（同 AccessController.check_permission）
        return self.acl.check_permission(user_id, effective_acl.doc_id, required_perm)
```

---

## 6.7 本章小结

知识库是 RAG 系统的数据基石。本章涵盖了知识库从构建到运维的完整生命周期：

1. **数据源集成**：介绍了文件系统、数据库 CDC、API（Confluence/Notion/SharePoint）、网络爬虫和消息队列 Kafka 等多种数据源的接入方式。核心原则是——根据数据的实时性要求和变更频率选择合适的接入策略。

2. **文档质量评估**：从完整性、可读性、信息密度、时效性等维度对文档进行质量评分，确保进入知识库的文档都有基本的内容保障。

3. **版本管理**：全量更新保证一致性，增量更新保证实时性，两者互为补充。版本回滚机制为系统提供安全网。

4. **更新机制**：变更检测是增量更新的前提，CDC 是实现近实时同步的最佳实践。

5. **维护监控**：从数据和性能两个维度监控知识库健康状态，设置合理的告警阈值，建立审计日志体系。

6. **访问控制**：基于 RBAC 的权限模型，支持文档级细粒度控制和 LDAP/SSO 集成，确保知识库安全可控。

在下一章中，我们将讨论如何通过查询重写、路由和检索增强技术来优化 RAG 系统的检索质量。
