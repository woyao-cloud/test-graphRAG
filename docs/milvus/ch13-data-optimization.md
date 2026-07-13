# 第13章: RAG数据层优化：Milvus专属知识库治理

## 13.1 引言

在RAG系统中，数据质量决定了检索质量的上限。即便拥有最先进的检索算法和模型，如果知识库本身存在噪声、冗余或过时信息，最终的生成效果也会大打折扣。Milvus作为知识库的存储和检索核心，提供了丰富的数据管理能力，但如何高效地构建和维护知识库，仍然需要系统性的方法论。本章将从智能切片、增量更新、数据去重、分区管理和多版本控制等角度，全面探讨基于Milvus的RAG知识库治理策略。

## 13.2 智能切片策略

文档切片（Chunking）是RAG知识库构建的第一步，也是最重要的一步。切片的粒度直接影响检索质量：切片过大，包含过多无关信息，降低精准度；切片过小，缺乏上下文语义，降低召回率。

### 13.2.1 固定长度切片及其局限

最简单的切片方式是按固定字符数切分，但这种方式存在明显缺陷：容易切断句子或段落，破坏语义完整性。

```python
# 基础固定长度切片（不推荐单独使用）
def fixed_chunk(text: str, chunk_size: int = 512, overlap: int = 64) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks
```

### 13.2.2 语义切片

语义切片利用嵌入模型的相似度检测语义边界，在语义不连贯的地方进行切分。

```python
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Tuple

class SemanticChunker:
    """基于语义相似度的智能切片器"""
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-zh-v1.5",
        min_chunk_size: int = 128,
        max_chunk_size: int = 1024,
        similarity_threshold: float = 0.75
    ):
        self.model = SentenceTransformer(model_name)
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.similarity_threshold = similarity_threshold
    
    def split_into_sentences(self, text: str) -> List[str]:
        """将文本切分为句子"""
        import re
        # 中文句子分割
        sentences = re.split(r'(?<=[。！？\n])', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def chunk(self, text: str) -> List[str]:
        sentences = self.split_into_sentences(text)
        
        if not sentences:
            return []
        
        # 对每个句子进行编码
        embeddings = self.model.encode(sentences)
        
        chunks = []
        current_chunk = [sentences[0]]
        current_embedding = embeddings[0]
        
        for i in range(1, len(sentences)):
            # 计算当前句子与当前块的语义相似度
            chunk_embedding = np.mean(
                [embeddings[j] for j in range(
                    i - len(current_chunk), i
                )] + [current_embedding],
                axis=0
            )
            similarity = np.dot(
                chunk_embedding, embeddings[i]
            ) / (np.linalg.norm(chunk_embedding) * np.linalg.norm(embeddings[i]))
            
            current_text = "".join(current_chunk)
            
            if similarity >= self.similarity_threshold or len(current_text) < self.min_chunk_size:
                # 语义连续，继续合并
                current_chunk.append(sentences[i])
            else:
                # 语义边界，保存当前块
                if len(current_text) >= self.min_chunk_size:
                    chunks.append(current_text)
                    current_chunk = [sentences[i]]
                else:
                    current_chunk.append(sentences[i])
            
            # 强制切分：超过最大长度
            if len("".join(current_chunk)) >= self.max_chunk_size:
                chunks.append("".join(current_chunk))
                current_chunk = []
        
        # 处理最后一个块
        if current_chunk:
            chunks.append("".join(current_chunk))
        
        return chunks
```

### 13.2.3 段落级切片

对于结构良好的文档（如技术文档、报告），按段落或标题层级进行切片可以保留文档的原始结构信息。

```python
import re
from typing import List, Dict

class HeadingBasedChunker:
    """基于标题结构的层级切片器"""
    
    def __init__(self, max_depth: int = 3, min_section_length: int = 100):
        self.max_depth = max_depth
        self.min_section_length = min_section_length
    
    def parse_headings(self, text: str) -> List[Dict]:
        """解析文档的标题结构"""
        # 匹配 Markdown 标题和普通文本标题
        pattern = r'^(#{1,%d})\s+(.+)$' % self.max_depth
        lines = text.split('\n')
        
        structure = []
        current_heading = None
        current_content = []
        
        for line in lines:
            match = re.match(pattern, line.strip())
            if match:
                # 保存上一个章节
                if current_heading is not None:
                    structure.append({
                        'level': len(current_heading) - 1,
                        'title': current_heading,
                        'content': '\n'.join(current_content)
                    })
                current_heading = match.group(2)
                current_content = []
            else:
                current_content.append(line)
        
        # 保存最后一个章节
        if current_heading is not None:
            structure.append({
                'level': 0,
                'title': current_heading,
                'content': '\n'.join(current_content)
            })
        
        return structure
    
    def chunk(self, text: str) -> List[Dict]:
        """基于标题结构进行切片，返回带元数据的块"""
        sections = self.parse_headings(text)
        chunks = []
        
        for section in sections:
            content = section['content'].strip()
            if len(content) < self.min_section_length:
                # 内容过短，合并到上一个块
                if chunks:
                    chunks[-1]['text'] += '\n' + content
                    chunks[-1]['metadata']['merged_sections'] = \
                        chunks[-1]['metadata'].get('merged_sections', []) + [section['title']]
                continue
            
            chunks.append({
                'text': content,
                'metadata': {
                    'heading': section['title'],
                    'heading_level': section['level'],
                    'char_count': len(content)
                }
            })
        
        return chunks
```

### 13.2.4 自适应切片策略

实际应用中，不同类型文档需要不同的切片策略。自适应切片根据文档特征自动选择最优策略。

```python
class AdaptiveChunker:
    """自适应切片器：根据文档特征自动选择切片策略"""
    
    def __init__(self):
        self.semantic_chunker = SemanticChunker()
        self.heading_chunker = HeadingBasedChunker()
    
    def detect_document_type(self, text: str) -> str:
        """检测文档类型"""
        # 检查是否包含标题结构
        heading_count = len(re.findall(r'^#{1,6}\s', text, re.MULTILINE))
        if heading_count >= 3:
            return "structured"
        
        # 检查平均句子长度
        sentences = re.split(r'[。！？\n]', text)
        avg_sentence_len = sum(len(s) for s in sentences) / max(len(sentences), 1)
        
        if avg_sentence_len > 100:
            return "narrative"  # 叙事型文本
        elif avg_sentence_len > 40:
            return "technical"  # 技术型文本
        else:
            return "conversational"  # 对话型文本
    
    def chunk(self, text: str) -> List[Dict]:
        doc_type = self.detect_document_type(text)
        
        if doc_type == "structured":
            raw_chunks = self.heading_chunker.chunk(text)
        else:
            raw_chunks = [
                {'text': t, 'metadata': {'chunk_type': doc_type}}
                for t in self.semantic_chunker.chunk(text)
            ]
        
        # 统一添加元数据
        for chunk in raw_chunks:
            chunk['metadata']['doc_type'] = doc_type
            chunk['metadata']['chunk_id'] = hash(chunk['text']) % (10 ** 10)
        
        return raw_chunks
```

## 13.3 增量更新机制

在真实业务场景中，知识库需要不断更新——新增文档、修改已有内容、删除过时信息。全量重建不仅耗时，而且浪费计算资源。Milvus支持高效的增量更新操作。

### 13.3.1 增量插入新文档

```python
from pymilvus import Collection
from datetime import datetime
import hashlib

class IncrementalUpdater:
    """增量更新管理器"""
    
    def __init__(self, collection: Collection, id_field: str = "id"):
        self.collection = collection
        self.id_field = id_field
    
    def generate_document_id(self, content: str, source: str = "") -> int:
        """基于内容生成唯一文档ID"""
        hash_input = f"{content}{source}".encode('utf-8')
        return int(hashlib.md5(hash_input).hexdigest()[:8], 16)
    
    def insert_new_documents(
        self,
        documents: List[Dict],
        embedding_model
    ) -> Tuple[int, int]:
        """
        增量插入新文档
        
        Returns:
            (插入数, 跳过数)
        """
        inserted = 0
        skipped = 0
        
        for doc in documents:
            doc_id = self.generate_document_id(doc['text'], doc.get('source', ''))
            
            # 检查文档是否已存在
            existing = self.collection.query(
                expr=f'{self.id_field} == {doc_id}',
                output_fields=[self.id_field]
            )
            
            if existing:
                skipped += 1
                continue
            
            # 生成向量
            embedding = embedding_model.encode(doc['text']).tolist()
            
            # 插入新文档
            self.collection.insert([
                [doc_id],
                [embedding],
                [doc['text']],
                [doc.get('source', '')],
                [int(datetime.now().timestamp())],
                [doc.get('category', 'default')]
            ])
            inserted += 1
        
        # 刷新索引
        if inserted > 0:
            self.collection.flush()
        
        return inserted, skipped
```

### 13.3.2 文档更新与版本追踪

对于需要更新已有文档的场景，采用软删除+新增的策略，保留历史版本。

```python
class VersionedUpdater:
    """带版本追踪的文档更新器"""
    
    def __init__(self, collection: Collection):
        self.collection = collection
        # 在 schema 中需要包含 version 和 is_latest 字段
    
    def update_document(
        self,
        doc_id: int,
        new_text: str,
        new_embedding: List[float],
        source: str = ""
    ) -> int:
        """
        更新文档，创建新版本
        
        Returns:
            新版本号
        """
        # 查询当前最新版本
        current = self.collection.query(
            expr=f'id == {doc_id} && is_latest == true',
            output_fields=["version"]
        )
        
        new_version = 1
        if current:
            # 标记旧版本为非最新
            self.collection.upsert([
                [doc_id],
                [current[0]['version']],
                [False]  # is_latest = false
            ])
            new_version = current[0]['version'] + 1
        
        # 插入新版本
        self.collection.insert([
            [doc_id],
            [new_version],
            [new_text],
            [new_embedding],
            [True],  # is_latest
            [source],
            [int(datetime.now().timestamp())]
        ])
        
        self.collection.flush()
        return new_version
    
    def get_document_history(self, doc_id: int) -> List[Dict]:
        """获取文档的所有历史版本"""
        results = self.collection.query(
            expr=f'id == {doc_id}',
            output_fields=["version", "text", "timestamp", "is_latest"],
            order_by="version desc"
        )
        return results
```

### 13.3.3 批量增量更新与性能优化

```python
class BatchIncrementalUpdater:
    """批量增量更新，优化写入性能"""
    
    def __init__(self, collection: Collection, batch_size: int = 1000):
        self.collection = collection
        self.batch_size = batch_size
    
    def batch_insert(self, documents: List[Dict], embedding_model) -> Dict:
        """
        批量增量插入
        
        Returns:
            统计信息
        """
        stats = {"total": len(documents), "inserted": 0, "skipped": 0, "errors": 0}
        batch_buffer = []
        
        for i, doc in enumerate(documents):
            try:
                embedding = embedding_model.encode(doc['text']).tolist()
                batch_buffer.append({
                    'embedding': embedding,
                    'text': doc['text'],
                    'source': doc.get('source', ''),
                    'timestamp': int(datetime.now().timestamp()),
                    'category': doc.get('category', 'default')
                })
                
                # 达到批处理大小，批量写入
                if len(batch_buffer) >= self.batch_size:
                    self._flush_batch(batch_buffer)
                    stats['inserted'] += len(batch_buffer)
                    batch_buffer = []
                    
            except Exception as e:
                stats['errors'] += 1
                print(f"处理文档 {i} 时出错: {e}")
        
        # 处理剩余的文档
        if batch_buffer:
            self._flush_batch(batch_buffer)
            stats['inserted'] += len(batch_buffer)
        
        return stats
    
    def _flush_batch(self, batch: List[Dict]):
        """批量写入 Milvus"""
        self.collection.insert([
            [d['embedding'] for d in batch],
            [d['text'] for d in batch],
            [d['source'] for d in batch],
            [d['timestamp'] for d in batch],
            [d['category'] for d in batch]
        ])
```

## 13.4 数据去重与降噪

### 13.4.1 向量级去重

基于向量相似度检测并移除重复或高度相似的文档。

```python
class VectorDeduplicator:
    """基于向量相似度的去重工具"""
    
    def __init__(
        self,
        collection: Collection,
        similarity_threshold: float = 0.95
    ):
        self.collection = collection
        self.similarity_threshold = similarity_threshold
    
    def find_duplicates(self, batch_size: int = 1000) -> List[Tuple[int, int, float]]:
        """
        查找知识库中的重复文档
        
        Returns:
            重复对列表 [(id1, id2, similarity), ...]
        """
        duplicates = []
        
        # 分批检索所有文档
        total = self.collection.num_entities
        for offset in range(0, total, batch_size):
            results = self.collection.query(
                expr=f'id >= 0',
                output_fields=["id", "embedding"],
                limit=batch_size,
                offset=offset
            )
            
            # 对每个文档，检索其最相似的文档
            for doc in results:
                similar = self.collection.search(
                    data=[doc['embedding']],
                    anns_field="embedding",
                    param={"metric_type": "IP", "params": {"nprobe": 16}},
                    limit=5,
                    expr=f'id != {doc["id"]}'
                )
                
                for hit in similar[0]:
                    if hit.score >= self.similarity_threshold:
                        duplicates.append((doc['id'], hit.id, hit.score))
        
        return duplicates
    
    def remove_duplicates(self, strategy: str = "keep_first") -> int:
        """
        移除重复文档
        
        Args:
            strategy: "keep_first" 或 "keep_latest"
        
        Returns:
            移除的文档数
        """
        duplicates = self.find_duplicates()
        to_delete = set()
        
        for id1, id2, score in duplicates:
            if strategy == "keep_first":
                to_delete.add(id2)
            else:
                to_delete.add(id1)
        
        # 删除重复文档
        for doc_id in to_delete:
            self.collection.delete(f'id == {doc_id}')
        
        self.collection.flush()
        return len(to_delete)
```

### 13.4.2 文本级降噪

在插入前对原始文本进行清洗，移除噪声内容。

```python
import re

class TextCleaner:
    """文本降噪处理器"""
    
    def __init__(self):
        self.noise_patterns = [
            (r'<script[^>]*>.*?</script>', '', re.DOTALL),  # HTML script
            (r'<style[^>]*>.*?</style>', '', re.DOTALL),    # CSS style
            (r'<!--.*?-->', '', re.DOTALL),                  # HTML 注释
            (r'\n{3,}', '\n\n'),                             # 多余空行
            (r'\s{2,}', ' '),                                # 多余空白
            (r'[\U0001F600-\U0001F64F]', ''),               # Emoji
            (r'[\U0001F300-\U0001F5FF]', ''),               # 符号
            (r'[\U0001F680-\U0001F6FF]', ''),               # 交通符号
        ]
        
        self.boilerplate_patterns = [
            r'版权所有.*?\n',
            r'免责声明.*?\n',
            r'点击这里.*?',
            r'更多信息.*?',
            r'关注我们.*?',
        ]
    
    def clean(self, text: str) -> str:
        """清洗文本"""
        # 移除噪声模式
        for pattern, replacement, *flags in self.noise_patterns:
            flag = flags[0] if flags else 0
            text = re.sub(pattern, replacement, text, flags=flag)
        
        # 移除模板化内容
        for pattern in self.boilerplate_patterns:
            text = re.sub(pattern, '', text)
        
        # 去除首尾空白
        text = text.strip()
        
        return text
    
    def is_quality_low(self, text: str, min_length: int = 20) -> bool:
        """检查文本质量是否过低"""
        if len(text) < min_length:
            return True
        
        # 检查信息密度（有效字符占比）
        total_chars = len(text)
        meaningful_chars = len(re.findall(r'[一-鿿\w]', text))
        density = meaningful_chars / total_chars if total_chars > 0 else 0
        
        return density < 0.3  # 有效字符占比低于30%视为低质量
```

## 13.5 分区冷热分离

### 13.5.1 基于时间的分区策略

Milvus支持集合分区（Partition），可以将数据按时间或其他维度划分到不同分区中。冷热分离策略将高频访问的热数据和小概率访问的冷数据分开存储。

```python
from datetime import datetime, timedelta

class HotColdPartitioner:
    """冷热数据分区管理器"""
    
    def __init__(self, collection: Collection):
        self.collection = collection
        self._ensure_partitions()
    
    def _ensure_partitions(self):
        """确保必要的分区存在"""
        partitions = self.collection.partitions
        existing = [p.name for p in partitions]
        
        required = ["hot_data", "warm_data", "cold_data"]
        for name in required:
            if name not in existing:
                self.collection.create_partition(name)
    
    def classify_and_insert(
        self,
        documents: List[Dict],
        embedding_model
    ):
        """根据文档时间戳分类并插入到对应分区"""
        now = datetime.now()
        
        for doc in documents:
            timestamp = doc.get('timestamp', now)
            doc_time = datetime.fromtimestamp(timestamp) if isinstance(timestamp, (int, float)) else timestamp
            age_days = (now - doc_time).days
            
            # 根据时间分类
            if age_days <= 30:
                partition = "hot_data"
            elif age_days <= 180:
                partition = "warm_data"
            else:
                partition = "cold_data"
            
            embedding = embedding_model.encode(doc['text']).tolist()
            
            self.collection.insert(
                [[doc.get('id')], [embedding], [doc['text']], [timestamp]],
                partition_name=partition
            )
    
    def search_with_partition_priority(
        self,
        query_vector: List[float],
        top_k: int = 10,
        **kwargs
    ):
        """
        按分区优先级检索
        
        优先检索热数据，不足时扩展到温数据，再到冷数据
        """
        results = []
        remaining_k = top_k
        
        for partition in ["hot_data", "warm_data", "cold_data"]:
            if remaining_k <= 0:
                break
            
            partition_results = self.collection.search(
                data=[query_vector],
                anns_field="embedding",
                param={"metric_type": "IP", "params": {"nprobe": 16}},
                limit=remaining_k,
                partition_names=[partition],
                **kwargs
            )
            
            results.extend(partition_results[0])
            remaining_k -= len(partition_results[0])
        
        return results
```

### 13.5.2 数据迁移策略

```python
class DataMigration:
    """冷热数据迁移工具"""
    
    def __init__(self, source_collection: Collection, target_collection: Collection = None):
        self.source = source_collection
        self.target = target_collection or source_collection
    
    def migrate_cold_to_hot(self, doc_ids: List[int]):
        """将冷数据迁移到热分区"""
        for doc_id in doc_ids:
            # 从冷分区查询
            doc = self.source.query(
                expr=f'id == {doc_id}',
                partition_names=["cold_data"]
            )
            if doc:
                # 复制到热分区
                self.target.insert(
                    [[d['id']], [d['embedding']], [d['text']], [d['timestamp']]
                     for d in doc],
                    partition_name="hot_data"
                )
                # 从冷分区删除
                self.source.delete(f'id == {doc_id}', partition_name="cold_data")
    
    def auto_migrate_by_age(self, max_hot_days: int = 30, max_warm_days: int = 180):
        """根据时间自动迁移数据"""
        now = datetime.now().timestamp()
        
        # 热数据迁移到温数据
        hot_docs = self.source.query(
            expr=f'timestamp < {now - max_hot_days * 86400}',
            partition_names=["hot_data"]
        )
        self._move_docs(hot_docs, "hot_data", "warm_data")
        
        # 温数据迁移到冷数据
        warm_docs = self.source.query(
            expr=f'timestamp < {now - max_warm_days * 86400}',
            partition_names=["warm_data"]
        )
        self._move_docs(warm_docs, "warm_data", "cold_data")
    
    def _move_docs(self, docs: List[Dict], from_part: str, to_part: str):
        """在分区之间移动文档"""
        if not docs:
            return
        ids = [d['id'] for d in docs]
        self.target.insert(docs, partition_name=to_part)
        for doc_id in ids:
            self.source.delete(f'id == {doc_id}', partition_name=from_part)
```

## 13.6 多版本知识库管理

### 13.6.1 集合级别的版本管理

通过创建不同版本的集合来管理知识库版本。

```python
class CollectionVersionManager:
    """集合版本管理器"""
    
    def __init__(self, milvus_client, base_name: str, dim: int = 768):
        self.client = milvus_client
        self.base_name = base_name
        self.dim = dim
        self.version_collections = {}
    
    def create_version(self, version: str) -> str:
        """创建新版本的知识库集合"""
        collection_name = f"{self.base_name}_v{version}"
        
        schema = CollectionSchema([
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="version", dtype=DataType.VARCHAR, max_length=32),
            FieldSchema(name="created_at", dtype=DataType.INT64),
        ])
        
        collection = Collection(name=collection_name, schema=schema)
        
        index_params = {
            "metric_type": "IP",
            "index_type": "HNSW",
            "params": {"M": 16, "efConstruction": 200}
        }
        collection.create_index("embedding", index_params)
        collection.load()
        
        self.version_collections[version] = collection
        return collection_name
    
    def switch_active_version(self, version: str):
        """切换当前活跃版本"""
        if version not in self.version_collections:
            raise ValueError(f"版本 {version} 不存在")
        
        # 释放旧版本
        for v, col in self.version_collections.items():
            if v != version:
                col.release()
        
        # 加载新版本
        self.version_collections[version].load()
    
    def compare_versions(
        self, version_a: str, version_b: str
    ) -> Dict:
        """比较两个版本的差异"""
        col_a = self.version_collections[version_a]
        col_b = self.version_collections[version_b]
        
        ids_a = set(
            r['id'] for r in col_a.query(expr='id >= 0', output_fields=['id'])
        )
        ids_b = set(
            r['id'] for r in col_b.query(expr='id >= 0', output_fields=['id'])
        )
        
        return {
            "added": list(ids_b - ids_a),
            "removed": list(ids_a - ids_b),
            "common": list(ids_a & ids_b),
            "version_a_count": len(ids_a),
            "version_b_count": len(ids_b)
        }
    
    def rollback_to(self, version: str):
        """回滚到指定版本"""
        self.switch_active_version(version)
```

### 13.6.2 版本标签与发布管理

```python
class VersionReleaseManager:
    """版本发布管理器"""
    
    def __init__(self):
        self.releases = {}  # {tag: version}
        self.release_notes = {}
    
    def tag_release(self, tag: str, version: str, notes: str = ""):
        """为版本打标签"""
        self.releases[tag] = version
        if notes:
            self.release_notes[tag] = {
                "version": version,
                "notes": notes,
                "timestamp": int(datetime.now().timestamp())
            }
        print(f"已发布标签 {tag} -> 版本 {version}")
    
    def list_releases(self) -> List[Dict]:
        """列出所有发布版本"""
        return [
            {
                "tag": tag,
                "version": ver,
                "notes": self.release_notes.get(tag, {}).get("notes", ""),
                "timestamp": self.release_notes.get(tag, {}).get("timestamp", 0)
            }
            for tag, ver in self.releases.items()
        ]
```

## 13.7 综合实践：知识库治理流水线

```python
class KnowledgeBasePipeline:
    """完整的知识库治理流水线"""
    
    def __init__(self, collection: Collection, embedding_model):
        self.collection = collection
        self.embedding_model = embedding_model
        self.chunker = AdaptiveChunker()
        self.cleaner = TextCleaner()
        self.updater = IncrementalUpdater(collection)
        self.partitioner = HotColdPartitioner(collection)
    
    def process_documents(
        self,
        documents: List[Dict],
        batch_size: int = 100
    ) -> Dict:
        """
        完整的文档处理流水线
        
        1. 清洗 -> 2. 切片 -> 3. 去重 -> 4. 向量化 -> 5. 分类插入
        """
        stats = {
            "input": len(documents),
            "cleaned": 0,
            "chunks": 0,
            "inserted": 0,
            "skipped": 0,
            "low_quality": 0
        }
        
        processed_chunks = []
        
        for doc in documents:
            # Step 1: 文本清洗
            cleaned_text = self.cleaner.clean(doc.get('text', ''))
            stats['cleaned'] += 1
            
            # Step 2: 质量检查
            if self.cleaner.is_quality_low(cleaned_text):
                stats['low_quality'] += 1
                continue
            
            # Step 3: 智能切片
            chunks = self.chunker.chunk(cleaned_text)
            stats['chunks'] += len(chunks)
            
            for chunk in chunks:
                processed_chunks.append({
                    'text': chunk['text'],
                    'metadata': {
                        **chunk['metadata'],
                        'source': doc.get('source', ''),
                        'timestamp': doc.get('timestamp', int(datetime.now().timestamp())),
                        'category': doc.get('category', 'default')
                    }
                })
        
        # Step 4: 批量向量化并插入
        for i in range(0, len(processed_chunks), batch_size):
            batch = processed_chunks[i:i+batch_size]
            embeddings = self.embedding_model.encode([b['text'] for b in batch])
            
            for j, chunk in enumerate(batch):
                self.collection.insert([
                    [hash(chunk['text']) % (10**10)],
                    [embeddings[j].tolist()],
                    [chunk['text']],
                    [chunk['metadata']['source']],
                    [chunk['metadata']['timestamp']],
                    [chunk['metadata']['category']]
                ])
                stats['inserted'] += 1
        
        self.collection.flush()
        return stats
```

## 13.8 本章小结

本章深入探讨了基于Milvus的RAG知识库治理策略，涵盖了智能切片、增量更新、数据去重与降噪、分区冷热分离以及多版本知识库管理等核心技术。高效的知识库治理是RAG系统长期稳定运行的基石。建议在实际项目中，根据数据规模、更新频率和业务需求，灵活组合使用这些策略，并建立持续的数据质量监控机制。一个治理良好的知识库，不仅能够提升检索质量，还能大幅降低运维成本。

下一章将探讨高级RAG架构，介绍基于Milvus的进阶落地方案。
