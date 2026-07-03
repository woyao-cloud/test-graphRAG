# 第14章 性能优化

## 14.1 引言

RAG（检索增强生成）系统的性能直接影响用户体验、运营成本和系统可扩展性。在实际生产环境中，RAG系统需要同时满足低延迟、高吞吐量和低成本的要求，这需要从多个维度进行系统性的优化。本章将深入探讨RAG系统的性能优化策略，包括缓存策略、并发执行、Token预算管理、内存优化和监控体系。

性能优化在RAG系统中具有特殊的重要性，因为RAG系统通常涉及多个组件（检索、排序、生成）的串行和并行操作，任何一个环节的瓶颈都会影响整体性能。与传统的搜索系统或单纯的LLM应用不同，RAG系统需要平衡检索质量、生成质量和响应速度三者之间的关系。

### 14.1.1 性能优化的核心目标

RAG系统性能优化的核心目标可以概括为三个维度：

1. **延迟（Latency）**：从用户提交查询到收到响应的时间，通常要求P99延迟在2秒以内
2. **吞吐量（Throughput）**：系统单位时间内能处理的查询数量，通常用QPS（Queries Per Second）衡量
3. **成本（Cost）**：每次查询的运营成本，包括API调用费、计算资源和存储成本

这三个目标相互制约，需要在具体业务场景中找到平衡点。例如，在客服场景中，延迟比成本更重要；而在批量文档处理场景中，吞吐量和成本则更为关键。

### 14.1.2 RAG系统的性能瓶颈分析

典型的RAG流水线包括以下环节，每个环节都可能成为性能瓶颈：

```
用户查询 → 查询嵌入 → 向量检索 → 结果重排序 → 上下文构建 → LLM生成 → 响应
```

各环节的典型延迟分布：

| 环节 | 典型延迟 | 瓶颈类型 | 优化空间 |
|------|---------|---------|---------|
| 查询嵌入 | 50-200ms | IO/计算 | 缓存、量化 |
| 向量检索 | 10-100ms | IO/内存 | 索引优化、近似搜索 |
| 结果重排序 | 20-100ms | 计算 | 模型裁剪、并行化 |
| 上下文构建 | 5-20ms | IO | 缓存、预计算 |
| LLM生成 | 500-5000ms | 计算/IO | 模型选择、流式输出 |
| 后处理 | 5-50ms | 计算 | 简化逻辑 |

## 14.2 缓存策略

缓存是提升RAG系统性能最直接有效的手段之一。合理的缓存策略可以显著减少重复计算和外部API调用，降低延迟和成本。

### 14.2.1 缓存层次结构

RAG系统的缓存可以分为三个层次：

```
┌─────────────────────────────────────┐
│          L1: 应用内存缓存            │  < 1ms, 容量小
├─────────────────────────────────────┤
│          L2: 本地Redis缓存           │  < 5ms, 容量中等
├─────────────────────────────────────┤
│          L3: 分布式缓存/数据库       │  < 20ms, 容量大
└─────────────────────────────────────┘
```

每一层缓存的访问速度和容量成反比，设计时需要在速度和命中率之间取得平衡。

### 14.2.2 查询缓存

查询缓存是最基础的缓存形式，它缓存用户查询及其对应的检索结果。查询缓存的粒度可以分为精确匹配缓存和语义缓存两种。

**精确匹配缓存**适用于缓存完全相同的查询，实现简单、命中判断快。在实际应用中，可以通过查询标准化（去除停用词、统一大小写、纠错）来提高命中率。

```python
import hashlib
import json
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

@dataclass
class QueryCacheEntry:
    """查询缓存条目"""
    query: str
    query_hash: str
    results: List[Dict[str, Any]]
    embedding: Optional[List[float]] = None
    created_at: datetime = None
    ttl: int = 3600  # 默认1小时
    access_count: int = 0
    last_access: datetime = None

class QueryCache:
    """查询缓存管理器"""
    
    def __init__(self, max_size: int = 10000, default_ttl: int = 3600):
        self.cache: Dict[str, QueryCacheEntry] = {}
        self.max_size = max_size
        self.default_ttl = default_ttl
    
    def _normalize_query(self, query: str) -> str:
        """标准化查询文本"""
        # 去除多余空格
        normalized = ' '.join(query.split())
        # 转小写
        normalized = normalized.lower()
        # 去除标点符号
        import re
        normalized = re.sub(r'[^\w\s]', '', normalized)
        return normalized
    
    def _hash_query(self, query: str) -> str:
        """对标准化后的查询进行哈希"""
        normalized = self._normalize_query(query)
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()
    
    def get(self, query: str) -> Optional[List[Dict[str, Any]]]:
        """获取缓存结果"""
        query_hash = self._hash_query(query)
        entry = self.cache.get(query_hash)
        
        if entry is None:
            return None
        
        # 检查TTL
        if self._is_expired(entry):
            del self.cache[query_hash]
            return None
        
        # 更新访问统计
        entry.access_count += 1
        entry.last_access = datetime.now()
        
        return entry.results
    
    def set(self, query: str, results: List[Dict[str, Any]], 
            ttl: Optional[int] = None):
        """设置缓存"""
        # 检查容量，必要时驱逐
        if len(self.cache) >= self.max_size:
            self._evict()
        
        query_hash = self._hash_query(query)
        entry = QueryCacheEntry(
            query=query,
            query_hash=query_hash,
            results=results,
            created_at=datetime.now(),
            ttl=ttl or self.default_ttl
        )
        self.cache[query_hash] = entry
    
    def _is_expired(self, entry: QueryCacheEntry) -> bool:
        """检查缓存是否过期"""
        if entry.ttl < 0:  # 永不过期
            return False
        elapsed = (datetime.now() - entry.created_at).total_seconds()
        return elapsed > entry.ttl
    
    def _evict(self):
        """LRU驱逐策略：驱逐最不活跃的条目"""
        sorted_entries = sorted(
            self.cache.items(),
            key=lambda x: (
                x[1].access_count, 
                x[1].last_access or x[1].created_at
            )
        )
        # 驱逐最不活跃的20%
        evict_count = max(1, len(self.cache) // 5)
        for key, _ in sorted_entries[:evict_count]:
            del self.cache[key]
    
    def clear_expired(self):
        """清理所有过期条目"""
        expired_keys = [
            key for key, entry in self.cache.items()
            if self._is_expired(entry)
        ]
        for key in expired_keys:
            del self.cache[key]
        return len(expired_keys)
    
    def stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "default_ttl": self.default_ttl,
            "expired_count": sum(
                1 for entry in self.cache.values() 
                if self._is_expired(entry)
            )
        }
```

**语义缓存**是一种更高级的缓存策略，它不仅缓存完全相同的查询，还缓存语义相似的查询。语义缓存通过比较查询嵌入向量的余弦相似度来判断是否命中。

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class SemanticQueryCache:
    """语义查询缓存"""
    
    def __init__(self, similarity_threshold: float = 0.95, max_size: int = 5000):
        self.cache: List[Dict[str, Any]] = []
        self.similarity_threshold = similarity_threshold
        self.max_size = max_size
    
    def get(self, query_embedding: np.ndarray) -> Optional[Dict[str, Any]]:
        """通过语义相似度查找缓存"""
        if not self.cache:
            return None
        
        # 提取所有缓存条目的嵌入向量
        cached_embeddings = np.array([
            entry['embedding'] for entry in self.cache
        ])
        
        # 计算相似度
        similarities = cosine_similarity([query_embedding], cached_embeddings)[0]
        
        # 找到最相似的缓存条目
        best_idx = np.argmax(similarities)
        if similarities[best_idx] >= self.similarity_threshold:
            entry = self.cache[best_idx]
            entry['access_count'] = entry.get('access_count', 0) + 1
            return entry
        
        return None
    
    def set(self, query: str, embedding: np.ndarray, 
            results: List[Dict[str, Any]]):
        """添加缓存条目"""
        if len(self.cache) >= self.max_size:
            # 按访问次数排序并驱逐不活跃的条目
            self.cache.sort(key=lambda x: x.get('access_count', 0))
            self.cache = self.cache[len(self.cache)//4:]
        
        self.cache.append({
            'query': query,
            'embedding': embedding,
            'results': results,
            'created_at': datetime.now(),
            'access_count': 0
        })
    
    def invalidate(self, query: str):
        """使特定查询的缓存失效"""
        self.cache = [
            entry for entry in self.cache 
            if entry['query'] != query
        ]
```

### 14.2.3 嵌入缓存

嵌入缓存用于缓存文本的向量表示，避免重复调用嵌入模型。在RAG系统中，文档嵌入通常是一次性计算的，而查询嵌入需要实时计算。因此嵌入缓存主要针对两种情况：重复的查询和频繁出现的文档片段。

```python
import numpy as np
from pathlib import Path
import pickle

class EmbeddingCache:
    """嵌入缓存（内存 + 磁盘两级缓存）"""
    
    def __init__(self, cache_dir: str = ".cache/embeddings", 
                 memory_capacity: int = 10000):
        self.memory_cache: Dict[str, np.ndarray] = {}
        self.memory_capacity = memory_capacity
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载磁盘缓存索引
        self.disk_index_path = self.cache_dir / "index.json"
        self.disk_index: Dict[str, str] = {}
        if self.disk_index_path.exists():
            with open(self.disk_index_path, 'r') as f:
                self.disk_index = json.load(f)
    
    def get(self, text: str) -> Optional[np.ndarray]:
        """获取嵌入向量"""
        text_hash = self._hash_text(text)
        
        # 1. 查内存缓存
        if text_hash in self.memory_cache:
            return self.memory_cache[text_hash]
        
        # 2. 查磁盘缓存
        if text_hash in self.disk_index:
            cache_path = self.cache_dir / self.disk_index[text_hash]
            if cache_path.exists():
                embedding = np.load(cache_path)
                # 同时加载到内存
                self._add_to_memory(text_hash, embedding)
                return embedding
        
        return None
    
    def set(self, text: str, embedding: np.ndarray):
        """存储嵌入向量"""
        text_hash = self._hash_text(text)
        
        # 存储到内存
        self._add_to_memory(text_hash, embedding)
        
        # 存储到磁盘
        filename = f"{text_hash}.npy"
        cache_path = self.cache_dir / filename
        np.save(cache_path, embedding)
        
        # 更新索引
        self.disk_index[text_hash] = filename
        self._save_index()
    
    def _add_to_memory(self, text_hash: str, embedding: np.ndarray):
        """添加到内存缓存，并管理容量"""
        if len(self.memory_cache) >= self.memory_capacity:
            # 简单策略：清空一半
            keys = list(self.memory_cache.keys())
            for key in keys[:len(keys)//2]:
                del self.memory_cache[key]
        
        self.memory_cache[text_hash] = embedding
    
    def _hash_text(self, text: str) -> str:
        """对文本进行哈希"""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]
    
    def _save_index(self):
        """保存磁盘索引"""
        with open(self.disk_index_path, 'w') as f:
            json.dump(self.disk_index, f)
    
    def clear(self):
        """清空所有缓存"""
        self.memory_cache.clear()
        self.disk_index = {}
        self._save_index()
        
        # 删除所有缓存文件
        for file in self.cache_dir.glob("*.npy"):
            file.unlink()
```

### 14.2.4 文档缓存

文档缓存用于缓存从文档存储中检索到的文档内容。不同来源的文档有不同的更新频率，因此需要分层TTL（Time-To-Live）策略。

```python
class DocumentCache:
    """文档缓存，支持分层TTL策略"""
    
    def __init__(self, redis_client=None):
        self.redis = redis_client
        
        # 不同文档类型的TTL配置（秒）
        self.ttl_config = {
            'static': -1,          # 静态文档，永不过期
            'daily': 86400,        # 每日更新
            'hourly': 3600,        # 每小时更新
            'realtime': 300,       # 5分钟更新
            'user_generated': 600  # 用户生成内容，10分钟
        }
        
        # 内存缓存（当Redis不可用时使用）
        self.memory_cache: Dict[str, Dict[str, Any]] = {}
    
    def get(self, doc_id: str, doc_type: str = 'static') -> Optional[Dict[str, Any]]:
        """获取文档缓存"""
        cache_key = f"doc:{doc_type}:{doc_id}"
        
        # 先查Redis
        if self.redis:
            data = self.redis.get(cache_key)
            if data:
                return json.loads(data)
        
        # 再查内存缓存
        entry = self.memory_cache.get(cache_key)
        if entry and not self._is_expired(entry):
            return entry['content']
        
        return None
    
    def set(self, doc_id: str, content: Dict[str, Any], 
            doc_type: str = 'static'):
        """设置文档缓存"""
        cache_key = f"doc:{doc_type}:{doc_id}"
        ttl = self.ttl_config.get(doc_type, 3600)
        
        # 存储到Redis
        if self.redis:
            serialized = json.dumps(content, ensure_ascii=False)
            if ttl > 0:
                self.redis.setex(cache_key, ttl, serialized)
            else:
                self.redis.set(cache_key, serialized)
        
        # 同时存储到内存
        self.memory_cache[cache_key] = {
            'content': content,
            'created_at': datetime.now(),
            'ttl': ttl
        }
    
    def _is_expired(self, entry: Dict[str, Any]) -> bool:
        """检查内存缓存是否过期"""
        if entry['ttl'] < 0:
            return False
        elapsed = (datetime.now() - entry['created_at']).total_seconds()
        return elapsed > entry['ttl']
    
    def invalidate(self, doc_id: str, doc_type: str = 'static'):
        """使特定文档的缓存失效"""
        cache_key = f"doc:{doc_type}:{doc_id}"
        
        if self.redis:
            self.redis.delete(cache_key)
        
        self.memory_cache.pop(cache_key, None)
    
    def invalidate_by_type(self, doc_type: str):
        """使特定类型的所有文档缓存失效"""
        pattern = f"doc:{doc_type}:*"
        
        if self.redis:
            keys = self.redis.keys(pattern)
            if keys:
                self.redis.delete(*keys)
        
        # 清除内存缓存
        self.memory_cache = {
            k: v for k, v in self.memory_cache.items()
            if not k.startswith(f"doc:{doc_type}:")
        }
```

### 14.2.5 缓存预热策略

缓存预热是在系统启动或数据更新时，主动将高频数据加载到缓存中。合理的预热策略可以显著减少冷启动阶段的延迟。

```python
class CacheWarmer:
    """缓存预热器"""
    
    def __init__(self, query_cache: QueryCache, 
                 embedding_cache: EmbeddingCache,
                 document_cache: DocumentCache):
        self.query_cache = query_cache
        self.embedding_cache = embedding_cache
        self.document_cache = document_cache
        self.is_warmed = False
    
    def warm_from_logs(self, log_path: str, top_k: int = 1000):
        """从历史查询日志中预热缓存"""
        logger.info(f"Starting cache warmup from logs: {log_path}")
        
        # 1. 分析历史查询日志
        hot_queries = self._analyze_query_logs(log_path, top_k)
        logger.info(f"Found {len(hot_queries)} hot queries")
        
        # 2. 预计算并缓存热门查询的嵌入
        for query_data in hot_queries:
            query = query_data['query']
            if not self.query_cache.get(query):
                # 执行查询并缓存结果
                results = self._simulate_query(query)
                self.query_cache.set(query, results, ttl=3600)
        
        self.is_warmed = True
        logger.info("Cache warmup completed")
    
    def warm_from_documents(self, documents: List[Dict], embed_func):
        """预热文档嵌入缓存"""
        for doc in documents:
            text = doc.get('text', '')
            if text and not self.embedding_cache.get(text):
                embedding = embed_func([text])[0]
                self.embedding_cache.set(text, embedding)
    
    def _analyze_query_logs(self, log_path: str, top_k: int) -> List[Dict]:
        """分析查询日志，找出高频查询"""
        from collections import Counter
        
        query_counter = Counter()
        
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    log_entry = json.loads(line)
                    query = log_entry.get('query', '')
                    if query:
                        query_counter[query] += 1
                except json.JSONDecodeError:
                    continue
        
        # 返回频率最高的top_k个查询
        return [
            {'query': q, 'frequency': c}
            for q, c in query_counter.most_common(top_k)
        ]
    
    def _simulate_query(self, query: str) -> List[Dict]:
        """模拟查询（实际实现应调用检索接口）"""
        # 这里只是占位，实际实现会调用检索服务
        return []
```

### 14.2.6 缓存一致性策略

当底层数据更新时，缓存中的数据可能变得过时。需要制定合适的缓存一致性策略：

```python
class CacheConsistencyManager:
    """缓存一致性管理器"""
    
    def __init__(self, query_cache, document_cache):
        self.query_cache = query_cache
        self.document_cache = document_cache
    
    def on_document_update(self, doc_id: str, doc_type: str = 'static'):
        """文档更新时的缓存处理"""
        # 1. 使文档缓存失效
        self.document_cache.invalidate(doc_id, doc_type)
        
        # 2. 对于静态文档，还需要使包含该文档的查询缓存失效
        if doc_type == 'static':
            self._invalidate_affected_queries(doc_id)
    
    def on_batch_update(self, doc_ids: List[str], doc_type: str):
        """批量更新时的缓存处理"""
        for doc_id in doc_ids:
            self.on_document_update(doc_id, doc_type)
    
    def _invalidate_affected_queries(self, doc_id: str):
        """使包含特定文档的查询缓存失效"""
        # 简化实现：记录文档到查询的映射关系
        # 实际应用中需要更复杂的追踪机制
        pass
```

## 14.3 并发执行

并发执行可以充分利用系统资源，减少总响应时间。RAG系统的多个环节（检索、排序、生成）可以并行或流水线执行。

### 14.3.1 异步处理

使用异步IO来并发处理多个独立任务。Python的`asyncio`是实现异步RAG流水线的核心工具。

```python
import asyncio
import time
from typing import List, Dict, Any, Callable, Optional

class AsyncPipeline:
    """异步RAG流水线"""
    
    def __init__(self, max_concurrency: int = 10):
        self.semaphore = asyncio.Semaphore(max_concurrency)
    
    async def parallel_retrieve(self, 
                                queries: List[str], 
                                retriever_func: Callable) -> List[Any]:
        """并行检索多个查询"""
        async def retrieve_one(query: str):
            async with self.semaphore:
                return await retriever_func(query)
        
        tasks = [retrieve_one(q) for q in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理异常
        processed_results = []
        for r in results:
            if isinstance(r, Exception):
                logger.error(f"Retrieval failed: {r}")
                processed_results.append([])
            else:
                processed_results.append(r)
        
        return processed_results
    
    async def parallel_rerank(self, 
                               documents: List[Dict], 
                               rerank_func: Callable) -> List[float]:
        """并行重排序"""
        async def rerank_one(doc: Dict):
            async with self.semaphore:
                return await rerank_func(doc)
        
        tasks = [rerank_one(doc) for doc in documents]
        scores = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理异常
        valid_scores = []
        for s in scores:
            if isinstance(s, Exception):
                valid_scores.append(0.0)
            else:
                valid_scores.append(s)
        
        return valid_scores
    
    async def rag_pipeline(self, query: str) -> Dict[str, Any]:
        """完整的异步RAG流水线"""
        start = time.time()
        
        # 并行执行多种检索策略
        retrieval_tasks = [
            self._vector_search(query),
            self._keyword_search(query),
            self._graph_search(query)
        ]
        
        vector_results, keyword_results, graph_results = await asyncio.gather(
            *retrieval_tasks, return_exceptions=True
        )
        
        # 处理检索异常
        all_results = []
        for results in [vector_results, keyword_results, graph_results]:
            if not isinstance(results, Exception):
                all_results.extend(results)
        
        if not all_results:
            return {
                'answer': '抱歉，未能找到相关信息。',
                'sources': [],
                'latency': time.time() - start
            }
        
        # 去重
        seen_ids = set()
        unique_results = []
        for doc in all_results:
            if doc.get('id') not in seen_ids:
                seen_ids.add(doc.get('id'))
                unique_results.append(doc)
        
        # 并行重排序
        rerank_scores = await self.parallel_rerank(
            unique_results, self._rerank
        )
        
        # 按分数排序
        ranked = [
            doc for _, doc in sorted(
                zip(rerank_scores, unique_results),
                key=lambda x: x[0],
                reverse=True
            )
        ]
        
        # 取Top-K并生成答案
        top_docs = ranked[:5]
        answer = await self._generate(query, top_docs)
        
        return {
            'answer': answer,
            'sources': [d.get('source') for d in top_docs],
            'latency': time.time() - start,
            'breakdown': {
                'retrieval_count': len(all_results),
                'rerank_count': len(unique_results)
            }
        }
    
    async def _vector_search(self, query: str) -> List[Dict]:
        """向量检索（模拟）"""
        await asyncio.sleep(0.1)
        return []
    
    async def _keyword_search(self, query: str) -> List[Dict]:
        """关键词检索（模拟）"""
        await asyncio.sleep(0.05)
        return []
    
    async def _graph_search(self, query: str) -> List[Dict]:
        """图检索（模拟）"""
        await asyncio.sleep(0.08)
        return []
    
    async def _rerank(self, doc: Dict) -> float:
        """重排序（模拟）"""
        await asyncio.sleep(0.01)
        return doc.get('score', 0.5)
    
    async def _generate(self, query: str, docs: List[Dict]) -> str:
        """生成答案（模拟）"""
        await asyncio.sleep(0.5)
        return f"基于{len(docs)}个文档生成的答案"
```

### 14.3.2 线程池

对于CPU密集型和IO密集型混合的任务，使用线程池管理并发。线程池适合处理阻塞IO操作（如文件读写、数据库查询）和轻量级计算任务。

```python
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from threading import Lock, local
import queue

class ThreadPoolPipeline:
    """线程池流水线"""
    
    def __init__(self, io_workers: int = 8, cpu_workers: int = 4):
        # IO密集型任务使用更多线程
        self.io_executor = ThreadPoolExecutor(
            max_workers=io_workers,
            thread_name_prefix="io-worker"
        )
        # CPU密集型任务使用进程池
        self.cpu_executor = ProcessPoolExecutor(
            max_workers=cpu_workers
        )
        
        self.stats_lock = Lock()
        self.stats = {
            'total_tasks': 0,
            'completed': 0,
            'failed': 0,
            'total_time': 0.0
        }
        
        # 线程本地存储
        self.thread_local = local()
    
    def batch_process_documents(self, documents: List[Dict]) -> List[Dict]:
        """批量处理文档（IO密集型）"""
        futures = []
        for doc in documents:
            future = self.io_executor.submit(self._process_document, doc)
            futures.append(future)
        
        results = []
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                with self.stats_lock:
                    self.stats['completed'] += 1
            except Exception as e:
                logger.error(f"Document processing failed: {e}")
                with self.stats_lock:
                    self.stats['failed'] += 1
        
        return results
    
    def batch_compute_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """批量计算嵌入（CPU密集型）"""
        futures = []
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            future = self.cpu_executor.submit(self._compute_embeddings, batch)
            futures.append(future)
        
        all_embeddings = []
        for future in as_completed(futures):
            try:
                embeddings = future.result()
                all_embeddings.extend(embeddings)
            except Exception as e:
                logger.error(f"Embedding computation failed: {e}")
        
        return all_embeddings
    
    def _process_document(self, doc: Dict) -> Dict:
        """处理单个文档"""
        # 文档分块
        chunks = self._chunk_document(doc['text'])
        
        # 提取元数据
        metadata = doc.get('metadata', {})
        
        return {
            'doc_id': doc.get('id'),
            'chunks': chunks,
            'metadata': metadata,
            'chunk_count': len(chunks)
        }
    
    def _chunk_document(self, text: str, chunk_size: int = 512) -> List[str]:
        """将文档分块"""
        chunks = []
        for i in range(0, len(text), chunk_size):
            chunks.append(text[i:i + chunk_size])
        return chunks
    
    def _compute_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """计算嵌入向量"""
        # 实际实现会调用嵌入模型
        return [np.random.rand(768) for _ in texts]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取流水线统计信息"""
        with self.stats_lock:
            return dict(self.stats)
```

### 14.3.3 连接池管理

管理外部服务（数据库、API、消息队列）的连接池，减少连接建立的开销。

```python
import redis
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool
import aiohttp

class ConnectionPoolManager:
    """连接池管理器"""
    
    def __init__(self):
        # Redis连接池
        self.redis_pool = redis.ConnectionPool(
            host='localhost',
            port=6379,
            db=0,
            max_connections=50,
            socket_keepalive=True,
            socket_connect_timeout=5,
            retry_on_timeout=True,
            health_check_interval=30
        )
        
        # 数据库连接池
        self.db_engine = create_engine(
            'postgresql://user:pass@localhost/db',
            poolclass=QueuePool,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            pool_recycle=3600,
            pool_timeout=30
        )
        
        # HTTP连接池（用于调用外部API）
        self.http_session = None
        self.http_connector = aiohttp.TCPConnector(
            limit=20,
            limit_per_host=10,
            ttl_dns_cache=300,
            enable_cleanup_closed=True
        )
    
    @property
    def redis_client(self):
        """获取Redis客户端"""
        return redis.Redis(connection_pool=self.redis_pool)
    
    @property
    def db_session(self):
        """获取数据库会话"""
        return self.db_engine.connect()
    
    async def get_http_session(self) -> aiohttp.ClientSession:
        """获取HTTP会话"""
        if self.http_session is None or self.http_session.closed:
            self.http_session = aiohttp.ClientSession(
                connector=self.http_connector,
                timeout=aiohttp.ClientTimeout(total=30)
            )
        return self.http_session
    
    async def close(self):
        """关闭所有连接"""
        if self.http_session and not self.http_session.closed:
            await self.http_session.close()
        self.redis_pool.disconnect()
        self.db_engine.dispose()
```

### 14.3.4 流水线并行化

除了任务级别的并行，还可以实现流水线级别的并行化。流水线并行将RAG流程分解为多个阶段，每个阶段由独立的处理器执行，数据在阶段之间通过队列传递。

```python
import asyncio
from asyncio import Queue

class PipelineStage:
    """流水线阶段基类"""
    
    def __init__(self, name: str, num_workers: int = 1):
        self.name = name
        self.num_workers = num_workers
        self.input_queue: Queue = Queue()
        self.output_queue: Queue = Queue()
        self.workers = []
    
    async def start(self):
        """启动工作协程"""
        for i in range(self.num_workers):
            worker = asyncio.create_task(self._worker_loop(i))
            self.workers.append(worker)
    
    async def _worker_loop(self, worker_id: int):
        """工作协程主循环"""
        while True:
            item = await self.input_queue.get()
            if item is None:  # 停止信号
                self.input_queue.task_done()
                break
            
            try:
                result = await self.process(item)
                await self.output_queue.put(result)
            except Exception as e:
                logger.error(f"Stage {self.name} worker {worker_id} error: {e}")
                await self.output_queue.put(None)
            finally:
                self.input_queue.task_done()
    
    async def process(self, item: Any) -> Any:
        """处理单个项目（子类实现）"""
        raise NotImplementedError
    
    async def stop(self):
        """停止所有工作协程"""
        for _ in range(self.num_workers):
            await self.input_queue.put(None)

class PipelinedRAG:
    """流水线并行RAG"""
    
    def __init__(self):
        # 定义流水线阶段
        self.stages = [
            PipelineStage("retrieval", num_workers=3),
            PipelineStage("rerank", num_workers=2),
            PipelineStage("generation", num_workers=1)
        ]
        
        # 连接阶段
        for i in range(len(self.stages) - 1):
            self.stages[i].output_queue = self.stages[i + 1].input_queue
    
    async def process_batch(self, queries: List[str]) -> List[Dict]:
        """批量处理查询（流水线模式）"""
        # 启动所有阶段
        for stage in self.stages:
            await stage.start()
        
        # 提交所有查询到第一个阶段
        for query in queries:
            await self.stages[0].input_queue.put(query)
        
        # 等待处理完成
        await self.stages[0].input_queue.join()
        
        # 停止所有阶段
        for stage in self.stages:
            await stage.stop()
        
        # 收集结果
        results = []
        while not self.stages[-1].output_queue.empty():
            result = await self.stages[-1].output_queue.get()
            if result is not None:
                results.append(result)
        
        return results
```

## 14.4 Token预算管理

Token预算管理是控制成本和延迟的关键策略。在RAG系统中，Token消耗主要集中在检索到的文档上下文和LLM生成部分。

### 14.4.1 Token计数

首先需要实现精确的Token计数功能：

```python
import tiktoken

class TokenCounter:
    """Token计数器"""
    
    # 常见模型的编码器映射
    MODEL_ENCODINGS = {
        'gpt-4': 'cl100k_base',
        'gpt-4-turbo': 'cl100k_base',
        'gpt-4o': 'o200k_base',
        'gpt-4o-mini': 'o200k_base',
        'gpt-3.5-turbo': 'cl100k_base',
        'text-embedding-3-small': 'cl100k_base',
        'text-embedding-3-large': 'cl100k_base',
        'deepseek-chat': 'cl100k_base',  # DeepSeek兼容
        'deepseek-coder': 'cl100k_base',
    }
    
    def __init__(self, model: str = 'gpt-4'):
        self.model = model
        encoding_name = self.MODEL_ENCODINGS.get(model, 'cl100k_base')
        self.encoder = tiktoken.get_encoding(encoding_name)
    
    def count(self, text: str) -> int:
        """计算文本的token数"""
        return len(self.encoder.encode(text))
    
    def count_messages(self, messages: List[Dict]) -> int:
        """计算消息列表的token数（包含格式开销）"""
        total = 0
        for message in messages:
            # 每条消息的格式开销
            total += 4  # <im_start>role\ncontent<im_end>
            for key, value in message.items():
                total += self.count(str(value))
                if key == 'name':
                    total -= 1  # name的偏移修正
        total += 2  # <im_start>assistant
        return total
    
    def truncate(self, text: str, max_tokens: int) -> str:
        """将文本截断到指定token数"""
        tokens = self.encoder.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self.encoder.decode(tokens[:max_tokens])
```

### 14.4.2 Token预算分配

```python
class TokenBudgetManager:
    """Token预算管理器"""
    
    def __init__(self, model: str = "gpt-4", max_total_tokens: int = 8000):
        self.counter = TokenCounter(model)
        self.max_total_tokens = max_total_tokens
        
        # 默认预算分配比例
        self.budget_allocation = {
            'system_prompt': 500,        # 系统提示词
            'retrieved_context': 4000,    # 检索到的上下文
            'chat_history': 1000,         # 对话历史
            'query': 500,                 # 当前查询
            'generation': 1500,           # 留给生成的token
            'buffer': 500                 # 安全缓冲
        }
        
        # 验证预算分配是否超过最大限制
        total_budget = sum(self.budget_allocation.values())
        assert total_budget <= max_total_tokens, \
            f"Budget allocation {total_budget} exceeds max {max_total_tokens}"
    
    def optimize_context_window(self, 
                                 query: str,
                                 documents: List[Dict],
                                 chat_history: Optional[List[Dict]] = None) -> Dict:
        """优化上下文窗口分配"""
        result = {
            'system_prompt': '',
            'retrieved_context': [],
            'chat_history': [],
            'query': query,
            'budget_usage': {}
        }
        
        # 1. 分配系统提示词
        result['budget_usage']['system_prompt'] = self.budget_allocation['system_prompt']
        
        # 2. 分配对话历史
        if chat_history:
            history_budget = self.budget_allocation['chat_history']
            truncated_history = self._truncate_chat_history(
                chat_history, history_budget
            )
            result['chat_history'] = truncated_history
            result['budget_usage']['chat_history'] = self.counter.count(
                str(truncated_history)
            )
        else:
            # 没有对话历史，将预算转给检索上下文
            self.budget_allocation['retrieved_context'] += \
                self.budget_allocation['chat_history']
        
        # 3. 分配检索上下文
        context_budget = self.budget_allocation['retrieved_context']
        selected_docs = self._allocate_context_budget(documents, context_budget)
        result['retrieved_context'] = selected_docs
        result['budget_usage']['retrieved_context'] = sum(
            doc['token_count'] for doc in selected_docs
        )
        
        # 4. 记录查询token数
        result['budget_usage']['query'] = self.counter.count(query)
        
        # 5. 计算剩余生成预算
        used = sum(result['budget_usage'].values())
        remaining = self.max_total_tokens - used
        result['budget_usage']['generation'] = min(
            remaining, self.budget_allocation['generation']
        )
        
        result['total_usage'] = used + result['budget_usage']['generation']
        
        return result
    
    def _allocate_context_budget(self, documents: List[Dict], 
                                  budget: int) -> List[Dict]:
        """在文档之间智能分配上下文预算"""
        if not documents:
            return []
        
        # 计算每个文档的token数
        for doc in documents:
            doc['token_count'] = self.counter.count(doc.get('text', ''))
        
        # 按相关性分数排序（降序）
        sorted_docs = sorted(
            documents, 
            key=lambda x: x.get('score', 0), 
            reverse=True
        )
        
        selected = []
        remaining_budget = budget
        
        for doc in sorted_docs:
            doc_tokens = doc['token_count']
            
            # 至少保留50个token给每个文档
            if doc_tokens < 50:
                continue
            
            # 如果预算不足且还没有选择任何文档，强制选择第一个
            if doc_tokens > remaining_budget and selected:
                break
            
            # 实际分配的token数
            allocated = min(doc_tokens, remaining_budget)
            
            truncated_text = self.counter.truncate(
                doc.get('text', ''), allocated
            )
            
            selected.append({
                **doc,
                'text': truncated_text,
                'original_tokens': doc_tokens,
                'allocated_tokens': allocated
            })
            
            remaining_budget -= allocated
            
            if remaining_budget < 100:  # 剩余预算太少，停止
                break
        
        return selected
    
    def _truncate_chat_history(self, history: List[Dict], 
                                budget: int) -> List[Dict]:
        """截断对话历史"""
        total_tokens = 0
        truncated = []
        
        for message in reversed(history):  # 从最新的消息开始保留
            msg_tokens = self.counter.count(str(message))
            if total_tokens + msg_tokens > budget:
                break
            total_tokens += msg_tokens
            truncated.insert(0, message)
        
        return truncated
```

### 14.4.3 动态预算调整

```python
class DynamicBudgetAdjuster:
    """动态预算调整器"""
    
    def __init__(self, initial_budget: Dict[str, int]):
        self.budget = initial_budget.copy()
        self.adjustment_factor = 0.1
        self.performance_history = []
    
    def adjust(self, performance_metrics: Dict[str, float]):
        """根据性能指标动态调整预算"""
        self.performance_history.append(performance_metrics)
        
        # 如果生成质量下降，增加生成预算
        gen_quality = performance_metrics.get('generation_quality', 1.0)
        if gen_quality < 0.8:
            transfer = int(self.budget['retrieved_context'] * self.adjustment_factor)
            self.budget['retrieved_context'] -= transfer
            self.budget['generation'] += transfer
        
        # 如果检索召回率低，增加检索预算
        recall = performance_metrics.get('recall', 1.0)
        if recall < 0.9:
            transfer = int(self.budget['generation'] * self.adjustment_factor)
            self.budget['generation'] -= transfer
            self.budget['retrieved_context'] += transfer
        
        # 如果上下文利用率低，减少上下文预算
        context_util = performance_metrics.get('context_utilization', 1.0)
        if context_util < 0.5:
            self.budget['retrieved_context'] = int(
                self.budget['retrieved_context'] * 0.8
            )
        
        return self.budget
    
    def get_current_budget(self) -> Dict[str, int]:
        """获取当前预算配置"""
        return self.budget.copy()
```

### 14.4.4 Token使用监控

```python
class TokenUsageMonitor:
    """Token使用监控器"""
    
    def __init__(self):
        self.usage_records = []
        self.daily_usage = defaultdict(lambda: defaultdict(int))
    
    def record(self, query: str, model: str, 
               input_tokens: int, output_tokens: int,
               stage: str = 'generation'):
        """记录Token使用情况"""
        record = {
            'timestamp': datetime.now(),
            'query_length': len(query),
            'model': model,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': input_tokens + output_tokens,
            'stage': stage
        }
        self.usage_records.append(record)
        
        # 更新日统计
        date = datetime.now().strftime('%Y-%m-%d')
        self.daily_usage[date]['total_input'] += input_tokens
        self.daily_usage[date]['total_output'] += output_tokens
        self.daily_usage[date]['query_count'] += 1
    
    def get_daily_stats(self, date: Optional[str] = None) -> Dict:
        """获取每日统计"""
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        usage = self.daily_usage.get(date, {})
        return {
            'date': date,
            'total_input_tokens': usage.get('total_input', 0),
            'total_output_tokens': usage.get('total_output', 0),
            'total_tokens': usage.get('total_input', 0) + usage.get('total_output', 0),
            'query_count': usage.get('query_count', 0),
            'avg_tokens_per_query': (
                (usage.get('total_input', 0) + usage.get('total_output', 0)) /
                max(usage.get('query_count', 1), 1)
            )
        }
    
    def get_cost_estimate(self, pricing: Dict[str, float]) -> float:
        """估算成本"""
        total_cost = 0.0
        for record in self.usage_records:
            model = record['model']
            model_pricing = pricing.get(model, {})
            input_cost = model_pricing.get('input', 0) * record['input_tokens'] / 1000
            output_cost = model_pricing.get('output', 0) * record['output_tokens'] / 1000
            total_cost += input_cost + output_cost
        return total_cost
```

## 14.5 内存优化

### 14.5.1 嵌入向量量化

通过降低数值精度来减少模型的内存占用。常见的量化方案包括int8、int4和二进制量化。

```python
import numpy as np
from typing import Tuple

class EmbeddingQuantizer:
    """嵌入向量量化器"""
    
    def __init__(self, bits: int = 8):
        assert bits in [1, 2, 4, 8], "仅支持1、2、4、8位量化"
        self.bits = bits
        self.max_int = 2 ** (bits - 1) - 1 if bits > 1 else 0
        self.min_int = -(2 ** (bits - 1)) if bits > 1 else 0
        
        # 量化参数（每个维度）
        self.scales: np.ndarray = None
        self.zero_points: np.ndarray = None
    
    def quantize(self, embeddings: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """量化嵌入向量"""
        if self.bits == 1:
            return self._binary_quantize(embeddings)
        elif self.bits == 8:
            return self._int8_quantize(embeddings)
        else:
            return self._general_quantize(embeddings)
    
    def _int8_quantize(self, embeddings: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """int8量化（逐维度）"""
        orig_dtype = embeddings.dtype
        shape = embeddings.shape
        
        # 计算每列的最小值和最大值
        min_vals = embeddings.min(axis=0)
        max_vals = embeddings.max(axis=0)
        
        # 计算缩放因子和零点
        scales = (max_vals - min_vals) / (self.max_int - self.min_int)
        scales = np.where(scales == 0, 1.0, scales)  # 避免除零
        zero_points = np.round(-min_vals / scales) + self.min_int
        
        # 量化
        quantized = np.round(embeddings / scales - zero_points + self.min_int)
        quantized = np.clip(quantized, self.min_int, self.max_int).astype(np.int8)
        
        params = {
            'scales': scales,
            'zero_points': zero_points.astype(np.int8),
            'orig_dtype': str(orig_dtype),
            'shape': shape
        }
        
        return quantized, params
    
    def _binary_quantize(self, embeddings: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """二进制量化（1位）"""
        # 使用符号函数
        quantized = np.where(embeddings > 0, 1, -1).astype(np.int8)
        
        # 压缩为bit-packed格式
        # 将8个1位值打包到一个字节中
        packed = np.packbits(quantized.reshape(-1) > 0)
        
        params = {
            'orig_shape': embeddings.shape,
            'orig_dtype': str(embeddings.dtype),
            'packed': True
        }
        
        return packed, params
    
    def dequantize(self, quantized: np.ndarray, params: Dict) -> np.ndarray:
        """反量化"""
        if self.bits == 1:
            return self._binary_dequantize(quantized, params)
        
        scales = params['scales']
        zero_points = params['zero_points']
        
        dequantized = (quantized.astype(np.float32) - self.min_int + zero_points) * scales
        return dequantized
    
    def _binary_dequantize(self, quantized: np.ndarray, 
                            params: Dict) -> np.ndarray:
        """二进制反量化"""
        unpacked = np.unpackbits(quantized)[:np.prod(params['orig_shape'])]
        unpacked = unpacked.astype(np.float32) * 2 - 1  # 0->-1, 1->1
        return unpacked.reshape(params['orig_shape'])
    
    def compression_ratio(self, original: np.ndarray) -> float:
        """计算压缩比"""
        original_bytes = original.nbytes
        quantized, params = self.quantize(original)
        
        if self.bits == 1:
            quantized_bytes = quantized.nbytes
        else:
            quantized_bytes = quantized.nbytes + \
                              params['scales'].nbytes + \
                              params['zero_points'].nbytes
        
        return original_bytes / quantized_bytes
```

### 14.5.2 产品量化（PQ）

产品量化是一种更高级的量化技术，它将高维向量分割成多个子空间，在每个子空间独立进行量化。

```python
from sklearn.cluster import KMeans

class ProductQuantizer:
    """产品量化器"""
    
    def __init__(self, num_subvectors: int = 8, num_centroids: int = 256):
        self.num_subvectors = num_subvectors
        self.num_centroids = num_centroids
        self.codebooks = []  # 每个子空间的码本
        self.subvector_dim = None
    
    def fit(self, embeddings: np.ndarray):
        """训练量化器"""
        n, d = embeddings.shape
        assert d % self.num_subvectors == 0, \
            f"维度{d}不能被{self.num_subvectors}整除"
        
        self.subvector_dim = d // self.num_subvectors
        
        # 对每个子空间进行KMeans聚类
        for i in range(self.num_subvectors):
            sub_vectors = embeddings[:, i * self.subvector_dim:(i + 1) * self.subvector_dim]
            kmeans = KMeans(
                n_clusters=self.num_centroids,
                random_state=42,
                n_init=1
            )
            kmeans.fit(sub_vectors)
            self.codebooks.append(kmeans.cluster_centers_)
    
    def encode(self, embeddings: np.ndarray) -> np.ndarray:
        """编码为压缩表示"""
        n = embeddings.shape[0]
        codes = np.zeros((n, self.num_subvectors), dtype=np.uint8)
        
        for i in range(self.num_subvectors):
            sub_vectors = embeddings[:, i * self.subvector_dim:(i + 1) * self.subvector_dim]
            # 找到最近的质心
            distances = np.linalg.norm(
                sub_vectors[:, np.newaxis] - self.codebooks[i][np.newaxis, :],
                axis=2
            )
            codes[:, i] = np.argmin(distances, axis=1)
        
        return codes
    
    def decode(self, codes: np.ndarray) -> np.ndarray:
        """解码回近似向量"""
        n = codes.shape[0]
        embeddings = np.zeros((n, self.num_subvectors * self.subvector_dim))
        
        for i in range(self.num_subvectors):
            embeddings[:, i * self.subvector_dim:(i + 1) * self.subvector_dim] = \
                self.codebooks[i][codes[:, i]]
        
        return embeddings
    
    def compression_ratio(self, original_dim: int) -> float:
        """计算压缩比"""
        original_bytes = original_dim * 4  # float32
        compressed_bytes = self.num_subvectors  # uint8 codes
        codebook_bytes = self.num_subvectors * self.num_centroids * \
                        self.subvector_dim * 4
        # 码本开销平摊到每个向量
        amortized = codebook_bytes / 10000  # 假设10000个向量
        return original_bytes / (compressed_bytes + amortized)
```

### 14.5.3 内存映射文件

使用内存映射文件处理大规模向量索引，避免将所有数据加载到内存中。内存映射文件允许操作系统按需加载页面，对于大型索引可以显著减少内存占用。

```python
import numpy as np
from pathlib import Path
import mmap

class MmapVectorIndex:
    """使用内存映射的向量索引"""
    
    def __init__(self, index_path: str, dimension: int, dtype=np.float32):
        self.index_path = Path(index_path)
        self.dimension = dimension
        self.dtype = dtype
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 初始化内存映射文件
        self._initialize_mmap()
    
    def _initialize_mmap(self):
        """初始化内存映射文件"""
        if not self.index_path.exists():
            # 初始分配
            initial_vectors = 10000
            shape = (initial_vectors, self.dimension)
            self.mmap = np.memmap(
                str(self.index_path), dtype=self.dtype, 
                mode='w+', shape=shape
            )
            self.size = 0
            self.capacity = initial_vectors
        else:
            # 打开现有文件
            file_size = self.index_path.stat().st_size
            bytes_per_vector = self.dimension * np.dtype(self.dtype).itemsize
            num_vectors = file_size // bytes_per_vector
            shape = (num_vectors, self.dimension)
            self.mmap = np.memmap(
                str(self.index_path), dtype=self.dtype, 
                mode='r+', shape=shape
            )
            self.size = num_vectors
            self.capacity = num_vectors
    
    def add_vectors(self, vectors: np.ndarray):
        """添加向量"""
        n = len(vectors)
        if self.size + n > self.capacity:
            # 扩展文件（翻倍）
            new_capacity = max(self.capacity * 2, self.size + n)
            self._resize(new_capacity)
        
        self.mmap[self.size:self.size + n] = vectors
        self.size += n
        self.mmap.flush()
    
    def get_vectors(self, indices: np.ndarray) -> np.ndarray:
        """获取指定索引的向量"""
        return self.mmap[indices].copy()
    
    def get_all_vectors(self) -> np.ndarray:
        """获取所有向量"""
        return self.mmap[:self.size].copy()
    
    def _resize(self, new_capacity: int):
        """调整容量"""
        # 保存现有数据
        data = self.mmap[:self.size].copy()
        
        # 重新创建更大的文件
        new_shape = (new_capacity, self.dimension)
        self.mmap = np.memmap(
            str(self.index_path), dtype=self.dtype, 
            mode='w+', shape=new_shape
        )
        self.mmap[:self.size] = data
        self.capacity = new_capacity
        self.mmap.flush()
    
    def close(self):
        """关闭内存映射文件"""
        self.mmap.flush()
        del self.mmap
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
```

### 14.5.4 增量索引

使用增量索引避免每次更新都重建整个索引。增量索引维护一个主索引和一个增量缓冲区，定期合并。

```python
from pathlib import Path
import pickle

class IncrementalIndex:
    """增量索引"""
    
    def __init__(self, base_path: str, merge_threshold: int = 1000):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        self.merge_threshold = merge_threshold
        
        # 主索引路径
        self.main_path = self.base_path / "main.idx"
        # 增量索引路径
        self.delta_path = self.base_path / "delta.idx"
        
        self.delta_size = 0
        self._load_metadata()
    
    def _load_metadata(self):
        """加载元数据"""
        metadata_path = self.base_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.delta_size = metadata.get('delta_size', 0)
        else:
            self.delta_size = 0
    
    def add_document(self, doc_id: str, vector: np.ndarray, 
                     metadata: Optional[Dict] = None):
        """添加文档到增量索引"""
        entry = {
            'doc_id': doc_id,
            'vector': vector.tolist(),
            'metadata': metadata or {}
        }
        
        # 追加到增量文件
        with open(self.delta_path, 'a') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        self.delta_size += 1
        
        # 触发合并
        if self.delta_size >= self.merge_threshold:
            self.merge()
    
    def search(self, query_vector: np.ndarray, top_k: int = 10) -> List[Dict]:
        """搜索最近邻"""
        # 搜索主索引
        main_results = self._search_main(query_vector, top_k)
        
        # 搜索增量索引
        delta_results = self._search_delta(query_vector, top_k)
        
        # 合并结果
        all_results = main_results + delta_results
        
        # 去重（按doc_id）
        seen = set()
        unique_results = []
        for result in sorted(all_results, key=lambda x: x['score'], reverse=True):
            if result['doc_id'] not in seen:
                seen.add(result['doc_id'])
                unique_results.append(result)
                if len(unique_results) >= top_k:
                    break
        
        return unique_results
    
    def merge(self):
        """合并增量索引到主索引"""
        if not self.delta_path.exists() or self.delta_size == 0:
            return
        
        logger.info(f"Merging {self.delta_size} entries into main index")
        
        # 读取增量条目
        delta_entries = []
        with open(self.delta_path, 'r') as f:
            for line in f:
                delta_entries.append(json.loads(line))
        
        # 追加到主索引
        with open(self.main_path, 'a') as f:
            for entry in delta_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        # 清空增量索引
        self.delta_path.unlink(missing_ok=True)
        self.delta_size = 0
        self._save_metadata()
        
        logger.info("Merge completed")
    
    def _search_main(self, query_vector: np.ndarray, 
                     top_k: int) -> List[Dict]:
        """在主索引中搜索"""
        if not self.main_path.exists():
            return []
        
        results = []
        with open(self.main_path, 'r') as f:
            for line in f:
                entry = json.loads(line)
                vector = np.array(entry['vector'])
                score = self._cosine_similarity(query_vector, vector)
                results.append({
                    'doc_id': entry['doc_id'],
                    'score': score,
                    'metadata': entry.get('metadata', {})
                })
        
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]
    
    def _search_delta(self, query_vector: np.ndarray, 
                       top_k: int) -> List[Dict]:
        """在增量索引中搜索"""
        if not self.delta_path.exists():
            return []
        
        results = []
        with open(self.delta_path, 'r') as f:
            for line in f:
                entry = json.loads(line)
                vector = np.array(entry['vector'])
                score = self._cosine_similarity(query_vector, vector)
                results.append({
                    'doc_id': entry['doc_id'],
                    'score': score,
                    'metadata': entry.get('metadata', {})
                })
        
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]
    
    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """计算余弦相似度"""
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))
    
    def _save_metadata(self):
        """保存元数据"""
        metadata_path = self.base_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump({'delta_size': self.delta_size}, f)
```

## 14.6 监控体系

完善的监控体系是性能优化的基础。没有精确的测量就无法进行有效的优化。

### 14.6.1 延迟监控

延迟是衡量RAG系统性能最重要的指标之一。通常需要监控P50（中位数）、P95（第95百分位）和P99（第99百分位）延迟。

```python
import time
from collections import deque
import statistics
import threading

class LatencyMonitor:
    """延迟监控器"""
    
    def __init__(self, window_size: int = 1000, name: str = "default"):
        self.latencies = deque(maxlen=window_size)
        self.name = name
        self.lock = threading.Lock()
    
    def record(self, latency_ms: float):
        """记录延迟"""
        with self.lock:
            self.latencies.append(latency_ms)
    
    def record_call(self, func):
        """装饰器：记录函数执行延迟"""
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed = (time.perf_counter() - start) * 1000
                self.record(elapsed)
        return wrapper
    
    def get_stats(self) -> Dict[str, float]:
        """获取延迟统计"""
        with self.lock:
            if not self.latencies:
                return {}
            
            sorted_latencies = sorted(self.latencies)
            n = len(sorted_latencies)
            
            return {
                'name': self.name,
                'avg': statistics.mean(sorted_latencies),
                'min': sorted_latencies[0],
                'max': sorted_latencies[-1],
                'p50': sorted_latencies[int(n * 0.50)],
                'p90': sorted_latencies[int(n * 0.90)],
                'p95': sorted_latencies[int(n * 0.95)],
                'p99': sorted_latencies[int(n * 0.99)],
                'count': n,
                'std': statistics.stdev(sorted_latencies) if n > 1 else 0
            }
    
    def reset(self):
        """重置统计数据"""
        with self.lock:
            self.latencies.clear()

class PipelineLatencyTracker:
    """全流水线延迟追踪"""
    
    def __init__(self):
        self.timers = {}
        self.breakdown = {}
        self.start_time = None
    
    def start(self):
        """开始追踪"""
        self.start_time = time.perf_counter()
        self.breakdown = {}
        return self
    
    def mark(self, stage: str):
        """标记阶段完成"""
        if self.start_time is None:
            return
        
        now = time.perf_counter()
        elapsed_ms = (now - self.start_time) * 1000
        self.breakdown[stage] = elapsed_ms
    
    def get_total(self) -> float:
        """获取总耗时（毫秒）"""
        if self.start_time is None:
            return 0.0
        return (time.perf_counter() - self.start_time) * 1000
    
    def get_breakdown(self) -> Dict[str, float]:
        """获取阶段耗时分解"""
        return dict(self.breakdown)
    
    def get_stage_latency(self, stage: str) -> float:
        """获取特定阶段的耗时"""
        stages = sorted(self.breakdown.items(), key=lambda x: x[1])
        result = {}
        prev = 0.0
        
        for s, t in stages:
            result[s] = t - prev
            prev = t
        
        return result.get(stage, 0.0)
```

### 14.6.2 QPS和吞吐量监控

```python
from collections import defaultdict, deque
import threading
import time

class QPSMonitor:
    """QPS（每秒查询数）监控器"""
    
    def __init__(self, window_seconds: int = 60):
        self.window_seconds = window_seconds
        self.requests = deque()
        self.success_count = 0
        self.failure_count = 0
        self.lock = threading.Lock()
    
    def record_request(self, success: bool = True):
        """记录请求"""
        with self.lock:
            self.requests.append(time.time())
            if success:
                self.success_count += 1
            else:
                self.failure_count += 1
    
    def get_qps(self) -> float:
        """获取当前QPS"""
        with self.lock:
            now = time.time()
            # 清理过期记录
            cutoff = now - self.window_seconds
            while self.requests and self.requests[0] < cutoff:
                self.requests.popleft()
            
            if not self.requests:
                return 0.0
            
            # 计算窗口内的QPS
            window = min(self.window_seconds, now - self.requests[0])
            if window <= 0:
                return 0.0
            return len(self.requests) / window
    
    def get_error_rate(self) -> float:
        """获取错误率"""
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.0
        return self.failure_count / total
    
    def get_stats(self) -> Dict[str, Any]:
        """获取完整统计"""
        return {
            'qps': self.get_qps(),
            'error_rate': self.get_error_rate(),
            'success_count': self.success_count,
            'failure_count': self.failure_count,
            'total_requests': self.success_count + self.failure_count,
            'window_seconds': self.window_seconds
        }
```

### 14.6.3 成本监控

```python
class CostMonitor:
    """成本监控器"""
    
    def __init__(self):
        # 模型定价（每1K token，美元）
        self.model_pricing = {
            'gpt-4': {'input': 0.03, 'output': 0.06},
            'gpt-4-turbo': {'input': 0.01, 'output': 0.03},
            'gpt-4o': {'input': 0.005, 'output': 0.015},
            'gpt-4o-mini': {'input': 0.00015, 'output': 0.0006},
            'gpt-3.5-turbo': {'input': 0.001, 'output': 0.002},
            'text-embedding-3-small': {'input': 0.00002},
            'text-embedding-3-large': {'input': 0.00013},
            'deepseek-chat': {'input': 0.00027, 'output': 0.0011},
            'deepseek-coder': {'input': 0.00014, 'output': 0.00028}
        }
        
        self.daily_cost = defaultdict(float)
        self.total_tokens = defaultdict(int)
        self.query_costs = []
        self.lock = threading.Lock()
    
    def record_llm_call(self, model: str, input_tokens: int, 
                        output_tokens: int = 0):
        """记录LLM调用成本"""
        pricing = self.model_pricing.get(model, {})
        input_cost = pricing.get('input', 0) * input_tokens / 1000
        output_cost = pricing.get('output', 0) * output_tokens / 1000
        total_cost = input_cost + output_cost
        
        with self.lock:
            today = datetime.now().strftime('%Y-%m-%d')
            self.daily_cost[today] += total_cost
            self.total_tokens['input'] += input_tokens
            self.total_tokens['output'] += output_tokens
            self.query_costs.append({
                'timestamp': datetime.now(),
                'model': model,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'cost': total_cost
            })
    
    def get_daily_cost(self, date: Optional[str] = None) -> float:
        """获取每日成本"""
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        return self.daily_cost.get(date, 0.0)
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """获取成本汇总"""
        with self.lock:
            total_cost = sum(self.daily_cost.values())
            today = datetime.now().strftime('%Y-%m-%d')
            
            return {
                'today_cost': self.daily_cost.get(today, 0.0),
                'total_cost': total_cost,
                'total_input_tokens': self.total_tokens.get('input', 0),
                'total_output_tokens': self.total_tokens.get('output', 0),
                'total_tokens': sum(self.total_tokens.values()),
                'avg_cost_per_query': (
                    total_cost / len(self.query_costs) if self.query_costs else 0
                ),
                'query_count': len(self.query_costs)
            }
    
    def get_model_breakdown(self) -> Dict[str, float]:
        """获取模型成本分解"""
        breakdown = defaultdict(float)
        for record in self.query_costs:
            breakdown[record['model']] += record['cost']
        return dict(breakdown)
```

### 14.6.4 综合性能仪表盘

```python
class PerformanceDashboard:
    """综合性能仪表盘"""
    
    def __init__(self):
        # 各阶段延迟监控
        self.stage_monitors = {
            'embedding': LatencyMonitor(name='embedding'),
            'retrieval': LatencyMonitor(name='retrieval'),
            'rerank': LatencyMonitor(name='rerank'),
            'generation': LatencyMonitor(name='generation'),
            'total': LatencyMonitor(name='total')
        }
        
        self.qps_monitor = QPSMonitor()
        self.cost_monitor = CostMonitor()
        
        # 系统资源监控
        self.resource_metrics = deque(maxlen=100)
    
    def record_pipeline(self, 
                         stage_times: Dict[str, float],
                         model: str,
                         input_tokens: int,
                         output_tokens: int,
                         success: bool = True):
        """记录一次完整的流水线性能"""
        for stage, time_ms in stage_times.items():
            if stage in self.stage_monitors:
                self.stage_monitors[stage].record(time_ms)
        
        self.qps_monitor.record_request(success)
        self.cost_monitor.record_llm_call(model, input_tokens, output_tokens)
    
    def get_full_report(self) -> Dict[str, Any]:
        """获取完整性能报告"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'latency': {
                name: monitor.get_stats()
                for name, monitor in self.stage_monitors.items()
            },
            'throughput': self.qps_monitor.get_stats(),
            'cost': self.cost_monitor.get_cost_summary(),
            'cost_by_model': self.cost_monitor.get_model_breakdown()
        }
        
        # 计算整体健康度
        report['health_score'] = self._calculate_health_score(report)
        
        return report
    
    def _calculate_health_score(self, report: Dict) -> float:
        """计算系统健康度（0-100）"""
        score = 100.0
        
        # 基于P99延迟扣分
        total_latency = report['latency'].get('total', {})
        p99 = total_latency.get('p99', 0)
        if p99 > 5000:  # 5秒
            score -= 30
        elif p99 > 2000:  # 2秒
            score -= 15
        elif p99 > 1000:  # 1秒
            score -= 5
        
        # 基于错误率扣分
        error_rate = report['throughput'].get('error_rate', 0)
        if error_rate > 0.1:  # 10%
            score -= 30
        elif error_rate > 0.05:  # 5%
            score -= 15
        elif error_rate > 0.01:  # 1%
            score -= 5
        
        return max(0, score)
    
    def get_alert(self) -> Optional[str]:
        """检查是否需要告警"""
        report = self.get_full_report()
        
        total_stats = report['latency'].get('total', {})
        if total_stats.get('p99', 0) > 5000:
            return f"ALERT: P99 latency {total_stats['p99']:.0f}ms exceeds 5s threshold"
        
        error_rate = report['throughput'].get('error_rate', 0)
        if error_rate > 0.1:
            return f"ALERT: Error rate {error_rate:.1%} exceeds 10% threshold"
        
        return None
```

### 14.6.5 性能剖析

使用性能剖析工具定位代码级别的瓶颈。

```python
import cProfile
import pstats
import io
import functools

class Profiler:
    """性能剖析器"""
    
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
    
    def profile(self, func):
        """剖析装饰器"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not self.enabled:
                return func(*args, **kwargs)
            
            profiler = cProfile.Profile()
            try:
                profiler.enable()
                result = func(*args, **kwargs)
                profiler.disable()
                
                # 输出统计
                s = io.StringIO()
                ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
                ps.print_stats(20)
                
                logger.info(f"Profile for {func.__name__}:\n{s.getvalue()}")
                
                return result
            finally:
                profiler.disable()
        
        return wrapper
    
    def profile_block(self, name: str = "block"):
        """代码块剖析上下文管理器"""
        class ProfileContext:
            def __enter__(self):
                self.profiler = cProfile.Profile()
                self.profiler.enable()
                return self
            
            def __exit__(self, *args):
                self.profiler.disable()
                s = io.StringIO()
                ps = pstats.Stats(self.profiler, stream=s).sort_stats('time')
                ps.print_stats(10)
                logger.info(f"Profile for {name}:\n{s.getvalue()}")
        
        return ProfileContext()
```

## 14.7 综合优化策略

### 14.7.1 分级优化框架

```python
class TieredOptimizer:
    """分级优化器，根据负载和查询复杂度动态调整策略"""
    
    def __init__(self):
        # 三种优化策略
        self.strategies = {
            'aggressive': {
                'description': '激进优化，最大化吞吐量',
                'cache_ttl': 3600,
                'max_docs': 3,
                'model': 'gpt-4o-mini',
                'embedding_dim': 256,
                'quantize': True,
                'use_semantic_cache': True,
                'max_concurrency': 20
            },
            'balanced': {
                'description': '平衡模式，权衡质量和性能',
                'cache_ttl': 1800,
                'max_docs': 5,
                'model': 'gpt-4o',
                'embedding_dim': 512,
                'quantize': False,
                'use_semantic_cache': True,
                'max_concurrency': 10
            },
            'quality': {
                'description': '高质量模式，优先保证回答质量',
                'cache_ttl': 600,
                'max_docs': 10,
                'model': 'gpt-4',
                'embedding_dim': 1536,
                'quantize': False,
                'use_semantic_cache': False,
                'max_concurrency': 5
            }
        }
    
    def select_strategy(self, 
                         current_load: float, 
                         query_complexity: str = 'medium',
                         user_tier: str = 'standard') -> str:
        """根据上下文选择策略"""
        # 高负载时使用激进策略
        if current_load > 0.8:
            return 'aggressive'
        
        # VIP用户使用高质量策略
        if user_tier == 'premium':
            return 'quality'
        
        # 根据查询复杂度选择
        complexity_map = {
            'simple': 'aggressive',
            'medium': 'balanced',
            'complex': 'quality'
        }
        
        return complexity_map.get(query_complexity, 'balanced')
    
    def get_strategy_config(self, strategy_name: str) -> Dict:
        """获取策略配置"""
        return self.strategies.get(strategy_name, self.strategies['balanced'])
```

### 14.7.2 预热与冷启动优化

```python
class ColdStartOptimizer:
    """冷启动优化器"""
    
    def __init__(self):
        self.is_warmed = False
        self.warmup_completed_at = None
        
        # 预热查询模板
        self.warmup_queries = [
            "什么是知识图谱",
            "RAG系统的工作原理",
            "向量数据库的用途",
            "如何构建知识图谱",
            "Neo4j图数据库介绍"
        ]
    
    def warmup(self, pipeline):
        """系统预热"""
        if self.is_warmed:
            return
        
        logger.info("Starting system warmup...")
        start = time.time()
        
        # 1. 预热嵌入模型
        _ = pipeline.embed(self.warmup_queries)
        
        # 2. 预热LLM（发送一次简单请求）
        _ = pipeline.generate("Hello", [])
        
        # 3. 预热缓存
        for query in self.warmup_queries:
            pipeline.query(query, use_cache=True)
        
        self.is_warmed = True
        self.warmup_completed_at = datetime.now()
        
        elapsed = time.time() - start
        logger.info(f"System warmup completed in {elapsed:.2f}s")
    
    def is_cold_start(self) -> bool:
        """判断是否为冷启动"""
        return not self.is_warmed
    
    def get_warmup_age(self) -> Optional[float]:
        """获取预热后的运行时间（秒）"""
        if self.warmup_completed_at:
            return (datetime.now() - self.warmup_completed_at).total_seconds()
        return None
```

### 14.7.3 性能基线管理

```python
class PerformanceBaseline:
    """性能基线管理器"""
    
    def __init__(self, baseline_file: str = "performance_baseline.json"):
        self.baseline_file = baseline_file
        self.baseline = self._load_baseline()
    
    def _load_baseline(self) -> Dict:
        """加载性能基线"""
        if os.path.exists(self.baseline_file):
            with open(self.baseline_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_baseline(self, metrics: Dict):
        """保存性能基线"""
        self.baseline = {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        }
        with open(self.baseline_file, 'w') as f:
            json.dump(self.baseline, f, indent=2)
    
    def compare_to_baseline(self, current_metrics: Dict) -> Dict:
        """与基线比较"""
        if not self.baseline:
            return {'status': 'no_baseline'}
        
        baseline_metrics = self.baseline.get('metrics', {})
        comparison = {}
        
        for key in ['p50', 'p95', 'p99', 'avg']:
            baseline_val = baseline_metrics.get(key, 0)
            current_val = current_metrics.get(key, 0)
            
            if baseline_val > 0:
                change_pct = ((current_val - baseline_val) / baseline_val) * 100
                comparison[key] = {
                    'baseline': baseline_val,
                    'current': current_val,
                    'change_pct': round(change_pct, 2),
                    'status': 'improved' if change_pct < 0 else 'degraded'
                }
        
        return comparison
```

## 14.8 性能优化检查清单

### 缓存优化
- [ ] 实现多级缓存（内存 -> Redis -> 数据库）
- [ ] 配置合理的TTL策略
- [ ] 实现缓存预热机制
- [ ] 监控缓存命中率
- [ ] 实现缓存一致性策略

### 并发优化
- [ ] 异步化IO密集型操作
- [ ] 配置合适的线程池大小
- [ ] 管理外部服务连接池
- [ ] 实现流水线并行化
- [ ] 避免死锁和竞态条件

### Token管理
- [ ] 实现精确的Token计数
- [ ] 配置合理的预算分配
- [ ] 动态调整上下文窗口
- [ ] 监控Token使用量和成本
- [ ] 实现Token使用告警

### 内存优化
- [ ] 实现嵌入向量量化
- [ ] 使用内存映射文件管理大索引
- [ ] 实现增量索引机制
- [ ] 配置内存使用上限
- [ ] 定期清理过期缓存

### 监控体系
- [ ] 部署延迟监控（P50/P95/P99）
- [ ] 部署QPS和吞吐量监控
- [ ] 部署成本监控
- [ ] 建立性能基线
- [ ] 配置告警阈值和通知

## 14.9 本章小结

性能优化是RAG系统从原型走向生产的关键环节。本章从缓存策略、并发执行、Token预算管理、内存优化和监控体系五个维度，系统地介绍了RAG系统的性能优化方法。

**缓存策略**方面，本章介绍了多级缓存架构（查询缓存、嵌入缓存、文档缓存），以及语义缓存、分层TTL和缓存预热等高级技术。合理的缓存策略可以将响应延迟降低50%-80%。

**并发执行**方面，通过异步IO、线程池、连接池管理和流水线并行化，可以充分利用系统资源，显著提升吞吐量。关键是要根据任务类型（CPU密集型vs IO密集型）选择合适的并发模型。

**Token预算管理**是控制成本和延迟的核心手段。通过精确的Token计数、智能的预算分配和动态调整机制，可以在保证回答质量的前提下，将每次查询的成本控制在预算范围内。

**内存优化**方面，嵌入向量量化（int8、二进制量化、产品量化）和内存映射文件技术，使大规模向量索引的内存占用变得可控。增量索引机制则保证了数据更新时的性能稳定性。

**监控体系**是持续优化的基础。通过对延迟（P50/P95/P99）、QPS、错误率和成本的实时监控，可以及时发现性能瓶颈并指导优化方向。完善的监控体系还能帮助建立性能基线，量化优化效果。

在实际部署中，建议根据具体的业务场景和SLA要求，选择适当的优化组合。通常的优化路径是：先实施缓存策略（见效最快），然后优化并发模型，再调整Token预算，最后进行内存优化。每次优化后都需要通过监控数据验证效果，形成"测量-优化-验证"的持续改进循环。
