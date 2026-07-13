# 第7章：Milvus检索原理

## 7.1 向量检索完整流程

理解Milvus的向量检索流程，有助于在RAG系统中进行精准的查询调优。一次完整的向量检索从客户端发起，经历请求分发、索引查询、结果聚合和返回四个阶段。

### 7.1.1 检索流程总览

```
客户端（SDK）
    │
    │ 1. 发送搜索请求（查询向量 + 参数）
    ▼
Proxy（接入层）
    │
    │ 2. 解析请求，查询RootCoord获取segment分布信息
    │ 3. 将请求广播到所有持有相关segment的QueryNode
    ▼
QueryNode（执行层，多个并行）
    │
    │ 4. 在每个加载的segment上执行ANN搜索
    │ 5. 应用标量过滤条件
    │ 6. 返回局部Top-K结果
    │
    ▼
Proxy（结果聚合）
    │
    │ 7. 合并所有QueryNode返回的局部结果
    │ 8. 全局排序，取最终Top-K
    │ 9. 根据output_fields拉取完整字段数据
    ▼
客户端（结果）
```

### 7.1.2 请求分发机制

当客户端发起搜索请求时，Proxy需要确定请求应该分发到哪些QueryNode。这个过程涉及两个关键步骤：

1. **Segment分布查询**：Proxy向QueryCoord查询当前所有segment的分布情况，包括每个segment所在的QueryNode
2. **请求广播**：根据segment分布信息，将搜索请求广播到所有持有相关segment的QueryNode

```python
# 搜索请求的内部流程示意（伪代码）
class SearchRequest:
    def __init__(self, collection_name, query_vector, top_k, params):
        self.collection = collection_name
        self.query_vector = query_vector
        self.top_k = top_k
        self.params = params
    
    def execute(self):
        # 1. 查询segment分布
        segments = query_coord.get_segments(self.collection)
        
        # 2. 分组：按QueryNode分组
        node_segments = {}
        for seg in segments:
            node_id = seg.query_node_id
            if node_id not in node_segments:
                node_segments[node_id] = []
            node_segments[node_id].append(seg)
        
        # 3. 并行向各QueryNode发送子请求
        sub_results = []
        for node_id, seg_list in node_segments.items():
            sub_result = query_node.search(
                node_id=node_id,
                segments=seg_list,
                query_vector=self.query_vector,
                top_k=self.top_k * 2  # 每个节点多查一些
            )
            sub_results.append(sub_result)
        
        # 4. 合并结果
        final_result = merge_results(sub_results, top_k=self.top_k)
        return final_result
```

### 7.1.3 查询参数详解

Milvus的搜索API支持丰富的查询参数，理解每个参数的作用是调优查询效果的关键。

```python
search_params = {
    # 距离度量方式
    "metric_type": "COSINE",  # COSINE | L2 | IP
    
    # ANN索引特定参数
    "params": {
        # HNSW: 搜索宽度
        "ef": 100,
        # IVF: 探访的聚类数
        "nprobe": 16,
    },
    
    # 偏移量（分页）
    "offset": 0,
    
    # 是否跳过重复结果
    "skip_duplicates": False,
    
    # 返回的Round Decimal精度
    "round_decimal": -1,
}

results = collection.search(
    data=[query_vector],      # 查询向量（支持批量）
    anns_field="embedding",    # 向量字段名
    param=search_params,       # 搜索参数
    limit=10,                  # 返回Top-K
    # 返回的字段列表
    output_fields=["text", "title", "source"],
    # 时间戳（实现时间旅行查询）
    travel_timestamp=None,
    # 分区列表（限定搜索范围）
    partition_names=None,
)
```

### 7.1.4 时间旅行查询

Milvus支持时间旅行查询（Time Travel），允许查询某个历史时间点的数据状态。这在RAG系统的数据版本回滚和AB测试中非常有用。

```python
import time

# 记录当前时间戳
current_ts = int(time.time() * 1000)

# 插入一批数据后，记录时间戳
mr = collection.insert([embeddings, texts])
collection.flush()
after_insert_ts = int(time.time() * 1000)

# 后续可以用时间戳查询插入时的数据状态
results = collection.search(
    data=[query_vector],
    anns_field="embedding",
    param=search_params,
    limit=10,
    travel_timestamp=after_insert_ts  # 查询插入时刻的数据
)
```

## 7.2 混合检索：向量+标量过滤

纯向量检索在处理需要精确条件过滤的场景时力不从心。Milvus原生支持在向量检索的同时应用标量过滤条件，这种能力被称为**混合检索**（Hybrid Search）。

### 7.2.1 表达式过滤

Milvus支持通过表达式（expr）在向量检索时附加标量过滤条件。表达式语法兼容SQL的WHERE子句。

```python
# 基本过滤：等值匹配
expr = 'doc_type == "report"'

# 范围过滤
expr = 'created_at >= 1700000000000 and created_at <= 1800000000000'

# IN过滤
expr = 'source in ["arxiv", "openreview"]'

# 字符串模糊匹配
expr = 'tags like "%deep_learning%"'

# 复合条件
expr = '(doc_type == "manual" or doc_type == "faq") and created_at > 1700000000000'

# 在搜索时应用过滤
results = collection.search(
    data=[query_vector],
    anns_field="embedding",
    param=search_params,
    limit=10,
    expr=expr,
    output_fields=["text", "title", "source"]
)
```

### 7.2.2 过滤执行策略

Milvus支持两种过滤执行策略：**预过滤**（IVF）和**后过滤**（HNSW）。

**预过滤（IVF）**：在向量检索之前先应用标量过滤，缩小搜索范围。这种策略适合过滤条件严格（选择性高）的场景，可以大幅减少向量计算量。

**后过滤（HNSW）**：先执行向量检索，再对结果应用标量过滤。这种策略适合过滤条件宽松的场景。HNSW默认使用后过滤，因为HNSW的图结构不支持高效的前置过滤。

```python
# IVF索引：预过滤
search_params_ivf = {
    "metric_type": "COSINE",
    "params": {"nprobe": 16}
}

# 对于IVF，过滤条件会在搜索时先应用到聚类上
# 只搜索满足条件的聚类中的向量
results_ivf = collection.search(
    data=[query_vector],
    anns_field="embedding",
    param=search_params_ivf,
    limit=10,
    expr='doc_type == "report"'
)

# HNSW索引：后过滤
search_params_hnsw = {
    "metric_type": "COSINE",
    "params": {"ef": 100}
}

# 对于HNSW，建议增大搜索范围以补偿后过滤的损耗
results_hnsw = collection.search(
    data=[query_vector],
    anns_field="embedding",
    param=search_params_hnsw,
    limit=50,  # 多取一些，因为过滤后可能不足10条
    expr='doc_type == "report"'
)
```

**过滤性能调优建议**：

| 过滤条件选择性 | 推荐索引 | 策略 | 说明 |
|--------------|---------|------|------|
| 严格（保留<10%） | IVF_FLAT | 预过滤 | 先过滤再检索，效率最高 |
| 中等（保留10~50%） | IVF_FLAT/HNSW | 都可 | 增大nprobe或ef |
| 宽松（保留>50%） | HNSW | 后过滤 | 先检索再过滤，增大top_k |

### 7.2.3 RAG场景中的典型过滤模式

```python
def rag_hybrid_search(collection, query_vector, filters, top_k=10):
    """RAG系统的混合检索封装"""
    search_params = {
        "metric_type": "COSINE",
        "params": {"ef": 100}
    }
    
    # 构建过滤表达式
    expr_parts = []
    
    # 1. 时间范围过滤
    if "start_time" in filters and "end_time" in filters:
        expr_parts.append(
            f"created_at >= {filters['start_time']} "
            f"and created_at <= {filters['end_time']}"
        )
    
    # 2. 文档类型过滤
    if "doc_types" in filters and filters["doc_types"]:
        types_str = ",".join([f'"{t}"' for t in filters["doc_types"]])
        expr_parts.append(f"doc_type in [{types_str}]")
    
    # 3. 来源过滤
    if "sources" in filters and filters["sources"]:
        sources_str = ",".join([f'"{s}"' for s in filters["sources"]])
        expr_parts.append(f"source in [{sources_str}]")
    
    # 4. 关键词过滤
    if "keyword" in filters:
        expr_parts.append(f'text like "%{filters["keyword"]}%"')
    
    expr = " and ".join(expr_parts) if expr_parts else None
    
    # 执行混合检索
    results = collection.search(
        data=[query_vector],
        anns_field="embedding",
        param=search_params,
        limit=top_k * 3 if expr else top_k,  # 有过滤时多取一些
        expr=expr,
        output_fields=["text", "title", "source", "doc_type", "created_at"]
    )
    
    return results
```

## 7.3 分页检索与TopK

### 7.3.1 基础TopK检索

TopK检索是最基本的查询模式，返回与查询向量最相似的K个结果。在RAG系统中，TopK的取值直接影响生成质量：

- **K太小**：可能遗漏重要信息，导致回答不完整
- **K太大**：引入噪声信息，增加LLM上下文负担和Token消耗

```python
# 不同场景的推荐TopK
TOP_K_RECOMMENDATIONS = {
    "qa": 5,           # 问答场景：精确匹配
    "summary": 10,     # 摘要场景：需要更多上下文
    "multi_doc": 15,   # 多文档综合分析
    "search": 20,      # 搜索场景：高召回率优先
}
```

### 7.3.2 分页检索（Offset）

Milvus支持通过offset参数实现分页查询，这在需要浏览大量检索结果的场景中非常有用。

```python
def paginated_search(collection, query_vector, page=1, page_size=10):
    """分页检索"""
    offset = (page - 1) * page_size
    
    results = collection.search(
        data=[query_vector],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"ef": 100}},
        limit=page_size,
        offset=offset,
        output_fields=["text", "title"]
    )
    
    return results
```

### 7.3.3 批量检索（多向量查询）

Milvus支持在一次搜索请求中传入多个查询向量，这在大规模查询或需要多角度检索的场景中非常高效。

```python
# 批量查询：一次搜索多个向量
multi_queries = np.array([query1, query2, query3])  # shape: (3, 1024)

results = collection.search(
    data=multi_queries,   # 多个查询向量
    anns_field="embedding",
    param={"metric_type": "COSINE", "params": {"ef": 100}},
    limit=10,
    output_fields=["text"]
)

# 结果为每个查询向量的结果列表
for i, hits in enumerate(results):
    print(f"查询 {i+1} 的结果:")
    for hit in hits:
        print(f"  ID={hit.id}, Score={hit.score:.4f}, Text={hit.entity.text[:50]}")
```

## 7.4 相似度阈值控制

在RAG系统中，低质量的检索结果会降低最终回答的质量。通过设置相似度阈值，可以过滤掉不相关的结果，降低LLM的"幻觉"风险。

### 7.4.1 阈值过滤

```python
def search_with_threshold(collection, query_vector, top_k=10, min_score=0.6):
    """带相似度阈值的检索"""
    results = collection.search(
        data=[query_vector],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"ef": 100}},
        limit=top_k * 3,  # 多取一些以便过滤
        output_fields=["text", "title", "source"]
    )
    
    # 过滤低分结果
    filtered = [
        hit for hit in results[0]
        if hit.score >= min_score
    ][:top_k]
    
    if not filtered:
        print("警告：所有结果均低于阈值，检索结果为空")
    
    return filtered
```

### 7.4.2 自适应阈值

不同的查询类型对相似度阈值的要求不同。语义明确的查询可以设置较低阈值，而模糊查询需要较高阈值。

```python
def adaptive_threshold(query_text):
    """根据查询特征自适应调整阈值"""
    # 短查询（关键词型）：需要较高阈值
    if len(query_text) < 10:
        return 0.7
    # 长查询（描述型）：可以接受较低阈值
    elif len(query_text) > 50:
        return 0.5
    # 中等长度查询
    else:
        return 0.6

def rag_search_with_adaptive_threshold(collection, query_text, query_vector):
    """带自适应阈值的RAG检索"""
    min_score = adaptive_threshold(query_text)
    
    results = search_with_threshold(
        collection, query_vector, 
        top_k=5, min_score=min_score
    )
    
    return {
        "results": results,
        "threshold": min_score,
        "result_count": len(results)
    }
```

## 7.5 多路检索融合

在高级RAG架构中，单一检索方式往往不够。多路检索融合（Multi-Route Retrieval Fusion）通过组合多种检索策略的结果，提升召回率和鲁棒性。

### 7.5.1 多路检索架构

```
用户查询
    │
    ├──→ [Milvus向量检索] ──→ 语义匹配结果
    ├──→ [BM25关键词检索] ──→ 精确匹配结果
    ├──→ [知识图谱检索]  ──→ 关系匹配结果
    │
    └──→ [RRF/加权融合]  ──→ 最终排序结果
```

### 7.5.2 Milvus多路检索

Milvus本身可以产生多路结果——通过不同索引类型或不同搜索参数获得互补的检索结果。

```python
def multi_route_search(collection, query_vector, top_k=10):
    """多路检索：使用不同搜索参数获取互补结果"""
    all_hits = []
    
    # 路由1：高召回率搜索（大ef）
    results1 = collection.search(
        data=[query_vector],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"ef": 200}},
        limit=top_k,
        output_fields=["text", "title"]
    )
    for hit in results1[0]:
        all_hits.append(("high_recall", hit))
    
    # 路由2：高精度搜索（小ef）
    results2 = collection.search(
        data=[query_vector],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"ef": 30}},
        limit=top_k,
        output_fields=["text", "title"]
    )
    for hit in results2[0]:
        all_hits.append(("high_precision", hit))
    
    # 路由3：带标量过滤的搜索
    # 假设按文档类型分别搜索
    for doc_type in ["report", "manual"]:
        results3 = collection.search(
            data=[query_vector],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"ef": 100}},
            limit=top_k,
            expr=f'doc_type == "{doc_type}"',
            output_fields=["text", "title"]
        )
        for hit in results3[0]:
            all_hits.append((f"filtered_{doc_type}", hit))
    
    return all_hits
```

### 7.5.3 RRF融合

RRF（Reciprocal Rank Fusion）是融合多路检索结果的标准算法。它基于排名位置而非原始分数进行融合，对不同检索器的分数尺度不敏感。

```python
from collections import defaultdict

def rrf_fusion(route_results, k=60, top_k=10):
    """RRF融合多路检索结果"""
    rrf_scores = defaultdict(float)
    
    for route_name, hits in route_results:
        for rank, hit in enumerate(hits):
            # 每个结果在不同路线中的排名
            doc_key = hit.id
            rrf_scores[doc_key] += 1.0 / (k + rank + 1)
    
    # 按RRF分数排序
    sorted_results = sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    return sorted_results[:top_k]

# 使用示例
route_results = multi_route_search(collection, query_vector)
final_ranking = rrf_fusion(route_results, top_k=10)
```

### 7.5.4 向量检索+BM25混合检索

在RAG系统中，最常见的多路检索组合是"向量检索 + BM25关键词检索"。向量检索擅长语义匹配，BM25擅长精确关键词匹配，两者互补。

```python
import jieba
from rank_bm25 import BM25Okapi

class HybridRetriever:
    """向量检索 + BM25混合检索器"""
    
    def __init__(self, milvus_collection, documents):
        self.collection = milvus_collection
        self.documents = documents
        self._build_bm25_index()
    
    def _build_bm25_index(self):
        """构建BM25索引"""
        tokenized_docs = [
            list(jieba.cut(doc.get("text", "")))
            for doc in self.documents
        ]
        self.bm25 = BM25Okapi(tokenized_docs)
    
    def search(self, query_text, query_vector, top_k=10, alpha=0.5):
        """混合检索：向量检索 + BM25"""
        # 1. 向量检索
        vector_results = self.collection.search(
            data=[query_vector],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"ef": 100}},
            limit=top_k * 2,
            output_fields=["text", "title"]
        )
        
        # 2. BM25检索
        query_tokens = list(jieba.cut(query_text))
        bm25_scores = self.bm25.get_scores(query_tokens)
        bm25_top_k = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True
        )[:top_k * 2]
        
        # 3. RRF融合
        rrf_scores = defaultdict(float)
        
        for rank, hit in enumerate(vector_results[0]):
            rrf_scores[hit.id] += alpha / (60 + rank + 1)
        
        for rank, idx in enumerate(bm25_top_k):
            doc_id = self.documents[idx].get("id")
            rrf_scores[doc_id] += (1 - alpha) / (60 + rank + 1)
        
        # 4. 排序输出
        sorted_results = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_results[:top_k]
```

## 7.6 本章小结

Milvus的检索原理贯穿请求分发、索引查询、结果聚合和结果返回四个阶段。理解每个阶段的机制有助于在RAG系统中进行精准的查询调优。

**向量检索完整流程**中，Proxy负责请求分发和结果聚合，QueryNode负责实际的ANN搜索。理解这一流程有助于定位性能瓶颈——当查询延迟高时，可以检查QueryNode的负载和segment分布是否均衡。

**混合检索（向量+标量过滤）** 是RAG系统的核心能力。IVF索引支持高效的预过滤，HNSW索引使用后过滤策略。在实际应用中，建议根据过滤条件的选择性选择合适的索引和策略。

**分页检索与TopK** 方面，offset参数支持分页浏览，批量检索支持多向量查询。TopK的取值需要在召回率和上下文质量之间权衡。

**相似度阈值控制** 通过过滤低分结果降低LLM的"幻觉"风险。自适应阈值策略可以根据查询类型动态调整阈值，在召回率和精准度之间取得更好的平衡。

**多路检索融合** 将向量检索与BM25等关键词检索的结果通过RRF算法融合，兼顾了语义匹配和精确匹配的优势。RRF算法基于排名位置进行融合，对不同检索器的分数尺度不敏感，是实现多路检索融合的首选方案。
