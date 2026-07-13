# 第22章 RAG系统量化评估体系

RAG系统的效果不能仅凭主观感受来评判，需要建立一套完整的量化评估体系。本章将系统介绍RAG评估的核心指标、Milvus检索性能压测方法，以及自动化评估脚本的实现。

## 22.1 核心评估指标

### 22.1.1 检索质量指标

检索阶段是RAG的基石，检索质量直接影响最终的回答效果。以下是核心的检索质量指标：

**召回率（Recall@K）**

召回率衡量在Top-K结果中，检索系统找到了多少比例的相关文档。公式如下：

```
Recall@K = |retrieved[:K] ∩ expected| / |expected|
```

其中 `retrieved[:K]` 是检索返回的前K个结果集合，`expected` 是预期相关文档集合。

```python
def recall_at_k(retrieved, expected, k):
    """计算 Recall@K"""
    if not expected:
        return 0.0
    retrieved_k = set(retrieved[:k])
    expected_set = set(expected)
    hits = len(retrieved_k & expected_set)
    return hits / len(expected_set)
```

**精确率（Precision@K）**

精确率衡量在Top-K结果中，有多少是真正相关的：

```
Precision@K = |retrieved[:K] ∩ expected| / K
```

```python
def precision_at_k(retrieved, expected, k):
    """计算 Precision@K"""
    if k == 0:
        return 0.0
    retrieved_k = set(retrieved[:k])
    expected_set = set(expected)
    hits = len(retrieved_k & expected_set)
    return hits / k
```

**平均倒数排名（MRR）**

MRR 衡量第一个相关结果出现的位置，适用于问答场景（用户通常关注第一个答案）：

```python
def reciprocal_rank(retrieved, expected):
    """计算 Reciprocal Rank"""
    expected_set = set(expected)
    for i, doc_id in enumerate(retrieved, 1):
        if doc_id in expected_set:
            return 1.0 / i
    return 0.0

def mean_reciprocal_rank(all_retrieved, all_expected):
    """计算 MRR"""
    if not all_retrieved:
        return 0.0
    rrs = [reciprocal_rank(ret, exp) for ret, exp in zip(all_retrieved, all_expected)]
    return sum(rrs) / len(rrs)
```

### 22.1.2 生成质量指标

检索之后，大模型基于检索结果生成答案。生成质量的评估指标包括：

**F1分数**

基于生成答案和参考答案的词级别重合度：

```python
def f1_score(candidate, reference):
    """计算 F1 分数（基于 Token 重叠）"""
    cand_tokens = set(re.findall(r"\w+", candidate.lower()))
    ref_tokens = set(re.findall(r"\w+", reference.lower()))

    if not cand_tokens or not ref_tokens:
        return 0.0

    common = cand_tokens & ref_tokens
    if not common:
        return 0.0

    precision = len(common) / len(cand_tokens)
    recall = len(common) / len(ref_tokens)
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
```

**BLEU分数**

衡量生成答案与参考答案的n-gram相似度，适用于翻译和生成任务：

```python
def bleu_like(candidate, reference, max_n=2):
    """简化版 BLEU 分数（n=1,2 的几何平均 + 简短惩罚）"""
    cand_tokens = re.findall(r"\w+", candidate.lower())
    ref_tokens = re.findall(r"\w+", reference.lower())

    if not cand_tokens or not ref_tokens:
        return 0.0

    precisions = []
    for n in range(1, max_n + 1):
        cand_ngrams = Counter(
            tuple(cand_tokens[i:i+n]) for i in range(len(cand_tokens) - n + 1)
        )
        ref_ngrams = Counter(
            tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens) - n + 1)
        )
        matches = sum(min(cand_ngrams[ng], ref_ngrams.get(ng, 0)) for ng in cand_ngrams)
        total = sum(cand_ngrams.values())
        precisions.append(matches / total if total > 0 else 0.0)

    # 几何平均
    if any(p == 0 for p in precisions):
        return 0.0
    geo_mean = math.exp(sum(math.log(p) for p in precisions) / len(precisions))

    # 简短惩罚
    c = len(cand_tokens)
    r = len(ref_tokens)
    bp = math.exp(1 - r / c) if c < r else 1.0

    return bp * geo_mean
```

### 22.1.3 性能指标

**QPS（Queries Per Second）**

衡量系统每秒能处理的查询数量：

```
QPS = 总查询数 / 总耗时（秒）
```

**延迟分布**

关注 P50（中位数）、P95、P99 延迟：

```
P50延迟 = median(所有请求延迟)
P99延迟 = sorted(所有请求延迟)[int(len(延迟列表) * 0.99)]
```

## 22.2 Milvus 检索性能压测

### 22.2.1 压测方案设计

完整的性能压测需要考虑以下维度：

- **数据规模**：测试不同数据量级（10万、100万、500万、1000万）
- **并发级别**：测试不同并发数（1、10、50、100、200）
- **索引类型**：对比 FLAT、IVF_FLAT、IVF_SQ8、HNSW 的性能差异
- **向量维度**：测试不同维度（128、256、512、768）的影响

### 22.2.2 压测脚本实现

```python
import time
import random
import threading
from statistics import mean, median, stdev
from pymilvus import MilvusClient, Collection, utility

class MilvusBenchmark:
    """Milvus 检索性能压测工具"""

    def __init__(self, uri="http://localhost:19530"):
        self.client = MilvusClient(uri=uri)
        self.results = {}

    def prepare_data(self, collection_name, num_vectors, dim=768,
                     index_type="HNSW"):
        """准备测试数据"""
        print(f"准备测试数据: {num_vectors} 条, 维度 {dim}")
        if self.client.has_collection(collection_name):
            self.client.drop_collection(collection_name)

        schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False)
        schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dim)
        schema.add_field("content", DataType.VARCHAR, max_length=256)

        client.create_collection(collection_name=collection_name, schema=schema)

        # 批量写入
        BATCH_SIZE = 10000
        for i in range(0, num_vectors, BATCH_SIZE):
            batch = []
            for _ in range(min(BATCH_SIZE, num_vectors - i)):
                batch.append({
                    "vector": [random.random() for _ in range(dim)],
                    "content": f"doc_{i}",
                })
            self.client.insert(collection_name=collection_name, data=batch)

        # 构建索引
        index_params = {
            "index_type": index_type,
            "metric_type": "IP",
            "params": {"M": 24, "efConstruction": 500} if index_type == "HNSW"
                      else {"nlist": 4096},
        }
        self.client.create_index(
            collection_name=collection_name,
            index_params=index_params,
        )
        self.client.load_collection(collection_name)
        print(f"数据准备完成")

    def benchmark_search(self, collection_name, concurrency=10,
                         num_queries=1000, top_k=100):
        """执行检索压测"""
        # 准备测试查询向量
        test_vectors = [[random.random() for _ in range(768)]
                        for _ in range(num_queries)]

        latencies = []
        errors = 0
        lock = threading.Lock()

        def worker():
            nonlocal errors
            for vec in test_vectors:
                start = time.time()
                try:
                    self.client.search(
                        collection_name=collection_name,
                        data=[vec],
                        limit=top_k,
                    )
                    latency = (time.time() - start) * 1000
                    with lock:
                        latencies.append(latency)
                except Exception as e:
                    with lock:
                        errors += 1

        threads = [threading.Thread(target=worker) for _ in range(concurrency)]
        start = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.time() - start

        total_queries = num_queries * concurrency
        qps = total_queries / elapsed if elapsed > 0 else 0

        report = {
            "concurrency": concurrency,
            "total_queries": total_queries,
            "elapsed": round(elapsed, 2),
            "qps": round(qps),
            "avg_latency": round(mean(latencies), 1) if latencies else 0,
            "p50_latency": round(median(latencies), 1) if latencies else 0,
            "p99_latency": round(sorted(latencies)[int(len(latencies)*0.99)], 1)
                           if latencies else 0,
            "errors": errors,
        }
        return report

    def run_full_benchmark(self, collection_name, num_vectors=1000000):
        """运行完整压测"""
        for concurrency in [1, 10, 50, 100]:
            report = self.benchmark_search(
                collection_name, concurrency=concurrency
            )
            print(f"并发={concurrency}: QPS={report['qps']}, "
                  f"P50={report['p50_latency']}ms, P99={report['p99_latency']}ms")
            self.results[f"concurrent_{concurrency}"] = report
        return self.results
```

### 22.2.3 压测结果解读

以下是一组典型压测结果（768维向量，HNSW索引，100万数据）：

| 并发数 | QPS | P50延迟 | P99延迟 | 错误数 |
|-------|-----|--------|--------|-------|
| 1 | 120 | 8ms | 15ms | 0 |
| 10 | 850 | 12ms | 28ms | 0 |
| 50 | 2800 | 18ms | 55ms | 0 |
| 100 | 4200 | 24ms | 120ms | 2 |

解读要点：

- **QPS 随并发数增长呈亚线性增长**：从 1 并发到 100 并发，QPS 从 120 增长到 4200（约 35 倍），说明系统具有良好的并发扩展能力。
- **P99 延迟随并发数增长显著增加**：100 并发时 P99 延迟达到 120ms，说明系统在高并发下出现队列等待。
- **错误数开始出现**：100 并发时出现少量错误，说明接近系统承载上限。

### 22.2.4 不同索引的性能对比

| 索引类型 | 1并发QPS | 100并发QPS | P99延迟(100并发) | 内存占用 |
|---------|---------|-----------|-----------------|---------|
| FLAT | 15 | 300 | 850ms | 3GB |
| IVF_FLAT (nlist=4096) | 200 | 3500 | 180ms | 3.2GB |
| IVF_SQ8 (nlist=4096) | 450 | 6000 | 80ms | 0.8GB |
| HNSW (M=24) | 120 | 4200 | 120ms | 4.5GB |

对比结论：

- **FLAT** 仅适合小规模数据（<10万），千万级不可用。
- **IVF_SQ8** 在内存占用和检索速度之间取得了最佳平衡，适合内存受限的场景。
- **HNSW** 在低并发下延迟最低（8ms），适合对延迟要求苛刻的在线服务。
- **IVF_FLAT** 综合表现中等，适合中等规模场景。

## 22.3 自动化测评脚本

### 22.3.1 测试数据集构建

自动化评估的第一步是构建包含"查询-预期文档-预期答案"三元组的测试集：

```python
# 测试查询数据集
TEST_QUERIES = [
    {
        "query": "什么是监督学习？",
        "expected_docs": ["doc1", "doc6"],
        "expected_answer": "监督学习是一种使用标注数据进行训练的机器学习方法。",
    },
    {
        "query": "CNN和RNN的区别",
        "expected_docs": ["doc2", "doc7"],
        "expected_answer": "CNN擅长图像处理，RNN适合序列数据处理。",
    },
    {
        "query": "常用的模型评估指标",
        "expected_docs": ["doc8"],
        "expected_answer": "常用评估指标包括精确率、召回率、F1分数和ROC曲线。",
    },
    {
        "query": "BERT和GPT是什么",
        "expected_docs": ["doc3"],
        "expected_answer": "BERT和GPT是重要的预训练语言模型，用于自然语言处理任务。",
    },
    {
        "query": "强化学习经典算法",
        "expected_docs": ["doc5"],
        "expected_answer": "Q学习和深度Q网络（DQN）是强化学习的经典算法。",
    },
]
```

### 22.3.2 完整评估流程

```python
def run_evaluation(client, collection_name, test_queries, top_k=5):
    """运行完整评估"""
    all_retrieved = []
    all_expected = []
    all_candidates = []
    all_references = []

    # 逐条评估
    for tq in test_queries:
        query = tq["query"]
        expected = tq["expected_docs"]
        expected_answer = tq["expected_answer"]

        # 执行检索
        q_vec = generate_embedding(query)
        results = client.search(
            collection_name=collection_name,
            data=[q_vec],
            limit=top_k,
            output_fields=["title", "content"],
        )
        retrieved = [r["id"] for r in results[0]] if results[0] else []
        all_retrieved.append(retrieved)
        all_expected.append(expected)

        # 模拟答案生成
        candidate = simulate_answer(query, retrieved)
        all_candidates.append(candidate)
        all_references.append(expected_answer)

    # 计算聚合指标
    avg_r1 = mean(recall_at_k(r, e, 1) for r, e in zip(all_retrieved, all_expected))
    avg_r3 = mean(recall_at_k(r, e, 3) for r, e in zip(all_retrieved, all_expected))
    avg_r5 = mean(recall_at_k(r, e, 5) for r, e in zip(all_retrieved, all_expected))
    avg_p1 = mean(precision_at_k(r, e, 1) for r, e in zip(all_retrieved, all_expected))
    avg_p3 = mean(precision_at_k(r, e, 3) for r, e in zip(all_retrieved, all_expected))
    mrr = mean_reciprocal_rank(all_retrieved, all_expected)
    avg_bleu = mean(bleu_like(c, r) for c, r in zip(all_candidates, all_references))
    avg_f1 = mean(f1_score(c, r) for c, r in zip(all_candidates, all_references))

    # 输出评估报告
    report = {
        "test_queries": len(test_queries),
        "retrieval": {
            "recall@1": round(avg_r1, 3),
            "recall@3": round(avg_r3, 3),
            "recall@5": round(avg_r5, 3),
            "precision@1": round(avg_p1, 3),
            "precision@3": round(avg_p3, 3),
            "mrr": round(mrr, 3),
        },
        "generation": {
            "bleu": round(avg_bleu, 3),
            "f1": round(avg_f1, 3),
        },
    }
    return report
```

### 22.3.3 持续迭代优化

评估体系的价值在于驱动持续优化。建议建立以下迭代流程：

```python
def optimization_loop(client, collection_name, test_queries):
    """评估驱动的优化循环"""
    best_score = 0
    best_config = None

    configs_to_try = [
        {"index_type": "IVF_FLAT", "nlist": 1024, "nprobe": 64},
        {"index_type": "IVF_FLAT", "nlist": 2048, "nprobe": 128},
        {"index_type": "IVF_SQ8", "nlist": 4096, "nprobe": 256},
        {"index_type": "HNSW", "M": 16, "efConstruction": 200, "ef": 128},
        {"index_type": "HNSW", "M": 24, "efConstruction": 500, "ef": 200},
    ]

    for config in configs_to_try:
        print(f"\n尝试配置: {config}")

        # 重新构建索引
        rebuild_index(client, collection_name, config)
        time.sleep(5)  # 等待索引生效

        # 评估
        report = run_evaluation(client, collection_name, test_queries)
        score = report["retrieval"]["recall@3"]

        print(f"  Recall@3: {score:.3f}, F1: {report['generation']['f1']:.3f}")

        if score > best_score:
            best_score = score
            best_config = config

    print(f"\n最佳配置: {best_config}, 最佳 Recall@3: {best_score:.3f}")
    return best_config
```

## 22.4 评估报告模板

一个完整的评估报告应包含以下内容：

```json
{
  "test_info": {
    "date": "2026-07-13",
    "data_size": 1000000,
    "dimension": 768,
    "index_type": "HNSW",
    "index_params": {"M": 24, "efConstruction": 500}
  },
  "retrieval_metrics": {
    "recall@1": 0.825,
    "recall@3": 0.943,
    "recall@5": 0.972,
    "precision@1": 0.825,
    "precision@3": 0.314,
    "mrr": 0.891
  },
  "generation_metrics": {
    "bleu": 0.673,
    "f1": 0.781
  },
  "performance": {
    "qps_1concurrent": 120,
    "qps_10concurrent": 850,
    "qps_50concurrent": 2800,
    "p50_latency_ms": 12,
    "p99_latency_ms": 55
  }
}
```

## 本章小结

量化评估是RAG系统从"能用"走向"好用"的关键环节。本章的核心要点包括：

1. **建立多维评估指标体系**：检索质量（Recall/Precision/MRR）、生成质量（BLEU/F1）、性能（QPS/延迟）三者缺一不可。
2. **性能压测覆盖全场景**：测试不同数据规模、并发级别、索引类型的组合，找到系统的最佳配置和瓶颈。
3. **评估驱动持续优化**：将评估结果作为迭代优化的基准，通过系统化的参数调优不断提升系统效果。
4. **自动化评估脚本**：将评估流程脚本化、自动化，确保每次变更都能得到量化的效果反馈。
