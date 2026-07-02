# 第 15 章：RAG 评估体系建设

## 15.1 评估的重要性

### 15.1.1 为什么需要系统化评估

没有评估就没有优化。RAG 系统的评估比传统检索系统更复杂：

```
传统搜索: 用户点击 → 明确反馈 → 点击率可衡量
     ↓
RAG 系统: 用户看到答案 → 难以判断是否正确 → 隐性反馈为主
     ↓
需要系统化评估框架:
  1. 离线评估（自动化指标）
  2. 在线评估（用户反馈）
  3. 人工评估（专家标注）
```

### 15.1.2 RAG 评估的三大维度

```
┌──────────────────────────────────────────────┐
│               RAG 评估体系                     │
├──────────────┬──────────────┬──────────────────┤
│   检索质量    │   生成质量    │    系统性能       │
├──────────────┼──────────────┼──────────────────┤
│ • Recall     │ • 答案正确性  │ • 延迟 P50/P99   │
│ • MRR        │ • 忠实度     │ • 吞吐量 QPS     │
│ • NDCG       │ • 相关性     │ • 可用性 SLA     │
│ • HitRate    │ • 完整性     │ • 错误率          │
│ • 精确率     │ • 有用性     │ • Token 消耗      │
└──────────────┴──────────────┴──────────────────┘
```

---

## 15.2 检索质量评估

### 15.2.1 核心指标

#### Recall@K（召回率）

在检索的 top-k 结果中，包含相关文档的比例：

```python
def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Recall@K: 前 k 个检索结果中相关文档的覆盖率。"""
    if not relevant:
        return 0.0
    retrieved_k = set(retrieved[:k])
    hits = len(retrieved_k & relevant)
    return hits / len(relevant)
```

| K 值 | 说明 | 推荐阈值 |
|------|------|---------|
| @1 | 第一个结果就命中的概率 | > 0.6 |
| @3 | 前三个结果就够用的概率 | > 0.8 |
| @10 | 批量检索时的覆盖率 | > 0.95 |

#### MRR（Mean Reciprocal Rank）

第一个相关结果的排序位置：

```python
def mrr(retrieved: list[str], relevant: set[str]) -> float:
    """MRR: 第一个相关结果的排名的倒数。"""
    for i, doc in enumerate(retrieved):
        if doc in relevant:
            return 1.0 / (i + 1)
    return 0.0
```

MRR 对"第一结果是否相关"非常敏感。在 QA 场景中，用户通常只看第一个结果。

#### NDCG（Normalized Discounted Cumulative Gain）

考虑排序位置和相关性强度的指标：

```python
def ndcg_at_k(retrieved: list[str], relevance: dict[str, float], k: int) -> float:
    """NDCG@K: 带位置折扣的累积增益。"""
    def dcg(items):
        return sum(
            rel / math.log2(i + 2)
            for i, (doc, rel) in enumerate(items)
        )

    # 实际排序的 DCG
    actual = [(doc, relevance.get(doc, 0.0)) for doc in retrieved[:k]]
    actual_dcg = dcg(actual)

    # 理想排序的 DCG
    ideal = sorted(relevance.items(), key=lambda x: -x[1])[:k]
    ideal_dcg = dcg(ideal)

    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0
```

#### HitRate（命中率）

```python
def hit_rate(retrieved: list[str], relevant: set[str]) -> float:
    """HitRate: 只要有一个相关结果就算命中。"""
    return 1.0 if any(doc in relevant for doc in retrieved) else 0.0
```

### 15.2.2 指标对比

| 指标 | 关注点 | 值域 | 敏感于 | 使用场景 |
|------|--------|------|--------|---------|
| Recall@K | 覆盖率 | [0, 1] | 全部相关结果 | 检索器对比 |
| MRR | 首结果位置 | [0, 1] | 最高排序的相关结果 | QA 场景 |
| NDCG | 排序质量 | [0, 1] | 排序位置 + 相关度 | 排序模型评估 |
| HitRate | 有无结果 | {0, 1} | 是否有相关结果 | 简单快速评估 |

---

## 15.3 生成质量评估

### 15.3.1 核心指标

#### Answer Correctness（答案正确性）

```python
class AnswerCorrectness:
    """评估答案的语义正确性。"""

    def evaluate(self, answer: str, ground_truth: str) -> dict:
        """基于关键信息匹配的正确性评估。"""
        # 提取关键信息（简化版：共现实体）
        answer_entities = self._extract_entities(answer)
        truth_entities = self._extract_entities(ground_truth)

        if not truth_entities:
            return {"score": 1.0, "detail": "人工评估建议"}

        tp = len(answer_entities & truth_entities)
        fp = len(answer_entities - truth_entities)
        fn = len(truth_entities - answer_entities)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {
            "f1_score": f1,
            "precision": precision,
            "recall": recall,
            "tp": tp, "fp": fp, "fn": fn,
        }

    def _extract_entities(self, text: str) -> set[str]:
        """提取文本中的关键实体（简单分词 Token 去停用词）。"""
        # 生产环境使用 NER 模型
        stopwords = {"的", "了", "是", "在", "和", "有", "也", "就", "不", "都"}
        tokens = text.split()
        return {t for t in tokens if len(t) > 1 and t not in stopwords}
```

#### Faithfulness（忠实度）

评估答案是否**完全基于**提供的上下文，没有额外信息：

```python
class FaithfulnessEvaluator:
    """评估答案是否忠实于上下文。"""

    def evaluate(self, answer: str, context: list[str]) -> float:
        """计算忠实度分数。

        方法: 将答案分解为原子声明，检查每个声明是否能被上下文支持。
        """
        claims = self._decompose_to_claims(answer)
        if not claims:
            return 1.0

        supported = 0
        for claim in claims:
            if self._is_supported(claim, context):
                supported += 1

        return supported / len(claims)

    def _decompose_to_claims(self, text: str) -> list[str]:
        """将答案分解为独立的声明句。"""
        # 按句号分割
        sentences = [s.strip() for s in text.replace("。", ".").split(".") if s.strip()]
        # 排除太短或不完整的句子
        return [s for s in sentences if len(s) > 5]

    def _is_supported(self, claim: str, context: list[str]) -> bool:
        """检查声明是否被上下文支持（简化版）。"""
        claim_words = set(claim.lower().split())
        for ctx in context:
            ctx_words = set(ctx.lower().split())
            # 声明中 70% 以上的词出现在上下文中
            overlap = len(claim_words & ctx_words)
            if overlap / len(claim_words) > 0.7:
                return True
        return False
```

#### Answer Relevance（答案相关性）

评估答案是否与问题相关：

```python
class RelevanceEvaluator:
    """评估答案与问题的相关性。"""

    def evaluate(self, question: str, answer: str) -> float:
        """基于关键词覆盖的相关性评估。"""
        q_words = self._tokenize(question)
        a_words = self._tokenize(answer)

        if not q_words:
            return 1.0

        # 核心词覆盖率
        covered = sum(1 for w in q_words if w in a_words)
        return covered / len(q_words)

    def _tokenize(self, text: str) -> set[str]:
        stopwords = {"的", "了", "是", "在", "和", "有", "也", "就", "不", "都"}
        return {w for w in text if len(w) > 1 and w not in stopwords}
```

### 15.3.2 自动评估框架对比

```python
class RAGASEvaluator:
    """RAGAS 风格评估（简化实现）。"""

    def evaluate(self, dataset: list[dict]) -> dict:
        """
        数据集格式:
        [
            {"question": "...", "answer": "...",
             "contexts": [...], "ground_truth": "..."},
        ]
        """
        faithfulness_scores = []
        relevance_scores = []
        correctness_scores = []

        faithful = FaithfulnessEvaluator()
        relevant = RelevanceEvaluator()
        correct = AnswerCorrectness()

        for item in dataset:
            faithfulness_scores.append(
                faithful.evaluate(item["answer"], item["contexts"])
            )
            relevance_scores.append(
                relevant.evaluate(item["question"], item["answer"])
            )
            if "ground_truth" in item:
                correctness_scores.append(
                    correct.evaluate(item["answer"], item["ground_truth"])["f1_score"]
                )

        return {
            "faithfulness": sum(faithfulness_scores) / len(faithfulness_scores),
            "answer_relevancy": sum(relevance_scores) / len(relevance_scores),
            "answer_correctness": (
                sum(correctness_scores) / len(correctness_scores)
                if correctness_scores else None
            ),
            "sample_count": len(dataset),
        }
```

---

## 15.4 评估数据集构建

### 15.4.1 数据集要求

| 维度 | 要求 | 说明 |
|------|------|------|
| 多样性 | 覆盖各类型问题 | 事实型、推理型、总结型 |
| 代表性 | 与真实查询分布一致 | 基于日志分析采样 |
| 规模 | > 200 条 | 统计显著性要求 |
| 标注一致性 | Cohen's Kappa > 0.7 | 多人标注需校验 |

### 15.4.2 自动生成测试集

```python
class TestSetGenerator:
    """基于文档自动生成测试集。"""

    def __init__(self, llm):
        self.llm = llm

    def generate_from_documents(self, documents: list[str], count: int = 50) -> list[dict]:
        """从文档自动生成测试问题。"""
        test_set = []

        for doc in documents[:count]:
            # 生成事实性问题
            factual = self._generate_factual_questions(doc)
            test_set.extend(factual)

            # 生成推理性问题（需要多步推理）
            reasoning = self._generate_reasoning_questions(doc)
            test_set.extend(reasoning)

        return test_set

    def _generate_factual_questions(self, doc: str) -> list[dict]:
        """从文档中提取事实性问题。"""
        questions = []
        # 提取实体名 → 生成"XX是什么/谁/哪里"问题
        entities = self._extract_key_entities(doc)
        for name, etype in entities[:10]:
            q = self._make_question(name, etype)
            questions.append({
                "question": q,
                "ground_truth": self._extract_answer(doc, name),
                "type": "factual",
                "difficulty": "easy",
            })
        return questions
```

---

## 15.5人工评估

### 15.5.1 评估维度

| 维度 | 5 分标准 | 说明 |
|------|---------|------|
| 正确性 | 5=完全正确, 3=部分正确, 1=完全错误 | 核心指标 |
| 完整性 | 5=全覆盖, 3=部分覆盖, 1=严重遗漏 | 是否回答了所有方面 |
| 有用性 | 5=直接可用, 3=需修改, 1=不可用 | 实际业务价值 |
| 安全性 | 5=完全安全, 3=有风险表述, 1=有害 | 合规要求 |

### 15.5.2 标注平台接口

```python
@dataclass
class AnnotationTask:
    question_id: str
    question: str
    answer: str
    contexts: list[str]
    annotator: str = ""
    scores: dict = field(default_factory=dict)
    comment: str = ""

class AnnotationPlatform:
    """人工标注平台接口。"""

    def __init__(self):
        self.tasks: list[AnnotationTask] = []
        self.results: list[AnnotationTask] = []

    def create_batch(self, sessions: list[dict], annotator: str):
        for sess in sessions:
            self.tasks.append(AnnotationTask(
                question_id=sess["id"],
                question=sess["question"],
                answer=sess["answer"],
                contexts=sess.get("contexts", []),
                annotator=annotator,
            ))

    def submit_score(self, task_id: str, scores: dict, comment: str = ""):
        for task in self.tasks:
            if task.question_id == task_id:
                task.scores = scores
                task.comment = comment
                self.results.append(task)
                self.tasks.remove(task)
                break

    def compute_inter_annotator_agreement(self) -> float:
        """计算标注者间一致性（Cohen's Kappa）。"""
        # 简化实现
        if len(self.results) < 2:
            return 1.0
        return 0.85  # 模拟值
```

---

## 15.6 线上监控

### 15.6.1 监控指标

```python
class OnlineMonitor:
    """线上监控。"""

    def __init__(self):
        self.query_log: list[dict] = []
        self.feedback_log: list[dict] = []

    def log_query(self, question: str, latency_ms: float, token_count: int, success: bool):
        self.query_log.append({
            "timestamp": time.time(),
            "question": question,
            "latency_ms": latency_ms,
            "tokens": token_count,
            "success": success,
        })

    def log_feedback(self, question_id: str, rating: int, feedback_type: str):
        """记录用户反馈。

        feedback_type: thumbs_up / thumbs_down / report
        rating: 1-5
        """
        self.feedback_log.append({
            "timestamp": time.time(),
            "question_id": question_id,
            "rating": rating,
            "type": feedback_type,
        })

    def daily_summary(self) -> dict:
        """日摘要。"""
        total = len(self.query_log)
        if total == 0:
            return {}
        failed = sum(1 for q in self.query_log if not q["success"])
        latencies = [q["latency_ms"] for q in self.query_log]
        sorted_lat = sorted(latencies)
        total_tokens = sum(q["tokens"] for q in self.query_log)

        return {
            "total_queries": total,
            "error_rate": failed / total,
            "p50_latency": sorted_lat[total // 2],
            "p99_latency": sorted_lat[int(total * 0.99)],
            "total_tokens": total_tokens,
            "avg_tokens_per_query": total_tokens / total,
        }
```

---

## 15.7 评估最佳实践

1. **先离线后在线**：离线指标达标后再上线，避免直接伤害用户体验
2. **多维度评估**：检索质量和生成质量分开评估，不要只看最终答案
3. **自动 + 人工结合**：自动评估跑量大面广，人工评估把控质量上限
4. **渐进式发布**：A/B 测试评估新策略，用统计显著性判断胜负
5. **监控反馈闭环**：收集用户隐式和显式反馈，持续优化
6. **基线对比**：每次改进与基线（如传统 BM25）对比，量化提升

---

*下一章 [第 16 章：典型问题处理方法](ch16-troubleshooting.md)*
