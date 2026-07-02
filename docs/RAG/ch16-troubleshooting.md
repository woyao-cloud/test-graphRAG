# 第 16 章：典型问题处理方法

## 16.1 RAG 系统的常见故障模式

实际生产中的 RAG 系统，问题通常可以归纳为几种典型模式：

```
用户投诉: "这个回答不对"

诊断路径:
  是检索结果有问题吗？
  ├─ 是 → 检索召回率低 → 16.4 低召回处理
  ├─ 是 → 检索结果不相关 → 16.1 Query 理解与改写
  │
  是生成结果有问题吗？
  ├─ 是 → 答案包含幻觉 → 16.5 幻觉抑制
  ├─ 是 → 答案遗漏关键信息 → 16.3 上下文窗口管理
  │
  是分块策略有问题吗？
  └─ 是 → 文档块粒度不合适 → 16.2 Chunk 优化
```

---

## 16.1 Query 理解与改写

### 16.1.1 拼写纠错

用户查询中常见的拼写问题：

```python
class SpellCorrector:
    """拼写纠错（基于编辑距离和领域词典）。"""

    def __init__(self):
        # 领域词典
        self.dictionary = {
            "恒瑞医药", "华海药业", "国药控股", "齐鲁制药",
            "紫杉醇", "奥沙利铂", "卡培他滨", "顺铂",
            "非小细胞肺癌", "乳腺癌", "结直肠癌", "卵巢癌",
        }

    def correct(self, query: str) -> str:
        """对查询进行拼写纠错。"""
        tokens = query.split()
        corrected = []
        for token in tokens:
            if token in self.dictionary:
                corrected.append(token)
            else:
                best = self._find_closest(token)
                corrected.append(best if best else token)
        return " ".join(corrected)

    def _find_closest(self, word: str, max_dist: int = 2) -> Optional[str]:
        """在词典中找编辑距离最近的词。"""
        best_word, best_dist = None, float("inf")
        for dict_word in self.dictionary:
            dist = self._edit_distance(word, dict_word)
            if dist < best_dist and dist <= max_dist:
                best_dist = dist
                best_word = dict_word
        return best_word

    def _edit_distance(self, a: str, b: str) -> int:
        """Levenshtein 编辑距离。"""
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1): dp[i][0] = i
        for j in range(n + 1): dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = 0 if a[i - 1] == b[j - 1] else 1
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
        return dp[m][n]
```

### 16.1.2 Query 扩展（Query Expansion）

通过添加同义词和相关词来扩大召回范围：

```python
class QueryExpander:
    """查询扩展。"""

    def __init__(self):
        # 领域同义词表
        self.synonyms = {
            "恒瑞医药": ["恒瑞制药", "江苏恒瑞"],
            "紫杉醇": ["Paclitaxel", "紫杉醇注射液"],
            "卡培他滨": ["Capecitabine", "希罗达"],
            "非小细胞肺癌": ["NSCLC", "肺鳞癌", "肺腺癌"],
            "药品": ["药物", "药品", "药剂"],
            "生产": ["制造", "研发", "生产"],
            "医院": ["医疗机构", "三甲医院", "医院"],
        }

    def expand(self, query: str) -> list[str]:
        """生成扩展查询列表。"""
        expanded_queries = [query]

        # 为查询中的每个词添加同义词
        for word, syns in self.synonyms.items():
            if word in query:
                for syn in syns:
                    if syn not in query:
                        expanded_queries.append(
                            query.replace(word, syn)
                        )

        return expanded_queries[:5]  # 限制最多 5 个变体
```

### 16.1.3 意图识别

```python
class IntentClassifier:
    """查询意图分类。"""

    INTENTS = {
        "factual": {
            "description": "事实性查询",
            "patterns": ["什么", "谁", "哪里", "多少", "什么时候"],
            "action": "向量检索 + 精确检索"
        },
        "reasoning": {
            "description": "推理型查询",
            "patterns": ["为什么", "如何", "怎样", "什么原因"],
            "action": "知识图谱 + 多步检索"
        },
        "comparison": {
            "description": "对比型查询",
            "patterns": ["对比", "比较", "区别", "vs", "versus"],
            "action": "分别检索 + 对比生成"
        },
        "summary": {
            "description": "总结型查询",
            "patterns": ["总结", "概述", "汇总", "列出所有"],
            "action": "全局搜索 + 社区摘要"
        },
    }

    def classify(self, query: str) -> str:
        """分类查询意图。"""
        for intent, config in self.INTENTS.items():
            if any(p in query for p in config["patterns"]):
                return intent
        return "factual"  # 默认
```

---

## 16.2 Chunk 优化

### 16.2.1 解决"分块粒度矛盾"

```
矛盾: 小块 → 检索精确但上下文不足 → LLM 无法理解
      大块 → 上下文丰富但检索噪音大 → 召回不精准

解决方案: 动态分块 + Small-to-Big
```

### 16.2.2 动态分块策略

```python
class DynamicChunker:
    """动态分块器。"""

    def __init__(self):
        # 自然断点（优先级从高到低）
        self.breakpoints = [
            ("\n## ", "h2"),
            ("\n### ", "h3"),
            ("\n\n", "paragraph"),
            ("\n", "line"),
            ("。", "sentence"),
        ]

    def chunk(self, text: str, min_size: int = 200, max_size: int = 800) -> list[dict]:
        """动态分块：尽量在自然断点处分块。"""
        chunks = []
        start = 0

        while start < len(text):
            # 在 [min_size, max_size] 范围内找最近的断点
            end = min(start + max_size, len(text))
            if end == len(text):
                chunks.append({"text": text[start:end], "start": start, "end": end})
                break

            # 从 end 往回找自然断点
            split_pos = self._find_breakpoint(text, start + min_size, end)

            chunks.append({"text": text[start:split_pos], "start": start, "end": split_pos})
            start = split_pos

        return chunks

    def _find_breakpoint(self, text: str, min_pos: int, max_pos: int) -> int:
        """在 [min_pos, max_pos] 范围内找最佳断点。"""
        # 按优先级检查断点
        for marker, _ in self.breakpoints:
            pos = text.rfind(marker, min_pos, max_pos)
            if pos != -1:
                return pos + len(marker)
        # 没有自然断点，在 max_pos 处强制分块
        return max_pos
```

### 16.2.3 重叠策略

```python
class OverlapChunker:
    """带重叠的分块器。"""

    def chunk(self, text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
        """按固定大小分块，带重叠。"""
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
            # 下一次的起始位置去掉重叠
            start += chunk_size - overlap
            if start >= len(text):
                break
        return chunks
```

### 16.2.4 分块参数调优

| 场景 | 最佳块大小 | 重叠 | 策略 |
|------|-----------|------|------|
| 技术文档 | 500-800 tokens | 50-100 | 按章节/标题分块 |
| 新闻文章 | 300-500 tokens | 50 | 按段落分块 |
| 医疗报告 | 200-400 tokens | 30-50 | 按句子分块 + Small-to-Big |
| 代码文件 | 100-300 lines | 10-20 lines | 按函数/类分块 |
| 法律合同 | 1000-2000 tokens | 100-200 | 按条款分块 |

---

## 16.3 上下文窗口管理

### 16.3.1 滑动窗口

```python
class SlidingWindowManager:
    """滑动窗口管理。"""

    def __init__(self, max_context_tokens: int = 4000):
        self.max_tokens = max_context_tokens

    def build_context(self, chunks: list[dict], query: str, max_chunks: int = 15) -> str:
        """构建上下文窗口，优先排布高相关性的块。"""
        # 已经按相关性排序的 chunks
        context_parts = []
        used_tokens = 0

        # 1. 先放入综述性内容（如果有）
        summary = self._get_summary(chunks)
        if summary:
            summary_tokens = self._count_tokens(summary)
            if used_tokens + summary_tokens <= self.max_tokens:
                context_parts.append(("summary", summary))
                used_tokens += summary_tokens

        # 2. 按相关性从高到低放入结果块
        for chunk in chunks[:max_chunks]:
            tokens = self._count_tokens(chunk["text"])
            if used_tokens + tokens > self.max_tokens:
                # 截断最后一个块
                remaining = self.max_tokens - used_tokens
                if remaining > 50:
                    truncated = self._truncate(chunk["text"], remaining)
                    context_parts.append(("chunk", truncated))
                break
            context_parts.append(("chunk", chunk["text"]))
            used_tokens += tokens

        return "\n\n".join(text for _, text in context_parts)

    def _count_tokens(self, text: str) -> int:
        return len(text) // 2  # 中文：约 2 字符 = 1 token

    def _truncate(self, text: str, max_tokens: int) -> str:
        max_chars = max_tokens * 2
        return text[:max_chars] + "..." if len(text) > max_chars else text
```

---

## 16.4 低召回处理

### 16.4.1 诊断步骤

```python
class RecallDiagnoser:
    """低召回诊断器。"""

    def diagnose(self, question: str, results: list[dict]) -> dict:
        issues = []

        # 检查 1: 是否有结果
        if len(results) == 0:
            issues.append("zero_results")
        elif len(results) < 3:
            issues.append("too_few_results")

        if issues:
            return {"diagnosis": issues}

        # 检查 2: 结果是否相关
        relevance_scores = [
            self._compute_relevance(question, r["text"])
            for r in results
        ]
        avg_relevance = sum(relevance_scores) / len(relevance_scores)

        if avg_relevance < 0.3:
            issues.append("irrelevant_results")
        elif avg_relevance < 0.6:
            issues.append("partially_relevant")

        # 检查 3: 是否覆盖问题中的实体
        question_entities = self._extract_entities(question)
        result_text = " ".join(r["text"] for r in results)
        missing = [e for e in question_entities if e not in result_text]
        if missing:
            issues.append(f"missing_entities: {missing}")

        return {"diagnosis": issues, "avg_relevance": avg_relevance}
```

### 16.4.2 多路补充策略

```python
class RecallEnhancer:
    """召回增强器。"""

    def enhance(self, question: str, initial_results: list) -> list:
        """多路补充检索。"""
        enhanced_results = list(initial_results)

        # 策略 1: Query 改写
        rewritten = self._rewrite_query(question)
        enhanced_results.extend(self._search(rewritten))

        # 策略 2: 子查询分解
        sub_queries = self._decompose(question)
        for sq in sub_queries[:3]:
            enhanced_results.extend(self._search(sq))

        # 策略 3: SSI（取消块大小限制）
        enhanced_results.extend(self._search_raw(question))

        # 去重 + 重排
        return self._dedup_and_rerank(question, enhanced_results)[:10]
```

---

## 16.5 幻觉抑制

### 16.5.1 幻觉类型

```
类型 1 — 内在幻觉（Intrinsic）:
  答案包含与上下文矛盾的信息
  例: 上下文说"紫杉醇治疗非小细胞肺癌"，答案说"紫杉醇治疗所有癌症"

类型 2 — 外在幻觉（Extrinsic）:
  答案包含上下文完全没有的信息
  例: 上下文没有提到 FDA，但答案说"该药已获得 FDA 批准"

类型 3 — 事实扭曲（Factual Distortion）:
  答案部分正确，但关键细节错误
  例: 上下文说"2019年获批"，答案说"2020年获批"
```

### 16.5.2 幻觉检测器

```python
class HallucinationDetector:
    """幻觉检测器。"""

    def detect(self, answer: str, contexts: list[str]) -> dict:
        """检测答案中的幻觉。"""
        claims = self._extract_claims(answer)
        context_text = " ".join(contexts)

        hallu_claims = []
        safe_claims = []

        for claim in claims:
            # 检查声明是否能被上下文支持
            support_score = self._compute_support(claim, context_text)
            if support_score < 0.3:
                hallu_claims.append({"claim": claim, "score": support_score})
            else:
                safe_claims.append({"claim": claim, "score": support_score})

        return {
            "total_claims": len(claims),
            "hallucinated": len(hallu_claims),
            "hallucination_rate": len(hallu_claims) / len(claims) if claims else 0,
            "hallucinated_claims": hallu_claims[:5],
            "safe_claims": safe_claims,
        }

    def _extract_claims(self, text: str) -> list[str]:
        """将答案拆分为原子声明。"""
        sentences = [s.strip() for s in text.replace("。", ".").split(".") if s.strip()]
        return [s for s in sentences if len(s) > 5]

    def _compute_support(self, claim: str, context: str) -> float:
        """计算声明被上下文支持的程度。"""
        claim_words = set(claim.lower().split())
        context_words = set(context.lower().split())
        if not claim_words:
            return 1.0
        overlap = len(claim_words & context_words)
        return overlap / len(claim_words)
```

### 16.5.3 幻觉抑制策略

```python
class HallucinationSuppressor:
    """幻觉抑制器。"""

    def suppress(self, answer: str, contexts: list[str]) -> str:
        """在答案中标记或移除幻觉内容。"""
        detector = HallucinationDetector()
        result = detector.detect(answer, contexts)

        if result["hallucination_rate"] < 0.2:
            return answer  # 幻觉率低，直接返回

        # 策略 1: 移除幻觉声明
        cleaned_sentences = []
        for item in result["safe_claims"]:
            cleaned_sentences.append(item["claim"])
        if cleaned_sentences:
            return "。".join(cleaned_sentences) + "。"

        # 策略 2: 添加不确定性声明
        return answer + "\n\n> 注意：以上部分信息在提供的文档中未找到直接支持，请核实。"

    def add_citations(self, answer: str, chunks: list[dict]) -> str:
        """为答案添加引用标记。"""
        # 将答案中的声明与源文档块关联
        cited_sentences = []
        for sent in answer.replace("。", ".").split("."):
            sent = sent.strip()
            if not sent:
                continue
            # 找到最相关的源块
            best_source, best_score = None, 0
            for chunk in chunks:
                overlap = len(set(sent) & set(chunk["text"]))
                if overlap > best_score:
                    best_score = overlap
                    best_source = chunk.get("source", "unknown")
            if best_source:
                cited_sentences.append(f"{sent}[{best_source}]")
            else:
                cited_sentences.append(sent)
        return "。".join(cited_sentences)
```

---

## 16.6 综合诊断流程

当 RAG 系统出现问题时，按以下流程排查：

```python
class RAGDiagnosticPipeline:
    """RAG 问题诊断流水线。"""

    def diagnose(self, question: str, answer: str, contexts: list[dict]) -> dict:
        issues = []

        # 1. 检查检索质量
        retrieval_diag = self._check_retrieval(question, contexts)
        if retrieval_diag["has_issue"]:
            issues.extend(retrieval_diag["issues"])

        # 2. 检查上下文完整性
        context_diag = self._check_context(question, contexts)
        if context_diag["has_issue"]:
            issues.extend(context_diag["issues"])

        # 3. 检查生成质量
        generation_diag = self._check_generation(question, answer, contexts)
        if generation_diag["has_issue"]:
            issues.extend(generation_diag["issues"])

        return {
            "has_issues": len(issues) > 0,
            "issues": issues,
            "severity": "high" if len(issues) > 2 else "medium" if issues else "low",
            "recommendations": self._get_recommendations(issues),
        }
```

---

## 16.7 各问题类型速查

| 症状 | 可能原因 | 快速检查 | 修复方案 |
|------|---------|---------|---------|
| 答案完全错误 | 检索不相关或幻觉 | 检查检索结果的 relevance score | Query 改写 + 多路补充 |
| 答案部分正确 | 关键信息被截断 | 检查上下文中是否含正确答案 | 增大 chunk size 或 Small-to-Big |
| 答案太笼统 | 检索只拿到摘要层 | 检查检索的层级信息 | 降低检索层级 Level |
| 答案太长啰嗦 | 上下文过多 | 检查上下文 Token 数 | 滑动窗口 + 截断 |
| 答案包含"我觉得" | 模型在自由发挥 | 检查上下文相关性 | 降低 temperature + Citation |
| 回答"不知道" | 检索结果太少 | 检查召回数量 | 降低 score threshold |
| 回答慢 | 上下文太大或模型慢 | 检查 LLM 延迟 | 缓存 + 缩减上下文 |

---

*下一章 [第 17 章：跨团队协作与落地推进](ch17-collaboration.md)*
