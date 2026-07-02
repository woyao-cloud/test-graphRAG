# 第 17 章：跨团队协作与落地推进

## 17.1 RAG 项目的团队角色

RAG 系统落地不是纯工程任务，而是涉及算法、后端、产品和业务的多角色协作：

```
RAG 项目团队组成

产品经理 ─── 定义需求和验收标准
    │
算法工程师 ─── 模型选型、评测、RAG 策略
    │
后端工程师 ─── API 设计、性能优化、部署运维
    │
前端工程师 ─── 交互设计、流式展示、可视化
    │
业务方 ─── 数据提供、场景定义、效果验收
```

### 17.1.1 各团队的核心关注点

| 角色 | 关注 | 常用语言 | 常用工具 |
|------|------|---------|---------|
| 算法 | 召回率、生成质量、模型选型 | Python | LangChain, LlamaIndex |
| 后端 | 延迟、吞吐量、可用性 | Go/Java/Python | FastAPI, Flask |
| 前端 | 交互体验、响应速度、可视化 | TypeScript | React, Vue |
| 产品 | 需求明确、效果可衡量、ROI | - | 需求文档 |
| 业务 | 解决实际问题、使用门槛低 | - | Excel |

---

## 17.2 与算法团队协作

### 17.2.1 模型选型决策矩阵

RAG 系统中涉及的模型选择需要算法和工程团队共同决策：

| 决策项 | 算法团队负责 | 工程团队负责 |
|--------|------------|------------|
| Embedding 模型 | 评测：在业务数据上的召回率 | 部署：推理延迟和成本 |
| LLM 模型 | Prompt 优化、few-shot 设计 | API 封装、缓存、负载均衡 |
| Reranker | 排序质量评估 | 推理加速、批处理 |
| Chunk 策略 | 块大小调优实验 | 工程实现、索引构建 |

### 17.2.2 联合评测流程

```python
class JointEvaluation:
    """算法-工程联合评测。"""

    def __init__(self):
        self.algo_metrics = {}  # 算法关注的指标
        self.eng_metrics = {}   # 工程关注的指标

    def run_evaluation(self, model_config: dict, test_set: list[dict]) -> dict:
        """同时输出算法和工程维度的评测结果。"""
        results = []

        for case in test_set:
            start = time.time()

            # 执行推理
            answer = self._infer(model_config, case["question"])

            latency = (time.time() - start) * 1000

            results.append({
                "question": case["question"],
                "answer": answer,
                "latency_ms": latency,
                "tokens": self._count_tokens(answer),
            })

        # 算法维度
        self.algo_metrics = self._compute_algo_metrics(results, test_set)
        # 工程维度
        self.eng_metrics = self._compute_eng_metrics(results)

        return {
            "algorithm": self.algo_metrics,
            "engineering": self.eng_metrics,
            "tradeoff_analysis": self._analyze_tradeoffs(),
        }

    def _compute_algo_metrics(self, results: list, test_set: list) -> dict:
        """计算算法指标（准确率、召回率等）。"""
        return {
            "answer_correctness": 0.85,
            "faithfulness": 0.92,
            "retrieval_recall": 0.88,
        }

    def _compute_eng_metrics(self, results: list) -> dict:
        """计算工程指标（延迟、吞吐量等）。"""
        latencies = [r["latency_ms"] for r in results]
        sorted_l = sorted(latencies)
        n = len(sorted_l)
        return {
            "p50_latency_ms": sorted_l[n // 2],
            "p99_latency_ms": sorted_l[int(n * 0.99)],
            "avg_tokens": sum(r["tokens"] for r in results) / n,
        }
```

---

## 17.3 与产品团队协作

### 17.3.1 验收标准定义

RAG 系统的验收标准与传统软件不同，需要明确定义"好"的标准：

```text
验收标准示例（知识库问答场景）:

[P0 - 必须通过]
1. 常见问题（FAQs）的准确率 > 90%
2. 端到端延迟 P99 < 5s
3. 不出现有害内容

[P1 - 应该通过]
4. 复杂推理问题的准确率 > 70%
5. "我不知道"的比例 < 10%（即有答案的问题不拒答）
6. 答案引用准确率 > 85%

[P2 - 期望达到]
7. 首次回答即可解决用户问题 > 60%
8. 用户满意度评分 > 4.0/5.0
```

### 17.3.2 需求优先级排布

```python
class RequirementPrioritizer:
    """基于 Value/Effort 的需求优先级排布。"""

    def prioritize(self, features: list[dict]) -> list[dict]:
        """按价值/工作量比排序。"""
        for f in features:
            f["value_effort_ratio"] = f["business_value"] / max(f["effort"], 1)
            f["priority"] = self._assign_priority(f)

        return sorted(features, key=lambda x: -x["value_effort_ratio"])

    def _assign_priority(self, feature: dict) -> str:
        ratio = feature["value_effort_ratio"]
        if ratio > 3.0: return "P0"
        if ratio > 1.5: return "P1"
        if ratio > 0.5: return "P2"
        return "P3"
```

---

## 17.4 与后端团队协作

### 17.4.1 API 契约

```python
# API 契约示例（OpenAPI 风格）

RAG_QUERY_API = {
    "endpoint": "POST /api/v1/rag/query",
    "request": {
        "question": "恒瑞医药生产哪些药品？",
        "options": {
            "mode": "auto",           # auto/local/global/drift
            "top_k": 5,
            "temperature": 0.3,
            "stream": True,            # 是否流式返回
        }
    },
    "response": {
        "answer": "...",
        "sources": [
            {"title": "doc1", "relevance": 0.92, "content": "...片段..."},
        ],
        "metrics": {
            "latency_ms": 2340,
            "tokens_used": 1250,
        },
    },
    "error_codes": {
        400: "无效请求参数",
        429: "请求频率超限",
        503: "LLM 服务暂时不可用",
    },
}
```

### 17.4.2 性能压测

```python
class LoadTester:
    """性能压测。"""

    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self.results = []

    def run_load_test(self, qps: int, duration_sec: int = 30):
        """执行负载测试。"""
        import concurrent.futures
        import threading

        stop_event = threading.Event()
        latencies = []
        errors = 0

        def worker():
            while not stop_event.is_set():
                start = time.time()
                try:
                    # 模拟请求
                    time.sleep(0.1)
                    latency = (time.time() - start) * 1000
                    latencies.append(latency)
                except Exception:
                    nonlocal errors
                    errors += 1

        # 启动 worker
        workers = min(qps * 2, 50)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(worker) for _ in range(workers)]
            time.sleep(duration_sec)
            stop_event.set()

        # 计算指标
        sorted_l = sorted(latencies)
        n = len(sorted_l)
        return {
            "total_requests": n,
            "actual_qps": n / duration_sec,
            "error_rate": errors / n if n > 0 else 0,
            "p50_latency_ms": sorted_l[n // 2] if n > 0 else 0,
            "p99_latency_ms": sorted_l[int(n * 0.99)] if n > 0 else 0,
            "max_latency_ms": max(latencies) if latencies else 0,
        }
```

---

## 17.5 与前端团队协作

### 17.5.1 流式输出

```python
# 后端流式接口（SSE）
class StreamResponse:
    """SSE 流式响应。"""

    @staticmethod
    async def generate(question: str, retriever, llm):
        """流式生成。"""
        # Step 1: 检索阶段
        yield {"type": "status", "content": "正在检索..."}
        chunks = await retriever.search(question)

        yield {"type": "status", "content": f"找到 {len(chunks)} 个相关文档"}

        # Step 2: 生成阶段（逐 token 流式返回）
        yield {"type": "stream_start"}
        async for token in llm.generate_stream(question, chunks):
            yield {"type": "token", "content": token}

        yield {"type": "stream_end"}

        # Step 3: 返回引用来源
        yield {"type": "sources", "content": [
            {"title": c.title, "relevance": c.score}
            for c in chunks[:3]
        ]}
```

### 17.5.2 前端展示建议

```text
┌──────────────────────────────────────────────┐
│   Q: 恒瑞医药生产哪些药品？                      │
│                                               │
│   ┌──────────────────────────────────────────┐│
│   │  恒瑞医药生产以下抗肿瘤药品：               ││
│   │  • 注射用紫杉醇 — 微管抑制剂              ││
│   │  • 奥沙利铂 — 铂类抗肿瘤药 [1]            ││
│   │  • 卡培他滨 — 口服抗肿瘤药 [2]            ││
│   │                                          ││
│   │  [参考文献]                               ││
│   │  [1] 恒瑞医药产品目录 (2024) · 相关性 0.92││
│   │  [2] 卡培他滨说明书 · 相关性 0.85         ││
│   └──────────────────────────────────────────┘│
│                                               │
│   反馈:  👍 👎  📋 举报                       │
│   延迟: 2.3s  ·  共 3 个来源                  │
└──────────────────────────────────────────────┘
```

---

## 17.6 落地路线图

### 17.6.1 四阶段推进

```
阶段 1: POC（技术验证） — 2-4 周
  目标: 在一个有限场景验证 RAG 可行性
  产出: 可演示的 Demo + 核心指标基线
  关键行动:
    • 选择一个业务场景（如客服问答）
    • 用 100-500 条文档快速搭建
    • 产出评测报告

阶段 2: 灰度（小范围上线） — 4-6 周
  目标: 在小范围用户中验证效果
  产出: A/B 测试数据 + 用户体验反馈
  关键行动:
    • 接入 10-20% 真实流量
    • 收集反馈和日志
    • 迭代优化

阶段 3: 全量（完整上线） — 4-8 周
  目标: 全量用户覆盖
  产出: 生产环境 + 监控告警
  关键行动:
    • 容量规划和压测
    • 部署高可用架构
    • 建立监控体系

阶段 4: 持续优化 — 持续进行
  目标: 持续提升效果和效率
  产出: 持续优化的闭环
  关键行动:
    • 定期模型升级
    • 知识库持续更新
    • 用户反馈闭环
```

---

*下一章 [第 18 章：端到端 RAG 系统实战](ch18-end-to-end-rag.md)*
