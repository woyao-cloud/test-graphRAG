# 第11章 RAG工程实践：跨团队协作与系统架构

## 11.1 引言

RAG系统的构建不仅是技术挑战，更是组织协作的考验。一个生产级RAG系统涉及算法研究、产品设计、后端开发、前端交互和数据工程等多个专业领域的协同工作。本章从工程实践角度出发，系统性地介绍RAG项目中的团队角色分工、项目生命周期管理和工程技术规范。

与单独的技术选型不同，跨团队协作的关键在于建立清晰的角色边界、高效的沟通机制和标准化的工程流程。我们以一个典型的RAG产品团队为例，详细阐述每个环节的最佳实践。

---

## 11.2 团队角色与职责

一个完整的RAG产品团队通常包含以下六个核心角色。值得注意的是，在小型团队中，一个人可能承担多个角色；而在大型组织中，每个角色可能由多人组成的子团队承担。

### 11.2.1 算法工程师

算法工程师是RAG系统的"大脑"，负责核心算法能力的构建和优化。

**核心职责：**

1. **模型选型与评估**
   - 评估不同嵌入模型（bge-m3、text-embedding-3-small、e5）在垂直领域的表现
   - 对比生成模型的推理质量（DeepSeek、GPT-4o、Claude、Llama）
   - 建立标准化的模型评估基准（Benchmark）

2. **检索策略优化**
   - 设计并实验不同的检索策略（向量搜索、关键词搜索、混合搜索）
   - 调优检索参数（Top-K、相似度阈值、分块策略）
   - 实现高级检索技术（Query Rewriting、HyDE、RAG-Fusion）

3. **提示工程（Prompt Engineering）**
   - 设计系统提示词和任务提示词
   - 优化少样本示例（Few-shot Examples）
   - 实验不同的提示策略（Chain-of-Thought、ReAct）

4. **质量评估**
   - 建立评估指标和评估流程
   - 进行Bad Case分析
   - 持续优化模型表现

**技能要求：**
```
- 扎实的机器学习/深度学习基础
- LLM和Embedding模型的深入理解
- 检索算法和排序算法的知识
- Python编程和数据科学能力
- 实验设计和统计分析能力
```

**典型工作交付物：**
```
- 模型评估报告（含对比实验数据）
- 检索策略设计方案
- 提示词模板库
- 算法性能监控报表
```

### 11.2.2 产品经理

产品经理是团队的"导航仪"，负责将技术能力转化为用户价值。

**核心职责：**

1. **需求分析与定义**
   - 深入理解用户场景和痛点
   - 定义产品功能和优先级
   - 编写产品需求文档（PRD）

2. **用户体验设计**
   - 定义问答交互流程
   - 设计检索结果的展示方式
   - 处理边缘情况（无结果、歧义问题、多轮对话）

3. **质量验收**
   - 制定质量标准（准确率、召回率、响应时间）
   - 组织UAT（用户验收测试）
   - 收集和分析用户反馈

4. **迭代规划**
   - 制定产品迭代路线图
   - 平衡新功能开发和质量优化
   - 管理版本发布

**典型工作交付物：**
```
- 产品需求文档（PRD）
- 交互原型和UI设计稿
- 用户反馈分析报告
- 产品迭代路线图
```

### 11.2.3 后端工程师

后端工程师是系统的"骨架"，负责构建稳定、高性能的服务架构。

**核心职责：**

1. **API设计与开发**
   - 设计RESTful API接口
   - 实现流式响应（Server-Sent Events）
   - 接口版本管理和兼容性

2. **性能优化**
   - 连接池管理（数据库、LLM服务）
   - 缓存策略设计（多级缓存、缓存预热）
   - 异步处理和并发控制

3. **系统可靠性**
   - 熔断和降级机制
   - 限流和负载均衡
   - 容错和重试策略

4. **部署运维**
   - CI/CD流水线搭建
   - 容器化和编排（Docker + Kubernetes）
   - 日志采集和监控告警

**技能要求：**
```
- 精通至少一种后端语言（Python/Go/Java）
- 分布式系统设计经验
- 数据库（PostgreSQL/Redis/Elasticsearch）优化经验
- DevOps工具链（Docker/K8s/CICD）
- API设计和性能优化经验
```

### 11.2.4 前端工程师

前端工程师负责用户与RAG系统的交互界面。

**核心职责：**

1. **对话界面开发**
   - 实现消息流式展示（打字机效果）
   - 支持富文本渲染（Markdown、代码高亮、表格）
   - 多轮对话管理

2. **检索结果可视化**
   - 展示引用来源（来源文档、片段高亮）
   - 知识图谱可视化（实体关系图）
   - 置信度展示和不确定性提示

3. **交互体验优化**
   - 首屏加载优化
   - 输入联想和建议
   - 移动端适配

**技能要求：**
```
- 精通React/Vue等现代前端框架
- WebSocket和SSE实时通信经验
- 数据可视化库使用经验（D3.js、ECharts）
- 性能优化意识
```

### 11.2.5 数据工程师

数据工程师是系统的"燃料"，负责数据的收集、清洗和管理。

**核心职责：**

1. **数据采集**
   - 设计数据爬取和导入流程
   - 支持多种数据源（数据库、API、文件上传）
   - 增量数据同步

2. **数据清洗**
   - 文档解析（PDF、Word、HTML）
   - 文本去重和标准化
   - 敏感信息脱敏

3. **数据标注**
   - 标注任务设计（实体标注、相关性标注）
   - 标注平台搭建
   - 标注质量审核

4. **数据管道维护**
   - ETL流水线搭建和维护
   - 数据质量监控
   - 数据版本管理

**技能要求：**
```
- 数据处理框架使用经验（Spark/Flink）
- 数据库和大数据存储技术
- ETL管道设计经验
- 数据质量意识
```

### 11.2.6 角色协作矩阵

| 活动 | 算法 | 产品 | 后端 | 前端 | 数据 |
|------|------|------|------|------|------|
| 需求评审 | 参与 | 主导 | 参与 | 参与 | 参与 |
| 模型选型 | 主导 | 参与 | 评审 | — | — |
| API设计 | 参与 | 参与 | 主导 | 参与 | — |
| 数据标注 | 指导 | 参与 | — | — | 主导 |
| 性能测试 | 参与 | — | 主导 | — | — |
| UAT测试 | — | 主导 | 支持 | 支持 | — |
| 线上监控 | 参与 | 参与 | 主导 | 参与 | 参与 |

---

## 11.3 项目生命周期

RAG项目的生命周期通常遵循"快速验证→持续迭代"的模式。与传统的瀑布式开发不同，RAG项目更适合敏捷开发方法，因为LLM的能力边界和用户期望都在快速变化。

### 11.3.1 需求阶段

**目标：** 明确业务需求，定义成功标准。

**关键活动：**
```
1. 业务场景调研
   - 用户访谈：了解目标用户的工作流程和痛点
   - 竞品分析：研究同类产品的优劣势
   - 可行性评估：评估RAG方案是否适合当前场景

2. 需求优先级排序
   - 核心功能（P0）：检索、问答、引用展示
   - 重要功能（P1）：多轮对话、文档管理、反馈机制
   - 增强功能（P2）：知识图谱、个性化推荐、数据看板

3. 技术方案选型
   - 嵌入模型选择：考虑领域适配性、成本、延迟
   - 向量数据库选择：Milvus/Pinecone/Weaviate/Qdrant
   - LLM选择：考虑推理能力、成本、响应速度
   - 部署方案：云端/私有化/混合部署
```

**成功标准：**
```
- 完成3-5个核心用户场景的定义
- 选定技术栈并获得团队共识
- 明确项目交付时间线
```

### 11.3.2 POC阶段

**目标：** 快速搭建最小可行系统，验证技术方案的可行性。

**时间线：** 2-4周

```python
# POC阶段的核心代码结构示例
class RAGPOC:
    """快速原型验证"""
    
    def __init__(self):
        self.embedding_model = self._load_embedding_model()
        self.llm = self._load_llm()
        self.vector_store = self._init_vector_store()
    
    def _load_embedding_model(self):
        """加载嵌入模型（POC阶段使用轻量模型）"""
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer('BAAI/bge-small-zh-v1.5')
    
    def _init_vector_store(self):
        """初始化向量存储（POC阶段使用内存存储）"""
        import faiss
        import numpy as np
        dimension = 512  # bge-small维度
        return faiss.IndexFlatIP(dimension)  # 内积索引
    
    def ingest(self, documents: List[str]):
        """文档入库"""
        embeddings = self.embedding_model.encode(documents)
        self.vector_store.add(embeddings)
        self.documents = documents
    
    def query(self, question: str) -> str:
        """检索并生成"""
        q_embedding = self.embedding_model.encode([question])
        scores, indices = self.vector_store.search(q_embedding, k=3)
        
        context = "\n".join([
            self.documents[i] for i in indices[0]
        ])
        
        prompt = f"基于以下信息回答问题：\n{context}\n\n问题：{question}"
        return self.llm.generate(prompt)
```

**POC评估清单：**
```
□ 检索质量：5个典型查询的人工评估通过
□ 响应时间：单次查询 < 3秒
□ 覆盖场景：覆盖至少3个核心用户场景
□ 技术风险：识别并记录所有技术风险点
```

### 11.3.3 评估阶段

**目标：** 系统性地评估RAG系统的各项指标，建立质量基线。

**关键活动：**

```python
# 评估阶段的核心流程
class RAGEvaluator:
    """RAG系统评估器"""
    
    def __init__(self):
        self.test_cases = []
        self.metrics = {}
    
    def build_test_set(self, questions: List[str], golden_answers: List[str]):
        """
        构建测试集。
        
        测试集要求：
        - 覆盖所有核心场景
        - 包含正例和反例
        - 包含边界情况（模糊问题、无答案问题）
        """
        self.test_cases = [
            {"question": q, "golden": a}
            for q, a in zip(questions, golden_answers)
        ]
    
    def evaluate(self, rag_system) -> dict:
        """执行评估"""
        results = []
        for case in self.test_cases:
            answer = rag_system.query(case["question"])
            
            # 计算各项指标
            metrics = self._compute_metrics(
                case["question"],
                answer,
                case["golden"],
            )
            results.append(metrics)
        
        return self._aggregate_results(results)
    
    def _compute_metrics(self, question, answer, golden) -> dict:
        """计算单条测试的指标"""
        return {
            "accuracy": self._check_accuracy(answer, golden),
            "hallucination": self._check_hallucination(answer, golden),
            "completeness": self._check_completeness(answer, golden),
            "latency": self._measure_latency(),
        }
```

**评估通过标准：**
```
- 准确率 >= 85%（基于人工评估）
- 幻觉率 <= 5%
- P95响应时间 <= 5秒
- 核心场景覆盖率 100%
```

### 11.3.4 优化阶段

**目标：** 基于评估结果进行针对性优化。

**优化循环：**
```
1. 分析Bad Case → 2. 定位根因 → 3. 设计优化方案 → 4. 实施优化 → 5. 验证效果
```

```python
class BadCaseAnalyzer:
    """Bad Case分析器"""
    
    def __init__(self):
        self.error_categories = {
            "retrieval": {
                "missing_context": "检索未召回相关文档",
                "wrong_context": "检索到不相关的文档",
                "insufficient_context": "检索结果不完整",
            },
            "generation": {
                "hallucination": "生成了上下文不支持的信息",
                "incomplete": "遗漏了关键信息",
                "misunderstanding": "误解了问题意图",
            },
            "integration": {
                "timeout": "LLM调用超时",
                "format_error": "输出格式错误",
                "contradiction": "回答自相矛盾",
            },
        }
    
    def analyze(self, bad_case: dict) -> dict:
        """分析Bad Case的根因"""
        question = bad_case["question"]
        answer = bad_case["answer"]
        context = bad_case.get("context", [])
        golden = bad_case.get("golden_answer")
        
        analysis = {
            "question": question,
            "categories": [],
            "root_causes": [],
            "suggestions": [],
        }
        
        # 检查检索问题
        if not context:
            analysis["categories"].append("retrieval")
            analysis["root_causes"].append("missing_context")
            analysis["suggestions"].append("增加检索Top-K或优化分块策略")
        
        # 检查生成问题
        if golden and self._has_hallucination(answer, golden):
            analysis["categories"].append("generation")
            analysis["root_causes"].append("hallucination")
            analysis["suggestions"].append("降低LLM temperature或加强上下文约束")
        
        return analysis
```

### 11.3.5 上线阶段

**目标：** 平稳地将系统部署到生产环境。

**上线检查清单：**
```
□ 性能基准测试通过
   - 单机QPS >= 10
   - P99延迟 <= 10秒
   - 错误率 <= 0.1%

□ 安全审计完成
   - 提示词注入防护
   - 敏感信息过滤
   - 访问控制配置

□ 监控系统就绪
   - 关键指标仪表盘
   - 告警规则配置
   - 日志收集正常

□ 回滚方案准备
   - 数据库备份完成
   - 历史版本可回滚
   - 灰度发布策略

□ 文档完善
   - API文档更新
   - 运维手册完成
   - 故障处理SOP
```

### 11.3.6 A/B测试

A/B测试是RAG系统持续优化的核心手段。与传统的A/B测试不同，RAG系统的A/B测试需要特别关注生成内容的不可控性：

```python
class RAGABTest:
    """RAG系统的A/B测试框架"""
    
    def __init__(self, redis_client):
        self.redis = redis_client
    
    async def route_request(self, user_id: str, query: str) -> str:
        """路由用户请求到不同的实验组"""
        experiment = await self._get_user_experiment(user_id)
        
        if experiment == "A":
            # 对照组：基线版本
            return await self._baseline_pipeline(query)
        else:
            # 实验组：新策略
            return await self._experiment_pipeline(query)
    
    async def record_feedback(self, user_id: str, rating: int):
        """记录用户反馈"""
        experiment = await self._get_user_experiment(user_id)
        await self.redis.lpush(
            f"ab_test:{experiment}:ratings",
            rating,
        )
    
    async def analyze_results(self) -> dict:
        """分析A/B测试结果"""
        ratings_a = await self.redis.lrange(
            "ab_test:A:ratings", 0, -1
        )
        ratings_b = await self.redis.lrange(
            "ab_test:B:ratings", 0, -1
        )
        
        return {
            "A": {
                "avg_rating": mean(ratings_a),
                "sample_size": len(ratings_a),
            },
            "B": {
                "avg_rating": mean(ratings_b),
                "sample_size": len(ratings_b),
            },
        }
```

---

## 11.4 工程实践规范

### 11.4.1 API设计

RAG系统的API设计需要在功能性、性能和可维护性之间取得平衡。

#### 11.4.1.1 RESTful API设计

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
import time
import logging

app = FastAPI(title="GraphRAG API", version="2.0.0")

# ============ 请求/响应模型 ============

class QueryRequest(BaseModel):
    """查询请求"""
    query: str = Field(..., min_length=1, max_length=2000,
                       description="用户查询文本")
    search_type: str = Field("auto", pattern="^(auto|local|global|drift)$",
                             description="检索策略类型")
    top_k: int = Field(10, ge=1, le=50,
                       description="检索结果数量")
    stream: bool = Field(False,
                         description="是否流式输出")
    conversation_id: Optional[str] = Field(None,
        description="会话ID，用于多轮对话上下文")
    user_id: Optional[str] = Field(None,
        description="用户ID，用于个性化")

class QueryResponse(BaseModel):
    """查询响应"""
    answer: str = Field(..., description="生成的回答")
    sources: List[dict] = Field(default_factory=list,
                                description="引用来源")
    search_type: str = Field(..., description="实际使用的检索策略")
    latency_ms: int = Field(..., description="总响应时间（毫秒）")
    token_usage: dict = Field(default_factory=dict,
                              description="Token使用统计")
    conversation_id: Optional[str] = None

class ErrorResponse(BaseModel):
    """错误响应"""
    error_code: str = Field(..., description="错误码")
    message: str = Field(..., description="错误描述")
    request_id: str = Field(..., description="请求唯一标识")
    timestamp: int = Field(..., description="错误时间戳")

# ============ API端点 ============

@app.post("/v2/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    RAG查询接口（v2版本）
    
    支持流式和非流式两种模式。流式模式使用SSE协议。
    """
    start_time = time.time()
    request_id = generate_request_id()
    
    try:
        # 查询处理
        result = await query_engine.search(
            query=request.query,
            search_type=request.search_type,
            top_k=request.top_k,
            conversation_id=request.conversation_id,
        )
        
        latency = int((time.time() - start_time) * 1000)
        
        return QueryResponse(
            answer=result["answer"],
            sources=result.get("sources", []),
            search_type=result.get("search_type", request.search_type),
            latency_ms=latency,
            token_usage=result.get("token_usage", {}),
            conversation_id=result.get("conversation_id"),
        )
    
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error_code="INTERNAL_ERROR",
                message="查询处理失败，请稍后重试",
                request_id=request_id,
                timestamp=int(time.time()),
            ).dict(),
        )

@app.post("/v2/query/stream")
async def query_stream(request: QueryRequest):
    """
    流式RAG查询接口（SSE协议）
    
    事件格式：
    - event: token     data: {"text": "部分回答"}
    - event: source    data: {"sources": [...]}
    - event: done      data: {"latency_ms": ..., "token_usage": ...}
    - event: error     data: {"error_code": "...", "message": "..."}
    """
    # 实现SSE流式响应
    pass

@app.post("/v2/documents/ingest")
async def ingest_documents(
    documents: List[dict],
    index_method: str = "standard",
):
    """
    文档入库接口
    
    支持三种索引方式：
    - standard: 标准索引（完整流程）
    - fast: 快速索引（跳过社区检测）
    - update: 增量更新
    """
    try:
        result = await indexing_pipeline.run(
            documents=documents,
            index_method=index_method,
        )
        return {
            "status": "success",
            "documents_indexed": result["documents_count"],
            "entities_extracted": result["entities_count"],
            "index_time_seconds": result["index_time"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v2/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "components": {
            "vector_store": check_vector_store(),
            "llm_service": check_llm_service(),
            "neo4j": check_neo4j(),
        },
        "uptime_seconds": get_uptime(),
    }
```

#### 11.4.1.2 流式响应实现

流式响应（SSE）对RAG系统的用户体验至关重要。它允许LLM在生成过程中逐步输出内容，大幅降低用户感知的等待时间：

```python
from fastapi.responses import StreamingResponse
import json
import asyncio

async def stream_query_response(query_request: QueryRequest):
    """流式查询响应的SSE生成器"""
    
    try:
        # 1. 发送检索信息
        yield f"event: search\ndata: {json.dumps({'status': 'searching'})}\n\n"
        
        # 2. 执行检索
        context = await retriever.retrieve(
            query_request.query,
            top_k=query_request.top_k,
        )
        
        yield f"event: search\ndata: {json.dumps({
            'status': 'complete',
            'sources_count': len(context),
        })}\n\n"
        
        # 3. 生成回答（流式）
        async for token in llm.stream_generate(
            query_request.query, context
        ):
            yield f"event: token\ndata: {json.dumps({'text': token})}\n\n"
        
        # 4. 发送完成事件
        yield f"event: done\ndata: {json.dumps({
            'latency_ms': compute_latency(),
        })}\n\n"
    
    except Exception as e:
        yield f"event: error\ndata: {json.dumps({
            'error_code': 'STREAM_ERROR',
            'message': str(e),
        })}\n\n"


@app.post("/v2/query/stream")
async def query_stream_endpoint(request: QueryRequest):
    """流式查询端点"""
    return StreamingResponse(
        stream_query_response(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 禁用Nginx缓冲
        },
    )
```

#### 11.4.1.3 错误码设计

标准化的错误码体系是API可维护性的基石：

```python
class ErrorCodes:
    """RAG系统错误码定义"""
    
    # 客户端错误（4xxx）
    INVALID_QUERY = ("4001", "查询内容无效")
    QUERY_TOO_LONG = ("4002", "查询内容超过长度限制")
    INVALID_SEARCH_TYPE = ("4003", "无效的检索策略类型")
    MISSING_PARAMETER = ("4004", "缺少必要参数")
    RATE_LIMITED = ("4290", "请求频率超过限制")
    
    # 服务端错误（5xxx）
    LLM_SERVICE_ERROR = ("5001", "LLM服务调用失败")
    LLM_TIMEOUT = ("5002", "LLM响应超时")
    VECTOR_STORE_ERROR = ("5003", "向量数据库异常")
    INDEXING_ERROR = ("5004", "索引构建失败")
    RETRIEVAL_ERROR = ("5005", "检索过程异常")
    NEO4J_ERROR = ("5006", "Neo4j数据库异常")
    
    # 配置错误（6xxx）
    MODEL_NOT_FOUND = ("6001", "模型不存在或未配置")
    EMBEDDING_MISMATCH = ("6002", "向量维度不匹配")
    CONFIG_ERROR = ("6003", "配置错误")
    
    @classmethod
    def to_http_status(cls, error_code: str) -> int:
        """将错误码映射为HTTP状态码"""
        if error_code.startswith("4"):
            return int(error_code[:3])
        elif error_code.startswith("5"):
            return 500
        elif error_code.startswith("6"):
            return 500
        return 500
```

### 11.4.2 日志与监控

#### 11.4.2.1 结构化日志

```python
import structlog
import json
from datetime import datetime

# 配置结构化日志
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# 在RAG流程中使用结构化日志
async def log_query_flow(query: str, search_type: str):
    """记录查询全流程日志"""
    start_time = datetime.now()
    
    logger.info("query.started", query=query, search_type=search_type)
    
    try:
        # 检索阶段
        retrieval_start = datetime.now()
        results = await retriever.retrieve(query)
        retrieval_time = (datetime.now() - retrieval_start).total_seconds()
        
        logger.info("retrieval.completed",
            results_count=len(results),
            retrieval_time_ms=retrieval_time * 1000,
            top_score=results[0]["score"] if results else 0,
        )
        
        # 生成阶段
        generation_start = datetime.now()
        answer = await llm.generate(query, results)
        generation_time = (datetime.now() - generation_start).total_seconds()
        
        total_time = (datetime.now() - start_time).total_seconds()
        
        logger.info("query.completed",
            total_time_ms=total_time * 1000,
            retrieval_time_ms=retrieval_time * 1000,
            generation_time_ms=generation_time * 1000,
            answer_length=len(answer),
            token_count=estimate_tokens(answer),
        )
        
        return answer
    
    except Exception as e:
        logger.error("query.failed",
            error=str(e),
            error_type=type(e).__name__,
            duration_ms=(datetime.now() - start_time).total_seconds() * 1000,
        )
        raise
```

#### 11.4.2.2 核心监控指标

```python
from prometheus_client import Counter, Histogram, Gauge
import time

# ============ 指标定义 ============

# 1. 延迟指标
query_latency = Histogram(
    'rag_query_latency_seconds',
    '查询延迟分布（秒）',
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
    labelnames=['search_type'],
)

retrieval_latency = Histogram(
    'rag_retrieval_latency_seconds',
    '检索阶段延迟分布（秒）',
    buckets=[0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
    labelnames=['search_type'],
)

generation_latency = Histogram(
    'rag_generation_latency_seconds',
    '生成阶段延迟分布（秒）',
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 20.0],
)

# 2. 流量指标
query_count = Counter(
    'rag_query_total',
    '总查询次数',
    labelnames=['search_type', 'status'],
)

token_usage = Counter(
    'rag_token_usage_total',
    'Token消耗总量',
    labelnames=['model', 'type'],  # type: input/output
)

# 3. 质量指标
answer_rating = Gauge(
    'rag_answer_rating',
    '用户评价平均分（1-5）',
)

hallucination_rate = Gauge(
    'rag_hallucination_rate',
    '幻觉率（基于自动检测）',
)

# 4. 资源指标
vector_store_size = Gauge(
    'rag_vector_store_size',
    '向量数据库文档总量',
)

cache_hit_rate = Gauge(
    'rag_cache_hit_rate',
    '缓存命中率',
)

# ============ 指标采集 ============

async def record_query_metrics(
    search_type: str,
    status: str,
    latency: float,
    ret_latency: float,
    gen_latency: float,
    tokens: dict,
):
    """记录查询指标"""
    query_latency.labels(search_type=search_type).observe(latency)
    retrieval_latency.labels(search_type=search_type).observe(ret_latency)
    generation_latency.observe(gen_latency)
    query_count.labels(search_type=search_type, status=status).inc()
    
    for model, token_count in tokens.items():
        token_usage.labels(model=model, type="input").inc(
            token_count.get("input", 0)
        )
        token_usage.labels(model=model, type="output").inc(
            token_count.get("output", 0)
        )


class MetricsMiddleware:
    """FastAPI中间件：自动采集指标"""
    
    async def __call__(self, request, call_next):
        start = time.time()
        
        response = await call_next(request)
        
        duration = time.time() - start
        query_latency.labels(
            search_type=request.headers.get("X-Search-Type", "auto")
        ).observe(duration)
        
        return response
```

### 11.4.3 异常处理

#### 11.4.3.1 分级降级策略

```python
class DegradationManager:
    """分级降级管理器"""
    
    def __init__(self):
        self.failure_counts = defaultdict(int)
        self.circuit_breakers = {}
        
        # 降级级别定义
        self.degradation_levels = {
            0: "normal",        # 正常模式
            1: "no_graph",      # 降级1：不使用知识图谱
            2: "no_vector",     # 降级2：不使用向量检索
            3: "no_llm",        # 降级3：不使用LLM
            4: "fallback",      # 降级4：仅返回缓存
        }
    
    async def execute_with_degradation(
        self,
        query: str,
        normal_pipeline,
        fallback_pipeline,
    ) -> str:
        """执行带降级策略的查询"""
        
        current_level = self._get_degradation_level()
        
        try:
            if current_level >= 4:
                return await fallback_pipeline(query)
            elif current_level >= 3:
                return await self._no_llm_pipeline(query)
            elif current_level >= 2:
                return await self._no_vector_pipeline(query)
            elif current_level >= 1:
                return await self._no_graph_pipeline(query)
            else:
                return await normal_pipeline(query)
        
        except Exception as e:
            # 发生错误，提升降级级别
            self._escalate_degradation()
            # 降级后重试
            return await self.execute_with_degradation(
                query, normal_pipeline, fallback_pipeline
            )
    
    def _get_degradation_level(self) -> int:
        """获取当前降级级别"""
        max_level = 0
        now = time.time()
        
        for component, breaker in self.circuit_breakers.items():
            if breaker["state"] == "open":
                if now - breaker["last_failure"] > breaker["recovery_time"]:
                    # 尝试恢复
                    breaker["state"] = "half-open"
                else:
                    component_level = breaker.get("degradation_level", 1)
                    max_level = max(max_level, component_level)
        
        return max_level
    
    def _escalate_degradation(self):
        """提升降级级别"""
        for component in self.circuit_breakers:
            breaker = self.circuit_breakers[component]
            if breaker["state"] == "closed":
                breaker["state"] = "open"
                breaker["last_failure"] = time.time()
                break  # 每次只降级一个组件
```

#### 11.4.3.2 重试策略

```python
import asyncio
from functools import wraps

def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exponential_base: float = 2.0,
    retryable_exceptions: tuple = (TimeoutError, ConnectionError),
):
    """
    带指数退避的重试装饰器
    
    Args:
        max_retries: 最大重试次数
        base_delay: 初始延迟（秒）
        max_delay: 最大延迟（秒）
        exponential_base: 指数基数
        retryable_exceptions: 可重试的异常类型
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = min(
                            base_delay * (exponential_base ** attempt),
                            max_delay,
                        )
                        # 增加抖动（jitter）
                        import random
                        delay *= 1 + random.random() * 0.1
                        
                        logger.warning(
                            "retry_attempt",
                            func=func.__name__,
                            attempt=attempt + 1,
                            delay=delay,
                            error=str(e),
                        )
                        await asyncio.sleep(delay)
            
            raise last_exception
        
        return wrapper
    
    return decorator


# 使用示例
class LLMClient:
    """LLM客户端（带重试和熔断）"""
    
    @retry_with_backoff(
        max_retries=3,
        base_delay=1.0,
        retryable_exceptions=(
            TimeoutError,
            ConnectionError,
            ServiceUnavailableError,
        ),
    )
    async def generate(self, prompt: str) -> str:
        """生成回答（带自动重试）"""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.llm_endpoint,
                json={"prompt": prompt},
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status == 503:
                    raise ServiceUnavailableError("LLM服务暂不可用")
                elif resp.status == 429:
                    # 限流：使用较长的退避
                    retry_after = int(resp.headers.get("Retry-After", 5))
                    await asyncio.sleep(retry_after)
                    raise TimeoutError("rate limited")
                
                data = await resp.json()
                return data["text"]
```

### 11.4.4 金丝雀发布

金丝雀发布（Canary Release）是RAG系统上线新版本时的最佳实践：

```python
class CanaryReleaseManager:
    """金丝雀发布管理器"""
    
    def __init__(self, redis_client):
        self.redis = redis_client
    
    async def deploy_canary(
        self,
        new_version: str,
        canary_percentage: float = 0.05,  # 5%流量
        evaluation_period: int = 3600,     # 1小时评估期
    ):
        """
        执行金丝雀发布
        
        流程：
        1. 部署新版本到金丝雀节点
        2. 将5%的流量路由到新版本
        3. 监控关键指标（延迟、错误率、用户反馈）
        4. 如果指标正常，逐步增加流量比例
        5. 如果指标异常，自动回滚
        """
        # 设置金丝雀配置
        canary_config = {
            "version": new_version,
            "percentage": canary_percentage,
            "started_at": time.time(),
            "status": "deploying",
            "metrics": {
                "error_rate": 0.0,
                "p95_latency": 0.0,
                "user_satisfaction": 0.0,
            },
        }
        
        await self.redis.set(
            f"canary:{new_version}:config",
            json.dumps(canary_config),
        )
        
        # 金丝雀评估循环
        while time.time() - canary_config["started_at"] < evaluation_period:
            await asyncio.sleep(30)  # 每30秒检查一次
            
            metrics = await self._collect_canary_metrics(new_version)
            
            # 检查是否需要回滚
            if self._should_rollback(metrics):
                await self._rollback(new_version)
                return {"status": "rolled_back", "reason": "指标异常"}
            
            # 逐步增加流量
            current_pct = metrics.get("current_percentage", 0.05)
            if metrics["stable_minutes"] > 10 and current_pct < 1.0:
                new_pct = min(current_pct * 2, 1.0)
                await self._update_traffic_split(new_version, new_pct)
                logger.info(f"金丝雀流量增加至 {new_pct:.1%}")
        
        # 全量发布
        await self._full_release(new_version)
        return {"status": "released"}
    
    def _should_rollback(self, metrics: dict) -> bool:
        """判断是否需要回滚"""
        # 回滚条件
        if metrics["error_rate"] > 0.05:        # 错误率超过5%
            return True
        if metrics["p95_latency"] > 10000:      # P95延迟超过10秒
            return True
        if metrics["user_satisfaction"] < 0.3:  # 用户满意度低于0.3
            return True
        return False
```

---

## 11.5 团队协作最佳实践

### 11.5.1 沟通机制

```
每日站会（15分钟）
- 算法：昨日实验进展、今日计划、阻塞项
- 产品：用户反馈、需求变更、优先级调整
- 后端：系统状态、性能数据、部署计划
- 前端：UI开发进度、交互优化
- 数据：数据质量报告、标注进度

每周同步会（30分钟）
- 本周成果演示
- 下周计划对齐
- 跨团队依赖确认
- 风险和问题同步

双周回顾会（1小时）
- 迭代回顾（做得好的/待改进的）
- 流程优化讨论
- 技术分享和知识沉淀
```

### 11.5.2 文档规范

```
项目文档结构
├── docs/
│   ├── architecture/      # 架构设计文档
│   ├── api/               # API文档
│   ├── algorithms/        # 算法设计文档
│   ├── operations/        # 运维手册
│   └── decisions/         # 技术决策记录（ADR）
```

### 11.5.3 代码规范

```
1. 统一的代码风格（Black/PEP 8）
2. 类型注解（Python type hints）
3. 详细的文档字符串（Google风格）
4. 单元测试覆盖率 >= 80%
5. 代码审查（Code Review）流程
6. 语义化版本号（SemVer）
```

---

## 11.6 本章小结

本章从工程实践角度，系统性地介绍了RAG项目中的团队协作和工程规范：

1. **团队角色**定义了算法、产品、后端、前端和数据工程师的职责边界和协作方式，清晰的职责划分是高效协作的基础。

2. **项目生命周期**覆盖了从需求分析到上线运维的全流程，强调快速验证和持续迭代的敏捷模式。POC阶段的快速验证能够尽早识别技术风险，降低项目失败的可能性。

3. **工程规范**包括API设计（RESTful + 流式SSE）、结构化日志和监控指标体系、分级降级和重试策略、以及金丝雀发布流程。这些规范确保了系统的可维护性和可靠性。

4. **团队协作**的最佳实践包括高效的沟通机制、完善的文档体系和统一的代码规范。

RAG系统的成功，20%取决于算法能力，80%取决于工程质量和团队协作。下一章将深入讨论RAG系统的评估体系，包括评估指标、评估数据集构建和持续优化策略。
