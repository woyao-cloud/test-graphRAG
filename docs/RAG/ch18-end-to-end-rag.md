# 第 18 章：端到端 RAG 系统实战

## 18.1 系统架构全景

本章将搭一个最小可用的企业级 RAG 系统，涵盖从 API 设计到部署运维的完整链路。

```
┌──────────────────────────────────────────────────────────┐
│                    客户端（Web / API）                     │
└─────────────────────────┬────────────────────────────────┘
                          │ HTTP / SSE
┌─────────────────────────▼────────────────────────────────┐
│                  API 网关（Nginx / Envoy）                  │
└─────────────────────────┬────────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────────┐
│                  RAG Service (FastAPI)                    │
├───────────────┬───────────────────┬──────────────────────┤
│  Query Router │  Context Builder  │   Response Generator │
│  (意图路由)    │  (上下文构建)       │   (答案生成 + 流式)    │
└───────┬───────┴─────────┬─────────┴──────────┬───────────┘
        │                 │                    │
┌───────▼──────┐ ┌────────▼───────┐ ┌─────────▼─────────┐
│  Retrieval   │ │  Reranker      │ │  LLM Service      │
│  (多路检索)   │ │  (Cross-Encoder)│ │  (OpenAI/DeepSeek)│
├──────────────┤ ├───────────────┤ ├───────────────────┤
│ • 向量检索    │ │ • 重排排序     │ │ • 流式生成         │
│ • BM25       │ │ • 相关度过滤   │ │ • Prompt 管理      │
│ • 知识图谱    │ │               │ │ • Token 计数       │
└───────┬──────┘ └───────────────┘ └───────────────────┘
        │
┌───────▼──────────────────────────────────────────────────┐
│                    存储层                                  │
├──────────────────┬──────────────────┬────────────────────┤
│  Vector DB       │  Search Engine   │  Cache (Redis)    │
│  (Milvus/Qdrant) │  (ES/Search)     │  (Embedding/Result)│
└──────────────────┴──────────────────┴────────────────────┘
```

---

## 18.2 核心代码实现

### 18.2.1 配置管理

```python
from pydantic import BaseSettings


class RAGConfig(BaseSettings):
    """RAG 系统配置。"""

    # LLM
    llm_model: str = "deepseek-v4-flash"
    llm_api_base: str = "https://api.deepseek.com/v1"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 4096

    # Embedding
    embedding_model: str = "bge-m3"
    embedding_dim: int = 1024

    # Retrieval
    top_k: int = 10
    rerank_top_k: int = 5
    chunk_size: int = 512
    chunk_overlap: int = 50
    retrieval_modes: list[str] = ["dense", "sparse", "kg"]

    # Vector DB
    vector_db_type: str = "milvus"  # milvus / qdrant / chroma
    vector_db_host: str = "localhost"
    vector_db_port: int = 19530

    # Cache
    redis_host: str = "localhost"
    redis_port: int = 6379
    cache_ttl: int = 300  # seconds

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    max_concurrent: int = 10
    rate_limit: int = 60  # requests per minute

    class Config:
        env_file = ".env"
        env_prefix = "RAG_"
```

### 18.2.2 检索器实现

```python
class BaseRetriever(ABC):
    """检索器基类。"""

    @abstractmethod
    def search(self, query: str, top_k: int) -> list[dict]:
        ...


class VectorRetriever(BaseRetriever):
    """向量检索器。"""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.embedding_model = self._load_model()

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        query_emb = self.embedding_model.encode(query)
        # 向量库搜索
        results = vector_db.search(
            embedding=query_emb,
            top_k=top_k,
        )
        return [
            {"id": r.id, "text": r.text, "score": r.score, "source": "dense"}
            for r in results
        ]


class HybridRetriever(BaseRetriever):
    """混合检索器。"""

    def __init__(self, config: RAGConfig):
        self.retrievers = {
            "dense": VectorRetriever(config),
            "sparse": Bm25Retriever(),
            "kg": KnowledgeGraphRetriever(),
        }

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        all_results = []
        for name, retriever in self.retrievers.items():
            results = retriever.search(query, top_k=top_k)
            all_results.extend(results)

        # 去重 + RRF 融合
        return self._rrf_fusion(all_results)[:top_k]
```

### 18.2.3 RAG Service

```python
class RAGService:
    """核心 RAG 服务。"""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.retriever = HybridRetriever(config)
        self.reranker = CrossEncoderReranker()
        self.cache = CacheClient(config.redis_host, config.redis_port)

    async def query(
        self,
        question: str,
        mode: str = "auto",
        stream: bool = False,
    ) -> dict:
        # 1. 检查缓存
        cache_key = f"rag:{hash(question)}"
        cached = await self.cache.get(cache_key)
        if cached:
            return json.loads(cached)

        start = time.time()

        # 2. 检索
        chunks = self.retriever.search(question, top_k=self.config.top_k)

        # 3. 重排
        chunks = self.reranker.rerank(question, chunks)
        chunks = chunks[:self.config.rerank_top_k]

        # 4. 构建上下文
        context = self._build_context(chunks)

        # 5. 生成
        if stream:
            return self._generate_stream(question, context)
        else:
            answer = await self._generate(question, context)

        elapsed = (time.time() - start) * 1000

        result = {
            "answer": answer,
            "sources": [
                {"id": c["id"], "text": c["text"][:200], "score": c["score"]}
                for c in chunks[:3]
            ],
            "metrics": {
                "latency_ms": round(elapsed, 0),
                "chunks_retrieved": len(chunks),
            },
        }

        # 写入缓存
        if elapsed < 5000:  # 只缓存快速查询
            await self.cache.set(cache_key, json.dumps(result), ttl=300)

        return result
```

### 18.2.4 FastAPI 接口

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


app = FastAPI(title="RAG Service", version="1.0.0")
rag_service = RAGService(RAGConfig())


class QueryRequest(BaseModel):
    question: str
    mode: str = "auto"
    stream: bool = False
    top_k: int = 5


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    metrics: dict


@app.post("/api/v1/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """执行 RAG 查询。"""
    try:
        return await rag_service.query(
            question=request.question,
            mode=request.mode,
            stream=request.stream,
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/health")
async def health():
    """健康检查。"""
    return {"status": "ok", "timestamp": time.time()}


@app.get("/metrics")
async def metrics():
    """Prometheus 指标。"""
    return rag_service.get_metrics()
```

---

## 18.3 CI/CD 管线

### 18.3.1 GitHub Actions

```yaml
# .github/workflows/rag-ci.yml
name: RAG CI/CD

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install dependencies
        run: pip install -e ".[dev]"

      - name: Run lint
        run: ruff check src/

      - name: Run type check
        run: mypy src/

      - name: Run tests
        run: pytest tests/ --cov=src/ --cov-report=term

      - name: Run RAG evaluation
        run: python scripts/evaluate.py --test-set tests/data/eval_set.json
```

### 18.3.2 Docker 部署

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY src/ src/
COPY models/ models/

# Run
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 18.3.3 docker-compose.yml

```yaml
version: "3.8"

services:
  rag-service:
    build: .
    ports:
      - "8000:8000"
    env_file: .env
    depends_on:
      - redis
      - milvus
    restart: always

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  milvus:
    image: milvusdb/milvus:v2.4.0
    ports:
      - "19530:19530"
    volumes:
      - milvus_data:/var/lib/milvus

volumes:
  redis_data:
  milvus_data:
```

---

## 18.4 监控与告警

### 18.4.1 Prometheus 指标

```python
from prometheus_client import Counter, Histogram, Gauge

# 请求计数
rag_requests_total = Counter(
    "rag_requests_total", "Total RAG requests", ["mode", "status"]
)

# 延迟分布
rag_latency_seconds = Histogram(
    "rag_latency_seconds", "RAG latency in seconds",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)

# Token 消耗
rag_tokens_total = Counter(
    "rag_tokens_total", "Total tokens used", ["type"]  # input/output
)

# 缓存命中率
rag_cache_hit_ratio = Gauge(
    "rag_cache_hit_ratio", "Cache hit ratio"
)
```

---

## 18.5 部署检查清单

```text
□ 性能基线已建立（P50/P99 延迟）
□ Redis 缓存已配置
□ 数据库连接池大小已调优
□ 日志级别已调整为生产级别
□ Prometheus 指标已暴露
□ 关键告警已配置
  - P99 延迟 > 5s
  - 错误率 > 1%
  - 缓存命中率 < 20%
□ 限流策略已启用
□ CORS 已配置
□ 健康检查端点已添加
□ Docker 资源限制已设置
□ .env 中的密钥已配置
```

---

*下一章 [第 19 章：客服领域 RAG 实践](ch19-customer-service.md)*
