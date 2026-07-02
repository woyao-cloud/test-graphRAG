"""
第 18 章 Demo：端到端 RAG 系统实战

演示一个最小可用的企业级 RAG 系统：
  FastAPI 服务 + 混合检索 + Reranker + LLM 生成 + 缓存 + Prometheus 指标

可独立运行（内置 HTTP 服务）。

用法：
  python end_to_end_rag.py                     # 启动服务 (http://localhost:8000)
  python end_to_end_rag.py --test              # 运行查询测试
  python end_to_end_rag.py --benchmark         # 运行压测
"""

import argparse
import json
import math
import os
import random
import time
import uuid
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import Optional


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class RAGConfig:
    llm_model: str = "deepseek-v4-flash"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 4096
    embedding_dim: int = 768
    top_k: int = 10
    rerank_top_k: int = 5
    chunk_size: int = 512
    chunk_overlap: int = 50
    retrieval_modes: list = field(default_factory=lambda: ["dense", "sparse"])
    host: str = "0.0.0.0"
    port: int = 8000


# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class Document:
    id: str
    text: str
    title: str = ""
    source: str = ""
    score: float = 0.0


@dataclass
class QueryResult:
    answer: str
    sources: list[Document]
    latency_ms: float = 0.0
    tokens_used: int = 0


# ============================================================================
# Mock Embedding & Vector DB
# ============================================================================


class MockEmbedding:
    def embed(self, text: str) -> list[float]:
        features = [0.0] * 32
        for i, ch in enumerate(text[:200]):
            features[hash(ch) % 32] += 1.0
        norm = math.sqrt(sum(v * v for v in features)) or 1.0
        return [v / norm for v in features]


class MockVectorDB:
    def __init__(self, embedding: MockEmbedding):
        self.embedding = embedding
        self.docs: list[Document] = []
        self.embeddings: list[list[float]] = []

    def add_documents(self, docs: list[Document]):
        self.docs = docs
        self.embeddings = [self.embedding.embed(d.text) for d in docs]

    def search(self, query: str, top_k: int = 10) -> list[Document]:
        q_emb = self.embedding.embed(query)
        scored = []
        for i, doc in enumerate(self.docs):
            sim = sum(av * bv for av, bv in zip(q_emb, self.embeddings[i]))
            scored.append((sim, doc))
        scored.sort(key=lambda x: -x[0])
        return [Document(id=d.id, text=d.text, title=d.title, score=s)
                for s, d in scored[:top_k]]


# ============================================================================
# BM25-like Sparse Retriever
# ============================================================================


class BM25Retriever:
    def __init__(self):
        self.docs: list[Document] = []
        self.avg_dl: float = 0.0
        self.k1 = 1.5
        self.b = 0.75

    def add_documents(self, docs: list[Document]):
        self.docs = docs
        self.avg_dl = sum(len(d.text) for d in docs) / max(len(docs), 1)

    def search(self, query: str, top_k: int = 10) -> list[Document]:
        q_terms = set(query.lower().split())
        scored = []
        for doc in self.docs:
            score = 0.0
            dl = len(doc.text)
            for term in q_terms:
                tf = doc.text.lower().count(term)
                if tf == 0:
                    continue
                score += (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * dl / self.avg_dl))
            scored.append((score, doc))
        scored.sort(key=lambda x: -x[0])
        return [Document(id=d.id, text=d.text, title=d.title, score=s)
                for s, d in scored[:top_k]]


# ============================================================================
# Reranker (Mock Cross-Encoder)
# ============================================================================


class Reranker:
    def rerank(self, query: str, docs: list[Document], top_k: int = 5) -> list[Document]:
        """模拟 Cross-Encoder 重排。"""
        for doc in docs:
            # 计算 query-doc 的交互特征（模拟）
            query_words = set(query.lower().split())
            doc_words = set(doc.text.lower().split())
            overlap = len(query_words & doc_words) / max(len(query_words), 1)
            doc.score = doc.score * 0.7 + overlap * 0.3
        docs.sort(key=lambda x: -x.score)
        return docs[:top_k]


# ============================================================================
# Cache
# ============================================================================


class SimpleCache:
    def __init__(self, ttl: int = 300, max_size: int = 1000):
        self.cache: OrderedDict[str, tuple[float, str]] = OrderedDict()
        self.ttl = ttl
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[str]:
        if key in self.cache:
            ts, value = self.cache[key]
            if time.time() - ts < self.ttl:
                self.cache.move_to_end(key)
                self.hits += 1
                return value
            else:
                del self.cache[key]
        self.misses += 1
        return None

    def set(self, key: str, value: str):
        if len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)
        self.cache[key] = (time.time(), value)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


# ============================================================================
# Metrics
# ============================================================================


class MetricsCollector:
    def __init__(self):
        self.latencies: list[float] = []
        self.errors = 0
        self.total = 0

    def record(self, latency_ms: float, success: bool):
        self.latencies.append(latency_ms)
        self.total += 1
        if not success:
            self.errors += 1

    def summary(self) -> dict:
        if not self.latencies:
            return {"total": 0}
        sorted_l = sorted(self.latencies)
        n = len(sorted_l)
        return {
            "total": self.total,
            "errors": self.errors,
            "error_rate": round(self.errors / self.total, 4),
            "p50_ms": sorted_l[n // 2],
            "p95_ms": sorted_l[int(n * 0.95)],
            "p99_ms": sorted_l[int(n * 0.99)],
            "avg_ms": round(sum(self.latencies) / n, 1),
        }


# ============================================================================
# LLM (Mock)
# ============================================================================


class MockLLM:
    def generate(self, question: str, context: list[Document]) -> str:
        """模拟 LLM 生成。"""
        time.sleep(0.3)  # 模拟延迟
        sources = [d.title or d.text[:20] for d in context[:3]]
        return (
            f"根据检索结果，回答: {question}\n\n"
            + "\n".join(f"- {d.text[:150]}" for d in context[:3])
            + f"\n\n参考来源: {', '.join(sources)}"
        )


# ============================================================================
# RAG Service
# ============================================================================


class RAGService:
    def __init__(self, config: RAGConfig = None):
        self.config = config or RAGConfig()
        self.embedding = MockEmbedding()
        self.vector_db = MockVectorDB(self.embedding)
        self.bm25 = BM25Retriever()
        self.reranker = Reranker()
        self.llm = MockLLM()
        self.cache = SimpleCache()
        self.metrics = MetricsCollector()
        self._init_documents()

    def _init_documents(self):
        docs = [
            Document("d1", "恒瑞医药是中国领先的制药企业，专注于抗肿瘤药物研发和生产。主要产品包括注射用紫杉醇、奥沙利铂和卡培他滨。", "恒瑞医药简介", "doc1"),
            Document("d2", "紫杉醇是一种微管抑制剂，通过促进微管蛋白聚合、抑制微管解聚发挥抗肿瘤作用。用于非小细胞肺癌、乳腺癌和卵巢癌。", "紫杉醇说明", "doc2"),
            Document("d3", "奥沙利铂是第三代铂类抗肿瘤药物，主要用于转移性结直肠癌的治疗，常与氟尿嘧啶联合使用。", "奥沙利铂说明", "doc2"),
            Document("d4", "卡培他滨是口服氟尿嘧啶类抗肿瘤药物，用于结直肠癌和乳腺癌的治疗。可单药或联合化疗。", "卡培他滨说明", "doc2"),
            Document("d5", "国药控股是恒瑞医药和齐鲁制药的分销合作伙伴，拥有覆盖全国的药品分销网络和冷链物流体系。", "国药控股简介", "doc3"),
            Document("d6", "华海药业为恒瑞医药提供紫杉醇 API 原料药。在原料药领域拥有丰富经验，年产能超过500公斤。", "华海药业简介", "doc4"),
            Document("d7", "北京协和医院肿瘤科使用注射用紫杉醇和顺铂等抗肿瘤药物，用于各类实体瘤的综合治疗。", "协和医院简介", "doc5"),
            Document("d8", "齐鲁制药生产顺铂和卡培他滨等抗肿瘤药物。顺铂是最广泛使用的铂类抗肿瘤药物之一。", "齐鲁制药简介", "doc6"),
        ]
        self.vector_db.add_documents(docs)
        self.bm25.add_documents(docs)

    def query(self, question: str, mode: str = "auto", stream: bool = False) -> QueryResult:
        start = time.time()

        # 1. Check cache
        cache_key = f"rag:{hash(question)}"
        cached = self.cache.get(cache_key)
        if cached:
            latency = (time.time() - start) * 1000
            self.metrics.record(latency, True)
            result = json.loads(cached)
            return QueryResult(**result)

        try:
            # 2. Multi-recall retrieval
            dense_results = self.vector_db.search(question, top_k=self.config.top_k)
            sparse_results = self.bm25.search(question, top_k=self.config.top_k)

            # 3. Merge & dedup
            seen = set()
            all_docs = []
            for doc in dense_results + sparse_results:
                if doc.id not in seen:
                    seen.add(doc.id)
                    all_docs.append(doc)

            # 4. Rerank
            reranked = self.reranker.rerank(question, all_docs, self.config.rerank_top_k)

            # 5. Generate
            answer = self.llm.generate(question, reranked)

            latency = (time.time() - start) * 1000
            self.metrics.record(latency, True)

            result = QueryResult(
                answer=answer,
                sources=reranked[:3],
                latency_ms=round(latency, 0),
                tokens_used=len(answer) // 2,
            )

            # Cache
            self.cache.set(cache_key, json.dumps({
                "answer": result.answer,
                "sources": [{"id": s.id, "text": s.text[:100], "title": s.title, "score": s.score} for s in result.sources],
                "latency_ms": result.latency_ms,
                "tokens_used": result.tokens_used,
            }))

            return result

        except Exception as e:
            latency = (time.time() - start) * 1000
            self.metrics.record(latency, False)
            return QueryResult(answer=f"错误: {str(e)}", sources=[], latency_ms=round(latency, 0))


# ============================================================================
# Simple HTTP Server
# ============================================================================


def run_server(service: RAGService, port: int = 8000):
    """运行简单 HTTP 服务。"""
    import http.server
    import urllib.parse

    class RAGHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)

            if parsed.path == "/health":
                self._json({"status": "ok", "timestamp": time.time()})
            elif parsed.path == "/metrics":
                self._json(service.metrics.summary())
            elif parsed.path == "/cache":
                self._json({"hit_rate": service.cache.hit_rate})
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            if self.path != "/api/v1/query":
                self.send_response(404)
                self.end_headers()
                return

            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                data = json.loads(body)
                question = data.get("question", "")
                if not question:
                    self._json({"error": "question is required"}, 400)
                    return

                result = service.query(question)
                self._json({
                    "answer": result.answer,
                    "sources": [
                        {"id": s.id, "title": s.title, "score": s.score}
                        for s in result.sources
                    ],
                    "metrics": {
                        "latency_ms": result.latency_ms,
                        "tokens": result.tokens_used,
                    },
                })
            except Exception as e:
                self._json({"error": str(e)}, 500)

        def _json(self, data: dict, status: int = 200):
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(data, ensure_ascii=False).encode())

        def log_message(self, format, *args):
            pass  # 安静运行

    server = http.server.HTTPServer(("0.0.0.0", port), RAGHandler)
    print(f"\n  RAG Service 启动于 http://localhost:{port}")
    print(f"  端点:")
    print(f"    POST /api/v1/query  - 查询 (JSON body: {{'question': '...'}})")
    print(f"    GET  /health        - 健康检查")
    print(f"    GET  /metrics       - 性能指标")
    print(f"    GET  /cache         - 缓存统计")
    print(f"\n  示例:")
    print(f"    curl -X POST http://localhost:{port}/api/v1/query \\")
    print(f'      -H "Content-Type: application/json" \\')
    print(f'      -d \'{{"question": "恒瑞医药生产哪些药品？"}}\'')
    print(f"\n  按 Ctrl+C 停止服务\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  服务已停止")
        server.server_close()


# ============================================================================
# Test & Benchmark
# ============================================================================


def run_test(service: RAGService):
    print("\n" + "=" * 60)
    print("RAG 查询测试")
    print("=" * 60)

    questions = [
        "恒瑞医药生产哪些药品？",
        "紫杉醇的治疗机制是什么？",
        "国药控股有哪些业务？",
        "北京协和医院使用哪些药品？",
        "华海药业与恒瑞医药的关系是什么？",
    ]

    for q in questions:
        print(f"\n  Q: {q}")
        result = service.query(q)
        print(f"  A: {result.answer[:120]}...")
        print(f"  Latency: {result.latency_ms:.0f}ms | Sources: {len(result.sources)}")

    # 测试缓存命中
    print(f"\n\n  缓存测试（重复查询）:")
    q = "恒瑞医药生产哪些药品？"
    result1 = service.query(q)
    result2 = service.query(q)
    print(f"  首次: {result1.latency_ms:.0f}ms | 缓存命中率: {service.cache.hit_rate:.2%}")


def run_benchmark(service: RAGService):
    print("\n" + "=" * 60)
    print("压测（50 次查询）")
    print("=" * 60)

    questions = [
        "恒瑞医药生产哪些药品？",
        "紫杉醇的治疗机制是什么？",
        "北京协和医院使用哪些药品？",
        "国药控股有哪些业务？",
        "华海药业供应什么？",
    ]

    print(f"\n  执行 50 次查询（10 轮 × 5 个问题）...\n")
    for i in range(10):
        for q in questions:
            result = service.query(q)
            if i == 0 and q == questions[0]:
                pass  # 预热

    summary = service.metrics.summary()
    print(f"  {'指标':<20} {'值':<15}")
    print(f"  {'─' * 35}")
    for key, value in summary.items():
        print(f"  {key:<20} {value:<15}")


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="端到端 RAG 系统 Demo")
    parser.add_argument("--serve", action="store_true", help="启动 HTTP 服务")
    parser.add_argument("--test", action="store_true", help="运行查询测试")
    parser.add_argument("--benchmark", action="store_true", help="运行压测")
    parser.add_argument("--port", type=int, default=8000, help="服务端口")
    args = parser.parse_args()

    service = RAGService()

    if args.serve:
        run_server(service, args.port)
    elif args.test:
        run_test(service)
    elif args.benchmark:
        run_benchmark(service)
    else:
        # Default: show summary
        print("=" * 60)
        print("端到端 RAG 系统 Demo")
        print("=" * 60)
        print(f"\n配置:")
        print(f"  模型: {service.config.llm_model}")
        print(f"  检索模式: {', '.join(service.config.retrieval_modes)}")
        print(f"  Top-K: {service.config.top_k} → Rerank: {service.config.rerank_top_k}")
        print(f"\n用法:")
        print(f"  python end_to_end_rag.py --test      运行查询测试")
        print(f"  python end_to_end_rag.py --benchmark 运行压测")
        print(f"  python end_to_end_rag.py --serve     启动 HTTP 服务 (端口 {args.port})")


if __name__ == "__main__":
    main()
