"""
第 2 章 Demo：基础 RAG 管线

演示 RAG 三阶段（检索 → 增强 → 生成）的完整流程。
可独立运行，使用本地 Ollama 或 OpenAI 兼容 API。

运行前：
  1. 启动 Ollama: ollama pull llama3.2 && ollama pull nomic-embed-text
  2. 或设置 OpenAI API Key: set OPENAI_API_KEY=sk-xxx

用法：
  python basic_rag_pipeline.py --query "恒瑞医药生产什么药物？"
  python basic_rag_pipeline.py --query "紫杉醇的供应链是怎样的？" --method graph
"""

import argparse
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Optional

# ============================================================================
# Part 1: Knowledge Base — 模拟知识库
# ============================================================================

SAMPLE_DOCUMENTS = [
    {
        "id": "doc_001",
        "title": "恒瑞医药产品目录",
        "content": (
            "恒瑞医药是中国领先的制药企业，主要生产和销售抗肿瘤药物。"
            "旗下核心产品包括注射用紫杉醇（白蛋白结合型）、阿帕替尼、"
            "卡瑞利珠单抗、吡咯替尼等。其中注射用紫杉醇是公司最畅销的产品之一。"
        ),
    },
    {
        "id": "doc_002",
        "title": "紫杉醇供应链分析",
        "content": (
            "注射用紫杉醇（白蛋白结合型）的原料药紫杉醇API由华海药业供应。"
            "恒瑞医药将原料药加工为成品制剂后，通过国药控股进行分销。"
            "国药控股负责华东区域的医院配送，覆盖包括北京协和医院在内的主要医疗机构。"
        ),
    },
    {
        "id": "doc_003",
        "title": "国药控股区域覆盖",
        "content": (
            "国药控股是中国最大的药品分销商，在华东区设有12个配送中心。"
            "主要合作制药企业包括恒瑞医药、齐鲁制药、正大天晴等。"
            "2023年华东区分销额超过200亿元人民币。"
        ),
    },
    {
        "id": "doc_004",
        "title": "北京协和医院采购记录",
        "content": (
            "北京协和医院2023年药品采购数据显示，抗肿瘤药物采购量同比增长35%。"
            "主要供应商包括恒瑞医药（紫杉醇注射剂）和齐鲁制药（吉非替尼片）。"
            "采购流程遵循国家药品集中采购政策。"
        ),
    },
]


@dataclass
class DocumentChunk:
    """文档块，元数据保留来源信息。"""
    chunk_id: str
    doc_id: str
    title: str
    content: str
    metadata: dict = field(default_factory=dict)


def chunk_documents(docs: list[dict], chunk_size: int = 200) -> list[DocumentChunk]:
    """简单滑动窗口切分（生产环境请使用语义分块）。"""
    chunks = []
    for doc in docs:
        content = doc["content"]
        for i in range(0, len(content), chunk_size):
            chunk_text = content[i : i + chunk_size]
            chunk_id = f"{doc['id']}_chunk_{i // chunk_size}"
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    doc_id=doc["id"],
                    title=doc["title"],
                    content=chunk_text,
                    metadata={"source": doc["title"], "chunk_index": i // chunk_size},
                )
            )
    return chunks


# ============================================================================
# Part 2: Embedding & Vector Storage — 简易向量存储
# ============================================================================


class SimpleVectorStore:
    """内存向量存储（生产环境请使用 Milvus/Qdrant/Chroma）。"""

    def __init__(self):
        self.chunks: list[DocumentChunk] = []
        self.embeddings: list[list[float]] = []
        self._model = None

    def _get_embedding_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                model_name = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
                self._model = SentenceTransformer(model_name)
                print(f"[INFO] 加载 Embedding 模型: {model_name}")
            except ImportError:
                # Fallback: 简单哈希模拟（仅演示用）
                self._model = "mock"
        return self._model

    def _mock_embed(self, text: str) -> list[float]:
        """仅用于演示的 mock embedding（不依赖任何库）。"""
        import hashlib

        h = hashlib.md5(text.encode()).hexdigest()
        return [int(h[i : i + 2], 16) / 255.0 for i in range(0, 32, 2)]

    def add_documents(self, chunks: list[DocumentChunk]):
        """批量添加文档并计算向量。"""
        model = self._get_embedding_model()
        self.chunks = chunks

        texts = [c.content for c in chunks]
        if model == "mock":
            self.embeddings = [self._mock_embed(t) for t in texts]
        else:
            self.embeddings = [model.encode(t).tolist() for t in texts]
        print(f"[INFO] 已索引 {len(chunks)} 个文档块")

    def cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """余弦相似度。"""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        return dot / (norm_a * norm_b) if norm_a * norm_b > 0 else 0.0

    def search(self, query: str, top_k: int = 3) -> list[tuple[DocumentChunk, float]]:
        """向量检索 — 余弦相似度 Top-K。"""
        model = self._get_embedding_model()
        if model == "mock":
            query_vec = self._mock_embed(query)
        else:
            query_vec = model.encode(query).tolist()

        scores = [(i, self.cosine_similarity(query_vec, emb)) for i, emb in enumerate(self.embeddings)]
        scores.sort(key=lambda x: x[1], reverse=True)

        results = [(self.chunks[i], score) for i, score in scores[:top_k]]
        return results


# ============================================================================
# Part 3: RAG Pipeline — 检索 → 增强 → 生成
# ============================================================================


class RAGPipeline:
    """基础 RAG 管线。"""

    def __init__(self, vector_store: SimpleVectorStore):
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 3) -> list[tuple[DocumentChunk, float]]:
        """检索阶段。"""
        return self.vector_store.search(query, top_k=top_k)

    def augment(self, query: str, chunks: list[tuple[DocumentChunk, float]]) -> str:
        """增强阶段：构建包含上下文的 Prompt。"""
        context_parts = []
        for i, (chunk, score) in enumerate(chunks, 1):
            context_parts.append(
                f"[{i}] 来源: {chunk.title} (相关性: {score:.3f})\n内容: {chunk.content}\n"
            )
        context = "\n---\n".join(context_parts)

        prompt = f"""你是一个知识库问答助手。
请基于以下提供的上下文信息回答问题。如果上下文信息不足以回答问题，
请明确表示不知道，不要编造。请引用信息来源（如 [1][2]）。

上下文：
{context}

问题：{query}

回答："""
        return prompt

    def generate(self, prompt: str) -> str:
        """生成阶段：调用 LLM。"""
        # 尝试多种 LLM 后端
        response = self._call_openai_compatible(prompt)
        if response:
            return response
        response = self._call_ollama(prompt)
        if response:
            return response
        return self._mock_generate(prompt)

    def _call_openai_compatible(self, prompt: str) -> Optional[str]:
        """调用 OpenAI 兼容 API（DeepSeek / vLLM / Ollama）。"""
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("GRAPHRAG_API_KEY")
        api_base = os.environ.get("OPENAI_API_BASE", "http://localhost:11434/v1")
        model = os.environ.get("CHAT_MODEL", "llama3.2")

        if not api_key:
            return None

        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key, base_url=api_base)
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1024,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[WARN] OpenAI API 调用失败: {e}")
            return None

    def _call_ollama(self, prompt: str) -> Optional[str]:
        """直接调用 Ollama HTTP API。"""
        try:
            import requests

            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": os.environ.get("OLLAMA_MODEL", "llama3.2"),
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.3},
                },
                timeout=60,
            )
            if response.status_code == 200:
                return response.json().get("response", "")
        except Exception as e:
            print(f"[WARN] Ollama 调用失败: {e}")
        return None

    def _mock_generate(self, prompt: str) -> str:
        """Mock 生成器（无 API 依赖，仅演示结构）。"""
        # 从 Prompt 中提取上下文用于 Mock 回答
        import re

        match = re.search(r"问题：(.+?)(?:\n回答：|$)", prompt, re.DOTALL)
        query = match.group(1).strip() if match else "未知问题"

        # 简单关键词匹配
        if "恒瑞" in query or "紫杉醇" in query:
            return (
                "根据资料，恒瑞医药生产抗肿瘤药物，核心产品包括注射用紫杉醇。"
                "紫杉醇的原料药由华海药业供应，成品通过国药控股分销至各医院。[1][2]"
            )
        elif "国药" in query or "分销" in query:
            return (
                "国药控股是华东区主要药品分销商，与恒瑞医药、齐鲁制药等企业合作，"
                "覆盖包括北京协和医院在内的主要医疗机构。[3][4]"
            )
        else:
            return f"基于提供的上下文，关于「{query}」的相关信息已在上文中列出。如有具体问题请进一步询问。"

    def ask(self, query: str, top_k: int = 3) -> dict:
        """完整 RAG 流程。"""
        import time

        t0 = time.time()

        # 1. 检索
        chunks = self.retrieve(query, top_k=top_k)
        t1 = time.time()

        # 2. 增强
        prompt = self.augment(query, chunks)
        t2 = time.time()

        # 3. 生成
        answer = self.generate(prompt)
        t3 = time.time()

        return {
            "query": query,
            "answer": answer,
            "sources": [
                {"title": c.title, "content": c.content, "score": s} for c, s in chunks
            ],
            "timings": {
                "retrieve_ms": round((t1 - t0) * 1000, 1),
                "augment_ms": round((t2 - t1) * 1000, 1),
                "generate_ms": round((t3 - t2) * 1000, 1),
                "total_ms": round((t3 - t0) * 1000, 1),
            },
        }


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="基础 RAG 管线演示")
    parser.add_argument("--query", default="恒瑞医药生产什么药物？", help="查询问题")
    parser.add_argument("--top-k", type=int, default=3, help="检索 Top-K")
    parser.add_argument("--json", action="store_true", help="输出 JSON 格式")
    args = parser.parse_args()

    # 1. 构建知识库
    print("=" * 60)
    print("[RAG Demo] 初始化知识库")
    print("=" * 60)
    chunks = chunk_documents(SAMPLE_DOCUMENTS)
    print(f"文档数: {len(SAMPLE_DOCUMENTS)}, 文档块数: {len(chunks)}")

    # 2. 构建向量索引
    store = SimpleVectorStore()
    store.add_documents(chunks)

    # 3. 创建 RAG 管线
    pipeline = RAGPipeline(store)

    # 4. 执行查询
    print("\n" + "=" * 60)
    print(f"[Query] {args.query}")
    print("=" * 60)

    result = pipeline.ask(args.query, top_k=args.top_k)

    # 5. 输出结果
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"\n[Answer] {result['answer']}")
        print(f"\n[Timings] Total: {result['timings']['total_ms']}ms"
              f"  (Retrieve: {result['timings']['retrieve_ms']}ms"
              f"  Augment: {result['timings']['augment_ms']}ms"
              f"  Generate: {result['timings']['generate_ms']}ms)")

        print(f"\n[Sources] ({len(result['sources'])}):")
        for i, src in enumerate(result["sources"], 1):
            print(f"  [{i}] {src['title']} (得分: {src['score']:.3f})")


if __name__ == "__main__":
    main()
