"""
第 12 章 Demo：Agentic RAG

演示多种 Agentic RAG 模式：
  ReAct 循环（思考→行动→观察→最终答案）
  Corrective RAG（检索质量评估 + 自动重试）
  Self-RAG（逐片段自省）
  多 Agent 协作（Orchestrator/Worker）

可独立运行，无需外部依赖。

用法：
  python agentic_rag.py
  python agentic_rag.py --query "恒瑞医药生产哪些药品？" --mode react
  python agentic_rag.py --query "紫杉醇的供应链是怎样的？" --mode multi-agent
"""

import argparse
import json
import math
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Callable


# ============================================================================
# Base Data Structures
# ============================================================================


@dataclass
class Chunk:
    id: str
    text: str
    source: str = ""


@dataclass
class Evidence:
    text: str
    source: str = ""
    relevance: float = 0.0


@dataclass
class Thought:
    step: int
    content: str
    action: Optional[str] = None
    observation: Optional[str] = None


@dataclass
class Message:
    sender: str
    receiver: str
    content: str
    msg_type: str = "info"


# ============================================================================
# Mock Embedding & Retriever
# ============================================================================


class SimpleEmbedding:
    def embed(self, text: str) -> list[float]:
        features = [0.0] * 24
        for i, ch in enumerate(text[:200]):
            features[hash(ch) % 24] += 1.0
        norm = math.sqrt(sum(v * v for v in features)) or 1.0
        return [v / norm for v in features]

    def similarity(self, a: list[float], b: list[float]) -> float:
        return sum(av * bv for av, bv in zip(a, b))


class SimpleRetriever:
    def __init__(self, embedding: SimpleEmbedding):
        self.embedding = embedding
        self.chunks: list[Chunk] = []
        self.embeddings: list[list[float]] = []

    def add_chunks(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.embeddings = [self.embedding.embed(c.text) for c in chunks]

    def search(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        q_emb = self.embedding.embed(query)
        scored = []
        for i, chunk in enumerate(self.chunks):
            score = self.embedding.similarity(q_emb, self.embeddings[i])
            scored.append((chunk, score))
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]


# ============================================================================
# Knowledge Graph (in-memory)
# ============================================================================


@dataclass
class KGEntity:
    name: str
    type: str
    description: str = ""


@dataclass
class KGRelation:
    source: str
    target: str
    rel_type: str
    description: str = ""


class KnowledgeGraph:
    """内存知识图谱。"""

    def __init__(self):
        self.entities: dict[str, KGEntity] = {}
        self.adjacency: dict[str, list[tuple[str, str, str]]] = defaultdict(list)

    def add_entity(self, name: str, type: str, desc: str = ""):
        self.entities[name] = KGEntity(name=name, type=type, description=desc)

    def add_relation(self, source: str, target: str, rel_type: str, desc: str = ""):
        if source in self.entities and target in self.entities:
            self.adjacency[source].append((rel_type, target, desc))

    def query(self, query: str) -> str:
        """基于查询返回结构化结果。"""
        # 提取查询中出现的实体
        mentioned = [e for e in self.entities if e in query]
        if not mentioned:
            return "知识图谱中未找到相关实体。"

        lines = []
        for name in mentioned[:3]:
            entity = self.entities[name]
            lines.append(f"实体: {name} ({entity.type}) — {entity.description}")

            if name in self.adjacency:
                for rel_type, target, desc in self.adjacency[name][:5]:
                    if target in self.entities:
                        t = self.entities[target]
                        lines.append(f"  -> [{rel_type}] {target} ({t.type})")

        return "\n".join(lines)


def build_sample_kg() -> KnowledgeGraph:
    """构建示例知识图谱。"""
    kg = KnowledgeGraph()
    entities = [
        ("恒瑞医药", "制药企业", "中国领先的抗肿瘤药物研发和生产企业"),
        ("华海药业", "原料药企业", "紫杉醇 API 原料药供应商"),
        ("国药控股", "分销商", "中国最大的药品分销商之一"),
        ("齐鲁制药", "制药企业", "大型制药企业，生产顺铂和卡培他滨"),
        ("北京协和医院", "医院", "三级甲等综合医院"),
        ("上海中山医院", "医院", "三级甲等综合医院"),
        ("注射用紫杉醇", "药品", "微管抑制剂类抗肿瘤化疗药物"),
        ("奥沙利铂", "药品", "铂类抗肿瘤化疗药物"),
        ("卡培他滨", "药品", "口服氟尿嘧啶类抗肿瘤药物"),
        ("顺铂", "药品", "应用最广泛的铂类抗肿瘤药物"),
        ("NMPA", "监管机构", "国家药品监督管理局"),
    ]
    for name, type_, desc in entities:
        kg.add_entity(name, type_, desc)

    relations = [
        ("恒瑞医药", "注射用紫杉醇", "生产"),
        ("恒瑞医药", "奥沙利铂", "生产"),
        ("恒瑞医药", "卡培他滨", "生产"),
        ("齐鲁制药", "顺铂", "生产"),
        ("齐鲁制药", "卡培他滨", "生产"),
        ("华海药业", "恒瑞医药", "供应API", "供应紫杉醇 API"),
        ("国药控股", "恒瑞医药", "分销"),
        ("国药控股", "齐鲁制药", "分销"),
        ("北京协和医院", "注射用紫杉醇", "使用"),
        ("北京协和医院", "顺铂", "使用"),
        ("上海中山医院", "注射用紫杉醇", "使用"),
        ("NMPA", "注射用紫杉醇", "监管"),
    ]
    for src, tgt, rel, *desc in relations:
        kg.add_relation(src, tgt, rel, desc[0] if desc else "")

    return kg


# ============================================================================
# Sample Documents
# ============================================================================


SAMPLE_CHUNKS = [
    Chunk("c1", "恒瑞医药是中国领先的制药企业，专注于抗肿瘤药物的研发和生产。主要产品包括注射用紫杉醇、奥沙利铂和卡培他滨。公司总部位于江苏省连云港市。", "doc1"),
    Chunk("c2", "注射用紫杉醇是一种微管抑制剂，通过促进微管蛋白聚合抑制肿瘤细胞分裂。主要用于非小细胞肺癌、乳腺癌和卵巢癌的治疗。", "doc2"),
    Chunk("c3", "奥沙利铂是第三代铂类抗肿瘤药物，主要用于转移性结直肠癌的治疗。常与氟尿嘧啶联合使用。", "doc2"),
    Chunk("c4", "卡培他滨是一种口服氟尿嘧啶类抗肿瘤药物。用于结直肠癌和乳腺癌的治疗。可单药或联合化疗方案中使用。", "doc2"),
    Chunk("c5", "齐鲁制药是中国大型制药企业，主要生产顺铂和卡培他滨。顺铂是应用最广泛的铂类抗肿瘤药物，用于肺癌、卵巢癌等多种实体瘤。", "doc3"),
    Chunk("c6", "国药控股是恒瑞医药和齐鲁制药的重要分销合作伙伴。拥有覆盖全国的药品分销网络和冷链物流体系。", "doc4"),
    Chunk("c7", "华海药业为恒瑞医药提供紫杉醇 API 原料药。其在原料药领域拥有丰富经验，年产能超过500公斤。", "doc5"),
    Chunk("c8", "北京协和医院肿瘤科广泛使用注射用紫杉醇和顺铂等抗肿瘤药物，用于各类实体瘤的综合治疗。", "doc6"),
]


# ============================================================================
# ReAct Agent
# ============================================================================


class ReActAgent:
    """ReAct 模式 Agent：思考→行动→观察→最终答案。"""

    def __init__(self, retriever: SimpleRetriever, kg: KnowledgeGraph, llm=None):
        self.retriever = retriever
        self.kg = kg
        self.thoughts: list[Thought] = []

    def _think(self, question: str, context: list[str]) -> str:
        """模拟思考过程（实际中由 LLM 完成）。"""
        # 基于问题关键词决定下一步
        question_lower = question.lower()

        # 如果是关系型问题，用知识图谱
        if any(w in question_lower for w in ["供应链", "关系", "合作", "分销", "供应"]):
            return "kg_query"

        # 如果提到了具体药品或公司名
        for name in self.kg.entities:
            if name in question:
                return "vector_search"

        # 默认走向量检索
        return "vector_search"

    def _act(self, action: str, query: str) -> str:
        """执行动作。"""
        if action == "kg_query":
            return self.kg.query(query)
        elif action == "vector_search":
            results = self.retriever.search(query, top_k=3)
            return "\n".join(f"[{c.source}]({score:.2f}) {c.text}" for c, score in results)
        else:
            return f"未知动作: {action}"

    def answer(self, question: str, max_steps: int = 5) -> str:
        """ReAct 循环生成答案。"""
        self.thoughts = []
        context = []

        print(f"\n  ReAct 推理过程:")
        print(f"  {'─' * 50}")

        for step in range(max_steps):
            # Think
            action_type = self._think(question, context)
            query = question  # 简化：直接用原问题

            # Print thought
            print(f"\n  步骤 {step + 1}: Thought — 需要执行 {action_type}")
            print(f"  Action — {action_type}('{query[:40]}...')")

            # Act
            observation = self._act(action_type, query)
            print(f"  Observation — {observation[:80]}...")

            thought = Thought(
                step=step + 1,
                content=f"执行 {action_type} 查询",
                action=f"{action_type}({query})",
                observation=observation,
            )
            self.thoughts.append(thought)
            context.append(observation)

            # 决定是否终止循环
            if step >= 1 and self._has_sufficient_info(question, context):
                print(f"  → 信息足够，终止推理")
                break

        # Final answer
        answer = self._generate_final_answer(question, context)
        print(f"\n  {'─' * 50}")

        return answer

    def _has_sufficient_info(self, question: str, context: list[str]) -> bool:
        """检查是否已有足够信息。"""
        combined = " ".join(context)
        # 检查是否覆盖了问题中的关键实体
        mentioned = 0
        for name in self.kg.entities:
            if name in question and name in combined:
                mentioned += 1

        question_entities = [e for e in self.kg.entities if e in question]
        return mentioned >= len(question_entities) if question_entities else len(context) >= 1

    def _generate_final_answer(self, question: str, context: list[str]) -> str:
        """基于上下文生成最终答案。"""
        combined = " ".join(context)
        lines = ["最终答案：\n"]

        if "药品" in question or "什么药品" in question or "哪些药品" in question:
            drugs = []
            for name, entity in self.kg.entities.items():
                if entity.type == "药品":
                    drugs.append(name)
            if drugs:
                lines.append(f"根据知识图谱，找到以下相关药品：{', '.join(drugs)}")
        elif "供应链" in question or "路径" in question:
            lines.append("供应链路径：")
            for thought in self.thoughts:
                if thought.observation:
                    lines.append(f"  {thought.observation}")
        else:
            lines.append(f"基于检索结果：{combined[:200]}")

        return "\n".join(lines)


# ============================================================================
# Corrective RAG (CRAG)
# ============================================================================


class CRAgent:
    """Corrective RAG：检索→评估→纠正的循环。"""

    def __init__(self, retriever: SimpleRetriever, kg: KnowledgeGraph):
        self.retriever = retriever
        self.kg = kg
        self.retry_count = 0

    def _evaluate_relevance(self, query: str, chunks: list[tuple[Chunk, float]]) -> str:
        """评估检索结果质量。"""
        if not chunks:
            return "incorrect"

        avg_score = sum(s for _, s in chunks) / len(chunks)

        # 检查是否包含问题中的关键实体
        question_entities = [e for e in self.kg.entities if e in query]
        if question_entities:
            combined_text = " ".join(c.text for c, _ in chunks)
            found = sum(1 for e in question_entities if e in combined_text)
            coverage = found / len(question_entities)
            if coverage < 0.5:
                return "incorrect"

        if avg_score > 0.6:
            return "correct"
        elif avg_score > 0.3:
            return "ambiguous"
        return "incorrect"

    def _rewrite_query(self, question: str) -> str:
        """重写查询以改善检索。"""
        self.retry_count += 1
        # 简化：提取关键实体重组查询
        entities = [e for e in self.kg.entities if e in question]
        if len(entities) >= 2:
            # 如果图谱中有关系，用关系描述重写
            for src in entities:
                if src in self.kg.adjacency:
                    for rel_type, tgt, desc in self.kg.adjacency[src][:1]:
                        if tgt in question:
                            return f"{src} {rel_type} {tgt}"
        return f"{question} 详细情况"

    def answer(self, question: str, max_retries: int = 3) -> str:
        """CRAG 流程。"""
        self.retry_count = 0

        print(f"\n  CRAG 纠正循环:")
        print(f"  {'─' * 50}")

        for attempt in range(max_retries):
            # 1. 检索（如果重试则用改写后的问题）
            current_query = self._rewrite_query(question) if attempt > 0 else question
            chunks = self.retriever.search(current_query, top_k=5)

            # 2. 评估
            verdict = self._evaluate_relevance(question, chunks)
            print(f"  尝试 {attempt + 1}: 查询='{current_query[:40]}...' → 评估={verdict}")

            if verdict == "correct":
                print(f"  → 检索质量合格，生成答案")
                return self._generate(question, chunks)

            elif verdict == "ambiguous":
                # 保留最佳结果，补充检索
                supplement = self.retriever.search(self._rewrite_query(question), top_k=2)
                print(f"  → 结果模糊，补充检索中...")
                return self._generate(question, chunks + supplement)

            # incorrect：继续重试
            print(f"  → 结果不相关，重写查询重试...")

        # 兜底：即使检索结果不理想也生成答案
        chunks = self.retriever.search(question, top_k=3)
        return self._generate(question, chunks, fallback=True)

    def _generate(self, question: str, chunks: list[tuple[Chunk, float]], fallback: bool = False) -> str:
        """基于检索结果生成答案。"""
        if not chunks:
            return "未能找到相关信息。请尝试其他问题。"

        text = "\n".join(f"[{c.source}]({s:.2f}) {c.text}" for c, s in chunks[:3])

        if fallback:
            return f"（基于有限信息）根据检索结果：\n{text}"

        lines = ["答案：\n"]
        for c, s in chunks[:3]:
            lines.append(f"  - {c.text} (相关性: {s:.2f})")
        return "\n".join(lines)


# ============================================================================
# Multi-Agent Orchestrator
# ============================================================================


class KnowledgeGraphWorker:
    """知识图谱 Worker。"""
    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg
    def run(self, query: str) -> str:
        return self.kg.query(query)


class VectorSearchWorker:
    """向量检索 Worker。"""
    def __init__(self, retriever: SimpleRetriever):
        self.retriever = retriever
    def run(self, query: str) -> str:
        results = self.retriever.search(query, top_k=3)
        return "\n".join(f"[{c.source}]({s:.2f}) {c.text}" for c, s in results)


class WebSearchWorker:
    """模拟网页搜索 Worker。"""
    def run(self, query: str) -> str:
        return f"[模拟网页搜索] 关于「{query}」的最新信息：暂无公开数据。"


class Orchestrator:
    """多 Agent 编排器。"""

    def __init__(self, kg: KnowledgeGraph, retriever: SimpleRetriever):
        self.workers = {
            "kg": KnowledgeGraphWorker(kg),
            "vector": VectorSearchWorker(retriever),
            "web": WebSearchWorker(),
        }

    def decompose(self, question: str) -> list[tuple[str, str, int]]:
        """分解问题为子任务：列表为 (worker_name, query, priority)。"""
        tasks = []

        # 知识图谱查询（高优先级）
        entities = [e for e in [
            "恒瑞医药", "华海药业", "国药控股", "齐鲁制药",
            "北京协和医院", "注射用紫杉醇", "奥沙利铂", "卡培他滨", "顺铂"
        ] if e in question]
        if entities:
            tasks.append(("kg", question, 1))

        # 向量检索（中优先级）
        tasks.append(("vector", question, 2))

        # 网页搜索（低优先级）— 仅当问题涉及"最新"内容
        if any(w in question for w in ["最新", "近期", "最近"]):
            tasks.append(("web", question, 3))

        return tasks

    def run(self, question: str) -> str:
        """编排多 Agent 并返回综合结果。"""
        tasks = self.decompose(question)
        tasks.sort(key=lambda x: x[2])  # 按优先级排序

        print(f"\n  编排器: 将问题分解为 {len(tasks)} 个子任务")
        print(f"  {'─' * 50}")

        results = {}
        for worker_name, query, priority in tasks:
            worker = self.workers[worker_name]
            result = worker.run(query)
            results[worker_name] = result
            print(f"  Worker [{worker_name}]: 完成查询")

        # 综合结果
        combined = []
        if "kg" in results:
            combined.append(f"【知识图谱】{results['kg']}")
        if "vector" in results:
            combined.append(f"【向量检索】{results['vector'][:100]}...")
        if "web" in results:
            combined.append(f"【网页】{results['web']}")

        return "\n\n".join(combined)


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Agentic RAG Demo")
    parser.add_argument("--query", default="恒瑞医药生产哪些药品？", help="查询问题")
    parser.add_argument(
        "--mode",
        choices=["react", "crag", "multi-agent", "all"],
        default="all",
        help="Agent 模式",
    )
    args = parser.parse_args()

    # 初始化
    embedding = SimpleEmbedding()
    retriever = SimpleRetriever(embedding)
    retriever.add_chunks(SAMPLE_CHUNKS)
    kg = build_sample_kg()

    query = args.query
    mode = args.mode

    print("=" * 60)
    print("Agentic RAG Demo")
    print("=" * 60)
    print(f"Query: {query}\n")

    modes_to_run = ["react", "crag", "multi-agent"] if mode == "all" else [mode]

    for m in modes_to_run:
        print(f"\n{'=' * 50}")
        print(f"[Mode: {m.upper()}]")
        print(f"{'=' * 50}")

        if m == "react":
            agent = ReActAgent(retriever, kg)
            result = agent.answer(query)

        elif m == "crag":
            agent = CRAgent(retriever, kg)
            result = agent.answer(query)

        elif m == "multi-agent":
            orch = Orchestrator(kg, retriever)
            result = orch.run(query)

        print(f"\n{result}")

    # 模式速查
    if mode == "all":
        print(f"\n{'=' * 50}")
        print("模式说明:")
        print("  --mode react       ReAct: 思考→行动→观察→最终答案")
        print("  --mode crag        CRAG: 检索→评估→纠正→答案")
        print("  --mode multi-agent 多Agent: 分解→并行执行→综合")
        print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
