"""
第 15 章 Demo：RAG 评估体系建设

演示 RAG 系统评估的完整框架：
  检索质量（Recall@K, MRR, NDCG, HitRate）
  生成质量（Correctness, Faithfulness, Relevance）
  自动评估框架（RAGAS 风格）
  人工标注模拟
  线上监控系统

可独立运行，无需外部依赖。

用法：
  python evaluation_framework.py
  python evaluation_framework.py --mode retrieval
  python evaluation_framework.py --mode generation
"""

import argparse
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


# ============================================================================
# Part 1: Retrieval Quality Metrics
# ============================================================================


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Recall@K: 前 k 个结果中相关文档的覆盖率。"""
    if not relevant:
        return 0.0
    retrieved_k = set(retrieved[:k])
    hits = len(retrieved_k & relevant)
    return hits / len(relevant)


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    """MRR: 第一个相关结果的排名的倒数。"""
    for i, doc in enumerate(retrieved):
        if doc in relevant:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved: list[str], relevance: dict[str, float], k: int) -> float:
    """NDCG@K: 考虑相关度分级的排序质量。"""

    def dcg(items: list[tuple[str, float]]) -> float:
        return sum(rel / math.log2(i + 2) for i, (_, rel) in enumerate(items))

    actual = [(doc, relevance.get(doc, 0.0)) for doc in retrieved[:k]]
    actual_dcg = dcg(actual)

    ideal = sorted(relevance.items(), key=lambda x: -x[1])[:k]
    ideal_dcg = dcg(ideal)

    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def hit_rate(retrieved: list[str], relevant: set[str]) -> float:
    """HitRate: 只要有相关结果就算命中。"""
    return 1.0 if any(doc in relevant for doc in retrieved) else 0.0


# ============================================================================
# Part 2: Generation Quality Metrics
# ============================================================================


class AnswerCorrectness:
    """答案正确性评估（基于关键信息匹配）。"""

    def evaluate(self, answer: str, ground_truth: str) -> dict:
        answer_info = self._extract_info(answer)
        truth_info = self._extract_info(ground_truth)

        if not truth_info:
            return {"score": 1.0, "detail": "ground_truth 无有效信息"}

        tp = len(answer_info & truth_info)
        fp = len(answer_info - truth_info)
        fn = len(truth_info - answer_info)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0)

        return {"f1": round(f1, 3), "precision": round(precision, 3),
                "recall": round(recall, 3), "tp": tp, "fp": fp, "fn": fn}

    def _extract_info(self, text: str) -> set[str]:
        """提取关键信息（关键词）。"""
        stopwords = {"的", "了", "是", "在", "和", "有", "也", "就", "不",
                     "都", "而", "且", "或", "与", "及", "但", "被", "把"}
        words = text.replace("。", " ").replace("，", " ").replace("、", " ").split()
        return {w for w in words if len(w) > 1 and w not in stopwords}


class FaithfulnessEvaluator:
    """忠实度评估。"""

    def evaluate(self, answer: str, contexts: list[str]) -> float:
        claims = self._decompose(answer)
        if not claims:
            return 1.0

        supported = sum(1 for c in claims if self._is_supported(c, contexts))
        return round(supported / len(claims), 3)

    def _decompose(self, text: str) -> list[str]:
        sents = [s.strip() for s in text.replace("。", ".").split(".") if s.strip()]
        return [s for s in sents if len(s) > 3]

    def _is_supported(self, claim: str, contexts: list[str]) -> bool:
        claim_words = set(claim.lower().split())
        if not claim_words:
            return True
        for ctx in contexts:
            ctx_words = set(ctx.lower().split())
            overlap = len(claim_words & ctx_words)
            if overlap / len(claim_words) > 0.6:
                return True
        return False


class RelevanceEvaluator:
    """答案与问题的相关性评估。"""

    def evaluate(self, question: str, answer: str) -> float:
        q_words = self._tokenize(question)
        a_words = self._tokenize(answer)
        if not q_words:
            return 1.0
        covered = sum(1 for w in q_words if w in a_words)
        return round(covered / len(q_words), 3)

    def _tokenize(self, text: str) -> set[str]:
        stopwords = {"的", "了", "是", "在", "和", "有", "也", "就"}
        return {w for w in text if len(w) > 1 and w not in stopwords}


# ============================================================================
# Part 3: RAGAS-style Auto Evaluation
# ============================================================================


class RAGASLikeEvaluator:
    """RAGAS 风格的综合评估。"""

    def __init__(self):
        self.faithful = FaithfulnessEvaluator()
        self.relevant = RelevanceEvaluator()
        self.correct = AnswerCorrectness()

    def evaluate(self, dataset: list[dict]) -> dict:
        f_scores, r_scores, c_scores = [], [], []

        for item in dataset:
            f_scores.append(self.faithful.evaluate(item["answer"], item["contexts"]))
            r_scores.append(self.relevant.evaluate(item["question"], item["answer"]))
            if "ground_truth" in item:
                c_scores.append(
                    self.correct.evaluate(item["answer"], item["ground_truth"])["f1"]
                )

        n = len(dataset)
        return {
            "faithfulness": round(sum(f_scores) / n, 3),
            "answer_relevancy": round(sum(r_scores) / n, 3),
            "answer_correctness": round(sum(c_scores) / n, 3) if c_scores else None,
            "sample_count": n,
        }


# ============================================================================
# Part 4: Simulated Annotation
# ============================================================================


@dataclass
class AnnotationTask:
    qid: str
    question: str
    answer: str
    contexts: list[str]
    scores: dict = field(default_factory=dict)
    comment: str = ""


class AnnotationSimulator:
    """模拟人工标注。"""

    def __init__(self):
        self.tasks: list[AnnotationTask] = []
        self.results: list[AnnotationTask] = []

    def create_batch(self, sessions: list[dict]):
        for sess in sessions:
            self.tasks.append(AnnotationTask(
                qid=sess["id"],
                question=sess["question"],
                answer=sess["answer"],
                contexts=sess.get("contexts", []),
            ))

    def simulate_annotation(self, seed: int = 42):
        """模拟多位标注者打分。"""
        random.seed(seed)
        for task in self.tasks:
            # 模拟 3 位标注者
            annotators = []
            for a in range(3):
                base = random.uniform(3.0, 5.0)
                annotators.append({
                    "correctness": min(5, max(1, round(base + random.gauss(0, 0.5), 1))),
                    "completeness": min(5, max(1, round(base + random.gauss(0, 0.3), 1))),
                    "helpfulness": min(5, max(1, round(base + random.gauss(0, 0.4), 1))),
                })
            # 取平均
            task.scores = {
                "correctness": round(sum(a["correctness"] for a in annotators) / 3, 1),
                "completeness": round(sum(a["completeness"] for a in annotators) / 3, 1),
                "helpfulness": round(sum(a["helpfulness"] for a in annotators) / 3, 1),
            }
            task.comment = "人工标注完成"
            self.results.append(task)

        self.tasks.clear()
        return self.results


# ============================================================================
# Part 5: Online Monitor
# ============================================================================


class OnlineMonitor:
    """线上监控模拟。"""

    def __init__(self):
        self.queries: list[dict] = []
        self.feedback: list[dict] = []

    def log_query(self, question: str, latency_ms: float, tokens: int, success: bool):
        self.queries.append({
            "time": time.time(), "question": question,
            "latency_ms": latency_ms, "tokens": tokens, "success": success,
        })

    def log_feedback(self, qid: str, rating: int, ftype: str):
        self.feedback.append({
            "time": time.time(), "qid": qid,
            "rating": rating, "type": ftype,
        })

    def summary(self) -> dict:
        total = len(self.queries)
        if total == 0:
            return {"total": 0}

        failed = sum(1 for q in self.queries if not q["success"])
        lats = sorted(q["latency_ms"] for q in self.queries)
        total_tokens = sum(q["tokens"] for q in self.queries)

        ratings = [f["rating"] for f in self.feedback]
        avg_rating = sum(ratings) / len(ratings) if ratings else 0

        return {
            "total_queries": total,
            "error_rate": round(failed / total, 4),
            "p50_latency_ms": lats[total // 2],
            "p99_latency_ms": lats[int(total * 0.99)],
            "avg_tokens_per_query": round(total_tokens / total, 0),
            "avg_user_rating": round(avg_rating, 2),
            "feedback_count": len(self.feedback),
        }


# ============================================================================
# Demo Functions
# ============================================================================


def generate_test_dataset() -> list[dict]:
    """生成测试数据集。"""
    return [
        {"id": "q1", "question": "恒瑞医药生产哪些药品？",
         "answer": "恒瑞医药生产注射用紫杉醇、奥沙利铂和卡培他滨。",
         "contexts": ["恒瑞医药是制药企业，生产注射用紫杉醇、奥沙利铂等抗肿瘤药物。",
                      "卡培他滨是恒瑞医药的产品之一。"],
         "ground_truth": "恒瑞医药生产注射用紫杉醇、奥沙利铂和卡培他滨。"},
        {"id": "q2", "question": "紫杉醇的治疗机制是什么？",
         "answer": "紫杉醇是一种微管抑制剂，通过促进微管蛋白聚合、抑制微管解聚来发挥抗肿瘤作用。",
         "contexts": ["紫杉醇是微管抑制剂，通过促进微管蛋白聚合来抑制肿瘤细胞分裂。"],
         "ground_truth": "紫杉醇是微管抑制剂，通过促进微管蛋白聚合来发挥抗肿瘤作用。"},
        {"id": "q3", "question": "北京协和医院使用哪些抗肿瘤药物？",
         "answer": "北京协和医院肿瘤科使用注射用紫杉醇和顺铂等抗肿瘤药物。",
         "contexts": ["北京协和医院肿瘤科使用注射用紫杉醇和顺铂。",
                      "北京协和医院是大型综合医院。"],
         "ground_truth": "北京协和医院使用注射用紫杉醇和顺铂。"},
        {"id": "q4", "question": "国药控股的业务范围是什么？",
         "answer": "国药控股是药品分销商，为恒瑞医药和齐鲁制药提供分销服务。",
         "contexts": ["国药控股是恒瑞医药和齐鲁制药的分销合作伙伴。"],
         "ground_truth": "国药控股是药品分销商，为恒瑞医药和齐鲁制药分销药品。"},
        {"id": "q5", "question": "NMPA 在医药行业中的角色是什么？",
         "answer": "NMPA 是监管机构，负责药品的审批和市场监管。",
         "contexts": ["NMPA 是国家药品监督管理局，负责药品审批和监管。",
                      "NMPA 监管注射用紫杉醇在中国市场的审批。"],
         "ground_truth": "NMPA 是国家药品监督管理局，负责药品审批和市场监管。"},
    ]


def demo_retrieval_metrics():
    """演示检索质量评估指标。"""
    print("\n" + "=" * 50)
    print("[Demo] Retrieval Quality Metrics")
    print("=" * 50)

    # 模拟检索结果
    all_docs = [f"doc_{i}" for i in range(20)]
    relevant = {"doc_0", "doc_2", "doc_5"}
    relevance_grades = {"doc_0": 3, "doc_2": 2, "doc_5": 1, "doc_1": 0, "doc_3": 0}

    # 两种检索策略
    retriever_a = ["doc_0", "doc_1", "doc_2", "doc_3", "doc_4", "doc_5"]
    retriever_b = ["doc_5", "doc_4", "doc_3", "doc_2", "doc_1", "doc_0"]

    print("\n  Related docs: doc_0, doc_2, doc_5")
    print(f"\n  {'Metric':<20} {'Retriever A':>12} {'Retriever B':>12} {'说明'}")
    print(f"  {'─' * 60}")

    for k in [1, 3, 5]:
        ra = recall_at_k(retriever_a, relevant, k)
        rb = recall_at_k(retriever_b, relevant, k)
        print(f"  {'Recall@' + str(k):<20} {ra:>12.3f} {rb:>12.3f} {'前' + str(k) + '个结果的覆盖率'}")

    ra_mrr = mrr(retriever_a, relevant)
    rb_mrr = mrr(retriever_b, relevant)
    print(f"  {'MRR':<20} {ra_mrr:>12.3f} {rb_mrr:>12.3f} {'第一个相关结果的排名倒数'}")

    for k in [3, 5]:
        ra_ndcg = ndcg_at_k(retriever_a, relevance_grades, k)
        rb_ndcg = ndcg_at_k(retriever_b, relevance_grades, k)
        print(f"  {'NDCG@' + str(k):<20} {ra_ndcg:>12.3f} {rb_ndcg:>12.3f} {'排序质量（考虑相关度分级）'}")

    ra_hit = hit_rate(retriever_a, relevant)
    rb_hit = hit_rate(retriever_b, relevant)
    print(f"  {'HitRate':<20} {ra_hit:>12.3f} {rb_hit:>12.3f} {'是否有相关结果'}")

    print(f"\n  分析: Retriever A 将最相关文档排在最前 → 各项指标优于 B")


def demo_generation_metrics():
    """演示生成质量评估。"""
    print("\n" + "=" * 50)
    print("[Demo] Generation Quality Metrics")
    print("=" * 50)

    dataset = generate_test_dataset()

    correct = AnswerCorrectness()
    faithful = FaithfulnessEvaluator()
    relevant = RelevanceEvaluator()

    # 逐条评估
    print(f"\n  {'Question':<30} {'Correct':>8} {'Faithful':>10} {'Relevant':>10}")
    print(f"  {'─' * 60}")

    for item in dataset:
        c = correct.evaluate(item["answer"], item["ground_truth"])
        f = faithful.evaluate(item["answer"], item["contexts"])
        r = relevant.evaluate(item["question"], item["answer"])
        q_short = item["question"][:28] + ".." if len(item["question"]) > 28 else item["question"]
        print(f"  {q_short:<30} {c['f1']:>8.3f} {f:>10.3f} {r:>10.3f}")

    # RAGAS 综合评估
    print(f"\n  RAGAS 风格综合评估:")
    evaluator = RAGASLikeEvaluator()
    result = evaluator.evaluate(dataset)
    for metric, value in result.items():
        if value is not None:
            print(f"    {metric:<22}: {value:.3f}")


def demo_annotation():
    """演示人工标注流程。"""
    print("\n" + "=" * 50)
    print("[Demo] Manual Annotation Simulation")
    print("=" * 50)

    dataset = generate_test_dataset()
    simulator = AnnotationSimulator()
    simulator.create_batch(dataset)
    results = simulator.simulate_annotation()

    print(f"\n  {'Question':<30} {'Correct':>9} {'Complete':>9} {'Helpful':>9}")
    print(f"  {'─' * 60}")
    for r in results[:3]:
        q_short = r.question[:28] + ".." if len(r.question) > 28 else r.question
        print(f"  {q_short:<30} {r.scores['correctness']:>9.1f} "
              f"{r.scores['completeness']:>9.1f} {r.scores['helpfulness']:>9.1f}")

    avg = {k: round(sum(r.scores[k] for r in results) / len(results), 1)
           for k in ["correctness", "completeness", "helpfulness"]}
    print(f"  {'─' * 60}")
    print(f"  {'平均分':<30} {avg['correctness']:>9.1f} {avg['completeness']:>9.1f} {avg['helpfulness']:>9.1f}")


def demo_online_monitoring():
    """演示线上监控。"""
    print("\n" + "=" * 50)
    print("[Demo] Online Monitoring")
    print("=" * 50)

    monitor = OnlineMonitor()

    # 模拟 100 次查询
    random.seed(42)
    print(f"\n  模拟 100 次线上查询...")
    for i in range(100):
        latency = random.gauss(1800, 400)  # LLM 生成延迟为主
        tokens = random.randint(200, 2000)
        success = random.random() > 0.03  # 3% 错误率
        monitor.log_query(
            f"查询_{i}", max(100, latency), tokens, success
        )
        if i % 10 == 0:
            monitor.log_feedback(f"q_{i}", random.randint(3, 5), "thumbs_up")

    summary = monitor.summary()
    print(f"\n  {'指标':<25} {'值':<15}")
    print(f"  {'─' * 40}")
    for key, value in summary.items():
        print(f"  {key:<25} {value:<15}")


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="RAG 评估体系 Demo")
    parser.add_argument("--mode", choices=["retrieval", "generation", "annotation",
                                           "monitoring", "all"],
                        default="all")
    args = parser.parse_args()

    print("=" * 60)
    print("RAG 评估体系建设 Demo")
    print("=" * 60)

    modes = (["retrieval", "generation", "annotation", "monitoring"]
             if args.mode == "all" else [args.mode])

    for m in modes:
        if m == "retrieval":
            demo_retrieval_metrics()
        elif m == "generation":
            demo_generation_metrics()
        elif m == "annotation":
            demo_annotation()
        elif m == "monitoring":
            demo_online_monitoring()

    print("\n" + "=" * 60)
    print("模式说明:")
    print("  --mode retrieval   检索质量指标 (Recall/MRR/NDCG/HitRate)")
    print("  --mode generation  生成质量指标 (Correctness/Faithfulness/Relevance)")
    print("  --mode annotation  人工标注模拟")
    print("  --mode monitoring  线上监控模拟")
    print("=" * 60)


if __name__ == "__main__":
    main()
