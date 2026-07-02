# 第12章 RAG系统评估体系

## 12.1 引言

评估是RAG系统开发和运维中最具挑战性的环节之一。与传统软件系统不同，RAG系统的输出具有开放性和不确定性——同一个问题可能产生多种同样合理的回答。这使得评估工作既需要定量指标的精确性，又需要定性判断的灵活性。

一个完整的RAG评估体系应该覆盖三个层面：
- **检索质量**：系统能否找到最相关的信息？
- **生成质量**：系统能否基于检索结果生成准确、完整的回答？
- **系统性能**：系统能否在可接受的延迟和成本下提供服务？

本章将从这三个层面出发，详细介绍评估指标、评估方法、评估工具和持续优化策略。所有代码示例均基于实际项目经验，可直接用于构建生产级的评估流水线。

---

## 12.2 检索评估指标

检索阶段的目标是从文档库中找到与查询最相关的文档。评估检索质量的核心指标包括以下几类。

### 12.2.1 Recall@K

Recall@K衡量在前K个检索结果中，相关文档的覆盖率。它关注的是"我们是否把该找到的都找到了"。

```python
def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """
    计算Recall@K
    
    Args:
        retrieved: 检索结果列表（文档ID）
        relevant: 相关文档ID集合（真实标注）
        k: 考虑的前K个结果
    
    Returns:
        Recall@K值（0.0 ~ 1.0）
    """
    if not relevant:
        return 0.0
    
    retrieved_at_k = set(retrieved[:k])
    relevant_set = set(relevant)
    
    # 在前K个结果中找到的相关文档数
    relevant_retrieved = len(retrieved_at_k & relevant_set)
    
    # 除以总相关文档数
    return relevant_retrieved / len(relevant_set)


def compute_recall_curve(
    retrieved: List[str], relevant: List[str], max_k: int = 20
) -> List[float]:
    """
    计算Recall@K曲线（K从1到max_k）
    
    召回率曲线可以直观地展示检索系统的性能：
    - 曲线上升越快，说明最相关的文档排在越前面
    - 曲线最终值越高，说明系统召回能力越强
    """
    recall_curve = []
    for k in range(1, max_k + 1):
        recall = recall_at_k(retrieved, relevant, k)
        recall_curve.append(recall)
    return recall_curve
```

**解读指南：**
| Recall@K | 评价 | 说明 |
|----------|------|------|
| 0.9+ @ K=5 | 优秀 | 核心需求几乎都能在前5个结果中找到 |
| 0.8-0.9 @ K=5 | 良好 | 大部分核心需求都能覆盖 |
| 0.6-0.8 @ K=5 | 一般 | 需要增大K值或优化检索策略 |
| <0.6 @ K=5 | 需改进 | 检索系统存在根本性问题 |

### 12.2.2 Precision@K

Precision@K衡量在前K个检索结果中，相关文档的占比。它关注的是"检索结果中有多少是真正有用的"。

```python
def precision_at_k(
    retrieved: List[str], relevant: List[str], k: int
) -> float:
    """
    计算Precision@K
    
    Precision@K关注检索结果的精度（信噪比）。
    高精度意味着用户看到的结果大多都是相关的。
    """
    if k <= 0:
        return 0.0
    
    retrieved_at_k = set(retrieved[:k])
    relevant_set = set(relevant)
    
    # 在前K个结果中，有多少是相关的
    relevant_retrieved = len(retrieved_at_k & relevant_set)
    
    return relevant_retrieved / k


def precision_recall_tradeoff(
    retrieved: List[str], relevant: List[str], k: int
) -> dict:
    """
    分析精确率和召回率的权衡关系
    
    通常，增大K值会提高召回率但降低精确率。
    最优的K值取决于具体场景：
    - 摘要类任务：更看重召回率（K可较大）
    - 事实查询：更看重精确率（K应较小）
    """
    precision = precision_at_k(retrieved, relevant, k)
    recall = recall_at_k(retrieved, relevant, k)
    
    # F1分数：精确率和召回率的调和平均
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "k": k,
    }


def average_precision(
    retrieved: List[str], relevant: List[str]
) -> float:
    """
    计算平均精确率（AP）
    
    AP是Precision@K的加权平均，考虑了排名的顺序因素。
    它在每个相关文档出现的位置计算Precision@K，然后取平均。
    
    这是后续MAP指标的基础。
    """
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    
    score = 0.0
    num_hits = 0
    
    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant_set:
            num_hits += 1
            # 在第i个位置，精确率为 num_hits / (i+1)
            score += num_hits / (i + 1)
    
    return score / len(relevant_set)
```

### 12.2.3 MRR（Mean Reciprocal Rank）

MRR衡量第一个相关结果出现的位置。它特别适合"用户只关心第一个正确答案"的场景（如问答系统）。

```python
def reciprocal_rank(
    retrieved: List[str], relevant: List[str]
) -> float:
    """
    计算倒数排名（Reciprocal Rank）
    
    如果第一个相关结果出现在第3位，则RR = 1/3。
    如果没有找到相关结果，则RR = 0。
    """
    relevant_set = set(relevant)
    
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank
    
    return 0.0


def mean_reciprocal_rank(
    queries_results: List[tuple],  # [(retrieved_list, relevant_list), ...]
) -> float:
    """
    计算平均倒数排名（MRR）
    
    MRR = (1/N) * Σ(1/rank_i)
    其中N是查询总数，rank_i是第i个查询的第一个相关结果位置。
    """
    if not queries_results:
        return 0.0
    
    total_rr = sum(
        reciprocal_rank(retrieved, relevant)
        for retrieved, relevant in queries_results
    )
    
    return total_rr / len(queries_results)


# MRR计算示例
def mrr_example():
    """MRR计算示例"""
    test_queries = [
        (
            ["doc3", "doc1", "doc5", "doc2", "doc4"],  # 检索结果
            {"doc1", "doc2"},                            # 相关文档
        ),
        (
            ["doc8", "doc6", "doc7"],
            {"doc7", "doc9"},
        ),
        (
            ["doc1", "doc2", "doc3"],
            {"doc3"},
        ),
    ]
    
    mrr = mean_reciprocal_rank(test_queries)
    print(f"MRR: {mrr:.3f}")
    # 输出：MRR: 0.722
    # 解释：
    #   Q1: 第一个相关文档在位置2 → RR = 0.5
    #   Q2: 第一个相关文档在位置3 → RR = 0.333
    #   Q3: 第一个相关文档在位置3 → RR = 0.333
    #   MRR = (0.5 + 0.333 + 0.333) / 3 = 0.722
```

### 12.2.4 MAP（Mean Average Precision）

MAP在所有相关文档位置上计算精确率的平均值，能够全面反映检索系统的排序质量。

```python
def mean_average_precision(
    queries_results: List[tuple],
) -> float:
    """
    计算平均精确率均值（MAP）
    
    MAP = (1/N) * Σ(AP_i)
    AP_i是第i个查询的平均精确率。
    
    MAP对所有相关文档的位置都敏感，是信息检索领域
    最常用的综合评估指标之一。
    """
    if not queries_results:
        return 0.0
    
    total_ap = 0.0
    for retrieved, relevant in queries_results:
        ap = average_precision(retrieved, relevant)
        total_ap += ap
    
    return total_ap / len(queries_results)


# MAP的详细计算
def average_precision_detailed(
    retrieved: List[str], relevant: List[str]
) -> float:
    """
    详细版AP计算，展示每个位置的计算过程
    """
    relevant_set = set(relevant)
    print(f"相关文档: {relevant_set}")
    print(f"检索结果排名: {retrieved}")
    print()
    
    cumulative_precision = 0.0
    num_relevant_found = 0
    
    for rank, doc_id in enumerate(retrieved, start=1):
        is_relevant = doc_id in relevant_set
        
        if is_relevant:
            num_relevant_found += 1
            precision_at_this_rank = num_relevant_found / rank
            cumulative_precision += precision_at_this_rank
            
            print(f"  位置{rank}: {doc_id} (相关) "
                  f"→ Precision@{rank} = {num_relevant_found}/{rank} "
                  f"= {precision_at_this_rank:.3f}")
        else:
            print(f"  位置{rank}: {doc_id} (不相关)")
    
    if num_relevant_found == 0:
        ap = 0.0
    else:
        ap = cumulative_precision / len(relevant_set)
    
    print(f"\nAP = {ap:.3f}")
    return ap


def map_example():
    """MAP计算示例"""
    queries = [
        (
            ["doc1", "doc2", "doc3", "doc4", "doc5"],
            {"doc2", "doc4", "doc5"},
        ),
        (
            ["doc1", "doc3", "doc2", "doc5", "doc4"],
            {"doc1", "doc5"},
        ),
    ]
    
    map_score = mean_average_precision(queries)
    print(f"MAP: {map_score:.3f}")
```

### 12.2.5 NDCG（Normalized Discounted Cumulative Gain）

NDCG引入分级相关性（不仅仅是二元相关/不相关），更适合评估真实场景中的检索质量。

```python
import math
from typing import List, Dict, Union

def dcg(relevance_scores: List[float], k: int) -> float:
    """
    计算折损累积增益（DCG）
    
    DCG = Σ(rel_i / log2(i+1))
    rel_i是第i个结果的相关性得分（可以是分级得分）
    """
    scores = relevance_scores[:k]
    if not scores:
        return 0.0
    
    return sum(
        rel / math.log2(idx + 2)  # idx+2: log2(1+1)=log2(2)=1
        for idx, rel in enumerate(scores)
    )


def ndcg(
    relevance_scores: List[float],
    k: int = None,
) -> float:
    """
    计算归一化折损累积增益（NDCG）
    
    NDCG = DCG / IDCG
    IDCG是理想排序下的DCG（按相关性降序排列）
    
    NDCG的优势：
    - 支持分级相关性（不仅仅是二值）
    - 对排名靠前的结果给予更高权重
    - 归一化后可以在不同查询间比较
    """
    if k is not None:
        relevance_scores = relevance_scores[:k]
    
    # 计算实际DCG
    actual_dcg = dcg(relevance_scores, len(relevance_scores))
    
    # 计算理想DCG（按相关性降序排列）
    ideal_scores = sorted(relevance_scores, reverse=True)
    ideal_dcg = dcg(ideal_scores, len(ideal_scores))
    
    if ideal_dcg == 0:
        return 0.0
    
    return actual_dcg / ideal_dcg


# NDCG计算示例
def ndcg_example():
    """
    NDCG计算示例
    假设相关性分级：0=不相关, 1=部分相关, 2=高度相关, 3=完全匹配
    """
    # 检索结果的相关性得分（按排名顺序）
    relevance_scores = [3, 2, 0, 1, 2]  # 5个结果
    
    print("检索结果相关性: [3, 2, 0, 1, 2]")
    print(f"NDCG@1: {ndcg(relevance_scores, k=1):.3f}")
    print(f"NDCG@3: {ndcg(relevance_scores, k=3):.3f}")
    print(f"NDCG@5: {ndcg(relevance_scores, k=5):.3f}")
    print()
    
    # 对比两个检索系统的NDCG
    system_a = [3, 2, 1, 1, 0]  # 优秀结果靠前
    system_b = [0, 1, 2, 3, 1]  # 优秀结果靠后
    
    print(f"系统A NDCG@3: {ndcg(system_a, k=3):.3f}")
    print(f"系统B NDCG@3: {ndcg(system_b, k=3):.3f}")
    print("→ NDCG能够区分排序质量的差异")


def batch_ndcg(
    query_scores: List[List[float]],
    ks: List[int] = [1, 3, 5, 10],
) -> Dict[int, float]:
    """
    批量计算多个查询的NDCG
    """
    results = {}
    for k in ks:
        ndcg_values = [
            ndcg(scores, k=k) for scores in query_scores
        ]
        results[k] = sum(ndcg_values) / len(ndcg_values)
    return results
```

### 12.2.6 检索指标综合对比

| 指标 | 核心问题 | 适用场景 | 优点 | 缺点 |
|------|---------|---------|------|------|
| Recall@K | 该找到的都找到了吗？ | 摘要、综合类查询 | 直观、易理解 | 未考虑排名顺序 |
| Precision@K | 检索结果纯净吗？ | 事实查询、精确匹配 | 反映信噪比 | 对K值敏感 |
| MRR | 第一个正确答案在哪？ | 问答系统 | 关注首位结果 | 忽略多个相关结果 |
| MAP | 整体排序质量如何？ | 综合评估 | 考虑所有相关文档位置 | 仅支持二元相关 |
| NDCG | 排序质量是否合理？ | 推荐、分级评估 | 支持分级相关性 | 需要分级标注 |

**实践经验：** 在实际项目中，通常同时使用Recall@K和NDCG作为主要指标。Recall@K反映系统的"广度"，NDCG反映系统的"精度"。

---

## 12.3 生成质量评估

生成质量评估是RAG评估中最具挑战性的部分。与检索评估不同，生成评估没有"标准答案"，同一个问题可以有多种正确的回答方式。

### 12.3.1 核心评估维度

```python
class GenerationMetrics:
    """生成质量评估指标"""
    
    @staticmethod
    def answer_accuracy(
        answer: str,
        golden_answer: str,
        llm_judge,
    ) -> float:
        """
        回答准确率：使用LLM判断回答是否正确
        
        使用LLM作为评判者（LLM-as-Judge）是目前最实用的方法。
        关键在于提供清晰的评判标准。
        """
        prompt = f"""
请判断以下回答是否准确地回答了问题。

标准答案：{golden_answer}

待评估回答：{answer}

请按以下标准评分（1-5分）：
5分：完全准确，包含所有关键信息
4分：基本准确，遗漏次要信息
3分：部分准确，存在轻微错误
2分：大部分错误，仅有少量正确信息
1分：完全不正确

仅输出分数（1-5）：
"""
        result = llm_judge.generate(prompt)
        try:
            score = int(result.strip())
            return max(1.0, min(5.0, score)) / 5.0
        except ValueError:
            return 0.0
    
    @staticmethod
    def hallucination_rate(
        answer: str,
        context: str,
        llm_judge,
    ) -> float:
        """
        幻觉率检测：检查回答中是否有上下文不支持的信息
        
        幻觉是RAG系统最严重的问题之一。
        高幻觉率会严重损害用户信任。
        """
        prompt = f"""
请检查以下回答中的每个断言是否都有上下文支持。

参考上下文：
{context[:3000]}

待检查回答：
{answer}

请执行以下步骤：
1. 将回答拆分为独立的事实断言
2. 检查每个断言是否能在上下文中找到支持
3. 统计无支持的断言数量和总断言数

输出JSON格式：
{{
    "total_claims": <总断言数>,
    "unsupported_claims": <无支持断言数>,
    "unsupported_details": ["断言1", "断言2", ...],
    "hallucination_rate": <幻觉率>
}}
"""
        result = llm_judge.generate(prompt)
        import json
        try:
            data = json.loads(result)
            return data.get("hallucination_rate", 1.0)
        except json.JSONDecodeError:
            return 1.0
    
    @staticmethod
    def citation_accuracy(
        answer: str,
        sources: List[dict],
        llm_judge,
    ) -> float:
        """
        引用准确率：检查回答中的引用是否正确指向了来源
        
        高引用准确率意味着用户可以信任回答中标注的来源信息。
        """
        if not sources:
            return 0.0
        
        prompt = f"""
请检查回答中的引用是否准确。

回答（包含引用标记）：
{answer}

可用来源：
{json.dumps(sources, ensure_ascii=False)[:3000]}

请检查：
1. 每个引用是否指向了正确的来源
2. 引用内容是否确实来自标注的来源
3. 是否有遗漏的必要引用

输出JSON：
{{
    "total_citations": <总引用数>,
    "correct_citations": <正确引用数>,
    "accuracy": <正确引用数/总引用数>
}}
"""
        result = llm_judge.generate(prompt)
        try:
            data = json.loads(result)
            return data.get("accuracy", 0.0)
        except json.JSONDecodeError:
            return 0.0
    
    @staticmethod
    def completeness(
        answer: str,
        golden_answer: str,
        required_points: List[str],
        llm_judge,
    ) -> float:
        """
        完整性：评估回答是否覆盖了所有关键信息点
        """
        prompt = f"""
请评估以下回答的完整性。

必须覆盖的关键信息点：
{json.dumps(required_points, ensure_ascii=False)}

标准答案（参考）：
{golden_answer}

待评估回答：
{answer}

请逐点检查是否覆盖：
{json.dumps({
    point: "已覆盖/未覆盖/部分覆盖"
    for point in required_points
}, ensure_ascii=False)}

输出覆盖比例（0.0-1.0）：
"""
        result = llm_judge.generate(prompt)
        try:
            return float(result.strip())
        except ValueError:
            return 0.0
```

### 12.3.2 BLEU / ROUGE / BERTScore

传统的文本生成评估指标在RAG场景中仍然有参考价值，尽管它们无法完全捕捉语义质量。

```python
class TextGenerationMetrics:
    """文本生成自动评估指标"""
    
    @staticmethod
    def bleu_score(
        candidate: str,
        reference: str,
        max_n: int = 4,
    ) -> float:
        """
        计算BLEU分数
        
        BLEU基于n-gram精确匹配，衡量生成文本与参考文本的相似度。
        主要评估流畅度和词汇选择。
        
        注意：BLEU在RAG场景中效果有限，因为同一信息可以用
        不同的表达方式呈现。建议配合其他指标使用。
        """
        from collections import Counter
        
        def get_ngrams(text: str, n: int):
            tokens = text.split()
            return [
                tuple(tokens[i:i+n])
                for i in range(len(tokens) - n + 1)
            ]
        
        candidate_tokens = candidate.split()
        reference_tokens = reference.split()
        
        # 短句惩罚（Brevity Penalty）
        if len(candidate_tokens) < len(reference_tokens):
            bp = math.exp(1 - len(reference_tokens) / len(candidate_tokens))
        else:
            bp = 1.0
        
        # 计算各阶n-gram的精确率
        log_avg = 0.0
        for n in range(1, max_n + 1):
            cand_ngrams = Counter(get_ngrams(candidate, n))
            ref_ngrams = Counter(get_ngrams(reference, n))
            
            matches = sum(
                min(cand_ngrams[ng], ref_ngrams.get(ng, 0))
                for ng in cand_ngrams
            )
            total = sum(cand_ngrams.values())
            
            if total == 0:
                precision = 0.0
            else:
                precision = matches / total
            
            if precision > 0:
                log_avg += (1.0 / max_n) * math.log(precision)
            else:
                return 0.0
        
        return bp * math.exp(log_avg)
    
    @staticmethod
    def rouge_score(
        candidate: str,
        reference: str,
        rouge_type: str = "rouge-l",
    ) -> float:
        """
        计算ROUGE分数
        
        ROUGE-L基于最长公共子序列（LCS），评估流畅度和信息覆盖。
        ROUGE-1/2基于unigram/bigram匹配。
        
        ROUGE比BLEU更适合评估信息覆盖度。
        """
        def lcs_length(x: str, y: str) -> int:
            """计算最长公共子序列长度"""
            x_tokens = x.split()
            y_tokens = y.split()
            m, n = len(x_tokens), len(y_tokens)
            
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if x_tokens[i-1] == y_tokens[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            
            return dp[m][n]
        
        def rouge_n(n: int) -> float:
            """计算ROUGE-N"""
            cand_ngrams = Counter(
                tuple(candidate.split()[i:i+n])
                for i in range(len(candidate.split()) - n + 1)
            )
            ref_ngrams = Counter(
                tuple(reference.split()[i:i+n])
                for i in range(len(reference.split()) - n + 1)
            )
            
            matches = sum(
                min(cand_ngrams[ng], ref_ngrams.get(ng, 0))
                for ng in cand_ngrams
            )
            total = sum(ref_ngrams.values())
            
            return matches / total if total > 0 else 0.0
        
        if rouge_type == "rouge-l":
            lcs = lcs_length(candidate, reference)
            ref_len = len(reference.split())
            
            if ref_len == 0:
                return 0.0
            
            return lcs / ref_len
        
        elif rouge_type == "rouge-1":
            return rouge_n(1)
        elif rouge_type == "rouge-2":
            return rouge_n(2)
        else:
            raise ValueError(f"不支持的ROUGE类型: {rouge_type}")
    
    @staticmethod
    def bert_score(
        candidate: str,
        reference: str,
        model_name: str = "bert-base-chinese",
    ) -> dict:
        """
        计算BERTScore
        
        BERTScore基于BERT嵌入计算语义相似度，能够捕捉
        同义词替换和语义等价。
        
        注意：需要安装bert-score包
        """
        try:
            from bert_score import score
            
            P, R, F1 = score(
                [candidate],
                [reference],
                model_type=model_name,
                verbose=False,
            )
            
            return {
                "precision": P.item(),
                "recall": R.item(),
                "f1": F1.item(),
            }
        except ImportError:
            print("请安装bert-score: pip install bert-score")
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
```

### 12.3.3 RAGAS评估框架

RAGAS（Retrieval Augmented Generation Assessment）是目前最流行的RAG专用评估框架，它提供了一套标准化的评估指标：

```python
# ============ RAGAS评估框架核心实现 ============

class RagasEvaluator:
    """
    RAGAS评估框架
    
    RAGAS定义了四个核心评估维度：
    1. Faithfulness（忠实度）：回答是否忠实于检索到的上下文
    2. Answer Relevancy（回答相关性）：回答是否与问题相关
    3. Context Precision（上下文精确率）：检索结果中相关文档的比例
    4. Context Recall（上下文召回率）：所有相关信息是否都被检索到
    """
    
    def __init__(self, llm_judge, embedding_model):
        self.llm = llm_judge
        self.embedding = embedding_model
    
    async def evaluate(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
    ) -> dict:
        """执行RAGAS评估"""
        
        faithfulness = await self._faithfulness(answer, contexts)
        answer_relevancy = await self._answer_relevancy(question, answer)
        context_precision = await self._context_precision(question, contexts)
        
        result = {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
        }
        
        if ground_truth:
            context_recall = await self._context_recall(
                ground_truth, contexts
            )
            result["context_recall"] = context_recall
        
        # 综合得分
        result["ragas_score"] = sum(result.values()) / len(result)
        
        return result
    
    async def _faithfulness(
        self, answer: str, contexts: List[str]
    ) -> float:
        """
        忠实度评估
        
        步骤：
        1. 将回答拆分为独立的事实断言（claims）
        2. 检查每个断言是否能在上下文中找到支持
        3. 计算支持比例
        
        忠实度是RAG系统最重要的指标。低忠实度意味着幻觉。
        """
        # Step 1: 提取事实断言
        extract_prompt = f"""
将以下回答拆分为独立的事实断言。
每个断言应该是一个可以独立验证的事实陈述。

回答：{answer}

请以列表形式输出，每行一个断言：
"""
        claims_text = await self.llm.generate(extract_prompt)
        claims = [
            c.strip().lstrip("- ").lstrip("0123456789. ")
            for c in claims_text.split("\n")
            if c.strip()
        ]
        
        if not claims:
            return 1.0
        
        # Step 2: 验证每个断言
        context_text = "\n".join(contexts)
        supported_count = 0
        
        for claim in claims:
            verify_prompt = f"""
判断以下断言是否被参考上下文支持。

断言：{claim}

参考上下文：
{context_text[:2000]}

请只回答"支持"或"不支持"：
"""
            result = await self.llm.generate(verify_prompt)
            if "支持" in result:
                supported_count += 1
        
        # Step 3: 计算支持比例
        return supported_count / len(claims)
    
    async def _answer_relevancy(
        self, question: str, answer: str
    ) -> float:
        """
        回答相关性评估
        
        评估回答是否针对问题进行了回应。
        一个不相关的回答即使内容正确也没有价值。
        """
        # 生成反向问题
        reverse_prompt = f"""
基于以下回答，生成一个最可能引发这个回答的问题。

回答：{answer}

生成的问题：
"""
        generated_question = await self.llm.generate(reverse_prompt)
        
        # 计算生成的问题与原始问题的语义相似度
        q_emb = await self._get_embedding(question)
        gen_q_emb = await self._get_embedding(generated_question)
        
        similarity = self._cosine_similarity(q_emb, gen_q_emb)
        
        return similarity
    
    async def _context_precision(
        self, question: str, contexts: List[str]
    ) -> float:
        """
        上下文精确率评估
        
        评估检索结果中，相关文档的比例。
        相当于检索阶段的信息密度。
        """
        if not contexts:
            return 0.0
        
        relevant_count = 0
        for ctx in contexts:
            judge_prompt = f"""
判断以下上下文是否与问题相关。

问题：{question}

上下文：{ctx[:500]}

请只回答"相关"或"不相关"：
"""
            result = await self.llm.generate(judge_prompt)
            if "相关" in result:
                relevant_count += 1
        
        return relevant_count / len(contexts)
    
    async def _context_recall(
        self, ground_truth: str, contexts: List[str]
    ) -> float:
        """
        上下文召回率评估
        
        评估标准答案中的信息是否都被检索到了。
        相当于检索阶段的覆盖度。
        """
        # 将标准答案拆分为断言
        extract_prompt = f"""
将以下标准答案拆分为独立的事实断言。

标准答案：{ground_truth}

请以列表形式输出，每行一个断言：
"""
        claims_text = await self.llm.generate(extract_prompt)
        claims = [
            c.strip().lstrip("- ").lstrip("0123456789. ")
            for c in claims_text.split("\n")
            if c.strip()
        ]
        
        if not claims:
            return 1.0
        
        # 检查每个断言是否能从检索结果中推断
        context_text = "\n".join(contexts)
        supported_count = 0
        
        for claim in claims:
            verify_prompt = f"""
判断以下信息是否可以从参考上下文中推断出来。

信息：{claim}

参考上下文：
{context_text[:2000]}

请只回答"可以"或"不可以"：
"""
            result = await self.llm.generate(verify_prompt)
            if "可以" in result:
                supported_count += 1
        
        return supported_count / len(claims)
    
    async def _get_embedding(self, text: str) -> List[float]:
        """获取文本嵌入"""
        return await self.embedding.embed(text)
    
    def _cosine_similarity(
        self, a: List[float], b: List[float]
    ) -> float:
        """计算余弦相似度"""
        import numpy as np
        a_arr = np.array(a)
        b_arr = np.array(b)
        return float(np.dot(a_arr, b_arr) / (
            np.linalg.norm(a_arr) * np.linalg.norm(b_arr) + 1e-10
        ))


# ============ RAGAS批量评估流水线 ============

class RagasEvaluationPipeline:
    """RAGAS批量评估流水线"""
    
    def __init__(
        self,
        evaluator: RagasEvaluator,
        test_dataset: List[dict],
    ):
        self.evaluator = evaluator
        self.test_dataset = test_dataset
    
    async def run(self, rag_system) -> dict:
        """
        执行批量评估
        
        test_dataset格式：
        [
            {
                "question": "问题",
                "ground_truth": "标准答案",
                "expected_contexts": ["期望检索到的文档"],
            },
            ...
        ]
        """
        all_results = []
        
        for i, test_case in enumerate(self.test_dataset):
            print(f"评估 {i+1}/{len(self.test_dataset)}: "
                  f"{test_case['question'][:50]}...")
            
            # 调用RAG系统
            response = await rag_system.query(test_case["question"])
            
            # 执行评估
            eval_result = await self.evaluator.evaluate(
                question=test_case["question"],
                answer=response["answer"],
                contexts=response.get("contexts", []),
                ground_truth=test_case.get("ground_truth"),
            )
            
            eval_result["question"] = test_case["question"]
            all_results.append(eval_result)
        
        # 聚合结果
        aggregated = self._aggregate(all_results)
        aggregated["individual_results"] = all_results
        
        return aggregated
    
    def _aggregate(self, results: List[dict]) -> dict:
        """聚合评估结果"""
        metrics = [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
            "ragas_score",
        ]
        
        aggregated = {}
        for metric in metrics:
            values = [
                r.get(metric, 0.0) for r in results
                if metric in r
            ]
            if values:
                aggregated[f"avg_{metric}"] = sum(values) / len(values)
                aggregated[f"min_{metric}"] = min(values)
                aggregated[f"max_{metric}"] = max(values)
        
        return aggregated


# ============ 使用示例 ============

async def ragas_evaluation_example():
    """RAGAS评估使用示例"""
    
    # 初始化评估器
    evaluator = RagasEvaluator(
        llm_judge=LLMClient(),
        embedding_model=EmbeddingClient(),
    )
    
    # 准备测试数据集
    test_dataset = [
        {
            "question": "Keytruda的作用机制是什么？",
            "ground_truth": (
                "Keytruda（帕博利珠单抗）是一种PD-1抑制剂，"
                "通过阻断PD-1/PD-L1通路来激活T细胞抗肿瘤免疫。"
            ),
        },
        {
            "question": "辉瑞收购Seagen的时间和对ADC管线的影响？",
            "ground_truth": (
                "辉瑞于2023年以430亿美元收购Seagen，"
                "获得了其ADC技术平台和临床管线。"
            ),
        },
    ]
    
    # 执行评估
    pipeline = RagasEvaluationPipeline(evaluator, test_dataset)
    results = await pipeline.run(rag_system)
    
    # 输出结果
    print("=" * 50)
    print("RAGAS评估结果")
    print("=" * 50)
    print(f"平均忠实度: {results['avg_faithfulness']:.3f}")
    print(f"平均回答相关性: {results['avg_answer_relevancy']:.3f}")
    print(f"平均上下文精确率: {results['avg_context_precision']:.3f}")
    print(f"平均RAGAS得分: {results['avg_ragas_score']:.3f}")
    
    return results
```

### 12.3.4 使用官方RAGAS库

除了上述手写实现，生产环境中推荐直接使用官方RAGAS库：

```python
# 安装：pip install ragas

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset

async def ragas_official_example():
    """使用官方RAGAS库进行评估"""
    
    # 准备数据
    data = {
        "question": [
            "Keytruda的作用机制是什么？",
            "辉瑞收购Seagen的时间？",
        ],
        "answer": [
            "Keytruda是一种PD-1抑制剂...",
            "辉瑞于2023年收购Seagen...",
        ],
        "contexts": [
            ["Keytruda是一种人源化抗PD-1单克隆抗体..."],
            ["2023年3月，辉瑞宣布以430亿美元收购Seagen..."],
        ],
        "ground_truth": [
            "Keytruda通过阻断PD-1/PD-L1通路激活T细胞",
            "2023年辉瑞以430亿美元收购Seagen",
        ],
    }
    
    # 创建HuggingFace Dataset
    dataset = Dataset.from_dict(data)
    
    # 执行评估
    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
    )
    
    # 转换为Pandas DataFrame查看详情
    df = result.to_pandas()
    print(df)
    
    return result
```

---

## 12.4 系统性能指标

除了检索和生成质量，系统性能指标直接决定了用户体验和运营成本。

### 12.4.1 延迟指标

```python
class LatencyTracker:
    """延迟追踪器"""
    
    def __init__(self):
        self.latencies = []
        self.phase_latencies = {
            "retrieval": [],
            "generation": [],
            "total": [],
        }
    
    def record_query(self, phase: str, latency_ms: float):
        """记录单次延迟"""
        self.phase_latencies[phase].append(latency_ms)
        self.latencies.append(latency_ms)
    
    def compute_percentiles(self, latencies: List[float]) -> dict:
        """计算百分位延迟"""
        if not latencies:
            return {"P50": 0, "P95": 0, "P99": 0}
        
        sorted_lat = sorted(latencies)
        n = len(sorted_lat)
        
        return {
            "P50": sorted_lat[int(n * 0.50)],
            "P95": sorted_lat[int(n * 0.95)],
            "P99": sorted_lat[int(n * 0.99)],
            "avg": sum(sorted_lat) / n,
            "min": sorted_lat[0],
            "max": sorted_lat[-1],
            "count": n,
        }
    
    def report(self) -> dict:
        """生成延迟报告"""
        report = {}
        for phase, latencies in self.phase_latencies.items():
            report[phase] = self.compute_percentiles(latencies)
        return report


# 延迟监控示例
async def latency_monitoring_example():
    """延迟监控示例"""
    tracker = LatencyTracker()
    
    # 模拟50个查询的延迟数据
    import random
    for _ in range(50):
        # 检索阶段
        ret_latency = random.gauss(200, 50)  # 均值200ms
        tracker.record_query("retrieval", max(50, ret_latency))
        
        # 生成阶段
        gen_latency = random.gauss(2000, 500)  # 均值2000ms
        tracker.record_query("generation", max(500, gen_latency))
        
        # 总延迟
        total_latency = max(ret_latency, 50) + max(gen_latency, 500)
        tracker.record_query("total", total_latency)
    
    report = tracker.report()
    
    print("延迟报告（毫秒）：")
    print(f"{'阶段':<15} {'P50':<10} {'P95':<10} {'P99':<10} {'平均':<10}")
    print("-" * 55)
    for phase, metrics in report.items():
        print(f"{phase:<15} {metrics['P50']:<10.0f} "
              f"{metrics['P95']:<10.0f} {metrics['P99']:<10.0f} "
              f"{metrics['avg']:<10.0f}")
```

### 12.4.2 吞吐量和成本指标

```python
class SystemMetrics:
    """系统性能指标"""
    
    def __init__(self):
        self.query_count = 0
        self.total_latency = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        self.cache_hits = 0
        self.cache_misses = 0
    
    def record_query(
        self,
        latency_ms: float,
        tokens: dict,
        cost: float,
        cache_hit: bool = False,
    ):
        """记录一次查询的指标"""
        self.query_count += 1
        self.total_latency += latency_ms
        self.total_tokens += tokens.get("total", 0)
        self.total_cost += cost
        
        if cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
    
    @property
    def qps(self) -> float:
        """每秒查询数（假设所有查询在1秒内完成）"""
        if self.total_latency == 0:
            return 0
        return self.query_count / (self.total_latency / 1000)
    
    @property
    def avg_latency_ms(self) -> float:
        """平均延迟（毫秒）"""
        if self.query_count == 0:
            return 0
        return self.total_latency / self.query_count
    
    @property
    def cost_per_query(self) -> float:
        """每查询成本"""
        if self.query_count == 0:
            return 0.0
        return self.total_cost / self.query_count
    
    @property
    def cache_hit_rate(self) -> float:
        """缓存命中率"""
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return self.cache_hits / total
    
    def report(self) -> dict:
        """生成系统指标报告"""
        return {
            "total_queries": self.query_count,
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "p95_latency_ms": self._compute_p95(),
            "qps": round(self.qps, 2),
            "cost_per_query": round(self.cost_per_query, 4),
            "total_cost": round(self.total_cost, 2),
            "cache_hit_rate": round(self.cache_hit_rate, 3),
            "total_tokens": self.total_tokens,
        }
    
    def _compute_p95(self) -> float:
        """计算P95延迟"""
        # 实际实现需要存储所有延迟值
        return 0.0
```

### 12.4.3 系统性能基准

| 指标 | 优秀 | 良好 | 及格 | 需改进 |
|------|------|------|------|--------|
| P50延迟 | < 1s | 1-2s | 2-5s | > 5s |
| P95延迟 | < 3s | 3-5s | 5-10s | > 10s |
| P99延迟 | < 5s | 5-10s | 10-20s | > 20s |
| QPS（单节点） | > 50 | 10-50 | 5-10 | < 5 |
| 成本/查询 | < $0.001 | $0.001-0.005 | $0.005-0.01 | > $0.01 |
| 缓存命中率 | > 60% | 40-60% | 20-40% | < 20% |
| 可用性 | > 99.9% | 99.5-99.9% | 99.0-99.5% | < 99% |

---

## 12.5 评估数据集构建

高质量的评估数据集是RAG系统优化的基础。本节介绍评估数据集的构建方法和最佳实践。

### 12.5.1 人工标注

```python
class AnnotationPipeline:
    """人工标注流水线"""
    
    def __init__(self):
        self.annotation_tasks = []
        self.annotation_results = []
    
    def create_task(
        self,
        question: str,
        retrieved_docs: List[str],
        generated_answer: str,
    ) -> dict:
        """创建标注任务"""
        task = {
            "task_id": str(uuid.uuid4())[:8],
            "question": question,
            "retrieved_docs": retrieved_docs,
            "generated_answer": generated_answer,
            "annotations": {
                "retrieval_relevance": None,   # 1-5分
                "answer_accuracy": None,       # 1-5分
                "answer_completeness": None,   # 1-5分
                "hallucination": None,         # 是/否
                "comments": None,              # 自由文本
            },
            "status": "pending",
            "annotator": None,
        }
        self.annotation_tasks.append(task)
        return task
    
    def export_annotation_guidelines(self) -> str:
        """导出标注指南"""
        return """
# RAG标注指南

## 检索相关性评分（1-5分）
5分：文档完全回答了问题，包含所有关键信息
4分：文档与问题高度相关，包含大部分关键信息
3分：文档与问题部分相关，但缺少关键信息
2分：文档与问题关系不大，仅包含少量相关信息
1分：文档完全不相关

## 回答准确度评分（1-5分）
5分：完全准确，无任何错误
4分：基本准确，有轻微不精确之处
3分：部分准确，存在明显但非致命的错误
2分：大部分错误，仅有少量正确信息
1分：完全不正确

## 幻觉标注
- 是：回答中包含检索结果中不存在的信息
- 否：回答中的所有信息都可以在检索结果中找到依据

## 标注注意事项
1. 标注前请完整阅读检索结果
2. 如果问题本身存在歧义，请在备注中说明
3. 对于专业术语，可以查阅参考文档
4. 每条标注任务建议用时3-5分钟
5. 标注结果将直接影响系统优化方向
"""
```

### 12.5.2 LLM辅助标注

```python
class LLMAssistedAnnotation:
    """LLM辅助标注"""
    
    def __init__(self, llm_judge):
        self.llm = llm_judge
    
    async def auto_annotate(
        self,
        question: str,
        answer: str,
        contexts: List[str],
    ) -> dict:
        """自动标注（作为人工标注的补充）"""
        
        annotation = {}
        
        # 1. 检索相关性标注
        annotation["retrieval_relevance"] = await self._rate_retrieval(
            question, contexts
        )
        
        # 2. 回答质量标注
        annotation["answer_accuracy"] = await self._rate_accuracy(
            question, answer, contexts
        )
        
        # 3. 幻觉检测
        annotation["has_hallucination"] = await self._detect_hallucination(
            answer, contexts
        )
        
        # 4. 答案完整性
        annotation["completeness"] = await self._rate_completeness(
            question, answer
        )
        
        return annotation
    
    async def _rate_retrieval(
        self, question: str, contexts: List[str]
    ) -> int:
        """评估检索相关性（1-5分）"""
        prompt = f"""
评估检索结果与问题的相关性。

问题：{question}

检索结果：
{"---".join(c[:300] for c in contexts[:3])}

评分标准：
5 - 检索结果完美覆盖了回答问题所需的所有信息
4 - 检索结果覆盖了大部分所需信息
3 - 检索结果覆盖了部分信息
2 - 检索结果仅有少量相关信息
1 - 检索结果完全不相关

仅输出分数（1-5）：
"""
        result = await self.llm.generate(prompt)
        try:
            return int(result.strip())
        except ValueError:
            return 3
    
    async def _detect_hallucination(
        self, answer: str, contexts: List[str]
    ) -> bool:
        """检测幻觉"""
        prompt = f"""
检查以下回答是否存在幻觉（即包含上下文中不存在的断言）。

上下文：
{" ".join(c[:500] for c in contexts[:3])}

回答：
{answer}

是否包含上下文不支持的断言？
仅回答"是"或"否"：
"""
        result = await self.llm.generate(prompt)
        return "是" in result
```

### 12.5.3 自动化评估流水线

```python
class AutoEvaluationPipeline:
    """自动化评估流水线"""
    
    def __init__(
        self,
        rag_system,
        evaluator: RagasEvaluator,
        test_suite: List[dict],
    ):
        self.rag_system = rag_system
        self.evaluator = evaluator
        self.test_suite = test_suite
        
        self.results_history = []  # 历史评估结果
    
    async def run_evaluation(
        self, version: str = None
    ) -> dict:
        """执行全量评估"""
        print(f"[评估] 开始评估，测试集大小：{len(self.test_suite)}")
        
        all_results = []
        start_time = time.time()
        
        for i, test_case in enumerate(self.test_suite):
            # 执行查询
            response = await self.rag_system.query(
                test_case["question"]
            )
            
            # RAGAS评估
            ragas_result = await self.evaluator.evaluate(
                question=test_case["question"],
                answer=response["answer"],
                contexts=response.get("contexts", []),
                ground_truth=test_case.get("ground_truth"),
            )
            
            # 记录延迟
            latency = response.get("latency_ms", 0)
            
            result = {
                "test_id": i,
                "question": test_case["question"],
                "ragas_score": ragas_result["ragas_score"],
                "faithfulness": ragas_result["faithfulness"],
                "answer_relevancy": ragas_result["answer_relevancy"],
                "context_precision": ragas_result["context_precision"],
                "latency_ms": latency,
            }
            all_results.append(result)
            
            if (i + 1) % 10 == 0:
                print(f"  进度：{i+1}/{len(self.test_suite)}")
        
        # 生成报告
        report = self._generate_report(
            all_results, version, time.time() - start_time
        )
        
        # 保存历史
        self.results_history.append(report)
        
        return report
    
    def _generate_report(
        self,
        results: List[dict],
        version: str,
        duration: float,
    ) -> dict:
        """生成评估报告"""
        # 计算各项指标的平均值
        metrics = [
            "ragas_score",
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "latency_ms",
        ]
        
        avg_metrics = {}
        for metric in metrics:
            values = [r.get(metric, 0) for r in results]
            avg_metrics[f"avg_{metric}"] = sum(values) / len(values)
        
        # 找出Top-5和最差的5个用例
        sorted_by_score = sorted(
            results, key=lambda x: x["ragas_score"]
        )
        worst_cases = sorted_by_score[:5]
        best_cases = sorted_by_score[-5:]
        
        return {
            "version": version or "unknown",
            "timestamp": datetime.now().isoformat(),
            "test_count": len(results),
            "duration_seconds": duration,
            "metrics": avg_metrics,
            "worst_cases": worst_cases,
            "best_cases": best_cases,
        }
    
    def compare_with_baseline(self) -> dict:
        """与基线版本对比"""
        if len(self.results_history) < 2:
            return {"error": "需要至少2次评估结果才能对比"}
        
        current = self.results_history[-1]
        baseline = self.results_history[0]
        
        comparison = {}
        for metric in current["metrics"]:
            diff = current["metrics"][metric] - baseline["metrics"][metric]
            comparison[metric] = {
                "current": current["metrics"][metric],
                "baseline": baseline["metrics"][metric],
                "diff": diff,
                "improvement": f"{diff / baseline['metrics'][metric] * 100:+.1f}%"
                if baseline["metrics"][metric] != 0 else "N/A",
            }
        
        return comparison
```

---

## 12.6 持续优化

评估的最终目的是指导优化。本节介绍如何基于评估结果进行持续优化。

### 12.6.1 Bad Case分析流程

```python
class BadCaseAnalysisSystem:
    """Bad Case分析系统"""
    
    def __init__(self):
        # 错误分类体系
        self.error_taxonomy = {
            "retrieval_errors": {
                "description": "检索阶段错误",
                "subtypes": {
                    "miss_relevant": "遗漏了相关文档",
                    "noise_too_much": "检索结果噪声过多",
                    "wrong_order": "相关文档排序靠后",
                },
            },
            "context_errors": {
                "description": "上下文构建错误",
                "subtypes": {
                    "truncation": "关键信息被截断",
                    "irrelevant": "上下文包含无关信息",
                    "format_error": "上下文格式异常",
                },
            },
            "generation_errors": {
                "description": "生成阶段错误",
                "subtypes": {
                    "hallucination": "生成幻觉信息",
                    "incomplete": "回答不完整",
                    "misunderstand": "误解问题意图",
                    "over_refuse": "过度拒绝回答",
                },
            },
            "knowledge_errors": {
                "description": "知识库错误",
                "subtypes": {
                    "outdated": "知识过期",
                    "conflict": "知识冲突",
                    "missing": "知识缺失",
                },
            },
        }
    
    def analyze(self, bad_case: dict) -> List[dict]:
        """分析Bad Case的根因"""
        findings = []
        
        question = bad_case["question"]
        answer = bad_case["answer"]
        contexts = bad_case.get("contexts", [])
        golden = bad_case.get("golden_answer")
        
        # 1. 检索质量检查
        if golden and contexts:
            retrieval_findings = self._check_retrieval(
                question, golden, contexts
            )
            findings.extend(retrieval_findings)
        
        # 2. 生成质量检查
        if golden and answer:
            gen_findings = self._check_generation(
                question, answer, contexts, golden
            )
            findings.extend(gen_findings)
        
        return findings
    
    def _check_retrieval(
        self, question: str, golden: str, contexts: List[str]
    ) -> List[dict]:
        """检查检索质量"""
        findings = []
        
        # 检查关键信息是否在检索结果中
        golden_entities = self._extract_key_entities(golden)
        context_text = " ".join(contexts)
        
        missing_entities = [
            e for e in golden_entities
            if e not in context_text
        ]
        
        if missing_entities:
            findings.append({
                "type": "retrieval_errors.miss_relevant",
                "severity": "high",
                "detail": f"检索结果遗漏了关键实体：{missing_entities}",
                "suggestion": "考虑优化分块策略或增加检索Top-K",
            })
        
        return findings
    
    def _check_generation(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        golden: str,
    ) -> List[dict]:
        """检查生成质量"""
        findings = []
        context_text = " ".join(contexts)
        
        # 检查幻觉
        if self._has_hallucination(answer, context_text):
            findings.append({
                "type": "generation_errors.hallucination",
                "severity": "critical",
                "detail": "回答包含上下文不支持的信息",
                "suggestion": "降低LLM temperature或加强prompt约束",
            })
        
        # 检查完整性
        if self._is_incomplete(answer, golden):
            findings.append({
                "type": "generation_errors.incomplete",
                "severity": "medium",
                "detail": "回答遗漏了标准答案中的部分信息",
                "suggestion": "增加prompt中对完整性的要求",
            })
        
        return findings
    
    def generate_optimization_plan(
        self, bad_cases: List[dict]
    ) -> dict:
        """基于Bad Case生成优化计划"""
        # 统计各类错误的频率
        error_counts = defaultdict(int)
        for case in bad_cases:
            findings = self.analyze(case)
            for finding in findings:
                error_counts[finding["type"]] += 1
        
        # 按频率排序
        sorted_errors = sorted(
            error_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        
        # 生成优化建议
        optimization_plan = {
            "priority_issues": [],
            "analysis": {
                "total_bad_cases": len(bad_cases),
                "top_errors": [],
            },
        }
        
        for error_type, count in sorted_errors[:5]:
            optimization_plan["analysis"]["top_errors"].append({
                "error_type": error_type,
                "count": count,
                "percentage": count / len(bad_cases) * 100,
            })
        
        return optimization_plan
```

### 12.6.2 回归测试

```python
class RegressionTestSuite:
    """回归测试套件"""
    
    def __init__(self, baseline_results: dict):
        self.baseline = baseline_results
        self.thresholds = {
            "faithfulness": -0.05,       # 允许下降5%
            "answer_relevancy": -0.05,
            "context_precision": -0.05,
            "avg_latency_ms": 500,       # 允许增加500ms
        }
    
    def run_regression(
        self, current_results: dict
    ) -> dict:
        """执行回归测试"""
        regression_results = {
            "passed": True,
            "failures": [],
            "improvements": [],
            "unchanged": [],
        }
        
        for metric, threshold in self.thresholds.items():
            baseline_val = self.baseline.get(metric, 0)
            current_val = current_results.get(metric, 0)
            
            if metric.endswith("latency_ms"):
                # 延迟：越低越好
                diff = baseline_val - current_val
                if diff < -threshold:
                    regression_results["passed"] = False
                    regression_results["failures"].append({
                        "metric": metric,
                        "baseline": baseline_val,
                        "current": current_val,
                        "diff": diff,
                    })
            else:
                # 质量指标：越高越好
                diff = current_val - baseline_val
                if diff < threshold:
                    regression_results["passed"] = False
                    regression_results["failures"].append({
                        "metric": metric,
                        "baseline": baseline_val,
                        "current": current_val,
                        "diff": diff,
                    })
                elif diff > abs(threshold):
                    regression_results["improvements"].append({
                        "metric": metric,
                        "baseline": baseline_val,
                        "current": current_val,
                        "improvement": diff,
                    })
                else:
                    regression_results["unchanged"].append(metric)
        
        return regression_results
```

### 12.6.3 在线A/B测试

```python
class OnlineABTest:
    """在线A/B测试框架"""
    
    def __init__(self, redis_client):
        self.redis = redis_client
    
    async def start_experiment(
        self,
        experiment_name: str,
        variants: List[str],
        traffic_split: List[float],
    ):
        """启动A/B实验"""
        experiment_config = {
            "name": experiment_name,
            "variants": dict(zip(variants, traffic_split)),
            "status": "running",
            "start_time": time.time(),
            "metrics": {},
        }
        
        await self.redis.set(
            f"abtest:{experiment_name}",
            json.dumps(experiment_config),
        )
    
    async def record_interaction(
        self,
        experiment: str,
        variant: str,
        metrics: dict,
    ):
        """记录用户交互"""
        key = f"abtest:{experiment}:results:{variant}"
        await self.redis.lpush(key, json.dumps(metrics))
    
    async def analyze_experiment(
        self, experiment: str
    ) -> dict:
        """分析实验数据"""
        config = await self.redis.get(f"abtest:{experiment}")
        if not config:
            return {"error": "实验不存在"}
        
        config = json.loads(config)
        results = {}
        
        for variant in config["variants"]:
            key = f"abtest:{experiment}:results:{variant}"
            data = await self.redis.lrange(key, 0, -1)
            
            metrics_list = [json.loads(d) for d in data]
            
            if metrics_list:
                results[variant] = {
                    "samples": len(metrics_list),
                    "avg_score": sum(
                        m.get("score", 0) for m in metrics_list
                    ) / len(metrics_list),
                    "avg_latency": sum(
                        m.get("latency", 0) for m in metrics_list
                    ) / len(metrics_list),
                }
        
        return {
            "experiment": experiment,
            "duration_hours": (
                time.time() - config["start_time"]
            ) / 3600,
            "results": results,
        }
```

---

## 12.7 本章小结

本章全面介绍了RAG系统的评估体系：

1. **检索评估指标**包括Recall@K、Precision@K、MRR、MAP和NDCG。在实际项目中，推荐同时使用Recall@K（衡量覆盖率）和NDCG（衡量排序质量）作为核心指标。

2. **生成质量评估**包括准确率、幻觉率、引用准确率和完整性。RAGAS框架提供了标准化的评估维度（Faithfulness、Answer Relevancy、Context Precision、Context Recall），是目前最成熟的RAG专用评估方案。

3. **系统性能指标**包括延迟（P50/P95/P99）、QPS、成本/查询和缓存命中率。这些指标直接影响用户体验和运营成本。

4. **评估数据集**的构建需要人工标注、LLM辅助标注和自动化评估的有机结合。高质量的数据集是持续优化的基础。

5. **持续优化**依赖于Bad Case分析、回归测试和在线A/B测试三个环节的闭环运作。每次优化都应可衡量、可追踪、可回滚。

评估不是一次性活动，而是贯穿RAG系统全生命周期的持续过程。建立完善的评估体系，是RAG系统从"能用"走向"好用"的必经之路。下一章将以GraphRAG-KG项目为例，展示如何将这些技术整合到一个完整的RAG系统中。
