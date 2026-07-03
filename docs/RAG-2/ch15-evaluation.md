# 第15章 评估体系

## 15.1 引言

评估是RAG系统开发和运维中的关键环节。没有科学的评估体系，就无法准确衡量系统的性能、发现系统的不足、指导优化的方向。RAG系统的评估比传统检索系统或单纯的LLM应用更为复杂，因为它涉及检索质量和生成质量两个维度的交叉评估。

RAG系统评估的挑战性体现在以下几个方面：

1. **多维度**：需要同时评估检索质量、生成质量和系统性能
2. **主观性**：生成质量的好坏往往具有主观性，不同评估者可能有不同标准
3. **任务依赖性**：不同的应用场景对质量的要求不同
4. **成本约束**：全面评估需要大量的人力或API调用成本
5. **持续变化**：LLM的更新、知识库的变化都可能导致评估结果的变化

本章将全面介绍RAG系统的评估体系，包括评估维度、自动化指标、评估框架、人工评估方法和持续优化流程。

### 15.1.1 评估体系总览

RAG系统评估体系可以分为三个主要层次：

```
第一层：检索质量评估
├── 检索精确性：Precision@K, Recall@K
├── 检索排序性：MRR, MAP, NDCG
└── 检索覆盖度：覆盖率、多样性

第二层：生成质量评估
├── 准确性：事实正确性、幻觉率
├── 相关性：答案相关性、引用准确性
├── 完整性：覆盖率、信息充分性
└── 语言质量：流畅度、一致性、可读性

第三层：系统性能评估
├── 延迟：P50, P95, P99
├── 吞吐量：QPS
├── 可用性：错误率、SLA达成率
└── 成本：每查询成本、Token消耗
```

## 15.2 检索质量评估

检索质量是RAG系统的基石。如果检索不到相关文档，生成阶段就不可能产生准确的答案。

### 15.2.1 核心指标

#### 15.2.1.1 Recall@K

Recall@K衡量在前K个检索结果中，相关文档的比例。它关注的是检索系统是否遗漏了重要信息。

```python
import numpy as np
from typing import List, Dict, Any
from collections import defaultdict

class RetrievalMetrics:
    """检索质量指标计算"""
    
    @staticmethod
    def recall_at_k(relevant_docs: List[str], 
                    retrieved_docs: List[str], 
                    k: int) -> float:
        """计算Recall@K"""
        if not relevant_docs:
            return 0.0
        
        retrieved_k = retrieved_docs[:k]
        relevant_retrieved = sum(
            1 for doc in relevant_docs if doc in retrieved_k
        )
        
        return relevant_retrieved / len(relevant_docs)
    
    @staticmethod
    def precision_at_k(relevant_docs: List[str],
                       retrieved_docs: List[str],
                       k: int) -> float:
        """计算Precision@K"""
        if k == 0:
            return 0.0
        
        retrieved_k = retrieved_docs[:k]
        relevant_retrieved = sum(
            1 for doc in retrieved_k if doc in relevant_docs
        )
        
        return relevant_retrieved / k
    
    @staticmethod
    def mean_reciprocal_rank(relevant_docs: List[str],
                              retrieved_docs: List[str]) -> float:
        """计算MRR"""
        for i, doc in enumerate(retrieved_docs, 1):
            if doc in relevant_docs:
                return 1.0 / i
        return 0.0
    
    @staticmethod
    def average_precision(relevant_docs: List[str],
                          retrieved_docs: List[str]) -> float:
        """计算AP"""
        hits = 0
        sum_precisions = 0.0
        
        for i, doc in enumerate(retrieved_docs, 1):
            if doc in relevant_docs:
                hits += 1
                sum_precisions += hits / i
        
        if hits == 0:
            return 0.0
        
        return sum_precisions / len(relevant_docs)
    
    @staticmethod
    def ndcg_at_k(relevant_docs: List[str],
                  retrieved_docs: List[str],
                  relevance_scores: Dict[str, float],
                  k: int) -> float:
        """计算NDCG@K"""
        retrieved_k = retrieved_docs[:k]
        
        # 计算DCG
        dcg = 0.0
        for i, doc in enumerate(retrieved_k, 1):
            rel = relevance_scores.get(doc, 0.0)
            dcg += (2 ** rel - 1) / np.log2(i + 1)
        
        # 计算IDCG
        ideal_rels = sorted(
            [relevance_scores.get(doc, 0.0) for doc in relevant_docs],
            reverse=True
        )[:k]
        
        idcg = 0.0
        for i, rel in enumerate(ideal_rels, 1):
            idcg += (2 ** rel - 1) / np.log2(i + 1)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    @staticmethod
    def compute_all_metrics(relevant_docs: List[str],
                            retrieved_docs: List[str],
                            relevance_scores: Dict[str, float] = None,
                            ks: List[int] = None) -> Dict[str, float]:
        """计算所有检索指标"""
        if ks is None:
            ks = [1, 3, 5, 10]
        
        if relevance_scores is None:
            relevance_scores = {doc: 1.0 for doc in relevant_docs}
        
        metrics = {}
        
        for k in ks:
            metrics[f'Recall@{k}'] = RetrievalMetrics.recall_at_k(
                relevant_docs, retrieved_docs, k
            )
            metrics[f'Precision@{k}'] = RetrievalMetrics.precision_at_k(
                relevant_docs, retrieved_docs, k
            )
            metrics[f'NDCG@{k}'] = RetrievalMetrics.ndcg_at_k(
                relevant_docs, retrieved_docs, relevance_scores, k
            )
        
        metrics['MRR'] = RetrievalMetrics.mean_reciprocal_rank(
            relevant_docs, retrieved_docs
        )
        metrics['MAP'] = RetrievalMetrics.average_precision(
            relevant_docs, retrieved_docs
        )
        
        return metrics


class RetrievalEvaluator:
    """检索评估器"""
    
    def __init__(self):
        self.metrics = RetrievalMetrics()
        self.results = []
    
    def evaluate_query(self, query: str,
                       retrieved_docs: List[str],
                       relevant_docs: List[str],
                       relevance_scores: Dict[str, float] = None) -> Dict:
        """评估单次查询"""
        metrics = self.metrics.compute_all_metrics(
            relevant_docs, retrieved_docs, relevance_scores
        )
        
        result = {
            'query': query,
            'num_relevant': len(relevant_docs),
            'num_retrieved': len(retrieved_docs),
            'metrics': metrics
        }
        
        self.results.append(result)
        return result
    
    def evaluate_batch(self, queries: List[Dict]) -> Dict:
        """批量评估"""
        for q in queries:
            self.evaluate_query(
                q['query'],
                q['retrieved_docs'],
                q['relevant_docs'],
                q.get('relevance_scores')
            )
        
        return self.aggregate_results()
    
    def aggregate_results(self) -> Dict:
        """聚合所有评估结果"""
        if not self.results:
            return {}
        
        # 计算各指标的平均值
        all_metrics = defaultdict(list)
        
        for result in self.results:
            for metric_name, value in result['metrics'].items():
                all_metrics[metric_name].append(value)
        
        aggregated = {
            'avg': {},
            'std': {},
            'min': {},
            'max': {},
            'num_queries': len(self.results)
        }
        
        for metric_name, values in all_metrics.items():
            aggregated['avg'][metric_name] = np.mean(values)
            aggregated['std'][metric_name] = np.std(values)
            aggregated['min'][metric_name] = np.min(values)
            aggregated['max'][metric_name] = np.max(values)
        
        return aggregated
    
    def get_bad_cases(self, threshold: float = 0.5) -> List[Dict]:
        """获取检索质量差的案例"""
        bad_cases = []
        
        for result in self.results:
            recall = result['metrics'].get('Recall@5', 0)
            if recall < threshold:
                bad_cases.append(result)
        
        return sorted(bad_cases, key=lambda x: x['metrics'].get('Recall@5', 0))
```

#### 15.2.1.2 检索多样性评估

除了精确性和排序质量，检索结果的多样性也是重要的评估维度：

```python
class DiversityMetrics:
    """多样性评估指标"""
    
    @staticmethod
    def document_coverage(retrieved_docs: List[str],
                          source_categories: Dict[str, str]) -> float:
        """文档来源覆盖率"""
        if not retrieved_docs:
            return 0.0
        
        covered_categories = set()
        for doc in retrieved_docs:
            category = source_categories.get(doc, 'unknown')
            covered_categories.add(category)
        
        return len(covered_categories) / len(set(source_categories.values()))
    
    @staticmethod
    def semantic_diversity(embeddings: np.ndarray) -> float:
        """语义多样性（嵌入向量平均距离）"""
        if len(embeddings) < 2:
            return 0.0
        
        from sklearn.metrics.pairwise import cosine_distances
        
        distances = cosine_distances(embeddings)
        # 排除对角线
        mask = ~np.eye(distances.shape[0], dtype=bool)
        avg_distance = distances[mask].mean()
        
        return avg_distance
    
    @staticmethod
    def information_novelty(retrieved_docs: List[str],
                            previous_docs: List[str]) -> float:
        """信息新颖度（与历史检索结果的不同程度）"""
        if not previous_docs:
            return 1.0
        
        new_docs = set(retrieved_docs) - set(previous_docs)
        return len(new_docs) / len(retrieved_docs) if retrieved_docs else 0.0
```

### 15.2.2 检索评估数据集

构建高质量的检索评估数据集是准确评估的基础：

```python
class RetrievalEvalDataset:
    """检索评估数据集"""
    
    def __init__(self):
        self.queries = []
        self.annotations = []
    
    def add_query(self, query: str, 
                  relevant_docs: List[str],
                  relevance_scores: Dict[str, float] = None):
        """添加带标注的查询"""
        entry = {
            'query': query,
            'relevant_docs': relevant_docs,
            'relevance_scores': relevance_scores or {
                doc: 1.0 for doc in relevant_docs
            }
        }
        self.queries.append(entry)
    
    def load_from_file(self, filepath: str):
        """从文件加载数据集"""
        import json
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for entry in data:
            self.add_query(
                entry['query'],
                entry['relevant_docs'],
                entry.get('relevance_scores')
            )
    
    def save_to_file(self, filepath: str):
        """保存数据集"""
        import json
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.queries, f, ensure_ascii=False, indent=2)
    
    def split(self, train_ratio: float = 0.8) -> tuple:
        """分割训练集和测试集"""
        import random
        
        shuffled = self.queries.copy()
        random.shuffle(shuffled)
        
        split_idx = int(len(shuffled) * train_ratio)
        
        train_dataset = RetrievalEvalDataset()
        train_dataset.queries = shuffled[:split_idx]
        
        test_dataset = RetrievalEvalDataset()
        test_dataset.queries = shuffled[split_idx:]
        
        return train_dataset, test_dataset
    
    @property
    def stats(self) -> Dict:
        """数据集统计"""
        if not self.queries:
            return {'num_queries': 0}
        
        avg_relevant = np.mean([
            len(q['relevant_docs']) for q in self.queries
        ])
        
        return {
            'num_queries': len(self.queries),
            'avg_relevant_docs': avg_relevant,
            'min_relevant': min(len(q['relevant_docs']) for q in self.queries),
            'max_relevant': max(len(q['relevant_docs']) for q in self.queries)
        }
```

## 15.3 生成质量评估

生成质量评估是RAG评估中最具挑战性的部分，因为它涉及自然语言理解和主观判断。

### 15.3.1 自动化评估指标

#### 15.3.1.1 基于参考的指标

```python
from rouge_score import rouge_scorer
from bert_score import BERTScorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import evaluate

class ReferenceBasedMetrics:
    """基于参考答案的评估指标"""
    
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=True
        )
        
        # 初始化BERTScore
        try:
            self.bert_scorer = BERTScorer(
                lang="zh",
                rescale_with_baseline=True
            )
        except:
            print("BERTScore初始化失败，使用备用模式")
            self.bert_scorer = None
    
    def compute_bleu(self, candidate: str, references: List[str]) -> Dict:
        """计算BLEU分数"""
        smoothing = SmoothingFunction().method4
        
        # 分词
        candidate_tokens = list(candidate)
        reference_tokens = [list(ref) for ref in references]
        
        bleu_scores = {}
        for n in [1, 2, 3, 4]:
            weights = tuple(1.0/n if i < n else 0.0 for i in range(4))
            score = sentence_bleu(
                reference_tokens, candidate_tokens,
                weights=weights,
                smoothing_function=smoothing
            )
            bleu_scores[f'BLEU-{n}'] = score
        
        return bleu_scores
    
    def compute_rouge(self, candidate: str, reference: str) -> Dict:
        """计算ROUGE分数"""
        scores = self.rouge_scorer.score(reference, candidate)
        
        return {
            'ROUGE-1': scores['rouge1'].fmeasure,
            'ROUGE-2': scores['rouge2'].fmeasure,
            'ROUGE-L': scores['rougeL'].fmeasure
        }
    
    def compute_bertscore(self, candidate: str, reference: str) -> Dict:
        """计算BERTScore"""
        if self.bert_scorer is None:
            return {'BERTScore': 0.0}
        
        P, R, F1 = self.bert_scorer.score([candidate], [reference])
        
        return {
            'BERTScore_Precision': P.item(),
            'BERTScore_Recall': R.item(),
            'BERTScore_F1': F1.item()
        }
    
    def compute_all(self, candidate: str, 
                    references: List[str]) -> Dict:
        """计算所有参考指标"""
        metrics = {}
        
        # BLEU
        bleu = self.compute_bleu(candidate, references)
        metrics.update(bleu)
        
        # ROUGE（使用第一个参考）
        if references:
            rouge = self.compute_rouge(candidate, references[0])
            metrics.update(rouge)
        
        # BERTScore
        if references:
            bert = self.compute_bertscore(candidate, references[0])
            metrics.update(bert)
        
        return metrics
```

#### 15.3.1.2 无参考指标

无参考指标不需要标准答案，而是基于检索上下文和查询来评估生成质量：

```python
class ReferenceFreeMetrics:
    """无参考评估指标"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def compute_faithfulness(self, answer: str, 
                              context: List[str]) -> float:
        """忠实度：评估答案是否忠实于检索上下文"""
        context_text = "\n".join(context)
        
        prompt = f"""评估以下回答是否忠实于提供的上下文。

上下文：
{context_text[:2000]}

回答：
{answer[:1000]}

请逐句检查回答是否在上下文中有依据。
输出JSON格式：
{{
    "faithfulness_score": 0.0-1.0,
    "supported_claims": ["有依据的陈述"],
    "unsupported_claims": ["无依据的陈述"],
    "hallucination_rate": 0.0-1.0
}}"""
        
        try:
            import json
            response = self.llm(prompt)
            result = json.loads(response)
            return result.get('faithfulness_score', 0.5)
        except:
            return 0.5
    
    def compute_answer_relevancy(self, answer: str, 
                                  query: str) -> float:
        """答案相关性：评估答案与查询的相关程度"""
        prompt = f"""评估以下回答与问题的相关程度。

问题：{query}

回答：
{answer[:1000]}

评分标准：
- 1.0: 完全相关，直接回答问题
- 0.8: 高度相关，回答了主要问题
- 0.6: 部分相关，只回答了部分内容
- 0.4: 低度相关，包含大量无关信息
- 0.2: 几乎不相关
- 0.0: 完全不相关

只输出分数（0.0-1.0）："""
        
        try:
            response = self.llm(prompt)
            score = float(response.strip())
            return max(0.0, min(1.0, score))
        except:
            return 0.5
    
    def compute_context_precision(self, query: str,
                                   contexts: List[str]) -> float:
        """上下文精确度：评估检索上下文中有多少信息是相关的"""
        if not contexts:
            return 0.0
        
        prompt = f"""评估以下检索结果中，每条对回答问题的重要性。

问题：{query}

检索结果：
{chr(10).join([f"[{i+1}] {c[:300]}" for i, c in enumerate(contexts)])}

请判断每条结果是否相关，输出JSON：
{{
    "relevant_indices": [相关索引],
    "precision_score": 0.0-1.0,
    "irrelevant_reasons": {{
        "索引": "不相关原因"
    }}
}}"""
        
        try:
            import json
            response = self.llm(prompt)
            result = json.loads(response)
            return result.get('precision_score', 0.5)
        except:
            return 0.5
    
    def compute_context_recall(self, query: str,
                                answer: str,
                                contexts: List[str]) -> float:
        """上下文召回率：评估答案所需信息是否在上下文中"""
        prompt = f"""检查回答所需的信息是否都在提供的上下文中。

问题：{query}
回答：{answer[:1000]}

上下文：
{chr(10).join([f"[{i+1}] {c[:300]}" for i, c in enumerate(contexts)])}

请判断回答中的关键信息是否都能在上下文中找到。
输出JSON：
{{
    "recall_score": 0.0-1.0,
    "missing_info": ["缺失的信息"],
    "found_info": ["找到的信息"]
}}"""
        
        try:
            import json
            response = self.llm(prompt)
            result = json.loads(response)
            return result.get('recall_score', 0.5)
        except:
            return 0.5
    
    def compute_all(self, query: str, answer: str,
                    contexts: List[str]) -> Dict:
        """计算所有无参考指标"""
        return {
            'faithfulness': self.compute_faithfulness(answer, contexts),
            'answer_relevancy': self.compute_answer_relevancy(answer, query),
            'context_precision': self.compute_context_precision(query, contexts),
            'context_recall': self.compute_context_recall(query, answer, contexts)
        }
```

### 15.3.2 幻觉检测

幻觉检测是评估生成质量的重要环节，特别是对于RAG系统：

```python
class HallucinationDetector:
    """幻觉检测器"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def detect_hallucinations(self, answer: str, 
                               context: List[str]) -> Dict:
        """检测答案中的幻觉"""
        # 将答案分解为声明（claim）
        claims = self._extract_claims(answer)
        
        # 逐句验证
        results = []
        hallucinated_claims = []
        supported_claims = []
        
        for claim in claims:
            verification = self._verify_claim(claim, context)
            results.append(verification)
            
            if verification['is_supported']:
                supported_claims.append(claim)
            else:
                hallucinated_claims.append({
                    'claim': claim,
                    'reason': verification.get('reason', '')
                })
        
        hallucination_rate = len(hallucinated_claims) / len(claims) if claims else 0
        
        return {
            'hallucination_rate': hallucination_rate,
            'total_claims': len(claims),
            'supported_claims': supported_claims,
            'hallucinated_claims': hallucinated_claims,
            'results': results
        }
    
    def _extract_claims(self, text: str) -> List[str]:
        """将文本分解为独立声明"""
        prompt = f"""将以下文本分解为独立的声明（事实性陈述）。

文本：{text[:1500]}

要求：
1. 每个声明应该是独立的事实陈述
2. 按句子或子句拆分
3. 排除观点性、建议性的内容

请以JSON列表格式输出："""
        
        try:
            import json
            response = self.llm(prompt)
            claims = json.loads(response)
            return [c if isinstance(c, str) else c.get('claim', str(c)) 
                    for c in claims]
        except:
            # 简单按句子拆分
            import re
            sentences = re.split(r'[。！？\n]', text)
            return [s.strip() for s in sentences if len(s.strip()) > 5]
    
    def _verify_claim(self, claim: str, context: List[str]) -> Dict:
        """验证单个声明"""
        context_text = "\n".join(context)[:2000]
        
        prompt = f"""验证以下声明是否在上下文中有所依据。

声明：{claim}

上下文：
{context_text}

输出JSON：
{{
    "is_supported": true/false,
    "confidence": 0.0-1.0,
    "evidence": "支持证据（如果有）",
    "reason": "不支持的原因（如果不支持）"
}}"""
        
        try:
            import json
            response = self.llm(prompt)
            return json.loads(response)
        except:
            return {
                "is_supported": False,
                "confidence": 0.0,
                "reason": "验证失败"
            }
    
    def get_hallucination_types(self, answer: str,
                                 context: List[str]) -> Dict:
        """幻觉类型分类"""
        result = self.detect_hallucinations(answer, context)
        
        # 分类幻觉类型
        type_counts = defaultdict(int)
        
        for hc in result.get('hallucinated_claims', []):
            claim = hc['claim']
            reason = hc.get('reason', '')
            
            hallucination_type = self._classify_hallucination(claim, reason)
            type_counts[hallucination_type] += 1
        
        return {
            'hallucination_types': dict(type_counts),
            'total_hallucinations': len(result.get('hallucinated_claims', []))
        }
    
    def _classify_hallucination(self, claim: str, reason: str) -> str:
        """分类幻觉类型"""
        # 简单规则分类
        if '矛盾' in reason or '不一致' in reason:
            return 'contradiction'
        elif '不存在' in reason or '未提及' in reason:
            return 'fabrication'
        elif '无关' in reason:
            return 'irrelevant'
        elif '模糊' in reason:
            return 'ambiguity'
        else:
            return 'unknown'
```

### 15.3.3 引用准确性评估

RAG系统通常会在答案中引用来源，引用准确性直接影响可信度：

```python
class CitationEvaluator:
    """引用评估器"""
    
    def evaluate_citation_accuracy(self, answer: str,
                                    documents: List[Dict]) -> Dict:
        """评估引用准确性"""
        # 提取答案中的引用标记
        citations = self._extract_citations(answer)
        
        if not citations:
            return {
                'citation_rate': 0.0,
                'citation_accuracy': 1.0,  # 没有引用算1.0
                'has_citations': False
            }
        
        # 验证每个引用
        results = []
        correct = 0
        
        for citation in citations:
            verification = self._verify_citation(citation, documents)
            results.append(verification)
            
            if verification['is_correct']:
                correct += 1
        
        return {
            'citation_rate': len(citations) / len(answer.split()),
            'citation_accuracy': correct / len(citations) if citations else 1.0,
            'citation_count': len(citations),
            'correct_citations': correct,
            'incorrect_citations': len(citations) - correct,
            'citation_details': results,
            'has_citations': True
        }
    
    def _extract_citations(self, text: str) -> List[Dict]:
        """提取引用标记"""
        import re
        
        citations = []
        
        # 匹配 [数字] 格式
        pattern = r'\[(\d+)\]'
        for match in re.finditer(pattern, text):
            citations.append({
                'index': int(match.group(1)),
                'position': match.start(),
                'surrounding_text': text[
                    max(0, match.start()-50):match.end()+50
                ]
            })
        
        # 匹配 [来源: xxx] 格式
        source_pattern = r'\[来源[：:]\s*([^\]]+)\]'
        for match in re.finditer(source_pattern, text):
            citations.append({
                'source': match.group(1),
                'position': match.start(),
                'surrounding_text': text[
                    max(0, match.start()-50):match.end()+50
                ]
            })
        
        return citations
    
    def _verify_citation(self, citation: Dict,
                          documents: List[Dict]) -> Dict:
        """验证单个引用"""
        if 'index' in citation:
            idx = citation['index']
            if idx <= len(documents):
                doc = documents[idx - 1]
                surrounding = citation['surrounding_text']
                
                # 检查引用内容是否与文档一致
                consistency = self._check_citation_consistency(
                    surrounding, doc.get('content', '')
                )
                
                return {
                    'is_correct': consistency,
                    'citation_index': idx,
                    'document_source': doc.get('source', ''),
                    'details': '引用与文档内容一致' if consistency else '引用内容与文档不符'
                }
            else:
                return {
                    'is_correct': False,
                    'citation_index': idx,
                    'details': f'引用索引 {idx} 超出文档范围'
                }
        
        if 'source' in citation:
            # 验证来源名称
            source_name = citation['source']
            matching_docs = [
                d for d in documents 
                if source_name in d.get('source', '')
            ]
            
            return {
                'is_correct': len(matching_docs) > 0,
                'source': source_name,
                'details': '来源匹配' if matching_docs else '来源未找到'
            }
        
        return {'is_correct': False, 'details': '无法解析的引用格式'}
    
    def _check_citation_consistency(self, text: str, 
                                     doc_content: str) -> bool:
        """检查引用内容与文档的一致性"""
        # 使用LLM评估
        prompt = f"""判断引用文本是否与文档内容一致。

引用周围文本：{text[:200]}

文档内容：{doc_content[:500]}

如果引用内容在文档中有依据，回答"是"，否则回答"否"："""
        
        try:
            response = self.llm(prompt)
            return '是' in response or 'Yes' in response
        except:
            return False
```

## 15.4 RAGAS框架

RAGAS（Retrieval Augmented Generation Assessment）是一个专门用于评估RAG系统的开源框架。它提供了一套全面的自动化评估指标。

### 15.4.1 RAGAS核心指标

```python
class RAGASEvaluator:
    """RAGAS评估器实现"""
    
    def __init__(self, llm, embeddings):
        self.llm = llm
        self.embeddings = embeddings
    
    def evaluate(self, dataset: List[Dict]) -> Dict:
        """执行RAGAS评估"""
        results = {}
        
        # 计算各指标
        results['faithfulness'] = self.compute_faithfulness(dataset)
        results['answer_relevancy'] = self.compute_answer_relevancy(dataset)
        results['context_precision'] = self.compute_context_precision(dataset)
        results['context_recall'] = self.compute_context_recall(dataset)
        results['answer_correctness'] = self.compute_answer_correctness(dataset)
        
        # 综合评分
        results['ragas_score'] = np.mean([
            v for v in results.values() if isinstance(v, float)
        ])
        
        return results
    
    def compute_faithfulness(self, dataset: List[Dict]) -> float:
        """计算忠实度"""
        scores = []
        
        for item in dataset:
            answer = item.get('answer', '')
            contexts = item.get('contexts', [])
            
            if not answer or not contexts:
                continue
            
            # 分解声明并验证
            score = self._score_faithfulness(answer, contexts)
            scores.append(score)
        
        return np.mean(scores) if scores else 0.0
    
    def compute_answer_relevancy(self, dataset: List[Dict]) -> float:
        """计算答案相关性"""
        scores = []
        
        for item in dataset:
            question = item.get('question', '')
            answer = item.get('answer', '')
            
            if not question or not answer:
                continue
            
            score = self._score_answer_relevancy(question, answer)
            scores.append(score)
        
        return np.mean(scores) if scores else 0.0
    
    def compute_context_precision(self, dataset: List[Dict]) -> float:
        """计算上下文精确度"""
        scores = []
        
        for item in dataset:
            question = item.get('question', '')
            contexts = item.get('contexts', [])
            
            if not question or not contexts:
                continue
            
            score = self._score_context_precision(question, contexts)
            scores.append(score)
        
        return np.mean(scores) if scores else 0.0
    
    def compute_context_recall(self, dataset: List[Dict]) -> float:
        """计算上下文召回率"""
        scores = []
        
        for item in dataset:
            question = item.get('question', '')
            contexts = item.get('contexts', [])
            answer = item.get('answer', '')
            ground_truth = item.get('ground_truth', '')
            
            if not question or not contexts:
                continue
            
            # 需要标准答案或答案来评估
            reference = ground_truth or answer
            if not reference:
                continue
            
            score = self._score_context_recall(question, contexts, reference)
            scores.append(score)
        
        return np.mean(scores) if scores else 0.0
    
    def compute_answer_correctness(self, dataset: List[Dict]) -> float:
        """计算答案正确性"""
        scores = []
        
        for item in dataset:
            if 'ground_truth' not in item or 'answer' not in item:
                continue
            
            score = self._score_answer_correctness(
                item['answer'], item['ground_truth']
            )
            scores.append(score)
        
        return np.mean(scores) if scores else 0.0
    
    def _score_faithfulness(self, answer: str, 
                             contexts: List[str]) -> float:
        """评分：忠实度"""
        prompt = f"""评估以下回答是否忠实于提供的上下文。

回答：{answer[:500]}

上下文：
{chr(10).join([c[:300] for c in contexts])}

请逐句判断，输出JSON：
{{
    "total_claims": 整数,
    "supported_claims": 整数,
    "faithfulness_score": 0.0-1.0
}}"""
        
        try:
            import json
            response = self.llm(prompt)
            result = json.loads(response)
            return result.get('faithfulness_score', 0.0)
        except:
            return 0.0
    
    def _score_answer_relevancy(self, question: str, 
                                 answer: str) -> float:
        """评分：答案相关性"""
        # 使用嵌入相似度作为相关性代理
        q_emb = self.embeddings.embed_query(question)
        a_emb = self.embeddings.embed_query(answer[:500])
        
        similarity = np.dot(q_emb, a_emb) / (
            np.linalg.norm(q_emb) * np.linalg.norm(a_emb)
        )
        
        return float(similarity)
    
    def _score_context_precision(self, question: str,
                                  contexts: List[str]) -> float:
        """评分：上下文精确度"""
        prompt = f"""评估以下上下文中有多少信息是与问题相关的。

问题：{question}

上下文列表：
{chr(10).join([f"[{i+1}] {c[:200]}" for i, c in enumerate(contexts)])}

请判断每条上下文的相关性，输出JSON：
{{
    "relevant_ratio": 0.0-1.0,
    "relevant_indices": [索引列表]
}}"""
        
        try:
            import json
            response = self.llm(prompt)
            result = json.loads(response)
            return result.get('relevant_ratio', 0.0)
        except:
            return 0.0
    
    def _score_context_recall(self, question: str,
                               contexts: List[str],
                               reference: str) -> float:
        """评分：上下文召回率"""
        prompt = f"""检查回答所需的信息是否都在提供的上下文中。

问题：{question}
标准答案：{reference[:500]}

上下文：
{chr(10).join([f"[{i+1}] {c[:300]}" for i, c in enumerate(contexts)])}

输出JSON：
{{
    "recall_score": 0.0-1.0,
    "covered_statements": ["覆盖的陈述"],
    "missing_statements": ["缺失的陈述"]
}}"""
        
        try:
            import json
            response = self.llm(prompt)
            result = json.loads(response)
            return result.get('recall_score', 0.0)
        except:
            return 0.0
    
    def _score_answer_correctness(self, answer: str,
                                   ground_truth: str) -> float:
        """评分：答案正确性"""
        # 使用BERTScore
        from bert_score import BERTScorer
        
        scorer = BERTScorer(lang="zh", rescale_with_baseline=True)
        P, R, F1 = scorer.score([answer], [ground_truth])
        
        return F1.item()
```

### 14.4.2 RAGAS评估流水线

```python
class RAGASPipeline:
    """RAGAS评估流水线"""
    
    def __init__(self, evaluator: RAGASEvaluator):
        self.evaluator = evaluator
    
    def run_evaluation(self, 
                       queries: List[str],
                       answers: List[str],
                       contexts: List[List[str]],
                       ground_truths: List[str] = None) -> Dict:
        """运行完整评估"""
        # 构建数据集
        dataset = []
        for i in range(len(queries)):
            item = {
                'question': queries[i],
                'answer': answers[i],
                'contexts': contexts[i]
            }
            if ground_truths:
                item['ground_truth'] = ground_truths[i]
            dataset.append(item)
        
        # 执行评估
        results = self.evaluator.evaluate(dataset)
        
        # 添加置信区间
        results['confidence_intervals'] = self._compute_confidence_intervals(
            dataset
        )
        
        return results
    
    def _compute_confidence_intervals(self, 
                                       dataset: List[Dict]) -> Dict:
        """计算置信区间"""
        from scipy import stats
        
        n = len(dataset)
        if n < 2:
            return {}
        
        # 对每个指标计算95%置信区间
        intervals = {}
        for metric in ['faithfulness', 'answer_relevancy', 
                       'context_precision', 'context_recall']:
            scores = [
                self.evaluator._score_faithfulness(
                    item.get('answer', ''), item.get('contexts', [])
                ) if metric == 'faithfulness' else 0
                for item in dataset
            ]
            
            if len(scores) > 1:
                mean = np.mean(scores)
                se = stats.sem(scores)
                ci = se * stats.t.ppf((1 + 0.95) / 2, n - 1)
                
                intervals[metric] = {
                    'mean': mean,
                    'ci_lower': mean - ci,
                    'ci_upper': mean + ci
                }
        
        return intervals
    
    def compare_models(self, model_a_results: Dict,
                        model_b_results: Dict) -> Dict:
        """比较两个模型的评估结果"""
        comparison = {}
        
        for metric in ['ragas_score', 'faithfulness', 'answer_relevancy',
                       'context_precision', 'context_recall']:
            if metric in model_a_results and metric in model_b_results:
                diff = model_a_results[metric] - model_b_results[metric]
                comparison[metric] = {
                    'model_a': model_a_results[metric],
                    'model_b': model_b_results[metric],
                    'difference': diff,
                    'improvement_pct': (
                        diff / model_b_results[metric] * 100 
                        if model_b_results[metric] != 0 else 0
                    )
                }
        
        return comparison
```

## 15.5 人工评估

尽管自动化评估指标可以快速评估系统性能，但人工评估仍然是评估生成质量的金标准。

### 15.5.1 评估维度设计

```python
class HumanEvaluationCriteria:
    """人工评估标准"""
    
    # 评估维度定义
    DIMENSIONS = {
        'accuracy': {
            'name': '准确性',
            'description': '回答中的事实信息是否正确',
            'scale': [
                (1, '存在严重事实错误'),
                (2, '存在明显错误'),
                (3, '基本正确，有小瑕疵'),
                (4, '大部分正确'),
                (5, '完全正确')
            ]
        },
        'relevance': {
            'name': '相关性',
            'description': '回答是否直接回应了用户问题',
            'scale': [
                (1, '完全不相关'),
                (2, '部分相关但偏离主题'),
                (3, '基本相关'),
                (4, '高度相关'),
                (5, '精确命中问题核心')
            ]
        },
        'completeness': {
            'name': '完整性',
            'description': '回答是否覆盖了问题的所有方面',
            'scale': [
                (1, '完全未覆盖'),
                (2, '覆盖很少'),
                (3, '覆盖主要方面'),
                (4, '覆盖大部分'),
                (5, '全面覆盖')
            ]
        },
        'clarity': {
            'name': '清晰度',
            'description': '回答是否易于理解、结构清晰',
            'scale': [
                (1, '难以理解'),
                (2, '表达混乱'),
                (3, '基本清晰'),
                (4, '表达清楚'),
                (5, '非常清晰流畅')
            ]
        },
        'conciseness': {
            'name': '简洁性',
            'description': '回答是否简洁、无冗余信息',
            'scale': [
                (1, '过度冗长'),
                (2, '比较冗长'),
                (3, '适中'),
                (4, '比较简洁'),
                (5, '非常精炼')
            ]
        },
        'citation_quality': {
            'name': '引用质量',
            'description': '引用是否准确、位置是否合理',
            'scale': [
                (1, '无引用或引用错误'),
                (2, '引用不准确'),
                (3, '引用基本正确'),
                (4, '引用准确'),
                (5, '引用精准且位置合理')
            ]
        }
    }

class HumanEvaluationPlatform:
    """人工评估平台"""
    
    def __init__(self):
        self.criteria = HumanEvaluationCriteria()
        self.evaluations = []
    
    def create_evaluation_task(self, 
                                query: str,
                                answer: str,
                                contexts: List[str],
                                evaluator_id: str) -> Dict:
        """创建评估任务"""
        return {
            'task_id': str(uuid.uuid4()),
            'query': query,
            'answer': answer,
            'contexts': contexts,
            'evaluator_id': evaluator_id,
            'dimensions': [
                {
                    'name': dim['name'],
                    'description': dim['description'],
                    'scale': dim['scale']
                }
                for dim in self.criteria.DIMENSIONS.values()
            ],
            'status': 'pending',
            'created_at': datetime.now().isoformat()
        }
    
    def submit_evaluation(self, task_id: str,
                          scores: Dict[str, int],
                          comments: str = "") -> Dict:
        """提交评估结果"""
        evaluation = {
            'task_id': task_id,
            'scores': scores,
            'average_score': np.mean(list(scores.values())),
            'comments': comments,
            'submitted_at': datetime.now().isoformat()
        }
        
        self.evaluations.append(evaluation)
        return evaluation
    
    def get_evaluator_agreement(self) -> float:
        """计算评估者间一致性"""
        # 使用Krippendorff's Alpha或Cohen's Kappa
        from sklearn.metrics import cohen_kappa_score
        
        # 简化实现：计算评分标准差
        if len(self.evaluations) < 2:
            return 1.0
        
        scores_by_dim = defaultdict(list)
        for eval_ in self.evaluations:
            for dim, score in eval_['scores'].items():
                scores_by_dim[dim].append(score)
        
        agreements = []
        for dim, scores in scores_by_dim.items():
            if len(scores) >= 2:
                std = np.std(scores)
                agreement = 1.0 - (std / 4.0)  # 4是评分范围的一半
                agreements.append(agreement)
        
        return np.mean(agreements) if agreements else 0.0
    
    def generate_report(self) -> Dict:
        """生成评估报告"""
        if not self.evaluations:
            return {}
        
        # 汇总各维度评分
        dim_scores = defaultdict(list)
        for eval_ in self.evaluations:
            for dim, score in eval_['scores'].items():
                dim_scores[dim].append(score)
        
        report = {
            'total_evaluations': len(self.evaluations),
            'average_scores': {
                dim: np.mean(scores)
                for dim, scores in dim_scores.items()
            },
            'overall_average': np.mean([
                eval_['average_score'] for eval_ in self.evaluations
            ]),
            'evaluator_agreement': self.get_evaluator_agreement(),
            'score_distributions': {
                dim: {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'min': np.min(scores),
                    'max': np.max(scores),
                    'median': np.median(scores)
                }
                for dim, scores in dim_scores.items()
            }
        }
        
        return report
```

### 15.5.2 LLM作为评估者

使用LLM模拟人工评估，可以大幅降低评估成本：

```python
class LLMJudge:
    """LLM评估者"""
    
    def __init__(self, llm, evaluation_criteria: Dict = None):
        self.llm = llm
        self.criteria = evaluation_criteria or {
            'accuracy': {
                'weight': 0.3,
                'prompt': '评估回答的事实准确性'
            },
            'relevance': {
                'weight': 0.25,
                'prompt': '评估回答与问题的相关性'
            },
            'completeness': {
                'weight': 0.2,
                'prompt': '评估回答的完整性'
            },
            'clarity': {
                'weight': 0.15,
                'prompt': '评估回答的清晰度'
            },
            'citation': {
                'weight': 0.1,
                'prompt': '评估回答的引用质量'
            }
        }
    
    def evaluate(self, query: str, answer: str,
                 contexts: List[str]) -> Dict:
        """LLM评估"""
        scores = {}
        
        for criterion, config in self.criteria.items():
            score = self._score_criterion(
                query, answer, contexts,
                criterion, config['prompt']
            )
            scores[criterion] = score
        
        # 计算加权总分
        weighted_score = sum(
            scores[c] * self.criteria[c]['weight']
            for c in self.criteria
        )
        
        return {
            'scores': scores,
            'weighted_score': weighted_score,
            'num_criteria': len(self.criteria)
        }
    
    def _score_criterion(self, query: str, answer: str,
                          contexts: List[str],
                          criterion: str, prompt_text: str) -> float:
        """评分单个维度"""
        context_text = "\n".join(contexts)[:2000]
        
        prompt = f"""作为RAG系统评估专家，请评估以下回答。

{prompt_text}

问题：{query}

回答：{answer[:1000]}

上下文（供参考）：
{context_text[:1000]}

请给出1-5分的评分（1最差，5最好），只输出数字："""
        
        try:
            response = self.llm(prompt)
            score = float(response.strip())
            return max(1.0, min(5.0, score))
        except:
            return 3.0  # 默认中等分数
    
    def evaluate_batch(self, samples: List[Dict]) -> Dict:
        """批量评估"""
        results = []
        
        for sample in samples:
            result = self.evaluate(
                sample['query'],
                sample['answer'],
                sample.get('contexts', [])
            )
            results.append(result)
        
        # 聚合
        avg_scores = defaultdict(list)
        for result in results:
            for criterion, score in result['scores'].items():
                avg_scores[criterion].append(score)
        
        return {
            'average_scores': {
                c: np.mean(s) for c, s in avg_scores.items()
            },
            'overall_average': np.mean([
                r['weighted_score'] for r in results
            ]),
            'num_samples': len(results)
        }
```

## 15.6 在线评估

在线评估（A/B测试）是在生产环境中评估RAG系统效果的重要手段。

### 15.6.1 A/B测试框架

```python
class ABTestFramework:
    """A/B测试框架"""
    
    def __init__(self, control_system, treatment_system,
                 traffic_split: float = 0.5):
        self.control = control_system
        self.treatment = treatment_system
        self.traffic_split = traffic_split
        
        self.results = {
            'control': [],
            'treatment': []
        }
    
    def route_request(self, user_id: str, query: str) -> tuple:
        """路由请求到实验组或对照组"""
        import hashlib
        
        # 基于用户ID一致性哈希
        hash_val = int(hashlib.md5(
            user_id.encode()
        ).hexdigest(), 16) % 100
        
        if hash_val < self.traffic_split * 100:
            # 实验组
            system = self.treatment
            group = 'treatment'
        else:
            # 对照组
            system = self.control
            group = 'control'
        
        return group, system.query(query)
    
    def record_result(self, group: str, query: str,
                      response: Dict, metrics: Dict):
        """记录结果"""
        self.results[group].append({
            'query': query,
            'response': response,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        })
    
    def analyze_results(self) -> Dict:
        """分析A/B测试结果"""
        analysis = {}
        
        # 计算各组的平均指标
        for group in ['control', 'treatment']:
            group_results = self.results[group]
            
            if not group_results:
                analysis[group] = {'error': '无数据'}
                continue
            
            # 聚合指标
            metrics_summary = defaultdict(list)
            for result in group_results:
                for metric, value in result['metrics'].items():
                    metrics_summary[metric].append(value)
            
            analysis[group] = {
                'num_queries': len(group_results),
                'average_metrics': {
                    metric: np.mean(values)
                    for metric, values in metrics_summary.items()
                }
            }
        
        # 统计显著性检验
        if 'control' in analysis and 'treatment' in analysis:
            analysis['significance'] = self._statistical_test()
        
        return analysis
    
    def _statistical_test(self) -> Dict:
        """统计显著性检验（t检验）"""
        from scipy import stats
        
        control_metrics = [
            r['metrics'].get('user_satisfaction', 0)
            for r in self.results['control']
        ]
        treatment_metrics = [
            r['metrics'].get('user_satisfaction', 0)
            for r in self.results['treatment']
        ]
        
        if len(control_metrics) < 2 or len(treatment_metrics) < 2:
            return {'can_compute': False}
        
        t_stat, p_value = stats.ttest_ind(
            control_metrics, treatment_metrics
        )
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'is_significant': p_value < 0.05,
            'improvement_pct': (
                (np.mean(treatment_metrics) - np.mean(control_metrics)) /
                np.mean(control_metrics) * 100
                if np.mean(control_metrics) != 0 else 0
            )
        }
```

### 15.6.2 用户反馈收集

```python
class UserFeedbackCollector:
    """用户反馈收集器"""
    
    def __init__(self):
        self.feedback = []
    
    def collect_explicit_feedback(self, query_id: str,
                                   rating: int,
                                   comment: str = "") -> Dict:
        """收集显式反馈"""
        feedback = {
            'query_id': query_id,
            'rating': rating,
            'comment': comment,
            'timestamp': datetime.now().isoformat(),
            'type': 'explicit'
        }
        self.feedback.append(feedback)
        return feedback
    
    def collect_implicit_feedback(self, query_id: str,
                                   actions: List[Dict]) -> Dict:
        """收集隐式反馈"""
        # 分析用户行为
        satisfaction_score = self._compute_satisfaction_from_actions(actions)
        
        feedback = {
            'query_id': query_id,
            'actions': actions,
            'satisfaction_score': satisfaction_score,
            'timestamp': datetime.now().isoformat(),
            'type': 'implicit'
        }
        self.feedback.append(feedback)
        return feedback
    
    def _compute_satisfaction_from_actions(self, 
                                            actions: List[Dict]) -> float:
        """从用户行为推断满意度"""
        score = 0.5  # 中性基准
        
        for action in actions:
            action_type = action.get('type', '')
            
            if action_type == 'copy_answer':
                score += 0.1
            elif action_type == 'click_source':
                score += 0.05
            elif action_type == 'regenerate':
                score -= 0.1
            elif action_type == 'switch_to_search':
                score -= 0.15
            elif action_type == 'feedback_positive':
                score += 0.2
            elif action_type == 'feedback_negative':
                score -= 0.2
            elif action_type == 'session_end':
                # 会话时长也可以作为信号
                pass
        
        return max(0.0, min(1.0, score))
    
    def get_satisfaction_trend(self, 
                                window: str = '7d') -> Dict:
        """获取满意度趋势"""
        from datetime import timedelta
        
        now = datetime.now()
        if window.endswith('d'):
            delta = timedelta(days=int(window[:-1]))
        elif window.endswith('h'):
            delta = timedelta(hours=int(window[:-1]))
        else:
            delta = timedelta(days=7)
        
        cutoff = now - delta
        
        recent_feedback = [
            f for f in self.feedback
            if datetime.fromisoformat(f['timestamp']) > cutoff
        ]
        
        if not recent_feedback:
            return {}
        
        # 按天聚合
        daily_scores = defaultdict(list)
        for fb in recent_feedback:
            day = fb['timestamp'][:10]
            score = fb.get('rating', fb.get('satisfaction_score', 0.5))
            daily_scores[day].append(score)
        
        return {
            'daily_average': {
                day: np.mean(scores)
                for day, scores in daily_scores.items()
            },
            'overall_average': np.mean([
                fb.get('rating', fb.get('satisfaction_score', 0.5))
                for fb in recent_feedback
            ]),
            'total_feedback': len(recent_feedback)
        }
```

## 15.7 评估数据集构建

高质量的评估数据集是评估体系的基础。

### 15.7.1 数据集构建方法

```python
class EvalDatasetBuilder:
    """评估数据集构建器"""
    
    def __init__(self, llm, document_store):
        self.llm = llm
        self.document_store = document_store
    
    def generate_qa_pairs(self, documents: List[Dict],
                           num_pairs: int = 100) -> List[Dict]:
        """从文档生成问答对"""
        qa_pairs = []
        
        for doc in documents:
            prompt = f"""基于以下文档生成问答对。

文档内容：
{doc['content'][:1000]}

生成3个问题-答案对，要求：
1. 问题需要检索文档才能回答
2. 答案必须完全基于文档内容
3. 覆盖文档的不同方面

输出JSON格式：
[
    {{
        "question": "问题",
        "answer": "答案",
        "difficulty": "easy/medium/hard",
        "source_doc": "文档ID"
    }}
]"""
            
            try:
                import json
                response = self.llm(prompt)
                pairs = json.loads(response)
                
                for pair in pairs:
                    pair['doc_content'] = doc['content'][:500]
                    qa_pairs.append(pair)
                
                if len(qa_pairs) >= num_pairs:
                    break
                    
            except:
                continue
        
        return qa_pairs[:num_pairs]
    
    def generate_hard_negatives(self, qa_pairs: List[Dict],
                                 num_negatives: int = 3) -> List[Dict]:
        """生成难负例"""
        for pair in qa_pairs:
            question = pair['question']
            correct_answer = pair['answer']
            
            # 检索相似但不相关的文档
            similar_docs = self.document_store.similarity_search(
                question, k=num_negatives + 5
            )
            
            negatives = []
            for doc in similar_docs:
                if doc.page_content != pair.get('doc_content', ''):
                    # 验证是否为真正的负例
                    if not self._is_relevant(doc.page_content, question):
                        negatives.append(doc.page_content[:500])
                
                if len(negatives) >= num_negatives:
                    break
            
            pair['hard_negatives'] = negatives
        
        return qa_pairs
    
    def _is_relevant(self, content: str, question: str) -> bool:
        """判断文档是否与问题相关"""
        prompt = f"""判断以下文档是否与问题相关。

问题：{question}

文档：{content[:300]}

回答"是"或"否"："""
        
        try:
            response = self.llm(prompt)
            return '是' in response or 'Yes' in response
        except:
            return False
    
    def build_retrieval_dataset(self, qa_pairs: List[Dict]) -> Dict:
        """构建检索评估数据集"""
        retrieval_data = {
            'queries': [],
            'relevant_docs': [],
            'all_docs': []
        }
        
        for pair in qa_pairs:
            query = pair['question']
            relevant = [pair.get('source_doc', '')]
            negatives = pair.get('hard_negatives', [])
            
            retrieval_data['queries'].append(query)
            retrieval_data['relevant_docs'].append(relevant)
            retrieval_data['all_docs'].append(negatives)
        
        return retrieval_data
    
    def build_generation_dataset(self, qa_pairs: List[Dict]) -> Dict:
        """构建生成评估数据集"""
        generation_data = {
            'queries': [],
            'ground_truths': [],
            'contexts': []
        }
        
        for pair in qa_pairs:
            generation_data['queries'].append(pair['question'])
            generation_data['ground_truths'].append(pair['answer'])
            generation_data['contexts'].append([
                pair.get('doc_content', '')
            ] + pair.get('hard_negatives', []))
        
        return generation_data
```

### 15.7.2 数据集质量控制

```python
class DatasetQualityControl:
    """数据集质量控制"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def validate_qa_pair(self, question: str, answer: str,
                          context: str) -> Dict:
        """验证问答对质量"""
        prompt = f"""验证以下问答对的质量。

问题：{question}
答案：{answer}
上下文：{context[:500]}

检查：
1. 答案是否完全基于上下文？
2. 问题是否清晰明确？
3. 答案是否正确？

输出JSON：
{{
    "is_valid": true/false,
    "issues": ["问题列表"],
    "quality_score": 0.0-1.0
}}"""
        
        try:
            import json
            response = self.llm(prompt)
            return json.loads(response)
        except:
            return {"is_valid": True, "quality_score": 0.8}
    
    def filter_low_quality(self, qa_pairs: List[Dict],
                           threshold: float = 0.7) -> List[Dict]:
        """过滤低质量数据"""
        valid_pairs = []
        
        for pair in qa_pairs:
            validation = self.validate_qa_pair(
                pair['question'],
                pair['answer'],
                pair.get('doc_content', '')
            )
            
            if validation.get('quality_score', 0) >= threshold:
                pair['validation'] = validation
                valid_pairs.append(pair)
        
        return valid_pairs
    
    def deduplicate(self, qa_pairs: List[Dict]) -> List[Dict]:
        """去重"""
        seen_questions = set()
        unique_pairs = []
        
        for pair in qa_pairs:
            question_normalized = pair['question'].lower().strip()
            if question_normalized not in seen_questions:
                seen_questions.add(question_normalized)
                unique_pairs.append(pair)
        
        return unique_pairs
```

## 15.8 持续优化流水线

### 15.8.1 评估-优化循环

```python
class ContinuousOptimizationPipeline:
    """持续优化流水线"""
    
    def __init__(self, system, evaluator, dataset):
        self.system = system
        self.evaluator = evaluator
        self.dataset = dataset
        
        self.history = []
        self.baseline = None
    
    def run_evaluation_cycle(self, 
                              system_config: Dict = None) -> Dict:
        """运行评估循环"""
        if system_config:
            self.system.configure(system_config)
        
        # 1. 获取系统输出
        outputs = []
        for item in self.dataset:
            result = self.system.query(item['question'])
            outputs.append({
                'question': item['question'],
                'answer': result.get('answer', ''),
                'contexts': result.get('contexts', []),
                'ground_truth': item.get('ground_truth', '')
            })
        
        # 2. 评估
        results = self.evaluator.evaluate(outputs)
        
        # 3. 记录历史
        self.history.append({
            'config': system_config,
            'results': results,
            'timestamp': datetime.now().isoformat()
        })
        
        # 4. 设置基线
        if self.baseline is None:
            self.baseline = results
        
        return results
    
    def compare_with_baseline(self) -> Dict:
        """与基线比较"""
        if not self.history or self.baseline is None:
            return {}
        
        latest = self.history[-1]['results']
        comparison = {}
        
        for metric in ['ragas_score', 'faithfulness', 'answer_relevancy',
                       'context_precision', 'context_recall']:
            if metric in latest and metric in self.baseline:
                diff = latest[metric] - self.baseline[metric]
                comparison[metric] = {
                    'baseline': self.baseline[metric],
                    'current': latest[metric],
                    'change': diff,
                    'change_pct': (
                        diff / self.baseline[metric] * 100
                        if self.baseline[metric] != 0 else 0
                    )
                }
        
        return comparison
    
    def identify_regression(self, threshold: float = -0.05) -> List[Dict]:
        """识别性能回退"""
        regressions = []
        
        if len(self.history) < 2:
            return regressions
        
        current = self.history[-1]['results']
        previous = self.history[-2]['results']
        
        for metric in current:
            if metric in previous:
                diff = current[metric] - previous[metric]
                if diff < threshold:
                    regressions.append({
                        'metric': metric,
                        'previous': previous[metric],
                        'current': current[metric],
                        'decline': diff,
                        'config': self.history[-1].get('config')
                    })
        
        return regressions
    
    def suggest_optimizations(self, results: Dict) -> List[Dict]:
        """基于评估结果建议优化方向"""
        suggestions = []
        
        # 检查忠实度
        if results.get('faithfulness', 1.0) < 0.7:
            suggestions.append({
                'target': 'faithfulness',
                'priority': 'high',
                'suggestion': '增强检索相关性过滤，改进上下文选择策略'
            })
        
        # 检查相关性
        if results.get('answer_relevancy', 1.0) < 0.7:
            suggestions.append({
                'target': 'answer_relevancy',
                'priority': 'high',
                'suggestion': '优化查询理解，改进检索查询生成'
            })
        
        # 检查上下文精确度
        if results.get('context_precision', 1.0) < 0.7:
            suggestions.append({
                'target': 'context_precision',
                'priority': 'medium',
                'suggestion': '改进检索排序，添加相关性重排序'
            })
        
        # 检查上下文召回率
        if results.get('context_recall', 1.0) < 0.7:
            suggestions.append({
                'target': 'context_recall',
                'priority': 'medium',
                'suggestion': '增加检索数量，使用查询扩展技术'
            })
        
        return suggestions
```

### 15.8.2 回归测试

```python
class RegressionTestSuite:
    """回归测试套件"""
    
    def __init__(self):
        self.test_cases = []
        self.regression_threshold = 0.05
    
    def add_test_case(self, query: str, 
                       expected_behavior: Dict):
        """添加测试用例"""
        self.test_cases.append({
            'query': query,
            'expected': expected_behavior
        })
    
    def run_regression(self, system) -> Dict:
        """运行回归测试"""
        results = {
            'passed': 0,
            'failed': 0,
            'details': []
        }
        
        for i, test_case in enumerate(self.test_cases):
            query = test_case['query']
            expected = test_case['expected']
            
            # 执行查询
            response = system.query(query)
            
            # 验证
            test_result = self._verify_test_case(
                response, expected
            )
            
            if test_result['passed']:
                results['passed'] += 1
            else:
                results['failed'] += 1
            
            results['details'].append({
                'test_id': i + 1,
                'query': query,
                'passed': test_result['passed'],
                'expected': expected,
                'actual': test_result['actual'],
                'errors': test_result.get('errors', [])
            })
        
        results['pass_rate'] = (
            results['passed'] / len(self.test_cases)
            if self.test_cases else 0
        )
        
        return results
    
    def _verify_test_case(self, response: Dict,
                           expected: Dict) -> Dict:
        """验证单个测试用例"""
        errors = []
        
        # 验证包含特定关键词
        if 'must_contain' in expected:
            for keyword in expected['must_contain']:
                if keyword not in response.get('answer', ''):
                    errors.append(f"缺少关键词: {keyword}")
        
        # 验证不包含特定关键词
        if 'must_not_contain' in expected:
            for keyword in expected['must_not_contain']:
                if keyword in response.get('answer', ''):
                    errors.append(f"包含不应出现的关键词: {keyword}")
        
        # 验证引用数
        if 'min_citations' in expected:
            citations = response.get('citations', [])
            if len(citations) < expected['min_citations']:
                errors.append(
                    f"引用数不足: {len(citations)} < {expected['min_citations']}"
                )
        
        # 验证延迟
        if 'max_latency_ms' in expected:
            latency = response.get('latency_ms', 0)
            if latency > expected['max_latency_ms']:
                errors.append(
                    f"延迟超标: {latency}ms > {expected['max_latency_ms']}ms"
                )
        
        return {
            'passed': len(errors) == 0,
            'errors': errors,
            'actual': response
        }
```

## 15.9 Bad Case分析

深入分析错误案例是改进系统最有效的方法之一。

### 15.9.1 Bad Case分类

```python
class BadCaseAnalyzer:
    """Bad Case分析器"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def analyze(self, query: str, answer: str,
                contexts: List[str], error_type: str = None) -> Dict:
        """分析Bad Case"""
        analysis = {
            'query': query,
            'error_type': error_type or self._classify_error(query, answer, contexts),
            'root_cause': self._identify_root_cause(query, answer, contexts),
            'impact': self._assess_impact(answer, query),
            'improvement_suggestions': []
        }
        
        analysis['improvement_suggestions'] = self._suggest_improvements(
            analysis['error_type'], analysis['root_cause']
        )
        
        return analysis
    
    def _classify_error(self, query: str, answer: str,
                         contexts: List[str]) -> str:
        """分类错误类型"""
        prompt = f"""分析以下RAG系统输出中的错误类型。

问题：{query}
回答：{answer[:500]}

可用上下文：
{chr(10).join([c[:200] for c in contexts])}

错误类型分类：
1. retrieval_failure: 检索结果不相关
2. hallucination: 生成内容无依据
3. incompleteness: 回答不完整
4. irrelevance: 回答与问题无关
5. contradiction: 回答自相矛盾
6. citation_error: 引用错误

输出JSON：
{{
    "error_type": "类型",
    "confidence": 0.0-1.0,
    "reason": "分类原因"
}}"""
        
        try:
            import json
            response = self.llm(prompt)
            result = json.loads(response)
            return result.get('error_type', 'unknown')
        except:
            return 'unknown'
    
    def _identify_root_cause(self, query: str, answer: str,
                              contexts: List[str]) -> Dict:
        """识别根因"""
        prompt = f"""识别以下RAG错误的根本原因。

问题：{query}
回答：{answer[:500]}

检索上下文：
{chr(10).join([c[:200] for c in contexts])}

分析根因，输出JSON：
{{
    "primary_cause": "主要根因",
    "contributing_factors": ["次要因素"],
    "evidence": ["支持证据"],
    "fix_priority": "high/medium/low"
}}"""
        
        try:
            import json
            response = self.llm(prompt)
            return json.loads(response)
        except:
            return {
                "primary_cause": "未知",
                "fix_priority": "medium"
            }
    
    def _assess_impact(self, answer: str, query: str) -> Dict:
        """评估影响"""
        prompt = f"""评估以下回答错误的影响程度。

问题：{query}
回答：{answer[:500]}

评估维度：
1. 用户满意度影响
2. 信息准确性影响
3. 任务完成度影响

输出JSON：
{{
    "user_satisfaction_impact": "high/medium/low",
    "information_accuracy_impact": "high/medium/low",
    "task_completion_impact": "high/medium/low",
    "overall_severity": "critical/major/minor/cosmetic"
}}"""
        
        try:
            import json
            response = self.llm(prompt)
            return json.loads(response)
        except:
            return {"overall_severity": "minor"}
    
    def _suggest_improvements(self, error_type: str,
                               root_cause: Dict) -> List[str]:
        """建议改进措施"""
        suggestions = {
            'retrieval_failure': [
                '改进查询扩展策略，增加同义词和近义词',
                '调整检索参数（chunk_size, top_k）',
                '添加混合检索（向量+关键词）',
                '优化嵌入模型或使用领域微调模型'
            ],
            'hallucination': [
                '增强上下文约束提示词',
                '添加生成后事实验证步骤',
                '限制生成范围，强制引用来源',
                '降低生成温度参数'
            ],
            'incompleteness': [
                '增加检索文档数量',
                '使用多步检索策略',
                '添加查询分解机制',
                '增强答案结构引导'
            ],
            'irrelevance': [
                '改进查询理解和意图识别',
                '添加检索结果相关性重排序',
                '优化提示词引导'
            ]
        }
        
        return suggestions.get(error_type, [
            '收集更多Bad Case进行根因分析',
            '检查评估数据的质量和覆盖度'
        ])
    
    def batch_analyze(self, bad_cases: List[Dict]) -> Dict:
        """批量分析"""
        analyses = []
        error_distribution = defaultdict(int)
        
        for case in bad_cases:
            analysis = self.analyze(
                case['query'],
                case['answer'],
                case.get('contexts', []),
                case.get('error_type')
            )
            analyses.append(analysis)
            error_distribution[analysis['error_type']] += 1
        
        return {
            'total_cases': len(bad_cases),
            'error_distribution': dict(error_distribution),
            'analyses': analyses,
            'top_issues': sorted(
                error_distribution.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
        }
```

### 15.9.2 Bad Case管理系统

```python
class BadCaseManager:
    """Bad Case管理器"""
    
    def __init__(self, storage_path: str = "bad_cases.json"):
        self.storage_path = storage_path
        self.bad_cases = []
        self.load()
    
    def add_case(self, case: Dict):
        """添加Bad Case"""
        case['id'] = str(uuid.uuid4())
        case['created_at'] = datetime.now().isoformat()
        case['status'] = 'open'
        self.bad_cases.append(case)
        self.save()
    
    def update_status(self, case_id: str, status: str,
                      resolution: str = ""):
        """更新状态"""
        for case in self.bad_cases:
            if case['id'] == case_id:
                case['status'] = status
                if resolution:
                    case['resolution'] = resolution
                case['updated_at'] = datetime.now().isoformat()
                break
        self.save()
    
    def get_statistics(self) -> Dict:
        """获取统计"""
        if not self.bad_cases:
            return {}
        
        status_count = Counter(c['status'] for c in self.bad_cases)
        type_count = Counter(
            c.get('error_type', 'unknown') for c in self.bad_cases
        )
        
        return {
            'total': len(self.bad_cases),
            'by_status': dict(status_count),
            'by_type': dict(type_count),
            'resolution_rate': (
                status_count.get('resolved', 0) / len(self.bad_cases)
                if self.bad_cases else 0
            )
        }
    
    def save(self):
        """持久化"""
        import json
        with open(self.storage_path, 'w', encoding='utf-8') as f:
            json.dump(self.bad_cases, f, ensure_ascii=False, indent=2)
    
    def load(self):
        """加载"""
        import json
        import os
        
        if os.path.exists(self.storage_path):
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                self.bad_cases = json.load(f)
```

## 15.10 本章小结

本章全面介绍了RAG系统的评估体系，涵盖从自动化指标到人工评估、从离线评估到在线评估的完整方法论。

**评估维度**方面，RAG系统需要在检索质量、生成质量和系统性能三个层面进行评估。检索质量关注文档检索的精确性、排序质量和覆盖度；生成质量关注答案的忠实度、相关性、完整性和语言质量；系统性能关注延迟、吞吐量和成本。

**自动化指标**方面，本章实现了完整的检索评估指标套件（Recall@K、Precision@K、MRR、MAP、NDCG）和生成评估指标（忠实度、相关性、幻觉检测、引用准确性）。自动化评估可以在开发过程中快速反馈系统性能。

**RAGAS框架**提供了专门针对RAG系统的评估指标体系，包括忠实度（Faithfulness）、答案相关性（Answer Relevancy）、上下文精确度（Context Precision）和上下文召回率（Context Recall）四个核心指标。

**人工评估**仍然是评估生成质量的金标准。本章设计了多维度的评估标准，并提供了LLM作为评估者的实现方案，在成本和准确性之间取得平衡。

**在线评估**通过在真实用户流量中进行A/B测试，获取最真实的系统性能评估。用户反馈收集（显式和隐式）为持续优化提供了重要信号。

**评估数据集构建**是评估体系的基础。本章介绍了从文档自动生成问答对、构建难负例和质量控制的方法。

**持续优化**将评估和优化形成闭环。通过定期评估、回归测试和Bad Case分析，不断发现系统不足并指导优化方向。

在实际应用中，建议建立多层次的评估策略：开发阶段使用自动化指标快速迭代；发布前进行人工评估确保质量；上线后通过在线评估和用户反馈持续监控。同时，建立Bad Case管理系统，将每个错误案例转化为改进的机会，形成数据驱动的持续优化文化。
