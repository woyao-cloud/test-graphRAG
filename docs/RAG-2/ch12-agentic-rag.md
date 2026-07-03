# 第12章 智能体RAG（Agentic RAG）

## 12.1 引言

传统的RAG（Retrieval-Augmented Generation）系统采用"检索-生成"的线性流水线架构，用户查询经过一次检索后直接送入大语言模型（LLM）生成答案。这种被动式的信息处理方式在面对复杂查询、多步骤推理和需要外部工具交互的场景时存在明显的局限性。Agentic RAG（智能体RAG）通过引入自主智能体的概念，赋予RAG系统主动思考、规划和行动的能力，从而突破了传统RAG的局限。

智能体RAG的核心思想是让LLM不再仅仅是一个"生成器"，而是成为一个"推理引擎"和"决策者"。它可以根据查询的需要，自主决定是否检索、检索什么、使用什么工具、如何推理、以及是否需要修正自己的输出。这种"推理+行动"（Reasoning + Acting）的模式，使RAG系统能够处理更复杂的任务，提供更准确的答案。

本章将深入探讨智能体RAG的核心技术，包括ReAct模式、Corrective RAG（CRAG）、Self-RAG、多智能体协作、工具使用、多步骤推理和自我纠正机制，并通过LangGraph实现模式提供完整的代码示例。

### 12.1.1 从传统RAG到智能体RAG的演进

传统RAG的局限性主要体现在以下几个方面：

1. **单次检索**：传统RAG只进行一次检索，如果检索结果不相关，最终答案也会不准确
2. **无反馈机制**：无法根据生成结果判断是否需要补充检索或修正
3. **缺乏工具使用能力**：无法调用计算器、代码执行器、API等外部工具
4. **无多步骤推理**：无法将复杂问题分解为多个子问题逐步求解
5. **被动响应**：缺乏主动规划和自我反思能力

智能体RAG通过引入智能体范式解决了这些问题：

```
传统RAG:  查询 → 检索 → 生成 → 答案
智能体RAG: 查询 → 思考(需要什么信息?) → 行动(检索/计算/查询) → 
           观察(结果如何?) → 思考(还需要什么?) → 行动(...) → 最终答案
```

### 12.1.2 智能体RAG的核心能力

一个完整的智能体RAG系统应具备以下核心能力：

| 能力 | 描述 | 实现方式 |
|------|------|---------|
| 推理规划 | 分解复杂问题，制定执行计划 | ReAct模式、思维链 |
| 工具使用 | 调用外部工具获取信息或执行操作 | 函数调用（Function Calling） |
| 多步检索 | 根据中间结果决定是否补充检索 | 迭代检索、自适应检索 |
| 自我反思 | 评估自身输出的质量和准确性 | Self-RAG、自我纠正 |
| 纠错机制 | 识别并修正错误或幻觉内容 | CRAG、验证-修正循环 |
| 记忆管理 | 维护对话历史和中间推理状态 | 上下文窗口、外部记忆 |

## 12.2 ReAct模式

ReAct（Reasoning + Acting）是智能体RAG的核心理念，由Shunyu Yao等人在2022年提出。它将推理（Reasoning）和行动（Acting）交替进行，使LLM能够基于推理结果采取行动，并根据行动结果更新推理。

### 12.2.1 ReAct原理

ReAct模式的核心思想是在每个步骤中交替进行"思考"（Thought）和"行动"（Action）：

```
思考: 用户想知道什么？我需要哪些信息？
行动: 调用检索工具搜索相关信息
观察: 找到了三条相关文档
思考: 这些信息已经足够回答问题了吗？
行动: 需要进一步查询知识图谱获取更多细节
观察: 知识图谱返回了实体关系
思考: 现在我可以综合所有信息来回答
行动: 生成最终答案
```

这种模式的优点在于：
- **透明度**：每一步的推理过程都显式可见
- **可控性**：可以在任意步骤介入或修正
- **鲁棒性**：某一步失败时可以通过重新规划恢复
- **可扩展性**：可以轻松添加新的工具和行动类型

### 12.2.2 ReAct提示词模板

实现ReAct模式需要精心设计的提示词模板，引导LLM按照"思考-行动-观察"的循环进行推理：

```python
REACT_SYSTEM_PROMPT = """你是一个具有推理和行动能力的智能助手。
你可以使用以下工具来获取信息或执行操作：

{tool_descriptions}

请遵循以下格式进行推理：
问题: 用户提出的问题
思考: 你需要思考当前状态，决定下一步行动
行动: 选择要使用的工具和输入参数
行动输入: 工具的输入
观察: 工具返回的结果
...（重复思考-行动-观察循环直到可以回答问题）
思考: 我现在可以回答这个问题了
最终答案: 对用户的最终回答

重要规则：
1. 每次只执行一个行动
2. 基于观察结果决定下一步
3. 当信息足够时，给出最终答案
4. 如果工具返回错误，尝试其他方法
5. 始终用中文回答用户问题"""

def build_react_prompt(query: str, 
                        tools: List[Dict],
                        max_steps: int = 5) -> List[Dict]:
    """构建ReAct提示词"""
    tool_descriptions = "\n\n".join([
        f"工具名称: {t['name']}\n"
        f"功能描述: {t['description']}\n"
        f"参数: {t.get('parameters', '无')}"
        for t in tools
    ])
    
    system_prompt = REACT_SYSTEM_PROMPT.format(
        tool_descriptions=tool_descriptions
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"问题: {query}"}
    ]
    
    return messages
```

### 12.2.3 ReAct循环实现

以下是ReAct循环的完整实现：

```python
import json
import re
from typing import List, Dict, Any, Optional, Callable

class ReActAgent:
    """ReAct智能体实现"""
    
    def __init__(self, 
                 llm: Callable,
                 tools: Dict[str, Callable],
                 max_steps: int = 10,
                 verbose: bool = True):
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps
        self.verbose = verbose
        self.trajectory = []
    
    def run(self, query: str) -> str:
        """执行ReAct循环"""
        messages = self._build_initial_messages(query)
        
        for step in range(self.max_steps):
            if self.verbose:
                print(f"\n=== Step {step + 1} ===")
            
            # 1. LLM生成思考/行动
            response = self.llm(messages)
            content = response['content']
            
            if self.verbose:
                print(f"LLM输出:\n{content}\n")
            
            # 记录轨迹
            self.trajectory.append({
                'step': step,
                'content': content
            })
            
            # 2. 检查是否包含最终答案
            if "最终答案:" in content:
                final_answer = self._extract_final_answer(content)
                return final_answer
            
            # 3. 解析行动
            action = self._parse_action(content)
            if action is None:
                # 没有有效行动，尝试直接回答
                messages.append({
                    "role": "assistant", 
                    "content": content
                })
                messages.append({
                    "role": "user",
                    "content": "请直接给出最终答案。"
                })
                continue
            
            # 4. 执行行动
            tool_name = action['name']
            tool_input = action['input']
            
            if tool_name not in self.tools:
                observation = f"错误: 未知工具 '{tool_name}'"
            else:
                try:
                    result = self.tools[tool_name](tool_input)
                    observation = self._format_observation(tool_name, result)
                except Exception as e:
                    observation = f"工具执行错误: {str(e)}"
            
            if self.verbose:
                print(f"行动: {tool_name}({tool_input})")
                print(f"观察: {observation[:200]}...\n")
            
            # 5. 将思考-行动-观察添加到消息
            messages.append({
                "role": "assistant",
                "content": content
            })
            messages.append({
                "role": "user",
                "content": f"观察: {observation}"
            })
        
        # 超过最大步骤数
        return "抱歉，我无法在允许的步骤内完成这个任务。"
    
    def _build_initial_messages(self, query: str) -> List[Dict]:
        """构建初始消息"""
        tool_descriptions = "\n\n".join([
            f"## {name}\n{func.__doc__ or '无描述'}"
            for name, func in self.tools.items()
        ])
        
        system_prompt = f"""你是一个具有推理和行动能力的智能助手。
你可以使用以下工具：

{tool_descriptions}

请遵循格式：
思考: 你的推理
行动: 工具名称
行动输入: 输入参数
观察: 工具返回结果
...（重复直到可以回答）
思考: 我现在可以回答了
最终答案: 你的回答"""
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
    
    def _parse_action(self, content: str) -> Optional[Dict]:
        """解析LLM输出中的行动指令"""
        # 匹配 "行动: xxx" 和 "行动输入: xxx"
        action_match = re.search(r'行动:\s*(.+?)(?:\n|$)', content)
        input_match = re.search(r'行动输入:\s*(.+?)(?:\n|$)', content)
        
        if action_match:
            return {
                'name': action_match.group(1).strip(),
                'input': input_match.group(1).strip() if input_match else ""
            }
        
        return None
    
    def _extract_final_answer(self, content: str) -> str:
        """提取最终答案"""
        match = re.search(r'最终答案:\s*(.+?)$', content, re.DOTALL)
        if match:
            return match.group(1).strip()
        return content
    
    def _format_observation(self, tool_name: str, result: Any) -> str:
        """格式化工具观察结果"""
        if isinstance(result, str):
            return result
        if isinstance(result, (list, dict)):
            return json.dumps(result, ensure_ascii=False, indent=2)[:500]
        return str(result)
    
    def get_trajectory(self) -> List[Dict]:
        """获取完整的推理轨迹"""
        return self.trajectory
    
    def reset(self):
        """重置智能体状态"""
        self.trajectory = []
```

### 12.2.4 ReAct工具定义示例

在RAG场景中，ReAct智能体可以使用多种检索工具：

```python
class RetrieverTools:
    """检索工具集合"""
    
    def __init__(self, vector_store, kg_client=None):
        self.vector_store = vector_store
        self.kg_client = kg_client
    
    def vector_search(self, query: str, top_k: int = 5) -> str:
        """向量检索工具
        
        通过语义相似度检索相关文档。
        输入参数: query - 查询文本
        返回: 检索到的文档列表
        """
        results = self.vector_store.similarity_search(query, k=top_k)
        
        formatted = []
        for i, doc in enumerate(results, 1):
            formatted.append(
                f"[文档{i}] 来源: {doc.metadata.get('source', '未知')}\n"
                f"内容: {doc.page_content[:300]}...\n"
                f"相关性: {doc.metadata.get('score', 0):.3f}"
            )
        
        return "\n\n".join(formatted) if formatted else "未找到相关文档"
    
    def keyword_search(self, query: str, top_k: int = 5) -> str:
        """关键词检索工具
        
        使用BM25算法进行关键词匹配检索。
        输入参数: query - 查询文本
        返回: 匹配的文档列表
        """
        # BM25检索实现
        results = self._bm25_search(query, top_k)
        
        if not results:
            return "未找到匹配的文档"
        
        return "\n\n".join([
            f"[文档{i}] 来源: {r['source']}\n内容: {r['content'][:300]}"
            for i, r in enumerate(results, 1)
        ])
    
    def kg_query(self, query: str) -> str:
        """知识图谱查询工具
        
        查询知识图谱获取结构化信息。
        输入参数: query - 自然语言查询或Cypher查询
        返回: 知识图谱查询结果
        """
        if self.kg_client is None:
            return "知识图谱服务不可用"
        
        try:
            result = self.kg_client.query(query)
            return json.dumps(result, ensure_ascii=False, indent=2)
        except Exception as e:
            return f"知识图谱查询失败: {str(e)}"
    
    def calculate(self, expression: str) -> str:
        """计算器工具
        
        执行数学计算。输入参数: expression - 数学表达式
        示例输入: "2 + 3 * 4"
        """
        try:
            # 安全执行数学表达式
            allowed_names = {
                k: v for k, v in math.__dict__.items()
                if not k.startswith("__")
            }
            result = eval(expression, {"__builtins__": {}}, allowed_names)
            return f"计算结果: {result}"
        except Exception as e:
            return f"计算错误: {str(e)}"
    
    def _bm25_search(self, query: str, top_k: int) -> List[Dict]:
        """BM25检索实现"""
        # 简化实现
        return []
```

### 12.2.5 多工具路由策略

ReAct智能体需要能够智能地选择使用哪个工具。以下是几种路由策略：

```python
class ToolRouter:
    """智能工具路由"""
    
    def __init__(self, tools: Dict[str, Callable]):
        self.tools = tools
        self.usage_stats = defaultdict(int)
    
    def select_tool(self, query: str, context: Dict = None) -> str:
        """选择最适合的工具"""
        query_lower = query.lower()
        
        # 基于查询特征的规则路由
        if self._needs_calculation(query_lower):
            return "calculate"
        elif self._needs_kg_query(query_lower):
            return "kg_query"
        elif self._is_factual_question(query_lower):
            return "vector_search"
        else:
            return "keyword_search"
    
    def _needs_calculation(self, query: str) -> bool:
        """判断是否需要计算"""
        calc_keywords = ['计算', '多少', '总和', '平均', '统计',
                        '加减乘除', '百分比', '概率']
        return any(kw in query for kw in calc_keywords)
    
    def _needs_kg_query(self, query: str) -> bool:
        """判断是否需要知识图谱"""
        kg_keywords = ['关系', '关联', '路径', '网络', '图谱',
                      '实体', '属性', '分类']
        return any(kw in query for kw in kg_keywords)
    
    def _is_factual_question(self, query: str) -> bool:
        """判断是否为事实性问题"""
        fact_keywords = ['什么', '谁', '哪里', '什么时候', '为什么',
                        '如何', '定义', '概念', '原理']
        return any(kw in query for kw in fact_keywords)
```

## 12.3 CRAG（Corrective RAG）

CRAG（Corrective RAG）由Yan等人于2024年提出，其核心思想是在检索结果质量不佳时进行纠正，而不是盲目使用检索结果。CRAG通过引入一个检索评估器来判断检索结果的相关性，并根据评估结果采取不同的行动。

### 12.3.1 CRAG架构

CRAG的工作流程包括三个主要阶段：

```
┌──────────┐    ┌──────────┐    ┌──────────┐
│  检索阶段  │ → │  评估阶段  │ → │  纠正阶段  │
└──────────┘    └──────────┘    └──────────┘
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
       相关        部分相关     不相关
          │          │          │
          ▼          ▼          ▼
     直接生成    知识分解重构   重新检索
                             或网络搜索
```

### 12.3.2 检索评估器

检索评估器是CRAG的核心组件，用于评估检索结果的相关性：

```python
class RetrievalEvaluator:
    """检索结果评估器"""
    
    def __init__(self, llm, threshold: float = 0.7):
        self.llm = llm
        self.threshold = threshold
    
    def evaluate(self, query: str, documents: List[Document]) -> List[Evaluation]:
        """评估检索结果的可靠性"""
        evaluations = []
        
        for doc in documents:
            # 计算相关性分数
            relevance_score = self._compute_relevance(query, doc)
            
            # 评估置信度
            confidence = self._assess_confidence(doc)
            
            # 检测潜在幻觉
            hallucination_risk = self._detect_hallucination_risk(query, doc)
            
            evaluation = Evaluation(
                doc_id=doc.id,
                relevance_score=relevance_score,
                confidence=confidence,
                hallucination_risk=hallucination_risk,
                is_relevant=relevance_score >= self.threshold,
                needs_correction=hallucination_risk > 0.3
            )
            evaluations.append(evaluation)
        
        return evaluations
    
    def _compute_relevance(self, query: str, doc: Document) -> float:
        """计算查询与文档的相关性"""
        prompt = f"""判断以下文档是否与查询相关。

查询: {query}

文档内容: {doc.page_content[:500]}

请给出0-1之间的相关性评分，只输出数字："""
        
        response = self.llm(prompt)
        try:
            score = float(response.strip())
            return max(0.0, min(1.0, score))
        except ValueError:
            return 0.5
    
    def _assess_confidence(self, doc: Document) -> float:
        """评估文档置信度"""
        # 基于文档元数据评估
        score = doc.metadata.get('score', 0.5)
        
        # 考虑文档来源可靠性
        source = doc.metadata.get('source', '')
        reliable_sources = ['official', 'verified', 'expert']
        if any(s in source.lower() for s in reliable_sources):
            score += 0.2
        
        # 考虑文档时效性
        if 'date' in doc.metadata:
            from datetime import datetime
            doc_date = datetime.fromisoformat(doc.metadata['date'])
            age = (datetime.now() - doc_date).days
            if age > 365:
                score -= 0.1
        
        return max(0.0, min(1.0, score))
    
    def _detect_hallucination_risk(self, query: str, doc: Document) -> float:
        """检测文档中的幻觉风险"""
        # 检查文档是否包含矛盾信息
        contradictions = self._check_contradictions(doc)
        
        # 检查文档是否包含模糊表述
        vague_terms = ['可能', '也许', '大概', '据说', '传闻']
        vague_count = sum(
            doc.page_content.count(term) for term in vague_terms
        )
        
        risk = (contradictions * 0.5 + min(vague_count / 10, 1.0) * 0.5)
        return min(risk, 1.0)
    
    def _check_contradictions(self, doc: Document) -> float:
        """检查文档中的矛盾程度"""
        # 简化实现
        return 0.0

@dataclass
class Evaluation:
    """评估结果"""
    doc_id: str
    relevance_score: float
    confidence: float
    hallucination_risk: float
    is_relevant: bool
    needs_correction: bool
```

### 12.3.3 CRAG纠正策略

根据评估结果，CRAG采取不同的纠正策略：

```python
class CorrectiveRAG:
    """CRAG纠正策略实现"""
    
    def __init__(self, 
                 retriever,
                 llm,
                 web_search_func: Optional[Callable] = None,
                 knowledge_decomposition: bool = True):
        self.retriever = retriever
        self.llm = llm
        self.evaluator = RetrievalEvaluator(llm)
        self.web_search = web_search_func
        self.knowledge_decomposition = knowledge_decomposition
    
    def retrieve_and_correct(self, query: str) -> Dict[str, Any]:
        """检索并纠正"""
        # 第一阶段：检索
        initial_docs = self.retriever.retrieve(query, k=10)
        
        # 第二阶段：评估
        evaluations = self.evaluator.evaluate(query, initial_docs)
        
        # 分析整体检索质量
        relevant_count = sum(1 for e in evaluations if e.is_relevant)
        needs_correction = any(e.needs_correction for e in evaluations)
        
        if relevant_count >= 3 and not needs_correction:
            # 情况1：检索结果质量好，直接生成
            return self._generate_with_docs(query, initial_docs, evaluations)
        
        elif relevant_count > 0 and relevant_count < 3:
            # 情况2：部分相关，使用知识分解重构
            return self._decompose_and_reconstruct(query, initial_docs, evaluations)
        
        else:
            # 情况3：完全不相关，重新检索
            return self._re_retrieve(query)
    
    def _generate_with_docs(self, query: str, 
                             docs: List[Document],
                             evaluations: List[Evaluation]) -> Dict[str, Any]:
        """基于高质量文档生成答案"""
        context = self._format_context(docs, evaluations)
        
        prompt = f"""基于以下信息回答问题。

上下文信息：
{context}

问题：{query}

请给出准确、完整的回答。"""
        
        response = self.llm(prompt)
        
        return {
            'answer': response,
            'sources': [d.metadata.get('source', '') for d in docs],
            'method': 'direct_generation',
            'evaluations': evaluations
        }
    
    def _decompose_and_reconstruct(self, query: str,
                                     docs: List[Document],
                                     evaluations: List[Evaluation]) -> Dict[str, Any]:
        """知识分解和重构"""
        # 分离相关和不相关文档
        relevant_docs = [
            d for d, e in zip(docs, evaluations) if e.is_relevant
        ]
        irrelevant_docs = [
            d for d, e in zip(docs, evaluations) if not e.is_relevant
        ]
        
        # 从相关文档中提取可靠知识
        reliable_knowledge = self._extract_reliable_knowledge(
            query, relevant_docs
        )
        
        # 对不相关文档进行纠错
        corrected_knowledge = self._correct_irrelevant_docs(
            query, irrelevant_docs
        )
        
        # 融合知识
        combined_knowledge = self._fuse_knowledge(
            reliable_knowledge, corrected_knowledge
        )
        
        # 生成答案
        prompt = f"""基于融合后的知识回答问题。

可靠知识：
{reliable_knowledge}

纠错知识：
{corrected_knowledge}

问题：{query}

请整合以上信息给出准确回答。"""
        
        response = self.llm(prompt)
        
        return {
            'answer': response,
            'method': 'decompose_reconstruct',
            'reliable_knowledge': reliable_knowledge,
            'corrected_knowledge': corrected_knowledge
        }
    
    def _re_retrieve(self, query: str) -> Dict[str, Any]:
        """重新检索（使用查询扩展或网络搜索）"""
        # 尝试查询扩展
        expanded_query = self._expand_query(query)
        
        # 重新检索
        new_docs = self.retriever.retrieve(expanded_query, k=10)
        
        # 如果还是没有结果，尝试网络搜索
        if not new_docs and self.web_search:
            web_results = self.web_search(query)
            return {
                'answer': self._generate_with_web_results(query, web_results),
                'method': 'web_search',
                'web_results': web_results
            }
        
        # 使用新检索结果
        evaluations = self.evaluator.evaluate(query, new_docs)
        return self._generate_with_docs(query, new_docs, evaluations)
    
    def _expand_query(self, query: str) -> str:
        """查询扩展"""
        prompt = f"""为以下查询生成一个扩展版本，包含同义词和相关概念。

原始查询: {query}

扩展查询:"""
        
        return self.llm(prompt).strip()
    
    def _extract_reliable_knowledge(self, query: str, 
                                     docs: List[Document]) -> str:
        """提取可靠知识"""
        if not docs:
            return "无可靠知识"
        
        prompt = f"""从以下文档中提取与查询相关的可靠事实。

查询: {query}

文档:
{self._format_simple_docs(docs)}

请只提取确定的事实信息，排除不确定或模糊的内容："""
        
        return self.llm(prompt)
    
    def _correct_irrelevant_docs(self, query: str,
                                   docs: List[Document]) -> str:
        """纠错不相关文档"""
        if not docs:
            return "无不相关文档需要纠错"
        
        prompt = f"""以下文档与查询不相关，请分析它们可能存在的错误。

查询: {query}

文档:
{self._format_simple_docs(docs)}

请指出文档中的错误或无关信息："""
        
        return self.llm(prompt)
    
    def _fuse_knowledge(self, reliable: str, corrected: str) -> str:
        """融合知识"""
        return f"{reliable}\n\n{corrected}"
    
    def _format_context(self, docs: List[Document], 
                         evaluations: List[Evaluation]) -> str:
        """格式化上下文"""
        parts = []
        for i, (doc, eval_) in enumerate(zip(docs, evaluations), 1):
            parts.append(
                f"[文档{i}] (相关性:{eval_.relevance_score:.2f}, "
                f"置信度:{eval_.confidence:.2f})\n"
                f"{doc.page_content[:500]}"
            )
        return "\n\n".join(parts)
    
    def _format_simple_docs(self, docs: List[Document]) -> str:
        """简化文档格式化"""
        return "\n\n".join([
            f"[{i+1}] {d.page_content[:300]}"
            for i, d in enumerate(docs)
        ])
    
    def _generate_with_web_results(self, query: str, 
                                    web_results: List[Dict]) -> str:
        """基于网络搜索结果生成答案"""
        context = "\n\n".join([
            f"[来源{i+1}] {r.get('snippet', '')}"
            for i, r in enumerate(web_results)
        ])
        
        prompt = f"""基于以下网络搜索结果回答问题。

搜索结果：
{context}

问题：{query}

请注意网络信息可能不完全可靠，请审慎回答。"""
        
        return self.llm(prompt)
```

### 12.3.4 CRAG评估决策流程

```python
class CRAGDecisionEngine:
    """CRAG决策引擎"""
    
    def __init__(self, llm, thresholds: Dict[str, float] = None):
        self.llm = llm
        self.thresholds = thresholds or {
            'relevance': 0.7,
            'confidence': 0.6,
            'hallucination': 0.3,
            'consensus': 0.5
        }
    
    def decide_strategy(self, 
                        query: str,
                        documents: List[Document],
                        evaluations: List[Evaluation]) -> str:
        """决定使用哪种策略"""
        # 计算整体质量指标
        metrics = self._compute_aggregate_metrics(evaluations)
        
        # 策略决策逻辑
        if metrics['avg_relevance'] >= self.thresholds['relevance']:
            if metrics['avg_hallucination'] < self.thresholds['hallucination']:
                return 'direct_generation'
            else:
                return 'knowledge_correction'
        
        elif metrics['avg_relevance'] >= 0.3:
            # 部分相关
            if self._check_diversity(documents):
                return 'decompose_reconstruct'
            else:
                return 'knowledge_decomposition'
        
        else:
            # 完全不相关
            if self._has_alternative_sources():
                return 're_retrieve'
            else:
                return 'web_search'
    
    def _compute_aggregate_metrics(self, 
                                    evaluations: List[Evaluation]) -> Dict:
        """计算聚合指标"""
        if not evaluations:
            return {
                'avg_relevance': 0,
                'avg_confidence': 0,
                'avg_hallucination': 0
            }
        
        n = len(evaluations)
        return {
            'avg_relevance': sum(e.relevance_score for e in evaluations) / n,
            'avg_confidence': sum(e.confidence for e in evaluations) / n,
            'avg_hallucination': sum(e.hallucination_risk for e in evaluations) / n
        }
    
    def _check_diversity(self, documents: List[Document]) -> bool:
        """检查文档多样性"""
        if len(documents) < 2:
            return False
        
        # 检查文档来源的多样性
        sources = set(d.metadata.get('source', '') for d in documents)
        return len(sources) >= 2
    
    def _has_alternative_sources(self) -> bool:
        """检查是否有备选检索源"""
        return True  # 简化实现
```

## 12.4 Self-RAG

Self-RAG（Self-Reflective RAG）由Asai等人于2023年提出，它让LLM在生成过程中进行自我反思，评估自身输出是否需要检索以及检索结果的可靠性。与CRAG不同，Self-RAG不仅评估检索结果，还反思生成内容本身。

### 12.4.1 Self-RAG核心机制

Self-RAG引入了三个关键的反思标记（Reflection Token）：

1. **检索标记**（Retrieve Token）：判断是否需要检索
2. **相关标记**（Relevance Token）：判断检索结果是否相关
3. **支持标记**（Support Token）：判断生成内容是否被检索结果支持

这些标记让LLM能够在生成过程中动态决定何时检索、是否使用检索结果，以及是否需要修正生成内容。

### 12.4.2 Self-RAG实现

```python
class SelfRAG:
    """Self-RAG实现"""
    
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever
    
    def generate(self, query: str, max_retrievals: int = 3) -> str:
        """Self-RAG生成"""
        segments = []
        current_input = query
        
        for iteration in range(max_retrievals):
            # 1. 判断是否需要检索
            retrieval_decision = self._decide_retrieval(current_input)
            
            if retrieval_decision['needs_retrieval']:
                # 2. 执行检索
                docs = self.retriever.retrieve(
                    retrieval_decision.get('refined_query', query),
                    k=5
                )
                
                # 3. 评估检索结果相关性
                relevant_docs = self._filter_relevant(query, docs)
                
                if not relevant_docs:
                    continue
                
                # 4. 基于检索结果生成
                segment = self._generate_with_reflection(
                    query, relevant_docs
                )
                
                # 5. 评估生成结果
                support_score = self._evaluate_support(
                    segment, relevant_docs
                )
                
                if support_score < 0.5:
                    # 支持度不够，继续检索
                    current_input = f"{query}\n之前的尝试不够充分，需要更多信息。"
                    continue
                
                segments.append(segment)
            else:
                # 不需要检索，直接生成
                segment = self.llm(current_input)
                segments.append(segment)
                break
        
        # 融合所有段落
        return self._merge_segments(segments, query)
    
    def _decide_retrieval(self, query: str) -> Dict:
        """判断是否需要检索"""
        prompt = f"""判断以下查询是否需要检索外部信息来回答。

查询: {query}

请输出JSON格式：
{{
    "needs_retrieval": true/false,
    "reason": "原因",
    "refined_query": "如果需要检索，优化后的查询"
}}"""
        
        response = self.llm(prompt)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"needs_retrieval": True, "refined_query": query}
    
    def _filter_relevant(self, query: str, 
                          docs: List[Document]) -> List[Document]:
        """过滤相关文档"""
        relevant = []
        
        for doc in docs:
            prompt = f"""判断以下文档是否与查询相关。

查询: {query}
文档: {doc.page_content[:300]}

只回答"相关"或"不相关"："""
            
            response = self.llm(prompt).strip()
            if '相关' in response:
                relevant.append(doc)
        
        return relevant
    
    def _generate_with_reflection(self, query: str,
                                   docs: List[Document]) -> str:
        """带反思的生成"""
        context = "\n\n".join([
            f"[文档{i+1}] {d.page_content}"
            for i, d in enumerate(docs)
        ])
        
        prompt = f"""基于以下信息回答问题。

参考文档：
{context}

问题：{query}

生成答案后，请反思：
1. 答案是否完全基于参考文档？
2. 是否有需要补充的信息？
3. 答案的确定性如何？"""
        
        return self.llm(prompt)
    
    def _evaluate_support(self, segment: str, docs: List[Document]) -> float:
        """评估生成内容被支持的程度"""
        prompt = f"""评估以下回答被参考文档支持的程度。

回答: {segment[:500]}

参考文档:
{"\n".join([d.page_content[:300] for d in docs])}

请给出0-1之间的支持度评分，只输出数字："""
        
        response = self.llm(prompt)
        try:
            return float(response.strip())
        except ValueError:
            return 0.5
    
    def _merge_segments(self, segments: List[str], query: str) -> str:
        """融合多个生成段落"""
        if len(segments) == 1:
            return segments[0]
        
        prompt = f"""将以下多个回答段落融合为一个完整、一致的答案。

问题: {query}

各段落:
{"\n\n".join([f"[段落{i+1}] {s}" for i, s in enumerate(segments)])}

请输出融合后的完整答案："""
        
        return self.llm(prompt)
```

### 12.4.3 Self-RAG的反思机制

Self-RAG的反思机制可以进一步细化为多个反思维度：

```python
class SelfReflection:
    """自我反思机制"""
    
    REFLECTION_PROMPTS = {
        'completeness': """评估回答的完整性。
        
回答: {answer}
问题: {query}

回答是否全面覆盖了问题的各个方面？
是否遗漏了重要信息？
评分(0-1):""",
        
        'accuracy': """评估回答的准确性。
        
回答: {answer}
参考文档: {context}

回答中的每个事实是否都有文档支持？
是否有任何事实与文档矛盾？
评分(0-1):""",
        
        'relevance': """评估回答的相关性。
        
回答: {answer}
问题: {query}

回答是否直接回应了问题？
是否包含无关信息？
评分(0-1):""",
        
        'clarity': """评估回答的清晰度。
        
回答: {answer}

表达是否清晰、有条理？
是否有歧义或模糊表述？
评分(0-1):"""
    }
    
    def __init__(self, llm):
        self.llm = llm
    
    def reflect(self, answer: str, query: str, 
                context: str = "") -> Dict[str, float]:
        """多维度反思"""
        scores = {}
        
        for dim, prompt_template in self.REFLECTION_PROMPTS.items():
            prompt = prompt_template.format(
                answer=answer,
                query=query,
                context=context
            )
            response = self.llm(prompt)
            
            try:
                score = float(response.strip())
                scores[dim] = max(0.0, min(1.0, score))
            except ValueError:
                scores[dim] = 0.5
        
        return scores
    
    def needs_revision(self, scores: Dict[str, float], 
                       thresholds: Dict[str, float] = None) -> bool:
        """判断是否需要修正"""
        if thresholds is None:
            thresholds = {
                'completeness': 0.7,
                'accuracy': 0.8,
                'relevance': 0.7,
                'clarity': 0.6
            }
        
        return any(
            scores.get(dim, 0) < threshold
            for dim, threshold in thresholds.items()
        )
```

## 12.5 多智能体协作

多智能体协作是Agentic RAG的高级形式，通过多个专业化的智能体协同工作来解决复杂任务。每个智能体负责特定的功能领域，通过消息传递和任务协调实现1+1>2的效果。

### 12.5.1 多智能体架构

典型的多智能体RAG系统包含以下角色：

| 智能体角色 | 职责 | 工具 |
|-----------|------|------|
| 主管智能体（Supervisor） | 任务规划、协调、最终决策 | 任务分配、结果评估 |
| 检索智能体（Retriever） | 文档检索和筛选 | 向量搜索、关键词搜索 |
| 推理智能体（Reasoner） | 逻辑推理和分析 | 思维链、数学计算 |
| 验证智能体（Verifier） | 事实核查和质量验证 | 交叉验证、来源检查 |
| 写作智能体（Writer） | 答案组织和润色 | 文本生成、格式化 |

### 12.5.2 多智能体协作框架实现

```python
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
import asyncio

class AgentRole(Enum):
    """智能体角色枚举"""
    SUPERVISOR = "supervisor"
    RETRIEVER = "retriever"
    REASONER = "reasoner"
    VERIFIER = "verifier"
    WRITER = "writer"

@dataclass
class Message:
    """智能体间消息"""
    sender: str
    receiver: str
    content: Any
    msg_type: str = "text"
    metadata: Dict = field(default_factory=dict)

@dataclass
class Task:
    """任务定义"""
    task_id: str
    description: str
    assigned_to: str
    status: str = "pending"
    result: Any = None
    dependencies: List[str] = field(default_factory=list)

class BaseAgent:
    """智能体基类"""
    
    def __init__(self, name: str, role: AgentRole, llm):
        self.name = name
        self.role = role
        self.llm = llm
        self.mailbox: List[Message] = []
        self.tasks: List[Task] = []
    
    async def send_message(self, receiver: 'BaseAgent', 
                           content: Any, msg_type: str = "text"):
        """发送消息"""
        message = Message(
            sender=self.name,
            receiver=receiver.name,
            content=content,
            msg_type=msg_type
        )
        await receiver.receive_message(message)
    
    async def receive_message(self, message: Message):
        """接收消息"""
        self.mailbox.append(message)
    
    async def process_tasks(self) -> List[Message]:
        """处理任务"""
        raise NotImplementedError

class SupervisorAgent(BaseAgent):
    """主管智能体"""
    
    def __init__(self, name: str, llm):
        super().__init__(name, AgentRole.SUPERVISOR, llm)
        self.sub_agents: Dict[str, BaseAgent] = {}
        self.workflow = []
    
    def register_agent(self, agent: BaseAgent):
        """注册子智能体"""
        self.sub_agents[agent.name] = agent
    
    async def plan_task(self, query: str) -> List[Task]:
        """规划任务"""
        prompt = f"""规划完成以下查询所需的步骤。

查询: {query}

可用智能体:
{chr(10).join([f"- {name}: {agent.role.value}" for name, agent in self.sub_agents.items()])}

请输出任务列表（JSON格式）：
[
    {{
        "task_id": "1",
        "description": "任务描述",
        "assigned_to": "智能体名称",
        "dependencies": []
    }}
]"""
        
        response = self.llm(prompt)
        try:
            tasks_data = json.loads(response)
            return [
                Task(**task_data) for task_data in tasks_data
            ]
        except json.JSONDecodeError:
            return [
                Task(task_id="1", description=query, 
                     assigned_to=list(self.sub_agents.keys())[0])
            ]
    
    async def execute_workflow(self, query: str) -> str:
        """执行工作流"""
        # 1. 规划任务
        tasks = await self.plan_task(query)
        self.tasks = tasks
        
        # 2. 按依赖关系执行任务
        completed = set()
        results = {}
        
        while len(completed) < len(tasks):
            for task in tasks:
                if task.task_id in completed:
                    continue
                
                # 检查依赖是否满足
                deps_met = all(dep in completed for dep in task.dependencies)
                if not deps_met:
                    continue
                
                # 分配任务
                agent = self.sub_agents.get(task.assigned_to)
                if agent is None:
                    continue
                
                # 执行任务
                task.status = "in_progress"
                context = {
                    'query': query,
                    'task': task,
                    'previous_results': results
                }
                
                result = await agent.process_task(task, context)
                task.result = result
                task.status = "completed"
                completed.add(task.task_id)
                results[task.task_id] = result
        
        # 3. 整合结果
        final_answer = await self._synthesize_results(query, results)
        return final_answer
    
    async def _synthesize_results(self, query: str, 
                                   results: Dict) -> str:
        """综合所有结果"""
        prompt = f"""综合以下所有任务结果回答用户问题。

问题: {query}

任务结果:
{json.dumps(results, ensure_ascii=False, indent=2)}

请给出最终答案："""
        
        return self.llm(prompt)

class RetrieverAgent(BaseAgent):
    """检索智能体"""
    
    def __init__(self, name: str, llm, vector_store, kg_client=None):
        super().__init__(name, AgentRole.RETRIEVER, llm)
        self.vector_store = vector_store
        self.kg_client = kg_client
    
    async def process_task(self, task: Task, context: Dict) -> Dict:
        """处理检索任务"""
        query = task.description
        results = {
            'vector_results': [],
            'kg_results': [],
            'combined': []
        }
        
        # 向量检索
        vector_docs = self.vector_store.similarity_search(query, k=5)
        results['vector_results'] = [
            {
                'content': doc.page_content[:500],
                'source': doc.metadata.get('source', ''),
                'score': doc.metadata.get('score', 0)
            }
            for doc in vector_docs
        ]
        
        # 知识图谱检索（如果可用）
        if self.kg_client:
            try:
                kg_results = self.kg_client.query(query)
                results['kg_results'] = kg_results
            except Exception as e:
                results['kg_results'] = {'error': str(e)}
        
        # 合并结果
        results['combined'] = self._merge_results(results)
        
        return results
    
    def _merge_results(self, results: Dict) -> List[Dict]:
        """合并多源检索结果"""
        all_results = []
        seen_sources = set()
        
        for doc in results['vector_results']:
            if doc['source'] not in seen_sources:
                seen_sources.add(doc['source'])
                all_results.append(doc)
        
        return sorted(
            all_results,
            key=lambda x: x.get('score', 0),
            reverse=True
        )

class VerifierAgent(BaseAgent):
    """验证智能体"""
    
    def __init__(self, name: str, llm):
        super().__init__(name, AgentRole.VERIFIER, llm)
    
    async def process_task(self, task: Task, context: Dict) -> Dict:
        """验证结果"""
        answer = task.description
        query = context.get('query', '')
        retrieval_results = context.get('previous_results', {})
        
        verification = {
            'factual_accuracy': await self._verify_facts(answer, retrieval_results),
            'completeness': await self._check_completeness(answer, query),
            'consistency': await self._check_consistency(answer),
            'hallucination_detection': await self._detect_hallucinations(
                answer, retrieval_results
            )
        }
        
        # 总体判断
        all_scores = [
            v['score'] for v in verification.values()
            if isinstance(v, dict) and 'score' in v
        ]
        verification['overall_score'] = (
            sum(all_scores) / len(all_scores) if all_scores else 0
        )
        
        return verification
    
    async def _verify_facts(self, answer: str, 
                             retrieval_results: Dict) -> Dict:
        """事实准确性验证"""
        # 提取检索结果中的上下文
        context = ""
        for task_id, result in retrieval_results.items():
            if isinstance(result, dict):
                combined = result.get('combined', [])
                context += "\n".join([
                    r.get('content', '') for r in combined
                ])
        
        prompt = f"""验证以下回答中的事实准确性。

回答: {answer[:1000]}

参考文档:
{context[:2000]}

请输出JSON：
{{
    "score": 0.0-1.0,
    "supported_claims": ["被支持的陈述"],
    "unsupported_claims": ["不被支持的陈述"],
    "corrections": ["需要的修正"]
}}"""
        
        response = self.llm(prompt)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"score": 0.5, "errors": ["验证失败"]}
    
    async def _check_completeness(self, answer: str, query: str) -> Dict:
        """完整性检查"""
        prompt = f"""检查回答是否完整覆盖了问题。

问题: {query}
回答: {answer[:1000]}

请输出JSON：
{{
    "score": 0.0-1.0,
    "covered_aspects": ["已覆盖的方面"],
    "missing_aspects": ["缺失的方面"]
}}"""
        
        response = self.llm(prompt)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"score": 0.5}
    
    async def _check_consistency(self, answer: str) -> Dict:
        """一致性检查"""
        prompt = f"""检查回答内部是否一致，是否存在矛盾。

回答: {answer[:1000]}

请输出JSON：
{{
    "score": 0.0-1.0,
    "contradictions": ["发现的矛盾"],
    "is_consistent": true/false
}}"""
        
        response = self.llm(prompt)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"score": 0.8, "is_consistent": True}
    
    async def _detect_hallucinations(self, answer: str,
                                      retrieval_results: Dict) -> Dict:
        """幻觉检测"""
        context = ""
        for task_id, result in retrieval_results.items():
            if isinstance(result, dict):
                combined = result.get('combined', [])
                context += "\n".join([
                    r.get('content', '') for r in combined
                ])
        
        prompt = f"""检测回答中的幻觉（没有文档支持的内容）。

回答: {answer[:1000]}

参考文档:
{context[:2000]}

请输出JSON：
{{
    "score": 0.0-1.0（分数越低表示幻觉越多）,
    "hallucinated_content": ["幻觉内容"],
    "supported_content": ["有支持的内容"]
}}"""
        
        response = self.llm(prompt)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"score": 0.5}

class MultiAgentOrchestrator:
    """多智能体编排器"""
    
    def __init__(self, llm, vector_store, kg_client=None):
        self.llm = llm
        
        # 创建智能体
        self.supervisor = SupervisorAgent("supervisor", llm)
        self.retriever = RetrieverAgent("retriever", llm, vector_store, kg_client)
        self.reasoner = BaseAgent("reasoner", AgentRole.REASONER, llm)
        self.verifier = VerifierAgent("verifier", llm)
        self.writer = BaseAgent("writer", AgentRole.WRITER, llm)
        
        # 注册智能体
        self.supervisor.register_agent(self.retriever)
        self.supervisor.register_agent(self.reasoner)
        self.supervisor.register_agent(self.verifier)
        self.supervisor.register_agent(self.writer)
    
    async def run(self, query: str) -> Dict:
        """运行多智能体系统"""
        # 主管规划并执行工作流
        answer = await self.supervisor.execute_workflow(query)
        
        return {
            'query': query,
            'answer': answer,
            'workflow': self.supervisor.workflow
        }
```

## 12.6 工具使用

工具使用是智能体RAG区别于传统RAG的关键能力之一。智能体可以调用各种外部工具来获取信息、执行计算或操作外部系统。

### 12.6.1 工具定义与注册

```python
from typing import Callable, Dict, Any, Optional
from pydantic import BaseModel, Field
import inspect

class Tool(BaseModel):
    """工具定义"""
    name: str = Field(description="工具名称")
    description: str = Field(description="工具功能描述")
    func: Callable = Field(description="工具函数")
    parameters: Dict = Field(description="参数schema", default_factory=dict)
    examples: List[str] = Field(description="使用示例", default_factory=list)
    
    class Config:
        arbitrary_types_allowed = True

class ToolRegistry:
    """工具注册中心"""
    
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
    
    def register(self, tool: Tool):
        """注册工具"""
        self._tools[tool.name] = tool
    
    def register_func(self, name: str, description: str, 
                      func: Callable, examples: List[str] = None):
        """注册函数为工具"""
        # 自动提取参数信息
        sig = inspect.signature(func)
        parameters = {}
        for param_name, param in sig.parameters.items():
            parameters[param_name] = {
                'type': str(param.annotation) if param.annotation != inspect.Parameter.empty else 'any',
                'required': param.default == inspect.Parameter.empty
            }
        
        tool = Tool(
            name=name,
            description=description,
            func=func,
            parameters=parameters,
            examples=examples or []
        )
        self.register(tool)
    
    def get(self, name: str) -> Optional[Tool]:
        """获取工具"""
        return self._tools.get(name)
    
    def list_tools(self) -> List[Dict]:
        """列出所有工具信息"""
        return [
            {
                'name': tool.name,
                'description': tool.description,
                'parameters': tool.parameters,
                'examples': tool.examples
            }
            for tool in self._tools.values()
        ]
    
    def execute(self, name: str, **kwargs) -> Any:
        """执行工具"""
        tool = self.get(name)
        if tool is None:
            raise ValueError(f"Unknown tool: {name}")
        
        return tool.func(**kwargs)

# 定义常用工具
def web_search(query: str, top_k: int = 5) -> List[Dict]:
    """网络搜索工具"""
    # 实际实现会调用搜索API
    return [{'title': '搜索结果', 'snippet': '...'}]

def python_repl(code: str) -> str:
    """Python代码执行工具"""
    try:
        local_vars = {}
        exec(code, {"__builtins__": __builtins__}, local_vars)
        return str(local_vars.get('result', '执行成功'))
    except Exception as e:
        return f"执行错误: {str(e)}"

def calculator(expression: str) -> str:
    """计算器工具"""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"计算错误: {str(e)}"

def current_datetime(format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """获取当前日期时间"""
    from datetime import datetime
    return datetime.now().strftime(format)

# 注册工具
registry = ToolRegistry()
registry.register_func("web_search", "搜索网络获取最新信息", web_search,
                       examples=["web_search(query='2024年奥运会')"])
registry.register_func("python_repl", "执行Python代码进行数据分析", python_repl,
                       examples=["python_repl(code='result = sum([1,2,3,4,5])')"])
registry.register_func("calculator", "执行数学计算", calculator,
                       examples=["calculator(expression='2 + 3 * 4')"])
registry.register_func("current_datetime", "获取当前日期和时间", current_datetime)
```

### 12.6.2 Function Calling集成

现代LLM支持Function Calling（函数调用），可以直接将工具定义为函数并让LLM自动选择调用：

```python
class FunctionCallingAgent:
    """Function Calling智能体"""
    
    def __init__(self, llm_client, tools: List[Dict]):
        self.client = llm_client
        self.tools = tools
    
    def run(self, query: str, max_turns: int = 5) -> str:
        """运行Function Calling智能体"""
        messages = [{"role": "user", "content": query}]
        
        for turn in range(max_turns):
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                tools=self.tools,
                tool_choice="auto"
            )
            
            message = response.choices[0].message
            
            if not message.tool_calls:
                # 没有工具调用，返回最终回答
                return message.content
            
            # 执行工具调用
            messages.append(message)
            
            for tool_call in message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                result = registry.execute(function_name, **function_args)
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result)
                })
        
        return "超过最大交互轮数"

# 工具定义（OpenAI格式）
OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "vector_search",
            "description": "通过语义相似度检索相关文档",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索查询"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回文档数量",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "kg_query",
            "description": "查询知识图谱获取结构化信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "自然语言查询或Cypher查询"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "执行数学计算",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式"
                    }
                },
                "required": ["expression"]
            }
        }
    }
]
```

## 12.7 多步骤推理

多步骤推理是智能体RAG处理复杂问题的关键能力。通过将复杂问题分解为多个子问题，逐步求解并综合答案，可以显著提高回答质量。

### 12.7.1 思维链（Chain-of-Thought）

思维链通过让LLM展示中间推理步骤来提高复杂问题的解决能力：

```python
class ChainOfThought:
    """思维链推理"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def solve(self, problem: str, num_steps: int = 3) -> Dict:
        """分步解决问题"""
        prompt = f"""请逐步解决以下问题，每一步都要展示推理过程。

问题: {problem}

请按以下格式回答：
步骤1: [推理过程]
步骤2: [推理过程]
...
最终答案: [完整答案]"""
        
        response = self.llm(prompt)
        
        # 解析步骤
        steps = self._parse_steps(response)
        final_answer = self._extract_answer(response)
        
        return {
            'steps': steps,
            'final_answer': final_answer,
            'full_response': response
        }
    
    def _parse_steps(self, response: str) -> List[Dict]:
        """解析推理步骤"""
        steps = []
        for line in response.split('\n'):
            if line.startswith('步骤') or line.startswith('Step'):
                steps.append({'content': line.strip()})
        return steps
    
    def _extract_answer(self, response: str) -> str:
        """提取最终答案"""
        lines = response.split('\n')
        for line in lines:
            if '最终答案' in line or 'Final Answer' in line:
                return line.split(':', 1)[1].strip() if ':' in line else line
        return response  # 返回全部作为答案
```

### 12.7.2 问题分解

将复杂问题分解为可独立解决的子问题：

```python
class QueryDecomposer:
    """查询分解器"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def decompose(self, query: str, max_sub_queries: int = 5) -> List[Dict]:
        """将复杂查询分解为子查询"""
        prompt = f"""将以下复杂问题分解为多个可以独立回答的子问题。

问题: {query}

要求：
1. 每个子问题应该独立且具体
2. 子问题之间应该有关联
3. 所有子问题合起来应覆盖原问题的所有方面

请输出JSON格式：
[
    {{
        "sub_query": "子问题1",
        "type": "factual/analytical/comparative",
        "dependencies": [],
        "expected_info": "期望获取的信息类型"
    }}
]"""
        
        response = self.llm(prompt)
        try:
            sub_queries = json.loads(response)
            return sub_queries[:max_sub_queries]
        except json.JSONDecodeError:
            return [{"sub_query": query, "type": "factual", "dependencies": []}]
    
    def aggregate_answers(self, query: str, 
                          sub_answers: Dict[str, str]) -> str:
        """聚合子答案"""
        context = "\n\n".join([
            f"子问题: {q}\n答案: {a}"
            for q, a in sub_answers.items()
        ])
        
        prompt = f"""综合以下子问题的答案来回答原始问题。

原始问题: {query}

子问题和答案：
{context}

请给出综合后的完整答案："""
        
        return self.llm(prompt)
```

### 12.7.3 自适应检索

根据当前检索结果的质量决定是否需要进一步检索：

```python
class AdaptiveRetriever:
    """自适应检索器"""
    
    def __init__(self, base_retriever, llm, max_iterations: int = 3):
        self.retriever = base_retriever
        self.llm = llm
        self.max_iterations = max_iterations
    
    def retrieve(self, query: str) -> List[Document]:
        """自适应检索"""
        all_docs = []
        current_query = query
        iteration = 0
        
        while iteration < self.max_iterations:
            # 检索
            docs = self.retriever.retrieve(current_query, k=5)
            all_docs.extend(docs)
            
            # 评估是否足够
            if self._is_sufficient(query, all_docs):
                break
            
            # 生成新的查询
            current_query = self._generate_follow_up(query, all_docs)
            iteration += 1
        
        # 去重
        return self._deduplicate(all_docs)
    
    def _is_sufficient(self, query: str, docs: List[Document]) -> bool:
        """判断检索结果是否足够"""
        context = "\n".join([d.page_content[:200] for d in docs])
        
        prompt = f"""判断以下检索结果是否足够回答用户问题。

问题: {query}
检索结果:
{context[:1500]}

如果信息充分回答"是"，否则回答"否"："""
        
        response = self.llm(prompt)
        return '是' in response or 'Yes' in response
    
    def _generate_follow_up(self, query: str, 
                             docs: List[Document]) -> str:
        """生成后续查询"""
        context = "\n".join([d.page_content[:200] for d in docs[-3:]])
        
        prompt = f"""基于当前检索结果，生成一个后续查询以获取更多信息。

原始问题: {query}
当前已获取的信息:
{context}

还缺少什么信息？生成新的查询："""
        
        return self.llm(prompt).strip()
    
    def _deduplicate(self, docs: List[Document]) -> List[Document]:
        """文档去重"""
        seen_content = set()
        unique_docs = []
        
        for doc in docs:
            content_hash = hash(doc.page_content[:200])
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_docs.append(doc)
        
        return unique_docs
```

## 12.8 LangGraph实现模式

LangGraph是一个用于构建有状态、多步骤LLM应用的框架，特别适合实现智能体RAG的复杂工作流。

### 12.8.1 LangGraph核心概念

LangGraph的核心概念包括：

1. **State（状态）**：整个工作流的共享状态
2. **Node（节点）**：工作流中的处理步骤
3. **Edge（边）**：节点之间的连接和条件路由
4. **Graph（图）**：节点和边的集合，构成完整的工作流

### 12.8.2 Agentic RAG的LangGraph实现

```python
from typing import TypedDict, List, Dict, Any, Optional, Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.documents import Document

# 定义状态类型
class AgentState(TypedDict):
    """智能体状态"""
    query: str
    messages: List[Dict]
    documents: List[Document]
    intermediate_answers: List[str]
    current_step: int
    max_steps: int
    final_answer: Optional[str]
    needs_retrieval: bool
    needs_correction: bool
    error: Optional[str]

class LangGraphAgenticRAG:
    """LangGraph智能体RAG"""
    
    def __init__(self, llm, retriever, tools: Dict[str, Callable] = None):
        self.llm = llm
        self.retriever = retriever
        self.tools = tools or {}
        
        # 构建图
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """构建工作流图"""
        workflow = StateGraph(AgentState)
        
        # 添加节点
        workflow.add_node("analyze_query", self._analyze_query)
        workflow.add_node("retrieve", self._retrieve)
        workflow.add_node("evaluate_docs", self._evaluate_documents)
        workflow.add_node("generate_answer", self._generate_answer)
        workflow.add_node("verify_answer", self._verify_answer)
        workflow.add_node("correct_answer", self._correct_answer)
        
        # 设置入口点
        workflow.set_entry_point("analyze_query")
        
        # 添加边
        workflow.add_edge("analyze_query", "retrieve")
        workflow.add_edge("retrieve", "evaluate_docs")
        
        # 条件边：根据文档质量决定下一步
        workflow.add_conditional_edges(
            "evaluate_docs",
            self._route_after_evaluation,
            {
                "generate": "generate_answer",
                "retry": "retrieve"
            }
        )
        
        # 生成后的验证
        workflow.add_edge("generate_answer", "verify_answer")
        
        # 条件边：根据验证结果决定是否需要修正
        workflow.add_conditional_edges(
            "verify_answer",
            self._route_after_verification,
            {
                "accept": END,
                "correct": "correct_answer",
                "retry": "retrieve"
            }
        )
        
        # 修正后结束
        workflow.add_edge("correct_answer", END)
        
        # 编译图
        return workflow.compile(
            checkpointer=MemorySaver()
        )
    
    def _analyze_query(self, state: AgentState) -> Dict:
        """分析查询节点"""
        query = state['query']
        
        prompt = f"""分析用户查询，判断：
1. 是否需要检索外部信息
2. 查询的类型（事实性/分析性/指令性）
3. 关键实体和概念

查询: {query}

输出JSON：
{{
    "needs_retrieval": true/false,
    "query_type": "factual/analytical/instructional",
    "key_concepts": ["概念1", "概念2"],
    "refined_query": "优化后的查询"
}}"""
        
        response = self.llm(prompt)
        try:
            analysis = json.loads(response)
        except json.JSONDecodeError:
            analysis = {
                "needs_retrieval": True,
                "query_type": "factual",
                "key_concepts": [],
                "refined_query": query
            }
        
        return {
            "needs_retrieval": analysis["needs_retrieval"],
            "current_step": state.get('current_step', 0) + 1,
            "messages": state.get('messages', []) + [
                {"role": "assistant", "content": f"分析结果: {json.dumps(analysis, ensure_ascii=False)}"}
            ]
        }
    
    def _retrieve(self, state: AgentState) -> Dict:
        """检索节点"""
        query = state['query']
        
        # 多路检索
        vector_docs = self.retriever.retrieve(query, k=5)
        
        # 如果有知识图谱，也进行图检索
        kg_docs = []
        if 'kg_query' in self.tools:
            try:
                kg_result = self.tools['kg_query'](query)
                if kg_result:
                    kg_docs = [Document(page_content=str(kg_result))]
            except:
                pass
        
        # 合并文档
        all_docs = vector_docs + kg_docs
        
        return {
            "documents": all_docs,
            "current_step": state.get('current_step', 0) + 1
        }
    
    def _evaluate_documents(self, state: AgentState) -> Dict:
        """评估文档节点"""
        docs = state.get('documents', [])
        
        if not docs:
            return {"needs_correction": True, "error": "未检索到文档"}
        
        # 评估每个文档的相关性
        prompt = f"""评估以下文档是否与查询相关。

查询: {state['query']}

文档列表:
{chr(10).join([f"[文档{i+1}] {d.page_content[:200]}" for i, d in enumerate(docs)])}

输出JSON：
{{
    "relevant_indices": [相关文档索引],
    "overall_quality": "high/medium/low",
    "needs_more_info": true/false
}}"""
        
        response = self.llm(prompt)
        try:
            evaluation = json.loads(response)
        except json.JSONDecodeError:
            evaluation = {
                "relevant_indices": list(range(len(docs))),
                "overall_quality": "medium",
                "needs_more_info": False
            }
        
        return {
            "needs_correction": evaluation.get('overall_quality') == 'low',
            "messages": state.get('messages', []) + [
                {"role": "assistant", "content": f"文档评估: {json.dumps(evaluation, ensure_ascii=False)}"}
            ]
        }
    
    def _generate_answer(self, state: AgentState) -> Dict:
        """生成答案节点"""
        docs = state.get('documents', [])
        query = state['query']
        
        context = "\n\n".join([
            f"[文档{i+1}] {d.page_content}"
            for i, d in enumerate(docs)
        ])
        
        prompt = f"""基于以下检索结果回答问题。

检索结果：
{context}

问题：{query}

要求：
1. 答案必须基于检索结果
2. 标注信息来源
3. 如果信息不足，明确指出
4. 保持客观准确"""
        
        answer = self.llm(prompt)
        
        return {
            "final_answer": answer,
            "current_step": state.get('current_step', 0) + 1
        }
    
    def _verify_answer(self, state: AgentState) -> Dict:
        """验证答案节点"""
        answer = state.get('final_answer', '')
        docs = state.get('documents', [])
        
        context = "\n".join([d.page_content[:500] for d in docs])
        
        prompt = f"""验证以下回答是否准确。

回答: {answer[:1000]}

参考文档:
{context[:2000]}

输出JSON：
{{
    "is_accurate": true/false,
    "hallucination_detected": true/false,
    "missing_info": ["缺失的信息"],
    "confidence": 0.0-1.0
}}"""
        
        response = self.llm(prompt)
        try:
            verification = json.loads(response)
        except json.JSONDecodeError:
            verification = {
                "is_accurate": True,
                "hallucination_detected": False,
                "missing_info": [],
                "confidence": 0.8
            }
        
        return {
            "needs_correction": not verification.get('is_accurate', True),
            "messages": state.get('messages', []) + [
                {"role": "assistant", "content": f"验证结果: {json.dumps(verification, ensure_ascii=False)}"}
            ]
        }
    
    def _correct_answer(self, state: AgentState) -> Dict:
        """修正答案节点"""
        original_answer = state.get('final_answer', '')
        docs = state.get('documents', [])
        
        context = "\n\n".join([d.page_content for d in docs])
        
        prompt = f"""修正以下回答中的错误。

原始回答: {original_answer}

正确参考信息:
{context}

请输出修正后的准确回答："""
        
        corrected_answer = self.llm(prompt)
        
        return {
            "final_answer": corrected_answer,
            "needs_correction": False
        }
    
    def _route_after_evaluation(self, state: AgentState) -> str:
        """评估后的路由"""
        if state.get('needs_correction', False):
            return "retry"
        return "generate"
    
    def _route_after_verification(self, state: AgentState) -> str:
        """验证后的路由"""
        if state.get('needs_correction', False):
            if state.get('current_step', 0) > 3:
                return "correct"
            return "retry"
        return "accept"
    
    def run(self, query: str, max_steps: int = 5) -> str:
        """运行Agentic RAG"""
        initial_state = AgentState(
            query=query,
            messages=[],
            documents=[],
            intermediate_answers=[],
            current_step=0,
            max_steps=max_steps,
            final_answer=None,
            needs_retrieval=True,
            needs_correction=False,
            error=None
        )
        
        # 配置
        config = {"configurable": {"thread_id": "1"}}
        
        # 执行工作流
        result = self.graph.invoke(initial_state, config)
        
        return result.get('final_answer', '生成失败')
```

### 12.8.3 带循环的Agent模式

```python
class LoopingAgentGraph:
    """带循环的智能体图"""
    
    def __init__(self, llm, tools: Dict[str, Callable]):
        self.llm = llm
        self.tools = tools
        self.graph = self._build_looping_graph()
    
    def _build_looping_graph(self) -> StateGraph:
        """构建带循环的图"""
        workflow = StateGraph(AgentState)
        
        # 节点
        workflow.add_node("think", self._think)
        workflow.add_node("act", self._act)
        
        workflow.set_entry_point("think")
        
        # 循环边：思考 -> 行动 -> 思考（直到条件满足）
        workflow.add_edge("think", "act")
        workflow.add_conditional_edges(
            "act",
            self._should_continue,
            {
                "continue": "think",
                "end": END
            }
        )
        
        return workflow.compile()
    
    def _think(self, state: AgentState) -> Dict:
        """思考节点"""
        query = state['query']
        previous_answers = state.get('intermediate_answers', [])
        
        prompt = f"""分析当前状态，决定下一步行动。

查询: {query}
已完成步骤: {len(previous_answers)}
之前的结果: {chr(10).join(previous_answers[-3:]) if previous_answers else "无"}

输出JSON：
{{
    "thought": "当前推理",
    "next_action": "要执行的行动",
    "action_input": "行动参数",
    "is_final": true/false
}}"""
        
        response = self.llm(prompt)
        try:
            decision = json.loads(response)
        except json.JSONDecodeError:
            decision = {
                "thought": "直接回答",
                "next_action": "answer",
                "action_input": "",
                "is_final": True
            }
        
        return {
            "intermediate_answers": previous_answers + [decision.get('thought', '')],
            "current_step": state.get('current_step', 0) + 1,
            "needs_retrieval": decision.get('next_action') == 'retrieve'
        }
    
    def _act(self, state: AgentState) -> Dict:
        """行动节点"""
        # 简化实现：直接检索并生成
        docs = []  # 实际应调用检索
        answer = "这是基于检索的答案"
        
        return {
            "final_answer": answer,
            "current_step": state.get('current_step', 0) + 1
        }
    
    def _should_continue(self, state: AgentState) -> str:
        """判断是否继续循环"""
        if state.get('final_answer') is not None:
            return "end"
        if state.get('current_step', 0) >= state.get('max_steps', 10):
            return "end"
        return "continue"
```

## 12.9 自我纠正机制

自我纠正是智能体RAG保证输出质量的重要手段。系统能够识别自己的错误并主动修正。

### 12.9.1 错误检测

```python
class ErrorDetector:
    """错误检测器"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def detect_errors(self, answer: str, query: str, 
                       context: str = "") -> Dict:
        """检测回答中的错误"""
        errors = {
            'factual_errors': self._check_factual_errors(answer, context),
            'logical_errors': self._check_logical_errors(answer),
            'relevance_errors': self._check_relevance(answer, query),
            'completeness_errors': self._check_completeness(answer, query)
        }
        
        errors['has_errors'] = any(
            e['has_error'] for e in errors.values()
        )
        
        return errors
    
    def _check_factual_errors(self, answer: str, context: str) -> Dict:
        """检查事实错误"""
        if not context:
            return {'has_error': False, 'details': []}
        
        prompt = f"""检查回答中的事实性错误。

回答: {answer[:1000]}

参考上下文:
{context[:2000]}

列出所有与上下文矛盾的事实陈述：
输出JSON格式的列表，每项包含"statement"和"correction"。"""
        
        response = self.llm(prompt)
        try:
            errors = json.loads(response)
            return {
                'has_error': len(errors) > 0,
                'details': errors
            }
        except json.JSONDecodeError:
            return {'has_error': False, 'details': []}
    
    def _check_logical_errors(self, answer: str) -> Dict:
        """检查逻辑错误"""
        prompt = f"""检查回答中的逻辑错误。

回答: {answer[:1000]}

检查：
1. 是否存在自相矛盾
2. 推理是否合理
3. 因果关系是否正确

输出JSON格式。"""
        
        response = self.llm(prompt)
        try:
            result = json.loads(response)
            return result
        except json.JSONDecodeError:
            return {'has_error': False, 'details': []}
    
    def _check_relevance(self, answer: str, query: str) -> Dict:
        """检查相关性"""
        prompt = f"""判断回答是否直接回应了问题。

问题: {query}
回答: {answer[:500]}

输出JSON：{{"has_error": true/false, "details": "说明"}}"""
        
        response = self.llm(prompt)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {'has_error': False, 'details': ''}
    
    def _check_completeness(self, answer: str, query: str) -> Dict:
        """检查完整性"""
        prompt = f"""检查回答是否完整。

问题: {query}
回答: {answer[:500]}

输出JSON：{{"has_error": true/false, "missing_aspects": ["缺失方面"]}}"""
        
        response = self.llm(prompt)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {'has_error': False, 'missing_aspects': []}

class SelfCorrectionEngine:
    """自我纠正引擎"""
    
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever
        self.detector = ErrorDetector(llm)
    
    def generate_with_correction(self, query: str, 
                                   max_corrections: int = 3) -> str:
        """生成并自我纠正"""
        # 初始生成
        answer = self._initial_generation(query)
        correction_count = 0
        
        while correction_count < max_corrections:
            # 检测错误
            errors = self.detector.detect_errors(answer, query)
            
            if not errors['has_errors']:
                break
            
            # 根据错误类型纠正
            answer = self._apply_correction(query, answer, errors)
            correction_count += 1
        
        return answer
    
    def _initial_generation(self, query: str) -> str:
        """初始生成"""
        docs = self.retriever.retrieve(query, k=5)
        context = "\n".join([d.page_content[:500] for d in docs])
        
        prompt = f"""基于以下信息回答问题。

{context}

问题: {query}"""
        
        return self.llm(prompt)
    
    def _apply_correction(self, query: str, answer: str, 
                           errors: Dict) -> str:
        """应用纠正"""
        error_description = self._format_errors(errors)
        
        prompt = f"""修正以下回答中的错误。

问题: {query}
当前回答: {answer}

发现的错误:
{error_description}

请输出修正后的完整回答："""
        
        return self.llm(prompt)
    
    def _format_errors(self, errors: Dict) -> str:
        """格式化错误描述"""
        parts = []
        for error_type, error_info in errors.items():
            if error_info.get('has_error'):
                parts.append(f"- {error_type}: {json.dumps(error_info.get('details', ''), ensure_ascii=False)}")
        
        return '\n'.join(parts)
```

## 12.10 本章小结

本章深入探讨了智能体RAG（Agentic RAG）的核心技术体系和实现方法。智能体RAG通过赋予系统主动推理和行动的能力，显著突破了传统RAG的局限性。

**ReAct模式**是智能体RAG的基石，通过"思考-行动-观察"的循环，让LLM能够自主规划、执行和调整策略。本章提供了完整的ReAct智能体实现，包括提示词模板、行动解析和轨迹追踪。

**CRAG（Corrective RAG）**通过引入检索评估器，在检索结果质量不佳时主动进行纠正。CRAG的三种策略——直接生成、知识分解重构和重新检索——覆盖了不同质量的检索结果场景。

**Self-RAG**通过反思标记（检索标记、相关标记、支持标记）让LLM在生成过程中进行自我反思，动态决定是否需要检索以及生成内容是否可靠。

**多智能体协作**通过主管、检索、推理、验证和写作等多个专业智能体的协同工作，实现了更复杂任务的分解和处理。本章提供了一个完整的多智能体框架实现。

**工具使用**方面，本章介绍了工具注册中心、Function Calling集成和多种常用工具（检索、计算、代码执行等）的实现，使智能体能够灵活调用外部能力。

**多步骤推理**通过思维链、查询分解和自适应检索等技术，使系统能够处理需要多步推理的复杂问题。

**自我纠正机制**通过错误检测和修正循环，确保系统输出的准确性和可靠性。

**LangGraph实现**部分展示了如何使用LangGraph框架构建有状态、多步骤的Agentic RAG工作流，包括条件路由、循环和状态管理。

在实际应用中，建议根据具体场景选择合适的智能体RAG策略组合：简单查询使用传统RAG即可，中等复杂度查询引入ReAct模式，复杂查询使用多智能体协作，对质量要求极高的场景启用CRAG和Self-RAG的纠正机制。
