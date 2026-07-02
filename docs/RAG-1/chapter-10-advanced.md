# 第10章 高级RAG技术：GraphRAG、混合检索与Agentic RAG

## 10.1 引言

基础RAG（检索增强生成）在简单问答场景中表现良好，但当面对复杂推理、跨文档聚合和多跳问题时，其局限性逐渐显现。本章深入探讨三类高级RAG范式：GraphRAG（图增强检索）、混合检索策略以及Agentic RAG（智能体驱动的检索增强生成）。这些技术代表了RAG领域的最新进展，能够显著提升系统在复杂场景下的表现。

我们将以实际项目为依托，展示每种技术的实现细节、调优策略和工程考量。本章涵盖的GraphRAG实践基于GraphRAG-KG项目（一个集成了知识图谱与RAG的生产级系统），所有代码示例均可直接复用或适配。

---

## 10.2 GraphRAG深度实践

GraphRAG由Microsoft Research提出，其核心思想是在文档预处理阶段构建实体关系知识图谱，并通过社区检测算法对图结构进行分层聚合。相比传统RAG仅依赖向量相似度检索，GraphRAG能够捕捉实体之间的高阶语义关系，在全局性问题、摘要生成和多文档推理任务上展现出显著优势。

### 10.2.1 实体提取优化

实体提取是GraphRAG的基石——提取质量直接决定后续所有环节的效果。GraphRAG-KG项目在实体提取方面提供了多层次的可配置选项。

#### 10.2.1.1 提示词调优（Prompt Tuning）

GraphRAG的实体提取本质上是LLM驱动的信息抽取任务。通过精心设计的提示词模板，可以显著提升提取的准确率和覆盖率。以下是GraphRAG-KG项目中使用的实体提取配置示例：

```python
# 实体提取配置：通过提示词模板控制提取行为
entity_extraction_config = {
    "strategy": "graph_intelligence",
    "llm": {
        "model": "deepseek-chat",
        "max_tokens": 6000,
        "temperature": 0.0,  # 提取任务使用低温度以确保一致性
    },
    "prompt_template": """
你是一个专业的命名实体识别（NER）系统。请从以下文本中提取所有实体及其关系。

实体类型定义：
- DRUG（药物）：包含药品名称、活性成分、化学物质
- COMPANY（公司）：包含制药企业、生物技术公司、合同研究组织
- DISEASE（疾病）：包含疾病名称、症状、病症
- REGULATORY（监管机构）：包含FDA、EMA、NMPA等监管实体
- CLINICAL_TRIAL（临床试验）：包含临床试验阶段、试验编号
- PERSON（人物）：包含研究人员、高管、关键意见领袖
- PROCESS（流程）：包含制造流程、检验流程、供应链流程
- LOCATION（地点）：包含国家、城市、生产基地位置

输出格式要求：
1. 实体列表：每个实体包含名称（name）、类型（type）、描述（description）
2. 关系列表：每条关系包含头实体（source）、尾实体（target）、关系类型（relationship）
3. 如果文本中没有符合条件的实体，请输出空列表
4. 不要编造任何信息

文本：
{input_text}
""",
    "entity_types": [
        "DRUG", "COMPANY", "DISEASE", "REGULATORY",
        "CLINICAL_TRIAL", "PERSON", "PROCESS", "LOCATION"
    ],
    "gleaning_rounds": 2,  # 多轮提取轮数
}
```

提示词调优的关键参数包括：

- **temperature**：实体提取是确定性任务，建议设置为0.0或接近0.0，以确保每次提取结果的一致性。
- **entity_types**：明确定义实体类型可以引导LLM关注领域相关的实体。类型定义应覆盖业务场景中的所有关键概念，但也不宜过多（建议8-12种），否则会增加LLM的认知负担。
- **max_tokens**：对于包含大量实体的长文档，需要足够大的输出窗口。根据经验，6000-8000 tokens通常能够覆盖绝大多数场景。

#### 10.2.1.2 实体类型定义策略

实体类型定义是领域适配的关键环节。GraphRAG-KG项目在制药供应链场景中定义了12种实体类型，以下是完整的定义配置：

```python
# 制药供应链领域的实体类型定义
PHARMA_ENTITY_TYPES = [
    {
        "name": "DRUG",
        "description": "药物、药品、候选化合物、生物制剂",
        "examples": ["帕博利珠单抗", "阿达木单抗", "辉瑞-BioNTech疫苗"]
    },
    {
        "name": "COMPANY",
        "description": "制药企业、生物科技公司、CRO/CDMO企业",
        "examples": ["辉瑞", "Moderna", "药明康德"]
    },
    {
        "name": "DISEASE",
        "description": "疾病、适应证、症状、并发症",
        "examples": ["非小细胞肺癌", "类风湿关节炎", "COVID-19"]
    },
    {
        "name": "REGULATORY",
        "description": "监管机构、法规、审批状态",
        "examples": ["FDA", "EMA", "NMPA", "孤儿药认定"]
    },
    {
        "name": "CLINICAL_TRIAL",
        "description": "临床试验、试验阶段、试验设计",
        "examples": ["I期临床试验", "III期关键试验", "随机双盲"]
    },
    {
        "name": "PERSON",
        "description": "科研人员、企业高管、医学专家",
        "examples": ["Albert Bourla", "Dr. Anthony Fauci"]
    },
    {
        "name": "PROCESS",
        "description": "制造流程、供应链流程、质量控制流程",
        "examples": ["冷链运输", "mRNA合成", "无菌灌装"]
    },
    {
        "name": "LOCATION",
        "description": "地理位置、生产基地、研发中心",
        "examples": ["上海", "波士顿", "马里兰州"]
    },
    {
        "name": "MATERIAL",
        "description": "原材料、辅料、包装材料、化学试剂",
        "examples": ["脂质纳米颗粒", "mRNA模板", "玻璃瓶"]
    },
    {
        "name": "EQUIPMENT",
        "description": "生产设备、检测设备、实验室仪器",
        "examples": ["生物反应器", "高效液相色谱仪", "低温冰箱"]
    },
    {
        "name": "CONTRACT",
        "description": "合同、协议、许可、专利",
        "examples": ["授权协议", "供应合同", "专利许可"]
    },
    {
        "name": "EVENT",
        "description": "行业会议、并购事件、产品上市事件",
        "examples": ["JPM医疗大会", "辉瑞收购Seagen"]
    }
]
```

设计实体类型时的最佳实践：

1. **领域驱动设计**：实体类型应直接映射业务领域中的核心概念，而非通用语言学分类。
2. **互斥性**：类型之间应尽可能互斥，避免模糊边界（如"药物"与"化合物"的区分）。
3. **覆盖度与粒度平衡**：过多的类型会增加提取难度，过少则信息粒度不足。建议从8种左右开始，逐步迭代扩展。
4. **提供示例**：每个类型提供3-5个具体示例，帮助LLM理解类型的边界。

#### 10.2.1.3 多轮提取（Gleaning Rounds）

单轮提取可能遗漏重要实体。Gleaning机制允许LLM对已提取的结果进行多轮补充，每次补充只关注"上一轮遗漏的实体"。配置参数如下：

```python
# Gleaning配置
gleaning_config = {
    "gleaning_rounds": 2,           # 补充轮数
    "gleaning_prompt": """
上一轮实体提取的结果如下：
{previous_output}

请检查以上结果，找出遗漏的重要实体或关系。注意：
1. 只补充遗漏的内容，不要重复已提取的实体
2. 重点关注：核心业务实体、关键人物、重要事件
3. 如果确认已完整提取，请直接输出空列表

补充提取结果：
""",
    "gleaning_min_entities": 5,     # 每轮最少补充实体数
    "gleaning_max_entities": 20,    # 每轮最多补充实体数
}
```

Gleaning轮数的选择需要在召回率和成本之间权衡：

| 轮数 | 召回率提升 | 额外LLM调用成本 | 适用场景 |
|------|-----------|----------------|---------|
| 0    | 基线      | 无             | 简单文档、测试环境 |
| 1    | +15-25%   | +100%          | 一般生产环境 |
| 2    | +25-35%   | +200%          | 知识密集型文档 |
| 3    | +30-38%   | +300%          | 高精度要求的场景 |

超过3轮后，边际收益迅速递减，不建议继续增加。

### 10.2.2 社区检测参数调优

实体提取完成后，GraphRAG需要对实体关系图进行社区检测（Community Detection），将紧密连接的实体聚合成社区。社区是后续查询理解的基本单元。GraphRAG使用Leiden算法进行社区检测，相关参数的调优直接影响检索效果。

#### 10.2.2.1 核心参数详解

```python
# 社区检测配置
community_config = {
    "max_cluster_size": 10,     # 最大社区大小（层级0）
    "seed": 42,                 # 随机种子
    "use_lcc": True,            # 仅使用最大连通分量
    "hierarchical_levels": 2,   # 层次级别数
    "resolution": 1.0,          # 分辨率参数
}
```

**max_cluster_size**：控制最大社区大小。较小的值产生更细粒度的社区划分，有利于局部精确检索；较大的值产生更粗粒度的社区，有利于全局理解。建议的调优策略：

- 对于技术文档（每个实体高度相关）：`max_cluster_size=5-8`
- 对于通用知识库：`max_cluster_size=10-15`
- 对于百科类数据：`max_cluster_size=15-20`

**seed**：随机种子控制社区检测的稳定性。Leiden算法包含随机性，相同的输入在不同seed下可能产生不同的社区划分。设置为固定值（如42）确保结果可重现。在A/B测试和回归测试中，固定seed至关重要。

**use_lcc（Largest Connected Component）**：是否只使用最大连通分量。知识图谱中可能存在孤立的连通分量（即与主图无连接的实体组）。`use_lcc=True`会过滤掉这些孤立分量，只保留最大的连通子图。适用场景：

- `True`：当预期所有实体都应在同一知识体系中时（如企业知识库）
- `False`：当需要保留所有实体信息时（如多领域混合知识库）

#### 10.2.2.2 层次化社区结构

GraphRAG构建层次化社区结构，每一层对应不同的聚合粒度：

```python
def build_hierarchical_communities(G, levels=2, max_cluster_size=10):
    """
    构建层次化社区结构。
    
    层级0：原始实体级别，每个实体是一个节点
    层级1：初级社区，紧密连接的实体聚合
    层级2：高级社区，初级社区的进一步聚合
    
    Parameters:
        G: NetworkX图对象
        levels: 层次级别数
        max_cluster_size: 最大社区大小
    """
    import networkx as nx
    from graspologic.partition import hierarchical_leiden
    
    # 使用Leiden层次聚类
    community_map = hierarchical_leiden(
        G,
        max_cluster_size=max_cluster_size,
        starting_communities=None,
        extra_forced_iterations=0,
        random_seed=42,
    )
    
    # 按层级组织社区
    hierarchical_communities = {}
    for community in community_map:
        level = community.level
        if level not in hierarchical_communities:
            hierarchical_communities[level] = {}
        
        cluster_id = community.cluster
        node = community.node
        
        if cluster_id not in hierarchical_communities[level]:
            hierarchical_communities[level][cluster_id] = []
        hierarchical_communities[level][cluster_id].append(node)
    
    return hierarchical_communities
```

查询时，根据问题的复杂度选择不同层级的社区：
- **局部问题**（"某药物的作用机制是什么"）→ 使用层级0-1的社区
- **全局问题**（"制药行业的主要趋势有哪些"）→ 使用层级1-2的社区

### 10.2.3 查询策略选择

GraphRAG提供了多种查询策略，每种策略适合不同的查询类型。GraphRAG-KG项目封装了统一的查询接口，自动根据查询特征选择最优策略。

#### 10.2.3.1 Local Search（局部搜索）

局部搜索专注于与查询实体直接相关的局部邻域，适用于具体、精确的查询。

```python
# GraphRAG-KG中的Local Search实现
from graphrag.query.local_search import LocalSearch

async def local_search_example():
    """使用Local Search进行局部查询"""
    local_search = LocalSearch(
        llm=llm,
        stream=True,
        context_builder="local",  # 使用局部上下文构建器
        token_encoder=token_encoder,
        system_prompt=local_search_system_prompt,
        response_type="multiple paragraphs",  # 输出格式
    )
    
    result = await local_search.search(
        query="Keytruda（帕博利珠单抗）在非小细胞肺癌治疗中的应用是什么？",
        conversation_history=None,  # 可选对话历史
    )
    
    return result
```

**工作原理**：
1. 从查询中提取关键实体
2. 在知识图谱中找到这些实体的直接邻居（1-hop）
3. 收集邻居实体的描述文本和关系
4. 使用LLM综合这些信息生成回答

**适用场景**：实体级问答、属性查询、关系查询

#### 10.2.3.2 Global Search（全局搜索）

全局搜索聚合多个社区摘要，适用于需要综合大量信息的查询。

```python
# GraphRAG-KG中的Global Search实现
from graphrag.query.global_search import GlobalSearch

async def global_search_example():
    """使用Global Search进行全局查询"""
    global_search = GlobalSearch(
        llm=llm,
        context_builder="global",
        token_encoder=token_encoder,
        dynamic_community_selection=True,  # 动态社区选择
        map_system_prompt=map_prompt,      # Map阶段的提示词
        reduce_system_prompt=reduce_prompt,  # Reduce阶段的提示词
    )
    
    result = await global_search.search(
        query="2024年全球制药行业的主要趋势和挑战是什么？",
        allow_general_knowledge=True,  # 允许使用通用知识补充
    )
    
    return result
```

**工作原理**：
1. 对每个社区摘要独立生成部分答案（Map阶段）
2. 合并所有部分答案生成最终回答（Reduce阶段）
3. 动态社区选择：根据查询与社区的相关性选择Top-K社区

**适用场景**：趋势分析、综述类问题、多文档综合

#### 10.2.3.3 Drift Search（漂移搜索）

Drift Search是GraphRAG中最先进的查询策略，通过多步实体漂移实现复杂推理路径的探索。

```python
# GraphRAG-KG中的Drift Search实现
from graphrag.query.drift_search import DriftSearch

async def drift_search_example():
    """使用Drift Search进行多步推理查询"""
    drift_search = DriftSearch(
        llm=llm,
        n=5,                    # 每步探索的实体数
        max_depth=3,            # 最大漂移深度
        drift_k=10,             # 每步保留的候选数
        primer_folds=3,         # 初始实体扩展轮数
        # 信噪比配置
        noise_threshold=0.3,    # 噪声阈值，低于此值停止漂移
        use_cosine_reranker=True,  # 使用余弦重排序
    )
    
    result = await drift_search.search(
        query="辉瑞收购Seagen对其ADC药物管线有什么影响？",
        # 种子实体，可选的初始实体列表
        seed_entities=["辉瑞", "Seagen"],
    )
    
    return result
```

**工作原理**：
1. **Primer阶段**：基于查询实体，在知识图谱中采样初始上下文
2. **漂移阶段**：从当前实体集出发，沿关系边漂移到相邻实体，扩展上下文
3. **评估阶段**：评估每个漂移步骤的信息增益，决定继续或终止
4. **聚合阶段**：将所有漂移路径的信息综合生成回答

**适用场景**：多跳推理、因果关系查询、影响分析

#### 10.2.3.4 查询策略自动路由

GraphRAG-KG的查询引擎实现了自动路由逻辑，根据查询特征选择最优策略：

```python
class QueryRouter:
    """查询策略路由引擎"""
    
    def __init__(self):
        self.strategies = {
            "local": LocalSearchStrategy(),
            "global": GlobalSearchStrategy(),
            "drift": DriftSearchStrategy(),
            "basic": BasicSearchStrategy(),
        }
    
    async def route(self, query: str) -> SearchStrategy:
        """
        根据查询特征自动路由到最优策略。
        
        路由逻辑：
        1. 短查询（<10词）且包含具体实体名 → local
        2. 抽象/总结类查询（含"趋势""概述""总结"等关键词）→ global
        3. 多跳推理查询（含"影响""关系""如何导致"等）→ drift
        4. 其他 → basic
        """
        query_lower = query.lower()
        word_count = len(query.split())
        
        # 关键词匹配规则
        global_keywords = ["趋势", "概述", "总结", "综述", "分析",
                          "trend", "overview", "summary", "survey"]
        drift_keywords = ["影响", "导致", "关系", "因果", "路径",
                         "impact", "cause", "relation", "pathway"]
        
        if word_count < 10 and self._has_entity(query):
            return self.strategies["local"]
        elif any(kw in query_lower for kw in global_keywords):
            return self.strategies["global"]
        elif any(kw in query_lower for kw in drift_keywords):
            return self.strategies["drift"]
        else:
            return self.strategies["basic"]
    
    def _has_entity(self, query: str) -> bool:
        """检查查询中是否包含已知实体"""
        # 实现：对查询进行NER，匹配知识库中的实体
        pass
```

### 10.2.4 成本分析

GraphRAG的LLM调用成本显著高于传统RAG，主要开销集中在索引构建阶段。以下是一个生产级系统的成本估算模型：

```python
# GraphRAG成本估算模型
def estimate_graphrag_cost(
    num_documents: int,
    avg_doc_tokens: int,
    entities_per_doc: float,
    gleaning_rounds: int,
    community_levels: int,
    num_queries_per_day: int,
    llm_cost_per_input_token: float = 0.000002,   # DeepSeek成本（美元/输入token）
    llm_cost_per_output_token: float = 0.000008,  # DeepSeek成本（美元/输出token）
):
    """估算GraphRAG的全流程成本"""
    
    # 1. 索引构建成本（一次性）
    # 实体提取：每个文档一次提取调用
    extraction_input_tokens = num_documents * avg_doc_tokens
    extraction_output_tokens = num_documents * entities_per_doc * 20  # 估计输出
    
    # Gleaning轮次
    gleaning_input_tokens = extraction_input_tokens * gleaning_rounds
    gleaning_output_tokens = extraction_output_tokens * gleaning_rounds * 0.3
    
    # 社区摘要生成
    num_communities = num_documents * entities_per_doc / 10  # 估计社区数
    community_input_tokens = num_communities * 2000  # 每个社区的上下文
    community_output_tokens = num_communities * 500  # 每个社区的摘要
    
    total_index_input = extraction_input_tokens + gleaning_input_tokens + community_input_tokens
    total_index_output = extraction_output_tokens + gleaning_output_tokens + community_output_tokens
    
    index_cost = (total_index_input * llm_cost_per_input_token +
                  total_index_output * llm_cost_per_output_token)
    
    # 2. 查询成本（每日）
    # Local Search
    local_input_per_query = 4000  # 局部上下文
    local_output_per_query = 800
    
    # Global Search（Map-Reduce）
    global_input_per_query = num_communities * 2000  # 所有社区摘要
    global_output_per_query = num_communities * 300 + 1000  # Map+Reduce
    
    # Drift Search
    drift_input_per_query = 6000
    drift_output_per_query = 1500
    
    # 假设查询分布：50% local, 20% global, 30% drift
    avg_input = (0.5 * local_input_per_query +
                 0.2 * global_input_per_query +
                 0.3 * drift_input_per_query)
    avg_output = (0.5 * local_output_per_query +
                  0.2 * global_output_per_query +
                  0.3 * drift_output_per_query)
    
    daily_query_cost = num_queries_per_day * (
        avg_input * llm_cost_per_input_token +
        avg_output * llm_cost_per_output_token
    )
    
    return {
        "index_build_cost": round(index_cost, 2),
        "daily_query_cost": round(daily_query_cost, 2),
        "monthly_cost": round(daily_query_cost * 30 + index_cost / 12, 2),
        "cost_breakdown": {
            "entity_extraction": round(extraction_input_tokens * llm_cost_per_input_token, 2),
            "gleaning": round(gleaning_input_tokens * llm_cost_per_input_token, 2),
            "community_summary": round(community_input_tokens * llm_cost_per_input_token, 2),
        }
    }

# 示例：1000篇文档的制药知识库
cost = estimate_graphrag_cost(
    num_documents=1000,
    avg_doc_tokens=3000,
    entities_per_doc=15,
    gleaning_rounds=2,
    community_levels=2,
    num_queries_per_day=10000,
)

print(f"索引构建成本: ${cost['index_build_cost']}")
print(f"日均查询成本: ${cost['daily_query_cost']}")
print(f"月均总成本: ${cost['monthly_cost']}")
```

成本优化建议：

1. **使用小模型进行实体提取**：实体提取等结构化任务可以使用DeepSeek-V3等性价比高的模型，仅在最终生成回答时使用更强的模型。
2. **缓存社区摘要**：社区摘要在文档更新前是不变的，应缓存复用。
3. **按需构建**：仅对需要查询的文档子集构建图谱，避免全量构建。
4. **Gleaning轮数自适应**：根据文档长度自动调整gleaning轮数，短文档0-1轮，长文档2轮。

---

## 10.3 层次化检索

层次化检索（Hierarchical Retrieval）通过构建文档的多粒度索引，实现从粗到精的渐进式上下文构建。这种方法在处理长文档和复杂查询时尤为有效。

### 10.3.1 文档树结构

层次化检索的核心是构建文档树：文档 → 章节 → 段落 → 句子。每一层都维护独立的索引，检索时从顶层开始，逐步深入到下层。

```python
from dataclasses import dataclass, field
from typing import List, Optional
import hashlib

@dataclass
class DocumentNode:
    """文档树节点"""
    node_id: str
    content: str
    level: int                # 0=文档, 1=章节, 2=段落, 3=句子
    embedding: Optional[List[float]] = None
    summary: Optional[str] = None
    children: List['DocumentNode'] = field(default_factory=list)
    parent_id: Optional[str] = None
    
    def __post_init__(self):
        if not self.node_id:
            self.node_id = hashlib.md5(
                self.content.encode()
            ).hexdigest()[:12]


class DocumentTreeBuilder:
    """文档树构建器"""
    
    def __init__(self, max_section_depth: int = 3):
        self.max_depth = max_section_depth
    
    def build_tree(self, document: str) -> DocumentNode:
        """
        将原始文档构建为树结构。
        支持Markdown标题、段落分隔符和句子边界检测。
        """
        doc_node = DocumentNode(
            content=document,
            level=0,
            summary=self._generate_summary(document),
        )
        
        # 按Markdown标题分割章节
        sections = self._split_by_headings(document)
        for section in sections:
            section_node = self._build_section(section, depth=1)
            section_node.parent_id = doc_node.node_id
            doc_node.children.append(section_node)
        
        return doc_node
    
    def _build_section(self, content: str, depth: int) -> DocumentNode:
        """递归构建章节节点"""
        node = DocumentNode(
            content=content,
            level=depth,
            summary=self._generate_summary(content),
        )
        
        if depth < self.max_depth:
            # 按段落分割
            paragraphs = self._split_paragraphs(content)
            for para in paragraphs:
                para_node = DocumentNode(
                    content=para,
                    level=depth + 1,
                    summary=self._generate_summary(para),
                )
                para_node.parent_id = node.node_id
                
                # 按句子分割
                sentences = self._split_sentences(para)
                for sent in sentences:
                    sent_node = DocumentNode(
                        content=sent,
                        level=depth + 2,
                    )
                    sent_node.parent_id = para_node.node_id
                    para_node.children.append(sent_node)
                
                node.children.append(para_node)
        
        return node
    
    def _generate_summary(self, text: str) -> str:
        """使用LLM生成节点摘要（可选，可降级为截取前200字）"""
        if len(text) < 100:
            return text
        return text[:200] + "..."
    
    def _split_by_headings(self, text: str) -> List[str]:
        """按Markdown/HTML标题分割文本"""
        import re
        # 支持 # ## ### === --- 等标题格式
        heading_pattern = r'(?:^|\n)(#{1,3}\s.+|\w[\w\s]+\n[-=]+\n)'
        splits = re.split(heading_pattern, text, flags=re.MULTILINE)
        return [s.strip() for s in splits if s.strip()]
    
    def _split_paragraphs(self, text: str) -> List[str]:
        """按空行分割段落"""
        paragraphs = [p.strip() for p in text.split('\n\n')]
        return [p for p in paragraphs if len(p) > 20]
    
    def _split_sentences(self, text: str) -> List[str]:
        """按句子边界分割"""
        import re
        sentences = re.split(r'[。！？.!?]\s*', text)
        return [s.strip() for s in sentences if s.strip()]
```

### 10.3.2 渐进式上下文构建

渐进式上下文构建（Progressive Context Building）模拟人类阅读的方式：先了解整体结构，再深入细节，逐步缩小搜索范围。

```python
class ProgressiveContextBuilder:
    """渐进式上下文构建器"""
    
    def __init__(self, doc_tree: DocumentNode):
        self.doc_tree = doc_tree
        self.context_hierarchy: List[DocumentNode] = []
    
    async def build_context(
        self,
        query: str,
        max_tokens: int = 4000,
        top_k_sections: int = 3,
        top_k_paragraphs: int = 5,
    ) -> str:
        """
        渐进式构建查询上下文。
        
        流程：
        1. 在文档层匹配 → 选择Top-K相关文档
        2. 在所选文档的章节层匹配 → 选择Top-K相关章节
        3. 在所选章节的段落层匹配 → 选择Top-K相关段落
        4. 在所选段落的句子层匹配 → 选择Top-K相关句子
        5. 拼接完整的层级化上下文
        """
        # 第一层：文档级匹配
        doc_matches = await self._retrieve_level(
            query, self.doc_tree, top_k=3
        )
        
        context_parts = []
        used_tokens = 0
        
        for doc in doc_matches:
            if used_tokens >= max_tokens:
                break
            
            # 添加文档摘要
            if doc.summary:
                context_parts.append(f"[文档摘要] {doc.summary}")
                used_tokens += len(doc.summary) // 2  # 粗略token计数
            
            # 第二层：章节级匹配
            section_matches = await self._retrieve_level(
                query, doc.children, top_k=top_k_sections
            )
            
            for section in section_matches:
                if used_tokens >= max_tokens:
                    break
                
                context_parts.append(f"\n[章节] {section.summary}")
                used_tokens += len(section.summary or "") // 2
                
                # 第三层：段落级匹配
                para_matches = await self._retrieve_level(
                    query, section.children, top_k=top_k_paragraphs
                )
                
                for para in para_matches:
                    if used_tokens >= max_tokens:
                        break
                    
                    # 截断段落以适应token限制
                    truncated = para.content[:max_tokens - used_tokens]
                    context_parts.append(f"\n{truncated}")
                    used_tokens += len(truncated) // 2
        
        return "\n".join(context_parts)
    
    async def _retrieve_level(
        self,
        query: str,
        nodes: List[DocumentNode],
        top_k: int,
    ) -> List[DocumentNode]:
        """在指定层级检索相关节点"""
        if not nodes:
            return []
        
        # 计算查询与节点的相关性得分
        scores = []
        for node in nodes:
            if node.embedding is not None:
                score = cosine_similarity(query, node.embedding)
            else:
                # 回退到关键词匹配
                score = self._keyword_match(query, node.content)
            scores.append((node, score))
        
        # 按得分排序并返回Top-K
        scores.sort(key=lambda x: x[1], reverse=True)
        return [node for node, _ in scores[:top_k]]
    
    def _keyword_match(self, query: str, content: str) -> float:
        """简单的关键词匹配得分"""
        query_words = set(query.lower().split())
        content_lower = content.lower()
        matches = sum(1 for w in query_words if w in content_lower)
        return matches / max(len(query_words), 1)
```

### 10.3.3 多级索引

多级索引为文档树的每一层建立独立的向量索引，实现高效的层次化检索。

```python
class MultiLevelIndex:
    """多级向量索引管理器"""
    
    def __init__(self, embedding_dim: int = 1024):
        self.embedding_dim = embedding_dim
        self.indices = {
            0: [],  # 文档级索引
            1: [],  # 章节级索引
            2: [],  # 段落级索引
            3: [],  # 句子级索引
        }
        self.id_to_node = {}
    
    def add_document(self, doc_node: DocumentNode):
        """递归添加文档树到各级索引"""
        # 为当前节点生成嵌入
        if doc_node.embedding is None:
            doc_node.embedding = self._generate_embedding(doc_node.content)
        
        level = doc_node.level
        self.indices[level].append({
            "id": doc_node.node_id,
            "embedding": doc_node.embedding,
            "content": doc_node.summary or doc_node.content[:200],
        })
        self.id_to_node[doc_node.node_id] = doc_node
        
        # 递归添加子节点
        for child in doc_node.children:
            self.add_document(child)
    
    async def hierarchical_search(
        self,
        query: str,
        query_embedding: List[float],
        top_k: int = 5,
    ) -> List[DocumentNode]:
        """
        层次化搜索：从顶层到底层逐步聚焦。
        
        返回从不同层级选出的最相关节点。
        """
        results = []
        
        # 从顶层开始
        for level in range(4):
            if not self.indices[level]:
                continue
            
            # 计算该层所有节点与查询的相似度
            similarities = []
            for item in self.indices[level]:
                sim = cosine_similarity(query_embedding, item["embedding"])
                similarities.append((item["id"], sim))
            
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # 选取Top-K
            for node_id, sim in similarities[:top_k]:
                node = self.id_to_node[node_id]
                results.append({
                    "node": node,
                    "score": sim,
                    "level": level,
                })
        
        return results
    
    def _generate_embedding(self, text: str) -> List[float]:
        """生成文本嵌入（调用嵌入模型）"""
        # 实际项目中调用Ollama bge-m3或OpenAI embeddings
        # 此处返回占位值
        import numpy as np
        return np.random.randn(self.embedding_dim).tolist()


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """计算余弦相似度"""
    import numpy as np
    a_arr = np.array(a)
    b_arr = np.array(b)
    return float(np.dot(a_arr, b_arr) / (
        np.linalg.norm(a_arr) * np.linalg.norm(b_arr) + 1e-10
    ))
```

层次化检索的核心优势：

1. **上下文质量提升**：渐进式构建确保上下文与查询的相关性，减少噪声
2. **Token效率**：只在需要的层级消耗token，避免"一刀切"的冗余
3. **可解释性**：检索路径（文档→章节→段落→句子）提供了清晰的可追溯链条

---

## 10.4 混合搜索

混合搜索（Hybrid Search）结合向量搜索的语义理解能力和关键词搜索的精确匹配能力，是生产级RAG系统的标配。

### 10.4.1 向量 + 关键词加权融合

```python
class HybridSearchEngine:
    """混合搜索引擎：向量搜索 + 关键词搜索加权融合"""
    
    def __init__(
        self,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
        vector_search: Optional[object] = None,
        keyword_search: Optional[object] = None,
    ):
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.vector_search = vector_search
        self.keyword_search = keyword_search
    
    def reciprocal_rank_fusion(
        self,
        vector_results: List[dict],
        keyword_results: List[dict],
        k: int = 60,
        top_n: int = 10,
    ) -> List[dict]:
        """
        使用倒数排名融合（RRF）合并两个结果集。
        
        RRF公式：score(d) = Σ(1 / (k + rank(d, i)))
        其中rank(d, i)是文档d在第i个检索器中的排名
        
        Args:
            vector_results: 向量搜索结果 [{id, score}]
            keyword_results: 关键词搜索结果 [{id, score}]
            k: RRF常数（通常60）
            top_n: 最终返回的文档数
        
        Returns:
            融合后的排名结果
        """
        from collections import defaultdict
        
        # 构建排名映射
        ranks = defaultdict(lambda: [float('inf'), float('inf')])
        
        for rank, doc in enumerate(vector_results):
            ranks[doc['id']][0] = rank + 1
        
        for rank, doc in enumerate(keyword_results):
            ranks[doc['id']][1] = rank + 1
        
        # 计算RRF得分
        scores = {}
        for doc_id, (r_vec, r_key) in ranks.items():
            score = 0.0
            if r_vec != float('inf'):
                score += self.vector_weight / (k + r_vec)
            if r_key != float('inf'):
                score += self.keyword_weight / (k + r_key)
            scores[doc_id] = score
        
        # 按得分排序
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        return [
            {"id": doc_id, "score": score}
            for doc_id, score in sorted_docs[:top_n]
        ]
    
    def weighted_sum_fusion(
        self,
        vector_results: List[dict],
        keyword_results: List[dict],
        top_n: int = 10,
    ) -> List[dict]:
        """
        使用加权和融合两个结果集。
        需要先将分数归一化到[0,1]区间。
        """
        from collections import defaultdict
        
        # 归一化分数
        vec_scores = self._min_max_normalize(
            {d['id']: d['score'] for d in vector_results}
        )
        key_scores = self._min_max_normalize(
            {d['id']: d['score'] for d in keyword_results}
        )
        
        # 加权融合
        combined = defaultdict(float)
        all_ids = set(vec_scores.keys()) | set(key_scores.keys())
        
        for doc_id in all_ids:
            combined[doc_id] = (
                self.vector_weight * vec_scores.get(doc_id, 0.0) +
                self.keyword_weight * key_scores.get(doc_id, 0.0)
            )
        
        sorted_docs = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        
        return [
            {"id": doc_id, "score": score}
            for doc_id, score in sorted_docs[:top_n]
        ]
    
    def _min_max_normalize(self, scores: dict) -> dict:
        """Min-Max归一化"""
        if not scores:
            return {}
        min_s = min(scores.values())
        max_s = max(scores.values())
        if max_s == min_s:
            return {k: 1.0 for k in scores}
        return {k: (v - min_s) / (max_s - min_s) for k, v in scores.items()}
```

### 10.4.2 自适应权重调整

不同查询类型适合不同的权重组合。自适应权重调整根据查询的语义特征动态调整向量搜索和关键词搜索的权重：

```python
class AdaptiveHybridSearch:
    """自适应权重混合搜索"""
    
    def __init__(self):
        self.base_weights = {
            "semantic": {"vector": 0.8, "keyword": 0.2},  # 语义查询
            "factual": {"vector": 0.4, "keyword": 0.6},   # 事实查询
            "entity": {"vector": 0.3, "keyword": 0.7},    # 实体查询
            "code": {"vector": 0.2, "keyword": 0.8},      # 代码/公式查询
            "mixed": {"vector": 0.6, "keyword": 0.4},     # 混合查询
        }
    
    def classify_query(self, query: str) -> str:
        """
        对查询进行分类，返回查询类型。
        
        分类特征：
        - 实体查询：包含专有名词、产品名、人名
        - 事实查询：包含数字、日期、具体属性
        - 代码查询：包含代码片段、特殊符号
        - 语义查询：抽象概念、比较、原因分析
        - 混合查询：无法明确分类的
        """
        import re
        
        # 检测实体查询
        entity_patterns = [
            r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)*\b',  # 驼峰命名
            r'[一-鿿]{2,}',               # 中文词汇
        ]
        entity_count = sum(
            len(re.findall(p, query)) for p in entity_patterns
        )
        
        # 检测事实查询
        fact_patterns = [r'\d+', r'\d+%', r'\d+年', r'\d+月']
        fact_count = sum(
            len(re.findall(p, query)) for p in fact_patterns
        )
        
        # 检测代码查询
        code_patterns = [r'[{}();]', r'\b(def|class|import|if)\b']
        code_count = sum(
            len(re.findall(p, query)) for p in code_patterns
        )
        
        # 根据特征得分分类
        query_len = len(query)
        
        if code_count / max(query_len, 1) > 0.05:
            return "code"
        if fact_count >= 2:
            return "factual"
        if entity_count >= 3:
            return "entity"
        if entity_count <= 1 and fact_count == 0:
            return "semantic"
        
        return "mixed"
    
    def get_weights(self, query: str) -> dict:
        """根据查询类型获取权重"""
        query_type = self.classify_query(query)
        return self.base_weights.get(query_type, self.base_weights["mixed"])
    
    async def search(
        self,
        query: str,
        vector_search_fn,
        keyword_search_fn,
        top_n: int = 10,
    ) -> List[dict]:
        """执行自适应混合搜索"""
        # 获取自适应权重
        weights = self.get_weights(query)
        self.vector_weight = weights["vector"]
        self.keyword_weight = weights["keyword"]
        
        # 执行两种搜索
        vector_results = await vector_search_fn(query, top_n * 2)
        keyword_results = await keyword_search_fn(query, top_n * 2)
        
        # 融合结果
        engine = HybridSearchEngine(
            vector_weight=self.vector_weight,
            keyword_weight=self.keyword_weight,
        )
        
        return engine.reciprocal_rank_fusion(
            vector_results, keyword_results, top_n=top_n
        )
```

### 10.4.3 两阶段检索

两阶段检索（粗排 + 精排）在性能和效果之间取得平衡：

```python
class TwoStageRetriever:
    """两阶段检索：粗排（高效）+ 精排（高精度）"""
    
    def __init__(
        self,
        coarse_top_k: int = 100,
        fine_top_k: int = 10,
        embedding_dim: int = 1024,
    ):
        self.coarse_top_k = coarse_top_k
        self.fine_top_k = fine_top_k
        self.embedding_dim = embedding_dim
    
    async def retrieve(
        self,
        query: str,
        index_store,  # 向量索引存储
    ) -> List[dict]:
        """两阶段检索流程"""
        
        # === 第一阶段：粗排 ===
        # 使用高效的近似最近邻搜索（ANN）
        coarse_candidates = await self._coarse_retrieval(
            query, index_store, top_k=self.coarse_top_k
        )
        
        print(f"[粗排] 从全量索引中召回 {len(coarse_candidates)} 个候选")
        
        # === 第二阶段：精排 ===
        # 使用更精确的重新排序模型
        fine_results = await self._fine_ranking(
            query, coarse_candidates, top_k=self.fine_top_k
        )
        
        print(f"[精排] 从 {len(coarse_candidates)} 个候选中选出 {len(fine_results)} 个")
        
        return fine_results
    
    async def _coarse_retrieval(
        self, query: str, index_store, top_k: int
    ) -> List[dict]:
        """
        粗排阶段：使用近似最近邻搜索（ANN）快速召回候选。
        
        常用方法：
        - FAISS IVF（倒排文件索引）
        - HNSW（分层可导航小世界图）
        - LSH（局部敏感哈希）
        """
        # 生成查询嵌入
        query_embedding = await self._encode_query(query)
        
        # 使用FAISS进行ANN搜索
        # 假设index_store包含一个FAISS索引
        distances, indices = index_store.faiss_index.search(
            query_embedding.reshape(1, -1),
            top_k,
        )
        
        candidates = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx != -1:  # FAISS返回-1表示无效索引
                doc = index_store.get_document(int(idx))
                candidates.append({
                    "id": doc["id"],
                    "content": doc["content"],
                    "coarse_score": float(1.0 / (1.0 + dist)),
                    "metadata": doc.get("metadata", {}),
                })
        
        return candidates
    
    async def _fine_ranking(
        self,
        query: str,
        candidates: List[dict],
        top_k: int,
    ) -> List[dict]:
        """
        精排阶段：使用交叉编码器（Cross-Encoder）重新排序。
        
        交叉编码器同时处理查询和文档，计算更精确的相关性得分，
        但速度比双编码器（Bi-Encoder）慢。
        """
        # 对每个候选计算精细的相关性得分
        for candidate in candidates:
            # 方法1：使用交叉编码器模型
            fine_score = await self._cross_encoder_score(
                query, candidate["content"]
            )
            
            # 方法2：融合多种信号（可选）
            # - BM25得分
            # - 语义相似度
            # - 实体覆盖度
            # - 位置权重
            
            candidate["fine_score"] = fine_score
            
            # 融合粗排和精排得分
            candidate["final_score"] = (
                0.3 * candidate["coarse_score"] +
                0.7 * candidate["fine_score"]
            )
        
        # 按最终得分排序
        candidates.sort(key=lambda x: x["final_score"], reverse=True)
        
        return candidates[:top_k]
    
    async def _encode_query(self, query: str) -> np.ndarray:
        """生成查询嵌入向量"""
        # 调用嵌入模型（如bge-m3）
        # 返回shape为(1, embedding_dim)的numpy数组
        pass
    
    async def _cross_encoder_score(
        self, query: str, document: str
    ) -> float:
        """
        使用交叉编码器计算查询-文档相关性得分。
        
        可用的交叉编码器模型：
        - BAAI/bge-reranker-v2-m3
        - cross-encoder/ms-marco-MiniLM-L-6-v2
        """
        # 实际实现中调用reranker模型
        # 此处返回占位值
        import random
        return random.uniform(0, 1)
```

---

## 10.5 Agentic RAG

Agentic RAG将智能体（Agent）引入RAG流程，使系统具备推理、规划和工具调用能力。相比传统RAG的单次检索-生成模式，Agentic RAG能够执行多步推理、自适应选择工具和自我纠错。

### 10.5.1 ReAct模式

ReAct（Reasoning + Acting）模式将推理步骤和行动步骤交织在一起，让Agent在思考过程中动态决定下一步行动。

```python
class ReActAgent:
    """
    ReAct模式智能体：推理(Reasoning) + 行动(Acting)
    
    工作流程：
    1. Thought: 分析当前问题和已有信息
    2. Action: 决定执行哪个工具
    3. Observation: 观察工具执行结果
    4. 重复1-3直到得出最终答案
    """
    
    def __init__(self, llm, tools: List[dict], max_steps: int = 10):
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps
        self.tool_map = {t["name"]: t for t in tools}
    
    async def run(self, query: str) -> str:
        """执行ReAct循环"""
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": query},
        ]
        
        for step in range(self.max_steps):
            # 调用LLM生成下一步（Thought + Action）
            response = await self.llm.chat(messages)
            
            # 检查是否包含最终答案
            if "Final Answer:" in response:
                return response.split("Final Answer:")[-1].strip()
            
            # 解析Action
            action = self._parse_action(response)
            if action is None:
                # 没有有效的Action，继续对话
                messages.append({"role": "assistant", "content": response})
                continue
            
            # 执行Action
            observation = await self._execute_action(action)
            
            # 将Thought-Action-Observation加入对话
            messages.append({"role": "assistant", "content": response})
            messages.append({
                "role": "user",
                "content": f"Observation: {observation}"
            })
        
        return "达到最大步骤数，无法给出确定答案。"
    
    def _build_system_prompt(self) -> str:
        """构建ReAct系统提示词"""
        tool_descriptions = "\n".join([
            f"- {t['name']}: {t['description']}"
            for t in self.tools
        ])
        
        return f"""你是一个具有推理和行动能力的AI助手。你可以使用以下工具：

{tool_descriptions}

请按照以下格式回复：

Thought: 分析当前情况，思考下一步该做什么
Action: 选择要使用的工具名称
Action Input: 工具的输入参数
Observation: 等待工具执行结果（由系统提供）
...（重复 Thought/Action/Observation）
Final Answer: 给出最终答案"""
    
    def _parse_action(self, response: str) -> Optional[dict]:
        """解析Action字段"""
        import re
        
        action_match = re.search(r'Action:\s*(\w+)', response)
        input_match = re.search(r'Action Input:\s*(.+?)(?:\n|$)', response)
        
        if action_match:
            action_name = action_match.group(1)
            action_input = input_match.group(1) if input_match else ""
            return {"name": action_name, "input": action_input}
        
        return None
    
    async def _execute_action(self, action: dict) -> str:
        """执行工具调用"""
        tool_name = action["name"]
        tool_input = action["input"]
        
        if tool_name not in self.tool_map:
            return f"错误：未知工具 '{tool_name}'，可用工具：{list(self.tool_map.keys())}"
        
        tool = self.tool_map[tool_name]
        try:
            result = await tool["fn"](tool_input)
            return str(result)
        except Exception as e:
            return f"工具执行错误：{str(e)}"
```

### 10.5.2 多工具编排

Agentic RAG的核心优势在于能够协调多个专业工具完成复杂任务。以下是GraphRAG-KG项目中使用的多工具编排系统：

```python
class MultiToolOrchestrator:
    """多工具编排器"""
    
    def __init__(self, llm, graph_query_engine, vector_store):
        self.llm = llm
        self.graph_query_engine = graph_query_engine
        self.vector_store = vector_store
        
        # 注册可用工具
        self.tools = [
            {
                "name": "graph_search",
                "description": "在知识图谱中搜索实体和关系，适合查询实体属性、关系路径",
                "fn": self._graph_search,
            },
            {
                "name": "vector_search",
                "description": "在文档库中进行语义搜索，适合查询概念、描述性内容",
                "fn": self._vector_search,
            },
            {
                "name": "calculator",
                "description": "执行数学计算，适合需要精确计算的查询",
                "fn": self._calculator,
            },
            {
                "name": "code_executor",
                "description": "执行Python代码片段并返回结果",
                "fn": self._code_executor,
            },
            {
                "name": "web_search",
                "description": "搜索互联网获取最新信息",
                "fn": self._web_search,
            },
            {
                "name": "sql_query",
                "description": "执行SQL查询，从数据库中获取结构化数据",
                "fn": self._sql_query,
            },
        ]
    
    async def _graph_search(self, input_str: str) -> str:
        """知识图谱搜索工具"""
        # 解析输入（支持JSON格式的参数）
        import json
        try:
            params = json.loads(input_str)
            query = params.get("query", input_str)
            search_type = params.get("type", "local")
        except json.JSONDecodeError:
            query = input_str
            search_type = "local"
        
        # 调用图查询引擎
        result = await self.graph_query_engine.search(
            query=query,
            search_type=search_type,
        )
        return result
    
    async def _vector_search(self, input_str: str) -> str:
        """向量搜索工具"""
        import json
        try:
            params = json.loads(input_str)
            query = params.get("query", input_str)
            top_k = params.get("top_k", 5)
        except json.JSONDecodeError:
            query = input_str
            top_k = 5
        
        results = await self.vector_store.similarity_search(query, k=top_k)
        return "\n".join([
            f"[{i+1}] {r.page_content[:200]}"
            for i, r in enumerate(results)
        ])
    
    async def _calculator(self, input_str: str) -> str:
        """计算器工具"""
        import math
        # 安全执行数学表达式
        allowed_names = {
            k: v for k, v in math.__dict__.items()
            if not k.startswith("__")
        }
        allowed_names.update({"abs": abs, "round": round, "sum": sum})
        
        try:
            result = eval(input_str, {"__builtins__": {}}, allowed_names)
            return f"计算结果：{result}"
        except Exception as e:
            return f"计算错误：{str(e)}"
    
    async def _code_executor(self, input_str: str) -> str:
        """代码执行工具（沙箱环境）"""
        try:
            # 在受限环境中执行
            local_vars = {}
            exec(input_str, {"__builtins__": {}}, local_vars)
            return str(local_vars)
        except Exception as e:
            return f"代码执行错误：{str(e)}"
    
    async def _web_search(self, input_str: str) -> str:
        """网络搜索工具"""
        # 调用搜索API
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"https://api.search.example.com/search",
                params={"q": input_str, "num": 5},
            ) as resp:
                data = await resp.json()
                return "\n".join([
                    f"[{r['title']}]({r['url']}): {r['snippet']}"
                    for r in data.get("results", [])
                ])
    
    async def _sql_query(self, input_str: str) -> str:
        """SQL查询工具"""
        # 实际项目中应使用数据库连接池
        import aiosqlite
        async with aiosqlite.connect(":memory:") as db:
            try:
                cursor = await db.execute(input_str)
                rows = await cursor.fetchall()
                columns = [d[0] for d in cursor.description]
                return "\n".join([
                    ", ".join(str(cell) for cell in row)
                    for row in rows[:20]
                ])
            except Exception as e:
                return f"SQL执行错误：{str(e)}"
    
    async def orchestrate(self, query: str) -> str:
        """执行多工具编排查询"""
        agent = ReActAgent(
            llm=self.llm,
            tools=self.tools,
            max_steps=10,
        )
        return await agent.run(query)
```

### 10.5.3 自纠正与反思

Agentic RAG的一个重要特性是具备自我纠正能力——当发现推理错误或检索结果不充分时，Agent可以自动调整策略。

```python
class SelfCorrectingAgent:
    """具备自纠正能力的Agent"""
    
    def __init__(self, llm, retriever, max_retries: int = 3):
        self.llm = llm
        self.retriever = retriever
        self.max_retries = max_retries
        self.attempt_history = []
    
    async def answer(self, query: str) -> str:
        """带自纠正的问答流程"""
        
        for attempt in range(self.max_retries):
            print(f"[尝试 {attempt + 1}/{self.max_retries}]")
            
            # 1. 检索
            context = await self.retriever.retrieve(query)
            
            # 2. 生成
            answer = await self._generate(query, context)
            
            # 3. 自我评估
            evaluation = await self._evaluate_answer(query, answer, context)
            
            self.attempt_history.append({
                "attempt": attempt,
                "context": context,
                "answer": answer,
                "evaluation": evaluation,
            })
            
            # 4. 判断是否接受
            if evaluation["is_satisfactory"]:
                return answer
            else:
                print(f"  - 不满意原因：{evaluation['reason']}")
                # 根据反馈优化查询策略
                query = await self._refine_query(query, evaluation)
        
        # 所有尝试失败，返回最佳结果
        return self.attempt_history[-1]["answer"]
    
    async def _evaluate_answer(
        self, query: str, answer: str, context: str
    ) -> dict:
        """评估回答质量"""
        eval_prompt = f"""
请评估以下回答的质量：

问题：{query}

参考上下文：
{context[:2000]}

回答：{answer}

评估维度（1-5分）：
1. 准确性：回答是否基于上下文，不包含幻觉信息
2. 完整性：是否全面回答了问题
3. 相关性：回答是否与问题相关
4. 引用：是否明确引用上下文中的信息

请按以下JSON格式输出：
{{
    "accuracy": <分数>,
    "completeness": <分数>,
    "relevance": <分数>,
    "citation": <分数>,
    "is_satisfactory": <true/false>,
    "reason": "<不满意的原因>",
    "missing_info": "<缺少的关键信息>"
}}
"""
        result = await self.llm.chat([{"role": "user", "content": eval_prompt}])
        
        import json
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            return {
                "is_satisfactory": True,
                "reason": "评估解析失败，默认接受",
            }
    
    async def _refine_query(self, original_query: str, evaluation: dict) -> str:
        """根据评估反馈优化查询"""
        refine_prompt = f"""
原始问题：{original_query}

评估反馈：{evaluation.get('reason', '信息不足')}

缺少的信息：{evaluation.get('missing_info', '')}

请生成一个改进版的查询，以获取更准确的信息。
改进查询应该：
1. 针对缺少的信息进行补充
2. 使用更精确的术语
3. 明确需要的信息类型

改进后的查询：
"""
        return await self.llm.chat([{"role": "user", "content": refine_prompt}])
    
    async def _generate(self, query: str, context: str) -> str:
        """基于上下文生成回答"""
        prompt = f"""基于以下上下文信息回答问题。如果上下文中没有相关信息，请明确说明。

上下文：
{context}

问题：{query}

回答："""
        
        result = await self.llm.chat([
            {"role": "system", "content": "你是一个严谨的AI助手，严格基于提供的信息回答问题。"},
            {"role": "user", "content": prompt},
        ])
        return result
```

### 10.5.4 多步推理

复杂问题往往需要分解为多个子问题，逐步求解。多步推理Agent将问题分解为子任务链，依次解决：

```python
class MultiStepReasoningAgent:
    """多步推理Agent"""
    
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever
    
    async def solve(self, question: str) -> str:
        """分解问题并逐步求解"""
        
        # 1. 问题分解
        sub_questions = await self._decompose(question)
        print(f"问题分解：{sub_questions}")
        
        # 2. 逐步求解
        intermediate_results = []
        
        for i, sub_q in enumerate(sub_questions):
            print(f"  子问题 {i+1}: {sub_q}")
            
            # 检索相关信息
            context = await self.retriever.retrieve(sub_q)
            
            # 结合已有结果求解
            sub_answer = await self._solve_subquestion(
                sub_q, context, intermediate_results
            )
            
            intermediate_results.append({
                "question": sub_q,
                "answer": sub_answer,
            })
        
        # 3. 综合所有中间结果
        final_answer = await self._synthesize(
            question, intermediate_results
        )
        
        return final_answer
    
    async def _decompose(self, question: str) -> List[str]:
        """将复杂问题分解为子问题序列"""
        prompt = f"""
将以下复杂问题分解为一系列简单子问题。子问题应该：
1. 可独立回答
2. 按照逻辑顺序排列
3. 每个子问题只包含一个查询点

问题：{question}

请以列表形式输出子问题，每行一个：
"""
        result = await self.llm.chat([{"role": "user", "content": prompt}])
        
        questions = [
            q.strip().lstrip("0123456789. ")
            for q in result.strip().split("\n")
            if q.strip()
        ]
        return questions[:5]  # 最多5个子问题
    
    async def _solve_subquestion(
        self,
        question: str,
        context: str,
        previous_results: List[dict],
    ) -> str:
        """求解单个子问题"""
        prev_context = "\n".join([
            f"Q: {r['question']}\nA: {r['answer']}"
            for r in previous_results
        ])
        
        prompt = f"""
上下文信息：
{context[:3000]}

已有结果：
{prev_context}

当前子问题：{question}

请基于上下文和已有结果回答当前子问题：
"""
        return await self.llm.chat([{"role": "user", "content": prompt}])
    
    async def _synthesize(
        self, original_question: str, results: List[dict]
    ) -> str:
        """综合所有子问题结果生成最终答案"""
        result_summary = "\n".join([
            f"步骤{i+1}：{r['question']}\n结论：{r['answer']}"
            for i, r in enumerate(results)
        ])
        
        prompt = f"""
基于以下逐步推理的结果，回答原始问题。

推理过程：
{result_summary}

原始问题：{original_question}

综合回答：
"""
        return await self.llm.chat([{"role": "user", "content": prompt}])
```

---

## 10.6 前沿趋势

### 10.6.1 Self-RAG

Self-RAG（自我反思式RAG）通过在生成过程中引入反思标记（Reflection Tokens），让模型自主判断是否需要检索以及检索结果是否相关：

```python
class SelfRAG:
    """
    Self-RAG：自我反思式检索增强生成
    
    核心思想：
    - 模型自主决定是否需要检索（通过IsREL标记）
    - 模型评估检索结果的相关性（通过IsSUP标记）
    - 模型只在需要时才使用检索结果
    """
    
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever
    
    async def generate(self, query: str) -> str:
        """Self-RAG生成流程"""
        
        # 1. 判断是否需要检索
        need_retrieval = await self._decide_retrieval(query)
        
        if not need_retrieval:
            # 不需要检索，直接使用模型内部知识
            return await self._generate_without_retrieval(query)
        
        # 2. 检索相关文档
        documents = await self.retriever.retrieve(query, top_k=5)
        
        # 3. 评估每个文档的相关性
        relevant_docs = []
        for doc in documents:
            is_relevant = await self._judge_relevance(query, doc)
            if is_relevant:
                relevant_docs.append(doc)
        
        if not relevant_docs:
            # 没有相关文档，回退到内部知识
            return await self._generate_without_retrieval(query)
        
        # 4. 基于相关文档生成回答
        return await self._generate_with_retrieval(query, relevant_docs)
    
    async def _decide_retrieval(self, query: str) -> bool:
        """判断是否需要检索"""
        prompt = f"""
问题：{query}

是否需要检索外部知识来回答这个问题？
只需要回答"是"或"否"：
"""
        result = await self.llm.chat([{"role": "user", "content": prompt}])
        return result.strip().startswith("是")
    
    async def _judge_relevance(self, query: str, doc: str) -> bool:
        """评估文档与查询的相关性"""
        prompt = f"""
问题：{query}

文档片段：{doc[:500]}

这个文档是否与问题相关？
只需要回答"相关"或"不相关"：
"""
        result = await self.llm.chat([{"role": "user", "content": prompt}])
        return result.strip().startswith("相关")
```

### 10.6.2 CRAG

CRAG（Corrective RAG）在发现检索结果质量不高时，自动触发纠错机制——包括重新检索、查询改写或网络搜索：

```python
class CRAG:
    """
    CRAG：纠正式RAG
    
    核心思想：
    1. 对检索结果进行质量评估
    2. 质量好 → 直接使用
    3. 质量中等 → 查询改写后重新检索
    4. 质量差 → 触发网络搜索或其他备选策略
    """
    
    async def retrieve_with_correction(
        self, query: str, retriever, web_search_fn
    ) -> List[dict]:
        """带纠错的检索流程"""
        
        # 初始检索
        results = await retriever.retrieve(query, top_k=10)
        
        # 评估检索质量
        quality = await self._assess_quality(query, results)
        
        if quality == "good":
            return results
        
        elif quality == "medium":
            # 改写查询后重新检索
            rewritten_query = await self._rewrite_query(query)
            return await retriever.retrieve(rewritten_query, top_k=10)
        
        else:  # quality == "poor"
            # 触发网络搜索
            web_results = await web_search_fn(query)
            # 将网络结果格式化为统一格式
            return self._format_web_results(web_results)
    
    async def _assess_quality(
        self, query: str, results: List[dict]
    ) -> str:
        """评估检索结果质量"""
        if not results:
            return "poor"
        
        # 计算平均相关性得分
        scores = [r.get("score", 0) for r in results]
        avg_score = sum(scores) / len(scores)
        
        if avg_score > 0.7:
            return "good"
        elif avg_score > 0.4:
            return "medium"
        else:
            return "poor"
```

### 10.6.3 RAPTOR

RAPTOR（Recursive Abstractive Processing for Tree-Organized Retrieval）通过递归摘要构建文档树，在检索时从粗到精选择合适粒度的信息：

```python
class RAPTOR:
    """
    RAPTOR：递归摘要树检索
    
    核心思想：
    1. 将文档分割为短文本块
    2. 对语义相近的块进行聚类
    3. 为每个聚类生成摘要
    4. 递归进行聚类和摘要，构建树结构
    5. 检索时从根节点开始，根据相关性向下遍历
    """
    
    def __init__(self, llm, embedding_model):
        self.llm = llm
        self.embedding_model = embedding_model
        self.tree = None
    
    async def build_tree(self, documents: List[str]):
        """构建递归摘要树"""
        # 初始化叶子节点
        leaves = [
            {"content": doc, "embedding": await self._embed(doc)}
            for doc in documents
        ]
        
        current_level = leaves
        self.tree = [current_level]
        
        # 递归聚类和摘要
        while len(current_level) > 1:
            # 聚类
            clusters = await self._cluster(current_level)
            
            # 为每个聚类生成摘要
            next_level = []
            for cluster in clusters:
                summary = await self._summarize(cluster)
                next_level.append({
                    "content": summary,
                    "embedding": await self._embed(summary),
                    "children": cluster,
                })
            
            self.tree.append(next_level)
            current_level = next_level
        
        return self.tree
    
    async def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        """从树中检索相关信息"""
        query_embedding = await self._embed(query)
        
        # 从顶层开始遍历
        results = []
        current_nodes = [self.tree[-1][0]] if self.tree else []
        
        for level in range(len(self.tree) - 1, -1, -1):
            # 计算当前层节点与查询的相似度
            scored = [
                (node, self._cosine_sim(query_embedding, node["embedding"]))
                for node in current_nodes
            ]
            scored.sort(key=lambda x: x[1], reverse=True)
            
            # 选择Top-K节点
            top_nodes = scored[:top_k]
            
            if level == 0:
                # 叶子节点：直接返回
                for node, score in top_nodes:
                    results.append(node["content"])
            else:
                # 非叶子节点：继续向下遍历
                current_nodes = [
                    child for node, _ in top_nodes
                    for child in node.get("children", [])
                ]
        
        return results[:top_k]
```

### 10.6.4 HippoRAG

HippoRAG受海马体记忆机制的启发，将长期记忆引入RAG系统，使得检索能够在不同会话间保持一致性：

```python
class HippoRAG:
    """
    HippoRAG：海马体启发的长期记忆RAG
    
    核心机制：
    - 情景记忆（Episodic Memory）：存储过去的查询和交互
    - 语义记忆（Semantic Memory）：存储知识图谱结构
    - 记忆整合：新信息与已有知识的整合
    """
    
    def __init__(self, llm, knowledge_graph):
        self.llm = llm
        self.knowledge_graph = knowledge_graph
        self.episodic_memory = []  # 情景记忆
        self.semantic_memory = {}   # 语义记忆
    
    async def retrieve_with_memory(
        self, query: str, top_k: int = 5
    ) -> List[str]:
        """结合长期记忆的检索"""
        
        # 1. 从情景记忆中检索相关历史
        similar_episodes = self._search_episodic_memory(query)
        
        # 2. 从知识图谱中检索相关实体
        graph_context = await self._search_knowledge_graph(query)
        
        # 3. 综合评分
        all_results = []
        
        for episode in similar_episodes:
            all_results.append({
                "content": episode["answer"],
                "score": episode["similarity"] * 0.3,
                "source": "episodic",
            })
        
        for entity in graph_context:
            all_results.append({
                "content": entity["description"],
                "score": entity["score"] * 0.4,
                "source": "graph",
            })
        
        # 4. 记忆整合与更新
        self._consolidate_memory(query, all_results)
        
        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results[:top_k]
    
    def _consolidate_memory(self, query: str, results: List[dict]):
        """记忆整合：将新的交互整合到长期记忆中"""
        self.episodic_memory.append({
            "query": query,
            "timestamp": time.time(),
            "results": results[:3],
        })
        
        # 记忆压缩：保留最近的100条记录
        if len(self.episodic_memory) > 100:
            self.episodic_memory = self.episodic_memory[-100:]
```

---

## 10.7 本章小结

本章深入探讨了三种高级RAG范式及其实现细节：

1. **GraphRAG**通过实体关系图谱和社区检测实现了对复杂语义关系的建模，在全局性问题和多文档推理任务上表现优异。关键优化点包括实体提取的提示词调优、社区检测参数调优和查询策略的智能路由。

2. **层次化检索**通过文档树结构和渐进式上下文构建实现了从粗到精的检索，在长文档处理和高精度检索场景中优势明显。

3. **混合搜索**通过向量与关键词的加权融合以及自适应权重调整，在不同查询类型下都能取得最优效果。两阶段检索策略（粗排+精排）在效率和精度之间取得了良好平衡。

4. **Agentic RAG**通过ReAct模式、多工具编排、自纠正和多步推理，实现了更接近人类思维方式的检索增强生成。这是RAG系统从"工具"向"智能助手"演进的关键一步。

5. **前沿趋势**如Self-RAG、CRAG、RAPTOR和HippoRAG代表了RAG技术的最新发展方向，它们分别从自我反思、纠错机制、递归摘要和长期记忆等角度推动着RAG能力的边界。

在下一章中，我们将讨论如何在实际工程环境中落地这些技术，包括团队协作、API设计、监控运维等工程实践。
