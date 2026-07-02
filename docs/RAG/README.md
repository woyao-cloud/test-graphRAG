# RAG 实战指南 —— 从入门到企业级部署

## 内容概览

本书系统性地介绍检索增强生成（RAG）技术的原理、实践与落地，共分为 **7 大部分、22 章**，覆盖从基础概念到企业级部署的完整知识体系。

---

## 第一部分：RAG 基础（1-3 章）

| 章节 | 内容 | 代码 |
|---|---|---|
| [第 1 章：RAG 概述与优势](chapter-01-intro.md) | RAG 的定义、起源、核心架构和应用价值 | - |
| [第 2 章：实现原理与传统 RAG 区别](chapter-02-principles.md) | 文档加载、文本分块、向量嵌入、检索生成的完整流程 | [demos/ch02](demos/ch02/basic_rag_pipeline.py) |
| [第 3 章：典型应用场景](chapter-03-scenarios.md) | 智能客服、医疗、金融、法律等十大场景分析 | - |

## 第二部分：工程基础（4-6 章）

| 章节 | 内容 | 代码 |
|---|---|---|
| [第 4 章：多源文档处理管线](ch04-document-processing.md) | 多格式解析、数据清洗、编码检测、文档切分策略 | [demos/ch04](demos/ch04/document_processing_pipeline.py) |
| [第 5 章：向量存储与检索](ch05-vector-storage.md) | 嵌入模型选型、向量数据库对比、索引构建策略 | - |
| [第 6 章：知识库构建](ch06-knowledge-base.md) | 数据源接入、文档质量评估、版本管理、维护监控 | - |

## 第三部分：检索与增强（7-9 章）

| 章节 | 内容 | 代码 |
|---|---|---|
| [第 7 章：多路召回管线](ch07-multi-recall.md) | 稠密检索、BM25、KG 检索、结构化检索、RRF 融合 | [demos/ch07](demos/ch07/multi_recall_pipeline.py) |
| [第 8 章：高级索引策略](ch08-advanced-indexing.md) | 向量量化、HNSW/IVF 索引、过滤预检、增量合并 | - |
| [第 9 章：知识图谱检索](ch09-kg-retrieval.md) | 实体链接、EGO 网络遍历、最短路径、KG 增强检索 | [demos/ch09](demos/ch09/kg_retrieval.py) |

## 第四部分：高级 RAG 架构（10-13 章）

| 章节 | 内容 | 代码 |
|---|---|---|
| [第 10 章：GraphRAG 深度实践](ch10-graphrag-deepseek.md) | 索引流水线、社区检测、Local/Global/DRIFT 搜索 | [demos/ch10](demos/ch10/graphrag_deepseek.py) |
| [第 11 章：层级检索与混合检索](ch11-hybrid-search.md) | 稠密+稀疏融合、Small-to-Big、Step-back Prompting | [demos/ch11](demos/ch11/hybrid_search.py) |
| [第 12 章：Agentic RAG](ch12-agentic-rag.md) | ReAct、CRAG、Self-RAG、多 Agent 协作 | [demos/ch12](demos/ch12/agentic_rag.py) |
| [第 13 章：知识图谱构建与应用落地](ch13-kg-construction.md) | Schema 设计、实体/关系抽取、质量检查、混合查询 | [demos/ch13](demos/ch13/kg_construction.py) |

## 第五部分：工程化（14-16 章）

| 章节 | 内容 | 代码 |
|---|---|---|
| [第 14 章：性能与内存管理](ch14-performance.md) | 缓存策略、并发执行、连接池、Token 预算、监控 | [demos/ch14](demos/ch14/performance_memory.py) |
| [第 15 章：RAG 评估体系建设](ch15-evaluation.md) | 评估维度、自动化指标、人工评估、在线 A/B 测试 | [demos/ch15](demos/ch15/evaluation_framework.py) |
| [第 16 章：典型问题处理方法](ch16-troubleshooting.md) | 幻觉、检索失败、长上下文、安全等问题的系统排查 | - |

## 第六部分：落地实战（17-20 章）

| 章节 | 内容 | 代码 |
|---|---|---|
| [第 17 章：跨团队协作与落地推进](ch17-collaboration.md) | 团队角色、评估标准、持续优化、风险管理 | - |
| [第 18 章：端到端 RAG 系统实战](ch18-end-to-end-rag.md) | 从需求分析到线上部署的完整项目流程 | [demos/ch18](demos/ch18/end_to_end_rag.py) |
| [第 19 章：客服领域 RAG 实践](ch19-customer-service.md) | 客服场景的知识库构建、多轮对话、人机协作 | - |
| [第 20 章：内部知识管理 RAG 实践](ch20-internal-km.md) | 企业知识库建设、权限管理、搜索增强 | - |

## 第七部分：拓展与实践（21-22 章）

| 章节 | 内容 | 代码 |
|---|---|---|
| [第 21 章：BI 分析与流程自动化 RAG 实践](ch21-bi-automation.md) | NL2SQL、时序分析、定时报表、异常预警 | - |
| [第 22 章：GraphRAG + DeepSeek 集成实战](ch22-graphrag-integration.md) | 系统架构、配置、索引/查询流程 | [demos/ch10](demos/ch10/graphrag_deepseek.py) |

---

## 代码使用说明

所有 Demo 代码位于 `demos/chNN/` 目录下，每个目录可独立运行。代码遵循"OpenAI API → Ollama API → Mock"的回退策略，无需任何外部依赖即可运行。

```bash
# 运行示例
python demos/ch10/graphrag_deepseek.py
python demos/ch11/hybrid_search.py --query "抗肿瘤药物"
python demos/ch18/end_to_end_rag.py
```

---

## 配套项目

本书配套的 GraphRAG-KG 开源项目提供了完整的知识图谱问答系统实现，支持 Neo4j + LanceDB 混合存储，可通过 CLI 工具快速搭建生产级 RAG 应用。详见项目根目录的 README.md。
