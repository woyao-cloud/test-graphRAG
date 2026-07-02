# 第 22 章：GraphRAG + DeepSeek 集成实战

## 22.1 集成背景

GraphRAG 是微软研究院推出的知识图谱增强 RAG 框架，DeepSeek 是深度求索推出的高性能大语言模型。将两者结合，可以构建一个端到端的知识图谱问答系统：GraphRAG 负责从文档中自动构建知识图谱并执行结构化查询，DeepSeek 作为底层的 LLM 驱动整个流程。

## 22.2 系统整体架构

整个系统的数据流可以分为三个阶段：

**索引阶段**：
1. 文档输入 → 文档处理管线（解析、清洗、切分）
2. 文本单元 → LLM 实体/关系抽取（使用 DeepSeek）
3. 实体关系 → 知识图谱构建 → 社区检测 → 社区摘要

**查询阶段**：
1. 用户问题 → 查询模式自动选择（Local/Global/DRIFT/Basic）
2. 检索结果 → 上下文组装 → 答案生成（使用 DeepSeek）

**应用阶段**：
1. API 服务层（FastAPI）
2. Web 管理界面
3. 监控与日志系统

## 22.3 核心配置

集成 GraphRAG 与 DeepSeek 的关键是正确配置 settings.yaml：

```yaml
# 使用 DeepSeek 作为所有 LLM 调用的后端
llm:
  api_key: ${GRAPHRAG_API_KEY}
  model: deepseek-chat
  api_base: https://api.deepseek.com/v1
  max_tokens: 4096
  temperature: 0.0  # 实体抽取使用零温度保证确定性

# 嵌入模型配置
embeddings:
  llm:
    api_key: ${GRAPHRAG_API_KEY}
    model: text-embedding-ada-002  # 使用 OpenAI 嵌入
    api_base: https://api.openai.com/v1
```

## 22.4 索引流程详解

### 22.4.1 文档处理

在 GraphRAG 的索引启动前，需要确保文档已被正确处理。使用 `graphrag-kg ingest run` 命令将原始文档转换为 GraphRAG 可处理的输入格式。

```bash
graphrag-kg ingest run --source ./documents
graphrag-kg index run --method standard
```

### 22.4.2 索引参数优化

对于使用 DeepSeek 的索引，推荐以下参数配置：

- **实体抽取温度**：0.0。实体抽取是确定性任务，不需要创造性输出。
- **Gleaning 轮数**：1-2。DeepSeek 的抽取能力较强，1 轮 Gleaning 通常已经足够。
- **分块大小**：1200 tokens。块大小影响实体共现关系的质量。
- **社区检测层级**：3 层（细粒度、中粒度、粗粒度）。

### 22.4.3 实体与关系抽取

GraphRAG 使用 DeepSeek 对每个文本单元进行实体和关系抽取。抽取提示词中包含 Schema 定义和 Few-shot 示例，引导模型提取符合要求的结构化信息。

```python
# 通过 DeepSeek API 调用的实体抽取示例
response = openai.ChatCompletion.create(
    model="deepseek-chat",
    messages=[{
        "role": "user",
        "content": f"从以下文本中提取实体和关系...\n\n{text_unit}"
    }],
    temperature=0.0
)
```

## 22.5 查询流程详解

### 22.5.1 Local Search

Local Search 模式适用于具体事实查询。查询流程：
1. 用户输入 → DeepSeek 嵌入 → 向量检索匹配相关实体
2. 图遍历 EGO 网络 → 收集邻居实体和关系
3. DeepSeek 基于结构化上下文生成答案

### 22.5.2 Global Search

Global Search 模式适用于总结性问题。使用 DeepSeek 进行社区摘要的语义匹配和 Map-Reduce 答案生成。

### 22.5.3 DRIFT Search

DRIFT Search 模式适用于多跳推理。DeepSeek 在路径遍历和答案生成两个环节提供推理支持。

### 22.5.4 查询优化建议

- **缓存策略**：开启查询结果缓存，重复问题直接返回缓存
- **温度设置**：事实查询用温度 0.0，创意分析用温度 0.3-0.7
- **上下文窗口**：DeepSeek 支持 32K 上下文，可以容纳多个社区摘要和路径信息

## 22.6 使用 DeepSeek v4 Flash 的注意事项

DeepSeek v4 Flash 是 DeepSeek 的高性能推理模型，与 GraphRAG 集成时需注意：

- **API 格式兼容性**：DeepSeek 的 API 兼容 OpenAI 格式，可以直接通过配置 `api_base` 接入
- **速率限制**：注意 API 调用的速率限制，推荐在 settings.yaml 中配置合理的并发数和重试策略
- **成本控制**：DeepSeek v4 Flash 的定价较低，但仍需通过 Token 预算管理来控制成本

## 22.7 本章小结

GraphRAG + DeepSeek 的集成构建了一个完整的端到端知识图谱问答系统。本章从系统架构、核心配置、索引流程、查询流程等方面详细介绍了集成方案。通过这种组合，开发者可以利用 GraphRAG 的知识图谱构建能力和 DeepSeek 的高性能推理能力，快速搭建面向复杂知识库的智能问答系统。

完整的可运行代码示例见 `demos/ch10/graphrag_deepseek.py`（简化版 GraphRAG 流水线）。
