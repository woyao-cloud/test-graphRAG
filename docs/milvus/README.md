# Milvus向量数据库理论与RAG实战

> **从底层原理到工业级检索增强落地**

![Milvus](https://img.shields.io/badge/Milvus-2.4+-00A1EA)
![Spring Boot](https://img.shields.io/badge/Spring%20Boot-3.3-6DB33F)
![Java](https://img.shields.io/badge/Java-17-ED8B00)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 书籍简介

《Milvus向量数据库理论与RAG实战：从底层原理到工业级检索增强落地》是一本系统讲解 **Milvus向量数据库** 与 **RAG（检索增强生成）** 技术的书籍。全书共分七大部分，从基础概念到工业级实践，覆盖了 Milvus 的核心原理、部署运维、生态集成以及 RAG 系统的完整构建流程。

本书配套 Java Spring Boot 演示代码，读者可直接运行体验 Milvus 与 RAG 的完整链路。

---

## 目录结构

```
docs/
├── Milvus/          # 本书 README 与相关文档
└── milvus/          # 书籍各章节 Markdown 源文件
    ├── plan.md           # 编写计划
    ├── part1-intro.md    # 第一部分：向量数据库基础
    ├── part2-milvus.md   # 第二部分：Milvus 核心原理
    ├── part3-ops.md      # 第三部分：部署与运维
    ├── part4-ecosystem.md # 第四部分：生态与集成
    ├── part5-rag.md      # 第五部分：RAG 实战
    ├── part6-advanced.md # 第六部分：高级主题
    └── part7-case.md     # 第七部分：案例与展望

documents/
└── milvs/
    └── milvus-docker-compose.yml   # Milvus 单机部署 Docker Compose 配置

java/spring-rag-demo/     # Java Spring Boot RAG 演示项目
```

---

## 七大部分概览

### 第一部分：向量数据库基础

> [part1-intro.md](../milvus/part1-intro.md)

介绍向量数据库的基本概念，包括：
- 向量与嵌入（Embedding）的原理
- 向量检索的核心技术（KNN、ANN、IVF、HNSW）
- 向量数据库与传统数据库的区别
- 主流向量数据库对比（Milvus、Pinecone、Qdrant、Weaviate）

### 第二部分：Milvus 核心原理

> [part2-milvus.md](../milvus/part2-milvus.md)

深入 Milvus 的内部架构：
- Milvus 整体架构设计（Proxy、Coordinator、Worker、DataNode、QueryNode）
- 数据模型与字段类型
- 索引类型详解（IVF_FLAT、IVF_SQ8、HNSW、DiskANN）
- 距离度量方式（L2、IP、COSINE）
- 分区（Partition）与分片（Sharding）机制
- 向量标量混合检索（混合搜索）

### 第三部分：部署与运维

> [part3-ops.md](../milvus/part3-ops.md)

Milvus 的生产环境部署指南：
- Docker Compose 单机部署（[配置文件](../milvs/milvus-docker-compose.yml)）
- Kubernetes Helm 集群部署
- Milvus Operator 管理
- 性能调优与监控
- 备份与恢复策略
- 安全配置（TLS、RBAC）

### 第四部分：生态与集成

> [part4-ecosystem.md](../milvus/part4-ecosystem.md)

Milvus 与周边工具的集成：
- Attu —— Milvus 可视化管理工具
- Milvus CDC —— 数据变更捕获
- 与 Elasticsearch 的对比与互补
- 与 OpenAI、HuggingFace 等 LLM 平台的集成
- 与 LangChain、LlamaIndex 等框架的整合

### 第五部分：RAG 实战

> [part5-rag.md](../milvus/part5-rag.md)

基于 Milvus 构建端到端 RAG 系统：
- RAG 架构概述（Indexing -> Retrieval -> Generation）
- 文档解析与切分策略
- 向量化与索引构建
- 语义检索与重排序
- LLM 生成与上下文增强
- **配套代码**：`java/spring-rag-demo/` 完整 Spring Boot 项目

### 第六部分：高级主题

> [part6-advanced.md](../milvus/part6-advanced.md)

进阶话题探讨：
- 多向量检索与混合搜索
- 过滤向量搜索（Filtered Search）
- 分组搜索（Grouping Search）
- Range Search 与 Iterator
- 批量操作与流式写入
- 多租户架构设计
- GPU 加速与性能优化

### 第七部分：案例与展望

> [part7-case.md](../milvus/part7-case.md)

行业案例分析与未来展望：
- Milvus 在推荐系统中的应用
- Milvus 在问答系统中的应用
- Milvus 在图片/视频检索中的应用
- Milvus 在药物发现中的应用
- 向量数据库 + LLM Agent 的未来趋势
- RAG 技术演进（Graph RAG、Agentic RAG）

---

## 演示代码使用指南

本项目提供了一个 **Java Spring Boot** 演示应用，位于 `java/spring-rag-demo/`，演示了基于 Milvus 的 RAG 完整链路。

### 技术栈

| 组件 | 技术 |
|---|---|
| 开发语言 | Java 17 |
| 框架 | Spring Boot 3.3 |
| 向量数据库 | Milvus 2.4+ |
| 构建工具 | Maven |
| 嵌入模型 | BGE-M3 / text2vec (可配置) |

### 项目结构

```
java/spring-rag-demo/
├── pom.xml                          # Maven 依赖配置
├── src/main/java/com/example/rag/
│   ├── RagApplication.java          # Spring Boot 启动类
│   ├── config/
│   │   └── MilvusConfig.java        # Milvus 连接配置
│   ├── controller/
│   │   └── RagController.java       # REST API 控制器
│   ├── dto/
│   │   ├── SearchRequest.java       # 搜索请求 DTO
│   │   └── SearchResult.java        # 搜索结果 DTO
│   ├── model/
│   │   └── Document.java            # 文档实体模型
│   └── service/
│       ├── MilvusService.java       # Milvus 向量操作服务
│       └── VectorService.java       # 向量化与嵌入服务
└── src/main/resources/
    └── application.yml              # 应用配置文件
```

---

## 快速开始

### 前置条件

- JDK 17+
- Docker & Docker Compose
- Maven 3.8+

### 第一步：启动 Milvus

```bash
# 使用项目中提供的 Docker Compose 文件启动 Milvus
docker compose -f documents/milvs/milvus-docker-compose.yml up -d
```

### 第二步：配置应用

编辑 `java/spring-rag-demo/src/main/resources/application.yml`，确认 Milvus 连接配置：

```yaml
milvus:
  host: localhost
  port: 19530
  collection: document_embeddings
  dimension: 768   # 根据嵌入模型调整
```

### 第三步：运行演示应用

```bash
cd java/spring-rag-demo
mvn spring-boot:run
```

### 第四步：测试 API

应用启动后，访问 REST API：

```bash
# 添加文档
curl -X POST http://localhost:8080/api/rag/documents \
  -H "Content-Type: application/json" \
  -d '{"id": "doc1", "content": "Milvus 是一款高性能向量数据库", "metadata": {"source": "book"}}'

# 搜索相似文档
curl -X POST http://localhost:8080/api/rag/search \
  -H "Content-Type: application/json" \
  -d '{"query": "向量数据库", "topK": 5}'
```

### API 端点

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/rag/documents` | 添加文档到 Milvus |
| POST | `/api/rag/search` | 基于向量相似度搜索 |
| GET  | `/api/rag/collections` | 查看集合状态 |

---

## Docker Compose 快速部署

项目根目录和 `documents/milvs/` 下均提供 Docker Compose 文件：

| 文件 | 说明 |
|---|---|
| `docker-compose.yml` | 根目录：包含 MinIO、Etcd、Milvus Standalone |
| `documents/milvs/milvus-docker-compose.yml` | Milvus 单机快速启动配置 |

```bash
# 方式一：使用根目录配置
docker compose up -d

# 方式二：使用独立 Milvus 配置
docker compose -f documents/milvs/milvus-docker-compose.yml up -d
```

---

## 许可

本项目采用 MIT 许可证。
