# GraphRAG-KG 项目架构文档

## 1. 项目概述

GraphRAG-KG 是一个基于 **Microsoft GraphRAG** 库构建的知识图谱问答系统，采用 **Neo4j + LanceDB** 混合存储架构。系统从多格式文档中提取实体和关系，构建知识图谱，并支持自然语言查询。

### 1.1 核心能力

- **多格式文档解析**：支持 PDF、Markdown、TXT、HTML 四种格式
- **知识图谱构建**：实体提取、关系抽取、社区检测、向量嵌入
- **混合存储**：Neo4j（图数据）+ LanceDB（向量嵌入）+ Parquet（备份/导出）
- **多模式查询**：Local、Global、DRIFT、Basic 四种搜索方法
- **REST API**：FastAPI 服务，支持 HTTP 查询

### 1.2 技术栈

| 组件 | 技术 | 用途 |
|---|---|---|
| 核心引擎 | Microsoft GraphRAG 3.x | 索引管道、查询引擎 |
| 图数据库 | Neo4j 5.22.0 | 实体、关系、社区存储 |
| 向量存储 | LanceDB | 文本/实体/社区嵌入 |
| 备份格式 | Apache Parquet | 索引输出、数据交换 |
| CLI 框架 | Typer + Rich | 命令行界面 |
| REST API | FastAPI + Uvicorn | HTTP 服务 |
| 文档解析 | PyMuPDF + BeautifulSoup4 | PDF/HTML 解析 |
| 配置管理 | Pydantic + PyYAML | 配置模型和验证 |
| 测试数据 | Faker + Jinja2 | 合成数据生成 |

---

## 2. 系统架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CLI Layer (Typer)                            │
│  data  │  init  │  ingest  │  index  │  graph  │  query  │  serve  │
└────────┴────────┴──────────┴─────────┴─────────┴─────────┴─────────┘
         │        │          │         │         │          │
         ▼        ▼          ▼         ▼         ▼          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        Application Layer                            │
│  ┌─────────┐ ┌──────────┐ ┌────────┐ ┌──────────┐ ┌───────────┐  │
│  │  Data   │ │  Ingest  │ │ Index  │ │  Graph   │ │  Query    │  │
│  │Generator│ │  Parser  │ │ Runner │ │  Sync    │ │  Engine   │  │
│  └─────────┘ └──────────┘ └────────┘ └──────────┘ └───────────┘  │
└─────────────────────────────────────────────────────────────────────┘
         │              │          │         │              │
         ▼              ▼          ▼         ▼              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        Storage Layer                                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────────────┐  │
│  │  Neo4j   │  │ LanceDB  │  │ Parquet  │  │  File System      │  │
│  │  Graph   │  │ Vectors  │  │ Backup   │  │  (input/output)   │  │
│  └──────────┘  └──────────┘  └──────────┘  └───────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.1 模块职责

| 模块 | 文件 | 职责 |
|---|---|---|
| `cli/` | `main.py` | CLI 入口，注册所有子命令 |
| `core/` | `config.py` | 配置模型（KGConfig、Neo4jConfig 等） |
| `core/` | `config_loader.py` | YAML 加载、环境变量替换、配置合并 |
| `core/` | `project.py` | 项目初始化、目录结构管理 |
| `core/` | `pipeline.py` | 全流程编排器（ingest→index→sync） |
| `data/` | `generator.py` | 测试数据生成器 |
| `data/` | `entities.py` | 实体工厂（Faker + 中文实体池） |
| `data/` | `relationships.py` | 关系工厂（规则驱动） |
| `data/` | `queries.py` | 测试问题工厂 |
| `data/` | `ground_truth.py` | Ground truth 模型 + JSON 序列化 |
| `ingest/` | `parsers.py` | PDF/MD/TXT/HTML 解析器 |
| `ingest/` | `loader.py` | 文件发现 + 批量加载 |
| `ingest/` | `converter.py` | 转换为 graphrag 输入格式 |
| `index/` | `runner.py` | graphrag 索引管道封装 |
| `index/` | `monitor.py` | Rich 进度监控 |
| `index/` | `updater.py` | 增量更新 |
| `graph/` | `connection.py` | Neo4j 驱动 + 连接池 |
| `graph/` | `schema.py` | 索引/约束管理 |
| `graph/` | `sync.py` | Parquet → Neo4j 批量同步 |
| `graph/` | `queries.py` | Cypher 查询库 |
| `graph/` | `traversal.py` | 图遍历（ego network、路径、影响分析） |
| `query/` | `engine.py` | QueryEngine 门面 + 自动路由 |
| `query/` | `local.py` | 本地搜索（Neo4j 增强） |
| `query/` | `global_search.py` | 全局搜索（社区报告 map-reduce） |
| `query/` | `drift.py` | DRIFT 搜索（层次化遍历） |
| `query/` | `basic.py` | 基础向量搜索 |
| `query/` | `context.py` | 引用提取 + 响应构建 |
| `query/` | `cache.py` | 查询结果缓存（TTL） |
| `storage/` | `parquet_store.py` | Parquet 文件读写 |
| `storage/` | `vector_store.py` | LanceDB 向量存储 |
| `api/` | `app.py` | FastAPI 应用 |
| `api/` | `models.py` | Pydantic 请求/响应模型 |
| `api/` | `middleware.py` | CORS + 日志中间件 |
| `prompts/` | `templates.py` | 提示词注册表 |

---

## 3. 项目结构

```
graphRAG/
├── src/graphrag_kg/          # 源代码
│   ├── cli/commands/         # 8 个 CLI 命令
│   ├── core/                 # 核心配置和项目管理
│   ├── data/                 # 测试数据生成器
│   ├── ingest/               # 文档导入
│   ├── index/                # 索引管道
│   ├── graph/                # Neo4j 图操作
│   ├── query/                # 查询引擎
│   ├── storage/              # 存储层
│   ├── api/                  # REST API
│   ├── prompts/              # LLM 提示词模板
│   └── utils/                # 工具函数
├── config/                   # 配置模板
├── tests/                    # 测试套件（149 个测试）
├── examples/                 # 使用示例
└── documents/                # 本文档
```

---

## 4. 配置模型

### 4.1 配置层次

```
环境变量 (.env)  ←  最低优先级
    ↓
YAML 文件 (settings.yaml)
    ↓
配置模板 (default/fast/production)
    ↓
CLI 参数覆盖  ←  最高优先级
```

### 4.2 核心配置类

```
KGConfig
├── project_name: str
├── neo4j: Neo4jConfig
│   ├── uri, username, password, database
│   ├── max_connection_pool_size, sync_batch_size
│   └── sync_create_indexes, store_embeddings
├── ingestion: IngestionConfig
│   ├── source_directories, file_patterns
│   ├── encoding, clean_html, extract_metadata
│   └── max_file_size_mb
├── pipeline: PipelineConfig
│   ├── auto_index_on_ingest, incremental
│   ├── backup_previous_output, max_workers
├── query: QueryConfig
│   ├── default_method, response_type
│   ├── max_context_tokens, include_sources
│   └── streaming, cache_ttl_seconds
├── chat_model, embedding_model
├── api_key, api_base
└── root_dir, input_dir, output_dir, ...
```

### 4.3 配置模板对比

| 模板 | Chat 模型 | Embedding | 查询方法 | 分块大小 | 适用场景 |
|---|---|---|---|---|---|
| `default` | gpt-4.1 | text-embedding-3-large | local | 1200 tokens | 通用 |
| `fast` | gpt-4.1-mini | text-embedding-3-small | basic | 600 tokens | 快速迭代 |
| `production` | gpt-4.1 | text-embedding-3-large | drift | 1200 tokens | 生产环境 |

---

## 5. 测试数据场景

| 场景 | 实体数 | 关系数 | 查询跳数 | 实体类型 | 用途 |
|---|---|---|---|---|---|
| `pharma_supply_chain` | 59 | 153 | 3-5 | 12 种 | 多跳供应链推理 |
| `tech_company` | 21 | 34 | 1-2 | 5 种 | 基础实体关系提取 |

### 5.1 pharma_supply_chain 场景

模拟完整的药品供应链：**原料药供应商 → 药品生产商 → 分销商 → 医院药房 → 临床科室 → 患者治疗**

实体类型：制药公司、药品、原料药、分销商、医院、临床科室、区域、监管机构、关键人物、药品批文、适应症、供应合同

关系网络包含监管层、生产层、流通层、使用层四个层次，支持 3-5 跳的多跳推理查询。
