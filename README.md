# GraphRAG-KG: Knowledge Graph Q&A System

基于 Microsoft `graphrag` 库构建的知识图谱问答系统，采用 **Neo4j + Milvus** 混合存储架构。

## Architecture

```
Documents (PDF/MD/TXT/HTML)
  → Ingest → graphrag Index → Parquet + Milvus
  → Neo4j Sync → Cypher Graph Traversal
  → Query Engine (Local/Global/DRIFT/Basic)
  → Grounded Answers + Citations + Graph Context
```

---

## 完整工作流

以下按顺序执行每个命令，从零搭建一个可查询的知识图谱系统。

---

### 1. 安装

```bash
pip install -e .
```

| 项目 | 说明 |
|---|---|
| **前置条件** | Python 3.10-3.12 |
| **副作用** | 安装 `graphrag-kg` 命令到系统 PATH；安装所有依赖 |
| **幂等性** | 是，重复执行不会重复安装 |

---

### 2. 启动 Docker 服务

```bash
docker-compose up -d
```

这会启动以下服务：

| 服务 | 端口 | 用途 |
|---|---|---|
| Neo4j | 7687 (Bolt), 7474 (HTTP) | 图数据库 |
| Milvus | 19530 (gRPC) | 向量数据库 |
| etcd | 2379 | Milvus 元数据存储 |
| MinIO | 9000 (API), 9001 (Console) | Milvus 底层存储 |

| 项目 | 说明 |
|---|---|
| **默认凭据** | Neo4j: `neo4j / password`；MinIO: `minioadmin / minioadmin` |
| **验证** | Neo4j: 浏览器访问 http://localhost:7474 ；Milvus: `python -c "from pymilvus import connections, utility; connections.connect(); print(utility.list_collections())"` |

---

### 3. 生成测试数据

```bash
# 生成医药供应链场景（强关联关系，推荐）
graphrag-kg data generate --scenario pharma_supply_chain

# 生成科技公司场景
graphrag-kg data generate --scenario tech_company

# 生成全部场景
graphrag-kg data generate --scenario all

# 自定义参数
graphrag-kg data generate --scenario pharma_supply_chain --seed 123 --entity-count 80 --doc-count 10
```

> **提示**: 如果 `graphrag-kg` 命令不可用，可以用 `python -m graphrag_kg.cli.main` 替代，例如：
> ```bash
> python -m graphrag_kg.cli.main data generate --scenario pharma_supply_chain
> ```

| 项目 | 说明 |
|---|---|
| **前置条件** | 无（纯本地生成，不需要 LLM API） |
| **副作用** | 在 `tests/fixtures/generated/<scenario>/` 下生成文档、ground truth 和测试问题 |
| **幂等性** | 相同 seed 产生完全相同的输出。重复执行会覆盖已有文件 |
| **数据量** | pharma_supply_chain：59 实体 / 153 关系 / 5 社区 / 8 测试问题 |

---

### 4. 初始化项目

```bash
# 在当前目录初始化
graphrag-kg init --name my-knowledge-graph

# 指定目录
graphrag-kg init --name my-kg --root ./my-project02

# 强制覆盖已有项目
graphrag-kg init --name my-kg --force
```

| 项目 | 说明 |
|---|---|
| **副作用** | 创建 `settings.yaml`、`.env`、`docker-compose.yml`、`project.json`，以及 `input/`、`output/`、`cache/`、`logs/`、`prompts/`、`documents/`、`config/` 目录 |
| **幂等性** | 默认拒绝覆盖。`--force` 可覆盖 `settings.yaml` 和 `docker-compose.yml`，但**不会覆盖 `.env`** |

---

### 5. 导入文档

```bash
# 从测试数据导入
graphrag-kg ingest run --source tests/fixtures/generated/pharma_supply_chain/documents
graphrag-kg ingest run --source tests/fixtures/generated/tech_company/documents

# 从自定义目录导入
graphrag-kg ingest run --source ./my-docs --patterns "**/*.pdf,**/*.md"

# 预览（不实际导入）
graphrag-kg ingest run --source ./my-docs --dry-run

# 追加模式（不清除 input/ 中已有文件）
graphrag-kg ingest run --source ./my-docs --no-clear

# 仅发现文件（不解析）
graphrag-kg ingest discover --source ./my-docs
```

| 项目 | 说明 |
|---|---|
| **前置条件** | 已完成 `init`；源目录中存在文档 |
| **支持格式** | PDF（pymupdf）、Markdown、TXT（chardet 自动编码检测）、HTML（beautifulsoup4） |
| **大小限制** | 默认跳过 >50MB 的文件（可通过 `--max-size` 调整） |

---

### 6. 构建知识图谱索引

```bash
# 快速索引（跳过社区报告，推荐，成本更低）
graphrag-kg index run --method fast

# 标准索引（完整流程，含社区报告，质量最高）
graphrag-kg index run --method standard

# 增量更新（仅处理新增文档）
graphrag-kg index run --method standard-update

# 预检查（不实际执行）
graphrag-kg index run --dry-run
```

| 项目 | 说明 |
|---|---|
| **前置条件** | `input/` 目录中存在 .txt 文件；已设置 `GRAPHRAG_API_KEY` |
| **副作用** | 生成 Parquet 文件到 `output/`；向量嵌入写入 **Milvus**（需确保 Milvus 运行中）|
| **方法对比** | `standard`：完整流程含社区报告，质量最高但成本高<br>`fast`：跳过社区报告，约节省 40% 成本 |

#### ⚠️ 注意：关于嵌入向量生成

某些 LLM（如 DeepSeek）不支持 `response_format` 参数，导致索引管道的 `generate_text_embeddings` 工作流失败、Milvus 中无嵌入数据。如果索引完成后查询返回"数据表为空"，运行以下备用脚本生成嵌入：

```bash
python _generate_embeddings.py
```

该脚本从本地 Ollama BGE-M3 模型生成嵌入并直接写入 Milvus。

---

### 7. 同步到 Neo4j

```bash
# 标准同步
graphrag-kg graph sync

# 清除 Neo4j 后重新同步
graphrag-kg graph sync --clear

# 查看 Neo4j 图统计
graphrag-kg graph status

# 清空 Neo4j 数据库（需确认）
graphrag-kg graph drop --force
```

| 项目 | 说明 |
|---|---|
| **前置条件** | Neo4j 已启动；`output/` 中存在 Parquet 索引文件 |
| **幂等性** | 使用 `MERGE` 语句，重复执行不会创建重复节点 |
| **批量大小** | 默认每批 1000 条（可在 `settings.yaml` 中调整 `neo4j.sync_batch_size`） |
| **验证** | `graphrag-kg graph status` 查看统计；浏览器访问 http://localhost:7474 可视化浏览 |

---

### 8. 查询知识图谱

```bash
# 自动选择最佳搜索方法（默认）
graphrag-kg query ask "恒瑞医药生产哪些药品？"

# 指定搜索方法
graphrag-kg query ask "紫杉醇的完整供应链是怎样的？" --method drift

# 全局搜索（适合总结性问题）
graphrag-kg query ask "这个数据集的主要主题是什么？" --method global

# 基础向量搜索（最快，无图上下文）
graphrag-kg query ask "什么是非小细胞肺癌？" --method basic

# 隐藏引用来源
graphrag-kg query ask "..." --no-sources

# 输出原始 JSON
graphrag-kg query ask "..." --raw

# 查看查询引擎状态
graphrag-kg query status
```

| 项目 | 说明 |
|---|---|
| **前置条件** | `output/` 中存在 Parquet 索引文件；已设置 `GRAPHRAG_API_KEY` |
| **搜索方法** | `auto`：根据问题类型自动选择（默认）<br>`local`：实体扩展 + 文本检索，适合具体事实查询<br>`global`：社区报告 map-reduce，适合总结/趋势问题<br>`drift`：层次化社区遍历，适合多跳推理/供应链追踪<br>`basic`：纯向量搜索，最快但无图上下文 |

---

### 9. 启动 REST API（可选）

```bash
# 启动服务器
graphrag-kg serve start --port 8000

# 开发模式（自动重载）
graphrag-kg serve start --port 8000 --reload
```

| 端点 | 说明 |
|---|---|
| `GET /health` | 健康检查（无需索引） |
| `GET /stats` | 系统统计 |
| `POST /query` | 提交查询 |
| `GET /index/status` | 索引状态 |
| `POST /index` | 触发索引 |
| `GET /graph/stats` | Neo4j 图统计 |
| `POST /graph/sync` | 同步到 Neo4j |

---

## 配置管理

```bash
# 查看当前配置
graphrag-kg config show

# 验证配置
graphrag-kg config validate

# 列出可用配置模板
graphrag-kg config profile list

# 查看某个模板
graphrag-kg config profile show fast

# 应用模板（仅显示，不修改文件）
graphrag-kg config profile apply fast
```

### 配置模板对比

| 模板 | Chat 模型 | Embedding 模型 | 查询方法 | 分块大小 | Gleanings | 适用场景 |
|---|---|---|---|---|---|---|
| `default` | gpt-4.1 | text-embedding-3-large | local | 1200 tokens | 1 | 通用 |
| `fast` | gpt-4.1-mini | text-embedding-3-small | basic | 600 tokens | 0 | 快速迭代、低成本 |
| `production` | gpt-4.1 | text-embedding-3-large | drift | 1200 tokens | 2 | 生产环境、最高质量 |

### 环境变量

在 `.env` 文件中配置（不会被 `init --force` 覆盖）：

```bash
# === LLM Configuration ===
GRAPHRAG_API_KEY=sk-xxx                          # 必填
GRAPHRAG_CHAT_MODEL=gpt-4.1                      # Chat 模型
GRAPHRAG_EMBEDDING_MODEL=text-embedding-3-large  # Embedding 模型
GRAPHRAG_API_BASE=https://api.openai.com/v1       # API 端点

# === Neo4j Configuration ===
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password

# === Milvus Configuration ===
MILVUS_HOST=localhost
MILVUS_PORT=19530
```

---

## 测试数据场景

| 场景 | 实体数 | 关系数 | 查询跳数 | 实体类型 | 用途 |
|---|---|---|---|---|---|
| `pharma_supply_chain` | 59 | 153 | 3-5 跳 | 12 种（药企、药品、分销商、医院等） | 测试多跳供应链推理 |
| `tech_company` | 21 | 34 | 1-2 跳 | 5 种（组织、人物、技术、地点、事件） | 测试基础实体关系提取 |

### pharma_supply_chain 测试问题

| # | 问题 | 跳数 | 搜索方法 | 测试目标 |
|---|---|---|---|---|
| 1 | 注射用紫杉醇主要用于治疗哪些疾病？ | 1 | local | 直接事实查找 |
| 2 | 恒瑞医药生产哪些主要药品？ | 1 | local | 单跳关系 |
| 3 | 北京协和医院使用哪些制药公司生产的抗肿瘤药物？ | 3 | local | 多跳推理 |
| 4 | 紫杉醇API从原料药到临床使用的完整供应链是怎样的？ | 4 | drift | 全链路追踪 |
| 5 | 国药控股在华东区分销哪些制药公司的哪些药品？ | 4 | global | 分销链路 |
| 6 | 哪些医院同时使用恒瑞医药和齐鲁制药的药品？ | 2 | local | 集合交集 |
| 7 | 如果注射用紫杉醇供应中断，会影响哪些医院的哪些科室？ | 3 | drift | 影响分析 |
| 8 | 恒瑞医药的注射用紫杉醇获得了哪些监管机构的审批？ | 2 | local | 监管链路 |

### tech_company 测试问题

| # | 问题 | 搜索方法 |
|---|---|---|
| 1 | ACME CORP 的公司概述？ | local |
| 2 | Satya Nadella 是谁？ | basic |
| 3 | ACME CORP 收购了哪家公司？ | local |
| 4 | CloudPeak Systems 和 ACME CORP 是什么关系？ | drift |

---

## 项目结构

```
src/graphrag_kg/
├── cli/commands/     # 8 个 CLI 命令模块
├── core/             # 核心模块（配置、加载、编排、项目初始化）
├── data/             # 测试数据生成器
├── ingest/           # 文档导入（PDF/MD/TXT/HTML 解析）
├── index/            # 索引管道（graphrag 封装、进度监控、增量更新）
├── graph/            # Neo4j 图操作（连接池、schema、同步、Cypher 查询、遍历）
├── query/            # 查询引擎（local/global/drift/basic + 缓存 + 上下文）
├── storage/          # 存储层（Parquet 读写）
├── api/              # FastAPI REST 服务
├── prompts/          # LLM 提示词模板
└── utils/            # 工具（日志、环境变量、进度条、验证）
```

---

## 依赖

| 包 | 版本 | 用途 |
|---|---|---|
| `graphrag` | >=3.0.0 | Microsoft GraphRAG 核心库 |
| `neo4j` | >=5.20.0 | Neo4j Python 驱动 |
| `pymilvus` | >=2.4.0 | Milvus 向量数据库 |
| `typer` | >=0.12.0 | CLI 框架 |
| `rich` | >=13.0.0 | 终端美化输出 |
| `pydantic` | >=2.0.0 | 配置模型和验证 |
| `pyyaml` | >=6.0 | YAML 配置解析 |
| `pymupdf` | >=1.23.0 | PDF 解析 |
| `beautifulsoup4` | >=4.12.0 | HTML 解析 |
| `pyarrow` | >=14.0.0 | Parquet 文件支持 |
| `pandas` | >=2.0.0 | 数据处理 |
| `fastapi` | >=0.110.0 | REST API（可选） |
| `uvicorn` | >=0.27.0 | ASGI 服务器（可选） |

---

## 快速启动（本地测试）

```bash
# 1. 安装
pip install -e .

# 2. 启动 Docker 服务
docker-compose up -d

# 3. 生成测试数据（可选）
graphrag-kg data generate --scenario all

# 4. 导入文档
graphrag-kg ingest run --source tests/fixtures/generated/pharma_supply_chain/documents
graphrag-kg ingest run --source tests/fixtures/generated/tech_company/documents --no-clear

# 5. 构建索引
graphrag-kg index run --method fast

# 6. 备用：如果嵌入未写入 Milvus（如 DeepSeek 用户），运行：
python _generate_embeddings.py

# 7. 同步到 Neo4j（可选）
graphrag-kg graph sync

# 8. 查询
graphrag-kg query ask "恒瑞医药生产哪些药品？"
graphrag-kg query ask "ACME CORP Company overview" --method basic
```


新增服务-monitor

┌────────────┬─────────────────────────┬──────┬────────────────────────────────────────────────┐
│    服务    │          镜像           │ 端口 │                      用途                      │
├────────────┼─────────────────────────┼──────┼────────────────────────────────────────────────┤
│ prometheus │ prom/prometheus:v2.51.0 │ 9090 │ 采集 Milvus 指标（从 standalone:9091/metrics） │
├────────────┼─────────────────────────┼──────┼────────────────────────────────────────────────┤
│ grafana    │ grafana/grafana:10.4.0  │ 3000 │ 可视化监控面板                                 │
└────────────┴─────────────────────────┴──────┴────────────────────────────────────────────────┘

配置文件

┌────────────────────────────────────┬──────────────────────────────────────────────────────────────────┐
│                文件                │                               用途                               │
├────────────────────────────────────┼──────────────────────────────────────────────────────────────────┤
│ prometheus/prometheus.yml          │ Prometheus 抓取配置，指向 Milvus standalone 的 9091/metrics 端点 │
├────────────────────────────────────┼──────────────────────────────────────────────────────────────────┤
│ grafana/datasources/datasource.yml │ 自动配置 Grafana 的 Prometheus 数据源                            │
├────────────────────────────────────┼──────────────────────────────────────────────────────────────────┤
│ grafana/dashboards/dashboard.yml   │ 自动加载 Milvus Dashboard 面板                                   │
└────────────────────────────────────┴──────────────────────────────────────────────────────────────────┘

使用方式

docker compose up -d

# 访问以下地址：
# - Prometheus: http://localhost:9090
# - Grafana:    http://localhost:3000 (admin/admin)
# - Attu:       http://localhost:8000 (admin/admin)
# - Milvus 指标: http://localhost:9091/metrics

Grafana 启动后会自动配置好 Prometheus 数据源，然后你可以从 Grafana Dashboard 市场 (https://grafana.com/grafana/dashboards/) 搜索 "Milvus" 导入官方面板，或者手动创建自定义面板来监控 Milvus 的 QPS、延迟、内存等指标。