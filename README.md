# GraphRAG-KG: Knowledge Graph Q&A System

A Python-based GraphRAG system for Knowledge Graph Q&A, built on Microsoft's `graphrag` library with **Neo4j + Milvus** hybrid storage.

## Architecture

```
Documents (PDF/MD/TXT/HTML)
  → Ingest → graphrag Index → Parquet + LanceDB
  → Neo4j Sync → Cypher Graph Traversal
  → Query Engine (Local/Global/DRIFT/Basic)
  → Grounded Answers + Citations + Graph Context
```

---

## 完整工作流

以下按顺序执行每个命令，从零搭建一个可查询的知识图谱系统。每个命令都标注了**前置条件**和**副作用**。

---

### 1. 安装

```bash
pip install -e .
```

| 项目 | 说明 |
|---|---|
| **前置条件** | Python 3.10-3.12 |
| **副作用** | 安装 `graphrag-kg` 命令到系统 PATH；安装所有依赖（graphrag, neo4j, lancedb, typer, fastapi 等） |
| **幂等性** | 是，重复执行不会重复安装 |

---

### 2. 启动 Neo4j

```bash
docker-compose up -d
```

| 项目 | 说明 |
|---|---|
| **前置条件** | Docker 已安装并运行 |
| **副作用** | 拉取 `neo4j:5.22.0` 镜像（首次约 500MB）；创建 `neo4j_data` 和 `neo4j_logs` 两个 Docker volume；启动 Neo4j 容器，监听 `bolt://localhost:7687` 和 `http://localhost:7474` |
| **数据持久性** | 数据存储在 Docker volume 中，`docker-compose down` 不会丢失数据。如需彻底清除：`docker-compose down -v` |
| **默认凭据** | 用户名 `neo4j`，密码 `password`（生产环境请修改 `.env` 中的 `NEO4J_PASSWORD`） |
| **验证** | 浏览器访问 http://localhost:7474 ，用 neo4j/password 登录 |

---

### 3. 生成测试数据

```bash
# 生成医药供应链场景（强关联关系，推荐）
graphrag-kg data generate --scenario pharma_supply_chain

#  无法识别graphrag-k命令解决方法一（推荐）：直接用 python -m 运行=

python -m graphrag_kg.cli.main data generate --scenario pharma_supply_chain

# 生成科技公司场景
graphrag-kg data generate --scenario tech_company
python -m graphrag_kg.cli.main data generate --scenario tech_company
# 生成全部场景
graphrag-kg data generate --scenario all
python -m graphrag_kg.cli.main data generate --scenario all

# 自定义参数
graphrag-kg data generate --scenario pharma_supply_chain --seed 123 --entity-count 80 --doc-count 10
```

| 项目 | 说明 |
|---|---|
| **前置条件** | 无（纯本地生成，不需要 LLM API） |
| **副作用** | 在 `tests/fixtures/generated/<scenario>/` 下创建以下文件：<br>• `documents/` — 7 篇文档 × 3 种格式（.md / .txt / .html），共 21 个文件<br>• `ground_truth.json` — 预期的实体、关系、社区、测试问题（标准答案）<br>• `queries.json` — 8 个预设测试问题，含预期答案和关系路径<br>• `README.md` — 场景说明 |
| **幂等性** | 是，相同 seed 产生完全相同的输出。重复执行会覆盖已有文件 |
| **数据量** | pharma_supply_chain：59 个实体（12 种类型）、153 条关系、5 个社区、8 个测试问题（3-5 跳） |
| **验证** | `graphrag-kg data ground-truth --scenario pharma_supply_chain` 查看预期数据 |

---

### 4. 初始化项目

```bash
graphrag-kg init --name my-knowledge-graph

# 指定目录=
graphrag-kg init --name my-kg --root ./my-project02
python -m graphrag_kg.cli.main  init --name my-kg --root ./my-project02

# 强制覆盖已有项目
graphrag-kg init --name my-kg --force
```

| 项目 | 说明 |
|---|---|
| **前置条件** | 无 |
| **副作用** | 在当前目录创建以下文件和目录：<br>• `settings.yaml` — 项目配置文件（含 Neo4j、LLM、ingest、query 等全部配置）<br>• `.env` — 环境变量模板（**仅在不存在时创建**，不会覆盖已有 .env）<br>• `docker-compose.yml` — Neo4j 容器配置<br>• `project.json` — 项目清单文件<br>• `input/` — 待索引的文本文件目录<br>• `output/` — 索引输出目录（parquet + lancedb）<br>• `cache/` — 缓存目录<br>• `logs/` — 日志目录<br>• `prompts/` — LLM 提示词模板目录<br>• `documents/` — 原始文档存放目录<br>• `config/` — 配置模板目录 |
| **幂等性** | 默认拒绝覆盖（报错 `Project already exists`）。使用 `--force` 可覆盖 `settings.yaml` 和 `docker-compose.yml`，但**不会覆盖 `.env`** |
| **验证** | `ls -la` 查看生成的文件；`graphrag-kg config show` 查看配置 |

---

### 5. 导入文档

```bash
# 从测试数据导入=
graphrag-kg ingest run --source tests/fixtures/generated/pharma_supply_chain/documents

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
| **副作用** | • **读取**：扫描源目录中的 PDF/MD/TXT/HTML 文件<br>• **写入**：将解析后的纯文本写入 `input/*.txt`（每个源文件生成一个 .txt）<br>• **删除**：默认先清空 `input/` 中已有的 .txt 文件（使用 `--no-clear` 可保留）<br>• 文件名格式：`<原文件名>_<格式>.txt`（如 `manufacturer_catalog.md_md.txt`），避免同名不同格式的文件冲突 |
| **幂等性** | 默认清除后重新生成，结果一致。`--no-clear` 模式下追加，可能产生重复 |
| **支持格式** | PDF（pymupdf）、Markdown、TXT（chardet 自动检测编码）、HTML（beautifulsoup4 去标签） |
| **大小限制** | 默认跳过 >50MB 的文件（可通过 `--max-size` 调整） |
| **验证** | `ls input/` 查看生成的 .txt 文件；检查文件内容是否完整 |

---

### 6. 构建知识图谱索引

```bash
# 标准索引（完整流程，质量最高）=
graphrag-kg index run --method standard

# 快速索引（跳过社区报告生成，成本更低）
graphrag-kg index run --method fast

# 增量更新（仅处理新增文档）
graphrag-kg index run --method standard-update

# 预检查（不实际执行）
graphrag-kg index run --dry-run
```

| 项目 | 说明 |
|---|---|
| **前置条件** | • `input/` 目录中存在 .txt 文件（已完成 `ingest`）<br>• 已设置 `GRAPHRAG_API_KEY` 环境变量（或在 `.env` 中配置）<br>• **会产生 LLM API 费用** |
| **副作用** | • **读取**：`input/*.txt` 全部文本文件<br>• **写入**：`output/` 目录下生成以下 Parquet 文件：<br>&nbsp;&nbsp;`entities.parquet` — 实体表<br>&nbsp;&nbsp;`relationships.parquet` — 关系表<br>&nbsp;&nbsp;`communities.parquet` — 社区表<br>&nbsp;&nbsp;`community_reports.parquet` — 社区报告表<br>&nbsp;&nbsp;`text_units.parquet` — 文本单元表<br>&nbsp;&nbsp;`documents.parquet` — 文档表<br>• **写入**：`output/lancedb/` — 向量嵌入数据库<br>• **写入**：`cache/` — 索引缓存<br>• **写入**：`logs/` — 索引日志 |
| **幂等性** | 重复执行会覆盖 `output/` 中的已有索引文件 |
| **方法对比** | `standard`：完整 10 步流程，含社区报告，质量最高，成本最高<br>`fast`：跳过社区报告生成，约节省 40% 成本<br>`standard-update`：仅处理新增文档，保留已有索引 |
| **耗时** | 取决于文档量和 LLM 速率限制。7 篇文档约 3-8 分钟 |
| **验证** | `ls output/*.parquet` 确认文件已生成；`graphrag-kg query status` 查看索引状态 |

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
| **前置条件** | • Neo4j 已启动（`docker-compose up -d`）<br>• `output/` 中存在 Parquet 索引文件（已完成 `index`） |
| **副作用** | • **读取**：`output/*.parquet` 全部索引文件<br>• **写入 Neo4j**：创建以下节点和关系：<br>&nbsp;&nbsp;`(:Entity)` 节点 — 实体（含 name, type, description, degree 等属性）<br>&nbsp;&nbsp;`(:Community)` 节点 — 社区（含 title, summary, level 等属性）<br>&nbsp;&nbsp;`(:Document)` 节点 — 文档<br>&nbsp;&nbsp;`(:TextUnit)` 节点 — 文本单元<br>&nbsp;&nbsp;`[:RELATES_TO]` 关系 — 实体间关系（含 description, weight）<br>&nbsp;&nbsp;`[:BELONGS_TO]` 关系 — 实体归属社区<br>• **写入 Neo4j**：创建索引和约束（加速查询）<br>• 使用 `--clear` 会**先删除 Neo4j 中所有节点和关系**，再重新同步 |
| **幂等性** | 使用 `MERGE` 语句，重复执行不会创建重复节点（基于 id 去重） |
| **批量大小** | 默认每批 1000 条（可在 `settings.yaml` 中调整 `neo4j.sync_batch_size`） |
| **验证** | `graphrag-kg graph status` 查看节点和关系数量；浏览器访问 http://localhost:7474 可视化浏览图 |

---

### 8. 查询知识图谱

```bash
# 自动选择最佳搜索方法
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
| **前置条件** | • `output/` 中存在 Parquet 索引文件（已完成 `index`）<br>• 已设置 `GRAPHRAG_API_KEY`（**每次查询都会调用 LLM，产生费用**） |
| **副作用** | • **读取**：`output/*.parquet` 和 `output/lancedb/`<br>• **读取 Neo4j**（仅 local 方法）：查询实体邻居和社区上下文<br>• **写入**：查询结果缓存到内存（TTL 默认 3600 秒），重复查询不重复调用 LLM<br>• **不修改**：任何磁盘文件或数据库内容 |
| **搜索方法** | `auto`：根据问题类型自动选择（默认）<br>`local`：实体扩展 + 文本检索，适合具体事实查询<br>`global`：社区报告 map-reduce，适合总结/趋势问题<br>`drift`：层次化社区遍历，适合多跳推理/供应链追踪<br>`basic`：纯向量搜索，最快但无图上下文 |
| **验证** | 答案应包含引用来源和实体信息 |

---

### 9. 启动 REST API（可选）

```bash
# 启动服务器
graphrag-kg serve start --port 8000

# 开发模式（自动重载）
graphrag-kg serve start --port 8000 --reload
```

| 项目 | 说明 |
|---|---|
| **前置条件** | 已完成 `index`（查询接口需要索引） |
| **副作用** | • 启动 uvicorn 服务器，监听指定端口<br>• 提供 Swagger 文档：http://localhost:8000/docs<br>• 提供 ReDoc 文档：http://localhost:8000/redoc |
| **端点** | `GET /health` — 健康检查（无需索引）<br>`GET /stats` — 系统统计<br>`POST /query` — 提交查询<br>`GET /index/status` — 索引状态<br>`POST /index` — 触发索引<br>`GET /graph/stats` — Neo4j 图统计<br>`POST /graph/sync` — 同步到 Neo4j |

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
# LLM（必填）
GRAPHRAG_API_KEY=sk-xxx

# LLM（可选，默认值如下）
GRAPHRAG_CHAT_MODEL=gpt-4.1
GRAPHRAG_EMBEDDING_MODEL=text-embedding-3-large
GRAPHRAG_API_BASE=https://api.openai.com/v1

# Neo4j（可选，默认值如下）
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password
```

---

## 测试数据场景

| 场景 | 实体数 | 关系数 | 查询跳数 | 实体类型 | 用途 |
|---|---|---|---|---|---|
| `pharma_supply_chain` | 59 | 153 | 3-5 跳 | 12 种（药企、药品、原料药、分销商、医院、科室、适应症...） | 测试多跳供应链推理 |
| `tech_company` | 21 | 34 | 1-2 跳 | 5 种（组织、人物、技术、地点、事件） | 测试基础实体关系提取 |

### pharma_supply_chain 场景的 8 个测试问题

| # | 问题 | 跳数 | 搜索方法 | 测试目标 |
|---|---|---|---|---|
| 1 | 注射用紫杉醇主要用于治疗哪些疾病？ | 1 | local | 直接事实查找 |
| 2 | 恒瑞医药生产哪些主要药品？ | 1 | local | 单跳关系 |
| 3 | 北京协和医院使用哪些制药公司生产的抗肿瘤药物？ | 3 | local | 多跳：医院→药品→生产商→适应症 |
| 4 | 紫杉醇API从原料药到临床使用的完整供应链是怎样的？ | 4 | drift | 全链路追踪 |
| 5 | 国药控股在华东区分销哪些制药公司的哪些药品？ | 4 | global | 分销商→区域→合同→药品→适应症 |
| 6 | 哪些医院同时使用恒瑞医药和齐鲁制药的药品？ | 2 | local | 集合交集 |
| 7 | 如果注射用紫杉醇供应中断，会影响哪些医院的哪些科室？ | 3 | drift | 影响分析 |
| 8 | 恒瑞医药的注射用紫杉醇获得了哪些监管机构的审批？ | 2 | local | 监管链路 |

---

## 项目结构

```
src/graphrag_kg/
├── cli/commands/     # 8 个 CLI 命令模块
│   ├── data_cmd.py       # graphrag-kg data
│   ├── init_cmd.py       # graphrag-kg init
│   ├── ingest_cmd.py     # graphrag-kg ingest
│   ├── index_cmd.py      # graphrag-kg index
│   ├── graph_cmd.py      # graphrag-kg graph
│   ├── query_cmd.py      # graphrag-kg query
│   ├── config_cmd.py     # graphrag-kg config
│   └── serve_cmd.py      # graphrag-kg serve
├── core/             # 核心模块
│   ├── config.py         # KGConfig Pydantic 模型
│   ├── config_loader.py  # YAML 加载 + 环境变量替换
│   ├── pipeline.py       # 全流程编排器
│   ├── project.py        # 项目初始化和管理
│   └── errors.py         # 异常层次结构
├── data/             # 测试数据生成器
│   ├── generator.py      # TestDataGenerator 主类
│   ├── entities.py       # 实体工厂（Faker + 中文实体池）
│   ├── relationships.py  # 关系工厂（规则驱动）
│   ├── queries.py        # 测试问题工厂
│   ├── ground_truth.py   # Ground truth 模型 + JSON 序列化
│   └── templates/        # Jinja2 文档模板（3 个场景）
├── ingest/           # 文档导入
│   ├── parsers.py        # PDF/MD/TXT/HTML 解析器
│   ├── loader.py         # 文件发现 + 批量加载
│   └── converter.py      # 转换为 graphrag 输入格式
├── index/            # 索引管道
│   ├── runner.py         # graphrag.index 封装
│   ├── monitor.py        # Rich 进度监控
│   └── updater.py        # 增量更新
├── graph/            # Neo4j 图操作
│   ├── connection.py     # 驱动 + 连接池 + 健康检查
│   ├── schema.py         # 索引/约束管理
│   ├── sync.py           # Parquet → Neo4j 批量同步
│   ├── queries.py        # Cypher 查询库
│   └── traversal.py      # 图遍历（ego network、路径、影响分析）
├── query/            # 查询引擎
│   ├── engine.py         # QueryEngine 门面 + 自动路由
│   ├── local.py          # 本地搜索（Neo4j 增强）
│   ├── global_search.py  # 全局搜索（社区报告 map-reduce）
│   ├── drift.py          # DRIFT 搜索（层次化遍历）
│   ├── basic.py          # 基础向量搜索
│   ├── context.py        # 引用提取 + 响应构建
│   └── cache.py          # 查询结果缓存（TTL）
├── storage/          # 存储层
│   ├── parquet_store.py  # Parquet 文件读写
│   └── vector_store.py   # LanceDB 向量存储
├── api/              # REST API
│   ├── app.py            # FastAPI 应用
│   ├── models.py         # Pydantic 请求/响应模型
│   ├── middleware.py      # CORS + 日志中间件
│   └── routes/           # 路由（health, query, index, graph）
├── prompts/          # LLM 提示词模板
│   ├── templates.py      # 提示词注册表
│   └── defaults/         # 10 个默认提示词文件
└── utils/            # 工具
    ├── logging.py        # 结构化日志
    ├── env.py            # 环境变量管理
    ├── progress.py       # Rich 进度条
    └── validators.py     # 输入验证
```

---

## 依赖

| 包 | 版本 | 用途 |
|---|---|---|
| `graphrag` | >=3.0.0 | Microsoft GraphRAG 核心库 |
| `neo4j` | >=5.20.0 | Neo4j Python 驱动 |
| `lancedb` | >=0.6.0 | 向量嵌入存储 |
| `typer` | >=0.12.0 | CLI 框架 |
| `rich` | >=13.0.0 | 终端美化输出 |
| `pydantic` | >=2.0.0 | 配置模型和验证 |
| `pyyaml` | >=6.0 | YAML 配置解析 |
| `pymupdf` | >=1.23.0 | PDF 解析 |
| `beautifulsoup4` | >=4.12.0 | HTML 解析 |
| `chardet` | >=5.0.0 | 文本编码检测 |
| `faker` | >=22.0.0 | 测试数据生成 |
| `jinja2` | >=3.1.0 | 文档模板渲染 |
| `pyarrow` | >=14.0.0 | Parquet 文件支持 |
| `pandas` | >=2.0.0 | 数据处理 |
| `fastapi` | >=0.110.0 | REST API（可选） |
| `uvicorn` | >=0.27.0 | ASGI 服务器（可选） |


## history 切换model运行以下命令
D:\claude-code-project\graphRAG>graphrag-kg index run --method standard

│ GraphRAG Indexing: standard                                                                                                                                                                                   │


[INFO] Input directory: D:\claude-code-project\graphRAG\input
[INFO] Output directory: D:\claude-code-project\graphRAG\output
[INFO] Chat model: deepseek-v4-flash
[INFO] Embedding model: bge-m3


│ Starting GraphRAG Indexing Pipeline                                                                                                                                                                           │
│ Method: standard | Workflows: │ Indexing Complete│ Total time: 137  6s
  Entities: 110
  Relationships: 468
  Communities: 24
  Text Units: 28

[INFO] Output: D:\claude-code-project\graphRAG\output
[OK] Indexing complete!

D:\claude-code-project\graphRAG>graphrag-kg query ask "恒瑞医药生产哪些药品？"

graphrag-kg query ask "注射用紫杉醇的完整供应链是怎样的？" 
graphrag-kg query ask "北京协和医院使用哪些制药公司的药品？"
graphrag-kg query ask "如果紫杉醇供应中断会影响哪些医院？" --method drift