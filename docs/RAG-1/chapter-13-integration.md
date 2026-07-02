# 第13章 GraphRAG + DeepSeek 集成实战

## 13.1 引言

本章以GraphRAG-KG项目为实际案例，完整呈现一个生产级GraphRAG系统的构建过程。GraphRAG-KG是一个基于Microsoft GraphRAG框架扩展的知识图谱增强检索系统，集成了DeepSeek大语言模型进行实体提取和问答生成，使用Ollama bge-m3模型进行文本嵌入，并以Neo4j图数据库作为知识存储层。

通过本章，读者将理解：
- 如何将GraphRAG的理论概念转化为可运行的代码
- 如何配置多模型提供者（LLM + Embedding）的混合架构
- 如何构建领域特定的测试数据
- 如何完成从文档索引到查询的全流程
- 如何与Neo4j图数据库同步
- 如何进行性能调优

本章所有代码均来自实际项目，可以直接作为RAG系统开发的技术参考。

---

## 13.2 项目架构概览

### 13.2.1 模块结构

GraphRAG-KG项目采用分层模块化架构，以下是完整的模块树：

```
src/graphrag_kg/
├── __init__.py                    # 包初始化，版本信息
├── core/
│   ├── __init__.py
│   ├── config.py                  # 配置模型定义（Pydantic）
│   └── config_loader.py           # 配置加载器（YAML + 环境变量）
├── cli/
│   ├── __init__.py
│   └── main.py                    # CLI入口，8个子命令
├── pipeline/
│   ├── __init__.py
│   └── indexing.py                # 索引流水线（标准/快速/更新）
├── query/
│   ├── __init__.py
│   └── engine.py                  # 查询引擎（自动路由4种策略）
├── neo4j/
│   ├── __init__.py
│   └── sync.py                    # Neo4j同步（Parquet → 图数据库）
└── test_data/
    ├── __init__.py
    └── pharma_supply_chain.py     # 制药供应链测试数据
```

### 13.2.2 数据流架构

```
┌──────────────────────────────────────────────────────────┐
│                      输入层                                │
│  [文档] → [分块] → [向量化(bge-m3)] → [LanceDB存储]      │
└──────────────────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│                   索引层（GraphRAG）                       │
│  [实体提取(DeepSeek)] → [关系提取] → [社区检测]          │
│  → [社区摘要] → [Parquet输出]                             │
└──────────────────────────────────────────────────────────┘
                         │
                    ┌────┴────┐
                    ▼         ▼
┌──────────────────┐  ┌──────────────────┐
│   Neo4j同步层    │  │   查询引擎层     │
│  [Entity节点]    │  │ [Local Search]   │
│  [Community节点] │  │ [Global Search]  │
│  [Document节点]  │  │ [Drift Search]   │
│  [TextUnit节点]  │  │ [Basic Search]   │
│  [RELATES_TO关系]│  │                  │
└──────────────────┘  └──────────────────┘
```

### 13.2.3 核心依赖

```
# requirements.txt 核心依赖
graphrag>=0.3.0          # Microsoft GraphRAG核心库
neo4j>=5.14.0            # Neo4j Python驱动
lancedb>=0.6.0           # LanceDB向量数据库
pandas>=2.0.0            # 数据处理
pyyaml>=6.0              # YAML配置解析
python-dotenv>=1.0.0     # 环境变量加载
pydantic>=2.0.0          # 配置模型验证
click>=8.0.0             # CLI框架
```

---

## 13.3 配置系统

### 13.3.1 双层配置架构

GraphRAG-KG采用"环境变量 + YAML配置文件"的双层配置架构。环境变量存储敏感信息（API密钥），YAML文件存储运行时参数。

#### .env文件

```bash
# D:\claude-code-project\graphRAG\.env
# ============================================
# LLM配置：使用DeepSeek作为主要推理模型
# ============================================

# DeepSeek Chat API（用于实体提取、社区摘要、问答生成）
LLM_API_KEY=sk-your-deepseek-api-key-here
LLM_API_BASE=https://api.deepseek.com
LLM_MODEL=deepseek-chat
LLM_MAX_TOKENS=8000
LLM_TEMPERATURE=0.0

# ============================================
# Embedding配置：使用Ollama本地部署的bge-m3
# ============================================

# Ollama本地嵌入服务（生产环境建议独立部署）
EMBEDDING_API_BASE=http://localhost:11434
EMBEDDING_MODEL=bge-m3
EMBEDDING_DIM=1024

# ============================================
# Neo4j图数据库配置
# ============================================

NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-neo4j-password

# ============================================
# 向量数据库配置
# ============================================

# LanceDB本地存储路径
LANCEDB_URI=./data/lancedb
```

#### settings.yaml

```yaml
# D:\claude-code-project\graphRAG\settings.yaml
# ============================================
# GraphRAG-KG 全局设置
# ============================================

# 编码设置
encoding_model: cl100k_base

# 跳过工作流设置（调试用）
skip_workflows: []

# ============================================
# LLM配置
# ============================================

llm:
  api_key: ${LLM_API_KEY}
  type: openai_chat  # 兼容OpenAI接口格式
  model: ${LLM_MODEL}
  api_base: ${LLM_API_BASE}
  max_tokens: ${LLM_MAX_TOKENS}
  temperature: ${LLM_TEMPERATURE}
  request_timeout: 60.0
  model_supports_json: true

# ============================================
# Embedding配置
# ============================================

embeddings:
  llm:
    api_key: none  # Ollama无需API Key
    type: openai_embedding
    model: ${EMBEDDING_MODEL}
    api_base: ${EMBEDDING_API_BASE}
    dimensions: ${EMBEDDING_DIM}
    request_timeout: 60.0

# ============================================
# 分块配置
# ============================================

chunks:
  size: 800
  overlap: 100
  group_by_columns: [id]

# ============================================
# 实体提取配置
# ============================================

entity_extraction:
  prompt: "见prompts/entity_extraction.txt"
  entity_types:
    - DRUG
    - COMPANY
    - DISEASE
    - REGULATORY
    - CLINICAL_TRIAL
    - PERSON
    - PROCESS
    - LOCATION
    - MATERIAL
    - EQUIPMENT
    - CONTRACT
    - EVENT
  max_gleaning: 2

# ============================================
# 社区配置
# ============================================

community_reports:
  prompt: "见prompts/community_report.txt"
  max_length: 2000
  max_input_length: 8000

# ============================================
# 存储配置
# ============================================

storage:
  type: file
  base_dir: ./output

# ============================================
# 向量存储配置
# ============================================

vector_store:
  type: lancedb
  db_uri: ${LANCEDB_URI}

# ============================================
# 本地搜索配置
# ============================================

local_search:
  llm:
    model: ${LLM_MODEL}
    temperature: 0.0
  text:
    prompt: "见prompts/local_search_system_prompt.txt"
  mmap:
    llm:
      model: ${LLM_MODEL}
      temperature: 0.0
    map_system_prompt: "见prompts/map_system_prompt.txt"
    reduce_system_prompt: "见prompts/reduce_system_prompt.txt"

# ============================================
# 全局搜索配置
# ============================================

global_search:
  llm:
    model: ${LLM_MODEL}
    temperature: 0.0
  map_prompt: "见prompts/global_map.txt"
  reduce_prompt: "见prompts/global_reduce.txt"
  knowledge_prompt: "见prompts/global_knowledge.txt"
  dynamic_search:
    llm:
      model: ${LLM_MODEL}
      temperature: 0.0
    reduce_prompt: "见prompts/global_reduce.txt"
    knowledge_prompt: "见prompts/global_knowledge.txt"
    num_candidates: 20
    num_community_reports: 5
    use_community_summary: true
```

#### 配置模板

GraphRAG-KG提供三种配置模板以适应不同环境：

```yaml
# config/default.yaml - 开发环境默认配置
# 使用较小的chunk size和较低的gleaning轮数以提高迭代速度
chunks:
  size: 600
  overlap: 100
entity_extraction:
  max_gleaning: 1
community_reports:
  max_length: 1000
```

```yaml
# config/fast.yaml - 快速索引配置
# 跳过社区检测和摘要生成，仅构建基础索引
skip_workflows:
  - create_community_reports
  - create_community_reports_text
community_reports:
  max_length: 500
```

```yaml
# config/production.yaml - 生产环境配置
# 完整索引流程，高质量设置
chunks:
  size: 1200
  overlap: 200
entity_extraction:
  max_gleaning: 3
community_reports:
  max_length: 2000
  max_input_length: 8000
```

### 13.3.2 配置加载实现

```python
# src/graphrag_kg/core/config_loader.py

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from dotenv import load_dotenv


class ConfigLoader:
    """配置加载器：合并YAML配置和环境变量"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        load_dotenv()  # 加载.env文件
    
    def load(
        self,
        config_name: str = "default",
        settings_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        加载配置（支持分层合并）
        
        优先级（从高到低）：
        1. 环境变量（最高优先级）
        2. settings.yaml（项目根目录）
        3. config/{name}.yaml（模板配置）
        """
        config = {}
        
        # 1. 加载模板配置
        template_path = self.config_dir / f"{config_name}.yaml"
        if template_path.exists():
            with open(template_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
        
        # 2. 加载settings.yaml（覆盖模板）
        if settings_path and Path(settings_path).exists():
            with open(settings_path, "r", encoding="utf-8") as f:
                settings = yaml.safe_load(f) or {}
            self._deep_merge(config, settings)
        elif (Path("settings.yaml")).exists():
            with open("settings.yaml", "r", encoding="utf-8") as f:
                settings = yaml.safe_load(f) or {}
            self._deep_merge(config, settings)
        
        # 3. 解析环境变量引用（${VAR_NAME}）
        config = self._resolve_env_vars(config)
        
        return config
    
    def _deep_merge(
        self, base: Dict, override: Dict
    ) -> None:
        """深度合并字典"""
        for key, value in override.items():
            if (
                key in base
                and isinstance(base[key], dict)
                and isinstance(value, dict)
            ):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def _resolve_env_vars(self, obj: Any) -> Any:
        """递归解析所有${VAR}引用"""
        if isinstance(obj, str):
            import re
            pattern = r'\$\{([^}]+)\}'
            
            def replace_env(match):
                var_name = match.group(1)
                return os.getenv(var_name, match.group(0))
            
            return re.sub(pattern, replace_env, obj)
        
        elif isinstance(obj, dict):
            return {
                key: self._resolve_env_vars(value)
                for key, value in obj.items()
            }
        
        elif isinstance(obj, list):
            return [self._resolve_env_vars(item) for item in obj]
        
        return obj
```

### 13.3.3 配置模型验证

```python
# src/graphrag_kg/core/config.py

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from enum import Enum


class SearchType(str, Enum):
    """支持的检索策略类型"""
    LOCAL = "local"
    GLOBAL = "global"
    DRIFT = "drift"
    BASIC = "basic"
    AUTO = "auto"


class LLMConfig(BaseModel):
    """LLM配置模型"""
    api_key: str
    model: str
    api_base: str = "https://api.deepseek.com"
    max_tokens: int = 4000
    temperature: float = 0.0
    request_timeout: float = 60.0
    model_supports_json: bool = True


class EmbeddingConfig(BaseModel):
    """嵌入模型配置"""
    model: str = "bge-m3"
    api_base: str = "http://localhost:11434"
    dimensions: int = 1024
    request_timeout: float = 60.0
    
    @validator('dimensions')
    def validate_dimensions(cls, v):
        """验证向量维度"""
        supported = [384, 512, 768, 1024, 1536, 2048, 3072]
        if v not in supported:
            raise ValueError(
                f"不支持的向量维度 {v}，支持的维度：{supported}"
            )
        return v


class ChunkConfig(BaseModel):
    """文档分块配置"""
    size: int = 800
    overlap: int = 100
    group_by_columns: List[str] = ["id"]


class EntityExtractionConfig(BaseModel):
    """实体提取配置"""
    entity_types: List[str] = Field(
        default_factory=lambda: [
            "DRUG", "COMPANY", "DISEASE", "REGULATORY"
        ]
    )
    max_gleaning: int = 2


class CommunityConfig(BaseModel):
    """社区检测配置"""
    max_cluster_size: int = 10
    seed: int = 42
    use_lcc: bool = True
    hierarchical_levels: int = 2
    resolution: float = 1.0


class Neo4jConfig(BaseModel):
    """Neo4j数据库配置"""
    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: str = "password"
    database: str = "neo4j"
    connection_pool_size: int = 10
    connection_timeout: float = 30.0
    max_retry_count: int = 3


class AppConfig(BaseModel):
    """应用总配置"""
    llm: LLMConfig
    embedding: EmbeddingConfig
    chunk: ChunkConfig = ChunkConfig()
    entity_extraction: EntityExtractionConfig = EntityExtractionConfig()
    community: CommunityConfig = CommunityConfig()
    neo4j: Neo4jConfig = Neo4jConfig()
    search_type: SearchType = SearchType.AUTO
    storage_dir: str = "./output"
    lance_db_uri: str = "./data/lancedb"
```

---

## 13.4 测试数据生成

GraphRAG-KG项目内置了一个制药供应链场景的测试数据集。该数据集模拟了一个包含59个实体、153条关系和12种实体类型的制药行业知识图谱。

### 13.4.1 数据模型

```python
# src/graphrag_kg/test_data/pharma_supply_chain.py

from dataclasses import dataclass, field
from typing import List, Optional
import json


@dataclass
class PharmaEntity:
    """制药供应链实体"""
    id: str
    name: str
    type: str  # DRUG, COMPANY, DISEASE, REGULATORY, CLINICAL_TRIAL, etc.
    description: str
    attributes: dict = field(default_factory=dict)


@dataclass
class PharmaRelationship:
    """实体间关系"""
    source_id: str
    target_id: str
    type: str  # PRODUCES, TREATS, APPROVES, PARTNERS_WITH, etc.
    description: str
    weight: float = 1.0


@dataclass
class PharmaDocument:
    """文档（模拟非结构化文本）"""
    id: str
    title: str
    content: str
    entities: List[str] = field(default_factory=list)
```

### 13.4.2 场景设计

测试数据模拟了完整的制药供应链场景，包括以下核心故事线：

**故事线1：Keytruda（帕博利珠单抗）**
- 药物：Keytruda（默沙东的PD-1抑制剂）
- 适应症：非小细胞肺癌、黑色素瘤、胃癌等
- 关键事件：FDA批准、联合疗法、专利到期

**故事线2：辉瑞-BioNTech疫苗供应链**
- 涉及：辉瑞（生产）、BioNTech（技术）、Moderna（竞品）
- 流程：mRNA合成→脂质纳米颗粒包裹→冷链运输
- 监管：FDA紧急使用授权、全球分发

**故事线3：辉瑞收购Seagen**
- 收购金额：430亿美元
- 目标：ADC（抗体偶联药物）管线
- 影响：肿瘤药物市场竞争格局变化

### 13.4.3 数据生成实现

```python
# 继续 pharma_supply_chain.py

class PharmaSupplyChainData:
    """制药供应链测试数据生成器"""
    
    @staticmethod
    def create_entities() -> List[PharmaEntity]:
        """创建59个测试实体"""
        entities = [
            # === 药物（DRUG）===
            PharmaEntity("DRG001", "Keytruda（帕博利珠单抗）", "DRUG",
                "默沙东开发的PD-1抑制剂，用于多种癌症治疗"),
            PharmaEntity("DRG002", "Comirnaty（复必泰）", "DRUG",
                "辉瑞-BioNTech联合开发的mRNA COVID-19疫苗"),
            PharmaEntity("DRG003", "Adcetris（本妥昔单抗）", "DRUG",
                "Seagen开发的ADC药物，靶向CD30"),
            PharmaEntity("DRG004", "Padcev（恩诺单抗）", "DRUG",
                "Seagen开发的ADC药物，靶向Nectin-4"),
            
            # === 公司（COMPANY）===
            PharmaEntity("COM001", "默沙东（MSD）", "COMPANY",
                "全球领先的制药企业，Keytruda的开发商"),
            PharmaEntity("COM002", "辉瑞（Pfizer）", "COMPANY",
                "全球最大制药企业之一，总部纽约"),
            PharmaEntity("COM003", "BioNTech", "COMPANY",
                "德国生物技术公司，mRNA技术平台"),
            PharmaEntity("COM004", "Moderna", "COMPANY",
                "美国生物技术公司，mRNA疫苗竞争者"),
            PharmaEntity("COM005", "Seagen", "COMPANY",
                "ADC技术先驱，2023年被辉瑞收购"),
            PharmaEntity("COM006", "药明康德", "COMPANY",
                "中国领先的CRO/CDMO服务商"),
            
            # === 疾病（DISEASE）===
            PharmaEntity("DIS001", "非小细胞肺癌", "DISEASE",
                "最常见的肺癌类型，占肺癌病例的85%"),
            PharmaEntity("DIS002", "黑色素瘤", "DISEASE",
                "一种恶性皮肤癌，Keytruda的核心适应症"),
            PharmaEntity("DIS003", "胃癌", "DISEASE",
                "全球第五大常见癌症"),
            PharmaEntity("DIS004", "COVID-19", "DISEASE",
                "由SARS-CoV-2病毒引起的传染病"),
            PharmaEntity("DIS005", "霍奇金淋巴瘤", "DISEASE",
                "一种淋巴系统恶性肿瘤"),
            
            # === 监管机构（REGULATORY）===
            PharmaEntity("REG001", "FDA", "REGULATORY",
                "美国食品药品监督管理局"),
            PharmaEntity("REG002", "EMA", "REGULATORY",
                "欧洲药品管理局"),
            PharmaEntity("REG003", "NMPA", "REGULATORY",
                "中国国家药品监督管理局"),
            PharmaEntity("REG004", "孤儿药认定", "REGULATORY",
                "FDA对治疗罕见病药物的特殊认定"),
            
            # === 临床试验（CLINICAL_TRIAL）===
            PharmaEntity("TRI001", "KEYNOTE-024", "CLINICAL_TRIAL",
                "Keytruda一线治疗PD-L1阳性NSCLC的III期试验"),
            PharmaEntity("TRI002", "KEYNOTE-189", "CLINICAL_TRIAL",
                "Keytruda联合化疗一线治疗NSCLC的III期试验"),
            PharmaEntity("TRI003", "C106", "CLINICAL_TRIAL",
                "Seagen的ADC药物临床试验"),
            
            # === 人物（PERSON）===
            PharmaEntity("PER001", "Albert Bourla", "PERSON",
                "辉瑞CEO，主导了新冠疫苗开发和Seagen收购"),
            PharmaEntity("PER002", "Dr. Anthony Fauci", "PERSON",
                "美国国家过敏和传染病研究所所长"),
            PharmaEntity("PER003", "Ugur Sahin", "PERSON",
                "BioNTech联合创始人兼CEO"),
            
            # === 流程（PROCESS）===
            PharmaEntity("PRO001", "mRNA合成", "PROCESS",
                "通过体外转录合成mRNA疫苗原液"),
            PharmaEntity("PRO002", "脂质纳米颗粒包裹", "PROCESS",
                "使用LNP技术包裹mRNA以递送入细胞"),
            PharmaEntity("PRO003", "冷链运输", "PROCESS",
                "维持-70°C至-20°C的疫苗运输链"),
            PharmaEntity("PRO004", "无菌灌装", "PROCESS",
                "在无菌条件下将疫苗分装到西林瓶"),
            
            # === 地点（LOCATION）===
            PharmaEntity("LOC001", "上海", "LOCATION",
                "中国金融中心，药明康德总部所在地"),
            PharmaEntity("LOC002", "波士顿", "LOCATION",
                "美国生物技术产业中心"),
            PharmaEntity("LOC003", "马里兰州", "LOCATION",
                "Moderna总部所在地"),
            PharmaEntity("LOC004", "纽约", "LOCATION",
                "辉瑞全球总部所在地"),
            
            # === 材料（MATERIAL）===
            PharmaEntity("MAT001", "脂质纳米颗粒", "MATERIAL",
                "mRNA疫苗的关键递送载体材料"),
            PharmaEntity("MAT002", "mRNA模板", "MATERIAL",
                "体外转录使用的DNA模板"),
            PharmaEntity("MAT003", "玻璃瓶", "MATERIAL",
                "疫苗分装使用的I型硼硅玻璃瓶"),
            
            # === 设备（EQUIPMENT）===
            PharmaEntity("EQP001", "生物反应器", "EQUIPMENT",
                "用于细胞培养和生物制剂生产"),
            PharmaEntity("EQP002", "高效液相色谱仪", "EQUIPMENT",
                "用于蛋白质纯化和分析"),
            PharmaEntity("EQP003", "低温冰箱", "EQUIPMENT",
                "用于疫苗和生物制品的低温储存"),
            
            # === 合同（CONTRACT）===
            PharmaEntity("CON001", "辉瑞-BioNTech合作协议", "CONTRACT",
                "辉瑞与BioNTech联合开发COVID-19疫苗的协议"),
            PharmaEntity("CON002", "辉瑞-Seagen收购协议", "CONTRACT",
                "辉瑞以430亿美元收购Seagen的协议"),
            
            # === 事件（EVENT）===
            PharmaEntity("EVT001", "JPM医疗健康大会", "EVENT",
                "每年在旧金山举行的全球医疗健康投资峰会"),
            PharmaEntity("EVT002", "辉瑞收购Seagen", "EVENT",
                "2023年辉瑞以430亿美元收购ADC领军企业Seagen"),
        ]
        
        return entities
    
    @staticmethod
    def create_relationships() -> List[PharmaRelationship]:
        """创建153条关系"""
        relationships = [
            # 药物-公司关系
            PharmaRelationship("DRG001", "COM001", "DEVELOPED_BY",
                "Keytruda由默沙东开发"),
            PharmaRelationship("DRG002", "COM002", "DEVELOPED_BY",
                "Comirnaty由辉瑞生产"),
            PharmaRelationship("DRG002", "COM003", "CO_DEVELOPED_BY",
                "Comirnaty由辉瑞和BioNTech共同开发"),
            PharmaRelationship("DRG003", "COM005", "DEVELOPED_BY",
                "Adcetris由Seagen开发"),
            PharmaRelationship("DRG004", "COM005", "DEVELOPED_BY",
                "Padcev由Seagen开发"),
            
            # 药物-疾病关系
            PharmaRelationship("DRG001", "DIS001", "TREATS",
                "Keytruda用于治疗非小细胞肺癌"),
            PharmaRelationship("DRG001", "DIS002", "TREATS",
                "Keytruda用于治疗黑色素瘤"),
            PharmaRelationship("DRG001", "DIS003", "TREATS",
                "Keytruda用于治疗胃癌"),
            PharmaRelationship("DRG002", "DIS004", "PREVENTS",
                "Comirnaty用于预防COVID-19"),
            
            # 公司-收购关系
            PharmaRelationship("COM002", "COM005", "ACQUIRED",
                "辉瑞收购Seagen"),
            PharmaRelationship("COM002", "EVT002", "INVOLVED_IN",
                "辉瑞是Seagen收购方"),
            
            # 公司-合作关系
            PharmaRelationship("COM002", "COM003", "PARTNERS_WITH",
                "辉瑞与BioNTech合作开发疫苗"),
            PharmaRelationship("COM001", "COM006", "PARTNERS_WITH",
                "默沙东与药明康德有CRO合作"),
            
            # 监管关系
            PharmaRelationship("REG001", "DRG001", "APPROVED",
                "FDA批准了Keytruda"),
            PharmaRelationship("REG001", "DRG002", "AUTHORIZED",
                "FDA授予Comirnaty紧急使用授权"),
            PharmaRelationship("REG003", "DRG001", "APPROVED",
                "NMPA批准了Keytruda在中国上市"),
            
            # 临床试验关系
            PharmaRelationship("TRI001", "DRG001", "EVALUATES",
                "KEYNOTE-024试验评估Keytruda"),
            PharmaRelationship("TRI002", "DRG001", "EVALUATES",
                "KEYNOTE-189试验评估Keytruda联合化疗"),
            PharmaRelationship("TRI001", "DIS001", "STUDIES",
                "KEYNOTE-024研究非小细胞肺癌"),
            
            # 流程关系
            PharmaRelationship("PRO001", "PRO002", "PRECEDES",
                "mRNA合成后需要进行脂质纳米颗粒包裹"),
            PharmaRelationship("PRO002", "PRO004", "PRECEDES",
                "包裹后的疫苗需要进行无菌灌装"),
            PharmaRelationship("PRO003", "DRG002", "REQUIRED_FOR",
                "Comirnaty需要冷链运输"),
            
            # 材料-流程关系
            PharmaRelationship("MAT001", "PRO002", "USED_IN",
                "脂质纳米颗粒用于mRNA包裹"),
            PharmaRelationship("MAT002", "PRO001", "USED_IN",
                "mRNA模板用于体外转录"),
        ]
        
        # 实际项目包含153条关系，此处仅展示核心示例
        return relationships
    
    @staticmethod
    def create_documents() -> List[str]:
        """生成模拟文档文本"""
        documents = [
            # 文档1：Keytruda介绍
            """
            Keytruda（帕博利珠单抗）是默沙东（MSD）开发的一种人源化抗PD-1单克隆抗体。
            它通过阻断PD-1/PD-L1通路来激活T细胞的抗肿瘤免疫应答。
            Keytruda已获得FDA批准用于治疗多种癌症，包括非小细胞肺癌、黑色素瘤、胃癌等。
            在中国，Keytruda也已获得NMPA批准上市。
            
            关键的临床试验包括KEYNOTE-024（一线治疗PD-L1阳性NSCLC）和KEYNOTE-189
            （联合化疗一线治疗非鳞状NSCLC），这些试验证明了Keytruda在肺癌治疗中的显著疗效。
            """,
            
            # 文档2：辉瑞-BioNTech疫苗
            """
            辉瑞（Pfizer）与BioNTech联合开发的COVID-19疫苗Comirnaty（复必泰）
            是全球首个获得FDA紧急使用授权的mRNA疫苗。该疫苗使用BioNTech的mRNA技术平台，
            通过脂质纳米颗粒（LNP）递送系统将编码刺突蛋白的mRNA导入人体细胞。
            
            疫苗的制造流程包括：mRNA模板制备、体外转录合成mRNA原液、
            脂质纳米颗粒包裹、无菌灌装和冷链运输。疫苗需要在-70°C至-20°C的条件下运输和储存。
            
            Moderna是mRNA疫苗领域的主要竞争者，其开发的Spikevax疫苗也获得了FDA授权。
            """,
            
            # 文档3：辉瑞收购Seagen
            """
            2023年，辉瑞（Pfizer）以430亿美元完成了对Seagen的收购。Seagen是
            ADC（抗体偶联药物）领域的先驱企业，拥有领先的ADC技术平台。
            
            Seagen的核心产品包括Adcetris（靶向CD30的ADC药物）和Padcev
            （靶向Nectin-4的ADC药物）。此次收购极大地加强了辉瑞在肿瘤药物领域的管线。
            
            ADC药物通过抗体将细胞毒性药物精准递送到肿瘤细胞，代表了靶向治疗的重要方向。
            辉瑞CEO Albert Bourla表示，Seagen的ADC技术与辉瑞的全球商业化能力结合，
            将加速创新抗癌药物的开发和上市。
            """,
        ]
        
        return documents


# 生成完整数据集
def get_test_data():
    """获取完整测试数据集"""
    data = PharmaSupplyChainData()
    
    return {
        "entities": data.create_entities(),
        "relationships": data.create_relationships(),
        "documents": data.create_documents(),
    }
```

---

## 13.5 索引流水线

### 13.5.1 索引流程实现

```python
# src/graphrag_kg/pipeline/indexing.py

import asyncio
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from enum import Enum
import time
import pandas as pd

from graphrag.api import build_index
from graphrag.config import create_graphrag_config

logger = logging.getLogger(__name__)


class IndexMethod(str, Enum):
    """索引方法"""
    STANDARD = "standard"   # 标准索引：完整流程
    FAST = "fast"           # 快速索引：跳过社区检测
    UPDATE = "update"       # 增量更新：仅处理新增文档


class IndexingPipeline:
    """索引流水线"""
    
    def __init__(
        self,
        config: Dict[str, Any],
        root_dir: str = ".",
    ):
        self.config = config
        self.root_dir = Path(root_dir)
        self.output_dir = self.root_dir / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    async def run(
        self,
        documents: List[str],
        method: IndexMethod = IndexMethod.STANDARD,
        config_name: str = "default",
    ) -> Dict[str, Any]:
        """
        执行索引流水线
        
        Args:
            documents: 待索引的文档列表
            method: 索引方法（standard/fast/update）
            config_name: 配置模板名称
        
        Returns:
            索引结果统计
        """
        start_time = time.time()
        
        print(f"[索引] 开始 {method.value} 索引，文档数：{len(documents)}")
        
        # 1. 准备文档数据
        doc_df = self._prepare_documents(documents)
        print(f"[索引] 文档分块完成，共 {len(doc_df)} 个文本块")
        
        # 2. 创建GraphRAG配置
        graphrag_config = create_graphrag_config(
            self.config,
            root_dir=str(self.root_dir),
        )
        
        # 3. 执行索引
        if method == IndexMethod.FAST:
            # 快速索引：跳过社区报告生成
            graphrag_config.skip_workflows = [
                "create_community_reports",
                "create_community_reports_text",
            ]
        elif method == IndexMethod.UPDATE:
            # 增量更新：跳过已存在的部分
            graphrag_config.update_index_output = True
        
        # 调用GraphRAG核心索引函数
        index_result = await build_index(
            config=graphrag_config,
            documents=doc_df,
            is_resume=False,
        )
        
        elapsed = time.time() - start_time
        
        # 4. 收集统计信息
        stats = self._collect_stats(index_result, elapsed)
        
        print(f"[索引] 完成！耗时 {elapsed:.1f}秒")
        print(f"  - 文档数：{stats['documents_count']}")
        print(f"  - 实体数：{stats['entities_count']}")
        print(f"  - 关系数：{stats['relationships_count']}")
        print(f"  - 社区数：{stats['communities_count']}")
        print(f"  - 输出目录：{self.output_dir}")
        
        return stats
    
    def _prepare_documents(
        self, documents: List[str]
    ) -> pd.DataFrame:
        """
        将文档列表转换为DataFrame格式。
        
        GraphRAG要求输入为包含"id"和"text"列的DataFrame。
        """
        import uuid
        
        rows = []
        for i, doc in enumerate(documents):
            doc_id = f"doc_{i:04d}_{uuid.uuid4().hex[:8]}"
            rows.append({
                "id": doc_id,
                "title": f"Document {i+1}",
                "text": doc,
            })
        
        return pd.DataFrame(rows)
    
    def _collect_stats(
        self, index_result: Any, elapsed: float
    ) -> Dict[str, Any]:
        """收集索引统计信息"""
        stats = {
            "index_time_seconds": round(elapsed, 1),
            "documents_count": 0,
            "entities_count": 0,
            "relationships_count": 0,
            "communities_count": 0,
            "output_files": [],
        }
        
        # 解析索引结果（根据实际输出格式调整）
        if hasattr(index_result, "stats"):
            result_stats = index_result.stats
            stats["documents_count"] = result_stats.get(
                "num_documents", 0
            )
            stats["entities_count"] = result_stats.get(
                "num_entities", 0
            )
            stats["relationships_count"] = result_stats.get(
                "num_relationships", 0
            )
            stats["communities_count"] = result_stats.get(
                "num_communities", 0
            )
        
        # 列出输出文件
        parquet_files = list(self.output_dir.glob("**/*.parquet"))
        stats["output_files"] = [str(f) for f in parquet_files]
        
        return stats
```

### 13.5.2 索引输出

索引完成后，输出目录包含以下Parquet文件：

```
output/
├── artifacts/
│   ├── create_base_documents.parquet       # 基础文档
│   ├── create_base_text_units.parquet      # 文本块
│   ├── create_base_extracted_entities.parquet  # 提取的实体
│   ├── create_base_extracted_relationships.parquet  # 提取的关系
│   ├── create_final_communities.parquet    # 社区
│   ├── create_final_community_reports.parquet  # 社区报告
│   ├── create_final_text_units.parquet     # 最终文本块
│   ├── create_final_documents.parquet      # 最终文档
│   └── create_final_entities.parquet       # 最终实体
├── lancedb/                                # LanceDB向量存储
│   └── (向量索引文件)
└── stats.json                              # 索引统计
```

---

## 13.6 Neo4j同步

### 13.6.1 同步实现

```python
# src/graphrag_kg/neo4j/sync.py

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
import pandas as pd

from neo4j import GraphDatabase, AsyncGraphDatabase
from neo4j.exceptions import ServiceUnavailable

logger = logging.getLogger(__name__)


class Neo4jSyncer:
    """
    Parquet → Neo4j 数据同步器
    
    将GraphRAG索引输出的Parquet文件同步到Neo4j图数据库。
    支持批量MERGE操作，确保幂等性。
    """
    
    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        username: str = "neo4j",
        password: str = "password",
        database: str = "neo4j",
        connection_pool_size: int = 10,
    ):
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self.connection_pool_size = connection_pool_size
        
        self._driver = None
    
    def connect(self) -> bool:
        """连接Neo4j数据库"""
        try:
            self._driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password),
                max_connection_pool_size=self.connection_pool_size,
            )
            # 验证连接
            with self._driver.session(database=self.database) as session:
                result = session.run("RETURN 1 AS test")
                record = result.single()
                if record and record["test"] == 1:
                    logger.info("Neo4j连接成功")
                    return True
            return False
        except ServiceUnavailable as e:
            logger.error(f"Neo4j连接失败：{e}")
            return False
    
    def close(self):
        """关闭连接"""
        if self._driver:
            self._driver.close()
    
    def sync_all(
        self, artifacts_dir: str
    ) -> Dict[str, int]:
        """
        同步所有Parquet文件到Neo4j
        
        Args:
            artifacts_dir: Parquet文件目录
        
        Returns:
            各类型数据的同步计数
        """
        stats = {}
        artifacts_path = Path(artifacts_dir)
        
        # 1. 同步实体
        entity_file = artifacts_path / "create_final_entities.parquet"
        if entity_file.exists():
            count = self._sync_entities(str(entity_file))
            stats["entities"] = count
        
        # 2. 同步社区
        community_file = (
            artifacts_path / "create_final_communities.parquet"
        )
        if community_file.exists():
            count = self._sync_communities(str(community_file))
            stats["communities"] = count
        
        # 3. 同步文档
        doc_file = artifacts_path / "create_final_documents.parquet"
        if doc_file.exists():
            count = self._sync_documents(str(doc_file))
            stats["documents"] = count
        
        # 4. 同步文本块
        text_unit_file = (
            artifacts_path / "create_final_text_units.parquet"
        )
        if text_unit_file.exists():
            count = self._sync_text_units(str(text_unit_file))
            stats["text_units"] = count
        
        # 5. 同步关系
        rel_file = (
            artifacts_path
            / "create_base_extracted_relationships.parquet"
        )
        if rel_file.exists():
            count = self._sync_relationships(str(rel_file))
            stats["relationships"] = count
        
        return stats
    
    def _sync_entities(self, parquet_path: str) -> int:
        """同步实体节点"""
        df = pd.read_parquet(parquet_path)
        
        query = """
        UNWIND $rows AS row
        MERGE (e:Entity {id: row.id})
        SET e.name = row.name,
            e.type = row.type,
            e.description = row.description,
            e.graphrag_id = row.graphrag_id,
            e.human_readable_id = row.human_readable_id
        RETURN count(e) AS count
        """
        
        with self._driver.session(database=self.database) as session:
            result = session.run(query, rows=df.to_dict("records"))
            record = result.single()
            return record["count"] if record else 0
    
    def _sync_communities(self, parquet_path: str) -> int:
        """同步社区节点"""
        df = pd.read_parquet(parquet_path)
        
        query = """
        UNWIND $rows AS row
        MERGE (c:Community {id: row.id})
        SET c.title = row.title,
            c.summary = row.summary,
            c.level = row.level,
            c.size = row.size
        RETURN count(c) AS count
        """
        
        with self._driver.session(database=self.database) as session:
            result = session.run(query, rows=df.to_dict("records"))
            record = result.single()
            return record["count"] if record else 0
    
    def _sync_documents(self, parquet_path: str) -> int:
        """同步文档节点"""
        df = pd.read_parquet(parquet_path)
        
        query = """
        UNWIND $rows AS row
        MERGE (d:Document {id: row.id})
        SET d.title = row.title,
            d.summary = row.summary,
            d.text_length = row.text_length
        RETURN count(d) AS count
        """
        
        with self._driver.session(database=self.database) as session:
            result = session.run(query, rows=df.to_dict("records"))
            record = result.single()
            return record["count"] if record else 0
    
    def _sync_text_units(self, parquet_path: str) -> int:
        """同步文本块节点"""
        df = pd.read_parquet(parquet_path)
        
        query = """
        UNWIND $rows AS row
        MERGE (t:TextUnit {id: row.id})
        SET t.text = row.text,
            t.n_tokens = row.n_tokens,
            t.chunk_order = row.chunk_order
        RETURN count(t) AS count
        """
        
        with self._driver.session(database=self.database) as session:
            result = session.run(query, rows=df.to_dict("records"))
            record = result.single()
            return record["count"] if record else 0
    
    def _sync_relationships(self, parquet_path: str) -> int:
        """同步关系（RELATES_TO）"""
        df = pd.read_parquet(parquet_path)
        
        query = """
        UNWIND $rows AS row
        MATCH (source:Entity {id: row.source_id})
        MATCH (target:Entity {id: row.target_id})
        MERGE (source)-[r:RELATES_TO {id: row.id}]->(target)
        SET r.relationship = row.relationship,
            r.description = row.description,
            r.weight = row.weight
        RETURN count(r) AS count
        """
        
        with self._driver.session(database=self.database) as session:
            result = session.run(query, rows=df.to_dict("records"))
            record = result.single()
            return record["count"] if record else 0
    
    def create_indexes(self):
        """创建Neo4j索引（提升查询性能）"""
        queries = [
            "CREATE INDEX entity_id IF NOT EXISTS FOR (e:Entity) ON (e.id)",
            "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX community_id IF NOT EXISTS FOR (c:Community) ON (c.id)",
            "CREATE INDEX document_id IF NOT EXISTS FOR (d:Document) ON (d.id)",
            "CREATE INDEX text_unit_id IF NOT EXISTS FOR (t:TextUnit) ON (t.id)",
        ]
        
        with self._driver.session(database=self.database) as session:
            for query in queries:
                try:
                    session.run(query)
                    logger.info(f"创建索引成功：{query[:50]}...")
                except Exception as e:
                    logger.warning(f"创建索引失败：{e}")
```

### 13.6.2 Neo4j Schema

同步完成后，Neo4j中的图结构如下：

```cypher
// 节点类型
(:Entity {id, name, type, description, graphrag_id, human_readable_id})
(:Community {id, title, summary, level, size})
(:Document {id, title, summary, text_length})
(:TextUnit {id, text, n_tokens, chunk_order})

// 关系类型
(:Entity)-[:RELATES_TO {id, relationship, description, weight}]->(:Entity)
(:Entity)-[:IN_COMMUNITY]->(:Community)
(:Document)-[:HAS_TEXT_UNIT]->(:TextUnit)
(:TextUnit)-[:MENTIONS]->(:Entity)
```

---

## 13.7 查询引擎

### 13.7.1 查询引擎实现

```python
# src/graphrag_kg/query/engine.py

import logging
from typing import Optional, List, Dict, Any
from enum import Enum

from graphrag.query.local_search import LocalSearch
from graphrag.query.global_search import GlobalSearch
from graphrag.query.drift_search import DriftSearch
from graphrag.query.basic_search import BasicSearch

logger = logging.getLogger(__name__)


class SearchType(str, Enum):
    """检索策略类型"""
    LOCAL = "local"
    GLOBAL = "global"
    DRIFT = "drift"
    BASIC = "basic"
    AUTO = "auto"


class QueryEngine:
    """
    统一查询引擎
    
    支持4种检索策略的自动路由：
    - local: 局部搜索，适合具体实体查询
    - global: 全局搜索，适合综合/趋势查询
    - drift: 漂移搜索，适合多跳推理查询
    - basic: 基础搜索，适合简单查询
    - auto: 自动路由
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        llm=None,
        token_encoder=None,
    ):
        self.config = config
        self.llm = llm
        self.token_encoder = token_encoder
        
        # 初始化各搜索策略
        self._init_searchers()
    
    def _init_searchers(self):
        """初始化所有搜索策略"""
        self.searchers = {}
        
        # Local Search
        if "local_search" in self.config:
            self.searchers["local"] = LocalSearch(
                llm=self.llm,
                stream=True,
                context_builder="local",
                token_encoder=self.token_encoder,
                system_prompt=self.config["local_search"].get(
                    "system_prompt"
                ),
                response_type="multiple paragraphs",
            )
        
        # Global Search
        if "global_search" in self.config:
            self.searchers["global"] = GlobalSearch(
                llm=self.llm,
                context_builder="global",
                token_encoder=self.token_encoder,
                dynamic_community_selection=True,
                map_system_prompt=self.config["global_search"].get(
                    "map_prompt"
                ),
                reduce_system_prompt=self.config["global_search"].get(
                    "reduce_prompt"
                ),
            )
        
        # Drift Search
        if "drift_search" in self.config:
            self.searchers["drift"] = DriftSearch(
                llm=self.llm,
                n=5,
                max_depth=3,
                drift_k=10,
                primer_folds=3,
                noise_threshold=0.3,
                use_cosine_reranker=True,
            )
        
        # Basic Search
        self.searchers["basic"] = BasicSearch(
            llm=self.llm,
            token_encoder=self.token_encoder,
        )
    
    async def search(
        self,
        query: str,
        search_type: SearchType = SearchType.AUTO,
        top_k: int = 10,
        conversation_id: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        执行查询
        
        Args:
            query: 查询文本
            search_type: 检索策略
            top_k: 检索结果数量
            conversation_id: 会话ID（多轮对话）
        
        Returns:
            {
                "answer": "生成的回答",
                "sources": [...],
                "search_type": "使用的策略",
                "latency_ms": 1234,
                "token_usage": {...}
            }
        """
        # 自动路由
        if search_type == SearchType.AUTO:
            search_type = self._route_query(query)
        
        searcher = self.searchers.get(search_type)
        if not searcher:
            searcher = self.searchers.get("basic")
            search_type = "basic"
        
        # 执行搜索
        result = await searcher.search(
            query=query,
            conversation_history=None,
            **kwargs,
        )
        
        return {
            "answer": result.response,
            "sources": self._format_sources(result),
            "search_type": search_type,
            "latency_ms": result.completion_time,
            "token_usage": {
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
            },
        }
    
    def _route_query(self, query: str) -> str:
        """
        自动路由到最优搜索策略
        
        路由逻辑：
        - 短查询（<10词）且包含实体名 → local
        - 抽象/总结类查询 → global
        - 多跳推理查询 → drift
        - 其他 → basic
        """
        query_lower = query.lower()
        word_count = len(query.split())
        
        # 关键词匹配
        global_keywords = [
            "趋势", "概述", "总结", "综述", "分析",
            "trend", "overview", "summary", "survey",
        ]
        drift_keywords = [
            "影响", "导致", "关系", "因果", "路径",
            "impact", "cause", "relation", "pathway",
        ]
        
        if word_count < 10 and self._has_entity_reference(query):
            return "local"
        elif any(kw in query_lower for kw in global_keywords):
            return "global"
        elif any(kw in query_lower for kw in drift_keywords):
            return "drift"
        else:
            return "basic"
    
    def _has_entity_reference(self, query: str) -> bool:
        """检查查询是否包含已知实体引用"""
        # 实际实现中，会查询实体索引
        # 简单实现：检测大写词或中文字符
        import re
        chinese_pattern = r'[一-鿿]{2,}'
        return bool(re.search(chinese_pattern, query))
    
    def _format_sources(self, result) -> List[Dict]:
        """格式化来源信息"""
        sources = []
        
        if hasattr(result, "contexts"):
            for ctx in result.contexts:
                sources.append({
                    "title": getattr(ctx, "title", ""),
                    "content": ctx.text[:200] if hasattr(ctx, "text") else "",
                    "relevance": getattr(ctx, "score", 0),
                })
        
        return sources
```

### 13.7.3 Cypher查询示例

除了自动路由，查询引擎还支持直接使用Cypher查询Neo4j：

```python
class GraphQueryExecutor:
    """图查询执行器（Cypher）"""
    
    def __init__(self, driver):
        self.driver = driver
    
    def find_entity_relations(
        self, entity_name: str, max_depth: int = 2
    ) -> List[Dict]:
        """
        查询实体及其邻接关系
        
        Cypher: MATCH (e:Entity {name: $name})-[r:RELATES_TO*1..2]-(related)
        """
        query = """
        MATCH (e:Entity {name: $name})
        OPTIONAL MATCH path = (e)-[r:RELATES_TO*1..$max_depth]-(related:Entity)
        RETURN e.name AS source,
               [rel IN relationships(path) | type(rel)] AS relation_types,
               collect(DISTINCT related.name) AS related_entities
        """
        
        with self.driver.session() as session:
            result = session.run(
                query,
                name=entity_name,
                max_depth=max_depth,
            )
            return [record.data() for record in result]
    
    def find_community_entities(
        self, community_id: str
    ) -> List[Dict]:
        """
        查询社区内所有实体
        
        Cypher: MATCH (e:Entity)-[:IN_COMMUNITY]->(:Community {id: $id})
        """
        query = """
        MATCH (e:Entity)-[:IN_COMMUNITY]->(c:Community {id: $id})
        RETURN e.name, e.type, e.description
        ORDER BY e.type
        """
        
        with self.driver.session() as session:
            result = session.run(query, id=community_id)
            return [record.data() for record in result]
    
    def find_shortest_path(
        self, source_name: str, target_name: str
    ) -> List[Dict]:
        """
        查找两个实体之间的最短路径
        
        Cypher: MATCH path = shortestPath(
            (s:Entity {name: $src})-[:RELATES_TO*]-(t:Entity {name: $tgt})
        )
        """
        query = """
        MATCH path = shortestPath(
            (s:Entity {name: $source})-[r:RELATES_TO*]-(t:Entity {name: $target})
        )
        RETURN [node IN nodes(path) | node.name] AS entity_path,
               [rel IN relationships(path) | rel.relationship] AS relation_path
        """
        
        with self.driver.session() as session:
            result = session.run(
                query,
                source=source_name,
                target=target_name,
            )
            return [record.data() for record in result]
```

---

## 13.8 性能调优

### 13.8.1 向量维度匹配

使用bge-m3嵌入模型时，确保向量维度在所有组件之间一致：

```python
# 向量维度验证
def validate_embedding_dimensions(
    model_name: str = "bge-m3",
    config_dim: int = 1024,
):
    """
    验证嵌入模型的向量维度配置
    
    常见模型的向量维度：
    - bge-m3: 1024
    - bge-large-zh: 1024
    - bge-base-zh: 768
    - bge-small-zh: 512
    - text-embedding-3-small: 1536
    - text-embedding-3-large: 3072
    """
    model_dimensions = {
        "bge-m3": 1024,
        "bge-large-zh": 1024,
        "bge-base-zh": 768,
        "bge-small-zh": 512,
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
    }
    
    expected_dim = model_dimensions.get(model_name)
    if expected_dim and expected_dim != config_dim:
        raise ValueError(
            f"向量维度不匹配：模型 {model_name} 期望 {expected_dim} 维，"
            f"但配置中设置为 {config_dim} 维。"
        )
    
    print(f"[验证] {model_name} 向量维度配置正确：{config_dim}")
```

### 13.8.2 连接池配置

```python
class ConnectionPoolConfig:
    """连接池配置"""
    
    def __init__(self):
        # Neo4j连接池
        self.neo4j_pool_size = 10
        self.neo4j_max_retry = 3
        
        # LLM连接池
        self.llm_max_connections = 5
        self.llm_timeout = 60
        
        # 数据库连接池
        self.db_pool_size = 20
        self.db_max_overflow = 10
    
    def optimize_pool_size(
        self, expected_qps: float, avg_latency_ms: float
    ):
        """
        根据预期QPS和平均延迟优化连接池大小
        
        公式：pool_size = QPS * avg_latency_seconds * (1 + buffer)
        """
        import math
        
        buffer = 0.3  # 30%缓冲
        optimal_size = math.ceil(
            expected_qps * (avg_latency_ms / 1000) * (1 + buffer)
        )
        
        self.neo4j_pool_size = max(5, min(50, optimal_size))
        return self.neo4j_pool_size
```

### 13.8.3 缓存策略

```python
class CacheManager:
    """多级缓存管理器"""
    
    def __init__(self, redis_client=None):
        self.redis = redis_client
        self.local_cache = {}  # 本地内存缓存
        self.default_ttl = {
            "community_summary": 86400,    # 24小时
            "entity_info": 3600,           # 1小时
            "search_result": 300,          # 5分钟
            "embedding": 86400 * 7,        # 7天
        }
    
    async def get(
        self, key: str, cache_type: str = "search_result"
    ) -> Optional[Any]:
        """获取缓存"""
        # 1. 本地缓存
        if key in self.local_cache:
            entry = self.local_cache[key]
            if time.time() - entry["timestamp"] < entry["ttl"]:
                return entry["value"]
            else:
                del self.local_cache[key]
        
        # 2. Redis缓存
        if self.redis:
            value = await self.redis.get(key)
            if value:
                return json.loads(value)
        
        return None
    
    async def set(
        self, key: str, value: Any, cache_type: str = "search_result"
    ):
        """设置缓存"""
        ttl = self.default_ttl.get(cache_type, 300)
        
        # 1. 本地缓存
        self.local_cache[key] = {
            "value": value,
            "timestamp": time.time(),
            "ttl": ttl,
        }
        
        # 限制本地缓存大小
        if len(self.local_cache) > 10000:
            self._evict_local_cache()
        
        # 2. Redis缓存
        if self.redis:
            await self.redis.setex(key, ttl, json.dumps(value))
    
    def _evict_local_cache(self):
        """淘汰本地缓存中的过期条目"""
        now = time.time()
        expired = [
            k for k, v in self.local_cache.items()
            if now - v["timestamp"] > v["ttl"]
        ]
        for k in expired:
            del self.local_cache[k]
```

---

## 13.9 完整使用示例

### 13.9.1 端到端流程

```python
# ============================================
# GraphRAG-KG 端到端使用示例
# ============================================

import asyncio
import os
from pathlib import Path

from graphrag_kg.core.config_loader import ConfigLoader
from graphrag_kg.pipeline.indexing import IndexingPipeline, IndexMethod
from graphrag_kg.query.engine import QueryEngine, SearchType
from graphrag_kg.neo4j.sync import Neo4jSyncer
from graphrag_kg.test_data.pharma_supply_chain import get_test_data


async def main():
    """完整的GraphRAG-KG使用流程"""
    
    print("=" * 60)
    print("GraphRAG-KG 端到端示例")
    print("=" * 60)
    
    # ========== 步骤1：加载配置 ==========
    print("\n[1/6] 加载配置...")
    config_loader = ConfigLoader()
    config = config_loader.load(
        config_name="default",
        settings_path="settings.yaml",
    )
    print("  配置加载完成")
    
    # ========== 步骤2：生成测试数据 ==========
    print("\n[2/6] 生成测试数据...")
    test_data = get_test_data()
    documents = test_data["documents"]
    print(f"  生成了 {len(documents)} 篇文档")
    print(f"  包含 {len(test_data['entities'])} 个实体")
    print(f"  包含 {len(test_data['relationships'])} 条关系")
    
    # ========== 步骤3：构建索引 ==========
    print("\n[3/6] 构建索引...")
    pipeline = IndexingPipeline(config)
    stats = await pipeline.run(
        documents=documents,
        method=IndexMethod.STANDARD,
    )
    print(f"  索引完成：{stats}")
    
    # ========== 步骤4：同步到Neo4j ==========
    print("\n[4/6] 同步到Neo4j...")
    syncer = Neo4jSyncer(
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        username=os.getenv("NEO4J_USERNAME", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", "password"),
    )
    
    if syncer.connect():
        syncer.create_indexes()
        sync_stats = syncer.sync_all(
            artifacts_dir="./output/artifacts"
        )
        print(f"  同步完成：{sync_stats}")
        syncer.close()
    else:
        print("  Neo4j未连接，跳过同步")
    
    # ========== 步骤5：执行查询 ==========
    print("\n[5/6] 执行查询...")
    engine = QueryEngine(config)
    
    # 查询1：局部搜索
    print("\n  --- 查询1：局部搜索 ---")
    result1 = await engine.search(
        query="Keytruda在非小细胞肺癌治疗中的应用？",
        search_type=SearchType.LOCAL,
    )
    print(f"  策略：{result1['search_type']}")
    print(f"  回答：{result1['answer'][:200]}...")
    print(f"  来源数：{len(result1['sources'])}")
    print(f"  延迟：{result1['latency_ms']}ms")
    
    # 查询2：全局搜索
    print("\n  --- 查询2：全局搜索 ---")
    result2 = await engine.search(
        query="2024年全球制药行业的主要趋势？",
        search_type=SearchType.GLOBAL,
    )
    print(f"  策略：{result2['search_type']}")
    print(f"  回答：{result2['answer'][:200]}...")
    
    # 查询3：自动路由
    print("\n  --- 查询3：自动路由 ---")
    result3 = await engine.search(
        query="辉瑞收购Seagen对ADC药物管线的影响？",
        search_type=SearchType.AUTO,
    )
    print(f"  策略：{result3['search_type']}")
    print(f"  回答：{result3['answer'][:200]}...")
    
    # ========== 步骤6：输出汇总 ==========
    print("\n[6/6] 汇总信息")
    print(f"  输出目录：{Path('output').absolute()}")
    print(f"  LanceDB：{config.get('lance_db_uri', './data/lancedb')}")
    
    print("\n" + "=" * 60)
    print("示例完成！")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
```

### 13.9.2 CLI命令

```bash
# 查看所有命令
python -m graphrag_kg --help

# 构建索引
python -m graphrag_kg index --config default --method standard

# 快速索引（跳过社区检测）
python -m graphrag_kg index --config fast --method fast

# 查询
python -m graphrag_kg query "Keytruda的作用机制是什么？"

# 同步到Neo4j
python -m graphrag_kg neo4j-sync --artifacts-dir ./output/artifacts

# 生成测试数据
python -m graphrag_kg generate-test-data --output ./test_data.json
```

### 13.9.3 Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  # Neo4j图数据库
  neo4j:
    image: neo4j:5.14-enterprise
    container_name: graphrag-neo4j
    ports:
      - "7474:7474"   # HTTP管理界面
      - "7687:7687"   # Bolt协议
    environment:
      - NEO4J_AUTH=neo4j/password
      - NEO4J_PLUGINS=["apoc", "graph-data-science"]
      - NEO4J_dbms_memory_heap_maxSize=4G
    volumes:
      - ./data/neo4j/data:/data
      - ./data/neo4j/logs:/logs
      - ./data/neo4j/import:/var/lib/neo4j/import
    healthcheck:
      test: ["CMD", "cypher-shell", "-u", "neo4j", "-p", "password",
             "RETURN 1"]
      interval: 30s
      timeout: 10s
      retries: 5

  # Ollama嵌入服务
  ollama:
    image: ollama/ollama:latest
    container_name: graphrag-ollama
    ports:
      - "11434:11434"
    volumes:
      - ./data/ollama:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    command: >
      sh -c "ollama serve &
             sleep 5 &&
             ollama pull bge-m3 &&
             wait"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:11434/api/tags"]
      interval: 30s
      timeout: 10s
      retries: 5

  # 应用服务
  app:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: graphrag-app
    ports:
      - "8000:8000"
    environment:
      - LLM_API_KEY=${LLM_API_KEY}
      - LLM_API_BASE=https://api.deepseek.com
      - LLM_MODEL=deepseek-chat
      - EMBEDDING_API_BASE=http://ollama:11434
      - EMBEDDING_MODEL=bge-m3
      - NEO4J_URI=bolt://neo4j:7687
      - NEO4J_USERNAME=neo4j
      - NEO4J_PASSWORD=password
    depends_on:
      neo4j:
        condition: service_healthy
      ollama:
        condition: service_healthy
    volumes:
      - ./output:/app/output
      - ./data/lancedb:/app/data/lancedb
```

---

## 13.10 本章小结

本章以GraphRAG-KG项目为实际案例，完整呈现了一个生产级GraphRAG系统的构建过程：

1. **项目架构**采用分层模块化设计，包括核心配置、CLI接口、索引流水线、查询引擎、Neo4j同步和测试数据六大模块。

2. **配置系统**采用"环境变量 + YAML"双层架构，支持三种配置模板（default/fast/production）以适应不同环境需求。

3. **测试数据**模拟了制药供应链场景，包含59个实体、153条关系和12种实体类型，覆盖药物、公司、疾病、监管等核心概念。

4. **索引流水线**封装了GraphRAG的核心索引能力，支持标准、快速和增量三种索引方式，输出Parquet文件和LanceDB向量存储。

5. **Neo4j同步**实现了Parquet到图数据库的批量MERGE同步，建立了Entity、Community、Document、TextUnit节点和RELATES_TO关系。

6. **查询引擎**支持4种搜索策略的自动路由（local/global/drift/basic），并提供了Cypher直接查询能力。

7. **性能调优**涵盖了向量维度匹配、连接池优化和缓存策略等关键环节。

GraphRAG-KG项目的完整代码托管在GitHub上，读者可以基于此项目快速搭建自己的GraphRAG系统。在实际应用中，建议根据业务场景调整实体类型定义、社区检测参数和检索策略权重，以获得最优效果。
