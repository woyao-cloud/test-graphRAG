# 第13章 知识图谱构建

## 13.1 引言

知识图谱（Knowledge Graph）作为一种结构化的知识表示方式，在RAG系统中扮演着越来越重要的角色。与传统的向量检索不同，知识图谱能够提供精确的实体关系查询、多跳推理和结构化信息检索能力，弥补了纯向量检索在处理精确匹配和复杂关系查询方面的不足。

知识图谱在RAG系统中的价值体现在以下几个方面：

1. **精确检索**：知识图谱可以精确地回答关于实体属性和关系的问题
2. **多跳推理**：通过图遍历，可以回答需要多步推理的复杂问题
3. **关系理解**：清晰展示实体之间的语义关系
4. **结构化输出**：支持结构化的查询结果，便于下游处理
5. **知识融合**：可以整合来自多个来源的结构化知识

本章将详细介绍知识图谱构建的完整流程，从Schema设计到实体关系提取，再到质量检查和混合查询，并提供基于Neo4j的完整实现方案。

### 13.1.1 知识图谱构建流程

完整的知识图谱构建流程包括以下步骤：

```
原始数据 → 实体提取 → 关系提取 → 质量检查 → 图存储 → 混合查询
   │           │           │           │          │          │
   ▼           ▼           ▼           ▼          ▼          ▼
文档/文本    命名实体    实体关系    完整性      Neo4j/    向量+图谱
  数据库      识别(LLM   提取(LLM    一致性     其他图      混合检索
              /NLP)      /规则)     准确性     数据库
```

### 13.1.2 知识图谱与传统RAG的互补

| 特性 | 向量检索 | 知识图谱检索 |
|------|---------|-------------|
| 匹配方式 | 语义相似度 | 精确匹配/模式匹配 |
| 关系查询 | 隐式（通过共现） | 显式（关系边） |
| 多跳推理 | 困难 | 自然支持 |
| 可解释性 | 低 | 高 |
| 更新成本 | 低（追加文档） | 高（需要结构化） |
| 查询灵活性 | 高（自然语言） | 中（需要结构化查询） |

## 13.2 Schema设计

Schema是知识图谱的骨架，定义了实体类型、关系类型和属性。良好的Schema设计是知识图谱质量和可用性的基础。

### 13.2.1 Schema设计原则

设计知识图谱Schema时，应遵循以下原则：

1. **业务驱动**：Schema应反映业务领域的概念和关系
2. **适度粒度**：实体类型既不能太粗（失去区分度）也不能太细（导致过度碎片化）
3. **可扩展性**：Schema应支持未来的扩展，避免频繁重构
4. **平衡复杂度**：在表达力和简洁性之间取得平衡
5. **命名规范**：使用清晰、一致的命名规则

### 13.2.2 实体类型设计

实体类型（Entity Type）是知识图谱中的节点类型。以下是一个供应链领域的实体类型设计示例：

```python
from typing import List, Dict, Optional
from enum import Enum
from datetime import datetime

class EntityType(str, Enum):
    """实体类型枚举"""
    COMPANY = "Company"              # 公司
    PRODUCT = "Product"              # 产品
    INGREDIENT = "Ingredient"        # 原料
    SUPPLIER = "Supplier"            # 供应商
    MANUFACTURER = "Manufacturer"    # 制造商
    DISTRIBUTOR = "Distributor"      # 分销商
    REGULATION = "Regulation"        # 法规
    CERTIFICATION = "Certification"  # 认证
    PATIENT = "Patient"              # 患者
    DISEASE = "Disease"              # 疾病
    DRUG = "Drug"                    # 药物
    CLINICAL_TRIAL = "ClinicalTrial" # 临床试验
    COUNTRY = "Country"              # 国家
    REGION = "Region"                # 地区
    FACILITY = "Facility"            # 设施

class EntitySchema:
    """实体类型Schema定义"""
    
    def __init__(self, name: str, description: str, 
                 properties: List[Dict], required_properties: List[str],
                 constraints: List[str] = None):
        self.name = name
        self.description = description
        self.properties = properties
        self.required_properties = required_properties
        self.constraints = constraints or []
    
    def to_dict(self) -> Dict:
        return {
            'type': self.name,
            'description': self.description,
            'properties': self.properties,
            'required': self.required_properties,
            'constraints': self.constraints
        }

# 定义实体Schema
ENTITY_SCHEMAS = {
    "Drug": EntitySchema(
        name="Drug",
        description="药物或药品",
        properties=[
            {"name": "name", "type": "string", "description": "药物名称"},
            {"name": "generic_name", "type": "string", "description": "通用名"},
            {"name": "brand_name", "type": "string", "description": "商品名"},
            {"name": "dosage_form", "type": "string", "description": "剂型"},
            {"name": "strength", "type": "string", "description": "规格"},
            {"name": "atc_code", "type": "string", "description": "ATC编码"},
            {"name": "approval_date", "type": "date", "description": "批准日期"},
            {"name": "status", "type": "string", "description": "状态（在售/停产等）"},
            {"name": "description", "type": "text", "description": "描述"}
        ],
        required_properties=["name"],
        constraints=["UNIQUE(name)"]
    ),
    "Company": EntitySchema(
        name="Company",
        description="制药公司或生物技术公司",
        properties=[
            {"name": "name", "type": "string", "description": "公司名称"},
            {"name": "type", "type": "string", "description": "公司类型"},
            {"name": "country", "type": "string", "description": "所在国家"},
            {"name": "founded_year", "type": "integer", "description": "成立年份"},
            {"name": "employees", "type": "integer", "description": "员工数"},
            {"name": "revenue", "type": "float", "description": "年收入"},
            {"name": "stock_symbol", "type": "string", "description": "股票代码"},
            {"name": "description", "type": "text", "description": "公司描述"}
        ],
        required_properties=["name"]
    ),
    "ClinicalTrial": EntitySchema(
        name="ClinicalTrial",
        description="临床试验",
        properties=[
            {"name": "trial_id", "type": "string", "description": "试验编号"},
            {"name": "title", "type": "string", "description": "试验标题"},
            {"name": "phase", "type": "string", "description": "试验阶段"},
            {"name": "status", "type": "string", "description": "试验状态"},
            {"name": "start_date", "type": "date", "description": "开始日期"},
            {"name": "end_date", "type": "date", "description": "结束日期"},
            {"name": "enrollment", "type": "integer", "description": "入组人数"},
            {"name": "conditions", "type": "list", "description": "研究疾病"},
            {"name": "interventions", "type": "list", "description": "干预措施"}
        ],
        required_properties=["trial_id", "title"]
    ),
    "Disease": EntitySchema(
        name="Disease",
        description="疾病或适应症",
        properties=[
            {"name": "name", "type": "string", "description": "疾病名称"},
            {"name": "icd_code", "type": "string", "description": "ICD编码"},
            {"name": "category", "type": "string", "description": "疾病类别"},
            {"name": "symptoms", "type": "list", "description": "症状列表"},
            {"name": "description", "type": "text", "description": "疾病描述"}
        ],
        required_properties=["name"]
    ),
    "Ingredient": EntitySchema(
        name="Ingredient",
        description="药物成分或原料药",
        properties=[
            {"name": "name", "type": "string", "description": "成分名称"},
            {"name": "cas_number", "type": "string", "description": "CAS编号"},
            {"name": "molecular_formula", "type": "string", "description": "分子式"},
            {"name": "molecular_weight", "type": "float", "description": "分子量"},
            {"name": "mechanism_of_action", "type": "text", "description": "作用机制"},
            {"name": "description", "type": "text", "description": "成分描述"}
        ],
        required_properties=["name"]
    )
}
```

### 13.2.3 关系类型设计

关系类型（Relation Type）定义了实体之间的连接方式：

```python
class RelationType(str, Enum):
    """关系类型枚举"""
    # 公司与产品
    MANUFACTURES = "MANUFACTURES"          # 公司-生产->产品
    DISTRIBUTES = "DISTRIBUTES"            # 公司-分销->产品
    SUPPLIES = "SUPPLIES"                  # 供应商-供应->原料
    
    # 产品与成分
    CONTAINS = "CONTAINS"                  # 产品-包含->成分
    HAS_ACTIVE_INGREDIENT = "HAS_ACTIVE_INGREDIENT"  # 产品-有效成分
    
    # 临床试验关系
    INVESTIGATES = "INVESTIGATES"          # 试验-研究->疾病
    USES_DRUG = "USES_DRUG"                # 试验-使用->药物
    SPONSORED_BY = "SPONSORED_BY"          # 试验-由->公司赞助
    CONDUCTED_AT = "CONDUCTED_AT"          # 试验-在->地点
    
    # 法规关系
    REGULATED_BY = "REGULATED_BY"          # 产品-受->法规监管
    HAS_CERTIFICATION = "HAS_CERTIFICATION" # 公司-有->认证
    APPROVED_IN = "APPROVED_IN"            # 产品-在->国家获批
    
    # 层次关系
    SUBSIDIARY_OF = "SUBSIDIARY_OF"        # 子公司-属于->母公司
    CATEGORIZED_AS = "CATEGORIZED_AS"      # 实体-分类为->类别
    RELATED_TO = "RELATED_TO"              # 通用关联关系

class RelationSchema:
    """关系Schema定义"""
    
    def __init__(self, name: str, source_types: List[str],
                 target_types: List[str], description: str,
                 properties: List[Dict] = None,
                 is_directed: bool = True):
        self.name = name
        self.source_types = source_types
        self.target_types = target_types
        self.description = description
        self.properties = properties or []
        self.is_directed = is_directed
    
    def to_dict(self) -> Dict:
        return {
            'type': self.name,
            'source_types': self.source_types,
            'target_types': self.target_types,
            'description': self.description,
            'properties': self.properties,
            'is_directed': self.is_directed
        }

# 定义关系Schema
RELATION_SCHEMAS = {
    "MANUFACTURES": RelationSchema(
        name="MANUFACTURES",
        source_types=["Company"],
        target_types=["Drug", "Product"],
        description="公司生产产品",
        properties=[
            {"name": "since_year", "type": "integer", "description": "开始生产年份"},
            {"name": "facility", "type": "string", "description": "生产设施"}
        ]
    ),
    "CONTAINS": RelationSchema(
        name="CONTAINS",
        source_types=["Drug", "Product"],
        target_types=["Ingredient"],
        description="产品包含成分",
        properties=[
            {"name": "concentration", "type": "string", "description": "浓度"},
            {"name": "is_active", "type": "boolean", "description": "是否为有效成分"}
        ]
    ),
    "INVESTIGATES": RelationSchema(
        name="INVESTIGATES",
        source_types=["ClinicalTrial"],
        target_types=["Disease"],
        description="临床试验研究疾病",
        properties=[
            {"name": "phase", "type": "string", "description": "试验阶段"},
            {"name": "primary_endpoint", "type": "string", "description": "主要终点"}
        ]
    ),
    "SPONSORED_BY": RelationSchema(
        name="SPONSORED_BY",
        source_types=["ClinicalTrial"],
        target_types=["Company"],
        description="试验由公司赞助",
        properties=[
            {"name": "role", "type": "string", "description": "赞助角色"}
        ]
    )
}
```

### 13.2.4 Schema验证

```python
class SchemaValidator:
    """Schema验证器"""
    
    def __init__(self, entity_schemas: Dict[str, EntitySchema],
                 relation_schemas: Dict[str, RelationSchema]):
        self.entity_schemas = entity_schemas
        self.relation_schemas = relation_schemas
    
    def validate_entity(self, entity_type: str, 
                        properties: Dict) -> List[str]:
        """验证实体数据"""
        errors = []
        
        if entity_type not in self.entity_schemas:
            errors.append(f"未知实体类型: {entity_type}")
            return errors
        
        schema = self.entity_schemas[entity_type]
        
        # 检查必填属性
        for prop_name in schema.required_properties:
            if prop_name not in properties:
                errors.append(f"缺少必填属性: {prop_name}")
        
        # 检查属性类型
        for prop_def in schema.properties:
            prop_name = prop_def['name']
            if prop_name in properties:
                prop_type = prop_def['type']
                value = properties[prop_name]
                
                if not self._validate_type(value, prop_type):
                    errors.append(
                        f"属性 {prop_name} 类型错误: "
                        f"期望 {prop_type}, 得到 {type(value).__name__}"
                    )
        
        return errors
    
    def validate_relation(self, relation_type: str,
                          source_type: str, target_type: str) -> List[str]:
        """验证关系"""
        errors = []
        
        if relation_type not in self.relation_schemas:
            errors.append(f"未知关系类型: {relation_type}")
            return errors
        
        schema = self.relation_schemas[relation_type]
        
        # 检查源实体类型
        if source_type not in schema.source_types:
            errors.append(
                f"源实体类型 {source_type} 不允许用于关系 {relation_type}"
            )
        
        # 检查目标实体类型
        if target_type not in schema.target_types:
            errors.append(
                f"目标实体类型 {target_type} 不允许用于关系 {relation_type}"
            )
        
        return errors
    
    def _validate_type(self, value, expected_type: str) -> bool:
        """验证属性值类型"""
        type_map = {
            'string': str,
            'integer': int,
            'float': float,
            'boolean': bool,
            'date': str,  # 日期使用ISO格式字符串
            'text': str,
            'list': list
        }
        
        expected = type_map.get(expected_type)
        if expected is None:
            return True  # 未知类型跳过检查
        
        return isinstance(value, expected)
```

## 13.3 实体提取

实体提取（Entity Extraction）是从非结构化文本中识别和提取命名实体的过程。在知识图谱构建中，实体提取是基础且关键的一步。

### 13.3.1 基于LLM的实体提取

使用大语言模型进行实体提取可以获得更高的准确率和灵活性：

```python
import json
from typing import List, Dict, Optional
from pydantic import BaseModel, Field

class ExtractedEntity(BaseModel):
    """提取的实体"""
    name: str = Field(description="实体名称")
    type: str = Field(description="实体类型")
    properties: Dict = Field(default_factory=dict, description="实体属性")
    source_text: str = Field(default="", description="来源文本")
    confidence: float = Field(default=1.0, description="提取置信度")

class LLMEntityExtractor:
    """基于LLM的实体提取器"""
    
    def __init__(self, llm, entity_types: Dict[str, EntitySchema],
                 batch_size: int = 5):
        self.llm = llm
        self.entity_types = entity_types
        self.batch_size = batch_size
    
    def extract_from_text(self, text: str) -> List[ExtractedEntity]:
        """从文本中提取实体"""
        prompt = self._build_extraction_prompt(text)
        
        try:
            response = self.llm(prompt)
            entities = self._parse_response(response)
            
            # 验证和过滤实体
            validated_entities = []
            for entity in entities:
                if self._validate_entity(entity):
                    validated_entities.append(entity)
            
            return validated_entities
            
        except Exception as e:
            print(f"实体提取失败: {e}")
            return []
    
    def extract_batch(self, texts: List[str]) -> List[List[ExtractedEntity]]:
        """批量提取实体"""
        results = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_results = []
            
            for text in batch:
                entities = self.extract_from_text(text)
                batch_results.append(entities)
            
            results.extend(batch_results)
        
        return results
    
    def _build_extraction_prompt(self, text: str) -> str:
        """构建提取提示词"""
        entity_descriptions = "\n".join([
            f"- {etype}: {schema.description}"
            for etype, schema in self.entity_types.items()
        ])
        
        prompt = f"""从以下文本中提取所有命名实体。

可提取的实体类型：
{entity_descriptions}

文本：
{text}

要求：
1. 只提取明确定义的实体类型
2. 每个实体必须包含name和type
3. 尽可能提取属性信息
4. 对不确定的实体标注较低的confidence值
5. 不要编造文本中不存在的信息

请以JSON格式输出实体列表：
[
    {{
        "name": "实体名称",
        "type": "实体类型",
        "properties": {{}},
        "source_text": "原文片段",
        "confidence": 0.95
    }}
]"""
        
        return prompt
    
    def _parse_response(self, response: str) -> List[ExtractedEntity]:
        """解析LLM响应"""
        try:
            # 尝试解析JSON
            data = json.loads(response)
            if isinstance(data, list):
                return [ExtractedEntity(**item) for item in data]
        except json.JSONDecodeError:
            # 尝试从文本中提取JSON
            import re
            json_match = re.search(r'\[.*?\]', response, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                    return [ExtractedEntity(**item) for item in data]
                except:
                    pass
        
        return []
    
    def _validate_entity(self, entity: ExtractedEntity) -> bool:
        """验证提取的实体"""
        if not entity.name or not entity.type:
            return False
        
        if entity.type not in self.entity_types:
            return False
        
        if entity.confidence < 0.3:
            return False
        
        return True

class EntityExtractionPipeline:
    """实体提取流水线"""
    
    def __init__(self, llm_extractor: LLMEntityExtractor,
                 nlp_extractor=None):
        self.llm_extractor = llm_extractor
        self.nlp_extractor = nlp_extractor
    
    def extract(self, text: str, 
                use_llm: bool = True,
                use_nlp: bool = True) -> List[ExtractedEntity]:
        """多策略实体提取"""
        all_entities = []
        
        # LLM提取
        if use_llm:
            llm_entities = self.llm_extractor.extract_from_text(text)
            all_entities.extend(llm_entities)
        
        # NLP提取（如果配置了）
        if use_nlp and self.nlp_extractor:
            nlp_entities = self.nlp_extractor.extract(text)
            all_entities.extend(nlp_entities)
        
        # 实体融合（去重和合并）
        merged = self._merge_entities(all_entities)
        
        return merged
    
    def _merge_entities(self, entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """合并来自不同来源的实体"""
        from collections import defaultdict
        
        # 按名称和类型分组
        groups = defaultdict(list)
        for entity in entities:
            key = (entity.name.lower(), entity.type)
            groups[key].append(entity)
        
        merged = []
        for key, group in groups.items():
            # 选择置信度最高的实体
            best = max(group, key=lambda e: e.confidence)
            
            # 合并属性
            combined_props = {}
            for entity in group:
                combined_props.update(entity.properties)
            
            best.properties = combined_props
            merged.append(best)
        
        return merged
```

### 13.3.2 基于NLP的实体提取

使用spaCy或HanLP等NLP工具进行实体提取，适合大规模文本的批量处理：

```python
import spacy
from collections import Counter
from typing import List, Tuple

class NLPEntityExtractor:
    """基于NLP的实体提取器"""
    
    def __init__(self, model_name: str = "zh_core_web_trf"):
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            print(f"模型 {model_name} 未找到，使用备用模型")
            self.nlp = spacy.load("zh_core_web_sm")
        
        # 自定义实体类型映射
        self.entity_type_mapping = {
            'PERSON': 'Person',
            'ORG': 'Company',
            'GPE': 'Country',
            'LOC': 'Region',
            'PRODUCT': 'Product',
            'DRUG': 'Drug',
            'DISEASE': 'Disease',
            'DATE': None,  # 日期不作为实体
            'MONEY': None,
            'PERCENT': None
        }
        
        # 领域特定词典
        self.domain_terms = self._load_domain_terms()
    
    def extract(self, text: str) -> List[ExtractedEntity]:
        """使用spaCy提取实体"""
        doc = self.nlp(text)
        entities = []
        
        seen = set()
        for ent in doc.ents:
            # 去重
            key = (ent.text.lower(), ent.label_)
            if key in seen:
                continue
            seen.add(key)
            
            # 映射实体类型
            mapped_type = self.entity_type_mapping.get(ent.label_)
            if mapped_type is None:
                continue
            
            entity = ExtractedEntity(
                name=ent.text,
                type=mapped_type,
                properties={'label': ent.label_},
                source_text=text[max(0, ent.start_char-50):ent.end_char+50],
                confidence=0.7
            )
            entities.append(entity)
        
        return entities
    
    def extract_phrases(self, text: str, min_freq: int = 2) -> List[str]:
        """提取高频短语（辅助实体识别）"""
        doc = self.nlp(text)
        phrases = []
        
        # 提取名词短语
        for chunk in doc.noun_chunks:
            if len(chunk.text) >= 2:
                phrases.append(chunk.text)
        
        # 统计频率
        phrase_freq = Counter(phrases)
        
        # 返回高频短语
        return [
            phrase for phrase, freq in phrase_freq.items()
            if freq >= min_freq
        ]
    
    def _load_domain_terms(self) -> Dict[str, str]:
        """加载领域词典"""
        return {
            '药品': 'Drug',
            '制药': 'Company',
            '临床': 'ClinicalTrial',
            '疾病': 'Disease',
            '成分': 'Ingredient'
        }
    
    def extract_domain_terms(self, text: str) -> List[ExtractedEntity]:
        """基于领域词典提取"""
        entities = []
        
        for term, entity_type in self.domain_terms.items():
            if term in text:
                entity = ExtractedEntity(
                    name=term,
                    type=entity_type,
                    confidence=0.6
                )
                entities.append(entity)
        
        return entities

# HanLP提取器示例
class HanLPExtractor:
    """HanLP实体提取器（中文NLP）"""
    
    def __init__(self):
        try:
            import hanlp
            self.ner = hanlp.load(hanlp.pretrained.ner.MSRA_NER_ELECTRA_SMALL_ZH)
            self.segmenter = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
        except ImportError:
            print("HanLP未安装，请执行: pip install hanlp")
            self.ner = None
            self.segmenter = None
    
    def extract(self, text: str) -> List[ExtractedEntity]:
        """使用HanLP提取实体"""
        if self.ner is None:
            return []
        
        entities = []
        
        try:
            # 分词和命名实体识别
            result = self.ner([list(text)])
            
            if result and len(result) > 0:
                for entity_info in result[0]:
                    if len(entity_info) >= 3:
                        entity_text, entity_type, start_end = entity_info[0], entity_info[1], entity_info[2]
                        
                        # 映射实体类型
                        mapped_type = self._map_entity_type(entity_type)
                        if mapped_type:
                            entity = ExtractedEntity(
                                name=entity_text,
                                type=mapped_type,
                                source_text=text[start_end[0]:start_end[1]],
                                confidence=0.8
                            )
                            entities.append(entity)
        
        except Exception as e:
            print(f"HanLP提取错误: {e}")
        
        return entities
    
    def _map_entity_type(self, hanlp_type: str) -> Optional[str]:
        """映射HanLP实体类型到系统类型"""
        mapping = {
            'NS': 'Country',     # 地名
            'NR': 'Person',      # 人名
            'NT': 'Company',     # 机构名
            'NI': 'Product',     # 产品名
        }
        return mapping.get(hanlp_type)
```

### 13.3.3 实体消歧

实体消歧解决同名实体指代不同现实世界对象的问题：

```python
class EntityDisambiguator:
    """实体消歧器"""
    
    def __init__(self, llm, knowledge_base: Dict = None):
        self.llm = llm
        self.knowledge_base = knowledge_base or {}
    
    def disambiguate(self, entity_name: str, 
                     context: str,
                     candidates: List[Dict]) -> Dict:
        """实体消歧"""
        if len(candidates) == 1:
            return candidates[0]
        
        if not candidates:
            return None
        
        prompt = f"""确定以下实体在给定上下文中的正确指代。

实体名称: {entity_name}

上下文:
{context[:500]}

候选实体:
{json.dumps(candidates, ensure_ascii=False, indent=2)}

请选择最匹配的候选实体，输出JSON：
{{
    "selected_index": 0,
    "confidence": 0.95,
    "reason": "选择原因"
}}"""
        
        try:
            response = self.llm(prompt)
            result = json.loads(response)
            idx = result.get('selected_index', 0)
            return candidates[idx]
        except:
            return candidates[0]  # 默认返回第一个
    
    def add_to_knowledge_base(self, entity: ExtractedEntity, 
                               context: str):
        """将实体加入知识库"""
        key = entity.name.lower()
        
        if key not in self.knowledge_base:
            self.knowledge_base[key] = []
        
        self.knowledge_base[key].append({
            'entity': entity,
            'context': context[:200],
            'timestamp': datetime.now().isoformat()
        })
```

## 13.4 关系提取

关系提取（Relation Extraction）是从文本中识别实体之间语义关系的过程。

### 13.4.1 基于LLM的关系提取

```python
class ExtractedRelation(BaseModel):
    """提取的关系"""
    source_entity: str = Field(description="源实体名称")
    target_entity: str = Field(description="目标实体名称")
    relation_type: str = Field(description="关系类型")
    properties: Dict = Field(default_factory=dict, description="关系属性")
    confidence: float = Field(default=1.0, description="置信度")
    source_text: str = Field(default="", description="来源文本")

class LLMRelationExtractor:
    """基于LLM的关系提取器"""
    
    def __init__(self, llm, relation_schemas: Dict[str, RelationSchema]):
        self.llm = llm
        self.relation_schemas = relation_schemas
    
    def extract(self, text: str, 
                entities: List[ExtractedEntity]) -> List[ExtractedRelation]:
        """从文本中提取实体间关系"""
        if not entities:
            return []
        
        # 构建实体列表
        entity_list = [
            {'name': e.name, 'type': e.type} 
            for e in entities
        ]
        
        prompt = f"""从以下文本中提取实体之间的关系。

文本：
{text}

文本中的实体：
{json.dumps(entity_list, ensure_ascii=False, indent=2)}

可提取的关系类型：
{self._format_relation_types()}

要求：
1. 只提取实体列表中存在的实体之间的关系
2. 关系类型必须来自预定义的关系类型
3. 每个关系必须标注置信度
4. 包含原文证据

请以JSON格式输出关系列表：
[
    {{
        "source_entity": "源实体名称",
        "target_entity": "目标实体名称",
        "relation_type": "关系类型",
        "properties": {{}},
        "confidence": 0.9,
        "source_text": "原文片段"
    }}
]"""
        
        try:
            response = self.llm(prompt)
            relations = self._parse_response(response)
            
            # 验证关系
            validated = []
            for rel in relations:
                if self._validate_relation(rel, entities):
                    validated.append(rel)
            
            return validated
            
        except Exception as e:
            print(f"关系提取失败: {e}")
            return []
    
    def _format_relation_types(self) -> str:
        """格式化关系类型说明"""
        parts = []
        for name, schema in self.relation_schemas.items():
            parts.append(
                f"- {name}: {schema.description} "
                f"(源: {schema.source_types}, 目标: {schema.target_types})"
            )
        return "\n".join(parts)
    
    def _parse_response(self, response: str) -> List[ExtractedRelation]:
        """解析LLM响应"""
        try:
            data = json.loads(response)
            if isinstance(data, list):
                return [ExtractedRelation(**item) for item in data]
        except:
            import re
            json_match = re.search(r'\[.*?\]', response, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                    return [ExtractedRelation(**item) for item in data]
                except:
                    pass
        return []
    
    def _validate_relation(self, relation: ExtractedRelation,
                           entities: List[ExtractedEntity]) -> bool:
        """验证提取的关系"""
        # 检查实体是否存在
        entity_names = {e.name for e in entities}
        if relation.source_entity not in entity_names:
            return False
        if relation.target_entity not in entity_names:
            return False
        
        # 检查关系类型是否有效
        if relation.relation_type not in self.relation_schemas:
            return False
        
        # 检查置信度
        if relation.confidence < 0.4:
            return False
        
        return True
```

### 13.4.2 基于共现PMI的关系提取

点互信息（Pointwise Mutual Information, PMI）是一种基于共现统计的关系提取方法：

```python
from collections import defaultdict
import math
from typing import Set

class PMIRelationExtractor:
    """基于PMI的关系提取器"""
    
    def __init__(self, window_size: int = 10, min_cooccurrence: int = 3,
                 pmi_threshold: float = 3.0):
        self.window_size = window_size
        self.min_cooccurrence = min_cooccurrence
        self.pmi_threshold = pmi_threshold
        
        # 统计信息
        self.entity_freq = Counter()           # 实体频率
        self.cooccurrence = defaultdict(Counter)  # 共现矩阵
    
    def train(self, texts: List[str], entities: List[List[ExtractedEntity]]):
        """训练PMI模型"""
        for text, entity_list in zip(texts, entities):
            # 获取实体位置
            entity_positions = self._find_entity_positions(text, entity_list)
            
            # 更新共现计数
            for i, (entity_i, pos_i) in enumerate(entity_positions):
                self.entity_freq[entity_i.name] += 1
                
                for j, (entity_j, pos_j) in enumerate(entity_positions):
                    if i >= j:
                        continue
                    
                    # 检查窗口距离
                    distance = abs(pos_i - pos_j)
                    if distance <= self.window_size:
                        pair = tuple(sorted([entity_i.name, entity_j.name]))
                        self.cooccurrence[entity_i.name][entity_j.name] += 1
                        self.cooccurrence[entity_j.name][entity_i.name] += 1
    
    def extract(self, text: str, 
                entities: List[ExtractedEntity]) -> List[ExtractedRelation]:
        """基于PMI提取关系"""
        if not self.cooccurrence:
            return []
        
        relations = []
        total_entities = sum(self.entity_freq.values())
        
        for i, entity_i in enumerate(entities):
            for j, entity_j in enumerate(entities):
                if i >= j:
                    continue
                
                # 计算PMI
                pmi = self._compute_pmi(entity_i.name, entity_j.name, total_entities)
                
                if pmi >= self.pmi_threshold:
                    relation = ExtractedRelation(
                        source_entity=entity_i.name,
                        target_entity=entity_j.name,
                        relation_type="RELATED_TO",
                        properties={'pmi_score': pmi},
                        confidence=min(pmi / 10.0, 0.95)
                    )
                    relations.append(relation)
        
        return relations
    
    def _compute_pmi(self, entity_a: str, entity_b: str, 
                     total: int) -> float:
        """计算PMI值"""
        freq_a = self.entity_freq.get(entity_a, 0)
        freq_b = self.entity_freq.get(entity_b, 0)
        cooc = self.cooccurrence.get(entity_a, {}).get(entity_b, 0)
        
        if freq_a == 0 or freq_b == 0 or cooc == 0:
            return 0.0
        
        # P(a,b) / (P(a) * P(b))
        p_ab = cooc / total
        p_a = freq_a / total
        p_b = freq_b / total
        
        pmi = math.log2(p_ab / (p_a * p_b))
        return max(0, pmi)
    
    def _find_entity_positions(self, text: str, 
                                entities: List[ExtractedEntity]) -> List[Tuple]:
        """查找实体在文本中的位置"""
        positions = []
        for entity in entities:
            pos = text.find(entity.name)
            if pos >= 0:
                positions.append((entity, pos))
        return positions
    
    def get_top_relations(self, entity_name: str, top_k: int = 10) -> List[Tuple]:
        """获取与指定实体最相关的实体"""
        if entity_name not in self.cooccurrence:
            return []
        
        coocs = self.cooccurrence[entity_name]
        sorted_entities = sorted(
            coocs.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_entities[:top_k]
```

### 13.4.3 基于规则的关系提取

对于特定领域，基于规则的方法往往能获得较高的准确率：

```python
import re
from typing import List, Pattern

class RuleBasedRelationExtractor:
    """基于规则的关系提取器"""
    
    def __init__(self):
        # 定义关系提取规则
        self.rules = self._define_rules()
    
    def _define_rules(self) -> List[Dict]:
        """定义提取规则"""
        return [
            {
                'relation_type': 'MANUFACTURES',
                'pattern': r'(?P<source>\w+公司|制药)\s*(?:生产|制造|研发)\s*(?P<target>\w+)',
                'source_types': ['Company'],
                'target_types': ['Drug', 'Product']
            },
            {
                'relation_type': 'CONTAINS',
                'pattern': r'(?P<source>\w+)\s*(?:含有|包含|成分包括)\s*(?P<target>\w+)',
                'source_types': ['Drug', 'Product'],
                'target_types': ['Ingredient']
            },
            {
                'relation_type': 'SPONSORED_BY',
                'pattern': r'(?P<source>\w+试验)\s*(?:由|被)\s*(?P<target>\w+)\s*(?:赞助|资助|支持)',
                'source_types': ['ClinicalTrial'],
                'target_types': ['Company']
            },
            {
                'relation_type': 'INVESTIGATES',
                'pattern': r'(?P<source>\w+试验)\s*(?:研究|针对|探索)\s*(?P<target>\w+)',
                'source_types': ['ClinicalTrial'],
                'target_types': ['Disease']
            },
            {
                'relation_type': 'SUPPLIES',
                'pattern': r'(?P<source>\w+)\s*(?:供应|提供|供给)\s*(?P<target>\w+)',
                'source_types': ['Supplier'],
                'target_types': ['Company', 'Manufacturer']
            },
            {
                'relation_type': 'APPROVED_IN',
                'pattern': r'(?P<source>\w+)\s*(?:获批|批准|通过)在\s*(?P<target>\w+)',
                'source_types': ['Drug'],
                'target_types': ['Country']
            }
        ]
    
    def extract(self, text: str, 
                entities: List[ExtractedEntity]) -> List[ExtractedRelation]:
        """基于规则提取关系"""
        relations = []
        entity_names = {e.name for e in entities}
        
        for rule in self.rules:
            pattern = re.compile(rule['pattern'])
            
            for match in pattern.finditer(text):
                source = match.group('source')
                target = match.group('target')
                
                # 验证实体存在
                if source not in entity_names or target not in entity_names:
                    continue
                
                relation = ExtractedRelation(
                    source_entity=source,
                    target_entity=target,
                    relation_type=rule['relation_type'],
                    source_text=match.group(),
                    confidence=0.9
                )
                relations.append(relation)
        
        return relations
    
    def add_custom_rule(self, relation_type: str, pattern: str,
                         source_types: List[str], target_types: List[str]):
        """添加自定义规则"""
        self.rules.append({
            'relation_type': relation_type,
            'pattern': pattern,
            'source_types': source_types,
            'target_types': target_types
        })
```

### 13.4.4 综合关系提取流水线

```python
class RelationExtractionPipeline:
    """综合关系提取流水线"""
    
    def __init__(self, llm_extractor: LLMRelationExtractor,
                 pmi_extractor: PMIRelationExtractor = None,
                 rule_extractor: RuleBasedRelationExtractor = None):
        self.llm_extractor = llm_extractor
        self.pmi_extractor = pmi_extractor
        self.rule_extractor = rule_extractor
    
    def extract(self, text: str, 
                entities: List[ExtractedEntity],
                strategies: List[str] = None) -> List[ExtractedRelation]:
        """多策略关系提取"""
        if strategies is None:
            strategies = ['llm', 'rule', 'pmi']
        
        all_relations = []
        
        if 'llm' in strategies and self.llm_extractor:
            llm_relations = self.llm_extractor.extract(text, entities)
            all_relations.extend(llm_relations)
        
        if 'rule' in strategies and self.rule_extractor:
            rule_relations = self.rule_extractor.extract(text, entities)
            all_relations.extend(rule_relations)
        
        if 'pmi' in strategies and self.pmi_extractor:
            pmi_relations = self.pmi_extractor.extract(text, entities)
            all_relations.extend(pmi_relations)
        
        # 融合和去重
        merged = self._merge_relations(all_relations)
        
        return merged
    
    def _merge_relations(self, relations: List[ExtractedRelation]) -> List[ExtractedRelation]:
        """合并和去重关系"""
        seen = set()
        merged = []
        
        for rel in sorted(relations, key=lambda r: r.confidence, reverse=True):
            key = (rel.source_entity, rel.relation_type, rel.target_entity)
            if key not in seen:
                seen.add(key)
                merged.append(rel)
        
        return merged
```

## 13.5 质量检查

知识图谱的质量直接影响下游应用的效果。质量检查需要在完整性、一致性和准确性三个维度进行。

### 13.5.1 完整性检查

```python
class CompletenessChecker:
    """完整性检查器"""
    
    def __init__(self, entity_schemas: Dict[str, EntitySchema]):
        self.schemas = entity_schemas
    
    def check_entity_completeness(self, entity: Dict) -> Dict:
        """检查实体完整性"""
        entity_type = entity.get('type')
        schema = self.schemas.get(entity_type)
        
        if schema is None:
            return {
                'is_complete': False,
                'score': 0.0,
                'missing': ['unknown_type'],
                'filled': 0,
                'total': 0
            }
        
        filled = 0
        total = len(schema.properties)
        missing = []
        
        for prop in schema.properties:
            prop_name = prop['name']
            if prop_name in entity.get('properties', {}):
                filled += 1
            elif prop_name in schema.required_properties:
                missing.append(prop_name)
        
        score = filled / total if total > 0 else 1.0
        
        return {
            'is_complete': len(missing) == 0,
            'score': score,
            'missing': missing,
            'filled': filled,
            'total': total
        }
    
    def check_graph_completeness(self, entities: List[Dict],
                                  relations: List[Dict]) -> Dict:
        """检查整个知识图谱的完整性"""
        entity_types = Counter(e.get('type') for e in entities)
        relation_types = Counter(r.get('type') for r in relations)
        
        # 检查孤立实体（没有关系的实体）
        connected_entities = set()
        for rel in relations:
            connected_entities.add(rel.get('source_entity'))
            connected_entities.add(rel.get('target_entity'))
        
        all_entity_names = {e.get('name') for e in entities}
        isolated = all_entity_names - connected_entities
        
        return {
            'total_entities': len(entities),
            'total_relations': len(relations),
            'entity_type_distribution': dict(entity_types),
            'relation_type_distribution': dict(relation_types),
            'isolated_entity_count': len(isolated),
            'isolated_entities': list(isolated)[:10],
            'avg_relations_per_entity': (
                len(relations) / len(entities) if entities else 0
            ),
            'completeness_score': self._compute_overall_score(
                entities, relations
            )
        }
    
    def _compute_overall_score(self, entities: List[Dict],
                                relations: List[Dict]) -> float:
        """计算总体完整性评分"""
        if not entities:
            return 0.0
        
        # 属性填充率
        prop_scores = []
        for entity in entities:
            result = self.check_entity_completeness(entity)
            prop_scores.append(result['score'])
        
        avg_prop_score = sum(prop_scores) / len(prop_scores)
        
        # 关系覆盖率
        connected = len(set(
            list(set(r.get('source_entity') for r in relations)) +
            list(set(r.get('target_entity') for r in relations))
        ))
        relation_coverage = connected / len(entities) if entities else 0
        
        # 综合评分
        return 0.6 * avg_prop_score + 0.4 * relation_coverage
```

### 13.5.2 一致性检查

```python
class ConsistencyChecker:
    """一致性检查器"""
    
    def __init__(self, entity_schemas, relation_schemas):
        self.entity_schemas = entity_schemas
        self.relation_schemas = relation_schemas
    
    def check_consistency(self, entities: List[Dict],
                          relations: List[Dict]) -> Dict:
        """全面一致性检查"""
        issues = []
        
        # 1. Schema一致性
        schema_issues = self._check_schema_consistency(entities, relations)
        issues.extend(schema_issues)
        
        # 2. 引用一致性
        ref_issues = self._check_reference_consistency(entities, relations)
        issues.extend(ref_issues)
        
        # 3. 关系对称性
        sym_issues = self._check_relation_symmetry(relations)
        issues.extend(sym_issues)
        
        # 4. 属性值一致性
        value_issues = self._check_value_consistency(entities)
        issues.extend(value_issues)
        
        return {
            'is_consistent': len(issues) == 0,
            'total_issues': len(issues),
            'issues': issues,
            'severity': self._compute_severity(issues)
        }
    
    def _check_schema_consistency(self, entities: List[Dict],
                                   relations: List[Dict]) -> List[Dict]:
        """检查Schema一致性"""
        issues = []
        
        for entity in entities:
            entity_type = entity.get('type')
            schema = self.entity_schemas.get(entity_type)
            
            if schema is None:
                issues.append({
                    'type': 'unknown_entity_type',
                    'severity': 'error',
                    'entity': entity.get('name'),
                    'message': f"未知实体类型: {entity_type}"
                })
                continue
            
            # 检查属性值类型
            for prop_def in schema.properties:
                prop_name = prop_def['name']
                value = entity.get('properties', {}).get(prop_name)
                
                if value is not None:
                    expected_type = prop_def['type']
                    if not self._check_type(value, expected_type):
                        issues.append({
                            'type': 'type_mismatch',
                            'severity': 'warning',
                            'entity': entity.get('name'),
                            'property': prop_name,
                            'message': f"属性 {prop_name} 类型不匹配: "
                                     f"期望 {expected_type}"
                        })
        
        # 检查关系类型
        for relation in relations:
            rel_type = relation.get('type')
            if rel_type not in self.relation_schemas:
                issues.append({
                    'type': 'unknown_relation_type',
                    'severity': 'error',
                    'message': f"未知关系类型: {rel_type}"
                })
        
        return issues
    
    def _check_reference_consistency(self, entities: List[Dict],
                                      relations: List[Dict]) -> List[Dict]:
        """检查引用一致性"""
        entity_names = {e.get('name') for e in entities}
        issues = []
        
        for relation in relations:
            source = relation.get('source_entity')
            target = relation.get('target_entity')
            
            if source not in entity_names:
                issues.append({
                    'type': 'dangling_reference',
                    'severity': 'error',
                    'message': f"关系引用不存在的源实体: {source}"
                })
            
            if target not in entity_names:
                issues.append({
                    'type': 'dangling_reference',
                    'severity': 'error',
                    'message': f"关系引用不存在的目标实体: {target}"
                })
        
        return issues
    
    def _check_relation_symmetry(self, relations: List[Dict]) -> List[Dict]:
        """检查关系对称性"""
        relation_pairs = set()
        issues = []
        
        for rel in relations:
            pair = (rel.get('source_entity'), rel.get('target_entity'), rel.get('type'))
            relation_pairs.add(pair)
        
        # 检查是否存在对称关系
        for rel in relations:
            if not self.relation_schemas.get(rel.get('type', ''), {}).get('is_directed', True):
                symmetric_pair = (rel.get('target_entity'), rel.get('source_entity'), rel.get('type'))
                if symmetric_pair not in relation_pairs:
                    issues.append({
                        'type': 'missing_symmetric_relation',
                        'severity': 'info',
                        'message': f"无向关系 {rel.get('type')} 缺少对称边"
                    })
        
        return issues
    
    def _check_value_consistency(self, entities: List[Dict]) -> List[Dict]:
        """检查属性值一致性"""
        issues = []
        
        # 检查同名实体的属性值是否一致
        entity_groups = defaultdict(list)
        for entity in entities:
            entity_groups[entity.get('name', '').lower()].append(entity)
        
        for name, group in entity_groups.items():
            if len(group) < 2:
                continue
            
            # 比较相同属性的值
            first = group[0].get('properties', {})
            for other in group[1:]:
                other_props = other.get('properties', {})
                
                for key, value in first.items():
                    if key in other_props and other_props[key] != value:
                        issues.append({
                            'type': 'value_conflict',
                            'severity': 'warning',
                            'entity': group[0].get('name'),
                            'property': key,
                            'values': [value, other_props[key]],
                            'message': f"实体 {name} 的属性 {key} 存在冲突值"
                        })
        
        return issues
    
    def _check_type(self, value, expected_type: str) -> bool:
        """检查值类型"""
        type_map = {
            'string': str,
            'integer': int,
            'float': (float, int),
            'boolean': bool,
            'list': list
        }
        
        expected = type_map.get(expected_type)
        if expected is None:
            return True
        
        return isinstance(value, expected)
    
    def _compute_severity(self, issues: List[Dict]) -> str:
        """计算总体严重程度"""
        severities = [i.get('severity') for i in issues]
        
        if 'error' in severities:
            return 'error'
        if 'warning' in severities:
            return 'warning'
        if severities:
            return 'info'
        return 'pass'
```

### 13.5.3 准确性检查

```python
class AccuracyChecker:
    """准确性检查器"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def check_entity_accuracy(self, entity: Dict, 
                               source_text: str) -> Dict:
        """检查实体准确性"""
        prompt = f"""验证以下从文本中提取的实体信息是否准确。

实体: {entity.get('name')}
类型: {entity.get('type')}
属性: {json.dumps(entity.get('properties', {}), ensure_ascii=False)}

原文: {source_text[:500]}

检查内容：
1. 实体名称是否正确
2. 实体类型是否准确
3. 属性值是否与原文一致

输出JSON：
{{
    "is_accurate": true/false,
    "score": 0.0-1.0,
    "errors": ["错误描述"],
    "suggestions": ["修正建议"]
}}"""
        
        try:
            response = self.llm(prompt)
            return json.loads(response)
        except:
            return {'is_accurate': True, 'score': 1.0, 'errors': []}
    
    def check_relation_accuracy(self, relation: Dict,
                                 source_text: str) -> Dict:
        """检查关系准确性"""
        prompt = f"""验证以下关系提取是否准确。

关系: {relation.get('source_entity')} -[{relation.get('type')}]-> {relation.get('target_entity')}

原文: {source_text[:500]}

输出JSON：
{{
    "is_accurate": true/false,
    "score": 0.0-1.0,
    "errors": ["错误描述"]
}}"""
        
        try:
            response = self.llm(prompt)
            return json.loads(response)
        except:
            return {'is_accurate': True, 'score': 1.0, 'errors': []}
    
    def sample_check(self, entities: List[Dict], relations: List[Dict],
                      sample_ratio: float = 0.1) -> Dict:
        """抽样检查准确性"""
        import random
        
        n_entities = max(1, int(len(entities) * sample_ratio))
        n_relations = max(1, int(len(relations) * sample_ratio))
        
        sampled_entities = random.sample(entities, n_entities)
        sampled_relations = random.sample(relations, n_relations)
        
        entity_results = []
        for entity in sampled_entities:
            result = self.check_entity_accuracy(entity, entity.get('source_text', ''))
            entity_results.append(result)
        
        relation_results = []
        for relation in sampled_relations:
            result = self.check_relation_accuracy(relation, relation.get('source_text', ''))
            relation_results.append(result)
        
        entity_accuracy = sum(
            r.get('score', 0) for r in entity_results
        ) / len(entity_results) if entity_results else 0
        
        relation_accuracy = sum(
            r.get('score', 0) for r in relation_results
        ) / len(relation_results) if relation_results else 0
        
        return {
            'entity_accuracy': entity_accuracy,
            'relation_accuracy': relation_accuracy,
            'overall_accuracy': (entity_accuracy + relation_accuracy) / 2,
            'sampled_entities': n_entities,
            'sampled_relations': n_relations,
            'entity_errors': [
                e for r in entity_results for e in r.get('errors', [])
            ],
            'relation_errors': [
                e for r in relation_results for e in r.get('errors', [])
            ]
        }
```

## 13.6 混合查询

混合查询结合了知识图谱的结构化查询能力和向量检索的语义匹配能力，是GraphRAG的核心能力。

### 13.6.1 查询路由

```python
class QueryRouter:
    """查询路由器 - 决定使用哪种查询策略"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def route(self, query: str) -> Dict:
        """路由查询到合适的处理策略"""
        prompt = f"""分析以下查询，决定最适合的检索策略。

查询: {query}

可选策略：
1. graph_only: 纯图查询（查询包含明确的实体和关系）
2. vector_only: 纯向量查询（语义搜索，不需要精确匹配）
3. hybrid: 混合查询（既需要精确匹配又需要语义理解）
4. graph_first: 先查图，再补向量（涉及多跳推理）
5. vector_first: 先查向量，再补图（开放域问题）

输出JSON：
{{
    "strategy": "策略名称",
    "reason": "选择原因",
    "key_entities": ["关键实体"],
    "query_type": "事实性/分析性/比较性"
}}"""
        
        try:
            response = self.llm(prompt)
            return json.loads(response)
        except:
            return {'strategy': 'hybrid', 'key_entities': []}
```

### 13.6.2 混合查询引擎

```python
class HybridQueryEngine:
    """混合查询引擎"""
    
    def __init__(self, graph_client, vector_store, llm,
                 router: QueryRouter = None):
        self.graph_client = graph_client
        self.vector_store = vector_store
        self.llm = llm
        self.router = router or QueryRouter(llm)
    
    def query(self, natural_query: str) -> Dict:
        """执行混合查询"""
        # 1. 路由决策
        route = self.router.route(natural_query)
        strategy = route.get('strategy', 'hybrid')
        
        # 2. 根据策略执行查询
        if strategy == 'graph_only':
            return self._graph_only_query(natural_query)
        elif strategy == 'vector_only':
            return self._vector_only_query(natural_query)
        elif strategy == 'graph_first':
            return self._graph_first_query(natural_query)
        elif strategy == 'vector_first':
            return self._vector_first_query(natural_query)
        else:
            return self._hybrid_query(natural_query)
    
    def _graph_only_query(self, query: str) -> Dict:
        """纯图查询"""
        # 将自然语言转换为Cypher
        cypher = self._nl_to_cypher(query)
        
        # 执行Cypher查询
        graph_results = self.graph_client.query(cypher)
        
        return {
            'strategy': 'graph_only',
            'graph_results': graph_results,
            'vector_results': [],
            'cypher': cypher,
            'answer': self._format_graph_answer(query, graph_results)
        }
    
    def _vector_only_query(self, query: str) -> Dict:
        """纯向量查询"""
        vector_results = self.vector_store.similarity_search(query, k=5)
        
        return {
            'strategy': 'vector_only',
            'graph_results': [],
            'vector_results': [
                {'content': d.page_content, 'score': d.metadata.get('score', 0)}
                for d in vector_results
            ],
            'answer': self._format_vector_answer(query, vector_results)
        }
    
    def _graph_first_query(self, query: str) -> Dict:
        """先图后向量"""
        # 先查询图谱
        cypher = self._nl_to_cypher(query)
        graph_results = self.graph_client.query(cypher)
        
        # 如果图谱结果不足，补充向量检索
        if not graph_results or len(graph_results) < 3:
            vector_results = self.vector_store.similarity_search(query, k=5)
        else:
            vector_results = []
        
        return {
            'strategy': 'graph_first',
            'graph_results': graph_results,
            'vector_results': [
                {'content': d.page_content, 'score': d.metadata.get('score', 0)}
                for d in vector_results
            ],
            'answer': self._format_hybrid_answer(
                query, graph_results, vector_results
            )
        }
    
    def _vector_first_query(self, query: str) -> Dict:
        """先向量后图"""
        # 先向量检索
        vector_results = self.vector_store.similarity_search(query, k=5)
        
        # 从向量结果中提取实体，补充图查询
        entities = self._extract_entities_from_docs(vector_results)
        graph_results = []
        
        if entities:
            for entity in entities[:3]:
                entity_query = f"MATCH (n) WHERE n.name = '{entity}' RETURN n"
                result = self.graph_client.query(entity_query)
                graph_results.extend(result)
        
        return {
            'strategy': 'vector_first',
            'graph_results': graph_results,
            'vector_results': [
                {'content': d.page_content, 'score': d.metadata.get('score', 0)}
                for d in vector_results
            ],
            'answer': self._format_hybrid_answer(
                query, graph_results, vector_results
            )
        }
    
    def _hybrid_query(self, query: str) -> Dict:
        """完全混合查询"""
        # 并行执行图和向量检索
        import asyncio
        
        async def parallel_query():
            cypher = self._nl_to_cypher(query)
            graph_task = asyncio.create_task(
                self._async_graph_query(cypher)
            )
            vector_task = asyncio.create_task(
                self._async_vector_query(query)
            )
            
            graph_results, vector_results = await asyncio.gather(
                graph_task, vector_task
            )
            
            return graph_results, vector_results
        
        # 由于这里不是异步上下文，使用同步方式
        cypher = self._nl_to_cypher(query)
        graph_results = self.graph_client.query(cypher)
        vector_results = self.vector_store.similarity_search(query, k=5)
        
        # 融合结果
        fused_answer = self._fuse_results(
            query, graph_results, vector_results
        )
        
        return {
            'strategy': 'hybrid',
            'graph_results': graph_results,
            'vector_results': [
                {'content': d.page_content, 'score': d.metadata.get('score', 0)}
                for d in vector_results
            ],
            'cypher': cypher,
            'answer': fused_answer
        }
    
    def _nl_to_cypher(self, query: str) -> str:
        """自然语言转Cypher"""
        prompt = f"""将以下自然语言查询转换为Cypher查询。

查询: {query}

图谱Schema:
- 节点类型: Drug, Company, ClinicalTrial, Disease, Ingredient
- 关系类型: MANUFACTURES, CONTAINS, INVESTIGATES, SPONSORED_BY
- 属性示例: name, status, phase, type

只输出Cypher查询语句，不要其他内容："""
        
        try:
            response = self.llm(prompt)
            # 清理输出
            cypher = response.strip()
            cypher = cypher.replace('```cypher', '').replace('```', '').strip()
            return cypher
        except:
            return f"MATCH (n) WHERE n.name CONTAINS '{query}' RETURN n LIMIT 10"
    
    def _extract_entities_from_docs(self, docs) -> List[str]:
        """从文档中提取实体名称"""
        entities = []
        for doc in docs:
            # 简单提取（实际应使用NER）
            text = doc.page_content[:200]
            # 假设实体以大写字母或特定格式开头
            words = text.split()
            for word in words:
                if word[0].isupper() and len(word) > 1:
                    entities.append(word)
        
        return list(set(entities))
    
    def _async_graph_query(self, cypher: str):
        """异步图查询"""
        return self.graph_client.query(cypher)
    
    def _async_vector_query(self, query: str):
        """异步向量查询"""
        return self.vector_store.similarity_search(query, k=5)
    
    def _format_graph_answer(self, query: str, results: List) -> str:
        """格式化图查询答案"""
        if not results:
            return "未在图谱中找到相关信息。"
        
        context = json.dumps(results, ensure_ascii=False, indent=2)[:1000]
        prompt = f"""基于图查询结果回答问题。

问题: {query}
图查询结果: {context}

请给出回答："""
        
        return self.llm(prompt)
    
    def _format_vector_answer(self, query: str, docs) -> str:
        """格式化向量查询答案"""
        if not docs:
            return "未在文档中找到相关信息。"
        
        context = "\n".join([d.page_content[:500] for d in docs])
        prompt = f"""基于检索到的文档回答问题。

问题: {query}
文档: {context}

请给出回答："""
        
        return self.llm(prompt)
    
    def _format_hybrid_answer(self, query: str, 
                               graph_results: List,
                               vector_docs) -> str:
        """格式化混合查询答案"""
        context_parts = []
        
        if graph_results:
            context_parts.append(
                f"图谱信息:\n{json.dumps(graph_results, ensure_ascii=False, indent=2)[:500]}"
            )
        
        if vector_docs:
            context_parts.append(
                f"文档信息:\n{chr(10).join([d.page_content[:300] for d in vector_docs])}"
            )
        
        context = "\n\n".join(context_parts)
        
        prompt = f"""综合图谱信息和文档信息回答问题。

问题: {query}

{context}

请给出准确、全面的回答："""
        
        return self.llm(prompt)
    
    def _fuse_results(self, query: str, graph_results: List,
                       vector_docs) -> str:
        """深度融合多源结果"""
        return self._format_hybrid_answer(query, graph_results, vector_docs)
```

## 13.7 Neo4j存储

Neo4j是最流行的图数据库之一，为知识图谱提供了高效存储和查询能力。

### 13.7.1 Neo4j连接管理

```python
from neo4j import GraphDatabase, Driver, Session, AsyncSession
from neo4j.exceptions import Neo4jError
from contextlib import contextmanager
from typing import Generator

class Neo4jManager:
    """Neo4j连接管理器"""
    
    def __init__(self, uri: str, user: str, password: str,
                 database: str = "neo4j",
                 max_connection_lifetime: int = 3600,
                 max_connection_pool_size: int = 50):
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.max_connection_lifetime = max_connection_lifetime
        self.max_connection_pool_size = max_connection_pool_size
        
        self.driver: Optional[Driver] = None
        self._connect()
    
    def _connect(self):
        """建立连接"""
        self.driver = GraphDatabase.driver(
            self.uri,
            auth=(self.user, self.password),
            max_connection_lifetime=self.max_connection_lifetime,
            max_connection_pool_size=self.max_connection_pool_size
        )
        
        # 验证连接
        try:
            with self.driver.session(database=self.database) as session:
                session.run("RETURN 1")
            print(f"成功连接到Neo4j: {self.uri}")
        except Exception as e:
            print(f"Neo4j连接失败: {e}")
            raise
    
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """获取会话"""
        if self.driver is None:
            self._connect()
        
        session = self.driver.session(database=self.database)
        try:
            yield session
        finally:
            session.close()
    
    def execute_query(self, cypher: str, 
                      parameters: Dict = None) -> List[Dict]:
        """执行Cypher查询"""
        with self.get_session() as session:
            try:
                result = session.run(cypher, parameters or {})
                return [record.data() for record in result]
            except Neo4jError as e:
                print(f"Cypher执行错误: {e}")
                raise
    
    def execute_transaction(self, queries: List[tuple]) -> bool:
        """执行事务"""
        with self.get_session() as session:
            tx = session.begin_transaction()
            try:
                for cypher, params in queries:
                    tx.run(cypher, params)
                tx.commit()
                return True
            except Exception as e:
                tx.rollback()
                print(f"事务失败: {e}")
                return False
    
    def create_constraints(self, constraints: List[str]):
        """创建约束"""
        with self.get_session() as session:
            for constraint in constraints:
                try:
                    session.run(constraint)
                    print(f"约束创建成功: {constraint}")
                except Neo4jError as e:
                    if "AlreadyExists" not in str(e):
                        print(f"约束创建失败: {e}")
    
    def create_indexes(self, indexes: List[str]):
        """创建索引"""
        with self.get_session() as session:
            for index in indexes:
                try:
                    session.run(index)
                    print(f"索引创建成功: {index}")
                except Neo4jError as e:
                    if "AlreadyExists" not in str(e):
                        print(f"索引创建失败: {e}")
    
    def close(self):
        """关闭连接"""
        if self.driver:
            self.driver.close()
            print("Neo4j连接已关闭")
```

### 13.7.2 批量导入

```python
from tqdm import tqdm
from typing import Iterator

class Neo4jBatchImporter:
    """Neo4j批量导入器"""
    
    def __init__(self, neo4j_manager: Neo4jManager,
                 batch_size: int = 1000):
        self.neo4j = neo4j_manager
        self.batch_size = batch_size
    
    def import_entities(self, entities: List[Dict]):
        """批量导入实体"""
        # 创建约束
        self._create_entity_constraints()
        
        # 分批导入
        for i in tqdm(range(0, len(entities), self.batch_size),
                      desc="导入实体"):
            batch = entities[i:i + self.batch_size]
            queries = []
            
            for entity in batch:
                cypher, params = self._build_entity_query(entity)
                queries.append((cypher, params))
            
            self.neo4j.execute_transaction(queries)
        
        print(f"成功导入 {len(entities)} 个实体")
    
    def import_relations(self, relations: List[Dict]):
        """批量导入关系"""
        # 分批导入
        for i in tqdm(range(0, len(relations), self.batch_size),
                      desc="导入关系"):
            batch = relations[i:i + self.batch_size]
            queries = []
            
            for relation in batch:
                cypher, params = self._build_relation_query(relation)
                queries.append((cypher, params))
            
            self.neo4j.execute_transaction(queries)
        
        print(f"成功导入 {len(relations)} 个关系")
    
    def import_full_graph(self, entities: List[Dict],
                          relations: List[Dict]):
        """完整导入图谱"""
        self.import_entities(entities)
        self.import_relations(relations)
        
        # 创建查询索引
        self._create_query_indexes()
        
        print("图谱导入完成")
    
    def _create_entity_constraints(self):
        """创建实体约束"""
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Drug) REQUIRE d.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Company) REQUIRE c.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (t:ClinicalTrial) REQUIRE t.trial_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Disease) REQUIRE d.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (i:Ingredient) REQUIRE i.name IS UNIQUE"
        ]
        self.neo4j.create_constraints(constraints)
    
    def _create_query_indexes(self):
        """创建查询索引"""
        indexes = [
            "CREATE INDEX IF NOT EXISTS FOR (d:Drug) ON (d.status)",
            "CREATE INDEX IF NOT EXISTS FOR (c:Company) ON (c.country)",
            "CREATE INDEX IF NOT EXISTS FOR (t:ClinicalTrial) ON (t.phase)",
            "CREATE INDEX IF NOT EXISTS FOR (t:ClinicalTrial) ON (t.status)"
        ]
        self.neo4j.create_indexes(indexes)
    
    def _build_entity_query(self, entity: Dict) -> tuple:
        """构建实体创建Cypher"""
        entity_type = entity.get('type', 'Entity')
        properties = entity.get('properties', {})
        name = entity.get('name')
        
        # 构建MERGE语句
        cypher = f"""
        MERGE (n:{entity_type} {{name: $name}})
        SET n += $properties
        """
        
        params = {
            'name': name,
            'properties': properties
        }
        
        return cypher, params
    
    def _build_relation_query(self, relation: Dict) -> tuple:
        """构建关系创建Cypher"""
        source = relation.get('source_entity')
        target = relation.get('target_entity')
        rel_type = relation.get('type', 'RELATED_TO')
        properties = relation.get('properties', {})
        
        cypher = f"""
        MATCH (source {{name: $source}})
        MATCH (target {{name: $target}})
        MERGE (source)-[r:{rel_type}]->(target)
        SET r += $properties
        """
        
        params = {
            'source': source,
            'target': target,
            'properties': properties
        }
        
        return cypher, params

class BulkCSVImporter:
    """CSV批量导入器（适用于大规模导入）"""
    
    def __init__(self, neo4j_manager: Neo4jManager):
        self.neo4j = neo4j_manager
    
    def import_from_csv(self, entities_file: str, relations_file: str):
        """从CSV文件批量导入"""
        # 使用LOAD CSV命令
        entity_cypher = f"""
        LOAD CSV WITH HEADERS FROM 'file:///{entities_file}' AS row
        CALL {{
            WITH row
            MERGE (n:{row.type} {{name: row.name}})
            SET n += row.properties
        }} IN TRANSACTIONS OF 1000 ROWS
        """
        
        relation_cypher = f"""
        LOAD CSV WITH HEADERS FROM 'file:///{relations_file}' AS row
        CALL {{
            WITH row
            MATCH (source {{name: row.source}})
            MATCH (target {{name: row.target}})
            MERGE (source)-[r:{row.type}]->(target)
            SET r += row.properties
        }} IN TRANSACTIONS OF 1000 ROWS
        """
        
        self.neo4j.execute_query(entity_cypher)
        self.neo4j.execute_query(relation_cypher)
        
        print(f"CSV导入完成")
    
    def export_to_csv(self, output_dir: str):
        """导出到CSV"""
        # 导出实体
        self.neo4j.execute_query(f"""
        MATCH (n)
        WITH n, labels(n)[0] AS label
        RETURN label AS type, n.name AS name, 
               properties(n) AS properties
        """)
        
        # 导出关系
        self.neo4j.execute_query(f"""
        MATCH (s)-[r]->(t)
        RETURN type(r) AS type, s.name AS source, 
               t.name AS target, properties(r) AS properties
        """)
```

## 13.8 知识图谱的增量更新

### 13.8.1 增量更新策略

```python
class IncrementalUpdater:
    """增量更新器"""
    
    def __init__(self, neo4j_manager: Neo4jManager):
        self.neo4j = neo4j_manager
    
    def upsert_entity(self, entity: Dict) -> bool:
        """插入或更新实体"""
        entity_type = entity.get('type')
        name = entity.get('name')
        properties = entity.get('properties', {})
        
        cypher = f"""
        MERGE (n:{entity_type} {{name: $name}})
        SET n += $properties
        SET n.updated_at = datetime()
        RETURN n
        """
        
        try:
            self.neo4j.execute_query(cypher, {
                'name': name,
                'properties': properties
            })
            return True
        except Exception as e:
            print(f"实体更新失败: {e}")
            return False
    
    def delete_entity(self, entity_type: str, name: str) -> bool:
        """删除实体及其所有关系"""
        cypher = f"""
        MATCH (n:{entity_type} {{name: $name}})
        DETACH DELETE n
        """
        
        try:
            self.neo4j.execute_query(cypher, {'name': name})
            return True
        except Exception as e:
            print(f"实体删除失败: {e}")
            return False
    
    def update_timestamp(self):
        """更新时间戳"""
        cypher = """
        MATCH (n)
        WHERE n.updated_at IS NULL
        SET n.updated_at = datetime()
        """
        self.neo4j.execute_query(cypher)
```

## 13.9 本章小结

本章详细介绍了知识图谱构建的完整流程，涵盖了从Schema设计到Neo4j存储的各个环节。

**Schema设计**是知识图谱的基石。本章介绍了实体类型、关系类型和属性的设计原则，并提供了Schema验证的实现，确保知识图谱的数据质量和一致性。

**实体提取**方面，本章介绍了基于LLM、基于NLP（spaCy/HanLP）和基于规则三种方法。LLM方法灵活且准确率高，适合复杂场景；NLP方法适合大规模批量处理；规则方法在特定领域具有高精度。实际应用中推荐采用多策略融合的方法。

**关系提取**是知识图谱构建中最具挑战性的环节。本章介绍了基于LLM、基于共现PMI和基于规则三种方法。PMI方法通过统计共现模式发现隐式关系，适合发现新关系；规则方法利用领域知识精确定位关系；LLM方法则提供最灵活的提取能力。

**质量检查**从完整性、一致性和准确性三个维度确保知识图谱的质量。完整性检查评估实体属性填充率和图结构完整性；一致性检查发现Schema违规、引用缺失和值冲突；准确性检查通过LLM验证实体和关系的正确性。

**混合查询**是知识图谱在RAG中发挥价值的关键。本章实现的混合查询引擎支持五种查询策略：纯图查询、纯向量查询、先图后向量、先向量后图和完全混合查询，根据查询特征自动选择最优策略。

**Neo4j存储**部分提供了完整的连接管理、批量导入和事务处理实现。批量导入支持分批次、事务性导入，确保大规模知识图谱构建的效率和可靠性。

在实际部署中，建议按照以下流程构建知识图谱：首先设计Schema，然后从小规模数据开始验证提取效果，逐步扩大到全量数据，最后通过质量检查评估并迭代优化。
