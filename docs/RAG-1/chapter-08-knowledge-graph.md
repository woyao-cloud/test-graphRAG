# 第8章 知识图谱构建与应用

## 8.1 知识图谱在 RAG 中的角色

### 8.1.1 实体-关系三元组

知识图谱（Knowledge Graph, KG）的核心是实体-关系三元组（Subject-Predicate-Object, SPO）：

```
(奥希替尼) --[用于治疗]--> (非小细胞肺癌)
(阿斯利康) --[生产]--> (奥希替尼)
(奥希替尼) --[靶点]--> (EGFR)
(非小细胞肺癌) --[属于]--> (肺癌)
```

三元组构成了有向图结构，其中：
- **实体（Entity）**：图中的节点，代表现实世界中的对象或概念
- **关系（Relation）**：图中的边，代表实体之间的语义关联
- **属性（Property）**：实体或关系的附加信息

### 8.1.2 KG 在 RAG 中的独特价值

| 维度 | 向量检索 | 知识图谱 |
|------|---------|---------|
| 语义匹配 | 强（语义相似） | 强（精确语义） |
| 精确推理 | 弱（无法多跳推理） | 强（路径遍历） |
| 关系发现 | 弱 | 强（N 跳邻居） |
| 结构化查询 | 有限 | 精确（Cypher/SPARQL） |
| 构建成本 | 低（只需 embedding） | 高（需要抽取+建模） |
| 可解释性 | 低（黑盒向量） | 高（可追踪路径） |
| 更新成本 | 低（新增文档自动生成） | 高（需要重新抽取） |

**向量检索 vs 知识图谱的互补性**：

- **向量检索**适合"模糊语义匹配"：用户问"肺癌怎么治"，检索到"肺腺癌化疗方案"
- **知识图谱**适合"精确关系查询"：用户问"阿斯利康的哪些靶向药用于治疗 EGFR 突变的非小细胞肺癌"
- **最佳实践**：向量检索做粗筛，知识图谱做精查和关系推理，两者结果融合后输入 LLM

### 8.1.3 知识图谱构建流程

```
原始文档
    │
    ├──→ 实体抽取
    │    ├── LLM 驱动
    │    ├── NLP 工具 (spaCy, HanLP)
    │    └── 规则匹配
    │
    ├──→ 实体链接 & 消歧
    │    ├── 同名实体合并
    │    └── 指代消解
    │
    ├──→ 关系抽取
    │    ├── 关系分类
    │    ├── 共现分析
    │    └── 规则模板
    │
    ├──→ 图存储
    │    ├── Neo4j
    │    ├── 索引构建
    │    └── 批量导入
    │
    └──→ 图检索
         ├── 实体查找
         ├── 路径遍历
         └── 社区查询
```

---

## 8.2 实体抽取

### 8.2.1 实体类型定义

在设计知识图谱之前，需要先定义实体类型体系。以下是企业知识图谱中常用的实体类型：

```python
from enum import Enum
from typing import List, Optional, Dict
from dataclasses import dataclass, field

class EntityType(Enum):
    """实体类型枚举"""
    PERSON = "Person"           # 人物
    ORGANIZATION = "Organization"  # 组织/公司
    LOCATION = "Location"       # 地理位置
    EVENT = "Event"             # 事件
    CONCEPT = "Concept"         # 抽象概念
    PRODUCT = "Product"         # 产品
    DRUG = "Drug"              # 药物（医疗领域）
    DISEASE = "Disease"        # 疾病（医疗领域）
    TECHNOLOGY = "Technology"   # 技术
    DOCUMENT = "Document"      # 文档


@dataclass
class EntitySchema:
    """实体模式定义"""
    type: EntityType
    description: str
    properties: List[str] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)
    
    def validate(self, entity: Dict) -> bool:
        """验证实体是否符合模式"""
        for prop in self.properties:
            if prop not in entity:
                return False
        return True


class EntitySchemaRegistry:
    """实体模式注册表"""
    
    SCHEMAS = {
        EntityType.PERSON: EntitySchema(
            type=EntityType.PERSON,
            description="人物",
            properties=["name", "title"],
            aliases=["人", "人物", "专家"]
        ),
        EntityType.ORGANIZATION: EntitySchema(
            type=EntityType.ORGANIZATION,
            description="组织/公司",
            properties=["name", "industry"],
            aliases=["公司", "组织", "机构", "企业"]
        ),
        EntityType.DRUG: EntitySchema(
            type=EntityType.DRUG,
            description="药物",
            properties=["name", "generic_name", "indication"],
            aliases=["药", "药物", "药品", "制剂"]
        ),
        EntityType.DISEASE: EntitySchema(
            type=EntityType.DISEASE,
            description="疾病",
            properties=["name", "icd_code"],
            aliases=["疾病", "病症", "综合征"]
        ),
        EntityType.TECHNOLOGY: EntitySchema(
            type=EntityType.TECHNOLOGY,
            description="技术",
            properties=["name", "field"],
            aliases=["技术", "方法", "算法"]
        ),
    }
    
    @classmethod
    def get_schema(cls, entity_type: EntityType) -> Optional[EntitySchema]:
        return cls.SCHEMAS.get(entity_type)
```

### 8.2.2 LLM 驱动的实体抽取

使用 LLM 进行实体抽取可以处理复杂语义，但需要精心设计 Prompt：

```python
from typing import List, Dict, Any
import json
import re

class LLMEntityExtractor:
    """基于 LLM 的实体抽取"""
    
    def __init__(self, llm_client, model: str = "gpt-4"):
        self.llm = llm_client
        self.model = model
    
    def extract_entities(self, text: str) -> List[Dict]:
        """
        从文本中抽取实体
        
        Args:
            text: 输入文本
            
        Returns:
            [{"name": "实体名", "type": "实体类型", "mentions": ["提及位置"]}, ...]
        """
        prompt = f"""你是一个知识图谱实体抽取专家。请从以下文本中抽取所有重要的实体。

文本：
{text}

请按以下 JSON 格式返回实体列表：
[
  {{
    "name": "实体名称（标准化后的名称）",
    "type": "实体类型（Person/Organization/Location/Event/Concept/Product/Drug/Disease/Technology）",
    "description": "实体的简要描述（从文本中提取）",
    "source_text": "实体在原文中的原始提及"
  }}
]

要求：
1. 只抽取明确的、有意义的实体
2. 实体名称使用标准名称（如"阿斯利康"而非"AstraZeneca"）
3. 只返回 JSON 数组，不要其他文字
"""
        
        response = self.llm.chat(prompt)
        
        # 提取 JSON
        entities = self._parse_json_response(response)
        
        # 去重（同名同类型的合并）
        entities = self._dedup_entities(entities)
        
        return entities
    
    def _parse_json_response(self, response: str) -> List[Dict]:
        """解析 LLM 返回的 JSON"""
        # 尝试直接解析
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # 尝试提取 JSON 代码块
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", response)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # 尝试提取 [] 中的内容
        array_match = re.search(r"\[[\s\S]*\]", response)
        if array_match:
            try:
                return json.loads(array_match.group())
            except json.JSONDecodeError:
                pass
        
        return []
    
    def _dedup_entities(self, entities: List[Dict]) -> List[Dict]:
        """去重实体"""
        seen = set()
        deduped = []
        
        for entity in entities:
            key = (entity.get("name", ""), entity.get("type", ""))
            if key not in seen:
                seen.add(key)
                deduped.append(entity)
        
        return deduped


class BatchEntityExtractor:
    """批量实体抽取（支持大文档分段处理）"""
    
    def __init__(self, extractor: LLMEntityExtractor,
                 max_chunk_size: int = 2000):
        self.extractor = extractor
        self.max_chunk_size = max_chunk_size
    
    def extract_from_document(self, title: str, content: str) -> List[Dict]:
        """
        从完整文档中抽取实体
        
        策略：将长文档分段抽取，然后合并去重
        """
        # 分段
        chunks = self._split_document(content)
        
        all_entities = []
        
        # 逐段抽取
        for chunk in chunks:
            chunk_text = f"标题：{title}\n\n内容：{chunk}"
            entities = self.extractor.extract_entities(chunk_text)
            all_entities.extend(entities)
        
        # 合并去重
        merged = self._merge_entities(all_entities)
        
        return merged
    
    def _split_document(self, content: str) -> List[str]:
        """将长文档分段"""
        if len(content) <= self.max_chunk_size:
            return [content]
        
        chunks = []
        # 按段落切分
        paragraphs = content.split("\n\n")
        current_chunk = ""
        
        for para in paragraphs:
            if len(current_chunk) + len(para) <= self.max_chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _merge_entities(self, entities: List[Dict]) -> List[Dict]:
        """合并跨段的同一实体"""
        # 按 (name, type) 分组
        groups = {}
        for entity in entities:
            key = (entity["name"], entity["type"])
            if key not in groups:
                groups[key] = entity
            else:
                # 合并描述
                existing = groups[key]
                if len(entity.get("description", "")) > len(existing.get("description", "")):
                    existing["description"] = entity["description"]
        
        return list(groups.values())
```

### 8.2.3 NLP 工具实体抽取（spaCy / HanLP）

对于需要离线部署、低延迟的场景，使用传统的 NLP 工具更为合适：

```python
class SpacyEntityExtractor:
    """基于 spaCy 的实体抽取"""
    
    def __init__(self, model_name: str = "zh_core_web_sm"):
        """
        Args:
            model_name: spaCy 模型名称
                - zh_core_web_sm: 中文小模型
                - zh_core_web_trf: 中文 Transformer 模型（更精确）
        """
        import spacy
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            print(f"[spaCy] 模型 {model_name} 未安装，下载中...")
            spacy.cli.download(model_name)
            self.nlp = spacy.load(model_name)
    
    def extract_entities(self, text: str) -> List[Dict]:
        """
        使用 spaCy NER 抽取实体
        
        spaCy 内置的实体类型：
        - PERSON: 人物
        - ORG: 组织
        - GPE: 地缘政治实体（国家/城市）
        - LOC: 位置
        - DATE: 日期
        - PRODUCT: 产品
        - EVENT: 事件
        - WORK_OF_ART: 艺术作品
        """
        doc = self.nlp(text)
        
        entities = []
        for ent in doc.ents:
            entities.append({
                "name": ent.text,
                "type": self._map_spacy_type(ent.label_),
                "start": ent.start_char,
                "end": ent.end_char,
                "source": "spacy"
            })
        
        return entities
    
    def _map_spacy_type(self, spacy_type: str) -> str:
        """映射 spaCy 类型到标准类型"""
        mapping = {
            "PERSON": "Person",
            "ORG": "Organization",
            "GPE": "Location",
            "LOC": "Location",
            "PRODUCT": "Product",
            "EVENT": "Event",
            "DATE": "Date",
            "WORK_OF_ART": "Concept",
            "LAW": "Concept",
            "FAC": "Location",
        }
        return mapping.get(spacy_type, "Concept")


class HanLPEntityExtractor:
    """基于 HanLP 的实体抽取"""
    
    def __init__(self):
        import hanlp
        # 加载多任务模型
        self.recognizer = hanlp.load(hanlp.pretrained.ner.MSRA_NER_ELECTRA_SMALL_ZH)
    
    def extract_entities(self, text: str) -> List[Dict]:
        """
        使用 HanLP 抽取中文实体
        
        HanLP MSRA NER 实体类型：
        - PERSON: 人名
        - ORG: 机构名
        - LOC: 地名
        """
        # HanLP 需要先分词
        import hanlp
        tokenizer = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
        tokens = tokenizer(text)
        
        # NER
        entities_raw = self.recognizer(tokens)
        
        entities = []
        for entity_type, entity_text, start, end in entities_raw:
            entities.append({
                "name": entity_text,
                "type": self._map_hanlp_type(entity_type),
                "position": (start, end),
                "source": "hanlp"
            })
        
        return entities
    
    def _map_hanlp_type(self, hanlp_type: str) -> str:
        mapping = {
            "PERSON": "Person",
            "ORG": "Organization",
            "LOC": "Location",
        }
        return mapping.get(hanlp_type, "Concept")
```

### 8.2.4 实体消歧与归一化

同名异义和异名同义是实体抽取中必须处理的问题：

```python
class EntityDisambiguator:
    """实体消歧器"""
    
    def __init__(self):
        # 同义词映射表
        self.synonym_map = {
            "阿斯利康": "AstraZeneca",
            "AZ": "AstraZeneca",
            "辉瑞": "Pfizer",
            "罗氏": "Roche",
            "非小细胞肺癌": "NSCLC",
            "NSCLC": "非小细胞肺癌",
        }
        
        # 实体上下文特征（用于消歧）
        self.entity_contexts = {
            "苹果": [
                {"context": "手机|iPhone|iOS|iPad", "type": "Organization"},
                {"context": "水果|吃|营养|维生素", "type": "Concept"},
            ]
        }
    
    def normalize_name(self, name: str) -> str:
        """归一化实体名称"""
        # 查同义词表
        if name in self.synonym_map:
            return self.synonym_map[name]
        
        # 反向查
        for standard, alias in self.synonym_map.items():
            if name == alias:
                return standard
        
        return name
    
    def disambiguate(self, entity_name: str,
                     context: str,
                     candidates: List[Dict]) -> Optional[Dict]:
        """
        实体消歧
        
        Args:
            entity_name: 实体名称
            context: 上下文文本
            candidates: 候选实体列表
            
        Returns:
            消歧后的实体
        """
        if len(candidates) == 1:
            return candidates[0]
        
        if len(candidates) == 0:
            return None
        
        # 使用上下文特征消歧
        contexts = self.entity_contexts.get(entity_name, [])
        
        for candidate in candidates:
            candidate_type = candidate.get("type", "")
            
            for ctx in contexts:
                if candidate_type == ctx["type"]:
                    # 检查上下文是否匹配
                    import re
                    if re.search(ctx["context"], context, re.IGNORECASE):
                        return candidate
        
        # 默认返回第一个
        return candidates[0]
```

---

## 8.3 关系建模与抽取

### 8.3.1 关系类型定义

```python
class RelationType(Enum):
    """关系类型枚举"""
    # 组织关系
    EMPLOYS = "employs"              # 雇佣
    BELONGS_TO = "belongs_to"        # 属于
    PARTNERS_WITH = "partners_with"  # 合作
    
    # 产品/药物关系
    PRODUCES = "produces"            # 生产
    DEVELOPS = "develops"            # 研发
    SUPPLIES = "supplies"            # 供应
    
    # 医疗关系
    TREATS = "treats"               # 治疗
    CAUSES = "causes"               # 导致
    DIAGNOSES = "diagnoses"         # 诊断
    TARGETS = "targets"             # 靶点
    SIDE_EFFECT = "side_effect"     # 副作用
    
    # 层级关系
    IS_A = "is_a"                    # 是一种
    PART_OF = "part_of"             # 组成部分
    HAS_SUBTYPE = "has_subtype"     # 有子类
    
    # 事件关系
    OCCURRED_AT = "occurred_at"     # 发生在
    INVOLVES = "involves"           # 涉及
    LEADS_TO = "leads_to"           # 导致
    
    # 通用关系
    RELATED_TO = "related_to"       # 相关
    REFERENCED_IN = "referenced_in" # 引用
    LOCATED_IN = "located_in"       # 位于


@dataclass
class RelationSchema:
    """关系模式"""
    type: RelationType
    name: str
    description: str
    domain_types: List[EntityType]  # 主语类型
    range_types: List[EntityType]   # 宾语类型
    is_directional: bool = True     # 是否有方向
    
    def validate(self, subject_type: EntityType,
                 object_type: EntityType) -> bool:
        """验证实体类型是否匹配关系模式"""
        return (subject_type in self.domain_types and 
                object_type in self.range_types)


class RelationSchemaRegistry:
    """关系模式注册表"""
    
    SCHEMAS = {
        RelationType.PRODUCES: RelationSchema(
            type=RelationType.PRODUCES,
            name="生产",
            description="组织/公司生产产品/药物",
            domain_types=[EntityType.ORGANIZATION],
            range_types=[EntityType.PRODUCT, EntityType.DRUG]
        ),
        RelationType.TREATS: RelationSchema(
            type=RelationType.TREATS,
            name="治疗",
            description="药物治疗疾病",
            domain_types=[EntityType.DRUG],
            range_types=[EntityType.DISEASE]
        ),
        RelationType.TARGETS: RelationSchema(
            type=RelationType.TARGETS,
            name="靶点",
            description="药物靶向基因/蛋白",
            domain_types=[EntityType.DRUG],
            range_types=[EntityType.CONCEPT]
        ),
        RelationType.IS_A: RelationSchema(
            type=RelationType.IS_A,
            name="是一种",
            description="子类与父类的关系",
            domain_types=[EntityType.DISEASE, EntityType.DRUG, EntityType.CONCEPT],
            range_types=[EntityType.DISEASE, EntityType.DRUG, EntityType.CONCEPT]
        ),
        RelationType.EMPLOYS: RelationSchema(
            type=RelationType.EMPLOYS,
            name="雇佣",
            description="组织雇佣人员",
            domain_types=[EntityType.ORGANIZATION],
            range_types=[EntityType.PERSON]
        ),
    }
```

### 8.3.2 LLM 关系抽取

```python
class LLMRelationExtractor:
    """基于 LLM 的关系抽取"""
    
    def __init__(self, llm_client):
        self.llm = llm_client
    
    def extract_relations(self, text: str,
                          entities: List[Dict]) -> List[Dict]:
        """
        从文本中抽取实体间的关系
        
        Args:
            text: 输入文本
            entities: 已抽取的实体列表
            
        Returns:
            [{"subject": "实体1", "relation": "关系类型", "object": "实体2", 
              "confidence": 0.95, "evidence": "原文证据"}, ...]
        """
        # 构建实体列表
        entity_names = [e["name"] for e in entities]
        entity_str = ", ".join(entity_names)
        
        prompt = f"""你是一个知识图谱关系抽取专家。请从以下文本中抽取实体之间的语义关系。

文本：
{text}

已知实体：{entity_str}

请按以下 JSON 格式返回关系列表：
[
  {{
    "subject": "主语实体名称",
    "relation": "关系名称（如：生产、治疗、靶点、是一种、属于、位于、研发等）",
    "object": "宾语实体名称",
    "confidence": 0.95,
    "evidence": "原文中支持该关系的文本片段"
  }}
]

要求：
1. 只抽取文本中明确表达的关系
2. subject 和 object 必须来自已知实体列表
3. 关系名称使用简短的中文动词
4. 只返回 JSON 数组，不要其他文字
"""
        
        response = self.llm.chat(prompt)
        relations = self._parse_json_response(response)
        
        # 验证关系
        valid_relations = self._validate_relations(relations, entities)
        
        return valid_relations
    
    def _parse_json_response(self, response: str) -> List[Dict]:
        """解析 LLM 返回的 JSON"""
        import json, re
        
        # 尝试直接解析
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # 尝试提取 JSON 块
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", response)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        return []
    
    def _validate_relations(self, relations: List[Dict],
                            entities: List[Dict]) -> List[Dict]:
        """验证抽取的关系是否有效"""
        entity_names = {e["name"] for e in entities}
        valid = []
        
        for rel in relations:
            subj = rel.get("subject", "")
            obj = rel.get("object", "")
            
            if subj in entity_names and obj in entity_names and subj != obj:
                valid.append(rel)
        
        return valid
```

### 8.3.3 基于共现的关系抽取

对于大规模文档集，共现分析是一种高效的关系发现方法：

```python
from collections import defaultdict, Counter
from typing import List, Tuple

class CooccurrenceRelationExtractor:
    """基于共现的关系抽取"""
    
    def __init__(self, window_size: int = 50):
        """
        Args:
            window_size: 共现窗口大小（字符数）
        """
        self.window_size = window_size
        self.cooccurrence_counts = defaultdict(Counter)  # entity -> {co_entity: count}
    
    def process_document(self, text: str, entities: List[Dict]):
        """
        处理单个文档，统计实体共现
        
        Args:
            text: 文档文本
            entities: 文档中的实体列表
        """
        # 获取实体位置
        entity_positions = []
        for entity in entities:
            name = entity["name"]
            # 查找实体在文本中的所有位置
            start = 0
            while True:
                pos = text.find(name, start)
                if pos == -1:
                    break
                entity_positions.append({
                    "name": name,
                    "type": entity.get("type", ""),
                    "start": pos,
                    "end": pos + len(name)
                })
                start = pos + 1
        
        # 统计窗口内的共现
        for i, ent1 in enumerate(entity_positions):
            for j, ent2 in enumerate(entity_positions):
                if i >= j:
                    continue
                
                # 计算距离
                distance = abs(ent1["start"] - ent2["start"])
                
                if distance <= self.window_size:
                    self.cooccurrence_counts[ent1["name"]][ent2["name"]] += 1
                    self.cooccurrence_counts[ent2["name"]][ent1["name"]] += 1
    
    def extract_relations(self, min_cooccurrence: int = 3) -> List[Dict]:
        """
        从共现统计中抽取关系
        
        Args:
            min_cooccurrence: 最小共现次数
            
        Returns:
            关系列表
        """
        relations = []
        
        for entity1, co_entities in self.cooccurrence_counts.items():
            for entity2, count in co_entities.items():
                if count >= min_cooccurrence and entity1 < entity2:
                    # 计算共现强度（点互信息 PMI）
                    total_ent1 = sum(self.cooccurrence_counts[entity1].values())
                    total_ent2 = sum(self.cooccurrence_counts[entity2].values())
                    total_all = sum(
                        sum(c.values()) for c in self.cooccurrence_counts.values()
                    ) // 2
                    
                    # PMI = log(P(x,y) / (P(x) * P(y)))
                    p_xy = count / total_all if total_all > 0 else 0
                    p_x = total_ent1 / total_all if total_all > 0 else 0
                    p_y = total_ent2 / total_all if total_all > 0 else 0
                    
                    import math
                    pmi = math.log(p_xy / (p_x * p_y) + 1e-10) if p_x > 0 and p_y > 0 else 0
                    
                    relations.append({
                        "subject": entity1,
                        "object": entity2,
                        "relation": "related_to",
                        "cooccurrence_count": count,
                        "pmi_score": round(pmi, 4),
                        "source": "cooccurrence"
                    })
        
        # 按 PMI 排序
        relations.sort(key=lambda r: r["pmi_score"], reverse=True)
        return relations
```

### 8.3.4 规则模板关系抽取

对于一些已知模式的关系，可以用正则表达式规则高效抽取：

```python
import re
from typing import List, Dict, Pattern

class RuleBasedRelationExtractor:
    """基于规则模板的关系抽取"""
    
    def __init__(self):
        # 中文关系抽取规则
        self.rules = [
            # 生产关系
            {
                "relation": "produces",
                "pattern": re.compile(
                    r"(?P<subject>[^，。；]+)(?:生产|研发|制造|开发)"
                    r"(?:了|出|的)?(?P<object>[^，。；]{2,30})"
                )
            },
            # 治疗关系
            {
                "relation": "treats",
                "pattern": re.compile(
                    r"(?P<subject>[^，。；]+)(?:用于治疗|治疗|对…有效)"
                    r"(?:的)?(?P<object>[^，。；]{2,30})"
                )
            },
            # 属于关系
            {
                "relation": "belongs_to",
                "pattern": re.compile(
                    r"(?P<subject>[^，。；]+)(?:属于|是|是一种|隶属于)"
                    r"(?:的)?(?P<object>[^，。；]{2,30})"
                )
            },
            # 靶点关系
            {
                "relation": "targets",
                "pattern": re.compile(
                    r"(?P<subject>[^，。；]+)(?:靶向|针对|作用于)"
                    r"(?:的)?(?P<object>[^，。；]{2,30})"
                )
            },
            # 位于关系
            {
                "relation": "located_in",
                "pattern": re.compile(
                    r"(?P<subject>[^，。；]+)(?:位于|坐落于|在)"
                    r"(?:的)?(?P<object>[^，。；]{2,30})"
                )
            },
        ]
    
    def extract_relations(self, text: str) -> List[Dict]:
        """
        使用规则抽取关系
        
        Args:
            text: 输入文本
            
        Returns:
            关系列表
        """
        relations = []
        
        for rule in self.rules:
            for match in rule["pattern"].finditer(text):
                subject = match.group("subject").strip()
                obj = match.group("object").strip()
                
                # 过滤过长的匹配
                if len(subject) > 50 or len(obj) > 50:
                    continue
                
                # 过滤无意义匹配
                if any(w in subject for w in ["的", "和", "与", "或"]):
                    continue
                
                relations.append({
                    "subject": subject,
                    "relation": rule["relation"],
                    "object": obj,
                    "evidence": match.group(0),
                    "source": "rule"
                })
        
        return relations
```

### 8.3.5 跨文档实体合并

同一实体可能出现在多个文档中，需要跨文档合并：

```python
class CrossDocumentEntityMerger:
    """跨文档实体合并"""
    
    def __init__(self):
        # 实体合并阈值（基于名称相似度）
        self.name_similarity_threshold = 0.85
        # 实体合并阈值（基于描述相似度）
        self.desc_similarity_threshold = 0.75
    
    def merge_entities(self, doc_entities: List[List[Dict]]) -> List[Dict]:
        """
        合并跨文档的同一实体
        
        Args:
            doc_entities: 每个文档的实体列表
            
        Returns:
            合并后的全局实体列表
        """
        from difflib import SequenceMatcher
        
        all_entities = []
        for entities in doc_entities:
            all_entities.extend(entities)
        
        merged = []
        
        for entity in all_entities:
            name = entity["name"]
            found = False
            
            for existing in merged:
                # 精确匹配
                if name == existing["name"]:
                    self._merge_entity_info(existing, entity)
                    found = True
                    break
                
                # 模糊匹配（同义词或缩写）
                similarity = SequenceMatcher(None, name, existing["name"]).ratio()
                if similarity >= self.name_similarity_threshold:
                    self._merge_entity_info(existing, entity)
                    found = True
                    break
            
            if not found:
                merged.append(dict(entity))
        
        return merged
    
    def _merge_entity_info(self, target: Dict, source: Dict):
        """合并实体信息"""
        # 合并提及次数
        target["mention_count"] = target.get("mention_count", 1) + 1
        
        # 合并描述（保留更详细的）
        target_desc = target.get("description", "")
        source_desc = source.get("description", "")
        if len(source_desc) > len(target_desc):
            target["description"] = source_desc
        
        # 合并来源文档
        target_sources = target.get("sources", [])
        source_sources = source.get("sources", [source.get("source", "")])
        target["sources"] = list(set(target_sources + source_sources))
```

---

## 8.4 图存储（Neo4j）

### 8.4.1 Neo4j 节点与关系模式设计

```python
class Neo4jSchemaManager:
    """Neo4j 模式管理器"""
    
    def __init__(self, driver):
        self.driver = driver
    
    def create_constraints(self):
        """创建唯一性约束和索引"""
        with self.driver.session() as session:
            # 实体唯一性约束
            session.run("""
                CREATE CONSTRAINT entity_name_unique IF NOT EXISTS
                FOR (e:Entity) REQUIRE e.name IS UNIQUE
            """)
            
            # 实体类型索引
            session.run("""
                CREATE INDEX entity_type_index IF NOT EXISTS
                FOR (e:Entity) ON (e.type)
            """)
            
            # 实体名称全文索引（支持模糊搜索）
            session.run("""
                CREATE FULLTEXT INDEX entity_name_fulltext IF NOT EXISTS
                FOR (n:Entity) ON EACH [n.name, n.synonyms]
            """)
            
            print("[Neo4j] 约束和索引创建完成")
    
    def create_schema(self):
        """创建节点标签和关系类型"""
        with self.driver.session() as session:
            # 设置实体标签
            session.run("""
                CALL db.createLabel("Entity")
            """)
            
            # 为每种实体类型创建子标签
            entity_types = ["Person", "Organization", "Location", 
                          "Drug", "Disease", "Technology", "Product"]
            
            for et in entity_types:
                session.run(f"""
                    CALL db.createLabel("{et}")
                """)
            
            print(f"[Neo4j] 模式创建完成: {len(entity_types)} 种实体类型")
    
    def get_schema_info(self) -> Dict:
        """获取数据库模式信息"""
        with self.driver.session() as session:
            # 节点标签
            labels = session.run("CALL db.labels()").values()
            
            # 关系类型
            rel_types = session.run("CALL db.relationshipTypes()").values()
            
            # 索引
            indexes = session.run("SHOW INDEXES").data()
            
            return {
                "labels": [l[0] for l in labels],
                "relationship_types": [r[0] for r in rel_types],
                "indexes": indexes
            }
```

### 8.4.2 批量数据导入

```python
from typing import List, Dict
from neo4j import GraphDatabase

class Neo4jBatchImporter:
    """Neo4j 批量导入器"""
    
    def __init__(self, uri: str, user: str, password: str,
                 batch_size: int = 500):
        """
        Args:
            uri: Neo4j URI
            user: 用户名
            password: 密码
            batch_size: 批量提交大小
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.batch_size = batch_size
    
    def close(self):
        self.driver.close()
    
    def import_entities(self, entities: List[Dict]):
        """
        批量导入实体节点
        
        使用 MERGE 避免重复创建
        
        Args:
            entities: 实体列表
        """
        with self.driver.session() as session:
            for i in range(0, len(entities), self.batch_size):
                batch = entities[i:i + self.batch_size]
                
                # 使用 UNWIND 批量处理
                result = session.run("""
                    UNWIND $entities AS entity
                    MERGE (e:Entity {name: entity.name})
                    SET e.type = entity.type,
                        e.description = entity.description,
                        e.synonyms = entity.synonyms,
                        e.source = entity.source
                    // 添加类型子标签
                    FOREACH (ignore IN CASE WHEN entity.type = 'Drug' THEN [1] ELSE [] END |
                        SET e:Drug
                    )
                    FOREACH (ignore IN CASE WHEN entity.type = 'Disease' THEN [1] ELSE [] END |
                        SET e:Disease
                    )
                    FOREACH (ignore IN CASE WHEN entity.type = 'Organization' THEN [1] ELSE [] END |
                        SET e:Organization
                    )
                    FOREACH (ignore IN CASE WHEN entity.type = 'Person' THEN [1] ELSE [] END |
                        SET e:Person
                    )
                    FOREACH (ignore IN CASE WHEN entity.type = 'Location' THEN [1] ELSE [] END |
                        SET e:Location
                    )
                    RETURN count(*) AS created
                """, entities=batch)
                
                created = result.single()["created"]
                print(f"[Neo4j] 导入实体: {i + len(batch)}/{len(entities)}")
    
    def import_relations(self, relations: List[Dict]):
        """
        批量导入关系
        
        Args:
            relations: 关系列表
        """
        with self.driver.session() as session:
            for i in range(0, len(relations), self.batch_size):
                batch = relations[i:i + self.batch_size]
                
                result = session.run("""
                    UNWIND $relations AS rel
                    MATCH (s:Entity {name: rel.subject})
                    MATCH (o:Entity {name: rel.object})
                    MERGE (s)-[r:RELATED {type: rel.relation}]->(o)
                    SET r.confidence = rel.confidence,
                        r.evidence = rel.evidence,
                        r.source = rel.source
                    RETURN count(*) AS created
                """, relations=batch)
                
                created = result.single()["created"]
                print(f"[Neo4j] 导入关系: {i + len(batch)}/{len(relations)}")
    
    def import_graph(self, entities: List[Dict],
                     relations: List[Dict]):
        """
        完整导入图数据
        
        Args:
            entities: 实体列表
            relations: 关系列表
        """
        print(f"[Neo4j] 开始导入: {len(entities)} 个实体, {len(relations)} 条关系")
        
        # 使用事务确保原子性
        tx = self.driver.session().begin_transaction()
        
        try:
            # 导入实体
            for i in range(0, len(entities), self.batch_size):
                batch = entities[i:i + self.batch_size]
                tx.run("""
                    UNWIND $entities AS entity
                    MERGE (e:Entity {name: entity.name})
                    SET e.type = entity.type,
                        e.description = entity.description
                """, entities=batch)
            
            # 导入关系
            for i in range(0, len(relations), self.batch_size):
                batch = relations[i:i + self.batch_size]
                tx.run("""
                    UNWIND $relations AS rel
                    MATCH (s:Entity {name: rel.subject})
                    MATCH (o:Entity {name: rel.object})
                    MERGE (s)-[r:RELATED {type: rel.relation}]->(o)
                    SET r.confidence = rel.confidence
                """, relations=batch)
            
            tx.commit()
            print(f"[Neo4j] 导入完成")
            
        except Exception as e:
            tx.rollback()
            print(f"[Neo4j] 导入失败，已回滚: {e}")
            raise
```

### 8.4.3 节点与关系属性设计

```python
@dataclass
class EntityNode:
    """实体节点属性"""
    name: str                    # 实体名称（唯一标识）
    type: str                    # 实体类型
    description: str = ""        # 描述
    aliases: List[str] = None    # 别名列表
    properties: Dict = None      # 附加属性
    source: str = ""             # 来源文档
    created_at: str = ""         # 创建时间
    updated_at: str = ""         # 更新时间
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "synonyms": ",".join(self.aliases or []),
            "source": self.source
        }


@dataclass
class RelationEdge:
    """关系边属性"""
    subject: str                 # 主语实体名称
    relation: str                # 关系类型
    object: str                  # 宾语实体名称
    confidence: float = 1.0      # 置信度
    evidence: str = ""           # 证据文本
    weight: float = 1.0          # 关系权重
    source: str = ""             # 来源
    properties: Dict = None      # 附加属性
    
    def to_dict(self) -> Dict:
        return {
            "subject": self.subject,
            "relation": self.relation,
            "object": self.object,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "weight": self.weight,
            "source": self.source
        }
```

---

## 8.5 图检索

### 8.5.1 Cypher 查询基础

```python
class GraphRetriever:
    """基于 Neo4j 的图检索器"""
    
    def __init__(self, driver):
        self.driver = driver
    
    def find_entity(self, name: str) -> List[Dict]:
        """
        查找实体
        
        MATCH (e:Entity)
        WHERE e.name CONTAINS $name
           OR e.synonyms CONTAINS $name
        RETURN e
        """
        with self.driver.session() as session:
            result = session.run("""
                MATCH (e:Entity)
                WHERE e.name CONTAINS $name
                   OR e.synonyms CONTAINS $name
                RETURN e.name AS name,
                       e.type AS type,
                       e.description AS description
                LIMIT 20
            """, name=name)
            
            return [record.data() for record in result]
    
    def get_neighbors(self, entity_name: str,
                      max_hop: int = 2,
                      relation_types: List[str] = None) -> Dict:
        """
        获取实体的 N 跳邻居
        
        Args:
            entity_name: 实体名称
            max_hop: 最大跳数
            relation_types: 关系类型过滤
            
        Returns:
            包含节点和边的子图
        """
        # 构建关系类型过滤
        rel_filter = ""
        if relation_types:
            rel_filter = "|".join(relation_types)
        
        with self.driver.session() as session:
            query = f"""
                MATCH path = (start:Entity {{name: $name}})
                            -[r:RELATED*1..{max_hop}]-
                            (neighbor:Entity)
                WHERE ALL(rel IN r WHERE 
                    CASE WHEN $rel_filter IS NOT NULL 
                    THEN rel.type IN $relation_types 
                    ELSE true END)
                RETURN start.name AS source,
                       [rel in r | rel.type] AS relation_chain,
                       neighbor.name AS target,
                       neighbor.type AS target_type,
                       length(path) AS hops,
                       [node in nodes(path) | {{name: node.name, type: node.type}}] AS path_nodes
                LIMIT 200
            """
            
            result = session.run(
                query,
                name=entity_name,
                rel_filter=rel_filter if relation_types else None,
                relation_types=relation_types or []
            )
            
            nodes = {}
            edges = []
            
            for record in result:
                data = record.data()
                
                # 收集所有路径节点
                for node in data["path_nodes"]:
                    if node["name"] not in nodes:
                        nodes[node["name"]] = node
                
                # 构建边
                for i, rel_type in enumerate(data["relation_chain"]):
                    if i < len(data["path_nodes"]) - 1:
                        source = data["path_nodes"][i]["name"]
                        target = data["path_nodes"][i + 1]["name"]
                        edges.append({
                            "source": source,
                            "target": target,
                            "type": rel_type,
                            "hops": data["hops"]
                        })
            
            return {
                "center": entity_name,
                "nodes": list(nodes.values()),
                "edges": edges,
                "node_count": len(nodes),
                "edge_count": len(edges)
            }
    
    def find_shortest_path(self, entity_a: str,
                           entity_b: str,
                           max_depth: int = 5) -> List[Dict]:
        """
        查找两个实体间的最短路径
        
        Args:
            entity_a: 起始实体
            entity_b: 目标实体
            max_depth: 最大深度
            
        Returns:
            最短路径列表
        """
        with self.driver.session() as session:
            result = session.run("""
                MATCH path = shortestPath(
                    (a:Entity)-[*1..{max_depth}]-(b:Entity)
                )
                WHERE a.name = $entity_a 
                  AND b.name = $entity_b
                RETURN [node in nodes(path) | node.name] AS path_nodes,
                       [rel in relationships(path) | rel.type] AS relation_types,
                       length(path) AS path_length
            """, entity_a=entity_a, entity_b=entity_b, max_depth=max_depth)
            
            return [record.data() for record in result]
    
    def community_query(self, entity_name: str,
                        depth: int = 2) -> str:
        """
        获取实体社区的上下文摘要
        
        Args:
            entity_name: 实体名称
            depth: 遍历深度
            
        Returns:
            自然语言描述的社区上下文
        """
        neighborhood = self.get_neighbors(entity_name, max_hop=depth)
        
        if neighborhood["node_count"] == 0:
            return f"实体 '{entity_name}' 在知识图谱中没有关联信息。"
        
        lines = [f"实体: {entity_name}"]
        lines.append(f"关联实体 ({neighborhood['node_count']} 个):")
        
        for node in neighborhood["nodes"]:
            if node["name"] != entity_name:
                lines.append(f"  - {node['name']} ({node.get('type', '未知')})")
        
        lines.append(f"\n关系 ({neighborhood['edge_count']} 条):")
        for edge in neighborhood["edges"][:20]:  # 限制显示数量
            lines.append(f"  {edge['source']} --[{edge['type']}]--> {edge['target']}")
        
        return "\n".join(lines)
```

### 8.5.2 高级图查询

```python
class AdvancedGraphQuery:
    """高级图查询"""
    
    def __init__(self, driver):
        self.driver = driver
    
    def find_common_neighbors(self, entity_a: str,
                              entity_b: str) -> List[Dict]:
        """
        查找两个实体的共同邻居
        
        MATCH (a:Entity {name: $a})--(common)--(b:Entity {name: $b})
        RETURN common
        """
        with self.driver.session() as session:
            result = session.run("""
                MATCH (a:Entity {name: $entity_a})
                      -[r1:RELATED]-
                      (common:Entity)
                      -[r2:RELATED]-
                      (b:Entity {name: $entity_b})
                RETURN common.name AS name,
                       common.type AS type,
                       type(r1) AS rel_to_a,
                       type(r2) AS rel_to_b
                LIMIT 20
            """, entity_a=entity_a, entity_b=entity_b)
            
            return [record.data() for record in result]
    
    def find_paths_with_conditions(self, start_type: str,
                                   end_type: str,
                                   max_depth: int = 4) -> List[Dict]:
        """
        带条件的路径查找
        
        例如：从 Drug 到 Disease，路径长度不超过 4
        """
        with self.driver.session() as session:
            result = session.run("""
                MATCH path = (start)-[r:RELATED*1..{max_depth}]-(end)
                WHERE start.type = $start_type 
                  AND end.type = $end_type
                  AND start <> end
                RETURN [node in nodes(path) | node.name] AS path_nodes,
                       [rel in r | rel.type] AS relation_types,
                       length(path) AS path_length,
                       start.name AS start_name,
                       end.name AS end_name
                LIMIT 50
            """, start_type=start_type, end_type=end_type, max_depth=max_depth)
            
            return [record.data() for record in result]
    
    def graph_statistics(self) -> Dict:
        """获取图统计信息"""
        with self.driver.session() as session:
            node_count = session.run(
                "MATCH (n:Entity) RETURN count(n) AS count"
            ).single()["count"]
            
            rel_count = session.run(
                "MATCH ()-[r:RELATED]->() RETURN count(r) AS count"
            ).single()["count"]
            
            type_dist = session.run("""
                MATCH (n:Entity)
                RETURN n.type AS type, count(*) AS count
                ORDER BY count DESC
            """).data()
            
            degree_dist = session.run("""
                MATCH (n:Entity)
                RETURN n.name AS name,
                       size((n)--()) AS degree
                ORDER BY degree DESC
                LIMIT 10
            """).data()
            
            return {
                "node_count": node_count,
                "relation_count": rel_count,
                "type_distribution": type_dist,
                "top_degree_nodes": degree_dist
            }
```

---

## 8.6 图增强的 RAG 集成

### 8.6.1 图增强的 Prompt 构建

```python
class GraphAugmentedRAG:
    """图增强的 RAG 系统"""
    
    def __init__(self, vector_retriever, graph_retriever):
        self.vector_retriever = vector_retriever
        self.graph_retriever = graph_retriever
    
    def retrieve(self, query: str, top_k: int = 10) -> Dict:
        """
        图增强的检索
        
        策略：
        1. 从查询中提取实体
        2. 从知识图谱获取实体邻居
        3. 从向量库检索相关文档
        4. 融合结果
        """
        # 1. 提取查询中的实体
        entities = self._extract_query_entities(query)
        
        # 2. 图检索：获取实体上下文
        graph_context = ""
        if entities:
            graph_parts = []
            for entity in entities[:3]:  # 最多 3 个实体
                community = self.graph_retriever.community_query(
                    entity, depth=2
                )
                graph_parts.append(community)
            graph_context = "\n\n".join(graph_parts)
        
        # 3. 向量检索：获取相关文档
        vector_results = self.vector_retriever.search(query, top_k=top_k)
        
        # 4. 构建增强上下文
        augmented_context = self._build_augmented_context(
            query, vector_results, graph_context
        )
        
        return {
            "documents": vector_results,
            "graph_context": graph_context,
            "augmented_context": augmented_context,
            "query_entities": entities
        }
    
    def _extract_query_entities(self, query: str) -> List[str]:
        """从查询中提取实体名称（简单匹配）"""
        # 在实际系统中，这里应该调用实体链接模块
        # 这里使用简单的字符串匹配演示
        known_entities = [
            "奥希替尼", "非小细胞肺癌", "阿斯利康",
            "EGFR", "靶向治疗", "免疫治疗"
        ]
        
        found = []
        for entity in known_entities:
            if entity in query:
                found.append(entity)
        
        return found
    
    def _build_augmented_context(self, query: str,
                                 documents: List[Dict],
                                 graph_context: str) -> str:
        """构建增强的 LLM 上下文"""
        parts = ["## 检索到的文档", ""]
        
        for i, doc in enumerate(documents[:5]):
            parts.append(f"### 文档 {i+1}")
            parts.append(doc.get("text", ""))
            parts.append("")
        
        if graph_context:
            parts.append("## 知识图谱上下文")
            parts.append(graph_context)
            parts.append("")
        
        return "\n".join(parts)
    
    def generate(self, query: str, llm) -> str:
        """
        图增强的生成
        
        完整流程：检索 -> 增强 -> 生成
        """
        # 检索
        retrieved = self.retrieve(query)
        
        # 构建 Prompt
        prompt = f"""你是一个知识问答助手。请基于以下信息回答用户的问题。

{retrieved['augmented_context']}

用户问题：{query}

请结合检索到的文档和知识图谱信息，给出准确、全面的回答。
如果知识图谱中提供了实体关系信息，请将其融入回答中。
"""
        
        # 生成
        response = llm.generate(prompt)
        
        return {
            "query": query,
            "response": response,
            "sources": len(retrieved["documents"]),
            "graph_entities": retrieved["query_entities"]
        }
```

### 8.6.2 实体链接 + 文本检索混合

```python
class HybridGraphTextRetriever:
    """图 + 文本混合检索器"""
    
    def __init__(self, vector_store, graph_retriever,
                 entity_linker):
        self.vector_store = vector_store
        self.graph = graph_retriever
        self.entity_linker = entity_linker
    
    def hybrid_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        混合检索
        
        融合策略：
        1. 文本检索结果（向量相似度）
        2. 图检索结果（实体相关性）
        3. RRF 融合排序
        """
        # 1. 实体链接
        entities = self.entity_linker.link_entities(query)
        
        # 2. 文本检索
        text_results = self.vector_store.search(query, top_k=top_k * 2)
        
        # 3. 图检索：获取与实体相关的文档
        graph_results = []
        if entities:
            for entity in entities[:3]:
                # 查找与实体关联的文档
                neighborhood = self.graph.get_neighbors(
                    entity, max_hop=2
                )
                # 将图节点映射到文档
                for node in neighborhood.get("nodes", []):
                    doc = self._node_to_document(node)
                    if doc:
                        graph_results.append(doc)
        
        # 4. RRF 融合
        from collections import defaultdict
        
        rrf_scores = defaultdict(float)
        k = 60
        
        for rank, doc in enumerate(text_results):
            doc_id = doc.get("doc_id", "")
            rrf_scores[doc_id] += 1.0 / (k + rank + 1)
        
        for rank, doc in enumerate(graph_results):
            doc_id = doc.get("doc_id", "")
            rrf_scores[doc_id] += 1.0 / (k + rank + 1)
        
        # 5. 排序输出
        ranked = sorted(rrf_scores.items(),
                        key=lambda x: x[1], reverse=True)[:top_k]
        
        results = []
        for doc_id, score in ranked:
            doc = self._get_document(doc_id)
            if doc:
                doc["fusion_score"] = score
                doc["source"] = "graph+text"
                results.append(doc)
        
        return results
    
    def _node_to_document(self, node: Dict) -> Optional[Dict]:
        """将图节点转换为文档"""
        # 实际系统中需要维护节点到文档的映射
        return None
    
    def _get_document(self, doc_id: str) -> Optional[Dict]:
        """根据文档 ID 获取文档内容"""
        # 实际系统中从文档存储中获取
        return None
```

### 8.6.3 完整构建流水线

```python
class KnowledgeGraphPipeline:
    """知识图谱构建完整流水线"""
    
    def __init__(self, llm_client, neo4j_uri: str,
                 neo4j_user: str, neo4j_password: str):
        self.llm = llm_client
        self.neo4j_driver = GraphDatabase.driver(
            neo4j_uri, auth=(neo4j_user, neo4j_password)
        )
        
        # 初始化各模块
        self.entity_extractor = LLMEntityExtractor(llm_client)
        self.relation_extractor = LLMRelationExtractor(llm_client)
        self.disambiguator = EntityDisambiguator()
        self.importer = Neo4jBatchImporter(neo4j_uri, neo4j_user, neo4j_password)
    
    def build_from_documents(self, documents: List[Dict]):
        """
        从文档集合构建知识图谱
        
        Args:
            documents: 文档列表，每项包含 id, title, content
        """
        print(f"[KG] 开始构建知识图谱: {len(documents)} 个文档")
        
        all_entities = []
        all_relations = []
        
        for doc in documents:
            print(f"[KG] 处理文档: {doc.get('title', 'unknown')}")
            
            # 1. 实体抽取
            entities = self.entity_extractor.extract_entities(
                doc.get("title", "") + "\n" + doc.get("content", "")
            )
            
            # 2. 实体消歧
            for entity in entities:
                entity["name"] = self.disambiguator.normalize_name(
                    entity["name"]
                )
            
            # 3. 关系抽取
            relations = self.relation_extractor.extract_relations(
                doc.get("content", ""), entities
            )
            
            all_entities.extend(entities)
            all_relations.extend(relations)
        
        # 4. 跨文档合并
        merger = CrossDocumentEntityMerger()
        merged_entities = merger.merge_entities([all_entities])
        
        # 5. 导入 Neo4j
        self.importer.import_entities(merged_entities)
        self.importer.import_relations(all_relations)
        
        print(f"[KG] 构建完成: {len(merged_entities)} 个实体, "
              f"{len(all_relations)} 条关系")
        
        return {
            "entity_count": len(merged_entities),
            "relation_count": len(all_relations)
        }
    
    def close(self):
        self.neo4j_driver.close()
        self.importer.close()
```

---

## 8.7 知识图谱最佳实践

### 8.7.1 构建策略选择

| 场景 | 推荐策略 | 说明 |
|------|---------|------|
| 小规模（<1K 文档） | LLM 抽取 | 质量高，成本可接受 |
| 中等规模（1K-10K） | LLM + 规则混合 | LLM 抽实体，规则抽关系 |
| 大规模（>10K） | NLP 工具 + 共现 | spaCy/HanLP 抽实体，共现抽关系 |
| 实时构建 | 流式处理 | Kafka + 轻量抽取 |
| 增量更新 | 新文档单独抽取 + 合并 | 避免全量重建 |

### 8.7.2 常见问题

| 问题 | 现象 | 解决方案 |
|------|------|---------|
| 实体过度抽取 | 把普通名词当作实体 | 限定实体类型范围，设置置信度阈值 |
| 关系冗余 | 同一关系被多次抽取 | 使用 MERGE 去重，合并置信度 |
| 图谱稀疏 | 实体多但关系少 | 增加共现分析，补充隐式关系 |
| 数据不一致 | 同一实体在不同文档中名称不同 | 建立同义词表，实体消歧 |
| 查询延迟 | N 跳邻居查询过慢 | 限制跳数（<=3），建立索引，使用图缓存 |
| 构建成本高 | LLM 调用费用过高 | 使用规则+共现做初筛，LLM 只做精校 |

---

## 本章小结

知识图谱为 RAG 系统提供了精确的关系推理能力，与向量检索的模糊语义匹配形成互补。本章从实体抽取（LLM 驱动、NLP 工具、规则）、关系建模与抽取（LLM、共现、规则模板）、图存储（Neo4j 模式设计、批量导入、索引优化）、图检索（Cypher 查询、N 跳邻居、社区查询）到 RAG 集成（图增强 Prompt、混合检索），完整覆盖了知识图谱在 RAG 中的构建与应用。

核心要点：
- 实体消歧和跨文档合并是知识图谱质量的保障
- 关系抽取优先使用规则和共现，LLM 用于复杂关系
- Neo4j 的 MERGE 和批量导入是性能关键
- 图检索限制在 2-3 跳以内，过深查询延迟高且信息稀释
- 知识图谱与向量检索是互补关系，融合使用效果最佳
