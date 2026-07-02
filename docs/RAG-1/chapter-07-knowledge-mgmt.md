# 第7章 知识体系管理

## 7.1 知识体系管理的必要性

### 7.1.1 从"数据堆积"到"知识管理"

当企业知识库中的文档数量从几百增长到几万、几十万时，一个根本性的问题浮现出来：**检索系统无法区分"相关的噪声"和"不相关的噪声"**。如果不做知识体系管理，检索结果将面临以下问题：

1. **领域混淆**：医疗查询可能检索到同名的法律文档
2. **权限越界**：非授权用户检索到机密文档
3. **信息过时**：旧版本文档仍然排在检索结果前列
4. **质量参差**：草稿文档与正式发布文档混在一起
5. **标签混乱**：同一概念被标记为不同的标签

知识体系管理（Knowledge Management System, KMS）的核心目标是建立结构化的知识组织方式，使检索系统能够：

- **精确过滤**：根据领域、类型、权限等维度筛选
- **高效排序**：优先展示高质量、高权威性的内容
- **生命周期管理**：跟踪知识的创建、更新、过期、废弃

```
无知识管理的 RAG：         有知识管理的 RAG：

用户查询                    用户查询
    │                          │
    ▼                          ▼
[所有文档]                [按领域过滤 → 按权限过滤
    │                        → 按新鲜度排序]
    ▼                          │
[随机排序结果]                 ▼
    │                    [高质量、相关、安全的结果]
    ▼                          │
LLM 生成                      ▼
                          LLM 生成（更好的上下文）
```

### 7.1.2 知识管理的维度模型

企业知识管理可以从以下维度构建：

| 维度 | 描述 | 示例 |
|------|------|------|
| 领域（Domain） | 知识所属的业务领域 | 医疗、法律、金融、技术 |
| 文档类型（DocType） | 文档的格式/用途分类 | 报告、手册、规范、指南 |
| 标签（Tag） | 多维度的自由标记 | #靶向治疗 #EGFR #2024版 |
| 权限（Permission） | 访问控制级别 | 公开、内部、机密、绝密 |
| 版本（Version） | 文档版本追踪 | v1.0, v2.0, draft-3 |
| 新鲜度（Freshness） | 信息时效性 | 最近更新、即将过期、已过期 |

---

## 7.2 知识分类

### 7.2.1 领域分类体系

建立领域分类体系是知识管理的首要任务。不同的业务领域对知识的组织方式有本质差异。

```python
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum

class Domain(Enum):
    """顶层领域枚举"""
    MEDICAL = "medical"
    LEGAL = "legal"
    FINANCE = "finance"
    TECHNOLOGY = "technology"
    MANUFACTURING = "manufacturing"
    EDUCATION = "education"
    HUMAN_RESOURCES = "human_resources"
    GENERAL = "general"


@dataclass
class DomainNode:
    """领域节点"""
    id: str
    name: str
    description: str
    parent: Optional[str] = None
    children: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)


class DomainTaxonomy:
    """领域分类体系"""
    
    # 预定义的中文领域分类树
    DOMAIN_TREE = {
        "medical": DomainNode(
            id="medical",
            name="医疗健康",
            description="医疗健康领域的知识文档",
            keywords=["医疗", "医院", "临床", "诊断", "治疗", "药物", "疾病"]
        ),
        "medical_oncology": DomainNode(
            id="medical_oncology",
            name="肿瘤学",
            description="肿瘤相关的诊断、治疗、药物知识",
            parent="medical",
            keywords=["肿瘤", "癌症", "化疗", "靶向", "免疫", "肺癌", "乳腺癌"]
        ),
        "medical_cardiology": DomainNode(
            id="medical_cardiology",
            name="心血管",
            description="心血管疾病相关的知识",
            parent="medical",
            keywords=["心脏", "心血管", "高血压", "冠心病", "心律失常"]
        ),
        "legal": DomainNode(
            id="legal",
            name="法律法规",
            description="法律法规和合规知识",
            keywords=["法律", "法规", "合规", "合同", "知识产权"]
        ),
        "legal_contract": DomainNode(
            id="legal_contract",
            name="合同法",
            description="合同相关的法律法规",
            parent="legal",
            keywords=["合同", "协议", "契约", "违约", "赔偿"]
        ),
        "finance": DomainNode(
            id="finance",
            name="金融财经",
            description="金融、财务、投资领域的知识",
            keywords=["金融", "财务", "投资", "会计", "审计", "税务"]
        ),
        "technology": DomainNode(
            id="technology",
            name="技术研发",
            description="技术研发和工程实践知识",
            keywords=["技术", "研发", "工程", "代码", "架构", "系统"]
        ),
        "technology_ai": DomainNode(
            id="technology_ai",
            name="人工智能",
            description="AI 和机器学习相关技术知识",
            parent="technology",
            keywords=["AI", "机器学习", "深度学习", "自然语言处理", "大模型"]
        ),
    }
    
    @classmethod
    def get_domain_path(cls, domain_id: str) -> List[str]:
        """获取领域的完整路径"""
        node = cls.DOMAIN_TREE.get(domain_id)
        if not node:
            return []
        
        path = [node.name]
        current = node
        while current.parent:
            parent = cls.DOMAIN_TREE.get(current.parent)
            if parent:
                path.insert(0, parent.name)
                current = parent
            else:
                break
        
        return path
    
    @classmethod
    def get_all_leaves(cls) -> List[str]:
        """获取所有叶子节点"""
        all_ids = set(cls.DOMAIN_TREE.keys())
        parent_ids = {
            node.parent for node in cls.DOMAIN_TREE.values()
            if node.parent
        }
        return list(all_ids - parent_ids)
    
    @classmethod
    def auto_classify(cls, text: str, confidence_threshold: float = 0.3) -> List[tuple]:
        """
        基于关键词自动分类文本到领域
        
        Args:
            text: 文本内容
            confidence_threshold: 置信度阈值
            
        Returns:
            [(domain_id, confidence), ...]
        """
        import jieba
        words = set(jieba.cut(text))
        
        scores = {}
        for domain_id, node in cls.DOMAIN_TREE.items():
            keyword_matches = sum(1 for kw in node.keywords if kw in words)
            if keyword_matches > 0:
                scores[domain_id] = keyword_matches / len(node.keywords)
        
        # 按置信度降序排列
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(d, s) for d, s in ranked if s >= confidence_threshold]
```

### 7.2.2 文档类型分类

除了业务领域，文档类型（DocType）提供了另一个重要的分类维度：

```python
@dataclass
class DocumentType:
    """文档类型"""
    id: str
    name: str
    description: str
    priority: int  # 排序优先级（越高越重要）
    icon: str = ""


class DocumentTypeClassifier:
    """文档类型分类器"""
    
    # 预定义的文档类型
    DOC_TYPES = {
        "guideline": DocumentType(
            id="guideline", name="指南/规范",
            description="官方发布的指南、标准、规范文件",
            priority=100
        ),
        "manual": DocumentType(
            id="manual", name="操作手册",
            description="产品使用手册、操作说明书",
            priority=90
        ),
        "report": DocumentType(
            id="report", name="报告",
            description="研究报告、分析报告、年度报告",
            priority=80
        ),
        "specification": DocumentType(
            id="specification", name="技术规格",
            description="技术规格书、需求规格说明",
            priority=85
        ),
        "tutorial": DocumentType(
            id="tutorial", name="教程",
            description="教学材料、培训文档、最佳实践",
            priority=70
        ),
        "faq": DocumentType(
            id="faq", name="FAQ",
            description="常见问题解答",
            priority=75
        ),
        "changelog": DocumentType(
            id="changelog", name="变更日志",
            description="版本更新记录、变更说明",
            priority=50
        ),
        "draft": DocumentType(
            id="draft", name="草稿",
            description="未完成的文档、讨论稿",
            priority=30
        ),
        "meeting_notes": DocumentType(
            id="meeting_notes", name="会议记录",
            description="会议纪要、讨论记录",
            priority=40
        ),
        "other": DocumentType(
            id="other", name="其他",
            description="未分类的文档",
            priority=10
        ),
    }
    
    # 文档类型识别规则
    TYPE_PATTERNS = {
        "guideline": ["指南", "规范", "标准", "规程", "SOP", "标准操作"],
        "manual": ["手册", "说明书", "用户指南", "操作指南"],
        "report": ["报告", "总结", "总结报告", "年报", "半年报"],
        "specification": ["规格", "需求规格", "技术要求", "技术方案"],
        "tutorial": ["教程", "教学", "培训", "最佳实践", "how-to"],
        "faq": ["FAQ", "常见问题", "常见问答", "Q&A"],
        "changelog": ["更新日志", "变更日志", "Release Notes", "版本说明"],
        "draft": ["草稿", "草案", "讨论稿", "征求意见稿"],
        "meeting_notes": ["会议记录", "会议纪要", "会议总结"],
    }
    
    @classmethod
    def classify_by_title(cls, title: str) -> str:
        """根据标题识别文档类型"""
        for doc_type, patterns in cls.TYPE_PATTERNS.items():
            for pattern in patterns:
                if pattern in title:
                    return doc_type
        return "other"
    
    @classmethod
    def classify_by_content(cls, text: str) -> str:
        """根据内容识别文档类型（更精确）"""
        import jieba
        words = " ".join(jieba.cut(text))
        
        scores = {}
        for doc_type, patterns in cls.TYPE_PATTERNS.items():
            score = sum(1 for p in patterns if p in words)
            if score > 0:
                scores[doc_type] = score
        
        if not scores:
            return "other"
        
        return max(scores, key=scores.get)
```

### 7.2.3 层次分类与标签分类

层次分类（Hierarchical）和标签分类（Tag-based）是两种主流的分类方式，各有优劣：

| 特性 | 层次分类 | 标签分类 |
|------|---------|---------|
| 结构 | 树状，有父子关系 | 扁平，多对多 |
| 灵活性 | 一个文档只能在一个叶子节点 | 一个文档可以有多个标签 |
| 导航性 | 易于浏览和钻取 | 适合精确过滤 |
| 维护成本 | 树结构调整成本高 | 新增标签成本低 |
| 典型场景 | 图书分类、产品目录 | 博客文章、知识库 |

**最佳实践：层次分类 + 标签分类组合使用**

```python
@dataclass
class KnowledgeClassification:
    """完整的知识分类"""
    domain: str                    # 领域（层次分类）
    domain_path: List[str]         # 领域路径
    doc_type: str                  # 文档类型
    tags: List[str]                # 自由标签
    importance: int = 5            # 重要性 1-10
    confidence: float = 1.0        # 分类置信度
    
    def to_dict(self) -> Dict:
        return {
            "domain": self.domain,
            "domain_path": self.domain_path,
            "doc_type": self.doc_type,
            "tags": self.tags,
            "importance": self.importance,
            "confidence": self.confidence
        }


class ClassifierPipeline:
    """分类流水线"""
    
    def __init__(self):
        self.domain_taxonomy = DomainTaxonomy()
        self.doc_type_classifier = DocumentTypeClassifier()
    
    def classify(self, title: str, content: str) -> KnowledgeClassification:
        """
        对文档进行多维度分类
        
        Args:
            title: 文档标题
            content: 文档内容
            
        Returns:
            完整的分类结果
        """
        # 1. 领域分类
        domain_scores = self.domain_taxonomy.auto_classify(content)
        best_domain = domain_scores[0][0] if domain_scores else "general"
        domain_confidence = domain_scores[0][1] if domain_scores else 0.0
        
        # 2. 文档类型分类
        doc_type = self.doc_type_classifier.classify_by_content(content)
        
        # 3. 自动标签生成（见 7.3 节）
        tags = self._auto_generate_tags(title, content)
        
        return KnowledgeClassification(
            domain=best_domain,
            domain_path=self.domain_taxonomy.get_domain_path(best_domain),
            doc_type=doc_type,
            tags=tags,
            confidence=domain_confidence
        )
    
    def _auto_generate_tags(self, title: str, content: str) -> List[str]:
        """自动生成标签（简化版）"""
        import jieba.analyse
        keywords = jieba.analyse.extract_tags(
            title + " " + content[:1000],
            topK=5
        )
        return keywords
```

---

## 7.3 标签系统

### 7.3.1 自动标签生成

标签系统是知识管理中灵活性最高的分类维度。自动标签生成可以大幅降低人工标注成本。

```python
from typing import List, Dict, Set
import jieba
import jieba.analyse
from collections import Counter

class AutoTagger:
    """自动标签生成器"""
    
    def __init__(self):
        # 领域词典
        self.domain_terms = {
            "medical": ["临床", "诊断", "治疗", "手术", "药物", "患者", "剂量"],
            "legal": ["合同", "条款", "违约", "诉讼", "赔偿", "法律", "合规"],
            "finance": ["营收", "利润", "资产", "负债", "现金流", "审计"],
            "tech": ["架构", "接口", "部署", "数据库", "API", "微服务"],
        }
    
    def extract_keywords(self, text: str, top_k: int = 10) -> List[str]:
        """
        使用 TF-IDF 提取关键词
        
        Args:
            text: 文本内容
            top_k: 提取的关键词数量
            
        Returns:
            关键词列表
        """
        keywords = jieba.analyse.extract_tags(text, topK=top_k)
        return keywords
    
    def extract_keywords_textrank(self, text: str, top_k: int = 10) -> List[str]:
        """
        使用 TextRank 提取关键词（考虑词间关系）
        
        TextRank 基于图模型，比 TF-IDF 更稳定
        """
        keywords = jieba.analyse.textrank(text, topK=top_k)
        return keywords
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        提取命名实体（需要 HanLP 或类似工具）
        
        Returns:
            {"PERSON": [...], "ORG": [...], "LOC": [...], ...}
        """
        try:
            import hanlp
            # 使用 HanLP 的多任务模型
            tokenizer = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
            ner = hanlp.load(hanlp.pretrained.ner.MSRA_NER_ELECTRA_SMALL_ZH)
            
            tokens = tokenizer(text)
            entities = ner(tokens)
            
            result = {}
            for entity_type, entity_text, start, end in entities:
                if entity_type not in result:
                    result[entity_type] = []
                result[entity_type].append(entity_text)
            
            return result
            
        except ImportError:
            print("[Tagger] HanLP 未安装，使用 jieba 词性标注替代")
            return self._jieba_entity_extraction(text)
    
    def _jieba_entity_extraction(self, text: str) -> Dict[str, List[str]]:
        """使用 jieba 词性标注做简单实体提取"""
        import jieba.posseg as pseg
        words = pseg.cut(text)
        
        result = {}
        type_map = {
            "nr": "PERSON",
            "ns": "LOC",
            "nt": "ORG",
            "nz": "CONCEPT"
        }
        
        for word, flag in words:
            entity_type = type_map.get(flag)
            if entity_type:
                if entity_type not in result:
                    result[entity_type] = []
                result[entity_type].append(word)
        
        return result
    
    def generate_tags(self, title: str, content: str,
                      max_tags: int = 8) -> List[Dict]:
        """
        综合生成标签
        
        Args:
            title: 标题
            content: 内容
            max_tags: 最大标签数
            
        Returns:
            [{"tag": "标签名", "source": "关键词/实体/领域", "weight": 0.9}, ...]
        """
        text = title + " " + content
        tags = []
        seen = set()
        
        # 1. 从标题提取（高权重）
        title_keywords = self.extract_keywords(title, top_k=3)
        for kw in title_keywords:
            if kw not in seen and len(kw) >= 2:
                tags.append({"tag": kw, "source": "title", "weight": 1.0})
                seen.add(kw)
        
        # 2. 从内容提取关键词
        content_keywords = self.extract_keywords(content, top_k=10)
        for kw in content_keywords:
            if kw not in seen and len(kw) >= 2:
                tags.append({"tag": kw, "source": "keyword", "weight": 0.8})
                seen.add(kw)
        
        # 3. 提取命名实体
        entities = self.extract_entities(text)
        entity_weight_map = {
            "PERSON": 0.9,
            "ORG": 0.9,
            "LOC": 0.7,
            "CONCEPT": 0.8
        }
        for entity_type, entity_list in entities.items():
            weight = entity_weight_map.get(entity_type, 0.6)
            for entity in entity_list[:3]:  # 每种类型最多 3 个
                if entity not in seen and len(entity) >= 2:
                    tags.append({"tag": entity, "source": f"entity:{entity_type}", "weight": weight})
                    seen.add(entity)
        
        # 4. 按权重排序并截断
        tags.sort(key=lambda x: x["weight"], reverse=True)
        return tags[:max_tags]


class LLMAutoTagger:
    """基于 LLM 的自动标签生成（更智能但更慢）"""
    
    def __init__(self, llm_client):
        self.llm = llm_client
    
    def generate_tags(self, title: str, content: str,
                      candidate_tags: List[str] = None) -> List[str]:
        """
        使用 LLM 生成标签
        
        Args:
            title: 文档标题
            content: 文档内容（截取前 2000 字）
            candidate_tags: 候选标签列表（标签库）
            
        Returns:
            标签列表
        """
        prompt = f"""你是一个知识库标签专家。请为以下文档生成 3-5 个标签。

文档标题：{title}
文档内容（摘要）：{content[:2000]}

"""
        if candidate_tags:
            prompt += f"""请从以下候选标签中选择最合适的（也可以新增）：
候选标签：{', '.join(candidate_tags)}

"""
        
        prompt += """要求：
1. 每个标签 2-6 个字
2. 标签要具体，不要太宽泛（如"技术"太宽泛，"知识图谱"合适）
3. 覆盖文档的核心主题
4. 按相关性从高到低排列

请只返回标签列表，每行一个标签。"""
        
        response = self.llm.generate(prompt)
        tags = [line.strip().strip("-").strip() 
                for line in response.split("\n") 
                if line.strip() and not line.startswith("```")]
        
        return tags[:8]
```

### 7.3.2 标签继承与传播

标签继承允许子分类自动获得父分类的标签：

```python
class TagInheritanceEngine:
    """标签继承引擎"""
    
    def __init__(self):
        # 定义标签继承规则
        self.inheritance_rules = {
            # 子分类 -> 继承自父分类的标签
            "medical_oncology": ["medical", "cancer", "treatment"],
            "medical_cardiology": ["medical", "heart", "cardiovascular"],
            "legal_contract": ["legal", "contract_law"],
            "technology_ai": ["technology", "artificial_intelligence"],
        }
        
        # 标签传播规则（基于文档相似度）
        self.propagation_threshold = 0.85
    
    def get_inherited_tags(self, domain: str) -> List[str]:
        """获取从父分类继承的标签"""
        return self.inheritance_rules.get(domain, [])
    
    def propagate_tags(self, source_doc: Dict,
                       target_doc: Dict,
                       similarity: float) -> List[str]:
        """
        在相似文档间传播标签
        
        Args:
            source_doc: 源文档
            target_doc: 目标文档
            similarity: 文档相似度
            
        Returns:
            传播的标签列表
        """
        if similarity < self.propagation_threshold:
            return []
        
        source_tags = set(source_doc.get("tags", []))
        target_tags = set(target_doc.get("tags", []))
        
        # 只传播源文档有但目标文档没有的标签
        new_tags = source_tags - target_tags
        
        # 权重随相似度衰减
        if similarity >= 0.95:
            return list(new_tags)
        elif similarity >= 0.90:
            # 只传播高置信度的标签
            return [t for t in new_tags if t in source_doc.get("high_confidence_tags", [])]
        else:
            return []
```

### 7.3.3 多维度标签体系

```python
@dataclass
class MultiDimensionalTag:
    """多维标签"""
    dimension: str   # 维度: "topic" | "entity" | "stage" | "priority" | "quality"
    value: str       # 标签值
    confidence: float = 1.0
    
    
class MultiDimensionalTagger:
    """多维度标签系统"""
    
    DIMENSIONS = {
        "topic": {
            "description": "主题维度",
            "examples": ["知识图谱", "向量检索", "文本生成"],
        },
        "entity": {
            "description": "实体维度",
            "examples": ["奥希替尼", "Neo4j", "BERT"],
        },
        "stage": {
            "description": "项目阶段",
            "values": ["需求", "设计", "开发", "测试", "运维"],
        },
        "priority": {
            "description": "优先级",
            "values": ["P0-紧急", "P1-重要", "P2-常规", "P3-低优先级"],
        },
        "quality": {
            "description": "质量等级",
            "values": ["reviewed", "draft", "archived"],
        },
        "audience": {
            "description": "目标读者",
            "values": ["开发者", "产品经理", "测试", "运维", "管理层"],
        }
    }
    
    @classmethod
    def tag_document(cls, doc: Dict) -> List[MultiDimensionalTag]:
        """为文档打多维标签"""
        tags = []
        
        # topic 维度
        if "domain" in doc:
            tags.append(MultiDimensionalTag(
                dimension="topic",
                value=doc["domain"],
                confidence=0.9
            ))
        
        # quality 维度
        if doc.get("is_reviewed", False):
            tags.append(MultiDimensionalTag(
                dimension="quality",
                value="reviewed",
                confidence=1.0
            ))
        
        return tags
```

---

## 7.4 权限管理

### 7.4.1 文档级与知识库级权限

企业知识库必须实现精细的访问控制，防止信息泄露。

```python
from enum import Enum
from typing import List, Set, Optional
from dataclasses import dataclass

class Permission(Enum):
    """权限级别"""
    DENY = 0           # 禁止访问
    READ = 1           # 只读
    WRITE = 2          # 读写
    ADMIN = 3          # 管理


@dataclass
class User:
    """用户"""
    id: str
    name: str
    department: str
    roles: List[str]
    security_clearance: int  # 安全等级 1-5


@dataclass
class DocumentACL:
    """文档访问控制列表"""
    doc_id: str
    owner: str                    # 所有者
    allowed_users: List[str]      # 允许的用户列表
    allowed_roles: List[str]      # 允许的角色列表
    allowed_departments: List[str] # 允许的部门
    min_security_clearance: int   # 最低安全等级
    is_public: bool = False       # 是否公开
    permission: Permission = Permission.READ


class AccessControlManager:
    """访问控制管理器"""
    
    def __init__(self):
        self.document_acls: Dict[str, DocumentACL] = {}
        self.users: Dict[str, User] = {}
    
    def register_user(self, user: User):
        """注册用户"""
        self.users[user.id] = user
    
    def set_document_acl(self, acl: DocumentACL):
        """设置文档权限"""
        self.document_acls[acl.doc_id] = acl
    
    def check_permission(self, user_id: str,
                         doc_id: str,
                         required: Permission = Permission.READ) -> bool:
        """
        检查用户是否有权限访问文档
        
        Args:
            user_id: 用户 ID
            doc_id: 文档 ID
            required: 需要的权限级别
            
        Returns:
            是否有权限
        """
        user = self.users.get(user_id)
        if not user:
            return False
        
        acl = self.document_acls.get(doc_id)
        if not acl:
            return False  # 默认拒绝
        
        # 公开文档
        if acl.is_public and required == Permission.READ:
            return True
        
        # 所有者
        if user_id == acl.owner:
            return True
        
        # 用户白名单
        if user_id in acl.allowed_users:
            return True
        
        # 角色检查
        if any(role in acl.allowed_roles for role in user.roles):
            return True
        
        # 部门检查
        if user.department in acl.allowed_departments:
            return True
        
        # 安全等级检查
        if user.security_clearance >= acl.min_security_clearance:
            return True
        
        return False
    
    def filter_accessible_docs(self, user_id: str,
                               doc_ids: List[str]) -> List[str]:
        """
        过滤用户可以访问的文档
        
        Args:
            user_id: 用户 ID
            doc_ids: 候选文档 ID 列表
            
        Returns:
            可访问的文档 ID 列表
        """
        return [
            doc_id for doc_id in doc_ids
            if self.check_permission(user_id, doc_id, Permission.READ)
        ]
```

### 7.4.2 RBAC 角色权限模型

RBAC（Role-Based Access Control）通过角色将用户与权限解耦，是企业中最常用的权限模型：

```python
class RBACManager:
    """RBAC 权限管理器"""
    
    def __init__(self):
        # role -> set of permissions
        self.role_permissions: Dict[str, Set[str]] = {}
        # user -> set of roles
        self.user_roles: Dict[str, Set[str]] = {}
        
        # 预定义角色
        self._init_default_roles()
    
    def _init_default_roles(self):
        """初始化默认角色"""
        self.role_permissions = {
            "admin": {"*"},  # 通配符，所有权限
            
            "manager": {
                "knowledge_base:create",
                "knowledge_base:update",
                "knowledge_base:delete",
                "document:create",
                "document:update",
                "document:delete",
                "document:read",
                "document:publish",
                "tag:manage",
                "user:view",
            },
            
            "editor": {
                "document:create",
                "document:update",
                "document:read",
                "document:delete_own",
                "tag:assign",
            },
            
            "viewer": {
                "document:read",
            },
            
            "auditor": {
                "document:read",
                "audit:view_logs",
            },
        }
    
    def add_role(self, role_name: str, permissions: Set[str]):
        """添加角色"""
        self.role_permissions[role_name] = permissions
    
    def assign_role(self, user_id: str, role_name: str):
        """为用户分配角色"""
        if role_name not in self.role_permissions:
            raise ValueError(f"角色不存在: {role_name}")
        
        if user_id not in self.user_roles:
            self.user_roles[user_id] = set()
        
        self.user_roles[user_id].add(role_name)
    
    def revoke_role(self, user_id: str, role_name: str):
        """撤销用户角色"""
        if user_id in self.user_roles:
            self.user_roles[user_id].discard(role_name)
    
    def has_permission(self, user_id: str, permission: str) -> bool:
        """
        检查用户是否有指定权限
        
        Args:
            user_id: 用户 ID
            permission: 权限字符串，如 "document:read"
            
        Returns:
            是否有权限
        """
        roles = self.user_roles.get(user_id, set())
        
        for role in roles:
            permissions = self.role_permissions.get(role, set())
            if "*" in permissions or permission in permissions:
                return True
        
        return False
    
    def get_user_permissions(self, user_id: str) -> Set[str]:
        """获取用户的所有权限"""
        roles = self.user_roles.get(user_id, set())
        permissions = set()
        
        for role in roles:
            role_perms = self.role_permissions.get(role, set())
            if "*" in role_perms:
                return {"*"}
            permissions.update(role_perms)
        
        return permissions
```

### 7.4.3 检索时权限过滤

在 RAG 检索阶段，权限过滤是保障安全的最后一道关卡：

```python
class PermissionAwareRetriever:
    """权限感知的检索器"""
    
    def __init__(self, base_retriever, acl_manager: AccessControlManager):
        self.base_retriever = base_retriever
        self.acl = acl_manager
    
    def search(self, query: str, user_id: str, top_k: int = 10) -> List[Dict]:
        """
        权限感知的检索
        
        策略：
        1. 先检索（不关心权限）
        2. 过滤无权限的文档
        3. 如果过滤后结果不足，补充检索
        
        Args:
            query: 查询文本
            user_id: 用户 ID
            top_k: 返回结果数
            
        Returns:
            权限过滤后的检索结果
        """
        # 1. 多检索一些（给权限过滤留空间）
        results = self.base_retriever.search(query, top_k=top_k * 3)
        
        # 2. 过滤权限
        filtered = [
            doc for doc in results
            if self.acl.check_permission(user_id, doc["doc_id"], Permission.READ)
        ]
        
        # 3. 如果过滤后结果不足，补充检索
        if len(filtered) < top_k:
            additional = self.base_retriever.search(
                query, top_k=top_k * 5
            )
            more_filtered = [
                doc for doc in additional
                if self.acl.check_permission(user_id, doc["doc_id"], Permission.READ)
                and doc not in filtered
            ]
            filtered.extend(more_filtered)
        
        return filtered[:top_k]
```

---

## 7.5 版本管理

### 7.5.1 文档版本追踪

```python
from datetime import datetime
from typing import List, Optional
import json

class DocumentVersionTracker:
    """文档版本追踪器"""
    
    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        self.versions: Dict[str, List[Dict]] = {}  # doc_id -> [versions]
    
    def create_version(self, doc_id: str, content: str,
                       metadata: Dict = None) -> Dict:
        """
        创建新版本
        
        Args:
            doc_id: 文档 ID
            content: 文档内容
            metadata: 版本元数据
            
        Returns:
            版本信息
        """
        if doc_id not in self.versions:
            self.versions[doc_id] = []
        
        current_versions = self.versions[doc_id]
        version_number = len(current_versions) + 1
        
        version = {
            "doc_id": doc_id,
            "version": version_number,
            "content_hash": hashlib.md5(content.encode()).hexdigest(),
            "content_length": len(content),
            "created_at": datetime.now().isoformat(),
            "metadata": metadata or {},
        }
        
        current_versions.append(version)
        self._persist(doc_id, version)
        
        return version
    
    def get_latest_version(self, doc_id: str) -> Optional[Dict]:
        """获取最新版本"""
        versions = self.versions.get(doc_id, [])
        return versions[-1] if versions else None
    
    def get_version(self, doc_id: str, version_number: int) -> Optional[Dict]:
        """获取指定版本"""
        versions = self.versions.get(doc_id, [])
        for v in versions:
            if v["version"] == version_number:
                return v
        return None
    
    def compare_versions(self, doc_id: str,
                         v1: int, v2: int) -> Dict:
        """
        比较两个版本的差异
        
        Returns:
            {"added": [...], "removed": [...], "changed": bool}
        """
        version1 = self.get_version(doc_id, v1)
        version2 = self.get_version(doc_id, v2)
        
        if not version1 or not version2:
            raise ValueError("版本不存在")
        
        return {
            "v1": v1,
            "v2": v2,
            "hash_match": version1["content_hash"] == version2["content_hash"],
            "size_diff": version2["content_length"] - version1["content_length"],
        }
    
    def _persist(self, doc_id: str, version: Dict):
        """持久化版本信息"""
        import os
        doc_dir = os.path.join(self.storage_path, doc_id)
        os.makedirs(doc_dir, exist_ok=True)
        
        version_file = os.path.join(
            doc_dir,
            f"v{version['version']}.json"
        )
        with open(version_file, "w", encoding="utf-8") as f:
            json.dump(version, f, ensure_ascii=False, indent=2)
```

### 7.5.2 快照与回滚策略

```python
class SnapshotManager:
    """快照管理器"""
    
    def __init__(self, knowledge_base_path: str):
        self.kb_path = knowledge_base_path
        self.snapshot_dir = os.path.join(knowledge_base_path, ".snapshots")
        os.makedirs(self.snapshot_dir, exist_ok=True)
    
    def create_snapshot(self, name: str = None) -> str:
        """
        创建知识库快照
        
        Args:
            name: 快照名称
            
        Returns:
            快照路径
        """
        if name is None:
            name = f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        snapshot_path = os.path.join(self.snapshot_dir, name)
        
        # 创建快照（硬链接以节省空间）
        import shutil
        shutil.copytree(
            self.kb_path,
            snapshot_path,
            ignore=shutil.ignore_patterns(".snapshots"),
            symlinks=False
        )
        
        print(f"[Snapshot] 创建快照: {name}")
        return snapshot_path
    
    def rollback(self, snapshot_name: str):
        """
        回滚到指定快照
        
        Args:
            snapshot_name: 快照名称
        """
        snapshot_path = os.path.join(self.snapshot_dir, snapshot_name)
        if not os.path.exists(snapshot_path):
            raise ValueError(f"快照不存在: {snapshot_name}")
        
        # 备份当前状态
        backup_name = f"pre_rollback_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.create_snapshot(backup_name)
        
        # 回滚
        for item in os.listdir(self.kb_path):
            item_path = os.path.join(self.kb_path, item)
            if item != ".snapshots":
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)
        
        for item in os.listdir(snapshot_path):
            src = os.path.join(snapshot_path, item)
            dst = os.path.join(self.kb_path, item)
            if os.path.isdir(src):
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
        
        print(f"[Snapshot] 已回滚到: {snapshot_name}")
    
    def list_snapshots(self) -> List[Dict]:
        """列出所有快照"""
        snapshots = []
        for name in sorted(os.listdir(self.snapshot_dir), reverse=True):
            snap_path = os.path.join(self.snapshot_dir, name)
            if os.path.isdir(snap_path):
                snapshots.append({
                    "name": name,
                    "created_at": datetime.fromtimestamp(
                        os.path.getctime(snap_path)
                    ).isoformat(),
                    "size": self._get_dir_size(snap_path)
                })
        return snapshots
    
    def _get_dir_size(self, path: str) -> int:
        """计算目录大小"""
        total = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total += os.path.getsize(fp)
        return total
```

---

## 7.6 新鲜度治理

### 7.6.1 过期检测与更新提醒

知识的新鲜度直接影响 RAG 输出的事实准确性：

```python
from datetime import datetime, timedelta
from typing import Optional
from enum import Enum

class FreshnessLevel(Enum):
    """新鲜度等级"""
    FRESH = "fresh"           # 最新
    RECENT = "recent"         # 较新
    AGING = "aging"           # 趋旧
    STALE = "stale"           # 过时
    EXPIRED = "expired"       # 过期


class FreshnessManager:
    """新鲜度管理器"""
    
    # 不同类型文档的 TTL（天）
    DEFAULT_TTL = {
        "guideline": 365,      # 指南：1年
        "manual": 180,         # 手册：6个月
        "report": 90,          # 报告：3个月
        "specification": 180,  # 规格：6个月
        "tutorial": 365,       # 教程：1年
        "faq": 90,             # FAQ：3个月
        "changelog": 30,       # 变更日志：1个月
        "draft": 30,           # 草稿：1个月
        "meeting_notes": 60,   # 会议记录：2个月
        "other": 180,          # 其他：6个月
    }
    
    def __init__(self, ttl_overrides: Dict[str, int] = None):
        self.ttl_config = {**self.DEFAULT_TTL, **(ttl_overrides or {})}
    
    def get_freshness(self, doc_type: str,
                      last_modified: datetime,
                      created_at: datetime = None) -> FreshnessLevel:
        """
        评估文档新鲜度
        
        Args:
            doc_type: 文档类型
            last_modified: 最后修改时间
            created_at: 创建时间
            
        Returns:
            新鲜度等级
        """
        ttl_days = self.ttl_config.get(doc_type, 180)
        now = datetime.now()
        age_days = (now - last_modified).days
        
        if age_days <= ttl_days * 0.3:
            return FreshnessLevel.FRESH
        elif age_days <= ttl_days * 0.6:
            return FreshnessLevel.RECENT
        elif age_days <= ttl_days * 0.9:
            return FreshnessLevel.AGING
        elif age_days <= ttl_days:
            return FreshnessLevel.STALE
        else:
            return FreshnessLevel.EXPIRED
    
    def needs_update(self, doc_type: str,
                     last_modified: datetime,
                     warning_days: int = 30) -> bool:
        """
        检查文档是否需要更新
        
        Args:
            doc_type: 文档类型
            last_modified: 最后修改时间
            warning_days: 提前警告天数
            
        Returns:
            是否需要更新
        """
        ttl_days = self.ttl_config.get(doc_type, 180)
        age_days = (datetime.now() - last_modified).days
        return age_days >= ttl_days - warning_days
    
    def get_expiration_date(self, doc_type: str,
                            last_modified: datetime) -> datetime:
        """计算文档过期日期"""
        ttl_days = self.ttl_config.get(doc_type, 180)
        return last_modified + timedelta(days=ttl_days)
    
    def freshness_score(self, doc_type: str,
                        last_modified: datetime) -> float:
        """
        计算新鲜度分数（0-1，越高越新鲜）
        
        用于检索排序中的新鲜度加权
        """
        ttl_days = self.ttl_config.get(doc_type, 180)
        age_days = (datetime.now() - last_modified).days
        
        if age_days >= ttl_days:
            return 0.0
        
        # 指数衰减
        import math
        return math.exp(-3 * age_days / ttl_days)
```

### 7.6.2 置信度评分

结合新鲜度、权威性、质量等多维度计算文档的置信度：

```python
@dataclass
class DocumentConfidence:
    """文档置信度"""
    doc_id: str
    freshness_score: float       # 新鲜度 0-1
    authority_score: float       # 权威性 0-1
    quality_score: float         # 质量 0-1
    completeness_score: float    # 完整性 0-1
    overall_score: float         # 综合得分 0-1


class ConfidenceScorer:
    """置信度评分器"""
    
    def __init__(self):
        self.freshness = FreshnessManager()
        
        # 作者权威性权重
        self.author_weights = {
            "admin": 1.0,
            "manager": 0.9,
            "editor": 0.8,
            "contributor": 0.6,
        }
        
        # 文档类型权威性
        self.doc_type_authority = {
            "guideline": 1.0,
            "specification": 0.95,
            "manual": 0.9,
            "report": 0.8,
            "tutorial": 0.7,
            "faq": 0.6,
            "meeting_notes": 0.4,
            "draft": 0.3,
        }
    
    def score(self, doc: Dict) -> DocumentConfidence:
        """
        计算文档的综合置信度
        
        Args:
            doc: 文档信息，包含 type, author_role, last_modified, 
                 review_status, completeness 等字段
        """
        # 1. 新鲜度
        freshness = self.freshness.freshness_score(
            doc.get("type", "other"),
            doc.get("last_modified", datetime.now())
        )
        
        # 2. 权威性
        author_role = doc.get("author_role", "contributor")
        doc_type = doc.get("type", "other")
        
        authority = (
            self.author_weights.get(author_role, 0.5) *
            self.doc_type_authority.get(doc_type, 0.5)
        )
        
        # 3. 质量
        review_status = doc.get("review_status", "unreviewed")
        quality_map = {
            "approved": 1.0,
            "reviewed": 0.8,
            "pending_review": 0.5,
            "unreviewed": 0.3,
            "rejected": 0.0
        }
        quality = quality_map.get(review_status, 0.3)
        
        # 4. 完整性
        completeness = doc.get("completeness", 0.5)
        
        # 5. 综合得分（加权平均）
        weights = {"freshness": 0.25, "authority": 0.25, 
                   "quality": 0.3, "completeness": 0.2}
        
        overall = (
            weights["freshness"] * freshness +
            weights["authority"] * authority +
            weights["quality"] * quality +
            weights["completeness"] * completeness
        )
        
        return DocumentConfidence(
            doc_id=doc.get("doc_id", ""),
            freshness_score=freshness,
            authority_score=authority,
            quality_score=quality,
            completeness_score=completeness,
            overall_score=overall
        )
    
    def rerank_by_confidence(self, docs: List[Dict]) -> List[Dict]:
        """根据置信度对检索结果重排序"""
        for doc in docs:
            confidence = self.score(doc)
            doc["confidence_score"] = confidence.overall_score
        
        return sorted(docs, key=lambda x: x["confidence_score"], reverse=True)
```

### 7.6.3 冷热数据分离

```python
class HotColdDataManager:
    """冷热数据分离管理器"""
    
    def __init__(self, hot_threshold_days: int = 30,
                 cold_threshold_days: int = 180):
        """
        Args:
            hot_threshold_days: 热数据阈值（近期访问过的天数）
            cold_threshold_days: 冷数据阈值（超过此天数未访问）
        """
        self.hot_threshold = hot_threshold_days
        self.cold_threshold = cold_threshold_days
    
    def classify_document(self, doc: Dict) -> str:
        """
        将文档分类为热/温/冷数据
        
        Args:
            doc: 文档信息，包含 last_access 字段
            
        Returns:
            "hot" | "warm" | "cold"
        """
        last_access = doc.get("last_access")
        if not last_access:
            return "warm"  # 未知访问时间默认为温数据
        
        days_since_access = (datetime.now() - last_access).days
        
        if days_since_access <= self.hot_threshold:
            return "hot"
        elif days_since_access <= self.cold_threshold:
            return "warm"
        else:
            return "cold"
    
    def optimize_storage(self, documents: List[Dict]) -> Dict:
        """
        根据冷热分类优化存储
        
        Returns:
            {"hot": [hot_docs], "warm": [warm_docs], "cold": [cold_docs]}
        """
        classified = {"hot": [], "warm": [], "cold": []}
        
        for doc in documents:
            category = self.classify_document(doc)
            classified[category].append(doc)
        
        # 冷数据建议：降维存储（降低向量维度或移到低成本存储）
        print(f"[Storage] 热数据: {len(classified['hot'])} 条")
        print(f"[Storage] 温数据: {len(classified['warm'])} 条")
        print(f"[Storage] 冷数据: {len(classified['cold'])} 条")
        
        return classified
```

---

## 7.7 知识体系管理最佳实践

### 7.7.1 实施路线图

| 阶段 | 目标 | 关键举措 |
|------|------|---------|
| 第一阶段 | 基础分类 | 建立领域分类树，实现文档类型自动识别 |
| 第二阶段 | 标签体系 | 实现自动标签生成，建立标签继承规则 |
| 第三阶段 | 权限管理 | 部署 RBAC，实现检索时权限过滤 |
| 第四阶段 | 版本管理 | 实现文档版本追踪和快照回滚 |
| 第五阶段 | 新鲜度治理 | 部署过期检测和置信度评分 |
| 第六阶段 | 持续优化 | 冷热分离，A/B 测试分类效果 |

### 7.7.2 常见问题

| 问题 | 现象 | 解决方案 |
|------|------|---------|
| 过度分类 | 分类太细导致检索不到 | 保持分类层级不超过 3 层，优先用标签 |
| 标签膨胀 | 标签数量失控 | 定期合并相似标签，建立标签同义词表 |
| 权限黑洞 | 权限过滤太严格导致零结果 | 默认公开 + 按需设密，而非默认保密 |
| 版本混乱 | 同一文档多版本共存 | 明确版本号规范，强制唯一活跃版本 |
| 新鲜度焦虑 | 频繁触发过期警告 | 合理设置 TTL，根据文档类型差异化配置 |

---

## 本章小结

知识体系管理是 RAG 系统从"能查"到"查得好"的关键基础设施。本章涵盖了知识分类（领域分类树、文档类型、层次+标签分类）、标签系统（自动生成、继承传播、多维度标签）、权限管理（文档级 ACL、RBAC 模型、检索时权限过滤）、版本管理（版本追踪、快照回滚）以及新鲜度治理（过期检测、置信度评分、冷热分离）五大核心模块。

核心原则：分类不要超过 3 层深，标签不怕多但要有合并机制，权限默认公开，新鲜度按类型差异化配置。一个好的知识管理体系能让检索系统在文档量增长 10 倍的情况下，检索质量不下降。
