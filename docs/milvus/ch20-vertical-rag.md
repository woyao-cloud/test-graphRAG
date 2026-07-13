# 第20章 垂直领域RAG实战

通用RAG方案可以满足大多数场景的知识库问答需求，但在特定垂直领域（如医疗、法律、政务、教育、电商）中，数据的专业性、结构的复杂性和检索的精准度要求远高于通用场景。本章将深入剖析四个典型垂直领域的RAG实战方案，涵盖数据建模、检索策略和领域特定的优化技巧。

## 20.1 政务/法律知识库RAG

### 20.1.1 领域特点与挑战

政务和法律领域的知识库具有以下鲜明特点：

- **条文精确性要求极高**：法律法规的引用必须准确无误，语义近似不能替代精确匹配。
- **版本管理复杂**：法律法规会修订、废止，知识库必须区分不同版本。
- **层级结构清晰**：法律条文具有"编-章-节-条-款-项"的严格层级。
- **检索需要多维过滤**：按效力级别、发布机关、施行日期、主题分类等多维度筛选。

### 20.1.2 数据模型设计

```python
from pymilvus import MilvusClient, DataType

COLLECTION_NAME = "legal_kb"
DIM = 768  # 使用 BGE-large 等专业 Embedding 模型

def create_legal_collection(client):
    schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False)

    schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=DIM)
    schema.add_field("title", DataType.VARCHAR, max_length=512)       # 法规名称
    schema.add_field("content", DataType.VARCHAR, max_length=4096)    # 条文内容
    schema.add_field("law_name", DataType.VARCHAR, max_length=256)    # 所属法规
    schema.add_field("chapter", DataType.VARCHAR, max_length=128)     # 章
    schema.add_field("article_no", DataType.VARCHAR, max_length=32)   # 条号
    schema.add_field("effective_date", DataType.VARCHAR, max_length=32)  # 施行日期
    schema.add_field("issuing_authority", DataType.VARCHAR, max_length=128)  # 发布机关
    schema.add_field("status", DataType.VARCHAR, max_length=16)       # 现行有效/已修订/已废止
    schema.add_field("category", DataType.VARCHAR, max_length=64)     # 法律/行政法规/司法解释

    index_params = MilvusClient.prepare_index_params()
    index_params.add_index("vector", index_type="IVF_FLAT", metric_type="IP", params={"nlist": 256})

    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
    )
```

### 20.1.3 精准条文检索策略

法律条文检索需要"向量语义检索+精确过滤"的混合策略：

```python
def legal_search(client, query, filters=None, top_k=5):
    """法律条文精准检索"""
    q_vec = generate_embedding(query, model="bge-large-zh")

    # 构建过滤条件
    filter_expr = 'status == "现行有效"'
    if filters:
        if "law_name" in filters:
            filter_expr += f' and law_name == "{filters["law_name"]}"'
        if "category" in filters:
            filter_expr += f' and category == "{filters["category"]}"'
        if "issuing_authority" in filters:
            filter_expr += f' and issuing_authority == "{filters["issuing_authority"]}"'

    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[q_vec],
        limit=top_k,
        filter=filter_expr,
        output_fields=["title", "content", "law_name", "article_no", "status"],
    )
    return results[0] if results else []
```

### 20.1.4 引用溯源

法律RAG与通用RAG的一个关键区别是：必须提供引用来源，并且引用必须是精确的条文编号：

```python
def format_legal_answer(results, llm_response):
    """格式化法律问答结果，附带精确引用"""
    references = []
    for r in results[:3]:
        e = r["entity"]
        ref = f"{e['law_name']} 第{e['article_no']}条（{e['status']}）"
        references.append(ref)

    return {
        "answer": llm_response,
        "references": references,
        "disclaimer": "以上内容仅供参考，不构成法律意见。如需专业法律服务，请咨询执业律师。",
    }
```

## 20.2 医疗文献RAG

### 20.2.1 领域特点与挑战

医疗文献RAG面临的核心挑战包括：

- **专业术语密集**：疾病名称、药物名称、基因符号等专业词汇繁多。
- **知识更新快速**：新药上市、治疗方案更新频繁。
- **安全性要求高**：错误的医疗建议可能导致严重后果。
- **数据多源异构**：临床指南、药品说明书、医学文献、病例报告等格式各异。

### 20.2.2 数据模型设计

医疗文献检索需要按疾病名称、药物名称、文献类别等多个维度组织数据：

```python
def create_medical_collection(client):
    """创建医疗文献集合"""
    COLLECTION_NAME = "medical_kb"
    DIM = 768

    schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=DIM)
    schema.add_field("title", DataType.VARCHAR, max_length=256)       # 文献标题
    schema.add_field("content", DataType.VARCHAR, max_length=4096)    # 文献摘要/内容
    schema.add_field("disease", DataType.VARCHAR, max_length=128)     # 相关疾病
    schema.add_field("drug", DataType.VARCHAR, max_length=128)        # 相关药物
    schema.add_field("category", DataType.VARCHAR, max_length=64)     # 治疗指南/专家共识/研究进展
    schema.add_field("publication_date", DataType.VARCHAR, max_length=32)  # 发表日期
    schema.add_field("source", DataType.VARCHAR, max_length=256)      # 来源期刊/机构

    index_params = MilvusClient.prepare_index_params()
    index_params.add_index("vector", index_type="IVF_FLAT", metric_type="IP", params={"nlist": 128})

    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
    )
```

### 20.2.3 疾病导向检索与药物导向检索

医疗RAG的典型检索模式有两种：

```python
def search_by_disease(client, disease, query="", top_k=5):
    """按疾病名称过滤 + 向量语义检索"""
    q_vec = generate_embedding(query) if query else generate_embedding(disease)
    results = client.search(
        collection_name="medical_kb",
        data=[q_vec],
        limit=top_k,
        filter=f'disease == "{disease}"',
        output_fields=["title", "content", "disease", "drug", "category"],
    )
    return results[0] if results else []

def search_by_drug(client, drug, query="", top_k=5):
    """按药物名称过滤 + 向量语义检索"""
    q_vec = generate_embedding(query) if query else generate_embedding(drug)
    results = client.search(
        collection_name="medical_kb",
        data=[q_vec],
        limit=top_k,
        filter=f'drug == "{drug}"',
        output_fields=["title", "content", "disease", "drug", "category"],
    )
    return results[0] if results else []
```

### 20.2.4 治疗方案推荐

基于检索结果，可以自动生成结构化的治疗建议：

```python
def generate_treatment_recommendation(client, disease):
    """生成结构化治疗推荐"""
    results = search_by_disease(client, disease, "治疗")

    drugs = set()
    categories = set()
    guidelines = []

    for r in results:
        e = r["entity"]
        drugs.add(e["drug"])
        categories.add(e["category"])
        guidelines.append({
            "category": e["category"],
            "title": e["title"],
            "drug": e["drug"],
            "summary": e["content"][:200],
        })

    return {
        "disease": disease,
        "related_drugs": list(drugs),
        "document_categories": list(categories),
        "guidelines": guidelines,
    }
```

## 20.3 电商商品知识库RAG

### 20.3.1 领域特点与挑战

电商商品知识库的典型应用场景包括智能客服、商品推荐和售前咨询。其挑战在于：

- **商品属性维度多**：品类、品牌、价格、规格、评价等属性复杂。
- **查询意图多样**：用户可能按功能、场景、价格区间等不同维度提问。
- **数据更新频繁**：商品上下架、价格变动、库存变化实时性要求高。
- **多模态需求**：商品图片、参数表、使用说明等多模态数据。

### 20.3.2 数据模型与检索策略

```python
def create_ecommerce_collection(client):
    """创建电商商品集合"""
    schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=True)

    schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=768)
    schema.add_field("product_name", DataType.VARCHAR, max_length=256)    # 商品名称
    schema.add_field("description", DataType.VARCHAR, max_length=2048)    # 商品描述
    schema.add_field("category", DataType.VARCHAR, max_length=128)        # 类目
    schema.add_field("brand", DataType.VARCHAR, max_length=128)           # 品牌
    schema.add_field("price", DataType.FLOAT)                             # 价格
    schema.add_field("rating", DataType.FLOAT)                            # 评分
    schema.add_field("tags", DataType.VARCHAR, max_length=512)            # 标签
    schema.add_field("status", DataType.VARCHAR, max_length=16)           # 在售/下架

    # 标量字段也创建索引，加速过滤
    index_params = MilvusClient.prepare_index_params()
    index_params.add_index("vector", index_type="IVF_SQ8", metric_type="IP", params={"nlist": 256})

    client.create_collection(
        collection_name="ecommerce_kb",
        schema=schema,
        index_params=index_params,
    )

def ecommerce_search(client, query, price_range=None, category=None, top_k=10):
    """电商商品多维检索"""
    q_vec = generate_embedding(query, model="bge-large-zh")

    filters = ['status == "在售"']
    if price_range:
        filters.append(f"price >= {price_range[0]} and price <= {price_range[1]}")
    if category:
        filters.append(f'category == "{category}"')

    results = client.search(
        collection_name="ecommerce_kb",
        data=[q_vec],
        limit=top_k,
        filter=" and ".join(filters),
        output_fields=["product_name", "description", "price", "brand", "rating"],
    )
    return results[0] if results else []
```

## 20.4 教育文档RAG

### 20.4.1 领域特点与挑战

教育领域的RAG应用包括教材问答、题库检索、知识点讲解等场景：

- **知识点体系化**：教材内容有明确的章节结构和知识递进关系。
- **题目与答案分离**：题库需要区分题目和答案，检索时只匹配题目。
- **难度分层**：同一知识点可能需要不同难度的讲解。
- **图文混排**：教材中常包含图表、公式等非纯文本内容。

### 20.4.2 智能切片策略

教育文档的切片策略需要与教材的章节结构对齐：

```python
def smart_chunk_educational_doc(doc):
    """基于章节结构的教育文档智能切片"""
    import re

    chunks = []
    # 按"第X章"和"第X节"分割
    chapters = re.split(r'(第[一二三四五六七八九十]+章\s+.*?)\n', doc)

    current_chapter = None
    for part in chapters:
        if part.startswith("第") and "章" in part:
            current_chapter = part.strip()
        elif current_chapter:
            # 进一步按"第X节"分割
            sections = re.split(r'(第[一二三四五六七八九十]+节\s+.*?)\n', part)
            current_section = None
            for sec in sections:
                if sec.startswith("第") and "节" in sec:
                    current_section = sec.strip()
                elif sec.strip():
                    chunks.append({
                        "chapter": current_chapter,
                        "section": current_section or "",
                        "content": sec.strip(),
                    })
    return chunks
```

### 20.4.3 数据模型

```python
def create_education_collection(client):
    """创建教育文档集合"""
    schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False)

    schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=768)
    schema.add_field("subject", DataType.VARCHAR, max_length=64)       # 科目：数学/物理/语文
    schema.add_field("grade", DataType.VARCHAR, max_length=32)         # 年级
    schema.add_field("chapter", DataType.VARCHAR, max_length=256)      # 章节
    schema.add_field("knowledge_point", DataType.VARCHAR, max_length=256)  # 知识点
    schema.add_field("content", DataType.VARCHAR, max_length=4096)     # 内容
    schema.add_field("difficulty", DataType.INT32)                     # 难度等级 1-5
    schema.add_field("doc_type", DataType.VARCHAR, max_length=32)      # 教材/习题/解析

    index_params = MilvusClient.prepare_index_params()
    index_params.add_index("vector", index_type="HNSW", metric_type="IP",
                           params={"M": 16, "efConstruction": 200})

    client.create_collection(
        collection_name="education_kb",
        schema=schema,
        index_params=index_params,
    )
```

### 20.4.4 知识点精准检索

教育场景中，按知识点和难度进行过滤检索是高频操作：

```python
def search_by_knowledge_point(client, knowledge_point, grade=None,
                               difficulty=None, top_k=5):
    """按知识点检索，支持年级和难度过滤"""
    q_vec = generate_embedding(knowledge_point)

    filters = [f'knowledge_point == "{knowledge_point}"']
    if grade:
        filters.append(f'grade == "{grade}"')
    if difficulty:
        filters.append(f"difficulty <= {difficulty}")

    results = client.search(
        collection_name="education_kb",
        data=[q_vec],
        limit=top_k,
        filter=" and ".join(filters),
        output_fields=["content", "chapter", "difficulty", "doc_type"],
    )
    return results[0] if results else []
```

## 20.5 垂直领域RAG通用优化策略

### 20.5.1 领域Embedding模型选型

| 领域 | 推荐Embedding模型 | 说明 |
|------|-----------------|------|
| 法律 | BAAI/bge-large-zh-v1.5 | 中文法律文本效果优秀 |
| 医疗 | BAAI/bge-large-zh-v1.5 / PubMedBERT | 医学专业词汇理解好 |
| 电商 | BAAI/bge-base-zh-v1.5 | 平衡效果与速度 |
| 教育 | BAAI/bge-large-zh-v1.5 | 知识点语义理解准确 |

### 20.5.2 标量过滤与分区策略

垂直领域的结构化数据通常有多个关键维度，利用 Milvus 的标量字段过滤可以显著提升检索精准度：

- **法律**：按效力级别、发布机关、施行日期过滤
- **医疗**：按疾病名称、药物名称、文献类别过滤
- **电商**：按品类、品牌、价格区间过滤
- **教育**：按科目、年级、难度等级过滤

### 20.5.3 领域特有后处理

垂直领域RAG的检索结果需要额外的后处理逻辑：

- **法律**：去重相同条文的不同版本，保留最新有效版本
- **医疗**：检查文献时效性，优先引用近3年的指南
- **电商**：去重同款商品，按综合排序（评分+销量）
- **教育**：按难度递进排列，先基础后进阶

## 本章小结

本章深入剖析了四个典型垂直领域的RAG实战方案：

1. **政务/法律RAG**：强调条文精确性和引用溯源，采用"向量检索+精确过滤"混合策略。
2. **医疗文献RAG**：聚焦疾病和药物双维度检索，自动生成结构化治疗推荐。
3. **电商商品RAG**：面向智能客服场景，实现多维属性过滤和动态数据更新。
4. **教育文档RAG**：基于教材章节结构的智能切片，支持知识点和难度分层检索。

垂直领域RAG的核心竞争力在于对领域知识的深度理解和针对性的数据建模。通用RAG框架提供了基础能力，而领域特定的数据模型、检索策略和后处理逻辑才是决定系统实际效果的关键。
