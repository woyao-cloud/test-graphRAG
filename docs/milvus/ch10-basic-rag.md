# 第10章: 搭建最简版Milvus-RAG问答系统

## 10.1 引言

在掌握了Milvus的CRUD操作之后，本章将把这些基础操作串联起来，搭建一个完整的、可运行的RAG问答系统。虽然这个系统是最简版本——使用字符频率向量替代深度学习Embedding、使用简单的句子匹配替代大模型生成——但它包含了RAG的全部核心环节：文档处理、向量化、向量存储、语义检索和答案生成。

理解最简版本的RAG系统，有助于读者在后续章节中理解更复杂的框架和优化方案时，清晰地把握每个环节的本质。本章内容对应项目`demos/ch10-basic-rag/main.py`中的完整代码实现。

## 10.2 项目架构设计

### 10.2.1 RAG全链路拆解

一个标准的RAG系统包含以下六个核心环节：

```
用户提问
    │
    ▼
┌─────────────────┐
│  1. 文档加载     │ ← PDF/Word/TXT/网页
└────────┬────────┘
         ▼
┌─────────────────┐
│  2. 文档切片     │ ← 将长文档分割为语义完整的片段
└────────┬────────┘
         ▼
┌─────────────────┐
│  3. Embedding    │ ← 将文本片段转换为向量
└────────┬────────┘
         ▼
┌─────────────────┐
│  4. 向量存储     │ ← Milvus存储向量+原文
└────────┬────────┘
         ▼
┌─────────────────┐
│  5. 语义检索     │ ← 在Milvus中搜索相似向量
└────────┬────────┘
         ▼
┌─────────────────┐
│  6. 答案生成     │ ← 拼接上下文+Prompt→LLM
└────────┬────────┘
         ▼
    最终回答
```

### 10.2.2 本章Demo的技术选型

由于本章Demo的目标是展示RAG原理而非追求生产级效果，因此采用以下轻量级方案：

| 环节 | 生产方案 | 本章Demo方案 |
|------|---------|-------------|
| Embedding | BGE/OpenAI/text2vec | 字符频率向量（TF-IDF风格） |
| 向量数据库 | Milvus集群 | Milvus单机/Embed模式 |
| 相似度算法 | IP/余弦相似度 | 余弦相似度 |
| 答案生成 | GPT-4/Claude/DeepSeek | 关键词重叠匹配 |

这种简化方案的好处是：无需任何外部模型依赖，安装pymilvus即可运行，让读者专注于理解RAG的流程逻辑。

## 10.3 文档处理实战

### 10.3.1 文档加载

在实际RAG项目中，文档加载是第一步。Python生态中有丰富的文档解析库：

```python
# PDF解析
import PyPDF2
def load_pdf(file_path):
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        return [page.extract_text() for page in reader.pages]

# Word解析
import docx
def load_docx(file_path):
    doc = docx.Document(file_path)
    return [p.text for p in doc.paragraphs if p.text.strip()]

# TXT解析
def load_txt(file_path, encoding="utf-8"):
    with open(file_path, "r", encoding=encoding) as f:
        return f.read()

# 统一加载接口
def load_document(file_path):
    if file_path.endswith(".pdf"):
        return "\n".join(load_pdf(file_path))
    elif file_path.endswith(".docx"):
        return "\n".join(load_docx(file_path))
    else:
        return load_txt(file_path)
```

### 10.3.2 文档切片策略

文档切片是RAG系统中影响检索效果的关键环节。切得太碎会丢失上下文语义，切得太整会降低检索精度。

**常见的切片策略**：

```python
def chunk_by_size(text, chunk_size=256, overlap=32):
    """固定长度切片，带重叠"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def chunk_by_paragraph(text):
    """按段落切片（以换行符为界）"""
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    return paragraphs

def chunk_by_sentence(text, max_chars=512):
    """按句子切片，合并短句到接近max_chars"""
    import re
    sentences = re.split(r'[。！？\n]', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    chunks = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) < max_chars:
            current += sent + "。"
        else:
            if current:
                chunks.append(current)
            current = sent + "。"
    if current:
        chunks.append(current)
    return chunks
```

**切片参数的经验值**：
- Embedding模型为BERT类（512 token限制）：切片长度256-512字符。
- Embedding模型为text2vec-large（1024 token限制）：切片长度512-1024字符。
- 重叠窗口：通常设为切片长度的10%-20%。
- 中文文档：建议按句子或段落切片，避免从中间截断语义。

### 10.3.3 文档去重

知识库中可能存在重复文档，需要去重处理：

```python
def deduplicate(chunks):
    """基于文本相似度的去重"""
    seen = set()
    unique = []
    for chunk in chunks:
        # 使用文本hash作为去重依据（简单方案）
        key = hash(chunk[:100])  # 取前100字符做hash
        if key not in seen:
            seen.add(key)
            unique.append(chunk)
    return unique
```

## 10.4 Embedding模型接入

### 10.4.1 字符频率向量（本章Demo方案）

本章Demo使用字符频率向量作为Embedding的简化替代。其核心思想是：统计文档中每个字符的出现频率，构成一个高维稀疏向量。虽然这种方案无法捕捉语义信息（"苹果"和"香蕉"的向量距离与"苹果"和"汽车"的向量距离没有本质区别），但它足以展示向量检索的完整流程。

```python
import math
from collections import Counter

def build_vocab(docs: list[str]) -> list[str]:
    """从文档集合中构建字符词表"""
    chars: set[str] = set()
    for doc in docs:
        chars.update(doc)
    return sorted(chars)

def char_freq_vector(text: str, vocab: list[str]) -> list[float]:
    """计算文本的字符频率向量（TF-IDF风格）"""
    text_len = len(text) if len(text) > 0 else 1
    counter = Counter(text)
    
    vector = []
    for ch in vocab:
        tf = counter.get(ch, 0) / text_len
        vector.append(tf)  # 简化版：仅使用TF
    
    # L2归一化
    norm = math.sqrt(sum(v * v for v in vector))
    if norm > 0:
        vector = [v / norm for v in vector]
    return vector
```

### 10.4.2 生产级Embedding模型接入

在实际RAG项目中，需要使用深度学习模型生成语义向量。以下是几种主流方案：

```python
# 方案一：HuggingFace开源模型（BGE）
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
vector = model.encode("你的文本内容").tolist()
# 向量维度：384（bge-small）或 768（bge-large）

# 方案二：OpenAI Embedding API
import openai
response = openai.embeddings.create(
    model="text-embedding-3-small",
    input="你的文本内容",
)
vector = response.data[0].embedding
# 向量维度：1536

# 方案三：智谱GLM Embedding
from zhipuai import ZhipuAI
client = ZhipuAI(api_key="your-key")
response = client.embeddings.create(
    model="embedding-2",
    input="你的文本内容",
)
vector = response.data[0].embedding
# 向量维度：1024
```

**Embedding模型选型建议**：
- 中文RAG场景首选BGE系列（BAAI/bge-large-zh-v1.5），在中文语义理解上表现优异。
- 英文或多语言场景选择OpenAI text-embedding-3-small，性价比最高。
- 离线部署场景选择text2vec-base-chinese，模型体积小，推理速度快。

## 10.5 向量入库与索引构建

### 10.5.1 创建集合并插入数据

将切分后的文档片段向量化后写入Milvus：

```python
from pymilvus import MilvusClient

# 连接Milvus
client = MilvusClient(uri="http://localhost:19530")
# 或使用Embed模式
# client = MilvusClient(uri="./rag_demo.db")

collection_name = "basic_rag_demo"
dim = len(vocab)  # 字符词表大小

# 创建集合
client.create_collection(
    collection_name=collection_name,
    dimension=dim,
    auto_id=False,
)

# 准备数据
data = [
    {
        "id": i,
        "vector": doc_vectors[i],
        "text": DOCUMENTS[i],
    }
    for i in range(len(DOCUMENTS))
]

# 批量插入
client.insert(collection_name=collection_name, data=data)
print(f"已插入 {len(data)} 条文档片段")
```

### 10.5.2 自动索引

使用`MilvusClient`创建集合时，如果未指定索引参数，Milvus会自动创建一个默认的FLAT索引。对于数据量小于10万条的小型RAG知识库，FLAT索引的性能完全可接受。

如果需要自定义索引：

```python
index_params = MilvusClient.prepare_index_params()
index_params.add_index(
    field_name="vector",
    index_type="IVF_FLAT",
    metric_type="IP",
    params={"nlist": 128},
)
client.create_index(collection_name, index_params)
client.load_collection(collection_name)
```

## 10.6 语义检索与Prompt工程

### 10.6.1 语义检索

用户提问时，将问题向量化后在Milvus中搜索最相似的文档片段：

```python
# 用户提问
query = "哪些药物可以治疗炎症和疼痛？"

# 将查询文本向量化
query_vector = char_freq_vector(query, vocab)

# 在Milvus中检索
search_result = client.search(
    collection_name=collection_name,
    data=[query_vector],
    limit=3,
    output_fields=["text"],
)

# 提取检索到的文本片段
retrieved_texts = []
for i, hit in enumerate(search_result[0]):
    text = hit["entity"]["text"]
    score = hit["distance"]
    retrieved_texts.append(text)
    print(f"  #{i+1} (相似度={score:.4f}): {text[:50]}...")
```

### 10.6.2 简单答案生成

本章Demo使用基于关键词重叠的句子匹配来生成答案，模拟LLM的"根据上下文回答问题"：

```python
def generate_answer(query: str, contexts: list[str]) -> str:
    """根据检索到的上下文生成答案"""
    query_chars = set(query)
    scored_sentences = []
    
    for text in contexts:
        for sentence in text.replace("。", "。|").split("|"):
            sentence = sentence.strip()
            if not sentence:
                continue
            sentence_chars = set(sentence)
            overlap = len(query_chars & sentence_chars)
            score = overlap / len(query_chars) if query_chars else 0
            scored_sentences.append((score, sentence))
    
    scored_sentences.sort(key=lambda x: -x[0])
    top = [s for _, s in scored_sentences[:3] if _ > 0]
    
    if not top:
        return "无法根据已有信息生成回答。"
    
    return "根据检索到的信息：" + "；".join(top) + "。"
```

### 10.6.3 生产级Prompt工程

在实际RAG系统中，答案生成由LLM完成，Prompt模板的设计至关重要：

```python
RAG_PROMPT_TEMPLATE = """你是一个知识问答助手。请根据以下检索到的上下文信息，回答用户的问题。

注意事项：
1. 如果上下文信息不足以回答问题，请如实回答"无法从已有知识库中找到相关信息"。
2. 不要编造上下文中没有的信息。
3. 请使用中文回答，语言简洁准确。

检索到的上下文：
{context}

用户问题：{question}

回答："""

def build_rag_prompt(question: str, contexts: list[str]) -> str:
    """构建RAG问答的Prompt"""
    context_text = "\n\n".join([f"文档{i+1}：{c}" for i, c in enumerate(contexts)])
    return RAG_PROMPT_TEMPLATE.format(context=context_text, question=question)

# 调用LLM（以OpenAI为例）
import openai
response = openai.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": build_rag_prompt(query, retrieved_texts)}],
)
answer = response.choices[0].message.content
```

## 10.7 调试：召回异常与问答幻觉

### 10.7.1 召回异常的排查思路

RAG系统中最常见的问题是检索结果与问题不相关。排查步骤如下：

**第一步：检查原始数据是否正确入库**

```python
# 查看集合中的数据是否完整
count = client.query(collection_name, output_fields=["count(*)"])
print(f"数据总量: {count}")

# 随机抽取几条查看内容
samples = client.query(collection_name, limit=5, output_fields=["id", "text"])
for s in samples:
    print(f"  id={s['id']}: {s['text'][:100]}...")
```

**第二步：检查向量质量**

```python
# 查看向量的统计信息
import numpy as np
sample = client.query(collection_name, limit=10, output_fields=["vector"])
vectors = [s["vector"] for s in sample]
print(f"向量维度: {len(vectors[0])}")
print(f"向量范数分布: min={min(np.linalg.norm(v) for v in vectors):.4f}, "
      f"max={max(np.linalg.norm(v) for v in vectors):.4f}")
```

如果向量的范数差异过大（未归一化），会导致余弦相似度或IP度量结果不稳定。

**第三步：检查检索参数**

```python
# 增大TopK和nprobe看是否能召回更多结果
results = client.search(
    collection_name=collection_name,
    data=[query_vector],
    limit=20,  # 增大TopK
    search_params={"params": {"nprobe": 64}},  # 增大检索深度
    output_fields=["text"],
)
print(f"增大TopK后召回数: {len(results[0])}")
```

**第四步：检查索引状态**

```python
# 查看索引信息
index_info = client.list_indexes(collection_name)
print(f"索引信息: {index_info}")
```

### 10.7.2 问答幻觉的排查

当LLM生成的答案出现幻觉（编造不存在的信息）时，排查方向如下：

1. **上下文不相关**：检查检索返回的TopK内容是否与问题相关。如果不相关，问题出在检索环节。
2. **上下文不足**：TopK太小导致信息不全，LLM被迫"脑补"。尝试增大TopK。
3. **Prompt指令不清晰**：Prompt中没有强调"不能编造"，LLM会倾向于给出看似合理的答案。
4. **LLM自身问题**：即使是GPT-4在面对模糊问题或缺失信息时也可能产生幻觉。可以通过温度参数（temperature=0）降低随机性。

### 10.7.3 调试利器：Attu可视化工具

在调试RAG系统时，Attu是非常高效的可视化工具。通过Attu可以直接查看：

- 集合中存储的文档内容和向量。
- 手动执行检索，对比应用层的检索结果。
- 索引构建状态和进度。

关于Attu的详细部署和使用方法，请参考第8章的8.6节。

## 10.8 本章小结

本章从零开始搭建了一个最简版的Milvus-RAG问答系统，涵盖了文档处理、Embedding、向量存储、语义检索和答案生成五大环节。虽然使用了字符频率向量这种极简方案替代深度学习Embedding，但RAG的核心链路和关键问题（切片策略、检索参数、召回异常排查、幻觉处理）在真实生产环境中同样存在。

通过本章的学习，读者应该能够：
1. 理解RAG系统的完整工作流程和每个环节的作用。
2. 掌握基本的文档切片策略和Embedding模型接入方法。
3. 熟练使用MilvusClient进行向量存储和语义检索。
4. 具备基本的RAG系统调试能力。

下一章将在此基础上，引入LangChain、LlamaIndex等主流框架，将RAG系统推向模块化和生产化。
