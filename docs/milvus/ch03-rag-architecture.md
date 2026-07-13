# 第3章：RAG完整架构与Milvus角色定位

## 3.1 标准 RAG 全链路拆解

### 3.1.1 数据预处理阶段

RAG 系统的第一步是构建知识库。这个阶段的质量直接影响后续所有环节的效果，是 RAG 系统的基石。

```python
# 完整的数据预处理流程
import os
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader

class DocumentProcessor:
    """文档预处理流水线"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
        )
    
    def load_documents(self, directory: str) -> List:
        """加载指定目录下的所有文档"""
        loader = DirectoryLoader(
            directory,
            glob="**/*.md",
            show_progress=True
        )
        return loader.load()
    
    def split_documents(self, documents: List) -> List:
        """将长文档切分为语义完整的片段"""
        chunks = self.splitter.split_documents(documents)
        
        # 为每个 chunk 生成唯一 ID
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = f"chunk_{i:06d}"
        
        return chunks

# 预处理流程
processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)
docs = processor.load_documents("./knowledge_base/")
chunks = processor.split_documents(docs)
print(f"原始文档数: {len(docs)}")
print(f"切分后片段数: {len(chunks)}")
```

**切分策略的核心考量**：

- **语义完整性**：切分应在自然断点（段落末尾、句号处）进行，避免打断完整的语义单元。
- **重叠窗口**：相邻片段之间保持一定重叠（overlap），确保边界处的重要信息不会丢失。
- **长度平衡**：片段不宜过短（丢失上下文）也不宜过长（超过 Embedding 模型的最大输入长度）。

### 3.1.2 向量化与入库阶段

切分后的文档片段需要转换为向量并存入向量数据库：

```python
from langchain_milvus import Milvus
from langchain_openai import OpenAIEmbeddings
from pymilvus import connections, CollectionSchema, FieldSchema, DataType

def create_milvus_collection():
    """创建 Milvus 集合并配置索引"""
    # 连接 Milvus
    connections.connect(host="localhost", port="19530")
    
    # 定义集合 schema
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=8192),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536)
    ]
    schema = CollectionSchema(fields, description="RAG 知识库集合")
    
    return schema

def index_documents(chunks: List, collection_name: str = "rag_knowledge_base"):
    """将文档片段向量化并存入 Milvus"""
    vector_store = Milvus(
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
        collection_name=collection_name,
        connection_args={"host": "localhost", "port": "19530"},
        index_params={
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024}
        }
    )
    
    # 批量写入
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    
    vector_store.add_texts(texts=texts, metadatas=metadatas)
    return vector_store
```

**向量化阶段的关键参数**：

| 参数 | 说明 | 推荐值 |
|------|------|-------|
| Embedding 模型 | 决定语义编码质量 | bge-large-zh / text-embedding-3-small |
| 向量维度 | 影响精度和存储 | 768-1536 |
| 索引类型 | 影响检索速度和召回率 | IVF_FLAT / HNSW |
| 相似度度量 | 影响语义匹配效果 | COSINE（文本推荐） |

### 3.1.3 检索与生成阶段

这是 RAG 系统的核心推理链路：

```python
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

class RAGPipeline:
    """完整的 RAG 推理流水线"""
    
    def __init__(self, vector_store: Milvus, llm_model: str = "gpt-4o"):
        self.vector_store = vector_store
        self.llm = ChatOpenAI(model=llm_model, temperature=0)
        
        # 自定义 Prompt 模板
        self.prompt_template = PromptTemplate(
            template="""你是一个专业的问答助手。请基于以下参考信息回答问题。

参考信息：
{context}

问题：{question}

要求：
1. 如果参考信息中包含答案，请直接回答
2. 如果参考信息不包含答案，请明确告知"无法从知识库中找到相关信息"
3. 不要编造信息，不要添加参考信息中没有的内容

答案：""",
            input_variables=["context", "question"]
        )
    
    def query(self, question: str, top_k: int = 5) -> dict:
        """执行完整 RAG 查询"""
        # 1. 检索相关文档
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k}
        )
        
        # 2. 构建 QA Chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",  # 将所有检索结果合并到上下文
            retriever=retriever,
            chain_type_kwargs={"prompt": self.prompt_template},
            return_source_documents=True
        )
        
        # 3. 执行查询
        result = qa_chain.invoke({"query": question})
        return result

# 使用示例
pipeline = RAGPipeline(vector_store=vector_store)
result = pipeline.query("Milvus 支持哪些索引类型？")

print(f"答案: {result['result']}")
print(f"参考来源: {len(result['source_documents'])} 篇文档")
```

### 3.1.4 RAG 全链路流程图

```
用户查询
    │
    ▼
┌─────────────┐     ┌──────────────────┐
│ 查询向量化   │────▶│  Milvus 向量检索  │
│ (Embedding) │     │  (ANN 搜索)       │
└─────────────┘     └────────┬─────────┘
                             │ Top-K 相关文档
                             ▼
┌─────────────┐     ┌──────────────────┐
│ 结果输出     │◀────│  LLM 生成答案     │
│ (Answer)    │     │  (基于上下文)     │
└─────────────┘     └──────────────────┘
```

## 3.2 传统 RAG 痛点

尽管 RAG 的基本流程看似简单，但在生产环境中面临着诸多挑战：

### 3.2.1 检索质量瓶颈

**痛点一：低召回率**

当问题复杂或表述模糊时，简单的向量检索可能遗漏重要信息。例如，问"如何使用 Python 连接 Milvus"，可能只匹配到包含"Python"和"Milvus"的文档，而忽略了直接回答问题的配置示例文档。

**痛点二：相关性排序不准**

Top-K 检索结果中，排序靠前的文档不一定是最相关的。单纯的向量相似度无法完全捕捉查询意图与文档内容的匹配度。

**解决方案**：混合检索（Hybrid Search），将向量检索与 BM25 关键词检索结合，通过 RRF（Reciprocal Rank Fusion）或重排序（Re-Ranker）模型进行二次排序：

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

def create_hybrid_retriever(vector_store, documents):
    """构建混合检索器：向量检索 + 关键词检索"""
    # 向量检索器
    vector_retriever = vector_store.as_retriever(
        search_kwargs={"k": 10}
    )
    
    # BM25 关键词检索器
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 10
    
    # 集成检索器（RRF 融合）
    hybrid_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.5, 0.5]
    )
    
    return hybrid_retriever
```

### 3.2.2 上下文管理难题

**痛点三：上下文窗口溢出**

当检索结果过多时，可能导致 LLM 的上下文窗口溢出。同时，大量不相关的检索结果会稀释有效信息。

**痛点四：信息冗余**

多个检索结果可能包含高度重叠的信息，浪费上下文空间。

**解决方案**：使用压缩检索器（Contextual Compression Retriever），在注入 LLM 之前对检索结果进行过滤和压缩：

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

def create_compression_retriever(vector_store, llm):
    """构建压缩检索器"""
    # 文档压缩器：提取每个文档中最相关的部分
    compressor = LLMChainExtractor.from_llm(llm)
    
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=vector_store.as_retriever(search_kwargs={"k": 10})
    )
    
    return compression_retriever
```

### 3.2.3 延迟与成本

**痛点五：端到端延迟**

RAG 流水线涉及 Embedding 推理、向量检索、LLM 生成三个串行步骤，端到端延迟通常在 2-10 秒。

**痛点六：Token 成本**

频繁调用 LLM API 的 Token 成本随检索文档数量线性增长。

## 3.3 Milvus 在 RAG 中的核心作用

### 3.3.1 高性能向量存储与检索

Milvus 作为 RAG 系统的向量存储层，承担着以下核心职责：

1. **海量向量存储**：支持 PB 级别的向量数据管理，通过分片和分区实现水平扩展。
2. **毫秒级检索**：十亿级数据集上的 ANN 搜索延迟控制在 100ms 以内。
3. **动态 Schema**：支持向量字段与标量字段的混合存储，便于存储文档元数据。

```python
# Milvus 在 RAG 中的核心配置
from pymilvus import MilvusClient, DataType

client = MilvusClient("http://localhost:19530")

# 创建优化的 RAG 集合
client.create_collection(
    collection_name="enterprise_rag",
    dimension=1024,  # 向量维度
    primary_field_name="id",
    vector_field_name="embedding",
    metric_type="COSINE",
    auto_id=True,
    # 存储文档元数据
    schema={
        "fields": [
            {"name": "id", "type": DataType.INT64, "is_primary": True, "auto_id": True},
            {"name": "embedding", "type": DataType.FLOAT_VECTOR, "params": {"dim": 1024}},
            {"name": "text", "type": DataType.VARCHAR, "max_length": 65535},
            {"name": "source", "type": DataType.VARCHAR, "max_length": 500},
            {"name": "category", "type": DataType.VARCHAR, "max_length": 100},
            {"name": "created_at", "type": DataType.INT64}
        ]
    }
)

# 创建 HNSW 索引实现毫秒级检索
client.create_index(
    collection_name="enterprise_rag",
    index_params={
        "metric_type": "COSINE",
        "index_type": "HNSW",
        "params": {
            "M": 16,         # 每个节点的最大连接数
            "efConstruction": 200  # 构建时的搜索范围
        }
    }
)
```

### 3.3.2 标量过滤增强检索

Milvus 支持在向量检索的同时进行标量字段过滤，这是工业级 RAG 系统的关键能力：

```python
def filtered_search(client, query_vector, category: str = None, date_range: tuple = None):
    """带过滤条件的向量检索"""
    filter_expr = ""
    
    if category:
        filter_expr += f'category == "{category}"'
    
    if date_range:
        if filter_expr:
            filter_expr += " and "
        filter_expr += f"created_at >= {date_range[0]} and created_at <= {date_range[1]}"
    
    results = client.search(
        collection_name="enterprise_rag",
        data=[query_vector],
        limit=10,
        filter=filter_expr if filter_expr else None,
        output_fields=["text", "source", "category"]
    )
    
    return results
```

### 3.3.3 增量更新与实时性

Milvus 支持实时的数据插入和删除操作，无需重建索引：

```python
# 实时增量更新
def add_documents_to_knowledge_base(client, texts: list, metadatas: list, embeddings: list):
    """向知识库实时添加新文档"""
    data = []
    for i in range(len(texts)):
        data.append({
            "embedding": embeddings[i],
            "text": texts[i],
            **metadatas[i]
        })
    
    client.insert(
        collection_name="enterprise_rag",
        data=data
    )
    print(f"成功添加 {len(data)} 条文档")
```

## 3.4 简易 RAG vs 工业级 RAG 差距

### 3.4.1 简易 RAG（适合原型开发）

```python
# 简易 RAG 实现（20 行代码）
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class SimpleRAG:
    def __init__(self, embedding_model, llm):
        self.embedding_model = embedding_model
        self.llm = llm
        self.documents = []
        self.embeddings = []
    
    def add_documents(self, texts):
        self.documents.extend(texts)
        self.embeddings.extend(self.embedding_model.encode(texts))
    
    def query(self, question):
        q_emb = self.embedding_model.encode([question])
        sims = cosine_similarity(q_emb, self.embeddings)[0]
        top_indices = sims.argsort()[-3:][::-1]
        context = "\n".join([self.documents[i] for i in top_indices])
        return self.llm.generate(f"Context: {context}\nQ: {question}\nA:")
```

**问题**：内存存储、线性搜索、无标量过滤、无分布式能力。

### 3.4.2 工业级 RAG 核心差距

| 维度 | 简易 RAG | 工业级 RAG |
|------|---------|-----------|
| 数据规模 | 百万级 | 十亿级 |
| 检索延迟 | 秒级 | 毫秒级 |
| 高可用 | 无 | 多副本、故障转移 |
| 增量更新 | 全量重建 | 实时增删改 |
| 安全控制 | 无 | RBAC、数据加密 |
| 监控告警 | 无 | 指标采集、链路追踪 |
| 成本优化 | 无 | 分级存储、冷热分离 |

### 3.4.3 工业级 RAG 架构示例

```python
# 工业级 RAG 架构的关键组件
class EnterpriseRAG:
    """工业级 RAG 系统"""
    
    def __init__(self):
        self.milvus_client = MilvusClient(
            uri="http://milvus-proxy:19530",
            token="your-auth-token"  # RBAC 认证
        )
        self.embedding_model = self._load_embedding_model()
        self.reranker = self._load_reranker()  # 重排序模型
        self.cache = self._init_cache()        # 结果缓存
        self.monitor = self._init_monitoring() # 监控系统
    
    def query(self, question: str, user_id: str = None):
        # 1. 缓存查询（降低成本）
        cache_key = self._generate_cache_key(question, user_id)
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        # 2. 向量化
        q_vector = self.embedding_model.encode(question)
        
        # 3. 带权限过滤的向量检索
        results = self.milvus_client.search(
            collection_name="enterprise_rag",
            data=[q_vector],
            limit=20,  # 先多检索
            filter=f"permission_group in {self._get_user_groups(user_id)}",
            output_fields=["text", "source", "score"]
        )
        
        # 4. 重排序提升精度
        reranked = self.reranker.rerank(question, results)[:5]
        
        # 5. LLM 生成
        answer = self._generate_answer(question, reranked)
        
        # 6. 缓存结果
        self.cache.set(cache_key, answer, ttl=3600)
        
        # 7. 监控上报
        self.monitor.record_query(question, user_id, latency=...)
        
        return answer
```

## 3.5 主流 RAG 框架适配 Milvus

### 3.5.1 LangChain + Milvus

LangChain 提供了对 Milvus 的一等支持：

```python
from langchain_milvus import Milvus
from langchain.embeddings import HuggingFaceEmbeddings

# 使用 HuggingFace 模型 + Milvus
vector_store = Milvus(
    embedding_function=HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-zh-v1.5"
    ),
    collection_name="langchain_rag",
    connection_args={"host": "localhost", "port": "19530"},
    index_params={
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {"M": 16, "efConstruction": 200}
    }
)
```

### 3.5.2 LlamaIndex + Milvus

LlamaIndex 通过 `MilvusVectorStore` 集成：

```python
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core import VectorStoreIndex, Document

vector_store = MilvusVectorStore(
    uri="http://localhost:19530",
    collection_name="llamaindex_rag",
    dim=1024,
    overwrite=False
)

index = VectorStoreIndex.from_documents(
    documents,
    vector_store=vector_store
)
```

### 3.5.3 框架选择建议

| 框架 | 优势 | 适合场景 |
|------|------|---------|
| LangChain | 生态丰富、组件齐全 | 快速构建完整 RAG 应用 |
| LlamaIndex | 索引策略多样、数据连接器丰富 | 复杂数据源整合 |
| Haystack | 生产管线、Pipeline 设计 | 需要完整搜索管线的场景 |
| 直接使用 Milvus SDK | 最大灵活性、无额外依赖 | 深度定制、性能优先 |

## 本章小结

本章从 RAG 的完整全链路出发，详细拆解了数据预处理、向量化入库和检索生成三个核心阶段。我们深入分析了传统 RAG 在检索质量、上下文管理和延迟成本方面的六大痛点及其解决方案。重点阐述了 Milvus 在 RAG 系统中的核心角色——高性能向量存储与检索、标量过滤增强检索和实时增量更新。通过对比简易 RAG 与工业级 RAG 的差距，展示了在生产环境中使用 Milvus 的必要性。最后，介绍了 LangChain、LlamaIndex 等主流 RAG 框架与 Milvus 的集成方式。

在附录中，我们将提供 Milvus 的常用命令和 SDK 速查表，方便日常开发和运维。
