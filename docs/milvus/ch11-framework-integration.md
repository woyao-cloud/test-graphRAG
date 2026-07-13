# 第11章: 基于主流框架的Milvus-RAG快速落地

## 11.1 引言

在第10章中，我们搭建了最简版的RAG系统，理解了RAG的核心链路。但在实际项目中，从头实现文档加载、切片、Embedding、检索、Prompt拼接等所有环节是非常低效的。LangChain、LlamaIndex等RAG框架将这些环节封装为可组合的模块，配合Milvus的向量存储能力，可以极大地加速RAG系统的开发效率。

本章将分别介绍LangChain + Milvus和LlamaIndex + Milvus两种主流集成方案，以及不依赖任何框架的轻量化原生RAG实现，最后给出多文档多格式知识库的统一接入方案。本章内容对应项目`demos/ch11-langchain-rag/main.py`中的完整代码实现。

## 11.2 LangChain + Milvus搭建

### 11.2.1 LangChain简介

LangChain是目前最流行的LLM应用开发框架，提供了标准化的接口来组合模型、检索器、文档处理器等组件。LangChain对Milvus有原生支持，通过`langchain_milvus`或`langchain_community.vectorstores.Milvus`即可将Milvus作为向量存储后端。

### 11.2.2 环境准备

```bash
pip install langchain langchain-community langchain-milvus pymilvus
# 如果使用OpenAI Embedding
pip install langchain-openai
# 如果使用HuggingFace Embedding
pip install sentence-transformers
```

### 11.2.3 基本集成模式

以下代码展示了LangChain + Milvus最核心的集成模式：

```python
from langchain_community.vectorstores import Milvus
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
import numpy as np

# 1. 定义Embedding模型
class SimulatedEmbeddings(Embeddings):
    """模拟Embedding（生产环境替换为真实模型）"""
    def __init__(self, dim: int = 128):
        self.dim = dim
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for t in texts:
            # 使用文本hash作为种子，保证相同文本生成相同向量
            seed = hash(t) % (2**31)
            rng = np.random.RandomState(seed)
            vec = rng.randn(self.dim)
            vec = vec / np.linalg.norm(vec)  # L2归一化
            vectors.append(vec.tolist())
        return vectors
    
    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

embeddings = SimulatedEmbeddings()

# 2. 准备文档
documents = [
    Document(page_content="LangChain是一个用于构建LLM应用的开源框架。"),
    Document(page_content="Milvus是一款高性能向量数据库，专为大规模相似度搜索而设计。"),
    Document(page_content="RAG（检索增强生成）将信息检索与LLM生成相结合。"),
    Document(page_content="向量Embedding将文本转换为数值表示，实现语义搜索。"),
    Document(page_content="LangChain提供了Milvus包装器，可将Milvus作为向量存储后端。"),
]

# 3. 通过Milvus向量存储创建集合并插入文档
vector_store = Milvus.from_documents(
    documents=documents,
    embedding=embeddings,
    connection_args={"uri": "http://localhost:19530"},
    collection_name="langchain_milvus_demo",
    drop_old=True,  # 如果集合已存在则删除重建
)

print(f"集合 '{vector_store.collection_name}' 创建完成，共 {len(documents)} 条文档")
```

### 11.2.4 语义检索

```python
# 方式一：相似度检索（返回Document对象）
query = "Milvus有什么用途？"
results = vector_store.similarity_search(query, k=2)
for i, doc in enumerate(results):
    print(f"结果 #{i+1}: {doc.page_content}")

# 方式二：带分数的相似度检索
results_with_score = vector_store.similarity_search_with_score(query, k=3)
for i, (doc, score) in enumerate(results_with_score):
    print(f"结果 #{i+1} (分数={score:.4f}): {doc.page_content}")
```

### 11.2.5 使用Retriever接口

LangChain的Retriever接口是构建RAG链的核心抽象：

```python
# 将向量存储转换为检索器
retriever = vector_store.as_retriever(
    search_kwargs={"k": 2}  # 每次检索返回2条结果
)

# 使用检索器
retrieved_docs = retriever.invoke("RAG是如何工作的？")
for i, doc in enumerate(retrieved_docs):
    print(f"检索 #{i+1}: {doc.page_content}")
```

### 11.2.6 动态添加文档

知识库通常需要增量更新：

```python
new_docs = [
    Document(page_content="LangChain Milvus包装器支持同步和异步操作。"),
    Document(page_content="向量搜索支持跨大规模数据集的语义相似度匹配。"),
]

vector_store.add_documents(new_docs)
print(f"新增 {len(new_docs)} 条文档")
```

### 11.2.7 构建完整RAG链

结合LangChain的LCEL语法，可以轻松构建端到端的RAG问答链：

```python
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# 1. 创建Prompt模板
prompt = ChatPromptTemplate.from_template("""
你是一个知识问答助手。请根据以下检索到的上下文回答问题。

上下文：
{context}

问题：{input}

回答：""")

# 2. 创建LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 3. 创建文档组合链
combine_docs_chain = create_stuff_documents_chain(llm, prompt)

# 4. 创建检索链
rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

# 5. 执行问答
response = rag_chain.invoke({"input": "请介绍Milvus向量数据库"})
print(f"答案: {response['answer']}")
```

## 11.3 LlamaIndex + Milvus

### 11.3.1 LlamaIndex简介

LlamaIndex（原名GPT Index）是另一个流行的LLM数据框架，专注于数据索引和检索。相比LangChain的"通用框架"定位，LlamaIndex在数据索引和检索方面更为深入。

### 11.3.2 环境准备

```bash
pip install llama-index llama-index-vector-stores-milvus pymilvus
```

### 11.3.3 基本集成模式

```python
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core.embeddings import BaseEmbedding
from typing import Any, List
import numpy as np

# 1. 自定义Embedding模型
class SimulatedEmbedding(BaseEmbedding):
    """模拟Embedding"""
    def __init__(self, dim: int = 128, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
    
    def _get_query_embedding(self, query: str) -> List[float]:
        seed = hash(query) % (2**31)
        rng = np.random.RandomState(seed)
        vec = rng.randn(self.dim)
        vec = vec / np.linalg.norm(vec)
        return vec.tolist()
    
    def _get_text_embedding(self, text: str) -> List[float]:
        return self._get_query_embedding(text)
    
    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

# 2. 设置全局Embedding
Settings.embed_model = SimulatedEmbedding()

# 3. 创建Milvus向量存储
vector_store = MilvusVectorStore(
    uri="http://localhost:19530",
    collection_name="llamaindex_demo",
    dim=128,
    overwrite=True,
)

# 4. 创建文档并构建索引
documents = [
    Document(text="LlamaIndex是一个用于构建LLM应用的数据框架。"),
    Document(text="Milvus向量数据库支持高效的相似度搜索。"),
    Document(text="RAG结合了检索系统和生成模型的能力。"),
]

index = VectorStoreIndex.from_documents(
    documents=documents,
    vector_store=vector_store,
)

# 5. 创建检索引擎
query_engine = index.as_query_engine(similarity_top_k=2)

# 6. 执行查询
response = query_engine.query("什么是RAG？")
print(f"答案: {response}")
```

### 11.3.4 LlamaIndex vs LangChain选型对比

| 维度 | LangChain | LlamaIndex |
|------|-----------|------------|
| 框架定位 | LLM应用通用框架 | 数据索引与检索框架 |
| 检索能力 | 基础，依赖第三方 | 深度，内置多种检索策略 |
| 文档处理 | 需要自行组装Pipeline | 内置丰富的NodeParser |
| 链式调用 | LCEL语法，灵活强大 | 通过QueryEngine组合 |
| 学习曲线 | 中等，概念较多 | 较低，API更专注 |
| Milvus集成 | langchain-milvus包 | llama-index-vector-stores-milvus包 |
| 适用场景 | 复杂LLM应用（Agent、Tool） | 纯RAG/知识库场景 |

**选型建议**：如果项目主要需求是RAG知识库问答，LlamaIndex的开箱体验更好；如果项目涉及Agent、Tool Calling、多步推理等复杂LLM编排，LangChain更合适。

## 11.4 自定义原生RAG框架

在某些场景下，使用LangChain或LlamaIndex会带来不必要的依赖和抽象开销。对于需要高度定制化的项目，可以基于MilvusClient构建轻量级原生RAG框架。

### 11.4.1 核心接口设计

```python
from dataclasses import dataclass, field
from typing import List, Optional, Callable
from pymilvus import MilvusClient
import numpy as np

@dataclass
class Document:
    """文档片段"""
    id: int
    text: str
    metadata: dict = field(default_factory=dict)

@dataclass
class RetrieverResult:
    """检索结果"""
    document: Document
    score: float

class EmbeddingFunction:
    """Embedding函数抽象（可替换为任何模型）"""
    def __call__(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

class NativeRAG:
    """轻量化原生RAG框架"""
    
    def __init__(
        self,
        milvus_uri: str = "http://localhost:19530",
        collection_name: str = "native_rag",
        embedding_fn: Optional[EmbeddingFunction] = None,
        dim: int = 768,
    ):
        self.client = MilvusClient(uri=milvus_uri)
        self.collection_name = collection_name
        self.embedding_fn = embedding_fn
        self.dim = dim
    
    def create_collection(self, drop_if_exists: bool = False):
        """创建集合"""
        if drop_if_exists and self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)
        
        if not self.client.has_collection(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                dimension=self.dim,
                auto_id=False,
            )
    
    def add_documents(self, documents: List[Document]):
        """添加文档到知识库"""
        if self.embedding_fn is None:
            raise ValueError("请先设置EmbeddingFunction")
        
        texts = [doc.text for doc in documents]
        vectors = self.embedding_fn(texts)
        
        data = [
            {
                "id": doc.id,
                "vector": vectors[i],
                "text": doc.text,
            }
            for i, doc in enumerate(documents)
        ]
        
        self.client.insert(self.collection_name, data)
    
    def retrieve(
        self, query: str, top_k: int = 5, output_fields: Optional[List[str]] = None
    ) -> List[RetrieverResult]:
        """检索最相关的文档片段"""
        if self.embedding_fn is None:
            raise ValueError("请先设置EmbeddingFunction")
        
        query_vector = self.embedding_fn([query])[0]
        
        if output_fields is None:
            output_fields = ["id", "text"]
        
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            limit=top_k,
            output_fields=output_fields,
        )
        
        retrieved = []
        for hit in results[0]:
            doc = Document(
                id=hit["entity"]["id"],
                text=hit["entity"]["text"],
            )
            retrieved.append(RetrieverResult(
                document=doc,
                score=hit["distance"],
            ))
        
        return retrieved
    
    def query(
        self,
        question: str,
        top_k: int = 5,
        llm_func: Optional[Callable[[str, List[str]], str]] = None,
    ) -> str:
        """完整RAG问答"""
        # 1. 检索
        results = self.retrieve(question, top_k=top_k)
        contexts = [r.document.text for r in results]
        
        # 2. 生成答案
        if llm_func:
            return llm_func(question, contexts)
        
        # 默认返回上下文拼接
        return "\n".join(contexts)
```

### 11.4.2 使用原生框架

```python
# 1. 定义Embedding函数
class BGEEmbedding(EmbeddingFunction):
    def __init__(self, model_name="BAAI/bge-small-zh-v1.5"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
    
    def __call__(self, texts):
        return self.model.encode(texts).tolist()

# 2. 初始化RAG系统
rag = NativeRAG(
    milvus_uri="http://localhost:19530",
    collection_name="my_knowledge_base",
    embedding_fn=BGEEmbedding(),
    dim=384,
)

# 3. 创建集合并添加文档
rag.create_collection(drop_if_exists=True)
rag.add_documents([
    Document(id=1, text="Milvus向量数据库支持多种索引类型。"),
    Document(id=2, text="RAG技术结合了检索和生成的优势。"),
])

# 4. 检索
results = rag.retrieve("Milvus支持哪些索引？", top_k=2)
for r in results:
    print(f"  分数={r.score:.4f}: {r.document.text}")
```

原生框架的优势在于完全可控、没有冗余抽象、依赖最少，适合对RAG流程有深入理解、需要精细控制的团队。

## 11.5 多文档多格式知识库统一接入

### 11.5.1 统一文档加载器

在实际项目中，知识库通常包含PDF、Word、Markdown、TXT等多种格式的文档。需要设计统一的加载接口：

```python
import os
from typing import List

class DocumentLoader:
    """统一文档加载器"""
    
    @staticmethod
    def load_pdf(file_path: str) -> List[str]:
        """加载PDF文件"""
        try:
            import PyPDF2
            texts = []
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text = page.extract_text()
                    if text.strip():
                        texts.append(text)
            return texts
        except ImportError:
            raise ImportError("请安装PyPDF2: pip install PyPDF2")
    
    @staticmethod
    def load_docx(file_path: str) -> List[str]:
        """加载Word文件"""
        try:
            import docx
            doc = docx.Document(file_path)
            return [p.text for p in doc.paragraphs if p.text.strip()]
        except ImportError:
            raise ImportError("请安装python-docx: pip install python-docx")
    
    @staticmethod
    def load_markdown(file_path: str) -> str:
        """加载Markdown文件"""
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    
    @staticmethod
    def load_txt(file_path: str, encoding: str = "utf-8") -> str:
        """加载纯文本文件"""
        with open(file_path, "r", encoding=encoding) as f:
            return f.read()
    
    @classmethod
    def load(cls, file_path: str) -> str:
        """统一加载入口"""
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            return "\n".join(cls.load_pdf(file_path))
        elif ext in (".docx", ".doc"):
            return "\n".join(cls.load_docx(file_path))
        elif ext in (".md", ".markdown"):
            return cls.load_markdown(file_path)
        elif ext == ".txt":
            return cls.load_txt(file_path)
        else:
            raise ValueError(f"不支持的文档格式: {ext}")
```

### 11.5.2 批量知识库导入

```python
import os
from pathlib import Path

def import_knowledge_base(
    rag_system,  # NativeRAG 实例
    base_path: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
):
    """批量导入整个知识库目录"""
    loader = DocumentLoader()
    all_chunks = []
    doc_id = 1
    
    for file_path in Path(base_path).rglob("*"):
        if file_path.suffix.lower() not in (".pdf", ".docx", ".md", ".txt"):
            continue
        
        try:
            # 1. 加载文档
            full_text = loader.load(str(file_path))
            
            # 2. 文档切片
            chunks = chunk_by_size(full_text, chunk_size, chunk_overlap)
            
            # 3. 创建Document对象
            for chunk in chunks:
                all_chunks.append(Document(
                    id=doc_id,
                    text=chunk,
                    metadata={"source": str(file_path)},
                ))
                doc_id += 1
            
            print(f"  [OK] {file_path.name} -> {len(chunks)} 个片段")
            
        except Exception as e:
            print(f"  [ERR] {file_path.name}: {e}")
    
    # 4. 批量写入Milvus
    if all_chunks:
        rag_system.add_documents(all_chunks)
        print(f"\n知识库导入完成！共 {len(all_chunks)} 个文档片段")
    else:
        print("\n未找到可导入的文档")
```

### 11.5.3 多知识源路由

当系统需要对接多个业务领域的知识库时，可以通过分区或不同集合来实现路由：

```python
class MultiSourceRAG:
    """多知识源RAG系统"""
    
    def __init__(self, milvus_uri: str, embedding_fn):
        self.milvus_uri = milvus_uri
        self.embedding_fn = embedding_fn
        self.knowledge_bases = {}  # name -> NativeRAG
    
    def add_knowledge_base(self, name: str, collection_name: str):
        """注册一个知识库"""
        rag = NativeRAG(
            milvus_uri=self.milvus_uri,
            collection_name=collection_name,
            embedding_fn=self.embedding_fn,
        )
        rag.create_collection()
        self.knowledge_bases[name] = rag
    
    def query(self, question: str, source: str = None, top_k: int = 5):
        """查询指定知识库，或遍历所有知识库"""
        if source:
            if source in self.knowledge_bases:
                return self.knowledge_bases[source].retrieve(question, top_k)
            raise ValueError(f"未知知识库: {source}")
        
        # 多源合并检索
        all_results = []
        for name, kb in self.knowledge_bases.items():
            results = kb.retrieve(question, top_k)
            for r in results:
                r.document.metadata["source"] = name
            all_results.extend(results)
        
        # 按分数排序返回TopK
        all_results.sort(key=lambda x: x.score, reverse=True)
        return all_results[:top_k]
```

## 11.6 本章小结

本章介绍了三种基于Milvus的RAG落地模式：

- **LangChain + Milvus**：适合需要复杂LLM编排（Agent、链式调用）的项目，通过`langchain_community.vectorstores.Milvus`实现快速集成。
- **LlamaIndex + Milvus**：适合纯RAG/知识库场景，内置丰富的数据索引和检索策略，开箱即用。
- **原生RAG框架**：适合对框架依赖敏感或需要深度定制的项目，基于MilvusClient构建轻量级RAG系统。

此外，还提供了多文档多格式知识库的统一接入方案，包括文档加载器、批量导入器和多源路由器的实现。

选择哪种方案取决于项目需求：如果需要快速验证RAG效果，推荐LangChain或LlamaIndex；如果项目对依赖管理有严格要求，或需要深度定制检索逻辑，推荐基于MilvusClient构建原生框架。从下一章开始，我们将进入RAG系统的高阶优化阶段，探讨如何提升召回率、检索速度和处理海量数据的能力。
