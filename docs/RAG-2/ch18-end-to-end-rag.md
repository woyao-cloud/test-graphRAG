# 第18章 端到端RAG项目实战

## 18.1 引言

前面的章节从多个角度深入介绍了RAG系统的各个组件和技术细节。本章将把这些知识整合起来，完整地介绍一个RAG项目从需求分析到生产部署的全流程。通过一个具体的端到端项目案例，展示架构设计、组件选择、集成模式、测试策略、部署方案和运维管理的完整过程。

### 18.1.1 项目案例背景

假设我们要为一家医药企业构建一个智能知识问答系统。该企业拥有大量的药品说明书、临床试验报告、法规文档和内部知识库，需要为员工提供一个统一的智能问答入口。

**项目目标**：
- 构建企业级RAG系统，覆盖药品信息、临床试验、法规政策等知识域
- 支持自然语言问答，准确率不低于85%
- P99延迟不超过3秒
- 支持100+并发用户

**约束条件**：
- 知识库包含敏感数据，需要权限控制
- 部分文档为中文，需要良好的中文理解能力
- 需要在6周内交付MVP

### 18.1.2 项目路线图

```
第1-2周：需求分析和架构设计
├── 需求调研和用例定义
├── 技术选型和架构设计
└── 评估指标定义

第3-4周：核心功能开发
├── 数据处理流水线
├── 检索系统实现
├── 生成系统实现
└── API开发

第5周：集成和测试
├── 组件集成
├── 性能测试
├── 安全测试
└── 用户验收测试

第6周：部署和上线
├── 环境部署
├── 监控配置
└── 上线评审
```

## 18.2 需求分析

### 18.2.1 用例定义

```python
class UseCaseManager:
    """用例管理器"""
    
    def __init__(self):
        self.use_cases = []
    
    def define_use_case(self, name: str, 
                         description: str,
                         actors: List[str],
                         preconditions: List[str],
                         postconditions: List[str],
                         main_flow: List[str],
                         alternative_flows: List[List[str]] = None):
        """定义用例"""
        use_case = {
            'id': f'UC-{len(self.use_cases) + 1}',
            'name': name,
            'description': description,
            'actors': actors,
            'preconditions': preconditions,
            'postconditions': postconditions,
            'main_flow': main_flow,
            'alternative_flows': alternative_flows or [],
            'priority': 'P0'
        }
        self.use_cases.append(use_case)
        return use_case
    
    def get_priority_use_cases(self, priority: str = 'P0') -> List[Dict]:
        """获取特定优先级的用例"""
        return [uc for uc in self.use_cases if uc['priority'] == priority]

# 定义核心用例
use_case_manager = UseCaseManager()

# UC1: 药品信息查询
use_case_manager.define_use_case(
    name="药品信息查询",
    description="用户通过药品名称查询详细信息",
    actors=["内部员工"],
    preconditions=["用户已登录", "知识库包含药品信息"],
    postconditions=["用户获得药品详细信息"],
    main_flow=[
        "用户输入药品名称或相关问题",
        "系统检索相关知识库",
        "系统生成结构化回答",
        "系统返回药品信息，包含来源引用"
    ],
    alternative_flows=[
        ["知识库中无相关信息", "系统提示无法找到，建议联系知识管理部门"]
    ]
)

# UC2: 多文档综合问答
use_case_manager.define_use_case(
    name="多文档综合问答",
    description="用户需要综合多个文档的信息回答复杂问题",
    actors=["研究员"],
    preconditions=["知识库包含相关多篇文档"],
    postconditions=["用户获得综合分析结果"],
    main_flow=[
        "用户提出需要综合分析的问题",
        "系统分解问题并检索多个相关文档",
        "系统综合分析并生成答案",
        "系统标注各信息来源"
    ],
    priority="P1"
)
```

### 18.2.2 非功能需求

```python
class NonFunctionalRequirements:
    """非功能需求"""
    
    def __init__(self):
        self.requirements = []
    
    def add_requirement(self, category: str, 
                         name: str,
                         description: str,
                         target: Any,
                         measurement: str):
        """添加非功能需求"""
        req = {
            'category': category,
            'name': name,
            'description': description,
            'target': target,
            'measurement': measurement,
            'status': 'defined'
        }
        self.requirements.append(req)
        return req
    
    def get_requirements_by_category(self, category: str) -> List[Dict]:
        """按类别获取需求"""
        return [r for r in self.requirements if r['category'] == category]

nfr = NonFunctionalRequirements()

# 性能需求
nfr.add_requirement('performance', '响应时间', 
    '用户查询的平均响应时间', '< 2s', 'P50延迟')
nfr.add_requirement('performance', '峰值响应时间', 
    '用户查询的P99响应时间', '< 5s', 'P99延迟')
nfr.add_requirement('performance', '并发用户', 
    '系统支持的最大并发用户数', '>= 100', '并发测试')

# 可用性需求
nfr.add_requirement('availability', '系统可用性', 
    '系统正常运行时间比例', '>= 99.5%', '监控统计')
nfr.add_requirement('availability', '数据备份', 
    '知识库数据的备份策略', '每日备份', '备份日志')

# 安全需求
nfr.add_requirement('security', '访问控制', 
    '文档级别权限控制', 'RBAC', '权限审计')
nfr.add_requirement('security', '数据加密', 
    '敏感数据传输和存储加密', 'TLS + AES-256', '安全扫描')

# 可维护性需求
nfr.add_requirement('maintainability', '日志记录', 
    '系统操作日志记录', '所有API调用', '日志分析')
nfr.add_requirement('maintainability', '监控告警', 
    '系统健康状态监控', '实时告警', '监控面板')
```

## 18.3 架构设计

### 18.3.1 系统架构概览

RAG系统的整体架构可以分为以下几个层次：

```
┌─────────────────────────────────────────────────┐
│                  接入层                          │
│     REST API / WebSocket / SDK                  │
├─────────────────────────────────────────────────┤
│                  网关层                          │
│     负载均衡 / 认证授权 / 限流 / 日志            │
├─────────────────────────────────────────────────┤
│                  编排层                          │
│   查询路由 / 多轮对话 / 工作流引擎               │
├──────────────┬──────────────────┬───────────────┤
│  检索服务    │  生成服务        │  评估服务      │
│  ┌────────┐  │  ┌────────┐     │  ┌────────┐   │
│  │向量检索│  │  │LLM调用 │     │  │质量评估│   │
│  │关键词  │  │  │提示词  │     │  │在线监控│   │
│  │知识图谱│  │  │后处理  │     │  │A/B测试 │   │
│  └────────┘  │  └────────┘     │  └────────┘   │
├──────────────┴──────────────────┴───────────────┤
│                  数据层                          │
│  向量数据库 / 文档存储 / 缓存 / 消息队列         │
├─────────────────────────────────────────────────┤
│                  基础设施                        │
│  Kubernetes / Docker / CI/CD / 监控              │
└─────────────────────────────────────────────────┘
```

### 18.3.2 组件选择

```python
class ComponentSelector:
    """组件选择器"""
    
    def __init__(self):
        self.components = {}
        self.selection_criteria = {}
    
    def evaluate_component(self, name: str, 
                            category: str,
                            candidates: List[Dict],
                            criteria: Dict[str, float]) -> Dict:
        """评估组件选择"""
        self.selection_criteria[name] = criteria
        
        scored_candidates = []
        for candidate in candidates:
            score = self._calculate_score(candidate, criteria)
            scored_candidates.append({
                **candidate,
                'score': score
            })
        
        scored_candidates.sort(key=lambda x: x['score'], reverse=True)
        
        selection = {
            'category': category,
            'candidates': scored_candidates,
            'recommended': scored_candidates[0] if scored_candidates else None,
            'rationale': self._generate_rationale(scored_candidates, criteria)
        }
        
        self.components[name] = selection
        return selection
    
    def _calculate_score(self, candidate: Dict, 
                          criteria: Dict[str, float]) -> float:
        """计算候选组件评分"""
        score = 0.0
        for criterion, weight in criteria.items():
            value = candidate.get(criterion, 0)
            score += value * weight
        return score
    
    def _generate_rationale(self, candidates: List[Dict],
                             criteria: Dict[str, float]) -> str:
        """生成选择理由"""
        if not candidates:
            return "无可用候选"
        
        best = candidates[0]
        return f"推荐{best['name']}，综合评分{best['score']:.2f}"

selector = ComponentSelector()

# 向量数据库选择
selector.evaluate_component(
    name="vector_db",
    category="向量数据库",
    candidates=[
        {'name': 'Milvus', 'performance': 9, 'ease_of_use': 7, 'community': 8, 'cost': 7},
        {'name': 'Pinecone', 'performance': 8, 'ease_of_use': 9, 'community': 7, 'cost': 5},
        {'name': 'Weaviate', 'performance': 8, 'ease_of_use': 8, 'community': 7, 'cost': 7},
        {'name': 'Qdrant', 'performance': 8, 'ease_of_use': 8, 'community': 6, 'cost': 8}
    ],
    criteria={'performance': 0.3, 'ease_of_use': 0.2, 'community': 0.2, 'cost': 0.3}
)

# LLM选择
selector.evaluate_component(
    name="llm",
    category="大语言模型",
    candidates=[
        {'name': 'GPT-4o', 'performance': 9, 'cost': 4, 'latency': 7, 'chinese': 8},
        {'name': 'DeepSeek-V3', 'performance': 8, 'cost': 7, 'latency': 8, 'chinese': 9},
        {'name': 'Qwen2.5', 'performance': 8, 'cost': 8, 'latency': 8, 'chinese': 9},
        {'name': 'Claude-3.5', 'performance': 9, 'cost': 5, 'latency': 7, 'chinese': 7}
    ],
    criteria={'performance': 0.3, 'cost': 0.25, 'latency': 0.2, 'chinese': 0.25}
)

# 嵌入模型选择
selector.evaluate_component(
    name="embedding",
    category="嵌入模型",
    candidates=[
        {'name': 'text-embedding-3-large', 'performance': 9, 'cost': 5, 'dimension': 3072},
        {'name': 'bge-m3', 'performance': 8, 'cost': 9, 'dimension': 1024},
        {'name': 'm3e-large', 'performance': 7, 'cost': 9, 'dimension': 768}
    ],
    criteria={'performance': 0.4, 'cost': 0.3, 'dimension': 0.3}
)
```

### 18.3.3 架构决策记录

```python
class ArchitectureDecisionRecord:
    """架构决策记录"""
    
    def __init__(self):
        self.decisions = []
    
    def record_decision(self, title: str, 
                         context: str,
                         decision: str,
                         alternatives: List[str],
                         consequences: List[str]):
        """记录架构决策"""
        adr = {
            'id': f'ADR-{len(self.decisions) + 1}',
            'title': title,
            'context': context,
            'decision': decision,
            'alternatives': alternatives,
            'consequences': consequences,
            'status': 'accepted',
            'date': datetime.now().isoformat()
        }
        self.decisions.append(adr)
        return adr

adr = ArchitectureDecisionRecord()

# ADR 1: 使用混合检索策略
adr.record_decision(
    title="采用混合检索策略",
    context="单一的向量检索在精确匹配场景下效果不佳，需要结合关键词检索",
    decision="采用向量检索+BM25关键词检索的混合策略，通过RRF（Reciprocal Rank Fusion）融合结果",
    alternatives=["仅使用向量检索", "仅使用关键词检索", "使用重排序模型"],
    consequences=["提高检索召回率", "增加系统复杂度", "需要维护两套索引"]
)

# ADR 2: 使用DeepSeek作为LLM
adr.record_decision(
    title="采用DeepSeek Chat作为主LLM",
    context="需要中文能力强、成本可控的LLM",
    decision="采用DeepSeek Chat作为主要LLM，GPT-4o作为备选",
    alternatives=["GPT-4o", "Claude-3.5", "Qwen2.5"],
    consequences=["更好的中文理解", "更低的成本", "需要处理API不稳定问题"]
)
```

## 18.4 核心实现

### 18.4.1 数据处理流水线

```python
class DataPipeline:
    """数据处理流水线"""
    
    def __init__(self, raw_data_dir: str, processed_data_dir: str):
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        self.steps = []
    
    def add_step(self, name: str, func: Callable, 
                 input_pattern: str, output_name: str):
        """添加处理步骤"""
        self.steps.append({
            'name': name,
            'func': func,
            'input': input_pattern,
            'output': output_name
        })
    
    def run(self):
        """运行流水线"""
        current_data = None
        
        for step in self.steps:
            print(f"执行步骤: {step['name']}")
            
            if current_data is None:
                # 读取原始数据
                input_files = list(self.raw_data_dir.glob(step['input']))
                current_data = self._load_files(input_files)
            
            # 执行处理
            current_data = step['func'](current_data)
            
            # 保存中间结果
            if step['output']:
                output_path = self.processed_data_dir / step['output']
                self._save_data(current_data, output_path)
        
        return current_data
    
    def _load_files(self, files: List[Path]) -> List[Dict]:
        """加载文件"""
        data = []
        for file in files:
            if file.suffix == '.pdf':
                data.extend(self._load_pdf(file))
            elif file.suffix == '.txt':
                data.extend(self._load_text(file))
            elif file.suffix == '.json':
                data.extend(self._load_json(file))
        return data
    
    def _load_pdf(self, file: Path) -> List[Dict]:
        """加载PDF文件"""
        # 使用PyMuPDF或pdfplumber
        try:
            import fitz
            doc = fitz.open(file)
            pages = []
            for page_num in range(len(doc)):
                page = doc[page_num]
                pages.append({
                    'content': page.get_text(),
                    'metadata': {
                        'source': str(file),
                        'page': page_num + 1,
                        'type': 'pdf'
                    }
                })
            return pages
        except ImportError:
            print("请安装PyMuPDF: pip install PyMuPDF")
            return []
    
    def _load_text(self, file: Path) -> List[Dict]:
        """加载文本文件"""
        with open(file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return [{
            'content': content,
            'metadata': {
                'source': str(file),
                'type': 'text'
            }
        }]
    
    def _load_json(self, file: Path) -> List[Dict]:
        """加载JSON文件"""
        import json
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            return data
        return [data]
    
    def _save_data(self, data: Any, path: Path):
        """保存数据"""
        import json
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

# 定义处理步骤
def clean_text(data: List[Dict]) -> List[Dict]:
    """清洗文本"""
    import re
    
    for item in data:
        content = item['content']
        # 去除多余空白
        content = re.sub(r'\s+', ' ', content)
        # 去除特殊字符
        content = re.sub(r'[^一-鿿\w\s\.\,\!\?\(\)\[\]]', '', content)
        item['content'] = content.strip()
    
    return data

def chunk_documents(data: List[Dict], 
                     chunk_size: int = 512,
                     overlap: int = 50) -> List[Dict]:
    """文档分块"""
    chunks = []
    
    for item in data:
        content = item['content']
        metadata = item['metadata']
        
        for i in range(0, len(content), chunk_size - overlap):
            chunk_content = content[i:i + chunk_size]
            if len(chunk_content) < 50:  # 忽略过短的块
                continue
            
            chunks.append({
                'content': chunk_content,
                'metadata': {
                    **metadata,
                    'chunk_index': len(chunks),
                    'chunk_start': i,
                    'chunk_end': i + len(chunk_content)
                }
            })
    
    return chunks

# 构建流水线
pipeline = DataPipeline(
    raw_data_dir="data/raw",
    processed_data_dir="data/processed"
)

pipeline.add_step("文本清洗", clean_text, "*.pdf", "cleaned.json")
pipeline.add_step("文档分块", chunk_documents, "cleaned.json", "chunks.json")
```

### 18.4.2 RAG核心引擎

```python
class RAGEngine:
    """RAG核心引擎"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # 初始化组件
        self.embedding_model = self._init_embedding_model()
        self.vector_store = self._init_vector_store()
        self.llm = self._init_llm()
        
        # 检索配置
        self.top_k = config.get('retrieval', {}).get('top_k', 5)
        self.min_score = config.get('retrieval', {}).get('min_score', 0.5)
    
    def _init_embedding_model(self):
        """初始化嵌入模型"""
        model_name = self.config['embedding']['model']
        # 根据配置初始化
        return None  # 实际实现
    
    def _init_vector_store(self):
        """初始化向量存储"""
        store_type = self.config['vector_store']['type']
        if store_type == 'milvus':
            # 初始化Milvus
            pass
        elif store_type == 'qdrant':
            # 初始化Qdrant
            pass
        return None
    
    def _init_llm(self):
        """初始化LLM"""
        provider = self.config['llm']['provider']
        model = self.config['llm']['model']
        
        if provider == 'openai':
            from openai import OpenAI
            return OpenAI(api_key=self.config['llm']['api_key'])
        elif provider == 'deepseek':
            from openai import OpenAI
            return OpenAI(
                api_key=self.config['llm']['api_key'],
                base_url="https://api.deepseek.com"
            )
        return None
    
    def query(self, question: str, 
              user_id: str = None,
              chat_history: List[Dict] = None) -> Dict:
        """执行RAG查询"""
        start_time = time.time()
        
        # 1. 查询嵌入
        query_embedding = self._embed_query(question)
        
        # 2. 向量检索
        vector_results = self._vector_search(query_embedding)
        
        # 3. 关键词检索（混合检索）
        keyword_results = self._keyword_search(question)
        
        # 4. 结果融合
        fused_results = self._fuse_results(vector_results, keyword_results)
        
        # 5. 过滤低分结果
        filtered_results = [
            r for r in fused_results 
            if r['score'] >= self.min_score
        ][:self.top_k]
        
        # 6. 构建上下文
        context = self._build_context(filtered_results)
        
        # 7. 生成答案
        answer = self._generate(question, context, chat_history)
        
        # 8. 后处理
        processed_answer = self._postprocess(answer, filtered_results)
        
        elapsed = time.time() - start_time
        
        return {
            'answer': processed_answer,
            'sources': [r['source'] for r in filtered_results],
            'confidence': self._calculate_confidence(filtered_results),
            'latency_ms': elapsed * 1000,
            'retrieval_count': len(filtered_results)
        }
    
    def _embed_query(self, text: str) -> List[float]:
        """嵌入查询"""
        # 实际实现
        return []
    
    def _vector_search(self, embedding: List[float]) -> List[Dict]:
        """向量检索"""
        # 实际实现
        return []
    
    def _keyword_search(self, query: str) -> List[Dict]:
        """关键词检索"""
        # 实际实现
        return []
    
    def _fuse_results(self, vector_results: List[Dict],
                       keyword_results: List[Dict]) -> List[Dict]:
        """RRF结果融合"""
        from collections import defaultdict
        
        scores = defaultdict(float)
        
        for rank, result in enumerate(vector_results, 1):
            doc_id = result.get('id', result.get('content', ''))
            scores[doc_id] += 1.0 / (60 + rank)  # RRF公式
        
        for rank, result in enumerate(keyword_results, 1):
            doc_id = result.get('id', result.get('content', ''))
            scores[doc_id] += 1.0 / (60 + rank)
        
        # 合并结果
        all_docs = {r.get('id', r.get('content', '')): r 
                    for r in vector_results + keyword_results}
        
        fused = [
            {**all_docs[doc_id], 'score': score}
            for doc_id, score in scores.items()
        ]
        
        fused.sort(key=lambda x: x['score'], reverse=True)
        return fused
    
    def _build_context(self, results: List[Dict]) -> str:
        """构建上下文"""
        parts = []
        for i, result in enumerate(results, 1):
            source = result.get('source', f'文档{i}')
            content = result.get('content', '')
            parts.append(f"[{i}] 来源: {source}\n{content}")
        
        return "\n\n".join(parts)
    
    def _generate(self, question: str, context: str,
                  chat_history: List[Dict] = None) -> str:
        """生成答案"""
        # 构建提示词
        messages = [
            {"role": "system", "content": self._get_system_prompt()}
        ]
        
        if chat_history:
            messages.extend(chat_history[-5:])  # 保留最近5轮
        
        messages.append({
            "role": "user",
            "content": f"""基于以下信息回答问题。

参考信息：
{context}

问题：{question}

请给出准确、完整的回答。如果参考信息不足以回答问题，请明确说明。"""
        })
        
        # 调用LLM
        response = self.llm.chat.completions.create(
            model=self.config['llm']['model'],
            messages=messages,
            temperature=0.3,
            max_tokens=1024
        )
        
        return response.choices[0].message.content
    
    def _get_system_prompt(self) -> str:
        """获取系统提示词"""
        return """你是一个专业的医药知识助手。请基于提供的参考信息回答问题。
要求：
1. 答案必须基于参考信息，不要添加额外信息
2. 在相关陈述后标注来源 [数字]
3. 如果信息不足，明确说明
4. 使用专业但易懂的语言"""
    
    def _postprocess(self, answer: str, 
                      results: List[Dict]) -> str:
        """后处理"""
        # 添加来源列表
        sources = []
        for i, result in enumerate(results, 1):
            source = result.get('source', f'文档{i}')
            sources.append(f"[{i}] {source}")
        
        if sources:
            answer += "\n\n**参考来源：**\n" + "\n".join(sources)
        
        return answer
    
    def _calculate_confidence(self, results: List[Dict]) -> float:
        """计算置信度"""
        if not results:
            return 0.0
        
        scores = [r.get('score', 0) for r in results]
        return sum(scores) / len(scores) * min(len(scores) / 5, 1.0)
```

### 18.4.3 API服务

```python
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional, List
import uvicorn

app = FastAPI(title="RAG Knowledge API", version="1.0.0")
security = HTTPBearer()

# 请求/响应模型
class QueryRequest(BaseModel):
    question: str
    user_id: Optional[str] = None
    chat_history: Optional[List[Dict]] = None
    top_k: Optional[int] = 5

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    confidence: float
    latency_ms: float

class DocumentIngestRequest(BaseModel):
    documents: List[Dict]
    source_type: str = "text"

# 全局引擎实例
rag_engine = None

@app.on_event("startup")
async def startup():
    """启动时初始化"""
    global rag_engine
    config = load_config()
    rag_engine = RAGEngine(config)

@app.post("/api/v1/query", response_model=QueryResponse)
async def query(request: QueryRequest,
                credentials: HTTPAuthorizationCredentials = Depends(security)):
    """RAG查询接口"""
    if not rag_engine:
        raise HTTPException(status_code=503, detail="服务未就绪")
    
    try:
        result = rag_engine.query(
            question=request.question,
            user_id=request.user_id,
            chat_history=request.chat_history
        )
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/documents/ingest")
async def ingest_documents(request: DocumentIngestRequest,
                           credentials: HTTPAuthorizationCredentials = Depends(security)):
    """文档导入接口"""
    # 实际实现
    return {"status": "success", "document_count": len(request.documents)}

@app.get("/api/v1/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

def start_server():
    """启动服务器"""
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        workers=4,
        log_level="info"
    )
```

## 18.5 测试策略

### 18.5.1 测试金字塔

```python
class TestStrategy:
    """测试策略"""
    
    def __init__(self):
        self.test_cases = []
    
    def add_unit_test(self, component: str, 
                       test_name: str,
                       input_data: Any,
                       expected_output: Any):
        """添加单元测试"""
        self.test_cases.append({
            'type': 'unit',
            'component': component,
            'name': test_name,
            'input': input_data,
            'expected': expected_output
        })
    
    def add_integration_test(self, scenario: str,
                              steps: List[Dict],
                              expected_behavior: str):
        """添加集成测试"""
        self.test_cases.append({
            'type': 'integration',
            'scenario': scenario,
            'steps': steps,
            'expected': expected_behavior
        })
    
    def add_evaluation_test(self, name: str,
                             test_data: List[Dict],
                             metrics: List[str],
                             thresholds: Dict[str, float]):
        """添加评估测试"""
        self.test_cases.append({
            'type': 'evaluation',
            'name': name,
            'test_data': test_data,
            'metrics': metrics,
            'thresholds': thresholds
        })
    
    def get_test_suite(self, test_type: str = None) -> List[Dict]:
        """获取测试套件"""
        if test_type:
            return [t for t in self.test_cases if t['type'] == test_type]
        return self.test_cases

test_strategy = TestStrategy()

# 单元测试
test_strategy.add_unit_test(
    component="文档分块",
    test_name="正常分块",
    input_data={"content": "A" * 1000, "metadata": {}},
    expected_output={"chunk_count": 2}
)

# 集成测试
test_strategy.add_integration_test(
    scenario="完整RAG查询流程",
    steps=[
        {"action": "导入测试文档", "data": "test_docs.json"},
        {"action": "执行查询", "query": "阿斯匹林的作用是什么"},
        {"action": "验证响应"}
    ],
    expected_behavior="返回包含引用的准确回答"
)

# 评估测试
test_strategy.add_evaluation_test(
    name="检索质量评估",
    test_data=[],  # 评估数据集
    metrics=["Recall@5", "Precision@5", "MRR"],
    thresholds={"Recall@5": 0.7, "MRR": 0.6}
)
```

### 18.5.2 自动化测试流水线

```python
class TestPipeline:
    """测试流水线"""
    
    def __init__(self):
        self.stages = [
            {'name': 'lint', 'command': 'ruff check .'},
            {'name': 'type_check', 'command': 'mypy .'},
            {'name': 'unit_tests', 'command': 'pytest tests/unit -v'},
            {'name': 'integration_tests', 'command': 'pytest tests/integration -v'},
            {'name': 'evaluation', 'command': 'pytest tests/evaluation -v'}
        ]
    
    def run(self, stage: str = None) -> Dict:
        """运行测试流水线"""
        results = {
            'passed': 0,
            'failed': 0,
            'details': []
        }
        
        stages_to_run = [s for s in self.stages 
                        if stage is None or s['name'] == stage]
        
        for stage in stages_to_run:
            print(f"运行 {stage['name']}...")
            
            import subprocess
            process = subprocess.run(
                stage['command'],
                shell=True,
                capture_output=True,
                text=True
            )
            
            stage_result = {
                'stage': stage['name'],
                'passed': process.returncode == 0,
                'output': process.stdout[-500:] if process.stdout else "",
                'error': process.stderr[-500:] if process.stderr else ""
            }
            
            results['details'].append(stage_result)
            
            if stage_result['passed']:
                results['passed'] += 1
            else:
                results['failed'] += 1
                
                # 如果单元测试失败，停止流水线
                if stage['name'] in ['unit_tests']:
                    break
        
        return results
```

## 18.6 部署方案

### 18.6.1 Docker化

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 安装Python依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY . .

# 配置环境变量
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 18.6.2 Docker Compose配置

```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - REDIS_URL=redis://redis:6379
      - MILVUS_HOST=milvus
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      - redis
      - milvus
    volumes:
      - ./data:/app/data
    restart: always
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: always

  milvus:
    image: milvusdb/milvus:v2.3.0
    ports:
      - "19530:19530"
    environment:
      - ETCD_ENDPOINTS=etcd:2379
      - MINIO_ADDRESS=minio:9000
    depends_on:
      - etcd
      - minio
    volumes:
      - milvus_data:/var/lib/milvus

  etcd:
    image: quay.io/coreos/etcd:v3.5.5
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296

  minio:
    image: minio/minio:RELEASE.2023-03-20T20-16-18Z
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    command: server /minio-data

volumes:
  redis_data:
  milvus_data:
```

### 18.6.3 Kubernetes部署

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-api
  labels:
    app: rag-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rag-api
  template:
    metadata:
      labels:
        app: rag-api
    spec:
      containers:
      - name: api
        image: rag-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: rag-secrets
              key: redis-url
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: rag-secrets
              key: openai-api-key
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /api/v1/health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /api/v1/health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: rag-api-service
spec:
  selector:
    app: rag-api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: rag-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: rag-api
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

## 18.7 CI/CD流水线

### 18.7.1 GitHub Actions配置

```yaml
# .github/workflows/ci-cd.yml
name: RAG CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Lint
      run: ruff check .
    
    - name: Type check
      run: mypy .
    
    - name: Unit tests
      run: pytest tests/unit -v --cov=app
    
    - name: Integration tests
      run: pytest tests/integration -v

  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
    - uses: actions/checkout@v3
    
    - name: Build Docker image
      run: docker build -t rag-api:${{ github.sha }} .
    
    - name: Push to registry
      run: |
        docker tag rag-api:${{ github.sha }} registry.example.com/rag-api:latest
        docker push registry.example.com/rag-api:latest

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
    - name: Deploy to Kubernetes
      run: |
        kubectl set image deployment/rag-api \
          api=registry.example.com/rag-api:${{ github.sha }}
        kubectl rollout status deployment/rag-api
```

## 18.8 监控告警

### 18.8.1 关键监控指标

```python
class MonitoringConfig:
    """监控配置"""
    
    def __init__(self):
        self.metrics = [
            {
                'name': 'query_latency_p50',
                'description': '查询P50延迟',
                'unit': 'ms',
                'warning': 2000,
                'critical': 5000
            },
            {
                'name': 'query_latency_p99',
                'description': '查询P99延迟',
                'unit': 'ms',
                'warning': 5000,
                'critical': 10000
            },
            {
                'name': 'qps',
                'description': '每秒查询数',
                'unit': 'qps',
                'warning': 80,
                'critical': 100
            },
            {
                'name': 'error_rate',
                'description': '错误率',
                'unit': '%',
                'warning': 1,
                'critical': 5
            },
            {
                'name': 'token_usage_per_query',
                'description': '每查询Token消耗',
                'unit': 'tokens',
                'warning': 4000,
                'critical': 8000
            }
        ]
    
    def get_alert_rules(self) -> List[Dict]:
        """获取告警规则"""
        rules = []
        for metric in self.metrics:
            rules.append({
                'metric': metric['name'],
                'warning': metric['warning'],
                'critical': metric['critical'],
                'for': '5m'  # 持续5分钟触发
            })
        return rules
```

### 18.8.2 日志管理

```python
import logging
from logging.handlers import RotatingFileHandler
import json

class RAGLogger:
    """RAG日志管理器"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # 配置日志
        self.logger = logging.getLogger('rag')
        self.logger.setLevel(logging.INFO)
        
        # 文件处理器
        file_handler = RotatingFileHandler(
            self.log_dir / "rag.log",
            maxBytes=100 * 1024 * 1024,  # 100MB
            backupCount=10
        )
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(file_handler)
    
    def log_query(self, query_id: str, question: str,
                  answer: str, latency_ms: float,
                  retrieval_count: int, confidence: float):
        """记录查询日志"""
        log_entry = {
            'type': 'query',
            'query_id': query_id,
            'question': question,
            'answer_length': len(answer),
            'latency_ms': latency_ms,
            'retrieval_count': retrieval_count,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat()
        }
        self.logger.info(json.dumps(log_entry, ensure_ascii=False))
    
    def log_error(self, error_type: str, error_msg: str,
                   query_id: str = None, details: Dict = None):
        """记录错误日志"""
        log_entry = {
            'type': 'error',
            'error_type': error_type,
            'error_msg': error_msg,
            'query_id': query_id,
            'details': details,
            'timestamp': datetime.now().isoformat()
        }
        self.logger.error(json.dumps(log_entry, ensure_ascii=False))
    
    def log_performance(self, metric: str, value: float,
                         tags: Dict = None):
        """记录性能指标"""
        log_entry = {
            'type': 'performance',
            'metric': metric,
            'value': value,
            'tags': tags,
            'timestamp': datetime.now().isoformat()
        }
        self.logger.info(json.dumps(log_entry))
```

## 18.9 成本估算

### 18.9.1 成本模型

```python
class CostEstimator:
    """成本估算器"""
    
    def __init__(self):
        self.pricing = {
            'deepseek-chat': {
                'input_per_1k': 0.00027,
                'output_per_1k': 0.0011
            },
            'text-embedding-3-small': {
                'input_per_1k': 0.00002
            },
            'milvus': {
                'per_gb_month': 0.50
            },
            'server': {
                'per_cpu_month': 30,
                'per_gb_ram_month': 5
            }
        }
    
    def estimate_monthly_cost(self, 
                               queries_per_day: int,
                               avg_input_tokens: int,
                               avg_output_tokens: int,
                               embedding_dim: int = 1536,
                               doc_count: int = 10000) -> Dict:
        """估算月度成本"""
        daily_queries = queries_per_day
        monthly_queries = daily_queries * 30
        
        # LLM成本
        input_cost = (
            monthly_queries * avg_input_tokens / 1000 * 
            self.pricing['deepseek-chat']['input_per_1k']
        )
        output_cost = (
            monthly_queries * avg_output_tokens / 1000 * 
            self.pricing['deepseek-chat']['output_per_1k']
        )
        llm_cost = input_cost + output_cost
        
        # 嵌入成本
        embed_cost = (
            doc_count * 500 / 1000 *  # 假设每文档500 tokens
            self.pricing['text-embedding-3-small']['input_per_1k']
        )
        
        # 存储成本
        storage_cost = (
            doc_count * embedding_dim * 4 / 1024 / 1024 *  # 向量大小
            self.pricing['milvus']['per_gb_month']
        )
        
        # 服务器成本
        server_cost = (
            4 * self.pricing['server']['per_cpu_month'] +  # 4 CPU
            8 * self.pricing['server']['per_gb_ram_month']  # 8GB RAM
        )
        
        total = llm_cost + embed_cost + storage_cost + server_cost
        
        return {
            'monthly_queries': monthly_queries,
            'cost_breakdown': {
                'llm': round(llm_cost, 2),
                'embedding': round(embed_cost, 2),
                'storage': round(storage_cost, 2),
                'server': round(server_cost, 2)
            },
            'total_monthly': round(total, 2),
            'cost_per_query': round(total / monthly_queries, 4),
            'currency': 'USD'
        }
```

## 18.10 上线检查清单

### 18.10.1 生产就绪检查

```python
class ProductionReadinessCheck:
    """生产就绪检查"""
    
    def __init__(self):
        self.checks = []
    
    def add_check(self, category: str, name: str,
                   description: str, check_func: Callable):
        """添加检查项"""
        self.checks.append({
            'category': category,
            'name': name,
            'description': description,
            'check': check_func,
            'status': 'pending'
        })
    
    def run_all(self) -> Dict:
        """运行所有检查"""
        results = {
            'passed': 0,
            'failed': 0,
            'warnings': 0,
            'details': []
        }
        
        for check in self.checks:
            try:
                result = check['check']()
                check['status'] = 'passed' if result['ok'] else 'failed'
                
                if result['ok']:
                    results['passed'] += 1
                else:
                    results['failed'] += 1
                
                results['details'].append({
                    'category': check['category'],
                    'name': check['name'],
                    'status': check['status'],
                    'message': result.get('message', '')
                })
            except Exception as e:
                check['status'] = 'error'
                results['failed'] += 1
                results['details'].append({
                    'category': check['category'],
                    'name': check['name'],
                    'status': 'error',
                    'message': str(e)
                })
        
        results['ready'] = results['failed'] == 0
        
        return results

# 创建生产就绪检查
readiness = ProductionReadinessCheck()

# 功能检查
readiness.add_check('function', '检索功能', '确认检索服务正常', 
    lambda: {'ok': check_retrieval_service()})
readiness.add_check('function', '生成功能', '确认生成服务正常',
    lambda: {'ok': check_generation_service()})

# 性能检查
readiness.add_check('performance', '响应时间', '确认P99延迟达标',
    lambda: {'ok': check_latency_sla()})
readiness.add_check('performance', '并发处理', '确认并发能力达标',
    lambda: {'ok': check_concurrency()})

# 安全检查
readiness.add_check('security', '访问控制', '确认权限控制生效',
    lambda: {'ok': check_access_control()})
readiness.add_check('security', '数据加密', '确认数据传输加密',
    lambda: {'ok': check_encryption()})

# 运维检查
readiness.add_check('operations', '监控告警', '确认监控系统正常',
    lambda: {'ok': check_monitoring()})
readiness.add_check('operations', '备份策略', '确认备份策略就绪',
    lambda: {'ok': check_backup()})
readiness.add_check('operations', '日志系统', '确认日志系统正常',
    lambda: {'ok': check_logging()})
```

## 18.11 本章小结

本章通过一个完整的医药企业智能知识问答系统案例，展示了RAG项目从需求分析到生产部署的全流程。

**需求分析**阶段明确了项目的目标和约束，通过用例定义和非功能需求分析，为后续的架构设计提供了清晰的输入。

**架构设计**阶段采用分层架构，将系统分为接入层、网关层、编排层、服务层和数据层。通过组件选择评估框架，基于性能、成本、易用性等维度选择了最合适的组件组合。架构决策记录（ADR）确保了设计决策的可追溯性。

**核心实现**包括数据处理流水线（清洗、分块）、RAG核心引擎（嵌入、检索、融合、生成）和API服务（FastAPI）。混合检索策略通过RRF融合向量检索和关键词检索的结果，提高了检索召回率。

**测试策略**采用测试金字塔方法，覆盖单元测试、集成测试和评估测试。自动化测试流水线在CI/CD中确保每次变更都经过充分验证。

**部署方案**提供了Docker化、Docker Compose和Kubernetes三种部署方式，适应不同规模的需求。Kubernetes部署支持自动扩缩容，确保系统能够应对流量波动。

**CI/CD流水线**通过GitHub Actions实现了从代码提交到生产部署的自动化。代码经过lint、类型检查、单元测试、集成测试后自动构建和部署。

**监控告警**覆盖延迟、QPS、错误率等关键指标，配合日志管理，确保系统运行状态可观测。

**成本估算**帮助团队在项目初期就了解运营成本，做好预算规划。基于DeepSeek的LLM方案相比GPT-4可以节省约60%的成本。

**上线检查清单**确保系统在进入生产环境前满足所有功能、性能、安全和运维要求。

在实际项目中，建议参考本章的方法论，但根据具体的业务需求、团队规模和技术栈进行调整。最重要的是建立"测量-优化-验证"的持续改进循环，让RAG系统在运行过程中不断进化。
