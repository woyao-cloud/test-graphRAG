# 第14章: 高级RAG架构：基于Milvus的进阶落地方案

## 14.1 引言

基础的RAG架构（检索+生成）虽然简单有效，但在面对复杂查询、多跳推理、大规模知识库等场景时，往往表现不佳。近年来，学术界和工业界提出了多种高级RAG架构，通过引入多级检索、子问题拆解、多路召回、多模态融合等技术，显著提升了RAG系统的能力边界。本章将深入探讨这些高级RAG架构，并结合Milvus提供完整的实现方案。

## 14.2 多级RAG：粗检索与精检索

### 14.2.1 两阶段检索架构

多级RAG（Multi-stage RAG）的核心思想是将检索过程分为两个阶段：第一阶段使用轻量级方法快速召回候选文档，第二阶段使用更精确的方法对候选文档进行精细化筛选。

```python
from pymilvus import Collection
from sentence_transformers import CrossEncoder, SentenceTransformer
from typing import List, Dict, Tuple
import numpy as np

class TwoStageRetriever:
    """两阶段检索器：粗检索 + 精检索"""
    
    def __init__(
        self,
        collection: Collection,
        coarse_model_name: str = "BAAI/bge-small-zh-v1.5",
        fine_model_name: str = "BAAI/bge-reranker-v2-m3",
        coarse_top_k: int = 50,
        fine_top_k: int = 10
    ):
        self.collection = collection
        
        # 粗检索使用轻量级嵌入模型
        self.coarse_encoder = SentenceTransformer(coarse_model_name)
        
        # 精检索使用交叉编码器重排序
        self.fine_reranker = CrossEncoder(fine_model_name)
        
        self.coarse_top_k = coarse_top_k
        self.fine_top_k = fine_top_k
    
    def search(self, query: str) -> List[Dict]:
        """两阶段检索"""
        
        # Stage 1: 粗检索 - 快速召回
        query_embedding = self.coarse_encoder.encode(query).tolist()
        
        coarse_results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param={"metric_type": "IP", "params": {"nprobe": 32}},
            limit=self.coarse_top_k,
            output_fields=["id", "text", "embedding"]
        )
        
        candidates = []
        for hits in coarse_results:
            for hit in hits:
                candidates.append({
                    "id": hit.id,
                    "text": hit.entity.get("text"),
                    "vector_score": hit.score
                })
        
        if not candidates:
            return []
        
        # Stage 2: 精检索 - 交叉编码器重排序
        pairs = [[query, doc["text"]] for doc in candidates]
        rerank_scores = self.fine_reranker.predict(pairs)
        
        for doc, score in zip(candidates, rerank_scores):
            doc["rerank_score"] = float(score)
        
        # 按精检索得分排序
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        return candidates[:self.fine_top_k]
```

### 14.2.2 三级检索架构

对于超大规模知识库，可以引入三级架构：轻量级嵌入检索 -> 中等规模嵌入检索 -> 交叉编码器重排序。

```python
class ThreeStageRetriever:
    """三级检索器"""
    
    def __init__(self, collections: Dict[str, Collection], models: Dict):
        """
        Args:
            collections: {
                "tiny": 轻量级集合（所有文档的摘要嵌入）
                "medium": 中等集合（段落嵌入）
                "full": 完整集合（详细文本）
            }
        """
        self.collections = collections
        self.models = models
    
    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        # Stage 1: 在摘要集合中快速检索
        tiny_emb = self.models['tiny'].encode(query).tolist()
        stage1 = self.collections['tiny'].search(
            data=[tiny_emb],
            anns_field="embedding",
            param={"metric_type": "IP", "params": {"nprobe": 16}},
            limit=200,  # 宽召回
            output_fields=["id", "doc_group_id"]
        )
        
        # 收集候选文档组
        candidate_group_ids = set()
        for hits in stage1:
            for hit in hits:
                candidate_group_ids.add(hit.entity.get("doc_group_id"))
        
        # Stage 2: 在段落集合中精细检索
        medium_emb = self.models['medium'].encode(query).tolist()
        group_filter = f"doc_group_id in {list(candidate_group_ids)}"
        
        stage2 = self.collections['medium'].search(
            data=[medium_emb],
            anns_field="embedding",
            param={"metric_type": "IP", "params": {"nprobe": 32}},
            limit=50,
            expr=group_filter,
            output_fields=["id", "text", "doc_group_id"]
        )
        
        candidates = []
        for hits in stage2:
            for hit in hits:
                candidates.append({
                    "id": hit.id,
                    "text": hit.entity.get("text"),
                    "vector_score": hit.score
                })
        
        # Stage 3: 交叉编码器最终排序
        pairs = [[query, d["text"]] for d in candidates]
        final_scores = self.models['reranker'].predict(pairs)
        
        for doc, score in zip(candidates, final_scores):
            doc["final_score"] = float(score)
        
        candidates.sort(key=lambda x: x["final_score"], reverse=True)
        return candidates[:top_k]
```

## 14.3 子问题拆解与多路召回

### 14.3.1 复杂查询的子问题拆解

对于需要多步推理的复杂问题，将其拆解为多个子问题分别检索，然后合并结果，可以显著提升回答质量。

```python
import json
from typing import List
import openai  # 或其他 LLM API

class QueryDecomposer:
    """查询拆解器：将复杂问题拆解为子问题"""
    
    def __init__(self, llm_client, model: str = "gpt-4o"):
        self.llm = llm_client
        self.model = model
    
    def decompose(self, complex_query: str) -> List[str]:
        """将复杂问题拆解为独立的子问题"""
        
        prompt = f"""请将以下复杂问题拆解为若干个独立的子问题。
每个子问题应该能够单独通过检索知识库来回答。
返回格式：JSON 数组 ["子问题1", "子问题2", ...]

复杂问题：{complex_query}

子问题："""
        
        response = self.llm.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        
        try:
            sub_queries = json.loads(response.choices[0].message.content)
            return sub_queries
        except:
            # 解析失败时返回原问题
            return [complex_query]


class MultiPathRetriever:
    """多路召回检索器"""
    
    def __init__(self, collection: Collection, retriever):
        self.collection = collection
        self.retriever = retriever
        self.decomposer = None  # 由外部设置
    
    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        # 拆解子问题
        sub_queries = self.decomposer.decompose(query)
        
        # 多路召回
        all_results = []
        for sub_q in sub_queries:
            results = self.retriever.search(sub_q, top_k=top_k)
            for r in results:
                r["sub_query"] = sub_q
            all_results.extend(results)
        
        # 合并去重
        seen_ids = set()
        merged = []
        for r in all_results:
            if r["id"] not in seen_ids:
                seen_ids.add(r["id"])
                merged.append(r)
        
        # 重新排序（保留每个子问题的最佳结果）
        merged.sort(key=lambda x: x.get("rerank_score", x.get("vector_score", 0)),
                   reverse=True)
        
        return merged[:top_k]
```

### 14.3.2 多路召回融合策略

不同检索路可能返回不同的候选文档，需要有效的融合策略。

```python
class FusionStrategy:
    """多路召回结果融合策略"""
    
    @staticmethod
    def reciprocal_rank_fusion(
        rankings: List[List[Dict]],
        k: int = 60
    ) -> List[Dict]:
        """
        使用 Reciprocal Rank Fusion (RRF) 算法融合多路召回结果
        
        RRF 得分 = sum(1 / (k + rank_i))
        """
        scores = {}
        doc_map = {}
        
        for rank_list in rankings:
            for rank, doc in enumerate(rank_list, 1):
                doc_id = doc["id"]
                if doc_id not in scores:
                    scores[doc_id] = 0.0
                    doc_map[doc_id] = doc
                scores[doc_id] += 1.0 / (k + rank)
        
        # 按 RRF 得分排序
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        return [doc_map[doc_id] for doc_id, _ in ranked]
    
    @staticmethod
    def weighted_fusion(
        path_results: List[Tuple[List[Dict], float]],
        top_k: int = 10
    ) -> List[Dict]:
        """
        加权融合多路召回结果
        
        Args:
            path_results: [(results, weight), ...]
        """
        scores = {}
        doc_map = {}
        
        for results, weight in path_results:
            for doc in results:
                doc_id = doc["id"]
                if doc_id not in scores:
                    scores[doc_id] = 0.0
                    doc_map[doc_id] = doc
                scores[doc_id] += doc.get("vector_score", 0) * weight
        
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_map[doc_id] for doc_id, _ in ranked[:top_k]]
```

### 14.3.3 完整的多路召回流水线

```python
class AdvancedMultiPathRAG:
    """高级多路召回 RAG 系统"""
    
    def __init__(self, collection: Collection, retriever, llm_client):
        self.collection = collection
        self.retriever = retriever
        self.llm = llm_client
        self.decomposer = QueryDecomposer(llm_client)
        self.retriever.decomposer = self.decomposer
        self.fusion = FusionStrategy()
    
    def query(self, complex_query: str, top_k: int = 10) -> Dict:
        # Step 1: 拆解子问题
        sub_queries = self.decomposer.decompose(complex_query)
        
        # Step 2: 对每个子问题执行多路检索
        # 路径1: 语义向量检索
        path1_results = []
        for sub_q in sub_queries:
            emb = self.retriever.coarse_encoder.encode(sub_q).tolist()
            results = self.collection.search(
                data=[emb],
                anns_field="embedding",
                param={"metric_type": "IP", "params": {"nprobe": 32}},
                limit=top_k,
                output_fields=["id", "text"]
            )
            for hits in results:
                for hit in hits:
                    path1_results.append({
                        "id": hit.id,
                        "text": hit.entity.get("text"),
                        "vector_score": hit.score,
                        "sub_query": sub_q
                    })
        
        # 路径2: 关键词检索（通过标量过滤）
        path2_results = []  # 关键词匹配结果
        
        # 路径3: 元数据过滤检索
        path3_results = []  # 根据子问题的实体过滤
        
        # Step 3: 融合多路结果
        all_rankings = [path1_results, path2_results, path3_results]
        all_rankings = [r for r in all_rankings if r]  # 过滤空结果
        fused_results = self.fusion.reciprocal_rank_fusion(all_rankings)
        
        # Step 4: 重排序
        pairs = [[complex_query, d["text"]] for d in fused_results[:top_k * 2]]
        rerank_scores = self.retriever.fine_reranker.predict(pairs)
        for doc, score in zip(fused_results[:top_k * 2], rerank_scores):
            doc["final_score"] = float(score)
        
        fused_results.sort(key=lambda x: x.get("final_score", 0), reverse=True)
        final_docs = fused_results[:top_k]
        
        # Step 5: 生成回答
        context = "\n\n".join([
            f"[文档{i+1}] {d['text']}"
            for i, d in enumerate(final_docs)
        ])
        
        response = self.llm.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "基于以下检索文档回答问题。"},
                {"role": "user", "content": f"问题：{complex_query}\n\n参考文档：\n{context}"}
            ]
        )
        
        return {
            "answer": response.choices[0].message.content,
            "sub_queries": sub_queries,
            "documents": final_docs,
            "context": context
        }
```

## 14.4 上下文窗口优化

### 14.4.1 上下文压缩

LLM的上下文窗口有限，通过压缩检索结果可以塞入更多有用信息。

```python
class ContextCompressor:
    """上下文压缩器：精简检索结果以适配上下文窗口"""
    
    def __init__(self, llm_client, max_tokens: int = 4000):
        self.llm = llm_client
        self.max_tokens = max_tokens
    
    def compress(self, query: str, documents: List[Dict]) -> List[Dict]:
        """压缩文档，保留与查询最相关的部分"""
        compressed = []
        total_tokens = 0
        
        for doc in documents:
            if len(doc['text']) < 100:
                # 短文档直接保留
                compressed.append(doc)
                total_tokens += len(doc['text'])
                continue
            
            # 使用 LLM 提取与查询相关的关键信息
            prompt = f"""查询：{query}

文档内容：{doc['text'][:2000]}

请从上述文档中提取与查询最相关的关键信息（不超过200字），
保持原始信息准确，不要添加新内容。"""
            
            response = self.llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300
            )
            
            summary = response.choices[0].message.content
            doc['compressed_text'] = summary
            compressed.append(doc)
            total_tokens += len(summary)
            
            if total_tokens > self.max_tokens * 2:  # 粗略估计
                break
        
        return compressed
```

### 14.4.2 滑动窗口上下文构建

对于需要多文档上下文的长文本场景，使用滑动窗口策略构建上下文。

```python
class SlidingWindowContext:
    """滑动窗口上下文构建器"""
    
    def __init__(self, max_chars: int = 8000, overlap_chars: int = 200):
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars
    
    def build_context(
        self,
        query: str,
        documents: List[Dict],
        sort_by_relevance: bool = True
    ) -> str:
        """构建适合 LLM 输入的上下文"""
        if sort_by_relevance:
            documents = sorted(
                documents,
                key=lambda x: x.get("rerank_score", x.get("vector_score", 0)),
                reverse=True
            )
        
        context_parts = []
        current_length = 0
        
        for doc in documents:
            doc_text = doc.get("compressed_text", doc.get("text", ""))
            
            if current_length + len(doc_text) > self.max_chars:
                # 截断最后一个文档
                remaining = self.max_chars - current_length - self.overlap_chars
                if remaining > 100:
                    context_parts.append(doc_text[:remaining])
                break
            
            context_parts.append(doc_text)
            current_length += len(doc_text)
        
        return "\n\n---\n\n".join(context_parts)
```

## 14.5 多模态RAG

### 14.5.1 图文混合检索

在Milvus中，可以将文本向量和图像向量存储在同一个集合中，实现图文混合检索。

```python
from pymilvus import Collection, CollectionSchema, FieldSchema, DataType

class MultiModalRetriever:
    """多模态检索器"""
    
    def __init__(
        self,
        collection: Collection,
        text_encoder,
        image_encoder,
        llm_client
    ):
        self.collection = collection
        self.text_encoder = text_encoder  # 如 BGE-M3
        self.image_encoder = image_encoder  # 如 CLIP ViT
        self.llm = llm_client
    
    def search_by_text(self, query: str, top_k: int = 10) -> List[Dict]:
        """文本查询多模态知识库"""
        text_emb = self.text_encoder.encode(query).tolist()
        
        results = self.collection.search(
            data=[text_emb],
            anns_field="text_embedding",
            param={"metric_type": "IP", "params": {"nprobe": 32}},
            limit=top_k,
            output_fields=["id", "text", "image_path", "modality"]
        )
        
        return [
            {
                "id": hit.id,
                "text": hit.entity.get("text"),
                "image_path": hit.entity.get("image_path"),
                "modality": hit.entity.get("modality"),
                "score": hit.score
            }
            for hits in results for hit in hits
        ]
    
    def search_by_image(self, image_path: str, top_k: int = 10) -> List[Dict]:
        """以图搜文 / 以图搜图"""
        image_emb = self.image_encoder.encode(image_path).tolist()
        
        # 在图像向量字段中检索
        results = self.collection.search(
            data=[image_emb],
            anns_field="image_embedding",
            param={"metric_type": "IP", "params": {"nprobe": 32}},
            limit=top_k,
            output_fields=["id", "text", "image_path"]
        )
        
        return [
            {
                "id": hit.id,
                "text": hit.entity.get("text"),
                "image_path": hit.entity.get("image_path"),
                "score": hit.score
            }
            for hits in results for hit in hits
        ]
    
    def multimodal_rag(
        self,
        query: str,
        top_k: int = 5
    ) -> Dict:
        """多模态 RAG 问答"""
        results = self.search_by_text(query, top_k=top_k)
        
        # 构建包含图文信息的上下文
        context_parts = []
        for r in results:
            part = f"[文本] {r['text']}"
            if r.get('image_path'):
                part += f"\n[图片] {r['image_path']}"
            context_parts.append(part)
        
        context = "\n\n".join(context_parts)
        
        response = self.llm.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system",
                 "content": "你是多模态AI助手。基于提供的文本和图片信息回答问题。"},
                {"role": "user",
                 "content": f"问题：{query}\n\n参考信息：\n{context}"}
            ]
        )
        
        return {
            "answer": response.choices[0].message.content,
            "retrieved_docs": results
        }
```

### 14.5.2 多模态知识库 Schema 设计

```python
# 多模态集合的 Schema 设计
multi_modal_schema = CollectionSchema([
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    
    # 文本嵌入（来自文本编码器）
    FieldSchema(name="text_embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
    
    # 图像嵌入（来自图像编码器）
    FieldSchema(name="image_embedding", dtype=DataType.FLOAT_VECTOR, dim=512),
    
    # 元数据
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=512),
    FieldSchema(name="modality", dtype=DataType.VARCHAR, max_length=16),  # text, image, mixed
    FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="timestamp", dtype=DataType.INT64),
])
```

## 14.6 Agent+RAG

### 14.6.1 智能Agent驱动的RAG

将Agent的规划能力与RAG的检索能力结合，让Agent自主决定何时检索、检索什么以及如何使用检索结果。

```python
from typing import Optional
import json

class AgentRAG:
    """Agent驱动的RAG系统"""
    
    def __init__(self, collection: Collection, llm_client, retriever):
        self.collection = collection
        self.llm = llm_client
        self.retriever = retriever
        self.conversation_history = []
    
    def _agent_think(self, user_query: str) -> Dict:
        """Agent 思考：决定下一步行动"""
        tools_desc = """
可用工具：
1. search_knowledge(query: str) - 在知识库中检索信息
2. get_document(id: int) - 获取指定文档的完整内容
3. filter_by_category(category: str, query: str) - 在特定类别中检索
4. final_answer(answer: str) - 给出最终答案
"""
        
        prompt = f"""你是一个智能检索Agent。你的任务是根据用户问题，决定需要执行哪些操作。

历史对话：{self.conversation_history[-5:] if self.conversation_history else "无"}

用户问题：{user_query}

可用工具：{tools_desc}

请分析问题，返回 JSON 格式的下一步行动：
{{
    "thought": "你的思考过程",
    "action": "工具名称",
    "action_input": {{"param": "value"}},
    "need_more_info": true/false
}}"""
        
        response = self.llm.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        
        try:
            return json.loads(response.choices[0].message.content)
        except:
            return {"action": "final_answer", "action_input": {},
                    "thought": "解析失败，直接回答"}
    
    def _execute_action(self, action: Dict) -> any:
        """执行 Agent 决策的行动"""
        action_type = action.get("action")
        action_input = action.get("action_input", {})
        
        if action_type == "search_knowledge":
            return self.retriever.search(
                action_input.get("query", ""),
                top_k=5
            )
        elif action_type == "get_document":
            doc_id = action_input.get("id")
            return self.collection.query(
                expr=f'id == {doc_id}',
                output_fields=["id", "text", "source"]
            )
        elif action_type == "filter_by_category":
            return self.retriever.search(
                action_input.get("query", ""),
                filters={"category": action_input.get("category")},
                top_k=5
            )
        elif action_type == "final_answer":
            return action_input.get("answer", "")
        else:
            return None
    
    def query(self, user_query: str, max_steps: int = 5) -> Dict:
        """Agent 驱动的多步 RAG 查询"""
        step_count = 0
        collected_info = []
        
        while step_count < max_steps:
            # Agent 思考
            action = self._agent_think(user_query)
            
            # 执行行动
            result = self._execute_action(action)
            
            if action.get("action") == "final_answer":
                # Agent 决定给出最终答案
                final_answer = result if isinstance(result, str) else str(result)
                return {
                    "answer": final_answer,
                    "steps": step_count,
                    "collected_info": collected_info
                }
            
            # 记录检索结果
            if result:
                collected_info.extend(result if isinstance(result, list) else [result])
                # 将结果加入对话历史
                self.conversation_history.append(
                    f"检索结果：{json.dumps([{'id': r.get('id'), 'text': r.get('text', '')[:200]} for r in (result if isinstance(result, list) else [result])], ensure_ascii=False)}"
                )
            
            step_count += 1
        
        # 达到最大步数，使用已收集信息生成答案
        context = "\n\n".join([
            f"[{i+1}] {r.get('text', str(r))}"
            for i, r in enumerate(collected_info[:10])
        ])
        
        response = self.llm.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system",
                 "content": "基于以下检索结果回答用户问题。"},
                {"role": "user",
                 "content": f"问题：{user_query}\n\n检索结果：\n{context}"}
            ]
        )
        
        return {
            "answer": response.choices[0].message.content,
            "steps": step_count,
            "collected_info": collected_info
        }
```

### 14.6.2 反思与自我纠错

Agent可以反思自己的检索结果，判断是否需要补充检索或修正回答。

```python
class ReflectiveAgentRAG(AgentRAG):
    """具备反思能力的 Agent RAG"""
    
    def _reflect(self, query: str, collected_info: List[Dict]) -> Dict:
        """反思当前检索结果是否充分"""
        
        context_summary = "\n".join([
            f"- {r.get('text', '')[:200]}"
            for r in collected_info[:5]
        ])
        
        prompt = f"""反思以下检索结果是否足够回答用户问题。

用户问题：{query}

已检索到的信息：
{context_summary}

请评估：
1. 这些信息是否足够回答用户问题？（是/否）
2. 如果不足，还需要哪些补充信息？
3. 当前信息是否存在矛盾？

返回 JSON：
{{
    "sufficient": true/false,
    "gaps": ["需要补充的信息1", "需要补充的信息2"],
    "contradictions": ["矛盾点1"],
    "next_query": "如果需要补充检索，应该搜索什么"
}}"""
        
        response = self.llm.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        
        try:
            return json.loads(response.choices[0].message.content)
        except:
            return {"sufficient": True, "gaps": [], "contradictions": []}
    
    def query(self, user_query: str, max_steps: int = 8) -> Dict:
        result = super().query(user_query, max_steps)
        
        # 反思阶段
        reflection = self._reflect(user_query, result.get("collected_info", []))
        
        if not reflection.get("sufficient") and len(result.get("collected_info", [])) > 0:
            # 补充检索
            if reflection.get("next_query"):
                extra = self.retriever.search(reflection["next_query"], top_k=3)
                result["collected_info"].extend(extra)
                # 重新生成回答
                context = "\n\n".join([
                    f"[{i+1}] {r.get('text', str(r))}"
                    for i, r in enumerate(result["collected_info"][:10])
                ])
                response = self.llm.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "基于以下信息回答问题。"},
                        {"role": "user",
                         "content": f"问题：{user_query}\n\n信息：\n{context}"}
                    ]
                )
                result["answer"] = response.choices[0].message.content
        
        result["reflection"] = reflection
        return result
```

## 14.7 综合实践：企业级高级RAG系统

```python
class EnterpriseAdvancedRAG:
    """企业级高级RAG系统"""
    
    def __init__(self, milvus_config: dict, llm_config: dict):
        # 初始化 Milvus 连接和集合
        self.collections = self._init_milvus(milvus_config)
        
        # 初始化检索器
        self.retriever = TwoStageRetriever(
            collection=self.collections['main'],
            coarse_model_name="BAAI/bge-small-zh-v1.5",
            fine_model_name="BAAI/bge-reranker-v2-m3"
        )
        
        # 初始化 Agent
        self.agent = ReflectiveAgentRAG(
            collection=self.collections['main'],
            llm_client=llm_config['client'],
            retriever=self.retriever
        )
        
        # 初始化多模态检索
        self.multi_modal = MultiModalRetriever(
            collection=self.collections.get('multimodal'),
            text_encoder=None,
            image_encoder=None,
            llm_client=llm_config['client']
        )
    
    def query(self, question: str, mode: str = "auto") -> Dict:
        """
        根据问题类型自动选择最佳策略
        
        Args:
            question: 用户问题
            mode: auto | direct | multi_step | multi_modal
        """
        if mode == "auto":
            mode = self._detect_query_mode(question)
        
        if mode == "direct":
            # 简单的单次检索
            docs = self.retriever.search(question, top_k=5)
            return {"answer": self._generate_answer(question, docs), "mode": mode}
        
        elif mode == "multi_step":
            # 多步 Agent 检索
            return {**self.agent.query(question), "mode": mode}
        
        elif mode == "multi_modal":
            # 多模态检索
            return {**self.multi_modal.multimodal_rag(question), "mode": mode}
    
    def _detect_query_mode(self, question: str) -> str:
        """检测问题类型，选择最佳检索模式"""
        prompt = f"""分析以下问题，判断最佳回答策略：

问题：{question}

请选择：
- "direct": 简单事实性问题，单次检索即可
- "multi_step": 复杂问题，需要多步推理和检索
- "multi_modal": 涉及图像或多模态内容

返回 JSON：{{"mode": "direct" | "multi_step" | "multi_modal"}}"""
        
        response = self.agent.llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        
        try:
            return json.loads(response.choices[0].message.content)["mode"]
        except:
            return "direct"
```

## 14.8 本章小结

本章深入探讨了基于Milvus的高级RAG架构，涵盖多级检索（粗检索+精检索）、子问题拆解与多路召回、上下文窗口优化、多模态RAG以及Agent+RAG等进阶方案。这些高级架构大幅提升了RAG系统在复杂场景下的能力：

1. **多级检索**通过分阶段筛选，在效率和精度之间取得平衡
2. **子问题拆解与多路召回**将复杂问题分解为可检索的子问题，提升回答的完整性
3. **上下文优化**确保在有限的上下文窗口内塞入最相关的信息
4. **多模态RAG**扩展了知识库的信息类型，支持图文混合检索
5. **Agent+RAG**赋予系统自主规划、反思和纠错的能力

在实际项目中，建议根据业务场景的复杂度，选择合适的架构组合，从小规模验证开始，逐步演进到企业级高级RAG系统。

下一章将探讨Milvus性能调优，重点关注高并发生产场景下的优化策略。
