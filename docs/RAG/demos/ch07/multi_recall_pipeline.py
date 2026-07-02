"""
第 7 章 Demo：多路召回管线

演示完整的多路召回 + 重排融合流程：
  Vector Recall → Keyword Recall → Score Fusion → Reranker
可独立运行，使用本地 mock 数据。

用法：
  python multi_recall_pipeline.py
  python multi_recall_pipeline.py --query "紫杉醇的供应链"
"""

import argparse
import json
import math
import re
import time
from dataclasses import dataclass, field
from typing import Optional


# ============================================================================
# Data Models
# ============================================================================


@dataclass
class SearchResult:
    id: str
    content: str
    source: str
    score: float = 0.0
    normalized_score: float = 0.0
    metadata: dict = field(default_factory=dict)


# ============================================================================
# Sample Knowledge Base
# ============================================================================

SAMPLE_DOCS = [
    {"id": "d1", "title": "恒瑞医药产品目录", "dept": "研发", "type": "产品手册",
     "content": "恒瑞医药是中国领先的制药企业，主要生产抗肿瘤药物。核心产品包括注射用紫杉醇、阿帕替尼、卡瑞利珠单抗。"},
    {"id": "d2", "title": "紫杉醇供应链分析", "dept": "供应链", "type": "分析报告",
     "content": "紫杉醇API由华海药业供应，恒瑞医药加工为成品，通过国药控股分销至北京协和医院等医疗机构。"},
    {"id": "d3", "title": "国药控股区域覆盖", "dept": "销售", "type": "运营数据",
     "content": "国药控股在华东区设有12个配送中心，主要合作企业包括恒瑞医药、齐鲁制药，2023年分销额超200亿元。"},
    {"id": "d4", "title": "北京协和医院采购记录", "dept": "采购", "type": "采购数据",
     "content": "2023年抗肿瘤药物采购量同比增长35%。主要供应商为恒瑞医药（紫杉醇）和齐鲁制药（吉非替尼）。"},
    {"id": "d5", "title": "抗肿瘤药物市场分析", "dept": "市场", "type": "行业报告",
     "content": "中国抗肿瘤药物市场规模2023年达2500亿元。紫杉醇类药物占市场份额约15%，年复合增长率12%。"},
    {"id": "d6", "title": "药品质量管理规范", "dept": "质量", "type": "制度规范",
     "content": "药品生产须符合GMP标准。原料药供应商需通过现场审计，成品需进行批次留样和稳定性考察。"},
    {"id": "d7", "title": "华海药业公司概况", "dept": "采购", "type": "供应商档案",
     "content": "华海药业是国内最大的紫杉醇API供应商之一，年产能5000kg，客户包括恒瑞医药、齐鲁制药等。"},
    {"id": "d8", "title": "华东区药品分销网络", "dept": "供应链", "type": "运营数据",
     "content": "华东区药品分销网络覆盖江浙沪三省市37家三甲医院。国药控股占华东市场份额约40%。"},
]


# ============================================================================
# Vector Retriever — 模拟向量检索
# ============================================================================


class MockVectorRetriever:
    """模拟向量检索（基于关键词重叠的简单相似度）。"""

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """基于 Query 词重叠的模拟向量检索。"""
        query_terms = set(self._tokenize(query))
        scored = []

        for doc in SAMPLE_DOCS:
            doc_terms = set(self._tokenize(doc["content"]))
            overlap = len(query_terms & doc_terms)
            if overlap > 0:
                # 模拟 cosine similarity [0, 1]
                sim = overlap / (math.sqrt(len(query_terms)) * math.sqrt(len(doc_terms)))
                scored.append(SearchResult(
                    id=doc["id"],
                    content=doc["content"],
                    source="vector",
                    score=round(sim, 4),
                    metadata={"title": doc["title"], "dept": doc["dept"]},
                ))

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]

    def _tokenize(self, text: str) -> list[str]:
        # 简单位置分词
        return re.findall(r"[\w\u4e00-\u9fff]+", text)


# ============================================================================
# Keyword Retriever (BM25)
# ============================================================================


class BM25Retriever:
    """BM25 关键词检索。"""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self._build_index()

    def _build_index(self):
        """构建倒排索引。"""
        self.doc_len = {}
        self.avgdl = 0
        self.df = {}  # document frequency
        self.tf = {}  # term frequency per doc

        total_len = 0
        for doc in SAMPLE_DOCS:
            tokens = self._tokenize(doc["content"])
            doc_id = doc["id"]
            self.doc_len[doc_id] = len(tokens)
            total_len += len(tokens)

            # TF
            term_counts = {}
            for t in tokens:
                term_counts[t] = term_counts.get(t, 0) + 1
            self.tf[doc_id] = term_counts

            # DF
            for t in set(tokens):
                self.df[t] = self.df.get(t, 0) + 1

        self.N = len(SAMPLE_DOCS)
        self.avgdl = total_len / self.N if self.N > 0 else 1

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        query_tokens = self._tokenize(query)
        scores = {}

        for doc in SAMPLE_DOCS:
            doc_id = doc["id"]
            doc_len = self.doc_len[doc_id]
            score = 0
            seen = set()
            for q_token in query_tokens:
                if q_token in seen:
                    continue
                seen.add(q_token)
                if q_token in self.df:
                    # IDF
                    idf = math.log((self.N - self.df[q_token] + 0.5) / (self.df[q_token] + 0.5) + 1)
                    # BM25 TF
                    tf = self.tf[doc_id].get(q_token, 0)
                    tf_norm = tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl))
                    score += idf * tf_norm

            if score > 0:
                scores[doc_id] = round(score, 4)

        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for doc_id, score in sorted_docs:
            doc = next(d for d in SAMPLE_DOCS if d["id"] == doc_id)
            results.append(SearchResult(
                id=doc_id, content=doc["content"], source="keyword",
                score=score, metadata={"title": doc["title"]},
            ))
        return results

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"[\w\u4e00-\u9fff]+", text)


# ============================================================================
# Structured Filter Retriever
# ============================================================================


class StructuredRetriever:
    """基于业务属性的结构化检索。"""

    def search(self, query: str, dept: Optional[str] = None, top_k: int = 5) -> list[SearchResult]:
        query_terms = set(re.findall(r"[\w\u4e00-\u9fff]+", query))
        scored = []
        for doc in SAMPLE_DOCS:
            if dept and doc["dept"] != dept:
                continue
            doc_terms = set(re.findall(r"[\w\u4e00-\u9fff]+", doc["content"]))
            overlap = len(query_terms & doc_terms)
            if overlap > 0:
                similarity = overlap / (math.sqrt(len(query_terms)) * math.sqrt(len(doc_terms)))
                scored.append(SearchResult(
                    id=doc["id"], content=doc["content"], source=f"structured(dept={doc['dept']})",
                    score=round(similarity, 4), metadata={"title": doc["title"], "dept": doc["dept"]},
                ))
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]


# ============================================================================
# KNN Retriever (Simulated)
# ============================================================================


class KGRetriever:
    """模拟知识图谱检索（基于实体关系）。"""

    def __init__(self):
        # 简单实体-关系图
        self.entity_relations = {
            "恒瑞医药": ["生产:注射用紫杉醇", "供应:华海药业", "分销:国药控股"],
            "紫杉醇": ["生产:恒瑞医药", "原料药:华海药业", "分销:国药控股"],
            "国药控股": ["分销:恒瑞医药", "分销:齐鲁制药", "覆盖:华东区"],
        }

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        # 从 query 中识别实体
        entities = [e for e in self.entity_relations if e in query]
        if not entities:
            return []

        # 获取相关关系文本
        related = set()
        for entity in entities:
            for rel in self.entity_relations[entity]:
                related.add(rel)

        # 匹配文档
        results = []
        for doc in SAMPLE_DOCS:
            doc_entities = [e for e in self.entity_relations if e in doc["content"]]
            if doc_entities:
                results.append(SearchResult(
                    id=doc["id"], content=doc["content"], source="kg",
                    score=0.8, metadata={
                        "title": doc["title"],
                        "matched_entities": doc_entities,
                        "relations": list(related),
                    },
                ))

        return results[:top_k]


# ============================================================================
# Score Fusion & Reranker
# ============================================================================


class ScoreFusion:
    """多路分数融合。"""

    @staticmethod
    def normalize(results: list[SearchResult]) -> list[SearchResult]:
        """Min-Max 归一化。"""
        if not results:
            return results
        scores = [r.score for r in results]
        min_s, max_s = min(scores), max(scores)
        if max_s == min_s:
            for r in results:
                r.normalized_score = 0.5
        else:
            for r in results:
                r.normalized_score = (r.score - min_s) / (max_s - min_s)
        return results

    @staticmethod
    def weighted_fusion(
        vector_results: list[SearchResult],
        keyword_results: list[SearchResult],
        struct_results: list[SearchResult],
        kg_results: list[SearchResult],
        weights: dict[str, float] = None,
    ) -> list[SearchResult]:
        """加权融合。"""
        w = weights or {"vector": 0.35, "keyword": 0.30, "struct": 0.15, "kg": 0.20}

        # 归一化
        ScoreFusion.normalize(vector_results)
        ScoreFusion.normalize(keyword_results)
        ScoreFusion.normalize(struct_results)
        ScoreFusion.normalize(kg_results)

        # 按 ID 融合
        fused = {}
        for name, results in [("vector", vector_results), ("keyword", keyword_results),
                              ("struct", struct_results), ("kg", kg_results)]:
            weight = w.get(name, 0.25)
            for r in results:
                if r.id not in fused:
                    r.score = 0
                    fused[r.id] = r
                fused[r.id].score += r.normalized_score * weight

        return sorted(fused.values(), key=lambda x: x.score, reverse=True)


class Reranker:
    """模拟 Cross-Encoder 重排。"""

    def rerank(self, query: str, results: list[SearchResult], top_k: int = 3) -> list[SearchResult]:
        """Cross-Encoder 风格重排（基于 query 与 content 的交互匹配）。"""
        query_terms = set(re.findall(r"[\w\u4e00-\u9fff]+", query))

        for r in results:
            # 计算交互匹配分数
            content_terms = set(re.findall(r"[\w\u4e00-\u9fff]+", r.content))
            exact_match = len(query_terms & content_terms)

            # 短语匹配加分
            phrase_bonus = 0
            for term in query_terms:
                if term in r.content and len(term) > 1:
                    phrase_bonus += 2
                elif term in r.content:
                    phrase_bonus += 1

            r.score = r.score * 0.5 + (exact_match + phrase_bonus) / 20 * 0.5

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]


# ============================================================================
# Main Pipeline
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="多路召回管线演示")
    parser.add_argument("--query", default="恒瑞医药的紫杉醇供应链", help="查询")
    parser.add_argument("--top-k", type=int, default=3, help="最终 Top-K")
    args = parser.parse_args()

    query = args.query
    top_k = args.top_k

    print("=" * 65)
    print(f"[Query] {query}")
    print("=" * 65)

    t0 = time.time()

    # 1. 向量召回
    vector_retriever = MockVectorRetriever()
    vector_results = vector_retriever.search(query, top_k=5)
    print(f"\n[Vector Recall] {len(vector_results)} results:")
    for r in vector_results:
        print(f"  [{r.id}] {r.metadata.get('title','')} (score={r.score:.3f})")

    # 2. 关键词召回
    bm25 = BM25Retriever()
    keyword_results = bm25.search(query, top_k=5)
    print(f"\n[Keyword Recall (BM25)] {len(keyword_results)} results:")
    for r in keyword_results:
        print(f"  [{r.id}] {r.metadata.get('title','')} (score={r.score:.3f})")

    # 3. 结构化检索
    struct_retriever = StructuredRetriever()
    struct_results = struct_retriever.search(query, dept="供应链", top_k=3)
    print(f"\n[Structured Recall] {len(struct_results)} results:")
    for r in struct_results:
        print(f"  [{r.id}] {r.metadata.get('title','')} (source={r.source})")

    # 4. 知识图谱检索
    kg_retriever = KGRetriever()
    kg_results = kg_retriever.search(query, top_k=3)
    print(f"\n[KG Recall] {len(kg_results)} results:")
    for r in kg_results:
        print(f"  [{r.id}] {r.metadata.get('title','')} (entities: {r.metadata.get('matched_entities', [])})")

    t1 = time.time()

    # 5. 分数融合
    fusion = ScoreFusion()
    all_results = fusion.weighted_fusion(vector_results, keyword_results, struct_results, kg_results)
    print(f"\n[Fused] {len(all_results)} unique results")

    # 6. 重排
    reranker = Reranker()
    final_results = reranker.rerank(query, all_results, top_k=top_k)
    t2 = time.time()

    print(f"\n" + "=" * 65)
    print(f"[Final Top-{top_k} after Rerank] (recall: {t1-t0:.3f}s, rerank: {t2-t1:.3f}s)")
    print("=" * 65)
    for i, r in enumerate(final_results, 1):
        print(f"\n  #{i} [{r.id}] {r.metadata.get('title','')}")
        print(f"      Source: {r.source}  |  Score: {r.score:.3f}")
        print(f"      Content: {r.content[:80]}...")
    print(f"\nTotal: {t2-t0:.3f}s")


if __name__ == "__main__":
    main()
