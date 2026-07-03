"""
ch07-multi-recall: 多策略检索演示 - RRF融合
============================================
实现密集检索（TF-IDF）、BM25检索、元数据过滤以及RRF融合。
仅依赖stdlib，可直接运行: python main.py
"""

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# 样例中文语料库（医药领域）
# ---------------------------------------------------------------------------

SAMPLE_CORPUS = [
    {
        "id": "doc_1",
        "title": "恒瑞医药2024年报",
        "content": (
            "恒瑞医药2024年实现营业收入280亿元，同比增长12%。"
            "研发投入达到60亿元，占营收比例21.4%。"
            "公司共有30余个创新药处于临床研究阶段，涵盖抗肿瘤、代谢、自身免疫等领域。"
        ),
        "tags": ["财报", "企业"],
        "category": "financial_report",
    },
    {
        "id": "doc_2",
        "title": "奥希替尼片临床研究进展",
        "content": (
            "奥希替尼（泰瑞沙）是第三代EGFR-TKI靶向药物，用于EGFR突变阳性非小细胞肺癌。"
            "临床研究表明其中位无进展生存期达18.9个月，显著优于第一代TKI药物。"
            "对于脑转移患者，奥希替尼同样表现出良好的颅内病灶控制效果。"
        ),
        "tags": ["肿瘤", "靶向药", "临床"],
        "category": "clinical_study",
    },
    {
        "id": "doc_3",
        "title": "注射用紫杉醇技术综述",
        "content": (
            "紫杉醇是一种微管抑制剂，通过促进微管聚合抑制细胞分裂。"
            "白蛋白结合型紫杉醇利用白蛋白作为载体，提高药物的靶向性和溶解度。"
            "该剂型减少了溶剂相关过敏反应，无需常规预防性给药。"
            "临床主要用于治疗乳腺癌、非小细胞肺癌和胰腺癌等实体瘤。"
        ),
        "tags": ["肿瘤", "化疗", "药物机制"],
        "category": "technical_review",
    },
    {
        "id": "doc_4",
        "title": "中国医药流通行业分析",
        "content": (
            "国药控股是中国最大的医药分销企业，市场份额约26%。"
            "公司拥有覆盖全国的分销网络，服务超过20000家医院和医疗机构。"
            "随着药品集中采购政策推进，医药流通行业集中度持续提升。"
            "数字化转型和供应链优化成为行业主要发展方向。"
        ),
        "tags": ["行业分析", "流通"],
        "category": "industry_report",
    },
    {
        "id": "doc_5",
        "title": "PD-1抑制剂市场格局",
        "content": (
            "PD-1/PD-L1抑制剂是肿瘤免疫治疗的重要药物类别。"
            "恒瑞医药的卡瑞利珠单抗是国内首个获批的国产PD-1抑制剂。"
            "国内已有10余款PD-1/PD-L1产品获批上市，竞争日趋激烈。"
            "适应症从后线治疗逐步拓展至一线治疗和围手术期治疗。"
        ),
        "tags": ["肿瘤", "免疫治疗", "市场"],
        "category": "market_analysis",
    },
    {
        "id": "doc_6",
        "title": "北京协和医院科研成就",
        "content": (
            "北京协和医院是中国医学科学院的重要临床基地。"
            "医院承担了大量国家级科研项目，包括国家重点研发计划和自然科学基金项目。"
            "在内分泌科、风湿免疫科等领域的临床研究处于国内领先水平。"
            "医院每年发表SCI论文超过1000篇，科研成果转化成效显著。"
        ),
        "tags": ["医院", "科研", "临床"],
        "category": "institution_profile",
    },
    {
        "id": "doc_7",
        "title": "国家药品集中采购政策解读",
        "content": (
            "国家药品集中采购（集采）政策旨在通过以量换价降低药品价格。"
            "自2018年首批4+7试点以来，已开展九批集采，覆盖超过300个品种。"
            "集采政策显著降低了患者用药负担，同时也推动了仿制药质量提升。"
            "对于药企而言，集采带来了价格压力和市场份额的重新分配。"
        ),
        "tags": ["政策", "价格", "行业影响"],
        "category": "policy_analysis",
    },
    {
        "id": "doc_8",
        "title": "ADC药物研发前沿",
        "content": (
            "抗体偶联药物（ADC）是近年来发展最快的抗肿瘤药物类别之一。"
            "ADC通过抗体特异性靶向肿瘤细胞，释放细胞毒性药物实现精准杀伤。"
            "国内已有多个ADC药物获批或处于临床后期阶段。"
            "恒瑞医药、荣昌生物等企业在该领域进行了重点布局。"
        ),
        "tags": ["肿瘤", "靶向药", "前沿技术"],
        "category": "technical_review",
    },
]


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------


def tokenize(text: str) -> List[str]:
    """基于单字符的简单中文分词。"""
    tokens = []
    for ch in text:
        if "一" <= ch <= "鿿" or ch.isalnum():
            tokens.append(ch)
    return tokens


# ---------------------------------------------------------------------------
# DenseRetriever (TF-IDF 向量检索)
# ---------------------------------------------------------------------------


@dataclass
class DenseRetriever:
    """基于TF-IDF的密集检索器（余弦相似度）。"""

    corpus: List[dict] = field(default_factory=list)
    _vocab: List[str] = field(default_factory=list)
    _idf: Dict[str, float] = field(default_factory=dict)
    _doc_vectors: List[List[float]] = field(default_factory=list)

    def fit(self, corpus: List[dict]):
        """在语料库上拟合IDF并构建文档向量。"""
        self.corpus = corpus
        N = len(corpus)
        tokenized_docs = [tokenize(d["content"]) for d in corpus]
        df: Counter = Counter()
        for tokens in tokenized_docs:
            for t in set(tokens):
                df[t] += 1

        self._vocab = sorted(df.keys())
        self._idf = {t: math.log((N + 1) / (df[t] + 1)) + 1 for t in self._vocab}
        self._doc_vectors = []
        for tokens in tokenized_docs:
            tf = Counter(tokens)
            max_tf = max(tf.values()) if tf else 1
            vec = []
            for t in self._vocab:
                vec.append((tf.get(t, 0) / max_tf) * self._idf[t])
            self._doc_vectors.append(vec)

    def _vectorize(self, text: str) -> List[float]:
        tokens = tokenize(text)
        tf = Counter(tokens)
        max_tf = max(tf.values()) if tf else 1
        vec = []
        for t in self._vocab:
            vec.append((tf.get(t, 0) / max_tf) * self._idf.get(t, 0))
        return vec

    @staticmethod
    def _cosine_sim(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb) if na and nb else 0.0

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float, str]]:
        """返回 (doc_id, score, title) 列表。"""
        qvec = self._vectorize(query)
        scores = []
        for i, dvec in enumerate(self._doc_vectors):
            sim = self._cosine_sim(qvec, dvec)
            scores.append((self.corpus[i]["id"], sim, self.corpus[i]["title"]))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ---------------------------------------------------------------------------
# BM25Retriever
# ---------------------------------------------------------------------------


@dataclass
class BM25Retriever:
    """BM25检索器（k1=1.5, b=0.75）。"""

    corpus: List[dict] = field(default_factory=list)
    k1: float = 1.5
    b: float = 0.75
    _doc_lens: List[int] = field(default_factory=list)
    _avgdl: float = 0.0
    _tokenized_docs: List[List[str]] = field(default_factory=list)
    _df: Counter = field(default_factory=Counter)
    _idf_cache: Dict[str, float] = field(default_factory=dict)
    N: int = 0

    def fit(self, corpus: List[dict]):
        self.corpus = corpus
        self.N = len(corpus)
        self._tokenized_docs = [tokenize(d["content"]) for d in corpus]
        self._doc_lens = [len(tokens) for tokens in self._tokenized_docs]
        self._avgdl = sum(self._doc_lens) / self.N if self.N else 0.0

        self._df = Counter()
        for tokens in self._tokenized_docs:
            for t in set(tokens):
                self._df[t] += 1

        self._idf_cache = {}
        for t in self._df:
            self._idf_cache[t] = math.log(
                (self.N - self._df[t] + 0.5) / (self._df[t] + 0.5) + 1.0
            )

    def _score_doc(self, query_tokens: List[str], doc_idx: int) -> float:
        """计算单个文档的BM25得分。"""
        tokens = self._tokenized_docs[doc_idx]
        tf = Counter(tokens)
        dl = self._doc_lens[doc_idx]
        score = 0.0
        for qt in query_tokens:
            if qt not in self._idf_cache:
                continue
            idf = self._idf_cache[qt]
            tf_val = tf.get(qt, 0)
            score += idf * (tf_val * (self.k1 + 1)) / (
                tf_val + self.k1 * (1 - self.b + self.b * dl / self._avgdl)
            )
        return score

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float, str]]:
        q_tokens = tokenize(query)
        scores = []
        for i in range(self.N):
            s = self._score_doc(q_tokens, i)
            scores.append((self.corpus[i]["id"], s, self.corpus[i]["title"]))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ---------------------------------------------------------------------------
# MetadataRetriever
# ---------------------------------------------------------------------------


@dataclass
class MetadataRetriever:
    """基于元数据（标签/类别）过滤的检索器。"""

    corpus: List[dict] = field(default_factory=list)

    def fit(self, corpus: List[dict]):
        self.corpus = corpus

    def search(
        self,
        query: str = "",
        top_k: int = 5,
        tags: Optional[List[str]] = None,
        category: Optional[str] = None,
    ) -> List[Tuple[str, float, str]]:
        """按元数据过滤文档，支持标签（取交集）和类别过滤。"""
        results = []
        for doc in self.corpus:
            match = True
            if tags:
                doc_tags = set(doc.get("tags", []))
                if not doc_tags.intersection(tags):
                    match = False
            if category and doc.get("category") != category:
                match = False
            if match:
                results.append((doc["id"], 1.0, doc["title"]))

        results.sort(key=lambda x: x[0])
        return results[:top_k]


# ---------------------------------------------------------------------------
# RRFusion (Reciprocal Rank Fusion)
# ---------------------------------------------------------------------------


@dataclass
class RRFusion:
    """使用倒数排名融合（RRF）合并多个检索结果列表。"""

    k: int = 60

    def fuse(
        self, result_lists: List[List[Tuple[str, float, str]]]
    ) -> List[Tuple[str, float, str]]:
        """融合多个排名列表。

        RRF得分 = sum(1 / (k + rank_i))，其中rank_i是文档在第i个列表中的排名（从1开始）。
        """
        rrf_scores: Dict[str, float] = {}
        doc_info: Dict[str, str] = {}

        for ranked_list in result_lists:
            for rank, (doc_id, score, title) in enumerate(ranked_list, start=1):
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (
                    self.k + rank
                )
                doc_info[doc_id] = title

        fused = [
            (doc_id, rrf_scores[doc_id], doc_info[doc_id])
            for doc_id in rrf_scores
        ]
        fused.sort(key=lambda x: x[1], reverse=True)
        return fused


# ---------------------------------------------------------------------------
# 打印工具
# ---------------------------------------------------------------------------


def print_results_table(
    label: str,
    results: List[Tuple[str, float, str]],
):
    """以表格形式打印检索结果。"""
    print(f"\n  {label}:")
    print(f"  {'排名':<6} {'文档ID':<12} {'得分':<12} {'标题'}")
    print(f"  {'-' * 6} {'-' * 12} {'-' * 12} {'-' * 30}")
    for rank, (doc_id, score, title) in enumerate(results, start=1):
        print(f"  {rank:<6} {doc_id:<12} {score:<12.4f} {title}")


def print_comparison_table(
    dense: List[Tuple[str, float, str]],
    bm25: List[Tuple[str, float, str]],
    meta: List[Tuple[str, float, str]],
    fusion: List[Tuple[str, float, str]],
):
    """打印四种检索结果的对比表。"""
    all_ids = sorted(
        set(
            list(d[0] for d in dense)
            + list(b[0] for b in bm25)
            + list(m[0] for m in meta)
            + list(f[0] for f in fusion)
        )
    )

    def _rank_of(doc_id, ranked_list):
        for i, (did, _, _) in enumerate(ranked_list, start=1):
            if did == doc_id:
                return i
        return "-"

    print(f"\n  {'=' * 60}")
    print(f"  检索结果对比表")
    print(f"  {'=' * 60}")
    print(f"  {'文档':<30} {'密集TF-IDF':<12} {'BM25':<12} {'元数据':<12} {'RRF融合':<12}")
    print(f"  {'-' * 30} {'-' * 12} {'-' * 12} {'-' * 12} {'-' * 12}")

    title_map = {d["id"]: d["title"] for d in SAMPLE_CORPUS}

    fusion_ids = [d[0] for d in fusion]
    other_ids = sorted(set(all_ids) - set(fusion_ids))
    sorted_ids = fusion_ids + other_ids

    for doc_id in sorted_ids:
        title = title_map.get(doc_id, doc_id)
        if len(title) > 28:
            title = title[:25] + "..."
        dr = _rank_of(doc_id, dense)
        br = _rank_of(doc_id, bm25)
        mr = _rank_of(doc_id, meta)
        fr = _rank_of(doc_id, fusion)
        print(f"  {title:<30} {str(dr):<12} {str(br):<12} {str(mr):<12} {str(fr):<12}")

    print(f"  {'-' * 60}")
    print(f"  注: 数字 = 排名, '-' = 未出现在Top-K中")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def main():
    print("=" * 60)
    print("  Multi-Strategy Retrieval Demo (RRF Fusion)")
    print("=" * 60)

    corpus = SAMPLE_CORPUS

    # 1. 拟合所有检索器
    print(f"\n[1] 拟合检索器 (语料库大小: {len(corpus)} 篇文档)")
    print(f"    {'文档ID':<12} {'标题':<30} {'标签':<30} {'类别'}")
    print(f"    {'-' * 12} {'-' * 30} {'-' * 30} {'-' * 20}")
    for doc in corpus:
        tags_str = ", ".join(doc["tags"])
        print(
            f"    {doc['id']:<12} {doc['title']:<30} {tags_str:<30} {doc['category']}"
        )

    dense = DenseRetriever()
    dense.fit(corpus)

    bm25 = BM25Retriever()
    bm25.fit(corpus)

    meta = MetadataRetriever()
    meta.fit(corpus)

    # 2. 定义查询
    query = "肺癌靶向治疗药物"
    print(f"\n[2] 查询: \"{query}\"")

    meta_tags = ["肿瘤"]
    meta_category = None
    print(f"    元数据过滤: tags={meta_tags}, category={meta_category}")

    # 3. 执行检索
    top_k = 5
    dense_results = dense.search(query, top_k=top_k)
    bm25_results = bm25.search(query, top_k=top_k)
    meta_results = meta.search(query, top_k=top_k, tags=meta_tags)

    # 4. 打印各检索器结果
    print("\n[3] 各检索器结果:")
    print_results_table("密集检索 (TF-IDF + 余弦相似度)", dense_results)
    print_results_table("BM25 检索 (k1=1.5, b=0.75)", bm25_results)
    print_results_table("元数据过滤 (tags包含'肿瘤')", meta_results)

    # 5. RRF融合
    fusion = RRFusion(k=60)
    fusion_results = fusion.fuse([dense_results, bm25_results, meta_results])

    print("\n[4] RRF融合 (k=60):")
    print_results_table("RRF融合结果", fusion_results)

    # 6. 对比表
    print("\n[5] 结果对比:")
    print_comparison_table(dense_results, bm25_results, meta_results, fusion_results)

    # 7. 分析
    print(f"\n[6] 分析:")
    dense_ids = set(d[0] for d in dense_results)
    bm25_ids = set(d[0] for d in bm25_results)
    meta_ids = set(d[0] for d in meta_results)
    fusion_ids = set(d[0] for d in fusion_results)

    all_methods = dense_ids & bm25_ids & meta_ids
    any_two = (dense_ids & bm25_ids) | (dense_ids & meta_ids) | (bm25_ids & meta_ids)

    print(f"    三种方法共同返回的文档数: {len(all_methods)}")
    print(f"    至少两种方法返回的文档数: {len(any_two)}")
    print(f"    融合结果包含的文档数: {len(fusion_ids)}")

    print("\n" + "=" * 60)
    print("  Demo 完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
