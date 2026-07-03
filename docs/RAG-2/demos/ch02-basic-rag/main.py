"""
ch02-basic-rag: 基础RAG演示 - 中文医药领域
============================================
一个完全自包含的中文医药RAG演示，仅依赖stdlib（jieba为可选增强）。
可直接运行: python main.py
"""

import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import List, Tuple

# ---------------------------------------------------------------------------
# 样例中文文档（医药领域）
# ---------------------------------------------------------------------------

SAMPLE_DOCS = [
    {
        "title": "恒瑞医药",
        "content": (
            "恒瑞医药是中国领先的创新药研发企业，成立于1970年，总部位于江苏省连云港市。"
            "公司主要从事抗肿瘤药物、麻醉药物、造影剂等领域的研发、生产和销售。"
            "恒瑞医药在肿瘤免疫治疗、靶向药物等前沿领域具有显著优势。"
            "公司每年投入大量资金用于新药研发，研发投入占营收比例超过20%。"
            "恒瑞医药已有多款创新药在中国获批上市，包括卡瑞利珠单抗、吡咯替尼等。"
        ),
    },
    {
        "title": "注射用紫杉醇（白蛋白结合型）",
        "content": (
            "注射用紫杉醇（白蛋白结合型）是一种用于治疗乳腺癌的靶向化疗药物。"
            "该药物由恒瑞医药研发生产，于2019年获得中国国家药品监督管理局批准上市。"
            "紫杉醇通过抑制微管解聚来阻止癌细胞分裂，白蛋白作为载体提高了药物的靶向性。"
            "临床研究显示该药物对转移性乳腺癌患者具有显著疗效。"
            "常见不良反应包括中性粒细胞减少、周围神经病变和疲劳等。"
        ),
    },
    {
        "title": "奥希替尼片（泰瑞沙）",
        "content": (
            "奥希替尼片（商品名：泰瑞沙）是阿斯利康研发的第三代EGFR-TKI靶向药物。"
            "主要用于治疗EGFR突变阳性的非小细胞肺癌患者。"
            "奥希替尼能够有效抑制EGFR敏感突变和T790M耐药突变。"
            "临床研究表明奥希替尼一线治疗无进展生存期显著优于第一代EGFR-TKI。"
            "该药物已被纳入中国国家医保目录，大大提高了患者的可及性。"
            "常见副作用包括皮疹、腹泻、甲沟炎和口腔炎等。"
        ),
    },
    {
        "title": "国药控股",
        "content": (
            "国药控股是中国最大的医药分销和供应链服务提供商之一。"
            "公司成立于2003年，是中国医药集团的核心企业。"
            "国药控股的业务涵盖药品分销、医疗器械分销、零售药店和物流服务。"
            "公司拥有覆盖全国的分销网络，为各级医疗机构提供药品供应保障。"
            "在新冠疫情期间，国药控股承担了重要医疗物资的储备和配送任务。"
            "公司正在积极推进数字化转型，通过智慧供应链提升运营效率。"
        ),
    },
    {
        "title": "北京协和医院",
        "content": (
            "北京协和医院是中国最著名的综合性三级甲等医院之一。"
            "医院成立于1921年，由洛克菲勒基金会创办，拥有百年历史。"
            "协和医院以疑难重症诊疗和高水平医学教育闻名于世。"
            "医院设有多个国家重点学科，包括内分泌科、风湿免疫科、妇产科等。"
            "北京协和医院还承担了大量国家级科研项目，是中国医学科学院的重要临床基地。"
            "医院秉承'严谨、求精、勤奋、奉献'的协和精神，服务广大患者。"
        ),
    },
]

# ---------------------------------------------------------------------------
# SimpleTfidfVectorizer
# ---------------------------------------------------------------------------


@dataclass
class SimpleTfidfVectorizer:
    """基于TF-IDF的简单向量化器，支持jieba（可选）或基于字符的分词。"""

    use_jieba: bool = False
    _idf: dict = field(default_factory=dict)
    _vocab: List[str] = field(default_factory=list)
    _corpus_tfidf: List[List[float]] = field(default_factory=list)

    def __post_init__(self):
        if self.use_jieba:
            try:
                import jieba

                self._jieba = jieba
            except ImportError:
                self.use_jieba = False

    def tokenize(self, text: str) -> List[str]:
        """将文本分词为tokens列表。"""
        if self.use_jieba:
            return list(self._jieba.cut(text))
        # 基于字符的简单分词（保留中文字符）
        tokens = []
        buf = []
        for ch in text:
            if "一" <= ch <= "鿿":
                if buf:
                    tokens.append("".join(buf))
                    buf = []
                tokens.append(ch)
            elif ch.isalnum():
                buf.append(ch)
            else:
                if buf:
                    tokens.append("".join(buf))
                    buf = []
        if buf:
            tokens.append("".join(buf))
        return [t for t in tokens if t.strip()]

    def fit(self, documents: List[str]) -> "SimpleTfidfVectorizer":
        """在文档语料上拟合IDF权重。"""
        N = len(documents)
        tokenized_docs = [self.tokenize(doc) for doc in documents]
        df: Counter = Counter()
        for tokens in tokenized_docs:
            for token in set(tokens):
                df[token] += 1

        self._vocab = sorted(df.keys())
        self._idf = {
            token: math.log((N + 1) / (df[token] + 1)) + 1
            for token in self._vocab
        }

        # 构建语料库的TF-IDF向量
        self._corpus_tfidf = []
        for tokens in tokenized_docs:
            tf = Counter(tokens)
            max_tf = max(tf.values()) if tf else 1
            vec = []
            for token in self._vocab:
                tf_val = tf.get(token, 0) / max_tf
                vec.append(tf_val * self._idf[token])
            self._corpus_tfidf.append(vec)
        return self

    def _vectorize(self, text: str) -> List[float]:
        """将文本转换为TF-IDF向量。"""
        tokens = self.tokenize(text)
        tf = Counter(tokens)
        max_tf = max(tf.values()) if tf else 1
        vec = []
        for token in self._vocab:
            tf_val = tf.get(token, 0) / max_tf
            vec.append(tf_val * self._idf.get(token, 0))
        return vec

    @staticmethod
    def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        """计算两个向量之间的余弦相似度。"""
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def search(
        self, query: str, top_k: int = 3
    ) -> List[Tuple[int, float]]:
        """在拟合的语料库中搜索与查询最相似的文档。"""
        query_vec = self._vectorize(query)
        scores = []
        for idx, doc_vec in enumerate(self._corpus_tfidf):
            sim = self.cosine_similarity(query_vec, doc_vec)
            scores.append((idx, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def chunk_documents(
    docs: List[dict], chunk_size: int = 50, overlap: int = 10
) -> List[dict]:
    """将文档列表按固定大小切块，块之间可重叠。

    Args:
        docs: 文档字典列表（需包含 'title' 和 'content' 字段）。
        chunk_size: 每个块的字符数。
        overlap: 相邻块之间的重叠字符数。

    Returns:
        块字典列表，每项包含 'title'（来源文档标题）、'content'（块文本）和 'chunk_id'。
    """
    chunks: List[dict] = []
    for doc in docs:
        content = doc["content"]
        start = 0
        chunk_id = 0
        while start < len(content):
            end = min(start + chunk_size, len(content))
            chunk_text = content[start:end]
            chunks.append(
                {
                    "title": doc["title"],
                    "content": chunk_text,
                    "chunk_id": chunk_id,
                }
            )
            chunk_id += 1
            if end >= len(content):
                break
            start += chunk_size - overlap
    return chunks


# ---------------------------------------------------------------------------
# 模拟答案生成
# ---------------------------------------------------------------------------


def generate_answer(query: str, retrieved_chunks: List[dict]) -> str:
    """基于检索到的文档块模拟生成答案（非真实LLM调用）。

    根据检索到的块内容生成模板化的回答。
    """
    if not retrieved_chunks:
        return "未检索到相关文档，无法生成回答。"

    sources = list(
        dict.fromkeys(c["title"] for c in retrieved_chunks)
    )
    source_text = "、".join(sources)

    # 从检索到的内容中提取关键信息
    evidence_parts = []
    for c in retrieved_chunks[:2]:
        snippet = c["content"][:60].replace("\n", "")
        evidence_parts.append(f"[{c['title']}] {snippet}...")

    answer = (
        f"【模拟回答】\n"
        f"查询: {query}\n"
        f"参考来源: {source_text}\n\n"
        f"基于检索到的{len(retrieved_chunks)}个文档块，相关回答如下：\n\n"
        f"{chr(10).join(f'  - {ep}' for ep in evidence_parts)}\n\n"
        f"（注：此为模拟回答。实际RAG系统会调用大语言模型根据检索内容生成答案。）"
    )
    return answer


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def main():
    print("=" * 60)
    print("  Chinese Pharma RAG Demo (Basic)")
    print("=" * 60)

    # 1. 加载文档
    t0 = time.perf_counter()
    docs = SAMPLE_DOCS
    t1 = time.perf_counter()
    print(f"\n[加载] 已加载 {len(docs)} 篇文档 ({t1-t0:.4f}s)")
    for d in docs:
        print(f"   - {d['title']} ({len(d['content'])}字符)")

    # 2. 文档分块
    t2 = time.perf_counter()
    chunks = chunk_documents(docs, chunk_size=50, overlap=10)
    t3 = time.perf_counter()
    print(f"\n[分块] 生成 {len(chunks)} 个块 (size=50, overlap=10) ({t3-t2:.4f}s)")
    for c in chunks:
        print(f"   [{c['title']}#{c['chunk_id']}] {c['content']}")

    # 3. 构建TF-IDF索引
    t4 = time.perf_counter()
    chunk_texts = [c["content"] for c in chunks]
    vectorizer = SimpleTfidfVectorizer(use_jieba=False)
    vectorizer.fit(chunk_texts)
    t5 = time.perf_counter()
    print(
        f"\n[索引] TF-IDF索引构建完成 "
        f"(词汇量={len(vectorizer._vocab)}) ({t5-t4:.4f}s)"
    )

    # 4. 搜索
    query = "肺癌靶向药物"
    print(f"\n[搜索] 查询: \"{query}\"")
    t6 = time.perf_counter()
    results = vectorizer.search(query, top_k=3)
    t7 = time.perf_counter()
    print(f"        耗时: {t7-t6:.4f}s")
    print(f"        结果:")
    for idx, score in results:
        chk = chunks[idx]
        print(f"           #{idx} [{chk['title']}] score={score:.4f}")
        print(f"           \"{chk['content']}\"")

    # 5. 生成答案
    t8 = time.perf_counter()
    retrieved = [chunks[idx] for idx, _ in results]
    answer = generate_answer(query, retrieved)
    t9 = time.perf_counter()
    print(f"\n[生成] 耗时: {t9-t8:.4f}s")
    print()
    print(answer)

    # 6. 更多查询示例
    print("\n" + "-" * 60)
    print("  更多查询示例:")
    print("-" * 60)
    for q in ["恒瑞医药主要业务", "紫杉醇作用机制", "北京协和医院历史"]:
        t = time.perf_counter()
        res = vectorizer.search(q, top_k=2)
        elapsed = time.perf_counter() - t
        top = chunks[res[0][0]]
        print(f"\n  查询: \"{q}\" ({elapsed:.4f}s)")
        print(f"    最佳匹配: [{top['title']}] (score={res[0][1]:.4f})")

    print("\n" + "=" * 60)
    print("  Demo 完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
