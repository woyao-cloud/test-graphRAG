"""
ch02-vector-basics: 向量基础理论演示
======================================
纯 Python 标准库实现，演示向量相似度计算、维度诅咒、归一化等核心概念。
可直接运行: python main.py
"""

import math
import random
from collections import Counter


# ======================================================================
# 1. 向量基础操作
# ======================================================================

def dot_product(a: list[float], b: list[float]) -> float:
    """计算向量点积。"""
    return sum(x * y for x, y in zip(a, b))


def vector_norm(v: list[float]) -> float:
    """计算向量 L2 范数（长度）。"""
    return math.sqrt(sum(x * x for x in v))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """余弦相似度：衡量两个向量的方向一致性，范围 [-1, 1]。
    在 RAG 中，余弦相似度是最常用的相似度度量。
    """
    dot = dot_product(a, b)
    norm_a = vector_norm(a)
    norm_b = vector_norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def euclidean_distance(a: list[float], b: list[float]) -> float:
    """欧氏距离：衡量两个向量的绝对距离，范围 [0, +∞)。
    在 RAG 中，距离越小表示越相似。
    """
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def inner_product(a: list[float], b: list[float]) -> float:
    """内积（点积）：衡量两个向量的相似度，范围 (-∞, +∞)。
    对于归一化向量，内积等价于余弦相似度。
    """
    return dot_product(a, b)


def normalize(v: list[float]) -> list[float]:
    """向量 L2 归一化：将向量长度缩放到 1。"""
    norm = vector_norm(v)
    if norm == 0:
        return v
    return [x / norm for x in v]


# ======================================================================
# 2. 文本向量化（词袋模型）
# ======================================================================

def text_to_vector(text: str, vocabulary: list[str]) -> list[float]:
    """将文本转换为词频向量（词袋模型）。"""
    words = text.lower().split()
    word_counts = Counter(words)
    return [float(word_counts.get(word, 0)) for word in vocabulary]


# ======================================================================
# 3. 维度诅咒演示
# ======================================================================

def demo_curse_of_dimensionality():
    """演示维度诅咒：随着维度增加，任意两点间的距离趋于一致。"""
    print("=" * 60)
    print("【维度诅咒演示】")
    print("=" * 60)
    print("生成 100 对随机向量，计算不同维度下的平均余弦相似度和欧氏距离。")
    print()

    for dim in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        cosines: list[float] = []
        distances: list[float] = []
        for _ in range(100):
            a = [random.random() for _ in range(dim)]
            b = [random.random() for _ in range(dim)]
            cosines.append(cosine_similarity(a, b))
            distances.append(euclidean_distance(a, b))

        avg_cos = sum(cosines) / len(cosines)
        std_cos = math.sqrt(sum((c - avg_cos) ** 2 for c in cosines) / len(cosines))
        avg_dist = sum(distances) / len(distances)
        std_dist = math.sqrt(sum((d - avg_dist) ** 2 for d in distances) / len(distances))

        print(f"  维度={dim:4d} | 平均余弦={avg_cos:.4f} ±{std_cos:.4f} | 平均距离={avg_dist:.4f} ±{std_dist:.4f}")

    print()
    print("结论：维度越高，向量间相似度趋于 0.5，距离趋于稳定。")
    print("这就是\"维度诅咒\"——高维空间中所有点看起来都差不多。")
    print()


# ======================================================================
# 4. 相似度对比演示
# ======================================================================

def demo_similarity_metrics():
    """演示三种相似度度量的差异。"""
    print("=" * 60)
    print("【相似度度量对比】")
    print("=" * 60)

    # 示例向量：词频表示
    vocabulary = ["抗肿瘤", "肺癌", "乳腺癌", "化疗", "靶向", "免疫", "手术", "放疗"]

    docs = [
        ("非小细胞肺癌靶向治疗", "非小细胞肺癌 靶向 治疗 抗肿瘤 药物"),
        ("乳腺癌化疗方案", "乳腺癌 化疗 方案 抗肿瘤 治疗"),
        ("免疫治疗在肺癌中的应用", "免疫 治疗 肺癌 应用 抗肿瘤"),
        ("心血管疾病预防", "心血管 疾病 预防 健康 饮食"),
    ]

    print(f"{'文档':<30} {'余弦':>8} {'欧氏':>8} {'内积':>8}")
    print("-" * 60)

    query_vec = text_to_vector("肺癌靶向治疗", vocabulary)
    for title, text in docs:
        doc_vec = text_to_vector(text, vocabulary)
        cos = cosine_similarity(query_vec, doc_vec)
        euc = euclidean_distance(query_vec, doc_vec)
        ip = inner_product(query_vec, doc_vec)
        print(f"  {title:<28} {cos:>8.4f} {euc:>8.4f} {ip:>8.4f}")

    print()
    print("余弦相似度：关注方向一致性，不受向量长度影响（最适合 RAG）。")
    print("欧氏距离：关注绝对差异，受向量长度影响大。")
    print("内积：对高频词更敏感，适合已归一化的向量。")
    print()


# ======================================================================
# 5. 归一化演示
# ======================================================================

def demo_normalization():
    """演示向量归一化对相似度的影响。"""
    print("=" * 60)
    print("【向量归一化演示】")
    print("=" * 60)

    # 创建两个方向相同但长度不同的向量
    v1 = [1.0, 2.0, 3.0]
    v2 = [2.0, 4.0, 6.0]  # v2 = 2 * v1

    n1 = normalize(v1)
    n2 = normalize(v2)

    print(f"  v1 = {v1}")
    print(f"  v2 = {v2}")
    print(f"  v1 范数 = {vector_norm(v1):.4f}")
    print(f"  v2 范数 = {vector_norm(v2):.4f}")
    print()
    print(f"  原始余弦相似度(v1, v2) = {cosine_similarity(v1, v2):.4f}")
    print(f"  原始欧氏距离(v1, v2)   = {euclidean_distance(v1, v2):.4f}")
    print()
    print(f"  归一化后 v1 = {[f'{x:.4f}' for x in n1]}")
    print(f"  归一化后 v2 = {[f'{x:.4f}' for x in n2]}")
    print(f"  归一化余弦相似度 = {cosine_similarity(n1, n2):.4f}")
    print(f"  归一化欧氏距离   = {euclidean_distance(n1, n2):.4f}")
    print()
    print("归一化后：余弦相似度不变（因为方向相同），欧氏距离变为 0。")
    print("在 RAG 中，向量归一化可以提升内积检索的效率。")
    print()


# ======================================================================
# 6. TopK 检索模拟
# ======================================================================

def demo_topk_retrieval():
    """模拟 RAG 中的 TopK 向量检索过程。"""
    print("=" * 60)
    print("【TopK 检索模拟】")
    print("=" * 60)

    # 模拟文档库
    documents = [
        "注射用紫杉醇是一种用于治疗乳腺癌的靶向化疗药物",
        "奥希替尼片用于治疗EGFR突变阳性的非小细胞肺癌",
        "贝伐珠单抗注射液是一种抗血管生成的靶向药物",
        "卡瑞利珠单抗是一种PD-1免疫检查点抑制剂",
        "吉非替尼片是第一代EGFR-TKI靶向药物",
        "来那度胺胶囊用于治疗多发性骨髓瘤",
    ]

    # 模拟嵌入向量（用随机向量代替真实嵌入）
    random.seed(42)
    doc_vectors = [[random.random() for _ in range(16)] for _ in documents]

    # 查询向量
    query = "肺癌靶向药物"
    random.seed(100)
    query_vec = [random.random() for _ in range(16)]

    # 计算相似度并排序
    scored = [(cosine_similarity(query_vec, dv), i) for i, dv in enumerate(doc_vectors)]
    scored.sort(reverse=True)

    print(f"  查询: \"{query}\"")
    print(f"  文档总数: {len(documents)}")
    print()
    print(f"  Top-3 检索结果:")
    print(f"  {'排名':<6} {'相似度':<10} {'文档'}")
    print(f"  {'-'*6} {'-'*10} {'-'*40}")
    for rank, (score, idx) in enumerate(scored[:3], 1):
        print(f"  #{rank:<4} {score:<10.4f} {documents[idx]}")
    print()


# ======================================================================
# Main
# ======================================================================

def main():
    print("=" * 60)
    print("  Milvus 实战指南 — Ch2: 向量基础理论")
    print("  本演示使用纯 Python 标准库实现")
    print("=" * 60)
    print()

    demo_similarity_metrics()
    demo_normalization()
    demo_curse_of_dimensionality()
    demo_topk_retrieval()

    print("=" * 60)
    print("  演示完成！")
    print("  关键概念回顾：")
    print("  - 余弦相似度：衡量方向一致性，RAG 中最常用")
    print("  - 欧氏距离：衡量绝对差异")
    print("  - 内积：归一化后等价于余弦相似度")
    print("  - 维度诅咒：高维空间下所有点趋于等距")
    print("  - 归一化：提升检索效率的重要预处理步骤")
    print("=" * 60)


if __name__ == "__main__":
    main()
