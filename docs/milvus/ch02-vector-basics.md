# 第2章：向量基础理论——RAG必备核心知识

## 2.1 向量与 Embedding 的底层逻辑

### 2.1.1 什么是向量

向量（Vector）是数学中最基础的概念之一，可以理解为一组有序排列的数字。在计算机科学中，一个向量通常表示为一个浮点数数组：

```
v = [0.12, 0.87, -0.33, 0.54, 0.01, ..., 0.76]
```

每个向量有固定的维度（Dimensionality），例如上述示例是一个 N 维向量。在 RAG 和向量检索的语境中，向量的维度通常在 128 到 4096 之间，具体取决于使用的 Embedding 模型。

向量的核心价值在于：**它可以将非结构化数据（文本、图像、音频）映射到数学可计算的数值空间**。在这个空间中，语义相似的数据在几何位置上彼此靠近。

### 2.1.2 Embedding 的本质

Embedding 是将高维离散数据（如单词、句子、图像）映射到低维连续向量空间的过程。这个映射不是随机的，而是通过神经网络学习得到的。

```python
# Embedding 过程示意
from sentence_transformers import SentenceTransformer

# 加载预训练的 Embedding 模型
model = SentenceTransformer('BAAI/bge-large-zh-v1.5')

# 将文本转换为向量
texts = [
    "今天天气真不错",
    "天气预报说明天有雨",
    "股票市场今日大涨"
]

embeddings = model.encode(texts)
print(f"向量维度: {embeddings.shape}")  # (3, 1024)
print(f"向量示例: {embeddings[0][:5]}") # [0.12, -0.34, 0.56, ...]
```

关键的直觉在于：Embedding 模型通过在大规模语料上的训练，学会了将语义信息编码为向量的位置和方向。"天气"相关文本的向量会聚集在语义空间的某个区域，而"金融"相关文本的向量则会聚集在另一个区域。

### 2.1.3 从词向量到句向量

Embedding 技术的发展经历了几个关键阶段：

**Word2Vec（2013）**：Google 提出的词向量训练方法。每个词被映射为一个固定维度的向量。"国王 - 男人 + 女人 ≈ 女王"是 Word2Vec 最著名的类比推理示例。

**BERT（2018）**：Google 提出的预训练模型，能够根据上下文生成动态的词向量。同一个词在不同语境下有不同的向量表示。"苹果好吃"和"苹果发布了新手机"中的"苹果"向量不同。

**Sentence Embedding（2019-至今）**：以 Sentence-BERT 为代表，直接生成整个句子的向量。这种方法使得语义相似度计算更加高效——不需要逐词比较，直接计算两个句向量的距离即可。

## 2.2 文本向量化原理

### 2.2.1 中文文本向量化的挑战

中文文本的向量化比英文更具挑战性：

1. **分词歧义**：中文没有天然的空格分隔。"南京市长江大桥"可以被分词为"南京市/长江大桥"或"南京市长/江大桥"。
2. **多义词丰富**：同一个词在不同语境下含义差异巨大。
3. **领域特异性**：通用 Embedding 模型在特定领域（医疗、法律、金融）的表现可能不理想。

### 2.2.2 主流 Embedding 模型

以下是在中文场景下常用的 Embedding 模型：

| 模型 | 维度 | 最大长度 | 特点 |
|------|------|---------|------|
| BAAI/bge-large-zh-v1.5 | 1024 | 512 | 中文语义理解出色 |
| text-embedding-3-small | 1536 | 8191 | OpenAI 出品，通用性强 |
| text-embedding-ada-002 | 1536 | 8191 | 经典模型，已逐步被替代 |
| moka-ai/m3e-base | 768 | 512 | 轻量级中文模型 |
| shibing624/text2vec-base-chinese | 768 | 128 | 中文语义相似度优化 |

### 2.2.3 Embedding 质量的关键因素

高质量的 Embedding 是 RAG 系统成功的基础。以下是影响 Embedding 质量的关键因素：

**1. 领域匹配度**

通用 Embedding 模型在特定领域可能表现不佳。如果你的知识库是医疗文献，使用在医学语料上微调的 Embedding 模型会显著提升检索质量。

```python
# 评估 Embedding 质量的简单方法
from sentence_transformers.util import cos_sim

def evaluate_embedding_quality(model_name: str, test_pairs: list) -> float:
    """评估 Embedding 模型在特定任务上的表现"""
    model = SentenceTransformer(model_name)
    scores = []
    
    for text1, text2, expected_similar in test_pairs:
        emb1 = model.encode(text1)
        emb2 = model.encode(text2)
        similarity = cos_sim(emb1, emb2).item()
        scores.append(similarity)
    
    return sum(scores) / len(scores)
```

**2. 输入长度控制**

大多数 Embedding 模型有最大输入长度限制（通常是 512 Token）。超过限制的文本会被截断，导致信息丢失。实践中，建议将文档切分为 200-500 Token 的段落再生成 Embedding。

**3. 归一化**

归一化是将向量缩放到单位长度的过程。归一化后的向量可以确保向量长度不影响相似度计算，让余弦相似度和内积结果等价：

```python
import numpy as np

def normalize(vector: np.ndarray) -> np.ndarray:
    """向量归一化"""
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm

# 归一化前后对比
v = np.array([3.0, 4.0])
print(f"原始向量: {v}")
print(f"归一化后: {normalize(v)}")
print(f"长度: {np.linalg.norm(normalize(v)):.1f}")  # 1.0
```

## 2.3 相似度算法详解

向量检索的核心是计算两个向量的相似度。Milvus 支持多种相似度度量方式，选择正确的度量方式对检索效果至关重要。

### 2.3.1 余弦相似度（Cosine Similarity）

余弦相似度衡量的是两个向量之间的夹角余弦值，取值范围为 [-1, 1]。值越接近 1，表示两个向量的方向越一致。

```python
import numpy as np

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """余弦相似度"""
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return dot_product / (norm_v1 * norm_v2)

# 示例
v1 = np.array([0.1, 0.2, 0.3])
v2 = np.array([0.15, 0.25, 0.28])  # 方向相近
v3 = np.array([-0.1, -0.2, -0.3])  # 方向相反

print(f"相似向量余弦相似度: {cosine_similarity(v1, v2):.4f}")  # ~0.99
print(f"相反向量余弦相似度: {cosine_similarity(v1, v3):.4f}")  # -1.0
```

**适用场景**：文本语义相似度。余弦相似度只关注方向，不受向量长度影响，非常适合文本检索。

**Milvus 配置**：`metric_type: "COSINE"`

### 2.3.2 欧氏距离（L2 Distance）

欧氏距离是向量空间中两点之间的直线距离。Milvus 中使用的是平方后的 L2 距离。

```python
def l2_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """L2 距离（欧氏距离）"""
    return np.sqrt(np.sum((v1 - v2) ** 2))

def squared_l2_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """平方 L2 距离（Milvus 默认使用）"""
    return np.sum((v1 - v2) ** 2)

# 示例
print(f"L2 距离: {l2_distance(v1, v2):.4f}")
print(f"平方 L2 距离: {squared_l2_distance(v1, v2):.4f}")
```

**适用场景**：图像检索、聚类分析等对绝对距离敏感的场景。

**Milvus 配置**：`metric_type: "L2"`

### 2.3.3 内积（Inner Product / IP）

内积是两个向量的点积。对于归一化后的向量，内积等价于余弦相似度。

```python
def inner_product(v1: np.ndarray, v2: np.ndarray) -> float:
    """内积"""
    return np.dot(v1, v2)

# 示例
print(f"内积: {inner_product(v1, v2):.4f}")

# 归一化后，内积等价于余弦相似度
v1_norm = normalize(v1)
v2_norm = normalize(v2)
print(f"归一化后内积: {inner_product(v1_norm, v2_norm):.4f}")
print(f"归一化后余弦相似度: {cosine_similarity(v1_norm, v2_norm):.4f}")
```

**适用场景**：归一化向量的相似度检索，或需要强调向量长度的场景（如推荐系统中的用户偏好向量）。

**Milvus 配置**：`metric_type: "IP"`

### 2.3.4 如何选择相似度度量

| 度量方式 | 取值范围 | 受向量长度影响 | 推荐场景 |
|---------|---------|--------------|---------|
| 余弦相似度 | [-1, 1] | 否 | 文本检索（推荐） |
| L2 距离 | [0, +∞) | 是 | 图像检索、聚类 |
| 内积 | (-∞, +∞) | 是 | 归一化向量检索 |

**最佳实践**：对于文本检索场景，统一使用余弦相似度。如果 Embedding 模型已经输出归一化向量，三种度量方式等价，此时可以选择内积（计算效率最高）。

## 2.4 维度诅咒与规避方案

### 2.4.1 什么是维度诅咒

维度诅咒（Curse of Dimensionality）是指随着向量维度的增加，数据在空间中变得极其稀疏，导致距离度量失去区分能力的现象。

以三维空间为例，假设数据在 [0, 1] 范围内均匀分布。当维度增加到 100 时，几乎所有数据点之间的距离都趋向于相等，最远点与最近点的距离比趋近于 1：

```python
import numpy as np
import matplotlib.pyplot as plt

def dimension_curse_demo(dim: int, num_points: int = 1000):
    """演示维度诅咒"""
    points = np.random.rand(num_points, dim)
    distances = []
    for i in range(min(100, num_points)):
        dist = np.linalg.norm(points[i] - points, axis=1)
        distances.append(dist)
    
    all_dists = np.concatenate(distances)
    return np.min(all_dists[all_dists > 0]), np.max(all_dists)

for dim in [2, 10, 100, 500, 1000]:
    min_dist, max_dist = dimension_curse_demo(dim)
    ratio = min_dist / max_dist
    print(f"维度={dim:4d}: 最小距离/最大距离 = {ratio:.4f}")
```

输出结果清晰地展示了：随着维度增加，所有点之间的距离趋于一致，区分度急剧下降。

### 2.4.2 规避策略

**1. 选择合适的 Embedding 模型**

并非维度越高越好。768 维或 1024 维的 Embedding 通常已经能够提供足够的表达能力。更高的维度（如 4096 维）虽然理论上能编码更多信息，但会加剧维度诅咒。

**2. 使用近似最近邻（ANN）搜索**

精确的 kNN 搜索在十亿级数据上是不现实的。ANN 算法通过牺牲少量精度换取巨大的性能提升：

```python
# Milvus 中使用 ANN 索引
from pymilvus import MilvusClient

client = MilvusClient("http://localhost:19530")

# 创建 IVF 索引，通过聚类近似搜索
index_params = {
    "metric_type": "COSINE",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 1024}
}

# 搜索时通过 nprobe 控制搜索精度
search_params = {
    "params": {"nprobe": 64}  # nprobe 越大越精确，但越慢
}
```

**3. 维度约减**

在特定场景下，可以使用 PCA（主成分分析）或 AutoEncoder 等方法降低向量维度，减少维度诅咒的影响。

## 2.5 向量量化基础

### 2.5.1 为什么需要量化

在大规模向量检索中，内存是最大的瓶颈之一。假设有 10 亿条 1024 维的向量，每个维度用 float32（4 字节）存储：

```
10亿 × 1024 × 4 字节 ≈ 4TB
```

这意味着仅向量数据就需要 4TB 内存，这在实践中是不可接受的。向量量化通过降低每个维度值的精度来压缩向量数据。

### 2.5.2 标量量化（SQ）

标量量化是最基础的量化方式，将 float32 的向量压缩为 int8：

```python
import numpy as np

def scalar_quantize(vector: np.ndarray) -> np.ndarray:
    """标量量化：float32 → int8"""
    # 找到向量范围
    v_min, v_max = vector.min(), vector.max()
    
    # 映射到 int8 范围 [-128, 127]
    scale = 255.0 / (v_max - v_min)
    quantized = np.round((vector - v_min) * scale - 128).astype(np.int8)
    
    return quantized, v_min, v_max

def scalar_dequantize(quantized: np.ndarray, v_min: float, v_max: float) -> np.ndarray:
    """反量化：int8 → float32"""
    scale = 255.0 / (v_max - v_min)
    return (quantized.astype(np.float32) + 128) / scale + v_min

# 示例
original = np.array([0.12, 0.87, -0.33, 0.54], dtype=np.float32)
quantized, v_min, v_max = scalar_quantize(original)
recovered = scalar_dequantize(quantized, v_min, v_max)

print(f"原始:   {original}")
print(f"量化后: {quantized}")
print(f"恢复:   {recovered}")
print(f"MSE:    {np.mean((original - recovered) ** 2):.6f}")
```

Milvus 的 IVF_SQ8 索引使用的就是这种量化方式，可将内存占用减少约 70%。

### 2.5.3 乘积量化（PQ）

乘积量化是更高级的量化方式，它将高维向量切分为多个低维子向量，对每个子向量分别量化：

```python
def product_quantize(vector: np.ndarray, m: int = 8) -> tuple:
    """乘积量化示例：将向量切分为 m 个子空间分别量化"""
    dim = len(vector)
    sub_dim = dim // m
    
    # 将向量切分为 m 个子向量
    sub_vectors = vector.reshape(m, sub_dim)
    
    # 对每个子向量独立量化（示例：直接取均值作为编码）
    codes = np.round(sub_vectors.mean(axis=1)).astype(np.int8)
    
    return codes

# 1024 维向量通过 PQ 压缩到 8 个 int8
vector_1024 = np.random.randn(1024)
pq_codes = product_quantize(vector_1024, m=8)
print(f"原始向量内存: {vector_1024.nbytes} 字节")
print(f"PQ 压缩后:    {pq_codes.nbytes} 字节")
print(f"压缩比:       {vector_1024.nbytes / pq_codes.nbytes:.0f}x")
```

Milvus 的 IVF_PQ 索引将内存压缩比提高到 10-20 倍，使得在有限内存中处理亿级数据成为可能。

### 2.5.4 索引类型对比

| 索引类型 | 量化方式 | 内存压缩比 | 召回率 | 推荐场景 |
|---------|---------|-----------|-------|---------|
| IVF_FLAT | 无 | 1x | 最高 | 百万级，精度优先 |
| IVF_SQ8 | 标量量化 | ~3-4x | 高 | 千万级，平衡方案 |
| IVF_PQ | 乘积量化 | ~10-20x | 中等 | 亿级，内存受限 |
| HNSW | 无（图结构） | ~1.5x（含图结构） | 最高 | 毫秒级延迟需求 |
| DISKANN | 无（硬盘存储） | 不限 | 高 | 十亿级，突破内存 |

## 本章小结

本章从向量和 Embedding 的基础概念出发，深入讲解了文本向量化的原理和主流模型。我们详细分析了三种核心相似度算法——余弦相似度、欧氏距离和内积——的数学原理、适用场景以及在 Milvus 中的配置方式。针对高维向量检索面临的维度诅咒问题，我们介绍了多种规避策略。最后，通过标量量化和乘积量化的原理讲解，揭示了 Milvus 如何通过量化技术在大规模数据场景下保持卓越性能。

掌握这些基础理论是理解 RAG 系统和 Milvus 运作机制的基石。下一章中，我们将深入 RAG 的完整架构，剖析 Milvus 在整个系统中的关键角色。
