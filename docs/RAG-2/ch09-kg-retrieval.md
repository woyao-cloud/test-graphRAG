# 第九章 知识图谱检索：结构化记忆的精确召回

## 9.1 知识图谱在 RAG 中的定位

### 9.1.1 向量检索的边界

传统的向量检索在非结构化文本中寻找语义相似的片段，其能力边界在以下场景中暴露无遗：

- **多跳推理**：用户问"恒瑞医药通过哪些分销商将药品送到北京协和医院？"，需要沿着"恒瑞医药 → 分销商 → 医院"的路径推理三跳，纯向量检索无法建模这种关系链。
- **精确关系查询**：问"哪些药品同时被恒瑞医药和齐鲁制药生产？"，需要精确的交集运算，向量相似度无法表达这种集合逻辑。
- **结构化约束**：问"2023年后获批的抗肿瘤药物中，年销量超过10万支的有哪些？"，涉及时间、类别、数值的多重过滤。

在这些场景下，向量检索的"语义相似"能力反而成为噪声——它会把"看起来像"但"实际上不是"的结果排在前面。

### 9.1.2 知识图谱的独特价值

知识图谱（Knowledge Graph, KG）以 **三元组 (头实体, 关系, 尾实体)** 的形式组织结构化知识，为 RAG 系统带来四个核心能力：

| 能力 | 描述 | 示例 |
|---|---|---|
| 精确实体匹配 | 基于实体 ID 的精确查找，而非模糊语义 | `(恒瑞医药, PRODUCES, 注射用紫杉醇)` |
| 多跳关系推理 | 沿关系路径进行 N 跳推理 | 恒瑞医药 → 生产 → 紫杉醇 → 采购 → 协和医院 |
| 结构化约束 | 按实体类型、关系类型、属性值过滤 | `WHERE drug.category = '抗肿瘤' AND sales > 100000` |
| 可解释性 | 每条推理链路可追溯、可验证 | 路径中的每个节点和关系都可展示 |

### 9.1.3 KG 与向量检索的互补关系

KG 检索和向量检索不是替代关系，而是互补关系：

```
问题类型谱系：
  开放域闲聊         事实性问答          结构化查询          精确关系推理
   |                    |                    |                    |
  向量检索 ←────────── 混合检索 ──────────→ 结构化检索 ←──────── KG 检索
  (语义匹配)          (互补融合)           (精确过滤)           (关系推理)
```

实践中，**混合检索架构**（Hybrid Retrieval Architecture）同时部署多条检索路径，通过融合策略综合各路结果。本章聚焦于 KG 检索这条路径，后续章节会讨论多路融合。

---

## 9.2 实体链接（Entity Linking）

实体链接是从用户查询中识别出命名实体，并将其映射到知识图谱中对应节点的过程。这是 KG 检索的第一步，也是最关键的一步——后续所有图遍历操作都依赖于此。

### 9.2.1 命名实体识别（NER）

NER（Named Entity Recognition）从文本中识别出人名、地名、机构名、药品名等命名实体。在 RAG 的 KG 检索场景中，常用的 NER 方法有三种：

**基于词典的 NER（Dictionary-based NER）**

预先构建实体词典，通过字符串匹配在文本中查找实体。这是最直接的方法：

```python
# ch09_dict_ner.py
"""
基于词典的命名实体识别
使用 AC 自动机（Aho-Corasick）实现高效多模式匹配
"""

from collections import defaultdict
from typing import List, Dict, Tuple


class AhoCorasickNode:
    """AC 自动机节点"""
    __slots__ = ("children", "fail", "output")

    def __init__(self):
        self.children: Dict[str, "AhoCorasickNode"] = {}
        self.fail: "AhoCorasickNode" = None
        self.output: List[str] = []


class AhoCorasick:
    """
    AC 自动机多模式匹配

    同时匹配多个实体名，时间复杂度 O(text_length + total_pattern_length)。
    比逐词匹配快一个数量级。
    """

    def __init__(self):
        self.root = AhoCorasickNode()

    def build(self, patterns: Dict[str, str]):
        """
        构建 AC 自动机

        Parameters
        ----------
        patterns : Dict[str, str]
            {实体名: 实体ID} 映射
        """
        # 1. 构建 Trie 树
        for word, entity_id in patterns.items():
            node = self.root
            for char in word:
                if char not in node.children:
                    node.children[char] = AhoCorasickNode()
                node = node.children[char]
            node.output.append(entity_id)

        # 2. 构建 fail 指针（BFS）
        from collections import deque
        queue = deque()

        # 第一层节点的 fail 指向 root
        for child in self.root.children.values():
            child.fail = self.root
            queue.append(child)

        while queue:
            current = queue.popleft()
            for char, child in current.children.items():
                # 计算 fail 指针
                fail = current.fail
                while fail and char not in fail.children:
                    fail = fail.fail
                child.fail = fail.children[char] if fail else self.root
                # 合并 output
                if child.fail:
                    child.output.extend(child.fail.output)
                queue.append(child)

    def search(self, text: str) -> List[Tuple[str, int, int]]:
        """
        在文本中搜索所有匹配的实体

        Returns
        -------
        List[Tuple[str, int, int]]
            (实体ID, 起始位置, 结束位置)
        """
        node = self.root
        results = []

        for i, char in enumerate(text):
            while node != self.root and char not in node.children:
                node = node.fail

            if char in node.children:
                node = node.children[char]
            else:
                continue

            for entity_id in node.output:
                results.append((entity_id, i, i))

        return results


class DictionaryNER:
    """
    基于词典的 NER

    支持：
    - AC 自动机快速匹配
    - 最长匹配优先
    - 重叠实体去重
    """

    def __init__(self):
        self.entity_dict: Dict[str, str] = {}
        self.name_to_entity: Dict[str, dict] = {}
        self.ac = AhoCorasick()

    def register_entities(self, entities: List[dict]):
        """
        注册实体词典

        Parameters
        ----------
        entities : List[dict]
            [{"id": "e1", "name": "恒瑞医药", "aliases": ["恒瑞"]}, ...]
        """
        patterns = {}
        for entity in entities:
            self.name_to_entity[entity["name"]] = entity
            patterns[entity["name"]] = entity["id"]
            for alias in entity.get("aliases", []):
                self.name_to_entity[alias] = {**entity, "name": alias}
                patterns[alias] = entity["id"]

        self.entity_dict = {e["name"]: e["id"] for e in entities}
        self.ac.build(patterns)

    def extract(self, text: str) -> List[dict]:
        """从文本中提取实体，最长匹配优先"""
        found = []
        matched_positions = set()

        for name in sorted(self.entity_dict.keys(), key=len, reverse=True):
            start = 0
            while True:
                pos = text.find(name, start)
                if pos == -1:
                    break
                if not any(p <= pos < p + len(name) for p in matched_positions):
                    matched_positions.add(pos)
                    entity_info = self.name_to_entity.get(name, {})
                    found.append({
                        "id": entity_info.get("id", ""),
                        "name": name,
                        "type": entity_info.get("type", ""),
                        "start": pos,
                        "end": pos + len(name),
                    })
                start = pos + 1

        found.sort(key=lambda x: x["start"])
        return found
```

**基于模型的 NER（Model-based NER）**

使用预训练语言模型（如 BERT、HanLP、spaCy）进行序列标注。泛化能力强，能识别词典中未收录的实体：

```python
# ch09_model_ner.py
"""
基于预训练模型的命名实体识别
"""

from typing import List, Dict, Optional


class ModelBasedNER:
    """基于模型的 NER，支持 HanLP 和 spaCy 后端"""

    def __init__(self, model_type: str = "hanlp"):
        self.model_type = model_type
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        if self.model_type == "hanlp":
            try:
                import hanlp
                self._model = hanlp.load(
                    hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_BASE_ZH
                )
            except ImportError:
                raise ImportError("请安装 hanlp: pip install hanlp")
        elif self.model_type == "spacy":
            try:
                import spacy
                self._model = spacy.load("zh_core_web_trf")
            except ImportError:
                raise ImportError("请安装 spacy: pip install spacy")

    def extract(self, text: str) -> List[dict]:
        """提取命名实体"""
        self._load_model()
        if self.model_type == "hanlp":
            result = self._model(text)
            return [
                {"text": text[e[0]:e[1]], "type": e[2], "start": e[0], "end": e[1]}
                for e in result.get("ner/msra", [])
            ]
        elif self.model_type == "spacy":
            doc = self._model(text)
            return [
                {"text": ent.text, "type": ent.label_, "start": ent.start_char, "end": ent.end_char}
                for ent in doc.ents
            ]


class HybridNER:
    """
    混合 NER（词典 + 模型）

    策略：先用模型做泛化识别，再用词典做精确匹配，模型结果优先。
    """

    def __init__(self, model_ner: ModelBasedNER, dict_ner: DictionaryNER):
        self.model_ner = model_ner
        self.dict_ner = dict_ner

    def extract(self, text: str) -> List[dict]:
        model_results = self.model_ner.extract(text)
        dict_results = self.dict_ner.extract(text)
        merged, covered = [], set()
        for r in model_results:
            key = (r["start"], r["end"])
            if key not in covered:
                r["source"] = "model"
                merged.append(r)
                covered.add(key)
        for r in dict_results:
            key = (r["start"], r["end"])
            if key not in covered:
                r["source"] = "dict"
                merged.append(r)
                covered.add(key)
        merged.sort(key=lambda x: x["start"])
        return merged
```

**LLM-based NER**

将实体识别视为生成任务，通过提示词直接让 LLM 提取实体。灵活但成本高：

```python
# ch09_llm_ner.py
class LLMNER:
    """基于 LLM 的 NER"""

    def extract(self, text: str, entity_types: List[str]) -> List[dict]:
        prompt = f"""请从以下文本中提取命名实体。

文本："{text}"

需要提取的实体类型：{', '.join(entity_types)}

请以 JSON 格式返回结果，格式为：
[
  {{"text": "实体文本", "type": "实体类型", "start": 起始位置, "end": 结束位置}}
]

只返回 JSON，不要额外解释。"""
        # 实际调用 LLM API
        return []  # Mock 返回值
```

**NER 方法对比：**

| 方法 | 精度 | 召回 | 速度 | 成本 | 泛化能力 | 适用场景 |
|---|---|---|---|---|---|---|
| 词典 NER | 高（已知实体） | 低（未见实体） | 极快 | 极低 | 无 | 固定领域、已知实体集 |
| 模型 NER | 高 | 中高 | 中 | 中（需 GPU） | 强 | 通用场景、开放域 |
| LLM NER | 高 | 高 | 慢 | 高 | 极强 | 灵活需求、长尾实体 |
| 混合 NER | 高 | 高 | 中 | 中 | 强 | 生产环境推荐 |

### 9.2.2 实体消歧（Entity Disambiguation）

实体消歧解决的是"同名不同义"的问题。例如，"苹果"可以指水果也可以指科技公司。

**上下文消歧（Contextual Disambiguation）：**

利用实体周围的上下文词汇来判断其所属类别：

```python
# ch09_entity_disambiguation.py
"""
实体消歧实现：上下文消歧 + 图谱消歧 + 混合消歧
"""

from typing import List, Dict, Optional, Tuple


class ContextualDisambiguator:
    """基于上下文信号词的实体消歧"""

    def __init__(self):
        self.context_signals: Dict[str, Dict[str, List[str]]] = {}

    def register_entity_context(
        self, entity_id: str, context_type: str, signal_words: List[str]
    ):
        if entity_id not in self.context_signals:
            self.context_signals[entity_id] = {}
        self.context_signals[entity_id][context_type] = signal_words

    def disambiguate(
        self, ambiguous_name: str, candidates: List[dict], context_text: str
    ) -> Optional[dict]:
        if len(candidates) == 1:
            return candidates[0]
        scores = []
        for candidate in candidates:
            eid = candidate["id"]
            if eid not in self.context_signals:
                scores.append((candidate, 0))
                continue
            score = sum(
                1 for _, signals in self.context_signals[eid].items()
                for signal in signals if signal in context_text
            )
            scores.append((candidate, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[0][0] if scores[0][1] > 0 else None


class GraphBasedDisambiguator:
    """基于图谱邻居的实体消歧"""

    def __init__(self, adjacency: Dict[str, List[Tuple[str, str]]]):
        self.adjacency = adjacency

    def disambiguate(
        self, ambiguous_name: str, candidates: List[dict], query_entity_ids: List[str]
    ) -> Optional[dict]:
        if len(candidates) == 1:
            return candidates[0]
        other_ids = set(query_entity_ids)
        best_candidate, best_overlap = None, -1
        for candidate in candidates:
            neighbors = set()
            for rel_type, neighbor_id in self.adjacency.get(candidate["id"], []):
                neighbors.add(neighbor_id)
            overlap = len(neighbors & other_ids)
            if overlap > best_overlap:
                best_overlap, best_candidate = overlap, candidate
        return best_candidate


class HybridDisambiguator:
    """混合消歧器：上下文优先，图谱兜底"""

    def __init__(self, contextual: ContextualDisambiguator, graph_based: GraphBasedDisambiguator):
        self.contextual = contextual
        self.graph_based = graph_based

    def disambiguate(
        self, ambiguous_name: str, candidates: List[dict],
        context_text: str, query_entity_ids: List[str]
    ) -> dict:
        result = self.contextual.disambiguate(ambiguous_name, candidates, context_text)
        if result:
            return result
        result = self.graph_based.disambiguate(ambiguous_name, candidates, query_entity_ids)
        return result or candidates[0]
```

### 9.2.3 基于 Embedding 的消歧

```python
# ch09_embedding_disambiguation.py
"""
基于 Embedding 的实体消歧
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity


class EmbeddingDisambiguator:
    """基于 Embedding 的消歧器"""

    def __init__(self, embedding_model=None, dim: int = 768):
        self.embedding_model = embedding_model
        self.dim = dim
        self.prototype_vectors: Dict[str, np.ndarray] = {}
        self.entity_info: Dict[str, dict] = {}

    def register_entity(self, entity_id: str, entity_info: dict, description: str = ""):
        self.entity_info[entity_id] = entity_info
        if self.embedding_model and description:
            vector = self.embedding_model.encode([description])[0]
        else:
            vector = np.random.randn(self.dim)
            vector /= np.linalg.norm(vector)
        self.prototype_vectors[entity_id] = vector

    def disambiguate(
        self, mention: str, candidates: List[dict], context: str
    ) -> Tuple[Optional[dict], float]:
        if len(candidates) == 1:
            return candidates[0], 1.0
        context_vector = (
            self.embedding_model.encode([context])[0]
            if self.embedding_model
            else np.random.randn(self.dim) / np.linalg.norm(np.random.randn(self.dim))
        )
        best_candidate, best_score = None, -1
        for candidate in candidates:
            eid = candidate["id"]
            if eid not in self.prototype_vectors:
                continue
            score = cosine_similarity(
                context_vector.reshape(1, -1),
                self.prototype_vectors[eid].reshape(1, -1),
            )[0][0]
            if score > best_score:
                best_score, best_candidate = score, candidate
        return best_candidate, float(best_score)
```

### 9.2.4 消歧评估指标

```python
# ch09_disambiguation_eval.py
def evaluate_disambiguation(
    predictions: List[Dict], ground_truth: List[Dict]
) -> Dict[str, float]:
    """
    评估实体消歧效果

    Returns: accuracy, precision, recall, f1
    """
    correct = sum(
        1 for g in ground_truth
        if g["mention"] in {p["mention"]: p["entity_id"] for p in predictions}
        and {p["mention"]: p["entity_id"] for p in predictions}.get(g["mention"]) == g["entity_id"]
    )
    total = len(ground_truth)
    pred_count = len(predictions)
    accuracy = correct / max(total, 1)
    precision = correct / max(pred_count, 1)
    recall = correct / max(total, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
```

---

## 9.3 图遍历（Graph Traversal）

完成实体链接后，下一步是以链接到的实体为起点，在知识图谱中进行遍历，收集相关子图信息。

### 9.3.1 自我网络（Ego Network）

EGO 网络是以某个实体为中心，向外扩展 N 跳的子图。这是 KG 检索中最基础也是最有用的操作。

**EGO 网络的定义：**

给定中心节点 v 和深度 k，EGO 网络 EGO(v, k) 包含：
- 所有从 v 出发在 k 跳内可达的节点
- 这些节点之间的所有边

**深度选择策略：**

| 深度 | 覆盖范围 | 适用查询 | 计算开销 |
|---|---|---|---|
| 1 跳 | 直接邻居 | "恒瑞医药生产哪些药品？" | 小 |
| 2 跳 | 间接关系 | "恒瑞医药的药品供应到哪些医院？" | 中 |
| 3 跳 | 多跳推理 | "恒瑞医药的供应商还供应哪些药企？" | 大 |

```python
# ch09_ego_network.py
"""
自我网络（EGO Network）遍历实现
"""

from typing import List, Dict, Set, Optional, Tuple
from collections import deque


class EgoNetworkTraverser:
    """EGO 网络遍历器"""

    def __init__(self, adjacency: Dict[str, List[Tuple[str, str, dict]]]):
        self.adjacency = adjacency

    def traverse(
        self,
        center_id: str,
        max_depth: int = 2,
        relation_filter: Optional[Set[str]] = None,
        direction: str = "both",
        max_neighbors_per_node: int = 50,
    ) -> dict:
        """
        获取实体的 EGO 网络

        Returns:
            {"center": id, "nodes": {id: {depth}}, "edges": [...],
             "depth_distribution": {depth: count}, "total_nodes": N, "total_edges": N}
        """
        visited: Dict[str, int] = {center_id: 0}
        queue = deque([(center_id, 0)])
        nodes: Dict[str, dict] = {}
        edges: List[dict] = []

        while queue:
            current_id, depth = queue.popleft()
            if current_id not in nodes:
                nodes[current_id] = {"depth": depth}
            if depth >= max_depth:
                continue

            neighbors = self.adjacency.get(current_id, [])
            if relation_filter:
                neighbors = [n for n in neighbors if n[0] in relation_filter]
            neighbors = neighbors[:max_neighbors_per_node]

            for rel_type, neighbor_id, props in neighbors:
                edges.append({
                    "source": current_id, "target": neighbor_id,
                    "type": rel_type, "properties": props,
                })
                if neighbor_id not in visited:
                    visited[neighbor_id] = depth + 1
                    nodes[neighbor_id] = {"depth": depth + 1}
                    queue.append((neighbor_id, depth + 1))

        depth_dist = {}
        for info in nodes.values():
            d = info["depth"]
            depth_dist[d] = depth_dist.get(d, 0) + 1

        return {
            "center": center_id, "nodes": nodes, "edges": edges,
            "depth_distribution": depth_dist,
            "total_nodes": len(nodes), "total_edges": len(edges),
        }

    def format_ego_network(self, network: dict, entity_names: Dict[str, str]) -> str:
        """将 EGO 网络格式化为可读文本"""
        lines = [
            f"中心实体: {entity_names.get(network['center'], network['center'])}",
            f"节点数: {network['total_nodes']}, 边数: {network['total_edges']}",
            f"深度分布: {network['depth_distribution']}", "",
        ]
        for depth in sorted(network["depth_distribution"].keys()):
            depth_nodes = [nid for nid, info in network["nodes"].items() if info["depth"] == depth]
            lines.append(f"--- 深度 {depth} ({len(depth_nodes)} 个节点) ---")
            if depth == 0:
                continue
            for nid in depth_nodes:
                for edge in network["edges"]:
                    if edge["target"] == nid:
                        lines.append(
                            f"  {entity_names.get(edge['source'], edge['source'])} "
                            f"--[{edge['type']}]--> {entity_names.get(nid, nid)}"
                        )
        return "\n".join(lines)
```

### 9.3.2 最短路径查找

最短路径用于回答"实体 A 和实体 B 之间如何关联"这类问题。例如，"华海药业和北京协和医院之间有什么关系？"

```python
# ch09_shortest_path.py
"""
最短路径查找：BFS、双向 BFS、Dijkstra、Top-K 路径
"""

from typing import List, Dict, Optional, Tuple
from collections import deque
import heapq


class ShortestPathFinder:
    """最短路径查找器"""

    def __init__(self, adjacency: Dict[str, List[Tuple[str, str, dict]]]):
        self.adjacency = adjacency

    def bfs_shortest_path(
        self, source_id: str, target_id: str, max_depth: int = 5
    ) -> Optional[List[str]]:
        """BFS 最短路径（无权图）"""
        if source_id == target_id:
            return [source_id]
        visited = {source_id}
        queue = deque([[source_id]])
        while queue:
            path = queue.popleft()
            current = path[-1]
            if len(path) - 1 >= max_depth:
                continue
            for rel_type, neighbor_id, _ in self.adjacency.get(current, []):
                if neighbor_id == target_id:
                    return path + [neighbor_id]
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append(path + [neighbor_id])
        return None

    def bidirectional_bfs(
        self, source_id: str, target_id: str, max_depth: int = 5
    ) -> Optional[List[str]]:
        """
        双向 BFS

        搜索空间从 O(b^d) 降为 O(b^(d/2))。
        b = 分支因子，d = 路径深度
        """
        if source_id == target_id:
            return [source_id]
        fwd_visited = {source_id: (None, None)}
        fwd_queue = deque([source_id])
        bwd_visited = {target_id: (None, None)}
        bwd_queue = deque([target_id])
        rev_adj = self._build_reverse_adjacency()

        for _ in range(max_depth // 2 + 1):
            meeting = self._bfs_layer(fwd_queue, fwd_visited, self.adjacency, bwd_visited)
            if meeting:
                return self._reconstruct_path(meeting, fwd_visited, bwd_visited)
            meeting = self._bfs_layer(bwd_queue, bwd_visited, rev_adj, fwd_visited)
            if meeting:
                return self._reconstruct_path(meeting, fwd_visited, bwd_visited)
        return None

    def _build_reverse_adjacency(self) -> Dict[str, List[Tuple[str, str, dict]]]:
        rev = {}
        for node, neighbors in self.adjacency.items():
            for rel_type, neighbor_id, props in neighbors:
                rev.setdefault(neighbor_id, []).append((rel_type, node, props))
        return rev

    def _bfs_layer(self, queue, visited, adj, other_visited):
        level_size = len(queue)
        for _ in range(level_size):
            current = queue.popleft()
            for rel_type, neighbor_id, _ in adj.get(current, []):
                if neighbor_id in visited:
                    continue
                visited[neighbor_id] = (current, rel_type)
                if neighbor_id in other_visited:
                    return neighbor_id
                queue.append(neighbor_id)
        return None

    def _reconstruct_path(self, meeting, fwd_visited, bwd_visited):
        fwd_path = []
        node = meeting
        while node is not None:
            fwd_path.append(node)
            prev, _ = fwd_visited.get(node, (None, None))
            node = prev
        fwd_path.reverse()
        bwd_path = []
        node = meeting
        while node is not None:
            prev, _ = bwd_visited.get(node, (None, None))
            if prev is not None:
                bwd_path.append(prev)
            node = prev
        return fwd_path + bwd_path

    def top_k_paths(
        self, source_id: str, target_id: str, k: int = 3, max_depth: int = 5
    ) -> List[List[str]]:
        """Top-K 最短路径（Yen 算法简化版）"""
        first = self.bfs_shortest_path(source_id, target_id, max_depth)
        if first is None:
            return []
        paths, candidates = [first], []
        for i in range(1, k):
            last = paths[-1]
            for j in range(len(last) - 1):
                spur_node = last[j]
                removed = []
                for p in paths:
                    if len(p) > j and p[:j+1] == last[:j+1]:
                        removed.append((p[j], p[j+1]))
                spur = self._find_spur(spur_node, target_id, last[:j], removed, max_depth - j)
                if spur:
                    candidates.append(last[:j] + spur)
            if not candidates:
                break
            candidates.sort(key=len)
            paths.append(candidates.pop(0))
        return paths[:k]

    def _find_spur(self, spur_node, target_id, root_path, removed_edges, max_depth):
        temp_adj = {}
        for node, neighbors in self.adjacency.items():
            if node in root_path and node != spur_node:
                continue
            temp_adj[node] = [(r, n, p) for r, n, p in neighbors if (node, n) not in removed_edges]
        finder = ShortestPathFinder(temp_adj)
        return finder.bfs_shortest_path(spur_node, target_id, max_depth)

    def dijkstra_shortest_path(
        self, source_id: str, target_id: str, weight_attr: str = "weight"
    ) -> Optional[Tuple[List[str], float]]:
        """Dijkstra 最短路径（带权图）"""
        distances = {source_id: 0.0}
        prev = {source_id: None}
        pq = [(0.0, source_id)]
        visited = set()
        while pq:
            dist, current = heapq.heappop(pq)
            if current in visited:
                continue
            visited.add(current)
            if current == target_id:
                path = []
                node = current
                while node is not None:
                    path.append(node)
                    node = prev[node]
                path.reverse()
                return path, dist
            for rel_type, neighbor_id, props in self.adjacency.get(current, []):
                if neighbor_id in visited:
                    continue
                weight = props.get(weight_attr, 1.0)
                new_dist = dist + weight
                if neighbor_id not in distances or new_dist < distances[neighbor_id]:
                    distances[neighbor_id] = new_dist
                    prev[neighbor_id] = current
                    heapq.heappush(pq, (new_dist, neighbor_id))
        return None

    def format_path(self, path: List[str], entity_names: Dict[str, str]) -> str:
        """格式化路径为可读字符串"""
        if not path:
            return "未找到路径"
        return " → ".join(entity_names.get(nid, nid) for nid in path)
```

### 9.3.3 路径搜索的剪枝策略

在大规模知识图谱中，不加限制的路径搜索会导致组合爆炸。常用的剪枝策略包括：

| 策略 | 方法 | 效果 |
|---|---|---|
| 方向约束 | 只沿出边或入边搜索 | 减少 50% 分支 |
| 关系类型过滤 | 只搜索特定关系 | 减少 70-90% 分支 |
| 跳数限制 | 设定最大深度（通常 ≤ 5） | 控制搜索空间 |
| 邻居采样 | 每层只取 Top-N 个邻居 | 控制分支因子 |
| 分数剪枝 | 只保留分数高于阈值的路径 | 聚焦高质量路径 |

```python
# ch09_path_pruning.py
class PrunedPathFinder:
    """带剪枝的路径查找器"""

    def __init__(
        self,
        adjacency: Dict[str, List[Tuple[str, str, dict]]],
        relation_scores: Optional[Dict[str, float]] = None,
    ):
        self.adjacency = adjacency
        self.relation_scores = relation_scores or {}

    def find_path_with_pruning(
        self, source_id: str, target_id: str,
        max_depth: int = 5, max_branch: int = 10,
        allowed_relations: Optional[Set[str]] = None,
    ) -> Optional[List[str]]:
        visited = {source_id}
        queue = deque([[source_id]])
        while queue:
            path = queue.popleft()
            current = path[-1]
            if len(path) - 1 >= max_depth:
                continue
            neighbors = self.adjacency.get(current, [])
            if allowed_relations:
                neighbors = [n for n in neighbors if n[0] in allowed_relations]
            neighbors.sort(key=lambda x: self.relation_scores.get(x[0], 0), reverse=True)
            neighbors = neighbors[:max_branch]
            for rel_type, neighbor_id, _ in neighbors:
                if neighbor_id == target_id:
                    return path + [neighbor_id]
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append(path + [neighbor_id])
        return None
```

### 9.3.4 子图匹配

对于更复杂的查询模式（如"找出所有同时使用恒瑞医药和齐鲁制药药品的医院"），需要子图匹配（Subgraph Matching）技术。

```python
# ch09_subgraph_match.py
class SubgraphMatcher:
    """子图匹配器"""

    def __init__(self, adjacency: Dict[str, List[Tuple[str, str, dict]]]):
        self.adjacency = adjacency

    def match_path_pattern(
        self, start_type: str, end_type: str,
        relation_chain: List[str], max_results: int = 10
    ) -> List[List[str]]:
        """匹配路径模式，如 Company-[PRODUCES]->Drug-[PURCHASES]->Hospital"""
        results = []
        start_entities = self._get_entities_by_type(start_type)
        for sid in start_entities:
            for path in self._expand_path(sid, relation_chain, 0, end_type):
                results.append(path)
                if len(results) >= max_results:
                    return results
        return results

    def match_star_pattern(
        self, center_type: str,
        leaf_patterns: List[Tuple[str, str]], max_results: int = 10
    ) -> List[dict]:
        """匹配星型模式"""
        results = []
        for center_id in self._get_entities_by_type(center_type):
            match = {"center": center_id}
            all_matched = True
            for rel_type, leaf_type in leaf_patterns:
                leaves = self._find_neighbors_by_type(center_id, rel_type, leaf_type)
                if not leaves:
                    all_matched = False
                    break
                match[f"leaf_{rel_type}"] = leaves
            if all_matched:
                results.append(match)
                if len(results) >= max_results:
                    return results
        return results

    def _get_entities_by_type(self, entity_type: str) -> List[str]:
        return []  # 需要实体类型索引

    def _find_neighbors_by_type(self, eid: str, rel_type: str, ntype: str) -> List[str]:
        return [nid for r, nid, _ in self.adjacency.get(eid, []) if r == rel_type]

    def _expand_path(self, cid: str, rels: List[str], depth: int, end_type: str) -> List[List[str]]:
        if depth >= len(rels):
            return [[cid]]
        results = []
        for r, nid, _ in self.adjacency.get(cid, []):
            if r == rels[depth]:
                for sub in self._expand_path(nid, rels, depth + 1, end_type):
                    results.append([cid] + sub)
        return results
```

---

## 9.4 KG 增强的检索（KG-Augmented Retrieval）

### 9.4.1 检索流程概览

KG 增强的检索流程分为以下步骤：

```
用户查询
    |
    v
[1. 实体链接] ──→ 从查询中识别实体
    |
    v
[2. 图遍历] ────→ 以识别到的实体为起点进行 N 跳遍历
    |
    v
[3. 上下文构建] ──→ 将图结构格式化为 LLM 可读的文本
    |
    v
[4. 答案生成] ────→ 将图谱上下文 + 文本段落送入 LLM
```

### 9.4.2 完整的 KG 检索器实现

```python
# ch09_kg_retriever.py
"""
KG 增强检索器完整实现
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque


@dataclass
class Entity:
    id: str
    name: str
    type: str
    description: str = ""
    aliases: List[str] = field(default_factory=list)
    properties: Dict = field(default_factory=dict)


@dataclass
class Relation:
    source_id: str
    target_id: str
    type: str
    properties: Dict = field(default_factory=dict)


class KnowledgeGraph:
    """内存知识图谱"""

    def __init__(self):
        self.entities: Dict[str, Entity] = {}
        self.relations: List[Relation] = []
        self.forward_adj: Dict[str, List[Tuple[str, str, dict]]] = {}
        self.backward_adj: Dict[str, List[Tuple[str, str, dict]]] = {}
        self.name_index: Dict[str, str] = {}
        self.type_index: Dict[str, List[str]] = {}

    def add_entity(self, entity: Entity):
        self.entities[entity.id] = entity
        self.name_index[entity.name] = entity.id
        for alias in entity.aliases:
            self.name_index[alias] = entity.id
        self.type_index.setdefault(entity.type, []).append(entity.id)

    def add_relation(self, relation: Relation):
        self.relations.append(relation)
        props = relation.properties
        self.forward_adj.setdefault(relation.source_id, []).append(
            (relation.type, relation.target_id, props)
        )
        self.backward_adj.setdefault(relation.target_id, []).append(
            (relation.type, relation.source_id, props)
        )


class KGRetriever:
    """KG 增强检索器"""

    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg

    def _link_entities(self, text: str) -> List[Tuple[str, str]]:
        """实体链接：最长匹配优先"""
        found = []
        for name in sorted(self.kg.name_index.keys(), key=len, reverse=True):
            if name in text:
                eid = self.kg.name_index[name]
                if eid not in [e[0] for e in found]:
                    found.append((eid, self.kg.entities[eid].name))
        return found

    def _ego_network(self, entity_id: str, max_depth: int = 2) -> dict:
        """EGO 网络遍历"""
        visited = {entity_id: 0}
        queue = deque([(entity_id, 0)])
        nodes, edges = {}, []
        while queue:
            current, depth = queue.popleft()
            if current not in nodes:
                ent = self.kg.entities.get(current)
                if ent:
                    nodes[current] = {"id": current, "name": ent.name, "type": ent.type}
            if depth >= max_depth:
                continue
            for rel_type, neighbor_id, props in self.kg.forward_adj.get(current, []):
                if neighbor_id not in visited:
                    visited[neighbor_id] = depth + 1
                    queue.append((neighbor_id, depth + 1))
                edges.append({"source": current, "target": neighbor_id, "type": rel_type, "direction": "forward"})
                if neighbor_id not in nodes:
                    n = self.kg.entities.get(neighbor_id)
                    if n:
                        nodes[neighbor_id] = {"id": neighbor_id, "name": n.name, "type": n.type}
            for rel_type, source_id, props in self.kg.backward_adj.get(current, []):
                if source_id not in visited:
                    visited[source_id] = depth + 1
                    queue.append((source_id, depth + 1))
                edges.append({"source": source_id, "target": current, "type": rel_type, "direction": "backward"})
                if source_id not in nodes:
                    s = self.kg.entities.get(source_id)
                    if s:
                        nodes[source_id] = {"id": source_id, "name": s.name, "type": s.type}
        return {"nodes": nodes, "edges": edges}

    def _shortest_path(self, source_id: str, target_id: str, max_depth: int = 5) -> Optional[List[str]]:
        if source_id == target_id:
            return [source_id]
        visited = {source_id}
        queue = deque([[source_id]])
        while queue:
            path = queue.popleft()
            current = path[-1]
            if len(path) - 1 >= max_depth:
                continue
            for r, nid, _ in self.kg.forward_adj.get(current, []):
                if nid == target_id:
                    return path + [nid]
                if nid not in visited:
                    visited.add(nid)
                    queue.append(path + [nid])
            for r, sid, _ in self.kg.backward_adj.get(current, []):
                if sid == target_id:
                    return path + [sid]
                if sid not in visited:
                    visited.add(sid)
                    queue.append(path + [sid])
        return None

    def retrieve(self, query: str, max_depth: int = 2) -> dict:
        """执行 KG 检索"""
        linked = self._link_entities(query)
        if not linked:
            return {"query": query, "entities_found": [], "contexts": [], "paths": [],
                    "formatted_context": "未在查询中识别到知识图谱实体。"}

        contexts = []
        for eid, ename in linked:
            network = self._ego_network(eid, max_depth)
            entity = self.kg.entities[eid]
            lines = [f"实体: {entity.name}", f"  类型: {entity.type}"]
            if entity.description:
                lines.append(f"  描述: {entity.description}")
            for edge in network["edges"]:
                if edge["direction"] == "forward" and edge["source"] == eid:
                    tgt_name = network["nodes"].get(edge["target"], {}).get("name", edge["target"])
                    lines.append(f"  -> [{edge['type']}] {tgt_name}")
                if edge["direction"] == "backward" and edge["target"] == eid:
                    src_name = network["nodes"].get(edge["source"], {}).get("name", edge["source"])
                    lines.append(f"  <- [{edge['type']}] {src_name}")
            contexts.append({"entity_id": eid, "entity_name": ename, "lines": lines})

        paths = []
        if len(linked) >= 2:
            for i in range(len(linked)):
                for j in range(i + 1, len(linked)):
                    path = self._shortest_path(linked[i][0], linked[j][0])
                    if path:
                        names = [self.kg.entities[pid].name for pid in path]
                        paths.append({"from": linked[i][1], "to": linked[j][1],
                                      "path": " → ".join(names), "length": len(path) - 1})

        formatted = self._format_context(contexts, paths, query)
        return {"query": query, "entities_found": [{"id": e[0], "name": e[1]} for e in linked],
                "contexts": contexts, "paths": paths, "formatted_context": formatted}

    def _format_context(self, contexts, paths, query) -> str:
        parts = [f"用户查询: {query}", ""]
        if not contexts:
            parts.append("未找到相关图谱信息。")
            return "\n".join(parts)
        parts.append("=== 知识图谱上下文 ===")
        parts.append("")
        for ctx in contexts:
            parts.extend(ctx["lines"])
            parts.append("")
        if paths:
            parts.append("--- 实体间关联路径 ---")
            for p in paths:
                parts.append(f"  {p['from']} → {p['to']}: {p['path']}（{p['length']} 跳）")
            parts.append("")
        return "\n".join(parts)
```

### 9.4.3 构建医药供应链知识图谱

```python
# ch09_build_pharma_kg.py
def build_pharma_kg() -> KnowledgeGraph:
    """构建医药供应链知识图谱"""
    kg = KnowledgeGraph()

    entities = [
        Entity("e1", "恒瑞医药", "company", "中国领先制药企业，专注于抗肿瘤药物", ["恒瑞", "恒瑞医药有限公司"]),
        Entity("e2", "齐鲁制药", "company", "中国主要制药企业，生产抗肿瘤和抗感染药物", ["齐鲁"]),
        Entity("e3", "注射用紫杉醇", "drug", "抗肿瘤药物，用于非小细胞肺癌和乳腺癌", ["紫杉醇", "紫杉醇注射剂"]),
        Entity("e4", "卡瑞利珠单抗", "drug", "PD-1抑制剂，用于霍奇金淋巴瘤治疗"),
        Entity("e5", "吉非替尼片", "drug", "EGFR-TKI靶向药，用于非小细胞肺癌", ["吉非替尼"]),
        Entity("e6", "华海药业", "company", "紫杉醇API原料药供应商，年产能5000kg", ["华海"]),
        Entity("e7", "国药控股", "company", "中国最大药品分销商，华东区覆盖37家三甲医院", ["国药"]),
        Entity("e8", "北京协和医院", "hospital", "三级甲等综合医院", ["协和医院"]),
        Entity("e9", "华东医院", "hospital", "三级甲等综合医院，位于上海"),
        Entity("e10", "紫杉醇API", "chemical", "紫杉醇原料药，用于制备注射用紫杉醇", ["原料药"]),
        Entity("e11", "国家药监局", "regulator", "药品监督管理机构", ["NMPA", "药监局"]),
        Entity("e12", "正大天晴", "company", "中国制药企业"),
        Entity("e13", "阿斯利康", "company", "跨国制药企业", ["AstraZeneca"]),
        Entity("e14", "奥希替尼", "drug", "第三代EGFR-TKI靶向药", ["泰瑞沙"]),
    ]
    for e in entities:
        kg.add_entity(e)

    relations = [
        Relation("e1", "e3", "PRODUCES", {"since": 2010, "share": "100%"}),
        Relation("e1", "e4", "PRODUCES", {"since": 2019}),
        Relation("e2", "e5", "PRODUCES", {"since": 2015}),
        Relation("e13", "e14", "PRODUCES", {"since": 2017}),
        Relation("e6", "e10", "SUPPLIES", {"annual_volume": "5000kg"}),
        Relation("e10", "e3", "IS_RAW_MATERIAL_OF", {}),
        Relation("e7", "e1", "DISTRIBUTES", {"region": "华东区", "contract_value": "5亿元/年"}),
        Relation("e7", "e2", "DISTRIBUTES", {"region": "华东区"}),
        Relation("e8", "e3", "PURCHASES", {"annual_volume": "50000支"}),
        Relation("e8", "e5", "PURCHASES", {"annual_volume": "20000盒"}),
        Relation("e9", "e3", "PURCHASES", {"annual_volume": "30000支"}),
        Relation("e11", "e3", "REGULATES", {"approval_number": "国药准字H20000001"}),
        Relation("e11", "e4", "REGULATES", {}),
        Relation("e1", "e12", "COOPERATES_WITH", {}),
        Relation("e1", "e13", "COOPERATES_WITH", {}),
        Relation("e1", "e2", "COMPETES_WITH", {}),
    ]
    for r in relations:
        kg.add_relation(r)

    print(f"[KG] 图谱构建完成: {len(kg.entities)} 个实体, {len(kg.relations)} 条关系")
    return kg
```

### 9.4.4 检索演示

```python
# ch09_retrieval_demo.py
def demo():
    kg = build_pharma_kg()
    retriever = KGRetriever(kg)

    queries = [
        "恒瑞医药生产哪些药品？",
        "北京协和医院采购了哪些药品？",
        "恒瑞医药的紫杉醇原料药从哪里来？",
    ]

    for q in queries:
        print(f"\n{'='*60}\n查询: {q}\n{'='*60}")
        result = retriever.retrieve(q)
        print(f"\n识别实体: {[e['name'] for e in result['entities_found']]}")
        print(f"\n{result['formatted_context']}")

    # 最短路径演示
    traverser = GraphTraverser(kg)
    print(f"\n{'='*60}\n路径: 华海药业 → 华东医院\n{'='*60}")
    path = traverser.shortest_path("e6", "e9")
    if path:
        print("  " + " → ".join([kg.entities[pid].name for pid in path]))


if __name__ == "__main__":
    demo()
```

---

## 9.5 图上下文构建与 Prompt 格式化

### 9.5.1 为什么需要图上下文格式化

知识图谱的数据结构（三元组、邻接表）对人类或 LLM 来说并不直观。将图结构转化为 LLM 能理解的自然语言是 KG 增强 RAG 的关键步骤。

### 9.5.2 多种格式化策略

```python
# ch09_context_formatter.py
class GraphContextFormatter:
    """图上下文格式化器"""

    def format(self, network: dict, entity_names: Dict[str, str],
               strategy: str = "hierarchical", max_entities: int = 20) -> str:
        if strategy == "hierarchical":
            return self._format_hierarchical(network, entity_names, max_entities)
        elif strategy == "triple_list":
            return self._format_triple_list(network, entity_names, max_entities)
        elif strategy == "narrative":
            return self._format_narrative(network, entity_names, max_entities)
        raise ValueError(f"未知策略: {strategy}")

    def _format_hierarchical(self, network, names, max_entities):
        nodes, edges = network.get("nodes", {}), network.get("edges", [])
        degree = {}
        for e in edges:
            degree[e["source"]] = degree.get(e["source"], 0) + 1
            degree[e["target"]] = degree.get(e["target"], 0) + 1
        center = max(degree, key=degree.get) if degree else list(nodes.keys())[0]
        lines = [f"# {names.get(center, center)} 的关系网络", ""]
        out_edges = [e for e in edges if e["source"] == center]
        if out_edges:
            lines.append(f"## {names.get(center, center)} 的关系")
            for e in out_edges[:max_entities]:
                lines.append(f"  - [{e['type']}] → {names.get(e['target'], e['target'])}")
            lines.append("")
        in_edges = [e for e in edges if e["target"] == center]
        if in_edges:
            lines.append(f"## 关联到 {names.get(center, center)} 的实体")
            for e in in_edges[:max_entities]:
                lines.append(f"  - {names.get(e['source'], e['source'])} → [{e['type']}]")
            lines.append("")
        return "\n".join(lines)

    def _format_triple_list(self, network, names, max_entities):
        lines = ["知识图谱三元组:"]
        for e in network.get("edges", [])[:max_entities]:
            lines.append(f"  ({names.get(e['source'], e['source'])}, {e['type']}, {names.get(e['target'], e['target'])})")
        return "\n".join(lines)

    def _format_narrative(self, network, names, max_entities):
        nodes, edges = network.get("nodes", {}), network.get("edges", [])
        degree = {}
        for e in edges:
            degree[e["source"]] = degree.get(e["source"], 0) + 1
            degree[e["target"]] = degree.get(e["target"], 0) + 1
        center = max(degree, key=degree.get) if degree else list(nodes.keys())[0]
        cname = names.get(center, center)
        produces, supplied_by, purchased_by = [], [], []
        for e in edges[:max_entities]:
            src, tgt = names.get(e["source"], e["source"]), names.get(e["target"], e["target"])
            if e["source"] == center and e["type"] in ("PRODUCES", "SUPPLIES"):
                produces.append(tgt)
            elif e["target"] == center and e["type"] == "SUPPLIES":
                supplied_by.append(src)
            elif e["target"] == center and e["type"] == "PURCHASES":
                purchased_by.append(src)
        parts = [f"关于{cname}:"]
        if produces:
            parts.append(f"  它生产/供应: {', '.join(produces)}。")
        if supplied_by:
            parts.append(f"  它的供应商: {', '.join(supplied_by)}。")
        if purchased_by:
            parts.append(f"  采购它的机构: {', '.join(purchased_by)}。")
        return "\n".join(parts)
```

### 9.5.3 Prompt 模板设计

```python
# ch09_prompt_templates.py
class KGPromptTemplates:
    """KG 增强 RAG 的 Prompt 模板"""

    @staticmethod
    def kg_qa(graph_context: str, text_context: str, query: str) -> str:
        return f"""你是一个知识图谱增强的问答助手。请基于以下信息回答问题。

## 知识图谱事实（精确关系信息）
{graph_context}

## 文本背景信息（详细描述）
{text_context}

## 问题
{query}

## 要求
1. 优先使用知识图谱中的精确事实进行回答
2. 使用文本背景信息补充细节
3. 如果两类信息冲突，说明冲突并给出判断
4. 如果信息不足，请明确说明

## 回答"""

    @staticmethod
    def multi_hop(path_description: str, query: str) -> str:
        return f"""你是一个多跳推理助手。请分析以下关系路径，然后回答问题。

## 关系路径
{path_description}

## 问题
{query}

## 要求
1. 沿着路径逐跳分析推理过程
2. 每跳都说明：从哪个实体、通过什么关系、到达哪个实体
3. 综合所有跳的信息得出结论

## 推理过程"""

    @staticmethod
    def compare_entities(entity_a: str, entity_b: str, graph_context: str, query: str) -> str:
        return f"""请比较以下两个实体的关系网络。

## 实体 A: {entity_a}
## 实体 B: {entity_b}

## 知识图谱信息
{graph_context}

## 问题
{query}

## 要求
1. 列出两个实体的共同点
2. 列出两个实体的不同点
3. 如果存在间接关系路径，描述出来

## 对比分析"""
```

---

## 9.6 Neo4j Cypher 查询

### 9.6.1 Neo4j 简介

Neo4j 是当前最流行的图数据库。在 KG 增强的 RAG 系统中，Neo4j 承担着"持久化知识图谱存储"的角色。与内存中的图结构相比，Neo4j 的优势在于：

- **持久化**：数据写入磁盘，不会因进程重启而丢失
- **查询语言**：Cypher 查询语言专门为图遍历优化
- **索引**：支持属性索引和全文索引
- **事务**：支持 ACID 事务
- **规模**：可处理亿级节点和关系

### 9.6.2 Neo4j 连接与操作

```python
# ch09_neo4j_operations.py
from neo4j import GraphDatabase, basic_auth


class Neo4jManager:
    """Neo4j 图数据库管理器"""

    def __init__(self, uri: str = "bolt://localhost:7687", user: str = "neo4j", password: str = "password"):
        self.uri, self.user, self.password = uri, user, password
        self.driver = None

    def connect(self):
        self.driver = GraphDatabase.driver(self.uri, auth=basic_auth(self.user, self.password))
        with self.driver.session() as session:
            result = session.run("RETURN 1 AS n")
            print(f"[Neo4j] 连接成功: {result.single()['n']}")

    def close(self):
        if self.driver:
            self.driver.close()

    def create_constraints(self):
        with self.driver.session() as session:
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE")
            session.run("CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.name)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.type)")
            print("[Neo4j] 约束和索引创建完成")

    def clear_database(self):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("[Neo4j] 数据库已清空")
```

### 9.6.3 Cypher 查询示例

以下是一组完整的 Cypher 查询示例，涵盖从数据创建到复杂图遍历的所有操作：

**1. Schema 定义与数据创建：**

```cypher
-- 创建实体节点
CREATE (e1:Company {id: 'e1', name: '恒瑞医药', type: 'company', description: '中国领先制药企业'})
CREATE (e3:Drug {id: 'e3', name: '注射用紫杉醇', type: 'drug', description: '抗肿瘤药物'})
CREATE (e8:Hospital {id: 'e8', name: '北京协和医院', type: 'hospital'})
CREATE (e11:Regulator {id: 'e11', name: '国家药监局', type: 'regulator'})

-- 创建关系
MATCH (e1:Company {id: 'e1'}), (e3:Drug {id: 'e3'})
CREATE (e1)-[:PRODUCES {since: 2010}]->(e3)

MATCH (e8:Hospital {id: 'e8'}), (e3:Drug {id: 'e3'})
CREATE (e8)-[:PURCHASES {annual_volume: '50000支'}]->(e3)

MATCH (e11:Regulator {id: 'e11'}), (e3:Drug {id: 'e3'})
CREATE (e11)-[:REGULATES {approval_number: '国药准字H20000001'}]->(e3)
```

**2. 实体查询：**

```cypher
-- 按名称查询
MATCH (e:Entity {name: '恒瑞医药'})
RETURN e.id, e.name, e.type, e.description

-- 按类型查询
MATCH (e:Entity) WHERE e.type = 'company'
RETURN e.name, e.description ORDER BY e.name

-- 模糊匹配
MATCH (e:Entity) WHERE e.name CONTAINS '紫杉醇'
RETURN e.name, e.type
```

**3. 邻居查询（EGO 网络）：**

```cypher
-- 1 跳邻居
MATCH (e:Company {name: '恒瑞医药'})-[r]->(neighbor)
RETURN e.name AS source, type(r) AS relation, neighbor.name AS target, neighbor.type AS target_type

-- 2 跳邻居（可变长度路径）
MATCH path = (e:Company {name: '恒瑞医药'})-[:*1..2]-(neighbor)
RETURN path LIMIT 50

-- 带关系类型过滤
MATCH (e:Company {name: '恒瑞医药'})-[:PRODUCES]->(drugs)
RETURN drugs.name AS drug_name, drugs.description
```

**4. 最短路径查询：**

```cypher
-- 使用 shortestPath 函数
MATCH path = shortestPath(
    (a:Entity {name: '华海药业'})-[:*]-(b:Entity {name: '华东医院'})
)
RETURN [node IN nodes(path) | node.name] AS entity_path,
       [rel IN relationships(path) | type(rel)] AS relation_path,
       length(path) AS path_length

-- 限制最大跳数
MATCH path = shortestPath(
    (a:Entity {name: '华海药业'})-[:*..5]-(b:Entity {name: '华东医院'})
)
RETURN path

-- 多条路径按长度排序
MATCH path = (a:Entity {name: '恒瑞医药'})-[:*..4]-(b:Entity {name: '北京协和医院'})
RETURN path, length(path) AS len ORDER BY len LIMIT 5
```

**5. 子图匹配：**

```cypher
-- 找"药品-公司-医院"三角形模式
MATCH (d:Drug)<-[:PRODUCES]-(c:Company),
      (d:Drug)<-[:PURCHASES]-(h:Hospital)
RETURN d.name AS drug, c.name AS company, h.name AS hospital LIMIT 10

-- 找同时分销多家公司的分销商
MATCH (d:Company)-[:DISTRIBUTES]->(c1:Company),
      (d:Company)-[:DISTRIBUTES]->(c2:Company)
WHERE c1.id < c2.id
RETURN d.name AS distributor, c1.name AS company1, c2.name AS company2
```

**6. 聚合查询：**

```cypher
-- 统计每种实体类型的数量
MATCH (e:Entity)
RETURN e.type AS entity_type, count(*) AS count ORDER BY count DESC

-- 统计每家公司生产的药品数量
MATCH (c:Company)-[:PRODUCES]->(d:Drug)
RETURN c.name AS company, count(d) AS drug_count ORDER BY drug_count DESC

-- 统计每家医院采购的药品数量
MATCH (h:Hospital)-[:PURCHASES]->(d:Drug)
RETURN h.name AS hospital, count(d) AS drug_count ORDER BY drug_count DESC
```

**7. 路径展开：**

```cypher
-- 将路径展开为行
MATCH path = (a:Entity {name: '恒瑞医药'})-[:*..3]-(b:Entity)
WITH path, nodes(path) AS ns, relationships(path) AS rs
UNWIND range(0, size(ns) - 2) AS i
RETURN ns[i].name AS from_node, type(rs[i]) AS relation, ns[i+1].name AS to_node, ns[i+1].type AS to_node_type
```

**8. 属性查询：**

```cypher
-- 带属性约束的关系查询
MATCH (e:Entity)-[r:PURCHASES]->(d:Drug)
WHERE r.annual_volume IS NOT NULL
  AND toInteger(split(r.annual_volume, '支')[0]) > 30000
RETURN e.name, d.name, r.annual_volume

-- 带时间范围的关系查询
MATCH (c:Company)-[r:PRODUCES]->(d:Drug)
WHERE r.since >= 2015
RETURN c.name, d.name, r.since ORDER BY r.since
```

**9. 全文搜索（需配置全文索引）：**

```cypher
-- 创建全文索引
CALL db.index.fulltext.createNodeIndex('entity_fulltext', ['Entity'], ['name', 'description'])

-- 使用全文搜索
CALL db.index.fulltext.queryNodes('entity_fulltext', '抗肿瘤')
YIELD node, score
RETURN node.name, node.type, score
ORDER BY score DESC LIMIT 10
```

### 9.6.4 Python 驱动的 Cypher 查询

```python
# ch09_neo4j_queries.py
class Neo4jKGRetriever:
    """基于 Neo4j 的 KG 检索器"""

    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def _run(self, query: str, params: dict = None) -> List[dict]:
        with self.driver.session() as session:
            return [record.data() for record in session.run(query, params or {})]

    def find_entity_by_name(self, name: str) -> Optional[dict]:
        results = self._run(
            "MATCH (e:Entity {name: $name}) RETURN e.id AS id, e.name AS name, e.type AS type, e.description AS description",
            {"name": name},
        )
        return results[0] if results else None

    def get_ego_network(self, entity_id: str, max_depth: int = 2) -> dict:
        query = """
        MATCH path = (e:Entity {id: $entity_id})-[:*..$max_depth]-(neighbor)
        WITH e, path, nodes(path) AS ns, relationships(path) AS rs
        UNWIND range(0, size(ns) - 2) AS i
        RETURN DISTINCT ns[i].id AS source_id, ns[i].name AS source_name, ns[i].type AS source_type,
               type(rs[i]) AS relation, ns[i+1].id AS target_id, ns[i+1].name AS target_name,
               ns[i+1].type AS target_type ORDER BY i
        """
        results = self._run(query, {"entity_id": entity_id, "max_depth": max_depth})
        nodes, edges = {}, []
        for row in results:
            for key, nid_key, name_key, type_key in [
                ("source_id", "source_name", "source_type"), ("target_id", "target_name", "target_type")
            ]:
                nid = row[key[0]]
                if nid not in nodes:
                    nodes[nid] = {"id": nid, "name": row[key[1]], "type": row[key[2]]}
            edges.append({"source": row["source_id"], "target": row["target_id"], "type": row["relation"]})
        return {"nodes": nodes, "edges": edges}

    def find_shortest_path(self, source_name: str, target_name: str, max_depth: int = 5) -> Optional[Dict]:
        query = """
        MATCH path = shortestPath(
            (a:Entity {name: $source_name})-[:*..$max_depth]-(b:Entity {name: $target_name})
        )
        RETURN [node IN nodes(path) | node.name] AS entity_names,
               [rel IN relationships(path) | type(rel)] AS relation_types,
               length(path) AS path_length
        """
        results = self._run(query, {"source_name": source_name, "target_name": target_name, "max_depth": max_depth})
        return results[0] if results else None

    def get_entity_statistics(self) -> Dict[str, int]:
        results = self._run("MATCH (e:Entity) RETURN e.type AS type, count(*) AS count ORDER BY count DESC")
        stats = {"total_entities": 0, "by_type": {}}
        for row in results:
            stats["by_type"][row["type"]] = row["count"]
            stats["total_entities"] += row["count"]
        return stats
```

### 9.6.5 Cypher 查询优化技巧

```cypher
-- 1. 使用节点标签过滤（不要只依赖属性）
-- 差：扫描所有节点
MATCH (e) WHERE e.name = '恒瑞医药' RETURN e
-- 好：利用标签索引
MATCH (e:Company {name: '恒瑞医药'}) RETURN e

-- 2. 使用 PROFILE 分析查询
PROFILE MATCH (e:Company {name: '恒瑞医药'})-[:PRODUCES]->(d:Drug) RETURN d

-- 3. 限制路径长度避免全表扫描
-- 好：限制最大深度
MATCH path = (a)-[:*..3]-(b) WHERE a.name = '恒瑞医药' RETURN path

-- 4. 使用方向约束减少搜索空间
-- 好：单向搜索
MATCH (a)-[:*..3]->(b) WHERE a.name = '恒瑞医药' RETURN b

-- 5. 使用 LIMIT 防止大结果集
MATCH (e:Entity) RETURN e.name LIMIT 100

-- 6. 使用参数化查询（防止 Cypher 注入）
MATCH (e:Entity {name: $name}) RETURN e
```

---

## 9.7 混合图+向量检索

### 9.7.1 混合检索架构

在真实生产环境中，KG 检索和向量检索各有优劣。混合检索架构将两者结合：

```
                   用户查询
                       |
            +----------+----------+
            |                     |
            v                     v
      [实体链接]              [向量编码]
            |                     |
            v                     v
      [KG 检索]              [向量检索]
      (精确关系)             (语义匹配)
            |                     |
            +----------+----------+
                       |
                    [结果融合]
                       |
                       v
                 [重排序 + 生成]
```

### 9.7.2 混合检索实现

```python
# ch09_hybrid_search.py
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class SearchResult:
    content: str
    score: float
    source: str  # "kg" or "vector"
    metadata: Dict = field(default_factory=dict)


class HybridRetriever:
    """
    混合检索器（KG + 向量）

    融合策略：
    - "rrf": 倒数排名融合
    - "weighted": 加权分数融合
    - "cascade": 级联（KG优先，向量兜底）
    """

    def __init__(self, kg_retriever, vector_retriever,
                 fusion_strategy: str = "rrf", kg_weight: float = 0.5):
        self.kg = kg_retriever
        self.vector = vector_retriever
        self.fusion_strategy = fusion_strategy
        self.kg_weight = kg_weight

    def retrieve(self, query: str, top_k: int = 10,
                 kg_top_k: int = 10, vector_top_k: int = 20) -> List[SearchResult]:
        kg_results = self._retrieve_kg(query, kg_top_k)
        vector_results = self._retrieve_vector(query, vector_top_k)

        if self.fusion_strategy == "rrf":
            return self._rrf_fusion(kg_results, vector_results, top_k)
        elif self.fusion_strategy == "weighted":
            return self._weighted_fusion(kg_results, vector_results, top_k)
        elif self.fusion_strategy == "cascade":
            return self._cascade_fusion(kg_results, vector_results, top_k)
        raise ValueError(f"未知融合策略: {self.fusion_strategy}")

    def _retrieve_kg(self, query: str, top_k: int) -> List[SearchResult]:
        kg_result = self.kg.retrieve(query)
        results = []
        for ctx in kg_result.get("contexts", []):
            content = ctx.get("context", "")
            if content:
                results.append(SearchResult(
                    content=content, score=1.0, source="kg",
                    metadata={"entity_id": ctx.get("entity_id"), "entity_name": ctx.get("entity_name")},
                ))
        return results[:top_k]

    def _retrieve_vector(self, query: str, top_k: int) -> List[SearchResult]:
        vector_results = self.vector.retrieve(query, top_k=top_k)
        results = []
        for doc, score in vector_results:
            text = doc.text if hasattr(doc, "text") else str(doc)
            results.append(SearchResult(
                content=text, score=score, source="vector",
                metadata={"doc_id": doc.id if hasattr(doc, "id") else ""},
            ))
        return results

    def _rrf_fusion(self, kg_results, vector_results, top_k, k: int = 60) -> List[SearchResult]:
        rrf_scores = defaultdict(float)
        for rank, r in enumerate(kg_results, 1):
            rrf_scores[r.content] += 1.0 / (k + rank)
        for rank, r in enumerate(vector_results, 1):
            rrf_scores[r.content] += 1.0 / (k + rank)
        scored = sorted(rrf_scores.items(), key=lambda x: -x[1])
        fused, seen = [], set()
        for content, score in scored[:top_k]:
            for r in kg_results + vector_results:
                if r.content == content and content not in seen:
                    r.score = score
                    fused.append(r)
                    seen.add(content)
                    break
        return fused[:top_k]

    def _weighted_fusion(self, kg_results, vector_results, top_k) -> List[SearchResult]:
        combined = defaultdict(lambda: {"score": 0.0, "sources": []})
        if kg_results:
            max_kg = max(r.score for r in kg_results)
            for r in kg_results:
                combined[r.content]["score"] += self.kg_weight * (r.score / max_kg if max_kg > 0 else 0)
                combined[r.content]["sources"].append("kg")
        if vector_results:
            max_vec = max(r.score for r in vector_results)
            for r in vector_results:
                combined[r.content]["score"] += (1 - self.kg_weight) * (r.score / max_vec if max_vec > 0 else 0)
                combined[r.content]["sources"].append("vector")
        sorted_results = sorted(combined.items(), key=lambda x: -x[1]["score"])[:top_k]
        fused = []
        for content, _ in sorted_results:
            for r in kg_results + vector_results:
                if r.content == content:
                    fused.append(r)
                    break
        return fused

    def _cascade_fusion(self, kg_results, vector_results, top_k) -> List[SearchResult]:
        if len(kg_results) >= top_k:
            return kg_results[:top_k]
        fused = list(kg_results)
        seen = {r.content for r in fused}
        for r in vector_results:
            if r.content not in seen and len(fused) < top_k:
                fused.append(r)
                seen.add(r.content)
        return fused
```

### 9.7.3 融合策略对比

| 策略 | 原理 | 优点 | 缺点 | 适用场景 |
|---|---|---|---|---|
| RRF | 基于排名的倒数融合 | 分数无关、鲁棒 | 忽略分数信息 | 各路分数尺度不同的场景 |
| 加权融合 | 归一化后加权平均 | 利用分数信息 | 对异常值敏感 | 各路分数可比时 |
| 级联 | KG 优先，向量兜底 | 简单、可解释 | 未真正融合 | KG 覆盖率高的场景 |

### 9.7.4 缓存策略

```python
# ch09_cache.py
import time, hashlib, json
from typing import Dict, Optional, Any


class KGCache:
    """KG 检索结果缓存（LRU + TTL）"""

    def __init__(self, capacity: int = 1000, ttl_seconds: int = 3600):
        self.capacity = capacity
        self.ttl = ttl_seconds
        self._cache: Dict[str, tuple] = {}

    def _make_key(self, query: str, **params) -> str:
        return hashlib.md5(f"{query}:{json.dumps(params, sort_keys=True)}".encode()).hexdigest()

    def get(self, query: str, **params) -> Optional[Any]:
        key = self._make_key(query, **params)
        if key not in self._cache:
            return None
        value, timestamp = self._cache[key]
        if time.time() - timestamp > self.ttl:
            del self._cache[key]
            return None
        return value

    def set(self, query: str, value: Any, **params):
        key = self._make_key(query, **params)
        if len(self._cache) >= self.capacity:
            oldest = min(self._cache, key=lambda k: self._cache[k][1])
            del self._cache[oldest]
        self._cache[key] = (value, time.time())

    def clear(self):
        self._cache.clear()


class CachedKGRetriever:
    """带缓存的 KG 检索器"""

    def __init__(self, retriever, cache: Optional[KGCache] = None):
        self.retriever = retriever
        self.cache = cache or KGCache()

    def retrieve(self, query: str, **kwargs) -> dict:
        cached = self.cache.get(query, **kwargs)
        if cached:
            return cached
        result = self.retriever.retrieve(query, **kwargs)
        self.cache.set(query, result, **kwargs)
        return result
```

---

## 9.8 综合案例：医药供应链问答系统

```python
# ch09_end_to_end.py
"""
端到端医药供应链问答系统
"""

from typing import List, Dict, Optional


class PharmaQA:
    """医药供应链问答系统"""

    def __init__(self, kg_retriever, vector_retriever=None, llm=None):
        self.kg = kg_retriever
        self.vector = vector_retriever
        self.llm = llm

    def answer(self, query: str, use_hybrid: bool = True) -> str:
        kg_result = self.kg.retrieve(query)
        text_context = ""
        if use_hybrid and self.vector:
            vector_results = self.vector.retrieve(query, top_k=5)
            text_context = "\n".join([
                doc[0] if isinstance(doc, tuple) else str(doc)
                for doc in vector_results
            ])
        prompt = self._build_prompt(query, kg_result["formatted_context"], text_context)
        if self.llm:
            return self.llm.generate(prompt)
        return f"[KG Context]\n{kg_result['formatted_context']}"

    def _build_prompt(self, query: str, kg_context: str, text_context: str) -> str:
        parts = ["你是一个医药供应链领域的智能问答助手。请基于以下信息回答问题。"]
        if kg_context:
            parts.append(f"\n## 知识图谱事实\n{kg_context}")
        if text_context:
            parts.append(f"\n## 文本背景\n{text_context}")
        parts.append(f"\n## 问题\n{query}")
        parts.append("\n## 回答要求")
        parts.append("1. 优先使用知识图谱中的精确事实")
        parts.append("2. 如果信息不足，请明确说明")
        parts.append("3. 展示推理过程（如果涉及多跳推理）")
        return "\n".join(parts)


if __name__ == "__main__":
    kg = build_pharma_kg()
    retriever = KGRetriever(kg)
    qa = PharmaQA(kg_retriever=retriever)

    test_queries = [
        "恒瑞医药生产哪些药品？",
        "恒瑞医药的紫杉醇原料药从哪里来？",
        "北京协和医院采购了恒瑞医药的哪些药品？",
        "华海药业和华东医院之间有什么关系？",
        "国药控股分销哪些公司的产品？",
    ]
    for q in test_queries:
        print(f"\n{'='*60}\nQ: {q}\n{'='*60}\nA: {qa.answer(q, use_hybrid=False)}")
```

---

## 9.9 本章小结

知识图谱检索为 RAG 系统提供了结构化推理能力，与向量检索形成互补。本章的核心要点如下：

1. **实体链接**是 KG 检索的入口。混合使用词典 NER 和模型 NER 可以获得最佳效果，实体消歧是提升准确率的关键。

2. **图遍历**以 EGO 网络为基础操作，最短路径查找支持多跳推理。BFS 适用于无权图，Dijkstra 适用于带权图。剪枝策略是控制搜索空间的关键。

3. **KG 增强检索**将图结构格式化为 LLM 可读的上下文。格式化策略的选择（层级化、三元组列表、叙事化）影响 LLM 的理解效果。

4. **Neo4j Cypher** 是生产环境中的标准图查询语言。shortestPath 函数和可变长度路径匹配是 KG 检索中最常用的 Cypher 特性。

5. **混合图+向量检索**融合了精确关系推理和语义匹配的优势。RRF 融合是推荐策略，但加权融合在分数可比时更优。

6. **Prompt 模板设计**影响 LLM 对图谱信息的利用效果。建议将 KG 信息作为"事实"，文本信息作为"背景"，分工明确。

7. **缓存策略**可以显著减少重复查询的图遍历开销，建议对热点查询启用缓存。

在下一章中，我们将深入探讨 GraphRAG 的深度实践，包括索引流水线、社区检测和 Local/Global/DRIFT 搜索策略。
