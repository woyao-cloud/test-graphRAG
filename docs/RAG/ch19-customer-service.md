# 第 19 章：客服领域 RAG 实践

## 19.1 场景概述

客服是 RAG 最具代表性的落地场景。它天然具有"知识密集、问答高频、容错敏感"的特点。

### 19.1.1 典型需求

| 需求 | 说明 | 传统方案 | RAG 方案 |
|------|------|---------|---------|
| 知识库问答 | 产品手册、政策文档的问答 | 关键词搜索 + 人工匹配 | 语义检索 + LLM 生成 |
| 工单辅助 | 根据历史工单推荐解决方案 | 专家系统 + 规则匹配 | 相似工单检索 + LLM 总结 |
| 智能导航 | 引导用户找到正确的服务入口 | 固定菜单 + 按键导航 | 意图识别 + 动态路由 |
| 话术辅助 | 为客服人员提供实时建议 | 话术手册（人工翻阅） | 实时检索 + 建议生成 |

### 19.1.2 客服 RAG 的特殊要求

```
传统 RAG 要求:     客服 RAG 额外要求:
• 检索准确         • 实时性（< 2s）
• 答案完整         • 安全性（不泄露客户数据）
• 引用可追溯       • 敏感词过滤
                   • 回答风格统一（品牌一致性）
                   • 多轮对话上下文理解
```

---

## 19.2 系统架构

```
用户消息
    │
├─ 意图识别 ────────────────────────
│  • 退货咨询 → 进入退货流程        │
│  • 产品查询 → RAG 知识库检索      │
│  • 投诉反馈 → 人工客服转接        │
└──────────────────────────────────
            │
        需要知识库 ──→ 向量检索 + 关键词检索
            │                    │
            ↓                    ↓
        Reranker ───────────→ 知识库文档
            │
        结合对话历史 ──→ 生成答案
            │
        敏感内容检查 ──→ 返回给用户
```

---

## 19.3 关键实现

### 19.3.1 多轮对话管理

```python
class ConversationContext:
    """客服对话上下文。"""

    def __init__(self, max_history: int = 5):
        self.history: list[dict] = []
        self.max_history = max_history

    def add_message(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        # 只保留最近 N 轮
        if len(self.history) > self.max_history * 2:
            self.history = self.history[-self.max_history * 2:]

    def build_query(self, current_question: str) -> str:
        """构建带上下文的查询。"""
        if not self.history:
            return current_question

        # 需要上下文消歧时，将历史压缩为前缀
        recent = self.history[-4:]  # 最近 2 轮对话
        context = " | ".join(
            f"{m['role']}: {m['content'][:50]}"
            for m in recent
        )
        return f"[上下文] {context} [当前] {current_question}"

    def detect_intent(self, message: str) -> str:
        """简单意图识别。"""
        if any(w in message for w in ["退", "换", " refund", "return"]):
            return "return"
        if any(w in message for w in ["投诉", "举报", "complain"]):
            return "complaint"
        if any(w in message for w in ["价格", "多少钱", "price"]):
            return "price_inquiry"
        return "general_query"
```

### 19.3.2 知识库结构

客服知识库的组织方式直接影响检索效果：

```python
class KnowledgeBase:
    """客服知识库。"""

    def __init__(self):
        self.documents = {
            "product": [],     # 产品信息
            "policy": [],      # 政策规则
            "process": [],     # 操作流程
            "faq": [],         # 常见问题
        }

    def add_document(self, category: str, doc: dict):
        self.documents[category].append(doc)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """按类别优先检索。"""
        results = []
        # 优先检索 FAQ（命中率高）
        for category in ["faq", "product", "process", "policy"]:
            docs = self.documents.get(category, [])
            scored = self._score_docs(query, docs)
            results.extend(scored[:2])
        return results[:top_k]

    def _score_docs(self, query: str, docs: list) -> list:
        """简单的关键词打分。"""
        q_words = set(query.lower().split())
        scored = []
        for doc in docs:
            text = f"{doc.get('title', '')} {doc.get('content', '')}"
            doc_words = set(text.lower().split())
            overlap = len(q_words & doc_words)
            scored.append((overlap, doc))
        scored.sort(key=lambda x: -x[0])
        return [d for _, d in scored]
```

---

## 19.4 客服 RAG 最佳实践

1. **知识库分层**：FAQ 层（高频问题精确匹配）→ 文档层（中频语义检索）→ 全库（低频兜底）
2. **对话历史压缩**：多轮对话中，过长历史会稀释当前问题的检索精度
3. **敏感词过滤**：在输入和输出两端进行过滤，防止个人信息泄露
4. **转人工策略**：当 RAG 置信度低于阈值时，平滑转接人工客服
5. **反馈闭环**：记录用户对答案的反馈（有用/无用），定期更新知识库
6. **A/B 测试**：新策略先在 10% 流量上验证，再全量上线

---

*下一章 [第 20 章：内部知识管理 RAG 实践](ch20-internal-km.md)*
