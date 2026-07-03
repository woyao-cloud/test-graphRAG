# 第19章 客户服务场景

## 19.1 引言

客户服务是RAG技术最具商业价值的应用场景之一。通过将RAG系统与客户服务平台集成，企业可以显著提升客服效率、降低运营成本、改善用户体验。与传统的人工客服或简单的FAQ系统相比，基于RAG的智能客服能够理解更复杂的用户问题，从知识库中检索最相关的信息，并生成准确的回答。

客户服务场景对RAG系统有独特的挑战：

1. **多轮对话管理**：客户服务通常涉及多轮交互，需要维护对话上下文
2. **实时性要求**：客户期望即时响应，对延迟敏感
3. **准确性要求**：错误的信息可能导致客户不满或业务损失
4. **情感处理**：需要识别和处理客户的情绪状态
5. **人机协作**：系统需要知道何时应转接人工客服

本章将详细介绍如何在客户服务场景中构建和优化RAG系统，包括知识库构建、多轮对话管理、人机协作、FAQ匹配和情感分析等关键环节。

### 19.1.1 客户服务RAG系统的核心能力

一个完整的客户服务RAG系统应具备以下核心能力：

| 能力 | 描述 | 优先级 |
|------|------|--------|
| 知识问答 | 从知识库中检索并回答客户问题 | P0 |
| 多轮对话 | 维护对话历史，理解上下文 | P0 |
| 转人工 | 在必要时无缝转接人工客服 | P0 |
| 情感识别 | 识别客户情绪并调整回复 | P1 |
| 工单创建 | 自动创建服务工单 | P1 |
| 满意度跟踪 | 收集并分析客户满意度 | P2 |

### 19.1.2 客户服务流程

典型的客户服务RAG流程如下：

```
客户消息 → 意图识别 → 知识检索 → 答案生成 → 质量检查 → 回复客户
    │          │            │           │           │
    ▼          ▼            ▼           ▼           ▼
  消息解析   意图分类    知识库查询    LLM生成    安全检查
               │                                          │
               ▼                                          ▼
           转人工(必要时)                             情感适配
```

## 19.2 知识库构建

### 19.2.1 知识来源与处理

```python
class CustomerServiceKnowledgeBase:
    """客服知识库构建器"""
    
    def __init__(self):
        self.knowledge_sources = []
        self.processed_docs = []
    
    def add_source(self, source_type: str, source_path: str,
                   category: str, priority: int = 5):
        """添加知识来源"""
        self.knowledge_sources.append({
            'type': source_type,
            'path': source_path,
            'category': category,
            'priority': priority,
            'processed': False
        })
    
    def process_all(self):
        """处理所有知识源"""
        for source in self.knowledge_sources:
            docs = self._process_source(source)
            self.processed_docs.extend(docs)
            source['processed'] = True
        
        return self.processed_docs
    
    def _process_source(self, source: Dict) -> List[Dict]:
        """处理单个知识源"""
        source_type = source['type']
        
        processors = {
            'faq': self._process_faq,
            'product_manual': self._process_manual,
            'policy_doc': self._process_policy,
            'chat_history': self._process_chat_history,
            'ticket': self._process_ticket
        }
        
        processor = processors.get(source_type, self._process_default)
        return processor(source['path'], source['category'])
    
    def _process_faq(self, path: str, category: str) -> List[Dict]:
        """处理FAQ数据"""
        import json
        
        with open(path, 'r', encoding='utf-8') as f:
            faqs = json.load(f)
        
        docs = []
        for faq in faqs:
            # 问题和答案分别作为文档
            docs.append({
                'content': f"问题：{faq['question']}\n答案：{faq['answer']}",
                'metadata': {
                    'type': 'faq',
                    'category': category,
                    'question': faq['question'],
                    'tags': faq.get('tags', []),
                    'priority': faq.get('priority', 5)
                }
            })
        
        return docs
    
    def _process_manual(self, path: str, category: str) -> List[Dict]:
        """处理产品手册"""
        # PDF或Word文档处理
        return self._extract_text_document(path, category, 'manual')
    
    def _process_policy(self, path: str, category: str) -> List[Dict]:
        """处理政策文档"""
        return self._extract_text_document(path, category, 'policy')
    
    def _process_chat_history(self, path: str, category: str) -> List[Dict]:
        """处理历史聊天记录"""
        import json
        
        docs = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    chat = json.loads(line)
                    # 提取客服-客户问答对
                    for i, msg in enumerate(chat['messages']):
                        if msg['role'] == 'agent' and i > 0:
                            customer_msg = chat['messages'][i-1]
                            if customer_msg['role'] == 'customer':
                                docs.append({
                                    'content': f"客户：{customer_msg['content']}\n客服：{msg['content']}",
                                    'metadata': {
                                        'type': 'chat_history',
                                        'category': category,
                                        'intent': chat.get('intent', ''),
                                        'resolution': chat.get('resolution', '')
                                    }
                                })
                except:
                    continue
        
        return docs
    
    def _process_ticket(self, path: str, category: str) -> List[Dict]:
        """处理工单数据"""
        import json
        
        with open(path, 'r', encoding='utf-8') as f:
            tickets = json.load(f)
        
        docs = []
        for ticket in tickets:
            docs.append({
                'content': f"问题描述：{ticket['description']}\n解决方案：{ticket['resolution']}",
                'metadata': {
                    'type': 'ticket',
                    'category': category,
                    'issue_type': ticket.get('issue_type', ''),
                    'severity': ticket.get('severity', 'medium')
                }
            })
        
        return docs
    
    def _process_default(self, path: str, category: str) -> List[Dict]:
        """默认处理"""
        return []
    
    def _extract_text_document(self, path: str, category: str,
                                doc_type: str) -> List[Dict]:
        """从文本文档提取内容"""
        # 简化实现
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 按段落分块
        paragraphs = content.split('\n\n')
        docs = []
        
        for i, para in enumerate(paragraphs):
            if len(para.strip()) < 20:
                continue
            
            docs.append({
                'content': para.strip(),
                'metadata': {
                    'type': doc_type,
                    'category': category,
                    'section_index': i,
                    'source': path
                }
            })
        
        return docs
```

### 19.2.2 知识分类与标签

```python
class KnowledgeClassifier:
    """知识分类器"""
    
    def __init__(self, llm):
        self.llm = llm
        
        self.categories = [
            '产品咨询',
            '订单问题',
            '售后服务',
            '账户管理',
            '支付问题',
            '投诉与建议',
            '技术支持',
            '其他'
        ]
    
    def classify_document(self, content: str) -> Dict:
        """分类文档"""
        prompt = f"""对以下客服知识文档进行分类。

文档内容：{content[:500]}

可选类别：
{chr(10).join([f"{i+1}. {c}" for i, c in enumerate(self.categories)])}

请选择最合适的类别，并提取3-5个标签。
输出JSON：
{{
    "category": "类别名称",
    "tags": ["标签1", "标签2", "标签3"],
    "confidence": 0.0-1.0
}}"""
        
        try:
            import json
            response = self.llm(prompt)
            result = json.loads(response)
            return result
        except:
            return {
                'category': '其他',
                'tags': ['未分类'],
                'confidence': 0.0
            }
    
    def batch_classify(self, documents: List[Dict]) -> List[Dict]:
        """批量分类"""
        for doc in documents:
            if 'metadata' not in doc:
                doc['metadata'] = {}
            
            classification = self.classify_document(doc['content'])
            doc['metadata']['category'] = classification['category']
            doc['metadata']['tags'] = classification['tags']
            doc['metadata']['classification_confidence'] = classification['confidence']
        
        return documents
```

## 19.3 多轮对话管理

### 19.3.1 对话状态管理

```python
class ConversationManager:
    """对话管理器"""
    
    def __init__(self, max_history: int = 10, 
                 session_timeout: int = 1800):
        self.sessions = {}
        self.max_history = max_history
        self.session_timeout = session_timeout
    
    def get_or_create_session(self, session_id: str) -> Dict:
        """获取或创建会话"""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            # 检查会话是否过期
            if self._is_expired(session):
                session = self._create_session(session_id)
        else:
            session = self._create_session(session_id)
        
        return session
    
    def _create_session(self, session_id: str) -> Dict:
        """创建新会话"""
        session = {
            'session_id': session_id,
            'messages': [],
            'context': {
                'intent': None,
                'entities': {},
                'resolved_issues': [],
                'pending_issues': [],
                'customer_info': {}
            },
            'created_at': datetime.now().isoformat(),
            'last_active': datetime.now().isoformat(),
            'turn_count': 0
        }
        self.sessions[session_id] = session
        return session
    
    def add_message(self, session_id: str, 
                     role: str, content: str,
                     metadata: Dict = None):
        """添加消息"""
        session = self.get_or_create_session(session_id)
        
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        session['messages'].append(message)
        session['last_active'] = datetime.now().isoformat()
        session['turn_count'] += 1
        
        # 限制历史消息数量
        if len(session['messages']) > self.max_history * 2:
            session['messages'] = session['messages'][-self.max_history * 2:]
        
        return message
    
    def get_context(self, session_id: str) -> Dict:
        """获取对话上下文"""
        session = self.get_or_create_session(session_id)
        
        # 构建上下文
        context = {
            'history': self._format_history(session['messages']),
            'current_intent': session['context'].get('intent'),
            'entities': session['context'].get('entities', {}),
            'turn_count': session['turn_count'],
            'customer_info': session['context'].get('customer_info', {})
        }
        
        return context
    
    def _format_history(self, messages: List[Dict]) -> List[Dict]:
        """格式化历史消息"""
        formatted = []
        for msg in messages[-self.max_history:]:
            formatted.append({
                'role': msg['role'],
                'content': msg['content']
            })
        return formatted
    
    def update_context(self, session_id: str, 
                        updates: Dict):
        """更新对话上下文"""
        session = self.get_or_create_session(session_id)
        session['context'].update(updates)
    
    def _is_expired(self, session: Dict) -> bool:
        """检查会话是否过期"""
        last_active = datetime.fromisoformat(session['last_active'])
        elapsed = (datetime.now() - last_active).total_seconds()
        return elapsed > self.session_timeout
    
    def cleanup_expired_sessions(self):
        """清理过期会话"""
        expired = []
        for session_id, session in self.sessions.items():
            if self._is_expired(session):
                expired.append(session_id)
        
        for session_id in expired:
            del self.sessions[session_id]
        
        return len(expired)

class ContextWindowManager:
    """上下文窗口管理器"""
    
    def __init__(self, max_tokens: int = 3000):
        self.max_tokens = max_tokens
    
    def build_context(self, history: List[Dict],
                       current_query: str,
                       retrieved_docs: List[Dict]) -> Dict:
        """构建LLM上下文"""
        # 优先级：当前查询 > 检索文档 > 历史消息
        components = {
            'query': current_query,
            'docs': self._format_docs(retrieved_docs),
            'history': self._format_history(history)
        }
        
        # 估算token数
        total_tokens = self._estimate_tokens(components)
        
        if total_tokens <= self.max_tokens:
            return components
        
        # 需要压缩：优先压缩历史
        while total_tokens > self.max_tokens and components['history']:
            components['history'] = components['history'][1:]
            total_tokens = self._estimate_tokens(components)
        
        # 如果仍然超出，压缩文档
        while total_tokens > self.max_tokens and components['docs']:
            components['docs'] = self._truncate_docs(components['docs'])
            total_tokens = self._estimate_tokens(components)
        
        return components
    
    def _estimate_tokens(self, components: Dict) -> int:
        """估算token数"""
        total = 0
        for key, value in components.items():
            if isinstance(value, str):
                total += len(value) // 2
            elif isinstance(value, list):
                for item in value:
                    total += len(str(item)) // 2
        return total
    
    def _format_docs(self, docs: List[Dict]) -> List[str]:
        """格式化文档"""
        return [f"[{i+1}] {d['content'][:300]}" for i, d in enumerate(docs)]
    
    def _format_history(self, history: List[Dict]) -> List[Dict]:
        """格式化历史"""
        return history[-6:]  # 最多保留6轮
    
    def _truncate_docs(self, docs: List[str]) -> List[str]:
        """截断文档"""
        if len(docs) <= 1:
            return []
        return docs[:-1]
```

### 19.3.2 意图识别与实体提取

```python
class IntentRecognizer:
    """意图识别器"""
    
    def __init__(self, llm):
        self.llm = llm
        
        self.intents = {
            'product_inquiry': '产品咨询',
            'order_status': '订单查询',
            'return_request': '退货申请',
            'complaint': '投诉',
            'account_issue': '账户问题',
            'payment_issue': '支付问题',
            'technical_support': '技术支持',
            'greeting': '问候',
            'farewell': '告别',
            'others': '其他'
        }
    
    def recognize(self, message: str, 
                  context: Dict = None) -> Dict:
        """识别意图"""
        history = context.get('history', []) if context else []
        
        prompt = f"""识别客户消息的意图。

对话历史：
{chr(10).join([f"{m['role']}: {m['content'][:100]}" for m in history[-3:]])}

当前消息：{message}

可选的意图：
{chr(10).join([f"{k}: {v}" for k, v in self.intents.items()])}

输出JSON：
{{
    "intent": "意图键值",
    "confidence": 0.0-1.0,
    "entities": {{
        "entity_type": "值"
    }},
    "needs_human": true/false
}}"""
        
        try:
            import json
            response = self.llm(prompt)
            result = json.loads(response)
            return result
        except:
            return {
                'intent': 'others',
                'confidence': 0.0,
                'entities': {},
                'needs_human': False
            }
    
    def needs_human_escalation(self, intent: str, 
                                confidence: float,
                                customer_sentiment: str) -> bool:
        """判断是否需要转人工"""
        # 高优先级意图
        high_priority = ['complaint', 'return_request']
        
        if intent in high_priority:
            return True
        
        if confidence < 0.5:
            return True
        
        if customer_sentiment == 'very_negative':
            return True
        
        return False
```

## 19.4 人机协作

### 19.4.1 转人工策略

```python
class HumanHandoff:
    """人机转接管理器"""
    
    def __init__(self):
        self.agent_queue = []
        self.agent_status = {}
    
    def should_escalate(self, 
                         intent_result: Dict,
                         confidence: float,
                         customer_sentiment: str,
                         turn_count: int) -> Dict:
        """判断是否需要转人工"""
        reasons = []
        
        # 1. 意图级别
        if intent_result.get('intent') in ['complaint', 'return_request']:
            reasons.append('high_priority_intent')
        
        # 2. 置信度低
        if confidence < 0.4:
            reasons.append('low_confidence')
        
        # 3. 情绪负面
        if customer_sentiment in ['negative', 'very_negative']:
            reasons.append('negative_sentiment')
        
        # 4. 轮次过多
        if turn_count > 5:
            reasons.append('too_many_turns')
        
        # 5. 用户明确要求
        # 在消息处理中检测
        
        return {
            'should_escalate': len(reasons) > 0,
            'reasons': reasons,
            'priority': self._calculate_priority(reasons, customer_sentiment)
        }
    
    def _calculate_priority(self, reasons: List[str],
                             sentiment: str) -> str:
        """计算优先级"""
        if 'complaint' in reasons or sentiment == 'very_negative':
            return 'urgent'
        elif len(reasons) >= 2:
            return 'high'
        elif reasons:
            return 'normal'
        return 'low'
    
    def create_handoff(self, session_id: str,
                        context: Dict,
                        reason: Dict) -> Dict:
        """创建转接工单"""
        handoff = {
            'session_id': session_id,
            'created_at': datetime.now().isoformat(),
            'priority': reason.get('priority', 'normal'),
            'reasons': reason.get('reasons', []),
            'context': {
                'history': context.get('history', []),
                'customer_info': context.get('customer_info', {}),
                'attempted_solutions': context.get('resolved_issues', [])
            },
            'status': 'waiting'
        }
        
        self.agent_queue.append(handoff)
        return handoff
    
    def get_queue_status(self) -> Dict:
        """获取队列状态"""
        waiting = [h for h in self.agent_queue if h['status'] == 'waiting']
        in_progress = [h for h in self.agent_queue if h['status'] == 'in_progress']
        
        return {
            'waiting_count': len(waiting),
            'in_progress_count': len(in_progress),
            'avg_wait_time': self._calculate_avg_wait_time(waiting),
            'urgent_count': len([h for h in waiting if h['priority'] == 'urgent'])
        }
    
    def _calculate_avg_wait_time(self, waiting: List[Dict]) -> float:
        """计算平均等待时间"""
        if not waiting:
            return 0.0
        
        now = datetime.now()
        wait_times = []
        for h in waiting:
            created = datetime.fromisoformat(h['created_at'])
            wait_seconds = (now - created).total_seconds()
            wait_times.append(wait_seconds)
        
        return sum(wait_times) / len(wait_times)
```

### 19.4.2 反馈循环

```python
class FeedbackLoop:
    """反馈循环"""
    
    def __init__(self):
        self.feedback_records = []
    
    def collect_feedback(self, session_id: str,
                          rating: int,
                          comment: str = "",
                          correction: str = "") -> Dict:
        """收集反馈"""
        feedback = {
            'session_id': session_id,
            'rating': rating,
            'comment': comment,
            'correction': correction,
            'timestamp': datetime.now().isoformat()
        }
        self.feedback_records.append(feedback)
        return feedback
    
    def analyze_feedback(self, period: str = '7d') -> Dict:
        """分析反馈"""
        from datetime import timedelta
        
        now = datetime.now()
        if period.endswith('d'):
            delta = timedelta(days=int(period[:-1]))
        else:
            delta = timedelta(days=7)
        
        cutoff = now - delta
        recent = [
            f for f in self.feedback_records
            if datetime.fromisoformat(f['timestamp']) > cutoff
        ]
        
        if not recent:
            return {'total': 0, 'avg_rating': 0}
        
        ratings = [f['rating'] for f in recent]
        corrections = [f['correction'] for f in recent if f.get('correction')]
        
        return {
            'total': len(recent),
            'avg_rating': sum(ratings) / len(ratings),
            'rating_distribution': {
                level: ratings.count(level)
                for level in set(ratings)
            },
            'correction_rate': len(corrections) / len(recent) if recent else 0,
            'common_issues': self._extract_common_issues(corrections)
        }
    
    def _extract_common_issues(self, corrections: List[str]) -> List[Dict]:
        """提取常见问题"""
        from collections import Counter
        
        # 简化实现
        return [{'issue': '需要人工标注', 'count': len(corrections)}]
    
    def update_knowledge_base(self, corrections: List[str]):
        """基于反馈更新知识库"""
        for correction in corrections:
            if correction:
                # 将修正内容添加到知识库
                pass
```

## 19.5 FAQ匹配

### 19.5.1 FAQ检索系统

```python
class FAQMatcher:
    """FAQ匹配器"""
    
    def __init__(self, vector_store, llm):
        self.vector_store = vector_store
        self.llm = llm
        self.faq_cache = {}
    
    def find_best_match(self, question: str, 
                         threshold: float = 0.7) -> Dict:
        """查找最佳FAQ匹配"""
        # 向量检索
        results = self.vector_store.similarity_search(
            question, k=5, filter={'type': 'faq'}
        )
        
        if not results:
            return {'found': False}
        
        best = results[0]
        similarity = best.metadata.get('score', 0)
        
        if similarity >= threshold:
            return {
                'found': True,
                'question': best.metadata.get('question', ''),
                'answer': best.page_content.split('答案：')[-1].strip(),
                'similarity': similarity,
                'source': 'faq'
            }
        
        # 低于阈值，尝试语义匹配
        return self._semantic_match(question, results)
    
    def _semantic_match(self, question: str,
                         candidates: List) -> Dict:
        """语义匹配"""
        prompt = f"""判断以下问题是否与FAQ中的问题语义相同。

用户问题：{question}

候选FAQ：
{chr(10).join([f"{i+1}. {c.metadata.get('question', '')}" for i, c in enumerate(candidates[:3])])}

如果存在语义相同的问题，输出对应的序号（1-based），否则输出0："""
        
        try:
            response = self.llm(prompt).strip()
            idx = int(response) - 1
            if 0 <= idx < len(candidates):
                candidate = candidates[idx]
                return {
                    'found': True,
                    'question': candidate.metadata.get('question', ''),
                    'answer': candidate.page_content.split('答案：')[-1].strip(),
                    'similarity': 0.8,
                    'source': 'faq_semantic'
                }
        except:
            pass
        
        return {'found': False}
```

### 19.5.2 FAQ更新机制

```python
class FAQUpdater:
    """FAQ更新器"""
    
    def __init__(self, faq_store):
        self.faq_store = faq_store
    
    def add_faq(self, question: str, answer: str,
                 category: str = None, tags: List[str] = None):
        """添加FAQ"""
        faq_entry = {
            'question': question,
            'answer': answer,
            'category': category or 'general',
            'tags': tags or [],
            'created_at': datetime.now().isoformat(),
            'usage_count': 0,
            'is_active': True
        }
        
        self.faq_store.add(faq_entry)
        return faq_entry
    
    def update_faq(self, faq_id: str, updates: Dict):
        """更新FAQ"""
        updates['updated_at'] = datetime.now().isoformat()
        self.faq_store.update(faq_id, updates)
    
    def increment_usage(self, faq_id: str):
        """增加使用计数"""
        faq = self.faq_store.get(faq_id)
        if faq:
            faq['usage_count'] = faq.get('usage_count', 0) + 1
            self.faq_store.update(faq_id, faq)
    
    def suggest_new_faq(self, unanswered_questions: List[str]) -> List[Dict]:
        """建议新增FAQ"""
        from collections import Counter
        
        question_counts = Counter(unanswered_questions)
        frequent_questions = [
            {'question': q, 'count': c}
            for q, c in question_counts.most_common(10)
            if c >= 3  # 出现3次以上建议新增
        ]
        
        return frequent_questions
```

## 19.6 情感分析集成

### 19.6.1 实时情感检测

```python
class SentimentAnalyzer:
    """情感分析器"""
    
    def __init__(self, llm):
        self.llm = llm
        
        self.sentiment_levels = [
            'very_positive', 'positive', 'neutral', 
            'negative', 'very_negative'
        ]
    
    def analyze(self, message: str, 
                context: Dict = None) -> Dict:
        """分析情感"""
        history = context.get('history', []) if context else []
        
        prompt = f"""分析客户消息的情感。

对话历史：
{chr(10).join([f"{m['role']}: {m['content'][:100]}" for m in history[-2:]])}

当前消息：{message}

输出JSON：
{{
    "sentiment": "情感级别",
    "score": 0.0-1.0,
    "emotions": ["检测到的情绪"],
    "urgency": "low/medium/high",
    "suggestion": "建议的应对方式"
}}

情感级别：{', '.join(self.sentiment_levels)}"""
        
        try:
            import json
            response = self.llm(prompt)
            result = json.loads(response)
            return result
        except:
            return {
                'sentiment': 'neutral',
                'score': 0.5,
                'emotions': [],
                'urgency': 'low',
                'suggestion': '正常回应'
            }
    
    def adapt_response(self, answer: str, 
                        sentiment: Dict) -> str:
        """根据情感调整回复"""
        sentiment_level = sentiment.get('sentiment', 'neutral')
        
        if sentiment_level in ['negative', 'very_negative']:
            # 增加安抚语气
            prompt = f"""将以下客服回复改写得更加温和、富有同理心。

原始回复：{answer}

要求：
1. 首先表达理解客户的感受
2. 保持专业和诚恳
3. 给出明确的解决方案
4. 语气要温和但不卑微

改写后的回复："""
            
            try:
                return self.llm(prompt)
            except:
                return answer
        
        return answer
```

### 19.6.2 情感趋势分析

```python
class SentimentTrendAnalyzer:
    """情感趋势分析器"""
    
    def __init__(self):
        self.sentiment_records = []
    
    def record_sentiment(self, session_id: str, 
                          sentiment: Dict,
                          intent: str):
        """记录情感数据"""
        record = {
            'session_id': session_id,
            'sentiment': sentiment.get('sentiment'),
            'score': sentiment.get('score'),
            'intent': intent,
            'timestamp': datetime.now().isoformat()
        }
        self.sentiment_records.append(record)
        return record
    
    def get_daily_trend(self, days: int = 7) -> Dict:
        """获取每日趋势"""
        from datetime import timedelta
        
        now = datetime.now()
        cutoff = now - timedelta(days=days)
        
        recent = [
            r for r in self.sentiment_records
            if datetime.fromisoformat(r['timestamp']) > cutoff
        ]
        
        # 按天聚合
        daily = defaultdict(list)
        for record in recent:
            day = record['timestamp'][:10]
            daily[day].append(record['score'])
        
        trend = {
            day: {
                'avg_score': sum(scores) / len(scores),
                'count': len(scores)
            }
            for day, scores in sorted(daily.items())
        }
        
        return {
            'trend': trend,
            'overall_avg': (
                sum(r['score'] for r in recent) / len(recent)
                if recent else 0
            ),
            'sample_count': len(recent)
        }
    
    def get_intent_sentiment_matrix(self) -> Dict:
        """获取意图-情感矩阵"""
        matrix = defaultdict(lambda: defaultdict(list))
        
        for record in self.sentiment_records:
            intent = record.get('intent', 'unknown')
            sentiment = record.get('sentiment', 'neutral')
            matrix[intent][sentiment].append(1)
        
        return {
            intent: {
                sentiment: len(records)
                for sentiment, records in sentiments.items()
            }
            for intent, sentiments in matrix.items()
        }
```

## 19.7 性能优化

### 19.7.1 响应时间优化

```python
class CustomerServiceOptimizer:
    """客服性能优化器"""
    
    def __init__(self):
        self.cache = {}
        self.fast_lane_enabled = True
    
    def fast_lane_query(self, question: str) -> Optional[Dict]:
        """快速通道：直接从缓存或FAQ匹配"""
        # 检查精确匹配缓存
        if question in self.cache:
            cache_entry = self.cache[question]
            if not self._is_expired(cache_entry):
                return cache_entry['response']
        
        return None
    
    def warmup_cache(self, frequent_questions: List[str], 
                     answer_generator: Callable):
        """预热缓存"""
        for question in frequent_questions:
            if question not in self.cache:
                answer = answer_generator(question)
                self.cache[question] = {
                    'response': answer,
                    'timestamp': datetime.now()
                }
    
    def _is_expired(self, cache_entry: Dict, 
                     ttl: int = 300) -> bool:
        """检查缓存是否过期"""
        elapsed = (datetime.now() - cache_entry['timestamp']).total_seconds()
        return elapsed > ttl
    
    def get_cache_stats(self) -> Dict:
        """获取缓存统计"""
        return {
            'cache_size': len(self.cache),
            'hit_rate': self._calculate_hit_rate()
        }
    
    def _calculate_hit_rate(self) -> float:
        """计算命中率"""
        return 0.8  # 简化实现
```

## 19.8 本章小结

本章详细介绍了RAG系统在客户服务场景中的应用方法和最佳实践。

**知识库构建**是客服RAG系统的基础。本章介绍了从FAQ、产品手册、政策文档、历史聊天记录和工单数据等多种来源构建知识库的方法。知识分类与标签系统使检索更加精准。

**多轮对话管理**通过对话状态管理器和上下文窗口管理器，实现了高效的对话历史维护和上下文构建。意图识别和实体提取模块能够准确理解用户需求，为后续的回答生成提供基础。

**人机协作**是客服RAG系统的关键能力。转人工策略基于意图优先级、置信度、客户情绪和对话轮次等多个因素综合判断。反馈循环机制将人工客服的修正反馈到系统中，形成持续改进的闭环。

**FAQ匹配**系统结合向量检索和语义匹配，能够快速找到与用户问题最匹配的FAQ条目。FAQ更新机制通过分析未回答的问题，自动建议新增FAQ条目。

**情感分析集成**通过实时情感检测和回复适配，使系统能够根据客户情绪状态调整回复语气。情感趋势分析帮助管理者了解整体服务质量。

在实际部署中，建议采用渐进式策略：先部署FAQ自动回复和简单的问答功能，逐步引入多轮对话和情感分析，最后实现完整的人机协作和反馈闭环。同时，建立完善的监控体系，持续跟踪关键指标（首次响应时间、解决率、用户满意度等），驱动系统的持续优化。
