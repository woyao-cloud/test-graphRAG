# 第20章 企业内部知识管理

## 20.1 引言

企业内部知识管理是企业数字化转型的核心组成部分。随着企业积累的文档、报告、流程规范和技术资料越来越多，如何高效地组织、检索和利用这些知识成为一大挑战。RAG技术为企业知识管理提供了全新的解决方案——它不仅能检索到相关文档，还能基于检索结果生成精确的答案，大大提升了知识获取的效率。

企业知识管理场景对RAG系统有独特的要求：

1. **权限管理**：不同部门和角色的员工应有不同的文档访问权限
2. **知识分类**：企业知识种类繁多，需要科学的分类体系
3. **时效性**：部分知识有时效性要求，需要定期更新
4. **精准性**：企业内部知识需要高度准确，错误信息可能造成业务损失
5. **搜索增强**：需要支持分面搜索、个性化排序等高级检索功能

### 20.1.1 企业知识管理系统的核心能力

一个企业级RAG知识管理系统应具备以下能力：

| 能力 | 描述 | 优先级 |
|------|------|--------|
| 文档管理 | 上传、分类、存储、版本管理 | P0 |
| 智能搜索 | 语义搜索、关键词搜索、混合搜索 | P0 |
| 知识问答 | 基于知识库的自然语言问答 | P0 |
| 权限控制 | 角色和文档级别的访问控制 | P0 |
| 知识更新 | 文档更新、过期检测、自动刷新 | P1 |
| 个性化 | 基于用户角色和历史的个性化排序 | P1 |

## 20.2 企业知识库构建

### 20.2.1 文档分类与标签

```python
class DocumentClassifier:
    """企业文档分类器"""
    
    def __init__(self, llm):
        self.llm = llm
        
        # 企业文档分类体系
        self.classification_schema = {
            '技术文档': {
                'description': '技术方案、架构设计、API文档',
                'subcategories': ['系统架构', '接口规范', '开发指南', '部署文档']
            },
            '产品文档': {
                'description': '产品需求、功能规格、用户手册',
                'subcategories': ['PRD', '功能说明', '用户手册', '版本发布']
            },
            '管理文档': {
                'description': '流程规范、制度文件、报告',
                'subcategories': ['流程规范', '制度文件', '周报月报', '会议纪要']
            },
            '知识资产': {
                'description': '最佳实践、经验总结、培训材料',
                'subcategories': ['最佳实践', '案例库', '培训材料', '研究笔记']
            },
            '商务文档': {
                'description': '合同、报价、客户资料',
                'subcategories': ['合同', '报价', '客户资料', '市场分析']
            }
        }
    
    def classify(self, document: Dict) -> Dict:
        """分类文档"""
        content = document.get('content', '')
        title = document.get('metadata', {}).get('title', '')
        
        prompt = f"""对以下企业文档进行分类。

文档标题：{title}
文档内容前200字：{content[:200]}

分类体系：
{json.dumps(self.classification_schema, ensure_ascii=False, indent=2)}

请输出JSON：
{{
    "category": "主类别",
    "subcategory": "子类别",
    "tags": ["标签1", "标签2", "标签3"],
    "confidence": 0.0-1.0,
    "sensitivity": "public/internal/confidential/secret"
}}"""
        
        try:
            import json
            response = self.llm(prompt)
            result = json.loads(response)
            return result
        except:
            return {
                'category': '知识资产',
                'subcategory': '其他',
                'tags': ['未分类'],
                'confidence': 0.0,
                'sensitivity': 'internal'
            }
    
    def batch_classify(self, documents: List[Dict]) -> List[Dict]:
        """批量分类"""
        for doc in documents:
            classification = self.classify(doc)
            if 'metadata' not in doc:
                doc['metadata'] = {}
            doc['metadata']['category'] = classification['category']
            doc['metadata']['subcategory'] = classification['subcategory']
            doc['metadata']['tags'] = classification['tags']
            doc['metadata']['sensitivity'] = classification['sensitivity']
        
        return documents
```

### 20.2.2 文档生命周期管理

```python
class DocumentLifecycleManager:
    """文档生命周期管理器"""
    
    def __init__(self):
        self.documents = {}
        self.version_history = defaultdict(list)
    
    def add_document(self, doc_id: str, content: str,
                     metadata: Dict) -> Dict:
        """添加文档"""
        document = {
            'id': doc_id,
            'content': content,
            'metadata': {
                **metadata,
                'version': 1,
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
                'status': 'active',
                'review_date': (
                    datetime.now() + timedelta(days=180)
                ).isoformat()
            }
        }
        
        self.documents[doc_id] = document
        self.version_history[doc_id].append({
            'version': 1,
            'content': content,
            'timestamp': datetime.now().isoformat()
        })
        
        return document
    
    def update_document(self, doc_id: str, new_content: str,
                         updated_metadata: Dict = None) -> Dict:
        """更新文档"""
        if doc_id not in self.documents:
            raise ValueError(f"文档 {doc_id} 不存在")
        
        old_doc = self.documents[doc_id]
        new_version = old_doc['metadata']['version'] + 1
        
        # 保存历史版本
        self.version_history[doc_id].append({
            'version': new_version,
            'content': new_content,
            'timestamp': datetime.now().isoformat()
        })
        
        # 更新文档
        self.documents[doc_id] = {
            'id': doc_id,
            'content': new_content,
            'metadata': {
                **old_doc['metadata'],
                **(updated_metadata or {}),
                'version': new_version,
                'updated_at': datetime.now().isoformat()
            }
        }
        
        return self.documents[doc_id]
    
    def archive_document(self, doc_id: str):
        """归档文档"""
        if doc_id in self.documents:
            self.documents[doc_id]['metadata']['status'] = 'archived'
    
    def get_document_history(self, doc_id: str) -> List[Dict]:
        """获取文档历史"""
        return self.version_history.get(doc_id, [])
    
    def check_expired_documents(self) -> List[Dict]:
        """检查过期文档"""
        expired = []
        now = datetime.now()
        
        for doc_id, doc in self.documents.items():
            if doc['metadata'].get('status') != 'active':
                continue
            
            review_date = datetime.fromisoformat(
                doc['metadata'].get('review_date', now.isoformat())
            )
            
            if review_date < now:
                expired.append({
                    'id': doc_id,
                    'title': doc['metadata'].get('title', ''),
                    'review_date': review_date.isoformat(),
                    'days_overdue': (now - review_date).days
                })
        
        return sorted(expired, key=lambda x: x['days_overdue'], reverse=True)
    
    def get_document_stats(self) -> Dict:
        """获取文档统计"""
        active = sum(
            1 for d in self.documents.values()
            if d['metadata'].get('status') == 'active'
        )
        archived = sum(
            1 for d in self.documents.values()
            if d['metadata'].get('status') == 'archived'
        )
        
        return {
            'total': len(self.documents),
            'active': active,
            'archived': archived,
            'total_versions': sum(len(v) for v in self.version_history.values())
        }
```

## 20.3 权限管理

### 20.3.1 RBAC实现

```python
class RBACManager:
    """基于角色的访问控制"""
    
    def __init__(self):
        self.roles = {}
        self.users = {}
        self.role_permissions = {}
    
    def define_role(self, role_name: str, 
                     permissions: List[str],
                     parent_role: str = None):
        """定义角色"""
        self.roles[role_name] = {
            'name': role_name,
            'permissions': permissions,
            'parent': parent_role,
            'created_at': datetime.now().isoformat()
        }
        
        # 如果有父角色，继承权限
        if parent_role and parent_role in self.role_permissions:
            inherited = self.role_permissions[parent_role]
            all_permissions = list(set(permissions + inherited))
        else:
            all_permissions = permissions
        
        self.role_permissions[role_name] = all_permissions
    
    def assign_role(self, user_id: str, role_name: str):
        """分配角色"""
        if role_name not in self.roles:
            raise ValueError(f"角色 {role_name} 不存在")
        
        self.users[user_id] = {
            'user_id': user_id,
            'role': role_name,
            'permissions': self.role_permissions[role_name]
        }
    
    def check_permission(self, user_id: str, 
                          permission: str) -> bool:
        """检查权限"""
        user = self.users.get(user_id)
        if not user:
            return False
        
        return permission in user.get('permissions', [])
    
    def get_user_permissions(self, user_id: str) -> List[str]:
        """获取用户权限"""
        user = self.users.get(user_id)
        return user.get('permissions', []) if user else []
    
    # 预定义企业角色
    def setup_default_roles(self):
        """设置默认角色"""
        # 管理员
        self.define_role('admin', [
            'doc:create', 'doc:read', 'doc:update', 'doc:delete',
            'doc:manage_permissions', 'user:manage',
            'search:all', 'search:personalized'
        ])
        
        # 部门经理
        self.define_role('dept_manager', [
            'doc:create', 'doc:read', 'doc:update',
            'search:all', 'search:personalized'
        ])
        
        # 普通员工
        self.define_role('employee', [
            'doc:create', 'doc:read',
            'search:personalized'
        ])
        
        # 实习生
        self.define_role('intern', [
            'doc:read',
            'search:basic'
        ])

class DocumentACL:
    """文档访问控制列表"""
    
    def __init__(self, rbac: RBACManager):
        self.rbac = rbac
        self.acls = {}
    
    def set_acl(self, doc_id: str, acl: Dict):
        """设置文档ACL"""
        self.acls[doc_id] = {
            'owner': acl.get('owner'),
            'allowed_roles': acl.get('allowed_roles', []),
            'allowed_users': acl.get('allowed_users', []),
            'denied_users': acl.get('denied_users', []),
            'department': acl.get('department'),
            'sensitivity': acl.get('sensitivity', 'internal')
        }
    
    def check_access(self, user_id: str, doc_id: str,
                     action: str = 'read') -> bool:
        """检查访问权限"""
        acl = self.acls.get(doc_id)
        if not acl:
            return True  # 无ACL默认可访问
        
        # 管理员有所有权限
        if self.rbac.check_permission(user_id, 'doc:manage_permissions'):
            return True
        
        # 检查拒绝列表
        if user_id in acl.get('denied_users', []):
            return False
        
        # 检查是否拥有者
        if user_id == acl.get('owner'):
            return True
        
        # 检查允许用户列表
        if user_id in acl.get('allowed_users', []):
            return True
        
        # 检查角色
        user_role = self.rbac.users.get(user_id, {}).get('role')
        if user_role in acl.get('allowed_roles', []):
            return True
        
        # 检查基本权限
        if action == 'read':
            return self.rbac.check_permission(user_id, 'doc:read')
        elif action == 'write':
            return self.rbac.check_permission(user_id, 'doc:update')
        
        return False
    
    def filter_accessible_docs(self, user_id: str,
                                documents: List[Dict]) -> List[Dict]:
        """过滤用户有权限的文档"""
        accessible = []
        for doc in documents:
            doc_id = doc.get('id', doc.get('doc_id', ''))
            if self.check_access(user_id, doc_id):
                accessible.append(doc)
        return accessible
```

## 20.4 搜索增强

### 20.4.1 分面搜索

```python
class FacetedSearch:
    """分面搜索"""
    
    def __init__(self, search_engine):
        self.search_engine = search_engine
        
        self.facets = {
            'category': {
                'label': '文档类别',
                'type': 'terms',
                'field': 'metadata.category'
            },
            'department': {
                'label': '所属部门',
                'type': 'terms',
                'field': 'metadata.department'
            },
            'author': {
                'label': '作者',
                'type': 'terms',
                'field': 'metadata.author'
            },
            'date': {
                'label': '创建日期',
                'type': 'date_range',
                'field': 'metadata.created_at',
                'ranges': [
                    ('最近7天', 7),
                    ('最近30天', 30),
                    ('最近90天', 90),
                    ('超过90天', -1)
                ]
            },
            'sensitivity': {
                'label': '敏感级别',
                'type': 'terms',
                'field': 'metadata.sensitivity'
            },
            'status': {
                'label': '文档状态',
                'type': 'terms',
                'field': 'metadata.status'
            }
        }
    
    def search(self, query: str, 
               selected_facets: Dict = None,
               page: int = 1,
               page_size: int = 20) -> Dict:
        """分面搜索"""
        selected_facets = selected_facets or {}
        
        # 执行搜索
        results = self.search_engine.search(query)
        
        # 应用分面过滤
        filtered_results = self._apply_facets(results, selected_facets)
        
        # 计算分面统计
        facet_counts = self._compute_facet_counts(
            results, selected_facets
        )
        
        # 分页
        total = len(filtered_results)
        start = (page - 1) * page_size
        end = start + page_size
        page_results = filtered_results[start:end]
        
        return {
            'results': page_results,
            'total': total,
            'page': page,
            'page_size': page_size,
            'total_pages': (total + page_size - 1) // page_size,
            'facets': facet_counts,
            'selected_facets': selected_facets
        }
    
    def _apply_facets(self, results: List[Dict],
                       selected_facets: Dict) -> List[Dict]:
        """应用分面过滤"""
        if not selected_facets:
            return results
        
        filtered = results
        for facet_name, facet_values in selected_facets.items():
            if not facet_values:
                continue
            
            facet_config = self.facets.get(facet_name)
            if not facet_config:
                continue
            
            field = facet_config['field']
            
            if facet_config['type'] == 'terms':
                filtered = [
                    r for r in filtered
                    if self._get_nested_value(r, field) in facet_values
                ]
            elif facet_config['type'] == 'date_range':
                filtered = self._filter_by_date_range(
                    filtered, field, facet_values
                )
        
        return filtered
    
    def _compute_facet_counts(self, results: List[Dict],
                                selected_facets: Dict) -> Dict:
        """计算分面统计"""
        from collections import Counter
        
        facet_counts = {}
        
        for facet_name, facet_config in self.facets.items():
            field = facet_config['field']
            
            if facet_config['type'] == 'terms':
                values = [
                    self._get_nested_value(r, field)
                    for r in results
                ]
                counter = Counter(values)
                facet_counts[facet_name] = dict(counter.most_common(10))
            
            elif facet_config['type'] == 'date_range':
                counts = {}
                for label, days in facet_config['ranges']:
                    count = self._count_in_range(results, field, days)
                    counts[label] = count
                facet_counts[facet_name] = counts
        
        return facet_counts
    
    def _get_nested_value(self, doc: Dict, field: str) -> Any:
        """获取嵌套字段值"""
        keys = field.split('.')
        value = doc
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key, '')
            else:
                return ''
        return value
    
    def _filter_by_date_range(self, results: List[Dict],
                               field: str, days: int) -> List[Dict]:
        """按日期范围过滤"""
        from datetime import timedelta
        
        cutoff = datetime.now() - timedelta(days=days) if days > 0 else None
        
        filtered = []
        for r in results:
            date_str = self._get_nested_value(r, field)
            if date_str:
                try:
                    doc_date = datetime.fromisoformat(date_str)
                    if cutoff is None or doc_date >= cutoff:
                        filtered.append(r)
                except:
                    filtered.append(r)
            else:
                filtered.append(r)
        
        return filtered
    
    def _count_in_range(self, results: List[Dict],
                         field: str, days: int) -> int:
        """统计日期范围内文档数"""
        from datetime import timedelta
        
        cutoff = datetime.now() - timedelta(days=days) if days > 0 else None
        count = 0
        
        for r in results:
            date_str = self._get_nested_value(r, field)
            if date_str:
                try:
                    doc_date = datetime.fromisoformat(date_str)
                    if cutoff is None or doc_date >= cutoff:
                        count += 1
                except:
                    count += 1
            else:
                count += 1
        
        return count
```

### 20.4.2 个性化排序

```python
class PersonalizedRanking:
    """个性化排序"""
    
    def __init__(self):
        self.user_profiles = {}
        self.user_actions = defaultdict(list)
    
    def build_user_profile(self, user_id: str,
                            actions: List[Dict]) -> Dict:
        """构建用户画像"""
        from collections import Counter
        
        # 分析用户行为
        category_interests = Counter()
        tag_interests = Counter()
        author_preferences = Counter()
        
        for action in actions:
            action_type = action.get('type', '')
            doc = action.get('document', {})
            
            if action_type in ['view', 'search_click', 'save']:
                weight = 2 if action_type == 'save' else 1
                
                category = doc.get('metadata', {}).get('category', '')
                if category:
                    category_interests[category] += weight
                
                tags = doc.get('metadata', {}).get('tags', [])
                for tag in tags:
                    tag_interests[tag] += weight
                
                author = doc.get('metadata', {}).get('author', '')
                if author:
                    author_preferences[author] += weight
        
        profile = {
            'user_id': user_id,
            'top_categories': [c for c, _ in category_interests.most_common(5)],
            'top_tags': [t for t, _ in tag_interests.most_common(10)],
            'top_authors': [a for a, _ in author_preferences.most_common(5)],
            'total_actions': len(actions),
            'last_updated': datetime.now().isoformat()
        }
        
        self.user_profiles[user_id] = profile
        return profile
    
    def rank_results(self, user_id: str,
                      results: List[Dict],
                      query: str = None) -> List[Dict]:
        """个性化排序"""
        profile = self.user_profiles.get(user_id)
        if not profile:
            return results  # 无画像，返回原始排序
        
        # 为每个结果计算个性化分数
        scored_results = []
        for result in results:
            base_score = result.get('score', 0.5)
            personal_score = self._compute_personal_score(
                result, profile
            )
            
            # 融合分数
            final_score = 0.7 * base_score + 0.3 * personal_score
            
            scored_results.append({
                **result,
                'original_score': base_score,
                'personal_score': personal_score,
                'score': final_score
            })
        
        # 按最终分数排序
        scored_results.sort(key=lambda x: x['score'], reverse=True)
        
        return scored_results
    
    def _compute_personal_score(self, document: Dict,
                                 profile: Dict) -> float:
        """计算个性化分数"""
        score = 0.0
        metadata = document.get('metadata', {})
        
        # 类别匹配
        category = metadata.get('category', '')
        if category in profile.get('top_categories', []):
            score += 0.4
        
        # 标签匹配
        doc_tags = set(metadata.get('tags', []))
        profile_tags = set(profile.get('top_tags', []))
        tag_overlap = len(doc_tags & profile_tags)
        if tag_overlap > 0:
            score += 0.3 * min(tag_overlap / 3, 1.0)
        
        # 作者匹配
        author = metadata.get('author', '')
        if author in profile.get('top_authors', []):
            score += 0.2
        
        # 近期活跃度
        last_updated = metadata.get('updated_at', '')
        if last_updated:
            try:
                updated = datetime.fromisoformat(last_updated)
                days_ago = (datetime.now() - updated).days
                if days_ago < 7:
                    score += 0.1
            except:
                pass
        
        return min(score, 1.0)
```

## 20.5 知识新鲜度管理

### 20.5.1 过期检测

```python
class FreshnessManager:
    """知识新鲜度管理器"""
    
    def __init__(self, llm):
        self.llm = llm
        
        # 不同类别文档的默认有效期限
        self.default_ttl = {
            '技术文档': {
                '系统架构': 180,
                '接口规范': 90,
                '开发指南': 120,
                '部署文档': 60
            },
            '产品文档': {
                'PRD': 90,
                '功能说明': 180,
                '用户手册': 365,
                '版本发布': 30
            },
            '管理文档': {
                '流程规范': 365,
                '制度文件': 365,
                '周报月报': 30,
                '会议纪要': 90
            },
            '知识资产': {
                '最佳实践': 365,
                '案例库': 365,
                '培训材料': 180,
                '研究笔记': 180
            },
            '商务文档': {
                '合同': 365,
                '报价': 30,
                '客户资料': 90,
                '市场分析': 90
            }
        }
    
    def check_freshness(self, document: Dict) -> Dict:
        """检查文档新鲜度"""
        metadata = document.get('metadata', {})
        category = metadata.get('category', '')
        subcategory = metadata.get('subcategory', '')
        
        # 获取有效期限
        ttl = self._get_ttl(category, subcategory)
        
        # 计算文档年龄
        created_at = metadata.get('created_at', '')
        if created_at:
            try:
                created = datetime.fromisoformat(created_at)
                age_days = (datetime.now() - created).days
            except:
                age_days = 0
        else:
            age_days = 0
        
        freshness_score = max(0, 1.0 - age_days / ttl) if ttl > 0 else 0.5
        
        return {
            'doc_id': document.get('id', ''),
            'age_days': age_days,
            'ttl_days': ttl,
            'freshness_score': freshness_score,
            'is_expired': age_days > ttl,
            'days_until_expiry': max(0, ttl - age_days),
            'needs_review': age_days > ttl * 0.8
        }
    
    def _get_ttl(self, category: str, subcategory: str) -> int:
        """获取有效期限"""
        category_ttl = self.default_ttl.get(category, {})
        return category_ttl.get(subcategory, 180)  # 默认180天
    
    def auto_update_check(self, document: Dict) -> Dict:
        """自动检查是否需要更新"""
        freshness = self.check_freshness(document)
        
        if not freshness['needs_review']:
            return {'needs_update': False, 'freshness': freshness}
        
        # 使用LLM检查内容是否过时
        prompt = f"""判断以下文档内容是否已经过时。

文档标题：{document.get('metadata', {}).get('title', '')}
文档内容：{document.get('content', '')[:500]}
文档创建日期：{document.get('metadata', {}).get('created_at', '')}
距离创建已过：{freshness['age_days']}天

请判断：
1. 文档内容是否可能已过时
2. 哪些部分需要更新

输出JSON：
{{
    "is_outdated": true/false,
    "confidence": 0.0-1.0,
    "outdated_parts": ["可能过时的部分"],
    "update_suggestion": "更新建议"
}}"""
        
        try:
            import json
            response = self.llm(prompt)
            result = json.loads(response)
            result['freshness'] = freshness
            return result
        except:
            return {
                'needs_update': freshness['is_expired'],
                'freshness': freshness
            }
```

### 20.5.2 自动刷新策略

```python
class KnowledgeRefreshScheduler:
    """知识刷新调度器"""
    
    def __init__(self, document_store, freshness_manager):
        self.document_store = document_store
        self.freshness_manager = freshness_manager
        self.refresh_queue = []
    
    def scan_for_updates(self):
        """扫描需要更新的文档"""
        all_docs = self.document_store.get_all_active()
        
        for doc in all_docs:
            freshness = self.freshness_manager.check_freshness(doc)
            
            if freshness['needs_review']:
                self.refresh_queue.append({
                    'doc': doc,
                    'freshness': freshness,
                    'priority': 'high' if freshness['is_expired'] else 'medium'
                })
        
        # 按优先级排序
        self.refresh_queue.sort(
            key=lambda x: (
                0 if x['priority'] == 'high' else 1,
                x['freshness']['days_until_expiry']
            )
        )
        
        return {
            'total_scanned': len(all_docs),
            'needs_review': len(self.refresh_queue),
            'expired': sum(1 for r in self.refresh_queue if r['freshness']['is_expired'])
        }
    
    def process_refresh_queue(self, batch_size: int = 10) -> List[Dict]:
        """处理刷新队列"""
        batch = self.refresh_queue[:batch_size]
        self.refresh_queue = self.refresh_queue[batch_size:]
        
        results = []
        for item in batch:
            doc = item['doc']
            
            # 通知文档所有者
            owner = doc.get('metadata', {}).get('owner', '')
            if owner:
                self._notify_owner(owner, doc)
            
            results.append({
                'doc_id': doc.get('id', ''),
                'owner': owner,
                'priority': item['priority'],
                'action': 'notified'
            })
        
        return results
    
    def _notify_owner(self, owner: str, document: Dict):
        """通知文档所有者"""
        # 实际实现：发送邮件或系统通知
        pass
```

## 20.6 搜索体验优化

### 20.6.1 搜索建议

```python
class SearchSuggestion:
    """搜索建议"""
    
    def __init__(self):
        self.popular_searches = []
        self.search_history = defaultdict(list)
    
    def get_suggestions(self, query: str, user_id: str = None) -> List[str]:
        """获取搜索建议"""
        suggestions = []
        
        # 1. 热门搜索
        popular = [s for s in self.popular_searches if query.lower() in s.lower()]
        suggestions.extend(popular[:3])
        
        # 2. 用户历史
        if user_id:
            history = self.search_history.get(user_id, [])
            relevant_history = [s for s in history if query.lower() in s.lower()]
            suggestions.extend(relevant_history[:2])
        
        # 3. 去重
        seen = set()
        unique_suggestions = []
        for s in suggestions:
            if s not in seen:
                seen.add(s)
                unique_suggestions.append(s)
        
        return unique_suggestions[:5]
    
    def record_search(self, user_id: str, query: str):
        """记录搜索"""
        self.search_history[user_id].append(query)
        
        # 限制历史长度
        if len(self.search_history[user_id]) > 100:
            self.search_history[user_id] = self.search_history[user_id][-100:]
    
    def update_popular_searches(self, queries: List[str]):
        """更新热门搜索"""
        from collections import Counter
        
        counter = Counter(queries)
        self.popular_searches = [
            q for q, _ in counter.most_common(20)
        ]
```

### 20.6.2 搜索结果高亮

```python
class SearchResultHighlighter:
    """搜索结果高亮"""
    
    def __init__(self):
        pass
    
    def highlight(self, text: str, query: str) -> str:
        """高亮匹配文本"""
        import re
        
        # 分词
        query_terms = self._tokenize(query)
        
        highlighted = text
        for term in query_terms:
            if len(term) < 2:
                continue
            
            # 不区分大小写替换
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            highlighted = pattern.sub(
                lambda m: f'<mark>{m.group()}</mark>',
                highlighted
            )
        
        return highlighted
    
    def get_snippet(self, text: str, query: str, 
                    max_length: int = 200) -> str:
        """获取带高亮的摘要"""
        import re
        
        query_terms = self._tokenize(query)
        
        # 找到匹配位置
        positions = []
        for term in query_terms:
            for match in re.finditer(re.escape(term), text, re.IGNORECASE):
                positions.append(match.start())
        
        if not positions:
            return text[:max_length] + "..."
        
        # 选择中心位置
        center = sum(positions) // len(positions)
        
        # 提取上下文
        start = max(0, center - max_length // 2)
        end = min(len(text), start + max_length)
        
        snippet = text[start:end]
        
        if start > 0:
            snippet = "..." + snippet
        if end < len(text):
            snippet = snippet + "..."
        
        # 高亮
        return self.highlight(snippet, query)
    
    def _tokenize(self, text: str) -> List[str]:
        """分词"""
        import re
        return re.findall(r'[一-鿿\w]+', text)
```

## 20.7 本章小结

本章详细介绍了RAG系统在企业内部知识管理场景中的应用方法和最佳实践。

**企业知识库构建**方面，本章设计了完整的文档分类体系，覆盖技术文档、产品文档、管理文档、知识资产和商务文档五大类别。文档生命周期管理器支持版本管理、过期检测和归档功能。

**权限管理**通过RBAC（基于角色的访问控制）和文档级ACL（访问控制列表）两层机制，确保不同角色和部门的员工只能访问其权限范围内的文档。这在实际企业环境中至关重要。

**搜索增强**实现了分面搜索和个性化排序两大功能。分面搜索支持按类别、部门、作者、日期和敏感级别等维度筛选结果。个性化排序基于用户行为画像，将用户偏好的文档排在前面，提升搜索效率。

**知识新鲜度管理**解决了企业知识时效性的问题。通过分类别设置不同的有效期限，结合LLM自动检测文档是否过时，确保知识库的信息始终是最新的。

**搜索体验优化**方面，搜索建议、结果高亮和智能摘要等功能提升了用户的搜索体验，让知识获取更加高效。

在实际部署中，建议先完成文档分类和权限体系的基础建设，再逐步引入搜索增强和个性化功能。知识新鲜度管理应该作为持续性工作，建立定期审核和自动提醒机制。同时，建议建立知识贡献激励机制，鼓励员工主动更新和维护知识库，形成良性的知识管理生态。
