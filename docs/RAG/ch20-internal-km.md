# 第 20 章：内部知识管理 RAG 实践

## 20.1 场景概述

企业知识管理是 RAG 的第二大落地场景，与客服场景的核心区别在于：**用户是内部员工，关注效率而非转化**。

### 20.1.1 典型需求

| 场景 | 用户 | 查询特点 | 知识来源 |
|------|------|---------|---------|
| 文档检索 | 全体员工 | 精确查找制度文件 | 内部 Wiki、SharePoint |
| 培训助手 | 新员工 | 入职培训相关问答 | 培训手册、课程视频 |
| 制度问答 | 全体员工 | HR/财务/法务政策 | 制度文档、流程文件 |
| 技术文档 | 研发团队 | API 文档、架构设计 | 内部文档、代码注释 |
| 项目经验 | 项目经理 | 历史项目复盘 | 项目文档、会议纪要 |

---

## 20.2 知识管理的核心挑战

### 20.2.1 权限控制

内部知识管理最核心的需求是**权限**：

```python
class KnowledgeAccessControl:
    """知识库访问控制（RBAC 模型）。"""

    def __init__(self):
        # 用户角色
        self.roles = {
            "admin": {"can_access": "*"},
            "manager": {"can_access": ["hr", "finance", "tech"]},
            "employee": {"can_access": ["tech", "training"]},
            "intern": {"can_access": ["training"]},
        }

        # 文档权限标签
        self.doc_permissions = {
            "hr_policy_2024": {"allowed_roles": ["admin", "manager"]},
            "salary_standard": {"allowed_roles": ["admin"]},
            "api_docs": {"allowed_roles": ["admin", "manager", "employee"]},
            "onboarding_guide": {"allowed_roles": "*"},
        }

    def can_access(self, user_role: str, doc_id: str) -> bool:
        """检查用户是否有权限访问文档。"""
        if user_role not in self.roles:
            return False

        role_config = self.roles[user_role]
        if role_config["can_access"] == "*":
            return True

        doc_perm = self.doc_permissions.get(doc_id, {"allowed_roles": "*"})
        if doc_perm["allowed_roles"] == "*":
            return True

        return user_role in doc_perm["allowed_roles"]

    def filter_documents(self, user_role: str, docs: list[dict]) -> list[dict]:
        """过滤用户有权访问的文档。"""
        return [d for d in docs if self.can_access(user_role, d["id"])]
```

### 20.2.2 知识版本管理

```python
class KnowledgeVersionManager:
    """知识版本管理。"""

    def __init__(self):
        self.versions: dict[str, list[dict]] = defaultdict(list)

    def add_version(self, doc_id: str, content: str, author: str, comment: str = ""):
        """添加新版本。"""
        version = {
            "version": len(self.versions[doc_id]) + 1,
            "content": content,
            "author": author,
            "comment": comment,
            "timestamp": time.time(),
        }
        self.versions[doc_id].append(version)

    def get_latest(self, doc_id: str) -> Optional[dict]:
        """获取最新版本。"""
        versions = self.versions.get(doc_id, [])
        return versions[-1] if versions else None

    def rollback(self, doc_id: str, version: int) -> bool:
        """回滚到指定版本。"""
        versions = self.versions.get(doc_id, [])
        target = next((v for v in versions if v["version"] == version), None)
        if target and versions:
            # 复制目标版本到最新
            self.versions[doc_id].append({
                "version": len(versions) + 1,
                "content": target["content"],
                "author": "system",
                "comment": f"回滚到版本 {version}",
                "timestamp": time.time(),
            })
            return True
        return False
```

---

## 20.3 企业搜索增强

内部知识管理场景对搜索精度要求更高，需要多种搜索方式结合：

```python
class EnterpriseSearch:
    """企业级搜索（混合检索 + 个性化排序）。"""

    def __init__(self):
        self.popularity: dict[str, float] = defaultdict(float)  # 文档热度
        self.user_preferences: dict[str, set[str]] = defaultdict(set)  # 用户偏好

    def search(self, query: str, user_id: str, top_k: int = 10) -> list[dict]:
        # 1. 基础检索（向量 + BM25）
        results = self._base_retrieval(query, top_k=top_k * 2)

        # 2. 个性化加权
        prefs = self.user_preferences.get(user_id, set())
        for doc in results:
            # 热度和偏好加分
            doc["score"] *= (1 + 0.1 * self.popularity.get(doc["id"], 0))
            if doc["category"] in prefs:
                doc["score"] *= 1.2

        # 3. 权限过滤
        results = self._filter_by_permission(user_id, results)

        results.sort(key=lambda x: -x["score"])
        return results[:top_k]

    def record_click(self, user_id: str, doc_id: str, category: str):
        """记录用户点击（用于个性化学习）。"""
        self.popularity[doc_id] += 1
        self.user_preferences[user_id].add(category)
```

---

## 20.4 最佳实践

| 策略 | 说明 | 效果 |
|------|------|------|
| 权限过滤嵌入检索 | 检索前先按权限过滤文档集合 | 防止越权访问 |
| 文档热度加权 | 高频访问文档提高排序权重 | 提升 30% 首答案命中率 |
| 个人偏好学习 | 基于历史点击记录调整搜索排序 | 提升 15% 满意度 |
| 版本追溯 | 答案附带文档版本号 | 降低 50% 因文档过期导致的错误 |
| 知识图谱增强 | 将部门架构、项目关系建模为图谱 | 提升跨部门检索效果 |

---

*下一章 [第 21 章：BI 分析与流程自动化 RAG 实践](ch21-bi-automation.md)*
