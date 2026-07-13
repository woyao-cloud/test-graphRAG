"""
ch19-enterprise-kb: Enterprise Knowledge Base Demo
Multi-department collections (HR, Tech, Finance), partition-based access control,
cross-department vs department-specific search, and document metadata management.
Uses MilvusClient API with sample Chinese enterprise documents.
"""

import hashlib
import random
import time
from datetime import datetime

import numpy as np
from pymilvus import MilvusClient

# ── Configuration ──────────────────────────────────────────────────────────────
MILVUS_URI = "http://localhost:19530"
COLLECTION_NAME = "enterprise_kb"
DIM = 64

# ── Sample Chinese Enterprise Documents ─────────────────────────────────────────

DOCUMENTS = {
    "HR": [
        {"title": "员工入职流程", "content": "新员工入职需要完成背景调查、签订劳动合同、领取办公设备、参加入职培训。试用期为三个月，期间进行绩效考核。"},
        {"title": "薪资福利政策", "content": "公司实行13薪制度，每年发放年终奖金。员工享有五险一金、带薪年假、补充医疗保险和年度体检。"},
        {"title": "考勤管理规定", "content": "工作时间为周一至周五9:00-18:00。员工需使用打卡系统记录考勤。迟到早退会影响月度绩效评估。"},
        {"title": "培训发展计划", "content": "公司每年提供培训预算。员工可申请参加专业技能培训、管理能力提升课程和行业会议。"},
        {"title": "离职管理流程", "content": "员工离职需提前30天提交书面申请。离职前需完成工作交接、归还公司资产、办理离职手续。"},
    ],
    "Tech": [
        {"title": "系统架构设计规范", "content": "系统采用微服务架构。服务间通过gRPC通信。每个服务需有独立的数据库和API网关。"},
        {"title": "代码审查指南", "content": "所有代码变更需通过代码审查。审查关注代码质量、安全漏洞和性能优化。使用Git进行版本控制。"},
        {"title": "数据库管理规范", "content": "生产数据库需定期备份。查询需使用索引优化性能。敏感数据需加密存储。数据库变更需提交变更申请。"},
        {"title": "部署发布流程", "content": "部署采用CI/CD流水线。代码需通过自动化测试和代码审查后才能合并到主分支。生产环境部署需审批。"},
        {"title": "安全运维手册", "content": "系统安全包括网络安全、应用安全和数据安全。定期进行安全审计和漏洞扫描。建立应急响应机制。"},
    ],
    "Finance": [
        {"title": "预算编制流程", "content": "各部门需在每年11月提交下年度预算。预算包括人员成本、运营费用和项目支出。财务部汇总后提交管理层审批。"},
        {"title": "报销管理制度", "content": "员工报销需提供合规发票。差旅费用按公司标准报销。大额支出需提前申请。"},
        {"title": "财务报表制度", "content": "每月5日前完成上月财务报表。季度报告需包含财务分析和预测。年度报告需经外部审计。"},
        {"title": "采购管理流程", "content": "采购需通过供应商评估。大额采购需招标。采购合同需法务审核。"},
        {"title": "资产管理规定", "content": "固定资产需登记台账。每年进行资产盘点。资产报废需提交申请并经审批。"},
    ],
}

DEPARTMENTS = ["HR", "Tech", "Finance"]


def generate_embedding(text: str, dim: int = DIM) -> list[float]:
    """Deterministic pseudo-embedding."""
    h = hashlib.md5(text.encode()).hexdigest()
    return [((int(h[i : i + 2], 16) / 255.0) * 2 - 1) for i in range(0, dim * 2, 2)]


# ── User Role Simulator ─────────────────────────────────────────────────────────


class User:
    def __init__(self, name: str, department: str, role: str):
        self.name = name
        self.department = department
        self.role = role  # "admin", "manager", "staff"

    def can_access(self, doc_department: str) -> bool:
        """Simulate RBAC: admin sees all; managers see own + HR; staff see only own."""
        if self.role == "admin":
            return True
        if doc_department == "HR":
            return self.department == "HR" or self.role == "manager"
        return self.department == doc_department

    def __str__(self) -> str:
        return f"{self.name} ({self.department}/{self.role})"


USERS = [
    User("Alice", "Tech", "admin"),
    User("Bob", "Tech", "manager"),
    User("Charlie", "Tech", "staff"),
    User("Diana", "HR", "staff"),
    User("Eve", "Finance", "manager"),
]


# ── Collection Setup ────────────────────────────────────────────────────────────


def ensure_collection(client: MilvusClient):
    """Create collection with partitions for each department."""
    if client.has_collection(COLLECTION_NAME):
        client.drop_collection(COLLECTION_NAME)

    schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=True)
    schema.add_field("id", datatype="INT64", is_primary=True, auto_id=True)
    schema.add_field("vector", datatype="FLOAT_VECTOR", dim=DIM)
    schema.add_field("title", datatype="VARCHAR", max_length=256)
    schema.add_field("content", datatype="VARCHAR", max_length=1024)
    schema.add_field("department", datatype="VARCHAR", max_length=32)
    schema.add_field("author", datatype="VARCHAR", max_length=64)
    schema.add_field("created_at", datatype="VARCHAR", max_length=32)
    schema.add_field("tags", datatype="VARCHAR", max_length=256)

    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(field_name="vector", index_type="FLAT", metric_type="L2")

    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
        num_partitions=len(DEPARTMENTS),
    )

    # Create partitions
    for dept in DEPARTMENTS:
        client.create_partition(collection_name=COLLECTION_NAME, partition_name=dept)

    print(f"Created collection '{COLLECTION_NAME}' with {len(DEPARTMENTS)} partitions")


def insert_documents(client: MilvusClient):
    """Insert sample documents into respective department partitions."""
    total = 0
    authors_by_dept = {
        "HR": ["张经理", "王主管"],
        "Tech": ["李工", "赵工", "刘架构"],
        "Finance": ["陈总监", "周会计"],
    }

    for dept, docs in DOCUMENTS.items():
        data = []
        for doc in docs:
            data.append({
                "vector": generate_embedding(doc["title"] + doc["content"]),
                "title": doc["title"],
                "content": doc["content"],
                "department": dept,
                "author": random.choice(authors_by_dept[dept]),
                "created_at": datetime.now().isoformat(),
                "tags": f"enterprise,{dept.lower()},kb",
            })
        result = client.insert(collection_name=COLLECTION_NAME, data=data, partition_name=dept)
        total += len(result)
        print(f"  Inserted {len(result)} documents into {dept} partition")

    print(f"  Total documents: {total}")
    return total


# ── Search Functions ────────────────────────────────────────────────────────────


def department_search(client: MilvusClient, user: User, query: str, top_k: int = 3):
    """Search within user's accessible partitions."""
    q_vec = generate_embedding(query)
    accessible = [d for d in DEPARTMENTS if user.can_access(d)]

    print(f"\n  User: {user}")
    print(f"  Accessible departments: {accessible}")

    all_results = []
    for dept in accessible:
        results = client.search(
            collection_name=COLLECTION_NAME,
            data=[q_vec],
            limit=top_k,
            partition_names=[dept],
            output_fields=["title", "content", "department", "author"],
        )
        if results[0]:
            all_results.extend(results[0])

    # Sort by distance
    all_results.sort(key=lambda x: x["distance"])
    return all_results[:top_k]


def cross_department_search(client: MilvusClient, query: str, top_k: int = 5):
    """Search across all departments (admin view)."""
    q_vec = generate_embedding(query)
    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[q_vec],
        limit=top_k,
        output_fields=["title", "content", "department", "author"],
    )
    return results[0] if results else []


def search_with_metadata_filter(
    client: MilvusClient,
    query: str,
    department: str | None = None,
    author: str | None = None,
    top_k: int = 5,
):
    """Search with metadata filters."""
    q_vec = generate_embedding(query)
    filters = []
    if department:
        filters.append(f'department == "{department}"')
    if author:
        filters.append(f'author == "{author}"')

    filter_expr = " and ".join(filters) if filters else None

    kwargs = {
        "collection_name": COLLECTION_NAME,
        "data": [q_vec],
        "limit": top_k,
        "output_fields": ["title", "content", "department", "author"],
    }
    if filter_expr:
        kwargs["filter"] = filter_expr
        if department:
            kwargs["partition_names"] = [department]

    results = client.search(**kwargs)
    return results[0] if results else []


# ── Main Demo ───────────────────────────────────────────────────────────────────


def main():
    print("=" * 60)
    print("ch19: Enterprise Knowledge Base Demo")
    print("=" * 60)

    print("\n[0] Connecting to Milvus...")
    client = MilvusClient(uri=MILVUS_URI)
    ensure_collection(client)
    insert_documents(client)

    # 1. Cross-department search (admin view)
    print("\n[1] Cross-Department Search (Admin View)")
    print("-" * 40)
    query = "员工福利和培训"
    results = cross_department_search(client, query)
    print(f"  Query: '{query}'")
    for r in results:
        print(f"    [{r['entity']['department']}] {r['entity']['title']} (dist: {r['distance']:.4f})")

    # 2. Role-based access (staff sees only own department)
    print("\n[2] Role-Based Access Simulation")
    print("-" * 40)
    query2 = "安全和管理规范"

    for user in USERS[:3]:
        results = department_search(client, user, query2)
        for r in results:
            print(f"    [{r['entity']['department']}] {r['entity']['title']} - {r['entity']['author']}")

    # 3. Department-specific search
    print("\n[3] Department-Specific Search")
    print("-" * 40)
    query3 = "流程"
    for dept in ["HR", "Tech", "Finance"]:
        results = client.search(
            collection_name=COLLECTION_NAME,
            data=[generate_embedding(query3)],
            limit=2,
            partition_names=[dept],
            output_fields=["title", "department"],
        )
        titles = [r["entity"]["title"] for r in results[0]]
        print(f"  {dept}: {', '.join(titles)}")

    # 4. Metadata-filtered search
    print("\n[4] Metadata-Filtered Search")
    print("-" * 40)
    results = search_with_metadata_filter(client, "系统", department="Tech")
    print(f"  Filter: department=Tech, query='系统'")
    for r in results:
        print(f"    {r['entity']['title']} (by {r['entity']['author']})")

    # 5. Document metadata report
    print("\n[5] Document Metadata Summary")
    print("-" * 40)
    for dept in DEPARTMENTS:
        results = client.query(
            collection_name=COLLECTION_NAME,
            filter=f'department == "{dept}"',
            output_fields=["title", "author", "department"],
            limit=10,
        )
        print(f"  {dept} ({len(results)} documents):")
        for r in results:
            print(f"    - {r['title']} [Author: {r['author']}]")

    print("\n" + "=" * 60)
    print("Enterprise KB Demo Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
