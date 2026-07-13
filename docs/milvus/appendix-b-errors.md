# 附录B 常见报错与故障排查手册

在 Milvus 的部署和使用过程中，各类报错和异常难以完全避免。本附录系统整理了最常遇到的九大类问题，提供详细的错误现象描述、根因分析和可复现的解决方案。

## B.1 Milvus 连接失败

### B.1.1 Connection refused

**错误现象**
```
pymilvus.exceptions.MilvusException: <MilvusException: (code=1, message=connection refused)>
```

**根因分析**
- Milvus 服务未启动或启动后崩溃退出
- 端口配置错误，默认端口 19530 被占用或未暴露
- 防火墙/安全组阻止了连接请求
- Docker 容器未正确映射端口

**解决方案**

```bash
# Step 1: 检查 Milvus 容器是否运行
docker ps | grep milvus

# Step 2: 检查容器日志，定位崩溃原因
docker logs milvus-standalone

# Step 3: 确认端口映射是否正确
docker port milvus-standalone
# 预期输出: 19530/tcp -> 0.0.0.0:19530

# Step 4: 测试连接
python -c "from pymilvus import MilvusClient; c = MilvusClient('http://localhost:19530'); print(c.list_collections())"
```

**Windows 特别处理**
```yaml
# docker-compose.yml 中正确配置端口映射
ports:
  - "19530:19530"
  - "9091:9091"

# 如果 WSL2 网络异常，重启 Docker Desktop
# 或在 PowerShell 中执行：
net stop com.docker.service
net start com.docker.service
```

### B.1.2 Connection timeout

**错误现象**
```
pymilvus.exceptions.MilvusException: <MilvusException: (code=1, message=timeout)>
```

**根因分析**
- 网络延迟过高，客户端默认超时时间（10秒）过短
- Milvus 负载过高，请求排队超时
- DNS 解析问题，主机名无法正确解析

**解决方案**

```python
# 方案1：增加超时时间
from pymilvus import connections

connections.connect(
    alias="default",
    host="192.168.1.100",
    port="19530",
    timeout=30,  # 默认10秒，增大到30秒
)

# 方案2：使用健康检查判断服务是否就绪
import time
from pymilvus import MilvusClient

def wait_for_milvus(uri, max_retries=30, interval=2):
    """等待 Milvus 服务就绪"""
    for i in range(max_retries):
        try:
            client = MilvusClient(uri=uri)
            client.list_collections()
            print(f"Milvus 已就绪（尝试 {i+1} 次）")
            return client
        except Exception as e:
            print(f"等待 Milvus 就绪...（{i+1}/{max_retries}）: {e}")
            time.sleep(interval)
    raise RuntimeError("Milvus 启动超时")
```

## B.2 索引构建失败

### B.2.1 nlist out of range

**错误现象**
```
MilvusException: (code=1, message=nlist should be in range [1, 999999])
```

**根因分析**
- nlist 参数超出 Milvus 允许的范围
- 对于 IVF 系列索引，nlist 的值不能超过数据集大小的平方根

**解决方案**

```python
# 正确的 nlist 取值建议
def suggest_nlist(num_vectors):
    """根据数据量推荐 nlist 值"""
    if num_vectors < 10000:
        return 128
    elif num_vectors < 100000:
        return 256
    elif num_vectors < 500000:
        return 512
    elif num_vectors < 1000000:
        return 1024
    elif num_vectors < 5000000:
        return 2048
    else:
        return 4096  # 最大建议值

# nlist 不应超过 sqrt(n) 的10倍
import math
nlist = min(suggest_nlist(num_vectors), int(math.sqrt(num_vectors) * 10))
```

### B.2.2 metric type not supported

**错误现象**
```
MilvusException: (code=1, message=metric type not supported)
```

**根因分析**
- 指定的距离度量类型不被当前索引类型支持
- 常见的不兼容组合：HNSW + L2 在某些旧版本中不可用

**解决方案**

```python
# 索引类型与度量类型的兼容性对照表
INDEX_METRIC_COMPATIBILITY = {
    "FLAT":     ["L2", "IP", "COSINE"],
    "IVF_FLAT": ["L2", "IP"],
    "IVF_SQ8":  ["L2", "IP"],
    "IVF_PQ":   ["L2", "IP"],
    "HNSW":     ["L2", "IP", "COSINE"],
    "DISKANN":  ["L2", "IP"],
}

# 检查兼容性
def check_index_metric(index_type, metric_type):
    supported = INDEX_METRIC_COMPATIBILITY.get(index_type, [])
    if metric_type not in supported:
        raise ValueError(
            f"索引类型 '{index_type}' 不支持度量类型 '{metric_type}'。"
            f"支持的度量类型: {supported}"
        )
```

## B.3 检索为空/召回错误

### B.3.1 Empty results

**错误现象**
- `search()` 返回空列表 `[]`
- 明明有数据，但检索不到任何结果

**根因分析**
- 集合未加载到内存：创建索引后未调用 `load_collection()`
- 查询向量维度与集合定义不一致
- 过滤条件过于严格，排除了所有数据
- 索引参数不合理，导致 ANN 搜索无法找到任何邻居

**解决方案**

```python
# Step 1: 确认集合已加载
client.load_collection(collection_name)

# Step 2: 检查集合状态
info = client.get_collection_stats(collection_name)
print(f"集合状态: {info}")
# 确认 row_count > 0

# Step 3: 先做无过滤条件的检索
results = client.search(
    collection_name=collection_name,
    data=[query_vector],
    limit=10,
    # 不设置 filter，排除过滤条件的影响
)

# Step 4: 检查查询向量的维度
assert len(query_vector) == DIM, f"维度不匹配: 期望 {DIM}, 实际 {len(query_vector)}"
```

### B.3.2 Wrong results（召回错误）

**错误现象**
- 检索返回了结果，但结果与查询语义不相关
- 排序不符合预期，不相似的结果排在前面

**根因分析**
- 向量未归一化，导致距离度量计算异常
- 索引参数（如 nprobe/ef）过小，ANN 搜索精度不足
- 数据本身质量问题，切片过长或过短
- Embedding 模型与检索领域不匹配

**解决方案**

```python
# 方案1：向量归一化
import numpy as np

def normalize_vector(vector):
    """L2 归一化"""
    vec = np.array(vector, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()

# 方案2：增加检索精度参数
search_params = {
    "metric_type": "IP",
    "params": {
        "nprobe": 256,  # IVF 索引：增大 nprobe 提升召回率
        "ef": 200,      # HNSW 索引：增大 ef 提升召回率
    }
}

# 方案3：使用 FLAT 索引对比验证
# FLAT 索引做暴力检索，验证 ANN 索引的召回率
flat_results = client.search(
    collection_name=collection_name,
    data=[query_vector],
    limit=100,
    search_params={"metric_type": "IP"},
)
```

## B.4 MinIO 鉴权失败

### B.4.1 Access denied / Signature mismatch

**错误现象**
- Milvus 日志中出现 `SignatureDoesNotMatch` 或 `AccessDenied`
- 容器启动后反复重启，检查日志发现 MinIO 连接失败
- 数据写入失败，报错与 S3 存储相关

**根因分析**
- MinIO 的 access key 和 secret key 配置不一致
- docker-compose.yml 中的 MINIO_ROOT_USER / MINIO_ROOT_PASSWORD 修改后未同步更新 Milvus 配置
- MinIO 数据卷权限问题，导致密钥文件无法读取

**解决方案**

```yaml
# docker-compose.yml 中统一配置 MinIO 认证信息
minio:
  image: minio/minio:RELEASE.2023-03-20T20-16-18Z
  environment:
    MINIO_ROOT_USER: minioadmin      # 统一使用 minioadmin
    MINIO_ROOT_PASSWORD: minioadmin  # 不要包含特殊字符
  volumes:
    - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/minio:/minio_data
  command: minio server /minio_data --console-address ":9001"

# Milvus 的配置需与 MinIO 完全一致
etcd:
  ...
standalone:
  environment:
    MINIO_ACCESS_KEY: minioadmin
    MINIO_SECRET_KEY: minioadmin
```

**密钥重置后的处理**
```bash
# 如果修改了 MinIO 密码，需要：
# 1. 删除 MinIO 数据卷（谨慎操作，会丢失数据）
docker volume rm <volume_name>

# 2. 或进入 MinIO 容器重置密码
docker exec -it milvus-minio bash
mc alias set local http://localhost:9000 minioadmin minioadmin
mc admin user info local minioadmin
```

## B.5 集合已存在/冲突

### B.5.1 Collection already exists

**错误现象**
```
MilvusException: (code=1, message=collection already exists: <name>)
```

**根因分析**
- 创建集合时未检查是否已存在同名集合
- 分布式环境下并发创建集合导致冲突
- 前一次运行未清理测试数据

**解决方案**

```python
# 方案1：安全创建集合（不存在则创建）
def safe_create_collection(client, collection_name, schema, index_params):
    """安全创建集合"""
    if client.has_collection(collection_name):
        print(f"集合 '{collection_name}' 已存在，跳过创建")
        return False
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params,
    )
    print(f"集合 '{collection_name}' 创建成功")
    return True

# 方案2：显式删除后重建（开发/测试环境）
def recreate_collection(client, collection_name, schema, index_params):
    """删除后重建（仅在确定可以删除时使用）"""
    if client.has_collection(collection_name):
        client.drop_collection(collection_name)
        print(f"已删除旧集合 '{collection_name}'")
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params,
    )

# 方案3：使用 if_not_exists 参数（Milvus 2.5+）
client.create_collection(
    collection_name=collection_name,
    schema=schema,
    index_params=index_params,
    if_not_exists=True,  # 不存在才创建
)
```

## B.6 维度不匹配

### B.6.1 Dimension mismatch

**错误现象**
```
MilvusException: (code=1, message=dimension mismatch, expected 768, got 512)
```

**根因分析**
- 插入数据的向量维度与集合定义的维度不一致
- 使用了不同的 Embedding 模型生成了不同维度的向量
- 同一个集合中混入了不同模型生成的向量

**解决方案**

```python
# 方案1：插入前验证维度
def validate_and_insert(client, collection_name, data, dim):
    """验证维度后插入"""
    for item in data:
        vec = item.get("vector", [])
        if len(vec) != dim:
            raise ValueError(
                f"向量维度错误: 期望 {dim}, 实际 {len(vec)}"
            )
    return client.insert(collection_name=collection_name, data=data)

# 方案2：获取集合的 Schema 信息
def get_collection_dimension(client, collection_name):
    """获取集合定义的向量维度"""
    schema = client.describe_collection(collection_name)
    for field in schema["fields"]:
        if field["type"] == "FLOAT_VECTOR":
            return field["params"]["dim"]
    raise ValueError("未找到向量字段")

# 方案3：统一 Embedding 模型管理
class EmbeddingManager:
    """统一管理 Embedding 模型和维度"""

    def __init__(self, model_name="BAAI/bge-large-zh-v1.5"):
        self.model_name = model_name
        self.dimension = self._get_dimension(model_name)

    def _get_dimension(self, model_name):
        dimensions = {
            "BAAI/bge-large-zh-v1.5": 1024,
            "BAAI/bge-base-zh-v1.5": 768,
            "BAAI/bge-small-zh-v1.5": 512,
            "text-embedding-ada-002": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
        }
        return dimensions.get(model_name, 768)
```

## B.7 超时/性能问题

### B.7.1 Timeout / Slow queries

**错误现象**
- 检索请求频繁超时
- QPS 远低于预期
- 单次检索延迟从毫秒级飙升到秒级

**根因分析**
- 集合未加载到内存，检索走了磁盘
- 索引参数不合理（如 IVF 的 nprobe 过大）
- 并发请求超过系统承载能力
- 硬件资源不足（内存、CPU）
- 数据量增长后未重建索引

**解决方案**

```python
# Step 1: 确认集合已加载到内存
loading_progress = utility.loading_progress(collection_name)
print(f"加载进度: {loading_progress}")

# Step 2: 检查索引状态
index_status = utility.index_building_progress(collection_name)
print(f"索引状态: {index_status}")

# Step 3: 优化检索参数
# IVF 索引：降低 nprobe（牺牲精度换取速度）
optimized_params = {
    "metric_type": "IP",
    "params": {"nprobe": 64},  # 从 256 降低到 64，速度提升 4 倍
}

# HNSW 索引：降低 ef
optimized_params = {
    "metric_type": "IP",
    "params": {"ef": 64},  # 从 200 降低到 64
}

# Step 4: 升级索引类型
# 如果当前使用 IVF_FLAT，切换到 IVF_SQ8 可大幅提升速度
# 如果当前使用 IVF_SQ8，切换到 HNSW 可提升速度

# Step 5: 资源监控
def check_system_resources():
    """检查系统资源使用情况"""
    import psutil
    memory = psutil.virtual_memory()
    cpu = psutil.cpu_percent(interval=1)
    print(f"内存: {memory.percent}% ({memory.used/1024**3:.1f}GB/{memory.total/1024**3:.1f}GB)")
    print(f"CPU: {cpu}%")
```

## B.8 数据不一致

### B.8.1 Data inconsistency

**错误现象**
- 插入数据后立即检索，结果不包含刚插入的数据
- 不同节点查询到的数据不一致
- 数据量统计不准确

**根因分析**
- Milvus 的数据可见性机制：数据插入后需要经过 Flush 才能被检索到
- 分布式环境下数据同步延迟
- 数据写入后未调用 flush 或等待时间不足

**解决方案**

```python
# 方案1：插入后强制 flush
client.insert(collection_name=collection_name, data=data)
client.flush(collection_name=collection_name)  # 强制刷新，确保数据可见

# 方案2：等待数据可见
import time

def insert_and_wait(client, collection_name, data, timeout=30):
    """插入数据并等待可见"""
    ids = client.insert(collection_name=collection_name, data=data)
    client.flush(collection_name=collection_name)

    # 等待数据可检索
    start = time.time()
    while time.time() - start < timeout:
        count = client.query(
            collection_name=collection_name,
            filter=f"id in {ids[:5]}",
            limit=1,
        )
        if count:
            print(f"数据已可见（等待 {time.time()-start:.1f}s）")
            return ids
        time.sleep(0.5)

    raise TimeoutError("数据写入后超时未见")

# 方案3：检查数据一致性
def verify_data_consistency(client, collection_name, expected_count):
    """验证数据一致性"""
    actual_count = client.get_collection_stats(collection_name).get("row_count", 0)
    if actual_count != expected_count:
        print(f"警告: 数据不一致！期望 {expected_count}，实际 {actual_count}")
        # 触发 compaction
        client.compact(collection_name=collection_name)
        time.sleep(5)
        actual_count = client.get_collection_stats(collection_name).get("row_count", 0)
        print(f"Compact 后: {actual_count}")
```

## B.9 容器启动失败

### B.9.1 Docker container exits

**错误现象**
- Docker 容器启动后立即退出
- `docker ps` 显示容器状态为 `Exited`
- `docker logs` 显示错误信息后退出

**根因分析**
- 端口冲突（19530、9091 等端口已被占用）
- 数据卷权限不足（MinIO 数据目录无写入权限）
- 内存不足（Milvus 默认需要 8GB+ 内存）
- Docker 版本过低或不兼容
- Windows WSL2 资源限制

**解决方案**

```bash
# Step 1: 检查端口占用
netstat -ano | findstr "19530"
# 如果端口被占用，终止占用进程或在 docker-compose.yml 中修改端口映射

# Step 2: 检查容器日志
docker logs milvus-standalone --tail 50

# Step 3: 检查数据卷权限（Windows 特别关注）
# Windows 上确保 volumes 目录已创建且有写入权限
mkdir -p volumes/minio volumes/etcd

# Step 4: 增加 Docker 资源限制（WSL2）
# 在 %USERPROFILE%\.wslconfig 中配置：
# [wsl2]
# memory=16GB
# processors=4

# Step 5: 使用更轻量的配置启动
# docker-compose.yml 中降低资源需求
standalone:
  environment:
    - CACHE_SIZE=2GB   # 默认 4GB，降低到 2GB
```

**Windows Docker Desktop 特别排查**
```powershell
# 检查 Docker 引擎状态
docker info

# 清理 Docker 缓存（释放磁盘空间）
docker system prune -a

# 重置 WSL2 网络
wsl --shutdown
# 重启 Docker Desktop

# 如果依然无法启动，检查 Docker Desktop 设置：
# Settings > Resources > Advanced > Memory 至少设置为 8GB
```

## B.10 通用排查流程

当遇到未在以上列表中列出的报错时，遵循以下通用排查流程：

```
1. 收集信息
   ├── 完整错误信息（code, message, stack trace）
   ├── 操作复现步骤
   ├── Milvus 版本和部署方式
   └── 最近变更记录

2. 检查基础环境
   ├── Milvus 服务状态（docker ps / systemctl status）
   ├── 磁盘空间（df -h）
   ├── 内存使用（free -m）
   └── 网络连通性（ping / telnet）

3. 定位问题
   ├── 客户端问题 → 检查代码、SDK版本
   ├── 服务端问题 → 检查 Milvus 日志
   ├── 网络问题 → 检查防火墙、代理
   └── 配置问题 → 检查 docker-compose.yml、milvus.yaml

4. 搜索解决方案
   ├── 官方文档：https://milvus.io/docs
   ├── GitHub Issues：https://github.com/milvus-io/milvus/issues
   └── 社区论坛：https://discord.gg/milvus
```

## 本章小结

本附录系统整理了 Milvus 部署和使用中最常见的九大类报错问题。核心经验总结如下：

1. **连接问题**大多源于网络配置和认证信息不一致，优先检查端口映射和 MinIO 鉴权。
2. **索引问题**通常与参数配置不当有关，nlist/nprobe/M/ef 需要根据数据规模合理设置。
3. **检索为空**时先检查集合是否已加载、查询向量维度是否正确。
4. **性能问题**需要从索引类型、检索参数、硬件资源三个维度逐一排查。
5. **容器问题**在 Windows 环境下尤其多发，建议优先检查 WSL2 资源分配和数据卷权限。

遇到报错时，最重要的是保持系统化的排查思路：先收集完整错误信息，再检查基础环境，然后逐步缩小问题范围，最后针对性解决。
