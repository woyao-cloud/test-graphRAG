# 第8章: Milvus多环境部署

## 8.1 引言

在RAG系统的开发与生产落地过程中，Milvus的部署是一个绕不开的基础环节。不同的开发阶段对部署形态的要求截然不同：本地开发追求快速启动和便捷调试，测试环境需要稳定可复现的配置，而生产环境则必须考虑高可用、弹性扩缩容和数据安全。本章将系统性地介绍Milvus在Windows、Linux、Kubernetes和Embed模式下的部署方案，并配套讲解可视化工具Attu的使用方法，最后汇总常见的部署踩坑与解决方案，帮助读者在RAG项目中快速搭建合适的Milvus环境。

## 8.2 Windows Docker Desktop部署

对于大多数RAG开发人员来说，本地Windows机器是第一开发环境。通过Docker Desktop在Windows上运行Milvus是最快捷的方式。

### 8.2.1 前置准备

在Windows上部署Milvus之前，需要确保以下环境已就绪：

- **Docker Desktop for Windows**：从官网下载并安装，确保版本在4.0以上。
- **WSL2后端**：Docker Desktop默认使用WSL2作为引擎，性能远优于Hyper-V。安装后务必在Docker Desktop设置中启用"Use the WSL 2 based engine"。
- **内存配置**：Milvus单机版推荐至少分配4GB内存给Docker引擎。可以在Docker Desktop的Settings -> Resources -> Advanced中调整。
- **磁盘空间**：确保C盘或Docker数据目录所在分区有至少10GB可用空间。

### 8.2.2 使用Docker Compose启动Milvus

Milvus官方提供了完整的Docker Compose配置文件。在Windows上推荐使用本项目的docker-compose.yml作为模板：

```yaml
version: '3.8'

services:
  etcd:
    container_name: milvus-etcd
    image: quay.io/coreos/etcd:v3.5.5
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
      - ETCD_SNAPSHOT_COUNT=50000
    volumes:
      - etcd_data:/etcd
    command: etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd
    networks:
      - milvus-net

  minio:
    container_name: milvus-minio
    image: minio/minio:RELEASE.2023-03-20T20-16-18Z
    ports:
      - "9000:9000"
      - "9001:9001"
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    volumes:
      - minio_data:/minio_data
    command: minio server /minio_data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3
    networks:
      - milvus-net

  milvus:
    container_name: milvus-standalone
    image: milvusdb/milvus:v2.2.11
    command: ["milvus", "run", "standalone"]
    ports:
      - "19530:19530"
      - "9091:9091"
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
    volumes:
      - milvus_data:/var/lib/milvus
    depends_on:
      - "etcd"
      - "minio"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9091/health"]
      interval: 10s
      timeout: 5s
      retries: 12
      start_period: 30s
    networks:
      - milvus-net

volumes:
  etcd_data:
    driver: local
  minio_data:
    driver: local
  milvus_data:
    driver: local

networks:
  milvus-net:
    name: milvus-net
```

在PowerShell或CMD中执行以下命令启动：

```powershell
cd docs/milvus/demos
docker compose up -d
```

启动后验证服务状态：

```powershell
docker compose ps
curl http://localhost:9091/health
```

如果返回`{"status":"OK"}`，说明Milvus已成功运行。

### 8.2.3 解决Windows特有坑点

**端口占用问题**：Windows上端口19530、2379（etcd）、9000（MinIO）可能被其他程序占用。可以使用`netstat -ano | findstr :19530`查看占用情况，或在docker-compose.yml中修改映射端口。

**WSL2磁盘性能**：Milvus数据卷默认挂载在WSL2的虚拟磁盘中，性能较好。但如果将数据挂载到Windows文件系统（如`C:\milvus-data`），IO性能会显著下降。建议使用Docker管理的volume而非bind mount。

**防火墙拦截**：Windows防火墙可能拦截Milvus的19530端口。在防火墙中添加入站规则放行即可。

## 8.3 Linux Docker Compose快速部署

Linux是Milvus最推荐的运行环境，性能最好，坑最少。以下是在Ubuntu 22.04上的快速部署流程。

### 8.3.1 安装Docker和Docker Compose

```bash
# 安装Docker
sudo apt update
sudo apt install -y docker.io docker-compose-v2

# 将当前用户加入docker组，避免每次使用sudo
sudo usermod -aG docker $USER
newgrp docker

# 验证安装
docker --version
docker compose version
```

### 8.3.2 启动Milvus

与Windows相同，使用同样的docker-compose.yml文件：

```bash
mkdir -p /data/milvus
cd /data/milvus
wget https://raw.githubusercontent.com/milvus-io/milvus/master/deployments/docker-compose/docker-compose.yml

# 或使用本项目提供的配置
cp /path/to/docs/milvus/demos/docker-compose.yml .

docker compose up -d
```

### 8.3.3 性能调优建议

Linux环境相比Windows有更好的性能表现，但仍需关注以下配置：

```bash
# 检查系统mmap上限（Milvus依赖mmap管理内存）
sysctl vm.max_map_count
# 如果小于262144，建议修改
sudo sysctl -w vm.max_map_count=262144
echo "vm.max_map_count=262144" | sudo tee -a /etc/sysctl.conf

# 检查磁盘IO调度器（SSD建议使用none或noop）
cat /sys/block/sda/queue/scheduler
```

**内存分配建议**：Milvus单机版推荐至少8GB内存。如果服务器内存有限，可以在Milvus的配置文件中限制cache大小。对于RAG开发测试环境，4GB内存也可运行，但索引构建时会较慢。

## 8.4 K8s集群部署

对于生产级RAG系统，Kubernetes集群部署是必然选择。Milvus官方提供了Helm Chart，支持一键部署到K8s集群。

### 8.4.1 使用Helm安装Milvus Operator

```bash
# 添加Milvus Helm仓库
helm repo add milvus https://milvus-io.github.io/milvus-helm/
helm repo update

# 安装Milvus Operator
helm install my-milvus milvus/milvus \
  --set cluster.enabled=true \
  --set persistence.enabled=true \
  --set persistence.storageClass=standard \
  --namespace milvus \
  --create-namespace
```

### 8.4.2 生产级配置

对于生产环境，需要根据业务规模调整资源配置：

```yaml
# milvus-prod-values.yaml
apiVersion: milvus.io/v1beta1
kind: MilvusCluster
metadata:
  name: milvus-prod
spec:
  components:
    # 查询节点配置（影响检索性能）
    queryNode:
      replicas: 3
      resources:
        requests:
          cpu: "4"
          memory: "8Gi"
        limits:
          cpu: "8"
          memory: "16Gi"
    
    # 数据节点配置（影响写入性能）
    dataNode:
      replicas: 2
      resources:
        requests:
          cpu: "2"
          memory: "4Gi"
    
    # 索引节点（独立资源，不影响在线检索）
    indexNode:
      replicas: 2
      resources:
        requests:
          cpu: "2"
          memory: "4Gi"
  
  # 存储配置
  persistence:
    enabled: true
    storageClass: "managed-premium"
    size: "500Gi"
```

使用自定义配置部署：

```bash
helm install milvus-prod milvus/milvus \
  -f milvus-prod-values.yaml \
  --namespace milvus
```

### 8.4.3 访问集群中的Milvus

在K8s集群内部，服务可以通过Service名称访问：

```bash
# 端口转发用于本地调试
kubectl port-forward -n milvus svc/my-milvus 19530:19530

# 查看所有Pod状态
kubectl get pods -n milvus -o wide
```

K8s部署的优势在于可以独立扩缩各个组件。当RAG系统的并发查询量增大时，只需要增加QueryNode副本数即可线性提升检索吞吐量。

## 8.5 轻量化Embed模式

对于纯RAG开发测试场景，Milvus提供了Embed模式（也称为轻量模式），可以直接在Python进程中嵌入运行，无需Docker或Kubernetes。

### 8.5.1 安装与使用

```bash
pip install pymilvus
```

在Python代码中直接创建MilvusClient，指定本地文件存储路径即可：

```python
from pymilvus import MilvusClient

# Embed模式：数据存储在本地文件中
client = MilvusClient(uri="./milvus_embed.db")

# 使用方式与连接远程Milvus完全一致
collection_name = "rag_demo"
client.create_collection(
    collection_name=collection_name,
    dimension=768,
    auto_id=False,
)

# 插入数据
client.insert(collection_name, {
    "id": 1,
    "vector": [0.1] * 768,
    "text": "这是Embed模式的测试数据"
})

# 检索
results = client.search(
    collection_name=collection_name,
    data=[[0.1] * 768],
    limit=5,
)
```

### 8.5.2 Embed模式适用场景

- **本地原型开发**：在笔记本电脑上快速验证RAG流程，无需启动Docker。
- **单元测试**：在CI/CD流水线中集成Milvus操作测试，无需外部服务依赖。
- **小规模知识库**：数据量在百万级以下时，Embed模式性能足够。
- **教学演示**：本书配套Demo代码大量使用Embed模式，方便读者零门槛运行。

Embed模式的底层基于Milvus Lite实现，它将Milvus的核心引擎编译为Python原生库，牺牲了一定的并发能力和数据容量，但换来了极致的部署便利性。

## 8.6 可视化工具：Attu

Attu是Milvus官方提供的可视化Web管理工具，可以直观地查看集合、索引、向量数据以及执行检索操作，非常适合RAG开发过程中对知识库数据进行可视化排查。

### 8.6.1 部署Attu

使用Docker一键部署Attu：

```bash
docker run -d \
  --name attu \
  -p 8000:3000 \
  -e MILVUS_URL=localhost:19530 \
  zilliz/attu:latest
```

启动后，在浏览器中访问`http://localhost:8000`即可打开Attu管理界面。在连接配置中输入Milvus地址（如果Attu和Milvus在不同机器，填写对应的IP和端口）。

### 8.6.2 Attu的RAG场景功能

- **集合浏览**：查看所有集合（对应RAG知识库中的不同数据源）的字段结构、行数、索引状态。
- **向量检索测试**：直接在Web界面中粘贴向量数据或文本，执行相似度搜索，实时查看检索结果。这对调试RAG召回效果非常方便。
- **数据管理**：浏览、编辑、删除集合中的记录，检查入库的文档是否正确。
- **索引管理**：查看索引构建进度，手动触发索引重建。
- **查询分析**：查看慢查询日志，分析检索性能瓶颈。

在RAG开发过程中，当发现问答效果不理想时，首先应该通过Attu检查知识库中的数据是否完整、向量是否已建索引、检索返回的结果是否相关。这比直接修改Prompt或尝试不同的LLM模型更有效率。

## 8.7 部署避坑指南

### 8.7.1 内存配置

Milvus对内存的需求取决于数据量和索引类型。常见内存估算公式如下：

| 数据量 | 索引类型 | 预估内存 | 建议配置 |
|--------|---------|---------|---------|
| 10万条x768维 | FLAT | ~600MB | 2GB |
| 100万条x768维 | IVF_SQ8 | ~800MB | 4GB |
| 1000万条x768维 | HNSW | ~6GB | 16GB |
| 1亿条x768维 | IVF_PQ | ~4GB | 32GB |

**避坑要点**：
- Docker Desktop的默认内存限制通常为2GB，需要手动增大。
- 如果Milvus容器反复重启，`docker logs milvus-standalone`中看到OOM Killer日志，说明内存不足。
- Embed模式下，Milvus Lite与Python进程共享内存，需要为Python进程预留足够内存。

### 8.7.2 磁盘挂载

Windows下的磁盘挂载有三大坑：

1. **WSL2跨文件系统性能问题**：Docker volume（由WSL2管理）性能优于bind mount到Windows路径。推荐始终使用volume。
2. **MinIO数据持久化**：MinIO存储向量数据文件，如果容器重启后MinIO数据丢失，Milvus会报"segment not found"错误。务必使用持久化卷。
3. **磁盘空间不足**：Milvus的写前日志（WAL）和合并操作会占用临时磁盘空间。建议监控磁盘使用率，设置Docker的日志轮转。

### 8.7.3 端口冲突

Milvus单机版使用以下端口，部署前需确认不被占用：

| 端口 | 用途 | 冲突常见原因 |
|------|------|------------|
| 19530 | Milvus gRPC服务 | 其他Java/Go应用 |
| 9091 | Milvus HTTP健康检查 | 监控系统 |
| 2379 | etcd | 其他etcd实例 |
| 9000/9001 | MinIO API/Console | 其他对象存储 |

如果端口被占用，修改docker-compose.yml中的ports映射，例如将Milvus端口映射为19531：

```yaml
ports:
  - "19531:19530"
```

Python客户端连接时相应地修改URI：

```python
client = MilvusClient(uri="http://localhost:19531")
```

### 8.7.4 WSL2适配问题

Windows上使用WSL2运行Docker时，有几个常见问题需要处理：

- **WSL2内存限制**：WSL2默认使用宿主机50%的内存。可以在`%UserProfile%\.wslconfig`中设置：

```ini
[wsl2]
memory=8GB
processors=4
```

- **Docker Desktop自动停止**：WSL2空闲时可能被系统回收。在Docker Desktop设置中关闭"Reduce resource usage when idle"。

- **跨网络访问**：如果需要在局域网其他机器访问Windows上的Milvus，注意Docker容器默认在WSL2的虚拟网络中。需要使用`docker run --network host`或在Docker Desktop中配置端口转发。

### 8.7.5 MinIO鉴权失败

Milvus依赖MinIO存储向量数据文件。如果在生产环境中修改了MinIO的默认密码（minioadmin/minioadmin），需要在Milvus配置中同步更新：

```yaml
environment:
  MINIO_ADDRESS: minio:9000
  MINIO_ACCESS_KEY: your_new_access_key
  MINIO_SECRET_KEY: your_new_secret_key
```

如果MinIO鉴权失败，Milvus启动日志中会出现`InvalidAccessKeyId`错误，此时向量数据无法读写，检索会返回空结果。

## 8.8 本章小结

本章详细介绍了Milvus在四种典型环境中的部署方案：Windows Docker Desktop适合本地开发调试，Linux Docker Compose是测试环境的理想选择，Kubernetes集群部署面向生产高可用场景，Embed模式则提供了最轻量的单机运行方案。同时介绍了Attu这一可视化工具的使用方法，并总结了内存、磁盘、端口和WSL2等方面的常见部署陷阱。

选择哪种部署方式取决于RAG项目的具体阶段和规模。对于本书后续章节的Demo代码，推荐使用Embed模式或Docker Compose方式启动Milvus，这两种方式能够覆盖从开发到测试的完整流程。下一章将在此基础上，介绍Milvus的基础CRUD操作，为构建RAG知识库打下坚实的基础。
