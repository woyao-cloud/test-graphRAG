# 第18章: Milvus集群高可用与扩容

## 18.1 引言

当RAG系统承载企业级业务时，单机版Milvus已无法满足高并发、高可用和海量数据的需求。检索服务的中断意味着整个RAG问答系统的瘫痪，存储瓶颈则直接限制知识库的规模上限。Milvus集群架构通过副本机制、动态扩容和负载均衡三大支柱，为企业级RAG系统提供了坚实的高可用基础。本章将从集群副本机制入手，深入探讨动态扩容、负载均衡和容灾备份策略，帮助读者构建稳定可靠的Milvus生产集群。

## 18.2 集群副本机制

### 18.2.1 副本的作用

在Milvus集群中，副本（Replica）是指同一份数据在多个QueryNode上的冗余拷贝。副本机制的核心价值体现在以下三个方面：

- **高可用**：当某个QueryNode宕机时，其他持有相同数据副本的节点可以接管检索请求，服务不中断。
- **负载分担**：检索请求可以分发到多个副本上并行处理，提升整体检索吞吐量。
- **故障隔离**：单个节点的硬件故障不会导致数据丢失，数据在其他节点上仍然可用。

### 18.2.2 副本组配置

Milvus 2.3+引入了副本组（Replica Group）的概念，可以精细控制哪些QueryNode承载哪些集合的副本：

```python
from pymilvus import Collection, connections

class ReplicaGroupManager:
    """Milvus 副本组管理器"""
    
    def __init__(self, host: str = "localhost", port: int = 19530):
        self.host = host
        self.port = port
        connections.connect(host=host, port=port)
    
    def create_collection_with_replicas(
        self,
        collection_name: str,
        dim: int,
        replica_number: int = 3,
        resource_groups: list = None,
    ):
        """创建带副本的集合"""
        from pymilvus import CollectionSchema, FieldSchema, DataType
        
        schema = CollectionSchema([
            FieldSchema("id", DataType.INT64, is_primary=True),
            FieldSchema("vector", DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema("text", DataType.VARCHAR, max_length=65535),
        ])
        
        collection = Collection(
            name=collection_name,
            schema=schema,
            shards_num=2,  # 分片数
        )
        
        # 创建索引
        collection.create_index(
            field_name="vector",
            index_params={
                "metric_type": "IP",
                "index_type": "HNSW",
                "params": {"M": 16, "efConstruction": 200}
            }
        )
        
        # 设置副本数并加载
        collection.load(replica_number=replica_number)
        print(f"集合 '{collection_name}' 已创建，副本数: {replica_number}")
        
        return collection
    
    def get_replica_info(self, collection_name: str) -> dict:
        """查看集合的副本分布信息"""
        collection = Collection(collection_name)
        info = collection.describe()
        
        # 获取副本分布（需要Milvus 2.3+）
        replica_info = {
            "collection": collection_name,
            "shards_num": info.get("shards_num", 0),
            "replicas": []
        }
        
        # 实际生产环境中通过Milvus API获取更详细的副本信息
        return replica_info
    
    def add_replica(self, collection_name: str, node_id: int):
        """向现有集合添加副本节点"""
        collection = Collection(collection_name)
        # 通过资源组将指定节点加入副本集
        # 需要Milvus 2.3+ 的资源组功能
        print(f"已将节点 {node_id} 添加到 '{collection_name}' 的副本集")


# 使用示例
manager = ReplicaGroupManager()
manager.create_collection_with_replicas(
    collection_name="rag_production",
    dim=768,
    replica_number=3,
)
```

### 18.2.3 资源组与副本调度

Milvus 2.3+引入了资源组（Resource Group）功能，可以将QueryNode划分到不同资源组，实现硬件隔离和流量隔离：

```yaml
# resource-group-config.yaml
# 通过 Milvus API 配置资源组

# 创建两个资源组
resource_groups:
  group_a:
    nodes:
      - querynode-1
      - querynode-2
    requests:
      node_num: 2
    
  group_b:
    nodes:
      - querynode-3
      - querynode-4
    requests:
      node_num: 2

# 将集合绑定到指定资源组
collections:
  rag_knowledge_base:
    resource_groups: ["group_a"]
    replica_number: 2
  
  rag_logs:
    resource_groups: ["group_b"]
    replica_number: 1
```

资源组的应用场景：
- **业务隔离**：不同RAG知识库使用不同的资源组，互不影响。
- **性能保障**：核心业务独占高配置节点，非核心业务共享通用节点。
- **分级服务**：付费用户和免费用户使用不同资源组，实现服务质量差异化。

### 18.2.4 副本一致性

Milvus采用最终一致性模型。数据写入主副本后，异步同步到其他副本。在数据写入到同步完成的窗口期内，不同副本可能返回不同的检索结果。

```python
class ConsistencyManager:
    """副本一致性管理器"""
    
    CONSISTENCY_LEVELS = {
        "Strong": "Strong",         # 强一致性：读取最新写入
        "Bounded": "BoundedStaleness",  # 有界一致性：容忍有限延迟
        "Session": "Session",       # 会话一致性：同一连接读取到自己的写入
        "Eventual": "Eventual",     # 最终一致性：可能读到旧数据
    }
    
    @staticmethod
    def create_collection_with_consistency(
        client,
        collection_name: str,
        dim: int,
        consistency_level: str = "BoundedStaleness",
    ):
        """创建指定一致性级别的集合"""
        schema = client.create_schema()
        schema.add_field("id", "INT64", is_primary=True)
        schema.add_field("vector", "FLOAT_VECTOR", dim=dim)
        
        # Milvus 2.4+ 支持在创建集合时指定一致性级别
        client.create_collection(
            collection_name=collection_name,
            schema=schema,
            consistency_level=consistency_level,
        )
        print(f"集合已创建，一致性级别: {consistency_level}")
        
        return collection_name
    
    @staticmethod
    def get_consistency_recommendation(scenario: str) -> str:
        """根据场景推荐一致性级别"""
        recommendations = {
            "实时RAG问答": "BoundedStaleness",
            "批量知识库构建": "Eventual",
            "金融风控检索": "Strong",
            "日志检索": "Eventual",
            "电商商品搜索": "BoundedStaleness",
        }
        return recommendations.get(scenario, "BoundedStaleness")
```

## 18.3 动态扩容

### 18.3.1 扩容维度

Milvus集群支持三个维度的动态扩容，分别解决不同的瓶颈：

| 扩容维度 | 操作对象 | 解决瓶颈 | 操作方式 |
|---------|---------|---------|---------|
| 节点扩容 | QueryNode / DataNode | 计算能力不足 | 增加Pod副本数 |
| 分片扩容 | Collection Shard | 写入吞吐瓶颈 | 增加分片数 |
| 存储扩容 | MinIO / etcd | 存储容量不足 | 扩展存储卷或增加存储节点 |

### 18.3.2 节点扩容

节点扩容是最常用、最有效的扩容方式，可以线性提升检索吞吐量。

**Kubernetes环境中的HPA自动扩缩容**：

```yaml
# querynode-hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: milvus-querynode-hpa
  namespace: milvus
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: milvus-prod-querynode
  minReplicas: 3
  maxReplicas: 20
  metrics:
    # 基于CPU使用率扩容
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    # 基于内存使用率扩容
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
    # 基于QPS扩容（需要自定义metrics）
    - type: Pods
      pods:
        metric:
          name: milvus_querynode_search_qps
        target:
          type: AverageValue
          averageValue: 1000
```

**Docker Compose环境的节点扩容**：

```yaml
# docker-compose-cluster.yml 集群模式
services:
  # 多个QueryNode实例
  querynode-1:
    image: milvusdb/milvus:v2.4.0
    command: ["milvus", "run", "querynode"]
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
    depends_on:
      - etcd
      - minio
  
  querynode-2:
    image: milvusdb/milvus:v2.4.0
    command: ["milvus", "run", "querynode"]
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
    depends_on:
      - etcd
      - minio
  
  querynode-3:
    image: milvusdb/milvus:v2.4.0
    command: ["milvus", "run", "querynode"]
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
    depends_on:
      - etcd
      - minio
```

**扩容后自动注册**：新的QueryNode启动后会自动向etcd注册，Milvus集群会将数据段自动调度到新节点上，无需手动操作。

### 18.3.3 分片扩容

分片（Shard）是Milvus数据分布的最小单元。增加分片数可以提升写入吞吐量，因为数据可以并行写入到多个分片中。

```python
class ShardManager:
    """分片管理器"""
    
    @staticmethod
    def create_sharded_collection(
        client,
        collection_name: str,
        dim: int,
        initial_shards: int = 2,
    ):
        """创建分片集合"""
        schema = client.create_schema()
        schema.add_field("id", "INT64", is_primary=True)
        schema.add_field("vector", "FLOAT_VECTOR", dim=dim)
        schema.add_field("text", "VARCHAR", max_length=65535)
        
        client.create_collection(
            collection_name=collection_name,
            schema=schema,
            shards_num=initial_shards,  # 初始分片数
        )
        
        print(f"集合 '{collection_name}' 已创建，初始分片数: {initial_shards}")
        return collection_name
    
    @staticmethod
    def estimate_shards(query_nodes: int, expected_qps: float) -> int:
        """估算推荐的分片数"""
        # 经验公式：分片数 = max(2, queryNodes * 2, expectedQPS / 500)
        recommended = max(2, query_nodes * 2, int(expected_qps / 500))
        return min(recommended, 64)  # Milvus最多支持64个分片
    
    @staticmethod
    def calculate_shard_distribution(
        shards: int,
        query_nodes: int,
    ) -> list:
        """计算分片在各QueryNode上的分布"""
        distribution = []
        for shard_id in range(shards):
            node_id = shard_id % query_nodes
            distribution.append({
                "shard_id": shard_id,
                "assigned_node": f"querynode-{node_id + 1}",
            })
        return distribution


# 估算推荐分片数
shard_count = ShardManager.estimate_shards(
    query_nodes=5,
    expected_qps=5000,
)
print(f"推荐分片数: {shard_count}")

# 查看分片分布
distribution = ShardManager.calculate_shard_distribution(8, 3)
for entry in distribution:
    print(f"  分片 {entry['shard_id']} -> {entry['assigned_node']}")
```

**分片扩容的最佳实践**：
- 初始分片数不宜过少（至少2个），否则无法利用多节点的并行能力。
- 分片数建议设为QueryNode数量的2倍，确保负载均匀分布。
- 分片数一旦设置，后续无法减少。如果后期需要扩容，建议创建新集合并迁移数据。
- 单个分片的数据量建议控制在200GB以内，超过此阈值应考虑增加分片。

### 18.3.4 存储扩容

Milvus的底层存储由etcd（元数据）和MinIO/S3（数据文件）提供。存储扩容方式取决于部署环境：

```yaml
# K8s环境：PVC自动扩容
persistence:
  enabled: true
  storageClass: "managed-premium"
  size: "500Gi"
  
  # 使用自动扩容的StorageClass
  # 云服务商通常提供自动扩容选项
  # Azure: managed-premium + autoGrow
  # AWS: gp3 + elastic
  # GCP: pd-ssd + autoResize

# 或者使用多个存储卷
extraVolumes:
  - name: milvus-data-extra
    persistentVolumeClaim:
      claimName: milvus-data-pvc-2
```

**MinIO集群模式**：生产环境中，MinIO本身也应以集群模式部署，避免单点故障：

```yaml
# minio-cluster.yml
services:
  minio-1:
    image: minio/minio:latest
    command: minio server --console-address ":9001" http://minio-{1...4}/data
    volumes:
      - minio-data-1:/data
    
  minio-2:
    image: minio/minio:latest
    command: minio server --console-address ":9001" http://minio-{1...4}/data
    volumes:
      - minio-data-2:/data
  
  minio-3:
    image: minio/minio:latest
    command: minio server --console-address ":9001" http://minio-{1...4}/data
    volumes:
      - minio-data-3:/data
  
  minio-4:
    image: minio/minio:latest
    command: minio server --console-address ":9001" http://minio-{1...4}/data
    volumes:
      - minio-data-4:/data
```

## 18.4 负载均衡

### 18.4.1 Milvus Proxy的负载均衡

Milvus的Proxy组件本身就是一个负载均衡器。在生产集群中，多个Proxy实例前端需要额外的负载均衡器（如Nginx、HAProxy或K8s Service）来分发客户端请求。

**使用Nginx作为Milvus Proxy的负载均衡器**：

```nginx
# nginx-milvus.conf
upstream milvus_proxy {
    # 使用最少连接算法
    least_conn;
    
    # 多个Proxy实例
    server proxy-1:19530 max_fails=3 fail_timeout=30s;
    server proxy-2:19530 max_fails=3 fail_timeout=30s;
    server proxy-3:19530 max_fails=3 fail_timeout=30s;
    
    # 保持长连接
    keepalive 32;
}

server {
    listen 19530 http2;
    
    location / {
        grpc_pass grpc://milvus_proxy;
        grpc_set_header Host $host;
        grpc_set_header X-Real-IP $remote_addr;
        
        # gRPC超时配置
        grpc_connect_timeout 10s;
        grpc_read_timeout 60s;
        grpc_send_timeout 60s;
    }
}

server {
    listen 9091;
    
    location / {
        proxy_pass http://milvus_proxy;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

**使用HAProxy的配置**：

```haproxy
# haproxy.cfg
global
    maxconn 4096

defaults
    mode tcp
    timeout connect 10s
    timeout client 60s
    timeout server 60s

frontend milvus_frontend
    bind *:19530
    default_backend milvus_backend

backend milvus_backend
    # 最少连接 + 权重
    balance leastconn
    
    server proxy-1 proxy-1:19530 weight 10 check inter 10s fall 3 rise 2
    server proxy-2 proxy-2:19530 weight 10 check inter 10s fall 3 rise 2
    server proxy-3 proxy-3:19530 weight 10 check inter 10s fall 3 rise 2
    
    # 健康检查
    option tcp-check
    tcp-check expect string OK
```

### 18.4.2 客户端侧负载均衡

对于Python客户端，可以在应用层实现简单的负载均衡：

```python
import random
from pymilvus import MilvusClient

class MilvusLoadBalancer:
    """Milvus 客户端负载均衡器"""
    
    def __init__(self, proxy_addresses: list):
        """
        Args:
            proxy_addresses: Proxy地址列表，如 ["http://proxy-1:19530", "http://proxy-2:19530"]
        """
        self.proxy_addresses = proxy_addresses
        self._current_index = 0
        self._clients = {}
        
        # 初始化所有连接
        for addr in proxy_addresses:
            self._clients[addr] = MilvusClient(uri=addr)
    
    def get_client(self, strategy: str = "round_robin") -> MilvusClient:
        """根据策略获取客户端连接"""
        if strategy == "round_robin":
            addr = self._round_robin()
        elif strategy == "random":
            addr = random.choice(self.proxy_addresses)
        else:
            raise ValueError(f"不支持的策略: {strategy}")
        
        return self._clients[addr]
    
    def _round_robin(self) -> str:
        addr = self.proxy_addresses[self._current_index]
        self._current_index = (self._current_index + 1) % len(self.proxy_addresses)
        return addr
    
    def search(self, collection_name: str, query_vector: list, top_k: int = 10):
        """带负载均衡的检索"""
        client = self.get_client(strategy="round_robin")
        return client.search(
            collection_name=collection_name,
            data=[query_vector],
            limit=top_k,
        )


# 使用示例
lb = MilvusLoadBalancer([
    "http://milvus-proxy-1.internal:19530",
    "http://milvus-proxy-2.internal:19530",
    "http://milvus-proxy-3.internal:19530",
])

# 检索请求自动分发到不同Proxy
results = lb.search("rag_knowledge_base", query_vector)
```

### 18.4.3 智能路由策略

在RAG生产环境中，可以根据请求特征实现更精细的智能路由：

```python
class SmartRouter:
    """智能路由策略"""
    
    def __init__(self, cluster_nodes: dict):
        """
        Args:
            cluster_nodes: 节点分组
                {
                    "high_perf": ["http://node-h1:19530", "http://node-h2:19530"],
                    "standard": ["http://node-s1:19530", "http://node-s2:19530"],
                    "batch": ["http://node-b1:19530"],
                }
        """
        self.nodes = cluster_nodes
        self.load_balancers = {
            group: MilvusLoadBalancer(addrs)
            for group, addrs in cluster_nodes.items()
        }
    
    def route_request(self, request_type: str, priority: str = "normal"):
        """根据请求类型和优先级路由"""
        
        if priority == "high":
            # 高优先级请求路由到高性能节点
            target_group = "high_perf"
        elif request_type == "batch":
            # 批量请求路由到批处理节点
            target_group = "batch"
        else:
            # 普通请求路由到标准节点
            target_group = "standard"
        
        lb = self.load_balancers[target_group]
        return lb.get_client(strategy="round_robin")
    
    def search(self, query_vector, top_k=10, priority="normal"):
        """执行带优先级的检索"""
        client = self.route_request("search", priority)
        return client.search(...)


# 使用示例
router = SmartRouter({
    "high_perf": ["http://node-h1:19530", "http://node-h2:19530"],
    "standard": ["http://node-s1:19530", "http://node-s2:19530"],
    "batch": ["http://node-b1:19530"],
})

# 普通用户请求
normal_results = router.search(query_vector, priority="normal")

# VIP用户请求，路由到高性能节点
vip_results = router.search(query_vector, priority="high")
```

## 18.5 容灾备份与异地多活

### 18.5.1 容灾策略层次

| 容灾级别 | 恢复时间目标（RTO） | 恢复点目标（RPO） | 适用场景 |
|---------|-------------------|-------------------|---------|
| 单节点容灾 | < 1分钟 | < 1秒 | 单机房内节点故障 |
| 同城容灾 | < 15分钟 | < 5分钟 | 单机房整体故障 |
| 异地容灾 | < 1小时 | < 15分钟 | 区域性灾难 |
| 异地多活 | < 1秒 | < 1秒 | 全球多地域服务 |

### 18.5.2 同城容灾方案

同城容灾通过在同一城市的不同可用区部署Milvus集群，实现机房级别的故障切换：

```yaml
# 同城双活架构
services:
  # 主集群（可用区A）
  milvus-primary:
    image: milvusdb/milvus:v2.4.0
    environment:
      ETCD_ENDPOINTS: etcd-primary:2379
      MINIO_ADDRESS: minio-primary:9000
  
  # 备集群（可用区B）
  milvus-secondary:
    image: milvusdb/milvus:v2.4.0
    environment:
      ETCD_ENDPOINTS: etcd-secondary:2379
      MINIO_ADDRESS: minio-secondary:9000
  
  # MinIO跨可用区同步
  minio-primary:
    image: minio/minio:latest
    command: minio server /data
    volumes:
      - minio-primary-data:/data
  
  minio-secondary:
    image: minio/minio:latest
    command: minio server /data
    volumes:
      - minio-secondary-data:/data
```

### 18.5.3 异地多活架构

异地多活是最高级别的容灾方案，要求多个地域的Milvus集群同时提供读写服务，数据实时同步：

```python
class MultiActiveManager:
    """异地多活管理器"""
    
    def __init__(self, regions: dict):
        """
        Args:
            regions: 地域集群配置
                {
                    "beijing": {"uri": "http://bj.milvus:19530", "weight": 50},
                    "shanghai": {"uri": "http://sh.milvus:19530", "weight": 30},
                    "shenzhen": {"uri": "http://sz.milvus:19530", "weight": 20},
                }
        """
        self.regions = regions
        self.clients = {
            name: MilvusClient(uri=cfg["uri"])
            for name, cfg in regions.items()
        }
    
    def get_nearest_client(self, user_region: str = None) -> MilvusClient:
        """根据用户地域获取最近的客户端"""
        if user_region and user_region in self.clients:
            return self.clients[user_region]
        
        # 按权重随机选择
        total_weight = sum(cfg["weight"] for cfg in self.regions.values())
        r = random.uniform(0, total_weight)
        cumulative = 0
        for name, cfg in self.regions.items():
            cumulative += cfg["weight"]
            if r <= cumulative:
                return self.clients[name]
        
        return list(self.clients.values())[0]
    
    def write_all_regions(self, collection_name: str, data: list):
        """跨地域写入（同步或异步）"""
        results = {}
        for region_name, client in self.clients.items():
            try:
                result = client.insert(collection_name, data)
                results[region_name] = {"status": "success", "count": len(data)}
            except Exception as e:
                results[region_name] = {"status": "failed", "error": str(e)}
        
        return results
    
    def read_from_region(self, collection_name: str, query_vector: list,
                         region: str = None, top_k: int = 10):
        """从指定地域（或最近地域）读取"""
        client = self.get_nearest_client(region)
        return client.search(
            collection_name=collection_name,
            data=[query_vector],
            limit=top_k,
        )
    
    def health_check_all(self) -> dict:
        """检查所有地域集群的健康状态"""
        status = {}
        for region_name, client in self.clients.items():
            try:
                version = client.get_server_version()
                status[region_name] = {"status": "healthy", "version": version}
            except Exception as e:
                status[region_name] = {"status": "unhealthy", "error": str(e)}
        return status


# 使用示例
multi_active = MultiActiveManager({
    "beijing": {"uri": "http://10.0.1.10:19530", "weight": 50},
    "shanghai": {"uri": "http://10.0.2.10:19530", "weight": 30},
    "shenzhen": {"uri": "http://10.0.3.10:19530", "weight": 20},
})

# 跨地域健康检查
health = multi_active.health_check_all()
for region, status in health.items():
    print(f"{region}: {status['status']}")

# 根据用户地域就近读取
results = multi_active.read_from_region(
    "rag_knowledge_base",
    query_vector,
    region="beijing",
)
```

### 18.5.4 跨地域数据同步方案

异地多活的核心挑战是数据同步。Milvus本身不提供跨地域数据同步功能，需要通过外部工具实现：

```python
class CrossRegionSync:
    """跨地域数据同步工具"""
    
    def __init__(self, source_client: MilvusClient, target_client: MilvusClient):
        self.source = source_client
        self.target = target_client
    
    def sync_collection(self, collection_name: str, batch_size: int = 1000):
        """同步集合数据从源集群到目标集群"""
        # 1. 检查目标集群是否已有该集合
        if not self.target.has_collection(collection_name):
            # 复制Schema
            desc = self.source.describe_collection(collection_name)
            schema = MilvusClient.create_schema(
                auto_id=desc.get("auto_id", False),
                enable_dynamic_field=desc.get("enable_dynamic_field", False),
            )
            for field in desc.get("fields", []):
                schema.add_field(
                    field["name"],
                    field["type"],
                    dim=field.get("dim"),
                    max_length=field.get("max_length"),
                    is_primary=field.get("is_primary", False),
                )
            
            self.target.create_collection(
                collection_name=collection_name,
                schema=schema,
            )
        
        # 2. 分批同步数据
        offset = 0
        total_synced = 0
        while True:
            batch = self.source.query(
                collection_name=collection_name,
                output_fields=["*"],
                limit=batch_size,
                offset=offset,
            )
            if not batch:
                break
            
            self.target.insert(collection_name, batch)
            total_synced += len(batch)
            offset += len(batch)
            print(f"  已同步 {total_synced} 条记录...")
        
        print(f"同步完成，共 {total_synced} 条记录")
        return total_synced
    
    def start_realtime_sync(self, collection_name: str, poll_interval: int = 60):
        """启动实时增量同步（基于轮询）"""
        import time
        last_count = self._get_entity_count(self.source, collection_name)
        
        while True:
            time.sleep(poll_interval)
            current_count = self._get_entity_count(self.source, collection_name)
            
            if current_count > last_count:
                # 有增量数据，执行增量同步
                print(f"检测到 {current_count - last_count} 条增量数据")
                self.sync_incremental(
                    collection_name,
                    offset=last_count,
                    limit=current_count - last_count,
                )
                last_count = current_count
    
    def _get_entity_count(self, client: MilvusClient, collection_name: str) -> int:
        count = client.query(collection_name, output_fields=["count(*)"])
        return count[0]["count(*)"] if count else 0
    
    def sync_incremental(self, collection_name: str, offset: int, limit: int):
        """增量同步"""
        data = self.source.query(
            collection_name=collection_name,
            output_fields=["*"],
            limit=limit,
            offset=offset,
        )
        if data:
            self.target.insert(collection_name, data)
            print(f"增量同步 {len(data)} 条记录")
```

### 18.5.5 自动故障切换

实现客户端级别的自动故障切换：

```python
class FailoverClient:
    """自动故障切换客户端"""
    
    def __init__(self, primary_uri: str, standby_uri: str, 
                 health_check_interval: int = 10):
        self.primary_uri = primary_uri
        self.standby_uri = standby_uri
        self.current_uri = primary_uri
        self.is_primary_active = True
        
        # 启动后台健康检查
        self._start_health_check(health_check_interval)
    
    def _start_health_check(self, interval: int):
        """启动后台健康检查线程"""
        import threading
        
        def check():
            while True:
                time.sleep(interval)
                primary_healthy = self._check_health(self.primary_uri)
                standby_healthy = self._check_health(self.standby_uri)
                
                if primary_healthy:
                    if not self.is_primary_active:
                        print("主集群已恢复，切换回主集群")
                        self.current_uri = self.primary_uri
                        self.is_primary_active = True
                else:
                    if self.is_primary_active and standby_healthy:
                        print("主集群不可用，切换到备集群")
                        self.current_uri = self.standby_uri
                        self.is_primary_active = False
        
        thread = threading.Thread(target=check, daemon=True)
        thread.start()
    
    def _check_health(self, uri: str) -> bool:
        try:
            client = MilvusClient(uri=uri, timeout=5)
            client.get_server_version()
            return True
        except:
            return False
    
    def get_client(self) -> MilvusClient:
        """获取当前活动的客户端"""
        return MilvusClient(uri=self.current_uri)
    
    def search(self, collection_name: str, query_vector: list, top_k: int = 10):
        """执行检索，支持自动重试"""
        try:
            client = self.get_client()
            return client.search(
                collection_name=collection_name,
                data=[query_vector],
                limit=top_k,
            )
        except Exception as e:
            if not self.is_primary_active:
                # 如果备集群也失败，尝试主集群
                try:
                    client = MilvusClient(uri=self.primary_uri, timeout=5)
                    return client.search(
                        collection_name=collection_name,
                        data=[query_vector],
                        limit=top_k,
                    )
                except:
                    pass
            raise Exception(f"所有集群均不可用: {e}")


# 使用示例
failover_client = FailoverClient(
    primary_uri="http://milvus-primary.internal:19530",
    standby_uri="http://milvus-standby.internal:19530",
    health_check_interval=10,
)

# 正常使用时无需关心主备切换
results = failover_client.search("rag_knowledge_base", query_vector)
```

## 18.6 生产集群部署清单

### 18.6.1 集群部署检查清单

在Milvus集群上线前，逐项检查以下配置：

| 检查项 | 要求 | 验证方式 |
|-------|------|---------|
| QueryNode副本数 | >= 3 | `kubectl get pods | grep querynode` |
| DataNode副本数 | >= 2 | `kubectl get pods | grep datanode` |
| IndexNode副本数 | >= 2 | `kubectl get pods | grep indexnode` |
| Proxy副本数 | >= 2 | `kubectl get pods | grep proxy` |
| etcd集群 | 3节点 | `docker exec etcd etcdctl member list` |
| MinIO集群 | 4节点（erasure code） | `docker exec minio mc admin info` |
| 存储持久化 | PVC配置 | `kubectl get pvc` |
| 资源限制 | CPU/Memory requests/limits | `kubectl describe pod` |
| HPA自动扩缩容 | 已配置 | `kubectl get hpa` |
| 监控告警 | Prometheus+Grafana | 验证Dashboard数据 |
| 备份策略 | 定时备份脚本 | 执行一次备份验证 |
| 故障切换 | 演练通过 | 手动停止主节点验证 |

### 18.6.2 生产集群性能基准

在不同集群规模下，预期的性能指标参考：

| 集群规模 | QueryNode数 | 数据量 | 预期QPS | P99延迟 |
|---------|------------|--------|---------|--------|
| 小型 | 3 | 100万x768维 | 1000 | < 50ms |
| 中型 | 6 | 1000万x768维 | 5000 | < 80ms |
| 大型 | 12 | 1亿x768维 | 20000 | < 150ms |
| 超大型 | 24+ | 10亿x768维 | 50000+ | < 200ms |

## 18.7 本章小结

本章系统性地介绍了Milvus集群高可用与扩容的核心技术和最佳实践。以下是要点总结：

1. **副本机制是高可用的基石**：通过设置多副本（推荐3副本以上），实现QueryNode级别的故障容错和检索负载分担。资源组功能可以实现业务隔离和硬件隔离。

2. **动态扩容是应对业务增长的保障**：节点扩容（增加QueryNode/DataNode）可以线性提升吞吐能力，分片扩容解决写入瓶颈，存储扩容支撑数据量增长。Kubernetes HPA可以实现基于CPU/内存/QPS的自动扩缩容。

3. **负载均衡是性能优化的关键**：多Proxy实例配合Nginx/HAProxy负载均衡器，可以实现请求的均匀分发。智能路由策略可以根据请求优先级和用户地域，将请求路由到最合适的节点。

4. **容灾备份是数据安全的最后防线**：同城容灾应对机房故障，异地多活应对区域性灾难。自动故障切换客户端可以实现主备集群的无缝切换，保障RAG服务的持续可用。

5. **从单机到集群的演进路径**：RAG系统从原型到生产的典型演进路径为：Embed模式 -> Docker单机 -> 多节点集群 -> 多副本高可用集群 -> 异地多活集群。每个阶段的选择取决于业务规模和服务等级要求。
