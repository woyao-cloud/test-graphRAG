# 第17章: Milvus监控、告警与问题排查

## 17.1 引言

RAG系统进入生产环境后，监控告警体系是保障服务稳定性的基石。没有有效的监控，运维人员就像在黑暗中驾驶——无法感知系统是否健康、无法预知风险、无法快速定位故障。Milvus原生集成了Prometheus指标暴露接口，配合Grafana可以搭建完整的可视化监控面板。本章将从Prometheus+Grafana监控面板搭建开始，系统性地介绍Milvus的核心监控指标，并针对RAG场景中常见的故障给出排查方案和典型报错的解决方法。本章参考代码位于`demos/ch17-monitoring/main.py`。

## 17.2 Prometheus+Grafana监控面板搭建

### 17.2.1 监控架构总览

Milvus的监控架构分为三层：数据采集层（Milvus metrics endpoint）、指标存储层（Prometheus）、可视化层（Grafana）。

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Milvus        │     │   Prometheus    │     │    Grafana      │
│   :9091/metrics │────▶│   :9090         │────▶│   :3000         │
│   (指标暴露)     │     │   (指标存储)     │     │   (可视化面板)   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

Milvus在每个节点的9091端口暴露Prometheus格式的指标数据。Prometheus定期拉取这些指标并存储到时间序列数据库中。Grafana从Prometheus读取数据，渲染为可视化的监控面板。

### 17.2.2 配置Prometheus

创建Prometheus配置文件，定义Milvus的抓取目标：

```yaml
# prometheus.yml
global:
  scrape_interval: 15s       # 抓取间隔
  evaluation_interval: 15s   # 规则评估间隔

scrape_configs:
  # Milvus 各节点指标
  - job_name: 'milvus'
    static_configs:
      - targets:
        - 'milvus-standalone:9091'   # Milvus standalone
        # 集群模式下还需要抓取各组件
        # - 'milvus-proxy:9091'
        # - 'milvus-querynode:9091'
        # - 'milvus-datanode:9091'
        # - 'milvus-indexnode:9091'
    metrics_path: '/metrics'
    scheme: 'http'

  # 系统资源指标（可选，通过node_exporter采集）
  - job_name: 'node'
    static_configs:
      - targets:
        - 'node-exporter:9100'

# 告警规则文件
rule_files:
  - 'alerts/*.yml'
```

### 17.2.3 Docker Compose集成部署

在docker-compose.yml中添加Prometheus和Grafana服务：

```yaml
version: '3.8'

services:
  etcd:
    # ... (与之前配置相同)

  minio:
    # ... (与之前配置相同)

  milvus:
    # ... (与之前配置相同)

  # ---------- 监控栈 ----------
  prometheus:
    image: prom/prometheus:v2.51.0
    container_name: milvus-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
      - ./prometheus/alerts:/etc/prometheus/alerts
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.retention.time=30d'
      - '--storage.tsdb.retention.size=50GB'
    networks:
      - milvus-net

  grafana:
    image: grafana/grafana:10.4.0
    container_name: milvus-grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_USER=admin
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_INSTALL_PLUGINS=grafana-piechart-panel
    volumes:
      - ./grafana/dashboards:/var/lib/grafana/dashboards
      - ./grafana/datasources:/etc/grafana/provisioning/datasources
      - grafana_data:/var/lib/grafana
    depends_on:
      - prometheus
    networks:
      - milvus-net

volumes:
  etcd_data:
  minio_data:
  milvus_data:
  prometheus_data:
  grafana_data:

networks:
  milvus-net:
    name: milvus-net
```

### 17.2.4 配置Grafana数据源

使用Grafana的provisioning功能自动配置Prometheus数据源：

```yaml
# grafana/datasources/datasource.yml
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: false
```

### 17.2.5 导入Milvus官方Dashboard

Milvus官方提供了完整的Grafana Dashboard JSON模板，可以从以下地址获取：

```bash
# 下载Milvus官方Dashboard
wget https://raw.githubusercontent.com/milvus-io/milvus/master/deployments/monitor/grafana/milvus-dashboard.json

# 移动到Grafana的dashboards目录
cp milvus-dashboard.json ./grafana/dashboards/
```

配置Grafana自动加载Dashboard：

```yaml
# grafana/dashboards/dashboard.yml
apiVersion: 1

providers:
  - name: 'Milvus'
    orgId: 1
    folder: 'Milvus'
    type: file
    disableDeletion: false
    editable: true
    options:
      path: /var/lib/grafana/dashboards
```

### 17.2.6 验证监控部署

启动所有服务后，通过以下方式验证监控栈是否正常工作：

```bash
# 启动所有服务
docker compose up -d

# 验证Milvus指标端点
curl http://localhost:9091/metrics | head -20

# 验证Prometheus是否正常抓取
curl http://localhost:9090/api/v1/targets

# 访问Grafana
# 浏览器打开 http://localhost:3000 (默认账号: admin/admin)
```

## 17.3 核心监控指标

### 17.3.1 检索性能指标

| 指标名 | 类型 | 含义 | 告警阈值 |
|-------|------|------|---------|
| `milvus_proxy_search_requests_count` | Counter | 检索请求总数 | 监控趋势 |
| `milvus_proxy_search_requests_duration_ms` | Histogram | 检索请求延迟分布 | P99 > 500ms |
| `milvus_querynode_search_qps` | Gauge | 每秒查询数 | 低于预期值 |
| `milvus_querynode_search_latency_ms` | Histogram | QueryNode检索延迟 | avg > 200ms |
| `milvus_proxy_search_requests_fail_count` | Counter | 检索失败次数 | > 0 |

使用Python程序从Milvus HTTP指标端点采集数据：

```python
import requests
import time
from collections import defaultdict

class MetricsCollector:
    """Milvus 指标采集器"""
    
    def __init__(self, metrics_url: str = "http://localhost:9091/metrics"):
        self.metrics_url = metrics_url
        self.metrics_cache = defaultdict(float)
    
    def fetch_metrics(self) -> dict:
        """从Milvus HTTP端点获取指标"""
        try:
            resp = requests.get(self.metrics_url, timeout=5)
            if resp.status_code != 200:
                return {"error": f"HTTP {resp.status_code}"}
            
            metrics = {}
            for line in resp.text.split("\n"):
                line = line.strip()
                if line.startswith("#") or not line:
                    continue
                if " " in line:
                    # 解析 "milvus_xxx{...} value" 格式
                    parts = line.rsplit(" ", 1)
                    if len(parts) == 2:
                        name = parts[0].split("{")[0]
                        try:
                            metrics[name] = float(parts[1])
                        except ValueError:
                            pass
            
            return metrics
        
        except requests.RequestException as e:
            return {"error": str(e)}
    
    def calculate_qps(self, interval: int = 10) -> float:
        """计算每秒查询数"""
        m1 = self.fetch_metrics()
        search_count_1 = m1.get("milvus_proxy_search_requests_count", 0)
        
        time.sleep(interval)
        
        m2 = self.fetch_metrics()
        search_count_2 = m2.get("milvus_proxy_search_requests_count", 0)
        
        qps = (search_count_2 - search_count_1) / interval
        return qps
    
    def get_latency_stats(self) -> dict:
        """获取延迟统计"""
        metrics = self.fetch_metrics()
        
        # 解析延迟直方图指标
        latency_metrics = {
            k: v for k, v in metrics.items()
            if "search_requests_duration" in k
        }
        
        return latency_metrics


# 使用示例
collector = MetricsCollector()
qps = collector.calculate_qps(interval=5)
print(f"当前 QPS: {qps:.2f}")
```

### 17.3.2 资源使用指标

| 指标名 | 类型 | 含义 | 告警阈值 |
|-------|------|------|---------|
| `milvus_querynode_memory_usage_bytes` | Gauge | QueryNode内存使用 | > 内存上限的85% |
| `milvus_datanode_memory_usage_bytes` | Gauge | DataNode内存使用 | > 内存上限的85% |
| `milvus_proxy_cpu_usage_percent` | Gauge | Proxy CPU使用率 | > 80% |
| `process_resident_memory_bytes` | Gauge | 进程常驻内存 | 监控趋势 |
| `milvus_storage_disk_usage_bytes` | Gauge | 磁盘使用量 | > 80% |

### 17.3.3 索引与存储指标

| 指标名 | 类型 | 含义 | 告警阈值 |
|-------|------|------|---------|
| `milvus_querynode_num_entities` | Gauge | 已加载的实体总数 | 监控趋势 |
| `milvus_querynode_collection_num` | Gauge | 已加载的集合数 | 监控趋势 |
| `milvus_datanode_flush_duration_seconds` | Histogram | 数据刷盘耗时 | avg > 30s |
| `milvus_indexnode_index_build_duration_seconds` | Histogram | 索引构建耗时 | 监控趋势 |

### 17.3.4 连接与请求指标

| 指标名 | 类型 | 含义 | 告警阈值 |
|-------|------|------|---------|
| `milvus_proxy_connected_num` | Gauge | 当前客户端连接数 | 超过连接池上限 |
| `milvus_proxy_connection_request_duration` | Histogram | 连接建立耗时 | > 5s |
| `milvus_proxy_req_total` | Counter | 请求总数 | 监控趋势 |
| `milvus_proxy_req_fail` | Counter | 请求失败数 | > 0 |

### 17.3.5 综合监控报告

参考`demos/ch17-monitoring/main.py`的代码模式，实现一个完整的监控报告生成器：

```python
"""
ch17-monitoring: 监控报告生成器
参考: demos/ch17-monitoring/main.py
"""

import json
import time
import requests
from datetime import datetime
from pymilvus import MilvusClient

class MilvusMonitorReport:
    """Milvus 综合监控报告生成器"""
    
    def __init__(self, milvus_uri: str = "http://localhost:19530",
                 metrics_uri: str = "http://localhost:9091/metrics"):
        self.milvus_uri = milvus_uri
        self.metrics_uri = metrics_uri
        self.client = MilvusClient(uri=milvus_uri)
    
    def collect_server_info(self) -> dict:
        """采集服务器基本信息"""
        info = {}
        try:
            info["version"] = self.client.get_server_version()
        except Exception as e:
            info["error"] = str(e)
        return info
    
    def collect_collection_stats(self) -> list:
        """采集所有集合的统计信息"""
        collections = self.client.list_collections()
        stats = []
        
        for col_name in collections:
            try:
                desc = self.client.describe_collection(col_name)
                count = self.client.query(
                    collection_name=col_name,
                    output_fields=["count(*)"]
                )
                row_count = count[0]["count(*)"] if count else 0
                
                stats.append({
                    "name": col_name,
                    "dimension": desc.get("dim", "N/A"),
                    "entity_count": row_count,
                    "index_status": desc.get("index_status", "N/A"),
                    "auto_id": desc.get("auto_id", False),
                })
            except Exception as e:
                stats.append({"name": col_name, "error": str(e)})
        
        return stats
    
    def collect_performance_metrics(self) -> dict:
        """采集性能指标"""
        metrics = {}
        try:
            resp = requests.get(self.metrics_uri, timeout=5)
            if resp.status_code == 200:
                for line in resp.text.split("\n"):
                    line = line.strip()
                    if line.startswith("#") or not line:
                        continue
                    if " " in line:
                        parts = line.rsplit(" ", 1)
                        if len(parts) == 2:
                            name = parts[0].split("{")[0]
                            try:
                                metrics[name] = float(parts[1])
                            except ValueError:
                                pass
        except Exception as e:
            metrics["error"] = str(e)
        
        return metrics
    
    def health_check(self) -> dict:
        """执行健康检查"""
        status = {"timestamp": datetime.now().isoformat()}
        
        # 连接检查
        try:
            self.client.get_server_version()
            status["connection"] = "OK"
        except Exception as e:
            status["connection"] = f"FAIL: {e}"
        
        # HTTP检查
        try:
            health_url = self.milvus_uri.replace(":19530", ":9091") + "/health"
            resp = requests.get(health_url, timeout=5)
            status["http_health"] = "OK" if resp.status_code == 200 else f"HTTP {resp.status_code}"
        except Exception as e:
            status["http_health"] = f"FAIL: {e}"
        
        # 读写检查
        try:
            collections = self.client.list_collections()
            status["read_check"] = f"OK ({len(collections)} collections)"
        except Exception as e:
            status["read_check"] = f"FAIL: {e}"
        
        return status
    
    def generate_report(self) -> str:
        """生成完整监控报告"""
        lines = []
        lines.append("=" * 60)
        lines.append("Milvus 监控报告")
        lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 60)
        
        # 服务器信息
        lines.append("\n[服务器信息]")
        server_info = self.collect_server_info()
        for k, v in server_info.items():
            lines.append(f"  {k}: {v}")
        
        # 健康检查
        lines.append("\n[健康检查]")
        health = self.health_check()
        for k, v in health.items():
            if k == "timestamp":
                continue
            status_symbol = "✓" if v.startswith("OK") else "✗"
            lines.append(f"  {status_symbol} {k}: {v}")
        
        # 集合统计
        lines.append("\n[集合统计]")
        col_stats = self.collect_collection_stats()
        for col in col_stats:
            lines.append(f"  集合: {col.get('name', 'N/A')}")
            lines.append(f"    维度: {col.get('dimension', 'N/A')}")
            lines.append(f"    实体数: {col.get('entity_count', 'N/A')}")
            lines.append(f"    索引状态: {col.get('index_status', 'N/A')}")
        
        # 性能指标摘要
        lines.append("\n[性能指标]")
        perf = self.collect_performance_metrics()
        if perf.get("milvus_proxy_search_requests_count"):
            lines.append(f"  总检索请求: {perf['milvus_proxy_search_requests_count']:.0f}")
        if perf.get("milvus_querynode_num_entities"):
            lines.append(f"  已加载实体: {perf['milvus_querynode_num_entities']:.0f}")
        if perf.get("process_resident_memory_bytes"):
            mem_mb = perf["process_resident_memory_bytes"] / 1024 / 1024
            lines.append(f"  进程内存: {mem_mb:.2f} MB")
        if perf.get("milvus_proxy_connected_num"):
            lines.append(f"  当前连接数: {perf['milvus_proxy_connected_num']:.0f}")
        
        lines.append("\n" + "=" * 60)
        return "\n".join(lines)
```

## 17.4 RAG场景常见故障排查

### 17.4.1 检索结果为空

这是RAG场景中最常见的故障，表现为向量检索返回零条结果。

**排查步骤**：

1. **检查数据是否已写入**：使用Attu或Python SDK查看集合中的实体数量。
2. **检查索引状态**：确认索引已创建并加载到内存。
3. **检查过滤条件**：标量过滤条件可能过于严格，导致无匹配结果。
4. **检查向量维度**：查询向量的维度是否与集合定义一致。

```python
def diagnose_empty_search(client, collection_name: str):
    """诊断检索为空的问题"""
    print(f"诊断集合 '{collection_name}' 检索为空问题...")
    
    # 1. 检查集合是否存在
    if not client.has_collection(collection_name):
        print("  ✗ 集合不存在")
        return
    
    # 2. 检查数据量
    count = client.query(collection_name, output_fields=["count(*)"])
    row_count = count[0]["count(*)"] if count else 0
    print(f"  - 实体数量: {row_count}")
    if row_count == 0:
        print("  ✗ 集合中没有数据，需要先写入数据")
        return
    
    # 3. 检查索引
    desc = client.describe_collection(collection_name)
    print(f"  - 向量维度: {desc.get('dim', 'N/A')}")
    print(f"  - 索引状态: {desc.get('index_status', 'N/A')}")
    
    # 4. 测试检索
    dim = desc.get("dim", 768)
    try:
        results = client.search(
            collection_name=collection_name,
            data=[[0.1] * dim],
            limit=3,
        )
        if results[0]:
            print(f"  ✓ 检索正常，返回 {len(results[0])} 条结果")
        else:
            print("  ✗ 检索返回空结果")
    except Exception as e:
        print(f"  ✗ 检索异常: {e}")
```

### 17.4.2 检索延迟过高

RAG系统的问答响应时间主要取决于检索延迟和LLM生成时间。当检索延迟过高时，需要从以下方面排查：

**常见原因及解决方案**：

| 原因 | 现象 | 解决方案 |
|------|------|---------|
| 未建索引 | 数据量>10万但使用FLAT | 创建IVF或HNSW索引 |
| nprobe过大 | IVF检索参数设置不当 | 调小nprobe（设为TopK的2-3倍） |
| 内存不足 | 数据未完全加载到内存 | 增加QueryNode内存或使用mmap |
| 并发过高 | QPS超过节点处理能力 | 扩容QueryNode或启用限流 |
| 网络延迟 | 客户端与Milvus跨网络 | 将Milvus部署在与应用同机房 |

```python
class LatencyDiagnostics:
    """检索延迟诊断工具"""
    
    def __init__(self, client: MilvusClient, collection_name: str):
        self.client = client
        self.collection_name = collection_name
    
    def run_diagnostics(self) -> dict:
        """执行延迟诊断"""
        result = {}
        
        # 1. 检查索引类型
        desc = self.client.describe_collection(self.collection_name)
        result["index_type"] = desc.get("index_type", "FLAT (no index)")
        
        # 2. 数据量评估
        count = self.client.query(self.collection_name, output_fields=["count(*)"])
        entity_count = count[0]["count(*)"] if count else 0
        result["entity_count"] = entity_count
        result["recommended_index"] = self._recommend_index(entity_count)
        
        # 3. 执行基准延迟测试
        dim = desc.get("dim", 768)
        latencies = []
        for _ in range(20):
            t0 = time.perf_counter()
            self.client.search(
                collection_name=self.collection_name,
                data=[[0.1] * dim],
                limit=10,
            )
            latencies.append((time.perf_counter() - t0) * 1000)
        
        avg_latency = sum(latencies) / len(latencies)
        result["avg_latency_ms"] = round(avg_latency, 2)
        result["min_latency_ms"] = round(min(latencies), 2)
        result["max_latency_ms"] = round(max(latencies), 2)
        
        # 4. 给出建议
        if avg_latency > 200:
            result["suggestion"] = "延迟过高，建议检查索引类型或扩容QueryNode"
        elif avg_latency > 100:
            result["suggestion"] = "延迟中等，可考虑优化索引参数"
        else:
            result["suggestion"] = "延迟正常"
        
        return result
    
    def _recommend_index(self, entity_count: int) -> str:
        if entity_count < 100000:
            return "FLAT 即可，无需额外索引"
        elif entity_count < 1000000:
            return "IVF_FLAT (nlist=128)"
        elif entity_count < 10000000:
            return "HNSW (M=16, efConstruction=200)"
        else:
            return "IVF_PQ 或 HNSW (根据内存和召回要求)"
```

### 17.4.3 写入失败

RAG知识库更新时，写入操作失败会导致知识库数据不完整：

```python
def diagnose_write_failure(client, collection_name: str):
    """诊断写入失败问题"""
    print("诊断写入失败...")
    
    # 1. 检查连接
    try:
        client.get_server_version()
        print("  ✓ 连接正常")
    except Exception as e:
        print(f"  ✗ 连接异常: {e}")
        return
    
    # 2. 检查磁盘空间
    try:
        # 通过HTTP接口获取磁盘信息
        resp = requests.get("http://localhost:9091/metrics", timeout=5)
        for line in resp.text.split("\n"):
            if "disk_usage" in line:
                print(f"  - {line}")
    except Exception as e:
        print(f"  - 无法获取磁盘信息: {e}")
    
    # 3. 检查批量大小
    print("  - 建议：单次批量写入控制在500-1000条")
    print("  - 建议：检查向量维度是否与集合Schema一致")
    print("  - 建议：检查写入的字段是否包含Schema中所有必填字段")
```

### 17.4.4 索引构建失败

```python
def diagnose_index_failure(client, collection_name: str):
    """诊断索引构建失败问题"""
    print(f"诊断 '{collection_name}' 索引问题...")
    
    # 检查索引状态
    desc = client.describe_collection(collection_name)
    index_status = desc.get("index_status", "unknown")
    print(f"  索引状态: {index_status}")
    
    # 检查是否有足够的实体构建索引
    count = client.query(collection_name, output_fields=["count(*)"])
    entity_count = count[0]["count(*)"] if count else 0
    print(f"  实体数量: {entity_count}")
    
    if entity_count < 10000:
        print("  提示: 数据量较小(<10000)，可以不建索引直接使用FLAT检索")
    
    # 常见索引失败原因
    print("\n  索引构建失败常见原因:")
    print("  1. 磁盘空间不足——确保有足够空间存储临时索引文件")
    print("  2. 内存不足——IVF_PQ/HNSW索引构建需要大量内存")
    print("  3. 参数错误——index_type拼写错误或参数不合法")
    print("  4. 数据为空——空集合无法构建索引")
```

## 17.5 典型报错解决方案

### 17.5.1 MinIO鉴权失败

**错误信息**：`InvalidAccessKeyId` 或 `The access key ID you provided does not exist`

**原因**：MinIO的访问密钥在Milvus配置中配置错误，或在MinIO中已被修改。

**解决方案**：

```yaml
# 1. 在docker-compose.yml中确认MinIO环境变量
services:
  minio:
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin

# 2. 在Milvus配置中同步更新
  milvus:
    environment:
      MINIO_ADDRESS: minio:9000
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
```

如果使用外部S3兼容存储：

```yaml
minio:
  address: s3.amazonaws.com:9000
  accessKey: "your-iam-access-key"
  secretKey: "your-iam-secret-key"
  useSSL: true
  bucketName: "milvus-prod-bucket"
```

### 17.5.2 MixCoord standby

**错误信息**：`MixCoord standby` 或 `coordinator is in standby mode`

**原因**：Milvus集群模式下，多个Coordinator节点中只有一个Active，其余为Standby。如果Active节点异常且Standby未能成功接管，会出现此状态。

**解决方案**：

```bash
# 1. 检查所有Coordinator Pod的状态
kubectl get pods -n milvus | grep coord

# 2. 查看Active Coordinator日志
kubectl logs -n milvus milvus-prod-rootcoord-0

# 3. 如果Standby无法接管，强制重启所有Coordinator
kubectl delete pods -n milvus -l app.kubernetes.io/component=coordinator

# 4. 检查etcd健康状态
docker exec milvus-etcd etcdctl endpoint health
```

### 17.5.3 索引失效

**错误信息**：`index not found` 或 `index out of date`

**原因**：插入新数据后索引未重建，或索引文件损坏。

**解决方案**：

```python
def rebuild_index(client, collection_name: str, field_name: str = "vector"):
    """重建索引"""
    print(f"开始重建 '{collection_name}' 的索引...")
    
    # 1. 先删除旧索引
    try:
        client.drop_index(collection_name, field_name)
        print("  旧索引已删除")
    except Exception as e:
        print(f"  删除索引时发生预期内异常: {e}")
    
    # 2. 重新创建索引
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name=field_name,
        index_type="HNSW",
        metric_type="IP",
        params={"M": 16, "efConstruction": 200}
    )
    client.create_index(collection_name, index_params)
    print("  新索引已创建")
    
    # 3. 重新加载集合
    client.load_collection(collection_name)
    print("  集合已重新加载")
```

### 17.5.4 连接超时

**错误信息**：`connection timeout` 或 `context deadline exceeded`

**原因**：客户端与Milvus服务器之间的网络延迟过高，或服务器负载过重无法及时响应。

**解决方案**：

```python
from pymilvus import MilvusClient

# 1. 增加连接超时
client = MilvusClient(
    uri="http://milvus-prod.internal:19530",
    timeout=60,  # 默认10秒，增加到60秒
)

# 2. 使用连接池避免频繁创建连接
# 3. 检查网络延迟
import subprocess

def check_network_latency(host: str = "localhost", port: int = 19530):
    """检查到Milvus的网络延迟"""
    import socket
    import time
    
    latencies = []
    for _ in range(5):
        t0 = time.perf_counter()
        try:
            sock = socket.create_connection((host, port), timeout=5)
            latencies.append((time.perf_counter() - t0) * 1000)
            sock.close()
        except Exception as e:
            print(f"连接失败: {e}")
            return None
    
    avg_latency = sum(latencies) / len(latencies)
    print(f"到 {host}:{port} 的平均网络延迟: {avg_latency:.2f}ms")
    
    if avg_latency > 50:
        print("警告：网络延迟较高，建议将应用和Milvus部署在同一机房")
    
    return avg_latency
```

### 17.5.5 OOM（内存溢出）

**错误信息**：日志中出现 `OOM Killer`，或容器被Kubernetes自动重启。

**原因**：加载到Milvus内存中的数据量超过可用内存。

**解决方案**：

```yaml
# 1. 限制QueryNode内存使用
queryNode:
  maxMemory: 32768           # 限制为32GB
  memoryWatermark: 0.75      # 75%水位线即触发驱逐
  
  # 2. 启用磁盘存储（Milvus 2.3+）
  enableDisk: true
  diskCacheCapacity: 10737418240  # 10GB磁盘缓存

# 3. 使用量化索引降低内存占用
index_params = {
    "index_type": "IVF_SQ8",   # SQ8可将内存降低75%
    "params": {"nlist": 1024},
    "metric_type": "IP",
}
```

### 17.5.6 数据不一致

**错误信息**：检索结果与预期不符，或同一查询在不同时间返回不同结果。

**原因**：数据写入后未刷新（flush），或数据在复制过程中存在延迟。

**解决方案**：

```python
def ensure_data_consistency(client, collection_name: str):
    """确保数据一致性"""
    
    # 1. 写入后立即刷新
    client.flush(collection_name)
    print("数据已刷新到磁盘")
    
    # 2. 等待索引更新
    import time
    desc = client.describe_collection(collection_name)
    while desc.get("index_status") != "Ready":
        print("等待索引就绪...")
        time.sleep(2)
        desc = client.describe_collection(collection_name)
    
    # 3. 重新加载集合
    client.load_collection(collection_name)
    print("集合已重新加载")
    
    # 4. 验证一致性
    count = client.query(collection_name, output_fields=["count(*)"])
    print(f"当前实体数: {count[0]['count(*)']}")
```

## 17.6 告警规则配置

### 17.6.1 Prometheus告警规则

创建告警规则文件，定义关键告警：

```yaml
# prometheus/alerts/milvus-alerts.yml
groups:
  - name: milvus_alerts
    interval: 30s
    rules:
      # 服务不可用告警
      - alert: MilvusServiceDown
        expr: up{job="milvus"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Milvus 服务不可用"
          description: "Milvus 实例 {{ $labels.instance }} 已下线超过1分钟"
      
      # 检索延迟过高告警
      - alert: HighSearchLatency
        expr: |
          histogram_quantile(0.99,
            rate(milvus_proxy_search_requests_duration_ms_bucket[5m])
          ) > 500
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "检索延迟过高"
          description: "P99 检索延迟超过 500ms"
      
      # 内存使用告警
      - alert: HighMemoryUsage
        expr: |
          milvus_querynode_memory_usage_bytes
          / milvus_querynode_memory_limit_bytes > 0.85
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "QueryNode 内存使用超过 85%"
      
      # 检索失败率告警
      - alert: HighSearchFailureRate
        expr: |
          rate(milvus_proxy_search_requests_fail_count[5m])
          / rate(milvus_proxy_search_requests_count[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "检索失败率超过 5%"
      
      # 磁盘空间告警
      - alert: DiskSpaceLow
        expr: milvus_storage_disk_usage_percent > 80
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "磁盘使用率超过 80%"
```

### 17.6.2 告警通知集成

告警可以通过Alertmanager发送到企业微信、钉钉、邮件等渠道：

```yaml
# alertmanager.yml
route:
  receiver: 'default'
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 4h
  routes:
    - match:
        severity: critical
      receiver: 'pager-duty'
    - match:
        severity: warning
      receiver: 'team-chat'

receivers:
  - name: 'default'
    webhook_configs:
      - url: 'http://alert-bridge:8080/webhook'
  
  - name: 'team-chat'
    webhook_configs:
      - url: 'https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=your-key'
        send_resolved: true
```

## 17.7 本章小结

本章系统性地介绍了Milvus监控告警体系的搭建方法和RAG场景中的常见故障排查方案。以下是要点总结：

1. **监控栈搭建**：Milvus暴露Prometheus格式指标（:9091/metrics），配合Prometheus存储和Grafana可视化，可以快速搭建完整的监控面板。使用Docker Compose一键部署监控栈是最快捷的方式。

2. **核心指标**：重点关注检索QPS、延迟分布（P50/P95/P99）、内存使用率、连接数和检索失败率。这些指标直接反映RAG系统的服务质量和健康状况。

3. **故障排查方法论**：检索为空、延迟过高、写入失败、索引失效等常见问题，都有固定的排查路径。遵循"连接检查->数据检查->索引检查->参数检查"的排查顺序，可以快速定位问题根源。

4. **告警体系**：基于Prometheus告警规则定义关键指标的告警阈值，通过Alertmanager集成企业微信、钉钉等通知渠道，实现7x24小时的自动化告警通知。

5. **监控驱动优化**：监控数据不仅是用于"发现问题"，更重要的是"驱动优化"。通过持续跟踪QPS和延迟变化，可以判断索引参数调整、节点扩容等优化措施的实际效果。
