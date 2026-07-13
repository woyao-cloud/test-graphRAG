# 附录C 故障排查流程与诊断工具

当 Milvus RAG 系统出现异常时，系统化的故障排查方法远比零散的"试错"高效。本附录将介绍一套完整的故障排查方法论，涵盖诊断命令、日志分析方法和性能诊断工具。

## C.1 系统性故障排查方法论

### C.1.1 分层排查法

Milvus RAG 系统的故障可能出现在多个层面。建议采用从上到下的分层排查策略：

```
┌──────────────────────────────────────┐
│  第一层：应用层                        │
│  Python SDK 调用是否正确？             │
│  参数配置是否合理？                     │
│  Embedding 模型是否正常？              │
├──────────────────────────────────────┤
│  第二层：API/网络层                    │
│  gRPC 连接是否正常？                   │
│  请求是否超时？                        │
│  负载均衡是否正常？                    │
├──────────────────────────────────────┤
│  第三层：Milvus 服务层                 │
│  各组件状态是否正常？                   │
│  索引是否已构建和加载？                 │
│  内存和磁盘使用是否合理？               │
├──────────────────────────────────────┤
│  第四层：基础设施层                    │
│  Docker/K8s 容器是否正常运行？          │
│  MinIO/Etcd/Pulsar 是否正常？          │
│  硬件资源是否充足？                    │
└──────────────────────────────────────┘
```

**排查原则**：从最外层的应用层开始排查，逐步深入到基础设施层。大部分问题其实出在应用层和 API 层，而非 Milvus 服务本身。

### C.1.2 黄金指标法

对于任何异常，首先检查以下五个黄金指标：

```python
GOLDEN_METRICS = {
    "延迟（Latency）": "请求响应时间，区分 P50/P95/P99",
    "流量（Traffic）": "QPS（每秒查询数），判断负载情况",
    "错误率（Errors）": "请求失败比例，正常应 <1%",
    "饱和度（Saturation）": "CPU/内存/磁盘使用率，是否接近瓶颈",
    "可用性（Availability）": "服务是否正常运行，能否接受请求",
}

def health_check(client, collection_name):
    """黄金指标健康检查"""
    report = {}

    # 1. 延迟
    import time
    start = time.time()
    try:
        client.list_collections()
        report["latency_ms"] = (time.time() - start) * 1000
    except Exception as e:
        report["latency_ms"] = -1
        report["error"] = str(e)

    # 2. 流量
    report["qps"] = "需要 Prometheus 数据"

    # 3. 错误率
    report["has_error"] = "latency_ms" not in report or report["latency_ms"] < 0

    # 4. 饱和度
    import psutil
    report["cpu_percent"] = psutil.cpu_percent()
    report["memory_percent"] = psutil.virtual_memory().percent
    report["disk_percent"] = psutil.disk_usage("/").percent

    # 5. 可用性
    report["service_available"] = client.service_available()

    return report
```

### C.1.3 问题定位决策树

```
问题出现
│
├─ 检索相关？
│   ├─ 结果为空 → 检查集合是否加载、维度是否匹配
│   ├─ 结果不相关 → 检查 Embedding 模型、索引参数
│   └─ 结果重复 → 检查数据去重逻辑
│
├─ 写入相关？
│   ├─ 写入失败 → 检查维度、类型、主键冲突
│   ├─ 写入慢 → 检查批次大小、索引状态
│   └─ 数据不可见 → 检查 Flush 和一致性级别
│
├─ 性能相关？
│   ├─ 延迟高 → 检查索引类型、加载状态、硬件资源
│   ├─ QPS 低 → 检查连接池、并发配置、负载均衡
│   └─ 资源高 → 检查内存泄漏、索引膨胀
│
└─ 部署相关？
    ├─ 容器退出 → 检查日志、端口冲突、资源限制
    ├─ 连接失败 → 检查网络、认证、防火墙
    └─ 数据丢失 → 检查持久化配置、备份策略
```

## C.2 Milvus 诊断命令

### C.2.1 核心诊断命令集

```bash
# ── 服务状态诊断 ──

# 1. 检查所有组件状态
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# 2. 查看容器资源使用
docker stats --no-stream milvus-standalone

# 3. 检查 Milvus 健康端点
curl -s http://localhost:9091/health | python -m json.tool
# 预期输出: {"status": "OK"}

# 4. 检查组件详细信息
curl -s http://localhost:9091/metrics | grep -E "milvus_(system|proxy)_(go|process)"

# 5. 检查索引构建进度
python -c "
from pymilvus import utility
print(utility.index_building_progress('enterprise_kb'))
"

# 6. 检查集合加载进度
python -c "
from pymilvus import utility
print(utility.loading_progress('enterprise_kb'))
"
```

### C.2.2 Python 诊断脚本

```python
"""
milvus_diagnostics.py — Milvus 综合诊断工具
运行方式: python milvus_diagnostics.py --uri http://localhost:19530
"""

import sys
import time
import argparse
from pymilvus import MilvusClient, utility

def diagnose_milvus(uri):
    """运行全面诊断"""
    print("=" * 60)
    print("Milvus 诊断报告")
    print("=" * 60)

    # 1. 连接诊断
    print("\n[1] 连接诊断")
    print("-" * 40)
    try:
        start = time.time()
        client = MilvusClient(uri=uri)
        connect_time = (time.time() - start) * 1000
        print(f"  连接成功 (耗时: {connect_time:.1f}ms)")
    except Exception as e:
        print(f"  连接失败: {e}")
        sys.exit(1)

    # 2. 服务版本诊断
    print("\n[2] 服务版本")
    print("-" * 40)
    try:
        version = utility.get_server_version()
        print(f"  Milvus 版本: {version}")
    except Exception as e:
        print(f"  获取版本失败: {e}")

    # 3. 集合诊断
    print("\n[3] 集合诊断")
    print("-" * 40)
    collections = client.list_collections()
    print(f"  集合数量: {len(collections)}")
    for name in collections:
        try:
            stats = client.get_collection_stats(name)
            schema = client.describe_collection(name)
            dim = None
            for field in schema["fields"]:
                if field["type"] in ("FLOAT_VECTOR", "BINARY_VECTOR"):
                    dim = field["params"].get("dim", "N/A")
            print(f"  - {name}: {stats.get('row_count', 'N/A')} 条, 维度 {dim}")
        except Exception as e:
            print(f"  - {name}: 诊断失败 - {e}")

    # 4. 索引诊断
    print("\n[4] 索引诊断")
    print("-" * 40)
    for name in collections:
        try:
            indexes = client.list_indexes(name)
            print(f"  {name} 的索引:")
            for idx in indexes:
                print(f"    - 字段: {idx['field']}, 类型: {idx['index_type']}, "
                      f"度量: {idx['metric_type']}")
        except Exception as e:
            print(f"  {name}: 获取索引失败 - {e}")

    # 5. 内存诊断
    print("\n[5] 内存状态")
    print("-" * 40)
    try:
        memory_info = utility.get_memory_info()
        print(f"  总内存: {memory_info.get('total', 'N/A')}MB")
        print(f"  已用内存: {memory_info.get('used', 'N/A')}MB")
    except Exception as e:
        print(f"  获取内存信息失败: {e}")

    print("\n" + "=" * 60)
    print("诊断完成")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Milvus 诊断工具")
    parser.add_argument("--uri", default="http://localhost:19530", help="Milvus 连接地址")
    args = parser.parse_args()
    diagnose_milvus(args.uri)
```

## C.3 日志分析方法

### C.3.1 Milvus 日志结构

Milvus 各组件的日志位于容器内的 `/var/lib/milvus/logs/` 目录：

| 日志文件 | 内容说明 | 排查重点 |
|---------|---------|---------|
| `proxy.log` | Proxy 组件的请求日志 | 请求量、错误率、延迟 |
| `querynode.log` | QueryNode 检索日志 | 索引加载、查询执行 |
| `datanode.log` | DataNode 写入日志 | 数据插入、Flush |
| `indexnode.log` | 索引构建日志 | 索引构建进度、失败原因 |
| `rootcoord.log` | RootCoord 协调日志 | 集合管理、DDL 操作 |
| `datacoord.log` | DataCoord 数据协调日志 | 数据段管理、Compaction |

### C.3.2 关键日志模式与解读

```bash
# 查看最近 100 条错误日志
docker exec milvus-standalone grep -E "(ERROR|FATAL|panic)" \
  /var/lib/milvus/logs/*.log | tail -100

# 查看索引构建相关日志
docker exec milvus-standalone grep -E "build index|index node" \
  /var/lib/milvus/logs/indexnode.log | tail -50

# 查看检索延迟异常
docker exec milvus-standalone grep -E "search request|slow query" \
  /var/lib/milvus/logs/querynode.log | tail -30

# 实时跟踪日志
docker logs -f milvus-standalone --tail 50
```

### C.3.3 常见日志模式解读

```
# 模式1：索引构建成功
"[index node] build index done, index id: xxx, index type: HNSW"
→ 正常状态，索引已构建完成

# 模式2：内存不足告警
"[query node] memory usage is high, current: 12.5GB, threshold: 8GB"
→ 需要增加内存或减少加载的数据量

# 模式3：查询超时
"[proxy] search request timeout, collection: xxx, timeout: 10000ms"
→ 查询耗时超过10秒，检查索引和资源

# 模式4：数据段合并
"[data coord] compaction triggered, segment: xxx, reason: too many small segments"
→ 正常的数据合并行为，小段合并为大段以优化检索性能

# 模式5：连接池耗尽
"[proxy] no available connection, all connections are busy"
→ 并发过高，需要增加 Proxy 节点或优化连接池
```

### C.3.4 日志级别调整

```yaml
# milvus.yaml 日志配置
log:
  level: info              # debug / info / warn / error / fatal
  file:
    rootPath: /var/lib/milvus/logs
    maxSize: 300           # 每个日志文件最大 300MB
    maxAge: 10             # 保留 10 天
    maxBackups: 20         # 最多 20 个备份文件

# 临时调整为 debug 级别（排查问题时使用）
# 排查结束后记得改回 info 级别，避免日志过多
```

## C.4 性能诊断工具

### C.4.1 内置诊断接口

Milvus 提供了丰富的诊断接口，可通过 HTTP 访问：

```bash
# 1. Prometheus 指标接口（核心诊断数据源）
curl -s http://localhost:9091/metrics | head -100

# 2. 关键指标查询
# 查询 QPS
curl -s http://localhost:9091/metrics | grep "milvus_proxy_search_requests_total"

# 查询延迟
curl -s http://localhost:9091/metrics | grep "milvus_proxy_search_duration_milliseconds"

# 查询缓存命中率
curl -s http://localhost:9091/metrics | grep "milvus_querynode_cache_hit_rate"

# 3. 健康检查
curl -s http://localhost:9091/health | python -m json.tool

# 4. 构建信息
curl -s http://localhost:9091/status | python -m json.tool
```

### C.4.2 性能基准测试脚本

```python
"""
benchmark_search.py — Milvus 检索性能基准测试
测试不同并发、不同 TopK 下的检索性能
"""

import time
import random
import threading
from statistics import mean, median
from pymilvus import MilvusClient


def benchmark(client, collection_name, dim=768,
              concurrency_list=[1, 10, 50],
              top_k_list=[10, 100, 500],
              queries_per_thread=100):
    """运行性能基准测试"""
    results = {}

    for concurrency in concurrency_list:
        for top_k in top_k_list:
            print(f"\n测试: 并发={concurrency}, TopK={top_k}")

            latencies = []
            errors = 0
            lock = threading.Lock()

            def worker():
                nonlocal errors
                for _ in range(queries_per_thread):
                    q_vec = [random.random() for _ in range(dim)]
                    start = time.time()
                    try:
                        client.search(
                            collection_name=collection_name,
                            data=[q_vec],
                            limit=top_k,
                        )
                        lat = (time.time() - start) * 1000
                        with lock:
                            latencies.append(lat)
                    except Exception as e:
                        with lock:
                            errors += 1

            threads = [threading.Thread(target=worker)
                       for _ in range(concurrency)]
            t_start = time.time()
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            elapsed = time.time() - t_start

            total_q = queries_per_thread * concurrency
            qps = total_q / elapsed if elapsed > 0 else 0

            key = f"concurrent_{concurrency}_topk_{top_k}"
            results[key] = {
                "qps": round(qps, 1),
                "avg_latency_ms": round(mean(latencies), 1),
                "p50_ms": round(median(latencies), 1),
                "p99_ms": round(sorted(latencies)[int(len(latencies)*0.99)], 1)
                          if latencies else 0,
                "errors": errors,
            }

            print(f"  QPS: {qps:.0f}, P50: {results[key]['p50_ms']}ms, "
                  f"P99: {results[key]['p99_ms']}ms, 错误: {errors}")

    return results
```

### C.4.3 Prometheus + Grafana 监控面板

生产环境建议使用 Prometheus + Grafana 搭建可视化监控：

```yaml
# docker-compose.yml 中添加监控组件
prometheus:
  image: prom/prometheus:v2.45.0
  volumes:
    - ./prometheus.yml:/etc/prometheus/prometheus.yml
  ports:
    - "9090:9090"

grafana:
  image: grafana/grafana:10.0.0
  environment:
    - GF_SECURITY_ADMIN_PASSWORD=admin
  ports:
    - "3000:3000"
  depends_on:
    - prometheus
```

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'milvus'
    scrape_interval: 10s
    static_configs:
      - targets: ['milvus-standalone:9091']
```

**Grafana 核心监控面板建议包含的图表**：

| 图表 | 数据来源 | 告警阈值 |
|------|---------|---------|
| QPS 趋势 | milvus_proxy_search_requests_total | 低于基线50% |
| P99 延迟趋势 | milvus_proxy_search_duration_milliseconds_bucket | >500ms |
| 内存使用率 | process_resident_memory_bytes | >80% |
| 磁盘使用率 | milvus_storage_disk_usage_bytes | >85% |
| 索引状态 | milvus_index_node_indexing_latency | 构建失败 |
| 错误率 | milvus_proxy_search_requests_fail_total | >1% |

### C.4.4 慢查询分析

```python
def analyze_slow_queries(client, collection_name, sample_count=100):
    """分析慢查询特征"""
    import time
    import numpy as np

    latencies = []
    for i in range(sample_count):
        q_vec = [random.random() for _ in range(768)]

        start = time.time()
        # 使用不同的 TopK 和过滤条件
        for top_k in [10, 50, 100, 200]:
            t0 = time.time()
            client.search(
                collection_name=collection_name,
                data=[q_vec],
                limit=top_k,
            )
            latencies.append({
                "top_k": top_k,
                "latency_ms": (time.time() - t0) * 1000,
            })

    # 分析结果
    for top_k in [10, 50, 100, 200]:
        group = [l for l in latencies if l["top_k"] == top_k]
        lats = [l["latency_ms"] for l in group]
        print(f"TopK={top_k}: avg={np.mean(lats):.1f}ms, "
              f"p50={np.median(lats):.1f}ms, p99={np.percentile(lats, 99):.1f}ms, "
              f"max={max(lats):.1f}ms")
```

## C.5 故障排查速查表

| 症状 | 排查步骤 | 常用命令 |
|------|---------|---------|
| 连接失败 | 1.检查容器状态 2.检查端口 3.检查网络 | `docker ps`, `telnet localhost 19530` |
| 检索为空 | 1.检查集合加载 2.检查索引 3.检查过滤条件 | `utility.loading_progress()` |
| 检索慢 | 1.检查索引类型 2.检查内存 3.检查参数 | `docker stats`, 调整 nprobe/ef |
| 写入失败 | 1.检查维度 2.检查主键 3.检查权限 | `client.describe_collection()` |
| 索引失败 | 1.检查参数 2.检查内存 3.检查数据量 | `utility.index_building_progress()` |
| 容器退出 | 1.检查日志 2.检查端口 3.检查资源 | `docker logs`, `docker stats` |
| 内存飙升 | 1.检查加载集合 2.检查索引 3.检查泄漏 | `docker stats`, 卸载不需要的集合 |
| 数据不一致 | 1.检查 flush 2.检查 compaction 3.检查同步 | `client.flush()`, `client.compact()` |

## 本章小结

系统化的故障排查能力是保障 Milvus RAG 生产环境稳定运行的关键。核心要点包括：

1. **分层排查法**：从应用层到基础设施层逐层排查，避免在错误层面浪费精力。
2. **黄金指标法**：延迟、流量、错误率、饱和度、可用性五个指标快速评估系统健康状态。
3. **日志分析**：不同组件日志各有侧重，Proxy 日志看请求量，QueryNode 日志看检索性能，IndexNode 日志看索引状态。
4. **性能诊断**：Prometheus + Grafana 是生产环境的标配监控方案，内置指标接口提供了丰富的诊断数据。
5. **基准测试**：建立系统的性能基线，每次变更后重新测试，量化评估变更影响。
