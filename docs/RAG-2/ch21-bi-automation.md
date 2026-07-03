# 第21章 BI自动化与数据分析

## 21.1 引言

BI（Business Intelligence）自动化是RAG技术在企业数据分析领域的重要应用。通过自然语言与数据系统交互，业务人员可以直接用中文提问获取数据分析结果，无需依赖数据工程师编写SQL或使用复杂的BI工具。RAG系统在BI自动化中的核心价值在于将自然语言查询转化为精确的数据操作，并将分析结果以易于理解的方式呈现。

BI自动化场景对RAG系统有独特的要求：

1. **精确性**：数据分析结果必须精确，错误的数据可能导致错误的决策
2. **实时性**：部分分析需要基于实时数据
3. **安全性**：数据访问需要严格的权限控制
4. **可解释性**：分析过程和结果需要清晰可解释
5. **交互性**：支持追问和数据钻取

### 21.1.1 BI自动化的核心流程

```
自然语言查询 → Schema理解 → SQL生成 → 结果解释 → 可视化
      │            │            │          │          │
      ▼            ▼            ▼          ▼          ▼
  "上月销售额"   表结构      SELECT       "上个月    柱状图
                  字段        SUM(...)    总销售额
                  关系        FROM ...    为XXX"
```

## 21.2 NL2SQL

NL2SQL（Natural Language to SQL）是BI自动化的核心技术，它将自然语言查询转换为SQL语句。

### 21.2.1 Schema链接

```python
class SchemaLinker:
    """Schema链接器：将自然语言映射到数据库Schema"""
    
    def __init__(self, llm, db_schema: Dict):
        self.llm = llm
        self.db_schema = db_schema
        self.table_descriptions = self._build_table_descriptions()
    
    def _build_table_descriptions(self) -> Dict:
        """构建表描述"""
        descriptions = {}
        for table_name, table_info in self.db_schema.items():
            columns_desc = []
            for col in table_info.get('columns', []):
                columns_desc.append(
                    f"  - {col['name']} ({col['type']}): {col.get('description', '')}"
                )
            
            descriptions[table_name] = {
                'description': table_info.get('description', ''),
                'columns': '\n'.join(columns_desc),
                'primary_key': table_info.get('primary_key', ''),
                'foreign_keys': table_info.get('foreign_keys', [])
            }
        
        return descriptions
    
    def link_schema(self, query: str) -> Dict:
        """将自然语言查询链接到Schema"""
        schema_text = self._format_schema()
        
        prompt = f"""将以下自然语言查询链接到数据库Schema。

查询：{query}

数据库Schema：
{schema_text}

请：
1. 识别查询涉及的表
2. 识别查询涉及的字段
3. 识别过滤条件
4. 识别聚合操作

输出JSON：
{{
    "relevant_tables": ["表名"],
    "relevant_columns": {{
        "表名": ["字段名"]
    }},
    "conditions": [
        {{
            "field": "字段路径",
            "operator": "操作符",
            "value": "值"
        }}
    ],
    "aggregations": ["SUM", "COUNT", "AVG"],
    "group_by": ["分组字段"],
    "order_by": ["排序字段"],
    "limit": 数字或null
}}"""
        
        try:
            import json
            response = self.llm(prompt)
            result = json.loads(response)
            return result
        except:
            return {
                'relevant_tables': [],
                'relevant_columns': {},
                'conditions': [],
                'aggregations': [],
                'group_by': [],
                'order_by': []
            }
    
    def _format_schema(self) -> str:
        """格式化Schema描述"""
        parts = []
        for table_name, desc in self.table_descriptions.items():
            parts.append(f"表: {table_name}")
            parts.append(f"  描述: {desc['description']}")
            parts.append(f"  主键: {desc['primary_key']}")
            parts.append(f"  字段:\n{desc['columns']}")
            if desc['foreign_keys']:
                parts.append(f"  外键:\n" + "\n".join(
                    [f"    {fk}" for fk in desc['foreign_keys']]
                ))
            parts.append("")
        
        return "\n".join(parts)

# 示例数据库Schema
SAMPLE_SCHEMA = {
    'sales': {
        'description': '销售记录表',
        'columns': [
            {'name': 'id', 'type': 'INT', 'description': '主键'},
            {'name': 'product_id', 'type': 'INT', 'description': '产品ID'},
            {'name': 'customer_id', 'type': 'INT', 'description': '客户ID'},
            {'name': 'amount', 'type': 'DECIMAL', 'description': '销售金额'},
            {'name': 'quantity', 'type': 'INT', 'description': '销售数量'},
            {'name': 'sale_date', 'type': 'DATE', 'description': '销售日期'},
            {'name': 'region', 'type': 'VARCHAR', 'description': '销售区域'}
        ],
        'primary_key': 'id',
        'foreign_keys': [
            'product_id -> products.id',
            'customer_id -> customers.id'
        ]
    },
    'products': {
        'description': '产品信息表',
        'columns': [
            {'name': 'id', 'type': 'INT', 'description': '主键'},
            {'name': 'name', 'type': 'VARCHAR', 'description': '产品名称'},
            {'name': 'category', 'type': 'VARCHAR', 'description': '产品类别'},
            {'name': 'price', 'type': 'DECIMAL', 'description': '单价'}
        ],
        'primary_key': 'id'
    },
    'customers': {
        'description': '客户信息表',
        'columns': [
            {'name': 'id', 'type': 'INT', 'description': '主键'},
            {'name': 'name', 'type': 'VARCHAR', 'description': '客户名称'},
            {'name': 'tier', 'type': 'VARCHAR', 'description': '客户等级'},
            {'name': 'industry', 'type': 'VARCHAR', 'description': '所属行业'}
        ],
        'primary_key': 'id'
    }
}
```

### 21.2.2 SQL生成

```python
class SQLGenerator:
    """SQL生成器"""
    
    def __init__(self, llm, db_schema: Dict):
        self.llm = llm
        self.db_schema = db_schema
        self.schema_linker = SchemaLinker(llm, db_schema)
    
    def generate(self, query: str) -> Dict:
        """生成SQL"""
        # 1. Schema链接
        schema_mapping = self.schema_linker.link_schema(query)
        
        # 2. 生成SQL
        sql = self._generate_sql(query, schema_mapping)
        
        # 3. 验证SQL
        validation = self._validate_sql(sql)
        
        return {
            'sql': sql,
            'schema_mapping': schema_mapping,
            'validation': validation,
            'is_valid': validation.get('is_valid', False)
        }
    
    def _generate_sql(self, query: str, 
                       schema_mapping: Dict) -> str:
        """生成SQL语句"""
        schema_text = self.schema_linker._format_schema()
        
        prompt = f"""将以下自然语言查询转换为SQL语句。

查询：{query}

数据库Schema：
{schema_text}

涉及的字段和条件：
{json.dumps(schema_mapping, ensure_ascii=False, indent=2)}

要求：
1. 只输出SQL语句
2. 使用标准SQL语法
3. 注意字段类型和JOIN条件
4. 添加适当的注释
5. 确保SQL可执行

SQL："""
        
        try:
            response = self.llm(prompt)
            sql = response.strip()
            # 清理markdown代码块
            sql = sql.replace('```sql', '').replace('```', '').strip()
            return sql
        except:
            return "-- SQL生成失败"
    
    def _validate_sql(self, sql: str) -> Dict:
        """验证SQL"""
        import re
        
        issues = []
        
        # 基本语法检查
        if not sql or sql.startswith('--'):
            issues.append("SQL为空")
        
        if not re.search(r'\bSELECT\b', sql, re.IGNORECASE):
            issues.append("缺少SELECT语句")
        
        # 安全检查
        dangerous_patterns = [
            (r'\bDROP\b', "包含DROP操作"),
            (r'\bDELETE\b', "包含DELETE操作"),
            (r'\bUPDATE\b', "包含UPDATE操作"),
            (r'\bINSERT\b', "包含INSERT操作"),
            (r'\bALTER\b', "包含ALTER操作"),
            (r';\s*SELECT', "多条SQL语句"),
            (r'--[^\n]*\n', "包含注释（可能隐藏恶意SQL）")
        ]
        
        for pattern, message in dangerous_patterns:
            if re.search(pattern, sql, re.IGNORECASE):
                issues.append(message)
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'sql_length': len(sql)
        }
    
    def generate_with_fix(self, query: str, 
                           max_attempts: int = 3) -> Dict:
        """生成并自动修正SQL"""
        for attempt in range(max_attempts):
            result = self.generate(query)
            
            if result['is_valid']:
                return result
            
            if attempt < max_attempts - 1:
                # 基于错误修正
                query = f"{query}\n注意：之前生成的SQL有问题：{result['validation']['issues']}"
        
        return result
```

### 21.2.3 结果解释

```python
class SQLResultInterpreter:
    """SQL结果解释器"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def interpret(self, query: str, sql: str,
                  result: List[Dict]) -> Dict:
        """解释SQL执行结果"""
        result_text = json.dumps(result[:20], ensure_ascii=False, indent=2)
        
        prompt = f"""解释数据分析结果。

用户查询：{query}
SQL：{sql}
查询结果（前20行）：
{result_text}

请输出：
1. 数据总结（关键发现）
2. 数值分析（趋势、异常、对比）
3. 业务洞察
4. 建议的可视化方式

输出JSON：
{{
    "summary": "一句话总结",
    "key_findings": ["发现1", "发现2"],
    "numbers": {{
        "total": 数值,
        "average": 数值,
        "max": 数值,
        "min": 数值,
        "change": "变化描述"
    }},
    "insights": ["洞察1", "洞察2"],
    "suggested_chart": {{
        "type": "bar/line/pie/table",
        "x_axis": "字段名",
        "y_axis": "字段名",
        "title": "图表标题"
    }}
}}"""
        
        try:
            import json
            response = self.llm(prompt)
            return json.loads(response)
        except:
            return {
                'summary': '结果解释失败',
                'key_findings': [],
                'suggested_chart': {'type': 'table'}
            }
    
    def generate_narrative(self, interpretation: Dict) -> str:
        """生成自然语言叙述"""
        parts = []
        
        parts.append(f"## 分析结果\n")
        parts.append(interpretation.get('summary', ''))
        
        key_findings = interpretation.get('key_findings', [])
        if key_findings:
            parts.append("\n**关键发现：**")
            for finding in key_findings:
                parts.append(f"- {finding}")
        
        insights = interpretation.get('insights', [])
        if insights:
            parts.append("\n**业务洞察：**")
            for insight in insights:
                parts.append(f"- {insight}")
        
        return "\n".join(parts)
```

## 21.3 时间序列分析

### 21.3.1 趋势分析

```python
class TimeSeriesAnalyzer:
    """时间序列分析器"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def analyze_trend(self, data: List[Dict], 
                       date_field: str,
                       value_field: str) -> Dict:
        """分析趋势"""
        # 计算基本统计
        values = [d[value_field] for d in data]
        dates = [d[date_field] for d in data]
        
        n = len(values)
        if n < 2:
            return {'trend': 'insufficient_data'}
        
        # 简单线性回归判断趋势
        x = list(range(n))
        y = values
        
        # 计算斜率
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_xx = sum(xi * xi for xi in x)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
        
        # 计算移动平均
        window = min(7, n // 3) if n >= 7 else n
        ma = self._moving_average(values, window)
        
        # 判断趋势
        if slope > 0:
            trend = '上升'
            strength = min(abs(slope) * 100 / (sum_y / n), 1.0)
        elif slope < 0:
            trend = '下降'
            strength = min(abs(slope) * 100 / (sum_y / n), 1.0)
        else:
            trend = '平稳'
            strength = 0
        
        prompt = f"""分析以下时间序列数据趋势。

数据点：{len(data)}个
时间范围：{dates[0]} 到 {dates[-1]}
计算趋势：{trend}（强度：{strength:.2f}）

数据示例：
{chr(10).join([f"{d[date_field]}: {d[value_field]}" for d in data[:10]])}

请输出JSON分析：
{{
    "trend": "上升/下降/平稳/周期性",
    "strength": 0.0-1.0,
    "seasonality": "是否存在周期性（是/否）",
    "anomalies": ["异常点描述"],
    "forecast": "短期趋势预测",
    "recommendation": "业务建议"
}}"""
        
        try:
            import json
            response = self.llm(prompt)
            analysis = json.loads(response)
            analysis['calculated_trend'] = trend
            analysis['slope'] = slope
            analysis['moving_average'] = ma[-1] if ma else values[-1]
            return analysis
        except:
            return {
                'trend': trend,
                'strength': strength,
                'calculated_trend': trend
            }
    
    def _moving_average(self, data: List[float], 
                         window: int) -> List[float]:
        """计算移动平均"""
        ma = []
        for i in range(len(data)):
            if i < window - 1:
                ma.append(sum(data[:i+1]) / (i+1))
            else:
                ma.append(sum(data[i-window+1:i+1]) / window)
        return ma
    
    def detect_seasonality(self, data: List[Dict],
                            value_field: str) -> Dict:
        """检测周期性"""
        values = [d[value_field] for d in data]
        n = len(values)
        
        if n < 14:
            return {'has_seasonality': False, 'period': None}
        
        # 自相关检测
        from collections import Counter
        
        # 计算周内模式
        weekly_pattern = Counter()
        for i, d in enumerate(data):
            try:
                dt = datetime.fromisoformat(d.get('date', ''))
                day_of_week = dt.weekday()
                weekly_pattern[day_of_week] += values[i]
            except:
                pass
        
        return {
            'has_seasonality': len(weekly_pattern) >= 5,
            'period': 'weekly' if len(weekly_pattern) >= 5 else None,
            'daily_pattern': dict(weekly_pattern)
        }
```

## 21.4 定时报表

### 21.4.1 报表调度器

```python
class ReportScheduler:
    """报表调度器"""
    
    def __init__(self):
        self.reports = []
        self.schedules = []
    
    def create_report(self, name: str, 
                       description: str,
                       sql: str,
                       schedule: str,
                       recipients: List[str],
                       visualization: Dict = None) -> Dict:
        """创建报表"""
        report = {
            'id': str(uuid.uuid4()),
            'name': name,
            'description': description,
            'sql': sql,
            'schedule': schedule,
            'recipients': recipients,
            'visualization': visualization or {'type': 'table'},
            'created_at': datetime.now().isoformat(),
            'last_run': None,
            'status': 'active'
        }
        self.reports.append(report)
        
        # 添加到调度
        self.schedules.append({
            'report_id': report['id'],
            'cron': self._parse_schedule(schedule),
            'next_run': self._calculate_next_run(schedule)
        })
        
        return report
    
    def _parse_schedule(self, schedule: str) -> str:
        """解析调度表达式"""
        schedule_map = {
            'daily_9am': '0 9 * * *',
            'daily_6pm': '0 18 * * *',
            'weekly_monday': '0 9 * * 1',
            'monthly_first': '0 9 1 * *',
            'hourly': '0 * * * *'
        }
        return schedule_map.get(schedule, schedule)
    
    def _calculate_next_run(self, schedule: str) -> str:
        """计算下次运行时间"""
        from datetime import timedelta
        
        now = datetime.now()
        
        if schedule == 'daily_9am':
            next_run = now.replace(hour=9, minute=0, second=0)
            if next_run <= now:
                next_run += timedelta(days=1)
        elif schedule == 'weekly_monday':
            next_run = now + timedelta(days=(7 - now.weekday()) % 7)
            next_run = next_run.replace(hour=9, minute=0, second=0)
        else:
            next_run = now + timedelta(hours=1)
        
        return next_run.isoformat()
    
    def get_due_reports(self) -> List[Dict]:
        """获取到期的报表"""
        now = datetime.now()
        due = []
        
        for schedule in self.schedules:
            next_run = datetime.fromisoformat(schedule['next_run'])
            if next_run <= now:
                report = next(
                    (r for r in self.reports if r['id'] == schedule['report_id']),
                    None
                )
                if report and report['status'] == 'active':
                    due.append(report)
                    # 更新下次运行时间
                    schedule['next_run'] = self._calculate_next_run(
                        self._get_schedule_str(report.get('schedule', ''))
                    )
        
        return due
    
    def _get_schedule_str(self, report_schedule: str) -> str:
        """获取调度字符串"""
        # 简化实现
        return report_schedule
    
    def execute_report(self, report_id: str, 
                        db_connector) -> Dict:
        """执行报表"""
        report = next(
            (r for r in self.reports if r['id'] == report_id),
            None
        )
        if not report:
            return {'error': '报表不存在'}
        
        try:
            # 执行SQL
            result = db_connector.execute(report['sql'])
            
            # 生成可视化
            visualization = self._generate_visualization(
                result, report.get('visualization', {})
            )
            
            report['last_run'] = datetime.now().isoformat()
            
            return {
                'report_id': report_id,
                'name': report['name'],
                'data': result,
                'visualization': visualization,
                'executed_at': report['last_run']
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _generate_visualization(self, data: List[Dict],
                                  config: Dict) -> Dict:
        """生成可视化配置"""
        viz_type = config.get('type', 'table')
        
        if viz_type == 'table':
            return {
                'type': 'table',
                'columns': list(data[0].keys()) if data else [],
                'rows': len(data)
            }
        elif viz_type == 'bar':
            return {
                'type': 'bar',
                'x_axis': config.get('x_axis', list(data[0].keys())[0]),
                'y_axis': config.get('y_axis', list(data[0].keys())[1]),
                'data_points': len(data)
            }
        
        return {'type': 'table'}
```

## 21.5 异常检测

### 21.5.1 实时异常检测

```python
class AnomalyDetector:
    """异常检测器"""
    
    def __init__(self, llm):
        self.llm = llm
        self.baselines = {}
        self.thresholds = {}
    
    def set_baseline(self, metric_name: str, 
                      historical_data: List[float]):
        """设置基线"""
        if not historical_data:
            return
        
        import numpy as np
        
        mean = np.mean(historical_data)
        std = np.std(historical_data)
        
        self.baselines[metric_name] = {
            'mean': mean,
            'std': std,
            'min': min(historical_data),
            'max': max(historical_data),
            'data_points': len(historical_data)
        }
        
        # 默认阈值：2倍标准差
        self.thresholds[metric_name] = {
            'warning': mean + 2 * std,
            'critical': mean + 3 * std
        }
    
    def detect(self, metric_name: str, 
                current_value: float,
                context: Dict = None) -> Dict:
        """检测异常"""
        baseline = self.baselines.get(metric_name)
        if not baseline:
            return {'is_anomaly': False, 'reason': '无基线数据'}
        
        threshold = self.thresholds.get(metric_name, {})
        
        # 计算偏离程度
        deviation = (current_value - baseline['mean']) / baseline['std'] if baseline['std'] > 0 else 0
        
        is_anomaly = abs(deviation) > 2
        
        # 判断严重程度
        if abs(deviation) > 3:
            severity = 'critical'
        elif abs(deviation) > 2:
            severity = 'warning'
        else:
            severity = 'normal'
        
        result = {
            'is_anomaly': is_anomaly,
            'metric': metric_name,
            'current_value': current_value,
            'baseline_mean': baseline['mean'],
            'baseline_std': baseline['std'],
            'deviation': deviation,
            'severity': severity,
            'timestamp': datetime.now().isoformat()
        }
        
        # 使用LLM分析异常原因
        if is_anomaly and context:
            result['analysis'] = self._analyze_anomaly(
                metric_name, current_value, baseline, context
            )
        
        return result
    
    def _analyze_anomaly(self, metric_name: str,
                          current_value: float,
                          baseline: Dict,
                          context: Dict) -> Dict:
        """分析异常原因"""
        prompt = f"""分析以下指标的异常情况。

指标名称：{metric_name}
当前值：{current_value:.2f}
基线平均值：{baseline['mean']:.2f}
基线标准差：{baseline['std']:.2f}
偏离程度：{(current_value - baseline['mean']) / baseline['std']:.2f} 标准差

上下文信息：
{json.dumps(context, ensure_ascii=False, indent=2)[:500]}

请分析可能的原因和推荐行动：
输出JSON：
{{
    "possible_causes": ["原因1", "原因2"],
    "impact": "业务影响描述",
    "recommended_actions": ["行动1", "行动2"],
    "priority": "high/medium/low"
}}"""
        
        try:
            import json
            response = self.llm(prompt)
            return json.loads(response)
        except:
            return {
                'possible_causes': ['分析失败'],
                'recommended_actions': ['人工排查']
            }
```

## 21.6 仪表盘集成

### 21.6.1 自然语言查询仪表盘

```python
class DashboardQueryInterface:
    """仪表盘自然语言查询接口"""
    
    def __init__(self, sql_generator: SQLGenerator,
                 result_interpreter: SQLResultInterpreter,
                 db_connector):
        self.sql_generator = sql_generator
        self.result_interpreter = result_interpreter
        self.db_connector = db_connector
        
        self.query_history = []
    
    def query(self, natural_query: str) -> Dict:
        """自然语言查询仪表盘"""
        start_time = time.time()
        
        # 1. 生成SQL
        sql_result = self.sql_generator.generate(natural_query)
        
        if not sql_result['is_valid']:
            return {
                'success': False,
                'error': '无法生成有效的SQL',
                'details': sql_result['validation']['issues'],
                'latency_ms': (time.time() - start_time) * 1000
            }
        
        # 2. 执行SQL
        try:
            db_result = self.db_connector.execute(sql_result['sql'])
        except Exception as e:
            return {
                'success': False,
                'error': f'SQL执行失败: {str(e)}',
                'sql': sql_result['sql'],
                'latency_ms': (time.time() - start_time) * 1000
            }
        
        # 3. 解释结果
        interpretation = self.result_interpreter.interpret(
            natural_query, sql_result['sql'], db_result
        )
        
        # 4. 记录查询
        self._record_query(natural_query, sql_result['sql'], 
                          len(db_result), time.time() - start_time)
        
        return {
            'success': True,
            'query': natural_query,
            'sql': sql_result['sql'],
            'data': db_result,
            'interpretation': interpretation,
            'narrative': self.result_interpreter.generate_narrative(interpretation),
            'row_count': len(db_result),
            'latency_ms': (time.time() - start_time) * 1000
        }
    
    def _record_query(self, query: str, sql: str,
                       result_count: int, latency: float):
        """记录查询"""
        self.query_history.append({
            'query': query,
            'sql': sql,
            'result_count': result_count,
            'latency': latency,
            'timestamp': datetime.now().isoformat()
        })
        
        # 限制历史数量
        if len(self.query_history) > 1000:
            self.query_history = self.query_history[-1000:]
    
    def get_popular_queries(self, top_k: int = 10) -> List[Dict]:
        """获取热门查询"""
        from collections import Counter
        
        query_counter = Counter(q['query'] for q in self.query_history)
        return [
            {'query': q, 'count': c}
            for q, c in query_counter.most_common(top_k)
        ]
```

## 21.7 本章小结

本章详细介绍了RAG系统在BI自动化和数据分析场景中的应用。

**NL2SQL**是BI自动化的核心技术。Schema链接将自然语言查询映射到数据库Schema，SQL生成器将映射结果转换为可执行的SQL语句，结果解释器将数据结果转化为业务洞察。安全验证确保生成的SQL不会执行危险操作。

**时间序列分析**支持趋势检测、周期性分析和异常发现。通过统计方法和LLM分析的结合，既能保证计算的精确性，又能提供业务层面的解释。

**定时报表**功能通过调度器实现报表的自动生成和分发。支持日报、周报、月报等多种调度频率，并集成可视化配置。

**异常检测**模块通过建立数据基线，实时监控关键指标的变化。当指标偏离超过阈值时，自动触发告警并分析可能的原因和推荐的应对措施。

**仪表盘集成**提供了统一的数据查询接口，支持自然语言查询、结果解释和查询历史管理。热门查询分析可以帮助了解用户的常见需求，持续优化系统。

在实际部署中，建议从简单的数据查询场景开始，逐步引入时间序列分析和异常检测等高级功能。NL2SQL的准确性需要通过充分的测试和迭代优化，建议建立常见查询的测试用例集。数据安全是BI自动化中的关键关注点，需要严格的权限控制和SQL安全验证。
