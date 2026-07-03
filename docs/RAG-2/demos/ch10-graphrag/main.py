#!/usr/bin/env python3
"""
ch10-graphrag: GraphRAG integration demo. Attempts to use the graphrag package
if API key is set; otherwise runs in dry-run mode printing configuration.
Self-contained; optional dependency: graphrag (pip install graphrag).
"""

import os
import tempfile
from typing import Any, Dict


def create_graphrag_config() -> Dict[str, Any]:
    """Return a GraphRAG configuration dict for DeepSeek."""
    return {
        "api_key": os.environ.get("GRAPHRAG_API_KEY", ""),
        "api_base": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
        "embedding_model": "text-embedding-ada-002",
        "encoding_model": "cl100k_base",
        "root_dir": "",
        "chunk": {
            "size": 1200,
            "overlap": 100,
            "group_by_columns": ["id"],
        },
        "local_search": {
            "max_tokens": 4000,
            "temperature": 0.0,
            "top_k": 10,
            "llm_max_tokens": 2000,
        },
        "global_search": {
            "max_tokens": 4000,
            "temperature": 0.0,
            "top_k": 20,
            "llm_max_tokens": 2000,
        },
        "entity_extraction": {
            "prompt": "从文本中提取医药领域的实体和关系。",
            "max_gleanings": 2,
        },
        "community_report": {
            "max_length": 2000,
            "max_input_length": 8000,
        },
        "summarize_descriptions": {
            "max_length": 500,
        },
        "claim_extraction": {
            "enabled": True,
            "max_gleanings": 1,
        },
    }


def prepare_test_data(dir_path: str) -> None:
    samples = {
        "恒瑞医药.txt": "恒瑞医药是中国领先的创新药企业。公司成立于1970年，总部位于江苏连云港。主要产品包括抗肿瘤药物、手术用药和造影剂。2024年研发投入超60亿元。",
        "奥希替尼片.txt": "奥希替尼（泰瑞沙）是第三代EGFR-TKI靶向药，用于非小细胞肺癌治疗。对EGFR敏感突变和T790M耐药突变均有效。FLAURA研究显示显著延长PFS。",
        "国药控股.txt": "国药控股是中国最大医药分销企业，覆盖全国31省市配送网络。2024年营收超6000亿元，服务2万家以上医疗机构。",
    }
    for filename, content in samples.items():
        path = os.path.join(dir_path, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)


def try_build_index(config: Dict[str, Any], data_dir: str) -> bool:
    api_key = config.get("api_key", "")
    if not api_key:
        return False
    try:
        import graphrag.api as graphrag_api
        config["root_dir"] = data_dir
        print("  正在构建 GraphRAG 索引...")
        graphrag_api.build_index(
            root_dir=data_dir,
            api_key=api_key,
            api_base=config["api_base"],
            model=config["model"],
            embedding_model=config["embedding_model"],
            chunk_size=config["chunk"]["size"],
            chunk_overlap=config["chunk"]["overlap"],
        )
        print("  GraphRAG 索引构建成功！")
        return True
    except ImportError:
        print("  graphrag 包未安装。请执行: pip install graphrag")
        return False
    except Exception as e:
        print(f"  索引构建失败: {e}")
        return False


def main():
    print("=" * 60)
    print("ch10-graphrag: GraphRAG 集成演示")
    print("=" * 60)

    print("\n[1/3] 生成 GraphRAG 配置...")
    config = create_graphrag_config()
    print(f"  API Base: {config['api_base']}")
    print(f"  Model: {config['model']}")
    print(f"  Embedding Model: {config['embedding_model']}")
    print(f"  Chunk Size: {config['chunk']['size']}")
    print(f"  Chunk Overlap: {config['chunk']['overlap']}")
    print(f"  API Key 已设置: {'是' if config['api_key'] else '否'}")

    print(f"\n[2/3] 准备测试数据...")
    tmpdir = tempfile.mkdtemp(prefix="graphrag_demo_")
    prepare_test_data(tmpdir)
    files = os.listdir(tmpdir)
    print(f"  临时目录: {tmpdir}")
    for f in files:
        fpath = os.path.join(tmpdir, f)
        size = os.path.getsize(fpath)
        print(f"    创建: {f} ({size} 字节)")

    print(f"\n[3/3] 构建索引...")
    success = try_build_index(config, tmpdir)

    if not success:
        print("\n  干运行模式 (dry-run):")
        print("  " + "-" * 50)
        print("  GraphRAG 索引构建需要以下步骤:")
        print("    1. 设置环境变量: set GRAPHRAG_API_KEY=your_deepseek_key")
        print("    2. 安装 graphrag: pip install graphrag")
        print("    3. 运行索引: python -m graphrag.index --root ./data")
        print("    4. 查询: python -m graphrag.query --root ./data --method local \"你的问题\"")
        print("  " + "-" * 50)

    print("\n" + "=" * 60)
    print("配置摘要")
    print("=" * 60)
    print(f"  LLM 模型:       {config['model']}")
    print(f"  API Base:       {config['api_base']}")
    print(f"  分块大小:       {config['chunk']['size']} tokens")
    print(f"  分块重叠:       {config['chunk']['overlap']} tokens")
    print(f"  本地搜索 top_k: {config['local_search']['top_k']}")
    print(f"  全局搜索 top_k: {config['global_search']['top_k']}")
    print(f"  实体提取:       启用, max_gleanings={config['entity_extraction']['max_gleanings']}")
    print(f"  声明提取:       {'启用' if config['claim_extraction']['enabled'] else '禁用'}")
    print(f"  状态:           {'索引已构建' if success else '干运行模式 (dry-run)'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
