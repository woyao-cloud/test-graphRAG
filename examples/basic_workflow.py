"""Basic end-to-end workflow example.

Demonstrates the complete GraphRAG-KG pipeline:
1. Generate test data
2. Initialize project
3. Ingest documents
4. Build knowledge graph index
5. Sync to Neo4j
6. Query the knowledge graph

Usage:
    python examples/basic_workflow.py
"""

from pathlib import Path


def main():
    print("=" * 60)
    print("GraphRAG-KG Basic Workflow")
    print("=" * 60)

    # Step 1: Generate test data
    print("\n[1/6] Generating test data...")
    from graphrag_kg.data.generator import TestDataGenerator

    output_dir = Path("tests/fixtures/generated/tech_company")
    generator = TestDataGenerator(scenario="tech_company", seed=42)
    gt = generator.generate(output_dir)
    print(f"  Generated {len(gt.entities)} entities, "
          f"{len(gt.relationships)} relationships, "
          f"{len(gt.test_queries)} test queries")

    # Step 2: Initialize project
    print("\n[2/6] Initializing project...")
    from graphrag_kg.core.project import ProjectManager

    project_dir = Path("./example_project")
    pm = ProjectManager(project_dir)
    config = pm.init(project_name="example-graphrag", force=True)
    print(f"  Project created at {project_dir}")

    # Step 3: Ingest documents
    print("\n[3/6] Ingesting documents...")
    from graphrag_kg.ingest.loader import DocumentLoader
    from graphrag_kg.ingest.converter import DocumentConverter

    loader = DocumentLoader()
    docs = loader.load(
        [output_dir / "documents"],
        file_patterns=["*.md"],
        recursive=False,
    )
    converter = DocumentConverter(input_dir=config.input_dir)
    paths = converter.convert_all(docs)
    print(f"  Ingested {len(docs)} documents -> {len(paths)} input files")

    # Step 4: Check index readiness
    print("\n[4/6] Checking index readiness...")
    from graphrag_kg.index.runner import IndexRunner

    runner = IndexRunner(config)
    report = runner.run(method="standard", dry_run=True)
    print(f"  Input files: {report['input_files']}")
    print(f"  Total characters: {report['total_characters']}")
    print(f"  Chat model: {report['chat_model']}")
    print("  (Dry run only — actual indexing requires LLM API key)")

    # Step 5: Check Neo4j status
    print("\n[5/6] Checking Neo4j status...")
    from graphrag_kg.graph.connection import Neo4jConnection

    conn = Neo4jConnection(config.neo4j)
    health = conn.health_check()
    if health["connected"]:
        print(f"  Neo4j connected: {health.get('neo4j_version', 'unknown')}")
    else:
        print(f"  Neo4j not available: {health.get('error', 'unknown')}")
        print("  Start Neo4j with: docker-compose up -d")

    # Step 6: Query engine status
    print("\n[6/6] Query engine status...")
    from graphrag_kg.query.engine import QueryEngine

    engine = QueryEngine(config)
    stats = engine.get_stats()
    print(f"  Ready: {stats['ready']}")
    print(f"  Default method: {stats['default_method']}")
    print(f"  Cache: {stats['cache_stats']}")

    print("\n" + "=" * 60)
    print("Workflow complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Set GRAPHRAG_API_KEY in .env")
    print("  2. Start Neo4j: docker-compose up -d")
    print("  3. Run indexing: graphrag-kg index")
    print("  4. Sync to Neo4j: graphrag-kg graph sync")
    print("  5. Query: graphrag-kg query ask 'Your question'")


if __name__ == "__main__":
    main()
