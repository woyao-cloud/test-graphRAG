   Here is Claude's plan:
  ╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌
   GraphRAG Knowledge Graph Q&A System — Implementation Plan

   Context

   Building a Python-based GraphRAG system for Knowledge Graph Q&A, using Microsoft's graphrag library as the foundation. The system ingests multi-format documents (PDF, Markdown, TXT, HTML), builds a knowledge graph via entity/relationship extraction + community detection, stores the graph in Neo4j for rich traversal queries, and supports natural language querying with grounded, cited answers.

   Tech Stack

   - Python 3.10-3.12 with graphrag (Microsoft GraphRAG library)
Neo4j — primary graph database for entities, relationships, communities (Cypher queries + native vector index)
   - LanceDB — dedicated vector store for text/entity/community embeddings
   - Parquet — backup/export format (compatible with graphrag native output)
   - Typer + Rich for CLI, FastAPI for optional REST API
   - LiteLLM (bundled with graphrag) for multi-provider LLM access
   - pymupdf, beautifulsoup4 for document parsing
   - Docker — Neo4j runs via docker-compose for easy local setup

   Architecture Overview

   A thin wrapper around the graphrag library — all indexing delegates to graphrag internals. After indexing, graph data is synced to Neo4j for production querying. The value-add is: document ingestion, Neo4j graph storage, Cypher-powered graph traversal in queries, configuration management, CLI ergonomics, and observability.

   Hybrid Storage Architecture

                      ┌─────────────────────────────┐
                      │       Query Engine           │
                      │  (Local/Global/Drift/Basic)  │
                      └──────────┬──────────────────┘
                            │
                 ┌───────────────┼───────────────┐
                 │               │               │
                 ▼               ▼               ▼                                                                                                                                                             ↓
        ┌────────────┐  ┌──────────────┐  ┌──────────────┐
        │   Neo4j    │  │   LanceDB    │  │   Parquet    │                                                                                                                                                     ↓
        │  Graph DB  │  │  Vector Store│  │  (backup)    │
        ├────────────┤  ├──────────────┤  ├──────────────┤                                                                                                                                                     ↓
        │ Entities   │  │ Text unit    │  │ Full dataset │
        │ Relations  │  │ embeddings   │  │ export/import│                                                                                                                                                     ↓
        │ Communities│  │ Entity       │  │ graphrag     │
        │ Graph      │  │ embeddings   │  │ compatibility│                                                                                                                                                     ↓
        │ traversals │  │ Community    │  │              │
        │ Cypher     │  │ embeddings   │  │              │                                                                                                                                                     ↓
        └────────────┘  └──────────────┘  └──────────────┘
                                                                                                                                                                                                               ↓
   Data Flow
                                                                                                                                                                                                               ↓
   Documents (PDF/MD/TXT/HTML)
     │                                                                                                                                                                                                         ↓
     ├─ Phase 1: Ingestion ──────────────────────────────
     │   DocumentLoader → FormatParsers → DocumentConverter                                                                                                                                                    ↓
     │   → ./input/*.txt
     │                                                                                                                                                                                                         ↓
     ├─ Phase 2: Indexing (graphrag pipeline) ────────────
     │   ./input/*.txt                                                                                                                                                                                         ↓
     │   → graphrag.index pipeline
     │     (entity extraction → graph finalize →                                                                                                                                                               ↓
     │      community detection → embeddings)
     │   → ./output/*.parquet + ./output/lancedb/                                                                                                                                                              ↓
     │
     ├─ Phase 3: Neo4j Sync ─────────────────────────────                                                                                                                                                      ↓
     │   ./output/*.parquet
     │   → Neo4jGraphSync                                                                                                                                                                                      ↓
     │     (entities → Neo4j nodes,
     │      relationships → Neo4j edges,                                                                                                                                                                       ↓
     │      communities → Neo4j node properties + community nodes)
     │   → Neo4j Graph Database                                                                                                                                                                                ↓
     │
     └─ Phase 4: Query ──────────────────────────────────                                                                                                                                                      ↓
         User question
         → QueryEngine                                                                                                                                                                                         ↓
           ├─ Neo4j Cypher traversal (entity/relationship lookup)
           ├─ LanceDB vector search (semantic similarity)                                                                                                                                                      ↓
           └─ Parquet fallback (full community reports)
         → QueryResponse (answer + citations + sources + graph paths)                                                                                                                                          ↓

   Project Structure                                                                                                                                                                                           ↓

   D:\claude-code-project\graphRAG\                                                                                                                                                                            ↓
   ├── pyproject.toml
   ├── requirements.txt                                                                                                                                                                                        ↓
   ├── README.md
   ├── .env.example                                                                                                                                                                                            ↓
   ├── .gitignore
   ├── docker-compose.yml              # Neo4j + optional services                                                                                                                                             ↓
   ├── config/
   │   ├── default.yaml                                                                                                                                                                                        ↓
   │   ├── fast.yaml
   │   └── production.yaml                                                                                                                                                                                     ↓
   ├── examples/
   │   ├── basic_workflow.py                                                                                                                                                                                   ↓
   │   ├── custom_entities.py
   │   └── incremental_update.py                                                                                                                                                                               ↓
   ├── scripts/
   │   ├── setup_env.ps1 / setup_env.sh                                                                                                                                                                        ↓
   │   └── run_tests.ps1 / run_tests.sh
   ├── src/                                                                                                                                                                                                    ↓
   │   └── graphrag_kg/
   │       ├── __init__.py                                                                                                                                                                                     ↓
   │       ├── cli/
   │       │   ├── main.py              # CLI entry point (Typer)                                                                                                                                              ↓
   │       │   ├── utils.py
   │       │   └── commands/                                                                                                                                                                                   ↓
   │       │       ├── init_cmd.py      # graphrag-kg init
   │       │       ├── ingest_cmd.py    # graphrag-kg ingest                                                                                                                                                   ↓
   │       │       ├── index_cmd.py     # graphrag-kg index
   │       │       ├── query_cmd.py     # graphrag-kg query                                                                                                                                                    ↓
   │       │       ├── serve_cmd.py     # graphrag-kg serve
   │       │       ├── config_cmd.py    # graphrag-kg config                                                                                                                                                   ↓
   │       │       └── graph_cmd.py     # graphrag-kg graph (Neo4j management)
   │       ├── core/                                                                                                                                                                                           ↓
   │       │   ├── config.py            # KGConfig Pydantic model
   │       │   ├── config_loader.py     # YAML loading, env var resolution                                                                                                                                     ↓
   │       │   ├── pipeline.py          # Full pipeline orchestrator
   │       │   ├── project.py           # Project structure management                                                                                                                                         ↓
   │       │   └── errors.py            # Custom exceptions
   │       ├── ingest/                                                                                                                                                                                         ↓
   │       │   ├── loader.py            # DocumentLoader
   │       │   ├── parsers.py           # PDF/MD/TXT/HTML parsers                                                                                                                                              ↓
   │       │   ├── chunker.py           # Pre-chunking
   │       │   └── converter.py         # Convert to graphrag input format                                                                                                                                     ↓
   │       ├── index/
   │       │   ├── runner.py            # IndexRunner (wraps graphrag.index)                                                                                                                                   ↓
   │       │   ├── updater.py           # Incremental index updates
   │       │   ├── monitor.py           # Progress monitoring with Rich                                                                                                                                        ↓
   │       │   └── workflows.py         # Custom workflow registry
   │       ├── graph/                   # Neo4j graph operations                                                                                                                                               ↓
   │       │   ├── __init__.py
   │       │   ├── connection.py        # Neo4j driver + connection pool                                                                                                                                       ↓
   │       │   ├── sync.py              # Neo4jGraphSync: parquet → Neo4j
   │       │   ├── schema.py            # Neo4j schema/constraints/index setup                                                                                                                                 ↓
   │       │   ├── queries.py           # Cypher query library
   │       │   └── traversal.py         # Graph traversal utilities                                                                                                                                            ↓
   │       ├── query/
   │       │   ├── engine.py            # QueryEngine facade                                                                                                                                                   ↓
   │       │   ├── local.py             # Local search (Neo4j + LanceDB)
   │       │   ├── global_search.py     # Global search (community reports)                                                                                                                                    ↓
   │       │   ├── drift.py             # DRIFT search (hierarchical)
   │       │   ├── basic.py             # Basic vector search (LanceDB)                                                                                                                                        ↓
   │       │   ├── context.py           # Citation/source builder
   │       │   └── cache.py             # Query result caching                                                                                                                                                 ↓
   │       ├── storage/
   │       │   ├── repository.py        # Abstract storage interface                                                                                                                                           ↓
   │       │   ├── parquet_store.py     # Parquet file management (backup/export)
   │       │   ├── vector_store.py      # LanceDB wrapper                                                                                                                                                      ↓
   │       │   └── graph_store.py       # GraphML/networkx access (optional)
   │       ├── api/                     # Phase 5                                                                                                                                                              ↓
   │       │   ├── app.py               # FastAPI application
   │       │   ├── models.py                                                                                                                                                                                   ↓
   │       │   ├── middleware.py
   │       │   └── routes/                                                                                                                                                                                     ↓
   │       │       ├── query.py, index.py, health.py
   │       ├── prompts/                                                                                                                                                                                        ↓
   │       │   ├── templates.py
   │       │   └── defaults/            # Default prompt .txt files                                                                                                                                            ↓
   │       └── utils/
   │           ├── logging.py, env.py, progress.py, validators.py                                                                                                                                              ↓
   └── tests/
       ├── conftest.py                                                                                                                                                                                         ↓
       ├── fixtures/                    # Sample docs, configs
       ├── test_core/                                                                                                                                                                                          ↓
       ├── test_ingest/
       ├── test_index/                                                                                                                                                                                         ↓
       ├── test_graph/
       ├── test_query/                                                                                                                                                                                         ↓
       ├── test_cli/
       └── test_storage/                                                                                                                                                                                       ↓

   Neo4j Graph Schema                                                                                                                                                                                          ↓

   Node Types                                                                                                                                                                                                  ↓

   (:Entity {                                                                                                                                                                                                  ↓
       id: String,              # Unique entity ID (from graphrag)
       name: String,            # Human-readable entity name                                                                                                                                                   ↓
       type: String,            # Entity type (organization, person, location, etc.)
       description: String,     # LLM-generated description                                                                                                                                                    ↓
       degree: Integer,         # Connection count
       community_ids: [Integer],# Communities this entity belongs to                                                                                                                                           ↓
       text_unit_ids: [String], # Source text units
       embedding: [Float],      # Vector embedding (optional, also in LanceDB)                                                                                                                                 ↓
       human_readable_id: Integer
   })                                                                                                                                                                                                          ↓

   (:Community {                                                                                                                                                                                               ↓
       id: Integer,             # Community ID (from Leiden clustering)
       title: String,           # Community title                                                                                                                                                              ↓
       level: Integer,          # Hierarchical level
       summary: String,         # Community summary                                                                                                                                                            ↓
       full_content: String,    # Full community report
       rating: Float,           # Community importance rating                                                                                                                                                  ↓
       entity_count: Integer,
       parent_community_id: Integer                                                                                                                                                                            ↓
   })
                                                                                                                                                                                                               ↓
   (:Document {
       id: String,                                                                                                                                                                                             ↓
       title: String,
       file_path: String,                                                                                                                                                                                      ↓
       text_unit_count: Integer
   })                                                                                                                                                                                                          ↓

   (:TextUnit {                                                                                                                                                                                                ↓
       id: String,
       text: String,                                                                                                                                                                                           ↓
       document_id: String,
       entity_ids: [String],                                                                                                                                                                                   ↓
       community_ids: [Integer]
   })                                                                                                                                                                                                          ↓

   Relationship Types                                                                                                                                                                                          ↓

   (:Entity)-[:RELATES_TO {                                                                                                                                                                                    ↓
       id: String,
       description: String,                                                                                                                                                                                    ↓
       weight: Float,
       text_unit_ids: [String],                                                                                                                                                                                ↓
       human_readable_id: Integer
   }]->(:Entity)                                                                                                                                                                                               ↓

   (:Entity)-[:BELONGS_TO]->(:Community)                                                                                                                                                                       ↓
   (:Community)-[:PARENT_OF]->(:Community)     # Hierarchical
   (:Entity)-[:MENTIONED_IN]->(:TextUnit)                                                                                                                                                                      ↓
   (:TextUnit)-[:PART_OF]->(:Document)
   (:Entity)-[:CO_OCCURS_WITH {                                                                                                                                                                                ↓
       count: Integer
   }]->(:Entity)                                # For graph analytics                                                                                                                                          ↓

   Key Cypher Queries                                                                                                                                                                                          ↓

   cypher                                                                                                                                                                                                      ↓
   // Find all entities connected to a given entity within N hops
   MATCH path = (e:Entity {name: $name})-[*1..3]-(connected)                                                                                                                                                   ↓
   RETURN path
                                                                                                                                                                                                               ↓
   // Get entity with its community context
   MATCH (e:Entity {name: $name})-[:BELONGS_TO]->(c:Community)                                                                                                                                                 ↓
   RETURN e, c
                                                                                                                                                                                                               ↓
   // Find shortest path between two entities
   MATCH path = shortestPath((a:Entity {name: $a})-[*]-(b:Entity {name: $b}))                                                                                                                                  ↓
   RETURN path
                                                                                                                                                                                                               ↓
   // Get community hierarchy
   MATCH (parent:Community)-[:PARENT_OF*]->(child:Community)                                                                                                                                                   ↓
   WHERE child.id = $community_id
   RETURN parent, child                                                                                                                                                                                        ↓

   // Hybrid: vector similarity + graph traversal                                                                                                                                                              ↓
   // 1. LanceDB finds top-k similar entities by embedding
   // 2. Neo4j traverses their neighborhoods                                                                                                                                                                   ↓

   CLI Design                                                                                                                                                                                                  ↓

   graphrag-kg init        # Initialize new project (incl. docker-compose.yml)                                                                                                                                 ↓
   graphrag-kg ingest      # Load documents → input/
   graphrag-kg index       # Build knowledge graph (graphrag pipeline)                                                                                                                                         ↓
   graphrag-kg graph sync  # Sync parquet data → Neo4j
   graphrag-kg graph status# Show Neo4j graph stats                                                                                                                                                            ↓
   graphrag-kg graph drop  # Clear Neo4j database
   graphrag-kg query "Q"   # Ask questions with citations + graph context                                                                                                                                      ↓
   graphrag-kg serve       # Start REST API (Phase 5)
   graphrag-kg config      # Manage configuration profiles                                                                                                                                                     ↓
   graphrag-kg status      # Project stats
                                                                                                                                                                                                               ↓
   Configuration
                                                                                                                                                                                                               ↓
   Neo4j Section in settings.yaml
                                                                                                                                                                                                               ↓
   kg:
     project_name: "my-knowledge-graph"                                                                                                                                                                        ↓

     neo4j:                                                                                                                                                                                                    ↓
       uri: "${NEO4J_URI:bolt://localhost:7687}"
       username: "${NEO4J_USERNAME:neo4j}"                                                                                                                                                                     ↓
       password: "${NEO4J_PASSWORD:password}"
       database: "neo4j"                                                                                                                                                                                       ↓
       # Connection pool
       max_connection_lifetime: 3600                                                                                                                                                                           ↓
       max_connection_pool_size: 50
       connection_acquisition_timeout: 60                                                                                                                                                                      ↓
       # Sync settings
       sync_batch_size: 1000                                                                                                                                                                                   ↓
       sync_create_indexes: true
       # Optional: store embeddings in Neo4j too (for hybrid ANN)                                                                                                                                              ↓
       store_embeddings: false  # default: use LanceDB for vectors
                                                                                                                                                                                                               ↓
     vector_store:
       type: lancedb                                                                                                                                                                                           ↓
       db_uri: "output/lancedb"
                                                                                                                                                                                                               ↓
   docker-compose.yml
                                                                                                                                                                                                               ↓
   version: '3.8'
   services:                                                                                                                                                                                                   ↓
     neo4j:
       image: neo4j:5.22.0                                                                                                                                                                                     ↓
       ports:
         - "7474:7474"  # HTTP                                                                                                                                                                                 ↓
         - "7687:7687"  # Bolt
       environment:                                                                                                                                                                                            ↓
         - NEO4J_AUTH=neo4j/password
         - NEO4J_PLUGINS=["apoc"]                                                                                                                                                                              ↓
       volumes:
         - neo4j_data:/data                                                                                                                                                                                    ↓
         - neo4j_logs:/logs
       healthcheck:                                                                                                                                                                                            ↓
         test: ["CMD", "cypher-shell", "-u", "neo4j", "-p", "password", "RETURN 1"]
         interval: 10s                                                                                                                                                                                         ↓
         timeout: 5s
         retries: 5                                                                                                                                                                                            ↓

   volumes:                                                                                                                                                                                                    ↓
     neo4j_data:
     neo4j_logs:                                                                                                                                                                                               ↓

   Implementation Phases                                                                                                                                                                                       ↓

   Phase 1: Project Setup & Foundation                                                                                                                                                                         ↓

   - pyproject.toml, requirements.txt, .gitignore, .env.example                                                                                                                                                ↓
   - docker-compose.yml for Neo4j
   - core/config.py — KGConfig Pydantic model with Neo4j settings                                                                                                                                              ↓
   - core/config_loader.py — YAML + env var resolution
   - core/project.py — Project directory init (incl. docker-compose)                                                                                                                                           ↓
   - core/errors.py — Custom exceptions
   - utils/ — logging, env helpers                                                                                                                                                                             ↓
   - config/ — default/fast/production YAML templates
   - Tests for config loading and project init                                                                                                                                                                 ↓

   Phase 2: Document Ingestion                                                                                                                                                                                 ↓

   - ingest/parsers.py — PDF (pymupdf), MD, TXT, HTML (beautifulsoup4)                                                                                                                                         ↓
   - ingest/loader.py — DocumentLoader with file discovery
   - ingest/converter.py — Normalize to graphrag input format                                                                                                                                                  ↓
   - cli/commands/ingest_cmd.py + wiring
   - Tests with sample documents of each format                                                                                                                                                                ↓

   Phase 3: Indexing Pipeline + Neo4j Integration                                                                                                                                                              ↓

   - index/runner.py — wraps graphrag.index.run_pipeline_with_config()                                                                                                                                         ↓
   - index/monitor.py — Rich progress bars via graphrag callbacks
   - index/updater.py — incremental updates (standard-update, fast-update)                                                                                                                                     ↓
   - storage/parquet_store.py + storage/vector_store.py (LanceDB)
   - graph/connection.py — Neo4j driver, connection pool, health check                                                                                                                                         ↓
   - graph/schema.py — Create indexes, constraints, node/relationship types
   - graph/sync.py — Neo4jGraphSync: read parquet → batch-insert into Neo4j                                                                                                                                    ↓
   - graph/queries.py — Cypher query library (entity lookup, traversal, paths)
   - cli/commands/init_cmd.py + cli/commands/index_cmd.py + cli/commands/graph_cmd.py                                                                                                                          ↓
   - End-to-end test: init → ingest → index → graph sync
                                                                                                                                                                                                               ↓
   Phase 4: Query Engine
                                                                                                                                                                                                               ↓
   - query/local.py — Local search with Neo4j entity expansion + LanceDB text retrieval
   - query/global_search.py — Community report map-reduce                                                                                                                                                      ↓
   - query/drift.py — Hierarchical search with Neo4j community traversal
   - query/basic.py — Pure vector search via LanceDB                                                                                                                                                           ↓
   - query/engine.py — QueryEngine facade with auto-routing
   - query/context.py — Citation extraction, graph path formatting                                                                                                                                             ↓
   - query/cache.py — Result caching
   - graph/traversal.py — Advanced graph traversal (shortest path, ego network, community neighborhood)                                                                                                        ↓
   - cli/commands/query_cmd.py + cli/commands/config_cmd.py
   - Prompt templates in prompts/defaults/                                                                                                                                                                     ↓
   - Tests for routing logic, Cypher queries, and context building
                                                                                                                                                                                                               ↓
   Phase 5: Advanced Features
                                                                                                                                                                                                               ↓
   - api/ — FastAPI REST server with /query, /index, /health, /stats, /graph
   - Streaming response support                                                                                                                                                                                ↓
   - core/pipeline.py — full pipeline orchestrator
   - Example scripts in examples/                                                                                                                                                                              ↓
   - Comprehensive README with quickstart
   - Performance profiling (Neo4j query optimization, batch sync tuning)                                                                                                                                       ↓

   Key Dependencies                                                                                                                                                                                            ↓

   # Core                                                                                                                                                                                                      ↓
   graphrag>=3.0.0,<4.0.0
                                                                                                                                                                                                               ↓
   # Graph Database
   neo4j>=5.20.0              # Neo4j Python driver                                                                                                                                                            ↓

   # CLI and UI                                                                                                                                                                                                ↓
   typer>=0.12.0
   rich>=13.0.0                                                                                                                                                                                                ↓

   # Configuration                                                                                                                                                                                             ↓
   pyyaml>=6.0
   pydantic>=2.0.0                                                                                                                                                                                             ↓
   pydantic-settings>=2.0.0
                                                                                                                                                                                                               ↓
   # Document parsing
   pymupdf>=1.23.0                                                                                                                                                                                             ↓
   beautifulsoup4>=4.12.0
   chardet>=5.0.0                                                                                                                                                                                              ↓

   # Storage                                                                                                                                                                                                   ↓
   pyarrow>=14.0.0
   pandas>=2.0.0                                                                                                                                                                                               ↓
   lancedb>=0.6.0
                                                                                                                                                                                                               ↓
   # API (Phase 5)
   fastapi>=0.110.0                                                                                                                                                                                            ↓
   uvicorn[standard]>=0.27.0
                                                                                                                                                                                                               ↓
   # Development
   pytest>=8.0.0                                                                                                                                                                                               ↓
   pytest-asyncio>=0.23.0
   pytest-mock>=3.12.0                                                                                                                                                                                         ↓
   mypy>=1.8.0
   ruff>=0.3.0                                                                                                                                                                                                 ↓

   Verification Strategy                                                                                                                                                                                       ↓

   - Phase 1: Unit tests pass for config loading, project init. docker-compose up starts Neo4j.                                                                                                                ↓
   - Phase 2: Ingest test documents, verify text files in input/.
   - Phase 3: Full pipeline: init → ingest → index → graph sync. Verify Neo4j contains all entities, relationships, communities. Run Cypher queries to confirm.                                                ↓
   - Phase 4: Query against indexed corpus with all four methods. Verify answers include both citations AND graph context (entity relationships, community info from Neo4j).
   - Phase 5: Full end-to-end: init → ingest → index → graph sync → query (CLI) → query (REST API). Walk through README quickstart.