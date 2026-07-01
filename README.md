# GraphRAG-KG: Knowledge Graph Q&A System

A Python-based GraphRAG system for Knowledge Graph Q&A, built on Microsoft's `graphrag` library with **Neo4j + LanceDB** hybrid storage.

## Architecture

```
Documents (PDF/MD/TXT/HTML)
  → Ingest → graphrag Index → Parquet + LanceDB
  → Neo4j Sync → Cypher Graph Traversal
  → Query Engine (Local/Global/DRIFT/Basic)
  → Grounded Answers + Citations + Graph Context
```

## Quickstart

### 1. Install

```bash
pip install -e .
```

### 2. Start Neo4j

```bash
docker-compose up -d
```

### 3. Generate Test Data

```bash
graphrag-kg data generate --scenario pharma_supply_chain
```

### 4. Initialize Project

```bash
graphrag-kg init --name my-kg
```

### 5. Ingest Documents

```bash
graphrag-kg ingest run --source tests/fixtures/generated/pharma_supply_chain/documents
```

### 6. Build Index (requires LLM API key)

```bash
# Set your API key
export GRAPHRAG_API_KEY=your-key

# Run indexing
graphrag-kg index run --method standard
```

### 7. Sync to Neo4j

```bash
graphrag-kg graph sync
```

### 8. Query

```bash
graphrag-kg query ask "What drugs does Hengrui Medicine produce?"
graphrag-kg query ask "Which hospitals use both Hengrui and Qilu drugs?" --method drift
```

## CLI Commands

| Command | Description |
|---|---|
| `graphrag-kg data generate` | Generate test data with ground truth |
| `graphrag-kg init` | Initialize a new project |
| `graphrag-kg ingest run` | Ingest documents into input/ |
| `graphrag-kg index run` | Build knowledge graph index |
| `graphrag-kg graph sync` | Sync to Neo4j |
| `graphrag-kg graph status` | Neo4j graph statistics |
| `graphrag-kg query ask "Q"` | Ask questions with citations |
| `graphrag-kg serve` | Start REST API server |
| `graphrag-kg config show` | Show configuration |

## REST API

```bash
graphrag-kg serve --port 8000
```

Endpoints:
- `POST /query` — Ask a question
- `GET /stats` — System statistics
- `POST /index` — Trigger indexing
- `GET /index/status` — Index status
- `POST /graph/sync` — Sync to Neo4j
- `GET /graph/stats` — Graph statistics
- `GET /health` — Health check

API docs: http://localhost:8000/docs

## Configuration

Edit `settings.yaml` or use profiles:

```bash
graphrag-kg config profile list
graphrag-kg config profile apply fast
graphrag-kg config show
```

### Profiles

| Profile | Chat Model | Embedding | Query Method | Use Case |
|---|---|---|---|---|
| `default` | gpt-4.1 | text-embedding-3-large | local | General purpose |
| `fast` | gpt-4.1-mini | text-embedding-3-small | basic | Quick iterations |
| `production` | gpt-4.1 | text-embedding-3-large | drift | Full quality |

## Test Data Scenarios

| Scenario | Entities | Relationships | Query Hops | Use Case |
|---|---|---|---|---|
| `pharma_supply_chain` | 59 (12 types) | 153 | 3-5 hops | Multi-hop supply chain |
| `tech_company` | 21 (5 types) | 34 | 1-2 hops | Company ecosystem |

## Project Structure

```
src/graphrag_kg/
├── cli/          # Typer CLI commands
├── core/         # Config, pipeline, project management
├── data/         # Test data generator
├── ingest/       # Document parsers + loader
├── index/        # GraphRAG indexing wrapper
├── graph/        # Neo4j connection, sync, queries, traversal
├── query/        # Query engine + 4 search methods
├── storage/      # Parquet + LanceDB stores
├── api/          # FastAPI REST server
├── prompts/      # LLM prompt templates
└── utils/        # Logging, env, progress, validators
```

## Dependencies

- **graphrag** >= 3.0.0 — Microsoft GraphRAG library
- **neo4j** >= 5.20.0 — Neo4j Python driver
- **lancedb** — Vector embeddings
- **typer + rich** — CLI
- **fastapi + uvicorn** — REST API
- **pymupdf + beautifulsoup4** — Document parsing
- **faker + jinja2** — Test data generation
