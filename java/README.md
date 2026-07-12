Java PoC: langchain4j + Milvus + Neo4j

This PoC provides a minimal Java project skeleton that demonstrates:

- connectivity to Neo4j (runs a simple test query)
- basic reachability check for Milvus (port test)
- project structure for later integration with langchain4j and a Milvus adapter

Files:
- docker-compose.yml — launches Milvus and Neo4j for local testing
- build.gradle — Gradle build file
- src/main/java/com/graphrag/poc/App.java — minimal PoC app

Quick start

1. Start Milvus and Neo4j:

```bash
docker compose -f java/docker-compose.yml up -d
```

2. Build and run the PoC (requires Gradle installed):

```bash
cd java
gradle run
```

Notes
- This PoC intentionally keeps external SDK usage minimal to remain buildable without additional native dependencies. It verifies Neo4j connectivity and Milvus reachability. Next steps: add langchain4j usage and implement a proper Milvus adapter.
