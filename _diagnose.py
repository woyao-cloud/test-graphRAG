"""Diagnose LanceDB and query pipeline."""
import lancedb
import pandas as pd
from pathlib import Path

output_dir = Path("D:/claude-code-project/graphRAG/output")
lancedb_path = output_dir / "lancedb"

print("=" * 60)
print("LANCE DB CHECK")
print("=" * 60)
db = lancedb.connect(str(lancedb_path))
tables = db.table_names()
print(f"Tables: {tables}")
for t in tables:
    tbl = db.open_table(t)
    count = tbl.count_rows()
    print(f"  {t}: {count} rows")
    if count > 0:
        sample = tbl.to_pandas().head(1)
        cols = list(sample.columns)
        print(f"    Columns: {cols}")
        if "vector" in cols:
            vec = sample["vector"].iloc[0]
            print(f"    Vector dim: {len(vec)}")
        if "text" in cols:
            print(f"    Text preview: {str(sample['text'].iloc(0))[:100]}" if count > 0 else "")

print()
print("=" * 60)
print("PARQUET CHECK")
print("=" * 60)
for name in ["entities", "relationships", "text_units", "community_reports"]:
    path = output_dir / f"{name}.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        print(f"{name}: {len(df)} rows")
    else:
        print(f"{name}: NOT FOUND")

print()
print("=" * 60)
print("CONFIG CHECK (to_graphrag_config)")
print("=" * 60)
import sys
sys.path.insert(0, "D:/claude-code-project/graphRAG/src")
from graphrag_kg.core.config_loader import ConfigLoader

loader = ConfigLoader(Path("D:/claude-code-project/graphRAG/settings.yaml"))
config = loader.load()
g_config = config.to_graphrag_config()
print(f"Vector store type: {g_config.vector_store.type}")
print(f"Vector store db_uri: {g_config.vector_store.db_uri}")
print(f"Vector size: {g_config.vector_store.vector_size}")
