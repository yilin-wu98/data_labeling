# scrub_parquet_metadata.py
import os
import pyarrow as pa
import pyarrow.parquet as pq

SRC = "./libero-100-small-parquet/data"      # folder containing chunk-000/, chunk-001/, ...
DST = "./libero-100-small-parquet-clean/data"  # write to a clean copy (safer)

for root, _, files in os.walk(SRC):
    rel = os.path.relpath(root, SRC)
    outdir = os.path.join(DST, rel)
    os.makedirs(outdir, exist_ok=True)

    for fn in files:
        if not fn.endswith(".parquet"):
            continue
        src_path = os.path.join(root, fn)
        dst_path = os.path.join(outdir, fn)

        table = pq.read_table(src_path)                    # read as Arrow Table
        table = table.replace_schema_metadata({})          # drop ALL schema metadata
        pq.write_table(table, dst_path)                    # write clean Parquet

print("Done. Cleaned Parquets in:", DST)
