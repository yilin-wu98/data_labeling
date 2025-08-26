# upload_libero_parquet.py
from huggingface_hub import HfApi, upload_folder, create_repo, hf_hub_url,upload_file
import os
from glob import glob

REPO_ID = "yilin-wu/libero-100-basket"                   # <-- change if needed
LOCAL_ROOT = "/home/yilinwu/Projects/data_labeling/libero-100-small-parquet-clean"  # folder that contains data/ and meta/
IS_DATASET = True

api = HfApi()

# 0) Ensure repo exists
create_repo(repo_id=REPO_ID, repo_type="dataset" if IS_DATASET else "model", exist_ok=True)

# 1) Upload .gitattributes first so parquet goes to LFS
gitattributes_path = os.path.join(LOCAL_ROOT, ".gitattributes")
if not os.path.exists(gitattributes_path):
    with open(gitattributes_path, "w") as f:
        f.write("""*.parquet filter=lfs diff=lfs merge=lfs -text
*.mp4 filter=lfs diff=lfs merge=lfs -text
""")

upload_file(
    path_or_fileobj=gitattributes_path,   # local file
    path_in_repo=".gitattributes",        # where it goes in repo
    repo_id=REPO_ID,
    repo_type="dataset",
    commit_message="Add .gitattributes for LFS",
)

# 2) Upload meta/ (small)
upload_folder(
    repo_id=REPO_ID,
    repo_type="dataset",
    folder_path=os.path.join(LOCAL_ROOT, "meta"),
    path_in_repo="meta",
    commit_message="Add meta files",
    allow_patterns=None,           # upload everything inside meta/
)

# 3) Upload each data/chunk-XXX as its own commit
data_root = os.path.join(LOCAL_ROOT, "data")
chunks = sorted([d for d in glob(os.path.join(data_root, "chunk-*")) if os.path.isdir(d)])

for chunk_dir in chunks:
    chunk_name = os.path.basename(chunk_dir)  # e.g., "chunk-000"
    print(f"Uploading {chunk_name} ...")
    upload_folder(
        repo_id=REPO_ID,
        repo_type="dataset",
        folder_path=chunk_dir,
        path_in_repo=f"data/{chunk_name}",
        commit_message=f"Add {chunk_name}",
        allow_patterns="*.parquet",      # only parquet files
        # ignore_patterns can be added if needed
    )

print("✅ Done. Repo:", hf_hub_url(REPO_ID, repo_type="dataset"))
