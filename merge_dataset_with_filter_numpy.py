import os
import numpy as np
from datasets import load_dataset, concatenate_datasets, DatasetDict, Image, Value
import json
# 1) Load
ds1 = load_dataset("/home/yilinwu/data/libero_10")
ds2 = load_dataset("/home/yilinwu/data/libero_90")


# 2) Make image columns non-decoding to avoid heavy I/O during ops
image_cols = ["image", "wrist_image"]  # adjust to your schema
for col in image_cols:
    if col in ds1["train"].column_names:
        ds1 = ds1.cast_column(col, Image(decode=False))
    if col in ds2["train"].column_names:
        ds2 = ds2.cast_column(col, Image(decode=False))

# 3) Ensure integer columns are ints
index_cols = ["task_index", "episode_index", "frame_index", "index"]
# for col in index_cols:
#     if col in ds1["train"].column_names: ds1 = ds1.cast_column(col, Value("int64"))
#     if col in ds2["train"].column_names: ds2 = ds2.cast_column(col, Value("int64"))

ds2_train = ds2["train"]

# ---------- FAST EPISODE-LEVEL FILTER ----------
# Keep only episodes whose task_index is in allowed_tasks
allowed_tasks = {7,8,19,29,54,61}  # <-- your set

# Pull the two small int columns as NumPy (cheap; no image decoding)
eps_arr   = np.asarray(ds2_train["episode_index"], dtype=np.int64)
tasks_arr = np.asarray(ds2_train["task_index"],   dtype=np.int64)

# Episodes to keep (unique episode_index values that have an allowed task)
keep_eps = np.unique(eps_arr[np.isin(tasks_arr, list(allowed_tasks))])

# Build a boolean mask for all rows, then get row indices to keep
mask = np.isin(eps_arr, keep_eps)
keep_indices = np.nonzero(mask)[0].tolist()

# Use Arrow slicing via select (much faster than Pythonic filter)
# (optional) write an indices cache file to speed repeated runs
ds2_f = ds2_train.select(
    keep_indices,
    keep_in_memory=False,
    # indices_cache_file_name="ds2_keep_indices.arrow"
)
# -----------------------------------------------

# ---------- MAKE INDICES CONSECUTIVE (FAST) ----------
# Reindex episode_index to 0..E-1, preserving order of first appearance
unique_eps_sorted = np.unique(np.asarray(ds2_f["episode_index"], dtype=np.int64))
ep_map = {old: new for new, old in enumerate(unique_eps_sorted)}
## dump ep_map for reference
ep_map_serializable = {int(k): int(v) for k, v in ep_map.items()}
os.makedirs("filter_ep_data", exist_ok=True)
with open("filter_ep_data/ep_map.json", "w") as f:
    json.dump(ep_map_serializable, f, indent=2)


ds2_f = ds2_f.map(lambda r: {"episode_index": ep_map[r["episode_index"]]},
                  desc="Reindex episode_index",
                  num_proc=os.cpu_count())  # parallel

# ------------------ Reindex task_index ------------------
unique_tasks_sorted = np.unique(np.asarray(ds2_f["task_index"], dtype=np.int64))
task_map = {int(old): i for i, old in enumerate(unique_tasks_sorted)}
task_map_serializable = {int(k): int(v) for k, v in task_map.items()}
with open("filter_ep_data/task_map.json", "w") as f:
    json.dump(task_map_serializable, f, indent=2)

ds2_f = ds2_f.map(lambda r: {"task_index": task_map[int(r["task_index"])]},
                  desc="Reindex task_index",
                  num_proc=os.cpu_count())

# Make global row index consecutive 0..N-1
ds2_f = ds2_f.map(lambda r, i: {"index": i},
                  with_indices=True,
                  desc="Reindex global index",
                  num_proc=os.cpu_count())

# # Cast back to int64 (map may infer int32 on some systems)
# ds2_f = ds2_f.cast_column("episode_index", Value("int64"))
# ds2_f = ds2_f.cast_column("index", Value("int64"))
# ------------------------------------------------------

# ---------- OFFSET & MERGE ----------
def safe_max(dset, col):
    return max(dset[col]) if (col in dset.column_names and dset.num_rows > 0) else -1

offsets = {
    "task_index":    safe_max(ds1["train"], "task_index") + 1,
    "episode_index": safe_max(ds1["train"], "episode_index") + 1,
    "index":         safe_max(ds1["train"], "index") + 1,
    # tip: usually do NOT offset frame_index (it’s per-episode)
}

def shift_indices(batch):
    for col, off in offsets.items():
        if col in batch:
            batch[col] = [x + off for x in batch[col]]
    return batch

ds2_shifted = ds2_f.map(shift_indices, batched=True, desc="Shift ds2 indices")

# Align schemas & concatenate
# ds2_shifted = ds2_shifted.cast(ds1["train"].features)
merged_train = concatenate_datasets([ds1["train"], ds2_shifted])
merged = DatasetDict({"train": merged_train})
# ------------------------------------
merged.save_to_disk("libero-100-small")

print('Saved!')

# Optional: save
# merged.save_to_disk("libero_100_merged_filtered")
