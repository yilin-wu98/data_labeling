from datasets import load_dataset, concatenate_datasets, DatasetDict, Image, Value
import json 
# 1) Load
ds1 = load_dataset("/home/yilinwu/data/libero_10")
ds2 = load_dataset("/home/yilinwu/data/libero_90")

# 2) Cast image/index cols (adjust names if needed)
image_cols = ["obs_agentview_image", "obs_wrist_image"]
index_cols = ["task_index", "episode_index", "frame_index", "index"]

for col in image_cols:
    if col in ds1["train"].column_names: ds1 = ds1.cast_column(col, Image(decode=True))
    if col in ds2["train"].column_names: ds2 = ds2.cast_column(col, Image(decode=True))

# for col in index_cols:
#     if col in ds1["train"].column_names: ds1 = ds1.cast_column(col, Value("int64"))
#     if col in ds2["train"].column_names: ds2 = ds2.cast_column(col, Value("int64"))

# 3) Filter ds2 by allowed tasks (episode-level keep)
allowed_tasks = {7,8,19,29,54,61}  # <-- your set
# first find episodes that match
keep_eps = set(ds2["train"].filter(lambda r: r["task_index"] in allowed_tasks)["episode_index"])
# then keep only rows whose episode_index is in keep_eps
ds2_f = ds2["train"].filter(lambda r: r["episode_index"] in keep_eps)

# 4) Make episode_index consecutive: 0..E-1 (preserve original order)
unique_eps_sorted = sorted(set(ds2_f["episode_index"]))  # already increasing, but sort to be safe
ep_map = {old: new for new, old in enumerate(unique_eps_sorted)}
## dump ep_map for reference
json.dump(ep_map, open("ep_map.json", "w"), indent=4)

ds2_f = ds2_f.map(lambda r: {"episode_index": ep_map[r["episode_index"]]},
                  desc="Reindex episode_index to consecutive")

# 5) Make index consecutive: 0..N-1 over the filtered ds2
ds2_f = ds2_f.map(lambda r, i: {"index": i}, with_indices=True, desc="Reindex global index")

# # (Optional) ensure dtypes after reassignment
# ds2_f = ds2_f.cast_column("episode_index", Value("int64"))
# ds2_f = ds2_f.cast_column("index", Value("int64"))

# 6) Compute offsets from ds1 so ds2 starts after ds1
def safe_max(dset, col):
    return max(dset[col]) if (col in dset.column_names and dset.num_rows > 0) else -1

offsets = {col: safe_max(ds1["train"], col) + 1 for col in ["task_index", "episode_index", "index"]}
# NOTE: typically do NOT globally offset frame_index (it’s within-episode). Omit unless you truly want global frames.

def shift_indices(batch):
    for col, off in offsets.items():
        if col in batch:
            batch[col] = [x + off for x in batch[col]]
    return batch

ds2_shifted = ds2_f.map(shift_indices, batched=True, desc="Shifting indices after merge")

# 7) Schema match + concatenate
ds2_shifted = ds2_shifted.cast(ds1["train"].features)
merged_train = concatenate_datasets([ds1["train"], ds2_shifted])
merged = DatasetDict({"train": merged_train})
## 8) Save
merged.save_to_disk("libero-100-small")
