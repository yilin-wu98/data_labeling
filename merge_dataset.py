## merge two huggingface dataset into one.

import os
import shutil
from datasets import load_dataset, concatenate_datasets
import imageio
from tqdm import tqdm
import json
import numpy as np
from datasets import load_dataset, concatenate_datasets, DatasetDict, Image, Value
def load_task_meta(path: str):

# 2) Read line-by-line, parse JSON, and insert into a dict
    task_dict = {}
    with open(path, "r") as f:
        for line in f:
            entry = json.loads(line)            # e.g. {"task_index": 35, "task": "..."}
            idx   = entry["task_index"]         # an integer
            desc  = entry["task"]               # the text string
            task_dict[idx] = desc
    return task_dict
def load_episode_meta(path: str):
    episode_dict = {}
    with open(path, "r") as f:
        for line in f:
            entry = json.loads(line)            # e.g. {"episode_index": 35, "task_index": 0, "video": "..."}
            idx   = entry["episode_index"]         # an integer
            tasks  = entry["tasks"] 
            length = entry["length"]
            episode_dict[idx] = (tasks, length)
    return episode_dict
def main(
    local_cache_dir: str = os.path.expanduser("/home/yilinwu/data"),
    dataset_name: str = "libero_10",
    added_dataset_name: str = "libero_90",
    split: str = "train",
    num_episodes: int = 10,
    output_folder: str = "exported_data",
):
    """
    1. Loads the “physical-intelligence/libero” dataset from local cache.
    2. Iterates over the first `num_episodes` in the specified `split`.
    3. For each record, locates the video file and the language instruction, then saves:
       - Video MP4 under `output_folder/videos/episode_<idx>.mp4`
       - Instruction text under `output_folder/instructions/episode_<idx>.txt`
    """

    # Ensure output subfolders exist
    videos_dir = os.path.join(output_folder, "videos_fps10")
    instr_dir = os.path.join(output_folder, "instructions")
    os.makedirs(videos_dir, exist_ok=True)
    os.makedirs(instr_dir, exist_ok=True)
    added_episode_meta = load_episode_meta(os.path.join(local_cache_dir, "libero_90/meta/episodes.jsonl"))
    added_task_meta = load_task_meta(os.path.join(local_cache_dir, "libero_90/meta/tasks.jsonl"))
    task_meta = load_task_meta(os.path.join(local_cache_dir, "libero_10/meta/tasks.jsonl"))
    episode_meta = load_episode_meta(os.path.join(local_cache_dir, "libero_10/meta/episodes.jsonl"))
    ## merge the task_meta and episode meta
    new_episode_meta = episode_meta.copy()
    new_task_meta = task_meta.copy()
    # breakpoint()
    for k, v in added_task_meta.items():
        new_task_meta[k + len(task_meta)] = v
        ## replace some task description
        if v == "close the microwave":
            new_task_meta[k + len(task_meta)] = "move away the yellow and white mug to close the microwave door"
        if 'basket' in v: 
            ## every sentence is like "pick up xxx and put it in the basket", extract xxx from v
            object_name = v.split('pick up ')[1].split(' and put it in the basket')[0]
            new_task_meta[k + len(task_meta)] = f"put {object_name} in the basket"
        if 'put it in the tray' in v :
            object_name = v.split('pick up ')[1].split(' and put it in the tray')[0]
            new_task_meta[k + len(task_meta)] = f"put {object_name} in the tray"
    # breakpoint()
    for k, v in added_episode_meta.items():
        # new_episode_meta[k + len(episode_meta)] = v
        desc = v[0]
        ## replace some task description
        if desc == "close the microwave":
            v[0] = "move away the yellow and white mug to close the microwave door"
        if 'basket' in desc: 
            ## every sentence is like "pick up xxx and put it in the basket", extract xxx from v
            object_name = v.split('pick up ')[1].split(' and put it in the basket')[0]
            v[0] = f"put {object_name} in the basket"
        if 'put it in the tray' in desc :
            object_name = v.split('pick up ')[1].split(' and put it in the tray')[0]
            v[0] = f"put {object_name} in the tray"
        new_episode_meta[k + len(episode_meta)] = v
   
    ## save the new meta data to local cache
    with open(os.path.join(local_cache_dir, "libero_100/meta/episodes.jsonl"), "w") as f:
        for k, v in new_episode_meta.items():
            f.write(json.dumps({"episode_index": k, "tasks": v[0], "length": v[1]}) + "\n")
    with open(os.path.join(local_cache_dir, "libero_100/meta/tasks.jsonl"), "w") as f:
        for k, v in new_task_meta.items():
            f.write(json.dumps({"task_index": k, "task": v}) + "\n")
    # Load the dataset from cache
    print(f"Loading '{dataset_name}' (split='{split}') from cache_dir={local_cache_dir} …")
    ## load from disk with local_cache_dir and the dataset name
    ds1 = load_dataset(os.path.join(local_cache_dir, dataset_name))
    print(f"✅ Dataset loaded. Number of episodes in this split: {len(ds1)}")
    # ## get the last episode_index and last task_index 
    # last_episode_index = max(ds['train']['episode_index'])
    # last_task_index = max(ds['train']['task_index'])
    # last_frame_index = max(ds['train']['frame_index'])
    # last_index = max(ds['train']['index'])
    # print(f"Last episode index: {last_episode_index}, Last task index: {last_task_index}") 
    ds2 = load_dataset(os.path.join(local_cache_dir, added_dataset_name))
    print(f"✅ Added Dataset loaded. Number of episodes in this split: {len(ds2)}")
    image_cols = ["image", "wrist_image"]  # extend as needed
    for col in image_cols:
        if col in ds1["train"].column_names:
            ds1 = ds1.cast_column(col, Image(decode=True))
        if col in ds2["train"].column_names:
            ds2 = ds2.cast_column(col, Image(decode=True))

    # 3) Ensure index columns are int64 on BOTH datasets (so arithmetic is safe)
    index_cols = ["task_index", "episode_index", "frame_index", "index"]  # add/remove as needed
    # for col in index_cols:
    #     if col in ds1["train"].column_names:
    #         ds1 = ds1.cast_column(col, Value("int64"))
    #     if col in ds2["train"].column_names:
    #         ds2 = ds2.cast_column(col, Value("int64"))

    # 4) Compute offsets from ds1 (so ds2 indices start AFTER ds1’s last values)
    # def safe_max(dset, col):
    #     if col in dset.column_names and dset.num_rows > 0:
    #         return dset.max(col)
    #     return -1  # so first value becomes 0 after +1
    def safe_max(dset, col):
        if col in dset.column_names and dset.num_rows > 0:
            # use Dataset[col] which gives a list-like of that column
            return max(dset[col])
        return -1

    offsets = {
        col: safe_max(ds1["train"], col) + 1
        for col in index_cols
    }

    # 5) Shift ds2 indices in a single pass (batched map = fast)
    def shift_indices(batch):
        for col, off in offsets.items():
            if col in batch:
                # batch[col] is a list; add offset elementwise
                batch[col] = [x + off for x in batch[col]]
        return batch

    ds2_shifted = ds2["train"].map(shift_indices, batched=True, desc="Shifting index columns")

    # # 6) (Optional) Make sure schemas match before concatenation.
    # #    If ds2 is missing some columns that ds1 has, add dummies first.
    # missing_in_ds2 = [c for c in ds1["train"].column_names if c not in ds2_shifted.column_names]
    # if missing_in_ds2:
    #     # Provide sensible defaults per type (customize if needed)
    #     defaults = {}
    #     f1 = ds1["train"].features
    #     for c in missing_in_ds2:
    #         if c in image_cols:
    #             # For Image feature, you usually don't want a dummy; instead consider dropping or filling with a real path/bytes.
    #             # If you must fill, you can put None and cast after, but many pipelines expect real images.
    #             defaults[c] = [None] * len(ds2_shifted)
    #         elif isinstance(f1[c], Value) and f1[c].dtype in ("int64", "int32"):
    #             defaults[c] = [0] * len(ds2_shifted)
    #         elif isinstance(f1[c], Value) and f1[c].dtype in ("float64", "float32"):
    #             defaults[c] = [0.0] * len(ds2_shifted)
    #         else:
    #             defaults[c] = [None] * len(ds2_shifted)
    #     ds2_shifted = ds2_shifted.with_columns(defaults)

    # # (Optional but recommended) cast ds2 to ds1’s exact features to avoid tiny schema mismatches
    # ds2_shifted = ds2_shifted.cast(ds1["train"].features)

    # 7) Concatenate
    merged_train = concatenate_datasets([ds1["train"], ds2_shifted])
    merged = DatasetDict({"train": merged_train})
    ## filter out part of the added_ds with only certin_task_index  
    # certain_task_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # new_task_list = []
    # for item in added_ds['train']:
    #     # if item['task_index'] in certain_task_index:
    #         ## change the episode index to continue from ds 
    #         # if item['task_index'] not in new_task_list:
    #             # new_task_list.append(item['task_index'])
    #             # item['task_index'] = last_task_index + len(new_task_list)
    #         item['task_index'] = last_task_index + item['task_index'] + 1
    #         item['episode_index'] = last_episode_index + item['episode_index'] + 1
    #         item['frame_index'] = item['frame_index'] + last_frame_index + 1
    #         item['index'] = item['index'] + last_index + 1
    #         ds['train'] = ds['train'].add_item(item)
   
    ## save the new dataset to local cache
    # ds.save_to_disk(os.path.join(local_cache_dir, "libero_100"))
    merged.save_to_disk(os.path.join(local_cache_dir,"libero_100"))
    print(f"New dataset saved to {os.path.join(local_cache_dir, 'libero_100')}")

if __name__ == "__main__":
    main()