from tqdm import tqdm
import json
import os
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
local_cache_dir = './'
added_episode_meta = load_episode_meta(os.path.join(local_cache_dir, "libero_90/episodes.jsonl"))
added_task_meta = load_task_meta(os.path.join(local_cache_dir, "libero_90/tasks.jsonl"))
task_meta = load_task_meta(os.path.join(local_cache_dir, "libero_10/tasks.jsonl"))
episode_meta = load_episode_meta(os.path.join(local_cache_dir, "libero_10/episodes.jsonl"))
## load_episode data mapping
ep_map = json.loads(''.join(open(os.path.join(local_cache_dir, "filter_ep_data/ep_map.json"), "r").readlines()))
task_map = json.loads(''.join(open(os.path.join(local_cache_dir, "filter_ep_data/task_map.json"), "r").readlines()))
# breakpoint()
## create filtered_task_meta and filtered_episode_meta
filtered_task_meta = {}
filtered_episode_meta = {}

## merge the task_meta and episode meta
new_episode_meta = episode_meta.copy()
new_task_meta = task_meta.copy()
# breakpoint()
for k, v in added_task_meta.items():
    # if str(k) not in task_map.keys():
    #     continue
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
    # if str(k) not in ep_map.keys():
    #     continue
    # new_episode_meta[k + len(episode_meta)] = v
    desc = v[0][0]
    v_new = list(v)
    # breakpoint()
    ## replace some task description
    if desc == "close the microwave":
        v_new[0] = ["move away the yellow and white mug to close the microwave door"]
    if 'basket' in desc: 
        ## every sentence is like "pick up xxx and put it in the basket", extract xxx from v
        object_name = desc.split('pick up ')[1].split(' and put it in the basket')[0]
        v_new[0] = [f"put {object_name} in the basket"]
    if 'put it in the tray' in desc :
        object_name = desc.split('pick up ')[1].split(' and put it in the tray')[0]
        v_new[0] = [f"put {object_name} in the tray"]
    new_episode_meta[k + len(episode_meta)] = (v_new[0], v_new[1])

## save the new meta data to local cache
with open(os.path.join(local_cache_dir, "libero-100/meta/episodes.jsonl"), "w") as f:
    for k, v in new_episode_meta.items():
        f.write(json.dumps({"episode_index": k, "tasks": v[0], "length": v[1]}) + "\n")
with open(os.path.join(local_cache_dir, "libero-100/meta/tasks.jsonl"), "w") as f:
    for k, v in new_task_meta.items():
        f.write(json.dumps({"task_index": k, "task": v}) + "\n")