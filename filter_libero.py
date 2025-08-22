import json
import os
import shutil
from datasets import load_dataset, DatasetDict
import numpy as np
from PIL import Image as PILImage
from tqdm import tqdm
# from lerobot.datasets.utils import load_image_as_numpy
from pathlib import Path
# def load_image_as_numpy(
#     fpath: str | Path, dtype: np.dtype = np.float32, channel_first: bool = True
# ) -> np.ndarray:
#     img = PILImage.open(fpath).convert("RGB")
#     img_array = np.array(img, dtype=dtype)
#     if channel_first:  # (H, W, C) -> (C, H, W)
#         img_array = np.transpose(img_array, (2, 0, 1))
#     if np.issubdtype(dtype, np.floating):
#         img_array /= 255.0
#     return img_array
# def estimate_num_samples(
#     dataset_len: int, min_num_samples: int = 100, max_num_samples: int = 10_000, power: float = 0.75
# ) -> int:
#     """Heuristic to estimate the number of samples based on dataset size.
#     The power controls the sample growth relative to dataset size.
#     Lower the power for less number of samples.

#     For default arguments, we have:
#     - from 1 to ~500, num_samples=100
#     - at 1000, num_samples=177
#     - at 2000, num_samples=299
#     - at 5000, num_samples=594
#     - at 10000, num_samples=1000
#     - at 20000, num_samples=1681
#     """
#     if dataset_len < min_num_samples:
#         min_num_samples = dataset_len
#     return max(min_num_samples, min(int(dataset_len**power), max_num_samples))
# def compute_episode_stats(dataset, features: dict) -> dict:
#     ep_stats = {}
#     ## logging the progress
#     for key in tqdm(dataset.column_names):
#         if features[key]["dtype"] == "string":
#             continue  # HACK: we should receive np.arrays of strings
#         elif features[key]["dtype"] in ["image", "video"]:
#             # Get image paths from dataset
#             image_paths = dataset[key]
#             ep_ft_array = sample_images(image_paths)  # data is a list of image paths
#             axes_to_reduce = (0, 2, 3)  # keep channel dim
#             keepdims = True
#         else:
#             # Convert dataset column to numpy array
#             ep_ft_array = np.array(dataset[key])
#             axes_to_reduce = 0  # compute stats over the first axis
#             keepdims = ep_ft_array.ndim == 1  # keep as np.array

#         ep_stats[key] = get_feature_stats(ep_ft_array, axis=axes_to_reduce, keepdims=keepdims)

#         # finally, we normalize and remove batch dim for images
#         if features[key]["dtype"] in ["image", "video"]:
#             ep_stats[key] = {
#                 k: v if k == "count" else np.squeeze(v / 255.0, axis=0) for k, v in ep_stats[key].items()
#             }

#     return ep_stats
# def sample_indices(data_len: int) -> list[int]:
#     num_samples = estimate_num_samples(data_len)
#     return np.round(np.linspace(0, data_len - 1, num_samples)).astype(int).tolist()


# def auto_downsample_height_width(img: np.ndarray, target_size: int = 150, max_size_threshold: int = 300):
#     _, height, width = img.shape

#     if max(width, height) < max_size_threshold:
#         # no downsampling needed
#         return img

#     downsample_factor = int(width / target_size) if width > height else int(height / target_size)
#     return img[:, ::downsample_factor, ::downsample_factor]

# def sample_images(image_paths: list[str]) -> np.ndarray:
#     sampled_indices = sample_indices(len(image_paths))

#     images = None
#     for i, idx in enumerate(sampled_indices):
#         path = image_paths[idx]
#         # we load as uint8 to reduce memory usage
#         img = load_image_as_numpy(path, dtype=np.uint8, channel_first=True)
#         img = auto_downsample_height_width(img)

#         if images is None:
#             images = np.empty((len(sampled_indices), *img.shape), dtype=np.uint8)

#         images[i] = img

#     return images
# def get_feature_stats(array: np.ndarray, axis: tuple, keepdims: bool) -> dict[str, np.ndarray]:
#     return {
#         "min": np.min(array, axis=axis, keepdims=keepdims),
#         "max": np.max(array, axis=axis, keepdims=keepdims),
#         "mean": np.mean(array, axis=axis, keepdims=keepdims),
#         "std": np.std(array, axis=axis, keepdims=keepdims),
#         # "count": np.array([len(array)]),
#     }


# dataset = load_dataset('/home/yilinw/.cache/huggingface/lerobot/physical-intelligence/libero-3')
# episode_data = dataset['train']

# # Create features dictionary
# features_dict = {}
# for column in tqdm(episode_data.column_names):
#     features_dict[column] = {"dtype": str(episode_data.features[column].dtype)}

# stats = compute_episode_stats(episode_data, features_dict)

# # Convert numpy arrays to lists for JSON serialization
# def convert_numpy(obj):
#     if isinstance(obj, np.ndarray):
#         return obj.tolist()
#     elif isinstance(obj, np.integer):
#         return int(obj)
#     elif isinstance(obj, np.floating):
#         return float(obj)
#     return obj

# # Convert stats to JSON-serializable format
# stats_json = {}
# for key, value in tqdm(stats.items()):
#     if isinstance(value, dict):
#         stats_json[key] = {k: convert_numpy(v) for k, v in value.items()}
#     else:
#         stats_json[key] = convert_numpy(value)

# ## save the stats as json
# os.makedirs('/home/yilinw/.cache/huggingface/lerobot/physical-intelligence/libero-3/meta', exist_ok=True)
# with open('/home/yilinw/.cache/huggingface/lerobot/physical-intelligence/libero-3/meta/stats.json', 'w') as f:
#     json.dump(stats_json, f, indent=2)

# 1. Load the original dataset
orig = load_dataset("/home/yilinw/.cache/huggingface/lerobot/physical-intelligence/libero-10")

# 2. Which task indices to keep
keep_indices = {2, 5, 6}

# 3. Remember old→new episode mapping for the train split
# old_idx_to_example = [
#     i for i, ex in enumerate(orig["train"])
#     if ex["task_index"] in keep_indices
# ]

# 3. Filter
print('filtering...')
filtered = DatasetDict({
    split: ds.filter(lambda ex: ex["task_index"] in keep_indices)
    for split, ds in orig.items()
})

# 4. remember the old-new episode mapping for the train split
episode_index = np.unique(np.array(filtered["train"]["episode_index"]))
old_idx_to_example = {
    old: new for new, old in enumerate(episode_index)
}
## load the episodes.jsonl 
with open('/home/yilinw/.cache/huggingface/lerobot/physical-intelligence/libero-10/meta/episodes.jsonl', 'r') as f:
    episodes = [json.loads(line) for line in f]

## episodes has three keys: episode_index, tasks, length
## filter the episodes.jsonl and change the episode_index to the new episode_index using the old_idx_to_example
filtered_episodes = []
for episode in episodes:
    if episode["episode_index"] in old_idx_to_example.keys():
        episode["episode_index"] = old_idx_to_example[episode["episode_index"]]
        filtered_episodes.append(episode)

## save the filtered episodes.jsonl
with open('/home/yilinw/.cache/huggingface/lerobot/physical-intelligence/libero-10/meta/filtered_episodes.jsonl', 'w') as f:
    for episode in filtered_episodes:
        f.write(json.dumps(episode) + '\n')








# ## change the episode_index to the new episode_index
# print('changing episode_index...')
# new_episode_indices = [old_idx_to_example[ep_idx] for ep_idx in filtered["train"]["episode_index"]]
# filtered["train"] = filtered["train"].remove_columns("episode_index").add_column("episode_index", new_episode_indices)

# # 5. Write out the filtered dataset to disk
# out_dir = "/home/yilinw/.cache/huggingface/lerobot/physical-intelligence/libero-3"
# if os.path.isdir(out_dir):
#     shutil.rmtree(out_dir)
# os.makedirs(out_dir, exist_ok=True)

# # Save each split with custom chunking structure
# for split_name, dataset in filtered.items():
#     split_dir = os.path.join(out_dir, split_name)
#     os.makedirs(split_dir, exist_ok=True)
    
#     # Group by episode_index
#     episode_groups = {}
#     for i, episode_idx in enumerate(dataset["episode_index"]):
#         if episode_idx not in episode_groups:
#             episode_groups[episode_idx] = []
#         episode_groups[episode_idx].append(i)
    
#     # Create chunks of episodes (e.g., 100 episodes per chunk)
#     chunk_size = 1000
#     episode_list = list(episode_groups.keys())
    
#     for chunk_idx in range(0, len(episode_list), chunk_size):
#         chunk_episodes = episode_list[chunk_idx:chunk_idx + chunk_size]
#         chunk_dir = os.path.join(split_dir, f"chunk-{chunk_idx//chunk_size:03d}")
#         os.makedirs(chunk_dir, exist_ok=True)
        
#         # Save each episode as a separate parquet file
#         for episode_idx in chunk_episodes:
#             episode_indices = episode_groups[episode_idx]
#             episode_data = dataset.select(episode_indices)
#             episode_file = os.path.join(chunk_dir, f"episode_{episode_idx:03d}.parquet")
#             episode_data.to_parquet(episode_file)

# # 6. Load & remap the old cot.json
# with open("cot.json", "r") as f:
#     cot = json.load(f)

# mapping = {old: new for new, old in enumerate(old_idx_to_example)}
# new_cot = {}
# for old_str, ann in cot.items():
#     if old_str == "vision_language_episode_idx":
#         continue
#     old = int(old_str)
#     if old in mapping:
#         new_cot[str(mapping[old])] = ann

# if "vision_language_episode_idx" in cot:
#     new_cot["vision_language_episode_idx"] = [
#         mapping[i] for i in cot["vision_language_episode_idx"]
#         if i in mapping
#     ]

# with open(os.path.join(out_dir, "cot.json"), "w") as f:
#     json.dump(new_cot, f, indent=2)

# print(f"Filtered dataset + cot.json written to ./{out_dir}")