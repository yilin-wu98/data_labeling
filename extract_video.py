
import os
import shutil
from datasets import load_dataset, concatenate_datasets
import imageio
from tqdm import tqdm
import json
import numpy as np
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

def main(
    local_cache_dir: str = os.path.expanduser("~/.cache/huggingface/lerobot"),
    dataset_name: str = "physical-intelligence/libero-10/data",
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

    task_meta = load_task_meta(os.path.join(local_cache_dir, "physical-intelligence/libero-10/meta/tasks.jsonl"))
    # Load the dataset from cache
    print(f"Loading '{dataset_name}' (split='{split}') from cache_dir={local_cache_dir} …")
    ## load from disk with local_cache_dir and the dataset name
    ds = load_dataset(os.path.join(local_cache_dir, dataset_name))
    print(f"✅ Dataset loaded. Number of episodes in this split: {len(ds)}")
    breakpoint()
    # Inspect available keys to find instruction field
    sample_keys = ds.column_names
    print(f"Dataset columns: {sample_keys}")
    # ds ['train'] has a key called task_index
    ## filter out the task_index column with only task_index >= 0 and <=9

    # possible_instruction_keys = ["instruction", "instructions", "text", "language", "lang", "task_description"]
    # instr_key = None
    # for key in possible_instruction_keys:
    #     if key in sample_keys:
    #         instr_key = key
    #         break
    # if instr_key is None:
    #     raise KeyError(f"Could not find any instruction field in columns: {sample_keys}")
    # print(f"Using instruction field: '{instr_key}'")
    ds_train = ds['train']
    ## get a random sample of 200 episodes
    # ds_train = ds_train.select(range(400))
    ## filter out the task_index column with only task_index >= 0 and <=9
    # ds_libero_10 = ds_train.filter(lambda x: x["task_index"] >= 0 and x["task_index"] <= 9)
    # grouped = ds_libero_10.groupby("episode_index").aggregate({"images": "list","task_index": "first"})
    # Iterate over the dataset
    # Group by episode_index and task_index
    grouped_data = {}
    # breakpoint()
    for item in tqdm(ds_train):
        key = (item['episode_index'], item['task_index'])
        if key not in grouped_data:
            grouped_data[key] = []
            ## convert the pil image to a numpy array
            item['image'] = np.array(item['image'])
        grouped_data[key].append(np.array(item['image']))
    print('Dataset grouped by episode and task index')
    # Sort by episode and task index
    sorted_episodes = sorted(grouped_data.keys())
    # breakpoint()
    # for idx, example in enumerate(sorted_episodes):
    ## iterate over the grouped_data with sorted_episodes
    for idx, example in enumerate(sorted_episodes):
        

        ## ds['train'] has a key called image and group the image with episode_idx and save the sequence of images as a video
        rollout_images = grouped_data[example]
        ## video path is the episode idx and the task index
        video_path = os.path.join(videos_dir, f"episode_{example[0]:04d}_task_{example[1]:04d}.mp4")

        # Save video file
        if video_path and not os.path.isfile(video_path):

            ## save the images as a video
            print(f"Saving video to {video_path}")
            video_writer = imageio.get_writer(video_path, fps=10)
            for img in rollout_images:
            
                video_writer.append_data(img)
            video_writer.close()
        

        # Extract and save instruction text
        instruction = task_meta[example[1]]
        if instruction is None:
            print(f"  [Episode {example[0]}] Instruction field '{example[1]}' is None; skipping instruction.")
            continue
       ## dump all the instruction to a file
        with open(os.path.join(instr_dir, f"episode_{example[0]:04d}_task_{example[1]:04d}.txt"), "w", encoding="utf-8") as tf:
            tf.write(instruction)

    print("\nExport complete. Check folder:", os.path.abspath(output_folder))


if __name__ == "__main__":
    main()