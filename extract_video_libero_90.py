
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
    local_cache_dir: str = os.path.expanduser("/home/yilinwu/data"),
    dataset_name: str = "libero_90",
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

    task_meta = load_task_meta(os.path.join(local_cache_dir, "libero_90/meta/tasks.jsonl"))
    # Load the dataset from cache
    print(f"Loading '{dataset_name}' (split='{split}') from cache_dir={local_cache_dir} …")
    ## load from disk with local_cache_dir and the dataset name
    ds = load_dataset(os.path.join(local_cache_dir, dataset_name))
    print(f"✅ Dataset loaded. Number of episodes in this split: {len(ds)}")
   
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
    ds_train = ds['train'].select(range(300127, 574571))
    # ds_train = ds['train'].select(range(300))
    
    # breakpoint()
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
            grouped_data[key] = {'image': [], 'gripper_info': []}
            ## convert the pil image to a numpy array
            # item['image'] = np.array(item['image'])
        gripper_state = item['state'][-2:]
        gripper_action = item['actions'][-1:]
        gripper_info = np.concatenate([gripper_state, gripper_action])
        
        grouped_data[key]['image'].append(np.array(item['image']))
        grouped_data[key]['gripper_info'].append(gripper_info)
    print('Dataset grouped by episode and task index')
    # Sort by episode and task index
    sorted_episodes = sorted(grouped_data.keys())
    # breakpoint()
    # for idx, example in enumerate(sorted_episodes):
    ## iterate over the grouped_data with sorted_episodes
    for idx, example in enumerate(sorted_episodes):
        
        # Extract and save instruction text
        instruction = task_meta[example[1]]
        if instruction is None:
            print(f"  [Episode {example[0]}] Instruction field '{example[1]}' is None; skipping instruction.")
            continue
        ## ds['train'] has a key called image and group the image with episode_idx and save the sequence of images as a video
        rollout_images = grouped_data[example]['image']
        rollout_gripper_info = grouped_data[example]['gripper_info']
        length = len(rollout_images)
        for i in range(1, length):
            current_gripper_action = rollout_gripper_info[i][-1]
            previous_gripper_action = rollout_gripper_info[i-1][-1]
           
            if current_gripper_action != previous_gripper_action:
                
                print('gripper action changed')
                imageio.imwrite(os.path.join(videos_dir, f"episode_{example[0]:04d}_task_{example[1]:04d}_step_{min(i+12, length-1):04d}.png"), rollout_images[min(length-1,i+12)])
            
        ## video path is the episode idx and the task index
        instruction_text = "_".join(instruction.split(" "))
        video_path = os.path.join(videos_dir, f"episode_{example[0]:04d}_task_{example[1]:04d}_instruction_{instruction_text}.mp4")

        # Save video file
        if video_path and not os.path.isfile(video_path):

            ## save the images as a video
            print(f"Saving video to {video_path}")
            video_writer = imageio.get_writer(video_path, fps=10)
            for img in rollout_images:
            
                video_writer.append_data(img)
            video_writer.close()
        
        ## save the gripper info as a npy file
        npy_path = os.path.join(videos_dir, f"episode_{example[0]:04d}_task_{example[1]:04d}.npy")
        np.save(npy_path, np.array(rollout_gripper_info))
        
       ## dump all the instruction to a file
        # with open(os.path.join(instr_dir, f"episode_{example[0]:04d}_task_{example[1]:04d}.txt"), "w", encoding="utf-8") as tf:
            # tf.write(instruction)

    print("\nExport complete. Check folder:", os.path.abspath(output_folder))


if __name__ == "__main__":
    main()