#!/usr/bin/env python3
import os
import re
import json
import argparse
from statistics import mean, stdev
from collections import defaultdict
import random 
## import deepcopy of the dictionary
import copy
FILENAME_RE = re.compile(r"episode_(\d+)_task_(\d+)_gemini\.txt")
FILENAME_RE_SCENE_DESCRIPTION = re.compile(r"task_(\d+).txt")
def parse_frames(path):
        """
    Loads a JSON object from a .txt (or .json) file, stripping:
      - Byte-order mark (BOM)
      - Markdown-style fences (```json ... ```)
      - Leading/trailing whitespace
      - Any garbage before the first '{' or after the last '}'
    """
        with open(path, 'r', encoding='utf-8') as f:
            raw = f.read()

        # 1) Strip BOM
        raw = raw.lstrip('\ufeff')

        # 2) Remove markdown fences
        #    ```json or ```  at start, and closing ``` at end
        raw = re.sub(r'^\s*```(?:json)?\s*', '', raw)
        raw = re.sub(r'\s*```\s*$', '', raw, flags=re.MULTILINE)

        # 3) Trim whitespace
        raw = raw.strip()

        # 4) If there’s any prefix/suffix around the JSON object,
        #    grab from first '{' to last '}'
        first = raw.find('{')
        last  = raw.rfind('}')
        if first == -1 or last == -1:
            raise ValueError(f"No JSON object found in {path!r}")
        json_str = raw[first:last+1]

        # 5) Parse
        data = json.loads(json_str, strict=False)
        # Expect data["critical moment"] to be a list of dicts with 'frame'
        ## if the frame is not in the list, return 0
        # breakpoint()
        plans = data.get("plans", [])
        if not plans:
            print(path)
        keys = ["critical moment", "critical_moments", "critical_moment"]
        for key in keys:
            critical_moment = data.get(key, [])
            if not (critical_moment):
                continue
            if 'frame' not in critical_moment[0]:
                return 0
            frames = [int(item["frame"]) for item in data.get(key, [])]
            break
        return plans, frames

def load_reasoning_interval(dir: str):
    cot_data = {}
     ## enumerate the files in the folder
    for file in sorted(os.listdir(dir)):
        if not file.endswith('.txt'):
            continue
        ## load the json from txt file
        plans, frames = parse_frames(os.path.join(dir, file))
        ## create a dict where the key is the episode_id and the value is the  (task_id, frames)
        m = FILENAME_RE.fullmatch(file)
        if not m:
            continue
        episode_id, task_id = m.groups()
        ## conver the episode id format from '0000 to '0'
        episode_id = int(episode_id)
        cot_data[str(episode_id)] = (int(task_id), plans, frames)
    return cot_data

def load_scene_description(path):
     ## load the json from txt file
        with open(path, 'r', encoding='utf-8') as f:
            raw = f.read()

        # 1) Strip BOM
        raw = raw.lstrip('\ufeff')

        # 2) Remove markdown fences
        #    ```json or ```  at start, and closing ``` at end
        raw = re.sub(r'^\s*```(?:json)?\s*', '', raw)
        raw = re.sub(r'\s*```\s*$', '', raw, flags=re.MULTILINE)

        # 3) Trim whitespace
        raw = raw.strip()

        # 4) If there’s any prefix/suffix around the JSON object,
        #    grab from first '{' to last '}'
        first = raw.find('{')
        last  = raw.rfind('}')
        if first == -1 or last == -1:
            raise ValueError(f"No JSON object found in {path!r}")
        json_str = raw[first:last+1]

        # 5) Parse
        data = json.loads(json_str, strict=False)
        ## change the data key from "0" to 0 
        # data = {int(k): v for k, v in data.items()}
        # data = [v for k, v in data.items()]
        ## the data is a dictionary of dictionary
        ## randomly sample one of the key from the dictionary and then convert the sampled ones into a list
        sampled_key = random.choice(list(data.keys()))
        selected_description = data[sampled_key]
        description_list = [v for k, v in selected_description.items()]
        # return data 
        return description_list

def load_scene_descriptions(dir: str):
    scene_descriptions = {}
    for file in os.listdir(dir):
        if not file.endswith('.txt'):
            continue
        task_id = FILENAME_RE_SCENE_DESCRIPTION.fullmatch(file).groups()[0]
        scene_descriptions[int(task_id)] = load_scene_description(os.path.join(dir, file))
    return scene_descriptions
def load_instructions(dir: str):
    ## load the tasks.jsonl file
    instructions = {}
    with open(os.path.join(dir, "tasks.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            task = json.loads(line)
            task_id = task["task_index"]
            instruction = task["task"]
            instructions[task_id] = instruction
    return instructions

def load_diverse_instructions(dir: str):
    ## load the instructions from the diverse_instructions folder
    instructions = {}
    with open(os.path.join(dir, "tasks_diverse.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            task = json.loads(line)
            task_id = task["task_index"]
            instruction = task["task"]
            if task_id not in instructions:
                instructions[task_id] = []
            instructions[task_id].append(instruction)
    return instructions
        

def load_episode_length(dir: str):
    lengths = {}
    with open(os.path.join(dir, "episodes.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            episode= json.loads(line)
            lengths[episode["episode_index"]] = episode["length"]

    return lengths

def construct_cot_data(scene_descriptions, reasoning_interval, diverse_instructions, lengths):
     ## initialize the dictionary
     cot_reasoning_data = {}
     ## enumerate the items in the reasoning_interval
     ## loading in the order the sorted episode_id key from 0 to larger number 
    #  for episode_id in sorted(reasoning_interval.keys()):
        # task_id, plans, frames = reasoning_interval[episode_id]
     for episode_id, (task_id, plans, frames) in reasoning_interval.items():
        ## get the scene description for the episode
        episode_data = {}
        ## start interval is 0 to a random number from 12 -14
        start_end = random.randint(12, 14)
        episode_data["episode_start_interval"] = [0, start_end]
        episode_data["segments"] = []
        scene_description = scene_descriptions[int(task_id)]
        episode_length = lengths[int(episode_id)]
        ## get the random instruction for the episode
        random_instruction = random.choice(diverse_instructions[task_id])
        ## check if the length of the plans is the same as the length of the frames
        if len(plans) != len(scene_description) - 1:
            if task_id == 1:
                ## remove the first frame
                print(f"episode_id {episode_id} task_id {task_id} has {len(frames)} frames and {len(plans)} plans, length of scene_description is {len(scene_description)}")
                
                ## remove the second scene description
                ## pop the key of 1 from scene_description
                # scene_description.pop(1)
                scene_description = [scene_description[0]] + scene_description[2:]
            elif task_id in [0, 2, 3, 5, 6,7]:
                ## remove 0, 2, 
                print(f"episode_id {episode_id} task_id {task_id} has {len(frames)} frames and {len(plans)} plans, length of scene_description is {len(scene_description)}")
              
                scene_description = [scene_description[0]]  + scene_description[2::2]
                # scene_description.pop(1)
                # scene_description.pop(3)
            else: 
                print(f"Task id {task_id} episode_id {episode_id} is not supported")
        # print(f"episode_id {episode_id} task_id {task_id} has {len(frames)} frames and {len(plans)} plans, scene_description has {len(scene_description)} items")
        frames[-1] = episode_length - 1
        ## join the plans into a string, each start with a number.
        plan_str = "\n".join([f"{i+1}. {plan}" for i, plan in enumerate(plans)])
        last_step = 0
        ## enumerate over the frames to create segments
        for i in range(len(plans)*2 + 1):
            ## create a segment
            segment = {}
            # segment["start_step"] = frames[i]
            # segment["end_step"] = frames[i+1]
            if i == 0:
                segment["start_step"] = 0
                segment["end_step"] = start_end
                segment["reference_start_step"] = 0
                segment["reference_end_step"] = start_end
                last_step = start_end
                content = f"Instruction: {random_instruction}. \nScene description: TBD.\nPlan: TBD.\n What I have done: TBD.\nNow I need to do: TBD.\n"
                updated_content =  f"Scene description: {scene_description[0]}\nPlan: {plan_str}\n What I have done: Nothing.\nNow I need to do: {plans[0]}\n"
                ## updated content with the instruction is "Instruction: {instructions[task_id]} \n" + updated_content
                updated_content_w_instruction = f"Instruction: {random_instruction}. \n" + updated_content
            elif i == len(plans)*2:
                segment["start_step"] = frames[-1]
                segment["end_step"] = -1
                segment["reference_start_step"] = last_segment["reference_start_step"]
                segment["reference_end_step"] = last_segment["reference_end_step"]
                last_step = segment["end_step"]
                content = last_segment["content"]
                plan_str_done = "\n".join([f"{i+1}. {plan}" for i, plan in enumerate(plans[:-1])])
                updated_content = f"Scene description: {scene_description[i//2]}\nPlan: {plan_str}\n What I have done: {plan_str}\nNow I need to do: Nothing. Task complete.\n"
                updated_content_w_instruction = f"Instruction: {random_instruction}. \n" + updated_content
                
            elif i % 2 == 0:
                segment["start_step"] = last_step
                segment["end_step"] = last_step + random.randint(8, 10)
                segment["reference_start_step"] = last_segment["reference_start_step"]
                segment["reference_end_step"] = last_segment["reference_end_step"]
                last_step = segment["end_step"]
                content = last_segment["content"]
                plan_str_done = "\n".join([f"{i+1}. {plan}" for i, plan in enumerate(plans[:i//2])])
                updated_content = f"Scene description: {scene_description[i//2]}\nPlan: {plan_str}\n What I have done: {plan_str_done}\nNow I need to do: {plans[i//2]}\n"
                updated_content_w_instruction = f"Instruction: {random_instruction}. \n" + updated_content
            else:
                segment["start_step"] = last_step
                segment["end_step"] = frames[i//2]-1
                if i == len(plans)*2 - 1:
                    segment["end_step"] = frames[i//2]
                segment["reference_start_step"] = last_segment["start_step"]
                segment["reference_end_step"] = last_segment["end_step"]
                last_step = segment["end_step"]
                content = last_segment["updated_content_w_instruction"]
                updated_content = None
           
              

            if i > 1 and i % 2 == 1:
                outdated_reference_start_step = last_segment["reference_start_step"]
                outdated_reference_end_step = last_segment["reference_end_step"]
                segment["outdated_reference_start_step"] = outdated_reference_start_step
                segment["outdated_reference_end_step"] = outdated_reference_end_step
            segment["content"] = content
            segment["updated_content"] = updated_content
            if updated_content is not None:
                segment["updated_content_w_instruction"] = updated_content_w_instruction
            episode_data["segments"].append(segment)
            last_segment = copy.deepcopy(segment)
        cot_reasoning_data[episode_id] = episode_data
    ## add the last item
     cot_reasoning_data["vision_language_episode_idx"] = []
     return cot_reasoning_data

def save_cot_data(cot_data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cot_data, f, indent=4)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default="gt_responses")
    args = parser.parse_args()
    ## load the scene description
    scene_description = load_scene_descriptions(os.path.join(args.folder, "../exported_data/diverse_scene_descriptions"))
    ## load the reasoning interval
    reasoning_interval = load_reasoning_interval(args.folder)
    ## load the instructions
    # instructions = load_instructions(os.path.join(args.folder, "../exported_data"))
    diverse_instructions = load_diverse_instructions(os.path.join(args.folder, "../exported_data"))
    ## load the episode length
    lengths = load_episode_length(os.path.join(args.folder, "../exported_data"))
    ## construct the cot data
    cot_data = construct_cot_data(scene_description, reasoning_interval, diverse_instructions, lengths)
    ## save the cot data
    save_cot_data(cot_data, os.path.join(args.folder, "../cot.json"))

if __name__ == "__main__":
    main()
