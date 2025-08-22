#!/usr/bin/env python3
import os
import re
import json
import argparse
from statistics import mean, stdev
from collections import defaultdict

FILENAME_RE = re.compile(r"episode_(\d+)_task_(\d+)_gemini\.txt")

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
        keys = ["critical moment", "critical_moments", "critical_moment"]
        for key in keys:
            critical_moment = data.get(key, [])
            if not (critical_moment):
                continue
            if 'frame' not in critical_moment[0]:
                return 0
            frames = [int(item["frame"]) for item in data.get(key, [])]
            break
       
       
     
        return frames

def compare_folders(folder_a, folder_b):
    """
    Compare files in folder_a vs folder_b. 
    Returns dict: task_id → list of all frame-differences across files of that task.
    """
    diffs_by_task = defaultdict(list)
    # total_pair_lengths = 0
    for fname in sorted(os.listdir(folder_a)):
        m = FILENAME_RE.fullmatch(fname)
        # print(fname)
        if not m:
            continue
        
        episode_id, task_id = m.groups()
        path_a = os.path.join(folder_a, fname)
        path_b = os.path.join(folder_b, fname)
        
        if not os.path.exists(path_b):
          
            print(f"Warning: {fname} found in {folder_a} but not in {folder_b}, skipping.")
            continue
        
        frames_a = parse_frames(path_a)
        frames_b = parse_frames(path_b)
        # print('frames_a', frames_a)
        # print('frames_b', frames_b)
        if frames_a == 0 or frames_b == 0:

            print('episode_id', episode_id, 'task_id', task_id, 'frames_a', frames_a, 'frames_b', frames_b)
            continue
        if len(frames_a) != len(frames_b):
            print('wrong length')
            print(f"Warning: different # of frames in {fname}: {len(frames_a)} vs {len(frames_b)}")
            print('episode_id', episode_id, 'task_id', task_id)
            continue
       
            # you could choose to skip or pair up to the min length:
        pair_len = min(len(frames_a), len(frames_b))
        # total_pair_lengths += pair_len
        # print('pair_len', pair_len)
        # accumulate frame_b - frame_a
        for i in range(pair_len-1):
            diffs_by_task[task_id].append(abs(frames_b[i] - frames_a[i]))
    # print('total pair lengths', total_pair_lengths)
    return diffs_by_task

def summarize(diffs_by_task):
    """
    Compute mean and stddev per task_id and return dict.
    """
    stats = {}
    for task_id, diffs in diffs_by_task.items():
        if len(diffs) == 0:
            continue
        mu = mean(diffs)
        sigma = stdev(diffs) if len(diffs) > 1 else 0.0
        stats[task_id] = {"mean_diff": mu, "std_diff": sigma, "count": len(diffs)}
    ## print total count
    print('total count', sum([len(diffs) for diffs in diffs_by_task.values()]))
    return stats

def main():
    parser = argparse.ArgumentParser(
        description="Compare 'frame' values in critical moment across two folders.")
    parser.add_argument("folder_a", help="First folder (baseline)")
    parser.add_argument("folder_b", help="Second folder (to compare against)")
    args = parser.parse_args()
    
    diffs_by_task = compare_folders(args.folder_a, args.folder_b)
    stats = summarize(diffs_by_task)
    
    # Print out results
    print(f"{'Task ID':<10} {'Count':>5} {'Mean Diff':>12} {'Std Dev':>10}")
    print("-" * 40)
    for task_id in sorted(stats, key=int):
        s = stats[task_id]
        print(f"{task_id:<10} {s['count']:>5} {s['mean_diff']:>12.3f} {s['std_diff']:>10.3f}")

if __name__ == "__main__":
    main()
