## load the txt data from the file and relabel the data from the png files
import os
import json
import imageio
from tqdm import tqdm
import re
import shutil
def load_annotations_txt(path):
    ## read the json from txt file
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
        keys = ["critical moment", "critical_moments", "critical_moment", "critical moments"]
        for key in keys:
            critical_moment = data.get(key, [])
            if not (critical_moment):
                continue
            if 'frame' not in critical_moment[0]:
                return 0
            frames = [int(item["frame"]) for item in data.get(key, [])]
            break
        
        return data , plans, frames
    
def modify_plans(data,  new_frames, filename):
    
    keys = ["critical moment", "critical_moments", "critical_moment", "critical moments"]
    for key in keys:
        critical_moment = data.get(key, [])
        if not (critical_moment):
            continue 
        for i in range(len(critical_moment)):
            critical_moment[i]["frame"] = int(new_frames[i])
    with open(os.path.join('relabel_data', filename), 'w') as f:
        json.dump(data, f, indent=4)    

def main():
    annotation_folder = 'gemini_responses'
    video_folder = 'exported_data/videos_fps10'
    output_folder = 'relabel_data'
    os.makedirs(output_folder, exist_ok=True)
    ## sort the filenames in the annotation folder
    for filename in sorted(os.listdir(annotation_folder))[3229:]:
        ## get the episode index from the filename
        episode_index = filename.split('_')[1]
        ## find the corresponding png file that starts with the same episode index
        png_files = [f for f in os.listdir(video_folder) if f.startswith(f'episode_{episode_index}') and f.endswith('.png')]
        data, plans, frames = load_annotations_txt(os.path.join(annotation_folder, filename)) 
        if len(frames) != len(png_files):
            print(f"Number of frames in {filename} does not match number of png files")
           
            print("plans:", plans)
            print("frames:", frames)
            ## load the video and check the length 
            vid_filename = filename.replace('_gemini.txt', '.mp4')
            vid_path = os.path.join(video_folder, vid_filename) 
            ## read the number of frames in the video
            reader = imageio.get_reader(vid_path)
            num_frames = reader.count_frames()
            print(f"Number of frames in video {vid_filename}: {num_frames}")
            
            new_frames = [int(png_file.split('_')[-1].split('.')[0]) for png_file in png_files]
            print(f"New frames: {new_frames}")
            user_input = input("Do you want to relabel the frames? (y/n): ")
            if user_input.lower() == 'y':
                new_input = input("The frame numbers to relabel")
                if new_input:
                    new_frames = new_input.split(',')
                    new_frames = [frame.strip() for frame in new_frames]
                    modify_plans(data, new_frames, filename)
                    print(f'Relabeled successfully {filename}')
            elif user_input.lower() == 'i':
                new_frames.append(num_frames - 1)
                ## sort the new frames
                new_frames = sorted(new_frames)
                print(new_frames)
                modify_plans(data, new_frames, filename)
                print(f'Relabeled successfully {filename}')
            else:
                ## copy the original file in annotation folder to the output folder
                
                shutil.copy(os.path.join(annotation_folder, filename), os.path.join(output_folder, filename))
                print(f"Skipping {filename}")
                continue
        else: 
            new_frames = [png_file.split('_')[-1].split('.')[0] for png_file in png_files]
            modify_plans(data, new_frames, filename)
            print(f"Relabeled {filename} successfully")
if __name__ == "__main__":
    main()