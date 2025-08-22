import os
import time
import openai
from openai import OpenAI
import json
import argparse
from tqdm import tqdm
from typing import List
import cv2 
import base64
from io import BytesIO
from PIL import Image
# -----------------------------------------------
# Configuration: adjust as needed
# -----------------------------------------------
IMG_FOLDER = os.path.join("/home/yilinw/Projects/OneTwoVLA/data/libero/videos", "verifier_image_dataset")
INSTR_FOLDER = os.path.join("exported_data", "instructions")
OUTPUT_FOLDER = "gpt_verifier_image_responses"
GCS_BUCKET = os.getenv("GCS_BUCKET", "your-gcs-bucket-name")
MODEL_NAME = "gpt-4o"
RPC_TIMEOUT = 120.0  # seconds

TASK_KEYWORD_MAPPLING = {
    "alphabet soup" : "alphabet soup can (blue cylindrical can)",
    "cream cheese" : "cream cheese (blue rectangular box)",
    "butter" : "butter (red rectangular box)",
    "tomato sauce": "tomato sauce (red and green cylindrical can)",
    "chocolate pudding": "chocolate pudding (brown rectangular box)",
    
}

def get_prompt(instruction_text: str, query_type: str) -> str:
    ## load the prompt from the file
    with open(f"prompt_templates/prompt_{query_type}.txt", "r", encoding="utf-8") as f:
        prompt = f.read()
    return prompt.format(TASK_INSTRUCTION=instruction_text)

def ensure_folder(path: str):
    os.makedirs(path, exist_ok=True)

def load_api_key(provider: str) -> str:
    with open("api_config.json") as f:
        config = json.load(f)
    api_key = config[provider]["api_key"]
    print(f"Loaded API key for {provider}: {api_key[:10]}...")
    return api_key

# def load_instruction(path: str) -> str:
#     with open(path, "r", encoding="utf-8") as f:
#         return f.read()
def load_instruction(file_name: str) -> str:
    ## separate file name with _ 
    instruction = file_name.split("_")[2]
    ## remove the 1. 
    instruction = instruction.split(".")[1]
    ## remove the . in the last
    instruction = instruction.rstrip(".")
    print(instruction)
    return instruction

def encode_image_to_base64(image):
    """Convert OpenCV image to base64 string"""
    success, buf = cv2.imencode(".jpg", image)
    if not success:
        raise RuntimeError("Could not encode image to JPEG")
    
    # Convert to base64
    img_base64 = base64.b64encode(buf).decode('utf-8')
    return img_base64

def call_gpt_with_images(client: OpenAI, img_files: List[str], instruction_text: str, timeout: float, query_type: str = 'moment') -> str:
    prompt = get_prompt(instruction_text, query_type = query_type)
    
    # Prepare messages with images
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }
    ]
    
    # Add images to the message
    for img_base64 in img_files:
        # img_files are now base64 strings directly
        messages[0]["content"].append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{img_base64}"
            }
        })
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=1000,
            temperature=0
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling GPT: {e}")
        raise

def get_img_files(img_file: str) -> List[str]:
    ## reading the video file and get the image files
    filename = img_file.split("/")[-1]
    task_id = filename.split("_")[1]
    variation_id = filename.split("_")[4]
    # print("name: ", f"task_{task_id}_variation_{variation_id}_initial_image.png")
    initial_img = cv2.imread(os.path.join("/".join(img_file.split("/")[:-1]), f"task_{task_id}_variation_{variation_id}_initial_image.png")) 
    # initial_img = cv2.imread(os.path.join("/home/yilinw/Projects/OneTwoVLA/data/libero/videos/verifier_image_dataset/task_0/","task_0_1. Pick up the alphabet soup._variation_22_thinking_phase_0_batch_0_initial_image.jpg"))
    agentview = cv2.imread(img_file)
    width = agentview.shape[1]
    agentview = agentview[:, :width//2, :]
    ## resize agentview to 256x256
    agentview = cv2.resize(agentview, (256, 256))
    separate_part = img_file.split("/")
    wristview = cv2.imread(os.path.join("/".join(separate_part[:-1]), separate_part[-1].replace("task", "wrist")))
    imgs = [initial_img, agentview, wristview]
    cv2.imshow('img', imgs[0])
    cv2.waitKey(0)
    cv2.imshow('img', imgs[1])
    cv2.waitKey(0)
    cv2.imshow('img', imgs[2])
    cv2.waitKey(0)

    # Convert images to base64 strings
    base64_images = []
    for img in imgs:
        img_base64 = encode_image_to_base64(img)
        base64_images.append(img_base64)
    
    return base64_images, imgs

def load_labels(label_file: str) -> List[str]:
    with open(label_file, "r", encoding="utf-8") as f:
        return f.read().splitlines()

def main():
    ensure_folder(OUTPUT_FOLDER)
    ## add argument with argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_type", type=str, default="verifier")
    args = parser.parse_args()
    # List all video files and match with corresponding instruction file
    image_files_by_task = {}
    for task in range(1):
        task_name = f"task_{task}"
        subfolder = os.path.join(IMG_FOLDER, task_name)
        img_files = [f for f in os.listdir(subfolder) if f.lower().endswith(".png") and "task" in f and 'initial' not in f]
        ## group those files with variation first, and then with phase, then lastly sort them in batch number
        ## each file name is task_{id}_{task_description}_variation_{variation_id}_thinking_phase_{phase_id}_batch_{batch_id}.mp4
        image_files_by_task[task_name] = {}
        for image_file in img_files:
            # Parse filename components
            parts = image_file.split('_')
            variation_id = parts[4]
            phase_id = parts[7]
            
            # Group by variation
            if variation_id not in image_files_by_task[task_name]:
                image_files_by_task[task_name][variation_id] = {}
                
            # Group by phase within variation
            if phase_id not in image_files_by_task[task_name][variation_id]:
                image_files_by_task[task_name][variation_id][phase_id] = []
                
            # Add to appropriate group
            image_files_by_task[task_name][variation_id][phase_id].append(image_file)
        
    # Sort each phase group by batch number
    for task in image_files_by_task.values():
        for variation in task.values():
            for phase_files in variation.values():
                phase_files.sort(key=lambda x: int(x.split('_batch_')[1].split('.')[0]))
            
    # Flatten the grouped and sorted files back into a single list
    image_files = []
    image_labels = []
    for task_name in sorted(image_files_by_task.keys()):
        for variation_id in sorted(image_files_by_task[task_name].keys()):
            ids = load_labels(os.path.join(IMG_FOLDER, f'{task_name}', f'variation_{variation_id}.txt'))
            for phase_id in sorted(image_files_by_task[task_name][variation_id].keys()):
                image_files.extend(image_files_by_task[task_name][variation_id][phase_id])
            image_labels.extend(ids)
            assert len(image_files) == len(image_labels), f"task_name: {task_name}, variation_id: {variation_id}, phase_id: {phase_id}, image_length: {len(image_files)}, image_labels_length: {len(image_labels)}"
    
    ## initialize the client
    api_key = load_api_key("openai")
    # Set API key as environment variable to avoid proxy issues
    os.environ["OPENAI_API_KEY"] = api_key
    client = OpenAI()
    # Prepare a log file to record video URIs
    # uri_log_path = os.path.join(OUTPUT_FOLDER, "video_uris.txt")
    # with open(uri_log_path, "w", encoding="utf-8") as uri_log:
        # uri_log.write("filename,video_uri\n")

    for idx, vid_filename in tqdm(enumerate(image_files)):
        # vid_path = os.path.join(VIDEO_FOLDER, vid_filename)
        # base_name = os.path.splitext(vid_filename)[0]
        # if idx not in [40, 47, 49]:
        if idx  not in [124]:
        # if idx not in [12, 15, 18, 19]:
            continue
        base_name = vid_filename.split('.png')[0]   
        # instr_filename = f"{base_name}.txt"
        # instr_path = os.path.join(INSTR_FOLDER, instr_filename)

        # print(f"\n[{idx+1}/{len(video_files)}] Processing {vid_filename} …")
        print(f"Processing {vid_filename} …")

        # if not os.path.isfile(instr_path):
            # print(f"  ⚠️ Instruction file not found: {instr_filename}. Skipping.")
            # continue

        # 1) Upload video to GCS
        # blob_name = f"libero_videos/{vid_filename}"
        # print(f"  • Uploading '{vid_filename}' …")
        # video_file = upload_to_google(vid_path, client)
        # print(f"  • Uploaded to {video_uri}")
        task_id = int(vid_filename.split("_")[1])
        img_files, imgs = get_img_files(os.path.join(IMG_FOLDER, f"task_{task_id}/{vid_filename}"))
        ## create a folder to save the images
        img_folder = os.path.join(OUTPUT_FOLDER, f"task_{task_id}")
        os.makedirs(img_folder, exist_ok=True)
        img_file_name = vid_filename.split(".mp4")[0]
        for i, img_file in enumerate(img_files):
            # cv2.imshow('img', imgs[i])
            # cv2.waitKey(0)
            cv2.imwrite(os.path.join(img_folder, f"{img_file_name.split('.png')[0]}_img_{i}.png"), imgs[i])
        label = image_labels[idx]
        # Log the filename and URI
        # uri_log.write(f"{vid_filename},{video_uri}\n")

        # 2) Load instruction
        instruction_text = load_instruction(vid_filename)
        instruction_text = get_prompt(instruction_text, query_type = args.query_type)

        ## add the task_instruction mapping keyword from TASK_KEYWORD_MAPPLING
        for keyword, mapping in TASK_KEYWORD_MAPPLING.items():
            if keyword in instruction_text:
                instruction_text = instruction_text.replace(keyword,  mapping)
        print('instruction_text: ', instruction_text)
        # 3) Call GPT with images
        try:
            start_time = time.time()
            gpt_output = call_gpt_with_images(
                client = client,
                img_files = img_files,
                instruction_text=instruction_text,
                timeout=RPC_TIMEOUT,
                query_type=args.query_type
            )
            print(f"  ✓ GPT response time: {time.time() - start_time} seconds")
            # 4) Write out the response as a .txt file
            resp_filename = f"{base_name}_gpt.txt"
            resp_path = os.path.join(OUTPUT_FOLDER, resp_filename)
            with open(resp_path, "w", encoding="utf-8") as rf:
                rf.write(gpt_output)
            print(f"  ✓ Wrote GPT response to {resp_path}")
            print(gpt_output)
            ## extract the label from the response
            ## get the last line of the response
            predicted_label = gpt_output.split("\n")[-1]

            if "success" in predicted_label.lower():
                predicted_label = 0
            else:
                predicted_label = 1
            print('predicted_label: ', predicted_label)
            print('label: ', label)
            # Optional: throttle between requests
            time.sleep(1)

        except FileNotFoundError as e:
            print(f"  ⚠️ File not found: {e}. Skipping this file.")
            continue
        except Exception as e:
            error_msg = str(e)
            if "insufficient_quota" in error_msg or "429" in error_msg:
                print(f"  ⚠️ API quota exceeded. Please check your OpenAI billing. Error: {e}")
                print("  Stopping execution due to quota limit.")
                break
            else:
                print(f"  ⚠️ Failed to process {vid_filename}: {e}")
                continue

    print("\nAll queries complete. Check responses and video_uris.txt in:", os.path.abspath(OUTPUT_FOLDER))


if __name__ == "__main__":
    main()
