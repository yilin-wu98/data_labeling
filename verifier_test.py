import os
import time
from google.cloud import storage
from google import genai
from google.genai.types import Part#, GenerateContentRequest
from google.api_core.exceptions import DeadlineExceeded
import json
import argparse
from tqdm import tqdm
from typing import List
import cv2 
# -----------------------------------------------
# Configuration: adjust as needed
# -----------------------------------------------
VIDEO_FOLDER = os.path.join("/home/yilinw/Projects/OneTwoVLA/data/libero/videos", "verifier_dataset")
INSTR_FOLDER = os.path.join("exported_data", "instructions")
OUTPUT_FOLDER = "gemini_verifier_responses"
GCS_BUCKET = os.getenv("GCS_BUCKET", "your-gcs-bucket-name")
MODEL_NAME = "gemini-2.5-pro-preview-06-05"
RPC_TIMEOUT = 120.0  # seconds

TASK_KEYWORD_MAPPLING = {
    "alphabet soup" : "(blue can)",
    "cream cheese" : "(blue box)",
    "butter" : "(red box)",
    "tomato sauce": "(red can)",
    "chocolate pudding": "(brown box)",
    
}

def get_prompt(instruction_text: str, query_type: str) -> str:
    ## load the prompt from the file
    with open(f"prompt_templates/prompt_{query_type}.txt", "r", encoding="utf-8") as f:
        prompt = f.read()
    return prompt.format(TASK_INSTRUCTION=instruction_text)

def ensure_folder(path: str):
    os.makedirs(path, exist_ok=True)


# def upload_to_gcs(local_path: str, bucket_name: str, destination_blob_name: str) -> str:
#     """
#     Uploads a local file to GCS and returns the `gs://` URI.
#     """
#     client = storage.Client()
#     bucket = client.bucket(bucket_name)
#     blob = bucket.blob(destination_blob_name)
#     blob.upload_from_filename(local_path)
#     return f"gs://{bucket_name}/{destination_blob_name}"
def upload_to_google(local_path: str, client: genai.Client) -> str:
    myfile = client.files.upload(file=local_path)
    time.sleep(3)
    return myfile

def load_api_key(provider: str) -> str:
    with open("api_config.json") as f:
        config = json.load(f)
    return config[provider]["api_key"]

# def load_instruction(path: str) -> str:
#     with open(path, "r", encoding="utf-8") as f:
#         return f.read()
def load_instruction(file_name: str) -> str:
    ## separate file name with _ 
    instruction = file_name.split("_")[2]
    ## remove the 1. 
    instruction = instruction.split("1.")[1]
    ## remove the . in the last
    instruction = instruction.rstrip(".")
    print(instruction)
    return instruction

def call_gemini_with_gcs_images(client: genai.Client, img_files: List[str], instruction_text: str, timeout: float, query_type: str = 'moment') -> str:
    prompt = get_prompt(instruction_text, query_type = query_type)
    contents = [
        
        
    ]
    for img_file in img_files:
        contents.append(img_file)
    contents.append(prompt)
    # request = GenerateContentRequest(
        # model=MODEL_NAME,
        # contents=contents,
    # )

    # response = client.transport.generate_content(request=request, timeout=timeout)
    #
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=contents,
        # timeout=timeout
    )
    return response.text

def get_img_files(video_file: str) -> List[str]:
    ## reading the video file and get the image files
    cap = cv2.VideoCapture(video_file)
    imgs = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        imgs.append(frame)
    cap.release()
    ## only keep the first, middle and last frame (3 images)
    length = len(imgs)
    ## only keep the left half of each image
    width = imgs[0].shape[1]
    imgs = [imgs[0][:, :width//2, :]] + [imgs[length//2][:, :width//2, :]] + [imgs[length-1][:, :width//2, :]]
    parts = []
    for img in imgs:
        success, buf = cv2.imencode(".jpg", img)
        if not success:
            raise RuntimeError("Could not encode image to JPEG")

        jpeg_bytes = buf.tobytes()

        img_file = Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg")
        parts.append(img_file)
    return parts, imgs

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
    video_files_by_task = {}
    for task in range(10):
        task_name = f"task_{task}"
        subfolder = os.path.join(VIDEO_FOLDER, task_name)
        video_files = [f for f in os.listdir(subfolder) if f.lower().endswith(".mp4") and "task" in f]
        ## group those files with variation first, and then with phase, then lastly sort them in batch number
        ## each file name is task_{id}_{task_description}_variation_{variation_id}_thinking_phase_{phase_id}_batch_{batch_id}.mp4
        video_files_by_task[task_name] = {}
        for video_file in video_files:
            # Parse filename components
            parts = video_file.split('_')
            variation_id = parts[4]
            phase_id = parts[7]
            
            # Group by variation
            if variation_id not in video_files_by_task[task_name]:
                video_files_by_task[task_name][variation_id] = {}
                
            # Group by phase within variation
            if phase_id not in video_files_by_task[task_name][variation_id]:
                video_files_by_task[task_name][variation_id][phase_id] = []
                
            # Add to appropriate group
            video_files_by_task[task_name][variation_id][phase_id].append(video_file)
        
    # Sort each phase group by batch number
    for task in video_files_by_task.values():
        for variation in task.values():
            for phase_files in variation.values():
                phase_files.sort(key=lambda x: int(x.split('_batch_')[1].split('.')[0]))
            
    # Flatten the grouped and sorted files back into a single list
    video_files = []
    video_labels = []
    for task_name in sorted(video_files_by_task.keys()):
        for variation_id in sorted(video_files_by_task[task_name].keys()):
            ids = load_labels(os.path.join(VIDEO_FOLDER, f'{task_name}', f'variation_{variation_id}.txt'))
            for phase_id in sorted(video_files_by_task[task_name][variation_id].keys()):
                video_files.extend(video_files_by_task[task_name][variation_id][phase_id])
            video_labels.extend(ids)
            assert len(video_files) == len(video_labels), f"task_name: {task_name}, variation_id: {variation_id}, phase_id: {phase_id}, video_length: {len(video_files)}, video_labels_length: {len(video_labels)}"
    
    ## initialize the client
    client = genai.Client(api_key = load_api_key("google"))
    # Prepare a log file to record video URIs
    # uri_log_path = os.path.join(OUTPUT_FOLDER, "video_uris.txt")
    # with open(uri_log_path, "w", encoding="utf-8") as uri_log:
        # uri_log.write("filename,video_uri\n")

    for idx, vid_filename in tqdm(enumerate(video_files)):
        # vid_path = os.path.join(VIDEO_FOLDER, vid_filename)
        # base_name = os.path.splitext(vid_filename)[0]
        base_name = vid_filename.split('.mp4')[0]   
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
        img_files, imgs = get_img_files(os.path.join(VIDEO_FOLDER, f"task_{task_id}/{vid_filename}"))
        ## create a folder to save the images
        img_folder = os.path.join(OUTPUT_FOLDER, f"task_{task_id}")
        os.makedirs(img_folder, exist_ok=True)
        img_file_name = vid_filename.split(".mp4")[0]
        for i, img_file in enumerate(img_files):
            cv2.imshow('img', imgs[i])
            cv2.waitKey(0)
            cv2.imwrite(os.path.join(img_folder, f"{img_file_name}_img_{i}.jpg"), imgs[i])
        label = video_labels[idx]
        # Log the filename and URI
        # uri_log.write(f"{vid_filename},{video_uri}\n")

        # 2) Load instruction
        instruction_text = load_instruction(vid_filename)
        instruction_text = get_prompt(instruction_text, query_type = args.query_type)

        ## add the task_instruction mapping keyword from TASK_KEYWORD_MAPPLING
        for keyword, mapping in TASK_KEYWORD_MAPPLING.items():
            if keyword in instruction_text:
                instruction_text = instruction_text.replace(keyword, keyword +" " + mapping)
        print('instruction_text: ', instruction_text)
        # 3) Call Gemini with GCS URI
        try:
            gemini_output = call_gemini_with_gcs_images(
                client = client,
                img_files = img_files,
                instruction_text=instruction_text,
                timeout=RPC_TIMEOUT,
                query_type=args.query_type
            )

            # 4) Write out the response as a .txt file
            resp_filename = f"{base_name}_gemini.txt"
            resp_path = os.path.join(OUTPUT_FOLDER, resp_filename)
            with open(resp_path, "w", encoding="utf-8") as rf:
                rf.write(gemini_output)
            print(f"  ✓ Wrote Gemini response to {resp_path}")
            ## extract the label from the response
            predicted_label = gemini_output.split("Verification: ")[1]
            if "success" in predicted_label.lower():
                predicted_label = 1
            else:
                predicted_label = 0
            print('predicted_label: ', predicted_label)
            print('label: ', label)
            # Optional: throttle between requests
            time.sleep(1)

        except DeadlineExceeded:
            print("  ⚠️ Request timed out. Consider increasing RPC_TIMEOUT.")
        except Exception as e:
            print(f"  ⚠️ Failed to process {vid_filename}: {e}")

    print("\nAll queries complete. Check responses and video_uris.txt in:", os.path.abspath(OUTPUT_FOLDER))


if __name__ == "__main__":
    main()
