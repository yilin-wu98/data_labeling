import os
import time
from google.cloud import storage
from google import genai
from google.genai.types import Part#, GenerateContentRequest
from google.api_core.exceptions import DeadlineExceeded
import json
import argparse
from tqdm import tqdm
# -----------------------------------------------
# Configuration: adjust as needed
# -----------------------------------------------
VIDEO_FOLDER = os.path.join("exported_data", "videos_fps10")
INSTR_FOLDER = os.path.join("exported_data", "instructions")
OUTPUT_FOLDER = "gemini_responses"
GCS_BUCKET = os.getenv("GCS_BUCKET", "your-gcs-bucket-name")
MODEL_NAME = "gemini-2.5-pro-preview-06-05"
RPC_TIMEOUT = 120.0  # seconds


def get_prompt(instruction_text: str, query_type: str) -> str:
    ## load the prompt from the file
    with open(f"prompt_templates/prompt_{query_type}.txt", "r", encoding="utf-8") as f:
        prompt = f.read()
    return prompt.format(task_description=instruction_text)

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

def load_instruction(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def call_gemini_with_gcs_video(client: genai.Client, video_file: str, instruction_text: str, timeout: float, query_type: str = 'moment') -> str:
    prompt = get_prompt(instruction_text, query_type = query_type)
    contents = [
        video_file,
        prompt
    ]

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


def main():
    ensure_folder(OUTPUT_FOLDER)
    ## add argument with argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_type", type=str, default="moment")
    args = parser.parse_args()
    # List all video files and match with corresponding instruction file
    video_files = sorted(f for f in os.listdir(VIDEO_FOLDER) if f.lower().endswith(".mp4"))
    if not video_files:
        print(f"No MP4 files found in '{VIDEO_FOLDER}'.")
        return

    ## initialize the client
    client = genai.Client(api_key = load_api_key("google"))
    # Prepare a log file to record video URIs
    # uri_log_path = os.path.join(OUTPUT_FOLDER, "video_uris.txt")
    # with open(uri_log_path, "w", encoding="utf-8") as uri_log:
        # uri_log.write("filename,video_uri\n")

    for idx, vid_filename in tqdm(enumerate(video_files)):
        vid_path = os.path.join(VIDEO_FOLDER, vid_filename)
        base_name = os.path.splitext(vid_filename)[0]
        instr_filename = f"{base_name}.txt"
        instr_path = os.path.join(INSTR_FOLDER, instr_filename)

        print(f"\n[{idx+1}/{len(video_files)}] Processing {vid_filename} …")

        if not os.path.isfile(instr_path):
            print(f"  ⚠️ Instruction file not found: {instr_filename}. Skipping.")
            continue

        # 1) Upload video to GCS
        # blob_name = f"libero_videos/{vid_filename}"
        print(f"  • Uploading '{vid_filename}' …")
        video_file = upload_to_google(vid_path, client)
        # print(f"  • Uploaded to {video_uri}")

        # Log the filename and URI
        # uri_log.write(f"{vid_filename},{video_uri}\n")

        # 2) Load instruction
        instruction_text = load_instruction(instr_path)

        # 3) Call Gemini with GCS URI
        try:
            gemini_output = call_gemini_with_gcs_video(
                client = client,
                video_file = video_file,
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

            # Optional: throttle between requests
            time.sleep(1)

        except DeadlineExceeded:
            print("  ⚠️ Request timed out. Consider increasing RPC_TIMEOUT.")
        except Exception as e:
            print(f"  ⚠️ Failed to process {vid_filename}: {e}")

    print("\nAll queries complete. Check responses and video_uris.txt in:", os.path.abspath(OUTPUT_FOLDER))


if __name__ == "__main__":
    main()
