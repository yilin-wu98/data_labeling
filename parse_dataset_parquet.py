import os, json, math
from collections import defaultdict
from tqdm import tqdm
from datasets import load_from_disk

# ------------------ CONFIG ------------------
# SRC_DIR   = "/home/yilinwu/data/libero_100"           # where you saved with save_to_disk()
# OUT_DIR   = "/home/yilinwu/data/libero_100_parquet"   # LeRobot-style export root
SRC_DIR = '/home/yilinwu/Projects/data_labeling/libero-100-small'
OUT_DIR = '/home/yilinwu/Projects/data_labeling/libero-100-small-parquet'
CHUNK_SIZE = 1000                         # episodes per chunk dir
PAD        = 6                            # filename zero-padding
FPS        = 10                           # dataset fps for info.json
ROBOT_TYPE = "panda"                      # info.json field
CODEBASE_VERSION = "v2.0"                 # info.json field
TOTAL_VIDEOS = 0                          # you can set !=0 if you also export videos

# If your dataset column names differ, map them to the target names:
# target -> source
# COLUMN_MAP = {
#     "image":         "obs_agentview_image",   # target 'image'
#     "wrist_image":   "obs_wrist_image",       # target 'wrist_image'
#     "state":         "state",
#     "actions":       "actions",
#     "timestamp":     "timestamp",
#     "frame_index":   "frame_index",
#     "episode_index": "episode_index",
#     "index":         "index",
#     "task_index":    "task_index",
# }
# --------------------------------------------

os.makedirs(os.path.join(OUT_DIR, "data"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "meta"), exist_ok=True)

ds = load_from_disk(SRC_DIR)["train"]

# Build mapping: episode_index -> row indices (each episode has variable length)
ep_to_rows = defaultdict(list)
for i, ep in enumerate(ds["episode_index"]):
    ep_to_rows[int(ep)].append(i)

episode_ids = sorted(ep_to_rows.keys())
num_eps = len(episode_ids)
total_frames = len(ds)  # number of rows across all episodes
total_chunks = math.ceil(num_eps / CHUNK_SIZE)
total_tasks = len(set(ds["task_index"])) if "task_index" in ds.column_names else 0

# Export per-episode Parquet
for eidx_pos, ep_id in enumerate(tqdm(episode_ids, desc="Writing episodes")):
    chunk_id = eidx_pos // CHUNK_SIZE
    chunk_dir = os.path.join(OUT_DIR, "data", f"chunk-{chunk_id:03d}")
    os.makedirs(chunk_dir, exist_ok=True)

    rows = ep_to_rows[ep_id]
    ep_ds = ds.select(rows)

    # Rename columns to target schema if needed
    # (only rename those that exist; others are ignored)
    # rename_map = {tgt: src for tgt, src in COLUMN_MAP.items() if src in ep_ds.column_names and tgt != src}
    # # Datasets.rename_columns expects {old_name: new_name}
    # rename_map_datasets = {v: k for k, v in rename_map.items()}
    # if rename_map_datasets:
    #     ep_ds = ep_ds.rename_columns(rename_map_datasets)

    # Ensure all target columns exist; if missing, add placeholder None/zeros
    # (adapt defaults as needed)
    # missing = [tgt for tgt in COLUMN_MAP.keys() if tgt not in ep_ds.column_names]
    # if missing:
    #     add_cols = {}
    #     n = len(ep_ds)
    #     for col in missing:
    #         if col in ("image", "wrist_image"):
    #             add_cols[col] = [None] * n
    #         elif col in ("state",):
    #             add_cols[col] = [[0.0]*8 for _ in range(n)]
    #         elif col in ("actions",):
    #             add_cols[col] = [[0.0]*7 for _ in range(n)]
    #         elif col in ("timestamp",):
    #             add_cols[col] = [0.0] * n
    #         elif col in ("frame_index","episode_index","index","task_index"):
    #             add_cols[col] = [0] * n
    #         else:
    #             add_cols[col] = [None] * n
    #     ep_ds = ep_ds.with_columns(add_cols)

    # Reorder columns to a canonical order (optional but nice)
    # target_order = ["image","wrist_image","state","actions","timestamp",
    #                 "frame_index","episode_index","index","task_index"]
    # keep = [c for c in target_order if c in ep_ds.column_names]
    # rest = [c for c in ep_ds.column_names if c not in keep]
    # ep_ds = ep_ds.select(keep + rest)

    ep_name = f"episode_{ep_id:0{PAD}d}.parquet"
    ep_path = os.path.join(chunk_dir, ep_name)
    ep_ds.to_parquet(ep_path)

# Write meta/info.json exactly as requested
info = {
    "codebase_version": CODEBASE_VERSION,
    "robot_type": ROBOT_TYPE,
    "total_episodes": int(num_eps),
    "total_frames": int(total_frames),
    "total_tasks": int(total_tasks),
    "total_videos": int(TOTAL_VIDEOS),
    "total_chunks": int(total_chunks),
    "chunks_size": int(CHUNK_SIZE),
    "fps": int(FPS),
    "splits": {
        "train": f"0:{num_eps}"
    },
    "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    "features": {
        "image": {
            "dtype": "image",
            "shape": [256, 256, 3],
            "names": ["height", "width", "channel"]
        },
        "wrist_image": {
            "dtype": "image",
            "shape": [256, 256, 3],
            "names": ["height", "width", "channel"]
        },
        "state": {
            "dtype": "float32",
            "shape": [8],
            "names": ["state"]
        },
        "actions": {
            "dtype": "float32",
            "shape": [7],
            "names": ["actions"]
        },
        "timestamp": {
            "dtype": "float32",
            "shape": [1],
            "names": None
        },
        "frame_index": {
            "dtype": "int64",
            "shape": [1],
            "names": None
        },
        "episode_index": {
            "dtype": "int64",
            "shape": [1],
            "names": None
        },
        "index": {
            "dtype": "int64",
            "shape": [1],
            "names": None
        },
        "task_index": {
            "dtype": "int64",
            "shape": [1],
            "names": None
        }
    }
}

with open(os.path.join(OUT_DIR, "meta", "info.json"), "w") as f:
    json.dump(info, f, indent=4)

print(f"✅ Export complete: {num_eps} episodes, {total_frames} frames")
print(f"   Wrote: {OUT_DIR}/meta/info.json")
