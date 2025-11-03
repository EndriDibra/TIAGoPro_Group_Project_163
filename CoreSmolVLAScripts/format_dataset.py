import ast
import gc
import os
import shutil
from tqdm import tqdm
import numpy as np
import pandas as pd
import cv2
import base64
from bagpy import bagreader
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ROSBAGDIR = "miniproject/rosbags"
OUTDIR = "miniproject/lerobot_full_dataset"
EPISODELENGTH = 30 # number of frames per episode
FPS = 4.0 # frames per second
IMAGESIZE = (256, 256) # Resize images to this size
ACTIONDIM = 32 # action vector dimension (2 used for linear/angular velocity, rest padded with zeros)

# Instruction generation mode: "static" or "dynamic"
INSTRUCTION_MODE = "dynamic"  # Set to "static" to use fixed instruction
STATIC_INSTRUCTION = "Continue along the path."

# Temporal camera offsets (in seconds before current frame)
# Set to None or empty list to disable temporal cameras
TEMPORAL_OFFSETS = [0.1, 0.2]  # camera2 at -0.1s, camera3 at -0.2s

# Build features dynamically based on temporal offsets
# State dimension: 2 (current) + 2 * num_temporal_offsets (past states)
state_dim = 2 + (2 * len(TEMPORAL_OFFSETS) if TEMPORAL_OFFSETS else 0)
state_names = ["linear_vel", "angular_vel"]
if TEMPORAL_OFFSETS:
    for i in range(len(TEMPORAL_OFFSETS)):
        state_names.extend([f"linear_vel_t-{i+1}", f"angular_vel_t-{i+1}"])

FEATURES = {
    "observation.images.camera1": {
        "dtype": "image",
        "shape": (3, IMAGESIZE[0], IMAGESIZE[1]),
        "names": ["channel", "height", "width"]
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (state_dim,),
        "names": state_names
    },
    "action": {
        "dtype": "float32",
        "shape": (ACTIONDIM,),
        "names": [f"action_{d}" for d in range(ACTIONDIM)]
    }
}

# Add temporal cameras if offsets are defined
if TEMPORAL_OFFSETS:
    for i, offset in enumerate(TEMPORAL_OFFSETS, start=2):
        FEATURES[f"observation.images.camera{i}"] = {
            "dtype": "image",
            "shape": (3, IMAGESIZE[0], IMAGESIZE[1]),
            "names": ["channel", "height", "width"]
        }

def detect_topics(bag_path):
    bag = bagreader(bag_path)
    topics = list(bag.topics)
    odom_topic = None
    img_topic = None
    
    # First pass: prioritize specific topic patterns
    for t in topics:
        if "odom" in t and "camera" not in t: 
            odom_topic = t
            break
    
    # Look for actual image topics (avoid camera_info)
    for t in topics:
        if "rgb" in t and "camera_info" not in t: 
            img_topic = t
            break
    
    if not img_topic:
        for t in topics:
            if "image_raw" in t and "camera_info" not in t: 
                img_topic = t
                break
    
    if not img_topic:
        for t in topics:
            if "image" in t and "camera_info" not in t and "compressed" not in t: 
                img_topic = t
                break
    
    if not img_topic:
        for t in topics:
            if "camera" in t and "camera_info" not in t and "image" in t: 
                img_topic = t
                break
    
    if not odom_topic:
        for t in topics:
            if "velocity" in t or "twist" in t: 
                odom_topic = t
                break
    
    if not odom_topic or not img_topic:
        print(f"[WARN] Could not detect topics for {bag_path}.")
        print(f"       Available topics: {topics}")
        print(f"       Selected odom: {odom_topic}, image: {img_topic}")
    
    return odom_topic, img_topic

def find_image_data_column(df):
    """Find the column name that contains image data."""
    possible_names = ['data', 'data.data', 'image.data', 'img.data']
    for name in possible_names:
        if name in df.columns:
            return name
    # If none found, look for any column containing 'data'
    for col in df.columns:
        if 'data' in col.lower():
            return col
    print(f"[WARN] Could not find image data column. Available columns: {df.columns.tolist()}")
    return None

def decode_ros_image(imgdata):
    try:
        # If imgdata is a string representation of bytes, convert it
        if isinstance(imgdata, str):
            if imgdata.startswith("b'") or imgdata.startswith('b"'):
                imgdata = ast.literal_eval(imgdata)
            else:
                # Try base64 or hex decoding
                try:
                    imgdata = base64.b64decode(imgdata)
                except Exception:
                    imgdata = bytes.fromhex(imgdata)
        
        # Now imgdata should be bytes
        np_arr = np.frombuffer(imgdata, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if img is None:
            print(f"[WARN] cv2.imdecode returned None")
            return None
        
        img = cv2.resize(img, IMAGESIZE)
        img = img.transpose(2, 0, 1)
        return img
    except Exception as e:
        print(f"[WARN] Failed to decode image: {e}")
        return None

def generate_instruction(start_pos, end_pos):
    """
    Generate a natural language instruction based on start and end positions.
    
    Args:
        start_pos: tuple (x, y) in meters
        end_pos: tuple (x, y) in meters
    
    Returns:
        str: Natural language instruction
    """
    dx = end_pos[0] - start_pos[0]  # forward/backward (positive = forward)
    dy = end_pos[1] - start_pos[1]  # left/right (positive = left)
    
    # Calculate distance and direction
    distance = np.sqrt(dx**2 + dy**2)
    
    # Determine forward/backward direction
    if abs(dx) < 0.5:
        forward_str = ""
    elif dx > 0:
        forward_str = f"go {abs(dx):.1f}m forward"
    else:
        forward_str = f"go {abs(dx):.1f}m backward"
    
    # Determine left/right direction
    if abs(dy) < 0.5:
        lateral_str = ""
    elif dy > 0:
        lateral_str = f"{abs(dy):.1f}m left"
    else:
        lateral_str = f"{abs(dy):.1f}m right"
    
    # Combine instructions
    if forward_str and lateral_str:
        instruction = f"{forward_str} and {lateral_str}"
    elif forward_str:
        instruction = forward_str
    elif lateral_str:
        instruction = f"move {lateral_str}"
    else:
        instruction = "stay in place"
    
    return instruction.capitalize()

def process_bag_to_episodes(bag_path):
    odom_topic, img_topic = detect_topics(bag_path)
    if not odom_topic or not img_topic:
        print(f"[SKIP] No suitable odometry/image topic for {bag_path}")
        return []
    bag = bagreader(bag_path)
    try:
        odom_csv = bag.message_by_topic(odom_topic)
        img_csv = bag.message_by_topic(img_topic)
    except Exception as e:
        print(f"[SKIP] Error fetching topics: {e}")
        return []
    if not odom_csv or not os.path.exists(odom_csv):
        print(f"[SKIP] Odom data missing, skipping {bag_path}")
        return []
    if not img_csv or not os.path.exists(img_csv):
        print(f"[SKIP] Image data missing, skipping {bag_path}")
        return []
    
    odomdf = pd.read_csv(odom_csv)
    imgdf = pd.read_csv(img_csv)
    
    # Find the actual image data column name
    img_data_col = find_image_data_column(imgdf)
    if img_data_col is None:
        print(f"[SKIP] Could not find image data column in {bag_path}")
        return []
    
    odomdf = odomdf.rename(columns={"Time": "timestamp"}).astype({"timestamp": float})
    imgdf = imgdf.rename(columns={"Time": "timestamp"}).astype({"timestamp": float})
    
    # Sort once for efficient searching
    odomdf = odomdf.sort_values("timestamp").reset_index(drop=True)
    imgdf = imgdf.sort_values("timestamp").reset_index(drop=True)
    
    # Sync at specified FPS
    sampling_interval = 1.0 / FPS  # Calculate interval from FPS
    starttime = max(odomdf["timestamp"].min(), imgdf["timestamp"].min())
    endtime = min(odomdf["timestamp"].max(), imgdf["timestamp"].max())
    samplets = np.arange(starttime, endtime, sampling_interval)
    
    valid_rows = []
    odom_timestamps = odomdf["timestamp"].values
    img_timestamps = imgdf["timestamp"].values
    
    # Helper function to get closest image at a given timestamp
    def get_image_at_time(target_ts, img_df, img_ts_array):
        img_idx = np.searchsorted(img_ts_array, target_ts)
        if img_idx > 0 and img_idx < len(img_ts_array):
            if abs(img_ts_array[img_idx-1] - target_ts) < abs(img_ts_array[img_idx] - target_ts):
                img_idx = img_idx - 1
        elif img_idx >= len(img_ts_array):
            img_idx = len(img_ts_array) - 1
        elif img_idx == 0:
            img_idx = 0
        return decode_ros_image(img_df.iloc[img_idx][img_data_col])
    
    # Helper function to get odometry state at a given timestamp
    def get_odom_at_time(target_ts, odom_df, odom_ts_array):
        odom_idx = np.searchsorted(odom_ts_array, target_ts)
        if odom_idx > 0 and odom_idx < len(odom_ts_array):
            if abs(odom_ts_array[odom_idx-1] - target_ts) < abs(odom_ts_array[odom_idx] - target_ts):
                odom_idx = odom_idx - 1
        elif odom_idx >= len(odom_ts_array):
            odom_idx = len(odom_ts_array) - 1
        elif odom_idx == 0:
            odom_idx = 0
        
        try:
            odom_row = odom_df.iloc[odom_idx]
            return {
                "linear_v": float(odom_row["twist.twist.linear.x"]),
                "angular_v": float(odom_row["twist.twist.angular.z"]),
                "pos_x": float(odom_row["pose.pose.position.x"]),
                "pos_y": float(odom_row["pose.pose.position.y"]),
            }
        except Exception:
            return None
    
    for ts in samplets:
        # Get current image (camera1)
        img = get_image_at_time(ts, imgdf, img_timestamps)
        if img is None:
            continue
        
        # Get current odometry
        odom_data = get_odom_at_time(ts, odomdf, odom_timestamps)
        if odom_data is None:
            continue
        
        # Get temporal images and states (camera2, camera3, etc.)
        temporal_imgs = {}
        temporal_states = []  # Changed to list for concatenation
        if TEMPORAL_OFFSETS:
            for i, offset in enumerate(TEMPORAL_OFFSETS, start=2):
                past_ts = ts - offset
                past_img = get_image_at_time(past_ts, imgdf, img_timestamps)
                if past_img is None:
                    # If past image unavailable, use current image
                    past_img = img.copy()
                temporal_imgs[f"camera{i}"] = past_img
                
                # Get past odometry state
                past_odom = get_odom_at_time(past_ts, odomdf, odom_timestamps)
                if past_odom is None:
                    # If past state unavailable, use current state
                    temporal_states.extend([odom_data["linear_v"], odom_data["angular_v"]])
                else:
                    temporal_states.extend([past_odom["linear_v"], past_odom["angular_v"]])
            
        try:
            rowdata = {
                "timestamp": ts,
                "linear_v": odom_data["linear_v"],
                "angular_v": odom_data["angular_v"],
                "temporal_velocities": temporal_states,  # Store as list
                "pos_x": odom_data["pos_x"],
                "pos_y": odom_data["pos_y"],
                "img": img,
            }
            # Add temporal images to rowdata
            rowdata.update(temporal_imgs)
            valid_rows.append(rowdata)
        except Exception as e:
            print(f"[WARN] Error in odometry extraction at timestamp {ts}: {e}, skipping frame")
            continue
    
    if len(valid_rows) == 0:
        print(f"[SKIP] No valid frames in {bag_path}, skipping this bag.")
        return []
    
    # Free memory from DataFrames
    del odomdf
    del imgdf
    gc.collect()
    
    # Build episodes directly from list, not DataFrame
    episodes = []
    N = len(valid_rows)
    start_idx = 0

    while start_idx < N:
        end_idx = min(start_idx + EPISODELENGTH, N)
        
        if end_idx <= start_idx:  # Safety check
            break
        
        # Generate instruction for this episode
        if INSTRUCTION_MODE == "dynamic":
            start_pos = (valid_rows[start_idx]['pos_x'], valid_rows[start_idx]['pos_y'])
            end_pos = (valid_rows[end_idx-1]['pos_x'], valid_rows[end_idx-1]['pos_y'])
            instruction = generate_instruction(start_pos, end_pos)
        else:
            instruction = STATIC_INSTRUCTION
            
        ep = []
        for i in range(start_idx, end_idx):
            # Concatenate current state with temporal states
            state_values = [
                valid_rows[i]['linear_v'],
                valid_rows[i]['angular_v'],
            ]
            # Add temporal velocities
            state_values.extend(valid_rows[i]['temporal_velocities'])
            state = np.array(state_values, dtype=np.float32)
            
            # Store only the current action as a single vector [32]
            action = np.zeros(ACTIONDIM, dtype=np.float32)
            action[0] = valid_rows[i]['linear_v']
            action[1] = valid_rows[i]['angular_v']
            # Remaining 30 dimensions stay as zeros
            
            assert action.shape == (ACTIONDIM,), f"Action shape mismatch: {action.shape}"
            assert state.shape == (state_dim,), f"State shape mismatch: expected {state_dim}, got {state.shape}"
            
            frame_data = {
                "observation.images.camera1": valid_rows[i]["img"],
                "observation.state": state,
                "action": action,
                "task": instruction,
            }
            
            # Add temporal cameras
            if TEMPORAL_OFFSETS:
                for j in range(2, 2 + len(TEMPORAL_OFFSETS)):
                    frame_data[f"observation.images.camera{j}"] = valid_rows[i][f"camera{j}"]
            
            ep.append(frame_data)
        
        # Pad or truncate episode to exactly EPISODELENGTH frames
        if len(ep) > 0:
            ep = pad_or_truncate_episode(ep, EPISODELENGTH)
            episodes.append(ep)
        
        start_idx = end_idx
        
    print(f"[INFO] Created {len(episodes)} episodes from {bag_path} with {N} valid frames.")
    
    return episodes

def pad_or_truncate_episode(ep, target_length):
    """
    Pad or truncate an episode to exactly target_length frames.
    Padding uses the last valid frame's values (zeros for actions to avoid unintended commands).
    """
    if len(ep) == 0:
        return []
    
    if len(ep) > target_length:
        # Truncate to target length
        return ep[:target_length]
    
    # Pad with frames based on the last valid frame
    while len(ep) < target_length:
        last_frame = ep[-1]
        pad_frame = {
            "observation.images.camera1": last_frame["observation.images.camera1"].copy(),
            "observation.state": last_frame["observation.state"].copy(),
            # Use zeros for actions with shape (32,)
            "action": np.zeros(ACTIONDIM, dtype=np.float32),
            "task": last_frame["task"],
        }
        
        # Add temporal cameras if they exist
        if TEMPORAL_OFFSETS:
            for i in range(2, 2 + len(TEMPORAL_OFFSETS)):
                camera_key = f"observation.images.camera{i}"
                if camera_key in last_frame:
                    pad_frame[camera_key] = last_frame[camera_key].copy()
        
        ep.append(pad_frame)
    
    return ep

def get_split(filename):
    if filename.startswith("B_Jackal_"): return "test"
    elif filename.startswith("A_Jackal_"): return "test"
    else: return "train"

def main():
    bags_by_split = {"train": [], "test": []}
    for bagfile in os.listdir(ROSBAGDIR):
        file_path = os.path.join(ROSBAGDIR, bagfile)
        if os.path.isfile(file_path) and bagfile.endswith('.bag'):
            split = get_split(bagfile)
            bags_by_split[split].append(file_path)
    
    for split, bagfiles in bags_by_split.items():
        split_dir = os.path.join(OUTDIR, split)
        if os.path.exists(split_dir):
            print(f"[INFO] Removing existing dataset directory: {split_dir}")
            shutil.rmtree(split_dir)
        print(f"Processing split: {split} ({len(bagfiles)} bags)")
        dataset = LeRobotDataset.create(
            repo_id=f"scand-{split}",
            fps=FPS,
            features=FEATURES,
            root=split_dir,
            use_videos=True,
            robot_type="jackal",
        )
        episodes_added = 0
        for bagfile in tqdm(bagfiles, desc="Processing bags"):
            episodes = process_bag_to_episodes(bagfile)
            for ep in episodes:
                # Verify episode length before adding frames
                assert len(ep) == EPISODELENGTH, f"Episode length {len(ep)} does not match EPISODELENGTH {EPISODELENGTH}"
                for frame in ep:
                    dataset.add_frame(frame)
                dataset.save_episode()
                episodes_added += 1
            del episodes
            gc.collect()

        dataset.finalize()
        print(f"[INFO] Saved split '{split}' with {episodes_added} episodes at {split_dir}")

    # Validation
    print("\nVALIDATION PHASE")
    for split in ["train", "test"]:
        print(f"\nValidating split: {split}")
        dataset_path = os.path.join(OUTDIR, split)
        
        # Check if dataset exists and has required metadata files
        meta_dir = os.path.join(dataset_path, "meta")
        tasks_file = os.path.join(meta_dir, "tasks.parquet")
        info_file = os.path.join(meta_dir, "info.json")
        
        if not os.path.exists(meta_dir):
            print(f"[SKIP] Split '{split}' metadata does not exist, skipping validation")
            continue
        
        if not os.path.exists(tasks_file) or not os.path.exists(info_file):
            print(f"[SKIP] Split '{split}' is missing required metadata files (likely empty dataset), skipping validation")
            continue
            
        try:
            # Load using the same repo_id and root as when creating
            ds = LeRobotDataset(f"scand-{split}", root=dataset_path)
            
            if len(ds) == 0:
                print(f"[WARN] Split '{split}' has no data")
                continue
                
            print("Keys:", ds[0].keys())
            print("Image shape:", ds[0]["observation.images.camera1"].shape)
            print("State shape:", ds[0]["observation.state"].shape)
            print("Action shape:", ds[0]["action"].shape)
            
            # Verify shapes
            assert ds[0]["observation.images.camera1"].shape == (3, IMAGESIZE[0], IMAGESIZE[1]), \
                f"Expected image shape (3, {IMAGESIZE[0]}, {IMAGESIZE[1]}), got {ds[0]['observation.images.camera1'].shape}"
            assert ds[0]["observation.state"].shape == (state_dim,), \
                f"Expected state shape ({state_dim},), got {ds[0]['observation.state'].shape}"
            assert ds[0]["action"].shape == (ACTIONDIM,), \
                f"Expected action shape ({ACTIONDIM},), got {ds[0]['action'].shape}"
            
            # Verify temporal cameras if enabled
            if TEMPORAL_OFFSETS:
                for i in range(2, 2 + len(TEMPORAL_OFFSETS)):
                    camera_key = f"observation.images.camera{i}"
                    assert camera_key in ds[0], f"Missing {camera_key}"
                    assert ds[0][camera_key].shape == (3, IMAGESIZE[0], IMAGESIZE[1]), \
                        f"Expected {camera_key} shape (3, {IMAGESIZE[0]}, {IMAGESIZE[1]}), got {ds[0][camera_key].shape}"
                    print(f"{camera_key} shape:", ds[0][camera_key].shape)
            
            # Test dataloader with smaller batch
            batch_size = min(8, len(ds))
            loader = torch.utils.data.DataLoader(ds, batch_size=batch_size)
            for batch in loader:
                print("Batch keys:", batch.keys())
                print("Batch action shape:", batch["action"].shape)
                print("Batch state shape:", batch["observation.state"].shape)
                print("Batch image shape:", batch["observation.images.camera1"].shape)
                assert batch["observation.state"].shape[-1] == state_dim
                assert batch["action"].shape[1] == ACTIONDIM
                assert batch["observation.images.camera1"].shape[1] == 3
                
                # Verify temporal cameras in batch
                if TEMPORAL_OFFSETS:
                    for i in range(2, 2 + len(TEMPORAL_OFFSETS)):
                        camera_key = f"observation.images.camera{i}"
                        assert batch[camera_key].shape[1] == 3, \
                            f"Expected {camera_key} to have 3 channels in batch"
                
                # task is a list of strings, not a tensor
                assert len(batch["task"]) == batch_size, f"Expected {batch_size} tasks, got {len(batch['task'])}"
                break
            print(f"Split '{split}' PASSED")
        except Exception as e:
            print(f"[ERROR] Validation failed for split '{split}': {e}")
            import traceback
            traceback.print_exc()
            
    print("Validation complete.")

if __name__ == "__main__":
    main()