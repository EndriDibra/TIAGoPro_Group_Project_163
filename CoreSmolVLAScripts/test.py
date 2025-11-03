import torch
import time
import os
import re
import json
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from transformers import AutoProcessor
from PIL import Image
import numpy as np
from tqdm import tqdm

# Settings
MODEL_PATH = os.path.abspath("miniproject/outputs/smolvla_base_test_finetune/checkpoints/001000/pretrained_model")
DATASET_PATH = os.path.abspath("miniproject/lerobot_dataset/test")
TRAIN_STATS_PATH = "miniproject/lerobot_dataset/train/meta/stats.json"
CAMERA_KEYS = ["observation.images.camera1"] #, "observation.images.camera2", "observation.images.camera3"]
MAX_SAMPLES = None  # Set to None to use all samples

# Load model and extract checkpoint number
checkpoint_match = re.search(r'/checkpoints/(\d+)/', MODEL_PATH)
checkpoint_num = int(checkpoint_match.group(1)) if checkpoint_match else 0

policy = SmolVLAPolicy.from_pretrained(MODEL_PATH).to("cuda")
policy.eval()

processor = AutoProcessor.from_pretrained(policy.config.vlm_model_name)
policy.language_tokenizer = processor.tokenizer

# Load test dataset
print(f"Loading test dataset from: {DATASET_PATH}")
dataset = LeRobotDataset(DATASET_PATH)
print(f"Test dataset size: {len(dataset)}")

# LOAD TRAIN STATISTICS
print(f"\nLoading TRAIN statistics from: {TRAIN_STATS_PATH}")
with open(TRAIN_STATS_PATH, 'r') as f:
    train_stats = json.load(f)

# Extract normalization parameters
state_mean = torch.tensor(train_stats['observation.state']['mean'], dtype=torch.float32)
state_std = torch.tensor(train_stats['observation.state']['std'], dtype=torch.float32)
action_mean = torch.tensor(train_stats['action']['mean'], dtype=torch.float32)
action_std = torch.tensor(train_stats['action']['std'], dtype=torch.float32)

print(f"\nTRAIN statistics:")
print(f"  State mean: {state_mean[:2].tolist()}")
print(f"  State std: {state_std[:2].tolist()}")
print(f"  Action mean: {action_mean[:2].tolist()}")
print(f"  Action std: {action_std[:2].tolist()}")

# Get expected dimensions
expected_state_dim = policy.model.state_proj.weight.shape[1]
print(f"\nExpected state dim: {expected_state_dim}")

# Pad normalization tensors to match expected dimensions
if len(state_mean) < expected_state_dim:
    state_mean_padded = torch.zeros(expected_state_dim)
    state_mean_padded[:len(state_mean)] = state_mean
    state_mean = state_mean_padded
    
    state_std_padded = torch.ones(expected_state_dim)
    state_std_padded[:len(state_std)] = state_std
    state_std = state_std_padded

state_mean = state_mean.to("cuda")
state_std = state_std.to("cuda")
action_mean = action_mean.to("cuda")
action_std = action_std.to("cuda")

def normalize_state(state):
    """Normalize state using TRAIN statistics"""
    return (state - state_mean) / (state_std + 1e-8)

def denormalize_action(action):
    """Denormalize action using TRAIN statistics"""
    return action * action_std + action_mean

# Run inference on test samples
num_samples = len(dataset) if MAX_SAMPLES is None else min(MAX_SAMPLES, len(dataset))
mse_errors = []
mae_errors = []
action_dim_errors = []
predicted_actions_all = []
gt_actions_all = []
avg_inference_time = 0.0

print(f"\nRunning inference on {num_samples} samples...")
for idx in tqdm(range(num_samples), desc="Inference", unit="sample"):
    sample = dataset[idx]
    
    # Get state and pad to expected dimension
    state = sample["observation.state"].unsqueeze(0).to("cuda")
    if state.shape[1] < expected_state_dim:
        padding = torch.zeros(1, expected_state_dim - state.shape[1], device="cuda")
        state = torch.cat([state, padding], dim=1)
    
    # NORMALIZE STATE using TRAIN statistics
    state_normalized = normalize_state(state)
    
    # Get temporal images (3 frames)
    img_tensors = []
    for cam_key in CAMERA_KEYS:
        img_tensor = sample[cam_key].permute(1,2,0).cpu().numpy()
        img_pil = Image.fromarray((img_tensor * 255).astype(np.uint8))
        img_tensors.append(img_pil)
    
    # Process all images together
    images = processor.image_processor(
        img_tensors,
        return_tensors="pt",
        do_rescale=False
    ).to("cuda")
    
    # Handle task/instruction
    if "task" in sample:
        task = [sample["task"]]
    elif "language_instruction" in sample:
        task = [sample["language_instruction"]]
    else:
        task = ["default task"]

    lang_tokens = processor.tokenizer(task, return_tensors="pt", padding=True).to("cuda")
    lang_attention_mask = lang_tokens["attention_mask"].bool()
    
    pixel_values = images["pixel_values"]
    pixel_values_list = [pixel_values[:, i] for i in range(pixel_values.shape[1])]
    img_masks_list = [torch.ones(1, dtype=torch.bool, device="cuda") for _ in range(len(pixel_values_list))]
    
    # Predict action (model outputs NORMALIZED actions)
    start_time = time.time()
    with torch.no_grad():
        predicted_action_normalized = policy.model.sample_actions(
            pixel_values_list, img_masks_list, 
            lang_tokens["input_ids"], lang_attention_mask, state_normalized
        )
    end_time = time.time()
    avg_inference_time += (end_time - start_time) / num_samples
    
    # DENORMALIZE ACTION to get physical units
    predicted_action = denormalize_action(predicted_action_normalized)
    
    # Get ground truth action (already in physical units)
    gt_action = sample["action"].to("cuda")
    
    # Store actions for trajectory analysis
    predicted_actions_all.append(predicted_action.squeeze().cpu().numpy())
    gt_actions_all.append(gt_action.cpu().numpy())
    
    # Calculate errors
    mse = torch.mean((predicted_action.squeeze() - gt_action) ** 2).item()
    mae = torch.mean(torch.abs(predicted_action.squeeze() - gt_action)).item()
    
    # Per-dimension absolute error
    dim_error = torch.abs(predicted_action.squeeze() - gt_action).cpu().numpy()
    action_dim_errors.append(dim_error)
    
    mse_errors.append(mse)
    mae_errors.append(mae)

# Convert to numpy array
action_dim_errors = np.array(action_dim_errors)
predicted_actions_all = np.array(predicted_actions_all)
gt_actions_all = np.array(gt_actions_all)

# Print results
print("\n" + "="*80)
print("INFERENCE RESULTS")
print("="*80)
print(f"Action dim errors shape: {action_dim_errors.shape}")
print(f"Predicted actions shape: {predicted_actions_all.shape}")
print(f"Ground truth actions shape: {gt_actions_all.shape}")
print(f"Prediction horizon: {action_dim_errors.shape[1]}")
print(f"Action dimensions: {action_dim_errors.shape[2]}")
print(f"\nMean Squared Error (MSE):  {np.mean(mse_errors):.6f} ± {np.std(mse_errors):.6f}")
print(f"Mean Absolute Error (MAE): {np.mean(mae_errors):.6f} ± {np.std(mae_errors):.6f}")
print(f"Max MSE: {np.max(mse_errors):.6f}")
print(f"Min MSE: {np.min(mse_errors):.6f}")
print(f"Avg inference time: {avg_inference_time:.6f} s")
print(f"Max GPU memory used: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")

# Per-dimension statistics
print(f"\nPer-dimension MAE statistics (averaged over all samples and horizon):")
print(f"  Dim 0 (Linear velocity) - TRAINED:")
print(f"    {action_dim_errors[:, :, 0].mean():.6f} ± {action_dim_errors[:, :, 0].std():.6f}")
if action_dim_errors.shape[2] > 1:
    print(f"  Dim 1 (Angular velocity) - TRAINED:")
    print(f"    {action_dim_errors[:, :, 1].mean():.6f} ± {action_dim_errors[:, :, 1].std():.6f}")
if action_dim_errors.shape[2] > 2:
    print(f"  Remaining dimensions (2-{action_dim_errors.shape[2]-1}) - TRAINED TO ZERO:")
    remaining_mean = action_dim_errors[:, :, 2:].mean()
    remaining_std = action_dim_errors[:, :, 2:].std()
    print(f"    Mean: {remaining_mean:.6f} ± {remaining_std:.6f}")

# Extract first-step errors
first_step_lin_errors = action_dim_errors[:, 0, 0]
first_step_ang_errors = action_dim_errors[:, 0, 1]

print(f"\nFirst-step prediction errors:")
print(f"  Linear velocity:  {first_step_lin_errors.mean():.6f} ± {first_step_lin_errors.std():.6f}")
print(f"  Angular velocity: {first_step_ang_errors.mean():.6f} ± {first_step_ang_errors.std():.6f}")

# Calculate prediction variance across horizon
lin_vel_variance = predicted_actions_all[:, :, 0].std(axis=1)
ang_vel_variance = predicted_actions_all[:, :, 1].std(axis=1)

print(f"\nPrediction consistency (std dev across horizon):")
print(f"  Linear velocity:  {lin_vel_variance.mean():.6f} ± {lin_vel_variance.std():.6f}")
print(f"  Angular velocity: {ang_vel_variance.mean():.6f} ± {ang_vel_variance.std():.6f}")

# Show sample predictions vs ground truth
print(f"\nSample predictions (first 5):")
for i in range(min(5, len(predicted_actions_all))):
    print(f"Sample {i}:")
    print(f"  GT:   lin={gt_actions_all[i, 0]:.4f}, ang={gt_actions_all[i, 1]:.4f}")
    print(f"  Pred: lin={predicted_actions_all[i, 0, 0]:.4f}, ang={predicted_actions_all[i, 0, 1]:.4f}")
    print(f"  Error: lin={first_step_lin_errors[i]:.4f}, ang={first_step_ang_errors[i]:.4f}")

print("="*80)

# Save results
output_dir = "miniproject/outputs/smolvla_base"
os.makedirs(output_dir, exist_ok=True)

output_file = f'{output_dir}/inference_results_step{checkpoint_num}.npz'
np.savez(output_file,
         action_dim_errors=action_dim_errors,
         mse_errors=np.array(mse_errors),
         mae_errors=np.array(mae_errors),
         first_step_lin_errors=first_step_lin_errors,
         first_step_ang_errors=first_step_ang_errors,
         lin_vel_variance=lin_vel_variance,
         ang_vel_variance=ang_vel_variance,
         predicted_actions=predicted_actions_all,
         gt_actions=gt_actions_all,
         checkpoint=checkpoint_num,
         num_samples=num_samples,
         avg_inference_time=avg_inference_time,
         train_state_mean=state_mean.cpu().numpy(),
         train_state_std=state_std.cpu().numpy(),
         train_action_mean=action_mean.cpu().numpy(),
         train_action_std=action_std.cpu().numpy())

print(f"\nResults saved to: {output_file}")