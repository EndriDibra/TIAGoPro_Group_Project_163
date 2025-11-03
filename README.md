# TIAGoPro_Group_Project_163
This is our group project from Aalborg University, regarding TIAGoPro robot, contributing to its Explainable Social Navigation approach.

## SmolVLA Core

### download_data_aria2.py
Downloads SCAND dataset rosbag files from Texas Digital Library using aria2c for optimized parallel downloads. Features include resumable downloads, multiple connections per file (16 by default), and concurrent file downloads (8 by default). Automatically verifies downloaded files and handles interruptions gracefully. Requires aria2c to be installed (`sudo apt-get install aria2`).

### format_dataset.py
Converts ROS bag files into LeRobot-compatible datasets with temporal multi-camera support. Processes odometry and image data at 4 FPS, creating episodes of 30 frames each. Supports dynamic instruction generation based on robot trajectory or static instructions. Features configurable temporal camera offsets for historical frame integration, automatic train/test splitting, and comprehensive validation. Outputs state vectors containing current and past velocities, with 32-dimensional action vectors (2D velocity commands + padding).

### finetune_notes.py
Shell command collection for training and evaluating SmolVLA models on the SCAND dataset. Contains three progressive training configurations: initial training (20k steps, batch size 32), full dataset training (30k steps, batch size 16), and fine-tuning with custom optimizer settings (50k steps, AdamW with gradient clipping). Includes evaluation and dataset visualization commands. All outputs are logged to training_log.txt.

### test.py
Inference evaluation script for trained SmolVLA policies. Loads checkpointed models and runs predictions on test datasets, computing MSE/MAE metrics across action dimensions. Features per-dimension error analysis, prediction consistency checks, and temporal trajectory evaluation. Uses training statistics for proper normalization/denormalization of states and actions. Outputs comprehensive metrics including first-step errors, horizon variance, and sample predictions, saving results to NumPy archives for further analysis.
