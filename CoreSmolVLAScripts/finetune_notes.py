"""
export TOKENIZERS_PARALLELISM=false

lerobot-train \
    --policy.path=lerobot/smolvla_base \
    --dataset.repo_id=scand-train \
    --dataset.root=miniproject/lerobot_dataset/train \
    --policy.push_to_hub=false \
    --policy.device=cuda \
    --policy.n_obs_steps=3 \
    --policy.use_amp=true \
    --rename_map='{"observation.images.camera1": "observation.images.camera1"}' \
    --dataset.image_transforms.enable=true \
    --batch_size=32 \
    --steps=20000 \
    --wandb.enable=false \
    --log_freq=100 \
    --save_freq=1000 \
    --eval_freq=1000 \
    --output_dir=outputs/smolvla_base > miniproject/outputs/smolvla_base/training_log.txt 2>&1

lerobot-train \
    --policy.path=lerobot/smolvla_base \
    --dataset.repo_id=scand-train \
    --dataset.root=miniproject/lerobot_full_dataset/train \
    --policy.push_to_hub=false \
    --policy.device=cuda \
    --policy.use_amp=true \
    --dataset.image_transforms.enable=true \
    --batch_size=16 \
    --steps=30000 \
    --wandb.enable=false \
    --log_freq=100 \
    --save_freq=1000 \
    --output_dir=outputs/smolvla_base > miniproject/outputs/smolvla_base/training_log.txt 2>&1

lerobot-train \
    --policy.path=lerobot/smolvla_base \
    --policy.pretrained_path=outputs/smolvla_base/checkpoints/last \
    --dataset.repo_id=scand-train \
    --dataset.root=miniproject/lerobot_full_dataset/train \
    --policy.push_to_hub=false \
    --resume=true \
    --policy.device=cuda \
    --policy.n_obs_steps=1 \
    --policy.use_amp=true \
    --dataset.image_transforms.enable=true \
    --batch_size=16 \
    --steps=50000 \
    --wandb.enable=false \
    --log_freq=100 \
    --save_freq=1000 \
    --eval_freq=1000 \
    --optimizer.type=adamw \
    --optimizer.lr=0.0001 \
    --optimizer.weight_decay=0.0000000001 \
    --optimizer.eps=0.00000001 \
    --optimizer.grad_clip_norm=10.0 \
    --optimizer.betas=[0.9,0.95] \
    --output_dir=outputs/smolvla_base > miniproject/outputs/smolvla_base/training_log.txt 2>&1


lerobot-eval \
    --policy.pretrained_path=miniproject/outputs/smolvla_base/checkpoints/last \
    --policy.dataset_paths=miniproject/lerobot_dataset/test \
    --eval.batch_size=32 \
    --output_dir=miniproject/outputs/smolvla_eval_test

  
lerobot-dataset-viz \
    --repo-id scand-train \
    --root miniproject/lerobot_dataset/train \
    --mode local \
    --episode-index 0

"""