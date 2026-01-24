#!/bin/bash

# Policy finetune training script (8 GPUs)

# Activate conda environment
source /root/FGA/miniconda3/bin/activate mitty

# Set environment variables
export PYTHONPATH=/root/data/Mitty:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=false

# Optional: better performance on H200
export TORCH_CUDNN_V8_API_ENABLED=1

cd /root/data/Mitty

python src/wan2_trainer_plus.py \
    --config _configs/train.yaml \
    --ckpt_path /root/data/Mitty/ckpt/epic_14B.ckpt
