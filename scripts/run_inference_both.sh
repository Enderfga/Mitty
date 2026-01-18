#!/bin/bash

cd /raid/fga/Mitty
export PYTHONPATH=/raid/fga/Mitty:$PYTHONPATH

# 基础模型
MODEL="/raid/fga/Mitty/ckpt/Wan2.2-T2V-A14B-Diffusers"

# LoRA checkpoint
LORA_EPIC="/home/cloud-user/fga/Mitty/output/policy_finetune/checkpoints/step=19000.ckpt"

# 源视频目录
SRC_VIDEO_DIR="/home/cloud-user/fga/Mitty/real_videos"
# 临时目录基础路径
TMP_BASE="/tmp/mitty_inference"
# 统一的输出目录名
OUTPUT_NAME="inference_all_real"

# 获取所有视频文件
VIDEOS=($(ls ${SRC_VIDEO_DIR}/*.mp4 | sort -V))
NUM_VIDEOS=${#VIDEOS[@]}

echo "[INFO] Found ${NUM_VIDEOS} videos to process"

# 可用GPU列表 (1-6 内存充足)
GPUS=(1 2 3 4 5 6 0)
NUM_GPUS=${#GPUS[@]}

# 为每个视频创建单独目录并在不同GPU上并行运行
for i in "${!VIDEOS[@]}"; do
    VIDEO="${VIDEOS[$i]}"
    VIDEO_NAME=$(basename "$VIDEO")
    GPU_IDX=$((i % NUM_GPUS))
    GPU=${GPUS[$GPU_IDX]}

    # 创建临时目录并链接视频
    TMP_DIR="${TMP_BASE}/gpu${GPU}_${VIDEO_NAME%.mp4}"
    mkdir -p "${TMP_DIR}"
    ln -sf "${VIDEO}" "${TMP_DIR}/${VIDEO_NAME}"

    # 创建临时配置文件
    CONFIG_TMP="${TMP_DIR}/inference.yaml"
    cat > "${CONFIG_TMP}" << EOF
experiment_project: 'Mitty'
experiment_name: '${OUTPUT_NAME}'
model_id: "${MODEL}"
output_root: '/raid/fga/Mitty/output'
use_DiffSynth: True
use_drop_text: True
use_lora: True

training:

dataset:
  video_root: '${TMP_DIR}/'
  video_root2: null
  first_root: null
  is_one2three: True
  height: 224
  width: 416
  sample_n_frames: 81
  fps: 24
  num_workers: 8
  shuffle: True
  drop_last: True
  pin_memory: True
EOF

    echo "[INFO] Starting video ${VIDEO_NAME} on GPU ${GPU}..."
    CUDA_VISIBLE_DEVICES=${GPU} python src/wan2_inference_plus.py \
        --config "${CONFIG_TMP}" \
        --ckpt_path "${LORA_EPIC}" \
        model_id="${MODEL}" \
        experiment_name="${OUTPUT_NAME}" \
        &

    # 如果GPU用完了，等待当前批次完成
    if [ $((($i + 1) % NUM_GPUS)) -eq 0 ] && [ $i -lt $((NUM_VIDEOS - 1)) ]; then
        echo "[INFO] Waiting for current batch to complete..."
        wait
    fi
done

wait

# 清理临时目录
rm -rf "${TMP_BASE}"

echo "[INFO] All ${NUM_VIDEOS} inference jobs completed!"
