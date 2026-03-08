#!/bin/bash

cd /home/guian/data/Mitty
export PYTHONPATH=/home/guian/data/Mitty:$PYTHONPATH

# 配置
MODEL="Wan-AI/Wan2.2-T2V-A14B-Diffusers"
CKPT="/home/guian/data/Mitty/ckpt/step=19000.ckpt"
INPUT_DIR="/home/guian/data/Mitty/robot/human/pick_and_place"
OUTPUT_BASE="sliding_window_inference"

# 可用GPU (排除1号卡)
GPUS=(0 2 3 4 5 6 7)
NUM_GPUS=${#GPUS[@]}

# 创建临时配置文件
CONFIG_TMP="/tmp/sliding_window_config.yaml"
cat > "${CONFIG_TMP}" << EOF
experiment_project: 'Mitty'
experiment_name: '${OUTPUT_BASE}'
model_id: "${MODEL}"
output_root: '/home/guian/data/Mitty/output'
use_DiffSynth: True
use_drop_text: True
use_lora: True

training:

dataset:
  height: 224
  width: 416
  sample_n_frames: 81
  fps: 24
  num_workers: 8
  shuffle: False
  drop_last: False
  pin_memory: True
EOF

# 查找所有video_L.mp4文件（按数字顺序排序）
VIDEOS=($(find "${INPUT_DIR}" -type f -name "video_L.mp4" | sort -V))
NUM_VIDEOS=${#VIDEOS[@]}

# 限制处理前105个视频
MAX_VIDEOS=105
if [ ${NUM_VIDEOS} -gt ${MAX_VIDEOS} ]; then
    echo "[INFO] Found ${NUM_VIDEOS} videos, processing first ${MAX_VIDEOS}"
    NUM_VIDEOS=${MAX_VIDEOS}
else
    echo "[INFO] Found ${NUM_VIDEOS} videos to process"
fi

echo "[INFO] Using ${NUM_GPUS} GPUs: ${GPUS[@]} for parallel processing"
echo "[INFO] Checkpoint: ${CKPT}"
echo "[INFO] Output: /home/guian/data/Mitty/output/${OUTPUT_BASE}"

# 按顺序在7张卡上并行处理
for i in $(seq 0 $((NUM_VIDEOS - 1))); do
    VIDEO="${VIDEOS[$i]}"
    GPU_IDX=$((i % NUM_GPUS))
    GPU=${GPUS[$GPU_IDX]}

    # 获取视频所在的子目录名称作为实验名
    SUBDIR=$(basename $(dirname "${VIDEO}"))
    EXP_NAME="${OUTPUT_BASE}/${SUBDIR}"

    echo "[INFO] [$(($i + 1))/${NUM_VIDEOS}] Starting ${SUBDIR}/video_L.mp4 on GPU ${GPU}"

    CUDA_VISIBLE_DEVICES=${GPU} python src/wan2_inference_sliding_window.py \
        --config "${CONFIG_TMP}" \
        --video_path "${VIDEO}" \
        --ckpt_path "${CKPT}" \
        model_id="${MODEL}" \
        experiment_name="${EXP_NAME}" \
        &

    # 每7个视频等待一批完成
    if [ $((($i + 1) % NUM_GPUS)) -eq 0 ] && [ $i -lt $((NUM_VIDEOS - 1)) ]; then
        echo "[INFO] Waiting for current batch (videos $(($i - NUM_GPUS + 2))-$(($i + 1))) to complete..."
        wait
    fi
done

echo "[INFO] All ${NUM_VIDEOS} inference jobs completed!"
echo "[INFO] Results saved to: /home/guian/data/Mitty/output/${OUTPUT_BASE}/"

bash /home/guian/data/Mitty/scripts/run_inference_with_prompt.sh
