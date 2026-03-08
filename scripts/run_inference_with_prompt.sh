#!/bin/bash

# 用 mitty 环境的绝对路径，避免多环境叠加冲突
PYTHON="/home/guian/data/miniconda3/envs/mitty/bin/python"

cd /home/guian/data/Mitty
export PYTHONPATH=/home/guian/data/Mitty:$PYTHONPATH

# 配置
MODEL="Wan-AI/Wan2.2-T2V-A14B-Diffusers"
CKPT="/home/guian/data/Mitty/output/human2robot_finetune/checkpoints/step=21100.ckpt"
INPUT_DIR="/home/guian/data/Mitty/dataset/human_data_for_openvla/front_hand_mitty_inference/pick_and_place"
OUTPUT_BASE="prompt_based_inference"

# 可用GPU (排除1号卡)
GPUS=(0 2 3 4 5 6 7)
NUM_GPUS=${#GPUS[@]}

# 三种 prompt (与训练一致: 480x640, 49帧, fps=16)
PROMPTS=("<sks> front view" "<sks> wrist view" "<sks> back view")
VIEW_TYPES=("front" "wrist" "back")
NUM_PROMPTS=${#PROMPTS[@]}

# 创建临时配置文件
CONFIG_TMP="/tmp/prompt_inference_config.yaml"
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
  height: 480
  width: 640
  sample_n_frames: 49
  fps: 16
  num_workers: 8
  shuffle: False
  drop_last: False
  pin_memory: True
EOF

# 查找所有子目录 (按数字排序)
SAMPLE_IDS=($(ls -1 "${INPUT_DIR}" | sort -n))
NUM_SAMPLES=${#SAMPLE_IDS[@]}

# 每个 sample 的视频约 357 帧, 49 帧一个 chunk ≈ 8 chunks
# 总 GPU 任务: NUM_SAMPLES x NUM_PROMPTS = 183, 每个任务内部滑窗处理所有 chunks
TOTAL_TASKS=$((NUM_SAMPLES * NUM_PROMPTS))

echo "[INFO] Found ${NUM_SAMPLES} samples in ${INPUT_DIR}"
echo "[INFO] ${TOTAL_TASKS} GPU tasks (${NUM_SAMPLES} samples x ${NUM_PROMPTS} views)"
echo "[INFO] Each task internally processes ~8 chunks (sliding window, 49 frames/chunk)"
echo "[INFO] Using ${NUM_GPUS} GPUs: ${GPUS[@]}"
echo "[INFO] Checkpoint: ${CKPT}"
echo "[INFO] Resolution: 480x640, frames: 49, fps: 16"
echo "[INFO] Output structure:"
echo "  /home/guian/data/Mitty/output/${OUTPUT_BASE}/{front,wrist,back}/{sample_id}/full.mp4"

# 任务计数器
task_idx=0

for sample_id in "${SAMPLE_IDS[@]}"; do
    VIDEO_PATH="${INPUT_DIR}/${sample_id}/video_L.mp4"

    if [ ! -f "${VIDEO_PATH}" ]; then
        echo "[WARN] ${VIDEO_PATH} not found, skipping"
        continue
    fi

    for prompt_idx in $(seq 0 $((NUM_PROMPTS - 1))); do
        PROMPT="${PROMPTS[$prompt_idx]}"
        VIEW_TYPE="${VIEW_TYPES[$prompt_idx]}"

        GPU_IDX=$((task_idx % NUM_GPUS))
        GPU=${GPUS[$GPU_IDX]}

        echo "[INFO] [$((task_idx + 1))/${TOTAL_TASKS}] sample=${sample_id} view=${VIEW_TYPE} GPU=${GPU}"

        CUDA_VISIBLE_DEVICES=${GPU} ${PYTHON} -u src/wan2_inference_with_prompt.py \
            --config "${CONFIG_TMP}" \
            --video_path "${VIDEO_PATH}" \
            --ckpt_path "${CKPT}" \
            --prompt "${PROMPT}" \
            --view_type "${VIEW_TYPE}" \
            --sample_id "${sample_id}" \
            model_id="${MODEL}" \
            experiment_name="${OUTPUT_BASE}" \
            &

        task_idx=$((task_idx + 1))

        # 每 NUM_GPUS 个任务等待一批完成
        if [ $((task_idx % NUM_GPUS)) -eq 0 ] && [ $task_idx -lt $TOTAL_TASKS ]; then
            echo "[INFO] Waiting for batch (tasks $((task_idx - NUM_GPUS + 1))-${task_idx}) ..."
            wait
        fi
    done
done

wait

echo "[INFO] All ${task_idx}/${TOTAL_TASKS} inference jobs completed!"
echo "[INFO] Results saved to:"
echo "  /home/guian/data/Mitty/output/${OUTPUT_BASE}/front/"
echo "  /home/guian/data/Mitty/output/${OUTPUT_BASE}/wrist/"
echo "  /home/guian/data/Mitty/output/${OUTPUT_BASE}/back/"

# ========== 压缩并上传到 HuggingFace ==========
OUTPUT_DIR="/home/guian/data/Mitty/output/${OUTPUT_BASE}"
UPLOAD_DIR="/home/guian/data/Mitty/output/${OUTPUT_BASE}_upload"
mkdir -p "${UPLOAD_DIR}"

echo "[INFO] Compressing view folders..."
for view in front wrist back; do
    if [ -d "${OUTPUT_DIR}/${view}" ]; then
        echo "[INFO] Compressing ${view}..."
        cd "${OUTPUT_DIR}"
        tar -czf "${UPLOAD_DIR}/${view}.tar.gz" "${view}/"
        echo "[INFO] Created ${UPLOAD_DIR}/${view}.tar.gz ($(du -sh "${UPLOAD_DIR}/${view}.tar.gz" | cut -f1))"
    fi
done

echo "[INFO] Uploading to HuggingFace: Enderfga/mitty_three_view ..."
${PYTHON} -c "
from huggingface_hub import HfApi
import os

api = HfApi()
upload_dir = '${UPLOAD_DIR}'

for fname in os.listdir(upload_dir):
    fpath = os.path.join(upload_dir, fname)
    if fname.endswith('.tar.gz'):
        print(f'Uploading {fname}...')
        api.upload_file(
            path_or_fileobj=fpath,
            path_in_repo=fname,
            repo_id='Enderfga/mitty_three_view',
            repo_type='dataset',
        )
        print(f'Uploaded {fname} successfully')

print('All uploads complete!')
"

echo "[INFO] Done! All videos compressed and uploaded to https://huggingface.co/datasets/Enderfga/mitty_three_view"
