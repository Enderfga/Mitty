#!/bin/bash
# 单视频推理脚本：选择一种 prompt (front/wrist/back) 对输入视频做推理
#
# 用法:
#   bash scripts/run_inference_single.sh --view front --video /path/to/input.mp4
#   bash scripts/run_inference_single.sh --view wrist --video /path/to/input.mp4 --gpu 2
#   bash scripts/run_inference_single.sh --view back  --video /path/to/input.mp4 --output my_output

set -e

PYTHON="/home/guian/data/miniconda3/envs/mitty/bin/python"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"

# 默认值
MODEL="Wan-AI/Wan2.2-T2V-A14B-Diffusers"
CKPT="${PROJECT_DIR}/output/human2robot_finetune/checkpoints/step=21100.ckpt"
GPU=0
OUTPUT_NAME="single_inference"
VIEW=""
VIDEO=""

# 解析命令行参数
usage() {
    echo "用法: $0 --view <front|wrist|back> --video <path> [--gpu <id>] [--output <name>]"
    echo ""
    echo "必选参数:"
    echo "  --view    选择 prompt 类型: front, wrist, back"
    echo "  --video   输入视频路径"
    echo ""
    echo "可选参数:"
    echo "  --gpu     GPU 编号 (默认: 0)"
    echo "  --output  输出目录名 (默认: single_inference)"
    echo "  --ckpt    checkpoint 路径 (默认: output/human2robot_finetune/checkpoints/step=21100.ckpt)"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --view)   VIEW="$2";        shift 2 ;;
        --video)  VIDEO="$2";       shift 2 ;;
        --gpu)    GPU="$2";         shift 2 ;;
        --output) OUTPUT_NAME="$2"; shift 2 ;;
        --ckpt)   CKPT="$2";       shift 2 ;;
        -h|--help) usage ;;
        *) echo "未知参数: $1"; usage ;;
    esac
done

# 参数校验
if [ -z "${VIEW}" ] || [ -z "${VIDEO}" ]; then
    echo "[ERROR] --view 和 --video 是必选参数"
    usage
fi

if [ ! -f "${VIDEO}" ]; then
    echo "[ERROR] 视频文件不存在: ${VIDEO}"
    exit 1
fi

if [ ! -f "${CKPT}" ]; then
    echo "[ERROR] checkpoint 不存在: ${CKPT}"
    exit 1
fi

# 根据 view 选择 prompt
case "${VIEW}" in
    front) PROMPT="<sks> front view" ;;
    wrist) PROMPT="<sks> wrist view" ;;
    back)  PROMPT="<sks> back view"  ;;
    *)
        echo "[ERROR] --view 必须是 front, wrist, back 之一，当前: ${VIEW}"
        exit 1
        ;;
esac

# 从视频文件名提取 sample_id
SAMPLE_ID="$(basename "${VIDEO}" .mp4)"

# 创建临时配置
CONFIG_TMP="/tmp/single_inference_config_$$.yaml"
cat > "${CONFIG_TMP}" << EOF
experiment_project: 'Mitty'
experiment_name: '${OUTPUT_NAME}'
model_id: "${MODEL}"
output_root: '${PROJECT_DIR}/output'
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

echo "============================================"
echo " Mitty 单视频推理"
echo "============================================"
echo " View:       ${VIEW}"
echo " Prompt:     ${PROMPT}"
echo " Video:      ${VIDEO}"
echo " Checkpoint: ${CKPT}"
echo " GPU:        ${GPU}"
echo " Output:     ${PROJECT_DIR}/output/${OUTPUT_NAME}/${VIEW}/${SAMPLE_ID}/"
echo "============================================"

CUDA_VISIBLE_DEVICES=${GPU} ${PYTHON} -u src/wan2_inference_with_prompt.py \
    --config "${CONFIG_TMP}" \
    --video_path "${VIDEO}" \
    --ckpt_path "${CKPT}" \
    --prompt "${PROMPT}" \
    --view_type "${VIEW}" \
    --sample_id "${SAMPLE_ID}" \
    model_id="${MODEL}" \
    experiment_name="${OUTPUT_NAME}"

# 清理临时配置
rm -f "${CONFIG_TMP}"

echo ""
echo "[DONE] 推理完成！输出目录: ${PROJECT_DIR}/output/${OUTPUT_NAME}/${VIEW}/${SAMPLE_ID}/"
