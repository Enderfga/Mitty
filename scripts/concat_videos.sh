#!/bin/bash

# 水平拼接 8 个推理结果视频
# 用法: ./concat_videos.sh <输入目录> [输出文件]
# 示例: ./concat_videos.sh /home/cloud-user/fga/Mitty/output/inference_all_Masquerade

INPUT_DIR="${1:?请提供输入目录路径}"
OUTPUT_FILE="${2:-${INPUT_DIR}/concat_all.mp4}"

# 视频列表（按数字顺序）
VIDEOS=(0 26 89 102 121 133 136 150)

echo "[INFO] Input directory: ${INPUT_DIR}"
echo "[INFO] Output file: ${OUTPUT_FILE}"

# 查找每个视频文件
INPUTS=""
FILTER=""
for i in "${!VIDEOS[@]}"; do
    idx="${VIDEOS[$i]}"
    # 查找子目录下的 mp4 文件
    VIDEO_FILE=$(find "${INPUT_DIR}/${idx}" -name "*.mp4" -type f 2>/dev/null | head -1)

    if [ -z "$VIDEO_FILE" ]; then
        echo "[ERROR] Video not found for index ${idx}"
        exit 1
    fi

    echo "[INFO] Found: ${VIDEO_FILE}"
    # -r 16 强制以 16fps 读取，保留所有帧
    INPUTS="${INPUTS} -r 16 -i ${VIDEO_FILE}"
    FILTER="${FILTER}[${i}:v]"
done

# 8 个视频水平拼接
FILTER="${FILTER}hstack=inputs=8[v]"

echo "[INFO] Concatenating 8 videos horizontally (16 fps, 81 frames, ~5s)..."
ffmpeg -y ${INPUTS} -filter_complex "${FILTER}" -map "[v]" -c:v libx264 -crf 18 "${OUTPUT_FILE}"

if [ $? -eq 0 ]; then
    echo "[DONE] Output saved to: ${OUTPUT_FILE}"
else
    echo "[ERROR] Failed to concatenate videos"
    exit 1
fi
