#!/bin/bash

# 从 Masquerade 数据集随机抽取 human/robot 配对视频，拼接成 16fps 5秒 GIF
# 用法: ./random_pair_gif.sh [输出文件]

DATA_DIR="/home/cloud-user/fga/Mitty/dataset/real"
OUTPUT="${1:-/tmp/random_pair.gif}"

# 获取所有可用的视频编号（取 human 和 robot 的交集）
HUMAN_DIR="${DATA_DIR}/human"
ROBOT_DIR="${DATA_DIR}/robot"

# 获取共有的视频编号
COMMON_IDS=($(comm -12 \
    <(ls "${HUMAN_DIR}"/*.mp4 2>/dev/null | xargs -n1 basename | sed 's/.mp4//' | sort -n) \
    <(ls "${ROBOT_DIR}"/*.mp4 2>/dev/null | xargs -n1 basename | sed 's/.mp4//' | sort -n)))

if [ ${#COMMON_IDS[@]} -eq 0 ]; then
    echo "[ERROR] No matching videos found"
    exit 1
fi

# 随机选择一个
RANDOM_IDX=$((RANDOM % ${#COMMON_IDS[@]}))
SELECTED_ID="${COMMON_IDS[$RANDOM_IDX]}"

HUMAN_VIDEO="${HUMAN_DIR}/${SELECTED_ID}.mp4"
ROBOT_VIDEO="${ROBOT_DIR}/${SELECTED_ID}.mp4"

echo "[INFO] Selected video ID: ${SELECTED_ID}"
echo "[INFO] Human: ${HUMAN_VIDEO}"
echo "[INFO] Robot: ${ROBOT_VIDEO}"

# 获取帧数用于等距采样
H_FRAMES=$(ffprobe -v error -select_streams v:0 -count_packets -show_entries stream=nb_read_packets -of csv=p=0 "${HUMAN_VIDEO}")
R_FRAMES=$(ffprobe -v error -select_streams v:0 -count_packets -show_entries stream=nb_read_packets -of csv=p=0 "${ROBOT_VIDEO}")

echo "[INFO] Human frames: ${H_FRAMES}, Robot frames: ${R_FRAMES}"

# 目标: 81帧 @ 16fps = 5.0625s
TARGET_FRAMES=81

# 计算采样步长
H_STEP=$(echo "scale=6; $H_FRAMES/$TARGET_FRAMES" | bc)
R_STEP=$(echo "scale=6; $R_FRAMES/$TARGET_FRAMES" | bc)

echo "[INFO] Generating GIF (16fps, ~5s)..."

# 等距采样 + 拼接 + 转 GIF
ffmpeg -y \
    -i "${HUMAN_VIDEO}" \
    -i "${ROBOT_VIDEO}" \
    -filter_complex "
        [0:v]select='lt(mod(n\,${H_STEP})\,1)',setpts=N/16/TB[h];
        [1:v]select='lt(mod(n\,${R_STEP})\,1)',setpts=N/16/TB[r];
        [h][r]hstack=inputs=2[v];
        [v]fps=16,split[s0][s1];
        [s0]palettegen=max_colors=256:stats_mode=diff[p];
        [s1][p]paletteuse=dither=bayer:bayer_scale=5
    " \
    -frames:v ${TARGET_FRAMES} \
    "${OUTPUT}"

if [ $? -eq 0 ]; then
    echo "[DONE] Output: ${OUTPUT}"
    echo "[INFO] Run again for a different random pair!"
else
    echo "[ERROR] Failed to generate GIF"
    exit 1
fi
