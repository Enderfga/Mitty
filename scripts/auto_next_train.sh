#!/bin/bash
#
# 自动化脚本：等待当前训练和数据处理完成，复制数据，启动新训练
# 用法: nohup bash scripts/auto_next_train.sh > auto_train.log 2>&1 &
#

set -e

# ==================== 配置 ====================
LOG_FILE="/raid/fga/Mitty/auto_train.log"
CHECK_INTERVAL=60  # 每60秒检查一次

# 源目录和目标目录
SRC_DIR="/home/cloud-user/fga/policy/danze/robot4_result/robot4"
DST_HUMAN="/home/cloud-user/fga/Mitty/dataset/Masquerade/human"
DST_ROBOT="/home/cloud-user/fga/Mitty/dataset/Masquerade/robot"

# 数据数量
TOTAL_SAMPLES=151  # 0-150

# ==================== 日志函数 ====================
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# ==================== 检测函数 ====================

check_training_done() {
    if pgrep -f "bash scripts/train_policy.sh" > /dev/null 2>&1; then
        return 1
    fi
    if pgrep -f "wan2_trainer_plus.py.*train.yaml" > /dev/null 2>&1; then
        return 1
    fi
    return 0
}

check_batch_process_done() {
    if pgrep -f "bash batch_process.sh" > /dev/null 2>&1; then
        return 1
    fi
    return 0
}

copy_available_data() {
    log "开始复制数据文件..."
    rm -f "$DST_HUMAN"/*.mp4 "$DST_ROBOT"/*.mp4 2>/dev/null || true

    local copied=0
    local skipped=""

    for i in $(seq 0 $((TOTAL_SAMPLES - 1))); do
        if [ -f "$SRC_DIR/$i/align.mp4" ] && [ -f "$SRC_DIR/$i/video_overlay_Panda_single_arm.mp4" ]; then
            cp "$SRC_DIR/$i/align.mp4" "$DST_HUMAN/$i.mp4"
            cp "$SRC_DIR/$i/video_overlay_Panda_single_arm.mp4" "$DST_ROBOT/$i.mp4"
            copied=$((copied + 1))
        else
            skipped="$skipped $i"
        fi
    done

    local human_count=$(ls "$DST_HUMAN"/*.mp4 2>/dev/null | wc -l)
    local robot_count=$(ls "$DST_ROBOT"/*.mp4 2>/dev/null | wc -l)

    log "复制完成: $copied 个 pair"
    log "human=$human_count robot=$robot_count"

    if [ -n "$skipped" ]; then
        log "跳过缺失的:$skipped"
    fi

    # 至少要有一些数据才能训练
    if [ $copied -lt 10 ]; then
        log "ERROR: 可用数据太少 ($copied < 10)"
        return 1
    fi
    return 0
}

# ==================== 主流程 ====================

main() {
    log "=========================================="
    log "自动训练脚本启动"
    log "=========================================="

    # 等待当前训练完成
    log "[1/4] 等待当前训练完成..."
    while ! check_training_done; do
        log "训练仍在运行，等待 ${CHECK_INTERVAL}s..."
        sleep $CHECK_INTERVAL
    done
    log "[1/4] 训练已完成!"

    # 等待数据处理完成
    log "[2/4] 等待数据处理完成..."
    while ! check_batch_process_done; do
        log "数据处理仍在运行，等待 ${CHECK_INTERVAL}s..."
        sleep $CHECK_INTERVAL
    done
    log "[2/4] 数据处理已完成!"

    # 复制可用数据（跳过缺失的）
    log "[3/4] 复制可用数据..."
    if ! copy_available_data; then
        log "ERROR: 数据复制失败"; exit 1
    fi
    log "[3/4] 数据复制完成!"

    # 启动新训练
    log "[4/4] 启动新训练..."
    cd /raid/fga/Mitty
    nohup bash scripts/train_policy.sh > /raid/fga/Mitty/train_stage2.log 2>&1 &
    log "训练已启动, 日志: /raid/fga/Mitty/train_stage2.log"

    log "=========================================="
    log "所有任务完成!"
    log "=========================================="
}

main
