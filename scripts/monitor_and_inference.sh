#!/bin/bash

# Monitor training and run inference on all checkpoints when done

CKPT_DIR="/home/cloud-user/fga/Mitty/output/policy_finetune/checkpoints"
MODEL="/raid/fga/Mitty/ckpt/Wan2.2-T2V-A14B-Diffusers"
CONFIG="_configs/inference.yaml"
OUTPUT_BASE="/home/cloud-user/fga/Mitty/output/policy_finetune/inference_results"

cd /raid/fga/Mitty
export PYTHONPATH=/raid/fga/Mitty:$PYTHONPATH
source /home/cloud-user/fga/miniconda3/bin/activate mitty

echo "[$(date)] Starting monitor script..."
echo "[$(date)] Waiting for training to complete..."

# Wait for training to finish (check if wan2_trainer_plus.py is running)
while pgrep -f "wan2_trainer_plus.py" > /dev/null; do
    echo "[$(date)] Training still running... checking again in 60s"
    sleep 60
done

echo "[$(date)] Training completed! Starting inference on all checkpoints..."

# Create output directory
mkdir -p "${OUTPUT_BASE}"

# Find all checkpoints and sort by step number
CHECKPOINTS=$(ls ${CKPT_DIR}/step=*.ckpt 2>/dev/null | sort -t= -k2 -n)

if [ -z "$CHECKPOINTS" ]; then
    echo "[$(date)] No checkpoints found in ${CKPT_DIR}"
    exit 1
fi

echo "[$(date)] Found checkpoints:"
echo "$CHECKPOINTS"

# Run inference on each checkpoint
for CKPT in $CHECKPOINTS; do
    STEP=$(basename "$CKPT" | sed 's/step=\([0-9]*\)\.ckpt/\1/')
    echo ""
    echo "=============================================="
    echo "[$(date)] Running inference for step=${STEP}"
    echo "=============================================="

    CUDA_VISIBLE_DEVICES=7 python src/wan2_inference_plus.py \
        --config ${CONFIG} \
        --ckpt_path "${CKPT}" \
        model_id="${MODEL}" \
        experiment_name="inference_step_${STEP}"

    echo "[$(date)] Finished inference for step=${STEP}"
done

echo ""
echo "=============================================="
echo "[$(date)] All inference jobs completed!"
echo "Results saved in: ${OUTPUT_BASE}"
echo "=============================================="
