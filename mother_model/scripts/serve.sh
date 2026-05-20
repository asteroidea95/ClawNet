#!/bin/bash
# 启动母模型推理服务
set -e

MODEL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$MODEL_DIR"

CHECKPOINT="${1:-./checkpoints/mother_model_final.pt}"

if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint not found at $CHECKPOINT"
    echo ""
    echo "Usage: $0 [checkpoint_path]"
    echo "  (default: ./checkpoints/mother_model_final.pt)"
    echo ""
    echo "First train the model:"
    echo "  bash scripts/train.sh"
    exit 1
fi

echo "Starting Mother Model service..."
echo "  Checkpoint: $CHECKPOINT"
echo ""

python inference/serve.py \
    --checkpoint "$CHECKPOINT" \
    --host 0.0.0.0 \
    --port 8888 \
    --device auto
