#!/bin/bash
# 量化训练好的母模型为 INT4
set -e

MODEL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$MODEL_DIR"

CHECKPOINT="${1:-./checkpoints/mother_model_final.pt}"

if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint not found at $CHECKPOINT"
    exit 1
fi

echo "Quantizing $CHECKPOINT to INT4..."
python inference/quantize.py \
    --checkpoint "$CHECKPOINT" \
    --output ./checkpoints/mother_int4.pt \
    --bits 4

echo ""
echo "INT4 model saved to ./checkpoints/mother_int4.pt"
echo "Model size should be ~84MB"
echo ""
echo "To serve quantized model:"
echo "  python inference/serve.py --checkpoint ./checkpoints/mother_int4.pt"
