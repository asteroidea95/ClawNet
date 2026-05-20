#!/bin/bash
# 生成训练数据 → 启动 SFT 训练
set -e

MODEL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$MODEL_DIR"

echo "=== Step 1: 生成训练数据 ==="
echo "Using DeepSeek API (如果有设置 DEEPSEEK_API_KEY 环境变量的话)"
echo ""

# 生成数据（如果没有 API key，只用模板也能生成 100+ 样本）
python data/synthesize.py \
    --languages go python ts \
    --samples 50 \
    --output ./data/train.jsonl \
    --no-compile

echo ""
echo "=== Step 2: 启动 SFT 微调 ==="
echo ""

python train/train_mother.py \
    --stage sft \
    --data ./data/train.jsonl \
    --batch-size 4 \
    --epochs 5 \
    --lr 2e-4 \
    --output-dir ./checkpoints \
    --log-interval 10

echo ""
echo "=== Done! ==="
echo "Training complete. Model saved to ./checkpoints/mother_model_final.pt"
echo "Run 'python inference/serve.py --checkpoint ./checkpoints/mother_model_final.pt' to start serving"
