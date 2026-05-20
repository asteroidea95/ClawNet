# 母模型（Mother Model）

**路由模型拼盘的核心组件** — 一个不写代码、只做归因的"产品经理"模型。

## 架构总览

```
3 个模块, ~140M 总参数, INT4 量化后 ~84MB

┌─ 意图编码器 (Intent Encoder, ~30M)
│  理解用户的自然语言指令
│  4 层 Transformer Encoder → 512-dim 意图向量
│
├─ 代码编码器 (Code Encoder, ~80M)
│  理解代码结构和逻辑
│  8 层 Transformer Encoder + AST-Aware 位置编码 → 768-dim 代码向量
│
├─ 融合解码器 (Fusion Decoder, ~30M)
│  意图 × 代码 交叉注意力 → 输出需求描述
│  2 层 Decoder → token 序列
│
└─ 轻量推理服务 HTTP API
```

## 快速开始

### 环境

```bash
pip install -r requirements.txt
# 可选（数据合成用）
export DEEPSEEK_API_KEY="your_key_here"
```

### 一键训练 + 量化 + 启动

```bash
chmod +x scripts/*.sh

# Stage 1: 合成训练数据 + 训练
bash scripts/train.sh

# Stage 2: 量化
bash scripts/quantize.sh

# Stage 3: 启动服务
bash scripts/serve.sh
```

或者分步执行：

```bash
# 1. 生成训练数据
python data/synthesize.py --languages go python ts --samples 50 --output ./data/train.jsonl

# 2. SFT 微调
python train/train_mother.py --stage sft --data ./data/train.jsonl --batch-size 4 --epochs 5

# 3. INT4 量化
python inference/quantize.py --checkpoint ./checkpoints/mother_model_final.pt --output ./checkpoints/mother_int4.pt

# 4. 启动推理服务
python inference/serve.py --checkpoint ./checkpoints/mother_int4.pt
```

### 调用 API

```bash
curl -X POST http://localhost:8888/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "intent": "看看这段代码有什么问题",
    "code": "package main\\nfunc main() {\\n    fmt.Println(undefinedVar)\\n}",
    "language": "go"
  }'
```

## 训练数据

纯合成，不需要人工标注：

1. **代码生成** → 用模板/DeepSeek API 生成正常代码
2. **Bug 注入** → 自动注入 9 种常见错误（undefined var, type mismatch, missing return...）
3. **编译验证** → Go/Python/TypeScript 编译器验证 + 正弦则静态分析兜底
4. **构造四元组** → {意图, 代码, 出错子模型, 错误描述}

**编译器就是你的标注员。** 不需要任何人工标注。

## 推理管线

```
用户输入 → 意图编码器 → [意图向量]
                            ↓
代码输入 → 代码编码器 → [代码向量]
                            ↓
                      交叉注意力融合
                            ↓
                      轻量 Decoder → 需求描述
                            ↓
                  回馈给子模型
```

## 硬件要求

| 硬件 | 能否跑 | 加载速度 | 推理速度 |
|------|--------|---------|---------|
| GTX 1060 4GB | ✅ INT4 峰值 ~300MB | ~1s | ~0.5s/turn |
| RTX 3060 12GB | ✅ 无压力 | <0.5s | <0.1s/turn |
| CPU (8 核) | ✅ 慢但可用 | ~2s | ~2s/turn |
| 手机/树莓派 | ✅ 需要额外优化 | ~3s | ~5s/turn |

## 项目结构

```
mother_model/
├── config.py                      # 全局配置
├── tokenizer/
│   └── code_tokenizer.py          # 代码结构感知分词器
├── model/
│   └── mother_model.py            # 核心模型架构
├── data/
│   └── synthesize.py              # 训练数据合成
├── train/
│   ├── dataset.py                 # 数据集
│   └── train_mother.py            # 训练脚本
├── inference/
│   ├── quantize.py                # INT4 量化
│   └── serve.py                   # 推理服务
├── scripts/
│   ├── train.sh                   # 一键训练
│   ├── quantize.sh                # 一键量化
│   └── serve.sh                   # 一键启动
└── requirements.txt
```
