"""
母模型简捷推理脚本
在云电脑 / 云服务器上直接跑，不启动 HTTP 服务

用法:
  python run_mother.py --checkpoint ../checkpoints/mother_int4.pt
"""

import sys, torch, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import MotherModelConfig
from model.mother_model import MotherModel
from tokenizer.code_tokenizer import CodeTokenizer


def encode_text(tokenizer, text, max_len=256):
    """编码自然语言文本 → input_ids + mask"""
    tokens = tokenizer.tokenize(text)
    ids = [1]  # <bos>
    for token_text, token_type, _, _ in tokens:
        if len(ids) >= max_len - 1:
            break
        tid = tokenizer._token_to_id(token_text, token_type)
        ids.append(tid)
    ids.append(2)  # <eos>
    ids = ids[:max_len]
    ids += [0] * (max_len - len(ids))
    mask = [i != 0 for i in ids]
    return ids, mask


def encode_code(tokenizer, code, max_len=2048):
    """编码代码 → input_ids + depth_ids + sibling_ids + mask"""
    code_ids, depth_ids, sibling_ids = tokenizer.encode(code)
    if len(code_ids) > max_len:
        code_ids = code_ids[:max_len]
        depth_ids = depth_ids[:max_len]
        sibling_ids = sibling_ids[:max_len]
    pad = max_len - len(code_ids)
    code_ids += [0] * pad
    depth_ids += [0] * pad
    sibling_ids += [0] * pad
    mask = [i != 0 for i in code_ids]
    return code_ids, depth_ids, sibling_ids, mask


def main():
    parser = argparse.ArgumentParser(description="Mother Model 推理")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/mother_int4.pt",
                        help="模型检查点路径")
    parser.add_argument("--intent", type=str, default="检查这段代码的安全性",
                        help="自然语言意图")
    parser.add_argument("--code", type=str, default="",
                        help="要分析的代码（可选，为空则只分析意图）")
    parser.add_argument("--max-new", type=int, default=256, help="最大生成 token 数")
    parser.add_argument("--temperature", type=float, default=0.7, help="采样温度")
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")
    
    # 1. 初始化模型
    print("加载模型...")
    config = MotherModelConfig()
    model = MotherModel(config)
    
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    
    # 支持多种 checkpoint 格式
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'quantized' in checkpoint:
        print("检测到量化权重，使用量化加载...")
        from inference.quantize import QuantizedLinear
        state_dict = {}
        for name, qdata in checkpoint['quantized'].items():
            state_dict[name] = qdata
        for name, param in checkpoint['non_quantized'].items():
            state_dict[name] = param
    else:
        state_dict = checkpoint
    
    try:
        model.load_state_dict(state_dict, strict=False)
    except Exception as e:
        print(f"load_state_dict 警告: {e}")
        print("尝试非严格模式继续...")
        model.load_state_dict(state_dict, strict=False)
    
    model = model.to(device)
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    by_module = model.get_params_by_module()
    print(f"模型参数量: {total_params:,}")
    for k, v in by_module.items():
        print(f"  {k}: {v:,}")
    
    # 2. 初始化 tokenizer
    tokenizer = CodeTokenizer()
    print(f"词表大小: {tokenizer.vocab_size:,}")
    
    # 3. 编码输入
    print(f"意图: {args.intent}")
    intent_ids, intent_mask = encode_text(tokenizer, args.intent)
    
    intent_tensor = torch.tensor([intent_ids], dtype=torch.long, device=device)
    intent_mask_tensor = torch.tensor([intent_mask], dtype=torch.bool, device=device)
    
    if args.code:
        print(f"代码: {args.code[:100]}...")
        code_ids, depth_ids, sibling_ids, code_mask = encode_code(tokenizer, args.code)
    else:
        print("无代码输入，仅编码意图")
        code_ids, depth_ids, sibling_ids, code_mask = [], [], [], []
    
    if code_ids:
        code_tensor = torch.tensor([code_ids], dtype=torch.long, device=device)
        depth_tensor = torch.tensor([depth_ids], dtype=torch.long, device=device)
        sibling_tensor = torch.tensor([sibling_ids], dtype=torch.long, device=device)
        code_mask_tensor = torch.tensor([code_mask], dtype=torch.bool, device=device)
    else:
        code_tensor = code_depth_tensor = code_sibling_tensor = None
        code_mask_tensor = None
    
    # 4. 推理
    print("生成中...")
    with torch.no_grad():
        # 编码意图
        intent_vec = model.intent_encoder(intent_tensor, intent_mask_tensor)
        print(f"  意图向量: {intent_vec.shape}")
        
        if code_tensor is not None:
            code_vec = model.code_encoder(
                code_tensor, depth_tensor, sibling_tensor, code_mask_tensor
            )
        else:
            code_vec = torch.zeros(1, config.code_encoder.output_dim, device=device)
        print(f"  代码向量: {code_vec.shape}")
        
        # FusionDecoder 自回归生成（内部处理 cross-attention）
        output_ids = model.fusion_decoder.generate(
            intent_vec, code_vec,
            max_new_tokens=args.max_new,
            temperature=args.temperature,
            device=device,
        )
    
    # 5. 解码输出
    output_text = tokenizer.decode(output_ids[0].tolist())
    print(f"\n输出: {output_text}")
    
    # 保存结果
    with open("mother_output.txt", "w", encoding="utf-8") as f:
        f.write(output_text)
    print("结果已保存到 mother_output.txt")


if __name__ == "__main__":
    main()
