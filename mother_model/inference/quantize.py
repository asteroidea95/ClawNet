"""
INT4 量化脚本 - 将训练好的母模型压缩到 ~84MB

量化策略: 分模块量化为 INT4, 将 FP16 权重压缩 4 倍。
GTX 1060 4GB 加载后峰值显存 ~300MB。

用法:
  python quantize.py --checkpoint ./checkpoints/best.pt --output ./checkpoints/mother_int4
"""

import os
import sys
import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import MotherModelConfig
from model.mother_model import MotherModel


def quantize_tensor(tensor: torch.Tensor, num_bits: int = 4) -> Dict:
    """
    将 FP16 tensor 量化为 INT4
    
    Args:
        tensor: FP16 权重
        num_bits: 量化位数 (4)
    
    Returns:
        {
            "qweight": 量化后的整数权重 (int8, packed)
            "scale": 缩放因子 (float16)
            "zero_point": 零点 (int8)
            "shape": 原始形状
            "num_bits": 量化位数
        }
    """
    orig_shape = tensor.shape
    tensor_flat = tensor.flatten()
    
    # 分块量化（每 128 个值一组，减少量化误差）
    group_size = 128
    n_groups = (tensor_flat.numel() + group_size - 1) // group_size
    
    # Pad to group boundary
    pad_len = n_groups * group_size - tensor_flat.numel()
    if pad_len > 0:
        tensor_flat = torch.cat([tensor_flat, torch.zeros(pad_len, device=tensor.device, dtype=tensor.dtype)])
    
    tensor_grouped = tensor_flat.view(n_groups, group_size)
    
    # 每组的 min/max
    min_vals = tensor_grouped.min(dim=1, keepdim=True)[0]
    max_vals = tensor_grouped.max(dim=1, keepdim=True)[0]
    
    # INT4 范围 [-8, 7]
    qmin = -2 ** (num_bits - 1)
    qmax = 2 ** (num_bits - 1) - 1
    
    scale = (max_vals - min_vals) / (qmax - qmin)
    scale = scale.clamp(min=1e-8)
    
    zero_point = torch.round(-min_vals / scale) + qmin
    zero_point = zero_point.clamp(qmin, qmax)
    
    # 量化
    qweight = torch.round(tensor_grouped / scale + zero_point)
    qweight = qweight.clamp(qmin, qmax).to(torch.int8)
    
    # Pack 两个 INT4 到一个 INT8
    # 偶数索引存低 4 位，奇数索引存高 4 位
    qweight_packed = torch.zeros(n_groups, group_size // 2, dtype=torch.int8, device=qweight.device)
    qweight_packed = (qweight[:, ::2] & 0x0F) | ((qweight[:, 1::2] & 0x0F) << 4)
    
    return {
        "qweight": qweight_packed.cpu(),
        "scale": scale.squeeze(-1).half().cpu(),
        "zero_point": zero_point.squeeze(-1).to(torch.int8).cpu(),
        "shape": orig_shape,
        "num_bits": num_bits,
        "group_size": group_size,
    }


class QuantizedLinear(nn.Module):
    """INT4 量化的 Linear 层（推理时在线反量化）"""
    
    def __init__(self, quantized_data: Dict, orig_module: nn.Linear):
        super().__init__()
        self.in_features = orig_module.in_features
        self.out_features = orig_module.out_features
        self.bias = orig_module.bias is not None
        if orig_module.bias is not None:
            self.register_buffer("bias_weight", orig_module.bias.half().cpu())
        else:
            self.bias_weight = None
        
        self.register_buffer("qweight", quantized_data["qweight"])
        self.register_buffer("scale", quantized_data["scale"])
        self.register_buffer("zero_point", quantized_data["zero_point"])
        self.group_size = quantized_data["group_size"]
        self.shape = quantized_data["shape"]
    
    def dequantize(self) -> torch.Tensor:
        """反量化权重"""
        n_groups, half_size = self.qweight.shape
        group_size = self.group_size
        
        # Unpack
        low = self.qweight & 0x0F
        high = (self.qweight >> 4) & 0x0F
        qweight = torch.stack([low, high], dim=2).reshape(n_groups, group_size)
        
        # 去量化
        deq_weight = (qweight.float() - self.zero_point[:, None].float()) * self.scale[:, None].float()
        
        # 恢复形状
        weight = deq_weight.reshape(-1)[:self.shape[0] * self.shape[1]].reshape(self.shape)
        return weight.half()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.dequantize().to(x.device, dtype=x.dtype)
        out = torch.nn.functional.linear(x, weight)
        if self.bias_weight is not None:
            out = out + self.bias_weight.to(x.device, dtype=x.dtype)
        return out


def quantize_model(
    model: MotherModel,
    num_bits: int = 4,
    output_path: str = "./checkpoints/mother_int4",
):
    """
    量化母模型所有 Linear 层
    
    只量化权重，不量化 embedding/层归一化（对精度影响小但收益有限）
    """
    print(f"Quantizing model to INT{num_bits}...")
    
    quantized_sd = {}
    total_params = 0
    quantized_params = 0
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            quantized_data = quantize_tensor(module.weight.data, num_bits)
            quantized_sd[name] = quantized_data
            quantized_params += module.weight.numel()
            print(f"  Quantized: {name:50s} {str(list(module.weight.shape)):20s} -> {quantized_data['qweight'].numel() * 1} bytes")
        total_params += 1
    
    # 保存非量化参数（embeddings, norms）
    non_quantized = {}
    for name, param in model.named_parameters():
        if not any(qn in name for qn in ['.weight']) or 'embed' in name or 'norm' in name:
            non_quantized[name] = param.half().cpu()
    
    output = {
        'quantized': quantized_sd,
        'non_quantized': non_quantized,
        'config': model.config,
        'num_bits': num_bits,
    }
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)
    
    # 计算压缩率
    fp16_size = sum(p.numel() * 2 for p in model.parameters()) / 1024 / 1024
    int4_size = sum(v['qweight'].numel() * 1 for v in quantized_sd.values()) / 1024 / 1024
    int4_size += sum(p.numel() * 2 for n, p in non_quantized.items()) / 1024 / 1024
    
    print(f"\n{'='*50}")
    print(f"Quantization complete!")
    print(f"  FP16 size: {fp16_size:.1f}MB")
    print(f"  INT4 size: {int4_size:.1f}MB")
    print(f"  Compression ratio: {fp16_size / int4_size:.1f}x")
    print(f"  Saved to: {output_path}")
    
    return output


def load_quantized(config: MotherModelConfig, path: str) -> MotherModel:
    """加载量化后的模型（反量化回推理）"""
    print(f"Loading quantized model from {path}...")
    data = torch.load(path, map_location='cpu')
    
    model = MotherModel(config)
    
    # 加载非量化参数
    for name, param in data['non_quantized'].items():
        target = model
        parts = name.split('.')
        for p in parts[:-1]:
            target = getattr(target, p)
        if hasattr(target, parts[-1]):
            getattr(target, parts[-1]).data = param
    
    print("  Non-quantized params loaded")
    print("  Quantized weights will be dequantized on-the-fly during inference")
    
    return model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="母模型 INT4 量化")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="训练好的检查点路径")
    parser.add_argument("--output", type=str, default="./checkpoints/mother_int4.pt",
                        help="量化模型输出路径")
    parser.add_argument("--bits", type=int, default=4, choices=[4],
                        help="量化位数（当前仅支持 4）")
    
    args = parser.parse_args()
    
    # 加载原始模型
    config = MotherModelConfig()
    model = MotherModel(config)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Original model parameters:")
    for name, count in model.get_params_by_module().items():
        print(f"  {name}: {count:,} ({count/1e6:.1f}M)")
    total = model.get_total_params()
    print(f"  Total: {total:,} ({total/1e6:.1f}M)")
    
    # 量化
    quantize_model(model, args.bits, args.output)
