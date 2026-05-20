"""
母模型训练脚本

训练两阶段:
  Stage 1: 预训练代码编码器（在 GitHub 代码语料上）
  Stage 2: SFT 全链路微调（用合成数据）

环境要求:
  - Python 3.10+
  - PyTorch 2.0+ (CUDA optional, CPU也能训但慢)
  - 8GB+ RAM (推荐)
  - 4GB+ VRAM (GTX 1060 可用)

用法:
  # Stage 1: 预训练代码编码器
  python train_mother.py --stage pretrain --data ./data/corpus.jsonl

  # Stage 2: SFT 微调
  python train_mother.py --stage sft --data ./data/train.jsonl

  # 完整流程
  python train_mother.py --stage all --pretrain-data ./data/corpus.jsonl --sft-data ./data/train.jsonl
"""

import os
import sys
import json
import math
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from pathlib import Path

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import MotherModelConfig
from model.mother_model import MotherModel
from train.dataset import MotherModelDataset, create_dataloaders


def get_optimizer(model: nn.Module, config: MotherModelConfig):
    """创建 AdamW 优化器 + weight decay"""
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'norm' in name or 'bias' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    return AdamW([
        {'params': decay_params, 'weight_decay': 0.1},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ], lr=config.learning_rate, betas=(0.9, 0.95))


def get_lr_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """余弦学习率 + warmup"""
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer,
    scheduler,
    scaler,
    config: MotherModelConfig,
    epoch: int,
    device: torch.device,
    log_interval: int = 10,
) -> float:
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    total_tokens = 0
    start_time = time.time()
    
    for batch_idx, batch in enumerate(loader):
        # 移到设备
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        
        # 前向
        with autocast(device_type=device.type, enabled=device.type == 'cuda'):
            outputs = model(
                intent_input_ids=batch['intent_input_ids'],
                intent_mask=batch['intent_mask'],
                code_input_ids=batch['code_input_ids'],
                code_depth_ids=batch['code_depth_ids'],
                code_sibling_ids=batch['code_sibling_ids'],
                code_mask=batch['code_mask'],
                target_ids=batch['target_ids'],
                target_mask=batch['target_mask'],
            )
            loss = outputs['loss']
        
        # 反向
        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            if scheduler is not None:
                scheduler.step()
        
        total_loss += loss.item() * batch['target_ids'].size(0)
        total_tokens += batch['target_ids'].size(0)
        
        if (batch_idx + 1) % log_interval == 0:
            elapsed = time.time() - start_time
            avg_loss = total_loss / max(1, total_tokens)
            lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch} | Batch {batch_idx + 1}/{len(loader)} | "
                  f"Loss: {avg_loss:.4f} | LR: {lr:.2e} | Time: {elapsed:.1f}s")
    
    return total_loss / max(1, total_tokens)


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """验证"""
    model.eval()
    total_loss = 0.0
    total = 0
    
    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        
        outputs = model(
            intent_input_ids=batch['intent_input_ids'],
            intent_mask=batch['intent_mask'],
            code_input_ids=batch['code_input_ids'],
            code_depth_ids=batch['code_depth_ids'],
            code_sibling_ids=batch['code_sibling_ids'],
            code_mask=batch['code_mask'],
            target_ids=batch['target_ids'],
            target_mask=batch['target_mask'],
        )
        
        total_loss += outputs['loss'].item() * batch['target_ids'].size(0)
        total += batch['target_ids'].size(0)
    
    return total_loss / max(1, total)


def save_checkpoint(model: nn.Module, optimizer, epoch: int, loss: float, path: str):
    """保存检查点"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': model.config,
    }, path)
    print(f"  Checkpoint saved: {path}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="母模型训练")
    parser.add_argument("--stage", type=str, default="sft",
                        choices=["pretrain", "sft", "all"],
                        help="训练阶段")
    parser.add_argument("--data", type=str, required=True,
                        help="训练数据路径")
    parser.add_argument("--val-data", type=str, default="",
                        help="验证数据路径（可选）")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="批次大小")
    parser.add_argument("--epochs", type=int, default=5,
                        help="训练轮数")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="学习率")
    parser.add_argument("--output-dir", type=str, default="./checkpoints",
                        help="检查点输出目录")
    parser.add_argument("--resume", type=str, default="",
                        help="从检查点恢复训练")
    parser.add_argument("--cpu", action="store_true",
                        help="强制使用 CPU")
    parser.add_argument("--log-interval", type=int, default=10,
                        help="日志打印间隔（step）")
    
    args = parser.parse_args()
    
    # 设备
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB)")
    else:
        print("Using CPU (training will be slower)")
    
    # 配置
    config = MotherModelConfig()
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    config.max_epochs = args.epochs
    config.output_dir = args.output_dir
    
    # 模型
    model = MotherModel(config)
    
    start_epoch = 1
    if args.resume and Path(args.resume).exists():
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from checkpoint: {args.resume} (epoch {checkpoint['epoch']})")
    
    model = model.to(device)
    
    params_per_module = model.get_params_by_module()
    total_params = model.get_total_params()
    print(f"\nModel parameters:")
    for name, count in params_per_module.items():
        print(f"  {name}: {count:,} ({count/1e6:.1f}M)")
    print(f"  Total: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"  INT4 estimate: {total_params * 4 / 8 / 1024 / 1024:.0f}MB")
    
    # 优化器
    optimizer = get_optimizer(model, config)
    if args.resume and Path(args.resume).exists():
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    scaler = GradScaler(enabled=(device.type == 'cuda'))
    
    # 数据
    train_loader, val_loader = create_dataloaders(config, args.data, args.val_data)
    print(f"\nTraining samples: {len(train_loader.dataset)}")
    if val_loader:
        print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Scheduler
    total_steps = len(train_loader) * args.epochs
    scheduler = get_lr_scheduler(optimizer, config.warmup_steps, total_steps)
    
    # 训练
    best_val_loss = float('inf')
    print(f"\n{'='*60}")
    print(f"Starting training (stage={args.stage}, epochs={args.epochs})")
    print(f"{'='*60}")
    
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            config, epoch, device, args.log_interval
        )
        
        if val_loader:
            val_loss = validate(model, val_loader, device)
            print(f"  Epoch {epoch} complete | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, epoch, val_loss,
                                f"{args.output_dir}/best.pt")
        else:
            print(f"  Epoch {epoch} complete | Train loss: {train_loss:.4f}")
        
        # 定期保存
        if epoch % 5 == 0:
            save_checkpoint(model, optimizer, epoch, train_loss,
                            f"{args.output_dir}/epoch_{epoch}.pt")
    
    # 保存最终模型（不含优化器，用于推理）
    final_path = f"{args.output_dir}/mother_model_final.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
    }, final_path)
    print(f"\n✅ Training complete! Final model saved: {final_path}")
    print(f"   Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    
    # 显示模型内存估计
    fp16_size = total_params * 2 / 1024 / 1024
    int4_size = total_params * 0.5 / 1024 / 1024
    print(f"   FP16 size: ~{fp16_size:.0f}MB")
    print(f"   INT4 size: ~{int4_size:.0f}MB")


if __name__ == "__main__":
    main()
