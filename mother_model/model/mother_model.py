"""
母模型 - 核心模型架构

架构:
  ┌─ 意图编码器 (Intent Encoder)
  │        ↓ 意图向量
  ├─ 代码编码器 (Code Encoder, AST-Aware)
  │        ↓ 代码语义向量
  ├─ 交叉注意力融合 + 轻量 Decoder
  │        ↓ 需求描述
  └─ LM Head → token 序列

总参数: ~140M (INT4 ~84MB)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass

from config import MotherModelConfig


# ============================================================
# 1. AST-Aware 位置编码
# ============================================================

class ASTPositionEncoding(nn.Module):
    """
    代码的 AST 深度感知位置编码。
    
    传统 RoPE/绝对位置编码把代码当成行号序列处理，
    但代码的"亲近关系"是按 AST 树算的：
    
    func main() {           # (depth=0, sibling=0)
        if x > 0 {          # (depth=1, sibling=0)
            fmt.Println(x)  # (depth=2, sibling=0)
        }                   # (depth=1, sibling=1)
    }                       # (depth=0, sibling=1)
    
    同一层级兄弟节点在编码空间里"距离近"，
    嵌套层次不同的代码块天然分离。
    """
    
    def __init__(self, hidden_dim: int, max_depth: int = 32, max_sibling: int = 256):
        super().__init__()
        self.depth_embed = nn.Embedding(max_depth, hidden_dim // 2)
        self.sibling_embed = nn.Embedding(max_sibling, hidden_dim // 2)
        
    def forward(self, depth_ids: torch.Tensor, sibling_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            depth_ids:   (batch, seq_len) AST 深度 ID
            sibling_ids: (batch, seq_len) 同级偏移 ID
        Returns:
            (batch, seq_len, hidden_dim) 位置编码
        """
        depth_pos = self.depth_embed(depth_ids)
        sibling_pos = self.sibling_embed(sibling_ids)
        return torch.cat([depth_pos, sibling_pos], dim=-1)


class RotaryPositionEncoding(nn.Module):
    """
    RoPE 位置编码（用于意图编码器和融合解码器）
    """
    def __init__(self, dim: int, max_seq_len: int = 4096):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len
        
    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        t = torch.arange(seq_len, device=device).float()
        freqs = torch.outer(t, self.inv_freq)
        return torch.cat([freqs.sin(), freqs.cos()], dim=-1)  # (seq_len, dim)


def apply_rotary_emb(x: torch.Tensor, pos_cos_sin: torch.Tensor) -> torch.Tensor:
    """应用 RoPE 到 x"""
    cos = pos_cos_sin[:, :, x.shape[-1] // 2:]
    sin = pos_cos_sin[:, :, :x.shape[-1] // 2]
    x_rot = torch.cat([-x[..., x.shape[-1] // 2:], x[..., :x.shape[-1] // 2]], dim=-1)
    return x * cos + x_rot * sin


# ============================================================
# 2. 注意力机制
# ============================================================

class GroupedQueryAttention(nn.Module):
    """
    GQA (Grouped Query Attention) - 比 MHA 省显存，比 MQA 质量好
    
    num_kv_heads < num_q_heads 时，多个 query head 共享一组 key/value
    """
    def __init__(self, hidden_dim: int, num_q_heads: int, num_kv_heads: Optional[int] = None,
                 dropout: float = 0.1, window_size: Optional[int] = None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads or num_q_heads
        self.head_dim = hidden_dim // num_q_heads
        self.window_size = window_size
        
        assert hidden_dim % num_q_heads == 0
        assert num_q_heads % self.num_kv_heads == 0
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        pos_cos_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        
        q = self.q_proj(x).view(batch, seq_len, self.num_q_heads, self.head_dim)
        k = self.k_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim)
        
        if pos_cos_sin is not None:
            q = apply_rotary_emb(q, pos_cos_sin)
            k = apply_rotary_emb(k, pos_cos_sin)
        
        # GQA: 扩展 kv heads 匹配 q heads
        n_groups = self.num_q_heads // self.num_kv_heads
        k = k[:, :, :, None, :].expand(-1, -1, -1, n_groups, -1).reshape(
            batch, seq_len, self.num_q_heads, self.head_dim
        )
        v = v[:, :, :, None, :].expand(-1, -1, -1, n_groups, -1).reshape(
            batch, seq_len, self.num_q_heads, self.head_dim
        )
        
        # 滑动窗口 mask
        if self.window_size is not None:
            window_mask = torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool).triu()
            window_mask = torch.logical_not(
                window_mask & (torch.arange(seq_len, device=x.device)[:, None] >
                               torch.arange(seq_len, device=x.device)[None, :] + self.window_size)
            )
            window_mask = window_mask[None, None, :, :]
            if attn_mask is not None:
                attn_mask = attn_mask & window_mask
            else:
                attn_mask = window_mask
        
        # Scaled Dot-Product Attention
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q.transpose(1, 2), k.transpose(1, 2).transpose(-2, -1)) * scale
        
        if attn_mask is not None:
            attn = attn.masked_fill(~attn_mask, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v.transpose(1, 2)).transpose(1, 2).contiguous()
        out = out.reshape(batch, seq_len, -1)
        out = self.o_proj(out)
        return out


class CrossAttention(nn.Module):
    """交叉注意力（用于融合模块：意图向量 Q, 代码向量 K/V）"""
    def __init__(self, q_dim: int, kv_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = q_dim // num_heads
        assert q_dim % num_heads == 0
        
        self.q_proj = nn.Linear(q_dim, q_dim, bias=False)
        self.k_proj = nn.Linear(kv_dim, q_dim, bias=False)
        self.v_proj = nn.Linear(kv_dim, q_dim, bias=False)
        self.o_proj = nn.Linear(q_dim, q_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, q_len, _ = query.shape
        kv_len = key_value.shape[1]
        
        q = self.q_proj(query).view(batch, q_len, self.num_heads, self.head_dim)
        k = self.k_proj(key_value).view(batch, kv_len, self.num_heads, self.head_dim)
        v = self.v_proj(key_value).view(batch, kv_len, self.num_heads, self.head_dim)
        
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q.transpose(1, 2), k.transpose(1, 2).transpose(-2, -1)) * scale
        
        if attn_mask is not None:
            attn = attn.masked_fill(~attn_mask, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v.transpose(1, 2)).transpose(1, 2).contiguous()
        out = out.reshape(batch, q_len, -1)
        out = self.o_proj(out)
        return out


# ============================================================
# 3. Transformer Block
# ============================================================

class FeedForward(nn.Module):
    """SwiGLU FFN"""
    def __init__(self, hidden_dim: int, ffn_dim: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        ffn_dim = ffn_dim or hidden_dim * 4
        self.gate_proj = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.dropout(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class TransformerBlock(nn.Module):
    """标准的 Pre-Norm Transformer Block"""
    def __init__(self, hidden_dim: int, num_heads: int, num_kv_heads: Optional[int] = None,
                 ffn_dim: Optional[int] = None, dropout: float = 0.1,
                 window_size: Optional[int] = None):
        super().__init__()
        self.attention = GroupedQueryAttention(
            hidden_dim, num_heads, num_kv_heads, dropout, window_size
        )
        self.feed_forward = FeedForward(hidden_dim, ffn_dim, dropout)
        self.norm1 = nn.RMSNorm(hidden_dim)
        self.norm2 = nn.RMSNorm(hidden_dim)
        
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None,
                pos_cos_sin: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attention(self.norm1(x), attn_mask, pos_cos_sin)
        x = x + self.feed_forward(self.norm2(x))
        return x


# ============================================================
# 4. 意图编码器（Intent Encoder）
# ============================================================

class IntentEncoder(nn.Module):
    """
    理解用户的自然语言指令 → 向量化意图表示
    
    输入: "看看这段代码有什么问题"
    输出: [512-dim 意图向量]
    
    架构: 4 层 Transformer Encoder + Pooling
    参数: ~30M
    """
    def __init__(self, config: MotherModelConfig, vocab_size: int):
        super().__init__()
        cfg = config.intent_encoder
        
        self.token_embed = nn.Embedding(vocab_size, cfg.hidden_dim)
        self.rope = RotaryPositionEncoding(cfg.hidden_dim, cfg.max_seq_len)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.hidden_dim, cfg.num_heads, dropout=cfg.dropout)
            for _ in range(cfg.num_layers)
        ])
        
        self.norm = nn.RMSNorm(cfg.hidden_dim)
        self.pooling = nn.Linear(cfg.hidden_dim, cfg.output_dim)
        
    def forward(self, input_ids: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq_len = input_ids.shape
        
        x = self.token_embed(input_ids)
        
        pos = self.rope(seq_len, input_ids.device)
        pos = pos[None, :, :]  # (1, seq_len, dim)
        
        for block in self.blocks:
            x = block(x, attn_mask, pos)
        
        x = self.norm(x)
        
        # Mean pooling（忽略 padding）
        if attn_mask is not None:
            x = (x * attn_mask.unsqueeze(-1)).sum(dim=1) / attn_mask.sum(dim=1, keepdim=True).clamp(min=1)
        else:
            x = x.mean(dim=1)
        
        return self.pooling(x)  # (batch, output_dim)


# ============================================================
# 5. 代码编码器（Code Encoder, AST-Aware）
# ============================================================

class CodeEncoder(nn.Module):
    """
    理解代码结构 → 向量化语义表示
    
    输入: handler.go 的代码内容（含 AST 深度/兄弟位置标记）
    输出: [768-dim 代码语义向量]
    
    关键设计:
      - AST-Aware Position Encoding（用 AST 深度替代行号）
      - 滑动窗口注意力（128 tokens 局部 + 每 4 层做全局）
      - 代码结构感知的 tokenizer 输入
    
    架构: 8 层 Transformer Encoder
    参数: ~80M
    """
    def __init__(self, config: MotherModelConfig, vocab_size: int):
        super().__init__()
        cfg = config.code_encoder
        
        self.token_embed = nn.Embedding(vocab_size, cfg.hidden_dim)
        self.ast_pos_encoding = ASTPositionEncoding(cfg.hidden_dim)
        
        self.blocks = nn.ModuleList()
        for i in range(cfg.num_layers):
            window_size = None if i % cfg.global_attention_every == 0 else cfg.local_window_size
            self.blocks.append(
                TransformerBlock(
                    cfg.hidden_dim, cfg.num_heads,
                    num_kv_heads=cfg.num_heads // 2,
                    dropout=cfg.dropout,
                    window_size=window_size,
                )
            )
        
        self.norm = nn.RMSNorm(cfg.hidden_dim)
        self.pooling = nn.Linear(cfg.hidden_dim, cfg.output_dim)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        depth_ids: torch.Tensor,    # AST 深度 ID
        sibling_ids: torch.Tensor,   # 同级偏移 ID
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, seq_len = input_ids.shape
        
        x = self.token_embed(input_ids)
        pos = self.ast_pos_encoding(depth_ids, sibling_ids)
        x = x + pos
        
        for block in self.blocks:
            x = block(x, attn_mask)
        
        x = self.norm(x)
        
        if attn_mask is not None:
            x = (x * attn_mask.unsqueeze(-1)).sum(dim=1) / attn_mask.sum(dim=1, keepdim=True).clamp(min=1)
        else:
            x = x.mean(dim=1)
        
        return self.pooling(x)  # (batch, output_dim)


# ============================================================
# 6. 融合解码器（Fusion Decoder）
# ============================================================

class FusionDecoder(nn.Module):
    """
    融合意图向量 + 代码语义向量 → 输出需求描述
    
    输入: 意图向量 (512-dim) + 代码向量 (768-dim)
    输出: "Go 子模型，第 15 行的 DBConn 未定义..."
    
    架构: Cross Attention + 2 层 Decoder
    参数: ~30M
    """
    def __init__(self, config: MotherModelConfig, vocab_size: int):
        super().__init__()
        cfg = config.fusion_decoder
        intent_dim = config.intent_encoder.output_dim
        code_dim = config.code_encoder.output_dim
        
        # 向量映射到同一空间
        self.intent_proj = nn.Linear(intent_dim, cfg.hidden_dim)
        self.code_proj = nn.Linear(code_dim, cfg.hidden_dim)
        
        # 交叉注意力：意图 Q × 代码 KV
        self.cross_attn = CrossAttention(cfg.hidden_dim, cfg.hidden_dim, cfg.num_heads, cfg.dropout)
        
        # 轻量 Decoder 层
        self.token_embed = nn.Embedding(vocab_size, cfg.hidden_dim)
        self.rope = RotaryPositionEncoding(cfg.hidden_dim, cfg.max_seq_len)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.hidden_dim, cfg.num_heads, dropout=cfg.dropout)
            for _ in range(cfg.num_layers)
        ])
        
        self.norm = nn.RMSNorm(cfg.hidden_dim)
        self.lm_head = nn.Linear(cfg.hidden_dim, vocab_size, bias=False)
        
    def forward(
        self,
        intent_vec: torch.Tensor,      # (batch, intent_dim)
        code_vec: torch.Tensor,        # (batch, code_dim)
        input_ids: torch.Tensor,       # (batch, tgt_seq_len) - 当前生成的部分
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, tgt_len = input_ids.shape
        
        # 映射融合向量
        intent_h = self.intent_proj(intent_vec).unsqueeze(1)  # (batch, 1, hidden_dim)
        code_h = self.code_proj(code_vec).unsqueeze(1)        # (batch, 1, hidden_dim)
        
        # Cross Attention: 用 intent 作为 query，code 作为 key/value
        # 意图说了"我想要什么"，代码说了"现状是什么"
        fused = self.cross_attn(intent_h, code_h)  # (batch, 1, hidden_dim)
        
        # 作为 Decoder 的起始 token 嵌入
        x = self.token_embed(input_ids)
        pos = self.rope(tgt_len, input_ids.device)
        pos = pos[None, :, :]
        
        # 将 fused 拼到序列开头作为引导
        # (batch, 1 + tgt_len, hidden_dim)
        x = torch.cat([fused, x], dim=1)
        
        # Causal mask
        full_len = 1 + tgt_len
        causal_mask = torch.triu(torch.ones(full_len, full_len, device=x.device, dtype=torch.bool), diagonal=1)
        if attn_mask is not None:
            # 扩展 attn_mask
            ext_mask = torch.ones(batch, 1, dtype=torch.bool, device=x.device)
            full_mask = torch.cat([ext_mask, attn_mask], dim=1)
            causal_mask = causal_mask & full_mask.unsqueeze(-1)
        
        for block in self.blocks:
            x = block(x, ~causal_mask, pos)
        
        x = self.norm(x)
        
        # 取 tgt 部分，排除 fused 起始 token
        logits = self.lm_head(x[:, 1:, :])  # (batch, tgt_len, vocab_size)
        return logits
    
    def generate(
        self,
        intent_vec: torch.Tensor,
        code_vec: torch.Tensor,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        eos_token_id: int = 2,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """自回归生成"""
        batch = intent_vec.shape[0]
        generated = torch.full((batch, 1), 1, dtype=torch.long, device=device)  # <bos>
        
        for _ in range(max_new_tokens):
            logits = self.forward(intent_vec, code_vec, generated)
            next_logits = logits[:, -1, :] / temperature
            
            # 采样
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated = torch.cat([generated, next_token], dim=-1)
            
            if (next_token == eos_token_id).all():
                break
        
        return generated[:, 1:]  # 去掉 <bos>


# ============================================================
# 7. 母模型（完整版）
# ============================================================

class MotherModel(nn.Module):
    """
    母模型完整架构（140M 参数）
    
    使用方式:
        model = MotherModel(config)
        
        # 训练
        loss = model(
            intent_input_ids=...,
            intent_mask=...,
            code_input_ids=...,
            code_depth_ids=...,
            code_sibling_ids=...,
            code_mask=...,
            target_ids=...,
            target_mask=...,
        )
        
        # 推理
        output = model.generate(
            intent_input_ids="看看这段代码问题在哪",
            code_input_ids="func main() { ... }",
            code_depth_ids=...,
            code_sibling_ids=...,
        )
    """
    
    def __init__(self, config: MotherModelConfig):
        super().__init__()
        self.config = config
        
        # 三个模块
        self.intent_encoder = IntentEncoder(config, config.intent_encoder.vocab_size)
        self.code_encoder = CodeEncoder(config, config.code_encoder.vocab_size)
        self.fusion_decoder = FusionDecoder(config, config.fusion_decoder.vocab_size)
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in [self.intent_encoder, self.code_encoder, self.fusion_decoder]:
            for name, param in module.named_parameters():
                if 'weight' in name and param.dim() >= 2:
                    nn.init.normal_(param, mean=0.0, std=self.config.fusion_decoder.hidden_dim ** -0.5)
                elif 'bias' in name:
                    nn.init.zeros_(param)
    
    def forward(
        self,
        # 意图输入
        intent_input_ids: torch.Tensor,
        intent_mask: Optional[torch.Tensor] = None,
        # 代码输入
        code_input_ids: Optional[torch.Tensor] = None,
        code_depth_ids: Optional[torch.Tensor] = None,
        code_sibling_ids: Optional[torch.Tensor] = None,
        code_mask: Optional[torch.Tensor] = None,
        # 目标输出
        target_ids: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Returns:
            训练: {"loss": loss}
            推理: {"intent_vec": ..., "code_vec": ..., "logits": ...}
        """
        # 1. 编码意图
        intent_vec = self.intent_encoder(intent_input_ids, intent_mask)
        
        # 2. 编码代码
        code_vec = self.code_encoder(code_input_ids, code_depth_ids, code_sibling_ids, code_mask)
        
        # 3. 生成需求描述
        logits = self.fusion_decoder(intent_vec, code_vec, target_ids, target_mask)
        
        if target_ids is not None:
            # 计算 loss
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = target_ids[:, 1:].contiguous()
            
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=0,  # 忽略 padding token
                reduction='mean',
            )
            return {"loss": loss}
        
        return {"intent_vec": intent_vec, "code_vec": code_vec, "logits": logits}
    
    @torch.no_grad()
    def generate(
        self,
        intent_input_ids: torch.Tensor,
        intent_mask: Optional[torch.Tensor] = None,
        code_input_ids: Optional[torch.Tensor] = None,
        code_depth_ids: Optional[torch.Tensor] = None,
        code_sibling_ids: Optional[torch.Tensor] = None,
        code_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        eos_token_id: int = 2,
    ) -> torch.Tensor:
        """推理生成"""
        device = intent_input_ids.device
        
        intent_vec = self.intent_encoder(intent_input_ids, intent_mask)
        
        if code_input_ids is not None:
            code_vec = self.code_encoder(code_input_ids, code_depth_ids, code_sibling_ids, code_mask)
        else:
            code_vec = torch.zeros(1, self.config.code_encoder.output_dim, device=device)
        
        return self.fusion_decoder.generate(
            intent_vec, code_vec, max_new_tokens, temperature, eos_token_id, device
        )
    
    def get_total_params(self) -> int:
        """返回总参数量"""
        return sum(p.numel() for p in self.parameters())
    
    def get_params_by_module(self) -> Dict[str, int]:
        """按模块展示参数分布"""
        return {
            "intent_encoder": sum(p.numel() for p in self.intent_encoder.parameters()),
            "code_encoder": sum(p.numel() for p in self.code_encoder.parameters()),
            "fusion_decoder": sum(p.numel() for p in self.fusion_decoder.parameters()),
        }
