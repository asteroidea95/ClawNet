"""母模型全局配置"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class IntentEncoderConfig:
    """意图理解模块配置"""
    vocab_size: int = 32000
    hidden_dim: int = 512
    num_layers: int = 4
    num_heads: int = 8
    max_seq_len: int = 1024
    dropout: float = 0.1
    output_dim: int = 512  # 意图向量维度


@dataclass
class CodeEncoderConfig:
    """代码理解模块配置"""
    vocab_size: int = 48000  # 更大的词表（含特殊代码 token）
    hidden_dim: int = 768
    num_layers: int = 8
    num_heads: int = 12
    max_seq_len: int = 4096
    local_window_size: int = 128  # 局部注意力窗口
    global_attention_every: int = 4  # 每几层做一次全局注意力
    dropout: float = 0.1
    output_dim: int = 768  # 代码语义向量维度


@dataclass
class FusionDecoderConfig:
    """融合输出模块配置"""
    hidden_dim: int = 512
    num_layers: int = 2
    num_heads: int = 8
    vocab_size: int = 32000
    max_seq_len: int = 1024
    dropout: float = 0.1


@dataclass
class MotherModelConfig:
    """母模型全局配置"""
    intent_encoder: IntentEncoderConfig = field(default_factory=IntentEncoderConfig)
    code_encoder: CodeEncoderConfig = field(default_factory=CodeEncoderConfig)
    fusion_decoder: FusionDecoderConfig = field(default_factory=FusionDecoderConfig)
    
    # 训练配置
    batch_size: int = 4
    learning_rate: float = 2e-4
    warmup_steps: int = 200
    max_epochs: int = 5
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    
    # 量化配置
    quantize_bits: int = 4  # INT4
    
    # 路径
    output_dir: str = "./checkpoints"
    data_dir: str = "./data"
    
    @property
    def total_params_m(self) -> float:
        """估算总参数量（百万）"""
        intent = (self.intent_encoder.vocab_size * self.intent_encoder.hidden_dim +
                  self.intent_encoder.num_layers * (
                      4 * self.intent_encoder.hidden_dim * self.intent_encoder.hidden_dim +
                      8 * self.intent_encoder.hidden_dim * self.intent_encoder.hidden_dim // self.intent_encoder.num_heads
                  )) / 1e6
        code = (self.code_encoder.vocab_size * self.code_encoder.hidden_dim +
                self.code_encoder.num_layers * (
                    4 * self.code_encoder.hidden_dim * self.code_encoder.hidden_dim +
                    8 * self.code_encoder.hidden_dim * self.code_encoder.hidden_dim // self.code_encoder.num_heads
                )) / 1e6
        fusion = (self.fusion_decoder.vocab_size * self.fusion_decoder.hidden_dim +
                  self.fusion_decoder.num_layers * (
                      4 * self.fusion_decoder.hidden_dim * self.fusion_decoder.hidden_dim +
                      8 * self.fusion_decoder.hidden_dim * self.fusion_decoder.hidden_dim // self.fusion_decoder.num_heads
                  )) / 1e6
        return round(intent + code + fusion, 1)
