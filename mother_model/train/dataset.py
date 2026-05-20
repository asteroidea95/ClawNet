"""训练数据集"""
import json
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from tokenizer.code_tokenizer import CodeTokenizer
from config import MotherModelConfig


class MotherModelDataset(Dataset):
    """
    母模型训练数据集
    
    每条数据:
      intent: 用户的自然语言指令
      code: 代码内容
      code_language: 代码语言
      target: 目标输出（需求描述）
      culprit_submodel: 出错的子模型
      error_type: 错误类型
      error_line: 错误行号
    """
    
    def __init__(
        self,
        data_path: str,
        config: MotherModelConfig,
        tokenizer: Optional[CodeTokenizer] = None,
        max_code_len: int = 2048,
        max_intent_len: int = 256,
        max_target_len: int = 512,
    ):
        self.config = config
        self.tokenizer = tokenizer or CodeTokenizer()
        self.max_code_len = max_code_len
        self.max_intent_len = max_intent_len
        self.max_target_len = max_target_len
        
        # 加载数据
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.data.append(json.loads(line))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        # 编码意图（用标准 tokenizer 或简单编码）
        intent_ids = self._encode_text(item["intent"], self.max_intent_len, is_intent=True)
        
        # 编码代码
        code_ids, depth_ids, sibling_ids = self._encode_code(
            item["code"], self.max_code_len
        )
        
        # 编码目标输出
        target_ids = self._encode_text(item["target"], self.max_target_len, is_intent=False)
        
        # 构建注意力掩码
        intent_mask = self._build_mask(intent_ids)
        code_mask = self._build_mask(code_ids)
        target_mask = self._build_mask(target_ids)
        
        return {
            "intent_input_ids": torch.tensor(intent_ids, dtype=torch.long),
            "intent_mask": torch.tensor(intent_mask, dtype=torch.bool),
            "code_input_ids": torch.tensor(code_ids, dtype=torch.long),
            "code_depth_ids": torch.tensor(depth_ids, dtype=torch.long),
            "code_sibling_ids": torch.tensor(sibling_ids, dtype=torch.long),
            "code_mask": torch.tensor(code_mask, dtype=torch.bool),
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
            "target_mask": torch.tensor(target_mask, dtype=torch.bool),
        }
    
    def _encode_text(self, text: str, max_len: int, is_intent: bool = True) -> List[int]:
        """
        编码自然语言文本
        
        简单实现：用 tokenizer 的分词能力 + 基本 ID 映射
        实际生产中可以换成 BPE tokenizer
        """
        tokens = self.tokenizer.tokenize(text)
        ids = [self.tokenizer.vocab.get('<bos>', 1)]
        
        for token_text, token_type, _, _ in tokens:
            if len(ids) >= max_len - 1:
                break
            tid = self.tokenizer._token_to_id(token_text, token_type)
            ids.append(tid)
        
        ids.append(self.tokenizer.vocab.get('<eos>', 2))
        
        # Pad to max_len
        pad_id = self.tokenizer.vocab.get('<pad>', 0)
        ids = ids[:max_len]
        ids += [pad_id] * (max_len - len(ids))
        
        return ids
    
    def _encode_code(self, code: str, max_len: int) -> Tuple[List[int], List[int], List[int]]:
        """编码代码"""
        input_ids, depth_ids, sibling_ids = self.tokenizer.encode(code)
        
        # 截断
        if len(input_ids) > max_len:
            input_ids = input_ids[:max_len]
            depth_ids = depth_ids[:max_len]
            sibling_ids = sibling_ids[:max_len]
        
        # Pad
        pad_id = self.tokenizer.vocab.get('<pad>', 0)
        pad_len = max_len - len(input_ids)
        input_ids += [pad_id] * pad_len
        depth_ids += [0] * pad_len
        sibling_ids += [0] * pad_len
        
        return input_ids, depth_ids, sibling_ids
    
    def _build_mask(self, ids: List[int]) -> List[bool]:
        """构建注意力掩码（1 = 有效，0 = pad）"""
        pad_id = self.tokenizer.vocab.get('<pad>', 0)
        return [id != pad_id for id in ids]


def create_dataloaders(
    config: MotherModelConfig,
    train_path: str,
    val_path: Optional[str] = None,
    tokenizer: Optional[CodeTokenizer] = None,
    batch_size: Optional[int] = None,
    num_workers: int = 0,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """创建训练/验证 DataLoader"""
    
    train_dataset = MotherModelDataset(train_path, config, tokenizer)
    
    val_loader = None
    if val_path and Path(val_path).exists():
        val_dataset = MotherModelDataset(val_path, config, tokenizer)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size or config.batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size or config.batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    
    return train_loader, val_loader
