"""
异机验证脚本 - 在你的服务器上运行
=================================
用法: python3 validate_remote.py
输出: 返回模型推理的归一化哈希值

与我本地沙箱跑的结果对比，如果哈希一致则证明异机同模型同输入同输出。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import hashlib
import re

# ========== 1. 模型定义（必须与本地完全一致） ==========
class TinyCodeLM(nn.Module):
    def __init__(self, vocab_size=96, dim=128, n_heads=4, n_layers=3, max_seq=128):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Embedding(max_seq, dim)
        decoder_layer = nn.TransformerDecoderLayer(dim, n_heads, dim*4, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, n_layers)
        self.lm_head = nn.Linear(dim, vocab_size)
        self.max_seq = max_seq
        
    def forward(self, x):
        seq_len = x.shape[1]
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = self.token_embed(x) + self.pos_embed(pos)
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)
        x = self.decoder(x, x, tgt_mask=mask)
        return self.lm_head(x)
    
    @torch.no_grad()
    def generate(self, prompt, max_new=64, temperature=1.0, seed=42):
        torch.manual_seed(seed)
        generated = prompt.clone()
        for _ in range(max_new):
            if generated.shape[1] > self.max_seq:
                generated = generated[:, -self.max_seq:]
            logits = self.forward(generated)[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            generated = torch.cat([generated, next_token], dim=-1)
        return generated


# ========== 2. AST 归一化（必须与本地完全一致） ==========
def ast_normalize(code_text):
    code = code_text.strip()
    code = re.sub(r'\s+', ' ', code)
    code = code.replace('{ ', '{').replace(' }', '}').replace('( ', '(').replace(' )', ')')
    
    keywords = {'func', 'return', 'if', 'else', 'for', 'range', 'var', 'int', 'bool',
                'string', 'float', 'def', 'import', 'class', 'and', 'or', 'not',
                'in', 'is', 'None', 'True', 'False', 'len', 'print', 'range'}
    
    var_counter = 0
    var_map = {}
    
    def rename_var(m):
        name = m.group(0)
        if name in keywords or name.isdigit():
            return name
        nonlocal var_counter
        if name not in var_map:
            var_map[name] = f"__VAR_{var_counter}__"
            var_counter += 1
        return var_map[name]
    
    code = re.sub(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', rename_var, code)
    return code


# ========== 3. 字符映射（必须与本地完全一致） ==========
chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 \n\t(){}[]<>=+-*/!&|,;:.@#$%^~_'\"`"
stoi = {c:i+1 for i,c in enumerate(chars)}
itos = {i+1:c for i,c in enumerate(chars)}
vocab_size = len(chars) + 1


# ========== 4. 主流程 ==========
def main():
    print("=" * 60)
    print("ClawNet 异机验证")
    print("=" * 60)
    print()
    
    # 加载模型
    print("[1/3] 加载模型权重...")
    model = TinyCodeLM(vocab_size=vocab_size)
    model.load_state_dict(torch.load("async_model.pt", map_location="cpu", weights_only=True))
    model.eval()
    print(f"  模型参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # 固定输入
    print("\n[2/3] 固定 prompt 推理（seed=42）...")
    prompt_str = "func add"
    prompt_tokens = torch.tensor([[stoi.get(c, 1) for c in prompt_str]], dtype=torch.long)
    
    # 推理
    gen = model.generate(prompt_tokens, max_new=64, temperature=1.0, seed=42)
    code_text = ''.join([itos[t.item()] for t in gen[0] if t.item() != 0])
    normalized = ast_normalize(code_text)
    hash_val = hashlib.sha256(normalized.encode()).hexdigest()
    
    print(f"  Prompt: '{prompt_str}'")
    print(f"  原始输出: {repr(code_text[:80])}")
    print(f"  归一化: {normalized[:80]}")
    print(f"  SHA-256: {hash_val}")
    
    # 保存结果
    print("\n[3/3] 保存结果...")
    with open("async_result.txt", "w") as f:
        f.write(f"HASH:{hash_val}\n")
        f.write(f"RAW:{code_text}\n")
        f.write(f"NORM:{normalized}\n")
    print("  已保存到 async_result.txt")
    print()
    print("=" * 60)
    print(f"你的服务器哈希: {hash_val}")
    print("把这个哈希发给沙箱那边比对即可验证异机一致性")
    print("=" * 60)


if __name__ == "__main__":
    main()
