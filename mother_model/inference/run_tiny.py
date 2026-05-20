"""
小模型推理脚本 — 用于 async_model.pt（834K 参数）
跨机器验证：同模型 + 同 seed → 同输出 → 同哈希

用法:
  python run_tiny.py
  python run_tiny.py --prompt "func main" --seed 42
"""

import sys, argparse, hashlib, re
import torch
import torch.nn as nn
import torch.nn.functional as F


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


def ast_normalize(code_text):
    """简易 AST 归一化"""
    code = code_text.strip()
    code = re.sub(r'\s+', ' ', code)
    keywords = {'func','return','if','else','for','range','var','int','bool',
                'string','float','def','import','class','and','or','not',
                'in','is','None','True','False','len','print','range'}
    var_counter = 0; var_map = {}
    def rename_var(m):
        name = m.group(0)
        if name in keywords or name.isdigit():
            return name
        nonlocal var_counter
        if name not in var_map:
            var_map[name] = f"v_{var_counter}"
            var_counter += 1
        return var_map[name]
    code = re.sub(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', rename_var, code)
    return code


def main():
    parser = argparse.ArgumentParser(description="小模型确定性验证")
    parser.add_argument("--model", default="../async_model.pt", help="模型文件路径")
    parser.add_argument("--prompt", default="func add", help="输入 prompt")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--max-new", type=int, default=64, help="最大生成 token 数")
    parser.add_argument("--temp", type=float, default=1.0, help="温度")
    args = parser.parse_args()
    
    # 字符映射（必须与训练时一致）
    # 字符映射 — 逐个定义，避免 shell/escaping 问题
    _cs = []
    # 字母
    for c in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ': _cs.append(c)
    # 数字
    for c in '0123456789': _cs.append(c)
    # 特殊字符 — 用 chr() 避开引号嵌套
    for o in [32, 10, 9, 40, 41, 123, 125, 91, 93, 60, 62, 61, 43, 45, 42, 47, 33, 38, 124, 44, 59, 58, 46, 64, 35, 36, 37, 94, 126, 95, 39, 34, 96]:
        _cs.append(chr(o))
    chars = ''.join(_cs)
    assert len(chars) == 95, f"Expected 95 chars, got {len(chars)}"
    stoi = {c:i+1 for i,c in enumerate(chars)}
    itos = {i+1:c for i,c in enumerate(chars)}
    vocab_size = len(chars) + 1
    
    # 加载模型
    print(f"加载模型: {args.model}")
    model = TinyCodeLM(vocab_size=vocab_size)
    model.load_state_dict(torch.load(args.model, map_location="cpu", weights_only=True))
    model.eval()
    params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {params:,}")
    
    # 编码 prompt
    prompt_ids = [stoi.get(c, 1) for c in args.prompt]
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long)
    print(f"Prompt: '{args.prompt}' ({len(prompt_ids)} tokens)")
    
    # 推理
    print(f"推理 (seed={args.seed}, temp={args.temp})...")
    gen = model.generate(prompt_tensor, max_new=args.max_new,
                         temperature=args.temp, seed=args.seed)
    
    # 解码
    code_text = ''.join([itos[t.item()] for t in gen[0] if t.item() != 0])
    print(f"\n输出:\n{code_text}")
    
    # AST 归一化 + 哈希
    normalized = ast_normalize(code_text)
    hash_val = hashlib.sha256(normalized.encode()).hexdigest()
    print(f"\nAST 归一化:\n{normalized[:120]}...")
    print(f"SHA-256: {hash_val}")
    
    return hash_val, code_text


if __name__ == "__main__":
    main()
