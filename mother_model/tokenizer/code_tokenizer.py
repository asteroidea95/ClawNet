"""
Code Tokenizer - 代码结构感知的分词器

标准 tokenizer 把代码当自然语言处理，不认识代码结构。
这个 tokenizer 在分词的同时提取 AST 结构信息：
  - 关键词（if, for, func）→ 独立 token
  - 标识符（变量名/函数名）→ 单独编码
  - 操作符（>, +, !=）→ 单独编码
  - 分隔符（{ } ( ) ;）→ 带嵌套深度标记
  - 字符串/注释 → 按块打包

同时输出三个 tensor：
  - input_ids: 分词后的 token IDs
  - depth_ids: 每个 token 对应的 AST 嵌套深度
  - sibling_ids: 每个 token 在同级中的偏移位置
"""

import re
import json
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict
from pathlib import Path


# 代码关键词（多语言）
KEYWORDS: Set[str] = {
    # Go
    'func', 'package', 'import', 'defer', 'go', 'chan', 'select',
    'type', 'struct', 'interface', 'map', 'range', 'return',
    'if', 'else', 'for', 'switch', 'case', 'default', 'break', 'continue',
    'var', 'const', 'nil', 'true', 'false', 'error', 'string', 'int',
    'bool', 'float64', 'byte', 'rune', 'uint64', 'int64', 'float32',
    'make', 'new', 'append', 'len', 'cap', 'delete', 'close', 'panic',
    'recover', 'fallthrough', 'goto',
    # Python
    'def', 'class', 'lambda', 'yield', 'with', 'as', 'async', 'await',
    'pass', 'raise', 'try', 'except', 'finally', 'import', 'from',
    'assert', 'global', 'nonlocal', 'del', 'elif', 'in', 'is', 'not',
    'and', 'or', 'None', 'True', 'False', 'self', 'cls',
    # JS/TS
    'function', 'const', 'let', 'var', 'async', 'await', 'export',
    'import', 'default', 'extends', 'class', 'new', 'this', 'super',
    'typeof', 'instanceof', 'void', 'delete', 'throw', 'catch',
    'finally', 'try', 'switch', 'case', 'break', 'continue', 'return',
    'if', 'else', 'for', 'while', 'do', 'of', 'in', 'as', 'from',
    'undefined', 'null', 'ArrowFunction',
    # SQL
    'SELECT', 'FROM', 'WHERE', 'INSERT', 'INTO', 'VALUES', 'UPDATE',
    'SET', 'DELETE', 'CREATE', 'TABLE', 'ALTER', 'DROP', 'INDEX',
    'JOIN', 'LEFT', 'RIGHT', 'INNER', 'OUTER', 'ON', 'AND', 'OR',
    'NOT', 'NULL', 'IS', 'LIKE', 'IN', 'BETWEEN', 'ORDER', 'BY',
    'GROUP', 'HAVING', 'LIMIT', 'OFFSET', 'AS', 'DISTINCT', 'COUNT',
    'SUM', 'AVG', 'MAX', 'MIN', 'EXISTS', 'UNION', 'ALL', 'PRIMARY',
    'KEY', 'FOREIGN', 'REFERENCES', 'CASCADE', 'INTEGER', 'VARCHAR',
    'BOOLEAN', 'TIMESTAMP', 'TEXT', 'FLOAT', 'DOUBLE', 'PRECISION',
    'BIGINT', 'SMALLINT', 'CHAR', 'DATE', 'DATETIME', 'AUTO_INCREMENT',
    'UNIQUE', 'CHECK', 'DEFAULT', 'INDEX', 'VIEW', 'TRIGGER',
    # Rust
    'fn', 'let', 'mut', 'match', 'impl', 'trait', 'pub', 'use',
    'mod', 'crate', 'self', 'Self', 'struct', 'enum', 'union',
    'unsafe', 'where', 'ref', 'move', 'dyn', 'static', 'const',
    'as', 'in', 'for', 'while', 'loop', 'if', 'else', 'return',
    'break', 'continue', 'true', 'false', 'Some', 'None', 'Ok', 'Err',
    'Option', 'Result', 'String', 'Vec', 'Box', 'Rc', 'Arc', 'Cell',
    'RefCell', 'HashMap', 'HashSet',
}

# 分隔符
DELIMITERS: Set[str] = {'{', '}', '(', ')', '[', ']', ';', ':', ',', '.'}

# 操作符
OPERATORS: Set[str] = {
    '=', '==', '!=', '<', '>', '<=', '>=', '+', '-', '*', '/', '%',
    '+=', '-=', '*=', '/=', '++', '--', '&&', '||', '!', '&', '|',
    '^', '~', '<<', '>>', '&^', ':=', '...', '->', '=>', '::',
}

# 字符串/注释标记
STRING_MARKS = {'"', "'", '`', '"""', "'''", '//', '/*', '#'}


class CodeTokenizer:
    """
    代码结构感知分词器
    
    用法:
        tokenizer = CodeTokenizer()
        tokens = tokenizer.tokenize("func main() {\n    return x\n}")
        # [('func', 'keyword', 0, 0), ('main', 'identifier', 0, 1), ...]
        
        input_ids, depth_ids, sibling_ids = tokenizer.encode("...")
    """
    
    def __init__(self, vocab_path: Optional[str] = None):
        self.special_tokens = {
            '<pad>': 0,
            '<bos>': 1,
            '<eos>': 2,
            '<unk>': 3,
            '<sep>': 4,
        }
        self.vocab: Dict[str, int] = {}
        self.inverse_vocab: Dict[int, str] = {}
        self.ast_tracker = ASTTracker()
        
        if vocab_path and Path(vocab_path).exists():
            self.load_vocab(vocab_path)
        else:
            self._build_vocab()
    
    def _build_vocab(self):
        """构建词表"""
        # Special tokens
        idx = 0
        for token, token_id in self.special_tokens.items():
            self.vocab[token] = token_id
            idx = max(idx, token_id + 1)
        
        # 关键词
        for kw in sorted(KEYWORDS):
            self.vocab[f'__{kw}__'] = idx
            idx += 1
        
        # 分隔符
        for delim in sorted(DELIMITERS):
            self.vocab[f'__{delim}__'] = idx
            idx += 1
        
        # 操作符
        for op in sorted(OPERATORS):
            self.vocab[f'__{op}__'] = idx
            idx += 1
        
        # 常见标记
        for marker in ['__NUMBER__', '__STRING__', '__COMMENT__', '__IDENTIFIER__', 
                       '__INDENT__', '__DEDENT__', '__NEWLINE__', '__EOF__']:
            self.vocab[marker] = idx
            idx += 1
        
        # BPE 子词 (前 20000 个常用子词占位)
        # 训练时从预训练语料中学习
        self.vocab['__UNUSED_BPE_START__'] = idx
        self.max_pretrained_id = idx
        
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
    
    @property
    def vocab_size(self) -> int:
        return len(self.vocab)
    
    def tokenize(self, code: str, file_extension: str = '') -> List[Tuple[str, str, int, int]]:
        """
        分词并提取 AST 结构信息
        
        Returns:
            List of (token_text, token_type, depth, sibling_offset)
        """
        tokens = []
        lines = code.split('\n')
        
        for line_idx, line in enumerate(lines):
            # 计算缩进深度
            stripped = line.lstrip()
            indent = len(line) - len(stripped)
            depth = max(0, indent // 4)  # 假设 4 空格缩进
            
            if line_idx > 0:
                tokens.append(('__NEWLINE__', 'structure', 0, line_idx - 1))
            
            pos = 0
            sibling = 0
            length = len(stripped)
            
            while pos < length:
                c = stripped[pos]
                
                # 跳过空格
                if c == ' ':
                    pos += 1
                    continue
                
                # 注释
                if stripped[pos:pos+2] == '//' or stripped[pos:pos+2] == '/*' or c == '#':
                    comment_end = length
                    if stripped[pos:pos+2] == '/*':
                        end_idx = stripped.find('*/', pos + 2)
                        if end_idx != -1:
                            comment_end = end_idx + 2
                    comment_text = stripped[pos:comment_end]
                    tokens.append((comment_text, 'comment', depth, sibling))
                    sibling += 1
                    pos = comment_end
                    continue
                
                # 字符串
                if c in ('"', "'", '`'):
                    quote = c
                    end = pos + 1
                    while end < length:
                        if stripped[end] == '\\':
                            end += 2
                            continue
                        if stripped[end] == quote:
                            end += 1
                            break
                        end += 1
                    string_text = stripped[pos:end]
                    tokens.append((string_text, 'string', depth, sibling))
                    sibling += 1
                    pos = end
                    continue
                
                # 分隔符
                if c in DELIMITERS:
                    tokens.append((c, 'delimiter', depth, sibling))
                    sibling += 1
                    pos += 1
                    # 花括号增减深度
                    continue
                
                # 操作符 (先匹配多字符)
                op_found = None
                for op_len in [3, 2, 1]:
                    if stripped[pos:pos+op_len] in OPERATORS:
                        op_found = stripped[pos:pos+op_len]
                        break
                if op_found:
                    tokens.append((op_found, 'operator', depth, sibling))
                    sibling += 1
                    pos += len(op_found)
                    continue
                
                # 关键词或标识符
                word_match = re.match(r'[a-zA-Z_][a-zA-Z0-9_]*', stripped[pos:])
                if word_match:
                    word = word_match.group()
                    if word in KEYWORDS:
                        tokens.append((word, 'keyword', depth, sibling))
                    else:
                        tokens.append((word, 'identifier', depth, sibling))
                    sibling += 1
                    pos += len(word)
                    continue
                
                # 数字
                num_match = re.match(r'\d+\.?\d*', stripped[pos:])
                if num_match:
                    tokens.append((num_match.group(), 'number', depth, sibling))
                    sibling += 1
                    pos += len(num_match.group())
                    continue
                
                # 其他字符
                tokens.append((c, 'other', depth, sibling))
                sibling += 1
                pos += 1
        
        tokens.append(('__EOF__', 'structure', 0, 0))
        return tokens
    
    def encode(self, code: str) -> Tuple[List[int], List[int], List[int]]:
        """
        编码代码为模型输入
        
        Returns:
            input_ids:   token IDs (list of int)
            depth_ids:   AST 深度 ID (list of int)
            sibling_ids: 同级偏移 (list of int)
        """
        raw_tokens = self.tokenize(code)
        input_ids = []
        depth_ids = []
        sibling_ids = []
        
        for token_text, token_type, depth, sibling in raw_tokens:
            token_id = self._token_to_id(token_text, token_type)
            input_ids.append(token_id)
            depth_ids.append(min(depth, 31))  # clip to max_depth=32
            sibling_ids.append(sibling % 256)
        
        return input_ids, depth_ids, sibling_ids
    
    def _token_to_id(self, token_text: str, token_type: str) -> int:
        """将 token 映射为 ID"""
        # Special tokens
        if token_text in self.special_tokens:
            return self.special_tokens[token_text]
        
        # 关键词
        if token_type == 'keyword' and f'__{token_text}__' in self.vocab:
            return self.vocab[f'__{token_text}__']
        
        # 分隔符
        if token_type == 'delimiter' and f'__{token_text}__' in self.vocab:
            return self.vocab[f'__{token_text}__']
        
        # 操作符
        if token_type == 'operator' and f'__{token_text}__' in self.vocab:
            return self.vocab[f'__{token_text}__']
        
        # 其他类型映射到 type markers
        type_marker = {
            'number': '__NUMBER__',
            'string': '__STRING__',
            'comment': '__COMMENT__',
            'structure': '__IDENTIFIER__',
        }.get(token_type, '__IDENTIFIER__')
        
        if type_marker in self.vocab:
            return self.vocab[type_marker]
        
        return self.vocab['<unk>']
    
    def decode(self, token_ids: List[int]) -> str:
        """将 token IDs 解码为文本（用于 decoder 输出）"""
        if not self.inverse_vocab:
            self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        
        texts = []
        for tid in token_ids:
            text = self.inverse_vocab.get(tid, '<unk>')
            # 去掉关键词包裹标记
            if text.startswith('__') and text.endswith('__') and text != '<pad>':
                inner = text[2:-2]
                if inner in KEYWORDS or inner in DELIMITERS or inner in OPERATORS:
                    texts.append(inner)
                elif inner in ('NUMBER', 'STRING', 'COMMENT', 'IDENTIFIER', 
                              'INDENT', 'DEDENT', 'NEWLINE', 'EOF'):
                    continue  # skip structural tokens
                else:
                    texts.append(text)
            else:
                texts.append(text)
        
        return ''.join(texts).replace('<pad>', '').replace('<bos>', '').replace('<eos>', '')
    
    def save_vocab(self, path: str):
        """保存词表"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'vocab': self.vocab,
                'inverse_vocab': {str(k): v for k, v in self.inverse_vocab.items()},
            }, f, ensure_ascii=False, indent=2)
    
    def load_vocab(self, path: str):
        """加载词表"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.vocab = data['vocab']
            self.inverse_vocab = {int(k): v for k, v in data['inverse_vocab'].items()}


class ASTTracker:
    """简单的 AST 嵌套深度追踪器（处理花括号/缩进）"""
    
    def __init__(self):
        self.brace_depth = 0
        self.paren_depth = 0
        self.bracket_depth = 0
        
    def push(self, char: str):
        if char == '{':
            self.brace_depth += 1
            return 'enter_brace'
        elif char == '}':
            self.brace_depth = max(0, self.brace_depth - 1)
            return 'exit_brace'
        elif char == '(':
            self.paren_depth += 1
            return 'enter_paren'
        elif char == ')':
            self.paren_depth = max(0, self.paren_depth - 1)
            return 'exit_paren'
        elif char == '[':
            self.bracket_depth += 1
            return 'enter_bracket'
        elif char == ']':
            self.bracket_depth = max(0, self.bracket_depth - 1)
            return 'exit_bracket'
        return None
    
    @property
    def current_depth(self) -> int:
        return self.brace_depth + self.paren_depth + self.bracket_depth
    
    def reset(self):
        self.__init__()


# Quick test
if __name__ == "__main__":
    tokenizer = CodeTokenizer()
    sample_code = """func main() {
    x := 42
    fmt.Println(x)
}"""
    
    tokens = tokenizer.tokenize(sample_code)
    print("Tokens with AST info:")
    for t in tokens[:20]:
        print(f"  {t[0]:20s}  type={t[1]:12s}  depth={t[2]}  sibling={t[3]}")
    print(f"\nTotal tokens: {len(tokens)}")
    print(f"Vocab size: {tokenizer.vocab_size}")
    
    input_ids, depth_ids, sibling_ids = tokenizer.encode("func test() { return 1 }")
    print(f"\nEncoded: {len(input_ids)} tokens")
    print(f"  input_ids: {input_ids[:10]}...")
    print(f"  depth_ids: {depth_ids[:10]}...")
    print(f"  sibling_ids: {sibling_ids[:10]}...")
