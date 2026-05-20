"""
BitCode 跨模型验证 - 基于 AST 归一化的哈希比对
对照你桌面上的 bitcode-compare.py 实现
"""
import ast
import hashlib
import json


class BitCodeNormalizer(ast.NodeTransformer):
    """AST 归一化器 - 变量名抹除 + 控制流标准化"""
    
    def __init__(self):
        self.var_map = {}
        self.counter = 0
    
    def _canonical_name(self, name):
        if name not in self.var_map:
            self.var_map[name] = f"v_{self.counter}"
            self.counter += 1
        return self.var_map[name]
    
    def visit_arg(self, node):
        node.arg = self._canonical_name(node.arg)
        return node
    
    def visit_Name(self, node):
        # 忽略内建名
        if node.id in dir(__builtins__):
            return node
        node.id = self._canonical_name(node.id)
        return node
    
    def visit_FunctionDef(self, node):
        # 归一化函数名
        node.name = self._canonical_name(node.name)
        
        # 移除 docstring
        if (node.body and isinstance(node.body[0], ast.Expr) and 
            isinstance(node.body[0].value, ast.Constant) and 
            isinstance(node.body[0].value.value, str)):
            node.body.pop(0)
        
        # 控制流标准化：三元表达式转 if-else
        new_body = []
        for stmt in node.body:
            if (isinstance(stmt, ast.Return) and 
                isinstance(stmt.value, ast.IfExp)):
                # a if cond else b → if cond: return a else: return b
                if_exp = stmt.value
                new_body.append(
                    ast.If(
                        test=if_exp.test,
                        body=[ast.Return(value=if_exp.body)],
                        orelse=[ast.Return(value=if_exp.orelse)]
                    )
                )
            else:
                new_body.append(stmt)
        node.body = new_body
        
        # 早期返回转 if-else
        # if cond: return a; return b → if cond: return a else: return b
        i = 0
        while i < len(node.body) - 1:
            curr = node.body[i]
            next_stmt = node.body[i + 1]
            if (isinstance(curr, ast.If) and 
                isinstance(next_stmt, ast.Return) and
                not curr.orelse):  # 当前 if 没有 else
                curr.orelse = [next_stmt]
                node.body.pop(i + 1)
            i += 1
        
        # 递归处理子节点
        self.generic_visit(node)
        return node


def normalize_code(source):
    """代码 → AST 归一化 → 代码字符串"""
    try:
        tree = ast.parse(source)
        n = BitCodeNormalizer()
        nt = n.visit(tree)
        ast.fix_missing_locations(nt)
        return ast.unparse(nt)
    except SyntaxError:
        return source


def bitcode_hash(source):
    return hashlib.sha256(normalize_code(source).encode('utf-8')).hexdigest()


def extract_code(raw):
    """从模型回复中提取代码"""
    lines = raw.strip().split('\n')
    code_lines = []
    in_code = False
    for line in lines:
        if line.startswith('```'):
            in_code = not in_code
            continue
        if in_code:
            code_lines.append(line)
    if not code_lines:
        # 没有代码块标记，尝试找 def 开头
        for i, line in enumerate(lines):
            if line.startswith('def ') or line.startswith('class '):
                code_lines = lines[i:]
                break
    return '\n'.join(code_lines).strip()


# ===== 模型输出 =====

BASELINE_HASH = "2707de8bce658bc0a1c07dbe3590e29667471cdbc0a1505d4a1e58d99af15759"

MODEL_OUTPUTS = {
    "DeepSeek (沙箱)": '''def get_primes(numbers):
    result = []
    for n in numbers:
        if n < 2:
            continue
        is_prime = True
        for i in range(2, int(n ** 0.5) + 1):
            if n % i == 0:
                is_prime = False
                break
        if is_prime:
            result.append(n)
    return result''',

    "DeepSeek (沙箱 v2 - 带独立函数)": '''def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True

def get_primes(numbers):
    result = []
    for n in numbers:
        if is_prime(n):
            result.append(n)
    return result''',

    "DeepSeek (沙箱 v3 - 列表推导式)": '''def is_prime(n):
    return n > 1 and all(n % i != 0 for i in range(2, int(n ** 0.5) + 1))

def get_primes(numbers):
    return [n for n in numbers if is_prime(n)]''',
}

# ===== 主流程 =====

print("=" * 60)
print("BitCode 跨模型验证")
print("=" * 60)

results = []
for model_name, raw_output in MODEL_OUTPUTS.items():
    code = extract_code(raw_output)
    h = bitcode_hash(code)
    match = "✅ MATCH" if h == BASELINE_HASH else "❌ MISMATCH"
    results.append(match == "✅ MATCH")
    
    print(f"\n--- {model_name} ---")
    print(f"代码:\n{code[:200]}")
    print(f"归一化后:\n{normalize_code(code)[:200]}")
    print(f"哈希: {h}")
    print(f"比对: {match}")

print("\n" + "=" * 60)
print(f"共识达成: {sum(results)}/{len(results)}")
if all(results):
    print("🎉 达成跨模型共识！")
else:
    print(f"💡 {len(results) - sum(results)} 个模型结构不同，未达成共识")
    print("  原因：内联 vs 独立函数 vs 列表推导式的 AST 结构本质不同")
    print("  修复方向：Level 4 语义归一化 — 函数内联展开 + 推导式转循环")
