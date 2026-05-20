"""
执行结果共识测试

核心思路：不对比代码长什么样，对比代码跑出来的结果。
AST 归一化比对不上就换赛道——执行验证。

每个大模型输出代码 → 用同一组输入跑 → 对比输出 → 等价的输出哈希一致 → 共识达成
"""
import hashlib
import json
from typing import Callable

# ===== 测试输入 =====
TEST_CASES = {
    "get_primes": [
        [10, 15, 3, 7, 22, 2],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [-5, 0, 1, 11, 13, 17, 19, 23],
        [97, 98, 99, 100, 101, 102],
        [],
        [2],
        [4, 6, 8, 10],
    ]
}

# ===== 验证引擎 =====
class ExecutionVerifier:
    """接收不同模型写的代码，跑同一组输入，比对执行结果"""
    
    def __init__(self):
        self.results = {}
    
    def test_code(self, name: str, code: str):
        """执行一段代码，返回输出哈希"""
        ns = {"__builtins__": __builtins__, "TEST_CASES": TEST_CASES}
        try:
            # 先把用户代码（函数定义）执行到命名空间
            exec(code, ns)
            
            # 再跑测试
            func_name = "get_primes"
            get_primes_fn = ns[func_name]
            
            output = {}
            for inp in TEST_CASES[func_name]:
                out = get_primes_fn(inp)
                output[repr(inp)] = out
            
            # 序列化输出 → 计算哈希
            serialized = json.dumps(output, sort_keys=True)
            h = hashlib.sha256(serialized.encode()).hexdigest()
            return h, output
        except Exception as e:
            import traceback
            return f"ERROR: {e}", None


# ===== 渲染不同的代码风格（模拟不同大模型的输出） =====
CODE_VARIANTS = {
    "内联循环版 (Claude风格)": """
def get_primes(numbers):
    result = []
    for n in numbers:
        if n < 2:
            continue
        prime = True
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                prime = False
                break
        if prime:
            result.append(n)
    return result
""",

    "独立函数版 (GPT风格)": """
def is_prime(num):
    if num < 2:
        return False
    for divisor in range(2, int(num**0.5) + 1):
        if num % divisor == 0:
            return False
    return True

def get_primes(numbers):
    primes = []
    for x in numbers:
        if is_prime(x):
            primes.append(x)
    return primes
""",

    "列表推导式版 (DeepSeek风格)": """
def is_prime(n):
    return n > 1 and all(n % i != 0 for i in range(2, int(n**0.5) + 1))

def get_primes(numbers):
    return [x for x in numbers if is_prime(x)]
""",

    "filter+lambda版 (极简风格)": """
def get_primes(numbers):
    return list(filter(lambda n: n > 1 and all(n % i != 0 for i in range(2, int(n**0.5) + 1)), numbers))
""",
    
    "while循环版 (保守风格)": """
def get_primes(numbers):
    primes = []
    idx = 0
    while idx < len(numbers):
        n = numbers[idx]
        if n >= 2:
            is_prime = True
            d = 2
            while d * d <= n:
                if n % d == 0:
                    is_prime = False
                    break
                d += 1
            if is_prime:
                primes.append(n)
        idx += 1
    return primes
""",
    # ===== 以下留空，等你粘贴真实模型输出 =====
}


# ===== 主测试 =====
def main():
    print("=" * 65)
    print("执行结果共识验证 — Execution Consensus Test")
    print("=" * 65)
    print(f"\n测试函数: get_primes")
    print(f"测试输入: {TEST_CASES['get_primes']}")
    print()
    
    verifier = ExecutionVerifier()
    all_hashes = []
    
    for name, code in CODE_VARIANTS.items():
        if code.strip() == "":
            continue
        h, output = verifier.test_code(name, code)
        
        # 用 AST 归一化再算个哈希做对照
        try:
            import ast
            class BitCodeNormalizer(ast.NodeTransformer):
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
                    if node.id in dir(__builtins__):
                        return node
                    node.id = self._canonical_name(node.id)
                    return node
                def visit_FunctionDef(self, node):
                    node.name = self._canonical_name(node.name)
                    if (node.body and isinstance(node.body[0], ast.Expr) and 
                        isinstance(node.body[0].value, ast.Constant) and 
                        isinstance(node.body[0].value.value, str)):
                        node.body.pop(0)
                    self.generic_visit(node)
                    return node
            def ast_hash(code):
                try:
                    tree = ast.parse(code)
                    n = BitCodeNormalizer()
                    nt = n.visit(tree)
                    ast.fix_missing_locations(nt)
                    return hashlib.sha256(ast.unparse(nt).encode()).hexdigest()
                except:
                    return "N/A"
            ast_h = ast_hash(code)
        except:
            ast_h = "N/A"
        
        status = "❌ ERROR" if h.startswith("ERROR") else "✅"
        
        if not h.startswith("ERROR"):
            all_hashes.append(h)
        
        print(f"\n--- {name} ---")
        print(f"  AST归一化哈希: {ast_h[:20]}...")
        print(f"  执行结果哈希:  {h[:20]}...  {status}")
        if output:
            # 只打印第一个测试用例的结果
            first_input = repr(TEST_CASES['get_primes'][0])
            first_output = output.get(first_input)
            print(f"  样例输出: get_primes({TEST_CASES['get_primes'][0]}) = {first_output}")
    
    # 共识判定
    success_hashes = [h for h in all_hashes if not h.startswith("ERROR")]
    unique_hashes = set(success_hashes)
    
    print("\n" + "=" * 65)
    print("共识判定")
    print("=" * 65)
    
    if len(all_hashes) < 2:
        print("⚠️  样本太少，无法判定共识")
    
    if len(unique_hashes) == 1 and len(success_hashes) >= 2:
        print(f"🎉🎉🎉 跨模型共识达成！")
        print(f"   所有 {len(success_hashes)} 个版本的执行结果哈希完全一致:")
        print(f"   {list(unique_hashes)[0][:20]}...")
        print(f"\n   AST 归一化可能对不上，但执行结果对上了")
        print(f"   → 代码怎么写不重要，跑出来一样就行")
    else:
        print(f"❌ 未达成共识")
        print(f"   共 {len(success_hashes)} 个成功执行版本")
        print(f"   产生了 {len(unique_hashes)} 个不同的执行结果哈希")
        for h in unique_hashes:
            count = success_hashes.count(h)
            print(f"   {h[:20]}... × {count}")
        print(f"\n   → 不同版本的代码算出了不同结果")

    print("=" * 65)
    print("\n📋 使用方法:")
    print("  1. 把 GPT-4/Claude/Gemini/DeepSeek 的回复贴到下面")
    print("  2. python execution_consensus_test.py")
    print("  3. 看几个模型跑出了相同的结果哈希")


if __name__ == "__main__":
    main()
