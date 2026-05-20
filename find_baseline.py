"""
暴力搜索：找出哪种代码结构能产生用户的基准哈希
"""
import ast, hashlib

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
        new_body = []
        for stmt in node.body:
            if (isinstance(stmt, ast.Return) and 
                isinstance(stmt.value, ast.IfExp)):
                if_exp = stmt.value
                new_body.append(ast.If(
                    test=if_exp.test,
                    body=[ast.Return(value=if_exp.body)],
                    orelse=[ast.Return(value=if_exp.orelse)]
                ))
            else:
                new_body.append(stmt)
        i = 0
        while i < len(new_body) - 1:
            curr = new_body[i]
            nxt = new_body[i+1]
            if (isinstance(curr, ast.If) and isinstance(nxt, ast.Return) and not curr.orelse):
                curr.orelse = [nxt]
                new_body.pop(i+1)
            i += 1
        node.body = new_body
        self.generic_visit(node)
        return node

def normalize_code(source):
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

BASELINE = "2707de8bce658bc0a1c07dbe3590e29667471cdbc0a1505d4a1e58d99af15759"

# 测试各种常见写法
variants = {
    "v1 标准内联-变量名a": '''def get_primes(numbers):
    primes = []
    for n in numbers:
        if n < 2:
            continue
        prime = True
        for i in range(2, int(n**0.5)+1):
            if n % i == 0:
                prime = False
                break
        if prime:
            primes.append(n)
    return primes''',

    "v2 标准内联-变量名b": '''def get_primes(numbers):
    result = []
    for n in numbers:
        if n < 2:
            continue
        prime_flag = True
        for divisor in range(2, int(n**0.5)+1):
            if n % divisor == 0:
                prime_flag = False
                break
        if prime_flag:
            result.append(n)
    return result''',

    "v3 独立函数-早期返回": '''def is_prime(num):
    if num < 2:
        return False
    for i in range(2, int(num**0.5)+1):
        if num % i == 0:
            return False
    return True

def get_primes(numbers):
    result = []
    for n in numbers:
        if is_prime(n):
            result.append(n)
    return result''',

    "v4 独立函数-布尔变量": '''def is_prime(n):
    if n < 2:
        return False
    is_prime = True
    for i in range(2, int(n**0.5)+1):
        if n % i == 0:
            is_prime = False
            break
    return is_prime

def filter_primes(numbers):
    result = []
    for n in numbers:
        if is_prime(n):
            result.append(n)
    return result''',

    "v5 单函数-带is_prime辅助在里面": '''def get_primes(numbers):
    result = []
    for num in numbers:
        if num >= 2:
            is_prime = True
            for i in range(2, int(num**0.5)+1):
                if num % i == 0:
                    is_prime = False
                    break
            if is_prime:
                result.append(num)
    return result''',

    "v6 while循环": '''def get_primes(numbers):
    result = []
    i = 0
    while i < len(numbers):
        n = numbers[i]
        if n >= 2:
            prime = True
            j = 2
            while j * j <= n:
                if n % j == 0:
                    prime = False
                    break
                j += 1
            if prime:
                result.append(n)
        i += 1
    return result''',

    "v7 列表推导式": '''def is_prime(n):
    return n > 1 and all(n % i != 0 for i in range(2, int(n**0.5)+1))

def get_primes(numbers):
    return [x for x in numbers if is_prime(x)]''',

    "v8 带early exit的is_prime": '''def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5)+1):
        if n % i == 0:
            return False
    return True

def get_primes(input_list):
    output = []
    for item in input_list:
        if is_prime(item):
            output.append(item)
    return output''',

    "v9 单函数 用flag变量 与v5不同变量名": '''def filter_primes(nums):
    out = []
    for x in nums:
        if x >= 2:
            ok = True
            for d in range(2, int(x**0.5)+1):
                if x % d == 0:
                    ok = False
                    break
            if ok:
                out.append(x)
    return out''',

    "v10 先排除偶数优化": '''def get_primes(numbers):
    result = []
    for n in numbers:
        if n < 2:
            continue
        if n == 2:
            result.append(n)
            continue
        if n % 2 == 0:
            continue
        is_prime = True
        for i in range(3, int(n**0.5)+1, 2):
            if n % i == 0:
                is_prime = False
                break
        if is_prime:
            result.append(n)
    return result''',

    "v11 用any/all": '''def get_primes(numbers):
    result = []
    for n in numbers:
        if n >= 2:
            if all(n % i != 0 for i in range(2, int(n**0.5)+1)):
                result.append(n)
    return result''',

    "v12 独立函数-用all": '''def is_prime(n):
    if n < 2:
        return False
    return all(n % i != 0 for i in range(2, int(n**0.5)+1))

def get_primes(numbers):
    result = []
    for n in numbers:
        if is_prime(n):
            result.append(n)
    return result''',
}

print("搜索基准哈希对应的代码结构...\n")
for name, code in variants.items():
    h = bitcode_hash(code)
    match = "🎯 MATCH!" if h == BASELINE else ""
    norm = normalize_code(code)
    print(f"{name}:")
    print(f"  hash={h[:20]}... {match}")
    if h == BASELINE:
        print(f"  ✅ 找到了！代码:\n{code}")
        print(f"  归一化后:\n{norm}")
        break
else:
    print("\n以上常见写法都没对上。尝试变体组合...")
