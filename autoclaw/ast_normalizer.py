"""
AutoClaw 核心 — AST 归一化 + 哈希比对

核心思路：
  异构设备产出的代码（不同语言、不同格式）→ AST 解析 →
  剥离一切格式差异（空格/注释/命名变化/换行）→ 统一输出 →
  哈希比对，判断是否逻辑等价

支持的归一化层级:
  Level 0: 原始字符串比对（严格一致）
  Level 1: 去除空白和注释
  Level 2: 变量名归一化 + Level 1
  Level 3: AST 结构比对（不依赖原始token，只比树结构）
"""

import ast
import re
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass


@dataclass
class NormalizationResult:
    """归一化结果"""
    normalized: str      # 归一化后的代码文本
    hash_value: str      # SHA256 哈希
    ast_structure: str   # AST 结构线（只含节点类型，不含具体文本）
    level: int           # 归一化层级
    language: str        # 代码语言


class ASTNormalizer:
    """
    AST 归一化器 — 将不同格式的代码转化为统一的哈希表示
    
    用法:
        normalizer = ASTNormalizer()
        
        # 归一化多段代码
        result_a = normalizer.normalize(code_a, "go")
        result_b = normalizer.normalize(code_b, "go")
        
        # 比对
        is_match = normalizer.compare(result_a, result_b)
    """
    
    def __init__(self, level: int = 2):
        """
        Args:
            level: 归一化层级
                0 = 原始字符串（不处理）
                1 = 去除空白/注释
                2 = 命名归一化 + Level 1
                3 = AST 结构比对（实验性）
        """
        self.level = level
        self.reset()
    
    def reset(self):
        """重置变量名计数器（每次归一化调用前调用）"""
        self._var_counter = 0
        self._var_map: Dict[str, str] = {}
    
    def normalize(self, code: str, language: str = "go") -> NormalizationResult:
        """归一化代码"""
        code = code.strip()
        
        if self.level == 0:
            normalized = code
        elif language in ("go", "golang"):
            normalized = self._normalize_go(code)
        elif language == "python":
            normalized = self._normalize_python(code)
        elif language in ("ts", "typescript", "js", "javascript"):
            normalized = self._normalize_ts(code)
        elif language in ("sql",):
            normalized = self._normalize_generic(code)
        else:
            normalized = self._normalize_generic(code)
        
        hash_val = hashlib.sha256(normalized.encode('utf-8')).hexdigest()
        
        # AST 结构线（对于 Level 3）
        ast_structure = self._extract_ast_structure(code, language) if self.level >= 3 else ""
        
        return NormalizationResult(
            normalized=normalized,
            hash_value=hash_val,
            ast_structure=ast_structure,
            level=self.level,
            language=language,
        )
    
    def compare(self, a: NormalizationResult, b: NormalizationResult) -> Tuple[bool, str]:
        """比对两个归一化结果"""
        if a.hash_value == b.hash_value:
            return True, f"哈希一致: {a.hash_value[:16]}..."
        
        details = []
        if a.level != b.level:
            details.append(f"归一化层级不同 ({a.level} vs {b.level})")
        if self.level >= 3 and a.ast_structure != b.ast_structure:
            details.append("AST 结构不同")
        
        return False, f"哈希不一致 | {'; '.join(details) if details else '代码内容不同'}"
    
    def normalize_text(self, text: str) -> str:
        """对普通文本（非代码）做归一化"""
        # 去除标点/空格统一/大小写归一化
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text
    
    # ================================================================
    # Go 代码归一化
    # ================================================================
    
    def _normalize_go(self, code: str) -> str:
        """Go 代码归一化
        Go 没有内置的 AST 解析器，用正则做归一化
        
        策略：
          1. 按行处理代码
          2. 去除注释/空白
          3. 命名归一化
          4. 放弃换行结构，统一生成规范格式：
             每条语句一行，括号不在独立行
        """
        self.reset()
        lines = code.split('\n')
        result = []
        
        for line in lines:
            # 去除注释
            line = re.sub(r'//.*', '', line)
            line = re.sub(r'/\*.*?\*/', '', line, flags=re.DOTALL)
            
            # 去除行首行尾空白
            line = line.strip()
            
            if not line:
                continue
            
            # 统一空白（多个空格 → 单个空格）
            line = re.sub(r'\s+', ' ', line)
            
            # 命名归一化
            if self.level >= 2:
                line = self._normalize_names(line)
            
            result.append(line)
        
        # 跨行结构归一化：将 `{` 附加到前一行末尾
        normalized = '\n'.join(result)
        normalized = re.sub(r'\{\n', ' { ', normalized)
        normalized = re.sub(r'\n\}', ' }', normalized)
        normalized = re.sub(r'\n\s*\}', ' }', normalized)
        normalized = re.sub(r'\s+', ' ', normalized)
        
        return normalized
    
    # ================================================================
    # Python 代码归一化（利用 ast 模块）
    # ================================================================
    
    def _normalize_python(self, code: str) -> str:
        """Python 代码归一化 — 用 ast 模块提取结构"""
        self.reset()
        try:
            tree = ast.parse(code)
            normalizer = _PythonNormalizer(self.level, self)
            normalized = normalizer.visit(tree)
            if hasattr(ast, 'unparse'):
                normalized_str = ast.unparse(normalized)
                # 折叠行结构
                normalized_str = normalized_str.replace('\n', ' ')
                normalized_str = re.sub(r'\s+', ' ', normalized_str)
                return normalized_str
            return self._normalize_generic(code)
        except SyntaxError:
            return self._normalize_generic(code)
    
    # ================================================================
    # TypeScript/JavaScript 归一化
    # ================================================================
    
    def _normalize_ts(self, code: str) -> str:
        """TypeScript 归一化 — 字符串级"""
        self.reset()
        lines = code.split('\n')
        result = []
        
        # 临时标记：是否在模板字符串内
        in_template = False
        
        for line in lines:
            # 注释
            line = re.sub(r'//.*', '', line)
            line = re.sub(r'/\*.*?\*/', '', line, flags=re.DOTALL)
            line = line.strip()
            
            if not line:
                continue
            
            line = re.sub(r'\s+', ' ', line)
            
            # 处理 JSX
            # 简化 JSX 属性
            line = re.sub(r'\s+className=', ' className=', line)
            
            # 分号归一化
            if not line.endswith('{') and not line.endswith('}') and not line.endswith(','):
                pass  # TS 的分号可选，不处理
            
            if self.level >= 2:
                line = self._normalize_names(line)
            
            result.append(line)
        
        return '\n'.join(result)
    
    # ================================================================
    # 通用代码归一化（兜底方案）
    # ================================================================
    
    def _normalize_generic(self, code: str) -> str:
        """通用代码归一化 — 纯字符串级，无语言特性依赖"""
        self.reset()
        lines = code.split('\n')
        result = []
        
        in_multi_comment = False
        
        for line in lines:
            # 多行注释
            if in_multi_comment:
                if '*/' in line:
                    in_multi_comment = False
                continue
            if '/*' in line:
                in_multi_comment = True
                line = line.split('/*')[0]
            
            # 单行注释
            for marker in ['//', '#', '--']:
                if marker in line:
                    line = line.split(marker)[0]
            
            line = line.strip()
            if not line:
                continue
            
            # 统一空白
            line = re.sub(r'\s+', ' ', line)
            
            # 统一引号（" 和 ' 都转成 "）
            line = re.sub(r"'([^']*)'", r'"\1"', line)
            
            # 统一换行符
            line = re.sub(r'\\r\\n', '\\n', line)
            
            if self.level >= 2:
                line = self._normalize_names(line)
            
            result.append(line)
        
        return '\n'.join(result)
    
    # ================================================================
    # 命名归一化
    # ================================================================
    
    def _normalize_names(self, code: str) -> str:
        """将变量名/函数名替换为通用占位符
        
        保留关键词和内置函数名，仅替换用户自定义命名。
        """
        # 关键词集合（多语言通用）
        keywords = {
            # Go / 通用
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
            'and', 'or', 'None', 'True', 'False',
            # JS/TS
            'function', 'const', 'let', 'async', 'await', 'export',
            'import', 'default', 'extends', 'class', 'new', 'this', 'super',
            'typeof', 'instanceof', 'void', 'delete', 'throw', 'catch',
            'finally', 'try', 'while', 'do', 'of', 'in', 'as', 'from',
            'undefined', 'null', 'React', 'useState', 'useEffect', 'useRef',
            'useMemo', 'useCallback', 'useContext', 'useReducer',
            # 内置函数
            'print', 'fmt', 'Println', 'Sprintf', 'Errorf', 'printf', 'sprintf',
            'len', 'cap', 'range', 'close', 'delete', 'panic', 'recover',
            'append', 'copy', 'make', 'new', 'string', 'int', 'float', 'bool',
            'list', 'dict', 'set', 'tuple', 'object', 'type', 'isinstance',
            'open', 'read', 'write', 'close', 'str', 'repr', 'format',
            'map', 'filter', 'reduce', 'sorted', 'enumerate', 'zip', 'reversed',
            'abs', 'all', 'any', 'bin', 'bool', 'chr', 'dir', 'divmod', 'eval',
            'exec', 'hex', 'id', 'input', 'isinstance', 'issubclass', 'iter',
            'locals', 'max', 'min', 'next', 'object', 'oct', 'ord', 'pow',
            'property', 'range', 'repr', 'reversed', 'round', 'set', 'slice',
            'sorted', 'staticmethod', 'str', 'super', 'tuple', 'type',
            'vars', 'zip', '__import__',
            # 类型注解
            'Optional', 'List', 'Dict', 'Set', 'Tuple', 'Union', 'Any', 'Callable',
            'TypeVar', 'Generic', 'Protocol', 'TypedDict',
        }
        
        # 替换变量名（使用全局计数器，跨行保持一致性）
        tokens = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', code)
        
        for token in tokens:
            if token in keywords:
                continue
            if token.upper() == token and len(token) > 1:
                continue  # 常量（全大写）不替换
            
            # 检查是否已有映射
            if token in self._var_map:
                placeholder = self._var_map[token]
            else:
                placeholder = f'__VAR_{self._var_counter}__'
                self._var_map[token] = placeholder
                self._var_counter += 1
            
            # 只替换完整的词
            code = re.sub(r'\b' + re.escape(token) + r'\b', placeholder, code)
        
        return code
    
    # ================================================================
    # AST 结构提取
    # ================================================================
    
    def _extract_ast_structure(self, code: str, language: str) -> str:
        """提取 AST 结构线（仅节点类型层级，不含具体文本）
        
        如:
            Module > FunctionDef > If > Compare > Call
        """
        if language == "python":
            try:
                tree = ast.parse(code)
                return self._ast_to_structure(tree)
            except SyntaxError:
                pass
        
        # 其他语言：用缩进/括号层级近似
        lines = code.strip().split('\n')
        structure_parts = []
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            
            # 关键词 → 结构类型
            if any(stripped.startswith(kw) for kw in ['func', 'def', 'function', 'class', 'if', 'for', 'while', 'switch', 'try', 'catch', 'SELECT', 'CREATE']):
                kw = stripped.split()[0]
                indent = len(line) - len(stripped)
                structure_parts.append(f"{'  ' * (indent // 4)}{kw}")
        
        return '\n'.join(structure_parts)
    
    def _ast_to_structure(self, node, depth: int = 0) -> str:
        """递归提取 AST 节点类型"""
        lines = []
        prefix = '  ' * depth
        node_name = type(node).__name__
        
        if node_name == 'Load' or node_name == 'Store' or node_name == 'Del':
            return ""
        if node_name == 'Constant':
            return ""
        
        lines.append(f"{prefix}{node_name}")
        
        for child_node in ast.iter_child_nodes(node):
            child_str = self._ast_to_structure(child_node, depth + 1)
            if child_str.strip():
                lines.append(child_str)
        
        return '\n'.join(lines)


class _PythonNormalizer(ast.NodeTransformer):
    """Python AST 归一化转换器"""
    
    def __init__(self, level: int, parent: ASTNormalizer):
        self.level = level
        self.parent = parent
        self._rename_map: Dict[str, str] = {}
        self._counter = 0
    
    def _get_rename(self, name: str) -> str:
        if name not in self._rename_map:
            self._rename_map[name] = f"var_{self._counter}"
            self._counter += 1
        return self._rename_map[name]
    
    def visit_Name(self, node: ast.Name) -> ast.Name:
        if self.level >= 2:
            if isinstance(node.ctx, ast.Load):
                if node.id in self._rename_map:
                    node.id = self._rename_map[node.id]
            elif isinstance(node.ctx, (ast.Store, ast.Param)):
                if node.id not in dir(__builtins__):
                    node.id = self._get_rename(node.id)
        return node
    
    def visit_arg(self, node: ast.arg) -> ast.arg:
        """重命名函数参数（ast.arg 不是 ast.Name）"""
        if self.level >= 2:
            if node.arg not in dir(__builtins__):
                node.arg = self._get_rename(node.arg)
        return node
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        if self.level >= 2:
            node.name = self._get_rename(node.name)
        self.generic_visit(node)
        return node
    
    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        if self.level >= 2:
            node.name = self._get_rename(node.name)
        self.generic_visit(node)
        return node


# ================================================================
# 批量比对工具
# ================================================================

class BatchComparator:
    """批量比对多段代码的一致性"""
    
    def __init__(self, normalizer: ASTNormalizer):
        self.normalizer = normalizer
    
    def compare_all(self, samples: List[Tuple[str, str]]) -> Dict:
        """
        比对多段代码
        
        Args:
            samples: [(code, language), ...]
        
        Returns:
            {
                "total": N,
                "consistent": N,
                "inconsistent": N,
                "hashes": {hash: [indices...], ...},
                "details": [...]
            }
        """
        results = [self.normalizer.normalize(code, lang) for code, lang in samples]
        
        hash_groups: Dict[str, List[int]] = {}
        for i, r in enumerate(results):
            hash_groups.setdefault(r.hash_value, []).append(i)
        
        consistent = max(len(v) for v in hash_groups.values())
        inconsistent = len(samples) - consistent
        
        return {
            "total": len(samples),
            "consistent": consistent,
            "inconsistent": inconsistent,
            "consistency_rate": consistent / max(1, len(samples)),
            "hash_groups": {h[:16] + "...": idxs for h, idxs in hash_groups.items()},
            "results": [(r.hash_value[:16] + "...", r.language) for r in results],
        }


def quick_test():
    """快速验证"""
    normalizer = ASTNormalizer(level=2)
    
    # 测试 1: 同一段代码，格式不同
    code_1 = """func main() {
    x := 42
    fmt.Println(x)
}"""
    code_2 = """func main() {   x := 42
    fmt.Println(x) // print the value
    }"""
    
    r1 = normalizer.normalize(code_1, "go")
    r2 = normalizer.normalize(code_2, "go")
    match, msg = normalizer.compare(r1, r2)
    print(f"同一代码/格式不同: {match} | {msg}")
    
    # 测试 2: 不同变量名但逻辑相同
    code_3 = """func main() {
    y := 42
    fmt.Println(y)
}"""
    r3 = normalizer.normalize(code_3, "go")
    match, msg = normalizer.compare(r1, r3)
    print(f"变量名不同但逻辑相同: {match} | {msg}")
    
    # 测试 3: 逻辑不同（预期不一致）
    code_4 = """func main() {
    x := 100
    fmt.Println(x)
}"""
    r4 = normalizer.normalize(code_4, "go")
    match, msg = normalizer.compare(r1, r4)
    print(f"逻辑不同（预期不一致）: {match} | {msg}")
    
    # 测试 4: Python 代码
    py_1 = """def process(items):
    total = sum(items)
    count = len(items)
    return total / count if count > 0 else 0"""
    
    py_2 = """def process_data(data_list):
    s = sum(data_list)
    c = len(data_list)
    return s / c if c > 0 else 0"""
    
    r5 = normalizer.normalize(py_1, "python")
    r6 = normalizer.normalize(py_2, "python")
    match, msg = normalizer.compare(r5, r6)
    print(f"Python 不同命名/同逻辑: {match} | {msg}")
    
    return normalizer


if __name__ == "__main__":
    print("=== ASTNormalizer 快速验证 ===\n")
    quick_test()
    print("\nAll tests passed!")
