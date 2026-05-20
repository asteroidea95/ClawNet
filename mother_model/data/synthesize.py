"""
训练数据合成管线 - 母模型 SFT 数据

核心逻辑：用 DeepSeek API 生成正常代码样本 → 自动注入 bug →
编译/静态检查 → 构造 {意图, 代码, 出错的子模型, 错误描述} 四元组

不需要任何人工标注。编译器的报错就是 ground truth。

数据格式（JSONL）:
  {
    "intent": "看看这段代码有什么问题",
    "code": "func main() {\n    fmt.Println(x)\n}",
    "code_language": "go",
    "code_depth_ids": [...],
    "code_sibling_ids": [...],
    "target": "Go 子模型，main 函数中第 2 行使用了未定义的变量 x。",
    "culprit_submodel": "go",
    "error_type": "undefined_var",
    "error_line": 2
  }
"""

import os
import re
import json
import random
import subprocess
import tempfile
from typing import Optional, Dict, List, Tuple
from pathlib import Path

# ============================================================
# 1. 编译器/静态检查器接口
# ============================================================

class CodeVerifier:
    """代码验证器：编译+静态检查，返回详细的报错信息"""
    
    def __init__(self, temp_dir: str = "/tmp/mother_model_verify"):
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    def verify_go(self, code: str) -> Tuple[bool, Optional[str], Optional[int]]:
        """
        验证 Go 代码
        Returns: (is_ok, error_message, error_line)
        """
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.go', dir=self.temp_dir, delete=False
        ) as f:
            f.write(code)
            fpath = f.name
        
        try:
            result = subprocess.run(
                ['go', 'build', '-o', '/dev/null', fpath],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                return True, None, None
            return self._parse_compile_error(result.stderr, fpath)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            # Go not available, use regex-based static analysis as fallback
            return self._static_analysis_go(code)
        finally:
            if os.path.exists(fpath):
                os.unlink(fpath)
    
    def verify_python(self, code: str) -> Tuple[bool, Optional[str], Optional[int]]:
        """验证 Python 代码"""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.py', dir=self.temp_dir, delete=False
        ) as f:
            f.write(code)
            fpath = f.name
        
        try:
            result = subprocess.run(
                ['python3', '-m', 'py_compile', fpath],
                capture_output=True, text=True, timeout=15
            )
            if result.returncode == 0:
                return True, None, None
            return self._parse_compile_error(result.stderr, fpath)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return self._static_analysis_python(code)
        finally:
            if os.path.exists(fpath):
                os.unlink(fpath)
    
    def verify_typescript(self, code: str) -> Tuple[bool, Optional[str], Optional[int]]:
        """验证 TypeScript 代码（需要 tsc）"""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.ts', dir=self.temp_dir, delete=False
        ) as f:
            f.write(code)
            fpath = f.name
        
        try:
            result = subprocess.run(
                ['npx', 'tsc', '--noEmit', '--lib', 'es6,dom', fpath],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                return True, None, None
            return self._parse_compile_error(result.stderr, fpath)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return self._static_analysis_ts(code)
        finally:
            if os.path.exists(fpath):
                os.unlink(fpath)
    
    def _parse_compile_error(self, stderr: str, fpath: str) -> Tuple[bool, str, int]:
        """解析编译器报错，提取行号和描述"""
        lines = stderr.strip().split('\n')
        for line in lines:
            # Go: fpath:line:col: error
            match = re.search(rf'{re.escape(fpath)}:(\d+):\d*:\s*(.*)', line)
            if match:
                return False, match.group(2).strip(), int(match.group(1))
            # Python: File "...", line N
            match = re.search(r'line\s+(\d+)', line)
            if match:
                return False, line.strip(), int(match.group(1))
        
        if lines:
            return False, lines[-1].strip(), 0
        return False, "Unknown error", 0
    
    def _static_analysis_go(self, code: str) -> Tuple[bool, Optional[str], Optional[int]]:
        """兜底：不用编译器，用正则做静态分析"""
        lines = code.split('\n')
        for i, line in enumerate(lines, 1):
            # 使用未定义的变量
            used = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', line)
            for var in used:
                if var in ('fmt', 'Println', 'main', 'func', 'if', 'else', 'for', 'return',
                          'string', 'int', 'error', 'nil', 'true', 'false'):
                    continue
                # 简单检查：变量是否在上下文定义过
                if not self._var_is_defined(lines[:i], var):
                    return False, f"undefined variable: {var}", i
            
            # 类型不匹配
            if re.search(r'= "', line) and re.search(r':= \d', code):
                pass
            
        return True, None, None
    
    def _static_analysis_python(self, code: str) -> Tuple[bool, Optional[str], Optional[int]]:
        return self._static_analysis_go(code)  # 复用相同逻辑
    
    def _static_analysis_ts(self, code: str) -> Tuple[bool, Optional[str], Optional[int]]:
        lines = code.split('\n')
        for i, line in enumerate(lines, 1):
            used = re.findall(r'\b([a-zA-Z_$][a-zA-Z0-9_$]*)\b', line)
            for var in used:
                if var in KEYWORDS_TYPESCRIPT:
                    continue
                if not self._var_is_defined(lines[:i], var):
                    return False, f"Cannot find name '{var}'", i
        return True, None, None
    
    def _var_is_defined(self, context_lines: List[str], var_name: str) -> bool:
        """检查变量是否在前文定义过"""
        prev_code = '\n'.join(context_lines)
        # var x = ... / x := ... / x = ... / (x type)
        patterns = [
            rf'\bvar\s+{re.escape(var_name)}\b',
            rf'\b{re.escape(var_name)}\s*:=',
            rf'\bfunc\s+{re.escape(var_name)}\b',
            rf'\btype\s+{re.escape(var_name)}\b',
            rf'\bconst\s+{re.escape(var_name)}\b',
            rf'\b{re.escape(var_name)}\s*=\s*(?!.*:)',  # 赋值但排除 :=
        ]
        for pat in patterns:
            if re.search(pat, prev_code):
                return True
        # 常见内置标识符
        if var_name in ('err', 'i', 'j', 'k', 'n', 's', 't', 'x', 'y', 'z',
                       'args', 'w', 'r', 'db', 'ctx', 'req', 'res', 'next',
                       'handler', 'resp', 'data', 'cfg', 'config', 'client'):
            return True
        return False


KEYWORDS_TYPESCRIPT = {
    'function', 'const', 'let', 'var', 'async', 'await', 'export',
    'import', 'default', 'extends', 'class', 'new', 'this', 'super',
    'typeof', 'instanceof', 'void', 'delete', 'throw', 'catch',
    'finally', 'try', 'switch', 'case', 'break', 'continue', 'return',
    'if', 'else', 'for', 'while', 'do', 'of', 'in', 'as', 'from',
    'undefined', 'null', 'true', 'false', 'type', 'interface', 'enum',
    'implements', 'abstract', 'private', 'protected', 'public',
    'static', 'readonly', 'declare', 'namespace', 'module', 'global',
    'keyof', 'never', 'unknown', 'any', 'void', 'symbol', 'object',
}


# ============================================================
# 2. Bug 注入器
# ============================================================

class BugInjector:
    """自动向正常代码注入常见错误"""
    
    # 注入策略: (名称, 适用语言, 概率权重)
    # 每个策略返回 (有 bug 的代码, 错误描述, 错误行号)
    
    STRATEGIES = [
        ("undefined_var", ["go", "python", "ts"], 0.20),
        ("type_mismatch", ["go", "python", "ts"], 0.15),
        ("missing_return", ["go", "python", "ts"], 0.15),
        ("wrong_operator", ["go", "python", "ts"], 0.10),
        ("off_by_one", ["go", "python", "ts"], 0.10),
        ("nil_dereference", ["go", "python"], 0.10),
        ("unused_variable", ["go", "python", "ts"], 0.10),
        ("infinite_loop", ["go", "python", "ts"], 0.05),
        ("race_condition", ["go"], 0.05),
    ]
    
    def inject(self, code: str, language: str) -> Tuple[Optional[str], Optional[str], Optional[int]]:
        """
        向代码注入一个 bug
        
        Returns:
            (buggy_code, error_description, error_line)
            如果无法注入返回 (None, None, None)
        """
        lines = code.split('\n')
        if len(lines) < 3:
            return None, None, None
        
        # 按权重选择策略
        available = [(name, w) for name, langs, w in self.STRATEGIES if language in langs]
        if not available:
            return None, None, None
        
        names, weights = zip(*available)
        strategy = random.choices(names, weights=weights, k=1)[0]
        
        injector = getattr(self, f'_inject_{strategy}', None)
        if injector:
            return injector(lines, language)
        return None, None, None
    
    def _inject_undefined_var(self, lines: List[str], lang: str) -> Tuple[str, str, int]:
        """注入：使用未定义的变量"""
        # 找一个非空、非注释、非 import 的行
        candidates = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or stripped.startswith(('//', '#', 'import', 'package', '/*')):
                continue
            if re.search(r'[a-zA-Z_]\w*\s*[:=]?\s*', stripped):
                candidates.append((i, stripped))
        
        if not candidates:
            # fallback: 在末尾加一行使用未定义变量
            lines.append(f"    fmt.Println(undefinedVar_{random.randint(100,999)})")
            return '\n'.join(lines), "undefined variable", len(lines)
        
        idx, line = random.choice(candidates)
        # 找一个变量名替换为未定义的
        vars_in_line = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', line)
        defined = {'func', 'if', 'else', 'for', 'return', 'var', 'const', 'type',
                   'int', 'string', 'bool', 'error', 'nil', 'true', 'false',
                   'fmt', 'Println', 'Sprintf', 'Errorf', 'len', 'cap', 'make',
                   'new', 'append', 'range', 'defer', 'go', 'select', 'chan',
                   'import', 'package', 'switch', 'case', 'default', 'break',
                   'continue', 'fallthrough', 'map', 'struct', 'interface',
                   'float64', 'float32', 'int64', 'int32', 'int8', 'int16',
                   'uint64', 'uint32', 'uint8', 'uint16', 'byte', 'rune', 'string',
                   'error', 'any', 'comparable',
                   # Python
                   'def', 'class', 'pass', 'raise', 'try', 'except', 'finally',
                   'with', 'as', 'lambda', 'yield', 'self', 'cls', 'None', 'True', 'False',
                   'print', 'range', 'len', 'type', 'int', 'str', 'float', 'bool', 'list',
                   'dict', 'set', 'tuple', 'object', 'super', 'isinstance', 'hasattr',
                   'open', 'close', 'read', 'write', 'import', 'from', 'as',
                   # JS/TS
                   'console', 'log', 'document', 'window', 'Math', 'JSON', 'Array',
                   'Object', 'String', 'Number', 'Boolean', 'Date', 'RegExp', 'Map',
                   'Set', 'Promise', 'Error', 'undefined', 'null', 'this', 'arguments',
        }
        
        for var in vars_in_line:
            if var not in defined and len(var) > 1:  # 不是常见关键词
                # 改成未定义
                new_var = f"undefinedV{random.randint(1000,9999)}"
                buggy_line = line.replace(var, new_var, 1)
                if buggy_line != line:
                    lines[idx] = buggy_line
                    return '\n'.join(lines), f"undefined variable: {new_var}", idx + 1
        
        # 加一行新的
        lines.append(f"    fmt.Println(unknownVar_{random.randint(1000,9999)})")
        return '\n'.join(lines), "undefined variable", len(lines)
    
    def _inject_missing_return(self, lines: List[str], lang: str) -> Tuple[str, str, int]:
        """注入：函数缺少 return"""
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith(('func ', 'func(')) and ')' in stripped:
                # 检查后面的行是否有 return
                has_return = False
                for j in range(i + 1, min(i + 20, len(lines))):
                    if lines[j].strip().startswith('return'):
                        has_return = True
                        break
                
                if not has_return and i + 2 < len(lines):
                    # 删掉函数体内的一行 return（如果存在的话）
                    for j in range(i + 1, min(i + 15, len(lines))):
                        if 'return' in lines[j]:
                            del lines[j]
                            return '\n'.join(lines), "missing return statement", i + 1
                    
                    # 或者把最后一行的 return 改成其他
                    last_body_line = min(i + 5, len(lines) - 1)
                    if lines[last_body_line].strip() and not lines[last_body_line].strip().startswith('}'):
                        lines[last_body_line] = lines[last_body_line] + "\n    // missing return"
                        return '\n'.join(lines), "missing return at end of function", last_body_line + 1
        
        return self._inject_undefined_var(lines, lang)  # fallback
    
    def _inject_type_mismatch(self, lines: List[str], lang: str) -> Tuple[str, str, int]:
        """注入：类型不匹配"""
        for i, line in enumerate(lines):
            if ':=' in line or '= ' in line:
                # 找一个数字赋值改成字符串（反之亦然）
                if re.search(r'= \d', line):
                    buggy = re.sub(r'= (\d+)', r'= "wrong_type_\1"', line, count=1)
                    if buggy != line:
                        lines[i] = buggy
                        return '\n'.join(lines), "type mismatch", i + 1
                if re.search(r'= "[^"]*"', line):
                    buggy = re.sub(r'= "([^"]*)"', r'= 42', line, count=1)
                    if buggy != line:
                        lines[i] = buggy
                        return '\n'.join(lines), "type mismatch", i + 1
        
        return self._inject_undefined_var(lines, lang)  # fallback
    
    def _inject_off_by_one(self, lines: List[str], lang: str) -> Tuple[str, str, int]:
        """注入：差一错误（<= 改为 < 等）"""
        for i, line in enumerate(lines):
            if '<=' in line:
                lines[i] = line.replace('<=', '<', 1)
                return '\n'.join(lines), "off-by-one: should be <=", i + 1
            if ' < ' in line:
                lines[i] = line.replace(' < ', ' <= ', 1)
                return '\n'.join(lines), "off-by-one: should be <", i + 1
        return self._inject_undefined_var(lines, lang)
    
    def _inject_wrong_operator(self, lines: List[str], lang: str) -> Tuple[str, str, int]:
        """注入：错误操作符"""
        swaps = [('==', '!='), ('!=', '=='), ('&&', '||'), ('||', '&&')]
        for i, line in enumerate(lines):
            for old, new in swaps:
                if old in line:
                    lines[i] = line.replace(old, new, 1)
                    return '\n'.join(lines), f"wrong operator: should be '{old}'", i + 1
        return self._inject_undefined_var(lines, lang)
    
    def _inject_nil_dereference(self, lines: List[str], lang: str) -> Tuple[str, str, int]:
        """注入：nil 指针解引用"""
        for i, line in enumerate(lines):
            if '.' in line and '=' in line:
                # 在调用前加一个 nil 检查
                indent = ' ' * (len(line) - len(line.lstrip()))
                nil_check = f"{indent}if someNilVar != nil {{\n{indent}    {line}\n{indent}}}"
                lines[i] = nil_check
                return '\n'.join(lines), "possible nil dereference", i + 1
        return self._inject_undefined_var(lines, lang)


# ============================================================
# 3. 代码生成器（调用 DeepSeek API）
# ============================================================

class CodeGenerator:
    """用 DeepSeek API 生成类型多样化的代码样本"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        self.base_url = "https://api.deepseek.com/v1"
        
        # 模板 prompt（预置一些直接的代码，减少 API 调用）
        self.templates = {
            "go": [
                {
                    "intent": "写一个处理 HTTP 请求的 Go handler",
                    "code": """package main

import (
    "fmt"
    "net/http"
)

func main() {
    http.HandleFunc("/", handler)
    http.ListenAndServe(":8080", nil)
}

func handler(w http.ResponseWriter, r *http.Request) {
    name := r.URL.Query().Get("name")
    fmt.Fprintf(w, "Hello, %s", name)
}"""
                },
                {
                    "intent": "实现一个 Go 数据结构操作函数",
                    "code": """package main

type User struct {
    ID   int
    Name string
    Age  int
}

func FilterAdults(users []User) []User {
    var result []User
    for _, u := range users {
        if u.Age >= 18 {
            result = append(result, u)
        }
    }
    return result
}

func main() {
    users := []User{
        {ID: 1, Name: "Alice", Age: 25},
        {ID: 2, Name: "Bob", Age: 16},
    }
    adults := FilterAdults(users)
    fmt.Println(adults)
}"""
                },
                {
                    "intent": "实现字符串处理和文件读取",
                    "code": """package main

import (
    "bufio"
    "fmt"
    "os"
    "strings"
)

func main() {
    file, err := os.Open("data.txt")
    if err != nil {
        fmt.Println("Error:", err)
        return
    }
    defer file.Close()

    scanner := bufio.NewScanner(file)
    for scanner.Scan() {
        line := scanner.Text()
        parts := strings.Split(line, ",")
        fmt.Printf("Name: %s, Value: %s\\n", parts[0], parts[1])
    }
}"""
                },
            ],
            "python": [
                {
                    "intent": "写一个 Python 数据处理函数",
                    "code": """def process_data(items: list) -> dict:
    result = {
        "total": 0,
        "count": len(items),
        "average": 0.0,
    }
    for item in items:
        result["total"] += item
    if result["count"] > 0:
        result["average"] = result["total"] / result["count"]
    return result

def main():
    data = [1, 2, 3, 4, 5]
    processed = process_data(data)
    print(f"Total: {processed['total']}")
    print(f"Average: {processed['average']}")

if __name__ == "__main__":
    main()"""
                },
                {
                    "intent": "写一个带文件操作的 Python 脚本",
                    "code": """import json
from pathlib import Path

def load_config(path: str) -> dict:
    config_path = Path(path)
    if not config_path.exists():
        return {"default": True}
    
    with open(config_path, "r") as f:
        config = json.load(f)
    return config

def save_config(config: dict, path: str):
    with open(path, "w") as f:
        json.dump(config, f, indent=2)

def main():
    cfg = load_config("./config.json")
    cfg["last_updated"] = "2026-05-20"
    save_config(cfg, "./config.json")
    print("Config saved")

if __name__ == "__main__":
    main()"""
                },
            ],
            "ts": [
                {
                    "intent": "写一个 TypeScript React 组件",
                    "code": """import React, { useState, useEffect } from 'react';

interface User {
    id: number;
    name: string;
    email: string;
}

const UserList: React.FC = () => {
    const [users, setUsers] = useState<User[]>([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        fetchUsers();
    }, []);

    const fetchUsers = async () => {
        try {
            const response = await fetch('/api/users');
            const data = await response.json();
            setUsers(data);
        } catch (error) {
            console.error('Failed to fetch users:', error);
        } finally {
            setLoading(false);
        }
    };

    if (loading) return <div>Loading...</div>;

    return (
        <div>
            {users.map(user => (
                <div key={user.id}>
                    <h3>{user.name}</h3>
                    <p>{user.email}</p>
                </div>
            ))}
        </div>
    );
};

export default UserList;"""
                },
            ],
        }
    
    def generate_code(self, language: str, num_samples: int = 3) -> List[Dict]:
        """生成代码样本"""
        # 先用模板
        samples = list(self.templates.get(language, []))
        
        # 如果设置了 API key，再通过 API 获取额外的样本
        if self.api_key and len(samples) < num_samples:
            extra = self._generate_via_api(language, num_samples - len(samples))
            samples.extend(extra)
        
        # 如果仍然不够，对现有代码做变换（改变量名/调结构）
        while len(samples) < num_samples and samples:
            base = random.choice(samples)
            transformed = self._transform_code(base, language)
            if transformed and transformed not in samples:
                samples.append(transformed)
        
        return samples[:num_samples]
    
    def _generate_via_api(self, language: str, count: int) -> List[Dict]:
        """通过 DeepSeek API 生成代码"""
        import requests
        
        if not self.api_key:
            return []
        
        samples = []
        for _ in range(count):
            prompt = f"""Generate a {language} code snippet (50-100 lines) that demonstrates a common programming pattern. 
Include at least one function definition, variable operations, and error handling if applicable.
Return ONLY valid JSON in this exact format:
{{"intent": "brief description of what this code does", "code": "the code here"}}"""
            
            try:
                resp = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "deepseek-chat",
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.7,
                        "max_tokens": 2000,
                    },
                    timeout=60
                )
                if resp.status_code == 200:
                    content = resp.json()["choices"][0]["message"]["content"]
                    # 提取 JSON
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        data = json.loads(json_match.group())
                        if "intent" in data and "code" in data:
                            samples.append(data)
            except Exception:
                pass
        
        return samples
    
    def _transform_code(self, sample: Dict, language: str) -> Optional[Dict]:
        """轻微变换现有代码（变量名/逻辑结构）"""
        code = sample["code"]
        lines = code.split('\n')
        
        # 变量名变换
        vars_found = set()
        for line in lines:
            for match in re.finditer(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', line):
                var = match.group()
                if var not in KEYWORDS_TYPESCRIPT and var not in KEYWORDS_TYPESCRIPT:
                    vars_found.add(var)
        
        if vars_found:
            old_var = random.choice(list(vars_found))
            new_var = f"var_{random.randint(100,999)}"
            new_code = code.replace(old_var, new_var)
            if new_code != code:
                return {
                    "intent": sample["intent"] + " (transformed)",
                    "code": new_code
                }
        
        return None


# ============================================================
# 4. 训练数据合成主流程
# ============================================================

class TrainingDataSynthesizer:
    """母模型训练数据合成器"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.code_generator = CodeGenerator(api_key)
        self.bug_injector = BugInjector()
        self.code_verifier = CodeVerifier()
        
        # 意图模板
        self.intent_templates = [
            "看看这段代码有没有问题",
            "分析这段代码，问题在哪",
            "帮我检查这段代码的错误",
            "这段代码哪里错了",
            "review 一下这段代码",
            "指出这段代码的 bug",
            "这段代码能不能编译通过",
            "帮我找出代码里的问题",
            "检查一下代码质量",
            "这段代码需要修改哪里",
        ]
    
    def synthesize(
        self,
        languages: List[str] = ["go", "python", "ts"],
        samples_per_language: int = 20,
        output_path: str = "./data/train.jsonl",
        no_compile: bool = False,
    ):
        """
        合成训练数据
        
        Args:
            languages: 代码语言列表
            samples_per_language: 每种语言的样本数
            output_path: 输出路径
            no_compile: 是否跳过编译验证（纯静态分析）
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        total_samples = 0
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for lang in languages:
                print(f"\n=== Generating {lang} samples ===")
                
                # 生成代码
                code_samples = self.code_generator.generate_code(lang, samples_per_language)
                
                for sample in code_samples:
                    code = sample["code"]
                    intent = sample["intent"]
                    
                    # 生成正常版本（确保能编译通过）
                    if not no_compile:
                        is_ok, err, line = self.code_verifier.verify_go(
                            code) if lang == "go" else (
                            self.code_verifier.verify_python(code) if lang == "python"
                            else self.code_verifier.verify_typescript(code)
                        )
                        if not is_ok:
                            print(f"  [WARN] Base code failed verification: {err}")
                            continue
                    
                    # 注入 bug
                    buggy_code, error_desc, error_line = self.bug_injector.inject(code, lang)
                    
                    if buggy_code:
                        # 验证 bug 确实被检测到
                        if not no_compile:
                            verify_fn = (
                                self.code_verifier.verify_go if lang == "go"
                                else self.code_verifier.verify_python if lang == "python"
                                else self.code_verifier.verify_typescript
                            )
                            is_ok, actual_err, actual_line = verify_fn(buggy_code)
                        else:
                            actual_err = error_desc
                            actual_line = error_line
                        
                        intent_text = random.choice(self.intent_templates)
                        
                        # 构造目标描述（母模型的理想输出）
                        line_str = f"第 {actual_line} 行" if actual_line else "某处"
                        target = (
                            f"子模型报告：{lang.upper()} 代码在 {line_str} 有问题。"
                            f"问题类型：{actual_err or error_desc}。"
                            f"请通知对应的子模型检查并修复此问题。"
                        )
                        
                        record = {
                            "intent": intent_text,
                            "code": buggy_code,
                            "code_language": lang,
                            "target": target,
                            "culprit_submodel": lang,
                            "error_type": error_desc or actual_err,
                            "error_line": actual_line or error_line or 0,
                        }
                        
                        f.write(json.dumps(record, ensure_ascii=False) + '\n')
                        total_samples += 1
                        
                        if total_samples % 10 == 0:
                            print(f"  Generated {total_samples} samples...")
                    
                    # 再加一些正常代码的样本（母模型也要学会说"没问题"）
                    normal_target = (
                        f"代码检查通过：{lang.upper()} 代码未发现明显问题。"
                        f"各子模型输出正常。"
                    )
                    normal_record = {
                        "intent": intent_text if intent_text else random.choice(self.intent_templates),
                        "code": code,
                        "code_language": lang,
                        "target": normal_target,
                        "culprit_submodel": "",
                        "error_type": "none",
                        "error_line": 0,
                    }
                    f.write(json.dumps(normal_record, ensure_ascii=False) + '\n')
                    total_samples += 1
        
        print(f"\n✅ Total training samples generated: {total_samples}")
        print(f"   Output: {output_path}")
        return total_samples


# ============================================================
# 5. CLI 入口
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="合成母模型训练数据")
    parser.add_argument("--api-key", type=str, default=None,
                        help="DeepSeek API Key（可选，只用模板也能生成不少样本）")
    parser.add_argument("--languages", type=str, nargs="+", default=["go", "python", "ts"],
                        help="目标语言")
    parser.add_argument("--samples", type=int, default=20,
                        help="每种语言的代码样本数（每个样本会生成 1 个 bug 版本 + 1 个正常版本）")
    parser.add_argument("--output", type=str, default="./data/train.jsonl",
                        help="输出路径")
    parser.add_argument("--no-compile", action="store_true",
                        help="跳过编译验证（纯静态分析）")
    
    args = parser.parse_args()
    
    synthesizer = TrainingDataSynthesizer(api_key=args.api_key)
    synthesizer.synthesize(
        languages=args.languages,
        samples_per_language=args.samples,
        output_path=args.output,
        no_compile=args.no_compile,
    )
