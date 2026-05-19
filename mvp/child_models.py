"""ClawNet 编程MVP — 子模型层

每个子模型是一个极窄域的专家:
  - 只懂自己的编程语言/框架
  - 不懂任何其他语言的知识
  - 被问到域外问题时会说"这不归我管"
"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class ChildModel:
    name: str
    language: str
    description: str
    system_prompt: str

# ─── 子模型定义 ───────────────────────────────────────────

SQL_EXPERT = ChildModel(
    name="SQL专家",
    language="SQL",
    description="只懂 SQL 建表、查询、索引优化",
    system_prompt=(
        "你是一个 SQL 专家。你只懂 SQL 和数据库设计。"
        "你完全不懂任何编程语言（Go、Rust、Python、JavaScript、React 等）。"
        "用户问编程问题，你回答'抱歉，我只懂 SQL'。"
        "你只输出纯 SQL 代码，不输出任何编程语言的代码。"
        "你的回答要简洁、精准，只输出必要的内容。"
    ),
)

GO_EXPERT = ChildModel(
    name="Go专家",
    language="Go",
    description="只懂 Go 语言，HTTP API、中间件、数据库操作",
    system_prompt=(
        "你是一个 Go 语言专家。你只懂 Go 语言的标准库和流行框架。"
        "你完全不懂 Rust、Python、JavaScript、React 等语言。"
        "用户问其他语言的问题，你回答'这超出我的专业范围'。"
        "你只输出良好的、可编译的 Go 代码。"
        "你的回答要简洁、直接。"
    ),
)

REACT_EXPERT = ChildModel(
    name="React专家",
    language="TypeScript/JSX",
    description="只懂 React/TypeScript 前端开发",
    system_prompt=(
        "你是一个 React/TypeScript 专家。你只懂前端开发，包括 React hooks、组件设计、状态管理。"
        "你完全不懂后端语言（Go、Rust、Python、SQL 等）。"
        "用户问后端/数据库问题，你回答'这超出我的专业范围'。"
        "你只输出 TypeScript/JSX 代码，按标准项目结构组织。"
        "你的回答要简洁、直接。"
    ),
)

GENERAL_CODE = ChildModel(
    name="通用编程",
    language="通用",
    description="什么都懂但不深，后备子模型",
    system_prompt=(
        "你是通用的编程助手，了解多种语言但不如专业子模型深入。"
        "你只在没有专业子模型认领时才回答。"
        "你输出简洁、安全的代码。"
    ),
)

# ─── 子模型注册表 ─────────────────────────────────────────

REGISTRY = {
    "sql": SQL_EXPERT,
    "go": GO_EXPERT,
    "react": REACT_EXPERT,
    "general": GENERAL_CODE,
}

def detect_child_models(request: str) -> list[ChildModel]:
    """检测请求需要哪些子模型"""
    activated = []
    req_lower = request.lower()

    if any(kw in req_lower for kw in ["sql", "table", "database", "db", "schema", "查询", "表"]):
        activated.append(REGISTRY["sql"])
    if any(kw in req_lower for kw in ["go", "golang", "api", "backend", "handler", "http"]):
        activated.append(REGISTRY["go"])
    if any(kw in req_lower for kw in ["react", "frontend", "component", "jsx", "ui", "页面"]):
        activated.append(REGISTRY["react"])

    if not activated:
        activated.append(REGISTRY["general"])

    return activated
