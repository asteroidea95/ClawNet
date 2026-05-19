"""ClawNet 编程MVP — 母模型 (代码产品经理)

职责:
  1. 拆需求 → 确定需要哪些子模型
  2. 排顺序 → 确定子模型的执行顺序
  3. 派任务 → 依次调用子模型
  4. 验证 → 检查编译/输出
  5. 回炉 → 发现问题定位到对应子模型
  6. 交付 → 输出最终结果
"""

from typing import Optional
from child_models import ChildModel, detect_child_models, REGISTRY
from model import call_model, MODE

class MomModel:
    """母模型 = 只会看代码的产品经理"""

    def __init__(self):
        self.artifacts: dict[str, str] = {}  # 子模型产出缓存
        self.build_log: list[str] = []        # 构建日志

    def decompose(self, request: str) -> list[ChildModel]:
        """拆需求 → 返回有序的子模型列表"""
        print(f"\n{'='*60}")
        print(f"📋 用户需求: {request}")
        print(f"{'='*60}\n")

        models = detect_child_models(request)

        # 确定执行顺序 (有依赖关系的)
        order_map = {"sql": 0, "go": 1, "react": 2, "general": 0}
        models.sort(key=lambda m: order_map.get(m.language.lower(), 0))

        print(f"🔍 母模型拆解: 需要 {len(models)} 个子模型")
        for i, m in enumerate(models):
            print(f"   {i+1}. {m.name} ({m.description})")
        print()

        return models

    def dispatch(self, child: ChildModel, task: str) -> str:
        """派任务给子模型并验证产出"""
        print(f"  ┌─▶ 派给 {child.name}")
        print(f"  │   任务: {task[:80]}...")

        output = call_model(child, task)
        self.artifacts[child.language] = output

        lines = output.strip().split('\n')
        print(f"  │   产出: {len(lines)} 行, {len(output)} 字节")
        print(f"  └── 完成\n")

        return output

    def validate_compile(self, code: str, language: str) -> tuple[bool, str]:
        """验证代码可编译性 (仿真: 语法检查 + 关键字校验)"""
        if language == "Go":
            # 仿真校验
            checks = [
                ("package", code.startswith("package")),
                ("func", "func " in code),
                ("import", "import" in code),
            ]
            failures = [c[0] for c in checks if not c[1]]
            if failures:
                return False, f"缺少: {', '.join(failures)}"
            return True, "通过"

        if language == "SQL":
            checks = [
                ("CREATE", "CREATE" in code.upper()),
                (";", ";" in code),
            ]
            failures = [c[0] for c in checks if not c[1]]
            if failures:
                return False, f"缺少: {', '.join(failures)}"
            return True, "通过"

        if "TypeScript" in language:
            checks = [
                ("import React", "import React" in code or "import React" in code),
                ("export", "export " in code),
                ("=>", "=>" in code or "function" in code),
            ]
            failures = [c[0] for c in checks if not c[1]]
            if failures:
                return False, f"缺少: {', '.join(failures)}"
            return True, "通过"

        return True, "通过(无需验证)"

    def validate_integration(self) -> tuple[bool, list[str]]:
        """验证子模型产出之间的对接一致性"""
        issues = []

        # 检查 API 接口对接
        if "Go" in self.artifacts and "TypeScript/JSX" in self.artifacts:
            go_code = self.artifacts["Go"]
            react_code = self.artifacts["TypeScript/JSX"]

            # Go handler 的名字
            import re
            go_handlers = re.findall(r'func (\w+Handler)', go_code)
            # React fetch 的 endpoint
            react_endpoints = re.findall(r"fetch\('([^']+)'", react_code)

            # 检查 React 调用的 endpoint 是否对应 Go handler
            if react_endpoints and go_handlers:
                for endpoint in react_endpoints:
                    # /api/login → LoginHandler?
                    path = endpoint.rstrip('/').split('/')[-1]
                    handler_match = any(path.capitalize() in h for h in go_handlers)
                    if not handler_match:
                        issues.append(f"对接警告: React 调 '{endpoint}', 但 Go 上未找到对应 handler")

        return (len(issues) == 0, issues)

    def run(self, request: str) -> dict:
        """完整流程：拆 → 派 → 验 → 回炉 → 交付"""
        self.artifacts = {}
        self.build_log = []

        # Step 1: 拆需求
        children = self.decompose(request)

        if not children:
            return {"status": "error", "message": "没有合适的子模型"}

        max_retries = 2

        # Step 2-5: 顺序推理
        for child in children:
            for attempt in range(max_retries + 1):
                # 派任务
                task = self._build_task(child, request)
                output = self.dispatch(child, task)

                # 编译验证
                ok, msg = self.validate_compile(output, child.language)
                self.build_log.append(f"[{child.language}] 第{attempt+1}次: {msg}")

                if ok:
                    print(f"  ✅ {child.name} → 编译验证通过\n")
                    break
                else:
                    print(f"  ❌ {child.name} → {msg} (重试 {attempt+1}/{max_retries})\n")
                    if attempt == max_retries:
                        print(f"  🚫 {child.name} → 达到最大重试次数\n")

        # 对接验证
        integration_ok, issues = self.validate_integration()
        if not integration_ok:
            for issue in issues:
                print(f"  ⚠  {issue}")
                self.build_log.append(f"[对接] {issue}")

        # 交付
        result = {
            "status": "success",
            "request": request,
            "artifacts": self.artifacts,
            "build_log": self.build_log,
            "validation": {
                "integration": "通过" if integration_ok else "有警告",
                "child_count": len(children),
            },
        }

        self._print_delivery(result)
        return result

    def _build_task(self, child: ChildModel, request: str) -> str:
        """为子模型构建具体的任务指令"""
        if child.language == "SQL":
            return f"请根据以下需求创建数据库表结构:\n\n{request}"
        if child.language == "Go":
            # 如果有之前的 SQL 产出，传给 Go 子模型参考
            extra = ""
            if "SQL" in self.artifacts:
                extra = f"\n参考数据库表结构:\n{self.artifacts['SQL'][:300]}"
            return f"请根据以下需求编写 Go 代码:\n\n{request}{extra}"
        if "TypeScript" in child.language:
            extra = ""
            if "Go" in self.artifacts:
                extra = f"\n参考后端 API:\n{self.artifacts['Go'][:300]}"
            return f"请根据以下需求编写 React 组件:\n\n{request}{extra}"
        return request

    def _print_delivery(self, result: dict):
        print(f"\n{'='*60}")
        print(f"📦 交付结果")
        print(f"{'='*60}")
        print(f"状态: {result['status']}")
        print(f"涉及子模型: {result['validation']['child_count']}")
        print(f"对接验证: {result['validation']['integration']}")
        for lang, code in result['artifacts'].items():
            lines = code.strip().split('\n')
            print(f"\n── {lang} ({len(lines)} 行) ──")
            print(code[:200] + ("..." if len(code) > 200 else ""))
