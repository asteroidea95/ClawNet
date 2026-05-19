#!/usr/bin/env python3
"""ClawNet 编程MVP — 入口

演示"母模型(产品经理) + N个子模型(编程专家)"架构。

支持的请求示例:
  python3 mvp.py --request "用 Go 写一个带 JWT 的登录 API，数据库用 Postgres"
  python3 mvp.py --request "写一个 React 登录页面调后端 API"
  python3 mvp.py --request "写一个完整的用户登录功能，前后端+数据库"

模式:
  CLAWNET_MODE=simulation  使用内置模拟响应 (默认)
  CLAWNET_MODE=api         调用 DeepSeek API (需设置 DEEPSEEK_API_KEY)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mom import MomModel

DEMO_REQUESTS = [
    "写一个完整的用户登录功能：Go 后端 JWT 认证，React 前端登录表单，Postgres 存储用户信息",
    "用 Go 写一个带中间件的 HTTP API 服务器",
    "创建一个 React 组件展示用户列表，调后端 API 获取数据",
]

def main():
    import argparse

    parser = argparse.ArgumentParser(description="ClawNet 编程MVP")
    parser.add_argument("--request", "-r", type=str, help="编程需求")
    parser.add_argument("--demo", "-d", type=int, default=None, help=f"运行示例需求 (0-{len(DEMO_REQUESTS)-1})")
    parser.add_argument("--list-demos", action="store_true", help="列出所有示例需求")

    args = parser.parse_args()

    if args.list_demos:
        print("示例需求:")
        for i, req in enumerate(DEMO_REQUESTS):
            print(f"  {i}: {req}")
        return

    request = args.request
    if args.demo is not None:
        if 0 <= args.demo < len(DEMO_REQUESTS):
            request = DEMO_REQUESTS[args.demo]
        else:
            print(f"无效的示例编号。可用: 0-{len(DEMO_REQUESTS)-1}")
            return

    if not request:
        print("用法: python3 mvp.py --request \"你的编程需求\"")
        print("或:   python3 mvp.py --demo 0")
        print("或:   python3 mvp.py --list-demos")
        return

    mode = os.getenv("CLAWNET_MODE", "simulation")
    print(f"ClawNet 编程MVP — 模式: {mode}")
    if mode == "simulation":
        print("(仿真模式: 使用内置模拟响应，无需 API 密钥)")
    print()

    mom = MomModel()
    mom.run(request)

if __name__ == "__main__":
    main()
