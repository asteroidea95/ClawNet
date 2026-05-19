"""ClawNet 编程MVP — 配置

运行模式:
  simulation  使用内置的模拟响应 (无需API密钥)
  api         使用 DeepSeek API (需设置 DEEPSEEK_API_KEY)
"""
import os

MODE = os.getenv("CLAWNET_MODE", "simulation")  # simulation | api
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
API_BASE = "https://api.deepseek.com/v1/chat/completions"
