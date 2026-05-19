# 贡献指南

欢迎！ClawNet 是一个民间项目，任何形式的贡献都受欢迎。

## 快速开始

```bash
cd ClawNet
cargo build
cargo test
cargo run -- start
```

## 贡献方式

### 🐛 报告 Bug

开 Issue，尽量包含：
- 复现步骤
- 预期行为 vs 实际行为
- 环境信息（OS、Rust版本）

### 💡 提想法

开 Issue 讨论新场景、新 worker 类型、协议改进。

### 🔧 提交代码

1. Fork 仓库
2. 创建 feature branch: `git checkout -b feat/your-feature`
3. 提交代码
4. 跑通测试: `cargo test`
5. 开 PR

### 📝 写文档

文档和代码一样重要。改 README、写教程、加注释都欢迎。

## 代码规范

- 用 `cargo fmt` 格式化
- 用 `cargo clippy` 检查
- 新功能要有测试

## 许可证

MIT — 提交即同意以 MIT 许可证发布。
