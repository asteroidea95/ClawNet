#!/usr/bin/env bash
# ClawNet — 双节点局域网演示
# 在一个终端跑 A，另一个终端跑 B，观察任务分发 + 令牌记账

set -e

echo "============================================"
echo " ClawNet 双节点演示"
echo "============================================"
echo ""
echo "这个演示需要两个终端窗口。"
echo ""
echo "终端 1（节点 A）:"
echo "  cd ClawNet && cargo run -- start --port 9876"
echo ""
echo "终端 2（节点 B）:"
echo "  cd ClawNet && cargo run -- start --port 9877 --seeds 127.0.0.1:9876"
echo ""
echo "预期行为："
echo "  1. B 通过 seed 发现 A，交换节点信息"
echo "  2. 两台节点互相知道对方在线"
echo "  3. 在一台节点提交任务，另一台认领、执行、回传"
echo "  4. 令牌台账通过 gossip 同步，两边一致"
echo "============================================"
