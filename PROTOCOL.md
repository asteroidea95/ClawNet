# ClawNet 协议规范 v0.1

> 草稿阶段，会随开发迭代。

---

## 1. 节点发现协议

基于 SWIM 的 gossip 成员管理。

### 1.1 节点标识

```proto
message NodeInfo {
    bytes  node_id = 1;          // Ed25519 公钥的哈希
    string host = 2;             // 公网 IP 或局域网地址
    uint32 port = 3;             // 监听端口
    double uptime_hours = 4;     // 累计在线时长
    uint32 token_balance = 5;    // 令牌余额（公告用近似值）
    repeated string caps = 6;    // 能力标签 ["dom", "gpu", "video-gen", ...]
}
```

### 1.2 成员协议

```
PING(seq)                     → ACK(seq)
PING-REQ(target, seq)         → ACK(seq) 或 NACK
INDIRECT-PING(target, seq)    → 由邻居转发探测
ANNOUNCE(node_info)           → 新节点 / 状态变更广播
```

### 1.3 故障检测

- 每个节点维持一个**怀疑列表**
- 节点在 K 轮 gossip 周期内无响应 → 标记为 `SUSPECT`
- 向随机邻居发起 `PING-REQ` 间接探测
- 间接探测失败 → 标记 `DEAD` → 全网广播

---

## 2. CRDT 令牌台账

使用 **G-Counter** (Grow-only Counter) 和 **AWORSet** (Add-Wins Observed-Removed Set) 的组合：

### 2.1 数据结构

```rust
struct TokenEntry {
    from: NodeId,          // 支付方
    to: NodeId,            // 接收方
    amount: u32,           // 令牌数
    signature: [u8; 64],   // 支付方的 Ed25519 签名
    timestamp: u64,        // 本地时间戳
    seq: u64,              // 单调递增序列号
}

// 每个节点维护：
type TokenLedger = Map<(NodeId, NodeId), GCounter>
// 合并规则：取各副本对应计数器的最大值
```

### 2.2 交易流程

```
1. 贡献者完成任务后，向发起者发送结果
2. 发起者验证结果 → 生成 TokenEntry
3. 发起者签名 TokenEntry → 发送给贡献者
4. 双方各自添加到本地台账
5. 台账随 gossip 同步到其他节点
```

### 2.3 冲突解决

CRDT 保证无冲突合并：
- G-Counter：两个副本合并时取每个元素的最大值
- AWORSet：元素同时被添加和删除时，添加胜出

---

## 3. 任务公告协议

### 3.1 任务公告

```rust
struct TaskAnnouncement {
    task_id: [u8; 32],        // 哈希
    task_type: TaskType,      // dom | image-gen | video-gen | inference
    description: String,      // 人类可读描述
    requirements: Requirements, // 硬件/能力要求
    reward_tokens: u32,       // 支付的令牌数
    deadline: u64,            // 截止时间戳
    initiator: NodeId,        // 发起者
    signature: [u8; 64],      // 发起者签名
}
```

### 3.2 任务认领

```rust
struct TaskClaim {
    task_id: [u8; 32],
    claimer: NodeId,
    capability_proof: String,  // 能力证明
}
```

### 3.3 状态机

```
CREATED → ASSIGNED → IN_PROGRESS → COMPLETED
    |          |            |
    v          v            v
EXPIRED   EXPIRED      FAILED
```

---

## 4. 传输层

### 4.1 小消息（gossip 通道）

大小 < 64KB，直接嵌入 gossip 消息体：

- 节点心跳
- 任务公告
- 令牌交易
- 任务认领

### 4.2 大文件（直连通道）

大小 > 64KB，通过直连传输：

- 视频帧序列
- 模型分片
- 批量搜索结果

直连协议：支持 libp2p 或 原生 TCP + STUN/TURN。

---

## 5. 能力标签

节点启动时声明自己的硬件能力：

| 标签 | 含义 | 要求 |
|---|---|---|
| `dom` | 浏览器DOM操作 | 有 Chromium/Firefox + Playwright |
| `webgpu` | WebGPU 推理 | 支持 WebGPU 的浏览器 |
| `coreml` | Apple Neural Engine | M系列芯片/A17+ |
| `cuda` | NVIDIA GPU 推理 | CUDA 可用 |
| `vulkan` | 跨平台 GPU 推理 | Vulkan 1.2+ |

---

## 6. 安全模型

### 6.1 通信安全

- 所有节点间通信使用 Ed25519 签名
- 可选 TLS 加密传输

### 6.2 Sybil 攻击防御

ClawNet 不预防 Sybil 攻击。因为令牌系统的经济模型使得**攻击的成本高于收益**：
- 新节点令牌余额为 0，优先级最低
- 大量创建假节点需要大量贡献真实算力才能获得令牌
- 恶意行为被举报后，该节点 ID 的信任积分清零

### 6.3 结果可用性

- 任务发起者同时将任务发给多个节点 → 冗余执行
- 至少取前 K 个返回结果
- 避免单点故障导致的等待
