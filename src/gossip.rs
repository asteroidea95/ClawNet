// Gossip 引擎 — 节点发现 + 状态传播

use crate::crypto::NodeId;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::mpsc;

/// 消息类型
#[derive(Clone, Serialize, Deserialize)]
pub enum GossipMessage {
    /// Ping 探测
    Ping { seq: u64 },
    /// Ping 响应
    Ack { seq: u64 },
    /// 间接探测请求
    PingReq { target: NodeId, seq: u64 },
    /// 间接探测响应
    AckIndirect { target: NodeId, seq: u64 },
    /// 节点公告
    Announce { node: NodeInfo },
    /// 令牌交易广播
    TransactionBroadcast { data: Vec<u8> },
    /// 任务公告
    TaskAnnouncement { data: Vec<u8> },
}

/// 节点信息
#[derive(Clone, Serialize, Deserialize)]
pub struct NodeInfo {
    pub node_id: NodeId,
    pub host: String,
    pub port: u16,
    pub uptime_hours: f64,
    pub token_balance_approx: u32,
    pub capabilities: Vec<String>,
}

/// 故障检测状态
#[derive(Clone, PartialEq)]
pub enum FailureState {
    Alive,
    Suspect,
    Dead,
}

/// 成员列表中的节点记录
#[derive(Clone)]
pub struct MemberEntry {
    pub info: NodeInfo,
    pub state: FailureState,
    pub last_seen: u64,
    pub incarnation: u64,
}

/// Gossip 引擎
pub struct Engine {
    node_id: NodeId,
    members: Arc<tokio::sync::RwLock<Vec<MemberEntry>>>,
    tx: mpsc::Sender<GossipMessage>,
    rx: mpsc::Receiver<GossipMessage>,
}

impl Engine {
    /// 创建并启动 gossip 引擎
    pub async fn new(
        node_id: NodeId,
        seed_nodes: &[String],
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let (tx, rx) = mpsc::channel(1024);

        let engine = Engine {
            node_id,
            members: Arc::new(tokio::sync::RwLock::new(Vec::new())),
            tx,
            rx,
        };

        // 连接种子节点
        for seed in seed_nodes {
            println!("  → 连接种子节点: {}", seed);
        }

        Ok(engine)
    }

    /// 广播消息到全网
    pub async fn broadcast(&self, msg: GossipMessage) {
        self.tx.send(msg).await.ok();
    }

    /// 接收消息
    pub async fn recv(&mut self) -> Option<GossipMessage> {
        self.rx.recv().await
    }
}

impl Clone for Engine {
    fn clone(&self) -> Self {
        // 注意：实际实现中需要共享 Arc
        unimplemented!("gossip engine clone not yet implemented")
    }
}
