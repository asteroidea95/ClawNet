// 传输层 — 节点间通信

use crate::crypto::NodeId;
use serde::{Deserialize, Serialize};
use tokio::net::TcpListener;

/// 传输层
pub struct Transport {
    listener: TcpListener,
}

impl Transport {
    pub async fn new(addr: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let listener = TcpListener::bind(addr).await?;
        println!("  → 监听: {}", addr);
        Ok(Transport { listener })
    }
}
