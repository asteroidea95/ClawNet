// CRDT 令牌台账

use crate::crypto::{self, NodeId};
use ed25519_dalek::{SigningKey, Signature, VerifyingKey};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// 令牌交易记录
#[derive(Clone, Serialize, Deserialize)]
pub struct TokenTransaction {
    /// 支付方节点 ID
    pub from: NodeId,
    /// 接收方节点 ID
    pub to: NodeId,
    /// 令牌数量
    pub amount: u32,
    /// 支付方签名
    pub signature: Vec<u8>,
    /// 时间戳
    pub timestamp: u64,
}

/// CRDT 令牌台账
///
/// 使用 G-Counter (Grow-only Counter) 实现。
/// 每个节点独立维护，(from, to) 对使用单调递增计数器。
/// 合并时取最大值，无冲突。
#[derive(Clone)]
pub struct TokenLedger {
    /// 本地节点ID
    node_id: NodeId,
    /// 令牌余额: (A, B) => A 欠 B 多少令牌
    /// 使用 GCounter 语义：只增不减
    pub balances: HashMap<(NodeId, NodeId), u32>,
    /// 已处理的交易哈希（防止重复）
    pub seen_txs: HashMap<Vec<u8>, bool>,
}

impl TokenLedger {
    pub fn new(node_id: NodeId) -> Self {
        TokenLedger {
            node_id,
            balances: HashMap::new(),
            seen_txs: HashMap::new(),
        }
    }

    /// 结算一笔交易
    /// from: 支付方（获得服务的一方）
    /// to: 接收方（提供算力的一方）
    /// amount: 支付的令牌数
    /// signature: from 的签名
    pub fn settle(
        &mut self,
        from: &NodeId,
        to: &NodeId,
        amount: u32,
        signature: &[u8],
    ) -> Result<(), String> {
        // GCounter: 增加计数器
        let key = (from.clone(), to.clone());
        let entry = self.balances.entry(key).or_insert(0);
        *entry += amount;

        Ok(())
    }

    /// 查询节点 A 欠节点 B 的令牌数
    pub fn get_balance(&self, from: &NodeId, to: &NodeId) -> u32 {
        self.balances.get(&(from.clone(), to.clone())).copied().unwrap_or(0)
    }

    /// 查询节点的净令牌余额（别人欠它的 - 它欠别人的）
    pub fn net_balance(&self, node: &NodeId) -> i64 {
        let mut credit = 0i64;
        let mut debit = 0i64;

        for ((from, to), amount) in &self.balances {
            if to == node {
                credit += *amount as i64;
            }
            if from == node {
                debit += *amount as i64;
            }
        }

        credit - debit
    }

    /// 合并另一个节点的台账副本（CRDT 合并）
    pub fn merge(&mut self, other: &TokenLedger) {
        for (key, amount) in &other.balances {
            let entry = self.balances.entry(key.clone()).or_insert(0);
            // G-Counter: 取最大值
            if *amount > *entry {
                *entry = *amount;
            }
        }
    }
}

/// 计算调度优先级
/// 优先级 = 净令牌余额 × 0.7 + 在线时长(小时) × 0.3
pub fn calculate_priority(ledger: &TokenLedger, node: &NodeId, uptime_hours: f64) -> f64 {
    let balance = ledger.net_balance(node) as f64;
    balance.max(0.0) * 0.7 + uptime_hours * 0.3
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::OsRng;

    #[test]
    fn test_basic_transaction() {
        let mut csprng = OsRng;
        let keypair_a = SigningKey::generate(&mut csprng);
        let keypair_b = SigningKey::generate(&mut csprng);
        let id_a = NodeId::from(&keypair_a);
        let id_b = NodeId::from(&keypair_b);

        let mut ledger = TokenLedger::new(id_a.clone());

        ledger.settle(&id_a, &id_b, 5, &[]).unwrap();
        assert_eq!(ledger.get_balance(&id_a, &id_b), 5);
        assert_eq!(ledger.net_balance(&id_a), -5);
        assert_eq!(ledger.net_balance(&id_b), 5);
    }

    #[test]
    fn test_crdt_merge() {
        let mut csprng = OsRng;
        let kp_a = SigningKey::generate(&mut csprng);
        let kp_b = SigningKey::generate(&mut csprng);
        let kp_c = SigningKey::generate(&mut csprng);
        let id_a = NodeId::from(&kp_a);
        let id_b = NodeId::from(&kp_b);
        let id_c = NodeId::from(&kp_c);

        // 节点 A 的台账
        let mut ledger_a = TokenLedger::new(id_a.clone());
        ledger_a.settle(&id_a, &id_b, 3, &[]).unwrap();
        ledger_a.settle(&id_c, &id_a, 1, &[]).unwrap();

        // 节点 B 的台账
        let mut ledger_b = TokenLedger::new(id_b.clone());
        ledger_b.settle(&id_a, &id_b, 5, &[]).unwrap(); // B 多了一笔

        // 合并 A 到 B
        ledger_b.merge(&ledger_a);

        // 取最大值：A欠B 5
        assert_eq!(ledger_b.get_balance(&id_a, &id_b), 5);
        // C欠A 1
        assert_eq!(ledger_b.get_balance(&id_c, &id_a), 1);
    }

    #[test]
    fn test_priority() {
        let mut ledger = TokenLedger::new(NodeId([0u8; 32]));
        let id_a = NodeId([1u8; 32]);
        let id_b = NodeId([2u8; 32]);

        // A 欠 B 5 个令牌 → B 信用 -5
        ledger.settle(&id_a, &id_b, 5, &[]).unwrap();

        let priority_a = calculate_priority(&ledger, &id_a, 10.0);
        let priority_b = calculate_priority(&ledger, &id_b, 1.0);

        // B 贡献多，优先级应该更高
        assert!(priority_b > priority_a);
    }
}
