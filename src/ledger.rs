// ClawNet — 扩展台账：令牌 + 观测记录 + 资产所有权
//
// 三个层次：
//   1. TokenLedger — 令牌收支（原版，保留）
//   2. ObservationChain — 观测记录链（证据锚定）
//   3. AssetRegistry — 数字资产所有权追踪（.soul）

use crate::crypto::{self, NodeId};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================
// 1. 令牌台账（原版保留）
// ============================================================

#[derive(Clone, Serialize, Deserialize)]
pub struct TokenEntry {
    pub from: NodeId,
    pub to: NodeId,
    pub amount: u32,
    pub signature: Vec<u8>,
    pub timestamp: u64,
}

pub type TokenBalanceMap = HashMap<(NodeId, NodeId), u32>;

#[derive(Clone)]
pub struct TokenLedger {
    node_id: NodeId,
    pub balances: TokenBalanceMap,
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

    pub fn settle(&mut self, from: &NodeId, to: &NodeId, amount: u32, _signature: &[u8]) -> Result<(), String> {
        let key = (from.clone(), to.clone());
        let entry = self.balances.entry(key).or_insert(0);
        *entry += amount;
        Ok(())
    }

    pub fn get_balance(&self, from: &NodeId, to: &NodeId) -> u32 {
        self.balances.get(&(from.clone(), to.clone())).copied().unwrap_or(0)
    }

    pub fn net_balance(&self, node: &NodeId) -> i64 {
        let mut credit = 0i64;
        let mut debit = 0i64;
        for ((from, to), amount) in &self.balances {
            if to == node { credit += *amount as i64; }
            if from == node { debit += *amount as i64; }
        }
        credit - debit
    }

    pub fn merge(&mut self, other: &TokenLedger) {
        for (key, amount) in &other.balances {
            let entry = self.balances.entry(key.clone()).or_insert(0);
            if *amount > *entry { *entry = *amount; }
        }
    }
}

// ============================================================
// 2. 观测记录链（证据锚定 — PROTOCOL.md §7）
// ============================================================

/// 一次观测事件的记录
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct Observation {
    /// 观测节点 ID
    pub node_id: NodeId,
    /// 任务类型
    pub task_type: String,
    /// 任务输入的哈希
    pub task_hash: [u8; 32],
    /// 观测结果（内容）的哈希
    pub data_hash: [u8; 32],
    /// 来源引用（URL、页面标题等）
    pub source_ref: Option<String>,
    /// 观测发生的时间戳（毫秒）
    pub timestamp: u64,
    /// 本节点上一条观测的哈希（形成个人观测链）
    pub prev_observation: [u8; 32],
    /// 本节点对这条记录的签名
    pub signature: Vec<u8>,
}

/// 观测链 — 每个节点独立维护
#[derive(Clone)]
pub struct ObservationChain {
    node_id: NodeId,
    /// 本节点的观测记录（按时间排序）
    pub local_observations: Vec<Observation>,
    /// 从 gossip 收到的其他节点观测
    pub peer_observations: HashMap<NodeId, Vec<Observation>>,
    /// 上一条观测的哈希（链式结构）
    last_hash: [u8; 32],
}

impl ObservationChain {
    pub fn new(node_id: NodeId) -> Self {
        ObservationChain {
            node_id,
            local_observations: Vec::new(),
            peer_observations: HashMap::new(),
            last_hash: [0u8; 32],
        }
    }

    /// 追加一条观测记录
    pub fn append(&mut self, obs: Observation) {
        self.local_observations.push(obs);
    }

    /// 从 gossip 同步另一节点的观测
    pub fn ingest(&mut self, from_node: &NodeId, obss: Vec<Observation>) {
        self.peer_observations.entry(from_node.clone()).or_insert_with(Vec::new).extend(obss);
    }

    /// 查询某个时间点是否有节点观测过某个哈希值
    pub fn query(&self, data_hash: &[u8; 32], after_timestamp: u64) -> Vec<&Observation> {
        let mut results = Vec::new();

        for obs in &self.local_observations {
            if &obs.data_hash == data_hash && obs.timestamp >= after_timestamp {
                results.push(obs);
            }
        }

        for (_, obss) in &self.peer_observations {
            for obs in obss {
                if &obs.data_hash == data_hash && obs.timestamp >= after_timestamp {
                    results.push(obs);
                }
            }
        }

        results
    }

    /// 负存在证明：查询某个时间点是否有任何观测记录
    pub fn has_any_observation(&self, before_timestamp: u64) -> bool {
        for obs in &self.local_observations {
            if obs.timestamp <= before_timestamp {
                return true;
            }
        }
        for (_, obss) in &self.peer_observations {
            for obs in obss {
                if obs.timestamp <= before_timestamp {
                    return true;
                }
            }
        }
        false
    }
}

// ============================================================
// 3. 数字资产所有权追踪（.soul）
// ============================================================

/// 数据资产的唯一标识
#[derive(Clone, Serialize, Deserialize, Debug, PartialEq, Eq, Hash)]
pub struct AssetId {
    /// 创建节点 ID
    pub creator: NodeId,
    /// 创建时间戳
    pub created_at: u64,
    /// 内容哈希
    pub content_hash: [u8; 32],
    /// 创建节点签名（证明此 ID 的合法性）
    pub signature: Vec<u8>,
}

/// 资产所有权转移记录
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct AssetTransfer {
    /// 资产 ID
    pub asset_id: AssetId,
    /// 转移序号（从 0 开始，每转移一次 +1）
    pub seq: u64,
    /// 前一手持有者（None = 首次创建）
    pub from: Option<NodeId>,
    /// 当前持有者
    pub to: NodeId,
    /// 时间戳
    pub timestamp: u64,
    /// from 的签名（证明其放弃所有权）
    pub from_sig: Vec<u8>,
    /// to 的签名（证明其接受所有权）
    pub to_sig: Vec<u8>,
}

/// 资产的持有链摘要
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct AssetSummary {
    pub asset_id: AssetId,
    pub current_owner: NodeId,
    pub chain_length: u64,
    pub last_transferred_at: u64,
}

/// 资产注册表（CRDT）
#[derive(Clone)]
pub struct AssetRegistry {
    node_id: NodeId,
    /// 资产 ID → 持有链（按 seq 排序）
    pub assets: HashMap<AssetId, Vec<AssetTransfer>>,
    /// 本地节点持有的资产列表
    pub local_assets: Vec<AssetId>,
}

impl AssetRegistry {
    pub fn new(node_id: NodeId) -> Self {
        AssetRegistry {
            node_id,
            assets: HashMap::new(),
            local_assets: Vec::new(),
        }
    }

    /// 创建一个新资产（.soul 文件出生）
    pub fn create_asset(
        &mut self,
        content_hash: [u8; 32],
        creator_sig: Vec<u8>,
    ) -> AssetId {
        let aid = AssetId {
            creator: self.node_id.clone(),
            created_at: current_time_millis(),
            content_hash,
            signature: creator_sig,
        };

        let transfer = AssetTransfer {
            asset_id: aid.clone(),
            seq: 0,
            from: None,
            to: self.node_id.clone(),
            timestamp: current_time_millis(),
            from_sig: vec![],
            to_sig: vec![],
        };

        self.assets.insert(aid.clone(), vec![transfer]);
        self.local_assets.push(aid.clone());
        aid
    }

    /// 转移资产所有权
    pub fn transfer(
        &mut self,
        asset_id: &AssetId,
        to: &NodeId,
        from_sig: Vec<u8>,
        to_sig: Vec<u8>,
    ) -> Result<(), String> {
        let chain = self.assets.get_mut(asset_id)
            .ok_or_else(|| "资产不存在".to_string())?;

        let last = chain.last().unwrap();
        if last.to != self.node_id {
            return Err("当前节点不是资产持有者".to_string());
        }

        let transfer = AssetTransfer {
            asset_id: asset_id.clone(),
            seq: last.seq + 1,
            from: Some(self.node_id.clone()),
            to: to.clone(),
            timestamp: current_time_millis(),
            from_sig,
            to_sig,
        };

        chain.push(transfer);

        // 从本地资产移除
        self.local_assets.retain(|a| a != asset_id);

        Ok(())
    }

    /// 查询资产的当前持有者
    pub fn current_owner(&self, asset_id: &AssetId) -> Option<NodeId> {
        self.assets.get(asset_id)
            .and_then(|chain| chain.last())
            .map(|t| t.to.clone())
    }

    /// 查询资产的完整持有历史
    pub fn ownership_history(&self, asset_id: &AssetId) -> Option<&Vec<AssetTransfer>> {
        self.assets.get(asset_id)
    }

    /// 获取资产的摘要信息
    pub fn summary(&self, asset_id: &AssetId) -> Option<AssetSummary> {
        self.assets.get(asset_id).and_then(|chain| {
            chain.last().map(|last| AssetSummary {
                asset_id: asset_id.clone(),
                current_owner: last.to.clone(),
                chain_length: chain.len() as u64,
                last_transferred_at: last.timestamp,
            })
        })
    }

    /// 合并另一个节点的资产注册表（CRDT 合并）
    pub fn merge(&mut self, other: &AssetRegistry) {
        for (aid, other_chain) in &other.assets {
            let entry = self.assets.entry(aid.clone()).or_insert_with(Vec::new);
            // 取更长的链（CRDT：更长的持有链胜出）
            if other_chain.len() > entry.len() {
                *entry = other_chain.clone();
            }
        }
    }

    /// 验证一份文件的持有链是否完整
    pub fn verify_ownership(&self, asset_id: &AssetId) -> bool {
        let chain = match self.assets.get(asset_id) {
            Some(c) => c,
            None => return false,
        };

        // 链必须非空
        if chain.is_empty() { return false; }

        // 首条记录必须是创建
        let first = &chain[0];
        if first.from.is_some() { return false; }

        // 后续转移必须连续
        for i in 1..chain.len() {
            let prev = &chain[i - 1];
            let curr = &chain[i];

            // seq 必须连续
            if curr.seq != prev.seq + 1 { return false; }
            // 前一条的 to 必须是当前的 from
            if curr.from.as_ref() != Some(&prev.to) { return false; }
        }

        true
    }
}

// ============================================================
// 辅助函数
// ============================================================

fn current_time_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

// ============================================================
// 统一台账（三层合一的对外接口）
// ============================================================

/// ClawNet 完整账本：令牌 + 观测 + 资产
#[derive(Clone)]
pub struct UnifiedLedger {
    pub tokens: TokenLedger,
    pub observations: ObservationChain,
    pub assets: AssetRegistry,
}

impl UnifiedLedger {
    pub fn new(node_id: NodeId) -> Self {
        UnifiedLedger {
            tokens: TokenLedger::new(node_id.clone()),
            observations: ObservationChain::new(node_id),
            assets: AssetRegistry::new(node_id),
        }
    }

    /// 合并来自另一个节点的完整账本
    pub fn merge_from(&mut self, other: &UnifiedLedger) {
        self.tokens.merge(&other.tokens);
        self.assets.merge(&other.assets);
        // 观测数据从 gossip 通道单独同步
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_basic() {
        let id_a = NodeId([1u8; 32]);
        let id_b = NodeId([2u8; 32]);
        let mut ledger = TokenLedger::new(id_a.clone());

        ledger.settle(&id_a, &id_b, 5, &[]).unwrap();
        assert_eq!(ledger.get_balance(&id_a, &id_b), 5);
        assert_eq!(ledger.net_balance(&id_a), -5);
        assert_eq!(ledger.net_balance(&id_b), 5);
    }

    #[test]
    fn test_token_crdt_merge() {
        let id_a = NodeId([1u8; 32]);
        let id_b = NodeId([2u8; 32]);
        let id_c = NodeId([3u8; 32]);

        let mut ledger_a = TokenLedger::new(id_a.clone());
        ledger_a.settle(&id_a, &id_b, 3, &[]).unwrap();
        ledger_a.settle(&id_c, &id_a, 1, &[]).unwrap();

        let mut ledger_b = TokenLedger::new(id_b.clone());
        ledger_b.settle(&id_a, &id_b, 5, &[]).unwrap();

        ledger_b.merge(&ledger_a);
        assert_eq!(ledger_b.get_balance(&id_a, &id_b), 5);
        assert_eq!(ledger_b.get_balance(&id_c, &id_a), 1);
    }

    #[test]
    fn test_observation_query() {
        let node = NodeId([1u8; 32]);
        let mut chain = ObservationChain::new(node.clone());

        chain.append(Observation {
            node_id: node.clone(),
            task_type: "dom-search".into(),
            task_hash: [0u8; 32],
            data_hash: [1u8; 32],
            source_ref: Some("https://example.com".into()),
            timestamp: 1000,
            prev_observation: [0u8; 32],
            signature: vec![],
        });

        let results = chain.query(&[1u8; 32], 500);
        assert_eq!(results.len(), 1);

        // 负存在证明：1000 之后有观测
        assert!(chain.has_any_observation(2000));
        // 500 之前没有观测
        assert!(!chain.has_any_observation(500));
    }

    #[test]
    fn test_asset_lifecycle() {
        let node_a = NodeId([1u8; 32]);
        let node_b = NodeId([2u8; 32]);
        let mut registry = AssetRegistry::new(node_a.clone());

        // 创建资产
        let aid = registry.create_asset([42u8; 32], vec![]);
        assert_eq!(registry.current_owner(&aid), Some(node_a.clone()));

        // 转移给 B
        registry.transfer(&aid, &node_b, vec![], vec![]).unwrap();
        assert_eq!(registry.current_owner(&aid), Some(node_b));

        // 验证持有链完整性
        assert!(registry.verify_ownership(&aid));
    }

    #[test]
    fn test_asset_merge() {
        let node_a = NodeId([1u8; 32]);
        let node_b = NodeId([2u8; 32]);
        let node_c = NodeId([3u8; 32]);

        let mut reg_a = AssetRegistry::new(node_a.clone());
        let mut reg_b = AssetRegistry::new(node_b.clone());

        // A 创建一个资产
        let aid = reg_a.create_asset([42u8; 32], vec![]);
        reg_a.transfer(&aid, &node_b, vec![], vec![]).unwrap();

        // B 有更长的链
        reg_b.merge(&reg_a);
        assert_eq!(reg_b.current_owner(&aid), Some(node_b));
        assert_eq!(reg_b.summary(&aid).unwrap().chain_length, 2);
    }
}
