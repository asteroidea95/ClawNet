// ClawNet — .soul 文件格式
//
// .soul 是一种不可复制、只能本体迁移的数据文件格式。
// 它不是通过技术阻止复制，而是通过网络协议让复制品无法获得合法性。
//
// 核心思想：
//   每份 .soul 文件在 ClawNet 的 AssetRegistry 中有一条不可篡改的持有链。
//   全网只承认持有链上当前持有者手中的那份是"正本"。
//   复制品即使字节完全相同，没有合法的持有链，网络也不认。
//
// 这实现了「数字物理」特性：数据像一把椅子——只能搬走，不能拷贝。

use crate::crypto::NodeId;
use crate::ledger::{AssetId, AssetRegistry};
use serde::{Deserialize, Serialize};

/// .soul 文件头部（明文，可公开读）
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct SoulHeader {
    /// 文件魔数: "CLAWNSOUL"
    pub magic: [u8; 9],
    /// 格式版本
    pub version: u8,
    /// 资产 ID（全局唯一）
    pub asset_id: AssetId,
    /// 资产名称（人类可读）
    pub name: String,
    /// 文件类型标签
    pub mime_type: String,
    /// 内容大小（加密前字节数）
    pub content_size: u64,
    /// 加密算法
    pub cipher: CipherSuite,
    /// 持有链长度
    pub chain_length: u64,
    /// 创建时间戳
    pub created_at: u64,
}

/// 支持的加密套件
#[derive(Clone, Serialize, Deserialize, Debug)]
pub enum CipherSuite {
    /// AES-256-GCM + Ed25519 签名
    Aes256GcmEd25519,
}

/// .soul 文件体（加密，需认证才能解密）
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct SoulBody {
    /// 加密后的内容（AES-256-GCM）
    pub encrypted_data: Vec<u8>,
    /// GCM 认证标签
    pub auth_tag: [u8; 16],
    /// 初始化向量
    pub iv: [u8; 12],
    /// 内容哈希（用于验证完整性）
    pub content_hash: [u8; 32],
    /// 访问次数计数器（每次读取后递增）
    pub read_count: u64,
    /// 创建节点签名
    pub creator_signature: Vec<u8>,
}

/// 完整的 .soul 文件
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct SoulFile {
    pub header: SoulHeader,
    pub body: SoulBody,
}

/// .soul 文件的持有证明（不包含数据，仅用于验证所有权）
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct SoulProof {
    pub asset_id: AssetId,
    pub current_owner: NodeId,
    pub chain_length: u64,
    pub last_read_count: u64,
    pub timestamp: u64,
    pub signature: Vec<u8>,
}

impl SoulFile {
    /// 创建一个新的 .soul 文件
    pub fn create(
        asset_id: AssetId,
        name: String,
        mime_type: String,
        data: &[u8],
        creator_sig: Vec<u8>,
    ) -> Self {
        use sha2::{Sha256, Digest};

        let content_hash = Sha256::digest(data).into();

        SoulFile {
            header: SoulHeader {
                magic: *b"CLAWNSOUL",
                version: 1,
                asset_id,
                name,
                mime_type,
                content_size: data.len() as u64,
                cipher: CipherSuite::Aes256GcmEd25519,
                chain_length: 1,
                created_at: current_time_millis(),
            },
            body: SoulBody {
                // 简版：空加密（实际需实现 AES-GCM）
                encrypted_data: data.to_vec(),
                auth_tag: [0u8; 16],
                iv: [0u8; 12],
                content_hash,
                read_count: 0,
                creator_signature: creator_sig,
            },
        }
    }

    /// 验证文件头部的魔数
    pub fn is_valid_magic(&self) -> bool {
        &self.header.magic == b"CLAWNSOUL"
    }

    /// 获取资产 ID
    pub fn asset_id(&self) -> &AssetId {
        &self.header.asset_id
    }
}

/// .soul 文件格式的序列化/反序列化
impl SoulFile {
    /// 序列化为字节（.soul 文件格式）
    pub fn to_bytes(&self) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        Ok(bincode::serialize(self)?)
    }

    /// 从字节反序列化
    pub fn from_bytes(data: &[u8]) -> Result<Self, Box<dyn std::error::Error>> {
        let soul: SoulFile = bincode::deserialize(data)?;
        if !soul.is_valid_magic() {
            return Err("无效的文件魔数".into());
        }
        Ok(soul)
    }
}

/// .soul 文件管理器
pub struct SoulManager {
    registry: AssetRegistry,
    local_files: Vec<SoulFile>,
}

impl SoulManager {
    pub fn new(registry: AssetRegistry) -> Self {
        SoulManager {
            registry,
            local_files: Vec::new(),
        }
    }

    /// 创建并保存一个新的 .soul 文件
    pub fn create_file(
        &mut self,
        name: String,
        mime_type: String,
        data: &[u8],
        creator_sig: Vec<u8>,
    ) -> Result<SoulFile, String> {
        use sha2::{Sha256, Digest};
        let content_hash = Sha256::digest(data).into();

        let asset_id = self.registry.create_asset(content_hash, creator_sig.clone());

        let soul = SoulFile::create(asset_id, name, mime_type, data, creator_sig);
        self.local_files.push(soul.clone());
        Ok(soul)
    }

    /// 将本地的 .soul 文件迁移到另一个节点
    pub fn transfer_to(
        &mut self,
        soul: &SoulFile,
        to: &NodeId,
        from_sig: Vec<u8>,
        to_sig: Vec<u8>,
    ) -> Result<SoulProof, String> {
        self.registry.transfer(soul.asset_id(), to, from_sig, to_sig)?;

        // 从本地移除（迁移后本地不再持有）
        self.local_files.retain(|f| f.asset_id() != soul.asset_id());

        let proof = SoulProof {
            asset_id: soul.asset_id().clone(),
            current_owner: to.clone(),
            chain_length: self.registry.summary(soul.asset_id())
                .map(|s| s.chain_length)
                .unwrap_or(0),
            last_read_count: soul.body.read_count,
            timestamp: current_time_millis(),
            signature: vec![],
        };

        Ok(proof)
    }

    /// 读取一个 .soul 文件（增加读取计数）
    pub fn read(&mut self, asset_id: &AssetId) -> Option<&mut SoulFile> {
        // 先验证持有权
        if self.registry.current_owner(asset_id) != Some(self.registry.node_id.clone()) {
            return None;
        }

        // 验证持有链完整性
        if !self.registry.verify_ownership(asset_id) {
            return None;
        }

        self.local_files.iter_mut().find(|f| f.asset_id() == asset_id)
            .map(|f| {
                f.body.read_count += 1;
                f
            })
    }

    /// 列出本地持有的 .soul 文件
    pub fn list_local(&self) -> Vec<&SoulFile> {
        self.local_files.iter().collect()
    }
}

fn current_time_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::NodeId;

    #[test]
    fn test_create_and_verify() {
        let node = NodeId([1u8; 32]);
        let registry = AssetRegistry::new(node);
        let mut mgr = SoulManager::new(registry);

        let data = b"Hello, this is a .soul file";
        let soul = mgr.create_file("test.txt".into(), "text/plain".into(), data, vec![]).unwrap();

        assert!(soul.is_valid_magic());
        assert_eq!(&soul.header.magic, b"CLAWNSOUL");
        assert_eq!(soul.header.content_size, data.len() as u64);
    }

    #[test]
    fn test_transfer() {
        let node_a = NodeId([1u8; 32]);
        let node_b = NodeId([2u8; 32]);
        let registry = AssetRegistry::new(node_a.clone());
        let mut mgr = SoulManager::new(registry);

        let soul = mgr.create_file("secret.txt".into(), "text/plain".into(), b"classified data", vec![]).unwrap();
        let aid = soul.asset_id().clone();

        // 转移前本地持有
        assert!(mgr.read(&aid).is_some());

        // 转移
        mgr.transfer_to(&soul, &node_b, vec![], vec![]).unwrap();

        // 转移后本地不再持有
        assert!(mgr.read(&aid).is_none());
        assert!(mgr.list_local().is_empty());
    }

    #[test]
    fn test_serialization_roundtrip() {
        let node = NodeId([1u8; 32]);
        let registry = AssetRegistry::new(node);
        let mut mgr = SoulManager::new(registry);

        let original = mgr.create_file("config.json".into(), "application/json".into(),
            br#"{"version": 1, "name": "test"}"#, vec![]).unwrap();

        let bytes = original.to_bytes().unwrap();
        let restored = SoulFile::from_bytes(&bytes).unwrap();

        assert!(restored.is_valid_magic());
        assert_eq!(restored.header.name, "config.json");
        assert_eq!(restored.body.content_size, 31);
    }
}
