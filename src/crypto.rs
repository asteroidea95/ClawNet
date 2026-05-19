// 密码学模块 — 节点身份标识与签名

use ed25519_dalek::{SigningKey, VerifyingKey, Signature, Signer, Verifier};
use rand::rngs::OsRng;
use std::fmt;
use std::fs;
use std::path::Path;

/// 节点 ID (由 Ed25519 公钥派生)
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct NodeId(pub [u8; 32]);

impl NodeId {
    pub fn from_bytes(bytes: &[u8; 32]) -> Self {
        NodeId(*bytes)
    }

    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // 取前 8 字节显示
        for byte in &self.0[..8] {
            write!(f, "{:02x}", byte)?;
        }
        Ok(())
    }
}

impl From<&VerifyingKey> for NodeId {
    fn from(key: &VerifyingKey) -> Self {
        NodeId(key.to_bytes())
    }
}

impl From<&SigningKey> for NodeId {
    fn from(key: &SigningKey) -> Self {
        NodeId(key.verifying_key().to_bytes())
    }
}

/// 加载或生成密钥对
pub fn load_or_generate_keypair(path: &str) -> Result<SigningKey, Box<dyn std::error::Error>> {
    let key_path = Path::new(&path);

    if key_path.exists() {
        let bytes = fs::read(key_path)?;
        let bytes: [u8; 32] = bytes.try_into().map_err(|_| "invalid key file")?;
        Ok(SigningKey::from_bytes(&bytes))
    } else {
        let mut csprng = OsRng;
        let keypair = SigningKey::generate(&mut csprng);

        if let Some(parent) = key_path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(key_path, &keypair.to_bytes())?;

        println!("🔑 新节点密钥已生成: {}", key_path.display());
        Ok(keypair)
    }
}

/// 签名消息
pub fn sign(keypair: &SigningKey, message: &[u8]) -> Signature {
    keypair.sign(message)
}

/// 验证签名
pub fn verify(key: &VerifyingKey, message: &[u8], signature: &Signature) -> bool {
    key.verify(message, signature).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sign_verify() {
        let mut csprng = OsRng;
        let keypair = SigningKey::generate(&mut csprng);

        let message = b"hello clawnet";
        let signature = sign(&keypair, message);
        assert!(verify(&keypair.verifying_key(), message, &signature));
    }

    #[test]
    fn test_node_id() {
        let mut csprng = OsRng;
        let keypair = SigningKey::generate(&mut csprng);
        let node_id = NodeId::from(&keypair);
        assert_eq!(node_id.as_bytes(), &keypair.verifying_key().to_bytes());
    }
}
