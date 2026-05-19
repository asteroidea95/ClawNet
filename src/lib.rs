// ClawNet — 核心库

pub mod crypto;
pub mod gossip;
pub mod ledger;
pub mod scheduler;
pub mod transport;
pub mod worker;

use std::sync::Arc;
use tokio::sync::RwLock;

/// ClawNet 节点实例
pub struct Node {
    pub id: crypto::NodeId,
    pub gossip: gossip::Engine,
    pub ledger: Arc<RwLock<ledger::TokenLedger>>,
    pub scheduler: scheduler::Scheduler,
    pub transport: transport::Transport,
    pub worker_registry: worker::Registry,
}

impl Node {
    /// 创建一个新节点并加入网络
    pub async fn new(config: Config) -> Result<Self, Box<dyn std::error::Error>> {
        // 1. 生成或加载节点密钥
        let keypair = crypto::load_or_generate_keypair(&config.key_path)?;
        let node_id = crypto::NodeId::from(&keypair);

        // 2. 启动 gossip 引擎（节点发现 + 状态同步）
        let gossip = gossip::Engine::new(node_id.clone(), &config.seed_nodes).await?;

        // 3. 初始化令牌台账
        let ledger = Arc::new(RwLock::new(ledger::TokenLedger::new(node_id.clone())));

        // 4. 启动调度器
        let scheduler = scheduler::Scheduler::new(node_id.clone(), ledger.clone(), gossip.clone());

        // 5. 启动传输层
        let transport = transport::Transport::new(&config.listen_addr).await?;

        // 6. 注册内置 worker 模块
        let mut worker_registry = worker::Registry::new();
        worker_registry.register("dom", worker::DomWorker::new()?);

        Ok(Node {
            id: node_id,
            gossip,
            ledger,
            scheduler,
            transport,
            worker_registry,
        })
    }

    /// 启动节点（阻塞）
    pub async fn run(&self) -> Result<(), Box<dyn std::error::Error>> {
        // gossip 循环
        // 调度器循环
        // 等待任务
        Ok(())
    }
}

/// 节点配置
pub struct Config {
    pub key_path: String,
    pub seed_nodes: Vec<String>,
    pub listen_addr: String,
    pub data_dir: String,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            key_path: "~/.clawnet/keypair".to_string(),
            seed_nodes: vec![],
            listen_addr: "0.0.0.0:9876".to_string(),
            data_dir: "~/.clawnet/data".to_string(),
        }
    }
}
