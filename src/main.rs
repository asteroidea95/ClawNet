// ClawNet CLI

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "clawnet", version, about = "C端去中心化算力网络")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// 启动节点
    Start {
        /// 种子节点地址
        #[arg(short, long)]
        seeds: Vec<String>,

        /// 监听端口
        #[arg(short, long, default_value = "9876")]
        port: u16,
    },

    /// 查看节点状态
    Status,

    /// 查看令牌余额
    Tokens {
        /// 可选：查看指定节点
        node_id: Option<String>,
    },

    /// 提交任务
    Task {
        #[command(subcommand)]
        action: TaskCommands,
    },
}

#[derive(Subcommand)]
enum TaskCommands {
    /// 提交新任务
    Submit {
        /// 任务类型: dom | image-gen | video-gen | inference
        #[arg(short, long)]
        r#type: String,

        /// 任务描述
        #[arg(short, long)]
        description: String,

        /// 支付令牌数
        #[arg(short, long, default_value = "1")]
        tokens: u32,
    },

    /// 列出已提交的任务
    List,

    /// 查看任务详情
    Show {
        task_id: String,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match &cli.command {
        Commands::Start { seeds, port } => {
            println!("🚀 ClawNet 节点启动中...");
            let config = clawnet_core::Config {
                seed_nodes: seeds.clone(),
                listen_addr: format!("0.0.0.0:{}", port),
                ..Default::default()
            };
            let node = clawnet_core::Node::new(config).await?;
            println!("✅ 节点已上线: {}", node.id);
            node.run().await?;
        }
        Commands::Status => {
            println!("📊 节点状态（待实现）");
        }
        Commands::Tokens { node_id } => {
            println!("💰 令牌余额（待实现）: {:?}", node_id);
        }
        Commands::Task { action } => match action {
            TaskCommands::Submit { r#type, description, tokens } => {
                println!("📋 提交任务: [{}] {} ({} tokens)", r#type, description, tokens);
            }
            TaskCommands::List => {
                println!("📋 任务列表（待实现）");
            }
            TaskCommands::Show { task_id } => {
                println!("📋 任务详情（待实现）: {}", task_id);
            }
        },
    }

    Ok(())
}
