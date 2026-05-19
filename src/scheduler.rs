// 调度器 — 任务拆解与分发

use crate::crypto::NodeId;
use crate::ledger::UnifiedLedger;
use std::sync::Arc;
use tokio::sync::RwLock;

/// 任务类型
#[derive(Clone, PartialEq)]
pub enum TaskType {
    Dom,          // 浏览器DOM操作
    ImageGen,     // 图片生成
    VideoGen,     // 视频渲染
    Inference,    // 通用推理
    Simulation,   // 物理模拟
}

/// 任务状态
#[derive(Clone, PartialEq)]
pub enum TaskState {
    Created,
    Assigned,
    InProgress,
    Completed,
    Failed,
    Expired,
}

/// 任务
pub struct Task {
    pub id: [u8; 32],
    pub task_type: TaskType,
    pub description: String,
    pub reward_tokens: u32,
    pub initiator: NodeId,
    pub assigned_to: Option<NodeId>,
    pub state: TaskState,
    pub created_at: u64,
    pub deadline: u64,
}

/// 调度器
pub struct Scheduler {
    node_id: NodeId,
    ledger: Arc<RwLock<UnifiedLedger>>,
    // 待办任务队列
    pending_tasks: Arc<RwLock<Vec<Task>>>,
}

impl Scheduler {
    pub fn new(
        node_id: NodeId,
        ledger: Arc<RwLock<UnifiedLedger>>,
        _gossip: crate::gossip::Engine,
    ) -> Self {
        Scheduler {
            node_id,
            ledger,
            pending_tasks: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// 提交新任务
    pub async fn submit_task(&self, task: Task) {
        let mut tasks = self.pending_tasks.write().await;
        tasks.push(task);
    }

    /// 领取下一个可用任务
    pub async fn claim_next_task(&self, worker_id: &NodeId) -> Option<Task> {
        let mut tasks = self.pending_tasks.write().await;
        // 按令牌余额排序，令牌多的优先
        tasks.sort_by(|a, b| b.reward_tokens.cmp(&a.reward_tokens));
        tasks.pop()
    }
}
