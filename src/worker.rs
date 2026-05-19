// Worker 模块 — 可插拔的任务执行器

use crate::crypto::NodeId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Worker 能力标签
#[derive(Clone, Serialize, Deserialize)]
pub enum WorkerCapability {
    Dom,
    WebGpu,
    CoreMl,
    Cuda,
    Vulkan,
}

/// Worker 模块接口
pub trait Worker: Send + Sync {
    fn name(&self) -> &str;
    fn capabilities(&self) -> Vec<WorkerCapability>;
    fn execute(&self, task: &str) -> Result<String, String>;
}

/// Worker 注册表
pub struct Registry {
    workers: HashMap<String, Box<dyn Worker>>,
}

impl Registry {
    pub fn new() -> Self {
        Registry {
            workers: HashMap::new(),
        }
    }

    pub fn register(&mut self, name: &str, worker: Box<dyn Worker>) {
        self.workers.insert(name.to_string(), worker);
    }

    pub fn get(&self, name: &str) -> Option<&Box<dyn Worker>> {
        self.workers.get(name)
    }

    pub fn list_capabilities(&self) -> Vec<String> {
        self.workers.keys().cloned().collect()
    }
}

/// DOM Worker — 浏览器自动化
pub struct DomWorker;

impl DomWorker {
    pub fn new() -> Result<Box<Self>, Box<dyn std::error::Error>> {
        Ok(Box::new(DomWorker))
    }
}

impl Worker for DomWorker {
    fn name(&self) -> &str {
        "dom"
    }

    fn capabilities(&self) -> Vec<WorkerCapability> {
        vec![WorkerCapability::Dom]
    }

    fn execute(&self, task: &str) -> Result<String, String> {
        // 集成 OpenClaw browser-automation
        // 实际执行由 ClawNet 框架调度后委托给 OpenClaw
        Ok(format!("dom worker received task: {}", task))
    }
}
