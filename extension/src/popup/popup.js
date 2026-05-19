// ClawNet — 弹窗 UI

document.addEventListener('DOMContentLoaded', async () => {
  const bg = chrome.extension.getBackgroundPage()?.__CLAWNET__;

  // 节点 ID
  if (bg?.getNodeId) {
    document.getElementById('node-id').textContent = (await bg.getNodeId()).slice(0, 8) + '...';
  }

  // 更新循环
  setInterval(async () => {
    const status = bg ? await bg.getStatus() : 'unknown';
    const badge = document.getElementById('status-badge');
    badge.textContent = status === 'connected' ? '🟢 在线' :
                        status === 'computing' ? '🔵 计算中' :
                        status === 'init' ? '⏳ 初始化中' : '🟠 离线';
    badge.className = 'status-badge ' + status;

    // 令牌
    if (bg?.getBalance) {
      const balance = bg.getBalance();
      document.getElementById('token-balance').textContent = balance;
    }

    // 连接数
    if (bg?.getPeers) {
      document.getElementById('peer-count').textContent = bg.getPeers();
    }

    // 队列长度
    if (bg?.getQueueLength) {
      document.getElementById('queue-length').textContent = bg.getQueueLength();
    }

    // 能力
    chrome.storage.local.get(['clawnet_capabilities'], (r) => {
      if (r.clawnet_capabilities) {
        document.getElementById('capabilities').textContent = r.clawnet_capabilities.join(', ');
      }
    });

    // 运行时间
    chrome.storage.local.get(['clawnet_uptime'], (r) => {
      if (r.clawnet_uptime) {
        document.getElementById('uptime').textContent = Math.floor((Date.now() - r.clawnet_uptime) / 60000);
      }
    });

    // 任务完成数
    chrome.storage.local.get(['clawnet_tasks_done'], (r) => {
      document.getElementById('tasks-completed').textContent = r.clawnet_tasks_done || 0;
    });

    // 当前任务
    chrome.storage.local.get(['clawnet_current_task'], (r) => {
      const el = document.getElementById('current-task');
      if (r.clawnet_current_task) {
        el.innerHTML = `<span class="task-desc">⏳ ${r.clawnet_current_task}</span>`;
      } else {
        el.innerHTML = `<span style="color: var(--text-dim);">空闲</span>`;
      }
    });
  }, 1000);

  // 提交任务按钮
  document.getElementById('btn-submit-task').addEventListener('click', () => {
    // 占位：弹出任务表单
    alert('任务提交功能即将上线\n\n目前支持远程节点代为搜索、处理网页内容');
  });
});
