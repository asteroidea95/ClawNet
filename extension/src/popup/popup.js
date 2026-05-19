/**
 * ClawNet — 弹窗 UI（完整功能版）
 */

// ============================================================
// 标签页切换
// ============================================================

document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    tab.classList.add('active');

    document.querySelectorAll('[id^="tab-"]').forEach(t => t.classList.add('hidden'));
    document.getElementById(`tab-${tab.dataset.tab}`).classList.remove('hidden');
  });
});

// ============================================================
// 日志系统
// ============================================================

function addLog(msg) {
  const logList = document.getElementById('log-list');
  const div = document.createElement('div');
  div.className = 'log-item';

  const time = new Date();
  const ts = `${time.getHours().toString().padStart(2,'0')}:${time.getMinutes().toString().padStart(2,'0')}:${time.getSeconds().toString().padStart(2,'0')}`;

  div.innerHTML = `<span class="log-time">[${ts}]</span><span class="log-msg">${msg}</span>`;
  logList.appendChild(div);

  // 移除"暂无日志"提示
  const empty = logList.querySelector('div:first-child');
  if (empty && empty.textContent === '暂无日志') empty.remove();

  // 只保留最后50条
  while (logList.children.length > 50) {
    logList.removeChild(logList.firstChild);
  }
}

document.getElementById('btn-clear-logs').addEventListener('click', () => {
  document.getElementById('log-list').innerHTML =
    '<div style="color:var(--text-dim);font-size:12px;">暂无日志</div>';
});

// ============================================================
// 自动接单开关
// ============================================================

const autoAcceptCheckbox = document.getElementById('auto-accept');
const toggleTrack = document.getElementById('toggle-track');
const toggleKnob = document.getElementById('toggle-knob');

autoAcceptCheckbox.addEventListener('change', () => {
  if (autoAcceptCheckbox.checked) {
    toggleTrack.style.background = 'var(--accent)';
    toggleKnob.style.right = '2px';
    toggleKnob.style.left = 'auto';
    addLog('⏺ 自动接单已开启');
  } else {
    toggleTrack.style.background = 'var(--border)';
    toggleKnob.style.left = '2px';
    toggleKnob.style.right = 'auto';
    addLog('⏸ 自动接单已关闭');
  }
});

// ============================================================
// 任务提交
// ============================================================

document.getElementById('task-reward').addEventListener('input', (e) => {
  document.getElementById('task-reward-label').textContent = e.target.value;
});

document.getElementById('btn-execute').addEventListener('click', async () => {
  const type = document.getElementById('task-type').value;
  const url = document.getElementById('task-url').value.trim();
  const desc = document.getElementById('task-desc').value.trim();
  const paramsText = document.getElementById('task-params').value.trim();
  const reward = parseInt(document.getElementById('task-reward').value);

  if (!desc && type !== 'dom-extract') {
    addLog('⚠️ 请输入任务描述');
    return;
  }

  const params = paramsText ? JSON.parse(paramsText) : {};
  const resultArea = document.getElementById('task-result-area');
  const resultDiv = document.getElementById('task-result');

  resultArea.classList.add('hidden');
  addLog(`📤 提交任务: [${type}] ${desc.slice(0, 50)}`);

  // 构造任务对象
  const task = {
    type,
    url,
    description: desc,
    params,
    reward
  };

  try {
    // 通过 background worker 执行
    const response = await chrome.runtime.sendMessage({
      type: 'clawnet-execute-task',
      task
    });

    if (response && response.error) {
      addLog(`❌ 执行失败: ${response.error}`);
      resultDiv.textContent = JSON.stringify(response, null, 2);
    } else {
      addLog(`✅ 任务完成`);
      resultDiv.textContent = JSON.stringify(response, null, 2);
    }

    resultArea.classList.remove('hidden');
  } catch (e) {
    addLog(`❌ 提交失败: ${e.message}`);
    resultDiv.textContent = e.message;
    resultArea.classList.remove('hidden');
  }
});

// ============================================================
// 状态更新循环
// ============================================================

async function updateStatus() {
  try {
    const status = await chrome.storage.local.get([
      'clawnet_status', 'clawnet_node_id', 'clawnet_queue_length',
      'clawnet_current_task', 'clawnet_tasks_done', 'clawnet_uptime',
      'clawnet_capabilities', 'clawnet_ledger', 'clawnet_webgpu'
    ]);

    // 节点 ID
    const idEl = document.getElementById('node-id');
    if (status.clawnet_node_id) {
      idEl.textContent = status.clawnet_node_id.slice(0, 8) + '...';
    }

    // 状态标签
    const badge = document.getElementById('status-badge');
    const s = status.clawnet_status || 'init';
    badge.textContent = s === 'connected' ? '🟢 在线' :
                        s === 'computing' ? '🔵 计算中' :
                        s === 'idle' ? '🟢 空闲' :
                        s === 'init' ? '⏳ 初始化' : '🟠 离线';
    badge.className = 'status-badge ' + (s === 'connected' || s === 'idle' ? 'connected' : s);

    // 令牌
    const ledger = status.clawnet_ledger || {};
    const nodeId = status.clawnet_node_id;
    let balance = 0;
    if (nodeId && ledger) {
      let credit = 0, debit = 0;
      for (const [key, amount] of Object.entries(ledger)) {
        const [from, to] = key.split('->');
        if (to === nodeId) credit += amount;
        if (from === nodeId) debit += amount;
      }
      balance = credit - debit;
    }
    document.getElementById('token-balance').textContent = balance;

    // 在线节点数（从 background 获取）
    try {
      const bg = await chrome.runtime.getBackgroundPage();
      const peers = bg?.__CLAWNET__?.getPeers ? bg.__CLAWNET__.getPeers() : 0;
      document.getElementById('peer-count').textContent = typeof peers === 'function' ? await peers() : peers;
    } catch {
      document.getElementById('peer-count').textContent = '?';
    }

    // 队列
    document.getElementById('queue-length').textContent = status.clawnet_queue_length || 0;

    // 能力
    const caps = status.clawnet_capabilities || [];
    document.getElementById('capabilities').textContent = caps.length > 0 ? caps.join(', ') : '--';

    // 当前任务
    const currTask = document.getElementById('current-task');
    if (status.clawnet_current_task) {
      currTask.textContent = `⏳ ${status.clawnet_current_task}`;
    } else {
      currTask.textContent = '空闲';
      currTask.style.color = 'var(--text-dim)';
    }

    // 贡献统计
    document.getElementById('uptime').textContent =
      status.clawnet_uptime ? Math.floor((Date.now() - status.clawnet_uptime) / 60000) : 0;
    document.getElementById('tasks-completed').textContent = status.clawnet_tasks_done || 0;

    // WebGPU 状态
    const wgpu = document.getElementById('webgpu-status');
    if (status.clawnet_webgpu === true) {
      wgpu.textContent = '✅ 可用';
      wgpu.className = 'value green';
    } else if (status.clawnet_webgpu === false) {
      wgpu.textContent = '❌ 不可用';
      wgpu.className = 'value orange';
    } else {
      wgpu.textContent = '检测中...';
      wgpu.className = 'value orange';
    }

  } catch (e) {
    // 忽略
  }
}

// 每秒更新
setInterval(updateStatus, 1000);

// ============================================================
// 初始加载
// ============================================================

updateStatus();
addLog('🚀 ClawNet 节点已启动');
