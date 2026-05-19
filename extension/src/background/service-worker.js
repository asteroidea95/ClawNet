/**
 * ClawNet — 后台服务 Worker
 * 
 * 职责：
 *   1. WebRTC P2P 组网（节点发现 + 连接管理）
 *   2. WebGPU 推理（执行算力任务）
 *   3. 令牌台账管理
 *   4. 任务队列管理
 */

// ============================================================
// 节点身份
// ============================================================

let nodeId = null;
let nodeIdPromise = null;

async function getNodeId() {
  if (nodeId) return nodeId;
  if (nodeIdPromise) return nodeIdPromise;

  nodeIdPromise = (async () => {
    const stored = await chrome.storage.local.get('clawnet_node_id');
    if (stored.clawnet_node_id) {
      nodeId = stored.clawnet_node_id;
      return nodeId;
    }

    // 生成 Ed25519 密钥对（通过 Web Crypto API）
    const keyPair = await crypto.subtle.generateKey(
      { name: 'Ed25519', namedCurve: 'Ed25519' },
      true,
      ['sign', 'verify']
    );
    const pubKey = await crypto.subtle.exportKey('raw', keyPair.publicKey);
    const nodeIdStr = Array.from(new Uint8Array(pubKey))
      .map(b => b.toString(16).padStart(2, '0'))
      .join('').slice(0, 16);  // 取前16位做展示ID

    await chrome.storage.local.set({
      clawnet_node_id: nodeIdStr,
      clawnet_key_pair: { publicKey: Array.from(new Uint8Array(pubKey)) }
    });

    nodeId = nodeIdStr;
    return nodeId;
  })();

  return nodeIdPromise;
}

// ============================================================
// 信令服务器连接
// ============================================================

const SIGNALING_SERVER = 'wss://signaling.clawnet.dev';
let signalingWs = null;
let peers = new Map();  // peerId -> RTCPeerConnection

async function connectSignaling() {
  const id = await getNodeId();

  try {
    signalingWs = new WebSocket(SIGNALING_SERVER);
  } catch (e) {
    // 没有信令服务器时，降级为 LAN 广播
    console.warn('[ClawNet] 信令服务器不可用，降级为局域网发现');
    startLanDiscovery();
    return;
  }

  signalingWs.onopen = () => {
    signalingWs.send(JSON.stringify({
      type: 'register',
      nodeId: id,
      capabilities: detectCapabilities()
    }));
    updateStatus('connected');
  };

  signalingWs.onmessage = async (event) => {
    const msg = JSON.parse(event.data);
    await handleSignalingMessage(msg);
  };

  signalingWs.onclose = () => {
    updateStatus('disconnected');
    setTimeout(connectSignaling, 5000);
  };
}

async function handleSignalingMessage(msg) {
  switch (msg.type) {
    case 'peer-list':
      // 收到在线节点列表
      for (const peer of msg.peers) {
        if (peer.nodeId !== await getNodeId() && !peers.has(peer.nodeId)) {
          connectToPeer(peer);
        }
      }
      break;

    case 'offer':
    case 'answer':
    case 'ice-candidate':
      // WebRTC 信令
      await handleRTCSignal(msg);
      break;
  }
}

// ============================================================
// WebRTC P2P 连接
// ============================================================

async function connectToPeer(peerInfo) {
  const config = {
    iceServers: [
      { urls: 'stun:stun.l.google.com:19302' }
    ]
  };

  const pc = new RTCPeerConnection(config);
  peers.set(peerInfo.nodeId, pc);

  // 用于传输任务数据的 DataChannel
  const dc = pc.createDataChannel('clawnet-tasks', {
    ordered: true
  });

  dc.onmessage = (event) => handleTaskMessage(event.data, peerInfo.nodeId);
  dc.onopen = () => console.log(`[ClawNet] P2P 连接已建立: ${peerInfo.nodeId}`);

  // 交换 ICE 候选
  pc.onicecandidate = (event) => {
    if (event.candidate && signalingWs) {
      signalingWs.send(JSON.stringify({
        type: 'ice-candidate',
        target: peerInfo.nodeId,
        candidate: event.candidate
      }));
    }
  };

  // 创建 Offer
  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);

  if (signalingWs) {
    signalingWs.send(JSON.stringify({
      type: 'offer',
      target: peerInfo.nodeId,
      sdp: offer
    }));
  }
}

// ============================================================
// LAN 发现（降级方案）
// ============================================================

function startLanDiscovery() {
  // 通过 WebRTC 不带信令的直连尝试
  // 或通过 mDNS / 局域网广播
  console.log('[ClawNet] LAN 发现模式（待实现）');
}

// ============================================================
// WebGPU 推理引擎（通过 Offscreen Document）
// ============================================================
//
// Service Worker 没有 navigator.gpu，所以 GPU 运算必须
// 在一个隐藏的 Offscreen Document 中运行。
// 这里封装了与 offscreen document 的通信。

let gpuAvailable = false;
let gpuInfo = null;

async function initWebGPU() {
  try {
    // 创建 Offscreen Document 并初始化 GPU
    await ensureGPUWorker();

    const result = await sendToGPUWorker({ action: 'init' });

    if (result.ready) {
      gpuAvailable = true;
      gpuInfo = result.info || { name: result.name };
      addLog(`WebGPU: ✅ ${result.name}`);
      await chrome.storage.local.set({
        clawnet_webgpu: true,
        clawnet_gpu_info: result.info
      });

      // 跑基准
      try {
        const bench = await sendToGPUWorker({ action: 'benchmark' });
        addLog(`GPU 基准: ${bench.flops} · ${bench.opsPerSecond} ops/s`);
        await chrome.storage.local.set({ clawnet_gpu_benchmark: bench });
      } catch (e) {
        addLog(`GPU 基准失败: ${e.message}`);
      }

      return true;
    } else {
      gpuAvailable = false;
      addLog('WebGPU: ❌ 不可用');
      await chrome.storage.local.set({ clawnet_webgpu: false });
      return false;
    }
  } catch (e) {
    gpuAvailable = false;
    addLog(`WebGPU 初始化失败: ${e.message}`);
    await chrome.storage.local.set({ clawnet_webgpu: false });
    return false;
  }
}

async function ensureGPUWorker() {
  // 尝试创建 Offscreen Document
  // 如果已存在会抛异常，忽略即可
  try {
    await chrome.offscreen.createDocument({
      url: 'src/background/offscreen.html',
      reasons: ['COMPUTE'],
      justification: 'ClawNet 使用 WebGPU 进行 AI 推理计算'
    });
  } catch (e) {
    // 如果已存在或已关闭，重新关闭再创建
    if (e.message?.includes('already exists')) {
      addLog('GPU Worker 已存在，复用');
      return;
    }
  }

  // 等待 GPU Worker 加载并发送就绪信号
  await new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      reject(new Error('GPU Worker 加载超时'));
    }, 15000);

    const handler = (msg) => {
      if (msg.type === 'gpu-status') {
        clearTimeout(timeout);
        chrome.runtime.onMessage.removeListener(handler);
        resolve();
      }
    };
    chrome.runtime.onMessage.addListener(handler);
  });
}

async function sendToGPUWorker(msg) {
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => reject(new Error('GPU 通信超时')), 30000);

    chrome.runtime.sendMessage(
      { ...msg, target: 'gpu-worker' },
      (response) => {
        clearTimeout(timeout);
        if (chrome.runtime.lastError) {
          reject(new Error(chrome.runtime.lastError.message));
        } else if (!response?.ok) {
          reject(new Error(response?.error || 'GPU 操作失败'));
        } else {
          resolve(response.result);
        }
      }
    );
  });
}

async function runInference(task) {
  if (!gpuAvailable) {
    return { error: 'WebGPU 不可用', fallback: 'dom-only' };
  }

  switch (task.type) {
    case 'image-gen':
      return {
        status: 'model-required',
        message: '图片生成需要加载模型（约 2GB），暂未自动下载',
        models: ['sd-turbo (2.1GB)', 'tiny-sd (890MB)']
      };
    case 'benchmark':
      return await sendToGPUWorker({ action: 'benchmark' });
    case 'gpu-detect':
      return await sendToGPUWorker({ action: 'detect' });
    default:
      return { error: `不支持的任务类型: ${task.type}` };
  }
}

// ============================================================
// 能力检测（不依赖 navigator，用 static analysis）
// ============================================================

function detectCapabilities() {
  const caps = ['dom'];

  // 无法在 Service Worker 中检测 navigator.gpu 或 deviceMemory
  // 这些由 offscreen document 报告
  if (gpuAvailable) caps.push('webgpu', 'gpu-ready');

  caps.push('service-worker');

  return caps;
}

// ============================================================
// 令牌台账（本地 CRDT）
// ============================================================

class TokenLedger {
  constructor() {
    this.balances = {};  // { [from-to]: amount }
    this.nodeId = null;
  }

  async init() {
    this.nodeId = await getNodeId();
    const stored = await chrome.storage.local.get('clawnet_ledger');
    if (stored.clawnet_ledger) {
      this.balances = stored.clawnet_ledger;
    }
  }

  async save() {
    await chrome.storage.local.set({ clawnet_ledger: this.balances });
  }

  settle(from, to, amount) {
    const key = `${from}->${to}`;
    this.balances[key] = (this.balances[key] || 0) + amount;
    this.save();
  }

  netBalance(nodeId) {
    let credit = 0, debit = 0;
    for (const [key, amount] of Object.entries(this.balances)) {
      const [from, to] = key.split('->');
      if (to === nodeId) credit += amount;
      if (from === nodeId) debit += amount;
    }
    return credit - debit;
  }

  merge(other) {
    for (const [key, amount] of Object.entries(other)) {
      const current = this.balances[key] || 0;
      if (amount > current) {
        this.balances[key] = amount;
      }
    }
    this.save();
  }
}

const ledger = new TokenLedger();

// ============================================================
// 任务处理
// ============================================================

let taskQueue = [];

async function handleTaskMessage(data, fromPeer) {
  const task = JSON.parse(data);

  // 加入队列
  taskQueue.push(task);
  updateTaskCount();

  // 按令牌优先级排序
  taskQueue.sort((a, b) => (b.reward || 0) - (a.reward || 0));

  // 如果队列变长，开始执行
  if (taskQueue.length === 1) {
    processNextTask();
  }
}

async function processNextTask() {
  if (taskQueue.length === 0) return;

  const task = taskQueue.shift();
  updateTaskCount();

  // 更新状态
  updateStatus('computing');
  updateCurrentTask(task);

  let result;
  switch (task.type) {
    case 'dom-search':
      result = await executeDomSearch(task);
      break;
    case 'image-gen':
      result = await runInference(task);
      break;
    default:
      result = { error: `未知任务类型: ${task.type}` };
  }

  // 回传结果
  const peer = peers.get(task.from);
  if (peer) {
    const dc = peer.createDataChannel('clawnet-result');
    dc.onopen = () => dc.send(JSON.stringify({
      taskId: task.id,
      result
    }));
  }

  // 记账
  ledger.settle(task.from, await getNodeId(), task.reward || 1);

  updateStatus('idle');
  updateCurrentTask(null);

  // 处理下一个
  if (taskQueue.length > 0) {
    processNextTask();
  }
}

async function executeDomSearch(task) {
  // 通过 content script 执行 DOM 搜索
  const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
  if (tabs.length === 0) return { error: '未找到活动标签页' };

  try {
    const result = await chrome.tabs.sendMessage(tabs[0].id, {
      type: 'clawnet-task',
      task
    });
    return result;
  } catch (e) {
    return { error: `DOM 执行失败: ${e.message}` };
  }
}

// ============================================================
// 状态同步
// ============================================================

async function syncState() {
  // 定期通过 gossip 同步令牌台账
  // 广播本节点状态给所有已连接的 peer
  const id = await getNodeId();
  const state = {
    type: 'state-sync',
    nodeId: id,
    ledger: ledger.balances,
    capabilities: detectCapabilities(),
    queueLength: taskQueue.length
  };

  for (const [peerId, pc] of peers) {
    try {
      const dc = pc.createDataChannel('clawnet-sync');
      dc.onopen = () => {
        dc.send(JSON.stringify(state));
        dc.close();
      };
    } catch (e) {
      // 连接可能已断开
    }
  }
}

// ============================================================
// UI 更新
// ============================================================

let statusCallbacks = [];

function onStatusUpdate(cb) {
  statusCallbacks.push(cb);
}

function updateStatus(status) {
  chrome.storage.local.set({ clawnet_status: status });
  statusCallbacks.forEach(cb => cb(status));
}

function updateTaskCount() {
  chrome.storage.local.set({ clawnet_queue_length: taskQueue.length });
}

function updateCurrentTask(task) {
  chrome.storage.local.set({ clawnet_current_task: task ? task.description || task.type : null });
}

// ============================================================
// 监听 Popup 消息
// ============================================================

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.type === 'clawnet-execute-task') {
    executeUserTask(request.task).then(sendResponse);
    return true;  // 异步响应
  }
});

async function executeUserTask(task) {
  updateStatus('computing');
  updateCurrentTask(task);

  addLog(`执行任务: [${task.type}] ${(task.description || '').slice(0, 50)}`);

  let result;

  if (task.type === 'image-gen' && webgpuDevice) {
    result = await webgpuImageGen(webgpuDevice, task);
  } else if (task.type.startsWith('dom-')) {
    result = await executeDomTask(task);
  } else {
    result = { error: `未知任务类型: ${task.type}` };
  }

  // 记录任务完成
  const stats = await chrome.storage.local.get('clawnet_tasks_done');
  await chrome.storage.local.set({ clawnet_tasks_done: (stats.clawnet_tasks_done || 0) + 1 });

  updateStatus('idle');
  updateCurrentTask(null);
  addLog(`✅ 任务完成`);

  return result;
}

async function executeDomTask(task) {
  // 支持三种模式：指定URL、当前活动标签页、所有标签页
  let tab;

  if (task.url) {
    // 模式1：打开或切换到指定URL
    const tabs = await chrome.tabs.query({ url: task.url.includes('*') ? task.url : undefined });
    if (tabs.length > 0) {
      tab = tabs[0];
      await chrome.tabs.update(tab.id, { active: true });
    } else {
      tab = await chrome.tabs.create({ url: task.url });
      // 等待页面加载
      await new Promise(r => setTimeout(r, 2000));
    }
  } else {
    // 模式2：使用当前活跃标签页
    const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
    if (tabs.length === 0) return { error: '没有打开的标签页' };
    tab = tabs[0];
  }

  try {
    const result = await chrome.tabs.sendMessage(tab.id, {
      type: 'clawnet-task',
      task: { action: task.type.replace('dom-', ''), ...task.params, keyword: task.description }
    });
    return result || { status: 'no-response' };
  } catch (e) {
    return { error: `DOM 执行失败: ${e.message}（页面可能不支持 content script）` };
  }
}

function addLog(msg) {
  const now = new Date();
  const ts = `${now.getHours().toString().padStart(2,'0')}:${now.getMinutes().toString().padStart(2,'0')}:${now.getSeconds().toString().padStart(2,'0')}`;
  console.log(`[ClawNet ${ts}] ${msg}`);
}

// ============================================================
// WebGPU 检测
// ============================================================

async function detectWebGPU() {
  if (!navigator.gpu) {
    await chrome.storage.local.set({ clawnet_webgpu: false });
    addLog('WebGPU: 不可用（浏览器不支持或未开启）');
    return false;
  }

  try {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) {
      await chrome.storage.local.set({ clawnet_webgpu: false });
      addLog('WebGPU: 无可用适配器');
      return false;
    }

    const device = await adapter.requestDevice();
    const name = adapter.name || 'unknown GPU';
    addLog(`WebGPU: ✅ ${name}`);
    await chrome.storage.local.set({ clawnet_webgpu: true });
    return true;
  } catch (e) {
    addLog(`WebGPU: ❌ ${e.message}`);
    await chrome.storage.local.set({ clawnet_webgpu: false });
    return false;
  }
}

// ============================================================
// 启动
// ============================================================

chrome.runtime.onInstalled.addListener(async () => {
  await ledger.init();

  // 记录启动时间
  await chrome.storage.local.set({ clawnet_uptime: Date.now() });

  // 检测 WebGPU
  await detectWebGPU();

  // 连接信令服务器
  await connectSignaling();

  // 上报能力
  const caps = detectCapabilities();
  await chrome.storage.local.set({ clawnet_capabilities: caps });
  addLog(`能力: ${caps.join(', ')}`);

  // 定时同步
  chrome.alarms.create('clawnet-sync', { periodInMinutes: 5 });

  // 心跳
  chrome.alarms.create('clawnet-heartbeat', { periodInMinutes: 1 });
});

chrome.alarms.onAlarm.addListener((alarm) => {
  if (alarm.name === 'clawnet-sync') syncState();
  if (alarm.name === 'clawnet-heartbeat') {
    updateStatus(peers.size > 0 ? 'connected' : 'disconnected');
  }
});

// 导出供 popup 使用
self.__CLAWNET__ = {
  getNodeId,
  getPeers: () => peers.size,
  getBalance: () => ledger.netBalance(nodeId),
  getStatus: () => chrome.storage.local.get('clawnet_status').then(s => s.clawnet_status || 'init'),
  getQueueLength: () => taskQueue.length,
  onStatusUpdate
};

addLog('🚀 ClawNet 后台服务已启动');
