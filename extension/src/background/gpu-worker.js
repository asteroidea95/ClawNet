/**
 * ClawNet — GPU Worker（Offscreen Document）
 * 
 * 在独立的隐藏文档中运行 WebGPU 推理。
 * 与 service worker 通过 chrome.runtime 通信。
 * 
 * 为什么不在 service worker 里跑？
 * → Service worker 没有 navigator.gpu
 */

import { WebGPUSession } from '../lib/webgpu.js';

let gpu = null;

// ============================================================
// 收到来自 service worker 的消息
// ============================================================

chrome.runtime.onMessage.addListener(async (request, sender, sendResponse) => {
  if (request.target !== 'gpu-worker') return false;

  try {
    const result = await handleMessage(request);
    sendResponse({ ok: true, result });
  } catch (e) {
    sendResponse({ ok: false, error: e.message });
  }
  return true;  // 异步响应
});

async function handleMessage(request) {
  switch (request.action) {
    case 'init':
      return await initGPU();
    case 'benchmark':
      return await runBenchmark();
    case 'matmul':
      return await runMatmul(request);
    case 'detect':
      return detectCapabilities();
    case 'ping':
      return { alive: true };
    default:
      throw new Error(`未知 GPU 操作: ${request.action}`);
  }
}

// ============================================================
// GPU 操作
// ============================================================

async function initGPU() {
  if (gpu && gpu.ready) return { ready: true, name: gpu.info.name };

  gpu = new WebGPUSession();
  const ok = await gpu.init();

  if (!ok) {
    close();  // 没有 GPU，关闭 offscreen 文档
    return { ready: false };
  }

  document.getElementById('status').textContent =
    `🦞 GPU: ${gpu.info.name}`;

  // 通知 service worker
  chrome.runtime.sendMessage({
    type: 'gpu-status',
    ready: true,
    info: gpu.info
  });

  return { ready: true, name: gpu.info.name, info: gpu.info };
}

async function runBenchmark() {
  if (!gpu || !gpu.ready) throw new Error('GPU 未就绪');
  const result = await gpu.benchmark();

  chrome.runtime.sendMessage({
    type: 'gpu-benchmark',
    result
  });

  return result;
}

async function runMatmul(request) {
  if (!gpu || !gpu.ready) throw new Error('GPU 未就绪');
  return await gpu.matmul(request.A, request.B, request.M, request.K, request.N);
}

function detectCapabilities() {
  if (!gpu || !gpu.ready) {
    return { webgpu: false };
  }
  return {
    webgpu: true,
    caps: gpu.getCapabilities(),
    info: gpu.info
  };
}

// ============================================================
// 启动
// ============================================================

document.getElementById('status').textContent = '🦞 ClawNet GPU Worker 已加载';
initGPU();
