/**
 * ClawNet — WebRTC 信令服务器
 * 
 * 轻量级 WebSocket 信令，用于浏览器节点间的 P2P 发现。
 * 只负责传递连接信息，不转发任何任务数据。
 */

const WebSocket = require('ws');
const http = require('http');
const fs = require('fs');
const path = require('path');

// ============================================================
// 配置
// ============================================================

const PORT = process.env.PORT || 8765;
const MAX_PEERS_PER_IP = 10;
const CLEANUP_INTERVAL = 30000;  // 30秒清理一次
const PEER_TIMEOUT = 60000;       // 60秒无心跳则断开

// ============================================================
// 节点注册表
// ============================================================

class PeerRegistry {
  constructor() {
    this.peers = new Map();  // nodeId -> { ws, info, lastSeen }
  }

  register(nodeId, ws, info) {
    // 清理旧连接
    if (this.peers.has(nodeId)) {
      const old = this.peers.get(nodeId);
      if (old.ws !== ws) {
        old.ws.close(1000, 'replaced');
      }
    }

    this.peers.set(nodeId, {
      ws,
      info: {
        nodeId,
        capabilities: info.capabilities || [],
        joinedAt: Date.now()
      },
      lastSeen: Date.now()
    });

    console.log(`[+] 节点上线: ${nodeId.slice(0, 8)}... (${(info.capabilities || []).join(', ')})`);
    this.broadcastPeerList();
  }

  unregister(nodeId) {
    this.peers.delete(nodeId);
    console.log(`[-] 节点离线: ${nodeId.slice(0, 8)}...`);
    this.broadcastPeerList();
  }

  heartbeat(nodeId) {
    const peer = this.peers.get(nodeId);
    if (peer) {
      peer.lastSeen = Date.now();
    }
  }

  getPeerList() {
    return Array.from(this.peers.values()).map(p => ({
      nodeId: p.info.nodeId,
      capabilities: p.info.capabilities,
      joinedAt: p.info.joinedAt
    }));
  }

  broadcastPeerList() {
    const list = this.getPeerList();
    const message = JSON.stringify({ type: 'peer-list', peers: list });

    for (const [id, peer] of this.peers) {
      if (peer.ws.readyState === WebSocket.OPEN) {
        peer.ws.send(message);
      }
    }
  }

  cleanup() {
    const now = Date.now();
    for (const [id, peer] of this.peers) {
      if (now - peer.lastSeen > PEER_TIMEOUT) {
        peer.ws.close(1000, 'timeout');
        this.unregister(id);
      }
    }
  }

  getStats() {
    return {
      online: this.peers.size,
      list: this.getPeerList()
    };
  }
}

const registry = new PeerRegistry();

// ============================================================
// HTTP 服务（健康检查 + 状态页）
// ============================================================

const server = http.createServer((req, res) => {
  if (req.url === '/health') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ status: 'ok', time: Date.now() }));
    return;
  }

  if (req.url === '/stats') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(registry.getStats(), null, 2));
    return;
  }

  res.writeHead(200, { 'Content-Type': 'text/html' });
  res.end(`<!DOCTYPE html>
<html><body style="font-family: sans-serif; padding: 2rem;">
  <h1>🦞 ClawNet 信令服务器</h1>
  <p>在线节点: <strong>${registry.peers.size}</strong></p>
  <p><a href="/stats">节点状态</a> · <a href="/health">健康检查</a></p>
  <pre>${JSON.stringify(registry.getStats(), null, 2)}</pre>
</body></html>`);
});

// ============================================================
// WebSocket 信令
// ============================================================

const wss = new WebSocket.Server({ server });

wss.on('connection', (ws, req) => {
  let nodeId = null;
  const clientIp = req.socket.remoteAddress;

  ws.on('message', (data) => {
    try {
      const msg = JSON.parse(data);

      switch (msg.type) {
        case 'register':
          nodeId = msg.nodeId;
          registry.register(nodeId, ws, msg);
          break;

        case 'heartbeat':
          if (nodeId) registry.heartbeat(nodeId);
          break;

        case 'offer':
        case 'answer':
        case 'ice-candidate':
          // 转发给目标节点
          if (msg.target) {
            const target = registry.peers.get(msg.target);
            if (target && target.ws.readyState === WebSocket.OPEN) {
              target.ws.send(JSON.stringify({
                ...msg,
                from: nodeId
              }));
            }
          }
          break;

        default:
          console.warn(`[?] 未知消息类型: ${msg.type}`);
      }
    } catch (e) {
      console.error(`[!] 消息解析失败: ${e.message}`);
    }
  });

  ws.on('close', () => {
    if (nodeId) registry.unregister(nodeId);
  });

  ws.on('error', (err) => {
    console.error(`[!] WebSocket 错误: ${err.message}`);
    if (nodeId) registry.unregister(nodeId);
  });
});

// ============================================================
// 定期清理
// ============================================================

setInterval(() => registry.cleanup(), CLEANUP_INTERVAL);

// ============================================================
// 启动
// ============================================================

server.listen(PORT, () => {
  console.log(`🦞 ClawNet 信令服务器`);
  console.log(`   HTTP:   http://0.0.0.0:${PORT}`);
  console.log(`   WS:     ws://0.0.0.0:${PORT}`);
  console.log(`   Health: http://0.0.0.0:${PORT}/health`);
  console.log(`   Stats:  http://0.0.0.0:${PORT}/stats`);
});
