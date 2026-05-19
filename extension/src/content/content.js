/**
 * ClawNet — 内容脚本
 * 
 * 在用户访问的每个页面中注入，负责：
 *   1. 监听来自 background worker 的 DOM 操作任务
 *   2. 执行页面内搜索、数据提取、内容精炼
 *   3. 将结果回传给 background worker
 */

// ============================================================
// 监听 background 发来的任务
// ============================================================

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.type === 'clawnet-task') {
    executeTask(request.task)
      .then(result => sendResponse(result))
      .catch(error => sendResponse({ error: error.message }));
    return true;  // 保持通道开放
  }

  if (request.type === 'clawnet-ping') {
    sendResponse({ alive: true, url: window.location.href });
    return true;
  }
});

// ============================================================
// 任务执行器
// ============================================================

async function executeTask(task) {
  switch (task.action) {
    case 'search':
      return await domSearch(task);
    case 'extract':
      return await domExtract(task);
    case 'summarize':
      return await domSummarize(task);
    case 'monitor':
      return await domMonitor(task);
    case 'fill':
      return await domFill(task);
    case 'click':
      return await domClick(task);
    default:
      throw new Error(`未知操作: ${task.action}`);
  }
}

// ============================================================
// 搜索页面内容
// ============================================================

async function domSearch(task) {
  const { keyword, selector } = task;

  // 全文搜索
  const results = [];
  const walker = document.createTreeWalker(
    document.body,
    NodeFilter.SHOW_TEXT,
    null,
    false
  );

  let node;
  while (node = walker.nextNode()) {
    const text = node.textContent.trim();
    if (text.length > 10 && text.includes(keyword)) {
      results.push(text.slice(0, 200));
      if (results.length >= (task.maxResults || 10)) break;
    }
  }

  return {
    type: 'search',
    keyword,
    url: window.location.href,
    title: document.title,
    results,
    totalFound: results.length
  };
}

// ============================================================
// 提取结构化数据
// ============================================================

async function domExtract(task) {
  const { selectors } = task;
  const result = {};

  for (const [key, sel] of Object.entries(selectors)) {
    try {
      const elements = document.querySelectorAll(sel);
      result[key] = Array.from(elements).slice(0, 20).map(el => {
        // 优先取 data-* 属性，其次 textContent
        return el.dataset.value || el.textContent.trim().slice(0, 500);
      });
    } catch (e) {
      result[key] = [];
    }
  }

  return {
    type: 'extract',
    url: window.location.href,
    title: document.title,
    data: result
  };
}

// ============================================================
// 精炼页面摘要
// ============================================================

async function domSummarize(task) {
  const { maxChars } = task;

  // 获取页面主要内容
  const article = document.querySelector('article') ||
                  document.querySelector('[class*="content"]') ||
                  document.querySelector('[class*="article"]') ||
                  document.querySelector('main');

  let text = '';
  if (article) {
    text = article.textContent.trim();
  } else {
    // 降级：取 body 文本
    text = document.body.textContent.trim();
  }

  if (maxChars && text.length > maxChars) {
    text = text.slice(0, maxChars);
  }

  return {
    type: 'summarize',
    url: window.location.href,
    title: document.title,
    meta: {
      description: document.querySelector('meta[name="description"]')?.content || '',
      keywords: document.querySelector('meta[name="keywords"]')?.content || '',
      author: document.querySelector('[rel="author"]')?.textContent || '',
    },
    text: text
  };
}

// ============================================================
// 监控页面变化（用于订阅检测）
// ============================================================

async function domMonitor(task) {
  const { selector, checkInterval } = task;

  return new Promise((resolve) => {
    const check = () => {
      const elements = document.querySelectorAll(selector);
      if (elements.length > 0) {
        resolve({
          type: 'monitor',
          url: window.location.href,
          found: true,
          count: elements.length,
          items: Array.from(elements).slice(0, 10).map(el => ({
            text: el.textContent.trim().slice(0, 200),
            href: el.href || el.closest('a')?.href || ''
          }))
        });
      } else {
        setTimeout(check, checkInterval || 5000);
      }
    };
    check();
  });
}

// ============================================================
// 填写表单
// ============================================================

async function domFill(task) {
  const { fields } = task;
  const filled = [];

  for (const [selector, value] of Object.entries(fields)) {
    try {
      const el = document.querySelector(selector);
      if (el) {
        el.value = value;
        el.dispatchEvent(new Event('input', { bubbles: true }));
        el.dispatchEvent(new Event('change', { bubbles: true }));
        filled.push({ selector, status: 'filled' });
      } else {
        filled.push({ selector, status: 'not-found' });
      }
    } catch (e) {
      filled.push({ selector, status: 'error', error: e.message });
    }
  }

  return { type: 'fill', filled };
}

// ============================================================
// 点击元素
// ============================================================

async function domClick(task) {
  const { selector } = task;

  try {
    const el = document.querySelector(selector);
    if (!el) return { type: 'click', selector, status: 'not-found' };

    el.click();
    return { type: 'click', selector, status: 'clicked' };
  } catch (e) {
    return { type: 'click', selector, status: 'error', error: e.message };
  }
}
