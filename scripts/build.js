#!/usr/bin/env node

/**
 * ClawNet 构建脚本
 * 
 * 用法: node scripts/build.js
 * 
 * 1. 生成 PNG 图标（从 SVG）
 * 2. 验证 manifest 完整性
 * 3. 打包成 .zip（可选）
 */

const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const EXTENSION_DIR = path.join(ROOT, 'extension');
const BUILD_DIR = path.join(ROOT, 'dist');

console.log('🦞 ClawNet 构建脚本\n');

// 1. 验证文件完整性
console.log('📋 验证文件完整性...');
const required = [
  'manifest.json',
  'src/background/service-worker.js',
  'src/popup/popup.html',
  'src/popup/popup.js',
  'src/content/content.js',
  'public/icon-16.svg',
  'public/icon-48.svg',
];

let allOk = true;
for (const file of required) {
  const p = path.join(EXTENSION_DIR, file);
  const exists = fs.existsSync(p);
  if (!exists) {
    console.log(`  ❌ ${file}`);
    allOk = false;
  } else {
    const size = fs.statSync(p).size;
    console.log(`  ✅ ${file} (${size} bytes)`);
  }
}

if (!allOk) {
  console.log('\n❌ 文件不完整，请检查后重试');
  process.exit(1);
}

// 2. 创建构建目录
if (!fs.existsSync(BUILD_DIR)) {
  fs.mkdirSync(BUILD_DIR, { recursive: true });
}

// 3. 生成 PNG 图标（使用 canvas，如果可用）
console.log('\n🖼 生成 PNG 图标...');
try {
  // 尝试用 Canvas 生成 PNG
  const { createCanvas } = require('canvas');
  const sizes = [16, 48, 128];

  for (const size of sizes) {
    const canvas = createCanvas(size, size);
    const ctx = canvas.getContext('2d');

    // 背景圆
    ctx.beginPath();
    ctx.arc(size / 2, size / 2, size / 2 - 1, 0, Math.PI * 2);
    ctx.fillStyle = '#161b22';
    ctx.fill();
    ctx.strokeStyle = '#58a6ff';
    ctx.lineWidth = size >= 48 ? 2 : 1;
    ctx.stroke();

    // 'C' 字母
    ctx.fillStyle = '#58a6ff';
    ctx.font = `bold ${Math.floor(size * 0.6)}px sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('C', size / 2, size / 2 + 1);

    const buffer = canvas.toBuffer('image/png');
    const outPath = path.join(EXTENSION_DIR, `public/icon-${size}.png`);
    fs.writeFileSync(outPath, buffer);
    console.log(`  ✅ icon-${size}.png (${buffer.length} bytes)`);
  }
} catch (e) {
  console.log(`  ⚠️  Canvas 不可用（${e.message}）`);
  console.log('  将在不存在 PNG 时使用 SVG 替代');
}

// 4. 打包
const zipPath = path.join(BUILD_DIR, 'clawnet-extension.zip');
console.log(`\n📦 打包到 ${zipPath}...`);

const archiver = require('archiver');
if (fs.existsSync(zipPath)) fs.unlinkSync(zipPath);

try {
  const output = fs.createWriteStream(zipPath);
  const archive = archiver('zip', { zlib: { level: 9 } });
  archive.pipe(output);
  archive.directory(EXTENSION_DIR, false);
  await archive.finalize();

  const stats = fs.statSync(zipPath);
  console.log(`  ✅ 打包完成 (${(stats.size / 1024).toFixed(1)} KB)`);
} catch (e) {
  console.log(`  ⚠️  archiver 未安装，跳过打包`);
  console.log('  手动打包: cd extension && zip -r ../dist/clawnet-extension.zip .');
}

console.log(`\n✅ 构建完成`);
