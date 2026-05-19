/**
 * ClawNet — WebGPU 推理引擎
 * 
 * 使用 WebGPU 在浏览器中运行 AI 模型。
 * 
 * 设计目标：
 *   1. 自动检测 GPU 能力（内存、Shader 支持）
 *   2. 支持小模型推理（分类、embedding）
 *   3. 可扩展到大模型（LLM、图片生成）
 *   4. 纯浏览器端，不需要服务器
 */

export class WebGPUSession {
  constructor() {
    this.device = null;
    this.adapter = null;
    this.info = {
      name: 'unknown',
      memory: 0,
      features: [],
      maxBufferSize: 0,
      maxComputeInvocationsPerWG: 0
    };
    this.ready = false;
  }

  // ============================================================
  // 初始化
  // ============================================================

  async init() {
    if (this.ready) return true;

    if (!navigator.gpu) {
      console.warn('[ClawNet WebGPU] navigator.gpu 不可用');
      return false;
    }

    this.adapter = await navigator.gpu.requestAdapter({
      powerPreference: 'high-performance'
    });

    if (!this.adapter) {
      console.warn('[ClawNet WebGPU] 无法获取 GPU Adapter');
      return false;
    }

    // 收集 GPU 信息
    const limits = this.adapter.limits;
    const adapterInfo = {};

    // 尝试获取 GPU 品牌/型号信息（Chrome 113+）
    try {
      const info = await this.adapter.requestAdapterInfo();
      adapterInfo.vendor = info.vendor;
      adapterInfo.architecture = info.architecture;
      adapterInfo.device = info.device;
      adapterInfo.description = info.description;
    } catch {
      adapterInfo.vendor = 'unknown';
    }

    this.info.name = adapterInfo.description || adapterInfo.vendor || this.adapter.name || 'unknown';
    this.info.memory = this.adapter.limits.maxStorageBufferBindingSize ? 
      Math.log2(this.adapter.limits.maxStorageBufferBindingSize) : 0;
    this.info.maxBufferSize = this.adapter.limits.maxBufferSize;
    this.info.maxComputeInvocationsPerWG = this.adapter.limits.maxComputeInvocationsPerWorkgroup;

    // 检测特性支持
    const features = this.adapter.features;
    this.info.features = Array.from(features || []).map(f => f.toString());

    // 创建设备
    this.device = await this.adapter.requestDevice({
      requiredFeatures: [],
      requiredLimits: {}
    });

    this.device.lost.then((reason) => {
      console.warn(`[ClawNet WebGPU] 设备丢失: ${reason}`);
      this.ready = false;
    });

    this.ready = true;
    console.log(`[ClawNet WebGPU] ✅ ${this.info.name}`);
    return true;
  }

  // ============================================================
  // 能力检测
  // ============================================================

  getCapabilities() {
    const caps = ['webgpu'];

    // 根据 GPU 信息判断等级
    const memScore = this.info.memory;
    if (memScore > 30) {  // > 1GB
      caps.push('gpu-high');
    } else if (memScore > 24) {  // > 16MB
      caps.push('gpu-medium');
    } else {
      caps.push('gpu-low');
    }

    return caps;
  }

  // ============================================================
  // 基准测试：计算吞吐
  // ============================================================

  async benchmark() {
    if (!this.ready) return null;

    const size = 4096;
    const workgroups = Math.ceil(size / 64);

    // 创建设备缓冲区
    const buffer = this.device.createBuffer({
      size: size * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    });

    // 写初始数据
    const initialData = new Float32Array(size).fill(1.0);
    this.device.queue.writeBuffer(buffer, 0, initialData);

    // 计算着色器：逐元素平方
    const shaderCode = `
      @group(0) @binding(0) var<storage, read_write> data: array<f32>;

      @compute @workgroup_size(64)
      fn main(@builtin(global_invocation_id) id: vec3<u32>) {
        let i = id.x;
        if (i < ${size}u) {
          data[i] = data[i] * data[i];
        }
      }
    `;

    const shader = this.device.createShaderModule({ code: shaderCode });

    const pipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: { module: shader, entryPoint: 'main' }
    });

    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer } }]
    });

    // 计时
    const iterations = 100;
    const start = performance.now();

    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);

    for (let i = 0; i < iterations; i++) {
      pass.dispatchWorkgroups(workgroups);
    }

    pass.end();
    const commands = encoder.finish();
    this.device.queue.submit([commands]);

    // 等待完成
    await this.device.queue.onSubmittedWorkDone();

    const elapsed = performance.now() - start;
    const ops = (size * iterations) / (elapsed / 1000);
    const flops = ops;  // 每次操作是一个 FMA 级别的运算

    this.info.benchmark = {
      opsPerSecond: Math.round(ops / 1e6) + 'M',
      flops: Math.round(flops / 1e9) + ' GFLOPS',
      latencyMs: Math.round(elapsed)
    };

    buffer.destroy();
    return this.info.benchmark;
  }

  // ============================================================
  // 运行计算着色器
  // ============================================================

  async runComputeShader(shaderCode, bindings, workgroupCount, workgroupSize = [64, 1, 1]) {
    if (!this.ready) throw new Error('WebGPU 未就绪');

    const shader = this.device.createShaderModule({ code: shaderCode });

    const pipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: { module: shader, entryPoint: 'main' }
    });

    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: bindings.map((b, i) => ({
        binding: i,
        resource: b.resource
      }))
    });

    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(...workgroupCount);
    pass.end();

    const commands = encoder.finish();
    this.device.queue.submit([commands]);
    await this.device.queue.onSubmittedWorkDone();
  }

  // ============================================================
  // 矩阵乘法（ML 推理的核心算子）
  // ============================================================

  async matmul(A: Float32Array, B: Float32Array, M: number, K: number, N: number) {
    if (!this.ready) throw new Error('WebGPU 未就绪');

    const shaderCode = `
      @group(0) @binding(0) var<storage, read> A: array<f32>;
      @group(0) @binding(1) var<storage, read> B: array<f32>;
      @group(0) @binding(2) var<storage, read_write> C: array<f32>;

      @compute @workgroup_size(8, 8)
      fn main(@builtin(global_invocation_id) id: vec3<u32>) {
        let row = id.x;
        let col = id.y;
        if (row >= ${M}u || col >= ${N}u) { return; }

        var sum = 0.0;
        for (var k = 0u; k < ${K}u; k++) {
          sum = sum + A[row * ${K}u + k] * B[k * ${N}u + col];
        }
        C[row * ${N}u + col] = sum;
      }
    `;

    // 创建 GPU 缓冲区
    const bufA = this.device.createBuffer({
      size: A.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    this.device.queue.writeBuffer(bufA, 0, A);

    const bufB = this.device.createBuffer({
      size: B.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    this.device.queue.writeBuffer(bufB, 0, B);

    const bufC = this.device.createBuffer({
      size: M * N * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });

    await this.runComputeShader(
      shaderCode,
      [
        { resource: { buffer: bufA } },
        { resource: { buffer: bufB } },
        { resource: { buffer: bufC } }
      ],
      [M, N, 1],
      [8, 8, 1]
    );

    // 读回结果
    const readBuf = this.device.createBuffer({
      size: M * N * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });

    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(bufC, 0, readBuf, 0, readBuf.size);
    this.device.queue.submit([encoder.finish()]);

    await readBuf.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(readBuf.getMappedRange());
    const out = Float32Array.from(result);
    readBuf.unmap();

    bufA.destroy();
    bufB.destroy();
    bufC.destroy();
    readBuf.destroy();

    return out;
  }

  // ============================================================
  // Release
  // ============================================================

  destroy() {
    if (this.device) {
      this.device.destroy();
      this.device = null;
    }
    this.ready = false;
  }
}

// ============================================================
// 快速 API
// ============================================================

let _session = null;

export async function getWebGPU() {
  if (_session && _session.ready) return _session;

  _session = new WebGPUSession();
  const ok = await _session.init();
  if (!ok) {
    _session = null;
    return null;
  }
  return _session;
}

export async function runBenchmark() {
  const gpu = await getWebGPU();
  if (!gpu) return { error: 'WebGPU 不可用' };
  return await gpu.benchmark();
}

export async function matmul(A, B, M, K, N) {
  const gpu = await getWebGPU();
  if (!gpu) return { error: 'WebGPU 不可用' };
  return await gpu.matmul(A, B, M, K, N);
}
