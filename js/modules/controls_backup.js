// =============================================================================
//  WebGPU Particle System - קוד צד לקוח (JavaScript)
//
//  קוד זה מנהל את מערכת החלקיקים ב-WebGPU:
//  - מאתחל את ה-device וה-pipelines.
//  - יוצר ומנהל את מאגרי הנתונים (buffers) עבור חלקיקים, סיבים ומשתנים גלובליים (uniforms).
//  - מריץ את שלבי החישוב (compute passes) בכל פריים.
//  - מספק ממשק לשליטה על פרמטרים של הסימולציה מהאפליקציה הראשית.
// =============================================================================

// --- קבועים עבור מבני נתונים (תואמים ל-WGSL Shaders) ---

// גודל מבנה חלקיק (Particle) ב-floats (מספרים עשרוניים)
// position(vec3) + velocity(vec3) + color(vec3) + life(f32) + initialTargetPos(vec3)
// כל vec3 תופס 4 floats בזיכרון בגלל ריפוד (padding) של 16 בתים.
// (3*4) + 1 + (3) = 13 floats? לא. (4+4+4) + 1 + 4 = 17? לא.
// נחשב בבתים: pos(12)+pad(4) + vel(12)+pad(4) + color(12)+pad(4) + life(4) + target(12)+pad(4)
// סה"כ: 16 + 16 + 16 + 4 + 16 = 68. זה לא מתחלק ב-4.
// בוא נבדוק את ה-layout ב-WGSL:
// struct Particle { position: vec3f, velocity: vec3f, color: vec3f, life: f32, initialTargetPos: vec3f }
// position: offset 0, size 12, align 16
// velocity: offset 16, size 12, align 16
// color: offset 32, size 12, align 16
// life: offset 44, size 4, align 4
// initialTargetPos: offset 48, size 12, align 16
// גודל כולל של המבנה: 60 בתים. 60 / 4 = 15 floats.
const PARTICLE_FLOAT_COUNT = 15;

// גודל מבנה סיב (Fiber) ב-floats
// start(u32) + end(u32) + strength(f32) + length(f32) + isActive(f32)
// 2 u32s ו-3 f32s, כל אחד 4 בתים. סה"כ 5 * 4 = 20 בתים.
const FIBER_FLOAT_COUNT = 5;

/**
 * מחלקה המנהלת את כל הלוגיקה והמשאבים של מערכת החלקיקים ב-WebGPU.
 */
class WebGPUParticleSystem {
  constructor() {
    this.device = null;
    this.particleCount = 25000;
    this.maxFibers = 50000;
    this.fibersEnabled = false;

    // Pipelines
    this.particleComputePipeline = null;
    this.fiberComputePipeline = null;

    // Buffers
    this.buffers = {
      particles: null,
      morphTargets: null,
      uniforms: null,
      fibers: null,
      fiberUniforms: null,
    };

    // Bind Groups
    this.bindGroups = {
      particleCompute: null,
      fiberCompute: null,
    };
  }

  /**
   * מאתחל את המערכת, יוצר את כל המשאבים הנדרשים.
   * @param {GPUDevice} device - ה-device של WebGPU.
   * @returns {Promise<boolean>} - מחזיר true אם האתחול הצליח.
   */
  async initialize(device) {
    if (!device) {
      console.error("WebGPU device is not provided.");
      return false;
    }
    this.device = device;

    await this.createPipelines();
    await this.createBuffers();
    this.createBindGroups();

    console.log('WebGPU Particle System initialized successfully.');
    return true;
  }

  /**
   * יוצר את ה-compute pipelines מה-WGSL shaders.
   */
  async createPipelines() {
    const particleShaderModule = this.device.createShaderModule({ code: particleComputeShader });
    this.particleComputePipeline = await this.device.createComputePipelineAsync({
      layout: 'auto',
      compute: {
        module: particleShaderModule,
        entryPoint: 'main',
      },
    });

    const fiberShaderModule = this.device.createShaderModule({ code: fiberComputeShader });
    this.fiberComputePipeline = await this.device.createComputePipelineAsync({
      layout: 'auto',
      compute: {
        module: fiberShaderModule,
        entryPoint: 'main',
      },
    });
  }

  /**
   * יוצר את כל מאגרי הנתונים (GPU Buffers).
   */
  async createBuffers() {
    // מאגר לחלקיקים
    const particleBufferSize = this.particleCount * PARTICLE_FLOAT_COUNT * 4; // 4 בתים ל-float
    this.buffers.particles = this.device.createBuffer({
      size: particleBufferSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });

    // מאגר למטרות מורפינג
    this.buffers.morphTargets = this.device.createBuffer({
      size: this.particleCount * 3 * 4, // vec3 לכל חלקיק
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // מאגר למשתנים גלובליים (uniforms) של סימולציית החלקיקים
    // גודל של 20 floats כדי להכיל את כל הפרמטרים עם ריפוד.
    this.buffers.uniforms = this.device.createBuffer({
      size: 20 * 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // מאגר לסיבים
    const fiberBufferSize = this.maxFibers * FIBER_FLOAT_COUNT * 4;
    this.buffers.fibers = this.device.createBuffer({
      size: fiberBufferSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // מאגר למשתנים גלובליים של סימולציית הסיבים
    this.buffers.fiberUniforms = this.device.createBuffer({
      size: 4 * 4, // 4 floats
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // אתחול נתונים ראשוני
    this.initializeParticleData();
  }

  /**
   * יוצר את קבוצות הקישור (Bind Groups) שמקשרות בין המאגרים ל-pipelines.
   */
  createBindGroups() {
    this.bindGroups.particleCompute = this.device.createBindGroup({
      layout: this.particleComputePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.buffers.particles } },
        { binding: 1, resource: { buffer: this.buffers.uniforms } },
        { binding: 2, resource: { buffer: this.buffers.morphTargets } },
      ],
    });

    this.bindGroups.fiberCompute = this.device.createBindGroup({
      layout: this.fiberComputePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.buffers.fibers } },
        { binding: 1, resource: { buffer: this.buffers.particles } },
        { binding: 2, resource: { buffer: this.buffers.fiberUniforms } },
      ],
    });
  }

  /**
   * מאתחל את הנתונים ההתחלתיים של החלקיקים וכותב אותם למאגר.
   */
  initializeParticleData() {
    const particleData = new Float32Array(this.particleCount * PARTICLE_FLOAT_COUNT);

    for (let i = 0; i < this.particleCount; i++) {
      const offset = i * PARTICLE_FLOAT_COUNT;
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      const radius = Math.random() * 5;

      // Position (x, y, z) - יישור ל-16 בתים (4 floats)
      particleData[offset + 0] = radius * Math.sin(phi) * Math.cos(theta);
      particleData[offset + 1] = radius * Math.sin(phi) * Math.sin(theta);
      particleData[offset + 2] = radius * Math.cos(phi);
      // particleData[offset + 3] is padding

      // Velocity (vx, vy, vz) - יישור ל-16 בתים
      particleData[offset + 4] = (Math.random() - 0.5) * 0.1;
      particleData[offset + 5] = (Math.random() - 0.5) * 0.1;
      particleData[offset + 6] = (Math.random() - 0.5) * 0.1;
      // particleData[offset + 7] is padding

      // Color (r, g, b) - יישור ל-16 בתים
      const color = this.hsl2rgb((i / this.particleCount) * 360, 80, 60);
      particleData[offset + 8] = color.r;
      particleData[offset + 9] = color.g;
      particleData[offset + 10] = color.b;
      // particleData[offset + 11] is padding

      // Life
      particleData[offset + 12] = 1.0 + Math.random() * 2.0;

      // Initial Target Position (tx, ty, tz)
      particleData[offset + 13] = particleData[offset + 0];
      particleData[offset + 14] = particleData[offset + 1];
      // particleData[offset + 15] is padding - wait, the struct is 60 bytes. 15 floats.
      // pos(3)+pad(1), vel(3)+pad(1), color(3)+pad(1), life(1), target(3)
      // pos(0,1,2), vel(4,5,6), color(8,9,10), life(12), target(13,14,15) -> This is 16 floats.
      // Let's recheck the layout from the improved shader:
      // pos(off 0), vel(off 16), color(off 32), life(off 44), target(off 48)
      // In floats: pos(off 0), vel(off 4), color(off 8), life(off 11), target(off 12)
      // This matches the 15 floats count.
      particleData[offset + 12] = 1.0 + Math.random() * 2.0; // Life
      particleData[offset + 13] = particleData[offset + 0]; // initialTargetPos.x
      particleData[offset + 14] = particleData[offset + 1]; // initialTargetPos.y
      // The last element is at index 14, so the size is 15 floats. Correct.
    }

    this.device.queue.writeBuffer(this.buffers.particles, 0, particleData);
  }

  /**
   * מעדכן את המשתנים הגלובליים (uniforms) לפני הרצת החישוב.
   * @param {number} time - זמן הסימולציה הכולל.
   * @param {number} deltaTime - הזמן שעבר מאז הפריים האחרון.
   * @param {object} params - אובייקט עם פרמטרים לשליטה בסימולציה.
   */
  updateUniforms(time, deltaTime, params) {
    const uniformData = new Float32Array([
      time,
      deltaTime,
      this.particleCount,
      params.gravity,
      params.turbulence,
      params.attraction,
      params.damping,
      params.mousePos.x,
      params.mousePos.y,
      params.hoverRadius,
      params.mouseStrength,
      params.morphProgress,
      params.respawnRate,
      ...params.slowColor, // r, g, b
      0.0, // padding
      ...params.fastColor, // r, g, b
    ]);
    this.device.queue.writeBuffer(this.buffers.uniforms, 0, uniformData);

    if (this.fibersEnabled) {
      const fiberUniformData = new Float32Array([
        params.connectionDistance,
        params.fiberStrength,
        params.fiberDamping,
        deltaTime,
      ]);
      this.device.queue.writeBuffer(this.buffers.fiberUniforms, 0, fiberUniformData);
    }
  }

  /**
   * מריץ את שלבי החישוב (compute passes) על ה-GPU.
   */
  compute() {
    const commandEncoder = this.device.createCommandEncoder();

    // שלב 1: עדכון חלקיקים
    const particlePass = commandEncoder.beginComputePass();
    particlePass.setPipeline(this.particleComputePipeline);
    particlePass.setBindGroup(0, this.bindGroups.particleCompute);
    const workgroupsX = Math.ceil(this.particleCount / 64);
    particlePass.dispatchWorkgroups(workgroupsX);
    particlePass.end();

    // שלב 2: עדכון סיבים (אם מופעל)
    if (this.fibersEnabled) {
      const fiberPass = commandEncoder.beginComputePass();
      fiberPass.setPipeline(this.fiberComputePipeline);
      fiberPass.setBindGroup(0, this.bindGroups.fiberCompute);
      const fiberWorkgroupsX = Math.ceil(this.maxFibers / 64);
      fiberPass.dispatchWorkgroups(fiberWorkgroupsX);
      fiberPass.end();
    }

    this.device.queue.submit([commandEncoder.finish()]);
  }

  /**
   * קורא את נתוני החלקיקים מה-GPU בחזרה ל-CPU.
   * @returns {Promise<Float32Array>} - מערך עם נתוני החלקיקים.
   */
  async readParticleData() {
    const stagingBuffer = this.device.createBuffer({
      size: this.buffers.particles.size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const commandEncoder = this.device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(
      this.buffers.particles, 0,
      stagingBuffer, 0,
      this.buffers.particles.size
    );
    this.device.queue.submit([commandEncoder.finish()]);

    await stagingBuffer.mapAsync(GPUMapMode.READ);
    const data = new Float32Array(stagingBuffer.getMappedRange().slice(0));
    stagingBuffer.unmap();
    stagingBuffer.destroy();

    return data;
  }

  /**
   * מאפשר או מנטרל את סימולציית הסיבים.
   * @param {boolean} enabled - האם להפעיל את הסיבים.
   */
  enableFibers(enabled) {
    if (this.fibersEnabled === enabled) return;
    this.fibersEnabled = enabled;
    if (enabled) {
      this.initializeFibers();
    }
  }

  /**
   * מאתחל את חיבורי הסיבים באופן אקראי.
   */
  initializeFibers() {
    const fiberDataView = new DataView(new ArrayBuffer(this.maxFibers * FIBER_FLOAT_COUNT * 4));
    let fiberCount = 0;

    // יוצר רשת חיבורים אקראית ודלילה
    for (let i = 0; i < this.particleCount && fiberCount < this.maxFibers; i++) {
      for (let j = 0; j < 5; j++) { // נסה לחבר כל חלקיק ל-5 שכנים אקראיים
          if (Math.random() < 0.5) { // סיכוי של 50% לחיבור
              const neighborIndex = Math.floor(Math.random() * this.particleCount);
              if (i === neighborIndex) continue;

              const offset = fiberCount * FIBER_FLOAT_COUNT * 4;
              fiberDataView.setUint32(offset + 0, i, true); // start index
              fiberDataView.setUint32(offset + 4, neighborIndex, true); // end index
              // strength - not used in this shader version
              fiberDataView.setFloat32(offset + 12, 0.5 + Math.random() * 0.5, true); // resting length
              fiberDataView.setFloat32(offset + 16, 1.0, true); // isActive
              fiberCount++;
              if (fiberCount >= this.maxFibers) break;
          }
      }
    }

    this.device.queue.writeBuffer(this.buffers.fibers, 0, fiberDataView.buffer);
    console.log(`Initialized ${fiberCount} fibers.`);
  }

  // --- פונקציות עזר ---

  hsl2rgb(h, s, l) {
    h /= 360; s /= 100; l /= 100;
    const a = s * Math.min(l, 1 - l);
    const f = n => {
      const k = (n + h * 12) % 12;
      return l - a * Math.max(Math.min(k - 3, 9 - k, 1), -1);
    };
    return { r: f(0), g: f(8), b: f(4) };
  }
}

// =============================================================================
//  אתחול המערכת
// =============================================================================

let webgpuSystem = null;

/**
 * פונקציית האתחול הראשית של WebGPU.
 * @returns {Promise<WebGPUParticleSystem|null>}
 */
async function initWebGPU() {
  if (!navigator.gpu) {
    console.warn('WebGPU not supported on this browser.');
    return null;
  }

  try {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) {
      console.warn('Failed to get WebGPU adapter.');
      return null;
    }

    const device = await adapter.requestDevice();
    webgpuSystem = new WebGPUParticleSystem();
    await webgpuSystem.initialize(device);

    return webgpuSystem;
  } catch (error) {
    console.error('WebGPU initialization failed:', error);
    return null;
  }
}

// ייצוא הפונקציות והמחלקה לשימוש במודולים אחרים
export { initWebGPU, webgpuSystem, WebGPUParticleSystem };
