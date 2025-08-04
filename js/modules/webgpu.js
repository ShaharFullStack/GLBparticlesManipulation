// WebGPU Detection
let webgpuDevice = null;
let webgpuContext = null;
let useWebGPU = false;

// WebGPU Compute Shaders
const particleComputeShader = `
  struct Particle {
    position: vec3<f32>,
    velocity: vec3<f32>,
    life: f32,
    size: f32,
    color: vec3<f32>,
    target: vec3<f32>
  };
  
  struct Uniforms {
    time: f32,
    deltaTime: f32,
    particleCount: f32,
    gravity: f32,
    turbulence: f32,
    attraction: f32,
    morphProgress: f32,
    mousePos: vec2<f32>
  };
  
  @group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
  @group(0) @binding(1) var<uniform> uniforms: Uniforms;
  @group(0) @binding(2) var<storage, read> targets: array<vec3<f32>>;
  
  fn noise3D(p: vec3<f32>) -> f32 {
    return fract(sin(dot(p, vec3<f32>(12.9898, 78.233, 54.53))) * 43758.5453);
  }
  
  @compute @workgroup_size(64)
  fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= u32(uniforms.particleCount)) {
      return;
    }
    
    var particle = particles[index];
    let dt = uniforms.deltaTime;
    
    // Turbulence
    let noisePos = particle.position * 0.1 + vec3<f32>(uniforms.time * 0.1);
    let turbulenceForce = vec3<f32>(
      noise3D(noisePos) - 0.5,
      noise3D(noisePos + vec3<f32>(100.0)) - 0.5,
      noise3D(noisePos + vec3<f32>(200.0)) - 0.5
    ) * uniforms.turbulence;
    
    // Gravity
    let gravityForce = vec3<f32>(0.0, -uniforms.gravity, 0.0);
    
    // Attraction to target
    let targetForce = (particle.target - particle.position) * uniforms.attraction;
    
    // Mouse attraction
    let mousePos3D = vec3<f32>(uniforms.mousePos.x, uniforms.mousePos.y, 0.0);
    let mouseDistance = distance(particle.position, mousePos3D);
    let mouseForce = normalize(mousePos3D - particle.position) * (1.0 / (mouseDistance + 1.0)) * 0.5;
    
    // Apply forces
    particle.velocity += (turbulenceForce + gravityForce + targetForce + mouseForce) * dt;
    particle.velocity *= 0.98; // Damping
    
    // Update position
    particle.position += particle.velocity * dt;
    
    // Update particle
    particles[index] = particle;
  }
`;

const fiberComputeShader = `
  struct Fiber {
    start: u32,
    end: u32,
    strength: f32,
    length: f32,
    active: f32
  };
  
  struct Particle {
    position: vec3<f32>,
    velocity: vec3<f32>,
    life: f32,
    size: f32,
    color: vec3<f32>,
    target: vec3<f32>
  };
  
  struct FiberUniforms {
    maxDistance: f32,
    springStrength: f32,
    damping: f32,
    time: f32
  };
  
  @group(0) @binding(0) var<storage, read_write> fibers: array<Fiber>;
  @group(0) @binding(1) var<storage, read> particles: array<Particle>;
  @group(0) @binding(2) var<uniform> uniforms: FiberUniforms;
  
  @compute @workgroup_size(64)
  fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&fibers)) {
      return;
    }
    
    var fiber = fibers[index];
    
    if (fiber.active > 0.5) {
      let startPos = particles[fiber.start].position;
      let endPos = particles[fiber.end].position;
      let distance = length(endPos - startPos);
      
      // Update fiber properties based on distance
      fiber.length = distance;
      
      // Deactivate if too far
      if (distance > uniforms.maxDistance) {
        fiber.active = 0.0;
      }
      
      // Spring force simulation
      let springForce = (distance - fiber.length) * uniforms.springStrength;
      fiber.strength = mix(fiber.strength, springForce, uniforms.damping);
    }
    
    fibers[index] = fiber;
  }
`;

// WebGPU System Class
class WebGPUParticleSystem {
  constructor() {
    this.particleCount = 2500;
    this.particles = null;
    this.fibers = null;
    this.uniformBuffer = null;
    this.fiberUniformBuffer = null;
    this.particleComputePipeline = null;
    this.fiberComputePipeline = null;
    this.renderPipeline = null;
    this.bindGroups = {};
    this.buffers = {};
    this.fibersEnabled = false;
    this.maxFibers = 5000;
  }
  
  async initialize(device, context) {
    this.device = device;
    this.context = context;
    
    // Create compute pipelines
    await this.createComputePipelines();
    
    // Initialize buffers
    await this.createBuffers();
    
    // Create bind groups
    await this.createBindGroups();
    
    console.log('WebGPU Particle System initialized');
    return true;
  }
  
  async createComputePipelines() {
    // Particle compute pipeline
    const particleShaderModule = this.device.createShaderModule({
      code: particleComputeShader
    });
    
    this.particleComputePipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: particleShaderModule,
        entryPoint: 'main'
      }
    });
    
    // Fiber compute pipeline
    const fiberShaderModule = this.device.createShaderModule({
      code: fiberComputeShader
    });
    
    this.fiberComputePipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: fiberShaderModule,
        entryPoint: 'main'
      }
    });
  }
  
  async createBuffers() {
    // Particle buffer (position, velocity, life, size, color, target)
    const particleBufferSize = this.particleCount * 12 * 4; // 12 floats per particle
    this.buffers.particles = this.device.createBuffer({
      size: particleBufferSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    });
    
    // Target positions buffer
    this.buffers.targets = this.device.createBuffer({
      size: this.particleCount * 3 * 4, // 3 floats per target
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    
    // Uniform buffer
    this.buffers.uniforms = this.device.createBuffer({
      size: 32, // 8 floats
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    
    // Fiber buffer
    const fiberBufferSize = this.maxFibers * 5 * 4; // 5 floats per fiber
    this.buffers.fibers = this.device.createBuffer({
      size: fiberBufferSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    });
    
    // Fiber uniform buffer
    this.buffers.fiberUniforms = this.device.createBuffer({
      size: 16, // 4 floats
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    
    // Initialize particle data
    await this.initializeParticleData();
  }
  
  async initializeParticleData() {
    const particleData = new Float32Array(this.particleCount * 12);
    
    for (let i = 0; i < this.particleCount; i++) {
      const offset = i * 12;
      
      // Position (random sphere)
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      const radius = Math.random() * 5;
      
      particleData[offset + 0] = radius * Math.sin(phi) * Math.cos(theta);
      particleData[offset + 1] = radius * Math.sin(phi) * Math.sin(theta);
      particleData[offset + 2] = radius * Math.cos(phi);
      
      // Velocity
      particleData[offset + 3] = (Math.random() - 0.5) * 0.1;
      particleData[offset + 4] = (Math.random() - 0.5) * 0.1;
      particleData[offset + 5] = (Math.random() - 0.5) * 0.1;
      
      // Life
      particleData[offset + 6] = 1.0;
      
      // Size
      particleData[offset + 7] = 0.02 + Math.random() * 0.04;
      
      // Color
      const hue = (i / this.particleCount) * 360;
      const color = this.hsl2rgb(hue, 80, 60);
      particleData[offset + 8] = color.r;
      particleData[offset + 9] = color.g;
      particleData[offset + 10] = color.b;
      
      // Target (same as initial position)
      particleData[offset + 11] = particleData[offset + 0];
    }
    
    this.device.queue.writeBuffer(this.buffers.particles, 0, particleData);
  }
  
  hsl2rgb(h, s, l) {
    h /= 360;
    s /= 100;
    l /= 100;
    
    const a = s * Math.min(l, 1 - l);
    const f = n => {
      const k = (n + h * 12) % 12;
      return l - a * Math.max(Math.min(k - 3, 9 - k, 1), -1);
    };
    
    return { r: f(0), g: f(8), b: f(4) };
  }
  
  async createBindGroups() {
    // Particle compute bind group
    this.bindGroups.particleCompute = this.device.createBindGroup({
      layout: this.particleComputePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.buffers.particles } },
        { binding: 1, resource: { buffer: this.buffers.uniforms } },
        { binding: 2, resource: { buffer: this.buffers.targets } }
      ]
    });
    
    // Fiber compute bind group
    this.bindGroups.fiberCompute = this.device.createBindGroup({
      layout: this.fiberComputePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.buffers.fibers } },
        { binding: 1, resource: { buffer: this.buffers.particles } },
        { binding: 2, resource: { buffer: this.buffers.fiberUniforms } }
      ]
    });
  }
  
  updateUniforms(time, deltaTime, params) {
    const uniformData = new Float32Array([
      time,
      deltaTime,
      this.particleCount,
      params.gravity || 0,
      params.turbulence || 0.1,
      params.attraction || 1,
      params.morphProgress || 0,
      0 // padding
    ]);
    
    this.device.queue.writeBuffer(this.buffers.uniforms, 0, uniformData);
    
    // Update fiber uniforms
    const fiberUniformData = new Float32Array([
      params.connectionDistance || 2,
      params.fiberStrength || 0.5,
      0.95, // damping
      time
    ]);
    
    this.device.queue.writeBuffer(this.buffers.fiberUniforms, 0, fiberUniformData);
  }
  
  updateTargets(targetData) {
    if (targetData && targetData.length >= this.particleCount * 3) {
      this.device.queue.writeBuffer(this.buffers.targets, 0, targetData);
    }
  }
  
  compute() {
    const encoder = this.device.createCommandEncoder();
    
    // Particle compute pass
    const particlePass = encoder.beginComputePass();
    particlePass.setPipeline(this.particleComputePipeline);
    particlePass.setBindGroup(0, this.bindGroups.particleCompute);
    const workgroupsX = Math.ceil(this.particleCount / 64);
    particlePass.dispatchWorkgroups(workgroupsX);
    particlePass.end();
    
    // Fiber compute pass (if enabled)
    if (this.fibersEnabled) {
      const fiberPass = encoder.beginComputePass();
      fiberPass.setPipeline(this.fiberComputePipeline);
      fiberPass.setBindGroup(0, this.bindGroups.fiberCompute);
      const fiberWorkgroupsX = Math.ceil(this.maxFibers / 64);
      fiberPass.dispatchWorkgroups(fiberWorkgroupsX);
      fiberPass.end();
    }
    
    this.device.queue.submit([encoder.finish()]);
  }
  
  async readParticleData() {
    // Create a staging buffer for reading
    const stagingBuffer = this.device.createBuffer({
      size: this.buffers.particles.size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });
    
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(
      this.buffers.particles, 0,
      stagingBuffer, 0,
      this.buffers.particles.size
    );
    this.device.queue.submit([encoder.finish()]);
    
    await stagingBuffer.mapAsync(GPUMapMode.READ);
    const data = new Float32Array(stagingBuffer.getMappedRange());
    const result = new Float32Array(data);
    stagingBuffer.unmap();
    stagingBuffer.destroy();
    
    return result;
  }
  
  setParticleCount(count) {
    // Note: Changing particle count requires buffer recreation
    if (count !== this.particleCount) {
      this.particleCount = Math.min(Math.max(count, 100), 10000);
      // Would need to recreate buffers - simplified for this demo
      console.log(`Particle count set to: ${this.particleCount}`);
    }
  }
  
  enableFibers(enabled) {
    this.fibersEnabled = enabled;
    if (enabled) {
      this.initializeFibers();
    }
  }
  
  async initializeFibers() {
    const fiberData = new Float32Array(this.maxFibers * 5);
    let fiberIndex = 0;
    
    // Create connections between nearby particles
    for (let i = 0; i < Math.min(this.particleCount, 1000) && fiberIndex < this.maxFibers; i++) {
      for (let j = i + 1; j < Math.min(this.particleCount, 1000) && fiberIndex < this.maxFibers; j++) {
        if (Math.random() < 0.1) { // 10% connection probability
          const offset = fiberIndex * 5;
          fiberData[offset + 0] = i; // start particle
          fiberData[offset + 1] = j; // end particle
          fiberData[offset + 2] = 0.5; // strength
          fiberData[offset + 3] = 2.0; // rest length
          fiberData[offset + 4] = 1.0; // active
          fiberIndex++;
        }
      }
    }
    
    this.device.queue.writeBuffer(this.buffers.fibers, 0, fiberData);
    console.log(`Initialized ${fiberIndex} fibers`);
  }
}

// Global WebGPU system instance
let webgpuSystem = null;

// Initialize WebGPU
async function initWebGPU() {
  if (!navigator.gpu) {
    console.log('WebGPU not supported, falling back to Three.js');
    return false;
  }
  
  try {
    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: 'high-performance'
    });
    
    if (!adapter) {
      console.log('WebGPU adapter not available');
      return false;
    }
    
    webgpuDevice = await adapter.requestDevice({
      requiredFeatures: [],
      requiredLimits: {
        maxComputeWorkgroupSizeX: 256,
        maxComputeInvocationsPerWorkgroup: 256
      }
    });
    
    // Create WebGPU system
    webgpuSystem = new WebGPUParticleSystem();
    await webgpuSystem.initialize(webgpuDevice, null);
    
    useWebGPU = true;
    console.log('WebGPU initialized successfully');
    return true;
    
  } catch (error) {
    console.error('WebGPU initialization failed:', error);
    return false;
  }
}

export { initWebGPU, webgpuSystem, useWebGPU, WebGPUParticleSystem };