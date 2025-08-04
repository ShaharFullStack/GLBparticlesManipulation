import * as THREE from 'three';

// WebGPU Detection
let webgpuDevice = null;
let webgpuContext = null;
let useWebGPU = false;

// =================================================================================================
// WebGPU Compute Shaders
// =================================================================================================

// Enhanced Particle Compute Shader
const particleComputeShader = `
  // Particle data structure
  struct Particle {
    position: vec3<f32>,
    velocity: vec3<f32>,
    life: f32,
    size: f32,
    color: vec3<f32>,
    target: vec3<f32>
  };
  
  // Uniforms to control the simulation from the CPU
  struct Uniforms {
    time: f32,
    deltaTime: f32,
    particleCount: f32,
    gravity: f32,
    turbulence: f32,
    attraction: f32,
    morphProgress: f32,
    mousePos: vec2<f32>,
    hoverRadius: f32,
    hoverStrength: f32,
    mouseInfluence: f32,
    hoverRepulsion: f32,
    respawnRate: f32,
  };
  
  // Bindings for data buffers
  @group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
  @group(0) @binding(1) var<uniform> uniforms: Uniforms;
  @group(0) @binding(2) var<storage, read> targets: array<vec3<f32>>;
  
  // Simplex Noise implementation for curl noise
  fn mod289_3(x: vec3<f32>) -> vec3<f32> { return x - floor(x * (1.0 / 289.0)) * 289.0; }
  fn permute(x: vec3<f32>) -> vec3<f32> { return mod289_3(((x*34.0)+1.0)*x); }
  fn taylorInvSqrt(r: vec3<f32>) -> vec3<f32> { return 1.79284291400159 - 0.85373472095314 * r; }

  fn snoise(v: vec3<f32>) -> f32 {
    let C = vec2<f32>(1.0/6.0, 1.0/3.0) ;
    let i  = floor(v + dot(v, C.yyy) );
    let x0 = v - i + dot(i, C.xxx) ;
    let g = step(x0.yzx, x0.xyz);
    let l = 1.0 - g;
    let i1 = min( g.xyz, l.zxy );
    let i2 = max( g.xyz, l.zxy );
    let x1 = x0 - i1 + C.xxx;
    let x2 = x0 - i2 + C.yyy;
    let x3 = x0 - 0.5;
    let i_mod = mod289_3(i);
    let p = permute( permute( i_mod.z + vec3<f32>(0.0, i1.z, i2.z )) + i_mod.y + vec3<f32>(0.0, i1.y, i2.y )) + i_mod.x + vec3<f32>(0.0, i1.x, i2.x );
    var m = max(0.5 - vec4<f32>(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), vec4<f32>(0.0));
    m = m*m;
    m = m*m;
    let x_ = 2.0 * fract(p * C.www) - 1.0;
    let h = abs(x_) - 0.5;
    let ox = floor(x_ + 0.5);
    let a0 = x_ - ox;
    m *= taylorInvSqrt(a0*a0 + h*h);
    let g_ = vec4<f32>(a0.x  * x0.x  + h.x  * x0.y,  a0.y  * x1.x  + h.y  * x1.y, a0.z  * x2.x  + h.z  * x2.y, a0.w  * x3.x  + h.w  * x3.y);
    return 130.0 * dot(m, g_);
  }

  // Curl Noise for fluid-like motion
  fn curlNoise(p: vec3<f32>) -> vec3<f32> {
    let e = 0.1;
    let dx = vec3<f32>(e, 0.0, 0.0);
    let dy = vec3<f32>(0.0, e, 0.0);
    let dz = vec3<f32>(0.0, 0.0, e);

    let p_x0 = snoise(p - dx);
    let p_x1 = snoise(p + dx);
    let p_y0 = snoise(p - dy);
    let p_y1 = snoise(p + dy);
    let p_z0 = snoise(p - dz);
    let p_z1 = snoise(p + dz);

    let x = (p_y1 - p_y0) - (p_z1 - p_z0);
    let y = (p_z1 - p_z0) - (p_x1 - p_x0);
    let z = (p_x1 - p_x0) - (p_y1 - p_y0);

    return normalize(vec3<f32>(x, y, z));
  }
  
  // Pseudo-random number generator
  fn hash(p: vec3<f32>) -> f32 {
    return fract(sin(dot(p, vec3<f32>(12.9898, 78.233, 54.53))) * 43758.5453);
  }

  // Main compute function
  @compute @workgroup_size(64)
  fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= u32(uniforms.particleCount)) {
      return;
    }
    
    var particle = particles[index];
    let dt = uniforms.deltaTime;
    
    // Particle Life Cycle & Respawning
    particle.life -= dt * uniforms.respawnRate;
    if (particle.life <= 0.0) {
        let randomVec = vec3<f32>(hash(particle.position + uniforms.time), hash(particle.velocity), hash(particle.target));
        let theta = randomVec.x * 2.0 * 3.14159;
        let phi = acos(2.0 * randomVec.y - 1.0);
        let radius = randomVec.z * 5.0;
        
        particle.position = vec3<f32>(radius * sin(phi) * cos(theta), radius * sin(phi) * sin(theta), radius * cos(phi));
        particle.velocity = vec3<f32>(0.0, 0.0, 0.0);
        particle.life = 1.0 + hash(particle.position) * 2.0;
    }
    
    // Forces Calculation
    let noisePos = particle.position * 0.2;
    let turbulenceForce = curlNoise(noisePos + vec3<f32>(uniforms.time * 0.05)) * uniforms.turbulence;
    let gravityForce = vec3<f32>(0.0, -uniforms.gravity, 0.0);
    
    var attractionTarget = particle.target;
    if (index < arrayLength(&targets)) {
        let morphTarget = targets[index];
        attractionTarget = mix(particle.target, morphTarget, uniforms.morphProgress);
    }
    let targetForce = (attractionTarget - particle.position) * uniforms.attraction;
    
    // Mouse Interaction
    var mouseForce = vec3<f32>(0.0);
    if (uniforms.hoverRadius > 0.0) {
        let mousePos3D = vec3<f32>(uniforms.mousePos.x, uniforms.mousePos.y, 0.0);
        let mouseVec = particle.position - mousePos3D;
        let mouseDistance = length(mouseVec);
        
        if (mouseDistance < uniforms.hoverRadius) {
            let direction = mouseVec / mouseDistance;
            let falloff = pow(1.0 - (mouseDistance / uniforms.hoverRadius), 2.0);
            let strength = falloff * uniforms.hoverStrength * uniforms.mouseInfluence;
            
            if (uniforms.hoverRepulsion > 0.5) {
                mouseForce = direction * strength;
            } else {
                mouseForce = -direction * strength;
            }
        }
    }
    
    // Apply Forces & Update State
    particle.velocity += (turbulenceForce + gravityForce + targetForce + mouseForce) * dt;
    particle.velocity *= 0.98; // Damping
    particle.position += particle.velocity * dt;
    
    // Update Visuals (Color based on velocity)
    let speed = clamp(length(particle.velocity), 0.0, 0.5);
    let slowColor = vec3<f32>(0.1, 0.4, 0.9);
    let fastColor = vec3<f32>(1.0, 0.9, 0.3);
    particle.color = mix(slowColor, fastColor, smoothstep(0.0, 0.5, speed));

    particles[index] = particle;
  }
`;

const fiberComputeShader = `
  struct Fiber { start: u32, end: u32, strength: f32, length: f32, isActive: f32 };
  struct Particle { position: vec3<f32>, velocity: vec3<f32>, life: f32, size: f32, color: vec3<f32>, target: vec3<f32> };
  struct FiberUniforms { maxDistance: f32, springStrength: f32, damping: f32, time: f32 };
  
  @group(0) @binding(0) var<storage, read_write> fibers: array<Fiber>;
  @group(0) @binding(1) var<storage, read> particles: array<Particle>;
  @group(0) @binding(2) var<uniform> uniforms: FiberUniforms;
  
  @compute @workgroup_size(64)
  fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&fibers)) { return; }
    
    var fiber = fibers[index];
    if (fiber.isActive > 0.5) {
      let startPos = particles[fiber.start].position;
      let endPos = particles[fiber.end].position;
      let distance = length(endPos - startPos);
      fiber.length = distance;
      if (distance > uniforms.maxDistance) { fiber.isActive = 0.0; }
      let springForce = (distance - fiber.length) * uniforms.springStrength;
      fiber.strength = mix(fiber.strength, springForce, uniforms.damping);
    }
    fibers[index] = fiber;
  }
`;

// =================================================================================================
// WebGPU System Class
// =================================================================================================
class WebGPUParticleSystem {
  constructor() {
    this.particleCount = 2500;
    this.particles = null;
    this.fibers = null;
    this.particleComputePipeline = null;
    this.fiberComputePipeline = null;
    this.bindGroups = {};
    this.buffers = {};
    this.fibersEnabled = false;
    this.maxFibers = 5000;
  }
  
  async initialize(device, context) {
    this.device = device;
    this.context = context;
    await this.createComputePipelines();
    await this.createBuffers();
    await this.createBindGroups();
    console.log('WebGPU Particle System initialized');
    return true;
  }
  
  async createComputePipelines() {
    const particleShaderModule = this.device.createShaderModule({ code: particleComputeShader });
    this.particleComputePipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: { module: particleShaderModule, entryPoint: 'main' }
    });
    
    const fiberShaderModule = this.device.createShaderModule({ code: fiberComputeShader });
    this.fiberComputePipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: { module: fiberShaderModule, entryPoint: 'main' }
    });
  }
  
  async createBuffers() {
    const particleStructSize = 14 * 4; // 14 floats * 4 bytes/float
    this.buffers.particles = this.device.createBuffer({
      size: this.particleCount * particleStructSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    });
    
    this.buffers.targets = this.device.createBuffer({
      size: this.particleCount * 3 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    
    this.buffers.uniforms = this.device.createBuffer({
      size: 16 * 4, // 16 floats for alignment
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    
    this.buffers.fibers = this.device.createBuffer({
      size: this.maxFibers * 5 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    });
    
    this.buffers.fiberUniforms = this.device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    
    await this.initializeParticleData();
  }
  
  async initializeParticleData() {
    const particleData = new Float32Array(this.particleCount * 14);
    for (let i = 0; i < this.particleCount; i++) {
      const offset = i * 14;
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      const radius = Math.random() * 5;
      
      // Position, Velocity, Life, Size, Color, Target
      particleData.set([radius * Math.sin(phi) * Math.cos(theta), radius * Math.sin(phi) * Math.sin(theta), radius * Math.cos(phi)], offset);
      particleData.set([(Math.random() - 0.5) * 0.1, (Math.random() - 0.5) * 0.1, (Math.random() - 0.5) * 0.1], offset + 3);
      particleData[offset + 6] = 1.0 + Math.random() * 2.0;
      particleData[offset + 7] = 0.02 + Math.random() * 0.04;
      const color = this.hsl2rgb((i / this.particleCount) * 360, 80, 60);
      particleData.set([color.r, color.g, color.b], offset + 8);
      particleData.set([particleData[offset], particleData[offset + 1], particleData[offset + 2]], offset + 11);
    }
    this.device.queue.writeBuffer(this.buffers.particles, 0, particleData);
  }
  
  hsl2rgb(h, s, l) {
    h /= 360; s /= 100; l /= 100;
    const a = s * Math.min(l, 1 - l);
    const f = n => {
      const k = (n + h * 12) % 12;
      return l - a * Math.max(Math.min(k - 3, 9 - k, 1), -1);
    };
    return { r: f(0), g: f(8), b: f(4) };
  }
  
  async createBindGroups() {
    this.bindGroups.particleCompute = this.device.createBindGroup({
      layout: this.particleComputePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.buffers.particles } },
        { binding: 1, resource: { buffer: this.buffers.uniforms } },
        { binding: 2, resource: { buffer: this.buffers.targets } }
      ]
    });
    
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
    const bloom = params.bloom || {};
    bloom.strength = 0.3;  // Reduced bloom strength
    
    // Write uniform data
    const uniformData = new Float32Array([
      time, deltaTime, this.particleCount,
      params.gravity || 0, params.turbulence || 0.5, params.attraction || 1,
      params.morphProgress || 0, params.mousePos?.x || 0, params.mousePos?.y || 0,
      params.hoverRadius || 3.0, params.hoverStrength || 2.0, params.mouseInfluence || 1.0,
      params.hoverRepulsion ? 1.0 : 0.0, params.respawnRate || 0.2, 0, 0
    ]);
    this.device.queue.writeBuffer(this.buffers.uniforms, 0, uniformData);
    
    const fiberUniformData = new Float32Array([params.connectionDistance || 2, params.fiberStrength || 0.5, 0.95, time]);
    this.device.queue.writeBuffer(this.buffers.fiberUniforms, 0, fiberUniformData);
  }
  async  updateParticles(particleData) {
    if (particleData && particleData.length >= this.particleCount * 14) {
      this.device.queue.writeBuffer(this.buffers.particles, 0, particleData);
    } else {
      console.warn('Insufficient particle data provided');
    }
  }

  updateTargets(targetData) {
    if (targetData && targetData.length >= this.particleCount * 3) {
      this.device.queue.writeBuffer(this.buffers.targets, 0, targetData);
    } else {
      console.warn('Insufficient target data provided');
    }
  }
  
  compute() {
    const encoder = this.device.createCommandEncoder();
    const particlePass = encoder.beginComputePass();
    particlePass.setPipeline(this.particleComputePipeline);
    particlePass.setBindGroup(0, this.bindGroups.particleCompute);
    particlePass.dispatchWorkgroups(Math.ceil(this.particleCount / 64));
    particlePass.end();
    
    if (this.fibersEnabled) {
      const fiberPass = encoder.beginComputePass();
      fiberPass.setPipeline(this.fiberComputePipeline);
      fiberPass.setBindGroup(0, this.bindGroups.fiberCompute);
      fiberPass.dispatchWorkgroups(Math.ceil(this.maxFibers / 64));
      fiberPass.end();
    }
    
    this.device.queue.submit([encoder.finish()]);
  }
  
  async readParticleData() {
    const stagingBuffer = this.device.createBuffer({
      size: this.buffers.particles.size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(this.buffers.particles, 0, stagingBuffer, 0, this.buffers.particles.size);
    this.device.queue.submit([encoder.finish()]);
    await stagingBuffer.mapAsync(GPUMapMode.READ);
    const data = new Float32Array(stagingBuffer.getMappedRange());
    const result = new Float32Array(data);
    stagingBuffer.unmap();
    stagingBuffer.destroy();
    return result;
  }
  
  setParticleCount(count) {
    if (count !== this.particleCount) {
      this.particleCount = Math.min(Math.max(count, 100), 10000);
      // In a real app, buffer recreation would be needed here.
      console.log(`Particle count set to: ${this.particleCount}`);
    }
  }
  
  enableFibers(enabled) {
    this.fibersEnabled = enabled;
    if (enabled) this.initializeFibers();
  }
  
  async initializeFibers() {
    const fiberData = new Float32Array(this.maxFibers * 5);
    let fiberIndex = 0;
    const limit = Math.min(this.particleCount, 1000);
    for (let i = 0; i < limit && fiberIndex < this.maxFibers; i++) {
      for (let j = i + 1; j < limit && fiberIndex < this.maxFibers; j++) {
        if (Math.random() < 0.1) {
          fiberData.set([i, j, 0.5, 2.0, 1.0], fiberIndex * 5); // [start, end, strength, length, isActive]
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
    console.log('WebGPU not supported.');
    return false;
  }
  try {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) {
      console.log('WebGPU adapter not available');
      return false;
    }
    webgpuDevice = await adapter.requestDevice();
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


// =================================================================================================
// Control Panel Logic
// =================================================================================================

// Mobile detection
const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);

// Control Panel State
const controlState = {
  particleCount: isMobile ? 1500 : 2500,
  particleSize: 0.03,
  particleOpacity: 0.9,
  fibersEnabled: true,
  fiberStrength: 4.5,
  connectionDistance: 2.0,
  interactiveFibersEnabled: true,
  mouseConnectionDistance: 2.0, // Shorter mouse connection distance
  fiberOpacity: 2,
  gravity: 0,
  turbulence: 0.1,
  attraction: 1.0,
  morphSpeed: 2.0,
  rotationSpeed: 0.005,
  bloomStrength: isMobile ? 1.0 : 1.4,
  bloomRadius: isMobile ? 0.6 : 0.8,
  colorPalette: 0,
  mousePos: { x: 0, y: 0 },
  hoverEnabled: false,
  hoverRadius: 3.0,
  hoverStrength: 2.0,
  hoverRepulsion: false,
  particleTrail: true,
  mouseInfluence: 1.0,
  autoRotate: false,
  respawnRate: 0.2 // Rate at which particles respawn
  

};

// Color palettes
const colorPalettes = [
  [new THREE.Color(0x00ffff), new THREE.Color(0xff0080), new THREE.Color(0x8000ff), new THREE.Color(0x00ff40), new THREE.Color(0xff4000), new THREE.Color(0x4080ff)], // Cyberpunk
  [new THREE.Color(0xff6b35), new THREE.Color(0xf7931e), new THREE.Color(0xffd23f), new THREE.Color(0xff8c42), new THREE.Color(0xff6b6b), new THREE.Color(0xffa726)], // Sunset
  [new THREE.Color(0x4ecdc4), new THREE.Color(0x44a08d), new THREE.Color(0x006064), new THREE.Color(0x26c6da), new THREE.Color(0x00acc1), new THREE.Color(0x0097a7)], // Ocean
  [new THREE.Color(0x667eea), new THREE.Color(0x764ba2), new THREE.Color(0x9c27b0), new THREE.Color(0x673ab7), new THREE.Color(0x3f51b5), new THREE.Color(0x5c6bc0)], // Galaxy
  [new THREE.Color(0xff9a9e), new THREE.Color(0xfecfef), new THREE.Color(0xff6b9d), new THREE.Color(0xf06292), new THREE.Color(0xe91e63), new THREE.Color(0xad1457)], // Rose
  [new THREE.Color(0xa8edea), new THREE.Color(0xfed6e3), new THREE.Color(0x81c784), new THREE.Color(0x66bb6a), new THREE.Color(0x4caf50), new THREE.Color(0x388e3c)]  // Mint
];

// Preset configurations
const presets = {
  organic: { particleCount: 3000, particleSize: 0.025, fibersEnabled: true, fiberStrength: 0.3, connectionDistance: 1.5, gravity: -0.2, turbulence: 0.15, attraction: 0.8, morphSpeed: 1.5, rotationSpeed: 0.003, colorPalette: 2 },
  digital: { particleCount: 5000, particleSize: 0.02, fibersEnabled: true, fiberStrength: 0.8, connectionDistance: 2.5, gravity: 0, turbulence: 0.05, attraction: 1.5, morphSpeed: 3.0, rotationSpeed: 0.008, colorPalette: 0 },
  cosmic: { particleCount: 4000, particleSize: 0.04, fibersEnabled: false, fiberStrength: 0.2, connectionDistance: 3.0, gravity: 0.1, turbulence: 0.2, attraction: 0.5, morphSpeed: 1.0, rotationSpeed: 0.002, colorPalette: 3 },
  minimal: { particleCount: 1000, particleSize: 0.035, fibersEnabled: false, fiberStrength: 0.1, connectionDistance: 1.0, gravity: 0, turbulence: 0.02, attraction: 2.0, morphSpeed: 2.5, rotationSpeed: 0.001, colorPalette: 4 }
};

// Control Panel Initialization
function initializeControlPanel(renderer, webgpuSystem, useWebGPU) {
  const toggleBtn = document.getElementById('togglePanel');
  const panel = document.getElementById('controlPanel');
  if (toggleBtn && panel) {
    toggleBtn.addEventListener('click', () => {
      panel.classList.toggle('open');
      toggleBtn.classList.toggle('open');
    });
  }
  
  initializeSliders(webgpuSystem, useWebGPU);
  initializeToggles(webgpuSystem, useWebGPU);
  initializeColorPalette();
  initializePresets(webgpuSystem, useWebGPU);
  
  document.addEventListener('mousemove', (e) => {
    const rect = renderer.domElement.getBoundingClientRect();
    controlState.mousePos.x = ((e.clientX - rect.left) / rect.width) * 2 + 1;
    controlState.mousePos.y = ((e.clientY - rect.top) / rect.height) * 2 + 1;
    controlState.mousePos.z = ((e.clientZ - rect.depth) / rect.height) * 2 + 1;
  });
  
  document.addEventListener('mouseleave', () => {
    controlState.mousePos.x = 0;
    controlState.mousePos.y = 0;
    controlState.mousePos.z = 0;
  });
}

function initializeSliders(webgpuSystem, useWebGPU) {
  const sliders = {
    particleCount: (v) => {
      controlState.particleCount = parseInt(v);
      if (useWebGPU && webgpuSystem) webgpuSystem.setParticleCount(controlState.particleCount);
      updateParticleCount(controlState.particleCount);
    },
    particleSize: (v) => { controlState.particleSize = parseFloat(v); updateParticleSize(v); },
    particleOpacity: (v) => { controlState.particleOpacity = parseFloat(v); updateParticleOpacity(v); },
    fiberStrength: (v) => { controlState.fiberStrength = parseFloat(v); },
    connectionDistance: (v) => { controlState.connectionDistance = parseFloat(v); },
    mouseConnectionDistance: (v) => { controlState.mouseConnectionDistance = parseFloat(v); },
    fiberOpacity: (v) => { 
      controlState.fiberOpacity = parseFloat(v); 
      if (window.interactiveFiberMaterial) {
        window.interactiveFiberMaterial.opacity = parseFloat(v);
      }
    },
    gravity: (v) => { controlState.gravity = parseFloat(v); },
    turbulence: (v) => { controlState.turbulence = parseFloat(v); },
    attraction: (v) => { controlState.attraction = parseFloat(v); },
    morphSpeed: (v) => { controlState.morphSpeed = parseFloat(v); },
    rotationSpeed: (v) => { controlState.rotationSpeed = parseFloat(v); },
    bloomStrength: (v) => { controlState.bloomStrength = parseFloat(v); updateBloomStrength(v); },
    bloomRadius: (v) => { controlState.bloomRadius = parseFloat(v); updateBloomRadius(v); },
    hoverRadius: (v) => { controlState.hoverRadius = parseFloat(v); },
    hoverStrength: (v) => { controlState.hoverStrength = parseFloat(v); },
    mouseInfluence: (v) => { controlState.mouseInfluence = parseFloat(v); }
  };
  
  Object.keys(sliders).forEach(key => {
    const slider = document.getElementById(key);
    const valueDisplay = document.getElementById(key + 'Value');
    if (slider && valueDisplay) {
      slider.addEventListener('input', (e) => {
        const value = e.target.value;
        valueDisplay.textContent = value;
        sliders[key](value);
      });
      valueDisplay.textContent = slider.value;
    }
  });
}

function initializeToggles(webgpuSystem, useWebGPU) {
  const toggles = {
    fibersEnabled: () => {
      controlState.fibersEnabled = !controlState.fibersEnabled;
      if (useWebGPU && webgpuSystem) webgpuSystem.enableFibers(controlState.fibersEnabled);
      if (window.toggleFiberVisibility) window.toggleFiberVisibility(controlState.fibersEnabled);
      const display = document.getElementById('fiberDisplay');
      if (display) display.textContent = controlState.fibersEnabled ? 'Active' : 'Disabled';
    },
    interactiveFibersEnabled: () => {
      controlState.interactiveFibersEnabled = !controlState.interactiveFibersEnabled;
      if (window.interactiveFiberLines) {
        window.interactiveFiberLines.visible = controlState.interactiveFibersEnabled;
      }
      const display = document.getElementById('interactiveFiberDisplay');
      if (display) display.textContent = controlState.interactiveFibersEnabled ? 'Active' : 'Disabled';
    },
    hoverEnabled: () => { controlState.hoverEnabled = !controlState.hoverEnabled; },
    hoverRepulsion: () => { controlState.hoverRepulsion = !controlState.hoverRepulsion; },
    particleTrail: () => { controlState.particleTrail = !controlState.particleTrail; },
    autoRotate: () => { controlState.autoRotate = !controlState.autoRotate; }
  };
  
  Object.keys(toggles).forEach(key => {
    const toggle = document.getElementById(key);
    if (toggle) {
      toggle.addEventListener('click', () => {
        toggles[key]();
        toggle.classList.toggle('active', controlState[key]);
      });
    }
  });
}

function initializeColorPalette() {
  const colorOptions = document.querySelectorAll('.color-option');
  colorOptions.forEach((option, index) => {
    option.addEventListener('click', () => {
      colorOptions.forEach(opt => opt.classList.remove('active'));
      option.classList.add('active');
      controlState.colorPalette = index;
      updateParticleColors(index);
    });
  });
}

function initializePresets(webgpuSystem, useWebGPU) {
  const presetButtons = document.querySelectorAll('.preset-btn');
  presetButtons.forEach(button => {
    button.addEventListener('click', () => {
      const presetName = button.dataset.preset;
      applyPreset(presetName, webgpuSystem, useWebGPU);
      presetButtons.forEach(btn => btn.classList.remove('active'));
      button.classList.add('active');
    });
  });
}

function applyPreset(presetName, webgpuSystem, useWebGPU) {
  const preset = presets[presetName];
  if (!preset) return;
  
  Object.keys(preset).forEach(key => {
    if (controlState.hasOwnProperty(key)) {
      controlState[key] = preset[key];
      const slider = document.getElementById(key);
      const valueDisplay = document.getElementById(key + 'Value');
      if (slider && valueDisplay) {
        slider.value = preset[key];
        valueDisplay.textContent = preset[key];
      }
    }
  });
  
  if (preset.fibersEnabled !== undefined) {
    const toggle = document.getElementById('fibersEnabled');
    if (toggle) toggle.classList.toggle('active', preset.fibersEnabled);
    if (window.toggleFiberVisibility) window.toggleFiberVisibility(preset.fibersEnabled);
  }
  if (preset.colorPalette !== undefined) {
    const colorOptions = document.querySelectorAll('.color-option');
    colorOptions.forEach((opt, idx) => opt.classList.toggle('active', idx === preset.colorPalette));
    updateParticleColors(preset.colorPalette);
  }
  
  if (useWebGPU && webgpuSystem) {
    webgpuSystem.setParticleCount(preset.particleCount);
    webgpuSystem.enableFibers(preset.fibersEnabled);
  }
  
  updateParticleCount(preset.particleCount);
  updateParticleSize(preset.particleSize);
}

// Stubs for main app to implement
function updateParticleColors(paletteIndex) { if (window.updateParticleColors) window.updateParticleColors(paletteIndex); }
function updateParticleCount(newCount) { if (window.updateParticleCount) window.updateParticleCount(newCount); }
function updateParticleSize(size) { if (window.updateParticleSize) window.updateParticleSize(size); }
function updateParticleOpacity(opacity) { if (window.updateParticleOpacity) window.updateParticleOpacity(opacity); }
function updateBloomStrength(strength) { if (window.updateBloomStrength) window.updateBloomStrength(strength); }
function updateBloomRadius(radius) { if (window.updateBloomRadius) window.updateBloomRadius(radius); }

// Combined exports
export { 
  initWebGPU, webgpuSystem, useWebGPU, WebGPUParticleSystem,
  controlState, colorPalettes, presets, initializeControlPanel,
  updateParticleColors, updateParticleCount, updateParticleSize,
  updateParticleOpacity, updateBloomStrength, updateBloomRadius
};
