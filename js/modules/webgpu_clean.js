// WebGPU Detection
let webgpuDevice = null;
let webgpuContext = null;
let useWebGPU = false;

// Enhanced Particle Compute Shader
//
// This shader introduces several visual improvements:
// 1.  **Simplex & Curl Noise**: Replaces simple random turbulence with a curl noise field based on
//     Simplex noise. This creates elegant, swirling, fluid-like motion.
// 2.  **Particle Life Cycle**: Particles now have a limited `life`. When a particle's life runs out,
//     it is "respawned" in a new position, creating a continuous and dynamic system.
// 3.  **Velocity-Based Color**: Particle color is now determined by its speed, shifting from
//     cool blues (slow) to warm yellows (fast), providing clear visual feedback on the simulation's dynamics.
// =================================================================================================
const particleComputeShader = `
  // Particle data structure
  struct Particle {
    position: vec3<f32>,
    velocity: vec3<f32>,
    life: f32,
    size: f32,
    color: vec3<f32>,
    targetPos: vec3<f32>
  }
  
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
    respawnRate: f32
  }
  
  // Bindings for data buffers
  @group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
  @group(0) @binding(1) var<uniform> uniforms: Uniforms;
  @group(0) @binding(2) var<storage, read> targets: array<vec3<f32>>;
  
  // =================================================================================
  // Noise Functions
  // Classic Simplex Noise implementation for high-quality, smooth noise fields.
  // This is the basis for our curl noise turbulence.
  // =================================================================================
  fn mod289_3(x: vec3<f32>) -> vec3<f32> { return x - floor(x * (1.0 / 289.0)) * 289.0; }
  fn permute(x: vec3<f32>) -> vec3<f32> { return mod289_3(((x*34.0)+1.0)*x); }
  fn taylorInvSqrt(r: vec3<f32>) -> vec3<f32> { return 1.79284291400159 - 0.85373472095314 * r; }

  fn snoise(v: vec3<f32>) -> f32 {
    let C = vec4<f32>(1.0/6.0, 1.0/3.0, 0.0, 0.5);
    let i  = floor(v + dot(v, C.yyy));
    let x0 = v - i + dot(i, C.xxx);
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
    let x_ = 2.0 * fract(p * C.yyy) - 1.0;
    let h = abs(x_) - 0.5;
    let ox = floor(x_ + 0.5);
    let a0 = x_ - ox;
    m *= taylorInvSqrt(a0*a0 + h*h);
    let g_ = vec4<f32>(a0.x  * x0.x  + h.x  * x0.y,  a0.y  * x1.x  + h.y  * x1.y, a0.z  * x2.x  + h.z  * x2.y, a0.w  * x3.x  + h.w  * x3.y);
    return 130.0 * dot(m, g_);
  }

  // =================================================================================
  // Curl Noise
  // Calculates the curl of the noise field. This creates divergence-free vector
  // fields, resulting in stable, swirling motion perfect for fluid dynamics.
  // =================================================================================
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
  
  // A simple pseudo-random number generator (hash) for respawning particles.
  fn hash(p: vec3<f32>) -> f32 {
    return fract(sin(dot(p, vec3<f32>(12.9898, 78.233, 54.53))) * 43758.5453);
  }

  // Main compute function, executed for each particle
  @compute @workgroup_size(64)
  fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= u32(uniforms.particleCount)) {
      return;
    }
    
    var particle = particles[index];
    let dt = uniforms.deltaTime;
    
    // --- 1. Particle Life Cycle ---
    particle.life -= dt * uniforms.respawnRate;
    if (particle.life <= 0.0) {
      // Respawn the particle at a new random position within a sphere
      let randomVec = vec3<f32>(hash(particle.position + uniforms.time), hash(particle.velocity), hash(particle.targetPos));
      let theta = randomVec.x * 2.0 * 3.14159;
      let phi = acos(2.0 * randomVec.y - 1.0);
      let radius = randomVec.z * 5.0;
      
      particle.position = vec3<f32>(
        radius * sin(phi) * cos(theta),
        radius * sin(phi) * sin(theta),
        radius * cos(phi)
      );
      particle.velocity = vec3<f32>(0.0, 0.0, 0.0);
      particle.life = 1.0 + hash(particle.position) * 2.0; // Life between 1 and 3 seconds
    }
    
    // --- 2. Calculate Forces ---
    
    // Curl Noise Turbulence
    let noisePos = particle.position * 0.2;
    let turbulenceForce = curlNoise(noisePos + vec3<f32>(uniforms.time * 0.05)) * uniforms.turbulence;
    
    // Gravity
    let gravityForce = vec3<f32>(0.0, -uniforms.gravity, 0.0);
    
    // Attraction to target position for morphing effects
    var attractionTarget = particle.targetPos;
    if (index < arrayLength(&targets)) {
      let morphTarget = targets[index];
      attractionTarget = mix(particle.targetPos, morphTarget, uniforms.morphProgress);
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
                mouseForce = direction * strength; // Repel
            } else {
                mouseForce = -direction * strength; // Attract
            }
        }
    }
    
    // --- 3. Apply Forces and Update State ---
    
    // Combine all forces and update velocity
    particle.velocity += (turbulenceForce + gravityForce + targetForce + mouseForce) * dt;
    particle.velocity *= 0.98; // Apply damping to prevent explosion
    
    // Update position based on velocity
    particle.position += particle.velocity * dt;
    
    // --- 4. Update Visuals ---
    
    // Update color based on velocity magnitude
    let speed = clamp(length(particle.velocity), 0.0, 0.5);
    let slowColor = vec3<f32>(0.1, 0.4, 0.9); // Deep Blue
    let fastColor = vec3<f32>(1.0, 0.9, 0.3); // Bright Yellow
    particle.color = mix(slowColor, fastColor, smoothstep(0.0, 0.5, speed));

    // Write the updated particle data back to the buffer
    particles[index] = particle;
  }
`;

// =================================================================================================
// Fiber Physics Compute Shader
//
// This shader simulates the physical connections (fibers) between particles.
// It applies spring and damping forces to create a stable, interconnected structure.
// 1.  **Spring Force (Hooke's Law)**: Pulls or pushes connected particles towards a defined
//     "resting length", making the connections behave like springs.
// 2.  **Damping Force**: Reduces oscillations and stabilizes the system by applying a force
//     proportional to the relative velocity of the connected particles.
// 3.  **Dynamic Connections**: Fibers can "break" (become inactive) if stretched too far,
//     allowing the structure to tear under stress.
//
// NOTE: This shader directly modifies particle velocities. This can cause a "race condition"
// if multiple threads update the same particle in one dispatch. For visual effects with
// a sparse fiber network, this is often an acceptable simplification.
// =================================================================================================
const fiberComputeShader = `
  // Particle data structure - must match the particle compute shader
  struct Particle {
    position: vec3<f32>,
    velocity: vec3<f32>,
    life: f32,
    size: f32,
    color: vec3<f32>,
    targetPos: vec3<f32>
  }
  
  // Fiber data structure, connecting two particles
  struct Fiber {
    start: u32,       // Index of the first particle
    end: u32,         // Index of the second particle
    strength: f32,    // Used to visualize force, can be updated
    length: f32,      // The resting length of the spring
    isActive: f32     // 1.0 if active, 0.0 if broken
  }
  
  // Uniforms to control the fiber simulation
  struct FiberUniforms {
    maxDistance: f32,
    springStrength: f32,
    springDamping: f32,
    deltaTime: f32
  }
  
  // Bindings for data buffers
  @group(0) @binding(0) var<storage, read_write> fibers: array<Fiber>;
  @group(0) @binding(1) var<storage, read_write> particles: array<Particle>;
  @group(0) @binding(2) var<uniform> uniforms: FiberUniforms;
  
  @compute @workgroup_size(64)
  fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&fibers)) {
      return;
    }
    
    var fiber = fibers[index];
    
    // Skip inactive (broken) fibers
    if (fiber.isActive < 0.5) {
      return;
    }
    
    let startIndex = fiber.start;
    let endIndex = fiber.end;
    
    // Bounds check
    if (startIndex >= arrayLength(&particles) || endIndex >= arrayLength(&particles)) {
      return;
    }
    
    // Get local copies of particle data
    var startParticle = particles[startIndex];
    var endParticle = particles[endIndex];
    
    let vecBetween = endParticle.position - startParticle.position;
    let currentDistance = length(vecBetween);
    
    // Deactivate fiber if it's stretched beyond the maximum allowed distance
    if (currentDistance > uniforms.maxDistance) {
      fiber.isActive = 0.0;
      fibers[index] = fiber; // Write back the change
      return;
    }
    
    // Avoid division by zero if particles are at the same position
    if (currentDistance < 0.0001) {
      return;
    }
    
    let direction = vecBetween / currentDistance;
    
    // --- 1. Calculate Spring Force (Hooke's Law) ---
    // The force that pulls/pushes particles to the resting length.
    let displacement = currentDistance - fiber.length;
    let springForceMagnitude = displacement * uniforms.springStrength;
    let springForce = direction * springForceMagnitude;
    
    // --- 2. Calculate Damping Force ---
    // The force that slows down oscillations, proportional to relative velocity.
    let relativeVelocity = endParticle.velocity - startParticle.velocity;
    let velocityAlongAxis = dot(relativeVelocity, direction);
    let dampingForceMagnitude = velocityAlongAxis * uniforms.springDamping;
    let dampingForce = direction * dampingForceMagnitude;
    
    // --- 3. Combine Forces ---
    let totalForce = springForce + dampingForce;
    
    // --- 4. Apply Forces to Particle Velocities ---
    // We apply the force integrated over deltaTime.
    // F = ma -> a = F/m. We assume mass (m) is 1.
    // delta_v = a * dt = F * dt.
    let deltaVelocity = totalForce * uniforms.deltaTime;
    
    startParticle.velocity += deltaVelocity;
    endParticle.velocity -= deltaVelocity; // Equal and opposite reaction
    
    // Update the visualization strength of the fiber
    fiber.strength = length(totalForce);
    
    // --- 5. Write Data Back to Buffers ---
    particles[startIndex] = startParticle;
    particles[endIndex] = endParticle;
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
    
    await this.createComputePipelines();
    await this.createBuffers();
    await this.createBindGroups();
    
    console.log('WebGPU Particle System initialized');
    return true;
  }
  
  async createComputePipelines() {
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
    // Particle struct is 14 floats * 4 bytes/float.
    const particleStructSize = 14 * 4;
    const particleBufferSize = this.particleCount * particleStructSize;
    this.buffers.particles = this.device.createBuffer({
      size: particleBufferSize,
      // Usage must allow for STORAGE in both pipelines, and copying.
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    });
    
    this.buffers.targets = this.device.createBuffer({
      size: this.particleCount * 3 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    
    // Uniform buffer for particle sim. 16 floats for alignment.
    this.buffers.uniforms = this.device.createBuffer({
      size: 16 * 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    
    // Fiber struct is 5 floats (2 u32, 3 f32) * 4 bytes/float
    const fiberStructSize = 5 * 4;
    const fiberBufferSize = this.maxFibers * fiberStructSize;
    this.buffers.fibers = this.device.createBuffer({
      size: fiberBufferSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    });
    
    // Fiber uniforms: maxDist, strength, damping, deltaTime (4 floats)
    this.buffers.fiberUniforms = this.device.createBuffer({
      size: 4 * 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    
    await this.initializeParticleData();
  }
  
  async initializeParticleData() {
    const particleData = new Float32Array(this.particleCount * 14); // 14 floats to match shader struct
    
    for (let i = 0; i < this.particleCount; i++) {
      const offset = i * 14;
      
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      const radius = Math.random() * 5;
      
      // Position (x, y, z)
      particleData[offset + 0] = radius * Math.sin(phi) * Math.cos(theta);
      particleData[offset + 1] = radius * Math.sin(phi) * Math.sin(theta);
      particleData[offset + 2] = radius * Math.cos(phi);
      
      // Velocity (vx, vy, vz)
      particleData[offset + 3] = (Math.random() - 0.5) * 0.1;
      particleData[offset + 4] = (Math.random() - 0.5) * 0.1;
      particleData[offset + 5] = (Math.random() - 0.5) * 0.1;
      
      // Life
      particleData[offset + 6] = 1.0 + Math.random() * 2.0;
      
      // Size
      particleData[offset + 7] = 0.02 + Math.random() * 0.04;
      
      // Color (r, g, b)
      const hue = (i / this.particleCount) * 360;
      const color = this.hsl2rgb(hue, 80, 60);
      particleData[offset + 8] = color.r;
      particleData[offset + 9] = color.g;
      particleData[offset + 10] = color.b;
      
      // Target (tx, ty, tz)
      particleData[offset + 11] = particleData[offset + 0];
      particleData[offset + 12] = particleData[offset + 1];
      particleData[offset + 13] = particleData[offset + 2];
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
    const uniformData = new Float32Array([
      time,
      deltaTime,
      this.particleCount,
      params.gravity || 0,
      params.turbulence || 0.5,
      params.attraction || 1,
      params.morphProgress || 0,
      params.mousePos?.x || 0,
      params.mousePos?.y || 0,
      params.hoverRadius || 3.0,
      params.hoverStrength || 2.0,
      params.mouseInfluence || 1.0,
      params.hoverRepulsion ? 1.0 : 0.0,
      params.respawnRate || 0.2, // New uniform for life cycle speed
      0, 0 // padding for 16-byte alignment
    ]);
    
    this.device.queue.writeBuffer(this.buffers.uniforms, 0, uniformData);
    
    // Update fiber uniforms with new physics parameters
    const fiberUniformData = new Float32Array([
      params.connectionDistance || 2.0, // maxDistance
      params.fiberStrength || 0.5,      // springStrength
      params.fiberDamping || 0.05,      // springDamping
      deltaTime                         // deltaTime for correct physics
    ]);
    
    this.device.queue.writeBuffer(this.buffers.fiberUniforms, 0, fiberUniformData);
  }
  
  updateTargets(targetData) {
    if (targetData && targetData.length >= this.particleCount * 3) {
      this.device.queue.writeBuffer(this.buffers.targets, 0, targetData);
    } else {
      console.warn('WebGPU target data insufficient:', targetData?.length, 'required:', this.particleCount * 3);
    }
  }
  
  compute() {
    const encoder = this.device.createCommandEncoder();
    
    // First Pass: Update particle positions based on noise, gravity, morphing, etc.
    const particlePass = encoder.beginComputePass();
    particlePass.setPipeline(this.particleComputePipeline);
    particlePass.setBindGroup(0, this.bindGroups.particleCompute);
    const workgroupsX = Math.ceil(this.particleCount / 64);
    particlePass.dispatchWorkgroups(workgroupsX);
    particlePass.end();
    
    // Second Pass: If fibers are enabled, apply spring forces between connected particles.
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
    if (count !== this.particleCount) {
      this.particleCount = Math.min(Math.max(count, 100), 10000);
      console.log(`Particle count set to: ${this.particleCount}`);
      // In a full application, you would need to recreate buffers and bind groups here.
    }
  }
  
  enableFibers(enabled) {
    this.fibersEnabled = enabled;
    if (enabled && this.device) {
      this.initializeFibers();
    }
  }
  
  async initializeFibers() {
    // Using Uint32 for indices and Float32 for the rest
    const fiberData = new Float32Array(this.maxFibers * 5);
    let fiberIndex = 0;
    
    // Create a random set of connections between the first 1000 particles
    for (let i = 0; i < Math.min(this.particleCount, 1000) && fiberIndex < this.maxFibers; i++) {
      for (let j = i + 1; j < Math.min(this.particleCount, 1000) && fiberIndex < this.maxFibers; j++) {
        // Randomly connect a small percentage of particle pairs
        if (Math.random() < 0.01) { // Lowered probability for a sparser network
          const offset = fiberIndex * 5;
          fiberData[offset + 0] = i; // start index
          fiberData[offset + 1] = j; // end index
          fiberData[offset + 2] = 0.5; // initial strength
          fiberData[offset + 3] = 1.0; // resting length
          fiberData[offset + 4] = 1.0; // isActive
          fiberIndex++;
        }
      }
    }
    
    this.device.queue.writeBuffer(this.buffers.fibers, 0, fiberData);
    console.log(`Initialized ${fiberIndex} fibers`);
  }
  
  updateFiberConnections() {
    // This method can be used to dynamically update fiber connections
    // For now, it's a placeholder that could be expanded to add/remove
    // connections based on particle proximity or other criteria
    if (this.fibersEnabled && this.device) {
      // Could implement dynamic fiber creation/destruction here
      // For example, connecting particles that come within a certain distance
    }
  }
  
  // Method to get fiber data for rendering
  async readFiberData() {
    if (!this.fibersEnabled || !this.device) {
      return null;
    }
    
    const stagingBuffer = this.device.createBuffer({
      size: this.buffers.fibers.size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });
    
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(
      this.buffers.fibers, 0,
      stagingBuffer, 0,
      this.buffers.fibers.size
    );
    this.device.queue.submit([encoder.finish()]);
    
    await stagingBuffer.mapAsync(GPUMapMode.READ);
    const data = new Float32Array(stagingBuffer.getMappedRange());
    const result = new Float32Array(data);
    stagingBuffer.unmap();
    stagingBuffer.destroy();
    
    return result;
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
