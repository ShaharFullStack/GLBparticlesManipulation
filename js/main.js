import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';
import { RGBShiftShader } from 'three/addons/shaders/RGBShiftShader.js';
import { FilmPass } from 'three/addons/postprocessing/FilmPass.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

// Import modules
import { initWebGPU, webgpuSystem, useWebGPU } from './modules/webgpu.js';
import { 
  controlState, 
  colorPalettes, 
  initializeControlPanel
} from './modules/controls.js';

// Mobile detection - must be defined early as it's used throughout
const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);

const gsapScript = document.createElement('script');
gsapScript.src = 'https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.2/gsap.min.js';
document.head.appendChild(gsapScript);

const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x000000, 0.0005);

const camera = new THREE.PerspectiveCamera(100, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.set(0, 0, 10);

// Optimize renderer settings for performance
const renderer = new THREE.WebGLRenderer({ 
  antialias: !isMobile, // Disable antialiasing on mobile for better performance
  powerPreference: 'high-performance',
  stencil: false, // Disable stencil buffer if not needed
  depth: true
});
renderer.setSize(window.innerWidth, window.innerHeight);

// Limit pixel ratio on mobile to prevent performance issues
const pixelRatio = isMobile ? Math.min(window.devicePixelRatio, 2) : window.devicePixelRatio;
renderer.setPixelRatio(pixelRatio);

// Enable additional optimizations
renderer.sortObjects = false; // Disable sorting for particles
renderer.outputColorSpace = THREE.SRGBColorSpace;

document.body.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;
controls.rotateSpeed = 0.5;
controls.minDistance = 1;
controls.maxDistance = 200;
controls.target.set(0, 0, 0);
controls.update();

scene.add(new THREE.AmbientLight(0x1a0033, 0.8));
const dirLight = new THREE.DirectionalLight(0xff66cc, 0.4);
dirLight.position.set(5, 10, 7.5);
scene.add(dirLight);

const dirLight2 = new THREE.DirectionalLight(0x00ffff, 0.3);
dirLight2.position.set(-5, -10, -7.5);
scene.add(dirLight2);

// Adaptive post-processing based on device capability
const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));

// Reduce bloom quality on mobile for better performance
const bloomResolution = isMobile ? 
  new THREE.Vector2(window.innerWidth * 0.5, window.innerHeight * 0.5) :
  new THREE.Vector2(window.innerWidth, window.innerHeight);
  
const bloom = new UnrealBloomPass(bloomResolution, 1.2, 0.6, 0.9);
bloom.threshold = isMobile ? 0.2 : 0.1; // Higher threshold on mobile
bloom.strength = isMobile ? 1.0 : 1.4;  // Reduced strength on mobile
bloom.radius = isMobile ? 0.6 : 0.8;    // Smaller radius on mobile
composer.addPass(bloom);

// Optional effects for desktop only
if (!isMobile) {
  composer.addPass(new FilmPass(0.5, 0.4, 1024, false));
  const rgbShift = new ShaderPass(RGBShiftShader);
  rgbShift.uniforms['amount'].value = 0.003;
  composer.addPass(rgbShift);
}


// GLB loading cache for performance
const glbCache = new Map();

async function createGLBPoints(filePath, totalPoints, scale = 10.0) {
  // Check cache first
  const cacheKey = `${filePath}_${totalPoints}_${scale}`;
  if (glbCache.has(cacheKey)) {
    return glbCache.get(cacheKey);
  }

  return new Promise((resolve, reject) => {
    const loader = new GLTFLoader();
    loader.load(
      filePath,
      (gltf) => {
        const vertices = [];
        const tempVertex = new THREE.Vector3();
        
        gltf.scene.traverse((child) => {
          if (child.isMesh && child.geometry) {
            const geometry = child.geometry;
            const positionAttribute = geometry.attributes.position;
            if (positionAttribute) {
              const matrix = child.matrixWorld;
              const array = positionAttribute.array;
              
              // Optimized vertex extraction using direct array access
              for (let i = 0; i < positionAttribute.count; i++) {
                const i3 = i * 3;
                tempVertex.set(array[i3], array[i3 + 1], array[i3 + 2]);
                tempVertex.applyMatrix4(matrix);
                tempVertex.multiplyScalar(scale);
                vertices.push(tempVertex.clone());
              }
            }
          }
        });

        if (vertices.length === 0) {
          const emptyArray = new Float32Array(totalPoints * 3);
          glbCache.set(cacheKey, emptyArray);
          resolve(emptyArray);
          return;
        }

        // Optimized centering and sampling
        const box = new THREE.Box3().setFromPoints(vertices);
        const center = box.getCenter(new THREE.Vector3());
        
        const pts = new Float32Array(totalPoints * 3);
        const vertexCount = vertices.length;
        
        if (vertexCount >= totalPoints) {
          // Uniform sampling for better distribution
          for (let i = 0; i < totalPoints; i++) {
            const idx = Math.floor((i / totalPoints) * vertexCount);
            const v = vertices[idx];
            const i3 = i * 3;
            pts[i3] = v.x - center.x;
            pts[i3 + 1] = v.y - center.y;
            pts[i3 + 2] = v.z - center.z;
          }
        } else {
          // Repeat vertices when we have fewer than needed
          for (let i = 0; i < totalPoints; i++) {
            const v = vertices[i % vertexCount];
            const i3 = i * 3;
            pts[i3] = v.x - center.x;
            pts[i3 + 1] = v.y - center.y;
            pts[i3 + 2] = v.z - center.z;
          }
        }

        glbCache.set(cacheKey, pts);
        resolve(pts);
      },
      (progress) => {
        // Optional: Add loading progress feedback
        console.log(`Loading ${filePath}: ${(progress.loaded / progress.total * 100).toFixed(1)}%`);
      },
      (error) => {
        console.error('Error loading GLB:', error);
        const fallbackArray = new Float32Array(totalPoints * 3);
        glbCache.set(cacheKey, fallbackArray);
        resolve(fallbackArray);
      }
    );
  });
}

const count = controlState.particleCount; // Use control state particle count
let targets = [];
let currentTargetIndex = 0;

// Mobile performance detection
const particleCount = isMobile ? Math.min(count, 1500) : count;

async function initializeTargets() {
  // Parallel loading for better performance
  const loadPromises = [
    createGLBPoints('./glb/logo.glb', particleCount, 12.0),
    createGLBPoints('./glb/life.glb', particleCount, 12.0),
    createGLBPoints('./glb/nati.glb', particleCount, 12.0)
  ];
  
  try {
    targets = await Promise.all(loadPromises);
    particles.userData.targets = targets;
    
    // Set initial position to first GLB model
    if (targets.length > 0) {
      const positionArray = particles.geometry.attributes.position.array;
      positionArray.set(targets[0]);
      particles.geometry.attributes.position.needsUpdate = true;
    }
    
    console.log(`Loaded ${targets.length} GLB models with ${particleCount} particles each`);
    return targets;
  } catch (error) {
    console.error('Failed to load GLB models:', error);
    return [];
  }
}

// Use actual particle count for geometry
const geom = new THREE.BufferGeometry();
geom.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(particleCount * 3), 3));

// Pre-allocate arrays for better memory efficiency
const colArr = new Float32Array(particleCount * 3);
const origCols = new Array(particleCount);
const twinkle = new Float32Array(particleCount);
const sizes = new Float32Array(particleCount);

// Pre-computed color palette for performance
const palette = [
  new THREE.Color(0x00ffff),
  new THREE.Color(0xff0080),
  new THREE.Color(0x8000ff),
  new THREE.Color(0x00ff40),
  new THREE.Color(0xff4000),
  new THREE.Color(0x4080ff)
];

// Optimized color generation with fewer object allocations
const tempColor = new THREE.Color();
for (let i = 0; i < particleCount; i++) {
  const t = i / Math.max(particleCount - 1, 1);
  const idx = t * (palette.length - 1);
  const floorIdx = Math.floor(idx);
  const ceilIdx = Math.min(Math.ceil(idx), palette.length - 1);
  const mix = idx % 1;
  
  tempColor.lerpColors(palette[floorIdx], palette[ceilIdx], mix).multiplyScalar(1.3);
  
  const i3 = i * 3;
  colArr[i3] = tempColor.r;
  colArr[i3 + 1] = tempColor.g;
  colArr[i3 + 2] = tempColor.b;
  
  origCols[i] = tempColor.clone();
  twinkle[i] = Math.random() < 0.5 ? Math.random() * 6 + 3 : 0;
  sizes[i] = 0.02 + Math.random() * 0.04;
}

geom.setAttribute('color', new THREE.Float32BufferAttribute(colArr, 3));
geom.setAttribute('size', new THREE.Float32BufferAttribute(sizes, 1));

// Enable frustum culling and set bounding sphere for better performance
geom.computeBoundingSphere();

const mat = new THREE.PointsMaterial({
  size: 0.03,
  vertexColors: true,
  sizeAttenuation: true,
  transparent: true,
  opacity: 0.9,
  blending: THREE.AdditiveBlending,
  depthWrite: false
});

const particles = new THREE.Points(geom, mat);
particles.userData = { targets: [], origCols, twinkle };
scene.add(particles);

let shapeIndex = 0;
const clock = new THREE.Clock();
let time = 0;

// Performance monitoring
let frameCount = 0;
let lastTime = performance.now();
let fps = 60;

// Optimized sparkle effect with reduced frequency updates
let sparkleFrameCounter = 0;
const sparkleUpdateFrequency = isMobile ? 3 : 2; // Update less frequently on mobile

function applySparkle(sys, t) {
  // Skip sparkle updates on some frames for better performance
  sparkleFrameCounter++;
  if (sparkleFrameCounter % sparkleUpdateFrequency !== 0) return;
  
  const cols = sys.geometry.attributes.color;
  const { origCols, twinkle } = sys.userData;
  const colArray = cols.array;
  
  // Use direct array access for better performance
  for (let i = 0; i < cols.count; i++) {
    if (twinkle[i] > 0) {
      const p = Math.pow(Math.abs(Math.sin(twinkle[i] * t + i * 0.1)), 10);
      const b = 1 + 4 * p;
      const oc = origCols[i];
      const i3 = i * 3;
      colArray[i3] = oc.r * b;
      colArray[i3 + 1] = oc.g * b;
      colArray[i3 + 2] = oc.b * b;
    }
  }
  cols.needsUpdate = true;
}

// Hover effects for Three.js fallback
function applyHoverEffects(particles, controlState, time) {
  if (!particles || !particles.geometry || !controlState.hoverEnabled) return;
  
  const positions = particles.geometry.attributes.position.array;
  const colors = particles.geometry.attributes.color.array;
  const mouse3D = new THREE.Vector3(controlState.mousePos.x * 10, controlState.mousePos.y * 10, 0);
  
  for (let i = 0; i < positions.length; i += 3) {
    const particle = new THREE.Vector3(positions[i], positions[i + 1], positions[i + 2]);
    const distance = particle.distanceTo(mouse3D);
    
    if (distance < controlState.hoverRadius) {
      const influence = (1 - distance / controlState.hoverRadius) * controlState.hoverStrength * 0.1;
      const direction = particle.clone().sub(mouse3D).normalize();
      
      if (controlState.hoverRepulsion) {
        // Repulsion effect
        positions[i] += direction.x * influence;
        positions[i + 1] += direction.y * influence;
        positions[i + 2] += direction.z * influence;
      } else {
        // Attraction effect
        positions[i] -= direction.x * influence;
        positions[i + 1] -= direction.y * influence;
        positions[i + 2] -= direction.z * influence;
      }
      
      // Color highlight effect
      const colorIntensity = influence * 2;
      const colorIndex = i;
      colors[colorIndex] = Math.min(1, colors[colorIndex] + colorIntensity);
      colors[colorIndex + 1] = Math.min(1, colors[colorIndex + 1] + colorIntensity * 0.5);
      colors[colorIndex + 2] = Math.min(1, colors[colorIndex + 2] + colorIntensity);
    }
  }
  
  particles.geometry.attributes.position.needsUpdate = true;
  particles.geometry.attributes.color.needsUpdate = true;
}

function morph(toIndex) {
  console.log('Morph function called with index:', toIndex);
  console.log('useWebGPU:', useWebGPU);
  console.log('webgpuSystem available:', !!webgpuSystem);
  
  const pos = particles.geometry.attributes.position.array;
  const dest = particles.userData.targets[toIndex];
  
  if (!dest) {
    console.error('No destination target found for index:', toIndex);
    return;
  }
  
  console.log('Destination points:', dest.length);
  console.log('Current position array length:', pos.length);
  
  // Update shape display
  const shapes = ['Logo', 'Life', 'Nati'];
  const shapeDisplay = document.getElementById('shapeDisplay');
  if (shapeDisplay) {
    shapeDisplay.textContent = shapes[toIndex] || `Shape ${toIndex + 1}`;
  }
  
  // Use control state for morph speed
  const duration = 4 / controlState.morphSpeed;
  
  // TEMPORARILY: Always use Three.js morphing to debug
  console.log('FORCED Three.js morphing for debugging');
  console.log('Starting position sample:', pos.slice(0, 9));
  console.log('Target position sample:', dest.slice(0, 9));
  
  // Fallback to Three.js morphing
  if (typeof gsap !== 'undefined') {
    gsap.killTweensOf(pos);
    gsap.to(pos, {
      endArray: dest,
      duration: duration,
      ease: 'power2.inOut',
      onUpdate: () => {
        particles.geometry.attributes.position.needsUpdate = true;
        console.log('GSAP morphing - position updated');
      },
      onComplete: () => {
        console.log('GSAP morphing complete');
      }
    });
  } else {
    // Manual morphing without GSAP
    console.log('Using manual morphing');
    manualMorph(pos, dest, duration);
  }
  
  /* ORIGINAL WebGPU CODE - COMMENTED OUT FOR DEBUGGING
  if (useWebGPU && webgpuSystem) {
    console.log('Using WebGPU morphing');
    // Update WebGPU targets
    webgpuSystem.updateTargets(dest);
    
    // Force immediate morph for testing
    controlState.morphProgress = 0.5; // Set to middle of morph
    
    // Animate morph progress for WebGPU
    if (typeof gsap !== 'undefined') {
      gsap.to(controlState, {
        morphProgress: 1,
        duration: duration,
        ease: 'power2.inOut',
        onComplete: () => {
          controlState.morphProgress = 0;
        }
      });
    } else {
      // Fallback animation without GSAP
      animateMorphProgress(duration);
    }
  }
  */
}

// Sync WebGPU computed particles with Three.js geometry for visual updates
async function syncWebGPUParticles() {
  if (!webgpuSystem || !particles) return;
  
  try {
    // Read particle data from WebGPU
    const particleData = await webgpuSystem.readParticleData();
    
    // Update Three.js geometry positions
    const positions = particles.geometry.attributes.position.array;
    const colors = particles.geometry.attributes.color.array;
    
    // WebGPU particle format: position(3) + velocity(3) + life(1) + size(1) + color(3) + target(3) = 14 floats
    for (let i = 0; i < particleData.length; i += 14) {
      const particleIndex = i / 14;
      const posIndex = particleIndex * 3;
      const colorIndex = particleIndex * 3;
      
      if (posIndex + 2 < positions.length) {
        // Update positions
        positions[posIndex] = particleData[i];     // x
        positions[posIndex + 1] = particleData[i + 1]; // y
        positions[posIndex + 2] = particleData[i + 2]; // z
        
        // Update colors if available
        if (colorIndex + 2 < colors.length) {
          colors[colorIndex] = particleData[i + 8];     // r
          colors[colorIndex + 1] = particleData[i + 9]; // g
          colors[colorIndex + 2] = particleData[i + 10]; // b
        }
      }
    }
    
    particles.geometry.attributes.position.needsUpdate = true;
    if (particles.geometry.attributes.color) {
      particles.geometry.attributes.color.needsUpdate = true;
    }
  } catch (error) {
    console.warn('Failed to sync WebGPU particles:', error);
  }
}

// Fallback morph animation without GSAP
function animateMorphProgress(duration) {
  const startTime = performance.now();
  
  function animate() {
    const elapsed = (performance.now() - startTime) / 1000;
    const progress = Math.min(elapsed / duration, 1);
    
    controlState.morphProgress = progress;
    
    if (progress < 1) {
      requestAnimationFrame(animate);
    } else {
      controlState.morphProgress = 0;
    }
  }
  
  animate();
}

// Manual morphing without GSAP
function manualMorph(pos, dest, duration) {
  console.log('Manual morph started, duration:', duration);
  const startPositions = [...pos];
  const startTime = performance.now();
  
  function animate() {
    const elapsed = (performance.now() - startTime) / 1000;
    const progress = Math.min(elapsed / duration, 1);
    
    // Easing function (power2.inOut equivalent)
    const easedProgress = progress < 0.5 
      ? 2 * progress * progress 
      : 1 - Math.pow(-2 * progress + 2, 2) / 2;
    
    for (let i = 0; i < pos.length; i++) {
      pos[i] = startPositions[i] + (dest[i] - startPositions[i]) * easedProgress;
    }
    
    particles.geometry.attributes.position.needsUpdate = true;
    
    if (progress < 1) {
      requestAnimationFrame(animate);
    } else {
      console.log('Manual morph complete');
    }
  }
  
  animate();
}

document.getElementById('morphButton').addEventListener('click', () => {
  console.log('Morph button clicked');
  console.log('Targets available:', particles.userData.targets?.length || 0);
  console.log('Current shapeIndex:', shapeIndex);
  console.log('GSAP available:', typeof gsap !== 'undefined');
  
  if (particles.userData.targets && particles.userData.targets.length > 0) {
    shapeIndex = (shapeIndex + 1) % particles.userData.targets.length;
    console.log('Morphing to shape index:', shapeIndex);
    
    // Visual feedback
    const morphButton = document.getElementById('morphButton');
    morphButton.style.transform = 'translateX(-50%) scale(0.9)';
    setTimeout(() => {
      morphButton.style.transform = 'translateX(-50%) scale(1)';
    }, 150);
    
    // Update button text or add indicator
    const shapes = ['Logo', 'Life', 'Nati'];
    morphButton.title = `Current: ${shapes[shapeIndex] || 'Shape ' + (shapeIndex + 1)}`;
    
    morph(shapeIndex);
  } else {
    console.error('No targets available for morphing');
    // Retry initializing targets if they're missing
    initializeTargets().then(() => {
      console.log('Targets reinitialized, try morphing again');
    });
  }
});

function animate() {
  requestAnimationFrame(animate);
  
  // Performance monitoring
  const currentTime = performance.now();
  frameCount++;
  if (currentTime - lastTime >= 1000) {
    fps = Math.round((frameCount * 1000) / (currentTime - lastTime));
    frameCount = 0;
    lastTime = currentTime;
    
    // Log performance occasionally (every 5 seconds)
    if (Math.random() < 0.002) {
      console.log(`Performance: ${fps} FPS, Particles: ${particleCount}`);
    }
  }
  
  const dt = clock.getDelta();
  time += dt;
  
  // Auto-rotate or manual rotation control
  if (controlState.autoRotate) {
    particles.rotation.y += controlState.rotationSpeed;
  }
  
  // Update WebGPU system if available
  if (useWebGPU && webgpuSystem) {
    webgpuSystem.updateUniforms(time, dt, controlState);
    webgpuSystem.compute();
    
    // Only sync during morphing to avoid performance hit
    if (controlState.morphProgress > 0) {
      syncWebGPUParticles();
    }
  }
  
  // Apply hover effects to Three.js particles (fallback when WebGPU is not available)
  if (!useWebGPU && controlState.hoverEnabled) {
    applyHoverEffects(particles, controlState, time);
  }
  
  // Update performance display
  const fpsDisplay = document.getElementById('fpsDisplay');
  if (fpsDisplay) {
    fpsDisplay.textContent = fps;
  }
  
  applySparkle(particles, time);
  controls.update();
  composer.render();
}

window.addEventListener('resize', () => {
  const w = window.innerWidth, h = window.innerHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
  composer.setSize(w, h);
});

// Memory cleanup utility
function cleanupUnusedResources() {
  if (renderer && renderer.info) {
    const memoryInfo = renderer.info.memory;
    console.log('GPU Memory - Geometries:', memoryInfo.geometries, 'Textures:', memoryInfo.textures);
  }
  
  // Force garbage collection if available (development only)
  if (window.gc && Math.random() < 0.001) {
    window.gc();
  }
}

// Update functions for control panel integration
function updateParticleColors(paletteIndex) {
  if (!particles || !particles.userData.origCols) return;
  
  const palette = colorPalettes[paletteIndex];
  const { origCols } = particles.userData;
  const cols = particles.geometry.attributes.color;
  
  for (let i = 0; i < origCols.length; i++) {
    const t = i / Math.max(origCols.length - 1, 1);
    const idx = t * (palette.length - 1);
    const floorIdx = Math.floor(idx);
    const ceilIdx = Math.min(Math.ceil(idx), palette.length - 1);
    const mix = idx % 1;
    
    const color = new THREE.Color().lerpColors(palette[floorIdx], palette[ceilIdx], mix);
    color.multiplyScalar(1.3);
    
    origCols[i] = color;
    
    const i3 = i * 3;
    cols.array[i3] = color.r;
    cols.array[i3 + 1] = color.g;
    cols.array[i3 + 2] = color.b;
  }
  
  cols.needsUpdate = true;
}

function updateParticleSize(size) {
  if (particles && particles.material) {
    particles.material.size = size;
  }
}

function updateParticleOpacity(opacity) {
  if (particles && particles.material) {
    particles.material.opacity = opacity;
  }
}

function updateBloomStrength(strength) {
  if (bloom) {
    bloom.strength = strength;
  }
}

function updateBloomRadius(radius) {
  if (bloom) {
    bloom.radius = radius;
  }
}

// Initialize application
gsapScript.onload = async () => {
  try {
    // Try to initialize WebGPU first
    const webgpuInitialized = await initWebGPU();
    
    if (webgpuInitialized) {
      const statusEl = document.getElementById('webgpuStatus');
      if (statusEl) {
        statusEl.textContent = 'Active';
        statusEl.style.color = '#00ff88';
      }
    } else {
      const statusEl = document.getElementById('webgpuStatus');
      if (statusEl) {
        statusEl.textContent = 'Fallback';
        statusEl.style.color = '#ffaa00';  
      }
    }
    
    await initializeTargets();
    initializeControlPanel(renderer, webgpuSystem, useWebGPU);
    
    // Set up control panel update functions
    window.updateParticleColors = updateParticleColors;
    window.updateParticleSize = updateParticleSize;
    window.updateParticleOpacity = updateParticleOpacity;
    window.updateBloomStrength = updateBloomStrength;
    window.updateBloomRadius = updateBloomRadius;
    
    animate();
    
    // Set up periodic cleanup for mobile devices
    if (isMobile) {
      setInterval(cleanupUnusedResources, 30000); // Every 30 seconds
    }
  } catch (error) {
    console.error('Failed to initialize application:', error);
  }
};