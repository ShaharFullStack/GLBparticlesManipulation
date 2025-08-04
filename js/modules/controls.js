import * as THREE from 'three';

// Mobile detection
const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);

// Control Panel State
const controlState = {
  particleCount: isMobile ? 1500 : 2500,
  particleSize: 0.03,
  particleOpacity: 0.9,
  fibersEnabled: false,
  fiberStrength: 0.5,
  connectionDistance: 2.0,
  gravity: 0,
  turbulence: 0.1,
  attraction: 1.0,
  morphSpeed: 2.0,
  rotationSpeed: 0.005,
  bloomStrength: isMobile ? 1.0 : 1.4,
  bloomRadius: isMobile ? 0.6 : 0.8,
  colorPalette: 0,
  mousePos: { x: 0, y: 0 },
  // Hover integration settings
  hoverEnabled: true,
  hoverRadius: 3.0,
  hoverStrength: 2.0,
  hoverRepulsion: false,
  particleTrail: false,
  mouseInfluence: 1.0,
  autoRotate: true
};

// Color palettes
const colorPalettes = [
  [ // Cyberpunk
    new THREE.Color(0x00ffff),
    new THREE.Color(0xff0080),
    new THREE.Color(0x8000ff),
    new THREE.Color(0x00ff40),
    new THREE.Color(0xff4000),
    new THREE.Color(0x4080ff)
  ],
  [ // Sunset
    new THREE.Color(0xff6b35),
    new THREE.Color(0xf7931e),
    new THREE.Color(0xffd23f),
    new THREE.Color(0xff8c42),
    new THREE.Color(0xff6b6b),
    new THREE.Color(0xffa726)
  ],
  [ // Ocean
    new THREE.Color(0x4ecdc4),
    new THREE.Color(0x44a08d),
    new THREE.Color(0x006064),
    new THREE.Color(0x26c6da),
    new THREE.Color(0x00acc1),
    new THREE.Color(0x0097a7)
  ],
  [ // Galaxy
    new THREE.Color(0x667eea),
    new THREE.Color(0x764ba2),
    new THREE.Color(0x9c27b0),
    new THREE.Color(0x673ab7),
    new THREE.Color(0x3f51b5),
    new THREE.Color(0x5c6bc0)
  ],
  [ // Rose
    new THREE.Color(0xff9a9e),
    new THREE.Color(0xfecfef),
    new THREE.Color(0xff6b9d),
    new THREE.Color(0xf06292),
    new THREE.Color(0xe91e63),
    new THREE.Color(0xad1457)
  ],
  [ // Mint
    new THREE.Color(0xa8edea),
    new THREE.Color(0xfed6e3),
    new THREE.Color(0x81c784),
    new THREE.Color(0x66bb6a),
    new THREE.Color(0x4caf50),
    new THREE.Color(0x388e3c)
  ]
];

// Preset configurations
const presets = {
  organic: {
    particleCount: 3000,
    particleSize: 0.025,
    fibersEnabled: true,
    fiberStrength: 0.3,
    connectionDistance: 1.5,
    gravity: -0.2,
    turbulence: 0.15,
    attraction: 0.8,
    morphSpeed: 1.5,
    rotationSpeed: 0.003,
    colorPalette: 2
  },
  digital: {
    particleCount: 5000,
    particleSize: 0.02,
    fibersEnabled: true,
    fiberStrength: 0.8,
    connectionDistance: 2.5,
    gravity: 0,
    turbulence: 0.05,
    attraction: 1.5,
    morphSpeed: 3.0,
    rotationSpeed: 0.008,
    colorPalette: 0
  },
  cosmic: {
    particleCount: 4000,
    particleSize: 0.04,
    fibersEnabled: false,
    fiberStrength: 0.2,
    connectionDistance: 3.0,
    gravity: 0.1,
    turbulence: 0.2,
    attraction: 0.5,
    morphSpeed: 1.0,
    rotationSpeed: 0.002,
    colorPalette: 3
  },
  minimal: {
    particleCount: 1000,
    particleSize: 0.035,
    fibersEnabled: false,
    fiberStrength: 0.1,
    connectionDistance: 1.0,
    gravity: 0,
    turbulence: 0.02,
    attraction: 2.0,
    morphSpeed: 2.5,
    rotationSpeed: 0.001,
    colorPalette: 4
  }
};

// Control Panel Initialization
function initializeControlPanel(renderer, webgpuSystem, useWebGPU) {
  const toggleBtn = document.getElementById('togglePanel');
  const panel = document.getElementById('controlPanel');
  let isPanelOpen = false;
  
  // Toggle panel visibility
  if (toggleBtn && panel) {
    toggleBtn.addEventListener('click', () => {
      isPanelOpen = !isPanelOpen;
      panel.classList.toggle('open', isPanelOpen);
      toggleBtn.classList.toggle('open', isPanelOpen);
    });
  }
  
  // Initialize all controls
  initializeSliders(webgpuSystem, useWebGPU);
  initializeToggles(webgpuSystem, useWebGPU);
  initializeColorPalette();
  initializePresets(webgpuSystem, useWebGPU);
  
  // Enhanced mouse tracking with hover integration
  let lastMouseTime = 0;
  const mouseTrail = [];
  
  document.addEventListener('mousemove', (e) => {
    const rect = renderer.domElement.getBoundingClientRect();
    const currentTime = performance.now();
    
    // Normalize mouse coordinates to WebGL space
    controlState.mousePos.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    controlState.mousePos.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
    
    // Calculate mouse velocity for dynamic effects
    const timeDiff = currentTime - lastMouseTime;
    if (timeDiff > 16) { // ~60fps throttling
      const mouseVelocity = Math.abs(controlState.mousePos.x - (controlState.lastMouseX || 0)) + 
                           Math.abs(controlState.mousePos.y - (controlState.lastMouseY || 0));
      
      controlState.mouseVelocity = mouseVelocity;
      controlState.lastMouseX = controlState.mousePos.x;
      controlState.lastMouseY = controlState.mousePos.y;
      lastMouseTime = currentTime;
      
      // Particle trail effect
      if (controlState.particleTrail && mouseTrail.length < 10) {
        mouseTrail.push({
          x: controlState.mousePos.x,
          y: controlState.mousePos.y,
          time: currentTime
        });
      }
      
      // Remove old trail points
      while (mouseTrail.length > 0 && currentTime - mouseTrail[0].time > 1000) {
        mouseTrail.shift();
      }
      
      controlState.mouseTrail = mouseTrail;
    }
  });
  
  // Mouse leave event to reset effects
  document.addEventListener('mouseleave', () => {
    controlState.mousePos.x = 0;
    controlState.mousePos.y = 0;
    controlState.mouseVelocity = 0;
  });
}

function initializeSliders(webgpuSystem, useWebGPU) {
  const sliders = {
    particleCount: (value) => {
      controlState.particleCount = parseInt(value);
      if (useWebGPU && webgpuSystem) {
        webgpuSystem.setParticleCount(controlState.particleCount);
      }
      updateParticleCount(controlState.particleCount);
    },
    particleSize: (value) => {
      controlState.particleSize = parseFloat(value);
      updateParticleSize(controlState.particleSize);
    },
    particleOpacity: (value) => {
      controlState.particleOpacity = parseFloat(value);
      updateParticleOpacity(controlState.particleOpacity);
    },
    fiberStrength: (value) => {
      controlState.fiberStrength = parseFloat(value);
    },
    connectionDistance: (value) => {
      controlState.connectionDistance = parseFloat(value);
    },
    gravity: (value) => {
      controlState.gravity = parseFloat(value);
    },
    turbulence: (value) => {
      controlState.turbulence = parseFloat(value);
    },
    attraction: (value) => {
      controlState.attraction = parseFloat(value);
    },
    morphSpeed: (value) => {
      controlState.morphSpeed = parseFloat(value);
    },
    rotationSpeed: (value) => {
      controlState.rotationSpeed = parseFloat(value);
    },
    bloomStrength: (value) => {
      controlState.bloomStrength = parseFloat(value);
      updateBloomStrength(controlState.bloomStrength);
    },
    bloomRadius: (value) => {
      controlState.bloomRadius = parseFloat(value);
      updateBloomRadius(controlState.bloomRadius);
    },
    // Hover integration sliders
    hoverRadius: (value) => {
      controlState.hoverRadius = parseFloat(value);
    },
    hoverStrength: (value) => {
      controlState.hoverStrength = parseFloat(value);
    },
    mouseInfluence: (value) => {
      controlState.mouseInfluence = parseFloat(value);
    }
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
      
      // Initialize display
      valueDisplay.textContent = slider.value;
    }
  });
}

function initializeToggles(webgpuSystem, useWebGPU) {
  const toggles = {
    fibersEnabled: () => {
      controlState.fibersEnabled = !controlState.fibersEnabled;
      if (useWebGPU && webgpuSystem) {
        webgpuSystem.enableFibers(controlState.fibersEnabled);
      }
      
      // Update display
      const fiberDisplay = document.getElementById('fiberDisplay');
      if (fiberDisplay) {
        fiberDisplay.textContent = controlState.fibersEnabled ? 'Active' : 'Disabled';
      }
    },
    hoverEnabled: () => {
      controlState.hoverEnabled = !controlState.hoverEnabled;
    },
    hoverRepulsion: () => {
      controlState.hoverRepulsion = !controlState.hoverRepulsion;
    },
    particleTrail: () => {
      controlState.particleTrail = !controlState.particleTrail;
    },
    autoRotate: () => {
      controlState.autoRotate = !controlState.autoRotate;
    }
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
      // Remove active class from all options
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
      
      // Update active state
      presetButtons.forEach(btn => btn.classList.remove('active'));
      button.classList.add('active');
    });
  });
}

function applyPreset(presetName, webgpuSystem, useWebGPU) {
  const preset = presets[presetName];
  if (!preset) return;
  
  // Apply all preset values with smooth transitions
  Object.keys(preset).forEach(key => {
    if (controlState.hasOwnProperty(key)) {
      controlState[key] = preset[key];
      
      // Update UI controls
      const slider = document.getElementById(key);
      const valueDisplay = document.getElementById(key + 'Value');
      
      if (slider && valueDisplay) {
        slider.value = preset[key];
        valueDisplay.textContent = preset[key];
      }
    }
  });
  
  // Update special controls
  if (preset.fibersEnabled !== undefined) {
    const fibersToggle = document.getElementById('fibersEnabled');
    if (fibersToggle) {
      fibersToggle.classList.toggle('active', preset.fibersEnabled);
    }
  }
  
  if (preset.colorPalette !== undefined) {
    const colorOptions = document.querySelectorAll('.color-option');
    colorOptions.forEach((opt, idx) => {
      opt.classList.toggle('active', idx === preset.colorPalette);
    });
    updateParticleColors(preset.colorPalette);
  }
  
  // Apply to systems
  if (useWebGPU && webgpuSystem) {
    webgpuSystem.setParticleCount(preset.particleCount);
    webgpuSystem.enableFibers(preset.fibersEnabled);
  }
  
  updateParticleCount(preset.particleCount);
  updateParticleSize(preset.particleSize);
  updateParticleOpacity(controlState.particleOpacity);
  updateBloomStrength(controlState.bloomStrength);
  updateBloomRadius(controlState.bloomRadius);
  
  console.log(`Applied preset: ${presetName}`);
}

// Functions that will be called by main app
function updateParticleColors(paletteIndex) {
  if (window.updateParticleColors) {
    window.updateParticleColors(paletteIndex);
  }
}

function updateParticleCount(newCount) {
  const display = document.getElementById('particleDisplay');
  if (display) {
    display.textContent = newCount;
  }
}

function updateParticleSize(size) {
  if (window.updateParticleSize) {
    window.updateParticleSize(size);
  }
}

function updateParticleOpacity(opacity) {
  if (window.updateParticleOpacity) {
    window.updateParticleOpacity(opacity);
  }
}

function updateBloomStrength(strength) {
  if (window.updateBloomStrength) {
    window.updateBloomStrength(strength);
  }
}

function updateBloomRadius(radius) {
  if (window.updateBloomRadius) {
    window.updateBloomRadius(radius);
  }
}

export { 
  controlState, 
  colorPalettes, 
  presets, 
  initializeControlPanel,
  updateParticleColors,
  updateParticleCount,
  updateParticleSize,
  updateParticleOpacity,
  updateBloomStrength,
  updateBloomRadius
};