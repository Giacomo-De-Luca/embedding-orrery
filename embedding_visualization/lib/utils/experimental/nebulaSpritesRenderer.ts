/**
 * NebulaSpritesRenderer — drop-in replacement for BloomRenderer.
 *
 * Instead of single-pixel bloom seeds + UnrealBloomPass post-processing,
 * this renders clouds of soft Gaussian sprites with additive blending.
 * Dense cluster regions naturally glow brighter as sprites accumulate.
 *
 * Architecture (unchanged from BloomRenderer):
 *   - Dedicated overlay <canvas> with its own WebGL2 context
 *   - Camera synced from Plotly's glplot matrices each frame
 *   - mix-blend-mode: screen on the canvas (black = invisible, bright = additive glow)
 *
 * No EffectComposer, no bloom pass, no render targets — just raw Points + a shader.
 */

import * as THREE from 'three';
import type { ClusterData } from './clusterGeometry';

// ---------------------------------------------------------------------------
// Shaders
// ---------------------------------------------------------------------------

const VERT = /* glsl */ `
  attribute float aSize;
  attribute float aOpacity;

  uniform float uSizeScale;

  varying float vOpacity;

  void main() {
    vOpacity = aOpacity;

    vec4 mvPos = modelViewMatrix * vec4(position, 1.0);

    // Size in pixels: data-space size → screen pixels via perspective.
    // projectionMatrix[1][1] ≈ 1/tan(fov/2), so this accounts for the
    // camera's field-of-view. uSizeScale carries viewport height / 2.
    gl_PointSize = aSize * uSizeScale * projectionMatrix[1][1] / -mvPos.z;

    // Clamp to GPU max (typically 64–256 px). Large sprites that hit this
    // limit still look fine — they just flatten out at maximum softness.
    gl_PointSize = clamp(gl_PointSize, 1.0, 512.0);

    gl_Position = projectionMatrix * mvPos;
  }
`;

const FRAG = /* glsl */ `
  uniform vec3 uColor;
  varying float vOpacity;

  void main() {
    // Distance from sprite center (gl_PointCoord is [0,1]²)
    float r = length(gl_PointCoord - vec2(0.5)) * 2.0; // normalized to [0,1]
    if (r > 1.0) discard;

    // Smooth Gaussian-ish falloff.
    // pow(1-r, exponent) controls the shape:
    //   1.0 = linear cone (harsh)
    //   2.0 = parabolic (good default)
    //   3.0 = tighter core, more diffuse edge
    float glow = pow(1.0 - r, 2.5);

    // Extra softness at the very edge to avoid visible circle boundaries
    // when sprites don't fully overlap.
    glow *= smoothstep(1.0, 0.8, r);

    gl_FragColor = vec4(uColor, glow * vOpacity);
  }
`;

// ---------------------------------------------------------------------------
// Sprite generation helpers
// ---------------------------------------------------------------------------

interface SpriteCloud {
  positions: Float32Array;   // xyz interleaved
  sizes: Float32Array;       // data-space radius per sprite
  opacities: Float32Array;   // base opacity per sprite
}

/**
 * Generate a cloud of sprites that traces the shape of a cluster.
 *
 * Strategy: sample positions from actual cluster points (so the nebula
 * follows the cluster's real shape, not just a sphere around the centroid),
 * then add Gaussian jitter so it extends slightly beyond the point cloud.
 *
 * Three tiers of sprites create layered depth:
 *   - Large diffuse (20%)  → outer halo / "gas"
 *   - Medium (40%)         → structural body
 *   - Small bright (40%)   → inner detail / bright core
 */
function generateSpriteCloud(cluster: ClusterData, opts: {
  /** Max sprites per cluster (scaled down for small clusters) */
  maxSprites?: number;
  /** Jitter multiplier on cluster std-dev (0 = no spread beyond points) */
  jitter?: number;
  /** Base size multiplier — tune to match your data scale */
  sizeScale?: number;
}): SpriteCloud {
  const {
    maxSprites = 400,
    jitter = 0.4,
    sizeScale = 1.0,
  } = opts;

  const n = cluster.points.length;
  // Scale sprite count with cluster size, but cap it
  const spriteCount = Math.min(Math.max(Math.round(n * 1.5), 40), maxSprites);

  const cx = cluster.centroid.x;
  const cy = cluster.centroid.y;
  const cz = cluster.centroid.z;

  // Compute per-axis std dev for anisotropic jitter
  let vx = 0, vy = 0, vz = 0;
  for (const p of cluster.points) {
    vx += (p.x - cx) ** 2;
    vy += (p.y - cy) ** 2;
    vz += (p.z - cz) ** 2;
  }
  const sx = Math.sqrt(vx / n) || 0.01;
  const sy = Math.sqrt(vy / n) || 0.01;
  const sz = Math.sqrt(vz / n) || 0.01;

  // Characteristic radius (geometric mean of axes) — used to scale sprite sizes
  const charRadius = Math.cbrt(sx * sy * sz);

  const positions: number[] = [];
  const sizes: number[] = [];
  const opacities: number[] = [];

  // Box-Muller for Gaussian jitter (avoids library dependency)
  const gaussRand = () => {
    const u1 = Math.random() || 1e-10;
    const u2 = Math.random();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  };

  for (let i = 0; i < spriteCount; i++) {
    // Pick a random source point from the cluster
    const src = cluster.points[Math.floor(Math.random() * n)];

    // Add anisotropic Gaussian jitter
    const px = src.x + gaussRand() * sx * jitter;
    const py = src.y + gaussRand() * sy * jitter;
    const pz = src.z + gaussRand() * sz * jitter;
    positions.push(px, py, pz);

    // Tier assignment — large & subtle, like a faint colored atmosphere
    const roll = Math.random();
    if (roll < 0.2) {
      // Large diffuse — outer halo
      sizes.push(charRadius * (3.0 + Math.random() * 3.0) * sizeScale);
      opacities.push(0.002 + Math.random() * 0.004);
    } else if (roll < 0.6) {
      // Medium — structural body
      sizes.push(charRadius * (1.2 + Math.random() * 1.6) * sizeScale);
      opacities.push(0.005 + Math.random() * 0.007);
    } else {
      // Small — inner detail
      sizes.push(charRadius * (0.4 + Math.random() * 0.6) * sizeScale);
      opacities.push(0.008 + Math.random() * 0.010);
    }
  }

  // Extra centroid glow: a handful of large, very soft sprites at the center
  const coreCount = Math.min(Math.ceil(spriteCount * 0.05), 8);
  for (let i = 0; i < coreCount; i++) {
    positions.push(
      cx + gaussRand() * sx * 0.15,
      cy + gaussRand() * sy * 0.15,
      cz + gaussRand() * sz * 0.15,
    );
    sizes.push(charRadius * (4.0 + Math.random() * 3.0) * sizeScale);
    opacities.push(0.003 + Math.random() * 0.003);
  }

  return {
    positions: new Float32Array(positions),
    sizes: new Float32Array(sizes),
    opacities: new Float32Array(opacities),
  };
}

// ---------------------------------------------------------------------------
// Renderer
// ---------------------------------------------------------------------------

export class NebulaSpritesRenderer {
  private renderer: THREE.WebGLRenderer;
  private scene: THREE.Scene;
  private camera: THREE.PerspectiveCamera;
  private group: THREE.Group;
  private lastW = 0;
  private lastH = 0;

  constructor(canvas: HTMLCanvasElement) {
    this.renderer = new THREE.WebGLRenderer({ canvas, alpha: false, antialias: false });
    this.renderer.setClearColor(0x000000, 1); // opaque black — invisible with screen blend
    this.renderer.setPixelRatio(1);            // match Plotly's CSS-pixel viewport

    this.scene = new THREE.Scene();

    this.camera = new THREE.PerspectiveCamera();
    this.camera.matrixAutoUpdate = false;

    this.group = new THREE.Group();
    this.group.matrixAutoUpdate = false;
    this.scene.add(this.group);

    const w = canvas.width || 1;
    const h = canvas.height || 1;
    this.lastW = w;
    this.lastH = h;
    this.renderer.setSize(w, h, false);
  }

  /**
   * Rebuild sprite clouds for all clusters.
   * Call when cluster data changes (new embedding, new clustering, etc.)
   */
  updateClusters(clusterDataMap: Map<string, ClusterData>): void {
    // Tear down existing geometry
    while (this.group.children.length > 0) {
      const child = this.group.children[0];
      this.group.remove(child);
      if (child instanceof THREE.Points) {
        child.geometry.dispose();
        (child.material as THREE.Material).dispose();
      }
    }

    for (const [, cluster] of clusterDataMap) {
      if (cluster.points.length < 10) continue;

      const cloud = generateSpriteCloud(cluster, {
        maxSprites: 400,
        jitter: 0.4,
        sizeScale: 1.0,
      });

      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(cloud.positions, 3));
      geo.setAttribute('aSize', new THREE.BufferAttribute(cloud.sizes, 1));
      geo.setAttribute('aOpacity', new THREE.BufferAttribute(cloud.opacities, 1));

      const color = new THREE.Color(cluster.color);

      const mat = new THREE.ShaderMaterial({
        vertexShader: VERT,
        fragmentShader: FRAG,
        uniforms: {
          uColor: { value: color },
          uSizeScale: { value: this.lastH * 0.5 },
        },
        transparent: true,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
        depthTest: true,
      });

      this.group.add(new THREE.Points(geo, mat));
    }
  }

  /** Resize the renderer + update the size uniform. */
  resize(width: number, height: number): void {
    if (width > 0 && height > 0 && (width !== this.lastW || height !== this.lastH)) {
      this.renderer.setSize(width, height, false);
      this.lastW = width;
      this.lastH = height;

      // Update the size scale uniform on all materials
      this.group.traverse((obj) => {
        if (obj instanceof THREE.Points) {
          const mat = obj.material as THREE.ShaderMaterial;
          if (mat.uniforms?.uSizeScale) {
            mat.uniforms.uSizeScale.value = height * 0.5;
          }
        }
      });
    }
  }

  /**
   * Render one frame. Called from glplot.onrender after Plotly has drawn.
   * Matrices come directly from Plotly's camera params.
   */
  render(projection: Float32Array, view: Float32Array, model: Float32Array): void {
    // Sync camera from Plotly's glplot
    this.camera.projectionMatrix.fromArray(projection);
    this.camera.projectionMatrixInverse.copy(this.camera.projectionMatrix).invert();
    this.camera.matrixWorldInverse.fromArray(view);
    this.camera.matrixWorld.copy(this.camera.matrixWorldInverse).invert();

    // Model matrix → group transform (data coords → scene coords)
    this.group.matrix.fromArray(model);
    this.group.matrixWorldNeedsUpdate = true;

    // No composer, no post-processing — just a straight render.
    // Additive blending on the sprites + screen blend on the canvas = nebula.
    this.renderer.render(this.scene, this.camera);
  }

  dispose(): void {
    this.group.traverse((obj) => {
      if (obj instanceof THREE.Points) {
        obj.geometry.dispose();
        (obj.material as THREE.Material).dispose();
      }
    });
    this.renderer.dispose();
  }
}
