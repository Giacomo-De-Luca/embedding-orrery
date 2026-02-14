/**
 * HazeRenderer — drop-in replacement for BloomRenderer.
 *
 * Uses the same technique as pickles976/GalaxyThreeJS:
 *   THREE.Sprite + feathered texture + distance-based opacity fade
 *
 * Instead of bloom post-processing, soft feathered sprites are placed at a
 * sampled subset of cluster points. Dense regions accumulate more sprites,
 * creating a natural nebula/haze glow. The overlay canvas with
 * mix-blend-mode: screen handles the additive compositing.
 *
 * Architecture (unchanged from BloomRenderer):
 *   - Dedicated overlay <canvas> with its own WebGL2 context
 *   - Camera synced from Plotly's glplot matrices each frame
 *   - mix-blend-mode: screen (black = invisible, bright = additive glow)
 *
 * No EffectComposer, no bloom pass, no render targets.
 */

import * as THREE from 'three';
import type { ClusterData } from './clusterGeometry';

// ---------------------------------------------------------------------------
// Config — tune these to taste
// ---------------------------------------------------------------------------

/**
 * Sprite count scales with cluster size (for spatial coverage) but opacity
 * scales inversely (to keep total accumulated brightness roughly constant).
 *
 * The "light budget" is: spriteCount × opacity ≈ constant
 *
 * Reference point: a 500-point cluster gets ~75 sprites at opacity 0.10
 * → budget ≈ 7.5
 *
 * A 50k-point cluster gets ~400 sprites at opacity ~0.019 → budget ≈ 7.6
 */
const LIGHT_BUDGET = 7.5;

/** Sprite count: sqrt scaling gives good coverage without explosion */
const SPRITE_COUNT_FACTOR = 3.5;  // sprites ≈ factor × √(pointCount)
const HAZE_MIN_COUNT = 8;
const HAZE_MAX_COUNT = 500;

/** Sprite scale range (multiplied by cluster extent) */
const HAZE_SCALE_MIN = 0.15;
const HAZE_SCALE_MAX = 0.5;

/** Opacity floor — never go fully invisible */
const HAZE_OPACITY_MIN = 0.001;
/** Opacity ceiling — small clusters don't blind you either */
const HAZE_OPACITY_MAX = 0.10;

/**
 * Small-cluster boost: scale up sprite size for clusters with few points
 * so they still look diffuse rather than tight/invisible.
 */
const SMALL_CLUSTER_THRESHOLD = 200;
const SMALL_CLUSTER_SCALE_BOOST = 2.0;

/** Extra large sprites at cluster centroid for a bright core */
const CORE_SPRITES = 3;
const CORE_SCALE_MULT = 1.5;

// ---------------------------------------------------------------------------
// Feathered circle texture (loaded once, shared across all sprites)
// ---------------------------------------------------------------------------

let hazeTexture: THREE.Texture | null = null;
function getHazeTexture(): THREE.Texture {
  if (!hazeTexture) {
    hazeTexture = new THREE.TextureLoader().load('/feathered60.png');
  }
  return hazeTexture;
}

// ---------------------------------------------------------------------------
// Renderer
// ---------------------------------------------------------------------------

export class HazeRenderer {
  private renderer: THREE.WebGLRenderer;
  private scene: THREE.Scene;
  private camera: THREE.PerspectiveCamera;
  private group: THREE.Group;
  private sprites: THREE.Sprite[] = [];
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
   * Rebuild haze sprites for all clusters.
   * Samples a subset of each cluster's points and places feathered sprites there,
   * plus a few extra at the centroid for a bright core.
   */
  updateClusters(clusterDataMap: Map<string, ClusterData>, totalPoints: number): void {
    // Tear down existing sprites
    this.clearSprites();

    const texture = getHazeTexture();
    const logTotal = Math.log2(Math.max(totalPoints, 2));

    for (const [, cluster] of clusterDataMap) {
      if (cluster.points.length < 10) continue;

      const color = new THREE.Color(cluster.color);

      // Compute cluster extent for sizing sprites
      const cx = cluster.centroid.x;
      const cy = cluster.centroid.y;
      const cz = cluster.centroid.z;

      let maxDist = 0;
      for (const p of cluster.points) {
        const d = Math.sqrt((p.x - cx) ** 2 + (p.y - cy) ** 2 + (p.z - cz) ** 2);
        if (d > maxDist) maxDist = d;
      }
      const extent = maxDist || 1;

      // --- Compute per-cluster adaptive parameters ---

      // Sprite count: sqrt scaling — covers shape without exploding
      const n = cluster.points.length;
      const hazeCount = Math.min(
        Math.max(Math.round(SPRITE_COUNT_FACTOR * Math.sqrt(n)), HAZE_MIN_COUNT),
        HAZE_MAX_COUNT,
      );

      // Opacity: per-cluster peak brightness ≈ LIGHT_BUDGET / log2(totalPoints)
      // Dividing by hazeCount ensures consistent peak regardless of cluster size.
      //   500 pts  → peak ≈ 0.83
      //   5k pts   → peak ≈ 0.61
      //   50k pts  → peak ≈ 0.48
      //   150k pts → peak ≈ 0.44
      const hazeOpacity = Math.min(
        Math.max(LIGHT_BUDGET / (hazeCount * logTotal), HAZE_OPACITY_MIN),
        HAZE_OPACITY_MAX,
      );

      // Scale boost for small clusters
      const scaleBoost = n < SMALL_CLUSTER_THRESHOLD
        ? SMALL_CLUSTER_SCALE_BOOST * (1 - n / SMALL_CLUSTER_THRESHOLD) + 1.0
        : 1.0;

      // Shuffle points (Fisher-Yates) then Poisson-disk reject to spread sprites evenly
      const indices = Array.from({ length: cluster.points.length }, (_, i) => i);
      for (let i = indices.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [indices[i], indices[j]] = [indices[j], indices[i]];
      }

      const minDist = extent * 0.03;
      const minDist2 = minDist * minDist;
      const accepted: { x: number; y: number; z: number }[] = [];

      for (const idx of indices) {
        const c = cluster.points[idx];
        let tooClose = false;
        for (const a of accepted) {
          if ((c.x - a.x) ** 2 + (c.y - a.y) ** 2 + (c.z - a.z) ** 2 < minDist2) {
            tooClose = true;
            break;
          }
        }
        if (!tooClose) accepted.push(c);
        if (accepted.length >= hazeCount) break;
      }

      // Place haze sprites at accepted positions
      for (const p of accepted) {

        const material = new THREE.SpriteMaterial({
          map: texture,
          color,
          opacity: hazeOpacity,
          transparent: true,
          depthTest: false,
          depthWrite: false,
          blending: THREE.AdditiveBlending,
        });

        const sprite = new THREE.Sprite(material);
        sprite.position.set(p.x, p.y, p.z);

        // Random scale within range, proportional to cluster extent, boosted for small clusters
        const scale = extent * (HAZE_SCALE_MIN + Math.random() * (HAZE_SCALE_MAX - HAZE_SCALE_MIN)) * scaleBoost;
        sprite.scale.set(scale, scale, 1);

        this.group.add(sprite);
        this.sprites.push(sprite);
      }

      // Centroid core sprites — larger, slightly brighter
      // Core opacity tracks the haze opacity
      const coreOpacity = hazeOpacity * 0.6;

      for (let i = 0; i < CORE_SPRITES; i++) {
        const material = new THREE.SpriteMaterial({
          map: texture,
          color,
          opacity: coreOpacity,
          transparent: true,
          depthTest: false,
          depthWrite: false,
          blending: THREE.AdditiveBlending,
        });

        const sprite = new THREE.Sprite(material);

        // Slight jitter around centroid
        sprite.position.set(
          cx + (Math.random() - 0.5) * extent * 0.1,
          cy + (Math.random() - 0.5) * extent * 0.1,
          cz + (Math.random() - 0.5) * extent * 0.1,
        );

        const scale = extent * HAZE_SCALE_MAX * CORE_SCALE_MULT * (0.8 + Math.random() * 0.4) * scaleBoost;
        sprite.scale.set(scale, scale, 1);

        this.group.add(sprite);
        this.sprites.push(sprite);
      }
    }
  }

  /** Resize the renderer. Width/height are CSS pixels. */
  resize(width: number, height: number): void {
    if (width > 0 && height > 0 && (width !== this.lastW || height !== this.lastH)) {
      this.renderer.setSize(width, height, false);
      this.lastW = width;
      this.lastH = height;
    }
  }

  /**
   * Render one frame. Called from glplot.onrender after Plotly has drawn.
   * Matrices come directly from Plotly's camera params.
   *
   * Optionally applies distance-based opacity fade (like pickles976's
   * Haze.updateScale) so sprites close to the camera fade out to avoid
   * obscuring points the user is inspecting.
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

    // Distance-based opacity fade (optional — comment out if not needed)
    // Fades haze when camera is very close, so points stay readable.
    const camPos = new THREE.Vector3();
    camPos.setFromMatrixPosition(this.camera.matrixWorld);
    for (const sprite of this.sprites) {
      const dist = sprite.position.distanceTo(camPos);
      const fadeStart = 0.5;  // below this distance, start fading
      const fadeFactor = Math.min(dist / fadeStart, 1.0);
      const baseMat = sprite.material as THREE.SpriteMaterial;
      // Preserve the original opacity intent (HAZE_OPACITY or CORE_OPACITY)
      // by storing it on the material.userData
      if (baseMat.userData.baseOpacity === undefined) {
        baseMat.userData.baseOpacity = baseMat.opacity;
      }
      baseMat.opacity = baseMat.userData.baseOpacity * fadeFactor * fadeFactor;
    }

    this.renderer.render(this.scene, this.camera);
  }

  dispose(): void {
    this.clearSprites();
    this.renderer.dispose();
  }

  private clearSprites(): void {
    for (const sprite of this.sprites) {
      this.group.remove(sprite);
      sprite.material.dispose();
    }
    this.sprites = [];
  }
}
