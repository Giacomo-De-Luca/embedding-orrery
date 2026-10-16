/**
 * DensityHazeRenderer — density-based nebula haze (experimental alternative to
 * the sprite-based `HazeRenderer`, same lifecycle contract).
 *
 * Per frame:
 *   1. Splat: ONE `THREE.Points` draw of every clustered point into a half-float,
 *      half-resolution render target with additive blending. Each point is a
 *      Gaussian whose sigma is set per cluster (data units, converted to pixels by
 *      the projection, so halos are world-space and zoom-consistent). RGB
 *      accumulates colour × density; alpha accumulates plain density, which the
 *      resolve does not read yet (reserved for a mean-colour mode).
 *   2. Halo: a separable Gaussian blur of that buffer at quarter resolution,
 *      added back at `haloWeight` — glow that spreads beyond the points at a cost
 *      independent of point count.
 *   3. Resolve: fullscreen quad → exposure → hue-preserving soft knee
 *      (1 − exp(−I^lift)) → the overlay canvas. Screen/multiply compositing with
 *      the Plotly canvas is unchanged.
 *
 * The brightness rule (kernel, per-cluster weight, exposure) lives in
 * `densityHazeMath.ts` and is unit-tested; this file only owns GL resources.
 *
 * Requires WebGL2 + EXT_color_buffer_float (`isSupported()`); callers fall back
 * to the sprite renderer otherwise.
 */

import * as THREE from 'three';
import type { ClusterData } from '../clusterGeometry';
import { isDensityHazeSupported } from './support';
import {
  buildDensityHazeBuffers,
  reweightDensityHazeBuffers,
  DEFAULT_DENSITY_HAZE_TUNING,
  SPRITE_SIGMA_SPAN,
  CLAMP_COMPENSATION_GLSL,
  type DensityHazeBuffers,
  type DensityHazeTuning,
} from './densityHazeMath';

export interface DensityHazeOptions extends DensityHazeTuning {
  /** Soft-knee exponent: out = 1 − exp(−I^lift). Lower lifts faint haze more. */
  lift: number;
  /** Weight of the blurred copy added back as the halo. */
  haloWeight: number;
  /** Halo blur sigma in canvas pixels. */
  haloBlurPx: number;
  /** Point-sprite diameter clamps in render-target pixels (a sprite spans ±3σ). */
  minPointPx: number;
  maxPointPx: number;
  /** Accumulation target resolution relative to the canvas. */
  resolutionScale: number;
  /** Light mode: multiply-blend strength of the inverted haze. */
  lightModeStrength: number;
}

export const DEFAULT_DENSITY_HAZE_OPTIONS: DensityHazeOptions = {
  ...DEFAULT_DENSITY_HAZE_TUNING,
  lift: 0.8,
  haloWeight: 0.5,
  haloBlurPx: 12,
  minPointPx: 2,
  maxPointPx: 32,
  resolutionScale: 0.5,
  lightModeStrength: 0.85,
};

// ---------------------------------------------------------------------------
// Shaders (GLSL 1 syntax; three.js rewrites attribute/varying/texture2D for WebGL2)
// ---------------------------------------------------------------------------

const SPLAT_VERT = /* glsl */ `
  attribute vec3 aColor;
  attribute float aWeight;
  attribute float aSigma;

  uniform float uViewportH;
  uniform float uMinPx;
  uniform float uMaxPx;

  varying vec3 vColor;
  varying float vWeight;

  ${CLAMP_COMPENSATION_GLSL}

  void main() {
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    // data → scene scale is uniform (see buildDataToSceneMatrix)
    float modelScale = length(vec3(modelMatrix[0].x, modelMatrix[0].y, modelMatrix[0].z));
    float depth = max(-mv.z, 1e-4);
    // scene-space sigma → pixels: NDC half-height = P[1][1] / depth, times half the viewport
    float sigmaPx = aSigma * modelScale * projectionMatrix[1][1] * uViewportH / (2.0 * depth);
    float ideal = ${SPRITE_SIGMA_SPAN.toFixed(1)} * sigmaPx;   // sprite spans ±(span/2)σ
    float size = clamp(ideal, uMinPx, uMaxPx);
    float comp = clampCompensation(ideal, size);   // undo the area gain of the minimum clamp only
    gl_PointSize = size;
    gl_Position = projectionMatrix * mv;
    vColor = aColor;
    vWeight = aWeight * comp;
  }
`;

const SPLAT_FRAG = /* glsl */ `
  varying vec3 vColor;
  varying float vWeight;

  void main() {
    vec2 u = (gl_PointCoord - 0.5) * ${SPRITE_SIGMA_SPAN.toFixed(1)};   // σ units across the sprite
    float r2 = dot(u, u);
    if (r2 > ${((SPRITE_SIGMA_SPAN / 2) ** 2).toFixed(1)}) discard;
    float g = exp(-0.5 * r2) * vWeight;
    gl_FragColor = vec4(vColor * g, g);
  }
`;

const QUAD_VERT = /* glsl */ `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = vec4(position.xy, 0.0, 1.0);
  }
`;

/** 17-tap separable Gaussian; `uStep` is one output pixel expressed in source UV units. */
const BLUR_FRAG = /* glsl */ `
  uniform sampler2D tSrc;
  uniform vec2 uStep;
  uniform float uSigma;
  varying vec2 vUv;

  void main() {
    vec4 acc = vec4(0.0);
    float wsum = 0.0;
    for (int i = -8; i <= 8; i++) {
      float fi = float(i);
      float w = exp(-0.5 * fi * fi / (uSigma * uSigma));
      acc += texture2D(tSrc, vUv + uStep * fi) * w;
      wsum += w;
    }
    gl_FragColor = acc / wsum;
  }
`;

const RESOLVE_FRAG = /* glsl */ `
  uniform sampler2D tAcc;
  uniform sampler2D tHalo;
  uniform float uExposure;
  uniform float uHaloWeight;
  uniform float uLift;
  uniform float uLightMode;
  uniform float uLightStrength;
  varying vec2 vUv;

  void main() {
    vec3 c = (texture2D(tAcc, vUv).rgb + uHaloWeight * texture2D(tHalo, vUv).rgb) * uExposure;
    float L = max(max(c.r, c.g), c.b);
    // hue-preserving soft knee: approaches but never reaches white
    float knee = L > 1e-6 ? (1.0 - exp(-pow(L, uLift))) / L : 0.0;
    if (uLightMode > 0.5) {
      // multiply-blended over the white page: pull white toward the haze hue by its strength
      vec3 hue = c / max(L, 1e-6);
      float a = L > 1e-6 ? (1.0 - exp(-pow(L, uLift))) * uLightStrength : 0.0;
      gl_FragColor = vec4(mix(vec3(1.0), hue, a), 1.0);
    } else {
      gl_FragColor = vec4(c * knee, 1.0);   // screen-blended in dark mode
    }
  }
`;

// ---------------------------------------------------------------------------
// Renderer
// ---------------------------------------------------------------------------

export class DensityHazeRenderer {
  /** WebGL2 with renderable float targets (see `support.ts`). */
  static isSupported(): boolean {
    return isDensityHazeSupported();
  }

  private readonly renderer: THREE.WebGLRenderer;
  private readonly camera = new THREE.PerspectiveCamera();
  private readonly splatScene = new THREE.Scene();
  private readonly group = new THREE.Group();
  private readonly points: THREE.Points;
  private readonly splatMaterial: THREE.ShaderMaterial;
  private readonly quadScene = new THREE.Scene();
  private readonly quadCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  private readonly quad: THREE.Mesh;
  private readonly blurMaterial: THREE.ShaderMaterial;
  private readonly resolveMaterial: THREE.ShaderMaterial;
  private readonly accTarget: THREE.WebGLRenderTarget;
  private readonly haloTargetA: THREE.WebGLRenderTarget;
  private readonly haloTargetB: THREE.WebGLRenderTarget;
  private readonly isDark: boolean;
  private options: DensityHazeOptions;
  private buffers: DensityHazeBuffers | null = null;
  private lastClusterMap: Map<string, ClusterData> | null = null;
  private lastW = 0;
  private lastH = 0;

  constructor(canvas: HTMLCanvasElement, isDark = true, options: Partial<DensityHazeOptions> = {}) {
    this.isDark = isDark;
    this.options = { ...DEFAULT_DENSITY_HAZE_OPTIONS, ...options };

    this.renderer = new THREE.WebGLRenderer({ canvas, alpha: false, antialias: false });
    this.renderer.setPixelRatio(1);            // match Plotly's CSS-pixel viewport
    this.renderer.autoClear = false;           // targets are cleared explicitly

    this.camera.matrixAutoUpdate = false;
    this.group.matrixAutoUpdate = false;
    this.splatScene.add(this.group);

    this.splatMaterial = new THREE.ShaderMaterial({
      vertexShader: SPLAT_VERT,
      fragmentShader: SPLAT_FRAG,
      uniforms: {
        uViewportH: { value: 1 },
        uMinPx: { value: this.options.minPointPx },
        uMaxPx: { value: this.options.maxPointPx },
      },
      transparent: true,
      depthTest: false,
      depthWrite: false,
      blending: THREE.CustomBlending,
      blendEquation: THREE.AddEquation,
      blendSrc: THREE.OneFactor,
      blendDst: THREE.OneFactor,
      blendSrcAlpha: THREE.OneFactor,
      blendDstAlpha: THREE.OneFactor,
    });
    this.points = new THREE.Points(new THREE.BufferGeometry(), this.splatMaterial);
    this.points.frustumCulled = false;
    this.group.add(this.points);

    this.blurMaterial = new THREE.ShaderMaterial({
      vertexShader: QUAD_VERT,
      fragmentShader: BLUR_FRAG,
      uniforms: {
        tSrc: { value: null },
        uStep: { value: new THREE.Vector2(0, 0) },
        uSigma: { value: 1 },
      },
      depthTest: false,
      depthWrite: false,
    });
    this.resolveMaterial = new THREE.ShaderMaterial({
      vertexShader: QUAD_VERT,
      fragmentShader: RESOLVE_FRAG,
      uniforms: {
        tAcc: { value: null },
        tHalo: { value: null },
        uExposure: { value: 1 },
        uHaloWeight: { value: this.options.haloWeight },
        uLift: { value: this.options.lift },
        uLightMode: { value: isDark ? 0 : 1 },
        uLightStrength: { value: this.options.lightModeStrength },
      },
      depthTest: false,
      depthWrite: false,
    });
    this.quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this.resolveMaterial);
    this.quad.frustumCulled = false;
    this.quadScene.add(this.quad);

    this.accTarget = DensityHazeRenderer.createTarget();
    this.haloTargetA = DensityHazeRenderer.createTarget();
    this.haloTargetB = DensityHazeRenderer.createTarget();

    this.resize(canvas.width || 1, canvas.height || 1);
  }

  private static createTarget(): THREE.WebGLRenderTarget {
    return new THREE.WebGLRenderTarget(1, 1, {
      type: THREE.HalfFloatType,
      format: THREE.RGBAFormat,
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      depthBuffer: false,
      stencilBuffer: false,
      generateMipmaps: false,
    });
  }

  /** Rebuild the point buffers for the given clusters. `totalPoints` is unused (sprite-API parity). */
  updateClusters(clusterDataMap: Map<string, ClusterData>, _totalPoints?: number): void {
    this.lastClusterMap = clusterDataMap;
    this.buffers = buildDensityHazeBuffers(clusterDataMap, this.options);
    this.uploadBuffers();
  }

  /** Live tuning: gamma/exposure reweight in place, a kernel change rebuilds. */
  setTuning(tuning: Partial<DensityHazeTuning>): void {
    const next = { ...this.options, ...tuning };
    const kernelChanged = next.kernelFrac !== this.options.kernelFrac;
    this.options = next;
    if (!this.buffers) return;
    if (kernelChanged && this.lastClusterMap) {
      this.buffers = buildDensityHazeBuffers(this.lastClusterMap, this.options);
      this.uploadBuffers();
      return;
    }
    this.buffers = reweightDensityHazeBuffers(this.buffers, this.options);
    const weightAttr = this.points.geometry.getAttribute('aWeight') as THREE.BufferAttribute | undefined;
    if (weightAttr) {
      weightAttr.array = this.buffers.weights;
      weightAttr.needsUpdate = true;
    }
    this.resolveMaterial.uniforms.uExposure.value = this.buffers.exposure;
  }

  private uploadBuffers(): void {
    const buffers = this.buffers;
    if (!buffers) return;
    // Replace the geometry wholesale so the previous GL buffers are released.
    this.points.geometry.dispose();
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(buffers.positions, 3));
    geometry.setAttribute('aColor', new THREE.BufferAttribute(buffers.colors, 3));
    geometry.setAttribute('aWeight', new THREE.BufferAttribute(buffers.weights, 1));
    geometry.setAttribute('aSigma', new THREE.BufferAttribute(buffers.sigmas, 1));
    geometry.setDrawRange(0, buffers.count);
    this.points.geometry = geometry;
    this.resolveMaterial.uniforms.uExposure.value = buffers.exposure;
  }

  /** Resize the canvas and the offscreen targets. Width/height are CSS pixels. */
  resize(width: number, height: number): void {
    if (width <= 0 || height <= 0 || (width === this.lastW && height === this.lastH)) return;
    this.lastW = width;
    this.lastH = height;
    this.renderer.setSize(width, height, false);

    const scale = this.options.resolutionScale;
    const accW = Math.max(1, Math.round(width * scale));
    const accH = Math.max(1, Math.round(height * scale));
    const haloW = Math.max(1, Math.round(accW / 2));
    const haloH = Math.max(1, Math.round(accH / 2));
    this.accTarget.setSize(accW, accH);
    this.haloTargetA.setSize(haloW, haloH);
    this.haloTargetB.setSize(haloW, haloH);

    this.splatMaterial.uniforms.uViewportH.value = accH;
    // blur sigma is specified in canvas px; the halo targets are at scale/2
    this.blurMaterial.uniforms.uSigma.value = Math.max(0.5, this.options.haloBlurPx * scale * 0.5);
  }

  /** Render one frame with Plotly's camera matrices (called from glplot.onrender). */
  render(projection: Float32Array, view: Float32Array, model: Float32Array): void {
    this.camera.projectionMatrix.fromArray(projection);
    this.camera.projectionMatrixInverse.copy(this.camera.projectionMatrix).invert();
    this.camera.matrixWorldInverse.fromArray(view);
    this.camera.matrixWorld.copy(this.camera.matrixWorldInverse).invert();
    this.group.matrix.fromArray(model);
    this.group.matrixWorldNeedsUpdate = true;

    const r = this.renderer;

    // 1. splat
    r.setRenderTarget(this.accTarget);
    r.setClearColor(0x000000, 0);
    r.clear(true, false, false);
    if (this.buffers && this.buffers.count > 0) r.render(this.splatScene, this.camera);

    // 2. halo: horizontal (half → quarter res) then vertical (quarter → quarter)
    const accW = this.accTarget.width;
    const haloH = this.haloTargetA.height;
    this.quad.material = this.blurMaterial;
    this.blurMaterial.uniforms.tSrc.value = this.accTarget.texture;
    this.blurMaterial.uniforms.uStep.value.set(2 / accW, 0);
    r.setRenderTarget(this.haloTargetA);
    r.render(this.quadScene, this.quadCamera);
    this.blurMaterial.uniforms.tSrc.value = this.haloTargetA.texture;
    this.blurMaterial.uniforms.uStep.value.set(0, 1 / haloH);
    r.setRenderTarget(this.haloTargetB);
    r.render(this.quadScene, this.quadCamera);

    // 3. resolve to the canvas
    this.quad.material = this.resolveMaterial;
    this.resolveMaterial.uniforms.tAcc.value = this.accTarget.texture;
    this.resolveMaterial.uniforms.tHalo.value = this.haloTargetB.texture;
    r.setRenderTarget(null);
    r.render(this.quadScene, this.quadCamera);
  }

  dispose(): void {
    this.points.geometry.dispose();
    this.splatMaterial.dispose();
    this.blurMaterial.dispose();
    this.resolveMaterial.dispose();
    this.quad.geometry.dispose();
    this.accTarget.dispose();
    this.haloTargetA.dispose();
    this.haloTargetB.dispose();
    // Blank the canvas to the neutral blend colour (black under screen, white
    // under multiply) so no stale frame lingers on the still-mounted overlay.
    this.renderer.setRenderTarget(null);
    this.renderer.setClearColor(this.isDark ? 0x000000 : 0xffffff, 1);
    this.renderer.clear(true, false, false);
    this.renderer.dispose();
    this.buffers = null;
    this.lastClusterMap = null;
  }
}
