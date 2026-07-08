/**
 * WebGL2 renderer for the 2D density overlay — a class-shaped port of
 * embedding-atlas's densityRenderCommand (webgl2_renderer/renderer.ts,
 * Apple MIT), replacing its Dataflow memoization harness with plain members.
 *
 * Pipeline per frame:
 *   splat points (1px GL_POINTS, additive, category → RGBA channel)
 *   → separable gaussian blur (σ≈20 device px, 8 passes, ping-pong)
 *   → quantized density-band paint + per-channel Sobel contours (linear RGB)
 *   → linear→sRGB gamma pass onto the canvas, cropping the blur safe margin.
 *
 * The canvas overlays the Plotly plot area (pointer-events: none), so the
 * intermediate buffer is cleared to transparent rather than Apple's opaque
 * white/black. See README.md.
 */

import {
  createProgram,
  createStaticBuffer,
  createFramebuffer,
  disposeFramebuffer,
  type GlProgram,
  type GlFramebuffer,
} from './glResources';
import {
  QUAD_VERTEX,
  FILL_COUNT_VERTEX_CATEGORY,
  FILL_COUNT_VERTEX_COLOR,
  FILL_COUNT_VERTEX_PLAIN,
  FILL_COUNT_FRAGMENT,
  BLUR_FRAGMENT,
  BLUR_FILTERS_R20,
  PAINT_DENSITY_MAP_FRAGMENT,
  PAINT_DENSITY_MEAN_FRAGMENT,
  PAINT_CONTOURS_FRAGMENT,
  PAINT_CONTOURS_MEAN_FRAGMENT,
  GAMMA_VERTEX,
  GAMMA_FRAGMENT,
  gaussianBlurR20PixelRadius,
} from './shaders';
import { matrix3Multiply, type Matrix3 } from './densityMath';

const DENSITY_BANDWIDTH = 20; // device px; the R20 blur is hardcoded to this
const SAFE_MARGIN = gaussianBlurR20PixelRadius(DENSITY_BANDWIDTH) + 1;

export interface DensityRenderProps {
  /** Data → clip matrix for the visible plot area (buildPositionMatrix). */
  positionMatrix: Matrix3;
  /** 16 floats from buildColorMatrix. */
  colorMatrix: number[];
  densityScaler: number;
  densityAlpha: number;
  contoursAlpha: number;
  quantizationStep?: number;
  gamma?: number;
  isDark: boolean;
}

const REQUIRED_EXTENSIONS = ['EXT_color_buffer_float', 'EXT_float_blend', 'OES_texture_float_linear'];

let supportCache: boolean | null = null;

export class DensityRenderer {
  private gl: WebGL2RenderingContext;
  private canvas: HTMLCanvasElement;

  private quadBuffer: WebGLBuffer;
  private fillCategoryProgram: GlProgram;
  private fillColorProgram: GlProgram;
  private fillPlainProgram: GlProgram;
  private blurProgram: GlProgram;
  private densityProgram: GlProgram;
  private densityMeanProgram: GlProgram;
  private contoursProgram: GlProgram;
  private contoursMeanProgram: GlProgram;
  private gammaProgram: GlProgram;

  private xBuffer: WebGLBuffer | null = null;
  private yBuffer: WebGLBuffer | null = null;
  private categoryBuffer: WebGLBuffer | null = null;
  private colorBuffer: WebGLBuffer | null = null;
  private pointCount = 0;
  private categoryCount = 1;

  private countFB: GlFramebuffer | null = null;
  private tempFB1: GlFramebuffer | null = null;
  private tempFB2: GlFramebuffer | null = null;
  private linearFB: GlFramebuffer | null = null;
  private deviceWidth = 0;
  private deviceHeight = 0;

  private disposed = false;

  /** True when the environment can run the float-framebuffer pipeline. */
  static isSupported(): boolean {
    if (supportCache !== null) return supportCache;
    try {
      const probe = document.createElement('canvas');
      const gl = probe.getContext('webgl2');
      supportCache = gl != null && REQUIRED_EXTENSIONS.every((ext) => gl.getExtension(ext) != null);
    } catch {
      supportCache = false;
    }
    return supportCache;
  }

  constructor(canvas: HTMLCanvasElement) {
    // preserveDrawingBuffer: the screenshot export copies this canvas outside
    // the render task; without it the buffer is cleared after compositing.
    const gl = canvas.getContext('webgl2', {
      antialias: false,
      premultipliedAlpha: true,
      preserveDrawingBuffer: true,
    });
    if (!gl) throw new Error('WebGL2 not available');
    for (const ext of REQUIRED_EXTENSIONS) {
      if (!gl.getExtension(ext)) throw new Error(`Missing WebGL extension: ${ext}`);
    }
    this.canvas = canvas;
    this.gl = gl;

    this.quadBuffer = createStaticBuffer(gl, new Float32Array([-1, -1, -1, 1, 1, -1, 1, 1]));
    this.fillCategoryProgram = createProgram(gl, FILL_COUNT_VERTEX_CATEGORY, FILL_COUNT_FRAGMENT);
    this.fillColorProgram = createProgram(gl, FILL_COUNT_VERTEX_COLOR, FILL_COUNT_FRAGMENT);
    this.fillPlainProgram = createProgram(gl, FILL_COUNT_VERTEX_PLAIN, FILL_COUNT_FRAGMENT);
    this.blurProgram = createProgram(gl, QUAD_VERTEX, BLUR_FRAGMENT);
    this.densityProgram = createProgram(gl, QUAD_VERTEX, PAINT_DENSITY_MAP_FRAGMENT);
    this.densityMeanProgram = createProgram(gl, QUAD_VERTEX, PAINT_DENSITY_MEAN_FRAGMENT);
    this.contoursProgram = createProgram(gl, QUAD_VERTEX, PAINT_CONTOURS_FRAGMENT);
    this.contoursMeanProgram = createProgram(gl, QUAD_VERTEX, PAINT_CONTOURS_MEAN_FRAGMENT);
    this.gammaProgram = createProgram(gl, GAMMA_VERTEX, GAMMA_FRAGMENT);
  }

  /**
   * Upload point data. Three mutually exclusive modes:
   * - `category` set: one-hot RGBA channels (≤4 categories), `categoryCount`
   *   bounds the per-channel contour passes;
   * - `colors` set: mean-color mode — 3 bytes of linearized RGB per point,
   *   density in the alpha channel (any number of categories);
   * - neither: single-channel splat (muted tint via colorMatrix[0]).
   */
  setData(data: {
    x: Float32Array;
    y: Float32Array;
    category?: Uint8Array | null;
    colors?: Uint8Array | null;
    categoryCount?: number;
  }): void {
    const gl = this.gl;
    this.deleteDataBuffers();
    this.pointCount = data.x.length;
    this.categoryCount = data.category ? Math.max(1, Math.min(4, data.categoryCount ?? 1)) : 1;
    if (this.pointCount === 0) return;
    this.xBuffer = createStaticBuffer(gl, data.x);
    this.yBuffer = createStaticBuffer(gl, data.y);
    // The category attribute is a signed-byte int in the shader.
    this.categoryBuffer = data.category ? createStaticBuffer(gl, new Int8Array(data.category)) : null;
    this.colorBuffer = !data.category && data.colors ? createStaticBuffer(gl, data.colors) : null;
  }

  /** Resize the canvas backing store and framebuffers. No-op if unchanged. */
  setSize(cssWidth: number, cssHeight: number, dpr: number): void {
    const w = Math.max(1, Math.round(cssWidth * dpr));
    const h = Math.max(1, Math.round(cssHeight * dpr));
    if (w === this.deviceWidth && h === this.deviceHeight) return;
    this.deviceWidth = w;
    this.deviceHeight = h;
    this.canvas.width = w;
    this.canvas.height = h;

    const gl = this.gl;
    this.deleteFramebuffers();
    const fbW = w + SAFE_MARGIN * 2;
    const fbH = h + SAFE_MARGIN * 2;
    this.countFB = createFramebuffer(gl, fbW, fbH, 4);
    this.tempFB1 = createFramebuffer(gl, fbW, fbH, 4);
    this.tempFB2 = createFramebuffer(gl, fbW, fbH, 4);
    this.linearFB = createFramebuffer(gl, fbW, fbH, 4);
  }

  render(props: DensityRenderProps): void {
    const gl = this.gl;
    const { countFB, tempFB1, tempFB2, linearFB } = this;
    if (this.disposed || !countFB || !tempFB1 || !tempFB2 || !linearFB) return;

    const showDensity = props.densityScaler > 0 && props.densityAlpha > 0;
    const showContours = props.densityScaler > 0 && props.contoursAlpha > 0;

    if (this.pointCount === 0 || (!showDensity && !showContours)) {
      // Nothing visible: just clear the canvas so no stale frame lingers.
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.viewport(0, 0, this.deviceWidth, this.deviceHeight);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      return;
    }

    // Safe-margin overscan: the FBs are larger than the canvas so the blur has
    // real data past the visible edge; shrink data toward the FB center now and
    // crop back in the gamma pass.
    const scalerX = this.deviceWidth / countFB.width;
    const scalerY = this.deviceHeight / countFB.height;
    const matrix = matrix3Multiply([scalerX, 0, 0, 0, scalerY, 0, 0, 0, 1], props.positionMatrix);

    // 1. Splat points into the count buffer (additive, 1px each).
    gl.bindFramebuffer(gl.FRAMEBUFFER, countFB.framebuffer);
    gl.viewport(0, 0, countFB.width, countFB.height);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    this.fillCountBuffer(matrix);

    // 2. Blur into tempFB1 (ping-pong via tempFB2).
    this.gaussianBlurR20(countFB.texture, tempFB1, tempFB2);

    // 3. Paint density bands + contours into the transparent linear buffer.
    gl.bindFramebuffer(gl.FRAMEBUFFER, linearFB.framebuffer);
    gl.viewport(0, 0, linearFB.width, linearFB.height);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    const quantizationStep = props.quantizationStep ?? 0.1;
    const meanColorMode = this.colorBuffer != null;
    if (showDensity) {
      if (meanColorMode) {
        this.paintDensityMean(tempFB1, props, quantizationStep);
      } else {
        this.paintDensityMap(tempFB1, props, quantizationStep);
      }
    }
    if (showContours) {
      if (meanColorMode) {
        this.paintContoursMean(tempFB1, props, quantizationStep);
      } else {
        for (let i = 0; i < this.categoryCount; i++) {
          this.paintContours(tempFB1, props, quantizationStep, i);
        }
      }
    }

    // 4. Linear→sRGB onto the canvas, cropping the safe margin.
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, this.deviceWidth, this.deviceHeight);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    this.gammaCorrection(linearFB.texture, props.gamma ?? 2.2, 1 / scalerX, 1 / scalerY);
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    const gl = this.gl;
    this.deleteDataBuffers();
    this.deleteFramebuffers();
    gl.deleteBuffer(this.quadBuffer);
    for (const p of [
      this.fillCategoryProgram, this.fillColorProgram, this.fillPlainProgram,
      this.blurProgram, this.densityProgram, this.densityMeanProgram,
      this.contoursProgram, this.contoursMeanProgram, this.gammaProgram,
    ]) {
      gl.deleteProgram(p.program);
    }
    // Clear the canvas so no stale frame lingers on the still-mounted element.
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, this.deviceWidth, this.deviceHeight);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
  }

  // -------------------------------------------------------------------------
  // Passes
  // -------------------------------------------------------------------------

  private fillCountBuffer(matrix: Matrix3): void {
    const gl = this.gl;
    const hasCategory = this.categoryBuffer != null;
    const hasColors = this.colorBuffer != null;
    const program = hasCategory
      ? this.fillCategoryProgram
      : hasColors
        ? this.fillColorProgram
        : this.fillPlainProgram;

    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE);
    gl.useProgram(program.program);

    gl.enableVertexAttribArray(0);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.xBuffer);
    gl.vertexAttribPointer(0, 1, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(1);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.yBuffer);
    gl.vertexAttribPointer(1, 1, gl.FLOAT, false, 0, 0);
    if (hasCategory) {
      gl.enableVertexAttribArray(2);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.categoryBuffer);
      gl.vertexAttribIPointer(2, 1, gl.BYTE, 0, 0);
    } else if (hasColors) {
      gl.enableVertexAttribArray(2);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.colorBuffer);
      gl.vertexAttribPointer(2, 3, gl.UNSIGNED_BYTE, true, 0, 0);
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, null);

    gl.uniformMatrix3fv(program.uniforms.matrix, false, matrix);
    gl.drawArrays(gl.POINTS, 0, this.pointCount);

    gl.disableVertexAttribArray(0);
    gl.disableVertexAttribArray(1);
    if (hasCategory || hasColors) gl.disableVertexAttribArray(2);
    gl.useProgram(null);
  }

  private bindQuad(): void {
    const gl = this.gl;
    gl.enableVertexAttribArray(0);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuffer);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
  }

  private unbindQuad(): void {
    const gl = this.gl;
    gl.useProgram(null);
    gl.bindTexture(gl.TEXTURE_2D, null);
    gl.disableVertexAttribArray(0);
  }

  /**
   * σ≈20 separable blur: 4 pre-baked filters × 2 directions, ping-ponging so
   * the even number of swaps lands the result in `destination`.
   */
  private gaussianBlurR20(texture: WebGLTexture, destination: GlFramebuffer, tmp: GlFramebuffer): void {
    const gl = this.gl;
    const program = this.blurProgram;
    gl.disable(gl.BLEND);
    this.bindQuad();

    gl.useProgram(program.program);
    gl.uniform2f(program.uniforms.resolution, destination.width, destination.height);
    gl.uniform1i(program.uniforms.image, 0);

    let src = texture;
    let target = tmp;
    let other = destination;
    for (let dir = 0; dir < 2; dir++) {
      gl.uniform2f(program.uniforms.direction, dir, 1 - dir);
      for (const [distances, weight0, weights] of BLUR_FILTERS_R20) {
        gl.bindFramebuffer(gl.FRAMEBUFFER, target.framebuffer);
        gl.bindTexture(gl.TEXTURE_2D, src);
        gl.uniform1fv(program.uniforms.weight0, weight0);
        gl.uniform3fv(program.uniforms.distances, distances);
        gl.uniform3fv(program.uniforms.weights, weights);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
        src = target.texture;
        const t = target;
        target = other;
        other = t;
      }
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    this.unbindQuad();
  }

  private paintDensityMap(input: GlFramebuffer, props: DensityRenderProps, quantizationStep: number): void {
    const gl = this.gl;
    const program = this.densityProgram;
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    this.bindQuad();
    gl.bindTexture(gl.TEXTURE_2D, input.texture);

    gl.useProgram(program.program);
    gl.uniform1i(program.uniforms.source, 0);
    gl.uniform2f(program.uniforms.resolution, input.width, input.height);
    gl.uniform1f(program.uniforms.densityScaler, props.densityScaler);
    gl.uniform1f(program.uniforms.quantizationStep, quantizationStep);
    gl.uniform1f(program.uniforms.globalAlpha, props.densityAlpha);
    gl.uniform1i(program.uniforms.isDarkMode, props.isDark ? 1 : 0);
    gl.uniformMatrix4fv(program.uniforms.colorMatrix, false, props.colorMatrix);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    this.unbindQuad();
  }

  private paintDensityMean(input: GlFramebuffer, props: DensityRenderProps, quantizationStep: number): void {
    const gl = this.gl;
    const program = this.densityMeanProgram;
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    this.bindQuad();
    gl.bindTexture(gl.TEXTURE_2D, input.texture);

    gl.useProgram(program.program);
    gl.uniform1i(program.uniforms.source, 0);
    gl.uniform2f(program.uniforms.resolution, input.width, input.height);
    gl.uniform1f(program.uniforms.densityScaler, props.densityScaler);
    gl.uniform1f(program.uniforms.quantizationStep, quantizationStep);
    gl.uniform1f(program.uniforms.globalAlpha, props.densityAlpha);
    gl.uniform1i(program.uniforms.isDarkMode, props.isDark ? 1 : 0);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    this.unbindQuad();
  }

  private paintContoursMean(input: GlFramebuffer, props: DensityRenderProps, quantizationStep: number): void {
    const gl = this.gl;
    const program = this.contoursMeanProgram;
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    this.bindQuad();
    gl.bindTexture(gl.TEXTURE_2D, input.texture);

    gl.useProgram(program.program);
    gl.uniform1i(program.uniforms.source, 0);
    gl.uniform2f(program.uniforms.resolution, input.width, input.height);
    gl.uniform1f(program.uniforms.densityScaler, props.densityScaler);
    gl.uniform1f(program.uniforms.quantizationStep, quantizationStep);
    gl.uniform1f(program.uniforms.globalAlpha, props.contoursAlpha);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    this.unbindQuad();
  }

  private paintContours(
    input: GlFramebuffer,
    props: DensityRenderProps,
    quantizationStep: number,
    channel: number,
  ): void {
    const gl = this.gl;
    const program = this.contoursProgram;
    const channelMask = [0, 0, 0, 0];
    channelMask[channel] = 1;

    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    this.bindQuad();
    gl.bindTexture(gl.TEXTURE_2D, input.texture);

    gl.useProgram(program.program);
    gl.uniform1i(program.uniforms.source, 0);
    gl.uniform2f(program.uniforms.resolution, input.width, input.height);
    gl.uniform1f(program.uniforms.densityScaler, props.densityScaler);
    gl.uniform1f(program.uniforms.quantizationStep, quantizationStep);
    gl.uniform1f(program.uniforms.globalAlpha, props.contoursAlpha);
    gl.uniform4fv(program.uniforms.channelMask, channelMask);
    gl.uniform4fv(program.uniforms.color, props.colorMatrix.slice(channel * 4, channel * 4 + 4));
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    this.unbindQuad();
  }

  private gammaCorrection(input: WebGLTexture, gamma: number, xScaler: number, yScaler: number): void {
    const gl = this.gl;
    const program = this.gammaProgram;
    // The linear buffer holds premultiplied colors over transparency; blend it
    // onto the (cleared) canvas instead of overwriting with opaque pixels.
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    this.bindQuad();
    gl.bindTexture(gl.TEXTURE_2D, input);

    gl.useProgram(program.program);
    gl.uniform1i(program.uniforms.source, 0);
    gl.uniform2f(program.uniforms.xyScaler, xScaler, yScaler);
    gl.uniform1f(program.uniforms.gamma, gamma);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    this.unbindQuad();
  }

  // -------------------------------------------------------------------------
  // Resource cleanup
  // -------------------------------------------------------------------------

  private deleteDataBuffers(): void {
    const gl = this.gl;
    if (this.xBuffer) gl.deleteBuffer(this.xBuffer);
    if (this.yBuffer) gl.deleteBuffer(this.yBuffer);
    if (this.categoryBuffer) gl.deleteBuffer(this.categoryBuffer);
    if (this.colorBuffer) gl.deleteBuffer(this.colorBuffer);
    this.xBuffer = null;
    this.yBuffer = null;
    this.categoryBuffer = null;
    this.colorBuffer = null;
    this.pointCount = 0;
  }

  private deleteFramebuffers(): void {
    const gl = this.gl;
    for (const fb of [this.countFB, this.tempFB1, this.tempFB2, this.linearFB]) {
      if (fb) disposeFramebuffer(gl, fb);
    }
    this.countFB = null;
    this.tempFB1 = null;
    this.tempFB2 = null;
    this.linearFB = null;
  }
}
