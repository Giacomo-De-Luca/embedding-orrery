/**
 * Test harness for the prebuilt stackgl bundle (`forked/plotly.js/stackgl_modules/index.js`).
 *
 * The bundle expects a browser: it reads `window` at load time and vectorize-text
 * grabs a 2D canvas from `document` when its module initializes. `StackglHarness.load()`
 * installs minimal stand-ins for both before the first require, so gl-scatter3d and
 * gl-plot3d can run in vitest's node environment against a `RecordingGl` context.
 *
 * Every fork test must load the bundle through this module: Node's require cache is
 * shared per worker, so the globals have to be in place before whichever test
 * requires the bundle first.
 */
import { createRequire } from 'node:module';

// ---------------------------------------------------------------------------
// Fake 2D canvas (glyph rasterization)
// ---------------------------------------------------------------------------

/**
 * 2D context stand-in. vectorize-text paints the glyph white-on-black, then reads the
 * pixels back; `getImageData` returns a filled disc so every glyph traces to a circle,
 * which is enough for gl-scatter3d to build a triangle mesh and an outline.
 */
class FakeCanvas2DContext {
  font = '64px normal';
  fillStyle = '#000';
  textAlign = 'start';
  textBaseline = 'alphabetic';

  fillRect(): void {}
  fillText(): void {}
  clearRect(): void {}

  measureText(text: string): { width: number } {
    return { width: 64 * Math.max(1, text.length) };
  }

  getImageData(_x: number, _y: number, width: number, height: number): { data: Uint8ClampedArray; width: number; height: number } {
    const data = new Uint8ClampedArray(width * height * 4);
    const cx = width / 2;
    const cy = height / 2;
    const r = Math.min(width, height) / 4;
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        if ((x - cx) ** 2 + (y - cy) ** 2 <= r * r) {
          const i = 4 * (y * width + x);
          data[i] = data[i + 1] = data[i + 2] = data[i + 3] = 255;
        }
      }
    }
    return { data, width, height };
  }
}

class FakeCanvasElement {
  width = 300;
  height = 150;
  private readonly context2d = new FakeCanvas2DContext();

  getContext(kind: string): FakeCanvas2DContext | null {
    return kind === '2d' ? this.context2d : null;
  }
}

// ---------------------------------------------------------------------------
// Recording WebGL context
// ---------------------------------------------------------------------------

/** Real WebGL enum values for every constant a test asserts on or the bundle branches on. */
const GL_ENUMS: Record<string, number> = {
  POINTS: 0, LINES: 1, LINE_LOOP: 2, LINE_STRIP: 3, TRIANGLES: 4, TRIANGLE_STRIP: 5, TRIANGLE_FAN: 6,
  DEPTH_BUFFER_BIT: 256, STENCIL_BUFFER_BIT: 1024, COLOR_BUFFER_BIT: 16384,
  BYTE: 5120, UNSIGNED_BYTE: 5121, SHORT: 5122, UNSIGNED_SHORT: 5123, INT: 5124, UNSIGNED_INT: 5125, FLOAT: 5126,
  ARRAY_BUFFER: 34962, ELEMENT_ARRAY_BUFFER: 34963,
  STREAM_DRAW: 35040, STATIC_DRAW: 35044, DYNAMIC_DRAW: 35048,
  FRAGMENT_SHADER: 35632, VERTEX_SHADER: 35633,
  COMPILE_STATUS: 35713, LINK_STATUS: 35714, VALIDATE_STATUS: 35715,
  ACTIVE_UNIFORMS: 35718, ACTIVE_ATTRIBUTES: 35721, MAX_VERTEX_ATTRIBS: 34921,
  TEXTURE_2D: 3553, RGBA: 6408, DEPTH_COMPONENT: 6402, DEPTH_COMPONENT16: 33189,
  FRAMEBUFFER: 36160, RENDERBUFFER: 36161, FRAMEBUFFER_COMPLETE: 36053,
  COLOR_ATTACHMENT0: 36064, DEPTH_ATTACHMENT: 36096,
  MAX_TEXTURE_SIZE: 3379, MAX_RENDERBUFFER_SIZE: 34024,
  BLEND: 3042, DEPTH_TEST: 2929, CULL_FACE: 2884, SCISSOR_TEST: 3089,
};

/** Default return values for GL query methods; anything not listed is a recorded no-op. */
const QUERY_PARAMETERS: Record<number, number> = {
  [GL_ENUMS.MAX_VERTEX_ATTRIBS]: 16,
  [GL_ENUMS.MAX_TEXTURE_SIZE]: 16384,
  [GL_ENUMS.MAX_RENDERBUFFER_SIZE]: 16384,
};

export interface GlCall {
  name: string;
  args: unknown[];
}

export interface DrawCall {
  mode: number;
  first: number;
  count: number;
}

/**
 * A WebGL context stand-in that records what the bundle does with it.
 *
 * Shader compilation and linking always succeed and report no active uniforms or
 * attributes. Buffers remember the bytes of their last `bufferData` upload, keyed by
 * the buffer bound to the target at the time.
 */
export class RecordingGl {
  readonly calls: GlCall[] = [];
  readonly drawCalls: DrawCall[] = [];
  readonly lineWidths: number[] = [];
  /** Latest uploaded bytes per buffer handle. */
  readonly bufferBytes = new Map<object, Uint8Array>();
  readonly gl: WebGLRenderingContext;

  private readonly boundBuffers = new Map<number, object | null>();
  private readonly extraEnums = new Map<string, number>();
  private nextHandleId = 1;

  constructor(
    readonly drawingBufferWidth = 1600,
    readonly drawingBufferHeight = 1200,
  ) {
    this.gl = new Proxy({}, {
      get: (_target, prop) => this.property(prop),
    }) as WebGLRenderingContext;
  }

  /** Clears recorded calls; uploaded buffer contents are kept. */
  resetCalls(): void {
    this.calls.length = 0;
    this.drawCalls.length = 0;
    this.lineWidths.length = 0;
  }

  private property(prop: string | symbol): unknown {
    if (typeof prop !== 'string') return undefined;
    if (prop === 'drawingBufferWidth') return this.drawingBufferWidth;
    if (prop === 'drawingBufferHeight') return this.drawingBufferHeight;
    if (prop === 'canvas') return null;
    if (/^[A-Z][A-Z0-9_]*$/.test(prop)) return this.enumValue(prop);
    return (...args: unknown[]) => this.call(prop, args);
  }

  private enumValue(name: string): number {
    if (name in GL_ENUMS) return GL_ENUMS[name];
    let value = this.extraEnums.get(name);
    if (value === undefined) {
      value = 0x10000 + this.extraEnums.size;
      this.extraEnums.set(name, value);
    }
    return value;
  }

  private call(name: string, args: unknown[]): unknown {
    this.calls.push({ name, args });
    if (name.startsWith('create')) return { handle: name, id: this.nextHandleId++ };
    switch (name) {
      case 'getShaderParameter':
        return true;
      case 'getProgramParameter':
        return args[1] === GL_ENUMS.LINK_STATUS || args[1] === GL_ENUMS.VALIDATE_STATUS ? true : 0;
      case 'getParameter':
        return QUERY_PARAMETERS[args[0] as number] ?? 0;
      case 'getExtension':
        return null;
      case 'getSupportedExtensions':
        return [];
      case 'checkFramebufferStatus':
        return GL_ENUMS.FRAMEBUFFER_COMPLETE;
      case 'isContextLost':
        return false;
      case 'getShaderInfoLog':
      case 'getProgramInfoLog':
        return '';
      case 'getAttribLocation':
        return 0;
      case 'getUniformLocation':
        return {};
      case 'getContextAttributes':
        return {};
      case 'bindBuffer':
        this.boundBuffers.set(args[0] as number, (args[1] as object | null) ?? null);
        return undefined;
      case 'bufferData':
        this.recordUpload(args[0] as number, args[1]);
        return undefined;
      case 'drawArrays':
        this.drawCalls.push({ mode: args[0] as number, first: args[1] as number, count: args[2] as number });
        return undefined;
      case 'lineWidth':
        this.lineWidths.push(args[0] as number);
        return undefined;
      default:
        return undefined;
    }
  }

  private recordUpload(target: number, data: unknown): void {
    const buffer = this.boundBuffers.get(target);
    if (!buffer) return;
    if (ArrayBuffer.isView(data)) {
      this.bufferBytes.set(buffer, new Uint8Array(data.buffer, data.byteOffset, data.byteLength).slice());
    } else if (typeof data === 'number') {
      this.bufferBytes.set(buffer, new Uint8Array(data));
    }
  }
}

// ---------------------------------------------------------------------------
// Bundle loading
// ---------------------------------------------------------------------------

/** The bundle's exports that the fork tests use (its full export map is wider). */
export interface StackglModules {
  gl_scatter3d: (options: Record<string, unknown>) => PointCloud;
  gl_plot3d: { createScene: (options: Record<string, unknown>) => unknown };
}

/** The gl-scatter3d object surface the tests read. */
export interface PointCloud {
  vertexCount: number;
  lineVertexCount: number;
  lineWidth: number;
  hasAlpha: boolean;
  pickId: number;
  pixelRatio: number;
  pointBuffer: { handle: object };
  colorBuffer: { handle: object };
  glyphBuffer: { handle: object };
  idBuffer: { handle: object };
  axes: { bounds: number[][] };
  update(options: Record<string, unknown>): void;
  draw(camera: Record<string, unknown>): void;
  drawTransparent(camera: Record<string, unknown>): void;
  drawPick(camera: Record<string, unknown>): void;
  isTransparent(): boolean;
  pick(selected: { id: number; value: number[] }): { index: number } | null;
}

export class StackglHarness {
  private static modules: StackglModules | null = null;

  /** Installs the browser stand-ins (once) and returns the bundle's exports. */
  static load(): StackglModules {
    if (StackglHarness.modules) return StackglHarness.modules;
    const g = globalThis as Record<string, unknown>;
    if (g.window === undefined) g.window = globalThis;
    if (g.document === undefined) {
      g.document = { createElement: () => new FakeCanvasElement() };
    }
    const require = createRequire(import.meta.url);
    StackglHarness.modules = require('../plotly.js/stackgl_modules/index.js') as StackglModules;
    return StackglHarness.modules;
  }
}
