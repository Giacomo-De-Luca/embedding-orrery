/**
 * Test harness for the prebuilt stackgl bundle (`forked/plotly.js/stackgl_modules/index.js`).
 *
 * The bundle expects a browser: it reads `window` at load time and vectorize-text
 * grabs a 2D canvas from `document` when its module initializes. `StackglHarness.load()`
 * installs minimal stand-ins for both before the first require, so gl-scatter3d and
 * gl-plot3d can run in vitest's node environment against a `RecordingGl` context.
 * gl-plot3d's `createScene` also needs `window.addEventListener`, a canvas that accepts
 * event listeners and `requestAnimationFrame`; for the last one the harness installs a
 * `FrameScheduler`, so tests step animation frames by hand.
 *
 * Every fork test must load the bundle through this module: the bundle captures the
 * globals at require time, and Node's require cache is shared by every test file that
 * runs in the same worker when vitest's `isolate` option is off.
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

type FakeEventListener = (event: Record<string, unknown>) => void;

/**
 * Canvas element stand-in. Besides the 2D context for glyph rasterization it keeps an
 * event-listener registry, so a test can `dispatch` the mouse events gl-plot3d listens
 * for (mouse-change reads `offsetX`, `offsetY` and `buttons` off the event object).
 * Scenes are created with `autoResize: false` and a fake camera, so no layout
 * properties are needed.
 */
export class FakeCanvasElement {
  width = 300;
  height = 150;
  private readonly context2d = new FakeCanvas2DContext();
  private readonly listeners = new Map<string, Set<FakeEventListener>>();

  getContext(kind: string): FakeCanvas2DContext | null {
    return kind === '2d' ? this.context2d : null;
  }

  addEventListener(type: string, listener: FakeEventListener): void {
    let set = this.listeners.get(type);
    if (!set) {
      set = new Set();
      this.listeners.set(type, set);
    }
    set.add(listener);
  }

  removeEventListener(type: string, listener: FakeEventListener): void {
    this.listeners.get(type)?.delete(listener);
  }

  /** Delivers `event` (with `type` and `target` filled in) to the listeners of `type`. */
  dispatch(type: string, event: Record<string, unknown> = {}): void {
    for (const listener of [...(this.listeners.get(type) ?? [])]) {
      listener({ type, target: this, ...event });
    }
  }
}

// ---------------------------------------------------------------------------
// Recording WebGL context
// ---------------------------------------------------------------------------

/** Real WebGL enum values for every constant a test asserts on or the bundle branches on. */
export const GL_ENUMS: Record<string, number> = {
  POINTS: 0, LINES: 1, LINE_LOOP: 2, LINE_STRIP: 3, TRIANGLES: 4, TRIANGLE_STRIP: 5, TRIANGLE_FAN: 6,
  DEPTH_BUFFER_BIT: 256, STENCIL_BUFFER_BIT: 1024, COLOR_BUFFER_BIT: 16384,
  BYTE: 5120, UNSIGNED_BYTE: 5121, SHORT: 5122, UNSIGNED_SHORT: 5123, INT: 5124, UNSIGNED_INT: 5125, FLOAT: 5126,
  ARRAY_BUFFER: 34962, ELEMENT_ARRAY_BUFFER: 34963,
  STREAM_DRAW: 35040, STATIC_DRAW: 35044, DYNAMIC_DRAW: 35048,
  FRAGMENT_SHADER: 35632, VERTEX_SHADER: 35633,
  COMPILE_STATUS: 35713, LINK_STATUS: 35714, VALIDATE_STATUS: 35715,
  ACTIVE_UNIFORMS: 35718, ACTIVE_ATTRIBUTES: 35721, MAX_VERTEX_ATTRIBS: 34921,
  TEXTURE_2D: 3553, RGBA: 6408, RGB: 6407, ALPHA: 6406, LUMINANCE: 6409, LUMINANCE_ALPHA: 6410,
  DEPTH_COMPONENT: 6402, DEPTH_COMPONENT16: 33189, DEPTH_STENCIL: 34041, STENCIL_INDEX8: 36168,
  // gl-texture2d validates filter and wrap modes against these, so they must be real.
  NEAREST: 9728, LINEAR: 9729, NEAREST_MIPMAP_NEAREST: 9984, LINEAR_MIPMAP_NEAREST: 9985,
  NEAREST_MIPMAP_LINEAR: 9986, LINEAR_MIPMAP_LINEAR: 9987,
  TEXTURE_MAG_FILTER: 10240, TEXTURE_MIN_FILTER: 10241, TEXTURE_WRAP_S: 10242, TEXTURE_WRAP_T: 10243,
  REPEAT: 10497, CLAMP_TO_EDGE: 33071, MIRRORED_REPEAT: 33648, TEXTURE0: 33984,
  FRAMEBUFFER: 36160, RENDERBUFFER: 36161, FRAMEBUFFER_COMPLETE: 36053,
  COLOR_ATTACHMENT0: 36064, DEPTH_ATTACHMENT: 36096, STENCIL_ATTACHMENT: 36128, DEPTH_STENCIL_ATTACHMENT: 33306,
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
 * the buffer bound to the target at the time. `readPixels` fills its target with
 * `readPixelsFill` when one is set, so a pick query can be made to hit or miss.
 */
export class RecordingGl {
  readonly calls: GlCall[] = [];
  readonly drawCalls: DrawCall[] = [];
  readonly lineWidths: number[] = [];
  /** Latest uploaded bytes per buffer handle. */
  readonly bufferBytes = new Map<object, Uint8Array>();
  /** RGBA bytes `readPixels` writes into every pixel of the requested region; null leaves the target untouched. */
  readPixelsFill: number[] | null = null;
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
    // `then` must stay undefined, or the context would look like a promise to `await`.
    if (typeof prop !== 'string' || prop === 'then') return undefined;
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
      case 'readPixels':
        this.fillReadback(args);
        return undefined;
      default:
        return undefined;
    }
  }

  /** `readPixels(x, y, width, height, format, type, target)`: fills `target` with `readPixelsFill`. */
  private fillReadback(args: unknown[]): void {
    const target = args[6];
    if (!this.readPixelsFill || !(target instanceof Uint8Array)) return;
    const bytes = (args[2] as number) * (args[3] as number) * 4;
    for (let i = 0; i < bytes; i++) target[i] = this.readPixelsFill[i & 3];
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
  gl_plot3d: { createScene: (options: Record<string, unknown>) => GlScene };
}

/** The gl-plot3d selection record its mouse handler fills in. */
export interface GlSelection {
  object: unknown;
  buttons: number;
  mouse: number[];
  distance: number;
  /** What the picked object's `pick()` returned; also passed to its `highlight()`. */
  data: unknown;
}

/** The gl-plot3d scene surface the tests read. */
export interface GlScene {
  selection: GlSelection;
  pickStats: { buffers: number; renders: number; deferredFrames: number; readbacks: number };
  pixelRatio: number;
  onselect: ((selection: GlSelection) => void) | null;
  add(object: unknown): void;
  update(options?: Record<string, unknown>): void;
  redraw(): void;
  dispose(): void;
}

/**
 * `requestAnimationFrame` stand-in. Callbacks queue until a test calls `step()`, which
 * runs one frame: everything queued so far, in order. A callback that re-queues itself
 * (gl-plot3d's render loop) therefore runs exactly once per step.
 */
export class FrameScheduler {
  private queue: Array<{ id: number; callback: (time: number) => void }> = [];
  private nextId = 1;
  private time = 0;

  readonly request = (callback: (time: number) => void): number => {
    const id = this.nextId++;
    this.queue.push({ id, callback });
    return id;
  };

  readonly cancel = (id: number): void => {
    this.queue = this.queue.filter((frame) => frame.id !== id);
  };

  /** Runs one animation frame. */
  step(): void {
    const batch = this.queue;
    this.queue = [];
    this.time += 16;
    for (const frame of batch) frame.callback(this.time);
  }

  /** Drops queued callbacks, e.g. from a scene a previous test left running. */
  reset(): void {
    this.queue = [];
  }
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
  /** The `requestAnimationFrame` queue; tests step it by hand. */
  static readonly frames = new FrameScheduler();

  /** Installs the browser stand-ins (once) and returns the bundle's exports. */
  static load(): StackglModules {
    if (StackglHarness.modules) return StackglHarness.modules;
    const g = globalThis as Record<string, unknown>;
    if (g.window === undefined) g.window = globalThis;
    const w = g.window as Record<string, unknown>;
    if (typeof w.addEventListener !== 'function') {
      // gl-plot3d and mouse-change register resize, blur and key listeners on window.
      w.addEventListener = () => {};
      w.removeEventListener = () => {};
    }
    if (g.requestAnimationFrame === undefined) {
      g.requestAnimationFrame = StackglHarness.frames.request;
      g.cancelAnimationFrame = StackglHarness.frames.cancel;
    }
    if (g.document === undefined) {
      g.document = { createElement: () => new FakeCanvasElement() };
    } else {
      // A DOM environment (e.g. jsdom) has a document whose canvases have no 2D context.
      const probe = (g.document as Document).createElement('canvas');
      if (!probe.getContext('2d')) {
        throw new Error('StackglHarness needs a canvas 2D context; run fork tests in the node environment.');
      }
    }
    const require = createRequire(import.meta.url);
    StackglHarness.modules = require('../plotly.js/stackgl_modules/index.js') as StackglModules;
    return StackglHarness.modules;
  }
}
