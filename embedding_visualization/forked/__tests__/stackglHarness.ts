/**
 * Test harness for the prebuilt stackgl bundle (`forked/plotly.js/stackgl_modules/index.js`).
 *
 * The bundle expects a browser: it reads `window` at load time and vectorize-text
 * grabs a 2D canvas from `document` when its module initializes. `StackglHarness.load()`
 * installs minimal stand-ins for both before the first require, so gl-scatter3d can
 * run in vitest's node environment against a `RecordingGl` context. gl-plot3d's
 * `createScene` also runs, with the scene stand-ins below (canvas, camera, frame
 * scheduler, pickable objects).
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

export interface ReadPixelsCall {
  width: number;
  height: number;
  /** Byte length of the destination's ArrayBuffer (typed-array pool views round it up to a power of two). */
  destinationBytes: number;
}

/**
 * A WebGL context stand-in that records what the bundle does with it.
 *
 * Shader compilation and linking always succeed and report no active uniforms or
 * attributes. Buffers remember the bytes of their last `bufferData` upload, keyed by
 * the buffer bound to the target at the time. `readPixels` leaves its destination
 * untouched unless `onReadPixels` fills it.
 */
export class RecordingGl {
  readonly calls: GlCall[] = [];
  readonly drawCalls: DrawCall[] = [];
  readonly lineWidths: number[] = [];
  readonly readPixelsCalls: ReadPixelsCall[] = [];
  /** Latest uploaded bytes per buffer handle. */
  readonly bufferBytes = new Map<object, Uint8Array>();
  readonly gl: WebGLRenderingContext;
  /** Writes the pixels a `readPixels` call returns (RGBA bytes, `width × height`). */
  onReadPixels: ((out: Uint8Array, width: number, height: number) => void) | null = null;
  /** What `isContextLost()` reports. */
  contextLost = false;

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
    this.readPixelsCalls.length = 0;
  }

  /** The latest uploaded bytes of a gl-scatter3d cloud's four vertex buffers. */
  uploadsOf(cloud: PointCloud): { position: Uint8Array; color: Uint8Array; glyph: Uint8Array; id: Uint8Array } {
    const bytes = (buffer: { handle: object }) => this.bufferBytes.get(buffer.handle) ?? new Uint8Array(0);
    return {
      position: bytes(cloud.pointBuffer),
      color: bytes(cloud.colorBuffer),
      glyph: bytes(cloud.glyphBuffer),
      id: bytes(cloud.idBuffer),
    };
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
        return this.contextLost;
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
        this.recordReadPixels(args[2] as number, args[3] as number, args[6] as Uint8Array);
        return undefined;
      default:
        return undefined;
    }
  }

  private recordReadPixels(width: number, height: number, out: Uint8Array): void {
    this.readPixelsCalls.push({ width, height, destinationBytes: out.buffer.byteLength });
    this.onReadPixels?.(out, width, height);
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
// gl-plot3d scene stand-ins
// ---------------------------------------------------------------------------

type FakeListener = (event: Record<string, unknown>) => void;

/** Minimal synchronous EventTarget. */
export class FakeEventTarget {
  private readonly listeners = new Map<string, Set<FakeListener>>();

  addEventListener(type: string, listener: FakeListener): void {
    let set = this.listeners.get(type);
    if (!set) this.listeners.set(type, (set = new Set()));
    set.add(listener);
  }

  removeEventListener(type: string, listener: FakeListener): void {
    this.listeners.get(type)?.delete(listener);
  }

  dispatch(type: string, event: Record<string, unknown> = {}): void {
    for (const listener of [...(this.listeners.get(type) ?? [])]) {
      listener({ type, target: this, ...event });
    }
  }
}

/**
 * The canvas a gl-plot3d scene draws to and listens on. Its parent's size is the CSS
 * size that `autoResize` reads; mouse events carry `offsetX`/`offsetY`, which
 * mouse-change uses as canvas-relative coordinates.
 */
export class FakeSceneCanvas extends FakeEventTarget {
  width = 0;
  height = 0;
  readonly style: Record<string, string> = {};
  readonly parentNode: { clientWidth: number; clientHeight: number };

  constructor(cssWidth: number, cssHeight: number) {
    super();
    this.parentNode = { clientWidth: cssWidth, clientHeight: cssHeight };
  }

  getBoundingClientRect(): { left: number; top: number; width: number; height: number } {
    return { left: 0, top: 0, width: this.parentNode.clientWidth, height: this.parentNode.clientHeight };
  }
}

/**
 * Manual `requestAnimationFrame`: `step()` runs the callbacks queued for one frame.
 * `install()` replaces the global `requestAnimationFrame` for the rest of the test file
 * (`cancelAnimationFrame` becomes a no-op). Render loops left queued on an older
 * scheduler never run again, so tests create a fresh scene per scheduler.
 */
export class FrameScheduler {
  private queue: ((time: number) => void)[] = [];
  private nextId = 1;

  install(): this {
    const g = globalThis as Record<string, unknown>;
    g.requestAnimationFrame = (callback: (time: number) => void) => {
      this.queue.push(callback);
      return this.nextId++;
    };
    g.cancelAnimationFrame = () => {};
    return this;
  }

  step(): void {
    const due = this.queue;
    this.queue = [];
    for (const callback of due) callback(0);
  }
}

/**
 * Stand-in for the orbit camera plotly passes as `cameraObject`. gl-plot3d only calls
 * `tick()` (true when the view changed since the last tick) and reads `matrix`.
 */
export class FakeCamera {
  readonly matrix = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
  private pendingMoves = 0;

  /** The next `frames` ticks report motion. */
  move(frames = 1): void {
    this.pendingMoves += frames;
  }

  tick(): boolean {
    if (this.pendingMoves === 0) return false;
    this.pendingMoves -= 1;
    return true;
  }
}

/** What `FakePickObject.pick` returns; `serial` is the alpha byte of the picked pixel. */
export interface FakePickResult {
  index: number;
  distance: number;
  position: number[];
  dataCoordinate: number[];
  serial: number;
}

/**
 * A pickable scene object with one pick slot that records its draw passes. `pick`
 * hits when the pixel's id matches the object's pick base, and passes the pixel's
 * alpha byte through as `serial`, so tests can tell which pick image a hit came from.
 */
export class FakePickObject {
  readonly pickSlots = 1;
  pickBase = 0;
  dirty = true;
  pixelRatio = 1;
  axes: unknown = null;
  readonly bounds = [[-1, -1, -1], [1, 1, 1]];
  drawCount = 0;
  /** `pixelRatio` at each `drawPick` call. */
  readonly pickPixelRatios: number[] = [];

  constructor(private readonly onDrawPick: () => void = () => {}) {}

  get drawPickCount(): number {
    return this.pickPixelRatios.length;
  }

  resetCounts(): void {
    this.drawCount = 0;
    this.pickPixelRatios.length = 0;
  }

  setPickBase(base: number): void {
    this.pickBase = base;
  }

  isOpaque(): boolean {
    return true;
  }

  isTransparent(): boolean {
    return false;
  }

  draw(): void {
    this.drawCount += 1;
  }

  drawTransparent(): void {}

  drawPick(): void {
    this.pickPixelRatios.push(this.pixelRatio);
    this.onDrawPick();
  }

  pick(result: { id: number; value: number[] }): FakePickResult | null {
    if (result.id !== this.pickBase) return null;
    return { index: 0, distance: 0, position: [0, 0, 0], dataCoordinate: [0, 0, 0], serial: result.value[2] };
  }

  highlight(): void {}

  dispose(): void {}
}

/** gl-plot3d's pick-pass counters (fork patch 4). */
export interface PickStats {
  renders: number;
  deferredFrames: number;
  readbacks: number;
  buffers: number;
}

/** The gl-plot3d scene surface the tests use. */
export interface Plot3dScene {
  selection: {
    object: unknown;
    distance: number;
    buttons: number;
    data: FakePickResult | null;
    mouse: number[];
  };
  pickStats?: PickStats;
  onrender: (() => void) | null;
  onselect: ((selection: Plot3dScene['selection']) => void) | null;
  add(object: unknown): void;
  remove(object: unknown): void;
  redraw(): void;
  dispose(): void;
}

// ---------------------------------------------------------------------------
// Bundle loading
// ---------------------------------------------------------------------------

/** The bundle's exports that the fork tests use (its full export map is wider). */
export interface StackglModules {
  gl_scatter3d: (options: Record<string, unknown>) => PointCloud;
  gl_plot3d: { createScene: (options: Record<string, unknown>) => Plot3dScene };
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
    // gl-plot3d and mouse-change register resize, blur and key listeners on `window`.
    if (typeof g.addEventListener !== 'function') {
      const windowEvents = new FakeEventTarget();
      g.addEventListener = windowEvents.addEventListener.bind(windowEvents);
      g.removeEventListener = windowEvents.removeEventListener.bind(windowEvents);
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
