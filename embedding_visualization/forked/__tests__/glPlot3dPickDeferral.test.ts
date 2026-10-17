/**
 * gl-plot3d pick deferral (fork patch 4 in forked/plotly_changes.md).
 *
 * gl-plot3d draws every object a second time into "pick" framebuffers and reads them
 * back to the CPU, so a mouse position can be resolved to a point. Upstream did that on
 * every frame the camera moved. The patch leaves the pick buffers alone while the camera
 * moves and renders them on the first still frame; a hover that arrives while they are
 * stale is remembered and replayed after the readback, and a press renders them on
 * demand so no click is lost. The buffers are also allocated at pick resolution, and
 * gl-select-static's double-readback bug (an inverted `clearTimeout` guard) is fixed.
 *
 * The scene runs against the recording GL stub with fake pick objects that count their
 * own draws, a fake camera whose `tick()` reports whatever the test says, hand-stepped
 * animation frames, and fake timers for the 1 ms readback.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { FakeCanvasElement, GL_ENUMS, RecordingGl, StackglHarness, type GlScene } from './stackglHarness';

const { gl_plot3d: glPlot3d } = StackglHarness.load();
const frames = StackglHarness.frames;

const PIXEL_RATIO = 2;
const DRAWING_BUFFER = { width: 1600, height: 1200 };
/** gl-plot3d renders picks at drawing-buffer size divided by the pixel ratio. */
const PICK_SHAPE = [DRAWING_BUFFER.width / PIXEL_RATIO, DRAWING_BUFFER.height / PIXEL_RATIO];
/** gl-plot3d packs objects into pick buffers of at most 254 ids. */
const IDS_PER_BUFFER = 254;
/** readPixels fills: R is the pick id (1 = the first object in a buffer), GBA the point index. */
const HIT = [1, 0, 0, 0];
const MISS = [255, 255, 255, 255];
const IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

const nextPow2 = (n: number) => 2 ** Math.ceil(Math.log2(n));

/** Camera stand-in: `tick()` is what redraw() asks, `matrix` what it reads. */
class FakeCamera {
  moving = false;
  matrix = IDENTITY.slice();

  tick(): boolean {
    return this.moving;
  }
}

/** The object surface gl-plot3d drives; counts its own draws instead of touching GL. */
class FakePickObject {
  pickSlots = 1;
  pickId = 0;
  /** False makes `pick()` miss, so a test can decide which buffer answers a query. */
  pickable = true;
  pixelRatio = 1;
  dirty = false;
  axes: unknown = null;
  axesBounds: unknown = null;
  clipBounds: unknown = null;
  bounds = [[-1, -1, -1], [1, 1, 1]];
  draws = 0;
  pickDraws = 0;
  highlighted: unknown = undefined;

  setPickBase(id: number): void {
    this.pickId = id;
  }

  draw(): void {
    this.draws++;
  }

  drawPick(): void {
    this.pickDraws++;
  }

  isOpaque(): boolean {
    return true;
  }

  isTransparent(): boolean {
    return false;
  }

  pick(result: { id: number; value: number[] }): { index: number; position: number[]; dataCoordinate: number[] } | null {
    if (!this.pickable || result.id !== this.pickId) return null;
    return { index: result.value[2], position: [0, 0, 0], dataCoordinate: [0, 0, 0] };
  }

  highlight(selection: unknown): void {
    this.highlighted = selection;
  }

  dispose(): void {}
}

let recorder: RecordingGl;
let canvas: FakeCanvasElement;
let camera: FakeCamera;
let scene: GlScene | null = null;
let objects: FakePickObject[] = [];
/** Objects passed to `onselect`, null for "nothing under the cursor". */
let selected: unknown[] = [];

beforeEach(() => {
  vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
  frames.reset();
  recorder = new RecordingGl(DRAWING_BUFFER.width, DRAWING_BUFFER.height);
  recorder.readPixelsFill = MISS;
  canvas = new FakeCanvasElement();
  camera = new FakeCamera();
  selected = [];
});

afterEach(() => {
  scene?.dispose();
  scene = null;
  vi.useRealTimers();
});

/** A scene with `count` objects, one still frame drawn, its pick read back, and counters cleared. */
function sceneWith(count: number): GlScene {
  scene = glPlot3d.createScene({
    gl: recorder.gl,
    canvas,
    cameraObject: camera,
    pixelRatio: PIXEL_RATIO,
    autoResize: false,
  });
  scene.onselect = (selection) => {
    selected.push(selection.object);
  };
  objects = Array.from({ length: count }, () => new FakePickObject());
  for (const object of objects) scene.add(object);
  frames.step();
  vi.runAllTimers();
  resetCounters();
  return scene;
}

function resetCounters(): void {
  for (const object of objects) {
    object.draws = 0;
    object.pickDraws = 0;
  }
  recorder.resetCalls();
}

/** Re-renders the pick with the camera still, so the buffers are fresh and hold `fill`. */
function refreshPick(s: GlScene, fill: number[]): void {
  recorder.readPixelsFill = fill;
  s.update();
  frames.step();
  vi.runAllTimers();
  resetCounters();
}

const pickDraws = () => objects.reduce((n, o) => n + o.pickDraws, 0);
const draws = () => objects.reduce((n, o) => n + o.draws, 0);
const readbacks = () => recorder.calls.filter((c) => c.name === 'readPixels');

function mouse(type: string, x: number, y: number, buttons: number): void {
  canvas.dispatch(type, { offsetX: x, offsetY: y, buttons });
}
const hover = (x: number, y: number) => mouse('mousemove', x, y, 0);
const press = (x: number, y: number) => mouse('mousedown', x, y, 1);
const dragTo = (x: number, y: number) => mouse('mousemove', x, y, 1);
const release = (x: number, y: number) => mouse('mouseup', x, y, 0);

describe('pick rendering while the camera moves', () => {
  it('renders no pick and reads nothing back on moving frames', () => {
    const s = sceneWith(1);
    camera.moving = true;
    frames.step();
    frames.step();
    frames.step();
    vi.runAllTimers();
    expect(draws()).toBe(3);
    expect(pickDraws()).toBe(0);
    expect(readbacks()).toHaveLength(0);
    expect(s.pickStats.deferredFrames).toBe(3);
  });

  it('renders the pick once on the first still frame and reads each buffer back once', () => {
    const s = sceneWith(IDS_PER_BUFFER + 46);
    camera.moving = true;
    frames.step();
    camera.moving = false;
    resetCounters();
    frames.step();
    expect(pickDraws()).toBe(objects.length);
    expect(objects.every((o) => o.pixelRatio === PIXEL_RATIO)).toBe(true);
    vi.runAllTimers();
    expect(readbacks()).toHaveLength(2);
    expect(s.pickStats.buffers).toBe(2);
  });

  it('draws nothing on idle frames once the pick is fresh', () => {
    sceneWith(1);
    camera.moving = true;
    frames.step();
    camera.moving = false;
    frames.step();
    vi.runAllTimers();
    resetCounters();
    frames.step();
    frames.step();
    expect(draws()).toBe(0);
    expect(pickDraws()).toBe(0);
    expect(readbacks()).toHaveLength(0);
  });

  it('renders the pick in the same frame as an object update while the camera is still', () => {
    const s = sceneWith(1);
    s.update();
    frames.step();
    expect(pickDraws()).toBe(1);
    expect(s.pickStats.deferredFrames).toBe(0);
  });

  it('defers the pick for an update while the camera moves, then renders it on the first still frame', () => {
    const s = sceneWith(1);
    camera.moving = true;
    s.update();
    frames.step();
    expect(pickDraws()).toBe(0);
    expect(s.pickStats.deferredFrames).toBe(1);
    camera.moving = false;
    frames.step();
    expect(pickDraws()).toBe(1);
  });

  it('still defers when redraw() is called synchronously while the camera moves', () => {
    const s = sceneWith(1);
    camera.moving = true;
    s.redraw();
    expect(draws()).toBe(1);
    expect(pickDraws()).toBe(0);
  });

  it('counts renders, deferred frames and readbacks', () => {
    const s = sceneWith(IDS_PER_BUFFER + 1);
    const before = { ...s.pickStats };
    camera.moving = true;
    frames.step();
    frames.step();
    camera.moving = false;
    frames.step();
    vi.runAllTimers();
    expect(s.pickStats).toEqual({
      buffers: 2,
      renders: before.renders + 1,
      deferredFrames: before.deferredFrames + 2,
      readbacks: before.readbacks + 2,
    });
  });
});

describe('pick readback', () => {
  it('reads each buffer back once when update and redraw run twice in one task', () => {
    const s = sceneWith(1);
    s.update();
    s.redraw();
    s.update();
    s.redraw();
    expect(pickDraws()).toBe(2);
    vi.runAllTimers();
    expect(readbacks()).toHaveLength(1);
  });

  it('leaves no pick framebuffer bound after a readback', () => {
    const s = sceneWith(1);
    s.update();
    frames.step();
    vi.runAllTimers();
    const calls = recorder.calls;
    const lastRead = calls.map((c) => c.name).lastIndexOf('readPixels');
    expect(lastRead).toBeGreaterThan(-1);
    const after = calls.slice(lastRead + 1);
    expect(after.some((c) => c.name === 'bindFramebuffer' && c.args[0] === GL_ENUMS.FRAMEBUFFER && c.args[1] === null)).toBe(true);
  });

  it('allocates pick buffers at pick resolution, not drawing-buffer resolution', () => {
    const s = sceneWith(1);
    s.update();
    frames.step();
    vi.runAllTimers();
    const [read] = readbacks();
    expect(read.args.slice(2, 4)).toEqual(PICK_SHAPE);
    const target = read.args[6] as Uint8Array;
    const bytes = PICK_SHAPE[0] * PICK_SHAPE[1] * 4;
    expect(target.length).toBe(bytes);
    expect(target.buffer.byteLength).toBe(nextPow2(bytes));
  });
});

describe('mouse events while the pick is stale', () => {
  it('resolves a hover against a fresh pick', () => {
    const s = sceneWith(1);
    refreshPick(s, HIT);
    hover(10, 10);
    expect(s.selection.object).toBe(objects[0]);
    expect(objects[0].highlighted).toBe(s.selection.data);
  });

  it('gives no stale hit for a mousemove after the camera moved', () => {
    const s = sceneWith(1);
    refreshPick(s, HIT);
    hover(10, 10);
    camera.moving = true;
    frames.step();
    hover(20, 20);
    expect(s.selection.object).toBeNull();
    expect(objects[0].highlighted).toBeNull();
  });

  it('gives no hit during a pending readback and replays the hover once it completes', () => {
    const s = sceneWith(1);
    recorder.readPixelsFill = HIT;
    camera.moving = true;
    frames.step();
    camera.moving = false;
    frames.step();
    hover(30, 30);
    expect(s.selection.object).toBeNull();
    vi.runAllTimers();
    expect(s.selection.object).toBe(objects[0]);
    expect(s.selection.mouse).toEqual([30, 30]);
  });

  it('replays only the latest deferred hover', () => {
    const s = sceneWith(1);
    recorder.readPixelsFill = HIT;
    camera.moving = true;
    frames.step();
    hover(10, 10);
    hover(40, 40);
    camera.moving = false;
    frames.step();
    vi.runAllTimers();
    expect(s.selection.mouse).toEqual([40, 40]);
    expect(selected.filter((o) => o !== null)).toHaveLength(1);
  });

  it('renders the pick on demand for a press while the camera moves, so the click is not lost', () => {
    const s = sceneWith(1);
    recorder.readPixelsFill = HIT;
    camera.moving = true;
    frames.step();
    resetCounters();
    press(50, 50);
    expect(pickDraws()).toBe(1);
    expect(readbacks()).toHaveLength(1);
    expect(s.selection.object).toBe(objects[0]);
    expect(s.selection.buttons).toBe(1);
  });

  it('reads back without a second pick pass for a press while only the readback is pending', () => {
    const s = sceneWith(1);
    recorder.readPixelsFill = HIT;
    camera.moving = true;
    frames.step();
    camera.moving = false;
    frames.step();
    resetCounters();
    press(50, 50);
    expect(pickDraws()).toBe(0);
    expect(readbacks()).toHaveLength(1);
    expect(s.selection.object).toBe(objects[0]);
    vi.runAllTimers();
    expect(readbacks()).toHaveLength(1);
  });

  it('replays a deferred hover only after every buffer has been read back', () => {
    const s = sceneWith(IDS_PER_BUFFER + 1);
    // Only the second buffer's object can answer the hover. The buffers read back in
    // order, so a replay after the first readback would query the second buffer's
    // stale copy, find nothing, and drop the hover.
    for (const object of objects.slice(0, IDS_PER_BUFFER)) object.pickable = false;
    recorder.readPixelsFill = HIT;
    camera.moving = true;
    frames.step();
    hover(30, 30);
    camera.moving = false;
    frames.step();
    vi.runAllTimers();
    expect(readbacks()).toHaveLength(2);
    expect(s.selection.object).toBe(objects[IDS_PER_BUFFER]);
  });

  it('resolves the release of a drag at its own position once the camera is still', () => {
    const s = sceneWith(1);
    recorder.readPixelsFill = HIT;
    camera.moving = true;
    frames.step();
    hover(10, 10);
    press(10, 10);
    dragTo(60, 60);
    release(60, 60);
    expect(s.selection.object).toBeNull();
    camera.moving = false;
    frames.step();
    vi.runAllTimers();
    expect(s.selection.object).toBe(objects[0]);
    expect(s.selection.mouse).toEqual([60, 60]);
    expect(s.selection.buttons).toBe(0);
  });
});
