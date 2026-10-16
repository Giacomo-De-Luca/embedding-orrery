/**
 * gl-scatter3d glyph outlines (fork patch 3 in forked/plotly_changes.md).
 *
 * Each marker is a filled glyph (triangles) plus an outline (line segments) sharing
 * the same four vertex buffers. The outline is only drawn when `lineWidth > 0`, and
 * plotly defaults `marker.line.width` to 0 (1 only when `marker.size` is an array), so
 * `update()` should not build or upload outline vertices at width 0. Everything that
 * is drawn must stay byte-identical.
 */
import { beforeEach, describe, expect, it } from 'vitest';
import { GL_ENUMS, RecordingGl, StackglHarness, type PointCloud } from './stackglHarness';

const { gl_scatter3d: createPointCloud } = StackglHarness.load();

/** Bytes per vertex in each buffer: position 3×f32, colour 4×f32, glyph offset 2×f32, id u32. */
const STRIDES = { position: 12, color: 16, glyph: 8, id: 4 };
const { TRIANGLES: GL_TRIANGLES, LINES: GL_LINES } = GL_ENUMS;
const CAMERA = {};

const POSITIONS = [[0, 0, 0], [1, 2, 3], [-1, 0.5, 2]];

let recorder: RecordingGl;

beforeEach(() => {
  recorder = new RecordingGl();
});

/** Options as plotly's scatter3d convert.js passes them (convert.js:416-429). */
function options(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    gl: recorder.gl,
    position: POSITIONS,
    color: [[0.2, 0.4, 0.6, 1], [0.8, 0.1, 0.1, 1], [0.1, 0.9, 0.3, 1]],
    size: 12,
    glyph: '●',
    opacity: 1,
    orthographic: true,
    lineWidth: 0,
    lineColor: [0, 0, 0, 1],
    project: [false, false, false],
    projectScale: [2 / 3, 2 / 3, 2 / 3],
    projectOpacity: [1, 1, 1],
    ...overrides,
  };
}

function build(overrides: Record<string, unknown> = {}): PointCloud {
  const cloud = createPointCloud(options(overrides));
  cloud.axes = { bounds: [[-10, -10, -10], [10, 10, 10]] };
  return cloud;
}

/** The latest uploaded bytes of each of the cloud's four buffers. */
function uploads(cloud: PointCloud): { position: Uint8Array; color: Uint8Array; glyph: Uint8Array; id: Uint8Array } {
  const bytes = (buffer: { handle: object }) => recorder.bufferBytes.get(buffer.handle) ?? new Uint8Array(0);
  return {
    position: bytes(cloud.pointBuffer),
    color: bytes(cloud.colorBuffer),
    glyph: bytes(cloud.glyphBuffer),
    id: bytes(cloud.idBuffer),
  };
}

function uploadLengths(cloud: PointCloud) {
  const u = uploads(cloud);
  return { position: u.position.length, color: u.color.length, glyph: u.glyph.length, id: u.id.length };
}

function lengthsFor(vertexCount: number) {
  return {
    position: vertexCount * STRIDES.position,
    color: vertexCount * STRIDES.color,
    glyph: vertexCount * STRIDES.glyph,
    id: vertexCount * STRIDES.id,
  };
}

/** Reference counts from a width-1 build; they depend on the fake glyph, so never hard-code them. */
function referenceCounts(): { tri: number; line: number } {
  const ref = build({ lineWidth: 1 });
  return { tri: ref.vertexCount, line: ref.lineVertexCount };
}

/** Draw calls issued by `draw`, `drawTransparent` and `drawPick` on a fresh recording. */
function drawCallsOf(cloud: PointCloud) {
  recorder.resetCalls();
  cloud.draw(CAMERA);
  cloud.drawTransparent(CAMERA);
  cloud.drawPick(CAMERA);
  return { draws: [...recorder.drawCalls], lineWidths: [...recorder.lineWidths] };
}

describe('gl-scatter3d outline geometry', () => {
  it('the fake glyph produces both a fill mesh and an outline', () => {
    const ref = referenceCounts();
    expect(ref.tri).toBeGreaterThan(0);
    expect(ref.line).toBeGreaterThan(0);
  });

  it('builds no outline vertices when the line width is 0', () => {
    const ref = referenceCounts();
    const cloud = build({ lineWidth: 0 });
    expect(cloud.lineVertexCount).toBe(0);
    expect(cloud.vertexCount).toBe(ref.tri);
    expect(uploadLengths(cloud)).toEqual(lengthsFor(ref.tri));
  });

  it.each([
    { label: 'opaque', opacity: 1 },
    { label: 'translucent (transparent pass)', opacity: 0.5 },
  ])('draws only triangles at width 0 in every pass: $label', ({ opacity }) => {
    const ref = referenceCounts();
    const cloud = build({ lineWidth: 0, opacity });
    const { draws, lineWidths } = drawCallsOf(cloud);
    expect(draws.length).toBeGreaterThan(0);
    for (const call of draws) {
      expect(call).toEqual({ mode: GL_TRIANGLES, first: 0, count: ref.tri });
    }
    expect(lineWidths).toEqual([]);
  });

  it('keeps fill and outline geometry when the line width is positive', () => {
    const ref = referenceCounts();
    const cloud = build({ lineWidth: 1 });
    expect(cloud.vertexCount).toBe(ref.tri);
    expect(cloud.lineVertexCount).toBe(ref.line);
    expect(uploadLengths(cloud)).toEqual(lengthsFor(ref.tri + ref.line));

    cloud.pixelRatio = 2;
    const { draws, lineWidths } = drawCallsOf(cloud);
    expect(draws).toContainEqual({ mode: GL_TRIANGLES, first: 0, count: ref.tri });
    expect(draws).toContainEqual({ mode: GL_LINES, first: ref.tri, count: ref.line });
    // draw (opaque pass) uses the cloud's pixel ratio; drawPick always uses 1.
    expect(lineWidths).toContain(2);
    expect(lineWidths).toContain(1);
  });

  it('uploads byte-identical fill data with and without outlines', () => {
    const ref = referenceCounts();
    const withOutline = uploads(build({ lineWidth: 1 }));
    const withoutOutline = uploads(build({ lineWidth: 0 }));
    for (const key of Object.keys(STRIDES) as (keyof typeof STRIDES)[]) {
      const fillBytes = ref.tri * STRIDES[key];
      expect(withoutOutline[key].subarray(0, fillBytes)).toEqual(withOutline[key].subarray(0, fillBytes));
    }
  });

  it('rebuilds outlines when the same object goes 1 → 0 → 1', () => {
    const fresh0 = uploads(build({ lineWidth: 0 }));
    const fresh1 = uploads(build({ lineWidth: 1 }));

    const ref = referenceCounts();
    const cloud = build({ lineWidth: 1 });
    cloud.update(options({ lineWidth: 0 }));
    expect(cloud.lineVertexCount).toBe(0);
    expect(uploads(cloud)).toEqual(fresh0);
    cloud.update(options({ lineWidth: 1 }));
    expect(cloud.lineVertexCount).toBe(ref.line);
    expect(uploads(cloud)).toEqual(fresh1);
  });

  it('a translucent line colour still marks the cloud transparent at width 0', () => {
    const cloud = build({ lineWidth: 0, lineColor: [0, 0, 0, 0.5] });
    expect(cloud.hasAlpha).toBe(true);
    expect(cloud.isTransparent()).toBe(true);
  });

  it('keeps pick ids aligned with point indices around invalid positions', () => {
    const cloud = build({
      lineWidth: 0,
      position: [[0, 0, 0], [NaN, 0, 0], [1, 1, 1]],
      color: [0.5, 0.5, 0.5, 1],
    });
    // Two valid points share the fill vertices; every vertex carries its point's index.
    const ids = Array.from(new Uint32Array(uploads(cloud).id.slice().buffer));
    const perPoint = cloud.vertexCount / 2;
    expect(Number.isInteger(perPoint)).toBe(true);
    expect(ids).toEqual([...Array(perPoint).fill(0), ...Array(perPoint).fill(2)]);
    // Smoke check: the pick result decodes to the third point.
    expect(cloud.pick({ id: cloud.pickId, value: [0, 0, 2] })?.index).toBe(2);
  });
});
