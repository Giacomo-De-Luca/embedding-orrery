# Label Placement & Collision Avoidance in 3D Scatter Plots

A guide to TensorBoard's two label rendering strategies and how to adapt them for the Plotly-based `ScatterPlot3D` component.

---

## TensorBoard's Two Modes

### Mode 1: Canvas 2D Labels (Default) — Collision-Aware

**Source:** `scatterPlotVisualizerCanvasLabels.ts` + `label.ts`

A transparent 2D `<canvas>` is overlaid on the WebGL scene. Every frame, all candidate labels are projected from 3D world space to 2D screen coordinates, then placed greedily using a **uniform spatial grid** (`CollisionGrid`) for fast overlap rejection.

#### How it works

1. **Project 3D → 2D.** Each labeled point's world position is projected to pixel coordinates via the camera's view-projection matrix.
2. **Cull behind-camera points.** A dot product check discards points behind the camera.
3. **Two-pass collision test:**
   - *Pass 1 (cheap):* Create a thin bounding box (width = 1px) and test against the grid. This avoids the expensive `measureText()` call for labels that would overlap anyway.
   - *Pass 2 (real):* If pass 1 succeeds, measure the actual text width, extend the bounding box, and attempt a real insert.
4. **Greedy first-come-first-served.** Labels are ordered by priority (hover > selected > neighbors). Higher-priority labels claim screen space first; lower-priority labels are suppressed if they overlap.
5. **Depth-fading.** In 3D mode, labels farther from the camera are drawn with lower opacity using a power-law scale.

#### `CollisionGrid` data structure

The screen is divided into a uniform grid of cells (roughly `screenWidth/25 × screenHeight/50` pixels each). Each cell stores an array of `BoundingBox` objects. When inserting:

1. Find all grid cells the new bounding box overlaps.
2. Test the new box against every existing box in those cells (AABB intersection).
3. If no collisions, insert the box into all overlapped cells.

This turns an O(n²) all-pairs check into roughly O(n) amortized — each label only tests against labels in nearby cells.

```
┌──────────────────────────────────┐
│  Screen divided into grid cells  │
│  ┌───┬───┬───┬───┬───┬───┐      │
│  │   │ A │   │   │   │   │      │
│  ├───┼───┼───┼───┼───┼───┤      │
│  │   │   │   │ B │   │   │  A and B are in different
│  ├───┼───┼───┼───┼───┼───┤  cells, so no collision
│  │   │   │   │   │ C │   │  test needed between them
│  └───┴───┴───┴───┴───┴───┘      │
└──────────────────────────────────┘
```

#### Pros
- Clean, readable labels that never overlap
- Priority system surfaces the most important labels
- Depth-fading provides spatial context
- Cheap — just 2D canvas draws, no extra GPU geometry

#### Cons
- Many labels suppressed at dense clusters (only the "winner" shows)
- Slight disconnect between label position and 3D depth (labels are flat overlay)
- Re-runs every frame (though the grid makes it fast)

---

### Mode 2: 3D Shader Labels — Billboarded Text Geometry

**Source:** `scatterPlotVisualizer3DLabels.ts`

Labels are rendered as **textured quads in world space** using a glyph atlas and custom vertex/fragment shaders. Each character is a pair of triangles textured from a single 8192×80 glyph sheet.

#### How it works

1. **Glyph atlas.** On init, render all 256 ASCII characters onto a single canvas to create a texture atlas. Store per-glyph width and UV offset.
2. **Geometry.** For each label string, build 6 vertices per character (2 triangles). The `posObj` attribute encodes the local offset of each letter (horizontally centered).
3. **Billboard vertex shader.** Extract the camera's right/up/at vectors from the `modelViewMatrix` to build a `pointToCamera` rotation. Each character quad is rotated to face the camera in world space.
4. **No collision detection.** Every point's label is always rendered. No overlap avoidance.

```glsl
// Billboard rotation in vertex shader
vec4 vRight = vec4(modelViewMatrix[0][0], modelViewMatrix[1][0], modelViewMatrix[2][0], 0);
vec4 vUp    = vec4(modelViewMatrix[0][1], modelViewMatrix[1][1], modelViewMatrix[2][1], 0);
mat4 pointToCamera = mat4(vRight, vUp, vAt, vec4(0, 0, 0, 1));
vec4 posRotated = pointToCamera * vec4(scaledPos, 0, 1);
```

#### Pros
- Labels participate in 3D depth (occlude / are occluded naturally)
- GPU-rendered — extremely fast even with many labels
- Natural billboarding (always face camera)

#### Cons
- All labels shown simultaneously → visual clutter in dense regions
- Requires custom shader pipeline
- No overlap avoidance

---

## Adapting for the Plotly `ScatterPlot3D` Component

Your current `ScatterPlot3D` renders labels via a Plotly `scatter3d` trace with `mode: 'text'` — which is essentially Mode 2 (all labels shown, no collision detection, rendered in 3D space). Plotly handles the billboarding internally.

Below are two implementation strategies.

---

### Strategy A: Canvas Overlay with CollisionGrid (Recommended)

This is the most impactful upgrade. Layer a 2D `<canvas>` on top of Plotly's WebGL canvas and draw labels yourself with collision avoidance.

#### Architecture

```
┌─────────────────────────────┐
│  containerRef (relative)    │
│  ┌───────────────────────┐  │
│  │  Plotly WebGL canvas   │  │
│  │  (points, markers)     │  │
│  └───────────────────────┘  │
│  ┌───────────────────────┐  │
│  │  Label canvas (abs)    │  │ ← pointer-events: none
│  │  (2D text overlay)     │  │
│  └───────────────────────┘  │
└─────────────────────────────┘
```

#### Step 1: Port the `CollisionGrid`

```typescript
// lib/utils/collisionGrid.ts

export interface BoundingBox {
  loX: number; loY: number;
  hiX: number; hiY: number;
}

export class CollisionGrid {
  private grid: BoundingBox[][];
  private numHorizCells: number;
  private numVertCells: number;

  constructor(
    private bound: BoundingBox,
    private cellWidth: number,
    private cellHeight: number
  ) {
    this.numHorizCells = Math.ceil((bound.hiX - bound.loX) / cellWidth);
    this.numVertCells = Math.ceil((bound.hiY - bound.loY) / cellHeight);
    this.grid = new Array(this.numHorizCells * this.numVertCells);
  }

  insert(box: BoundingBox, justTest = false): boolean {
    // Reject out-of-bounds
    if (box.hiX < this.bound.loX || box.loX > this.bound.hiX ||
        box.hiY < this.bound.loY || box.loY > this.bound.hiY) {
      return false;
    }

    const minCX = this.cellX(box.loX);
    const maxCX = this.cellX(box.hiX);
    const minCY = this.cellY(box.loY);
    const maxCY = this.cellY(box.hiY);

    // Test for conflicts
    for (let j = minCY; j <= maxCY; j++) {
      for (let i = minCX; i <= maxCX; i++) {
        const cell = this.grid[j * this.numHorizCells + i];
        if (cell) {
          for (const existing of cell) {
            if (this.intersects(box, existing)) return false;
          }
        }
      }
    }

    if (justTest) return true;

    // Insert into overlapped cells
    for (let j = minCY; j <= maxCY; j++) {
      for (let i = minCX; i <= maxCX; i++) {
        const idx = j * this.numHorizCells + i;
        if (!this.grid[idx]) this.grid[idx] = [box];
        else this.grid[idx].push(box);
      }
    }
    return true;
  }

  private cellX(x: number) { return Math.floor((x - this.bound.loX) / this.cellWidth); }
  private cellY(y: number) { return Math.floor((y - this.bound.loY) / this.cellHeight); }
  private intersects(a: BoundingBox, b: BoundingBox) {
    return !(a.loX > b.hiX || a.loY > b.hiY || a.hiX < b.loX || a.hiY < b.loY);
  }
}
```

#### Step 2: Project 3D → 2D Screen Coordinates

Plotly's internal camera is accessible via `graphDiv._fullLayout.scene._scene.glplot.camera`. You need to project each 3D point to pixel coordinates:

```typescript
// lib/utils/project3D.ts
import * as THREE from 'three';

/**
 * Project a 3D world point to 2D screen coordinates using
 * the camera matrices extracted from Plotly's gl-plot3d instance.
 */
export function projectToScreen(
  point: { x: number; y: number; z: number },
  camera: any,       // glplot.camera (has view, projection matrices)
  width: number,
  height: number
): { x: number; y: number; behind: boolean } {
  // gl-plot3d stores camera as 4x4 matrices in flat arrays
  // Alternative: reconstruct from Plotly's scene.camera (eye, center, up)
  const eye = camera.eye || [0, 0, 1];
  const center = camera.center || [0, 0, 0];

  // Use THREE.js for the math
  const cam = new THREE.PerspectiveCamera(45, width / height);
  cam.position.set(eye[0], eye[1], eye[2]);
  cam.lookAt(center[0], center[1], center[2]);
  cam.updateMatrixWorld();

  const vec = new THREE.Vector3(point.x, point.y, point.z);
  vec.project(cam);

  return {
    x: (vec.x * 0.5 + 0.5) * width,
    y: (-vec.y * 0.5 + 0.5) * height,  // flip Y
    behind: vec.z > 1,
  };
}
```

> **Note:** The tricky part with Plotly is that it normalizes data coordinates to fit the scene cube. You'll need to apply the same normalization Plotly uses internally, or read the projected coordinates from Plotly's `_scene` internals. A simpler alternative is described in the "Practical Shortcut" section below.

#### Step 3: Render Labels on Each Frame

```tsx
// Inside ScatterPlot3D component

const labelCanvasRef = useRef<HTMLCanvasElement>(null);

// Create the overlay canvas
useEffect(() => {
  if (!containerRef.current || labelCanvasRef.current) return;
  const canvas = document.createElement('canvas');
  canvas.style.position = 'absolute';
  canvas.style.left = '0';
  canvas.style.top = '0';
  canvas.style.pointerEvents = 'none';
  canvas.style.zIndex = '10';
  containerRef.current.style.position = 'relative';
  containerRef.current.appendChild(canvas);
  labelCanvasRef.current = canvas;
  return () => { canvas.remove(); };
}, []);

// Render labels with collision avoidance on camera move
const renderLabels = useCallback(() => {
  const canvas = labelCanvasRef.current;
  if (!canvas || !highlightedIndices?.size) return;

  const dpr = window.devicePixelRatio;
  canvas.width = width * dpr;
  canvas.height = height * dpr;
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;

  const ctx = canvas.getContext('2d')!;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.scale(dpr, dpr);
  ctx.font = '11px Inter, system-ui, sans-serif';
  ctx.textBaseline = 'middle';

  const grid = new CollisionGrid(
    { loX: 0, hiX: width, loY: 0, hiY: height },
    width / 25,
    height / 50
  );

  const labelMargin = 4;
  const xShift = 8; // offset right of the point

  // Sort by priority: selected first, then by similarity desc
  const candidates = points
    .filter(p => highlightedIndices.has(p.index))
    .sort((a, b) => {
      if (a.index === selectedPoint?.index) return -1;
      if (b.index === selectedPoint?.index) return 1;
      return (highlightedIndices.get(b.index) ?? 0) - (highlightedIndices.get(a.index) ?? 0);
    });

  for (const point of candidates) {
    const screen = projectToScreen(point, camera, width, height);
    if (screen.behind) continue;

    const x = screen.x + xShift;
    const y = screen.y;
    const labelHeight = 14;

    // Pass 1: cheap test with width=1
    const testBox: BoundingBox = {
      loX: x - labelMargin,
      hiX: x + 1 + labelMargin,
      loY: y - labelHeight / 2 - labelMargin,
      hiY: y + labelHeight / 2 + labelMargin,
    };

    if (!grid.insert(testBox, true)) continue;

    // Pass 2: measure real width
    const text = point.label || point.id;
    const textWidth = ctx.measureText(text).width;
    testBox.hiX = x + textWidth + labelMargin;

    if (!grid.insert(testBox)) continue;

    // Draw label with stroke + fill (like TensorBoard)
    const similarity = highlightedIndices.get(point.index) ?? 1;
    const opacity = 0.5 + similarity * 0.5;

    ctx.strokeStyle = `rgba(0, 0, 0, ${opacity * 0.8})`;
    ctx.lineWidth = 3;
    ctx.strokeText(text, x, y);

    ctx.fillStyle = isDark
      ? `rgba(226, 232, 240, ${opacity})`
      : `rgba(30, 41, 59, ${opacity})`;
    ctx.fillText(text, x, y);
  }
}, [width, height, points, highlightedIndices, selectedPoint, isDark]);
```

#### Step 4: Hook into Camera Changes

Call `renderLabels()` on every camera move and on frame updates:

```tsx
// In the handleRelayout callback
const handleRelayout = useCallback((e: Readonly<PlotRelayoutEvent>) => {
  if (isAnimatingRef.current) return;
  const sceneCamera = (e as any)['scene.camera'];
  if (sceneCamera) {
    if (sceneCamera.eye) currentCameraRef.current.eye = sceneCamera.eye;
    if (sceneCamera.center) currentCameraRef.current.center = sceneCamera.center;
  }
  renderLabels(); // ← re-render labels after camera move
}, [renderLabels]);
```

---

### Practical Shortcut: Use Plotly's Own Projection

Instead of reconstructing the camera matrices, you can read projected 2D positions from Plotly's internals after each render. The `_scene.glplot` object has access to the clip-space transform. Alternatively, during label rendering, iterate `graphDiv.querySelectorAll('.point3d')` — but this is fragile.

A more robust shortcut uses `Plotly.toImage` or the scene's internal `project` function:

```typescript
// Access Plotly's scene projection
const scene = graphDivRef.current?._fullLayout?.scene?._scene;
const project = scene?.glplot?.cameraParams; // varies by Plotly version
```

If access proves too fragile, the simplest path is to add `three` as a dependency (it's small) and reconstruct from `scene.camera.eye/center/up` as shown above.

---

### Strategy B: Enhanced Plotly Text Traces (Simpler, Less Control)

Keep using Plotly's native `mode: 'text'` traces but pre-filter which labels to show using a screen-space collision grid computed **before** building traces.

#### How it works

1. Before building `labelTraces`, project all candidate points to 2D.
2. Run them through the `CollisionGrid`.
3. Only include surviving labels in the Plotly text trace.
4. Re-run on camera changes (via `onRelayout`).

```tsx
const labelTraces = useMemo((): PlotlyData[] => {
  if (!showLabels || !highlightedIndices?.size) return [];

  const candidates = points.filter(p => highlightedIndices.has(p.index));

  // Project to 2D and filter with collision grid
  const grid = new CollisionGrid(
    { loX: 0, hiX: width, loY: 0, hiY: height },
    width / 25, height / 50
  );

  const surviving = candidates.filter(p => {
    const screen = projectToScreen(p, currentCamera, width, height);
    if (screen.behind) return false;

    const text = p.label || p.id;
    const textWidth = text.length * 7; // approximate
    const box: BoundingBox = {
      loX: screen.x - 4, hiX: screen.x + textWidth + 4,
      loY: screen.y - 10, hiY: screen.y + 10,
    };
    return grid.insert(box);
  });

  if (!surviving.length) return [];

  return [{
    x: surviving.map(p => p.x),
    y: surviving.map(p => p.y),
    z: surviving.map(p => p.z),
    mode: 'text' as const,
    type: 'scatter3d' as const,
    text: surviving.map(p => p.label || p.id),
    textposition: 'top center' as const,
    textfont: {
      size: 11,
      color: isDark ? '#e2e8f0' : '#1e293b',
    },
    hoverinfo: 'skip' as const,
    showlegend: false,
  }];
}, [showLabels, highlightedIndices, points, isDark, width, height, currentCamera]);
```

#### Tradeoff

| | Strategy A (Canvas Overlay) | Strategy B (Filtered Plotly Traces) |
|---|---|---|
| Visual quality | Full control (stroke, opacity, styling) | Limited to Plotly's text rendering |
| Performance | Lightweight canvas draws | Plotly trace rebuild triggers full replot |
| Collision accuracy | Per-pixel with measured text widths | Approximate (estimated widths) |
| Camera sync | Smooth per-frame updates | Only updates on `relayout` events |
| Depth integration | Labels are flat 2D overlay | Labels have 3D depth (Plotly handles it) |
| Implementation effort | Medium (canvas + projection math) | Low (filter before trace creation) |

---

## Recommendation

**Start with Strategy B** — it integrates with your existing code with minimal changes and gives you collision avoidance immediately. The main addition is the `CollisionGrid` class and a projection function.

**Graduate to Strategy A** when you want:
- Per-frame label updates during camera animation (your cinematic zoom)
- Depth-based opacity fading
- Typographic control (stroke outlines, shadows, custom fonts)
- Labels that don't cause Plotly trace count inflation

Both strategies use the same `CollisionGrid` — TensorBoard's approach is elegant and fast enough for 10k+ labels per frame.

---

## Key Parameters to Tune

| Parameter | TensorBoard Default | Suggested for your use case |
|---|---|---|
| Grid cell width | `screenWidth / 25` | Same — roughly 30-40px cells |
| Grid cell height | `screenHeight / 50` | Same — roughly 12-15px cells |
| Label margin | 2px | 4-6px (your labels are more spaced) |
| Max labels | 10,000 | 500-1000 (you show only highlights) |
| Depth opacity range | `[0.1, 1.0]` | `[0.3, 1.0]` (keep readable) |
| Priority order | hover → selected → neighbors | selected → high similarity → low similarity |
