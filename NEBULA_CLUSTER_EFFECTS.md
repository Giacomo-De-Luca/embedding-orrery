# Nebula Cluster Effects for ScatterPlot3D

## Context

The 3D scatter plot uses Plotly.js WebGL (`scatter3d` traces) to render up to 150k points with categorical/sequential coloring, highlight glow effects, and smooth camera animations. Clusters are currently shown only as differently-colored point groups. The goal is to add nebula/gas cloud visual effects around clusters to reinforce the galaxy theme — soft, glowing, organic shapes that visually delineate cluster boundaries without hard edges.

---

## Accessing Plotly's WebGL Internals

### The GraphDiv → Scene → glplot Chain

Plotly.js 3D scatter plots render via `gl-plot3d`, an internal library that wraps `regl` (a WebGL abstraction). The rendering pipeline is accessible through a chain of internal properties on the Plotly graph div element:

```
graphDiv (HTMLDivElement)
  └── _fullLayout
        └── scene
              ├── camera          ← Layout-level camera state {eye, center, up}
              └── _scene
                    └── glplot    ← The gl-plot3d instance (WebGL renderer)
```

### Getting a Reference to the Graph Div

The existing code already stores this reference via Plotly's `onInitialized` callback:

```typescript
// ScatterPlot3D.tsx:49-61 — Type definition
interface PlotlyGraphDiv extends HTMLDivElement {
  _fullLayout?: {
    scene?: {
      camera?: any;
      _scene?: {
        glplot?: {
          camera?: any;
          draw?: () => void;
        };
      };
    };
  };
}

// ScatterPlot3D.tsx:118 — Ref declaration
const graphDivRef = useRef<PlotlyGraphDiv | null>(null);

// ScatterPlot3D.tsx:719-721 — Ref assignment
<Plot
  onInitialized={(_figure, graphDiv) => {
    graphDivRef.current = graphDiv as PlotlyGraphDiv;
    setPlotReady(true);
  }}
/>
```

### The `glplot` Object — Properties and Methods

Once the plot is initialized, the `glplot` object exposes the full WebGL rendering infrastructure:

```typescript
const glplot = graphDivRef.current._fullLayout?.scene?._scene?.glplot;
```

| Property/Method | Type | Description |
|----------------|------|-------------|
| `glplot.gl` | `WebGLRenderingContext` | The raw WebGL context for the canvas |
| `glplot.canvas` | `HTMLCanvasElement` | The canvas element Plotly renders to |
| `glplot.camera` | `object` | Camera state — `eye`, `center`, `up` (array or object format) |
| `glplot.cameraParams` | `object` | Camera matrices — `projection`, `view`, `model` (Float32Array, 4x4) |
| `glplot.draw()` | `function` | Forces a WebGL redraw |
| `glplot.camera.update()` | `function` | Recomputes camera matrices from eye/center/up |
| `glplot.onrender` | `callback \| null` | Hook called every frame during rendering — **your entry point for custom draws** |
| `glplot.objects` | `array` | Collection of rendered objects (traces as WebGL objects) |
| `glplot.axes` | `object` | Axis configuration and bounds |
| `glplot.selection` | `object` | Currently selected/hovered object info |

### Accessing the Canvas Element Directly

Even without the `glplot` reference, the canvas is in the DOM:

```typescript
const canvas = graphDivRef.current?.querySelector('canvas');
// This returns the same canvas as glplot.canvas
```

**Important**: A canvas can only have ONE WebGL context. Calling `canvas.getContext('webgl')` returns the *existing* context (the one Plotly already created), not a new one. This is how you can get a reference to the same GL context Plotly uses.

### Camera State — Two Access Points

**1. Layout-level camera** (for reading user-driven changes):
```typescript
// Available via relayout events (ScatterPlot3D.tsx:286-293)
const handleRelayout = (e: PlotRelayoutEvent) => {
  const sceneCamera = (e as any)['scene.camera'];
  if (sceneCamera?.eye) currentCameraRef.current.eye = sceneCamera.eye;
  if (sceneCamera?.center) currentCameraRef.current.center = sceneCamera.center;
};
```

**2. glplot-level camera** (for direct manipulation during animations):
```typescript
// ScatterPlot3D.tsx:245-263 — Used during camera animation
const glplot = currentScene?.glplot as any;
if (glplot?.camera) {
  // Array format: [x, y, z]
  glplot.camera.eye = [newEye.x, newEye.y, newEye.z];
  glplot.camera.center = [newCenter.x, newCenter.y, newCenter.z];
  glplot.camera.up = [0, 0, 1];
  glplot.camera.update();  // Recompute matrices
  glplot.draw();           // Trigger render
}
```

### Camera Matrices — Projection, View, Model

The `glplot.cameraParams` object provides the standard 3D rendering matrices as `Float32Array[16]` (column-major 4x4):

```typescript
const { projection, view, model } = glplot.cameraParams;
// projection: Perspective projection matrix (FOV, aspect, near, far)
// view:       View matrix (eye position, look-at target, up vector)
// model:      Model matrix (identity in most cases, or axis scaling)
```

These are the same matrices used by all traces in the scene, so custom draws using these matrices will be in the exact same coordinate space as the data points.

### The `onrender` Hook — Custom Drawing Entry Point

The `glplot.onrender` callback fires on every frame during Plotly's render loop:

```typescript
// Basic pattern for hooking into the render loop
const originalOnRender = glplot.onrender;
glplot.onrender = () => {
  if (originalOnRender) originalOnRender();  // Preserve existing behavior
  // Your custom WebGL draw calls here
};

// Cleanup on unmount
return () => { glplot.onrender = originalOnRender; };
```

### GL State Management

When drawing custom WebGL content, you must save and restore GL state to avoid breaking Plotly's own rendering:

```typescript
// Save state before custom draws
const prevBlend = gl.isEnabled(gl.BLEND);
const prevBlendSrc = gl.getParameter(gl.BLEND_SRC_RGB);
const prevBlendDst = gl.getParameter(gl.BLEND_DST_RGB);
const prevDepthMask = gl.getParameter(gl.DEPTH_WRITEMASK);
const prevProgram = gl.getParameter(gl.CURRENT_PROGRAM);

// ... custom draws ...

// Restore state after
if (prevBlend) gl.enable(gl.BLEND); else gl.disable(gl.BLEND);
gl.blendFunc(prevBlendSrc, prevBlendDst);
gl.depthMask(prevDepthMask);
gl.useProgram(prevProgram);
```

### Coordinate System

Plotly 3D with `aspectmode: 'data'` uses the raw data coordinates. The current layout configuration:

```typescript
// ScatterPlot3D.tsx:613-628
scene: {
  aspectmode: 'data',  // No normalization — 1 unit in data = 1 unit in scene
  camera: {
    eye: defaultEye,       // {x, y, z} relative to scene center
    center: defaultCenter,  // {x, y, z} — look-at target
    up: { x: 0, y: 0, z: 1 }
  },
  xaxis: { range: [bounds.xMin - pad, bounds.xMax + pad], showgrid: false, ... },
  yaxis: { range: [bounds.yMin - pad, bounds.yMax + pad], showgrid: false, ... },
  zaxis: { range: [bounds.zMin - pad, bounds.zMax + pad], showgrid: false, ... },
}
```

This means if your data points are at coordinates like `(3.5, -1.2, 0.8)`, your custom draws at those same coordinates will overlap the data points exactly.

### Limitations and Caveats

1. **Undocumented internals**: `_fullLayout`, `_scene`, `glplot` are not part of Plotly's public API and could change between versions. The current code already relies on these (for camera animation), so this is an accepted trade-off.

2. **regl state**: gl-plot3d uses `regl` which manages WebGL state automatically. After custom GL calls, regl's internal state cache may be stale. Calling `glplot.draw()` after your custom render should force regl to re-sync.

3. **Canvas can only have one WebGL context**: You cannot create a separate Three.js renderer on the same canvas. Three.js would need its own overlay canvas.

4. **Render timing**: `onrender` fires after Plotly has drawn its traces. Custom draws appear on top of Plotly's content. To draw behind (e.g., nebula behind points), use `depthTest: true` and place nebula geometry slightly behind the data points, or use alpha blending with very low opacity so the points remain visible.

---

## Plan A: Plotly Volume Traces (Simpler, Public API)

### Overview

Use Plotly's built-in `volume` trace type to render 3D density fields as translucent volumetric clouds around each cluster. This stays entirely within Plotly's documented API — no custom WebGL needed.

The `volume` and `isosurface` trace types are confirmed to be included in the `plotly.js-dist-min` bundle (v3.3.0) used by this project.

### How Volume Traces Work

Plotly's `volume` trace takes a 3D scalar field (density values on a grid) and renders it as a stack of semi-transparent isosurfaces. It supports:
- Custom `colorscale` for per-cluster coloring
- `opacityscale` mapping density values to opacity (key for nebula look)
- `surface.count` controlling how many isosurface layers are rendered
- `isomin`/`isomax` for filtering out low-density regions

The visual result is layered transparent surfaces that create a volumetric cloud appearance.

### Data Flow

```
Points with topic_id metadata
  → Group by topic_id (skip -1/noise)
  → For each cluster:
      → Compute 3D density grid via KDE
      → Create a volume trace with cluster color + low opacity
  → Add volume traces to baseTraces (before scatter3d traces)
```

### Step 1: Cluster Grouping

Reuse the existing categorical grouping pattern from ScatterPlot3D.tsx:421-449:

```typescript
// Group points by topic_id
const clusterGroups: Record<string, Point3D[]> = {};
displayPoints.forEach(point => {
  const topicId = String(point.metadata?.topic_id ?? '');
  if (topicId === '-1' || topicId === '') return; // Skip noise
  if (!clusterGroups[topicId]) clusterGroups[topicId] = [];
  clusterGroups[topicId].push(point);
});
```

### Step 2: 3D Density Grid Computation

Create a new utility at `lib/utils/clusterGeometry.ts`:

```typescript
interface DensityGrid {
  x: number[];     // Flattened grid x-coordinates (length = gridSize³)
  y: number[];     // Flattened grid y-coordinates
  z: number[];     // Flattened grid z-coordinates
  value: number[]; // Density values at each grid point
  maxValue: number; // Max density (for isomax)
}

function computeDensityGrid(
  points: Point3D[],
  gridSize: number = 30,
  paddingFactor: number = 1.5
): DensityGrid
```

**Algorithm:**

1. **Compute cluster statistics**:
   ```typescript
   const centroid = { x: mean(xs), y: mean(ys), z: mean(zs) };
   const std = { x: stddev(xs), y: stddev(ys), z: stddev(zs) };
   ```

2. **Define grid bounds** (extend beyond cluster by `paddingFactor * std`):
   ```typescript
   const xMin = centroid.x - paddingFactor * std.x;
   const xMax = centroid.x + paddingFactor * std.x;
   // Same for y, z
   ```

3. **Create regular grid** (30x30x30 = 27,000 cells):
   ```typescript
   const dx = (xMax - xMin) / (gridSize - 1);
   for (let i = 0; i < gridSize; i++)
     for (let j = 0; j < gridSize; j++)
       for (let k = 0; k < gridSize; k++) {
         x.push(xMin + i * dx);
         y.push(yMin + j * dy);
         z.push(zMin + k * dz);
       }
   ```

4. **Compute density at each grid point** using Gaussian KDE:
   ```typescript
   // Silverman's rule for 3D bandwidth
   const bandwidth = Math.cbrt(std.x * std.y * std.z) * Math.pow(points.length, -1/7);
   const bw2 = 2 * bandwidth * bandwidth;

   for (let idx = 0; idx < gridSize ** 3; idx++) {
     let density = 0;
     for (const p of points) {
       const dx = x[idx] - p.x;
       const dy = y[idx] - p.y;
       const dz = z[idx] - p.z;
       const distSq = dx*dx + dy*dy + dz*dz;
       if (distSq < 9 * bw2) {  // 3σ cutoff for performance
         density += Math.exp(-distSq / bw2);
       }
     }
     value.push(density);
   }
   ```

5. **Normalize** density values to [0, 1] range.

**Performance**: For a 1000-point cluster with 30³ grid and 3σ cutoff, only nearby points contribute to each cell. Typical computation: <50ms per cluster. Memoize with `useMemo` keyed on `[points, colorByField]`.

### Step 3: Volume Trace Generation

Add volume traces to the `baseTraces` useMemo in ScatterPlot3D:

```typescript
// Inside baseTraces computation, before scatter3d traces
if (showNebula && colorBy === 'category') {
  Object.entries(clusterGroups).forEach(([topicId, clusterPoints]) => {
    if (clusterPoints.length < 10) return; // Skip tiny clusters

    const grid = computeDensityGrid(clusterPoints);
    const clusterColor = colorMap[topicId] || '#7f7f7f';
    const dimColor = adjustColorBrightness(clusterColor, 0.3);

    traces.push({
      type: 'volume' as any,
      x: grid.x,
      y: grid.y,
      z: grid.z,
      value: grid.value,
      isomin: grid.maxValue * 0.05,   // Hide very low density regions
      isomax: grid.maxValue,
      opacity: 0.12,
      surface: { count: 15 },          // More layers = smoother volume
      opacityscale: [
        [0, 0],          // Zero density = invisible
        [0.2, 0.03],     // Low density = barely visible
        [0.5, 0.08],     // Medium density = faint
        [1.0, 0.2],      // High density = moderately visible
      ] as any,
      colorscale: [
        [0, dimColor],
        [0.5, clusterColor],
        [1, lightenColor(clusterColor, 0.3)],
      ] as any,
      showscale: false,
      hoverinfo: 'skip' as any,
      showlegend: false,
    });
  });
}
```

### Step 4: Layered Volumes for Richer Effect

For a more convincing nebula, add two volume traces per cluster at different scales:

```typescript
// Inner core — tighter, brighter, more opaque
traces.push({
  type: 'volume',
  // ... grid at 1.0x padding ...
  opacity: 0.15,
  surface: { count: 10 },
  opacityscale: [[0, 0], [0.3, 0.05], [1, 0.25]],
  colorscale: [[0, clusterColor], [1, lightenColor(clusterColor, 0.4)]],
});

// Outer halo — wider, dimmer, more transparent
const outerGrid = computeDensityGrid(clusterPoints, 25, 2.5); // Larger padding
traces.push({
  type: 'volume',
  // ... grid at 2.5x padding ...
  opacity: 0.06,
  surface: { count: 8 },
  opacityscale: [[0, 0], [0.1, 0.02], [1, 0.1]],
  colorscale: [[0, dimColor], [1, clusterColor]],
});
```

### Visual Tuning Parameters

| Parameter | Effect | Recommended Range |
|-----------|--------|-------------------|
| `opacity` | Global transparency | 0.05 - 0.2 |
| `surface.count` | Number of isosurface layers | 8 - 20 (more = smoother) |
| `opacityscale` | Maps density to per-layer opacity | Keep max ≤ 0.3 to avoid artifacts |
| `isomin` | Minimum density threshold | 5-15% of max density |
| `gridSize` | Grid resolution | 25-40 (tradeoff: detail vs. performance) |
| `paddingFactor` | How far nebula extends beyond cluster | 1.2 - 2.5 |
| `bandwidth` | KDE smoothing | Silverman's rule ± manual adjustment |

### Known Limitations

1. **Depth sorting artifacts**: Plotly's WebGL has imperfect depth sorting for overlapping transparent surfaces. When opacity >= 0.5 on two overlapping surfaces, rendering artifacts appear. Keeping max opacity below 0.3 mitigates this.

2. **No additive blending**: Volume traces use standard alpha blending, not additive. Overlapping nebulae don't "glow brighter" — they just blend. This limits the luminous nebula look.

3. **Grid discretization**: The nebula shape is limited by grid resolution. At 30^3, fine structures are lost. Increasing to 50^3 improves detail but adds 125k array elements per cluster.

4. **Performance with many clusters**: Each volume trace is a separate WebGL render pass. With 15+ clusters x 2 layers each = 30 volume traces, rendering may slow down. Consider limiting to the largest N clusters.

---

## Plan B: Direct WebGL Custom Rendering (More Control)

### Overview

Hook into Plotly's render loop via `glplot.onrender` to draw custom particle nebula effects using the raw WebGL context and Plotly's camera matrices. This gives full control over blending, shaders, and particle placement.

### Architecture

```
ScatterPlot3D.tsx
  ├── <Plot /> (unchanged)
  └── useEffect (nebula WebGL hook)
        ├── Creates shader program once
        ├── Creates particle buffers per cluster
        └── Hooks into glplot.onrender
              ├── Saves GL state
              ├── Sets additive blending
              ├── Draws nebula particles using Plotly's camera matrices
              └── Restores GL state
```

All custom WebGL code lives in a new utility file that manages the shader lifecycle.

### Step 1: Shader Program

Create `lib/utils/nebulaRenderer.ts`:

```typescript
// Vertex shader — billboard sprites with size attenuation
const VERTEX_SHADER = `
  attribute vec3 aPosition;
  attribute float aOpacity;
  attribute float aSize;

  uniform mat4 uProjection;
  uniform mat4 uView;
  uniform mat4 uModel;

  varying float vOpacity;

  void main() {
    vOpacity = aOpacity;
    vec4 mvPosition = uView * uModel * vec4(aPosition, 1.0);
    gl_PointSize = aSize * (300.0 / -mvPosition.z);  // Size attenuation
    gl_Position = uProjection * mvPosition;
  }
`;

// Fragment shader — soft gaussian sprite with additive blending
const FRAGMENT_SHADER = `
  precision mediump float;

  uniform vec3 uColor;
  varying float vOpacity;

  void main() {
    // Distance from sprite center (gl_PointCoord is [0,1] for each sprite)
    float dist = length(gl_PointCoord - vec2(0.5));

    // Discard pixels outside the circle
    if (dist > 0.5) discard;

    // Gaussian-like falloff: smooth center to edge
    float alpha = smoothstep(0.5, 0.0, dist);
    alpha *= alpha;  // Quadratic falloff for softer edges
    alpha *= vOpacity;

    // Output: color with computed alpha
    // With additive blending (gl.blendFunc(SRC_ALPHA, ONE)),
    // this creates glow where particles overlap
    gl_FragColor = vec4(uColor * alpha, alpha);
  }
`;
```

### Step 2: Nebula Particle Generation

Instead of a density grid, sample particle positions directly from each cluster's shape:

```typescript
interface NebulaParticles {
  positions: Float32Array;  // [x, y, z, x, y, z, ...] — interleaved
  opacities: Float32Array;  // Per-particle opacity
  sizes: Float32Array;      // Per-particle sprite size
}

function sampleNebulaParticles(
  points: Point3D[],
  particleCount: number = 300
): NebulaParticles
```

**Algorithm:**
1. Compute centroid and covariance matrix of the cluster
2. Eigendecompose the 3x3 covariance matrix (gives ellipsoidal shape)
3. For each particle:
   - Sample `(u, v, w)` from standard normal distribution (Box-Muller)
   - Scale by `sqrt(eigenvalue[i]) * 1.5` per axis
   - Rotate by eigenvector matrix
   - Translate by centroid
   - Compute opacity: `0.02 + 0.13 * exp(-distance^2 / (2 * spread^2))` (brighter near center)
   - Compute size: `20 + 40 * exp(-distance^2 / (2 * spread^2))` (larger near center)

### Step 3: WebGL Buffer Setup

```typescript
class NebulaRenderer {
  private gl: WebGLRenderingContext;
  private program: WebGLProgram;
  private positionBuffer: WebGLBuffer;
  private opacityBuffer: WebGLBuffer;
  private sizeBuffer: WebGLBuffer;
  private particleCount: number;

  // Uniform locations
  private uProjection: WebGLUniformLocation;
  private uView: WebGLUniformLocation;
  private uModel: WebGLUniformLocation;
  private uColor: WebGLUniformLocation;

  constructor(gl: WebGLRenderingContext) {
    this.gl = gl;
    this.program = compileShaderProgram(gl, VERTEX_SHADER, FRAGMENT_SHADER);
    // ... get attribute/uniform locations, create buffers ...
  }

  updateParticles(particles: NebulaParticles) {
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, particles.positions, gl.STATIC_DRAW);
    // ... same for opacity and size buffers ...
    this.particleCount = particles.positions.length / 3;
  }

  draw(
    projection: Float32Array,
    view: Float32Array,
    model: Float32Array,
    color: [number, number, number]
  ) {
    const gl = this.gl;

    gl.useProgram(this.program);

    // Set matrices
    gl.uniformMatrix4fv(this.uProjection, false, projection);
    gl.uniformMatrix4fv(this.uView, false, view);
    gl.uniformMatrix4fv(this.uModel, false, model);
    gl.uniform3fv(this.uColor, color);

    // Bind position attribute
    gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);
    gl.vertexAttribPointer(aPositionLoc, 3, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(aPositionLoc);

    // Bind opacity attribute
    gl.bindBuffer(gl.ARRAY_BUFFER, this.opacityBuffer);
    gl.vertexAttribPointer(aOpacityLoc, 1, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(aOpacityLoc);

    // Bind size attribute
    gl.bindBuffer(gl.ARRAY_BUFFER, this.sizeBuffer);
    gl.vertexAttribPointer(aSizeLoc, 1, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(aSizeLoc);

    // Draw particles
    gl.drawArrays(gl.POINTS, 0, this.particleCount);
  }

  dispose() {
    // Delete buffers and program
  }
}
```

### Step 4: Integration with ScatterPlot3D

Add a `useEffect` hook in ScatterPlot3D that manages the nebula renderer lifecycle:

```typescript
// In ScatterPlot3D component
useEffect(() => {
  if (!plotReady || !showNebula || !graphDivRef.current) return;

  const glplot = graphDivRef.current._fullLayout?.scene?._scene?.glplot as any;
  if (!glplot?.gl) return;

  const gl = glplot.gl as WebGLRenderingContext;

  // Create one renderer per cluster
  const clusterRenderers: Array<{
    renderer: NebulaRenderer;
    color: [number, number, number];
  }> = [];

  Object.entries(clusterGroups).forEach(([topicId, clusterPoints]) => {
    if (clusterPoints.length < 10) return;
    const r = new NebulaRenderer(gl);
    const particles = sampleNebulaParticles(clusterPoints);
    r.updateParticles(particles);
    const [red, green, blue] = hexToRgbNormalized(colorMap[topicId] || '#7f7f7f');
    clusterRenderers.push({ renderer: r, color: [red, green, blue] });
  });

  // Hook into render loop
  const originalOnRender = glplot.onrender;
  glplot.onrender = () => {
    if (originalOnRender) originalOnRender();

    const { projection, view, model } = glplot.cameraParams;

    // Save GL state
    const prevBlend = gl.isEnabled(gl.BLEND);
    const prevBlendSrc = gl.getParameter(gl.BLEND_SRC_RGB);
    const prevBlendDst = gl.getParameter(gl.BLEND_DST_RGB);
    const prevDepthMask = gl.getParameter(gl.DEPTH_WRITEMASK);
    const prevProgram = gl.getParameter(gl.CURRENT_PROGRAM);

    // Configure for nebula rendering
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE);  // ADDITIVE blending — the key to glow
    gl.depthMask(false);                  // Don't write depth — particles layer freely

    // Draw each cluster's nebula
    clusterRenderers.forEach(({ renderer, color }) => {
      renderer.draw(projection, view, model, color);
    });

    // Restore GL state
    gl.depthMask(true);
    gl.blendFunc(prevBlendSrc, prevBlendDst);
    if (!prevBlend) gl.disable(gl.BLEND);
    gl.useProgram(prevProgram);
  };

  return () => {
    glplot.onrender = originalOnRender;
    clusterRenderers.forEach(({ renderer }) => renderer.dispose());
  };
}, [plotReady, showNebula, clusterGroups, colorMap]);
```

### Why Additive Blending Creates Nebula Glow

Standard alpha blending: `result = src * alpha + dst * (1 - alpha)` — overlapping transparent objects get darker.

Additive blending: `result = src * alpha + dst * 1` — overlapping objects get BRIGHTER. This is exactly how real nebulae work: denser regions accumulate more light.

With additive blending + gaussian sprite falloff:
- **Cluster core** (many overlapping particles): Bright glow
- **Cluster edge** (sparse particles): Faint haze
- **Between clusters**: No effect
- The glow intensity naturally maps to point density

### Comparison: Plan A vs Plan B

| Feature | Volume Traces (Plan A) | Direct WebGL (Plan B) |
|---------|----------------------|---------------------|
| Blending | Standard alpha only | **Additive** (glow effect) |
| Edge quality | Discrete isosurface steps | **Smooth gaussian falloff** |
| Grid artifacts | Visible at low resolution | **None** (particles are continuous) |
| Color control | Colorscale mapping | **Exact per-cluster color** |
| API stability | Public Plotly API | Undocumented internals |
| Implementation effort | Lower | Higher |
| Depth sorting | Has known artifacts | Depth-test-free (additive) |

### Performance Considerations

- 300 particles x 20 clusters = 6,000 points per frame — trivial for WebGL
- One shader program shared across all clusters
- Particle buffers created once, updated only when data changes
- `onrender` callback adds ~0.5ms per frame
- No impact on Plotly's own rendering performance

### Risk: Undocumented API Breakage

The `glplot.onrender`, `glplot.gl`, and `glplot.cameraParams` properties are internal to Plotly/gl-plot3d and could change between major versions. However:
- The existing ScatterPlot3D code already depends on `glplot.camera`, `glplot.draw()` for animation (lines 245-263)
- This is an accepted trade-off in the project already
- gl-plot3d is mature and rarely changes (last major update was years ago)

---

## Files to Create

| File | Purpose |
|------|---------|
| `lib/utils/clusterGeometry.ts` | Cluster statistics + density grid computation (Plan A) + particle sampling (Plan B) |
| `lib/utils/nebulaRenderer.ts` | *(Plan B only)* WebGL shader program, buffer management, draw calls |

## Files to Modify

| File | Change |
|------|--------|
| `ScatterPlot3D.tsx` | Add nebula rendering (volume traces or WebGL hook), add `showNebula` prop, extend `PlotlyGraphDiv` interface |
| `types.ts` | Add `showNebula?: boolean` to `VisualizationState` |
| `VisualizationControls.tsx` | Add "Show nebula" toggle (visible when mode=3D + coloring by topic field) |
| `DashboardPanel.tsx` | Pass `showNebula` state to ScatterPlot3D |

## Existing Code to Reuse

- `buildCategoryColorMap()` from `categoryColors.ts` — cluster colors
- Categorical point grouping pattern from `ScatterPlot3D.tsx:421-449`
- `PlotlyGraphDiv` interface from `ScatterPlot3D.tsx:49-61` — extend with `gl`, `cameraParams`
- `currentCameraRef` pattern from `ScatterPlot3D.tsx:149`

## Implementation Sequence

1. Create `clusterGeometry.ts` with cluster stats + density grid / particle sampling
2. **Plan A first**: Add volume traces to `baseTraces` in ScatterPlot3D
3. Add `showNebula` toggle to types + controls + dashboard
4. Evaluate visual quality — if sufficient, stop here
5. **Plan B if needed**: Create `nebulaRenderer.ts`, add `onrender` hook to ScatterPlot3D
6. Tune visual parameters (opacity, particle count, colorscale)

## Verification

1. Load a dataset with topic extraction → nebulae should appear around each cluster
2. Rotate/zoom/pan → nebulae should track the data correctly
3. Toggle "Show nebula" on/off
4. Verify existing features still work: coloring modes, highlights, hover tooltip, point click, camera animation, hide unclustered, muted categories
5. Test with varying cluster sizes and counts (5 clusters, 15 clusters, 30+ clusters)
6. Performance check with large datasets (50k+ points)
7. Dark/light theme rendering
