/**
 * Composite the on-screen visualization (Plotly plot + overlay canvases) into a
 * single canvas at screen resolution × devicePixelRatio.
 *
 * The on-screen image is a stack of layers that no single API captures:
 *  - the Plotly plot (2D: WebGL points + SVG axes; 3D: everything in one GL canvas)
 *  - the 3D haze/nebula overlay (Three.js WebGL canvas, CSS mix-blend-mode)
 *  - the label overlay (2D canvas, cluster/point labels)
 * The legend is composited separately by the caller (see drawLegendOverlay).
 */

interface Capture2DOptions {
  gd: any;
  plotlyLib: any;
  container: HTMLElement;
  labelCanvas: HTMLCanvasElement | null;
  isDark: boolean;
}

interface Capture3DOptions {
  gd: any;
  container: HTMLElement;
  hazeCanvas: HTMLCanvasElement | null;
  labelCanvas: HTMLCanvasElement | null;
  isDark: boolean;
}

function loadImage(dataUrl: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error('Failed to decode captured plot image'));
    img.src = dataUrl;
  });
}

function createOutputCanvas(
  container: HTMLElement,
  isDark: boolean,
): { canvas: HTMLCanvasElement; ctx: CanvasRenderingContext2D; cssW: number; cssH: number; dpr: number } {
  const dpr = window.devicePixelRatio || 1;
  const rect = container.getBoundingClientRect();
  const cssW = rect.width;
  const cssH = rect.height;
  const canvas = document.createElement('canvas');
  canvas.width = Math.round(cssW * dpr);
  canvas.height = Math.round(cssH * dpr);
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('Could not create 2D context for screenshot');
  // The plot/paper backgrounds are transparent in dark mode, so fill explicitly
  ctx.fillStyle = isDark ? '#000000' : '#ffffff';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  return { canvas, ctx, cssW, cssH, dpr };
}

/** Copy a live canvas into a detached snapshot synchronously (WebGL buffers may be cleared after the current task). */
function snapshotCanvas(source: HTMLCanvasElement): HTMLCanvasElement | null {
  if (source.width === 0 || source.height === 0) return null;
  const copy = document.createElement('canvas');
  copy.width = source.width;
  copy.height = source.height;
  copy.getContext('2d')?.drawImage(source, 0, 0);
  return copy;
}

/**
 * Capture the 2D scatter plot. scattergl axes/gridlines/ticks are SVG layers,
 * so the plot is re-rendered via Plotly.toImage (which composites SVG + GL),
 * then the label overlay is drawn on top.
 */
export async function capture2DPlot(opts: Capture2DOptions): Promise<HTMLCanvasElement> {
  const { gd, plotlyLib, container, labelCanvas, isDark } = opts;
  const { canvas, ctx, cssW, cssH, dpr } = createOutputCanvas(container, isDark);
  // toImage clones gd synchronously at call time, so interaction during the await is safe
  const dataUrl = await plotlyLib.toImage(gd, {
    format: 'png',
    width: Math.round(cssW),
    height: Math.round(cssH),
    scale: dpr,
  });
  const plotImg = await loadImage(dataUrl);
  ctx.drawImage(plotImg, 0, 0, canvas.width, canvas.height);
  if (labelCanvas) {
    ctx.drawImage(labelCanvas, 0, 0, canvas.width, canvas.height);
  }
  return canvas;
}

/**
 * Rasterize the live legend DOM node and composite it onto the export canvas
 * at its on-screen position. The Card is transparent with backdrop-blur, which
 * foreignObject rasterization cannot reproduce, so the plot pixels beneath the
 * card are blurred first. Elements marked `data-export-exclude` (e.g. the
 * resize drag handle) are left out.
 */
export async function drawLegendOverlay(canvas: HTMLCanvasElement, legendEl: HTMLElement): Promise<void> {
  const parent = legendEl.offsetParent as HTMLElement | null;
  const ctx = canvas.getContext('2d');
  if (!parent || !ctx) return;
  // Rasterize the Card inside the wrapper, not the wrapper itself: the
  // wrapper's computed `position: absolute; top/right` offsets survive into
  // the cloned foreignObject and push the content outside the SVG viewBox,
  // yielding a blank raster.
  const target = (legendEl.firstElementChild as HTMLElement | null) ?? legendEl;
  const parentRect = parent.getBoundingClientRect();
  const rect = target.getBoundingClientRect();
  if (parentRect.width === 0 || rect.width === 0 || rect.height === 0) return;
  // The legend overlay and the plot layer share the same positioned container,
  // so on-screen offsets translate directly; scale maps CSS px → canvas px.
  const scale = canvas.width / parentRect.width;

  const { toCanvas } = await import('html-to-image');
  const legendCanvas = await toCanvas(target, {
    pixelRatio: scale,
    filter: (node) => !(node instanceof HTMLElement && 'exportExclude' in node.dataset),
  });

  const x = (rect.left - parentRect.left) * scale;
  const y = (rect.top - parentRect.top) * scale;
  const w = rect.width * scale;
  const h = rect.height * scale;

  // Frosted backdrop (backdrop-blur-sm on the rounded-xl Card)
  if ('filter' in ctx) {
    ctx.save();
    ctx.beginPath();
    if (typeof ctx.roundRect === 'function') {
      ctx.roundRect(x, y, w, h, 12 * scale);
    } else {
      ctx.rect(x, y, w, h);
    }
    ctx.clip();
    ctx.filter = `blur(${4 * scale}px)`;
    ctx.drawImage(canvas, 0, 0);
    ctx.restore();
  }
  ctx.drawImage(legendCanvas, x, y, w, h);
}

/**
 * Capture the 3D scatter plot from the live GL scene.
 *
 * `scene.toImage()` synchronously forces a glplot redraw (which also fires the
 * `onrender` hook that repaints the haze canvas) and reads the pixels before
 * returning — this sidesteps both `preserveDrawingBuffer: false` and the
 * stale-camera problem (fly-to manipulates `glplot.camera` imperatively, so
 * the layout's camera can lag the live view). The haze canvas is snapshotted
 * in the same task, before the browser composites and clears its GL buffer.
 */
export async function capture3DPlot(opts: Capture3DOptions): Promise<HTMLCanvasElement> {
  const { gd, container, hazeCanvas, labelCanvas, isDark } = opts;
  const scene = gd?._fullLayout?.scene?._scene;
  const glplot = scene?.glplot;
  const glCanvas = (glplot?.gl?.canvas as HTMLCanvasElement | undefined) ?? null;
  if (!scene || !glCanvas) throw new Error('3D scene is not ready for capture');

  // --- Synchronous phase: read all GL-backed pixels before any await ---
  let glDataUrl: string | null = null;
  let glSnapshot: HTMLCanvasElement | null = null;
  if (typeof scene.toImage === 'function') {
    glDataUrl = scene.toImage('png');
  } else {
    // Defensive fallback: redraw then copy the live canvas in the same task
    glplot.redraw?.();
    glSnapshot = snapshotCanvas(glCanvas);
  }
  // scene.toImage's redraw fired onrender → the haze canvas was just repainted
  const hazeSnapshot = hazeCanvas ? snapshotCanvas(hazeCanvas) : null;
  const containerRect = container.getBoundingClientRect();
  const glRect = glCanvas.getBoundingClientRect();
  const hazeRect = hazeCanvas?.getBoundingClientRect() ?? null;

  // --- Async phase: compositing ---
  const { canvas, ctx, dpr } = createOutputCanvas(container, isDark);
  const glImage = glDataUrl ? await loadImage(glDataUrl) : glSnapshot;
  if (glImage) {
    ctx.drawImage(
      glImage,
      (glRect.left - containerRect.left) * dpr,
      (glRect.top - containerRect.top) * dpr,
      glRect.width * dpr,
      glRect.height * dpr,
    );
  }
  if (hazeSnapshot && hazeRect) {
    // Replicate the overlay's CSS mix-blend-mode compositing
    ctx.globalCompositeOperation = isDark ? 'screen' : 'multiply';
    ctx.drawImage(
      hazeSnapshot,
      (hazeRect.left - containerRect.left) * dpr,
      (hazeRect.top - containerRect.top) * dpr,
      hazeRect.width * dpr,
      hazeRect.height * dpr,
    );
    ctx.globalCompositeOperation = 'source-over';
  }
  if (labelCanvas) {
    ctx.drawImage(labelCanvas, 0, 0, canvas.width, canvas.height);
  }
  return canvas;
}
