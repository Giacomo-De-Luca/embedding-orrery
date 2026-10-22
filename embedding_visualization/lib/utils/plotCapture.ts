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

import { PremultipliedPixels, type Rgb255 } from './premultipliedPixels';

/** The plot/paper backgrounds are transparent in dark mode, so the export fills its own. */
function exportBackground(isDark: boolean): Rgb255 {
  return isDark ? [0, 0, 0] : [255, 255, 255];
}

interface Capture2DOptions {
  gd: any;
  plotlyLib: any;
  container: HTMLElement;
  /** 2D density overlay canvas (covers the plot area only), if active */
  densityCanvas?: HTMLCanvasElement | null;
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
  ctx.fillStyle = `rgb(${exportBackground(isDark).join(', ')})`;
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

/** Read the default framebuffer's raw (premultiplied, bottom-up) pixels; call in the task that drew them. */
function readDrawingBuffer(gl: WebGLRenderingContext): { pixels: Uint8Array; width: number; height: number } {
  const width = gl.drawingBufferWidth;
  const height = gl.drawingBufferHeight;
  const pixels = new Uint8Array(width * height * 4);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
  return { pixels, width, height };
}

/**
 * Capture the 2D scatter plot. scattergl axes/gridlines/ticks are SVG layers,
 * so the plot is re-rendered via Plotly.toImage (which composites SVG + GL),
 * then the label overlay is drawn on top.
 */
export async function capture2DPlot(opts: Capture2DOptions): Promise<HTMLCanvasElement> {
  const { gd, plotlyLib, container, densityCanvas, labelCanvas, isDark } = opts;
  // Snapshot GL-backed pixels synchronously, before any await, and record the
  // overlay's on-screen rect (it covers the plot area only, not the container).
  const densitySnapshot = densityCanvas ? snapshotCanvas(densityCanvas) : null;
  const containerRect = container.getBoundingClientRect();
  const densityRect = densityCanvas?.getBoundingClientRect() ?? null;

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
  if (densitySnapshot && densityRect) {
    ctx.drawImage(
      densitySnapshot,
      (densityRect.left - containerRect.left) * dpr,
      (densityRect.top - containerRect.top) * dpr,
      densityRect.width * dpr,
      densityRect.height * dpr,
    );
  }
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
 * `glplot.redraw()` renders synchronously (also firing the `onrender` hook
 * that repaints the haze canvas) and the drawing buffer is read in the same
 * task — this sidesteps both `preserveDrawingBuffer: false` and the
 * stale-camera problem (fly-to manipulates `glplot.camera` imperatively, so
 * the layout's camera can lag the live view). The raw pixels are flattened
 * onto the export background the way the browser composites the canvas,
 * not through Plotly's `scene.toImage`, whose straight-alpha conversion
 * clamps away most of the search glow (see PremultipliedPixels). The haze
 * canvas is snapshotted in the same task, before the browser composites and
 * clears its GL buffer.
 */
// async on purpose: the not-ready throw and GL errors must reject into the caller's .catch (ScatterPlot3D's screenshot handler)
export async function capture3DPlot(opts: Capture3DOptions): Promise<HTMLCanvasElement> {
  const { gd, container, hazeCanvas, labelCanvas, isDark } = opts;
  const glplot = gd?._fullLayout?.scene?._scene?.glplot;
  const gl = (glplot?.gl as WebGLRenderingContext | undefined) ?? null;
  if (!glplot || !gl) throw new Error('3D scene is not ready for capture');
  const glCanvas = gl.canvas as HTMLCanvasElement;

  // --- Synchronous phase: read all GL-backed pixels in the task that draws them ---
  glplot.redraw();
  const glPixels = readDrawingBuffer(gl);
  // The redraw fired onrender → the haze canvas was just repainted
  const hazeSnapshot = hazeCanvas ? snapshotCanvas(hazeCanvas) : null;
  const containerRect = container.getBoundingClientRect();
  const glRect = glCanvas.getBoundingClientRect();
  const hazeRect = hazeCanvas?.getBoundingClientRect() ?? null;

  // --- Compositing ---
  const { canvas, ctx, dpr } = createOutputCanvas(container, isDark);
  if (glPixels.width > 0 && glPixels.height > 0) {
    const flat = PremultipliedPixels.flattenOnto(glPixels.pixels, glPixels.width, glPixels.height, exportBackground(isDark));
    const glImage = document.createElement('canvas');
    glImage.width = glPixels.width;
    glImage.height = glPixels.height;
    glImage.getContext('2d')?.putImageData(new ImageData(flat, glPixels.width, glPixels.height), 0, 0);
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
