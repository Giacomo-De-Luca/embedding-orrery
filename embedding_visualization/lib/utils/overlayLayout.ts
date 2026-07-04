/**
 * Pure geometry helpers for positioning an absolutely-positioned overlay <canvas>
 * (nebula haze / labels) on top of Plotly's WebGL canvas.
 *
 * Plotly's GL canvas can sit at an offset within the plot container (margins,
 * modebar), so both the haze overlay and the label overlay need the GL canvas's
 * offset + CSS size relative to the container. This centralises that math so it
 * isn't duplicated across the two overlay sync sites in ScatterPlot3D, and so it
 * can be unit-tested independently of the DOM.
 */

/** Minimal shape of a DOMRect — lets tests avoid constructing real DOM rects. */
export interface RectLike {
  left: number;
  top: number;
  width: number;
  height: number;
}

export interface OverlayLayout {
  /** GL canvas left offset within the container, CSS px */
  offsetX: number;
  /** GL canvas top offset within the container, CSS px */
  offsetY: number;
  /** GL canvas width, CSS px */
  cssW: number;
  /** GL canvas height, CSS px */
  cssH: number;
}

/**
 * Compute the overlay canvas layout (offset within container + size) from the
 * container and GL canvas bounding rects. All values are CSS pixels.
 */
export function computeOverlayLayout(containerRect: RectLike, glRect: RectLike): OverlayLayout {
  return {
    offsetX: glRect.left - containerRect.left,
    offsetY: glRect.top - containerRect.top,
    cssW: glRect.width,
    cssH: glRect.height,
  };
}

/**
 * True when two layouts are exactly equal in all four fields. Used to skip
 * redundant per-frame DOM writes (style + canvas backing-store) when nothing
 * about the GL canvas position/size has changed.
 */
export function overlayLayoutEqual(a: OverlayLayout | null, b: OverlayLayout | null): boolean {
  if (a === b) return true;
  if (!a || !b) return false;
  return (
    a.offsetX === b.offsetX &&
    a.offsetY === b.offsetY &&
    a.cssW === b.cssW &&
    a.cssH === b.cssH
  );
}
