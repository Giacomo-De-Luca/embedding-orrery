/**
 * Tests for flattening WebGL read-back pixels (premultiplied alpha, bottom-up
 * rows) onto an opaque export background, the way the browser composites the
 * live 3D canvas. Pure array math — no DOM or WebGL required.
 */

import { describe, it, expect } from 'vitest';
import { PremultipliedPixels, type Rgb255 } from '../premultipliedPixels';

const BLACK: Rgb255 = [0, 0, 0];
const WHITE: Rgb255 = [255, 255, 255];

/** One-pixel image helper: returns the flattened RGBA of a single read-back pixel. */
function flattenOne(pixel: [number, number, number, number], background: Rgb255): number[] {
  return Array.from(PremultipliedPixels.flattenOnto(new Uint8Array(pixel), 1, 1, background));
}

describe('PremultipliedPixels.flattenOnto', () => {
  it('flips rows: gl.readPixels returns the bottom row first', () => {
    // 1×2 image, read-back order: bottom pixel (red) then top pixel (blue), both opaque
    const readBack = new Uint8Array([255, 0, 0, 255, 0, 0, 255, 255]);
    const out = PremultipliedPixels.flattenOnto(readBack, 1, 2, BLACK);
    expect(Array.from(out)).toEqual([0, 0, 255, 255, 255, 0, 0, 255]);
  });

  it('keeps every row intact when flipping a wider image', () => {
    // 2×2, bottom row [A, B], top row [C, D] → output top-down [C, D, A, B]
    const A = [10, 0, 0, 255];
    const B = [20, 0, 0, 255];
    const C = [30, 0, 0, 255];
    const D = [40, 0, 0, 255];
    const out = PremultipliedPixels.flattenOnto(new Uint8Array([...A, ...B, ...C, ...D]), 2, 2, BLACK);
    expect(Array.from(out)).toEqual([...C, ...D, ...A, ...B]);
  });

  it('passes opaque pixels through unchanged', () => {
    expect(flattenOne([12, 200, 99, 255], WHITE)).toEqual([12, 200, 99, 255]);
  });

  it('turns fully transparent pixels into the background', () => {
    expect(flattenOne([0, 0, 0, 0], BLACK)).toEqual([0, 0, 0, 255]);
    expect(flattenOne([0, 0, 0, 0], WHITE)).toEqual([255, 255, 255, 255]);
    expect(flattenOne([0, 0, 0, 0], [10, 20, 30])).toEqual([10, 20, 30, 255]);
  });

  it('composites valid premultiplied pixels with source-over', () => {
    // Premultiplied 50% red: (128, 0, 0, 128)
    expect(flattenOne([128, 0, 0, 128], BLACK)).toEqual([128, 0, 0, 255]);
    // Over white: 128 + (255 − 128) = 255 red, 0 + 127 green/blue
    expect(flattenOne([128, 0, 0, 128], WHITE)).toEqual([255, 127, 127, 255]);
  });

  it('keeps the light of super-luminous pixels (RGB above alpha) — the search glow', () => {
    // gl-scatter3d writes colour × opacity without multiplying by the colour's own
    // alpha: the dark outer glow rgba(140, 180, 230, 0.15) at trace opacity 0.15
    // reads back as roughly (21, 27, 35, 6). On screen it shows as (21, 27, 35);
    // un-premultiplying and clamping (Plotly's scene.toImage) leaves about (6, 6, 6).
    expect(flattenOne([21, 27, 35, 6], BLACK)).toEqual([21, 27, 35, 255]);
  });

  it('clamps at 255 like the browser compositor', () => {
    expect(flattenOne([200, 40, 0, 20], WHITE)).toEqual([255, 255, 235, 255]);
  });

  it('always outputs opaque pixels', () => {
    const readBack = new Uint8Array([0, 0, 0, 0, 5, 5, 5, 3, 50, 60, 70, 128, 1, 2, 3, 255]);
    const out = PremultipliedPixels.flattenOnto(readBack, 2, 2, BLACK);
    for (let i = 3; i < out.length; i += 4) expect(out[i]).toBe(255);
  });

  it('rejects a buffer whose length does not match the size', () => {
    expect(() => PremultipliedPixels.flattenOnto(new Uint8Array(8), 3, 1, BLACK)).toThrow();
  });
});
