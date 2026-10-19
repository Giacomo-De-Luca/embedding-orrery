/**
 * Perceptual colour math shared by the palette tooling and tests.
 *
 * Everything works on `#rrggbb` strings. Distances are Euclidean in OKLab
 * (×100, so ~2 is a just-noticeable difference and ≥15 is comfortably
 * distinct); colour-vision deficiency is simulated with the
 * Machado–Oliveira–Fernandes (2009) severity-1.0 matrices in linear sRGB,
 * the same model the dataviz palette validator uses.
 */

import { hexToRgb, rgbToHex } from './plotUtils';

export type Rgb = [number, number, number];
/** OKLab triple: lightness 0–1, then the two opponent axes. */
export type Oklab = [number, number, number];
/** OKLCH triple: lightness 0–1, chroma (≈0–0.37 in sRGB), hue in degrees. */
export type Oklch = [number, number, number];
export type CvdKind = 'protan' | 'deutan' | 'tritan';


const MACHADO: Record<CvdKind, number[][]> = {
  protan: [[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]],
  deutan: [[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.01182, 0.04294, 0.968881]],
  tritan: [[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602], [0.004733, 0.691367, 0.3039]],
};

const clamp01 = (c: number) => Math.max(0, Math.min(1, c));
const srgbToLinear = (c: number) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
const linearToSrgb = (c: number) => { const v = clamp01(c); return v <= 0.0031308 ? 12.92 * v : 1.055 * v ** (1 / 2.4) - 0.055; };

export class ColorMath {
  /** `#rrggbb` (any case) → sRGB 0–1. Throws on any other format, like `hexToRgb`. */
  static hexToSrgb(hex: string): Rgb {
    const { r, g, b } = hexToRgb(hex);
    return [r / 255, g / 255, b / 255];
  }

  /** sRGB 0–1 (clamped) → lower-case `#rrggbb`. */
  static srgbToHex(rgb: Rgb): string {
    const [r, g, b] = rgb.map((c) => Math.round(clamp01(c) * 255));
    return rgbToHex(r, g, b);
  }

  static hexToLinear(hex: string): Rgb {
    return ColorMath.hexToSrgb(hex).map(srgbToLinear) as Rgb;
  }

  static oklabFromLinear([r, g, b]: Rgb): Oklab {
    const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
    const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
    const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
    return [
      0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s,
      1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s,
      0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s,
    ];
  }

  static linearFromOklab([L, a, b]: Oklab): Rgb {
    const l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3;
    const m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3;
    const s = (L - 0.0894841775 * a - 1.291485548 * b) ** 3;
    return [
      4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
      -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
      -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s,
    ];
  }

  static oklab(hex: string): Oklab {
    return ColorMath.oklabFromLinear(ColorMath.hexToLinear(hex));
  }

  static oklch(hex: string): Oklch {
    const [L, a, b] = ColorMath.oklab(hex);
    return [L, Math.hypot(a, b), (((Math.atan2(b, a) * 180) / Math.PI) % 360 + 360) % 360];
  }

  /**
   * OKLCH → hex, or `null` when the colour falls outside the sRGB gamut by more
   * than `tolerance` (linear units). The default absorbs round-trip error:
   * exact white converts back to ~1e-16 outside the gamut. Pass 0 for gamut searches.
   */
  static oklchToHex(L: number, C: number, hueDeg: number, tolerance = 0.002): string | null {
    const h = (hueDeg * Math.PI) / 180;
    const rgb = ColorMath.linearFromOklab([L, C * Math.cos(h), C * Math.sin(h)]);
    if (rgb.some((c) => c < -tolerance || c > 1 + tolerance)) return null;
    return ColorMath.srgbToHex(rgb.map(linearToSrgb) as Rgb);
  }

  /** Largest in-gamut chroma at this lightness and hue (bisection, 24 steps). */
  static maxChroma(L: number, hueDeg: number): number {
    let lo = 0, hi = 0.4;
    for (let i = 0; i < 24; i++) {
      const mid = (lo + hi) / 2;
      if (ColorMath.oklchToHex(L, mid, hueDeg, 0)) lo = mid; else hi = mid;
    }
    return lo;
  }

  static relativeLuminance(hex: string): number {
    const [r, g, b] = ColorMath.hexToLinear(hex);
    return 0.2126 * r + 0.7152 * g + 0.0722 * b;
  }

  /** WCAG 2 contrast ratio (1–21). */
  static contrast(a: string, b: string): number {
    const [hi, lo] = [ColorMath.relativeLuminance(a), ColorMath.relativeLuminance(b)].sort((x, y) => y - x);
    return (hi + 0.05) / (lo + 0.05);
  }

  /** Linear sRGB of `hex` as seen with the given dichromacy. */
  static simulateCvd(hex: string, kind: CvdKind): Rgb {
    const [r, g, b] = ColorMath.hexToLinear(hex);
    const M = MACHADO[kind];
    return [0, 1, 2].map((i) => clamp01(M[i][0] * r + M[i][1] * g + M[i][2] * b)) as Rgb;
  }

  /** OKLab distance ×100 between two colours, optionally under a simulated deficiency. */
  static deltaE(a: string, b: string, kind?: CvdKind): number {
    const la = ColorMath.oklabFromLinear(kind ? ColorMath.simulateCvd(a, kind) : ColorMath.hexToLinear(a));
    const lb = ColorMath.oklabFromLinear(kind ? ColorMath.simulateCvd(b, kind) : ColorMath.hexToLinear(b));
    return 100 * Math.hypot(la[0] - lb[0], la[1] - lb[1], la[2] - lb[2]);
  }

  /**
   * Separation used when designing palettes: the plain distance, capped by the
   * protan and deutan distances scaled ×1.6 so that a pair which collapses for
   * red–green deficient readers counts as close even if it looks distinct otherwise.
   */
  static separation(a: string, b: string): number {
    return Math.min(ColorMath.deltaE(a, b), 1.6 * ColorMath.deltaE(a, b, 'protan'), 1.6 * ColorMath.deltaE(a, b, 'deutan'));
  }
}
