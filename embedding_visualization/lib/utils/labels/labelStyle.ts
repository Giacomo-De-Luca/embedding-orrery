import { desaturateHex } from '../categoryColors';

/** What label code needs from a canvas context to measure text (a 2D context has it). */
export interface TextMeasurer {
  font: string;
  measureText(text: string): { width: number };
}

/** The parts of `document.fonts` the label fonts use. */
export interface FontLoader {
  check(font: string): boolean;
  load(font: string): Promise<unknown>;
}

/**
 * Canvas label fonts. next/font registers Geist Mono under the family name
 * `GeistMono` and publishes its stack as `--font-geist-mono` on `<html>`; canvas
 * text must use that registered name ("Geist Mono" silently falls back to the
 * system monospace).
 */
export class LabelFonts {
  static readonly FALLBACK_FAMILY = 'GeistMono, ui-monospace, monospace';
  static readonly CLUSTER_SIZE = 13;
  static readonly POINT_SIZE = 11;

  private static cachedFamily: string | null = null;

  /** The font-family list for label text. */
  static family(): string {
    if (LabelFonts.cachedFamily) return LabelFonts.cachedFamily;
    if (typeof document === 'undefined') return LabelFonts.FALLBACK_FAMILY;
    const value = getComputedStyle(document.documentElement).getPropertyValue('--font-geist-mono');
    LabelFonts.cachedFamily = LabelFonts.resolveFamily(value);
    return LabelFonts.cachedFamily;
  }

  /** The family list from the CSS variable's value, or the fallback when it is unset. */
  static resolveFamily(value: string | null | undefined): string {
    return value?.trim() || LabelFonts.FALLBACK_FAMILY;
  }

  static cluster(size = LabelFonts.CLUSTER_SIZE): string {
    return `bold ${size}px ${LabelFonts.family()}`;
  }

  static point(size = LabelFonts.POINT_SIZE): string {
    return `${size}px ${LabelFonts.family()}`;
  }

  /** Whether every label font can be drawn now (true outside the browser). */
  static isReady(fonts: FontLoader | undefined = LabelFonts.documentFonts()): boolean {
    return !fonts || LabelFonts.all().every(font => fonts.check(font));
  }

  /**
   * Resolves once every label font has loaded or failed to load. Text measured
   * before then used a fallback font, so its widths are wrong.
   */
  static async whenReady(fonts: FontLoader | undefined = LabelFonts.documentFonts()): Promise<void> {
    if (!fonts) return;
    await Promise.all(LabelFonts.all().map(font => fonts.load(font).catch(() => undefined)));
  }

  /** A new offscreen context for measuring label text, or null outside the browser. */
  static createMeasurer(): TextMeasurer | null {
    return typeof document === 'undefined' ? null : document.createElement('canvas').getContext('2d');
  }

  private static all(): string[] {
    return [LabelFonts.cluster(), LabelFonts.point()];
  }

  private static documentFonts(): FontLoader | undefined {
    return typeof document === 'undefined' ? undefined : document.fonts;
  }
}

export type LabelView = '2d' | '3d';

/** A text outline, drawn under the fill for contrast against points and haze. */
export interface LabelStroke {
  color: string;
  width: number;
}

/**
 * How much of a cluster colour's saturation the label text gives up, per view
 * and theme (`desaturateHex`). 2D and 3D differ on purpose until one pair is chosen.
 */
const CLUSTER_DESATURATION: Record<LabelView, { dark: number; light: number }> = {
  '2d': { dark: 0.3, light: 0.45 },
  '3d': { dark: 0.1, light: 0.35 },
};

/** Colours and outlines of canvas labels for one view and theme. */
export class LabelStyle {
  readonly clusterStroke: LabelStroke;
  readonly pointStroke: LabelStroke;
  readonly pointFill: string;

  constructor(readonly isDark: boolean, readonly view: LabelView) {
    this.clusterStroke = isDark
      ? { color: 'rgba(15, 23, 42, 0.85)', width: 4 }
      : { color: 'rgba(255, 255, 255, 0.6)', width: 3 };
    this.pointStroke = { color: isDark ? 'rgba(15, 23, 42, 0.8)' : 'rgba(255, 255, 255, 0.8)', width: 3 };
    this.pointFill = isDark ? '#e2e8f0' : '#1e293b';
  }

  /** Cluster-label text colour for a cluster's colour. */
  clusterFill(hex: string): string {
    const amount = CLUSTER_DESATURATION[this.view];
    return desaturateHex(hex, this.isDark ? amount.dark : amount.light, this.isDark);
  }

  /** Search-label opacity: the selected point's label is opaque, the rest fade with similarity. */
  static pointAlpha(similarity: number, selected: boolean): number {
    return selected ? 1 : 0.4 + 0.6 * similarity;
  }
}
