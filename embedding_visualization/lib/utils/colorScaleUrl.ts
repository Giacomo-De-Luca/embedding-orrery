import type { ColorScale, SequentialScaleName, DivergingScaleName } from '../types/types';

/**
 * Serialization of the color-scheme state (the `ColorScale` discriminated union
 * plus the separate `categoricalPalette`) to / from URL query parameters so a
 * shared link reproduces the exact look of the scatter plot.
 *
 * The union is flattened into compact params:
 *   ?scale=sequential&scaleName=viridis
 *   ?scale=monochrome&color=%2300aaff
 *   ?scale=categorical&palette=tableau10
 *
 * `palette` is emitted independently of `scale` — it carries `categoricalPalette`,
 * which the chart components use regardless of the active scale type.
 */

export interface ColorUrlParams {
  scale?: string;      // ColorScale.type discriminant
  scaleName?: string;  // sequential / diverging variants
  color?: string;      // monochrome baseColor
  palette?: string;    // categoricalPalette (any scale type)
}

/** Flatten a ColorScale + categorical palette into URL param values. */
export function serializeColorScale(
  scale: ColorScale,
  palette: string | undefined,
): ColorUrlParams {
  const out: ColorUrlParams = { scale: scale.type };
  switch (scale.type) {
    case 'sequential':
    case 'diverging':
      out.scaleName = scale.scaleName;
      break;
    case 'monochrome':
      out.color = scale.baseColor;
      break;
    case 'categorical':
      break;
  }
  if (palette) out.palette = palette;
  return out;
}

/**
 * Reconstruct a ColorScale from URL param values. Returns null when the params
 * don't describe a valid scale (so callers can fall back to the recommended
 * scale for the field). The `palette` is read separately by the caller.
 */
export function deserializeColorScale(params: {
  scale: string | null;
  scaleName: string | null;
  color: string | null;
}): ColorScale | null {
  switch (params.scale) {
    case 'categorical':
      return { type: 'categorical' };
    case 'sequential':
      return params.scaleName
        ? { type: 'sequential', scaleName: params.scaleName as SequentialScaleName }
        : null;
    case 'diverging':
      return params.scaleName
        ? { type: 'diverging', scaleName: params.scaleName as DivergingScaleName }
        : null;
    case 'monochrome':
      return params.color ? { type: 'monochrome', baseColor: params.color } : null;
    default:
      return null;
  }
}

/**
 * A colour scheme persisted as a per-collection default in `extra_metadata`.
 * Same flattened shape as the URL params, plus the field to colour by. Stored as
 * a JSON string under `default_color_scheme` (matching the `field_analysis`
 * convention) and applied on collection load when no URL `colorBy` is present.
 */
export interface DefaultColorScheme extends ColorUrlParams {
  colorBy: string;
}

/** Build a persisted default from the live colour state (field + scale + palette). */
export function serializeDefaultColorScheme(
  colorBy: string,
  scale: ColorScale,
  palette: string | undefined,
): DefaultColorScheme {
  return { colorBy, ...serializeColorScale(scale, palette) };
}

/**
 * Resolve a persisted default into the pieces needed to apply it. Returns null
 * when there is no field to colour by. `scale` is null when the stored scale is
 * absent/invalid, letting the caller fall back to the field's recommended scale.
 */
export function resolveDefaultColorScheme(
  scheme: DefaultColorScheme | null | undefined,
): { field: string; scale: ColorScale | null; palette: string | null } | null {
  if (!scheme?.colorBy) return null;
  return {
    field: scheme.colorBy,
    scale: deserializeColorScale({
      scale: scheme.scale ?? null,
      scaleName: scheme.scaleName ?? null,
      color: scheme.color ?? null,
    }),
    palette: scheme.palette ?? null,
  };
}
