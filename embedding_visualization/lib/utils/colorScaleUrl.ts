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
