/**
 * View model for the field-association heatmap: what each cell encodes in the
 * selected mode, its colour, its compact label, and the summaries shown in the
 * Analytics sidebar.
 *
 * Colours follow the dataviz reference palette: one blue ramp for magnitude
 * (count, share) and a blue / red pair around a neutral gray for enrichment,
 * each with its own steps for the dark theme rather than an inverted copy.
 */

import { interpolateLab, piecewise } from 'd3-interpolate';
import { lab, rgb } from 'd3-color';
import { ContingencyTable } from './contingencyTable';
import { AxisOrdering } from './axisOrdering';
import { CategoryKey } from './categoryKey';
import { getCategoryLabel } from './categoryColors';
import type { HeatmapModel } from './heatmapModel';

/**
 * What a cell encodes: the Pearson residual, the raw count, or the share of
 * the cell within its row or within its column.
 */
export type AssociationMode = 'enrichment' | 'count' | 'rowShare' | 'colShare';

export interface ThemeColors {
  /** Magnitude ramp, from near zero to the maximum. */
  sequential: readonly string[];
  /** Diverging midpoint ("as expected"), also the fill of empty cells. */
  neutral: string;
  /** Pole for fewer points than expected. */
  under: string;
  /** Pole for more points than expected. */
  over: string;
}

export const ASSOCIATION_COLORS: { light: ThemeColors; dark: ThemeColors } = {
  light: {
    sequential: ['#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'],
    neutral: '#f0efec',
    under: '#2a78d6',
    over: '#e34948',
  },
  dark: {
    sequential: ['#0d366b', '#1c5cab', '#2a78d6', '#6da7ec', '#b7d3f6'],
    neutral: '#383835',
    under: '#3987e5',
    over: '#e66767',
  },
};

export interface SpreadRow {
  key: string;
  label: string;
  /** Points in the row. */
  count: number;
  /** Effective number of columns the row spreads over (see `ContingencyTable.effectiveCount`). */
  effective: number;
}

/** Display label of a row or column key (defaults to {@link AssociationView.axisLabel}). */
export interface AxisLabels {
  row: (key: string) => string;
  col: (key: string) => string;
}

export interface AssociationLegend {
  /** CSS gradient from the low end of the scale to the high end. */
  gradient: string;
  start: string;
  middle: string | null;
  end: string;
}

const compactNumber = new Intl.NumberFormat('en', { notation: 'compact', maximumFractionDigits: 1 });

/** Fill lightness (CIELAB L) above which a cell label switches to dark ink. */
const DARK_INK_MIN_LIGHTNESS = 60;

const clamp = (value: number, low: number, high: number) => Math.min(high, Math.max(low, value));

/** Colour scale of one view: a magnitude ramp, or a diverging scale symmetric around 0. */
export class AssociationScale {
  private readonly ramp: (t: number) => string;
  private readonly underRamp: (t: number) => string;
  private readonly overRamp: (t: number) => string;

  constructor(
    readonly kind: 'sequential' | 'diverging',
    /** Value mapped to the end of the ramp (sequential) or to either pole (diverging). */
    readonly limit: number,
    private readonly colors: ThemeColors,
  ) {
    this.ramp = AssociationScale.sampler(colors.sequential);
    this.underRamp = AssociationScale.sampler([colors.neutral, colors.under]);
    this.overRamp = AssociationScale.sampler([colors.neutral, colors.over]);
  }

  color(value: number): string {
    const t = this.limit > 0 ? value / this.limit : 0;
    if (this.kind === 'sequential') return this.ramp(clamp(t, 0, 1));
    return t < 0 ? this.underRamp(clamp(-t, 0, 1)) : this.overRamp(clamp(t, 0, 1));
  }

  /** CSS linear gradient across the scale's range, for the legend strip. */
  gradient(steps = 11): string {
    const low = this.kind === 'sequential' ? 0 : -this.limit;
    const stops = Array.from({ length: steps }, (_, k) => {
      const value = low + ((this.limit - low) * k) / (steps - 1);
      return `${this.color(value)} ${((100 * k) / (steps - 1)).toFixed(0)}%`;
    });
    return `linear-gradient(to right, ${stops.join(', ')})`;
  }

  /** Whether text drawn on `fill` needs light ink to stay readable. */
  static needsLightInk(fill: string): boolean {
    return lab(fill).l < DARK_INK_MIN_LIGHTNESS;
  }

  /** Lab interpolation through `stops`, returning the stops themselves exactly at the ends. */
  private static sampler(stops: readonly string[]): (t: number) => string {
    const interpolate = piecewise(interpolateLab, [...stops]);
    const first = rgb(stops[0]).toString();
    const last = rgb(stops[stops.length - 1]).toString();
    return t => (t <= 0 ? first : t >= 1 ? last : rgb(interpolate(t)).toString());
  }
}

export class AssociationView implements HeatmapModel {
  /** Smallest enrichment limit: residuals within ±2 are what chance alone produces. */
  static readonly ENRICHMENT_FLOOR = 2;
  /** The enrichment limit is this quantile of |residual|, so one extreme cell can't wash out the rest. */
  private static readonly ENRICHMENT_QUANTILE = 0.95;

  /** Fill of cells with no points in the magnitude modes (the neutral colour). */
  readonly emptyColor: string;
  readonly rowLabels: string[];
  readonly colLabels: string[];

  private constructor(
    readonly table: ContingencyTable,
    readonly mode: AssociationMode,
    readonly scale: AssociationScale,
    colors: ThemeColors,
    labels: AxisLabels,
  ) {
    this.emptyColor = rgb(colors.neutral).toString();
    this.rowLabels = table.rows.map(labels.row);
    this.colLabels = table.cols.map(labels.col);
  }

  static of(
    table: ContingencyTable,
    mode: AssociationMode,
    isDark: boolean,
    labels: AxisLabels = {
      row: key => AssociationView.axisLabel(table.rowField, key),
      col: key => AssociationView.axisLabel(table.colField, key),
    },
  ): AssociationView {
    const colors = isDark ? ASSOCIATION_COLORS.dark : ASSOCIATION_COLORS.light;
    const values: number[] = [];
    for (let i = 0; i < table.rows.length; i++) {
      for (let j = 0; j < table.cols.length; j++) values.push(AssociationView.valueOf(table, mode, i, j));
    }

    if (mode === 'enrichment') {
      const magnitudes = values.map(Math.abs).sort((a, b) => a - b);
      const rank = Math.ceil(AssociationView.ENRICHMENT_QUANTILE * magnitudes.length) - 1;
      const limit = Math.max(AssociationView.ENRICHMENT_FLOOR, magnitudes[rank] ?? 0);
      return new AssociationView(table, mode, new AssociationScale('diverging', limit, colors), colors, labels);
    }
    const limit = values.reduce((max, v) => Math.max(max, v), 0);
    return new AssociationView(table, mode, new AssociationScale('sequential', limit, colors), colors, labels);
  }

  private static valueOf(table: ContingencyTable, mode: AssociationMode, i: number, j: number): number {
    switch (mode) {
      case 'enrichment':
        return table.residual(i, j);
      case 'rowShare':
        return table.rowShare(i, j);
      case 'colShare':
        return table.colShare(i, j);
      case 'count':
        return table.counts[i][j];
    }
  }

  /** The value cell (i, j) encodes in this mode. */
  value(i: number, j: number): number {
    return AssociationView.valueOf(this.table, this.mode, i, j);
  }

  isEmpty(i: number, j: number): boolean {
    return this.table.counts[i][j] === 0;
  }

  /** Fill colour. Empty cells take the neutral colour in magnitude modes; enrichment colours them too (under-representation). */
  color(i: number, j: number): string {
    if (this.mode !== 'enrichment' && this.isEmpty(i, j)) return this.emptyColor;
    return this.scale.color(this.value(i, j));
  }

  /** Whether a label drawn on the cell needs light ink to stay readable. */
  usesLightInk(i: number, j: number): boolean {
    return AssociationScale.needsLightInk(this.color(i, j));
  }

  /** Short text for the cell: count, share, or observed/expected ratio. */
  cellLabel(i: number, j: number): string {
    switch (this.mode) {
      case 'enrichment':
        return AssociationView.formatRatio(this.table.ratio(i, j));
      case 'rowShare':
      case 'colShare':
        return AssociationView.formatShare(this.value(i, j));
      case 'count':
        return compactNumber.format(this.table.counts[i][j]);
    }
  }

  legend(): AssociationLegend {
    const gradient = this.scale.gradient();
    if (this.mode === 'enrichment') {
      return { gradient, start: 'Fewer than expected', middle: 'As expected', end: 'More than expected' };
    }
    const end = this.mode === 'count'
      ? compactNumber.format(this.scale.limit)
      : AssociationView.formatShare(this.scale.limit);
    return { gradient, start: this.mode === 'count' ? '0' : '0%', middle: null, end };
  }

  static formatShare(share: number): string {
    if (share <= 0) return '0%';
    const pct = share * 100;
    return pct < 0.5 ? '<1%' : `${Math.round(pct)}%`;
  }

  static formatRatio(ratio: number): string {
    if (ratio <= 0) return '0×';
    return ratio >= 10 ? `${Math.round(ratio)}×` : `${ratio.toFixed(1)}×`;
  }

  /** Display label of an axis key: Other, Missing values, or the field preset's label. */
  static axisLabel(field: string, key: string): string {
    if (key === ContingencyTable.OTHER) return ContingencyTable.OTHER_LABEL;
    if (key === CategoryKey.MISSING) return CategoryKey.MISSING_LABEL;
    return getCategoryLabel(field, key);
  }

  /** Rows by how few columns they spread over (most concentrated first), Other and Missing last. */
  static spreadRows(
    table: ContingencyTable,
    labelOf: (key: string) => string = key => AssociationView.axisLabel(table.rowField, key),
  ): SpreadRow[] {
    const effective = table.rows.map((_, i) => table.effectiveCount(i));
    const order = AxisOrdering.withPinnedLast(
      table.rows,
      (a, b) => effective[a] - effective[b] || table.rowTotals[b] - table.rowTotals[a],
    );
    return order.map(i => ({
      key: table.rows[i],
      label: labelOf(table.rows[i]),
      count: table.rowTotals[i],
      effective: effective[i],
    }));
  }
}
