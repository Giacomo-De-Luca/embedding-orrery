/**
 * Fields of the field-association view: which metadata fields can be
 * compared, how each becomes a table axis (categorical values as they are,
 * numeric values binned into ordered ranges), and which statistic measures
 * the relationship of a pair on the raw values.
 */

import type { MetadataPoint } from '../types/types';
import { CategoryKey } from './categoryKey';
import { getUnclusteredValues, isTopicLikeField } from './categoryColors';
import { CategoricalFields, type ColorFieldOption } from './fieldAnalysis';
import { FieldCorrelation } from './fieldCorrelation';
import { NumericBins, type BinMethod } from './numericBins';
import type { ContingencyTable } from './contingencyTable';
import { AssociationView } from './associationView';

export type AssociationFieldKind = 'categorical' | 'numeric';

export interface AssociationFieldOption {
  field: string;
  displayName: string;
  /** `numeric`: binned on the axis; `categorical`: one row/column per value. */
  kind: AssociationFieldKind;
  uniqueCount: number;
  /** The values are numbers: shown as categories, they sort by value (-3 before 2 before 10). */
  numericValues?: boolean;
}

/** How numeric bins are labelled: their value ranges or their percentile bands. */
export type BinLabelStyle = 'range' | 'percentile';

export interface BinSettings {
  count: number;
  method: BinMethod;
  labels: BinLabelStyle;
}

export type AssociationStatisticKind = 'cramersV' | 'eta' | 'spearman';
export type AssociationStrength = 'negligible' | 'weak' | 'moderate' | 'strong';

export interface AssociationStatistic {
  kind: AssociationStatisticKind;
  /** In [0, 1] for V and η, [-1, 1] for ρ; null when undefined (too few values, no variance). */
  value: number | null;
  /** Pearson r alongside Spearman ρ (numeric × numeric only). */
  pearson: number | null;
  /** Points the statistic was computed on. */
  n: number;
}

const toOption = (option: ColorFieldOption, kind: AssociationFieldKind): AssociationFieldOption => ({
  field: option.field,
  displayName: option.displayName,
  kind,
  uniqueCount: option.uniqueCount,
  numericValues: option.valueType === 'numeric',
});

const NO_VALUES: ReadonlySet<string> = new Set();

/** Numeric fields, except topic ids (numbers that are really labels). */
const isNumericOption = (option: ColorFieldOption) =>
  option.valueType === 'numeric' && !isTopicLikeField(option.field);

/**
 * The Group and Across fields. Defaults group by the colour field (never a
 * topic or numeric field) and compare across topics, so a fresh view answers
 * "which topics does each source cover?".
 */
export class AssociationFields {
  static readonly PREFERRED_ACROSS = 'topic_label';
  /**
   * Most values on a heatmap axis (smaller ones fold into "Other"). A numeric
   * field with no more distinct values than this is shown as its values by default.
   */
  static readonly MAX_AXIS_VALUES = 50;
  /** Fields the backend's `topicSimilarityOrder` can order (see `topic_order_service.py`). */
  private static readonly SIMILARITY_FIELDS: ReadonlySet<string> = new Set(CategoryKey.TOPIC_FIELDS);

  /** Whether the heatmap can offer a Similarity sort for `field`. */
  static supportsSimilarity(field: string | null | undefined): boolean {
    return !!field && AssociationFields.SIMILARITY_FIELDS.has(field);
  }

  /** Cross-tab fields: categorical ones, then continuous numeric ones (binned on the axis). */
  static options(colorFieldOptions: readonly ColorFieldOption[] | undefined): AssociationFieldOption[] {
    const categorical = CategoricalFields.options(colorFieldOptions).map(o => toOption(o, 'categorical'));
    const numeric = (colorFieldOptions ?? [])
      .filter(o => isNumericOption(o) && o.recommendedScale !== 'categorical')
      .map(o => toOption(o, 'numeric'));
    return [...categorical, ...numeric];
  }

  /**
   * Whether a numeric field is shown as its individual values (a category per
   * value) rather than binned: the user's choice in `overrides`, otherwise
   * yes when it has few enough distinct values. False for categorical fields.
   */
  static showsValues(option: AssociationFieldOption, overrides: Readonly<Record<string, boolean>>): boolean {
    if (option.kind !== 'numeric') return false;
    return overrides[option.field] ?? option.uniqueCount <= AssociationFields.MAX_AXIS_VALUES;
  }

  /** The options with every numeric field shown as values turned categorical; others unchanged (same objects). */
  static effective(
    options: readonly AssociationFieldOption[],
    overrides: Readonly<Record<string, boolean>>,
  ): AssociationFieldOption[] {
    return options.map(o =>
      AssociationFields.showsValues(o, overrides) ? { ...o, kind: 'categorical', numericValues: true } : o,
    );
  }

  /** Correlation-matrix fields: every numeric field, discrete ones (e.g. 0/1 flags) included. */
  static numericOptions(colorFieldOptions: readonly ColorFieldOption[] | undefined): AssociationFieldOption[] {
    return (colorFieldOptions ?? []).filter(isNumericOption).map(o => toOption(o, 'numeric'));
  }

  /** The picks when still available and distinct, otherwise the defaults. */
  static resolve(
    group: string | null,
    across: string | null,
    options: readonly AssociationFieldOption[],
    colorByField: string | null | undefined,
  ): { group: AssociationFieldOption | null; across: AssociationFieldOption | null } {
    const find = (field: string | null) => (field ? options.find(o => o.field === field) ?? null : null);
    const resolvedGroup = find(group) ?? AssociationFields.defaultGroup(options, colorByField);
    const pickedAcross = find(across);
    return {
      group: resolvedGroup,
      across: pickedAcross && pickedAcross !== resolvedGroup
        ? pickedAcross
        : AssociationFields.defaultAcross(options, resolvedGroup),
    };
  }

  private static defaultGroup(
    options: readonly AssociationFieldOption[],
    colorByField: string | null | undefined,
  ): AssociationFieldOption | null {
    const categorical = options.filter(o => o.kind === 'categorical');
    const candidates = categorical.filter(o => !isTopicLikeField(o.field));
    return candidates.find(o => o.field === colorByField) ?? candidates[0] ?? categorical[0] ?? options[0] ?? null;
  }

  private static defaultAcross(
    options: readonly AssociationFieldOption[],
    group: AssociationFieldOption | null,
  ): AssociationFieldOption | null {
    const others = options.filter(o => o !== group);
    return others.find(o => o.field === AssociationFields.PREFERRED_ACROSS)
      ?? others.find(o => o.kind === 'categorical')
      ?? others[0]
      ?? null;
  }
}

/** One field as a table axis: how a raw value becomes an axis key. */
export class FieldAxis {
  private constructor(
    readonly field: string,
    readonly kind: AssociationFieldKind,
    readonly bins: NumericBins | null,
    /** Value order of a categorical axis whose values are numbers. */
    private readonly numericOrder: readonly string[] | null = null,
  ) {}

  /**
   * A numeric field is binned over all `points`, so filtering never moves the
   * bin edges; a categorical field with numeric values keeps their value order.
   */
  static of(
    option: AssociationFieldOption,
    points: readonly MetadataPoint[],
    bins: Pick<BinSettings, 'count' | 'method'>,
  ): FieldAxis {
    if (option.kind === 'categorical') {
      const order = option.numericValues ? FieldAxis.numericKeyOrder(points, option.field) : null;
      return new FieldAxis(option.field, 'categorical', null, order);
    }
    const values: number[] = [];
    for (const point of points) {
      const value = NumericBins.numericValue(point.metadata?.[option.field]);
      if (value !== null) values.push(value);
    }
    return new FieldAxis(option.field, 'numeric', NumericBins.fromValues(values, bins.count, bins.method));
  }

  /** Axis key of a raw value: its category, or its bin's range label. */
  keyOf = (raw: unknown): string => (this.bins ? this.bins.keyOf(raw) : CategoryKey.of(raw));

  /** Display label of an axis key: a bin's range or percentile band, otherwise the usual category label. */
  labelOf(key: string, style: BinLabelStyle): string {
    const percentile = style === 'percentile' ? this.bins?.percentileLabel(key) : undefined;
    return percentile ?? AssociationView.axisLabel(this.field, key);
  }

  /** Keys low to high: a numeric axis's bins, or a numeric-valued categorical axis's values; otherwise null. */
  get valueOrder(): readonly string[] | null {
    return this.bins?.keys ?? this.numericOrder;
  }

  /** Distinct keys of a field's numeric values over all points, by value. */
  private static numericKeyOrder(points: readonly MetadataPoint[], field: string): string[] {
    const valueOf = new Map<string, number>();
    for (const point of points) {
      const raw = point.metadata?.[field];
      const value = NumericBins.numericValue(raw);
      if (value !== null) valueOf.set(CategoryKey.of(raw), value);
    }
    return [...valueOf.keys()].sort((a, b) => valueOf.get(a)! - valueOf.get(b)!);
  }

  /** Values left off the axis (unclustered topic noise). */
  get excluded(): ReadonlySet<string> {
    return this.bins ? NO_VALUES : getUnclusteredValues(this.field);
  }
}

/** The statistic of a field pair: Cramér's V, η, or Spearman ρ (with Pearson r). */
export class AssociationStatistics {
  static of(
    points: readonly MetadataPoint[],
    group: FieldAxis,
    across: FieldAxis,
    table: ContingencyTable,
    exclude?: ReadonlySet<number> | null,
  ): AssociationStatistic {
    if (group.kind === 'categorical' && across.kind === 'categorical') {
      return { kind: 'cramersV', value: table.cramersV(), pearson: null, n: table.total };
    }
    if (group.kind === 'numeric' && across.kind === 'numeric') {
      return AssociationStatistics.numericPair(points, group.field, across.field, exclude);
    }

    // η over the categories as the table folds them (the heatmap's rows or columns), so it
    // describes what is shown and the group count stays bounded.
    const groupIsCategorical = group.kind === 'categorical';
    const [categorical, numeric] = groupIsCategorical ? [group, across] : [across, group];
    const foldedKey = (key: string) => (groupIsCategorical ? table.rowKeyFor(key) : table.colKeyFor(key));
    const groups: string[] = [];
    const values: number[] = [];
    for (const point of AssociationStatistics.visible(points, exclude)) {
      const value = NumericBins.numericValue(point.metadata?.[numeric.field]);
      const key = value === null ? undefined : foldedKey(categorical.keyOf(point.metadata?.[categorical.field]));
      if (value !== null && key !== undefined) {
        groups.push(key);
        values.push(value);
      }
    }
    return { kind: 'eta', value: FieldCorrelation.correlationRatio(groups, values), pearson: null, n: values.length };
  }

  /** Spearman ρ with Pearson r between two numeric fields; independent of any binning. */
  static numericPair(
    points: readonly MetadataPoint[],
    fieldA: string,
    fieldB: string,
    exclude?: ReadonlySet<number> | null,
  ): AssociationStatistic {
    const xs: number[] = [];
    const ys: number[] = [];
    for (const point of AssociationStatistics.visible(points, exclude)) {
      const x = NumericBins.numericValue(point.metadata?.[fieldA]);
      const y = NumericBins.numericValue(point.metadata?.[fieldB]);
      if (x !== null && y !== null) {
        xs.push(x);
        ys.push(y);
      }
    }
    return {
      kind: 'spearman',
      value: FieldCorrelation.spearman(xs, ys),
      pearson: FieldCorrelation.pearson(xs, ys),
      n: xs.length,
    };
  }

  private static visible(points: readonly MetadataPoint[], exclude?: ReadonlySet<number> | null) {
    return exclude?.size ? points.filter(p => !exclude.has(p.index)) : points;
  }

  static label(kind: AssociationStatisticKind): string {
    return kind === 'cramersV' ? "Cramér's V" : kind === 'eta' ? 'η' : 'Spearman ρ';
  }

  /** What the statistic measures, for tooltips. */
  static explanation(kind: AssociationStatisticKind): string {
    switch (kind) {
      case 'cramersV':
        return "Cramér's V (bias-corrected): 0 when the fields are unrelated, 1 when one field's value determines the other's.";
      case 'eta':
        return 'η (correlation ratio, bias-corrected): 0 when the numeric field has the same mean in every category, 1 when the categories fully separate its values.';
      case 'spearman':
        return 'Spearman ρ: -1 to +1, how consistently one field rises (or falls) as the other rises. Pearson r measures the linear part.';
    }
  }

  /** One line, e.g. "η 0.41 · moderate" or "Spearman ρ −0.62 · strong (Pearson r −0.58)"; null when undefined. */
  static summary(statistic: AssociationStatistic): string | null {
    const { kind, value, pearson } = statistic;
    if (value === null) return null;
    const shown = kind === 'spearman' ? FieldCorrelation.format(value) : value.toFixed(2);
    const line = `${AssociationStatistics.label(kind)} ${shown} · ${AssociationStatistics.strength(value)}`;
    return pearson === null ? line : `${line} (Pearson r ${FieldCorrelation.format(pearson)})`;
  }

  /** Conventional reading of a coefficient's magnitude. */
  static strength(value: number | null): AssociationStrength | null {
    if (value === null) return null;
    const magnitude = Math.abs(value);
    if (magnitude < 0.1) return 'negligible';
    if (magnitude < 0.3) return 'weak';
    if (magnitude < 0.5) return 'moderate';
    return 'strong';
  }
}
