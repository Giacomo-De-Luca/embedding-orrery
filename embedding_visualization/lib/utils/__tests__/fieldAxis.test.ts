import { describe, it, expect } from 'vitest';
import {
  AssociationFields,
  AssociationStatistics,
  FieldAxis,
  type AssociationFieldOption,
} from '../fieldAxis';
import { ContingencyTable } from '../contingencyTable';
import { FieldCorrelation } from '../fieldCorrelation';
import { CategoryKey } from '../categoryKey';
import { getUnclusteredValues } from '../categoryColors';
import type { ColorFieldOption } from '../fieldAnalysis';

const colorOption = (
  field: string,
  valueType: ColorFieldOption['valueType'],
  recommendedScale: ColorFieldOption['recommendedScale'],
): ColorFieldOption => ({ field, displayName: field, valueType, uniqueCount: 5, recommendedScale });

const COLOR_OPTIONS = [
  colorOption('category', 'string', 'categorical'),
  colorOption('source', 'string', 'categorical'),
  colorOption('ironic', 'numeric', 'categorical'),
  colorOption('topic_label', 'string', 'categorical'),
  colorOption('topic_id', 'numeric', 'sequential'),
  colorOption('concreteness', 'numeric', 'sequential'),
];

const fieldsOf = (options: readonly AssociationFieldOption[]) => options.map(o => `${o.field}:${o.kind}`);
const BINS = { count: 2, method: 'quantile' as const };

describe('AssociationFields', () => {
  const options = AssociationFields.options(COLOR_OPTIONS);

  it('offers categorical fields, then continuous numeric fields (never topic ids)', () => {
    expect(fieldsOf(options)).toEqual([
      'category:categorical',
      'source:categorical',
      'ironic:categorical',
      'topic_label:categorical',
      'concreteness:numeric',
    ]);
  });

  it('marks the fields whose values are numbers', () => {
    const numericValues = Object.fromEntries(options.map(o => [o.field, o.numericValues]));
    expect(numericValues).toEqual({ category: false, source: false, ironic: true, topic_label: false, concreteness: true });
  });

  it('shows few-valued numeric fields as their values and many-valued ones as bins, unless overridden', () => {
    const fewValues = { ...colorOption('century', 'numeric', 'sequential'), uniqueCount: 30 };
    const manyValues = { ...colorOption('score', 'numeric', 'sequential'), uniqueCount: 800 };
    const base = AssociationFields.options([colorOption('source', 'string', 'categorical'), fewValues, manyValues]);
    const kinds = (overrides: Record<string, boolean>) =>
      AssociationFields.effective(base, overrides).map(o => `${o.field}:${o.kind}`);
    expect(kinds({})).toEqual(['source:categorical', 'century:categorical', 'score:numeric']);
    expect(kinds({ century: false, score: true })).toEqual(['source:categorical', 'century:numeric', 'score:categorical']);
    expect(AssociationFields.effective(base, {})[0]).toBe(base[0]);
    expect(AssociationFields.showsValues(base[1], {})).toBe(true);
    expect(AssociationFields.showsValues(base[2], {})).toBe(false);
  });

  it('offers every numeric field except topic ids for correlations', () => {
    expect(AssociationFields.numericOptions(COLOR_OPTIONS).map(o => o.field)).toEqual(['ironic', 'concreteness']);
  });

  it('groups by the colour field and compares across topics by default', () => {
    const { group, across } = AssociationFields.resolve(null, null, options, 'source');
    expect([group?.field, across?.field]).toEqual(['source', 'topic_label']);
  });

  it('never groups by a topic field or a numeric field by default', () => {
    expect(AssociationFields.resolve(null, null, options, 'topic_label').group?.field).toBe('category');
    expect(AssociationFields.resolve(null, null, options, 'concreteness').group?.field).toBe('category');
  });

  it('falls back to another categorical field when there is no topic field', () => {
    const noTopics = options.filter(o => o.field !== 'topic_label');
    expect(AssociationFields.resolve(null, null, noTopics, null).across?.field).toBe('source');
  });

  it('keeps valid picks, numeric ones included, and replaces stale ones', () => {
    expect(AssociationFields.resolve('source', 'concreteness', options, null).across?.field).toBe('concreteness');
    expect(AssociationFields.resolve('gone', 'source', options, null).group?.field).toBe('category');
  });

  it('never returns the same field twice', () => {
    const { group, across } = AssociationFields.resolve('source', 'source', options, null);
    expect([group?.field, across?.field]).toEqual(['source', 'topic_label']);
  });

  it('offers the Similarity sort only on the fields the backend can order', () => {
    expect(['topic_label', 'topic_id', 'subtopic_label', 'subtopic_id'].every(AssociationFields.supportsSimilarity)).toBe(true);
    expect(AssociationFields.supportsSimilarity('topic')).toBe(false);
    expect(AssociationFields.supportsSimilarity('source')).toBe(false);
  });

  it('returns nulls when fewer than two fields exist', () => {
    const only = options.slice(0, 1);
    expect(AssociationFields.resolve(null, null, only, null).across).toBeNull();
    expect(AssociationFields.resolve(null, null, [], null)).toEqual({ group: null, across: null });
  });
});

describe('FieldAxis', () => {
  const points = [1, 2, 3, 4].map(v => ({ index: v, metadata: { conc: v, topic_label: 't' } }));

  it('bins a numeric field over all points and orders its keys by value', () => {
    const axis = FieldAxis.of({ field: 'conc', displayName: 'Conc', kind: 'numeric', uniqueCount: 4 }, points, BINS);
    expect(axis.valueOrder).toHaveLength(2);
    expect(axis.keyOf(1)).toBe(axis.valueOrder![0]);
    expect(axis.keyOf(4)).toBe(axis.valueOrder![1]);
    expect(axis.keyOf(undefined)).toBe(CategoryKey.MISSING);
    expect(axis.excluded.size).toBe(0);
  });

  it('labels numeric bins by range or by percentile, other keys as usual', () => {
    const axis = FieldAxis.of({ field: 'conc', displayName: 'Conc', kind: 'numeric', uniqueCount: 4 }, points, BINS);
    const [first, second] = axis.valueOrder!;
    expect(axis.labelOf(first, 'range')).toBe('1–2');
    expect(axis.labelOf(second, 'percentile')).toBe('50–100%');
    expect(axis.labelOf(CategoryKey.MISSING, 'percentile')).toBe(CategoryKey.MISSING_LABEL);
    const topic = FieldAxis.of({ field: 'topic_label', displayName: 'Topic', kind: 'categorical', uniqueCount: 1 }, points, BINS);
    expect(topic.labelOf('Cats', 'percentile')).toBe('Cats');
  });

  it('orders the values of a numeric field shown as categories by value, not as text', () => {
    const centuries = [-12, 3, -3, 10, 3, 2].map((century, index) => ({ index, metadata: { century } }));
    const axis = FieldAxis.of(
      { field: 'century', displayName: 'Century', kind: 'categorical', uniqueCount: 5, numericValues: true },
      [...centuries, { index: 9, metadata: {} }],
      BINS,
    );
    expect(axis.kind).toBe('categorical');
    expect(axis.valueOrder).toEqual(['-12', '-3', '2', '3', '10']);
    expect(axis.keyOf(10)).toBe('10');
    expect(axis.keyOf(undefined)).toBe(CategoryKey.MISSING);
  });

  it('keeps categorical values as they are and drops topic noise', () => {
    const axis = FieldAxis.of({ field: 'topic_label', displayName: 'Topic', kind: 'categorical', uniqueCount: 1 }, points, BINS);
    expect(axis.valueOrder).toBeNull();
    expect(axis.keyOf('Cats')).toBe('Cats');
    expect(axis.excluded).toBe(getUnclusteredValues('topic_label'));
  });
});

describe('AssociationStatistics', () => {
  const categorical = (field: string): AssociationFieldOption => ({ field, displayName: field, kind: 'categorical', uniqueCount: 2 });
  const numeric = (field: string): AssociationFieldOption => ({ field, displayName: field, kind: 'numeric', uniqueCount: 9 });

  const points = [
    { src: 'a', x: 1, y: 10 },
    { src: 'a', x: 2, y: 8 },
    { src: 'a', x: 3, y: 9 },
    { src: 'b', x: 7, y: 3 },
    { src: 'b', x: 8, y: 1 },
    { src: 'b', x: 9, y: 2 },
  ].map((metadata, index) => ({ index, metadata }));

  function statOf(group: AssociationFieldOption, across: AssociationFieldOption, exclude?: Set<number>) {
    const g = FieldAxis.of(group, points, BINS);
    const a = FieldAxis.of(across, points, BINS);
    const table = ContingencyTable.fromPoints(points, g.field, a.field, { rowKey: g.keyOf, colKey: a.keyOf, exclude });
    return { stat: AssociationStatistics.of(points, g, a, table, exclude), table };
  }

  it("uses Cramér's V when a numeric field is shown as its values", () => {
    const valued: AssociationFieldOption = { field: 'x', displayName: 'x', kind: 'categorical', uniqueCount: 6, numericValues: true };
    const { stat, table } = statOf(categorical('src'), valued);
    expect(stat.kind).toBe('cramersV');
    expect(stat.value).toBe(table.cramersV());
  });

  it("uses Cramér's V between two categorical fields", () => {
    const { stat, table } = statOf(categorical('src'), categorical('x'));
    expect(stat.kind).toBe('cramersV');
    expect(stat.value).toBe(table.cramersV());
  });

  it('uses eta between a categorical and a numeric field, whichever side it is on', () => {
    const left = statOf(categorical('src'), numeric('x')).stat;
    const right = statOf(numeric('x'), categorical('src')).stat;
    expect(left.kind).toBe('eta');
    expect(left.value!).toBeGreaterThan(0.9);
    expect(right.value).toBeCloseTo(left.value!);
  });

  it('computes eta over the categories as the table folds them', () => {
    const many = [
      ...[1, 2, 3].map(x => ({ src: 'a', x })),
      ...[7, 8, 9].map(x => ({ src: 'b', x })),
      { src: 'c', x: 4 },
      { src: 'd', x: 6 },
    ].map((metadata, index) => ({ index, metadata }));
    const g = FieldAxis.of(categorical('src'), many, BINS);
    const a = FieldAxis.of(numeric('x'), many, BINS);
    const table = ContingencyTable.fromPoints(many, 'src', 'x', { rowKey: g.keyOf, colKey: a.keyOf, maxValues: 3 });
    const stat = AssociationStatistics.of(many, g, a, table);
    const folded = ['a', 'a', 'a', 'b', 'b', 'b', 'other', 'other'];
    expect(stat.value).toBeCloseTo(FieldCorrelation.correlationRatio(folded, [1, 2, 3, 7, 8, 9, 4, 6])!);
  });

  it('has no eta when the filters leave a single category', () => {
    expect(statOf(categorical('src'), numeric('x'), new Set([3, 4, 5])).stat.value).toBeNull();
  });

  it('uses Spearman rho, with Pearson r alongside, between two numeric fields', () => {
    const { stat } = statOf(numeric('x'), numeric('y'));
    expect(stat.kind).toBe('spearman');
    expect(stat.value!).toBeLessThan(-0.7);
    expect(stat.pearson!).toBeLessThan(-0.7);
    expect(stat.n).toBe(6);
  });

  it('counts only the points left visible by the filters', () => {
    expect(statOf(numeric('x'), numeric('y'), new Set([0, 1])).stat.n).toBe(4);
  });

  it('summarises a statistic in one line', () => {
    expect(AssociationStatistics.summary({ kind: 'cramersV', value: 0.4234, pearson: null, n: 9 })).toBe("Cramér's V 0.42 · moderate");
    expect(AssociationStatistics.summary({ kind: 'eta', value: 0.05, pearson: null, n: 9 })).toBe('η 0.05 · negligible');
    expect(AssociationStatistics.summary({ kind: 'spearman', value: -0.62, pearson: -0.581, n: 9 }))
      .toBe('Spearman ρ −0.62 · strong (Pearson r −0.58)');
    expect(AssociationStatistics.summary({ kind: 'eta', value: null, pearson: null, n: 1 })).toBeNull();
  });

  it('names and reads the strength of a statistic, using its magnitude', () => {
    expect(AssociationStatistics.label('cramersV')).toBe("Cramér's V");
    expect(AssociationStatistics.label('eta')).toBe('η');
    expect(AssociationStatistics.label('spearman')).toBe('Spearman ρ');
    expect(AssociationStatistics.strength(null)).toBeNull();
    expect(AssociationStatistics.strength(0.05)).toBe('negligible');
    expect(AssociationStatistics.strength(0.2)).toBe('weak');
    expect(AssociationStatistics.strength(-0.4)).toBe('moderate');
    expect(AssociationStatistics.strength(-0.7)).toBe('strong');
  });
});
