/**
 * Tests for probe hyperparameter helpers: which params apply per kind, and
 * building the GraphQL TrainProbeInput (defaults omitted so the backend's
 * defaults stay authoritative).
 */
import { describe, it, expect } from 'vitest';

import {
  DEFAULT_PROBE_PARAMS,
  probeParamFields,
  isBinaryKind,
  buildTrainProbeInput,
  type ProbeParams,
} from '../probeParams';

describe('probeParamFields', () => {
  it('returns the relevant controls per kind', () => {
    expect(probeParamFields('ridge')).toEqual(['alpha']);
    expect(probeParamFields('lasso')).toEqual(['alpha']);
    expect(probeParamFields('svr')).toEqual(['c', 'kernel']);
    expect(probeParamFields('logreg')).toEqual(['c', 'classWeight']);
    expect(probeParamFields('mlp')).toEqual(['hiddenSize', 'epochs', 'patience']);
    expect(probeParamFields('massmean')).toEqual([]);
    expect(probeParamFields('massmean_cov')).toEqual([]);
  });
});

describe('isBinaryKind', () => {
  it('is true only for logreg', () => {
    expect(isBinaryKind('logreg')).toBe(true);
    expect(isBinaryKind('ridge')).toBe(false);
    expect(isBinaryKind('svr')).toBe(false);
  });
});

describe('buildTrainProbeInput', () => {
  const base = { ...DEFAULT_PROBE_PARAMS };

  it('sends only collection/field/kind at defaults', () => {
    expect(buildTrainProbeInput('col', 'f', 'ridge', base)).toEqual({
      collectionName: 'col',
      targetField: 'f',
      kind: 'ridge',
    });
  });

  it('includes a changed alpha for ridge', () => {
    const p: ProbeParams = { ...base, alpha: 0.5 };
    expect(buildTrainProbeInput('col', 'f', 'ridge', p).alpha).toBe(0.5);
  });

  it('includes svr C + kernel only when changed', () => {
    const p: ProbeParams = { ...base, c: 2.0, kernel: 'linear' };
    const v = buildTrainProbeInput('col', 'f', 'svr', p);
    expect(v.c).toBe(2.0);
    expect(v.kernel).toBe('linear');
  });

  it('does not leak params from other kinds', () => {
    // c is changed but the kind is ridge, which does not use c.
    const p: ProbeParams = { ...base, c: 9.0, alpha: 2.0 };
    const v = buildTrainProbeInput('col', 'f', 'ridge', p);
    expect(v.c).toBeUndefined();
    expect(v.alpha).toBe(2.0);
  });

  it('maps logreg classWeight balanced, omits none', () => {
    expect(
      buildTrainProbeInput('col', 'f', 'logreg', { ...base, classWeight: 'balanced' }).classWeight,
    ).toBe('balanced');
    expect(
      buildTrainProbeInput('col', 'f', 'logreg', { ...base, classWeight: 'none' }).classWeight,
    ).toBeUndefined();
  });

  it('wraps mlp hidden size into hiddenDims', () => {
    const v = buildTrainProbeInput('col', 'f', 'mlp', { ...base, hiddenSize: 128 });
    expect(v.hiddenDims).toEqual([128]);
  });

  it('sends shared seed/split when non-default', () => {
    const v = buildTrainProbeInput('col', 'f', 'ridge', { ...base, seed: 9, trainSplit: 0.7 });
    expect(v.seed).toBe(9);
    expect(v.trainSplit).toBe(0.7);
  });

  it('default seed matches the backend default and is omitted', () => {
    // The popover shows the backend's own default (42); sending nothing must
    // therefore train with exactly the displayed value.
    expect(DEFAULT_PROBE_PARAMS.seed).toBe(42);
    expect(buildTrainProbeInput('col', 'f', 'ridge', base).seed).toBeUndefined();
  });

  it('closed-form massmean_cov sends kind only at defaults', () => {
    expect(buildTrainProbeInput('col', 'f', 'massmean_cov', base)).toEqual({
      collectionName: 'col',
      targetField: 'f',
      kind: 'massmean_cov',
    });
  });

  it('lasso sends a changed alpha', () => {
    expect(buildTrainProbeInput('col', 'f', 'lasso', { ...base, alpha: 0.01 }).alpha).toBe(0.01);
  });

  it('sends mlp patience only when changed, and only for mlp', () => {
    expect(buildTrainProbeInput('col', 'f', 'mlp', base).patience).toBeUndefined();
    expect(
      buildTrainProbeInput('col', 'f', 'mlp', { ...base, patience: 5 }).patience,
    ).toBe(5);
    // Patience is an mlp knob; other kinds never send it.
    expect(
      buildTrainProbeInput('col', 'f', 'ridge', { ...base, patience: 5 }).patience,
    ).toBeUndefined();
  });

  it('sends shared maxTrainSamples when non-default, for any kind', () => {
    expect(buildTrainProbeInput('col', 'f', 'ridge', base).maxTrainSamples).toBeUndefined();
    expect(
      buildTrainProbeInput('col', 'f', 'massmean', { ...base, maxTrainSamples: 10000 })
        .maxTrainSamples,
    ).toBe(10000);
  });
});
