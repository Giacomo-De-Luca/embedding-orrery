import { describe, it, expect } from 'vitest';
import type { SaeModelInfo } from '@/lib/types/types';
import {
  parseSaesParam,
  serializeSaesParam,
  resolveSelectionFromParams,
} from '../saeSelection';

const MODELS: SaeModelInfo[] = [
  { modelId: 'gemma-3-4b-it', saeId: '9-gemmascope-2-res-16k', featureCount: 16384, activationCount: 16380 },
  { modelId: 'gemma-3-4b-it', saeId: '9-gemmascope-2-res-65k', featureCount: 65536, activationCount: 65487 },
  { modelId: 'gemma-3-4b-it', saeId: '22-gemmascope-2-res-16k', featureCount: 16384, activationCount: 16379 },
  { modelId: 'gemma-3-1b', saeId: '22-gemmascope-2-res-16k', featureCount: 16384, activationCount: 16253 },
];

const EMPTY_PARAMS = {
  saes: null,
  model: null,
  modelId: null,
  saeId: null,
  layer: null,
  hookType: null,
  width: null,
};

describe('saes URL param round trip', () => {
  it('serializes and parses back', () => {
    const ids = ['9-gemmascope-2-res-16k', '22-gemmascope-2-res-16k'];
    expect(parseSaesParam(serializeSaesParam(ids))).toEqual(ids);
  });

  it('parses null/empty to []', () => {
    expect(parseSaesParam(null)).toEqual([]);
    expect(parseSaesParam('')).toEqual([]);
    expect(parseSaesParam(',,')).toEqual([]);
  });
});

describe('resolveSelectionFromParams', () => {
  it('legacy modelId+saeId cross-link resolves to a single-SAE selection', () => {
    const sel = resolveSelectionFromParams(
      { ...EMPTY_PARAMS, modelId: 'gemma-3-4b-it', saeId: '9-gemmascope-2-res-16k' },
      MODELS,
    );
    expect(sel).toEqual({ modelId: 'gemma-3-4b-it', saeIds: ['9-gemmascope-2-res-16k'] });
  });

  it('multi format model+saes resolves the listed SAEs', () => {
    const sel = resolveSelectionFromParams(
      {
        ...EMPTY_PARAMS,
        model: 'gemma-3-4b-it',
        saes: '9-gemmascope-2-res-16k,9-gemmascope-2-res-65k',
      },
      MODELS,
    );
    expect(sel).toEqual({
      modelId: 'gemma-3-4b-it',
      saeIds: ['9-gemmascope-2-res-16k', '9-gemmascope-2-res-65k'],
    });
  });

  it('old dimension format filters the model SAEs by layer/width', () => {
    const sel = resolveSelectionFromParams(
      { ...EMPTY_PARAMS, model: 'gemma-3-4b-it', layer: '9', width: '16k' },
      MODELS,
    );
    expect(sel).toEqual({ modelId: 'gemma-3-4b-it', saeIds: ['9-gemmascope-2-res-16k'] });
  });

  it('old dimension format with model only selects all of its SAEs', () => {
    const sel = resolveSelectionFromParams(
      { ...EMPTY_PARAMS, model: 'gemma-3-4b-it' },
      MODELS,
    );
    expect(sel?.saeIds).toHaveLength(3);
  });

  it('old dimension format without a model falls back to the first model', () => {
    const sel = resolveSelectionFromParams({ ...EMPTY_PARAMS, width: '65k' }, MODELS);
    expect(sel).toEqual({ modelId: 'gemma-3-4b-it', saeIds: ['9-gemmascope-2-res-65k'] });
  });

  it('returns null when no params are present', () => {
    expect(resolveSelectionFromParams(EMPTY_PARAMS, MODELS)).toBeNull();
  });
});
