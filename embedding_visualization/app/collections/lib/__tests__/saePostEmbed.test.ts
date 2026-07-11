import { describe, it, expect, vi } from 'vitest';
import {
  parseSaeSelection,
  runPostEmbedSaeStep,
  type SaePostEmbedParams,
} from '../embeddingFormUtils';
import type { EmbedDatasetResult } from '@/lib/graphql/mutations';

describe('parseSaeSelection', () => {
  it('splits a modelId::saeId composite', () => {
    expect(parseSaeSelection('gemma-3-4b-it::9-gemmascope-2-res-16k')).toEqual({
      sae_model_id: 'gemma-3-4b-it',
      sae_id: '9-gemmascope-2-res-16k',
    });
  });

  it('returns null for empty or malformed selections', () => {
    expect(parseSaeSelection(null)).toBeNull();
    expect(parseSaeSelection('')).toBeNull();
    expect(parseSaeSelection('no-separator')).toBeNull();
    expect(parseSaeSelection('::missing-model')).toBeNull();
    expect(parseSaeSelection('missing-sae::')).toBeNull();
  });
});

describe('runPostEmbedSaeStep', () => {
  const okResult: EmbedDatasetResult = {
    collectionName: 'my_coll',
    totalEmbedded: 10,
    embeddingDim: 384,
    device: 'cpu',
    durationSeconds: 1,
    projectionsComputed: true,
    error: null,
    embeddingProvider: null,
    embeddingModel: null,
  };

  function deps() {
    return {
      updateCollectionMetadata: vi.fn().mockResolvedValue({ error: null }),
      computeDocumentActivations: vi.fn().mockResolvedValue({ error: null }),
    };
  }

  const params: SaePostEmbedParams = {
    enabled: true,
    selection: 'gemma-3-4b-it::9-gemmascope-2-res-16k',
  };

  it('links then computes, in that order, on a successful embed', async () => {
    const d = deps();
    const calls: string[] = [];
    d.updateCollectionMetadata.mockImplementation(async () => { calls.push('link'); return { error: null }; });
    d.computeDocumentActivations.mockImplementation(async () => { calls.push('compute'); return { error: null }; });

    const ran = await runPostEmbedSaeStep(okResult, params, d);

    expect(ran).toBe(true);
    expect(calls).toEqual(['link', 'compute']);
    expect(d.updateCollectionMetadata).toHaveBeenCalledWith('my_coll', {
      sae_model_id: 'gemma-3-4b-it',
      sae_id: '9-gemmascope-2-res-16k',
    });
    expect(d.computeDocumentActivations).toHaveBeenCalledWith('my_coll');
  });

  it('does nothing when disabled', async () => {
    const d = deps();
    const ran = await runPostEmbedSaeStep(okResult, { ...params, enabled: false }, d);
    expect(ran).toBe(false);
    expect(d.updateCollectionMetadata).not.toHaveBeenCalled();
    expect(d.computeDocumentActivations).not.toHaveBeenCalled();
  });

  it('does nothing when no SAE is selected', async () => {
    const d = deps();
    const ran = await runPostEmbedSaeStep(okResult, { enabled: true, selection: null }, d);
    expect(ran).toBe(false);
    expect(d.updateCollectionMetadata).not.toHaveBeenCalled();
  });

  it('does nothing when the embed failed or returned null', async () => {
    const d = deps();
    expect(await runPostEmbedSaeStep(null, params, d)).toBe(false);
    expect(await runPostEmbedSaeStep({ ...okResult, error: 'boom' }, params, d)).toBe(false);
    expect(d.updateCollectionMetadata).not.toHaveBeenCalled();
  });

  it('skips compute when linking fails, and reports not-run', async () => {
    const d = deps();
    d.updateCollectionMetadata.mockResolvedValue({ error: 'metadata write failed' });
    const ran = await runPostEmbedSaeStep(okResult, params, d);
    expect(ran).toBe(false);
    expect(d.computeDocumentActivations).not.toHaveBeenCalled();
  });

  it('still counts as run when compute reports an error (embed itself succeeded)', async () => {
    const d = deps();
    d.computeDocumentActivations.mockResolvedValue({ error: 'model load failed' });
    const ran = await runPostEmbedSaeStep(okResult, params, d);
    expect(ran).toBe(true);
  });
});
