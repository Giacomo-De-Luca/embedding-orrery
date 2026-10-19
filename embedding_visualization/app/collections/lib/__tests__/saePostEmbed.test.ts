import { describe, it, expect, vi } from 'vitest';
import {
  parseSaeSelection,
  runPostEmbedSaeStep,
  saeLinkFromParams,
} from '../embeddingFormUtils';
import type { EmbedDatasetResult, SaeLinkInput } from '@/lib/graphql/mutations';
import { JOB_CANCELLED_MESSAGE } from '@/lib/utils/jobCancellation';

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

describe('saeLinkFromParams', () => {
  it('builds the embed input link when the step is enabled with a selection', () => {
    expect(saeLinkFromParams({ enabled: true, selection: 'gemma-3-4b-it::9-gemmascope-2-res-16k' })).toEqual({
      saeModelId: 'gemma-3-4b-it',
      saeId: '9-gemmascope-2-res-16k',
      computeActivations: true,
    });
  });

  it('is null when disabled or nothing valid is selected', () => {
    expect(saeLinkFromParams({ enabled: false, selection: 'm::s' })).toBeNull();
    expect(saeLinkFromParams({ enabled: true, selection: null })).toBeNull();
    expect(saeLinkFromParams({ enabled: true, selection: 'broken' })).toBeNull();
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
  const link: SaeLinkInput = {
    saeModelId: 'gemma-3-4b-it',
    saeId: '9-gemmascope-2-res-16k',
    computeActivations: true,
  };

  function compute() {
    return vi.fn().mockResolvedValue({ error: null });
  }

  it('computes activations after a successful embed that linked an SAE', async () => {
    const computeDocumentActivations = compute();

    const ran = await runPostEmbedSaeStep(okResult, link, { computeDocumentActivations });

    expect(ran).toBe(true);
    expect(computeDocumentActivations).toHaveBeenCalledWith('my_coll');
  });

  it('does not compute for a link-only SAE (e.g. a resumed SAE-tab feature collection)', async () => {
    const computeDocumentActivations = compute();
    const linkOnly = { ...link, computeActivations: false };
    expect(await runPostEmbedSaeStep(okResult, linkOnly, { computeDocumentActivations })).toBe(false);
    expect(computeDocumentActivations).not.toHaveBeenCalled();
  });

  it('does nothing without a link', async () => {
    const computeDocumentActivations = compute();
    expect(await runPostEmbedSaeStep(okResult, null, { computeDocumentActivations })).toBe(false);
    expect(await runPostEmbedSaeStep(okResult, undefined, { computeDocumentActivations })).toBe(false);
    expect(computeDocumentActivations).not.toHaveBeenCalled();
  });

  it('does nothing when the embed failed, returned null or was cancelled', async () => {
    const computeDocumentActivations = compute();
    const deps = { computeDocumentActivations };
    expect(await runPostEmbedSaeStep(null, link, deps)).toBe(false);
    expect(await runPostEmbedSaeStep({ ...okResult, error: 'boom' }, link, deps)).toBe(false);
    expect(await runPostEmbedSaeStep({ ...okResult, error: JOB_CANCELLED_MESSAGE }, link, deps)).toBe(false);
    expect(computeDocumentActivations).not.toHaveBeenCalled();
  });

  it('still counts as run when compute reports an error (embed itself succeeded)', async () => {
    const computeDocumentActivations = vi.fn().mockResolvedValue({ error: 'model load failed' });
    expect(await runPostEmbedSaeStep(okResult, link, { computeDocumentActivations })).toBe(true);
  });
});
