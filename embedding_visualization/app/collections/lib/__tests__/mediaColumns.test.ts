import { describe, expect, it } from 'vitest';

import { DATA_TYPE_LABELS, MEDIA_KINDS, MediaColumns } from '../mediaColumns';

const HF_FEATURES = [
  { name: 'img', dtype: 'image' },
  { name: 'clip', dtype: 'video' },
  { name: 'speech', dtype: 'audio' },
  { name: 'caption', dtype: 'string' },
  { name: 'label', dtype: 'class_label' },
];

describe('MediaColumns.kindsOf', () => {
  it('reads media kinds from HuggingFace feature dtypes', () => {
    expect(MediaColumns.kindsOf(HF_FEATURES)).toEqual({ img: 'image', clip: 'video', speech: 'audio' });
  });

  it("reads a local file's kinds from its preview, ignoring anything but image / audio / video", () => {
    const columns = [{ name: 'photo', dtype: 'string' }, { name: 'doc', dtype: 'string' }];
    expect(MediaColumns.kindsOf(columns, { photo: 'image', doc: 'pdf' })).toEqual({ photo: 'image' });
    expect(MediaColumns.kindsOf(columns, null)).toEqual({});
  });
});

describe('MediaColumns selections', () => {
  const kinds = MediaColumns.kindsOf(HF_FEATURES);

  it('counts the selected columns per kind', () => {
    expect(MediaColumns.counts(['img', 'clip', 'caption'], kinds)).toEqual({ image: 1, video: 1 });
    expect(MediaColumns.counts([], kinds)).toEqual({});
  });

  it('lists the selected kinds in a fixed order', () => {
    expect(MediaColumns.selectedKinds(['speech', 'img'], kinds)).toEqual(['image', 'audio']);
    expect(MEDIA_KINDS).toEqual(['image', 'audio', 'video']);
  });

  it('keeps media columns out of the text and metadata columns', () => {
    expect(MediaColumns.withoutMedia(['img', 'caption', 'label'], kinds)).toEqual(['caption', 'label']);
  });

  it('starts a multimodal embed with every media column and the rest as metadata', () => {
    expect(MediaColumns.defaultSelection(HF_FEATURES.map((f) => f.name), kinds)).toEqual({
      media: ['img', 'clip', 'speech'],
      metadata: ['caption', 'label'],
    });
  });
});

describe('MediaColumns.localColumns', () => {
  it('labels local columns with their media kind, else the type of their previewed values', () => {
    const rows = [
      { photo: '[image 12 KB]', title: null, score: 0.5, tags: ['a'], flag: true, meta: { a: 1 } },
      { photo: 'b.png', title: 'Cat', score: 2, tags: [], flag: false, meta: null },
    ];
    expect(
      MediaColumns.localColumns(['photo', 'title', 'score', 'tags', 'flag', 'meta', 'empty'], rows, { photo: 'image' }),
    ).toEqual([
      { name: 'photo', dtype: 'image' },
      { name: 'title', dtype: 'string' },
      { name: 'score', dtype: 'number' },
      { name: 'tags', dtype: 'list' },
      { name: 'flag', dtype: 'boolean' },
      { name: 'meta', dtype: 'object' },
      { name: 'empty', dtype: 'unknown' },
    ]);
  });
});

describe('multimodal collections', () => {
  it('are recognised by their recorded data type', () => {
    expect(MediaColumns.isMultimodalCollection({ data_type: 'multimodal' })).toBe(true);
    expect(MediaColumns.isMultimodalCollection({ data_type: 'text' })).toBe(false);
    expect(MediaColumns.isMultimodalCollection(undefined)).toBe(false);
  });

  it('carry the "Image / Audio / Video / Mixed" label', () => {
    expect(DATA_TYPE_LABELS.MULTIMODAL).toBe('Image / Audio / Video / Mixed');
  });
});
