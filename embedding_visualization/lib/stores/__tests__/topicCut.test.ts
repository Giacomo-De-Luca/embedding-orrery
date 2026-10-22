import { beforeEach, describe, expect, it, vi } from 'vitest';
import { DEFAULT_TOPIC_CUT, useVisualizationStore } from '../useVisualizationStore';

// zustand's persist middleware attaches only with a storage (it reads
// window.localStorage): give the node test environment an in-memory one
// before the store module loads.
vi.hoisted(() => {
  const data = new Map<string, string>();
  const storage = {
    getItem: (key: string) => data.get(key) ?? null,
    setItem: (key: string, value: string) => void data.set(key, String(value)),
    removeItem: (key: string) => void data.delete(key),
    clear: () => data.clear(),
    key: () => null,
    length: 0,
  };
  Object.defineProperty(globalThis, 'window', { value: { localStorage: storage }, configurable: true, writable: true });
});

const store = useVisualizationStore;

describe('topic-tree cut state', () => {
  beforeEach(() => {
    store.getState().resetForCollectionChange();
    store.getState().setColorByField('topic_label', 'categorical');
  });

  it('clears category mutes when a preview starts, moves or ends (labels are renamed)', () => {
    store.getState().setMutedCategories(['Cats']);
    store.getState().setTopicCut({ previewing: true, nTopics: 5 });
    expect(store.getState().mutedCategories).toEqual([]);

    store.getState().setMutedCategories(['Cats +2']);
    store.getState().setTopicCut({ nTopics: 4 });
    expect(store.getState().mutedCategories).toEqual([]);

    store.getState().setMutedCategories(['Dogs +1']);
    store.getState().setTopicCut({ previewing: false, nTopics: null });
    expect(store.getState().mutedCategories).toEqual([]);
  });

  it('keeps the mutes array when there is nothing to clear, so a slider step renders once', () => {
    store.getState().setTopicCut({ previewing: true, nTopics: 5 });
    const mutes = store.getState().mutedCategories;
    store.getState().setTopicCut({ nTopics: 4 });
    store.getState().setTopicCut({ nTopics: 3 });
    expect(store.getState().mutedCategories).toBe(mutes);
  });

  it('keeps mutes when only the linkage of a non-previewing cut changes', () => {
    store.getState().setMutedCategories(['Cats']);
    store.getState().setTopicCut({ linkage: 'ward' });
    expect(store.getState().mutedCategories).toEqual(['Cats']);
  });

  it('ends the preview when the colour field changes, keeping the linkage', () => {
    store.getState().setTopicCut({ linkage: 'ward', previewing: true, nTopics: 3 });
    store.getState().setColorByField('pos', 'categorical');
    expect(store.getState().topicCut).toEqual({ ...DEFAULT_TOPIC_CUT, linkage: 'ward' });
  });

  it('resets on collection change', () => {
    store.getState().setTopicCut({ linkage: 'density', previewing: true, nTopics: 3, absorbNoise: true });
    store.getState().resetForCollectionChange();
    expect(store.getState().topicCut).toEqual(DEFAULT_TOPIC_CUT);
  });

  it('never persists the cut or the colour pins', () => {
    const persisted = store.persist.getOptions().partialize?.(store.getState()) as Record<string, unknown>;
    expect(persisted).not.toHaveProperty('topicCut');
    expect(persisted).not.toHaveProperty('categoryColorPins');
  });
});
