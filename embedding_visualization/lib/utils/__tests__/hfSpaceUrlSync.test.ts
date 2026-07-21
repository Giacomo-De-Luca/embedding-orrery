import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  HF_PARENT_ORIGIN,
  isEmbedded,
  buildSyncMessage,
  createHfSpaceUrlSync,
  type EmbeddedWindowLike,
} from '../hfSpaceUrlSync';

function makeWindow(embedded: boolean) {
  const self = {};
  const postMessage = vi.fn();
  const win: EmbeddedWindowLike = {
    self,
    top: embedded ? {} : self,
    parent: { postMessage },
  };
  return { win, postMessage };
}

describe('isEmbedded', () => {
  it('is false when self === top', () => {
    expect(isEmbedded(makeWindow(false).win)).toBe(false);
  });
  it('is true when self !== top', () => {
    expect(isEmbedded(makeWindow(true).win)).toBe(true);
  });
  it('treats a throwing top accessor as embedded (cross-origin)', () => {
    const win = { self: {} } as EmbeddedWindowLike;
    Object.defineProperty(win, 'top', {
      get() {
        throw new Error('cross-origin');
      },
    });
    expect(isEmbedded(win)).toBe(true);
  });
});

describe('buildSyncMessage', () => {
  it('strips the leading ? from the query string', () => {
    expect(buildSyncMessage('?collection=emotion')).toEqual({ queryString: 'collection=emotion' });
    expect(buildSyncMessage('collection=emotion')).toEqual({ queryString: 'collection=emotion' });
    expect(buildSyncMessage('')).toEqual({ queryString: '' });
  });
});

describe('createHfSpaceUrlSync', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  it('never posts when not embedded', () => {
    const { win, postMessage } = makeWindow(false);
    const sync = createHfSpaceUrlSync(win);
    sync.post('?collection=emotion');
    vi.runAllTimers();
    expect(postMessage).not.toHaveBeenCalled();
  });

  it('posts the message shape to the exact HF origin', () => {
    const { win, postMessage } = makeWindow(true);
    const sync = createHfSpaceUrlSync(win, 100);
    sync.post('?collection=emotion&colorBy=label');
    vi.advanceTimersByTime(100);
    expect(postMessage).toHaveBeenCalledExactlyOnceWith(
      { queryString: 'collection=emotion&colorBy=label' },
      HF_PARENT_ORIGIN,
    );
  });

  it('skips reposting an unchanged query string', () => {
    const { win, postMessage } = makeWindow(true);
    const sync = createHfSpaceUrlSync(win, 100);
    sync.post('?a=1');
    vi.advanceTimersByTime(100);
    sync.post('?a=1');
    vi.advanceTimersByTime(100);
    expect(postMessage).toHaveBeenCalledTimes(1);
  });

  it('collapses a burst into one trailing post with the latest value', () => {
    const { win, postMessage } = makeWindow(true);
    const sync = createHfSpaceUrlSync(win, 100);
    sync.post('?a=1');
    vi.advanceTimersByTime(50);
    sync.post('?a=2');
    sync.post('?a=3');
    vi.advanceTimersByTime(100);
    expect(postMessage).toHaveBeenCalledExactlyOnceWith({ queryString: 'a=3' }, HF_PARENT_ORIGIN);
  });

  it('dispose cancels a pending post', () => {
    const { win, postMessage } = makeWindow(true);
    const sync = createHfSpaceUrlSync(win, 100);
    sync.post('?a=1');
    sync.dispose();
    vi.runAllTimers();
    expect(postMessage).not.toHaveBeenCalled();
  });
});
