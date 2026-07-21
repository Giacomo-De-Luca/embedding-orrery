/**
 * Parent-URL sync for the Hugging Face Space demo.
 *
 * On huggingface.co the app runs inside an iframe. HF propagates the parent
 * page's query string into the iframe on initial load, but updates made by the
 * embedded app (our `router.replace` view-state sync) only reach the parent —
 * and thus the user's address bar / share button — via `postMessage`:
 *   window.parent.postMessage({ queryString }, "https://huggingface.co")
 * (documented in HF's "How to handle URL parameters in Spaces").
 *
 * Pure core, node-testable: the window is a duck-typed argument.
 */

import { stripQueryPrefix } from './urlViewParams';

export const HF_PARENT_ORIGIN = 'https://huggingface.co';

export interface EmbeddedWindowLike {
  self?: unknown;
  top?: unknown;
  parent?: { postMessage: (message: unknown, targetOrigin: string) => void } | null;
}

/** True when running inside any iframe (cross-origin access throws ⇒ embedded). */
export function isEmbedded(win: EmbeddedWindowLike): boolean {
  try {
    return win.self !== win.top;
  } catch {
    return true;
  }
}

/** HF message shape: query string without the leading `?`. */
export function buildSyncMessage(search: string): { queryString: string } {
  return { queryString: stripQueryPrefix(search) };
}

export interface HfSpaceUrlSync {
  post: (search: string) => void;
  dispose: () => void;
}

/**
 * Debounced (trailing) poster. No-op entirely outside an iframe; skips posts
 * whose query string matches the last one sent.
 */
export function createHfSpaceUrlSync(win: EmbeddedWindowLike, debounceMs = 250): HfSpaceUrlSync {
  if (!isEmbedded(win)) {
    return { post: () => {}, dispose: () => {} };
  }
  let lastPosted: string | null = null;
  let pending: string | null = null;
  let timer: ReturnType<typeof setTimeout> | null = null;

  const flush = () => {
    timer = null;
    if (pending === null || pending === lastPosted) return;
    lastPosted = pending;
    win.parent?.postMessage(buildSyncMessage(pending), HF_PARENT_ORIGIN);
  };

  return {
    post(search: string) {
      const queryString = stripQueryPrefix(search);
      if (queryString === lastPosted && timer === null) return;
      pending = queryString; // trailing value wins within the debounce window
      if (timer === null) timer = setTimeout(flush, debounceMs);
    },
    dispose() {
      if (timer !== null) clearTimeout(timer);
      timer = null;
      pending = null;
    },
  };
}
