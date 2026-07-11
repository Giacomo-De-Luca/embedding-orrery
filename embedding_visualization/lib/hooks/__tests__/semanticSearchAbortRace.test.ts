/**
 * Regression tests for the "click selects a point but semantic search
 * silently returns nothing" bug.
 *
 * Root cause chain (pre-fix):
 * 1. gl3d emits plotly_click from its per-frame pick pass, so one physical
 *    press over a point could fire several click events (one per GL frame).
 * 2. Each search aborted the previous in-flight one via a shared
 *    AbortController, and ran through one shared useLazyQuery ObservableQuery
 *    with Apollo query deduplication ON: an identical query starting right
 *    after the abort joined the in-flight operation it had just killed and
 *    inherited its AbortError (real fetch rejects a microtask AFTER abort(),
 *    so the doomed operation was still in the dedup map).
 * 3. The AbortError was swallowed to `null`, which the caller turned into
 *    empty results — point selected, search results silently blank.
 *
 * The fix (`createSemanticSearchClient`) uses one-shot client.query calls
 * with per-request abort and queryDeduplication disabled, returns [] for
 * genuinely-empty results, null only for aborts, and throws real errors.
 *
 * The mock link rejects asynchronously on abort — that microtask gap is
 * load-bearing for reproducing the original dedup-join failure.
 */
import { describe, it, expect } from 'vitest';
import { ApolloClient, ApolloLink, InMemoryCache, Observable } from '@apollo/client';
import { createSemanticSearchClient, isAbortError } from '../../utils/semanticSearchClient';

interface MockBehavior {
  delayMs: (itemId: string) => number;
  respond?: (itemId: string) => unknown;
}

/** Link that resolves after a delay, erroring with AbortError (asynchronously,
 * like real fetch) if context.fetchOptions.signal aborts first. */
function makeMockLink({ delayMs, respond }: MockBehavior) {
  const defaultRespond = (itemId: string) => ({
    data: {
      semanticSearchById: [
        { id: `similar-to-${itemId}`, document: 'doc', metadata: {}, similarity: 0.9, distance: 0.1 },
      ],
    },
  });
  return new ApolloLink((operation) => {
    const signal: AbortSignal | undefined = operation.getContext().fetchOptions?.signal;
    const itemId = operation.variables.itemId as string;
    return new Observable((observer) => {
      const timer = setTimeout(() => {
        observer.next((respond ?? defaultRespond)(itemId) as any);
        observer.complete();
      }, delayMs(itemId));
      const onAbort = () => {
        clearTimeout(timer);
        queueMicrotask(() =>
          observer.error(new DOMException('The operation was aborted.', 'AbortError')),
        );
      };
      if (signal) {
        if (signal.aborted) onAbort();
        else signal.addEventListener('abort', onAbort);
      }
      return () => {
        clearTimeout(timer);
        signal?.removeEventListener('abort', onAbort);
      };
    });
  });
}

function makeSearcher(behavior: MockBehavior) {
  const client = new ApolloClient({ cache: new InMemoryCache(), link: makeMockLink(behavior) });
  const searcher = createSemanticSearchClient(client);
  return (itemId: string) =>
    searcher.searchById({ collectionName: 'c', itemId, nResults: 20, similarityMeasure: 'COSINE' });
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const resultFor = (itemId: string) => [
  { id: `similar-to-${itemId}`, document: 'doc', metadata: {}, similarity: 0.9, distance: 0.1 },
];

describe('semantic search abort race (click while previous search in flight)', () => {
  it('clicking point B while A is in flight: B gets its results, A resolves null', async () => {
    const searchById = makeSearcher({ delayMs: (id) => (id === 'A' ? 100 : 30) });

    const first = searchById('A'); // click point A — slow search
    await sleep(10);
    const second = searchById('B'); // click point B before A resolves

    const [resultA, resultB] = await Promise.all([first, second]);
    expect(resultA).toBeNull(); // superseded — caller leaves results untouched
    expect(resultB).toEqual(resultFor('B'));
  });

  it('re-clicking the SAME point while its search is in flight still gets results', async () => {
    const searchById = makeSearcher({ delayMs: () => 100 });

    const first = searchById('A'); // click point A — slow search
    await sleep(10);
    const second = searchById('A'); // duplicate click (gl3d frame re-emit or impatient user)

    const [resultA1, resultA2] = await Promise.all([first, second]);
    expect(resultA1).toBeNull(); // superseded
    // Pre-fix: query deduplication joined this to the in-flight operation its
    // own abort had just killed → AbortError → null → blank results.
    expect(resultA2).toEqual(resultFor('A'));
  });

  it('a single click resolves normally', async () => {
    const searchById = makeSearcher({ delayMs: () => 20 });
    expect(await searchById('A')).toEqual(resultFor('A'));
  });

  it('a genuinely empty result is [] (distinct from aborted null)', async () => {
    const searchById = makeSearcher({
      delayMs: () => 10,
      respond: () => ({ data: { semanticSearchById: [] } }),
    });
    expect(await searchById('A')).toEqual([]);
  });

  it('real errors are thrown, not swallowed to null', async () => {
    const client = new ApolloClient({
      cache: new InMemoryCache(),
      link: new ApolloLink(
        () =>
          new Observable((observer) => {
            observer.error(new Error('backend exploded'));
          }),
      ),
    });
    const searcher = createSemanticSearchClient(client);
    await expect(
      searcher.searchById({ collectionName: 'c', itemId: 'A', nResults: 20, similarityMeasure: 'COSINE' }),
    ).rejects.toThrow(/backend exploded/);
  });
});

describe('isAbortError', () => {
  it('detects raw and wrapped abort errors, rejects others', () => {
    const abort = new DOMException('aborted', 'AbortError');
    expect(isAbortError(abort)).toBe(true);
    expect(isAbortError(Object.assign(new Error('net'), { cause: abort }))).toBe(true);
    expect(isAbortError(Object.assign(new Error('net'), { networkError: abort }))).toBe(true);
    expect(isAbortError(new Error('other'))).toBe(false);
    expect(isAbortError(null)).toBe(false);
  });
});
