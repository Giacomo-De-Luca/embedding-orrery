/**
 * Repro for the "click selects a point but semantic search silently returns
 * nothing" bug: useSemanticSearch shares one AbortController across calls and
 * (pre-fix) executed queries through a single useLazyQuery ObservableQuery.
 * Clicking point B while point A's search is still in flight aborts A's fetch
 * on the shared observable; the AbortError can surface on B's freshly started
 * execution, which the hook swallows into `null` → empty results.
 *
 * The harness below replicates Apollo Client 4's useLazyQuery execute
 * semantics (watchQuery with standby + reobserve on ONE ObservableQuery)
 * without React, against a mock link that honors context.fetchOptions.signal.
 */
import { describe, it, expect } from 'vitest';
import {
  ApolloClient,
  ApolloLink,
  InMemoryCache,
  Observable,
  gql,
} from '@apollo/client';

const QUERY = gql`
  query Search($itemId: String!) {
    search(itemId: $itemId) {
      id
    }
  }
`;

/** Link that resolves after `delayMs`, erroring with AbortError if the
 * context.fetchOptions.signal aborts first — like a real fetch. */
function makeMockLink(delayMs: (itemId: string) => number) {
  return new ApolloLink((operation) => {
    const signal: AbortSignal | undefined =
      operation.getContext().fetchOptions?.signal;
    const itemId = operation.variables.itemId as string;
    return new Observable((observer) => {
      const timer = setTimeout(() => {
        observer.next({ data: { search: [{ id: `similar-to-${itemId}`, __typename: 'Result' }] } });
        observer.complete();
      }, delayMs(itemId));
      const onAbort = () => {
        clearTimeout(timer);
        observer.error(new DOMException('The operation was aborted.', 'AbortError'));
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

/** Mimic @apollo/client v4 useLazyQuery: one shared ObservableQuery,
 * execute() = reobserve() with new variables/context. */
function makeLazyExecutor(client: ApolloClient) {
  const observable = client.watchQuery({
    query: QUERY,
    initialFetchPolicy: 'no-cache',
    fetchPolicy: 'standby' as any,
  } as any);
  // useLazyQuery keeps a live subscription via useSyncExternalStore
  observable.subscribe({ next: () => {}, error: () => {} });
  return (variables: Record<string, unknown>, context: Record<string, unknown>) => {
    let fetchPolicy = observable.options.fetchPolicy;
    if (fetchPolicy === 'standby') fetchPolicy = (observable.options as any).initialFetchPolicy;
    return observable.reobserve({ fetchPolicy, variables, context } as any);
  };
}

/** The hook's findSimilarById logic, verbatim in shape. */
function makeFindSimilarById(
  execute: (vars: Record<string, unknown>, ctx: Record<string, unknown>) => Promise<any>,
) {
  const abortControllerRef: { current: AbortController | null } = { current: null };
  return async (itemId: string): Promise<unknown[] | null> => {
    abortControllerRef.current?.abort();
    const controller = new AbortController();
    abortControllerRef.current = controller;
    try {
      const result = await execute(
        { itemId },
        { fetchOptions: { signal: controller.signal } },
      );
      if (result.data?.search) return result.data.search;
      return null;
    } catch (err) {
      if (err instanceof DOMException && err.name === 'AbortError') return null;
      throw err;
    }
  };
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

describe('semantic search abort race (click while previous search in flight)', () => {
  it('second click still gets its results when it aborts the first search', async () => {
    const client = new ApolloClient({
      cache: new InMemoryCache(),
      link: makeMockLink((itemId) => (itemId === 'A' ? 100 : 30)),
    });
    const findSimilarById = makeFindSimilarById(makeLazyExecutor(client));

    const first = findSimilarById('A'); // click point A — slow search
    await sleep(10);
    const second = findSimilarById('B'); // click point B before A resolves

    const [resultA, resultB] = await Promise.all([first, second]);

    // A was superseded: null is fine (caller discards via requestId guard).
    expect(resultA).toBeNull();
    // B must get ITS results — pre-fix this is null because A's AbortError
    // surfaces on the shared ObservableQuery and is swallowed.
    expect(resultB).toEqual([{ id: 'similar-to-B', __typename: 'Result' }]);
  });

  it('re-clicking the SAME point while its search is in flight still gets results', async () => {
    const client = new ApolloClient({
      cache: new InMemoryCache(),
      link: makeMockLink(() => 100),
    });
    const findSimilarById = makeFindSimilarById(makeLazyExecutor(client));

    const first = findSimilarById('A'); // click point A — slow search
    await sleep(10);
    const second = findSimilarById('A'); // impatient re-click on the same point

    const [resultA1, resultA2] = await Promise.all([first, second]);
    expect(resultA1).toBeNull(); // superseded, discarded by requestId guard
    // Pre-fix: query deduplication joins the re-click to the in-flight request
    // that its own abort just killed → AbortError → swallowed to null.
    expect(resultA2).toEqual([{ id: 'similar-to-A', __typename: 'Result' }]);
  });

  it('a single click resolves normally', async () => {
    const client = new ApolloClient({
      cache: new InMemoryCache(),
      link: makeMockLink(() => 20),
    });
    const findSimilarById = makeFindSimilarById(makeLazyExecutor(client));
    expect(await findSimilarById('A')).toEqual([{ id: 'similar-to-A', __typename: 'Result' }]);
  });
});
