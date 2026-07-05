'use client';

import { useEffect, useRef, useState } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';

export const COLLECTION_TABS = ['huggingface', 'local', 'manage', 'sae'] as const;
export type DataSourceTab = (typeof COLLECTION_TABS)[number];

const DEFAULT_TAB: DataSourceTab = 'huggingface';

function isDataSourceTab(value: string | null): value is DataSourceTab {
  return value !== null && (COLLECTION_TABS as readonly string[]).includes(value);
}

export interface CollectionsUrlState {
  tab: DataSourceTab;
  collection: string | null;
}

/**
 * Resolve ?tab= / ?collection= into the initial page state.
 * A ?collection= without a valid ?tab= implies the manage tab; the collection
 * param is only meaningful on the manage tab.
 */
export function parseCollectionsParams(params: {
  get(name: string): string | null;
}): CollectionsUrlState {
  const rawTab = params.get('tab');
  const collection = params.get('collection');
  const tab = isDataSourceTab(rawTab) ? rawTab : collection ? 'manage' : DEFAULT_TAB;
  return { tab, collection: tab === 'manage' ? collection : null };
}

/**
 * Build the query string reflecting `next`, preserving unrelated params.
 * The default tab is omitted from the URL; `collection` is dropped whenever
 * the tab is not `manage`. Returns null when the result equals
 * `currentSearch` (no navigation needed), otherwise the new search string
 * ('' means "no params").
 */
export function buildCollectionsSearch(
  currentSearch: string,
  next: CollectionsUrlState
): string | null {
  const params = new URLSearchParams(currentSearch);
  if (next.tab === DEFAULT_TAB) {
    params.delete('tab');
  } else {
    params.set('tab', next.tab);
  }
  if (next.tab === 'manage' && next.collection) {
    params.set('collection', next.collection);
  } else {
    params.delete('collection');
  }
  const serialized = params.toString();
  const normalizedCurrent = new URLSearchParams(currentSearch).toString();
  if (serialized === normalizedCurrent) return null;
  return serialized ? `?${serialized}` : '';
}

/**
 * Page-local URL state for /collections: active tab + manage-tab selection.
 *
 * READ-ONCE CONTRACT: params are parsed on first render only, then state is
 * the source of truth and is mirrored into the URL via router.replace (no
 * history entries). An in-app navigation to /collections?... while already
 * mounted will NOT update the state — if such a link is ever added, this
 * hook must additionally sync from useSearchParams. Must be used under a
 * <Suspense> boundary (useSearchParams requirement).
 */
export function useCollectionsUrlState() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const initialRef = useRef<CollectionsUrlState | null>(null);
  if (initialRef.current === null) {
    initialRef.current = parseCollectionsParams(searchParams);
  }

  const [tab, setTab] = useState<DataSourceTab>(initialRef.current.tab);
  const [managedCollection, setManagedCollection] = useState<string | null>(
    initialRef.current.collection
  );

  useEffect(() => {
    const newSearch = buildCollectionsSearch(window.location.search, {
      tab,
      collection: managedCollection,
    });
    if (newSearch !== null) {
      router.replace(`${window.location.pathname}${newSearch}`, { scroll: false });
    }
  }, [tab, managedCollection, router]);

  return { tab, setTab, managedCollection, setManagedCollection };
}
