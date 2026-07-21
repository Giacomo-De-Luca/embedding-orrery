'use client';

import { useEffect, useMemo } from 'react';
import { useSearchParams } from 'next/navigation';
import { createHfSpaceUrlSync } from '../utils/hfSpaceUrlSync';

/**
 * Keeps the Hugging Face Space's parent-page URL in sync with the app's query
 * string (see lib/utils/hfSpaceUrlSync.ts). No-op outside an iframe, so it is
 * safe to mount unconditionally.
 */
export function useHfSpaceUrlSync(): void {
  const searchParams = useSearchParams();
  const sync = useMemo(
    () => (typeof window === 'undefined' ? null : createHfSpaceUrlSync(window)),
    [],
  );
  const search = searchParams.toString();

  useEffect(() => {
    sync?.post(search);
  }, [sync, search]);

  useEffect(() => () => sync?.dispose(), [sync]);
}
