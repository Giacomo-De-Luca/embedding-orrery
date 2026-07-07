"use client";

import { useEffect } from 'react';
import { ApolloProvider } from '@apollo/client/react';
import { ThemeProvider } from '@/lib/utils/theme-provider';
import { apolloClient } from '@/lib/utils/apollo-client';
import { Toaster } from '@/lib/ui-primitives/sonner';
import { useVisualizationStore } from '@/lib/stores/useVisualizationStore';

/** Applies persisted visualization preferences after hydration (the store uses
 *  skipHydration so server HTML and first client render agree on defaults). */
function StoreHydrator() {
  useEffect(() => {
    useVisualizationStore.persist.rehydrate();
  }, []);
  return null;
}

export function Providers({ children }: { children: React.ReactNode }) {
  return (
    <ApolloProvider client={apolloClient}>
      <ThemeProvider attribute="class" defaultTheme="system" enableSystem disableTransitionOnChange>
        <StoreHydrator />
        {children}
        {/* Inside ThemeProvider: the sonner wrapper reads useTheme() */}
        <Toaster />
      </ThemeProvider>
    </ApolloProvider>
  );
}
