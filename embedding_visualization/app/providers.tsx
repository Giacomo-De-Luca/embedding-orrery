"use client";

import { ApolloProvider } from '@apollo/client/react';
import { ThemeProvider } from '@/lib/utils/theme-provider';
import { apolloClient } from '@/lib/utils/apollo-client';
import { Toaster } from '@/lib/ui-primitives/sonner';

export function Providers({ children }: { children: React.ReactNode }) {
  return (
    <ApolloProvider client={apolloClient}>
      <ThemeProvider attribute="class" defaultTheme="system" enableSystem disableTransitionOnChange>
        {children}
        {/* Inside ThemeProvider: the sonner wrapper reads useTheme() */}
        <Toaster />
      </ThemeProvider>
    </ApolloProvider>
  );
}
