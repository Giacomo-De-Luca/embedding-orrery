'use client';

import { Moon, Sun, Search, Settings2, BarChart3 } from 'lucide-react';
import { useTheme } from 'next-themes';
import { useState, KeyboardEvent } from 'react';
import { PageNav } from './PageNav';
import {
  Combobox,
  ComboboxInput,
  ComboboxContent,
  ComboboxList,
  ComboboxItem,
  ComboboxEmpty,
} from '@/lib/ui-primitives/combobox';
import { Spinner } from '@/lib/ui-primitives/spinner';
import { Badge } from '@/lib/ui-primitives/badge';
import { Separator } from '@/lib/ui-primitives/separator';
import { Button } from '@/lib/ui-primitives/button';
import { cn } from '@/lib/utils/utils';
import { Input } from '@/lib/ui-primitives/input';
import type { CollectionsManifest } from '../../lib/types/types';
import { getSaeInfo } from '../../lib/utils/saeCollections';

function ModeToggle() {
  const { resolvedTheme, setTheme } = useTheme();
  const isDark = (resolvedTheme ?? 'light') === 'dark';

  return (
    <Button
      variant="circular"
      size="icon"
      className="relative"
      onClick={() => setTheme(isDark ? 'light' : 'dark')}
      aria-label={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
      suppressHydrationWarning
    >
      <Sun className="h-[1.2rem] w-[1.2rem] scale-100 rotate-0 transition-all dark:scale-0 dark:-rotate-90" />
      <Moon className="absolute h-[1.2rem] w-[1.2rem] scale-0 rotate-90 transition-all dark:scale-100 dark:rotate-0" />
    </Button>
  );
}

type ActivePanel = 'controls' | 'search' | 'analytics' | null;

interface AppHeaderProps {
  collections: CollectionsManifest | null;
  collectionsLoading: boolean;
  collectionsError: Error | null;
  selectedCollection: string | null;
  onCollectionChange: (collection: string) => void;
  totalWords?: number;
  embeddingDim?: number;
  onSemanticSearch?: (query: string) => void;
  searchLoading?: boolean;
  activePanel?: ActivePanel;
  onToggleControls?: () => void;
  onToggleSearch?: () => void;
  onToggleAnalytics?: () => void;
  saeInfo?: { modelId: string; saeId: string } | null;
}

export function AppHeader({
  collections,
  collectionsLoading,
  collectionsError,
  selectedCollection,
  onCollectionChange,
  totalWords,
  embeddingDim,
  onSemanticSearch,
  searchLoading = false,
  activePanel,
  onToggleControls,
  onToggleSearch,
  onToggleAnalytics,
  saeInfo: saeInfoProp,
}: AppHeaderProps) {
  const [searchQuery, setSearchQuery] = useState('');
  const [collectionFilter, setCollectionFilter] = useState('');

  const handleSearch = () => {
    if (searchQuery.trim() && onSemanticSearch) {
      onSemanticSearch(searchQuery.trim());
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      handleSearch();
    }
  };

  const isExpanded = activePanel !== null;

  // Context-carrying deep link: keep model/SAE params when the collection has a mapping
  const sae = saeInfoProp ?? getSaeInfo(selectedCollection ?? null);
  const saeHref = sae
    ? `/sae?modelId=${encodeURIComponent(sae.modelId)}&saeId=${encodeURIComponent(sae.saeId)}`
    : undefined;

  return (
    <header
      className={cn(
        "flex bg-transparent h-16 shrink-0 items-center transition-all duration-300 ease-in-out",
        "pl-0"
        // isExpanded ? "pl-84" : "pl-0"
      )}
    >
      <div className="flex w-full items-center gap-2 px-4">
        <Button
          variant={activePanel === 'controls' ? 'circular' : 'circularghost'}
          size="icon"
          onClick={onToggleControls}
          aria-label="Toggle controls panel"
          className="-ml-1"
        >
          <Settings2 className="h-4 w-4" />
        </Button>
        <Button
          variant={activePanel === 'search' ? 'circular' : 'circularghost'}
          size="icon"
          onClick={onToggleSearch}
          aria-label="Toggle search panel"
        >
          <Search className="h-4 w-4" />
        </Button>
        <Button
          variant={activePanel === 'analytics' ? 'circular' : 'circularghost'}
          size="icon"
          onClick={onToggleAnalytics}
          aria-label="Toggle analytics panel"
        >
          <BarChart3 className="h-4 w-4" />
        </Button>
        <Separator orientation="vertical" className="mr-2 h-4" />

        <div className="flex items-center gap-3 flex-1">
          <div className="flex items-center gap-3">
            {totalWords && embeddingDim && (
              <>
                <Badge variant="secondary" className="hidden backdrop-blur-sm sm:inline-flex">
                  {totalWords.toLocaleString()} points
                </Badge>
                <Badge variant="outline" className="hidden md:inline-flex">
                  {embeddingDim}D
                </Badge>
              </>
            )}
          </div>

          {/* Semantic Search */}
          {onSemanticSearch && (
            <div className="relative max-w-xs flex-1">
              <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Semantic search..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onKeyDown={handleKeyDown}
                className="pl-8 pr-9"
                disabled={searchLoading}
              />
              <button
                onClick={handleSearch}
                disabled={!searchQuery.trim() || searchLoading}
                className={cn(
                  "absolute right-1.5 top-1/2 -translate-y-1/2 rounded-full h-6 w-6 flex items-center justify-center transition-colors focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-1",
                  searchQuery.trim()
                    ? "backdrop-blur-sm bg-foreground/80 text-background hover:bg-foreground/90"
                    : "backdrop-blur-sm bg-muted/40 text-muted-foreground"
                )}
                aria-label="Search"
              >
                {searchLoading ? <Spinner className="h-3.5 w-3.5" /> : <Search className="h-3.5 w-3.5" />}
              </button>
            </div>
          )}
        </div>

        {/* Collection Selector */}
        <div className="ml-auto flex items-center gap-2">
          {collectionsLoading ? (
            <div className="flex items-center gap-2 px-3 py-2 border rounded-md bg-muted min-w-[200px]">
              <Spinner className="h-4 w-4" />
              <span className="text-sm">Loading...</span>
            </div>
          ) : collectionsError ? (
            <div className="px-3 py-2 border border-destructive/50 rounded-md bg-destructive/10 min-w-[200px]">
              <span className="text-sm text-destructive">Error</span>
            </div>
          ) : collections && Object.keys(collections).length > 0 ? (
            <Combobox
              value={selectedCollection || ''}
              onValueChange={(val) => {
                if (val) onCollectionChange(val);
              }}
              items={Object.keys(collections)}
              itemToStringLabel={(key) => collections[key]?.display_name ?? key}
              filter={(key, query) => {
                const name = collections[key]?.display_name ?? key;
                return name.toLowerCase().includes(query.toLowerCase());
              }}
              onInputValueChange={(val) => setCollectionFilter(val)}
            >
              <ComboboxInput
                placeholder="Search collections..."
                className="w-[200px] lg:w-[280px] backdrop-blur-sm bg-transparent dark:bg-transparent [&_input]:bg-transparent [&_input]:dark:bg-transparent"
              />
              <ComboboxContent>
                {collectionFilter.trim() && (
                  <ComboboxEmpty>No collections found.</ComboboxEmpty>
                )}
                <ComboboxList>
                  {(key) => (
                    <ComboboxItem key={key} value={key}>
                      <span className="font-medium">{collections[key]?.display_name ?? key}</span>
                      <span className="text-xs text-muted-foreground ml-2">
                        ({collections[key]?.count.toLocaleString()})
                      </span>
                    </ComboboxItem>
                  )}
                </ComboboxList>
              </ComboboxContent>
            </Combobox>
          ) : null}
          <PageNav variant="glass" saeHref={saeHref} />
          <ModeToggle />
        </div>
      </div>
    </header>
  );
}
