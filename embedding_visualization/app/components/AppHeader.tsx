'use client';

import { Moon, Sun, Search, Settings2, BarChart3, CircleHelp, Github } from 'lucide-react';
import { useState, KeyboardEvent } from 'react';
import { PageNav } from './PageNav';
import { HeaderMenu } from './HeaderMenu';
import { useThemeToggle } from '@/lib/hooks/useThemeToggle';
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
import { Button } from '@/lib/ui-primitives/button';
import { cn } from '@/lib/utils/utils';
import { Input } from '@/lib/ui-primitives/input';
import { REPO_URL } from '@/lib/utils/siteMetadata';
import type { CollectionsManifest } from '../../lib/types/types';
import { getSaeInfo } from '../../lib/utils/saeCollections';
import { textQueryUnavailableReason } from '../../lib/utils/queryPrompt';

function ModeToggle() {
  const { isDark, toggle } = useThemeToggle();

  return (
    <Button
      variant="circular"
      size="icon"
      className="relative"
      onClick={toggle}
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
  /** Reopens the demo welcome dialog; renders the header Help button when set. */
  onOpenIntro?: () => void;
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
  onOpenIntro,
}: AppHeaderProps) {
  const [searchQuery, setSearchQuery] = useState('');
  const [collectionFilter, setCollectionFilter] = useState('');

  // Semantic (text-query) search needs the collection's own model (the backend never falls back).
  const queryUnavailable = textQueryUnavailableReason(
    selectedCollection ? collections?.[selectedCollection] : null,
  );

  const handleSearch = () => {
    if (queryUnavailable) return;
    if (searchQuery.trim() && onSemanticSearch) {
      onSemanticSearch(searchQuery.trim());
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      handleSearch();
    }
  };

  // Context-carrying deep link: keep model/SAE params when the collection has a mapping
  const sae = saeInfoProp ?? getSaeInfo(selectedCollection ?? null);
  const saeHref = sae
    ? `/sae?modelId=${encodeURIComponent(sae.modelId)}&saeId=${encodeURIComponent(sae.saeId)}`
    : undefined;

  // Four cells (toggles, search, collection, actions) on one grid. Placement
  // lives in globals.css (`.explore-header`): it moves the cells per breakpoint
  // and per build, keeps them clear of the HF Space chrome, and publishes the
  // header's height for the panels, legend and modebar below it. The header
  // lets pointer events through to the plot; only the cells capture them.
  return (
    <header className="explore-header pointer-events-none">
      <div data-header-cell="toggles" className="pointer-events-auto flex items-center gap-2">
        <Button
          variant={activePanel === 'controls' ? 'circular' : 'circularghost'}
          size="icon"
          onClick={onToggleControls}
          aria-label="Toggle controls panel"
          title="Visualization controls"
          className="-ml-1"
          data-tour="toggle-controls"
        >
          <Settings2 className="h-4 w-4" />
        </Button>
        <Button
          variant={activePanel === 'search' ? 'circular' : 'circularghost'}
          size="icon"
          onClick={onToggleSearch}
          aria-label="Toggle search panel"
          title="Search & filters"
          data-tour="toggle-search"
        >
          <Search className="h-4 w-4" />
        </Button>
        <Button
          variant={activePanel === 'analytics' ? 'circular' : 'circularghost'}
          size="icon"
          onClick={onToggleAnalytics}
          aria-label="Toggle analytics panel"
          title="Analytics"
          data-tour="toggle-analytics"
        >
          <BarChart3 className="h-4 w-4" />
        </Button>
        {totalWords && embeddingDim && (
          <div className="ml-1 hidden items-center gap-3 lg:flex">
            <Badge variant="secondary" className="backdrop-blur-sm">
              {totalWords.toLocaleString()} points
            </Badge>
            <Badge variant="outline">{embeddingDim}D</Badge>
          </div>
        )}
      </div>

      {/* Semantic Search */}
      {onSemanticSearch && (
        <div
          data-header-cell="search"
          data-tour="search-input"
          className="pointer-events-auto relative w-full sm:max-w-xs"
          // On the cell: a disabled input shows no tooltip.
          title={queryUnavailable ?? undefined}
        >
          <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder={queryUnavailable ? 'Semantic search unavailable' : 'Semantic search...'}
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            className="pl-8 pr-9"
            disabled={searchLoading || queryUnavailable !== null}
          />
          <button
            onClick={handleSearch}
            disabled={!searchQuery.trim() || searchLoading || queryUnavailable !== null}
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

      {/* Collection Selector */}
      <div
        data-header-cell="collection"
        data-tour="collection-selector"
        className="pointer-events-auto flex min-w-0 items-center"
      >
        {collectionsLoading ? (
          <div className="flex w-full items-center gap-2 px-3 py-2 border rounded-md bg-muted sm:w-auto sm:min-w-[200px]">
            <Spinner className="h-4 w-4" />
            <span className="text-sm">Loading...</span>
          </div>
        ) : collectionsError ? (
          <div className="w-full px-3 py-2 border border-destructive/50 rounded-md bg-destructive/10 sm:w-auto sm:min-w-[200px]">
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
              className="w-full sm:w-[200px] lg:w-[280px] backdrop-blur-sm bg-transparent dark:bg-transparent [&_input]:bg-transparent [&_input]:dark:bg-transparent"
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
      </div>

      {/* Help stays visible at every width (it is the demo's way back to the
          mission menu); nav, theme and GitHub fold into a menu below `xl`.
          On phones this cell sits under the panel toggles and the menu leads,
          so it lines up under the first toggle. */}
      <div data-header-cell="actions" className="pointer-events-auto flex items-center gap-2">
        {onOpenIntro && (
          <Button
            variant="circularghost"
            size="icon"
            onClick={onOpenIntro}
            aria-label="About this demo"
            title="About this demo"
            data-tour="intro-button"
          >
            <CircleHelp className="h-4 w-4" />
          </Button>
        )}
        <HeaderMenu saeHref={saeHref} className="xl:hidden max-sm:order-first" />
        <div className="hidden items-center gap-2 xl:flex">
          {/* New tab is required, not cosmetic: inside the HF Space iframe a
              same-tab navigation would load GitHub into the frame, which its
              X-Frame-Options then blanks — taking the demo with it. */}
          <Button
            variant="circularghost"
            size="icon"
            asChild
            aria-label="Source code on GitHub"
            title="Source code on GitHub"
          >
            <a href={REPO_URL} target="_blank" rel="noopener noreferrer">
              <Github className="h-4 w-4" />
            </a>
          </Button>
          <ModeToggle />
          <PageNav variant="glass" saeHref={saeHref} />
        </div>
      </div>
    </header>
  );
}
