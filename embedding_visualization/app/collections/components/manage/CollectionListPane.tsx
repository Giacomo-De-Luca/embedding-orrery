'use client';

import { useMemo, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/lib/ui-primitives/card';
import { Input } from '@/lib/ui-primitives/input';
import { Badge } from '@/lib/ui-primitives/badge';
import { Button } from '@/lib/ui-primitives/button';
import { Spinner } from '@/lib/ui-primitives/spinner';
import { ScrollArea } from '@/lib/ui-primitives/scroll-area';
import { RefreshCw, Search, Tags } from 'lucide-react';
import { cn } from '@/lib/utils';
import type { CollectionInfo } from '../CollectionManagerTab';
import { filterCollections, providerShortLabel, formatItemCount } from '../../lib/collectionFilter';

interface CollectionListPaneProps {
  collections: CollectionInfo[];
  collectionsLoading: boolean;
  selectedCollection: string | null;
  onSelectCollection: (name: string | null) => void;
  onRefresh: () => void | Promise<void>;
}

/**
 * Master pane of the Manage tab: searchable, scannable collection list.
 * Selection is controlled by the page (synced to ?collection=).
 */
export function CollectionListPane({
  collections,
  collectionsLoading,
  selectedCollection,
  onSelectCollection,
  onRefresh,
}: CollectionListPaneProps) {
  const [query, setQuery] = useState('');

  const filtered = useMemo(
    () => filterCollections(collections, query),
    [collections, query]
  );

  return (
    <Card className="self-start">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle>Collections</CardTitle>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => onRefresh()}
            disabled={collectionsLoading}
          >
            <RefreshCw className={`h-4 w-4 ${collectionsLoading ? 'animate-spin' : ''}`} />
          </Button>
        </div>
        <div className="relative">
          <Search className="absolute left-2.5 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search collections..."
            className="pl-8 h-8"
          />
        </div>
      </CardHeader>
      <CardContent className="p-2 pt-0">
        {collectionsLoading && collections.length === 0 ? (
          <div className="flex items-center gap-2 text-muted-foreground p-3 text-sm">
            <Spinner className="h-4 w-4" />
            <span>Loading collections...</span>
          </div>
        ) : collections.length === 0 ? (
          <p className="text-muted-foreground text-sm p-3">
            No collections found. Create one using the HuggingFace or Local Files tabs.
          </p>
        ) : filtered.length === 0 ? (
          <p className="text-muted-foreground text-sm p-3">
            No collections match &quot;{query}&quot;.
          </p>
        ) : (
          <ScrollArea className="max-h-[60vh]">
            <div className="space-y-1">
              {filtered.map((collection) => {
                const isSelected = collection.name === selectedCollection;
                const provider = providerShortLabel(collection.embeddingProvider);
                const hasTopics = !!collection.metadata?.has_topics;
                return (
                  <button
                    key={collection.name}
                    onClick={() =>
                      onSelectCollection(isSelected ? null : collection.name)
                    }
                    className={cn(
                      'w-full text-left rounded-md px-3 py-2 transition-colors hover:bg-muted/60 cursor-pointer',
                      isSelected && 'bg-accent hover:bg-accent'
                    )}
                  >
                    <div className="font-mono text-sm truncate">{collection.name}</div>
                    <div className="mt-1 flex flex-wrap items-center gap-1">
                      <Badge variant="secondary" className="text-xs">
                        {formatItemCount(collection.numItems)} items
                      </Badge>
                      {provider && (
                        <Badge variant="outline" className="text-xs">
                          {provider}
                        </Badge>
                      )}
                      {hasTopics && (
                        <Badge variant="outline" className="text-xs gap-1">
                          <Tags className="h-3 w-3" />
                          topics
                        </Badge>
                      )}
                    </div>
                  </button>
                );
              })}
            </div>
          </ScrollArea>
        )}
      </CardContent>
    </Card>
  );
}
