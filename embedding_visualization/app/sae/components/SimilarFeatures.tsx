'use client';

import { useEffect, useMemo } from 'react';
import { useLazyQuery } from '@apollo/client/react';
import { SEMANTIC_SEARCH } from '@/lib/graphql/queries';
import { Spinner } from '@/lib/ui-primitives/spinner';
import { CollapsibleSection } from './CollapsibleSection';
import {
  FeatureSearchResults,
  type SemanticFeatureResult,
  type SelectedFeatureRef,
} from './FeatureSearchResults';

interface SimilarFeaturesProps {
  collectionName: string;
  featureIndex: number;
  featureLabel: string | null;
  onSelectFeature: (index: number, modelId?: string, saeId?: string) => void;
  selectedFeature: SelectedFeatureRef | null;
}

interface SemanticSearchResult {
  id: string;
  document: string | null;
  metadata: Record<string, unknown>;
  similarity: number;
  distance: number;
}

export function SimilarFeatures({
  collectionName,
  featureIndex,
  featureLabel,
  onSelectFeature,
  selectedFeature,
}: SimilarFeaturesProps) {
  const [fetchSimilar, { data, loading }] = useLazyQuery<{
    semanticSearch: SemanticSearchResult[];
  }>(SEMANTIC_SEARCH, { fetchPolicy: 'cache-first' });

  // Fire when label changes
  useEffect(() => {
    if (featureLabel && collectionName) {
      fetchSimilar({
        variables: {
          collectionName,
          query: featureLabel,
          nResults: 11, // +1 for potential self-match
        },
      });
    }
  }, [featureLabel, collectionName, fetchSimilar]);

  const results: SemanticFeatureResult[] = useMemo(() => {
    if (!data?.semanticSearch) return [];
    return data.semanticSearch
      .filter((r) => {
        const idx = r.metadata?.index;
        return idx != null && Number(idx) !== featureIndex;
      })
      .slice(0, 10)
      .map((r) => ({
        featureIndex: Number(r.metadata.index),
        label: r.document ?? null,
        density: (r.metadata?.density as number) ?? null,
        similarity: r.similarity,
      }));
  }, [data, featureIndex]);

  if (!featureLabel) return null;

  return (
    <CollapsibleSection title="Similar Features" count={results.length} defaultOpen>
      {loading ? (
        <div className="flex justify-center py-4">
          <Spinner className="h-4 w-4" />
        </div>
      ) : results.length > 0 ? (
        <FeatureSearchResults
          results={[]}
          onSelect={onSelectFeature}
          selectedFeature={selectedFeature}
          mode="semantic"
          semanticResults={results}
        />
      ) : (
        <p className="text-xs text-muted-foreground">No similar features found.</p>
      )}
    </CollapsibleSection>
  );
}
