'use client';

import { useState, useEffect } from 'react';
import type { EmbeddingData } from '../types/types';
import {
  detectDisplayConfig,
  analyzeColorFields,
  type ColorFieldOption,
} from '../utils/fieldAnalysis';

interface UseEmbeddingDataResult {
  data: EmbeddingData | null;
  loading: boolean;
  error: Error | null;
  colorFieldOptions: ColorFieldOption[];
}

export function useEmbeddingData(collectionName: string | null): UseEmbeddingDataResult {
  const [data, setData] = useState<EmbeddingData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);
  const [colorFieldOptions, setColorFieldOptions] = useState<ColorFieldOption[]>([]);

  useEffect(() => {
    if (!collectionName) {
      setLoading(false);
      return;
    }

    async function loadData() {
      try {
        setLoading(true);
        setError(null);

        // Query GraphQL endpoint - generic structure, no legacy fields
        const response = await fetch('http://localhost:8000/graphql', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            query: `
              query GetCollection($name: String!) {
                collection(name: $name) {
                  ids
                  documents
                  itemMetadata
                  availableFields
                  pca2d
                  pca3d
                  umap2d
                  umap3d
                  metadata {
                    totalItems
                    embeddingDim
                    timestamp
                    pca2dVariance
                    pca3dVariance
                    sourceDataset
                    sourceSplit
                    sourceFile
                    hasProjections
                    embeddingProvider
                    embeddingModel
                    embeddingPrompt
                    embeddingPromptName
                  }
                }
              }
            `,
            variables: { name: collectionName },
          }),
        });

        if (!response.ok) {
          throw new Error(`Failed to load data: ${response.statusText}`);
        }

        const result = await response.json();

        if (result.errors) {
          console.error('GraphQL errors:', result.errors);
          throw new Error(result.errors[0].message);
        }

        if (!result.data?.collection) {
          throw new Error('No data returned from GraphQL');
        }

        const collectionData = result.data.collection;

        // Parse item metadata
        const itemMetadata: Record<string, unknown>[] = collectionData.itemMetadata || [];

        // Detect display configuration dynamically
        const displayConfig = detectDisplayConfig(
          collectionData.availableFields || [],
          itemMetadata
        );

        // Compute color field options with proper type detection
        const fieldOptions = analyzeColorFields(
          collectionData.availableFields || [],
          itemMetadata
        );
        setColorFieldOptions(fieldOptions);

        console.log('Loaded collection:', {
          ids: collectionData.ids?.length,
          documents: collectionData.documents?.length,
          availableFields: collectionData.availableFields,
          displayConfig,
          colorFieldOptions: fieldOptions,
        });

        const embeddingData: EmbeddingData = {
          ids: collectionData.ids,
          documents: collectionData.documents,
          itemMetadata,
          availableFields: collectionData.availableFields || [],
          displayConfig,
          projections: {
            pca_2d: collectionData.pca2d,
            pca_3d: collectionData.pca3d,
            umap_2d: collectionData.umap2d,
            umap_3d: collectionData.umap3d,
          },
          metadata: {
            total_items: collectionData.metadata.totalItems,
            embedding_dim: collectionData.metadata.embeddingDim,
            timestamp: collectionData.metadata.timestamp,
            pca_2d_variance: collectionData.metadata.pca2dVariance,
            pca_3d_variance: collectionData.metadata.pca3dVariance,
            source_dataset: collectionData.metadata.sourceDataset,
            source_split: collectionData.metadata.sourceSplit,
            source_file: collectionData.metadata.sourceFile,
            has_projections: collectionData.metadata.hasProjections,
            embedding_provider: collectionData.metadata.embeddingProvider,
            embedding_model: collectionData.metadata.embeddingModel,
            embedding_prompt: collectionData.metadata.embeddingPrompt,
            embedding_prompt_name: collectionData.metadata.embeddingPromptName,
          },
        };

        setData(embeddingData);
      } catch (err) {
        setError(err as Error);
        console.error('Error loading embedding data:', err);
      } finally {
        setLoading(false);
      }
    }

    loadData();
  }, [collectionName]);

  return { data, loading, error, colorFieldOptions };
}
