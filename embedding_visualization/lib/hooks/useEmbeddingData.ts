'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { useLazyQuery, useMutation } from '@apollo/client/react';
import { GET_COLLECTION_DATA } from '../graphql/queries';
import { UPDATE_COLLECTION_METADATA } from '../graphql/mutations';
import type { EmbeddingData, ProjectionMethod, DimensionMode, DisplayConfig } from '../types/types';
import {
  detectDisplayConfig,
  analyzeColorFields,
  type ColorFieldOption,
} from '../utils/fieldAnalysis';

/** Compute smart default tooltip fields based on available collection fields. */
function computeSmartDefaults(availableFields: string[]): string[] {
  const defaults: string[] = [];
  if (availableFields.includes('topic_label')) defaults.push('topic_label');
  if (availableFields.includes('date')) {
    defaults.push('date');
  } else if (availableFields.includes('year')) {
    defaults.push('year');
  }
  return defaults;
}

/** Determine which projection types are needed for the current method + mode. */
function getNeededProjections(method: ProjectionMethod, mode: DimensionMode): string[] {
  const prefix = method === 'umap' ? 'umap' : 'pca'; // manual falls back to pca
  // Always need 2D (used for density clustering). Add 3D if in 3D mode.
  return mode === '2d' ? [`${prefix}_2d`] : [`${prefix}_2d`, `${prefix}_3d`];
}

interface UseEmbeddingDataResult {
  data: EmbeddingData | null;
  loading: boolean;
  error: Error | null;
  colorFieldOptions: ColorFieldOption[];
  defaultTooltipFields: string[];
}

export function useEmbeddingData(
  collectionName: string | null,
  method: ProjectionMethod = 'umap',
  mode: DimensionMode = '3d',
): UseEmbeddingDataResult {
  const [data, setData] = useState<EmbeddingData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);
  const [colorFieldOptions, setColorFieldOptions] = useState<ColorFieldOption[]>([]);
  const [defaultTooltipFields, setDefaultTooltipFields] = useState<string[]>([]);

  // Track accumulated projections across multiple fetches for the same collection.
  // When the user switches method/mode, we fetch only the new projection and merge it here.
  const projectionsRef = useRef<{
    collection: string | null;
    pca_2d: number[][] | null;
    pca_3d: number[][] | null;
    umap_2d: number[][] | null;
    umap_3d: number[][] | null;
  }>({ collection: null, pca_2d: null, pca_3d: null, umap_2d: null, umap_3d: null });

  // Track core (non-projection) data separately so projection fetches don't re-process it
  const coreDataRef = useRef<{
    collection: string | null;
    ids: string[];
    documents: string[];
    itemMetadata: Record<string, unknown>[];
    availableFields: string[];
    metadata: EmbeddingData['metadata'];
    displayConfig: DisplayConfig;
  } | null>(null);

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [executeQuery] = useLazyQuery<{ collection: Record<string, any> }>(GET_COLLECTION_DATA, {
    fetchPolicy: 'no-cache', // We manage caching ourselves via projectionsRef — skip Apollo's InMemoryCache to avoid duplicating 200k items in RAM
  });

  const [updateMetadata] = useMutation(UPDATE_COLLECTION_METADATA);

  // Process a GraphQL response and update state
  const processResponse = useCallback((
    collectionData: Record<string, unknown>,
    requestedProjections: string[],
    isNewCollection: boolean,
  ) => {
    // If this is a new collection, reset projections and parse core data
    if (isNewCollection) {
      projectionsRef.current = {
        collection: collectionData._collectionName as string,
        pca_2d: null, pca_3d: null, umap_2d: null, umap_3d: null,
      };

      const itemMetadata: Record<string, unknown>[] =
        (collectionData.itemMetadata as Record<string, unknown>[]) || [];
      const availableFields: string[] =
        (collectionData.availableFields as string[]) || [];
      const meta = collectionData.metadata as Record<string, unknown>;

      const embeddingMetadata: EmbeddingData['metadata'] = {
        total_items: meta.totalItems as number,
        embedding_dim: meta.embeddingDim as number,
        timestamp: meta.timestamp as string,
        pca_2d_variance: meta.pca2dVariance as number[] | undefined,
        pca_3d_variance: meta.pca3dVariance as number[] | undefined,
        source_dataset: meta.sourceDataset as string | undefined,
        source_split: meta.sourceSplit as string | undefined,
        source_file: meta.sourceFile as string | undefined,
        has_projections: meta.hasProjections as boolean | undefined,
        embedding_provider: meta.embeddingProvider as string | undefined,
        embedding_model: meta.embeddingModel as string | undefined,
        embedding_prompt: meta.embeddingPrompt as string | null | undefined,
      };

      // Phase 1: Determine display config — check field_analysis cache first
      const fieldAnalysis = meta.fieldAnalysis as {
        display_config?: {
          label_field: string | null;
          category_field: string | null;
          category_values: string[];
          category_name: string;
        };
        color_field_options?: ColorFieldOption[];
        default_tooltip_fields?: string[];
      } | null;

      let displayConfig: DisplayConfig;

      if (fieldAnalysis?.display_config) {
        // Cached field analysis — use immediately (instant)
        displayConfig = {
          labelField: fieldAnalysis.display_config.label_field,
          categoryField: fieldAnalysis.display_config.category_field,
          categoryValues: fieldAnalysis.display_config.category_values,
          categoryName: fieldAnalysis.display_config.category_name,
        };
        if (fieldAnalysis.color_field_options) {
          // Sanitize cached options: old caches may have null uniqueCount (from Infinity serialization)
          const sanitized = (fieldAnalysis.color_field_options as ColorFieldOption[]).map(opt => ({
            ...opt,
            uniqueCount: opt.uniqueCount ?? 0,
          }));
          setColorFieldOptions(sanitized);
        }
        if (fieldAnalysis.default_tooltip_fields) {
          setDefaultTooltipFields(fieldAnalysis.default_tooltip_fields);
        }
      } else {
        // No cached field analysis — use minimal defaults now, compute async later
        displayConfig = detectDisplayConfig(availableFields, []);
        // Schedule heavy field analysis as non-blocking (yields to renderer first)
        setTimeout(() => {
          const fieldOptions = analyzeColorFields(availableFields, itemMetadata);
          setColorFieldOptions(fieldOptions);

          const fullDisplayConfig = detectDisplayConfig(availableFields, itemMetadata);
          const smartDefaults = computeSmartDefaults(availableFields);
          setDefaultTooltipFields(smartDefaults);

          // Update data with the full display config
          setData(prev => prev ? { ...prev, displayConfig: fullDisplayConfig } : prev);

          // Persist field analysis for next time (fire and forget)
          const collName = collectionData._collectionName as string;
          if (collName) {
            updateMetadata({
              variables: {
                collectionName: collName,
                metadata: {
                  field_analysis: JSON.stringify({
                    display_config: {
                      label_field: fullDisplayConfig.labelField,
                      category_field: fullDisplayConfig.categoryField,
                      category_values: fullDisplayConfig.categoryValues,
                      category_name: fullDisplayConfig.categoryName,
                    },
                    color_field_options: fieldOptions,
                    default_tooltip_fields: smartDefaults,
                  }),
                },
              },
            }).catch((err: unknown) => console.warn('Failed to persist field analysis:', err));
          }
        }, 0);
      }

      coreDataRef.current = {
        collection: collectionData._collectionName as string,
        ids: collectionData.ids as string[],
        documents: collectionData.documents as string[],
        itemMetadata,
        availableFields,
        metadata: embeddingMetadata,
        displayConfig,
      };
    }

    // Merge newly fetched projections into the accumulated ref
    const projRef = projectionsRef.current;
    for (const pt of requestedProjections) {
      const key = pt as keyof typeof projRef;
      const camelKey = pt.replace(/_(\w)/g, (_, c) => c.toUpperCase()) as
        'pca2d' | 'pca3d' | 'umap2d' | 'umap3d';
      const projData = collectionData[camelKey] as number[][] | null;
      if (projData && key in projRef) {
        (projRef as Record<string, unknown>)[key] = projData;
      }
    }

    // Assemble the full EmbeddingData from core data + accumulated projections
    const core = coreDataRef.current!;
    const embeddingData: EmbeddingData = {
      ids: core.ids,
      documents: core.documents,
      itemMetadata: core.itemMetadata,
      availableFields: core.availableFields,
      displayConfig: core.displayConfig,
      projections: {
        pca_2d: projRef.pca_2d,
        pca_3d: projRef.pca_3d,
        umap_2d: projRef.umap_2d,
        umap_3d: projRef.umap_3d,
      },
      metadata: core.metadata,
    };

    setData(embeddingData);
    setLoading(false);
    setError(null);

    console.log('Loaded collection:', {
      collection: core.collection,
      items: core.ids?.length,
      loadedProjections: Object.entries(projRef)
        .filter(([k, v]) => k !== 'collection' && v !== null)
        .map(([k]) => k),
    });
  }, [updateMetadata]);

  // Fetch projections when collection or needed projections change
  useEffect(() => {
    if (!collectionName) {
      setLoading(false);
      setData(null);
      return;
    }

    const needed = getNeededProjections(method, mode);
    const isNewCollection = projectionsRef.current.collection !== collectionName;

    // Check which projections we actually need to fetch
    const missing = isNewCollection
      ? needed
      : needed.filter(pt => {
          const key = pt as keyof typeof projectionsRef.current;
          return projectionsRef.current[key] === null;
        });

    // If all needed projections are already loaded, just reassemble data
    if (!isNewCollection && missing.length === 0 && coreDataRef.current) {
      // Reassemble from cached data — no network request needed
      const core = coreDataRef.current;
      const projRef = projectionsRef.current;
      setData({
        ids: core.ids,
        documents: core.documents,
        itemMetadata: core.itemMetadata,
        availableFields: core.availableFields,
        displayConfig: core.displayConfig,
        projections: {
          pca_2d: projRef.pca_2d,
          pca_3d: projRef.pca_3d,
          umap_2d: projRef.umap_2d,
          umap_3d: projRef.umap_3d,
        },
        metadata: core.metadata,
      });
      return;
    }

    // Need to fetch — show loading for initial collection load, not for projection additions
    if (isNewCollection) {
      setLoading(true);
      setError(null);
      setColorFieldOptions([]);
      setDefaultTooltipFields([]);
    }

    executeQuery({
      variables: {
        name: collectionName,
        projectionTypes: missing,
      },
    }).then((queryResult) => {
      if (queryResult.error) {
        setError(queryResult.error);
        setLoading(false);
        return;
      }
      if (!queryResult.data?.collection) {
        setError(new Error('No data returned from GraphQL'));
        setLoading(false);
        return;
      }
      processResponse(
        { ...queryResult.data.collection, _collectionName: collectionName },
        missing,
        isNewCollection,
      );
    }).catch((err: unknown) => {
      setError(err instanceof Error ? err : new Error(String(err)));
      setLoading(false);
      console.error('Error loading embedding data:', err);
    });
  }, [collectionName, method, mode, executeQuery, processResponse]);

  return { data, loading, error, colorFieldOptions, defaultTooltipFields };
}
