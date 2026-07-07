'use client';

import { useState, useEffect, useCallback } from 'react';
import Link from 'next/link';
import { useQuery } from '@apollo/client/react';
import { toast } from 'sonner';
import { Button, buttonVariants } from '@/lib/ui-primitives/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/lib/ui-primitives/card';
import { Spinner } from '@/lib/ui-primitives/spinner';
import { Label } from '@/lib/ui-primitives/label';
import { Separator } from '@/lib/ui-primitives/separator';
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/lib/ui-primitives/collapsible';
import { ScrollArea, ScrollBar } from '@/lib/ui-primitives/scroll-area';
import { Popover, PopoverContent, PopoverTrigger } from '@/lib/ui-primitives/popover';
import { Trash2, ChevronDown, ChevronRight, ExternalLink } from 'lucide-react';
import type { UpdateCollectionMetadataResult, TopicConfigInput, ExtractTopicsResult, ReduceTopicsInput, ReduceTopicsResult, GenerateLlmLabelsInput, GenerateLlmLabelsResult, ComputeDocumentActivationsResult } from '@/lib/graphql/mutations';
import { GET_COLLECTION_PREVIEW } from '@/lib/graphql/queries';
import { InlineEditableField, SelectOption } from '../InlineEditableField';
import { AddFieldForm } from '../AddFieldForm';
import { TopicExtractionCard } from '../TopicExtractionCard';
import { SaeLinkSection } from '../SaeLinkSection';
import { ProgressModal } from '../EmbeddingProgressModal';
import { DeleteCollectionDialog } from './DeleteCollectionDialog';
import type { CollectionInfo } from '../CollectionManagerTab';

interface CollectionPreviewItem {
  id: string;
  document: string | null;
  metadata: Record<string, unknown> | null;
}

interface CollectionPreviewData {
  embeddings: CollectionPreviewItem[];
}

// Read-only fields that cannot be edited (computed/system)
const READ_ONLY_FIELDS = new Set([
  'embedding_dim',
  'has_projections',
  'pca_2d_variance',
  'pca_3d_variance',
  'hnsw:space',
  'projections_computed_at',
  'created_at',
]);

// Core fields handled separately in the UI
const CORE_FIELDS = new Set([
  'embedding_provider',
  'embedding_model',
]);

// Fields that should show as collapsible (first line + expand chevron)
const EXPANDABLE_FIELDS = new Set([
  'field_analysis',
  'topic_summary',
  'topic_hierarchy',
]);

// Provider options for the select dropdown
const PROVIDER_OPTIONS: SelectOption[] = [
  { value: 'SENTENCE_TRANSFORMERS', label: 'SentenceTransformers' },
  { value: 'OPENAI', label: 'OpenAI' },
  { value: 'COHERE', label: 'Cohere' },
  { value: 'OLLAMA', label: 'Ollama' },
  { value: 'HUGGINGFACE_API', label: 'HuggingFace API' },
];

/**
 * Preview table cell: values truncated at 100 chars open a popover with the
 * full content on click.
 */
function PreviewCell({ value }: { value: unknown }) {
  const shortText = typeof value === 'object'
    ? JSON.stringify(value) ?? ''
    : String(value ?? '');
  const isTruncated = shortText.length > 100;

  if (!isTruncated) {
    return <td className="p-2 max-w-xs truncate">{shortText}</td>;
  }

  const fullText = typeof value === 'object' ? JSON.stringify(value, null, 2) ?? '' : shortText;

  return (
    <td className="p-2 max-w-xs">
      <Popover>
        <PopoverTrigger asChild>
          <button
            className="block w-full truncate text-left cursor-pointer underline decoration-dotted decoration-muted-foreground/50 underline-offset-2 hover:decoration-foreground"
            title="Click to view full content"
          >
            {shortText.slice(0, 100)}...
          </button>
        </PopoverTrigger>
        <PopoverContent align="start" className="w-[480px] max-w-[90vw]">
          <ScrollArea className="[&>[data-radix-scroll-area-viewport]>div]:block!" viewportClassName="max-h-72">
            <pre className="text-xs whitespace-pre-wrap break-words font-mono leading-relaxed">
              {fullText}
            </pre>
          </ScrollArea>
        </PopoverContent>
      </Popover>
    </td>
  );
}

/** Shows just the first line of a long value, with a chevron to expand/collapse the full content. */
function ExpandableMetadataValue({
  fieldKey,
  value,
  isSaving,
  error,
  showDeleteButton,
  onDelete,
}: {
  fieldKey: string;
  value: unknown;
  isSaving: boolean;
  error?: string | null;
  showDeleteButton?: boolean;
  onDelete?: (key: string) => Promise<boolean>;
}) {
  const [expanded, setExpanded] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);

  const fullText = value === null || value === undefined
    ? ''
    : typeof value === 'object'
      ? JSON.stringify(value, null, 2)
      : String(value);

  const firstLine = fullText.split('\n')[0] || fullText.slice(0, 80);
  const isMultiline = fullText.includes('\n') || fullText.length > 80;

  const handleDelete = async (e: React.MouseEvent) => {
    e.stopPropagation();
    if (!onDelete) return;
    setIsDeleting(true);
    try {
      await onDelete(fieldKey);
    } finally {
      setIsDeleting(false);
    }
  };

  return (
    <div className="space-y-1 group">
      <div className="flex items-center justify-between">
        <label className="text-muted-foreground text-xs">{fieldKey}</label>
        {showDeleteButton && onDelete && (
          <Button
            size="sm"
            variant="ghost"
            className="h-6 w-6 p-0 text-muted-foreground hover:text-destructive opacity-40 group-hover:opacity-100 transition-opacity"
            onClick={handleDelete}
            disabled={isDeleting || isSaving}
          >
            {isDeleting ? (
              <Spinner className="h-3 w-3" />
            ) : (
              <Trash2 className="h-3 w-3" />
            )}
          </Button>
        )}
      </div>
      {isSaving ? (
        <div className="flex items-center gap-2 py-1.5 px-2 -mx-2">
          <Spinner className="h-4 w-4" />
        </div>
      ) : (
        <Collapsible open={expanded} onOpenChange={setExpanded}>
          <CollapsibleTrigger asChild>
            <button
              className="flex items-center gap-2 py-1.5 px-2 -mx-2 rounded transition-colors hover:bg-muted/50 w-full text-left cursor-pointer"
            >
              {isMultiline && (
                expanded
                  ? <ChevronDown className="h-3.5 w-3.5 text-muted-foreground flex-shrink-0" />
                  : <ChevronRight className="h-3.5 w-3.5 text-muted-foreground flex-shrink-0" />
              )}
              <span className="font-medium text-sm truncate">
                {firstLine}
                {!expanded && isMultiline && '...'}
              </span>
            </button>
          </CollapsibleTrigger>
          <CollapsibleContent>
            <ScrollArea className="max-h-48 overflow-hidden mt-1 rounded-md border bg-muted/30">
              <pre className="text-xs p-3 whitespace-pre-wrap break-words font-mono leading-relaxed">
                {fullText}
              </pre>
              <ScrollBar orientation="vertical" />
            </ScrollArea>
          </CollapsibleContent>
        </Collapsible>
      )}
      {error && (
        <p className="text-xs text-destructive animate-in fade-in slide-in-from-top-1">
          {error}
        </p>
      )}
    </div>
  );
}

export interface CollectionDetailPaneProps {
  collection: CollectionInfo;
  refreshCollections: () => Promise<void>;
  deleteCollection: (name: string) => Promise<boolean>;
  updateCollectionMetadata: (
    collectionName: string,
    metadata: Record<string, unknown>
  ) => Promise<UpdateCollectionMetadataResult | null>;
  onDeleted: () => void;
  extractTopics: (collectionName: string, config?: TopicConfigInput) => Promise<ExtractTopicsResult | null>;
  topicsLoading: boolean;
  lastTopicsResult: ExtractTopicsResult | null;
  error: string | null;
  clearError: () => void;
  reduceTopics: (input: ReduceTopicsInput) => Promise<ReduceTopicsResult | null>;
  reduceTopicsLoading: boolean;
  lastReduceResult: ReduceTopicsResult | null;
  generateLlmLabels: (input: GenerateLlmLabelsInput) => Promise<GenerateLlmLabelsResult | null>;
  llmLabelsLoading: boolean;
  lastLlmLabelsResult: GenerateLlmLabelsResult | null;
  renameTopicLabel: (collectionName: string, topicId: number, newLabel: string, isSubtopic?: boolean) => Promise<{ error?: string | null } | null>;
  regenerateTopicLabel: (collectionName: string, topicId: number, llmConfig?: string) => Promise<{ error?: string | null; newLabel?: string } | null>;
  fetchCollectionTopics: (collectionName: string) => Promise<ExtractTopicsResult | null>;
  computeDocumentActivations?: (collectionName: string) => Promise<ComputeDocumentActivationsResult | null>;
  docActivationsLoading?: boolean;
  lastDocActivationsResult?: ComputeDocumentActivationsResult | null;
}

/**
 * Detail pane of the Manage tab. Header surfaces the most common actions
 * (view in visualization, delete); below it the preview, metadata editor,
 * SAE sections, and topic extraction follow.
 */
export function CollectionDetailPane({
  collection,
  refreshCollections,
  deleteCollection,
  updateCollectionMetadata,
  onDeleted,
  extractTopics,
  topicsLoading,
  lastTopicsResult,
  error,
  clearError,
  reduceTopics,
  reduceTopicsLoading,
  lastReduceResult,
  generateLlmLabels,
  llmLabelsLoading,
  lastLlmLabelsResult,
  renameTopicLabel,
  regenerateTopicLabel,
  fetchCollectionTopics,
  computeDocumentActivations,
  docActivationsLoading,
  lastDocActivationsResult,
}: CollectionDetailPaneProps) {
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const [detailsOpen, setDetailsOpen] = useState(true);

  // Track saving state and errors per field
  const [savingFields, setSavingFields] = useState<Set<string>>(new Set());
  const [fieldErrors, setFieldErrors] = useState<Record<string, string>>({});

  const collectionName = collection.name;
  const metadata = collection.metadata || {};

  // Fetch collection preview
  const { data: previewData, loading: previewLoading } = useQuery<CollectionPreviewData>(GET_COLLECTION_PREVIEW, {
    variables: { collectionName, limit: 5 },
  });

  const previewItems: CollectionPreviewItem[] = previewData?.embeddings || [];

  // Reset state when the selected collection changes
  useEffect(() => {
    setDeleteError(null);
    setFieldErrors({});
    setDetailsOpen(true);
  }, [collectionName]);

  // Auto-load previously-extracted topics when a collection with topics is selected
  useEffect(() => {
    if (metadata.has_topics && lastTopicsResult?.collectionName !== collectionName) {
      fetchCollectionTopics(collectionName);
    }
  }, [collectionName, metadata.has_topics, lastTopicsResult?.collectionName, fetchCollectionTopics]);

  // Handle saving a single field
  const handleFieldSave = useCallback(async (
    key: string,
    value: unknown
  ): Promise<boolean> => {
    setSavingFields(prev => new Set(prev).add(key));
    setFieldErrors(prev => {
      const { [key]: _removed, ...rest } = prev;
      void _removed;
      return rest;
    });

    try {
      const result = await updateCollectionMetadata(collectionName, {
        [key]: value,
      });

      if (result?.error) {
        setFieldErrors(prev => ({ ...prev, [key]: result.error! }));
        return false;
      }

      await refreshCollections();
      return true;
    } catch (err) {
      setFieldErrors(prev => ({
        ...prev,
        [key]: err instanceof Error ? err.message : 'Save failed',
      }));
      return false;
    } finally {
      setSavingFields(prev => {
        const next = new Set(prev);
        next.delete(key);
        return next;
      });
    }
  }, [collectionName, updateCollectionMetadata, refreshCollections]);

  // Handle deleting a field (set to null)
  const handleFieldDelete = useCallback(async (key: string): Promise<boolean> => {
    return handleFieldSave(key, null); // null signals deletion
  }, [handleFieldSave]);

  // Handle collection deletion (called from the confirmation dialog)
  const handleDelete = useCallback(async () => {
    setDeleteError(null);
    try {
      const success = await deleteCollection(collectionName);
      if (success) {
        toast.success(`Deleted collection "${collectionName}"`);
        await refreshCollections();
        onDeleted();
      } else {
        setDeleteError('Failed to delete collection');
      }
    } catch (err) {
      setDeleteError(err instanceof Error ? err.message : 'Failed to delete collection');
    }
  }, [collectionName, deleteCollection, refreshCollections, onDeleted]);

  // Format metadata value for display
  const formatMetadataValue = (value: unknown): string => {
    if (value === null || value === undefined) return '';
    if (typeof value === 'object') return JSON.stringify(value);
    return String(value);
  };

  const existingMetadataKeys = Object.keys(metadata);

  const readOnlyFields = Object.entries(metadata).filter(
    ([key]) => READ_ONLY_FIELDS.has(key)
  );
  const customFields = Object.entries(metadata).filter(
    ([key]) => !READ_ONLY_FIELDS.has(key) && !CORE_FIELDS.has(key)
  );

  // Get preview table columns from the first item's metadata
  const previewColumns = previewItems.length > 0
    ? ['id', 'document', ...Object.keys(previewItems[0]?.metadata || {}).filter(k => k !== 'row_index')]
    : [];

  return (
    <div className="space-y-6 min-w-0">
      {/* Header: identity + primary actions */}
      <Card>
        <CardHeader>
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div className="min-w-0">
              <CardTitle className="font-mono truncate">{collectionName}</CardTitle>
              <CardDescription>
                {collection.numItems.toLocaleString()} items
                {collection.embeddingModel && ` · ${collection.embeddingModel}`}
              </CardDescription>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <Link
                href={`/?collection=${encodeURIComponent(collectionName)}`}
                className={buttonVariants({ variant: 'outline', size: 'sm' })}
              >
                <ExternalLink className="h-4 w-4 mr-1" />
                View in Visualization
              </Link>
              {!!metadata.has_topics && (
                <Link
                  href={`/?collection=${encodeURIComponent(collectionName)}&colorBy=topic_label`}
                  className={buttonVariants({ variant: 'outline', size: 'sm' })}
                >
                  View by Topics
                </Link>
              )}
              <DeleteCollectionDialog
                collectionName={collectionName}
                numItems={collection.numItems}
                onConfirm={handleDelete}
              />
            </div>
          </div>
          {deleteError && (
            <div className="text-destructive text-sm p-2 bg-destructive/10 rounded">
              {deleteError}
            </div>
          )}
        </CardHeader>
      </Card>

      {/* Data Preview */}
      <Card>
        <CardHeader>
          <CardTitle>Data Preview</CardTitle>
          <CardDescription>First 5 rows from the collection</CardDescription>
        </CardHeader>
        <CardContent>
          {previewLoading ? (
            <div className="flex items-center gap-2 text-muted-foreground py-4">
              <Spinner className="h-4 w-4" />
              <span>Loading preview...</span>
            </div>
          ) : previewItems.length > 0 ? (
            <ScrollArea className="border rounded-md">
              <div className="w-max min-w-full">
                <table className="w-full text-sm">
                  <thead className="bg-muted">
                    <tr>
                      {previewColumns.map((col) => (
                        <th key={col} className="text-left p-2 font-medium whitespace-nowrap">
                          {col}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {previewItems.map((item, i) => (
                      <tr key={item.id || i} className="border-t">
                        {previewColumns.map((col) => {
                          let value: unknown;
                          if (col === 'id') {
                            value = item.id;
                          } else if (col === 'document') {
                            value = item.document;
                          } else {
                            value = item.metadata?.[col];
                          }
                          return <PreviewCell key={col} value={value} />;
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <ScrollBar orientation="horizontal" />
            </ScrollArea>
          ) : (
            <p className="text-muted-foreground text-sm py-4">
              No data available for preview.
            </p>
          )}
        </CardContent>
      </Card>

      {/* Collection Details (Collapsible) */}
      <Collapsible open={detailsOpen} onOpenChange={setDetailsOpen}>
        <Card>
          <CardHeader className="pb-3">
            <CollapsibleTrigger asChild>
              <Button variant="ghost" className="p-0 h-auto hover:bg-transparent justify-start">
                <div className="flex items-center gap-2">
                  {detailsOpen ? (
                    <ChevronDown className="h-4 w-4" />
                  ) : (
                    <ChevronRight className="h-4 w-4" />
                  )}
                  <CardTitle>Collection Details</CardTitle>
                </div>
              </Button>
            </CollapsibleTrigger>
            <CardDescription>
              Click any field to edit it directly
            </CardDescription>
          </CardHeader>
          <CollapsibleContent>
            <CardContent className="space-y-4 pt-0">
              {/* Editable core fields */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <InlineEditableField
                  fieldKey="embedding_provider"
                  label="Embedding Provider"
                  value={collection.embeddingProvider}
                  type="select"
                  selectOptions={PROVIDER_OPTIONS}
                  isSaving={savingFields.has('embedding_provider')}
                  error={fieldErrors['embedding_provider']}
                  onSave={handleFieldSave}
                />

                <InlineEditableField
                  fieldKey="embedding_model"
                  label="Embedding Model"
                  value={collection.embeddingModel}
                  type="text"
                  isSaving={savingFields.has('embedding_model')}
                  error={fieldErrors['embedding_model']}
                  onSave={handleFieldSave}
                />
              </div>

              {/* Custom Metadata Fields */}
              {customFields.length > 0 && (
                <>
                  <Separator />
                  <div>
                    <Label className="text-muted-foreground text-xs mb-3 block">Additional Metadata</Label>
                    <div className="space-y-3">
                      {customFields.map(([key, value]) => (
                        EXPANDABLE_FIELDS.has(key) ? (
                          <ExpandableMetadataValue
                            key={key}
                            fieldKey={key}
                            value={value}
                            isSaving={savingFields.has(key)}
                            error={fieldErrors[key]}
                            showDeleteButton
                            onDelete={handleFieldDelete}
                          />
                        ) : (
                          <div key={key} className="group">
                            <InlineEditableField
                              fieldKey={key}
                              label={key}
                              value={value}
                              type="text"
                              isSaving={savingFields.has(key)}
                              error={fieldErrors[key]}
                              showDeleteButton
                              onSave={handleFieldSave}
                              onDelete={handleFieldDelete}
                            />
                          </div>
                        )
                      ))}
                    </div>
                  </div>
                </>
              )}

              {/* Read-only System Fields */}
              {readOnlyFields.length > 0 && (
                <>
                  <Separator />
                  <div>
                    <Label className="text-muted-foreground text-xs mb-3 block">System Fields (Read-only)</Label>
                    <div className="space-y-3">
                      {readOnlyFields.map(([key, value]) => (
                        <InlineEditableField
                          key={key}
                          fieldKey={key}
                          label={key}
                          value={formatMetadataValue(value)}
                          type="text"
                          readOnly
                          onSave={handleFieldSave}
                        />
                      ))}
                    </div>
                  </div>
                </>
              )}

              {/* Add Field Form */}
              <Separator />
              <AddFieldForm
                existingKeys={existingMetadataKeys}
                onAdd={(key, value) => handleFieldSave(key, value)}
                disabled={savingFields.size > 0}
              />
            </CardContent>
          </CollapsibleContent>
        </Card>
      </Collapsible>

      {/* SAE Link */}
      <SaeLinkSection
        collectionName={collectionName}
        currentModelId={(metadata.sae_model_id as string) ?? null}
        currentSaeId={(metadata.sae_id as string) ?? null}
        onUpdate={async (meta) => { await updateCollectionMetadata(collectionName, meta); await refreshCollections(); }}
      />

      {/* SAE Document Activations */}
      {!!(metadata.sae_model_id && metadata.sae_id) && computeDocumentActivations && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">SAE Document Activations</CardTitle>
            <CardDescription>
              Run SAE inference on all documents to enable feature-based search
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <Button
              onClick={() => computeDocumentActivations(collectionName)}
              disabled={docActivationsLoading}
            >
              {docActivationsLoading && <Spinner className="mr-2 h-4 w-4" />}
              Compute Document Activations
            </Button>
            {lastDocActivationsResult && !lastDocActivationsResult.error && (
              <p className="text-sm text-green-600 dark:text-green-400">
                Processed {lastDocActivationsResult.itemsProcessed} / {lastDocActivationsResult.totalItems} items
                in {lastDocActivationsResult.durationSeconds.toFixed(1)}s
              </p>
            )}
            {lastDocActivationsResult?.error && (
              <p className="text-sm text-destructive">{lastDocActivationsResult.error}</p>
            )}
          </CardContent>
        </Card>
      )}
      {docActivationsLoading && (
        <ProgressModal
          jobId={`${collectionName}_sae_activations`}
          title="Computing SAE Document Activations"
          subtitle="Running SAE inference on each document."
          itemsLabel="documents"
        />
      )}

      {/* Topic Extraction */}
      {!!metadata.has_projections && (
        <TopicExtractionCard
          collectionName={collectionName}
          hasTopics={!!metadata.has_topics}
          topicCount={metadata.topic_count as number | null}
          extractTopics={extractTopics}
          topicsLoading={topicsLoading}
          lastTopicsResult={lastTopicsResult}
          error={error}
          clearError={clearError}
          onTopicsExtracted={refreshCollections}
          reduceTopics={reduceTopics}
          reduceTopicsLoading={reduceTopicsLoading}
          lastReduceResult={lastReduceResult}
          generateLlmLabels={generateLlmLabels}
          llmLabelsLoading={llmLabelsLoading}
          lastLlmLabelsResult={lastLlmLabelsResult}
          hasSubtopics={!!metadata.topic_hierarchy}
          renameTopicLabel={renameTopicLabel}
          regenerateTopicLabel={regenerateTopicLabel}
        />
      )}
    </div>
  );
}
