'use client';

import { useState, useEffect, useCallback } from 'react';
import { useQuery } from '@apollo/client/react';
import { Button } from '@/lib/ui-primitives/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/lib/ui-primitives/card';
import { Spinner } from '@/lib/ui-primitives/spinner';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/lib/ui-primitives/select';
import { Label } from '@/lib/ui-primitives/label';
import { Separator } from '@/lib/ui-primitives/separator';
import { Badge } from '@/lib/ui-primitives/badge';
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/lib/ui-primitives/collapsible';
import { ScrollArea, ScrollBar } from '@/lib/ui-primitives/scroll-area';
import { Trash2, RefreshCw, ChevronDown, ChevronRight } from 'lucide-react';
import type { UpdateCollectionMetadataResult, TopicConfigInput, ExtractTopicsResult, ReduceTopicsInput, ReduceTopicsResult, GenerateLlmLabelsInput, GenerateLlmLabelsResult } from '@/lib/graphql/mutations';
import { GET_COLLECTION_PREVIEW } from '@/lib/graphql/queries';
import { InlineEditableField, SelectOption } from './InlineEditableField';
import { AddFieldForm } from './AddFieldForm';
import { TopicExtractionCard } from './TopicExtractionCard';

export interface CollectionInfo {
  name: string;
  numItems: number;
  embeddingProvider?: string | null;
  embeddingModel?: string | null;
  metadata?: Record<string, unknown>;
}

interface CollectionPreviewItem {
  id: string;
  document: string | null;
  metadata: Record<string, unknown> | null;
}

interface CollectionPreviewData {
  embeddings: CollectionPreviewItem[];
}

interface CollectionManagerTabProps {
  collections: CollectionInfo[];
  collectionsLoading: boolean;
  refreshCollections: () => Promise<void>;
  deleteCollection: (name: string) => Promise<boolean>;
  updateCollectionMetadata: (
    collectionName: string,
    metadata: Record<string, unknown>
  ) => Promise<UpdateCollectionMetadataResult | null>;
  onCollectionDeleted?: () => void;
  extractTopics: (collectionName: string, config?: TopicConfigInput) => Promise<ExtractTopicsResult | null>;
  topicsLoading: boolean;
  lastTopicsResult: ExtractTopicsResult | null;
  error: string | null;
  clearError: () => void;
  // Topic reduction
  reduceTopics: (input: ReduceTopicsInput) => Promise<ReduceTopicsResult | null>;
  reduceTopicsLoading: boolean;
  lastReduceResult: ReduceTopicsResult | null;
  // LLM label generation
  generateLlmLabels: (input: GenerateLlmLabelsInput) => Promise<GenerateLlmLabelsResult | null>;
  llmLabelsLoading: boolean;
  lastLlmLabelsResult: GenerateLlmLabelsResult | null;
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

/** Shows just the first line of a long value, with a chevron to expand/collapse the full content. */
function ExpandableMetadataValue({
  fieldKey,
  value,
  isSaving,
  error,
  showDeleteButton,
  onSave,
  onDelete,
}: {
  fieldKey: string;
  value: unknown;
  isSaving: boolean;
  error?: string | null;
  showDeleteButton?: boolean;
  onSave: (key: string, value: unknown) => Promise<boolean>;
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
            className="h-6 w-6 p-0 text-muted-foreground hover:text-destructive opacity-0 group-hover:opacity-100 transition-opacity"
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

// Provider options for the select dropdown
const PROVIDER_OPTIONS: SelectOption[] = [
  { value: 'SENTENCE_TRANSFORMERS', label: 'SentenceTransformers' },
  { value: 'OPENAI', label: 'OpenAI' },
  { value: 'COHERE', label: 'Cohere' },
  { value: 'OLLAMA', label: 'Ollama' },
  { value: 'HUGGINGFACE_API', label: 'HuggingFace API' },
];

export function CollectionManagerTab({
  collections,
  collectionsLoading,
  refreshCollections,
  deleteCollection,
  updateCollectionMetadata,
  onCollectionDeleted,
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
}: CollectionManagerTabProps) {
  const [selectedCollection, setSelectedCollection] = useState<string | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const [detailsOpen, setDetailsOpen] = useState(true);

  // Track saving state and errors per field
  const [savingFields, setSavingFields] = useState<Set<string>>(new Set());
  const [fieldErrors, setFieldErrors] = useState<Record<string, string>>({});

  // Get selected collection details
  const selectedCollectionInfo = collections.find(c => c.name === selectedCollection);

  // Fetch collection preview when a collection is selected
  const { data: previewData, loading: previewLoading } = useQuery<CollectionPreviewData>(GET_COLLECTION_PREVIEW, {
    variables: { collectionName: selectedCollection, limit: 5 },
    skip: !selectedCollection,
  });

  const previewItems: CollectionPreviewItem[] = previewData?.embeddings || [];

  // Reset state when selection changes
  useEffect(() => {
    setShowDeleteConfirm(false);
    setDeleteError(null);
    setFieldErrors({});
    setDetailsOpen(true);
  }, [selectedCollection]);

  // Handle saving a single field
  const handleFieldSave = useCallback(async (
    key: string,
    value: unknown
  ): Promise<boolean> => {
    if (!selectedCollection) return false;

    setSavingFields(prev => new Set(prev).add(key));
    setFieldErrors(prev => {
      const { [key]: _removed, ...rest } = prev;
      void _removed; // suppress unused variable warning
      return rest;
    });

    try {
      const result = await updateCollectionMetadata(selectedCollection, {
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
  }, [selectedCollection, updateCollectionMetadata, refreshCollections]);

  // Handle deleting a field (set to null)
  const handleFieldDelete = useCallback(async (key: string): Promise<boolean> => {
    if (!selectedCollection) return false;

    setSavingFields(prev => new Set(prev).add(key));
    setFieldErrors(prev => {
      const { [key]: _removed, ...rest } = prev;
      void _removed; // suppress unused variable warning
      return rest;
    });

    try {
      const result = await updateCollectionMetadata(selectedCollection, {
        [key]: null, // null signals deletion
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
        [key]: err instanceof Error ? err.message : 'Delete failed',
      }));
      return false;
    } finally {
      setSavingFields(prev => {
        const next = new Set(prev);
        next.delete(key);
        return next;
      });
    }
  }, [selectedCollection, updateCollectionMetadata, refreshCollections]);

  // Handle adding a new field
  const handleAddField = useCallback(async (
    key: string,
    value: string
  ): Promise<boolean> => {
    return handleFieldSave(key, value);
  }, [handleFieldSave]);

  // Handle collection deletion
  const handleDelete = useCallback(async () => {
    if (!selectedCollection) return;

    setIsDeleting(true);
    setDeleteError(null);

    try {
      const success = await deleteCollection(selectedCollection);
      if (success) {
        setSelectedCollection(null);
        await refreshCollections();
        onCollectionDeleted?.();
      } else {
        setDeleteError('Failed to delete collection');
      }
    } catch (err) {
      setDeleteError(err instanceof Error ? err.message : 'Failed to delete collection');
    } finally {
      setIsDeleting(false);
    }
  }, [selectedCollection, deleteCollection, refreshCollections, onCollectionDeleted]);

  // Format metadata value for display
  const formatMetadataValue = (value: unknown): string => {
    if (value === null || value === undefined) return '';
    if (typeof value === 'object') return JSON.stringify(value);
    return String(value);
  };

  // Get all metadata keys for add field validation
  const existingMetadataKeys = selectedCollectionInfo?.metadata
    ? Object.keys(selectedCollectionInfo.metadata)
    : [];

  // Categorize metadata fields
  const metadata = selectedCollectionInfo?.metadata || {};
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
    <div className="space-y-6">
      {/* Collection Selector */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>Manage Collections</CardTitle>
              <CardDescription>
                View, edit, or delete existing embedding collections
              </CardDescription>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={() => refreshCollections()}
              disabled={collectionsLoading}
            >
              <RefreshCw className={`h-4 w-4 mr-2 ${collectionsLoading ? 'animate-spin' : ''}`} />
              Refresh
            </Button>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="collection-select">Select Collection</Label>
            <Select
              value={selectedCollection || ''}
              onValueChange={(v) => setSelectedCollection(v || null)}
              disabled={collectionsLoading}
            >
              <SelectTrigger id="collection-select">
                <SelectValue placeholder="Choose a collection..." />
              </SelectTrigger>
              <SelectContent>
                {collections.map((collection) => (
                  <SelectItem key={collection.name} value={collection.name}>
                    <div className="flex items-center gap-2">
                      <span>{collection.name}</span>
                      <Badge variant="secondary" className="text-xs">
                        {collection.numItems.toLocaleString()} items
                      </Badge>
                    </div>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {collections.length === 0 && !collectionsLoading && (
            <p className="text-muted-foreground text-sm">
              No collections found. Create one using the HuggingFace or Local File tabs.
            </p>
          )}

          {collectionsLoading && (
            <div className="flex items-center gap-2 text-muted-foreground">
              <Spinner className="h-4 w-4" />
              <span>Loading collections...</span>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Data Preview */}
      {selectedCollectionInfo && (
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
                            const displayValue = typeof value === 'object'
                              ? JSON.stringify(value)?.slice(0, 100)
                              : String(value ?? '').slice(0, 100);
                            return (
                              <td key={col} className="p-2 max-w-xs truncate">
                                {displayValue}
                                {String(value ?? '').length > 100 && '...'}
                              </td>
                            );
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
      )}

      {/* Collection Details (Collapsible) */}
      {selectedCollectionInfo && (
        <Collapsible open={detailsOpen} onOpenChange={setDetailsOpen}>
          <Card>
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between">
                <CollapsibleTrigger asChild>
                  <Button variant="ghost" className="p-0 h-auto hover:bg-transparent">
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
                {!showDeleteConfirm && (
                  <Button
                    variant="destructive"
                    size="sm"
                    disabled={isDeleting}
                    onClick={() => setShowDeleteConfirm(true)}
                  >
                    <Trash2 className="h-4 w-4 mr-2" />
                    Delete
                  </Button>
                )}
              </div>
              <CardDescription>
                Click any field to edit it directly
              </CardDescription>
            </CardHeader>
            <CollapsibleContent>
              <CardContent className="space-y-4 pt-0">
                    {/* Delete Confirmation */}
                    {showDeleteConfirm && (
                      <div className="p-4 border border-destructive rounded-md bg-destructive/5">
                        <h4 className="font-semibold text-destructive mb-2">Delete Collection?</h4>
                        <p className="text-sm mb-4">
                          Are you sure you want to delete <strong>{selectedCollectionInfo.name}</strong>?
                          This will permanently remove all {selectedCollectionInfo.numItems.toLocaleString()} embeddings.
                          This action cannot be undone.
                        </p>
                        <div className="flex gap-2">
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => setShowDeleteConfirm(false)}
                            disabled={isDeleting}
                          >
                            Cancel
                          </Button>
                          <Button
                            variant="destructive"
                            size="sm"
                            onClick={handleDelete}
                            disabled={isDeleting}
                          >
                            {isDeleting ? <Spinner className="h-4 w-4 mr-2" /> : null}
                            Yes, Delete Collection
                          </Button>
                        </div>
                      </div>
                    )}

                    {deleteError && (
                      <div className="text-destructive text-sm p-2 bg-destructive/10 rounded">
                        {deleteError}
                      </div>
                    )}

                    <Separator />

                    {/* Core Fields */}
                    <div className="space-y-4">
                      {/* Read-only core fields */}
                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <Label className="text-muted-foreground text-xs">Name</Label>
                          <p className="font-medium py-1.5">{selectedCollectionInfo.name}</p>
                        </div>
                        <div>
                          <Label className="text-muted-foreground text-xs">Items</Label>
                          <p className="font-medium py-1.5">{selectedCollectionInfo.numItems.toLocaleString()}</p>
                        </div>
                      </div>

                      {/* Editable core fields */}
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <InlineEditableField
                          fieldKey="embedding_provider"
                          label="Embedding Provider"
                          value={selectedCollectionInfo.embeddingProvider}
                          type="select"
                          selectOptions={PROVIDER_OPTIONS}
                          isSaving={savingFields.has('embedding_provider')}
                          error={fieldErrors['embedding_provider']}
                          onSave={handleFieldSave}
                        />

                        <InlineEditableField
                          fieldKey="embedding_model"
                          label="Embedding Model"
                          value={selectedCollectionInfo.embeddingModel}
                          type="text"
                          isSaving={savingFields.has('embedding_model')}
                          error={fieldErrors['embedding_model']}
                          onSave={handleFieldSave}
                        />
                      </div>
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
                                  onSave={handleFieldSave}
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
                      onAdd={handleAddField}
                      disabled={savingFields.size > 0}
                    />
              </CardContent>
            </CollapsibleContent>
          </Card>
        </Collapsible>
      )}

      {/* Topic Extraction */}
      {selectedCollectionInfo && !!metadata.has_projections && (
        <TopicExtractionCard
          collectionName={selectedCollectionInfo.name}
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
        />
      )}

      {/* Quick Links */}
      {selectedCollectionInfo && (
        <Card>
          <CardHeader>
            <CardTitle>Quick Actions</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-2">
              <a
                href={`/?collection=${encodeURIComponent(selectedCollectionInfo.name)}`}
                className="inline-flex items-center justify-center rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:opacity-50 border border-input bg-background hover:bg-accent hover:text-accent-foreground h-9 px-4"
              >
                View in 3D Visualization
              </a>
              {!!metadata.has_topics && (
                <a
                  href={`/?collection=${encodeURIComponent(selectedCollectionInfo.name)}&colorBy=topic_label`}
                  className="inline-flex items-center justify-center rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:opacity-50 border border-input bg-background hover:bg-accent hover:text-accent-foreground h-9 px-4"
                >
                  View by Topics
                </a>
              )}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
