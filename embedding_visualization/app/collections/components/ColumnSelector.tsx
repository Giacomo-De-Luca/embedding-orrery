'use client';

import { useMemo } from 'react';
import { Label } from '@/lib/ui-primitives/label';
import { Checkbox } from '@/lib/ui-primitives/checkbox';
import { Textarea } from '@/lib/ui-primitives/textarea';
import { Badge } from '@/lib/ui-primitives/badge';
import { ScrollArea } from '@/lib/ui-primitives/scroll-area';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/lib/ui-primitives/select';
import type { DataType } from '@/lib/graphql/mutations';
import { placeholderIssues, templateFields } from '../lib/taskPrefixes';
import { MEDIA_KIND_LABELS, MediaColumns, type ColumnInfo, type MediaKind } from '../lib/mediaColumns';

interface ColumnSelectorProps {
  columns: ColumnInfo[];
  /** Text columns (MULTIMODAL: the optional text embedded with the media); VECTOR: the vector column */
  selectedEmbeddingColumns: string[];
  selectedMetadataColumns: string[];
  onEmbeddingColumnsChange: (columns: string[]) => void;
  onMetadataColumnsChange: (columns: string[]) => void;
  /** Why the embedding column was pre-selected (useEmbedFormState.selectionNote). */
  selectionNote?: string | null;
  textTemplate: string;
  onTemplateChange: (template: string) => void;
  idColumn: string;
  onIdColumnChange: (column: string) => void;
  dataType?: DataType;
  /** MULTIMODAL: the source's media columns and their kinds */
  mediaKinds?: Record<string, MediaKind>;
  selectedMediaColumns?: string[];
  onMediaColumnsChange?: (columns: string[]) => void;
  /**
   * Hide the metadata pane / ID selector for flows where they have no
   * effect (re-embed carries both over from the source dataset).
   */
  showMetadataColumns?: boolean;
  showIdColumn?: boolean;
  /** Prefix of the element ids, unique per form (several tabs stay mounted). */
  idPrefix?: string;
}

interface ColumnChecklistProps {
  columns: ColumnInfo[];
  selected: string[];
  onChange: (columns: string[]) => void;
  idPrefix: string;
  /** Badge text of a column (defaults to its dtype). */
  badge?: (column: ColumnInfo) => string;
}

/** A scrollable list of column checkboxes, each with a type badge. */
function ColumnChecklist({ columns, selected, onChange, idPrefix, badge }: ColumnChecklistProps) {
  const toggle = (name: string, checked: boolean) =>
    onChange(checked ? [...selected, name] : selected.filter((c) => c !== name));
  return (
    <ScrollArea
      className="border rounded-md [&>[data-radix-scroll-area-viewport]>div]:block!"
      viewportClassName="max-h-60"
    >
      <div className="space-y-2 p-3">
        {columns.map((col) => (
          <div key={col.name} className="flex items-center space-x-2">
            <Checkbox
              id={`${idPrefix}${col.name}`}
              checked={selected.includes(col.name)}
              onCheckedChange={(checked) => toggle(col.name, checked === true)}
            />
            <Label
              htmlFor={`${idPrefix}${col.name}`}
              className="flex-1 font-normal cursor-pointer flex items-center gap-2"
            >
              <span>{col.name}</span>
              <Badge variant="outline" className="text-xs">
                {badge ? badge(col) : col.dtype}
              </Badge>
            </Label>
          </div>
        ))}
      </div>
    </ScrollArea>
  );
}

export function ColumnSelector({
  columns,
  selectedEmbeddingColumns,
  selectedMetadataColumns,
  onEmbeddingColumnsChange,
  onMetadataColumnsChange,
  selectionNote = null,
  textTemplate,
  onTemplateChange,
  idColumn,
  onIdColumnChange,
  dataType = 'TEXT',
  mediaKinds = {},
  selectedMediaColumns = [],
  onMediaColumnsChange,
  showMetadataColumns = true,
  showIdColumn = true,
  idPrefix = '',
}: ColumnSelectorProps) {
  const isVectorMode = dataType === 'VECTOR';
  const isMultimodal = dataType === 'MULTIMODAL';
  // Multimodal: media columns are embedded as media, never as text or
  // metadata (the backend stores each cell's path or placeholder itself).
  const mediaColumns = isMultimodal ? columns.filter((col) => mediaKinds[col.name]) : [];
  const otherColumns = isMultimodal ? columns.filter((col) => !mediaKinds[col.name]) : columns;
  const hasText = selectedEmbeddingColumns.length > 0;

  // Validation: same grammar and rules as the backend (taskPrefixes.ts)
  const templateValidation = useMemo(() => {
    if (!textTemplate) return null;

    const [issue] = placeholderIssues({ template: textTemplate, embeddingColumns: selectedEmbeddingColumns });
    if (issue) return { valid: false, message: issue };
    if (templateFields(textTemplate).fields.length === 0) {
      return { valid: false, message: 'Template should include {column_name} placeholders' };
    }
    return { valid: true, message: null };
  }, [textTemplate, selectedEmbeddingColumns]);

  if (columns.length === 0) {
    return null;
  }

  const mediaSelected = MediaColumns.selectedKinds(selectedMediaColumns, mediaKinds)
    .map((kind) => MEDIA_KIND_LABELS[kind].toLowerCase());

  return (
    <div className="space-y-6">
      {/* Column selection grid */}
      <div className={`grid grid-cols-1 gap-6 ${showMetadataColumns ? 'md:grid-cols-2' : ''}`}>
        <div className="space-y-6">
          {/* Media columns (multimodal) */}
          {isMultimodal && (
            <div className="space-y-3">
              <div>
                <Label className="text-base">Media Columns</Label>
                <p className="text-xs text-muted-foreground mt-1">
                  Image, audio and video columns to embed; a row&apos;s media and text become one input
                </p>
              </div>
              {mediaColumns.length > 0 ? (
                <ColumnChecklist
                  columns={mediaColumns}
                  selected={selectedMediaColumns}
                  onChange={(cols) => onMediaColumnsChange?.(cols)}
                  idPrefix={`${idPrefix}media-`}
                  badge={(col) => MEDIA_KIND_LABELS[mediaKinds[col.name]]}
                />
              ) : (
                <p className="text-xs text-muted-foreground border rounded-md p-3">
                  No image, audio or video columns found. Image, Audio and Video features of
                  HuggingFace datasets are recognised, and local columns holding media bytes or file paths.
                </p>
              )}
              {mediaColumns.length > 0 && selectedMediaColumns.length === 0 && (
                <p className="text-xs text-destructive">Select at least one media column</p>
              )}
            </div>
          )}

          {/* Embedding (text) / Vector column */}
          <div className="space-y-3">
            <div>
              <Label className="text-base">
                {isVectorMode ? 'Vector Column' : isMultimodal ? 'Text Columns (optional)' : 'Embedding Columns'}
              </Label>
              <p className="text-xs text-muted-foreground mt-1">
                {isVectorMode
                  ? 'Column containing pre-computed embedding vectors'
                  : isMultimodal
                    ? 'Text embedded together with the media, e.g. a caption or a title'
                    : 'Columns to combine for embedding text'}
              </p>
            </div>
            {isVectorMode ? (
              <Select
                value={selectedEmbeddingColumns[0] || ''}
                onValueChange={(value) => onEmbeddingColumnsChange([value])}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select vector column..." />
                </SelectTrigger>
                <SelectContent>
                  {columns.map((col) => (
                    <SelectItem key={col.name} value={col.name}>
                      {col.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            ) : (
              <ColumnChecklist
                columns={otherColumns}
                selected={selectedEmbeddingColumns}
                onChange={onEmbeddingColumnsChange}
                idPrefix={`${idPrefix}embed-`}
              />
            )}
            {!isMultimodal && selectedEmbeddingColumns.length === 0 && (
              <p className="text-xs text-destructive">
                {isVectorMode ? 'Select a vector column' : 'Select at least one column'}
              </p>
            )}
            {selectionNote && <p className="text-xs text-muted-foreground">{selectionNote}</p>}
          </div>
        </div>

        {/* Metadata columns */}
        {showMetadataColumns && (
          <div className="space-y-3">
            <div>
              <Label className="text-base">Metadata Columns</Label>
              <p className="text-xs text-muted-foreground mt-1">
                Additional fields to store (categories, labels, etc.)
              </p>
            </div>
            <ColumnChecklist
              columns={otherColumns}
              selected={selectedMetadataColumns}
              onChange={onMetadataColumnsChange}
              idPrefix={`${idPrefix}meta-`}
            />
          </div>
        )}
      </div>

      {/* Column selection info (embed-flow copy — talks about metadata
          duplication rules, which don't apply when the pane is hidden) */}
      {showMetadataColumns && isMultimodal && selectedMediaColumns.length > 0 && (
        <div className="text-xs text-muted-foreground bg-muted p-3 rounded-md">
          Each row embeds its <strong>{mediaSelected.join(', ')}</strong>
          {hasText ? <> with the text of <strong>{selectedEmbeddingColumns.join(', ')}</strong></> : null}
          {' '}as one input. Media cells are stored as their file path, or a short placeholder, in the
          item metadata; rows without text are labelled by their media in the plot.
        </div>
      )}
      {showMetadataColumns && !isMultimodal && selectedEmbeddingColumns.length > 0 && (
        <div className="text-xs text-muted-foreground bg-muted p-3 rounded-md">
          {isVectorMode ? (
            <>
              Vector column: <strong>{selectedEmbeddingColumns[0]}</strong>. Its pre-computed embeddings are loaded directly.
            </>
          ) : selectedEmbeddingColumns.length === 1 ? (
            <>
              Single column selected: <strong>{selectedEmbeddingColumns[0]}</strong> will be embedded and{' '}
              <strong>not</strong> added to metadata (to avoid duplication)
            </>
          ) : (
            <>
              Multiple columns selected: <strong>{selectedEmbeddingColumns.join(', ')}</strong> will be
              embedded and <strong>also added to metadata</strong> to preserve original data
            </>
          )}
        </div>
      )}

      {/* Text template (text, and multimodal rows with text) */}
      {!isVectorMode && (!isMultimodal || hasText) && (
        <div className="space-y-2">
          <Label htmlFor={`${idPrefix}text-template`}>Text Template</Label>
          <Textarea
            id={`${idPrefix}text-template`}
            value={textTemplate}
            onChange={(e) => onTemplateChange(e.target.value)}
            placeholder={
              selectedEmbeddingColumns.length > 0
                ? `Example: {${selectedEmbeddingColumns[0]}}${selectedEmbeddingColumns.length > 1 ? `, {${selectedEmbeddingColumns[1]}}` : ''}`
                : '{column1}, {column2}'
            }
            rows={3}
            className="font-mono text-sm"
          />
          <p className="text-xs text-muted-foreground">
            Use {'{column_name}'} to reference columns. Example: {'{text}: {title}'}
          </p>
          {templateValidation && !templateValidation.valid && (
            <p className="text-xs text-destructive">{templateValidation.message}</p>
          )}
          {templateValidation && templateValidation.valid && (
            <p className="text-xs text-green-600 dark:text-green-400">✓ Template valid</p>
          )}
        </div>
      )}

      {/* ID column selector */}
      {showIdColumn && (
        <div className="space-y-2">
          <Label htmlFor={`${idPrefix}id-column`}>ID Column</Label>
          <Select value={idColumn} onValueChange={onIdColumnChange}>
            <SelectTrigger id={`${idPrefix}id-column`}>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="auto">Auto (sequential IDs)</SelectItem>
              {otherColumns.map((col) => (
                <SelectItem key={col.name} value={col.name}>
                  {col.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <p className="text-xs text-muted-foreground">
            {idColumn === 'auto'
              ? 'IDs will be generated automatically (collection_name_0, collection_name_1, ...)'
              : `Use values from the '${idColumn}' column as unique identifiers`}
          </p>
        </div>
      )}
    </div>
  );
}
