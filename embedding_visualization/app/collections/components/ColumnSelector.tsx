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
import { placeholderIssues, templateFields } from '../lib/taskPrefixes';
import { TaskPrefixField, type TaskPrefixControl } from './TaskPrefixField';

interface ColumnInfo {
  name: string;
  dtype: string;
}

interface ColumnSelectorProps {
  columns: ColumnInfo[];
  selectedEmbeddingColumns: string[];
  selectedMetadataColumns: string[];
  onEmbeddingColumnsChange: (columns: string[]) => void;
  onMetadataColumnsChange: (columns: string[]) => void;
  textTemplate: string;
  onTemplateChange: (template: string) => void;
  idColumn: string;
  onIdColumnChange: (column: string) => void;
  dataType?: 'TEXT' | 'IMAGE' | 'VECTOR';
  /**
   * Hide the metadata pane / ID selector for flows where they have no
   * effect (re-embed carries both over from the source dataset).
   */
  showMetadataColumns?: boolean;
  showIdColumn?: boolean;
  /** Task prefix of the selected model (text mode); its placeholders may name any column. */
  taskPrefix?: TaskPrefixControl;
}

export function ColumnSelector({
  columns,
  selectedEmbeddingColumns,
  selectedMetadataColumns,
  onEmbeddingColumnsChange,
  onMetadataColumnsChange,
  textTemplate,
  onTemplateChange,
  idColumn,
  onIdColumnChange,
  dataType = 'TEXT',
  showMetadataColumns = true,
  showIdColumn = true,
  taskPrefix,
}: ColumnSelectorProps) {
  const isVectorMode = dataType === 'VECTOR';
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

  const handleEmbeddingColumnToggle = (columnName: string, checked: boolean) => {
    if (checked) {
      onEmbeddingColumnsChange([...selectedEmbeddingColumns, columnName]);
    } else {
      onEmbeddingColumnsChange(selectedEmbeddingColumns.filter(c => c !== columnName));
    }
  };

  const handleMetadataColumnToggle = (columnName: string, checked: boolean) => {
    if (checked) {
      onMetadataColumnsChange([...selectedMetadataColumns, columnName]);
    } else {
      onMetadataColumnsChange(selectedMetadataColumns.filter(c => c !== columnName));
    }
  };

  if (columns.length === 0) {
    return null;
  }

  return (
    <div className="space-y-6">
      {/* Column selection grid */}
      <div className={`grid grid-cols-1 gap-6 ${showMetadataColumns ? 'md:grid-cols-2' : ''}`}>
        {/* Embedding / Vector column */}
        <div className="space-y-3">
          <div>
            <Label className="text-base">{isVectorMode ? 'Vector Column' : 'Embedding Columns'}</Label>
            <p className="text-xs text-muted-foreground mt-1">
              {isVectorMode
                ? 'Column containing pre-computed embedding vectors'
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
            <ScrollArea
              className="border rounded-md [&>[data-radix-scroll-area-viewport]>div]:block!"
              viewportClassName="max-h-60"
            >
              <div className="space-y-2 p-3">
                {columns.map((col) => (
                  <div key={col.name} className="flex items-center space-x-2">
                    <Checkbox
                      id={`embed-${col.name}`}
                      checked={selectedEmbeddingColumns.includes(col.name)}
                      onCheckedChange={(checked) => handleEmbeddingColumnToggle(col.name, checked as boolean)}
                    />
                    <Label
                      htmlFor={`embed-${col.name}`}
                      className="flex-1 font-normal cursor-pointer flex items-center gap-2"
                    >
                      <span>{col.name}</span>
                      <Badge variant="outline" className="text-xs">
                        {col.dtype}
                      </Badge>
                    </Label>
                  </div>
                ))}
              </div>
            </ScrollArea>
          )}
          {selectedEmbeddingColumns.length === 0 && (
            <p className="text-xs text-destructive">
              {isVectorMode ? 'Select a vector column' : 'Select at least one column'}
            </p>
          )}
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
            <ScrollArea
              className="border rounded-md [&>[data-radix-scroll-area-viewport]>div]:block!"
              viewportClassName="max-h-60"
            >
              <div className="space-y-2 p-3">
                {columns.map((col) => (
                  <div key={col.name} className="flex items-center space-x-2">
                    <Checkbox
                      id={`meta-${col.name}`}
                      checked={selectedMetadataColumns.includes(col.name)}
                      onCheckedChange={(checked) => handleMetadataColumnToggle(col.name, checked as boolean)}
                    />
                    <Label
                      htmlFor={`meta-${col.name}`}
                      className="flex-1 font-normal cursor-pointer flex items-center gap-2"
                    >
                      <span>{col.name}</span>
                      <Badge variant="outline" className="text-xs">
                        {col.dtype}
                      </Badge>
                    </Label>
                  </div>
                ))}
              </div>
            </ScrollArea>
          </div>
        )}
      </div>

      {/* Column selection info (embed-flow copy — talks about metadata
          duplication rules, which don't apply when the pane is hidden) */}
      {showMetadataColumns && selectedEmbeddingColumns.length > 0 && (
        <div className="text-xs text-muted-foreground bg-muted p-3 rounded-md">
          {isVectorMode ? (
            <>
              Vector column: <strong>{selectedEmbeddingColumns[0]}</strong> — pre-computed embeddings will be loaded directly
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

      {/* Task prefix of the model, before the text it prefixes (TEXT mode only) */}
      {!isVectorMode && taskPrefix && (
        <div className="space-y-2">
          <Label className="text-base">Task Prefix</Label>
          {taskPrefix.info && taskPrefix.value ? (
            <TaskPrefixField
              info={taskPrefix.info}
              value={taskPrefix.value}
              onChange={taskPrefix.onChange}
              columns={columns.map((col) => col.name)}
              template={textTemplate}
              embeddingColumns={selectedEmbeddingColumns}
            />
          ) : (
            <p className="text-xs text-muted-foreground">
              Not used by the selected model. EmbeddingGemma and Gemini Embedding 2 read their task
              from a prefix: pick one of them under Embedding Model to set it here.
            </p>
          )}
        </div>
      )}

      {/* Text template (TEXT mode only) */}
      {!isVectorMode && (
        <div className="space-y-2">
          <Label htmlFor="text-template">Text Template</Label>
          <Textarea
            id="text-template"
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
          <Label htmlFor="id-column">ID Column</Label>
          <Select value={idColumn} onValueChange={onIdColumnChange}>
            <SelectTrigger id="id-column">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="auto">Auto (sequential IDs)</SelectItem>
              {columns.map((col) => (
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
