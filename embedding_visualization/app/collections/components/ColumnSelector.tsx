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
}: ColumnSelectorProps) {
  const isVectorMode = dataType === 'VECTOR';
  // Validation: check if template references non-selected columns
  const templateValidation = useMemo(() => {
    if (!textTemplate) return null;

    const matches = textTemplate.match(/\{(\w+)\}/g);
    if (!matches) {
      return { valid: false, message: 'Template should include {column_name} placeholders' };
    }

    const referencedColumns = matches.map(m => m.slice(1, -1));
    const invalidColumns = referencedColumns.filter(col => !selectedEmbeddingColumns.includes(col));

    if (invalidColumns.length > 0) {
      return {
        valid: false,
        message: `Template references non-selected columns: ${invalidColumns.join(', ')}`,
      };
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
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
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
            <ScrollArea className="space-y-2 max-h-60 overflow-y-auto border rounded-md p-3">
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
            </ScrollArea>
          )}
          {selectedEmbeddingColumns.length === 0 && (
            <p className="text-xs text-destructive">
              {isVectorMode ? 'Select a vector column' : 'Select at least one column'}
            </p>
          )}
        </div>

        {/* Metadata columns */}
        <div className="space-y-3">
          <div>
            <Label className="text-base">Metadata Columns</Label>
            <p className="text-xs text-muted-foreground mt-1">
              Additional fields to store (categories, labels, etc.)
            </p>
          </div>
          <div className="space-y-2 max-h-60 overflow-y-auto border rounded-md p-3">
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
        </div>
      </div>

      {/* Column selection info */}
      {selectedEmbeddingColumns.length > 0 && (
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
    </div>
  );
}
