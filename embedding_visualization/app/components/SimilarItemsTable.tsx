'use client';

import * as React from 'react';
import {
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  type ColumnDef,
  type SortingState,
  useReactTable,
} from '@tanstack/react-table';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/lib/ui-primitives/table';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/lib/ui-primitives/card';
import { Badge } from '@/lib/ui-primitives/badge';
import { ArrowUpDown } from 'lucide-react';
import { Button } from '@/lib/ui-primitives/button';
import type { SemanticSearchResult } from '../../lib/types/types';
import { getCategoryLabel, getCategoryDisplayName } from '../../lib/utils/categoryColors';

interface SimilarItemsTableProps {
  results: SemanticSearchResult[] | null;
  queryLabel: string | null;
  categoryField?: string | null;
}

// Fields to exclude from dynamic metadata columns
const EXCLUDE_METADATA_FIELDS = new Set([
  'row_index',
  'source_split',
  'source_file',
  'source_dataset',
  // Projection coordinates (internal)
  'pca_2d',
  'pca_3d',
  'umap_2d',
  'umap_3d',
  // Common label fields (already shown in Label column)
  'word',
  'title',
  'name',
  'label',
  'text',
]);

// Convert field name to display name
function fieldToDisplayName(field: string): string {
  if (field === 'pos') return 'Part of Speech';
  return field
    .replace(/_/g, ' ')
    .replace(/([a-z])([A-Z])/g, '$1 $2')
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

export function SimilarItemsTable({ results, queryLabel, categoryField }: SimilarItemsTableProps) {
  const [sorting, setSorting] = React.useState<SortingState>([
    { id: 'similarity', desc: true },
  ]);

  const hasCategory = results?.some(r => r.category && r.category.length > 0);

  // Detect available metadata fields from the first result
  const metadataFields = React.useMemo(() => {
    if (!results || results.length === 0) return [];

    const firstResult = results[0];
    if (!firstResult.metadata) return [];

    return Object.keys(firstResult.metadata).filter(field => {
      // Exclude technical fields and label fields
      if (EXCLUDE_METADATA_FIELDS.has(field)) return false;
      // Exclude the current category field (already shown in Category column)
      if (categoryField && field === categoryField) return false;
      return true;
    });
  }, [results, categoryField]);

  const columns: ColumnDef<SemanticSearchResult>[] = React.useMemo(() => {
    const cols: ColumnDef<SemanticSearchResult>[] = [
      // ID column - shows actual item ID
      {
        accessorKey: 'id',
        header: 'ID',
        cell: ({ row }) => (
          <div className="font-mono text-xs">{row.getValue('id')}</div>
        ),
      },
      // Label column - shows friendly name (word/title/name)
      {
        accessorKey: 'label',
        header: 'Label',
        cell: ({ row }) => (
          <div className="font-medium">{row.getValue('label')}</div>
        ),
      },
      // Similarity column with progress bar
      {
        accessorKey: 'similarity',
        header: ({ column }) => {
          return (
            <Button
              variant="ghost"
              onClick={() => column.toggleSorting(column.getIsSorted() === 'asc')}
              className="-ml-4"
            >
              Similarity
              <ArrowUpDown className="ml-2 h-4 w-4" />
            </Button>
          );
        },
        cell: ({ row }) => {
          const similarity = parseFloat(row.getValue('similarity'));
          return (
            <div className="flex items-center gap-2">
              <div className="w-full max-w-[200px] bg-muted rounded-full h-2">
                <div
                  className="bg-primary h-2 rounded-full transition-all"
                  style={{ width: `${similarity * 100}%` }}
                />
              </div>
              <span className="text-sm font-medium tabular-nums">
                {(similarity * 100).toFixed(1)}%
              </span>
            </div>
          );
        },
      },
    ];

    // Add category column if we have category data
    if (hasCategory) {
      cols.push({
        accessorKey: 'category',
        header: getCategoryDisplayName(categoryField ?? null),
        cell: ({ row }) => {
          const category = row.getValue('category') as string;
          return category ? (
            <Badge variant="outline" className="uppercase">
              {getCategoryLabel(categoryField ?? null, category)}
            </Badge>
          ) : null;
        },
      });
    }

    // Add document/content column
    cols.push({
      accessorKey: 'document',
      header: 'Content',
      cell: ({ row }) => (
        <div className="max-w-[400px] truncate text-sm text-muted-foreground">
          {row.getValue('document')}
        </div>
      ),
    });

    // Add dynamic metadata columns at the end
    for (const field of metadataFields) {
      cols.push({
        id: `metadata_${field}`,
        header: fieldToDisplayName(field),
        cell: ({ row }) => {
          const value = row.original.metadata?.[field];
          if (value === null || value === undefined) {
            return <span className="text-muted-foreground">-</span>;
          }
          // Handle arrays (like "answers" in squad)
          if (Array.isArray(value)) {
            const preview = value.slice(0, 2).map(v =>
              typeof v === 'object' ? JSON.stringify(v) : String(v)
            ).join(', ');
            return (
              <div className="max-w-[200px] truncate text-sm">
                {preview}{value.length > 2 ? ` (+${value.length - 2} more)` : ''}
              </div>
            );
          }
          // Handle objects
          if (typeof value === 'object') {
            return (
              <div className="max-w-[200px] truncate text-sm font-mono">
                {JSON.stringify(value)}
              </div>
            );
          }
          // Handle primitives
          return (
            <div className="max-w-[200px] truncate text-sm">
              {String(value)}
            </div>
          );
        },
      });
    }

    return cols;
  }, [hasCategory, categoryField, metadataFields]);

  const table = useReactTable({
    data: results || [],
    columns,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    onSortingChange: setSorting,
    state: {
      sorting,
    },
  });

  if (!results || results.length === 0) {
    return null;
  }

  return (
    <Card className="min-w-0 backdrop-blur-sm">
      <CardHeader className="flex flex-row items-center gap-4">
        <CardTitle>Similar Items</CardTitle>
        {queryLabel && (
          <CardDescription className="ml-2">
            Items semantically similar to <span className="font-semibold text-foreground">{queryLabel}</span>
          </CardDescription>
        )}
      </CardHeader>
      <CardContent>
        <div className="rounded-md border overflow-x-auto">
          <Table>
            <TableHeader>
              {table.getHeaderGroups().map((headerGroup) => (
                <TableRow key={headerGroup.id}>
                  {headerGroup.headers.map((header) => (
                    <TableHead key={header.id}>
                      {header.isPlaceholder
                        ? null
                        : flexRender(
                            header.column.columnDef.header,
                            header.getContext()
                          )}
                    </TableHead>
                  ))}
                </TableRow>
              ))}
            </TableHeader>
            <TableBody>
              {table.getRowModel().rows?.length ? (
                table.getRowModel().rows.map((row) => (
                  <TableRow
                    key={row.id}
                    data-state={row.getIsSelected() && 'selected'}
                  >
                    {row.getVisibleCells().map((cell) => (
                      <TableCell key={cell.id}>
                        {flexRender(
                          cell.column.columnDef.cell,
                          cell.getContext()
                        )}
                      </TableCell>
                    ))}
                  </TableRow>
                ))
              ) : (
                <TableRow>
                  <TableCell
                    colSpan={columns.length}
                    className="h-24 text-center"
                  >
                    No results.
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </div>
      </CardContent>
    </Card>
  );
}

// Re-export with old name for backwards compatibility
export { SimilarItemsTable as SimilarWordsTable };
