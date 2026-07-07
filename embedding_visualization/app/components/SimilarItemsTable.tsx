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
import { ArrowUpDown, X } from 'lucide-react';
import { Button } from '@/lib/ui-primitives/button';
import type { SemanticSearchResult } from '../../lib/types/types';
import { getCategoryLabel, getCategoryDisplayName } from '../../lib/utils/categoryColors';
import { ScrollArea, ScrollBar } from '@/lib/ui-primitives/scroll-area';

interface SimilarItemsTableProps {
  results: SemanticSearchResult[] | null;
  queryLabel: string | null;
  categoryField?: string | null;
  onClose?: () => void;
  /** When true, column headers adapt for SAE prompt activation results. */
  isActivationResults?: boolean;
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

// Overlay-scrollbar wrapper for tall cell content (replaces native overflow-y-auto,
// whose permanent track looks heavy inside cells). Text styles go on the root and
// inherit; the height cap must be on the viewport or the cell silently stops scrolling.
function CellScroll({ className, children }: { className?: string; children: React.ReactNode }) {
  return (
    <ScrollArea className={className} viewportClassName="max-h-20">
      {children}
    </ScrollArea>
  );
}

// Convert field name to display name
function fieldToDisplayName(field: string): string {
  if (field === 'pos') return 'Part of Speech';
  return field
    .replace(/_/g, ' ')
    .replace(/([a-z])([A-Z])/g, '$1 $2')
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

export function SimilarItemsTable({ results, queryLabel, categoryField, onClose, isActivationResults }: SimilarItemsTableProps) {
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
      // ID column
      {
        accessorKey: 'id',
        header: 'ID',
        size: 130,
        minSize: 80,
        maxSize: 150,
        cell: ({ row }) => (
          <CellScroll className="font-mono text-xs">
            {row.getValue('id')}
          </CellScroll>
        ),
      },
      // Label column
      {
        accessorKey: 'label',
        header: 'Label',
        size: 150,
        minSize: 100,
        maxSize: 250,
        cell: ({ row }) => (
          <CellScroll className="font-medium">
            {row.getValue('label')}
          </CellScroll>
        ),
      },
      // Similarity column with progress bar
      {
        accessorKey: 'similarity',
        size: 200,
        minSize: 180,
        maxSize: 250,
        header: ({ column }) => {
          return (
            <Button
              variant="ghost"
              onClick={() => column.toggleSorting(column.getIsSorted() === 'asc')}
              className="-ml-4"
            >
              {isActivationResults ? 'Activation' : 'Similarity'}
              <ArrowUpDown className="ml-2 h-4 w-4" />
            </Button>
          );
        },
        cell: ({ row }) => {
          const similarity = parseFloat(row.getValue('similarity'));
          return (
            <div className="flex items-center gap-2">
              <div className="flex-1 bg-muted rounded-full h-2 min-w-[60px]">
                <div
                  className="bg-primary h-2 rounded-full transition-all"
                  style={{ width: `${similarity * 100}%` }}
                />
              </div>
              <span className="text-sm font-medium tabular-nums whitespace-nowrap">
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
        size: 120,
        minSize: 80,
        maxSize: 150,
        cell: ({ row }) => {
          const category = row.getValue('category') as string;
          return category ? (
            <CellScroll>
              <Badge variant="outline" className="uppercase">
                {getCategoryLabel(categoryField ?? null, category)}
              </Badge>
            </CellScroll>
          ) : null;
        },
      });
    }

    // Add document/content column - allows text wrapping
    cols.push({
      accessorKey: 'document',
      header: 'Content',
      size: 300,
      minSize: 200,
      maxSize: 500,
      cell: ({ row }) => (
        <CellScroll className="text-sm text-muted-foreground whitespace-normal">
          {row.getValue('document')}
        </CellScroll>
      ),
    });

    // Add dynamic metadata columns at the end
    for (const field of metadataFields) {
      cols.push({
        id: `metadata_${field}`,
        header: fieldToDisplayName(field),
        size: 180,
        minSize: 150,
        maxSize: 350,
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
              <CellScroll className="text-sm whitespace-normal">
                {preview}{value.length > 2 ? ` (+${value.length - 2} more)` : ''}
              </CellScroll>
            );
          }
          // Handle objects
          if (typeof value === 'object') {
            return (
              <CellScroll className="text-sm font-mono whitespace-normal">
                {JSON.stringify(value)}
              </CellScroll>
            );
          }
          // Handle primitives
          return (
            <CellScroll className="text-sm whitespace-normal">
              {String(value)}
            </CellScroll>
          );
        },
      });
    }

    return cols;
  }, [hasCategory, categoryField, metadataFields]);

  const table = useReactTable({
    data: results || [],
    columns,
    columnResizeMode: 'onChange',
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    onSortingChange: setSorting,
    state: {
      sorting,
    },
  });

  const headerRef = React.useRef<HTMLDivElement>(null);
  const tableWrapRef = React.useRef<HTMLDivElement>(null);

  // Keep the header description aligned over the second table column. The table uses
  // table-layout auto, so the ID column can render wider than its configured size —
  // measure the real width and set the title cell's min-width accordingly.
  React.useLayoutEffect(() => {
    const headerEl = headerRef.current;
    const wrapEl = tableWrapRef.current;
    if (!headerEl || !wrapEl) return;
    const firstTh = wrapEl.querySelector<HTMLTableCellElement>('thead th');
    if (!firstTh) return;

    const update = () => {
      // 1 = table wrapper border, 8 = cell px-2, 32 = header pl-8, 16 = header gap-4
      const labelTextLeft = wrapEl.getBoundingClientRect().left + 1 + firstTh.offsetWidth + 8;
      const titleLeft = headerEl.getBoundingClientRect().left + 32;
      headerEl.style.setProperty('--title-col', `${Math.max(labelTextLeft - titleLeft - 16, 0)}px`);
    };
    update();
    const ro = new ResizeObserver(update);
    ro.observe(firstTh);
    ro.observe(wrapEl);
    return () => ro.disconnect();
  }, [columns, results]);

  if (!results || results.length === 0) {
    return null;
  }

  return (
    <Card className="h-full flex flex-col min-w-0 backdrop-blur-sm py-3 gap-2">
      {/* pl-8 puts the title on the ID cell text; the title cell's min-width is set by
          the measurement effect (--title-col) so the description starts over the second
          column's cell text, tracking the ID column's real rendered width. */}
      <CardHeader ref={headerRef} className="grid grid-cols-[max-content_1fr_auto] grid-rows-1 items-center gap-4 shrink-0 pl-8">
        <CardTitle className="text-base leading-5 py-1 min-w-(--title-col,115px)">{isActivationResults ? 'Top Activated Features' : 'Similar Items'}</CardTitle>
        {queryLabel && (
          <CardDescription>
            {isActivationResults
              ? <>Features activated by <span className="font-semibold text-foreground">&ldquo;{queryLabel}&rdquo;</span></>
              : <>Items semantically similar to <span className="font-semibold text-foreground">{queryLabel}</span></>
            }
          </CardDescription>
        )}
        {onClose && (
          <Button variant="ghost" size="icon" className="ml-auto h-7 w-7" onClick={onClose}>
            <X className="h-4 w-4" />
            <span className="sr-only">Close results</span>
          </Button>
        )}
      </CardHeader>
      <CardContent className="flex-1 min-h-0">
        <div ref={tableWrapRef} className="h-full rounded-md border overflow-hidden">
          <ScrollArea className="h-full">
            <Table style={{ minWidth: table.getTotalSize() }}>
              <TableHeader>
                {table.getHeaderGroups().map((headerGroup) => (
                  <TableRow key={headerGroup.id}>
                    {headerGroup.headers.map((header) => (
                      <TableHead
                        key={header.id}
                        className="relative"
                        style={{
                          width: header.getSize(),
                          minWidth: header.column.columnDef.minSize,
                          maxWidth: header.column.columnDef.maxSize,
                        }}
                      >
                        {header.isPlaceholder
                          ? null
                          : flexRender(
                              header.column.columnDef.header,
                              header.getContext()
                            )}
                        {/* Resize handle */}
                        <div
                          onDoubleClick={() => header.column.resetSize()}
                          onMouseDown={header.getResizeHandler()}
                          onTouchStart={header.getResizeHandler()}
                          className={`table-resizer ${header.column.getIsResizing() ? 'isResizing' : ''}`}
                        />
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
                        <TableCell
                          key={cell.id}
                          className="align-top whitespace-normal"
                          style={{
                            width: cell.column.getSize(),
                            minWidth: cell.column.columnDef.minSize,
                            maxWidth: cell.column.columnDef.maxSize,
                          }}
                        >
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
            <ScrollBar orientation="horizontal" />
          </ScrollArea>
        </div>
      </CardContent>
    </Card>
  );
}
