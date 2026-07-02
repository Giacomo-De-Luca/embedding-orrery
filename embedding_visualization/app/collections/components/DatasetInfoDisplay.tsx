'use client';

import { Badge } from '@/lib/ui-primitives/badge';
import type { HFDatasetInfo, HFDatasetPreview, LocalFileInfo, LocalFilePreview } from '@/lib/graphql/mutations';

interface DatasetInfoDisplayProps {
  type: 'huggingface' | 'local';
  info: HFDatasetInfo | LocalFileInfo | null;
  preview: HFDatasetPreview | LocalFilePreview | null;
}

export function DatasetInfoDisplay({ type, info, preview }: DatasetInfoDisplayProps) {
  if (!info) return null;

  const isHF = type === 'huggingface';
  const hfInfo = isHF ? (info as HFDatasetInfo) : null;
  const localInfo = !isHF ? (info as LocalFileInfo) : null;

  return (
    <div className="space-y-4">
      {/* Description (HF only) */}
      {hfInfo?.description && (
        <p className="text-sm text-muted-foreground">
          {hfInfo.description.slice(0, 300)}
          {hfInfo.description.length > 300 && '...'}
        </p>
      )}

      {/* Splits info (HF only) */}
      {hfInfo?.configs && hfInfo.configs.length > 0 && (
        <div className="space-y-2">
          <h4 className="text-sm font-medium">Available Splits:</h4>
          <div className="flex flex-wrap gap-2">
            {hfInfo.configs[0].splits.map((split) => (
              <Badge key={split.name} variant="outline">
                {split.name}
                {split.numRows && ` (${split.numRows.toLocaleString()} rows)`}
              </Badge>
            ))}
          </div>
        </div>
      )}

      {/* Features/Columns */}
      <div className="space-y-2">
        <h4 className="text-sm font-medium">
          {isHF ? 'Features:' : 'Columns:'}
        </h4>
        <div className="flex flex-wrap gap-2">
          {isHF && hfInfo?.configs[0]?.features.map((feature) => (
            <Badge
              key={feature.name}
              variant={feature.dtype === 'string' || feature.dtype === 'str' ? 'default' : 'secondary'}
            >
              {feature.name}: {feature.dtype}
            </Badge>
          ))}
          {localInfo?.columns.map((column) => (
            <Badge key={column} variant="secondary">
              {column}
            </Badge>
          ))}
        </div>
      </div>

      {/* File metadata (local only) */}
      {localInfo && (
        <div className="text-xs text-muted-foreground space-y-1">
          <div>File type: {localInfo.fileType}</div>
          <div>Rows: {localInfo.numRows.toLocaleString()}</div>
          <div>Size: {(localInfo.fileSizeBytes / 1024 / 1024).toFixed(2)} MB</div>
        </div>
      )}

      {/* Preview table */}
      {preview && preview.rows.length > 0 && (
        <div className="space-y-2">
          <h4 className="text-sm font-medium">Preview ({preview.rows.length} rows):</h4>
          <div className="overflow-x-auto border rounded-md">
            <table className="w-full text-sm">
              <thead className="bg-muted">
                <tr>
                  {preview.columns.map((col) => (
                    <th key={col} className="text-left p-2 font-medium whitespace-nowrap">
                      {col}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {preview.rows.map((row, i) => (
                  <tr key={i} className="border-t">
                    {preview.columns.map((col) => (
                      <td key={col} className="p-2 max-w-xs truncate">
                        {typeof row[col] === 'object'
                          ? JSON.stringify(row[col]).slice(0, 100)
                          : String(row[col] ?? '').slice(0, 100)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
