'use client';

import type { SaeFeatureSearchResult } from '@/lib/types/types';
import { Badge } from '@/lib/ui-primitives/badge';
import { parseSaeId, HOOK_TYPE_SHORT } from '@/lib/utils/saeCollections';

export interface SemanticFeatureResult {
  featureIndex: number;
  label: string | null;
  density: number | null;
  similarity: number;
  modelId?: string;
  saeId?: string;
}

interface NormalizedRow {
  featureIndex: number;
  label: string | null;
  value: number | null;
  modelId?: string;
  saeId?: string;
  saeBadge?: string;
}

interface FeatureSearchResultsProps {
  results: SaeFeatureSearchResult[];
  onSelect: (featureIndex: number, modelId?: string, saeId?: string) => void;
  selectedIndex: number | null;
  mode?: 'text' | 'semantic' | 'prompt';
  semanticResults?: SemanticFeatureResult[];
  /** Show an SAE badge per row (for cross-SAE search). */
  showSaeBadge?: boolean;
}

/** Build a compact badge label from a saeId, e.g. "L9 res 16k". */
function saeLabel(saeId: string): string {
  const p = parseSaeId(saeId);
  return `L${p.layerIndex} ${HOOK_TYPE_SHORT[p.hookType] ?? p.hookType} ${p.width}`;
}

/**
 * Table of feature search results. Click a row to navigate to that feature.
 * Supports text mode (shows density) and semantic mode (shows similarity).
 * When showSaeBadge is true, displays which SAE each result belongs to.
 */
export function FeatureSearchResults({
  results,
  onSelect,
  selectedIndex,
  mode = 'text',
  semanticResults,
  showSaeBadge = false,
}: FeatureSearchResultsProps) {
  const isSemanticMode = (mode === 'semantic' || mode === 'prompt') && semanticResults;

  // Normalize rows to a common shape
  const rows: NormalizedRow[] = isSemanticMode
    ? semanticResults.map((r) => ({
        featureIndex: r.featureIndex,
        label: r.label,
        value: r.similarity,
        modelId: r.modelId,
        saeId: r.saeId,
        saeBadge: r.saeId ? saeLabel(r.saeId) : undefined,
      }))
    : results.map((r) => ({
        featureIndex: r.feature.featureIndex,
        label: r.feature.label,
        value: r.feature.density,
        modelId: r.feature.modelId,
        saeId: r.feature.saeId,
        saeBadge: r.feature.saeId ? saeLabel(r.feature.saeId) : undefined,
      }));

  if (rows.length === 0) {
    return <p className="text-sm text-muted-foreground py-4 text-center">No features found.</p>;
  }

  const valueLabel = mode === 'prompt' ? 'Activation' : isSemanticMode ? 'Similarity' : 'Density';

  return (
    <div className="border rounded-md overflow-hidden">
      <table className="w-full text-sm table-fixed">
        <thead>
          <tr className="border-b bg-muted/50">
            <th className="text-left px-3 py-2 font-medium text-xs text-muted-foreground w-14">#</th>
            {showSaeBadge && (
              <th className="text-left px-3 py-2 font-medium text-xs text-muted-foreground w-20">SAE</th>
            )}
            <th className="text-left px-3 py-2 font-medium text-xs text-muted-foreground">Label</th>
            <th className="text-right px-3 py-2 font-medium text-xs text-muted-foreground w-20">
              {valueLabel}
            </th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row, idx) => {
            const isSelected = row.featureIndex === selectedIndex && !showSaeBadge;
            // Use index as part of key since featureIndex may repeat across SAEs
            const rowKey = showSaeBadge
              ? `${row.modelId}::${row.saeId}::${row.featureIndex}`
              : `${row.featureIndex}-${idx}`;
            return (
              <tr
                key={rowKey}
                className={`border-b last:border-0 cursor-pointer transition-colors hover:bg-muted/50 ${
                  isSelected ? 'bg-accent' : ''
                }`}
                onClick={() => onSelect(row.featureIndex, row.modelId, row.saeId)}
              >
                <td className="px-3 py-2">
                  <Badge variant="outline" className="font-mono text-xs">
                    {row.featureIndex}
                  </Badge>
                </td>
                {showSaeBadge && (
                  <td className="px-3 py-2">
                    <Badge variant="secondary" className="font-mono text-[10px]">
                      {row.saeBadge}
                    </Badge>
                  </td>
                )}
                <td className="px-3 py-2 text-xs truncate overflow-hidden" title={row.label ?? ''}>
                  {row.label ?? <span className="text-muted-foreground italic">no label</span>}
                </td>
                <td className="px-3 py-2 text-right font-mono text-xs text-muted-foreground tabular-nums whitespace-nowrap">
                  {row.value != null
                    ? isSemanticMode
                      ? row.value.toFixed(4)
                      : (row.value < 0.001 ? row.value.toExponential(2) : row.value.toFixed(4))
                    : '-'}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
