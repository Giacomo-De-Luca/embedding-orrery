'use client';

import { useMemo, useState, useCallback, useEffect, useRef } from 'react';
import { X } from 'lucide-react';
import { cn } from '@/lib/utils/utils';
import type { LayerActivationsResult, ActiveFeatureResult } from '@/lib/graphql/mutations';

export interface SelectedTokenInfo {
  tokenIdx: number;
  token: string;
  features: ActiveFeatureResult[];
  /** SAE identity of the layer tab the token was selected in. */
  modelId?: string;
  saeId?: string;
}

/** Layer result optionally tagged with its SAE identity (attachSaeIdentity). */
export type TokenStripLayer = LayerActivationsResult & {
  modelId?: string;
  saeId?: string;
};

interface PromptTokenActivationsProps {
  layers: TokenStripLayer[];
  tokenStrings: string[];
  /** Controlled token selection — index within the active layer tab. */
  selectedTokenIdx: number | null;
  /** Called when a token is selected (click) or deselected (click again / tab switch). */
  onTokenSelect?: (info: SelectedTokenInfo | null) => void;
  /** When set, color tokens by this feature's activation instead of max across all. */
  highlightedFeatureIndex?: number | null;
  /** SAE of the highlighted feature — auto-focuses the matching layer tab. */
  highlightedFeatureSaeId?: string | null;
  /** Label of the highlighted feature (for the indicator). */
  highlightedFeatureLabel?: string | null;
  /** Called when the user clears the feature highlight. */
  onClearHighlight?: () => void;
}

/**
 * Interactive token strip with heatmap coloring by activation intensity.
 * Hover shows numeric activation value. Click selects/deselects a token
 * (selection is controlled by the parent via `selectedTokenIdx`).
 *
 * When `highlightedFeatureIndex` is set, colors tokens by that single
 * feature's activation per token (instead of max across all features).
 */
export function PromptTokenActivations({
  layers,
  tokenStrings,
  selectedTokenIdx,
  onTokenSelect,
  highlightedFeatureIndex,
  highlightedFeatureSaeId,
  highlightedFeatureLabel,
  onClearHighlight,
}: PromptTokenActivationsProps) {
  const [selectedLayer, setSelectedLayer] = useState<number>(0);
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);
  const lastAutoFocusRef = useRef<string | null>(null);

  const layerData = layers[selectedLayer];
  const isFeatureHighlighted = highlightedFeatureIndex != null;
  // Single-feature coloring only applies on the tab of the feature's own SAE —
  // the same index in another SAE is an unrelated feature.
  const highlightAppliesToLayer =
    isFeatureHighlighted &&
    (!highlightedFeatureSaeId || layerData?.saeId === highlightedFeatureSaeId);

  // Two widths at one layer are distinct tabs — disambiguate the labels
  const showWidth = useMemo(() => new Set(layers.map((l) => l.width)).size > 1, [layers]);
  const tabLabel = useCallback(
    (l: TokenStripLayer) => (showWidth ? `${l.layer} · ${l.width}` : `${l.layer}`),
    [showWidth],
  );

  // Auto-focus the layer tab matching the highlighted feature's SAE — once
  // per highlight identity, so the user can still switch tabs afterward.
  useEffect(() => {
    if (highlightedFeatureIndex == null || !highlightedFeatureSaeId) {
      lastAutoFocusRef.current = null;
      return;
    }
    const key = `${highlightedFeatureSaeId}::${highlightedFeatureIndex}`;
    if (lastAutoFocusRef.current === key) return;
    lastAutoFocusRef.current = key;
    const idx = layers.findIndex((l) => l.saeId === highlightedFeatureSaeId);
    if (idx >= 0 && idx !== selectedLayer) {
      setSelectedLayer(idx);
      onTokenSelect?.(null);
    }
  }, [highlightedFeatureIndex, highlightedFeatureSaeId, layers, selectedLayer, onTokenSelect]);

  // Per-token activation values: either for a single feature or max across all
  const tokenActivations = useMemo(() => {
    if (!layerData) return [];
    if (highlightAppliesToLayer) {
      return layerData.tokens.map((t) => {
        const feat = t.features.find((f) => f.index === highlightedFeatureIndex);
        return feat?.activation ?? 0;
      });
    }
    return layerData.tokens.map((t) => {
      if (t.features.length === 0) return 0;
      return Math.max(...t.features.map((f) => f.activation));
    });
  }, [layerData, highlightedFeatureIndex, highlightAppliesToLayer]);

  const globalMax = useMemo(() => Math.max(...tokenActivations, 0.01), [tokenActivations]);

  const handleTokenClick = useCallback((i: number) => {
    if (!layerData) return;
    const newIdx = i === selectedTokenIdx ? null : i;
    if (newIdx === null) {
      onTokenSelect?.(null);
    } else {
      const tokenData = layerData.tokens[newIdx];
      if (tokenData) {
        const sorted = [...tokenData.features].sort((a, b) => b.activation - a.activation);
        onTokenSelect?.({
          tokenIdx: newIdx,
          token: tokenStrings[tokenData.position] ?? tokenData.token,
          features: sorted,
          modelId: layerData.modelId,
          saeId: layerData.saeId,
        });
      }
    }
  }, [selectedTokenIdx, layerData, tokenStrings, onTokenSelect]);

  if (!layerData) return null;

  return (
    <div className="space-y-1">
      {/* Layer selector (if multiple layers) */}
      {layers.length > 1 && (
        <div className="flex items-center gap-2">
          <span className="text-[10px] text-muted-foreground">Layer:</span>
          <div className="flex gap-1 flex-wrap">
            {layers.map((l, i) => (
              <button
                key={`${l.layer}-${l.width}`}
                onClick={() => { setSelectedLayer(i); onTokenSelect?.(null); }}
                className={cn(
                  'px-2 py-0.5 text-[10px] rounded border transition-colors',
                  i === selectedLayer
                    ? 'bg-primary text-primary-foreground border-primary'
                    : 'bg-muted/50 text-muted-foreground border-border hover:bg-muted',
                )}
              >
                {tabLabel(l)}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Token strip */}
      <div className="border rounded-md p-2 bg-card">
        <p className="text-[10px] text-muted-foreground mb-1">
          Click a token to see its features (Layer {layerData.layer}
          {showWidth ? `, ${layerData.width}` : ''})
        </p>
        <div className="relative leading-relaxed font-mono text-xs flex flex-wrap">
          {layerData.tokens.map((tokenData, i) => {
            const intensity = globalMax > 0 ? tokenActivations[i] / globalMax : 0;
            const r = 255;
            const g = Math.round(165 - intensity * 100);
            const b = Math.round(50 - intensity * 50);
            const a = Math.min(0.9, intensity * 0.85 + 0.05);
            const bgColor = intensity > 0.01 ? `rgba(${r}, ${g}, ${b}, ${a})` : 'transparent';
            const isSelected = i === selectedTokenIdx;
            const isHovered = i === hoveredIdx;

            return (
              <span
                key={i}
                className={cn(
                  'px-[1px] py-[1px] cursor-pointer transition-all duration-75 whitespace-pre',
                  isSelected && 'ring-2 ring-primary rounded-sm',
                )}
                style={{ backgroundColor: bgColor }}
                onClick={() => handleTokenClick(i)}
                onMouseEnter={() => setHoveredIdx(i)}
                onMouseLeave={() => setHoveredIdx(null)}
              >
                <span className={intensity > 0.5 ? 'text-white dark:text-white' : 'text-foreground'}>
                  {tokenStrings[tokenData.position] ?? tokenData.token}
                </span>
                {isHovered && (
                  <span className="absolute z-10 -mt-7 px-1.5 py-0.5 text-[10px] bg-popover text-popover-foreground border rounded shadow-md whitespace-nowrap pointer-events-none">
                    {tokenActivations[i].toFixed(2)}
                    {highlightAppliesToLayer
                      ? ` — #${highlightedFeatureIndex}`
                      : ` (${tokenData.features.length} features)`}
                  </span>
                )}
              </span>
            );
          })}
        </div>
      </div>

      {/* Mode indicator */}
      <div className="flex items-center gap-1 min-h-[16px]">
        {isFeatureHighlighted ? (
          <>
            <span className="text-[10px] text-muted-foreground truncate">
              Feature #{highlightedFeatureIndex}
              {highlightedFeatureLabel && `: ${highlightedFeatureLabel}`}
            </span>
            {onClearHighlight && (
              <button
                onClick={onClearHighlight}
                className="shrink-0 p-0.5 rounded hover:bg-muted text-muted-foreground hover:text-foreground transition-colors"
                title="Show all features"
              >
                <X className="h-3 w-3" />
              </button>
            )}
          </>
        ) : (
          <span className="text-[10px] text-muted-foreground">
            Max activation across all features
          </span>
        )}
      </div>
    </div>
  );
}
