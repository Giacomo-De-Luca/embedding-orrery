'use client';

import { Badge } from '@/lib/ui-primitives/badge';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/lib/ui-primitives/select';
import type { SaeSelectors } from '../hooks/useSaeSelectors';
import { HOOK_TYPE_DISPLAY } from '@/lib/utils/saeCollections';

const ALL_SENTINEL = '__ALL__';

interface SaeSelectorProps {
  selectors: SaeSelectors;
  modelOptions: string[];
  layerOptions: string[];
  hookTypeOptions: string[];
  widthOptions: string[];
  onModelChange: (v: string | null) => void;
  onLayerChange: (v: string | null) => void;
  onHookTypeChange: (v: string | null) => void;
  onWidthChange: (v: string | null) => void;
  resolvedCount: number;
}

function toValue(v: string | null): string {
  return v ?? ALL_SENTINEL;
}

function fromValue(v: string): string | null {
  return v === ALL_SENTINEL ? null : v;
}

/**
 * Four cascading selectors for SAE dimensions: Model, Layer, Hook, Width.
 * Each has an "All" option. Shows a badge with the number of resolved SAEs.
 */
export function SaeSelector({
  selectors,
  modelOptions,
  layerOptions,
  hookTypeOptions,
  widthOptions,
  onModelChange,
  onLayerChange,
  onHookTypeChange,
  onWidthChange,
  resolvedCount,
}: SaeSelectorProps) {
  return (
    <div className="flex items-center gap-2 flex-wrap">
      {/* Model */}
      <Select value={toValue(selectors.model)} onValueChange={(v) => onModelChange(fromValue(v))}>
        <SelectTrigger className="w-44 h-8 text-xs">
          <SelectValue placeholder="Model" />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value={ALL_SENTINEL}>
            <span className="text-muted-foreground">All models</span>
          </SelectItem>
          {modelOptions.map((m) => (
            <SelectItem key={m} value={m}>
              <span className="font-mono text-xs">{m}</span>
            </SelectItem>
          ))}
        </SelectContent>
      </Select>

      {/* Layer */}
      <Select value={toValue(selectors.layer)} onValueChange={(v) => onLayerChange(fromValue(v))}>
        <SelectTrigger className="w-28 h-8 text-xs">
          <SelectValue placeholder="Layer" />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value={ALL_SENTINEL}>
            <span className="text-muted-foreground">All layers</span>
          </SelectItem>
          {layerOptions.map((l) => (
            <SelectItem key={l} value={l}>
              <span className="font-mono text-xs">Layer {l}</span>
            </SelectItem>
          ))}
        </SelectContent>
      </Select>

      {/* Hook type */}
      <Select value={toValue(selectors.hookType)} onValueChange={(v) => onHookTypeChange(fromValue(v))}>
        <SelectTrigger className="w-32 h-8 text-xs">
          <SelectValue placeholder="Hook" />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value={ALL_SENTINEL}>
            <span className="text-muted-foreground">All hooks</span>
          </SelectItem>
          {hookTypeOptions.map((h) => (
            <SelectItem key={h} value={h}>
              <span className="text-xs">{HOOK_TYPE_DISPLAY[h as keyof typeof HOOK_TYPE_DISPLAY] ?? h}</span>
            </SelectItem>
          ))}
        </SelectContent>
      </Select>

      {/* Width */}
      <Select value={toValue(selectors.width)} onValueChange={(v) => onWidthChange(fromValue(v))}>
        <SelectTrigger className="w-24 h-8 text-xs">
          <SelectValue placeholder="Width" />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value={ALL_SENTINEL}>
            <span className="text-muted-foreground">All</span>
          </SelectItem>
          {widthOptions.map((w) => (
            <SelectItem key={w} value={w}>
              <span className="font-mono text-xs">{w}</span>
            </SelectItem>
          ))}
        </SelectContent>
      </Select>

      {/* Resolved SAE count badge */}
      <Badge variant="secondary" className="text-[10px] shrink-0">
        {resolvedCount === 1 ? '1 SAE' : `${resolvedCount} SAEs`}
      </Badge>
    </div>
  );
}
