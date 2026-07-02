'use client';

import { useState } from 'react';
import { ChevronDown, ChevronRight } from 'lucide-react';
import type { SaeActivation, SaeActivationQuantileGroup } from '@/lib/types/types';
import { TokenStrip } from './TokenStrip';
import { ToggleGroup, ToggleGroupItem } from '@/lib/ui-primitives/toggle-group';

interface ActivationExamplesProps {
  activations: SaeActivation[];
  quantileGroups?: SaeActivationQuantileGroup[];
  quantileLoading?: boolean;
  onRequestQuantiles?: () => void;
  onHoverActivation?: (value: number | null) => void;
}

/**
 * Renders activation examples as token strip heatmaps.
 * Supports "Top only" (flat list) and "All quantiles" (grouped by activation level).
 */
export function ActivationExamples({
  activations,
  quantileGroups,
  quantileLoading,
  onRequestQuantiles,
  onHoverActivation,
}: ActivationExamplesProps) {
  const [viewMode, setViewMode] = useState<'top' | 'quantiles'>('top');

  const handleModeChange = (mode: string) => {
    if (!mode) return;
    setViewMode(mode as 'top' | 'quantiles');
    if (mode === 'quantiles' && !quantileGroups && onRequestQuantiles) {
      onRequestQuantiles();
    }
  };

  if (activations.length === 0 && (!quantileGroups || quantileGroups.length === 0)) {
    return <p className="text-sm text-muted-foreground">No activation examples available.</p>;
  }

  const globalMax = Math.max(
    ...activations.map((a) => a.maxValue),
    ...(quantileGroups?.flatMap((g) => g.activations.map((a) => a.maxValue)) ?? []),
  );

  return (
    <div className="space-y-2">
      {onRequestQuantiles && (
        <ToggleGroup
          type="single"
          value={viewMode}
          onValueChange={handleModeChange}
          variant="outline"
          className="justify-start"
        >
          <ToggleGroupItem value="top" className="text-xs h-7 px-2">Top only</ToggleGroupItem>
          <ToggleGroupItem value="quantiles" className="text-xs h-7 px-2">All quantiles</ToggleGroupItem>
        </ToggleGroup>
      )}

      {viewMode === 'top' ? (
        <ActivationList activations={activations} globalMax={globalMax} onHoverActivation={onHoverActivation} />
      ) : quantileLoading ? (
        <p className="text-xs text-muted-foreground py-4 text-center">Loading quantile data...</p>
      ) : quantileGroups && quantileGroups.length > 0 ? (
        <QuantileGroupList groups={quantileGroups} globalMax={globalMax} onHoverActivation={onHoverActivation} />
      ) : (
        <ActivationList activations={activations} globalMax={globalMax} onHoverActivation={onHoverActivation} />
      )}
    </div>
  );
}

function ActivationList({
  activations,
  globalMax,
  onHoverActivation,
}: {
  activations: SaeActivation[];
  globalMax: number;
  onHoverActivation?: (value: number | null) => void;
}) {
  return (
    <div className="space-y-3">
      {activations.map((act) => (
        <ActivationCard key={act.id} act={act} globalMax={globalMax} onHoverActivation={onHoverActivation} />
      ))}
    </div>
  );
}

function QuantileGroupList({
  groups,
  globalMax,
  onHoverActivation,
}: {
  groups: SaeActivationQuantileGroup[];
  globalMax: number;
  onHoverActivation?: (value: number | null) => void;
}) {
  const nGroups = groups.length;

  return (
    <div className="space-y-2">
      {groups.map((group) => {
        const pctHigh = Math.round(100 - ((group.quantile - 1) / nGroups) * 100);
        const pctLow = Math.round(100 - (group.quantile / nGroups) * 100);
        const label = `${pctLow}th–${pctHigh}th percentile`;
        const rangeLabel = `(${group.binMin.toFixed(2)} – ${group.binMax.toFixed(2)})`;
        return (
          <QuantileSection
            key={group.quantile}
            label={label}
            rangeLabel={rangeLabel}
            activations={group.activations}
            globalMax={globalMax}
            defaultExpanded={group.quantile === 1}
            onHoverActivation={onHoverActivation}
          />
        );
      })}
    </div>
  );
}

function QuantileSection({
  label,
  rangeLabel,
  activations,
  globalMax,
  defaultExpanded,
  onHoverActivation,
}: {
  label: string;
  rangeLabel: string;
  activations: SaeActivation[];
  globalMax: number;
  defaultExpanded: boolean;
  onHoverActivation?: (value: number | null) => void;
}) {
  const [expanded, setExpanded] = useState(defaultExpanded);

  return (
    <div className="border rounded-md bg-card">
      <button
        className="flex items-center gap-2 w-full px-3 py-2 text-left hover:bg-muted/50 transition-colors rounded-md"
        onClick={() => setExpanded(!expanded)}
      >
        {expanded ? <ChevronDown className="h-3 w-3 text-muted-foreground" /> : <ChevronRight className="h-3 w-3 text-muted-foreground" />}
        <span className="text-xs font-medium">{label}</span>
        <span className="text-[10px] text-muted-foreground">{rangeLabel}</span>
        <span className="text-[10px] text-muted-foreground ml-auto">{activations.length} examples</span>
      </button>
      {expanded && (
        <div className="px-3 pb-3 space-y-3">
          {activations.map((act) => (
            <ActivationCard key={act.id} act={act} globalMax={globalMax} onHoverActivation={onHoverActivation} />
          ))}
        </div>
      )}
    </div>
  );
}

function ActivationCard({
  act,
  globalMax,
  onHoverActivation,
}: {
  act: SaeActivation;
  globalMax: number;
  onHoverActivation?: (value: number | null) => void;
}) {
  return (
    <div className="rounded-md border p-2 bg-card">
      <div className="flex items-center justify-between mb-1.5">
        <span className="text-[10px] font-mono text-muted-foreground">
          max: {act.maxValue.toFixed(3)} @ token {act.maxValueTokenIndex}
        </span>
      </div>
      <div className="max-h-40 overflow-y-auto">
        <TokenStrip
          tokens={act.tokens}
          values={act.values}
          maxValueTokenIndex={act.maxValueTokenIndex}
          globalMax={globalMax}
          onHoverActivation={onHoverActivation}
        />
      </div>
    </div>
  );
}
