'use client';

import { useState } from 'react';
import { ChevronDown, ChevronRight } from 'lucide-react';
import type { SaeFeature, SaeActivation } from '@/lib/types/types';
import { ActivationHistogram } from './ActivationHistogram';
import { LogitHistogram } from './LogitHistogram';
import { DensityHistogram } from './DensityHistogram';

interface FeatureStatisticsProps {
  feature: SaeFeature;
  activations: SaeActivation[];
  allDensities: number[];
  densitiesLoading: boolean;
  hoveredActivationValue?: number | null;
}

export function FeatureStatistics({
  feature,
  activations,
  allDensities,
  densitiesLoading,
  hoveredActivationValue,
}: FeatureStatisticsProps) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="border rounded-lg bg-card">
      <button
        className="flex items-center gap-2 w-full px-4 py-2.5 text-left hover:bg-muted/50 transition-colors rounded-lg"
        onClick={() => setExpanded(!expanded)}
      >
        {expanded ? (
          <ChevronDown className="h-3.5 w-3.5 text-muted-foreground" />
        ) : (
          <ChevronRight className="h-3.5 w-3.5 text-muted-foreground" />
        )}
        <span className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
          Statistics
        </span>
      </button>
      {expanded && (
        <div className="px-4 pb-4 grid grid-cols-1 md:grid-cols-3 gap-4">
          <ActivationHistogram
            activations={activations}
            hoveredValue={hoveredActivationValue}
          />
          <LogitHistogram
            topLogits={feature.topLogits}
            bottomLogits={feature.bottomLogits}
          />
          <DensityHistogram
            allDensities={allDensities}
            currentDensity={feature.density}
            loading={densitiesLoading}
          />
        </div>
      )}
    </div>
  );
}
