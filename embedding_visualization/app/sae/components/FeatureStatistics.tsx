'use client';

import type { SaeFeature, SaeActivation } from '@/lib/types/types';
import { CollapsibleSection } from './CollapsibleSection';
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
  return (
    <CollapsibleSection title="Statistics">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
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
    </CollapsibleSection>
  );
}
