'use client';

import { useCallback, useState } from 'react';
import { useMutation } from '@apollo/client/react';
import { toast } from 'sonner';
import type { SaeFeature } from '@/lib/types/types';
import { Badge } from '@/lib/ui-primitives/badge';
import { UPDATE_SAE_FEATURE_LABEL } from '@/lib/graphql/mutations';
import { InlineEditableField } from '@/app/test-embed/components/InlineEditableField';
import { LogitBarChart } from './LogitBarChart';

interface FeatureDetailCardProps {
  feature: SaeFeature;
  onLabelUpdated?: (newLabel: string) => void;
}

export function FeatureDetailCard({ feature, onLabelUpdated }: FeatureDetailCardProps) {
  const [updateLabel] = useMutation<{ updateSaeFeatureLabel: boolean }>(UPDATE_SAE_FEATURE_LABEL);
  const [isSaving, setIsSaving] = useState(false);

  const handleSaveLabel = useCallback(async (_key: string, value: unknown): Promise<boolean> => {
    const newLabel = String(value).trim();
    if (!newLabel) return false;
    setIsSaving(true);
    try {
      const { data } = await updateLabel({
        variables: {
          modelId: feature.modelId,
          saeId: feature.saeId,
          featureIndex: feature.featureIndex,
          label: newLabel,
        },
      });
      if (data?.updateSaeFeatureLabel) {
        toast.success('Label updated');
        onLabelUpdated?.(newLabel);
        return true;
      }
      toast.error('Feature not found');
      return false;
    } catch (err: unknown) {
      toast.error(err instanceof Error ? err.message : 'Failed to update label');
      return false;
    } finally {
      setIsSaving(false);
    }
  }, [feature.modelId, feature.saeId, feature.featureIndex, updateLabel, onLabelUpdated]);

  return (
    <div className="space-y-4">
      {/* Header: index + density */}
      <div className="flex items-start gap-3">
        <Badge variant="outline" className="font-mono text-sm shrink-0">
          #{feature.featureIndex}
        </Badge>
        {feature.density != null && (
          <Badge variant="secondary" className="font-mono text-xs shrink-0">
            density: {feature.density < 0.001
              ? feature.density.toExponential(2)
              : feature.density.toFixed(4)}
          </Badge>
        )}
      </div>

      {/* Editable label */}
      <InlineEditableField
        fieldKey="label"
        label="Label"
        value={feature.label ?? ''}
        type="text"
        isSaving={isSaving}
        onSave={handleSaveLabel}
      />

      {/* Logit charts side by side */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <h4 className="text-xs font-medium text-muted-foreground mb-2 uppercase tracking-wide">
            Top Logits
          </h4>
          <LogitBarChart
            entries={feature.topLogits ?? []}
            variant="positive"
          />
        </div>
        <div>
          <h4 className="text-xs font-medium text-muted-foreground mb-2 uppercase tracking-wide">
            Bottom Logits
          </h4>
          <LogitBarChart
            entries={feature.bottomLogits ?? []}
            variant="negative"
          />
        </div>
      </div>
    </div>
  );
}
