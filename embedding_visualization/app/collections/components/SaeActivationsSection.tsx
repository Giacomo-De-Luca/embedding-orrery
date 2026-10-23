'use client';

import { useMemo } from 'react';
import { useQuery } from '@apollo/client/react';
import { Badge } from '@/lib/ui-primitives/badge';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/lib/ui-primitives/select';
import { GET_SAE_MODELS } from '@/lib/graphql/queries';
import type { SaeModelInfo } from '@/lib/types/types';
import type { EmbeddingModelState } from '../lib/useEmbeddingModelState';
import { OptionalStepCard } from './CollapsibleCard';

interface SaeActivationsSectionProps {
  model: Pick<
    EmbeddingModelState,
    'enableSaeActivations' | 'setEnableSaeActivations' | 'saeSelection' | 'setSaeSelection'
  >;
  idPrefix?: string;
}

/**
 * Optional final step of the embed flow: link the new collection to an SAE
 * and compute per-document activations once embedding (and topics, if
 * enabled) finish. Runs client-side after the embed mutation resolves — see
 * runPostEmbedSaeStep in ../lib/embeddingFormUtils.ts.
 */
export function SaeActivationsSection({ model, idPrefix = '' }: SaeActivationsSectionProps) {
  const { data: modelsData } = useQuery<{ saeModels: SaeModelInfo[] }>(GET_SAE_MODELS, {
    skip: !model.enableSaeActivations,
  });
  const saeModels = useMemo(() => modelsData?.saeModels ?? [], [modelsData]);

  return (
    <OptionalStepCard
      title="SAE Activations"
      description="Optional: link an SAE and collect each document's feature activations, for feature search on Explore."
      checkboxId={`${idPrefix}enable-sae-activations`}
      checkboxLabel="Collect SAE activations after embedding"
      enabled={model.enableSaeActivations}
      onEnabledChange={model.setEnableSaeActivations}
    >
      <div className="space-y-2">
        <p className="text-xs text-muted-foreground">
          Loads the model and runs it over every document: this may take a while on large collections.
        </p>
        <Select
          value={model.saeSelection ?? ''}
          onValueChange={(v) => model.setSaeSelection(v || null)}
        >
          <SelectTrigger id={`${idPrefix}sae-activations-model`} className="w-full">
            <SelectValue placeholder="Select an SAE model..." />
          </SelectTrigger>
          <SelectContent>
            {saeModels.map((m) => (
              <SelectItem key={`${m.modelId}::${m.saeId}`} value={`${m.modelId}::${m.saeId}`}>
                <span className="flex items-center gap-2">
                  {m.modelId} / {m.saeId}
                  <Badge variant="secondary" className="text-[10px] px-1 py-0">
                    {m.featureCount.toLocaleString()} features
                  </Badge>
                </span>
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        {modelsData && saeModels.length === 0 && (
          <p className="text-xs text-muted-foreground">
            No SAE models ingested yet. Use the SAE tab to download feature data first.
          </p>
        )}
        {saeModels.length > 0 && !model.saeSelection && (
          <p className="text-xs text-amber-600 dark:text-amber-500">
            Select an SAE model: otherwise this step is skipped.
          </p>
        )}
      </div>
    </OptionalStepCard>
  );
}
