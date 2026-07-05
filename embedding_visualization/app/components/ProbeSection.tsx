'use client';

import { useEffect, useMemo, useState } from 'react';
import { useSubscription } from '@apollo/client/react';
import { Trash2 } from 'lucide-react';

import { Card, CardContent, CardHeader, CardTitle } from '@/lib/ui-primitives/card';
import { Button } from '@/lib/ui-primitives/button';
import { Badge } from '@/lib/ui-primitives/badge';
import { Spinner } from '@/lib/ui-primitives/spinner';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/lib/ui-primitives/select';
import { EMBEDDING_PROGRESS_SUBSCRIPTION } from '@/lib/graphql/queries';
import type { JobProgress } from '@/lib/graphql/mutations';
import { useVisualizationStore } from '@/lib/stores/useVisualizationStore';
import type { ColorFieldOption } from '@/lib/utils/fieldAnalysis';
import type { ProbeInfo } from '@/lib/utils/probeFields';
import type { UseProbesReturn } from '@/lib/hooks/useProbes';

const PROBE_KINDS = [
  { value: 'ridge', label: 'Ridge (linear)' },
  { value: 'massmean', label: 'Mass-mean' },
  { value: 'mlp', label: 'MLP (nonlinear)' },
] as const;

interface ProbeSectionProps {
  probes: UseProbesReturn;
  colorFieldOptions?: ColorFieldOption[];
}

interface SubscriptionData {
  embeddingProgress: JobProgress;
}

const fmtMetric = (value: number | null | undefined) =>
  value === null || value === undefined ? '—' : value.toFixed(2);

/** Compact badge: validation R² (massmean's is calibrated) + Spearman ρ. */
function metricsBadge(probe: ProbeInfo): string {
  const m = probe.metrics ?? {};
  const parts: string[] = [];
  if (m.val_r2 !== undefined) parts.push(`R² ${fmtMetric(m.val_r2)}`);
  if (m.val_spearman !== undefined) parts.push(`ρ ${fmtMetric(m.val_spearman)}`);
  return parts.join(' · ') || 'no metrics';
}

/** Native tooltip listing every stored validation metric. */
function metricsTitle(probe: ProbeInfo): string {
  const m = probe.metrics ?? {};
  const labels: [string, string][] = [
    ['val_r2', 'R² (validation)'],
    ['val_pearson', 'Pearson r (validation)'],
    ['val_spearman', 'Spearman ρ (validation)'],
    ['val_mse', 'MSE (validation)'],
    ['val_mae', 'MAE (validation)'],
    ['train_r2', 'R² (train)'],
  ];
  const lines = labels
    .filter(([key]) => m[key] !== undefined)
    .map(([key, label]) => `${label}: ${fmtMetric(m[key])}`);
  return lines.join('\n') || 'No metrics recorded';
}

/**
 * "Direction probes" block for the Analytics sidebar: pick a numeric metadata
 * field + probe kind, fit server-side, then color the map by the probe's
 * score (sequential) or residual (diverging). Training progress arrives over
 * the shared job subscription; completion is driven by the mutation promise
 * inside useProbes (the subscription is display-only).
 */
export function ProbeSection({ probes, colorFieldOptions }: ProbeSectionProps) {
  const colorByField = useVisualizationStore((s) => s.colorByField);
  const setColorByField = useVisualizationStore((s) => s.setColorByField);

  const [selectedField, setSelectedField] = useState<string | null>(null);
  const [kind, setKind] = useState<string>('ridge');
  const [progress, setProgress] = useState<JobProgress | null>(null);

  // Numeric, non-derived fields are probe targets (never probe a probe).
  const targetOptions = useMemo(
    () =>
      (colorFieldOptions ?? []).filter(
        (o) => o.valueType === 'numeric' && !o.field.startsWith('probe_'),
      ),
    [colorFieldOptions],
  );

  // Follow the active color field by default, like CategoryBarList — but a
  // switch TO a probe-derived field is the auto-recolor after training, not
  // the user picking a new target: keep the selection so they can retrain
  // the same field with another kind.
  useEffect(() => {
    if (colorByField?.startsWith('probe_')) return;
    setSelectedField(null);
  }, [colorByField]);

  const followedField =
    colorByField && targetOptions.some((o) => o.field === colorByField)
      ? colorByField
      : null;
  const effectiveField = selectedField ?? followedField ?? null;

  const { data: progressData } = useSubscription<SubscriptionData>(
    EMBEDDING_PROGRESS_SUBSCRIPTION,
    {
      variables: { jobId: probes.jobId },
      skip: !probes.jobId,
    },
  );
  useEffect(() => {
    if (progressData?.embeddingProgress) {
      setProgress(progressData.embeddingProgress);
    }
  }, [progressData]);
  useEffect(() => {
    if (!probes.training) setProgress(null);
  }, [probes.training]);

  if (targetOptions.length === 0 && probes.probes.length === 0) {
    return null;
  }

  return (
    <Card className="gap-0 border-0 bg-transparent py-0 shadow-none">
      <CardHeader className="gap-1 px-0 pb-3">
        <CardTitle className="text-sm">Direction Probes</CardTitle>
        <p className="text-xs text-muted-foreground">
          Fit a probe on a numeric field, then color the map by how well the
          embedding encodes it.
        </p>
      </CardHeader>
      <CardContent className="space-y-3 px-0">
        {targetOptions.length > 0 && (
          <div className="space-y-2">
            {/* value falls back to '' (not undefined) so the Select stays
                controlled and never shows a stale internal selection. */}
            <Select
              value={effectiveField ?? ''}
              onValueChange={(val) => setSelectedField(val)}
            >
              <SelectTrigger size="sm" className="h-7 w-full text-xs">
                <SelectValue placeholder="Select numeric field..." />
              </SelectTrigger>
              <SelectContent>
                {targetOptions.map((opt) => (
                  <SelectItem key={opt.field} value={opt.field}>
                    <span className="text-xs">{opt.displayName}</span>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <div className="flex items-center gap-2">
              <Select value={kind} onValueChange={setKind}>
                <SelectTrigger size="sm" className="h-7 flex-1 text-xs">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {PROBE_KINDS.map((k) => (
                    <SelectItem key={k.value} value={k.value}>
                      <span className="text-xs">{k.label}</span>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <Button
                size="sm"
                className="h-7 shrink-0 text-xs"
                disabled={!effectiveField || probes.training}
                onClick={() => {
                  if (effectiveField) void probes.train(effectiveField, kind);
                }}
              >
                {probes.training ? <Spinner className="size-3" /> : 'Fit probe'}
              </Button>
            </div>
          </div>
        )}

        {probes.training && (
          <p className="flex items-center gap-2 text-xs text-muted-foreground">
            <Spinner className="size-3 shrink-0" />
            {progress?.message ?? 'Training probe...'}
          </p>
        )}

        {probes.trainingError && (
          <p className="text-xs text-destructive">{probes.trainingError}</p>
        )}

        {probes.probes.length > 0 && (
          <div className="space-y-2">
            {probes.probes.map((probe) => {
              const isActive =
                colorByField === probe.scoreField ||
                colorByField === probe.residualField;
              return (
                <div
                  key={`${probe.targetField}::${probe.kind}`}
                  className="space-y-1 rounded-md border border-border/60 px-2 py-1.5"
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="min-w-0 truncate text-xs font-medium">
                      {probe.targetField} · {probe.kind}
                    </span>
                    <Badge
                      variant={isActive ? 'default' : 'secondary'}
                      className="shrink-0 font-mono text-[10px]"
                      title={metricsTitle(probe)}
                    >
                      {metricsBadge(probe)}
                    </Badge>
                  </div>
                  <div className="flex items-center gap-1">
                    <Button
                      size="sm"
                      variant={colorByField === probe.scoreField ? 'secondary' : 'ghost'}
                      className="h-6 px-2 text-[11px]"
                      onClick={() => setColorByField(probe.scoreField, 'sequential')}
                    >
                      Score
                    </Button>
                    {probe.residualField && (
                      <Button
                        size="sm"
                        variant={colorByField === probe.residualField ? 'secondary' : 'ghost'}
                        className="h-6 px-2 text-[11px]"
                        onClick={() => setColorByField(probe.residualField!, 'diverging')}
                      >
                        Residual
                      </Button>
                    )}
                    <span className="ml-auto text-[10px] text-muted-foreground">
                      n={probe.nTrain + probe.nVal}
                    </span>
                    {/* Disabled while training: deleting mid-run races the
                        server-side persist (the finished run would resurrect
                        the probe or strand score rows). */}
                    <Button
                      size="icon"
                      variant="ghost"
                      className="size-6 text-muted-foreground hover:text-destructive"
                      aria-label={`Delete ${probe.targetField} ${probe.kind} probe`}
                      disabled={probes.training}
                      onClick={() => void probes.deleteProbe(probe)}
                    >
                      <Trash2 className="size-3" />
                    </Button>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
