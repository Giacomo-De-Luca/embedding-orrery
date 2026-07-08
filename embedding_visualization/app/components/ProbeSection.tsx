'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import { useSubscription } from '@apollo/client/react';
import { Settings2, Trash2 } from 'lucide-react';

import { Card, CardContent, CardHeader, CardTitle } from '@/lib/ui-primitives/card';
import { Button } from '@/lib/ui-primitives/button';
import { Badge } from '@/lib/ui-primitives/badge';
import { Input } from '@/lib/ui-primitives/input';
import { Label } from '@/lib/ui-primitives/label';
import { Popover, PopoverContent, PopoverTrigger } from '@/lib/ui-primitives/popover';
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
import {
  formatTargetMapping,
  isProbeTargetOption,
  resolveProbeTargetSelection,
  type ProbeInfo,
} from '@/lib/utils/probeFields';
import {
  DEFAULT_PROBE_PARAMS,
  PROBE_KIND_OPTIONS,
  buildTrainProbeInput,
  isBinaryKind,
  probeParamFields,
  type ProbeKind,
  type ProbeParams,
} from '@/lib/utils/probeParams';
import type { UseProbesReturn } from '@/lib/hooks/useProbes';

interface ProbeSectionProps {
  probes: UseProbesReturn;
  colorFieldOptions?: ColorFieldOption[];
}

interface SubscriptionData {
  embeddingProgress: JobProgress;
}

const fmtMetric = (value: number | null | undefined) =>
  value === null || value === undefined ? '—' : value.toFixed(2);

/** Compact badge: classification kinds show AUC + accuracy; regression kinds
 * show validation R² (massmean's is calibrated) + Spearman ρ. */
function metricsBadge(probe: ProbeInfo): string {
  const m = probe.metrics ?? {};
  const parts: string[] = [];
  if (m.val_auc !== undefined || m.val_accuracy !== undefined) {
    if (m.val_auc !== undefined) parts.push(`AUC ${fmtMetric(m.val_auc)}`);
    if (m.val_accuracy !== undefined) parts.push(`acc ${fmtMetric(m.val_accuracy)}`);
    return parts.join(' · ');
  }
  if (m.val_r2 !== undefined) parts.push(`R² ${fmtMetric(m.val_r2)}`);
  if (m.val_spearman !== undefined) parts.push(`ρ ${fmtMetric(m.val_spearman)}`);
  return parts.join(' · ') || 'no metrics';
}

/** Native tooltip: every stored validation metric + the 0/1 class mapping. */
function metricsTitle(probe: ProbeInfo): string {
  const m = probe.metrics ?? {};
  const labels: [string, string][] = [
    ['val_r2', 'R² (validation)'],
    ['val_pearson', 'Pearson r (validation)'],
    ['val_spearman', 'Spearman ρ (validation)'],
    ['val_mse', 'MSE (validation)'],
    ['val_mae', 'MAE (validation)'],
    ['train_r2', 'R² (train)'],
    ['val_auc', 'ROC-AUC (validation)'],
    ['val_accuracy', 'Accuracy (validation)'],
    ['val_f1_weighted', 'F1 weighted (validation)'],
  ];
  const lines = labels
    .filter(([key]) => m[key] !== undefined)
    .map(([key, label]) => `${label}: ${fmtMetric(m[key])}`);
  const mapping = formatTargetMapping(probe.targetMapping);
  if (mapping) lines.push(`Classes: ${mapping}`);
  return lines.join('\n') || 'No metrics recorded';
}

/** One compact labeled number input row for the settings popover.

 * Holds local text state so the field can be cleared while typing
 * (`Number('') === 0` would otherwise instantly commit 0 on backspace);
 * valid parses are clamped to [min, max] and committed as you type, and the
 * text resyncs to the committed value on blur. */
function ParamNumberRow({
  label,
  value,
  onChange,
  step = 1,
  min,
  max,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  step?: number;
  min?: number;
  max?: number;
}) {
  const [text, setText] = useState(String(value));
  useEffect(() => {
    setText(String(value));
  }, [value]);
  return (
    <div className="flex items-center justify-between gap-2">
      <Label className="text-xs font-normal text-muted-foreground">{label}</Label>
      <Input
        type="number"
        className="h-7 w-24 text-right text-xs"
        value={text}
        step={step}
        min={min}
        max={max}
        onChange={(e) => {
          setText(e.target.value);
          if (e.target.value.trim() === '') return;
          let v = Number(e.target.value);
          if (!Number.isFinite(v)) return;
          if (min !== undefined) v = Math.max(min, v);
          if (max !== undefined) v = Math.min(max, v);
          onChange(v);
        }}
        onBlur={() => setText(String(value))}
      />
    </div>
  );
}

/**
 * Gear-icon popover exposing the hyperparameters of the selected probe kind
 * plus shared advanced settings. Everything defaults to the backend's own
 * defaults; only changed values are sent (buildTrainProbeInput).
 */
function ProbeSettingsPopover({
  kind,
  params,
  onChange,
}: {
  kind: ProbeKind;
  params: ProbeParams;
  onChange: (p: ProbeParams) => void;
}) {
  const fields = probeParamFields(kind);
  const set = (patch: Partial<ProbeParams>) => onChange({ ...params, ...patch });

  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button
          size="icon"
          variant="ghost"
          className="size-7 shrink-0 text-muted-foreground"
          aria-label="Probe parameters"
        >
          <Settings2 className="size-3.5" />
        </Button>
      </PopoverTrigger>
      <PopoverContent align="end" className="w-60 space-y-2 p-3">
        <p className="text-xs font-medium">Probe parameters</p>
        {fields.includes('alpha') && (
          <ParamNumberRow
            label="Alpha (L2)"
            value={params.alpha}
            step={0.1}
            min={0}
            onChange={(v) => set({ alpha: v })}
          />
        )}
        {fields.includes('c') && (
          <ParamNumberRow
            label="C"
            value={params.c}
            step={0.1}
            min={0.001}
            onChange={(v) => set({ c: v })}
          />
        )}
        {fields.includes('kernel') && (
          <div className="flex items-center justify-between gap-2">
            <Label className="text-xs font-normal text-muted-foreground">Kernel</Label>
            <Select
              value={params.kernel}
              onValueChange={(v) => set({ kernel: v as ProbeParams['kernel'] })}
            >
              <SelectTrigger size="sm" className="h-7 w-24 text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="rbf">rbf</SelectItem>
                <SelectItem value="linear">linear</SelectItem>
              </SelectContent>
            </Select>
          </div>
        )}
        {fields.includes('classWeight') && (
          <div className="flex items-center justify-between gap-2">
            <Label className="text-xs font-normal text-muted-foreground">Class weight</Label>
            <Select
              value={params.classWeight}
              onValueChange={(v) => set({ classWeight: v as ProbeParams['classWeight'] })}
            >
              <SelectTrigger size="sm" className="h-7 w-24 text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="none">none</SelectItem>
                <SelectItem value="balanced">balanced</SelectItem>
              </SelectContent>
            </Select>
          </div>
        )}
        {fields.includes('hiddenSize') && (
          <ParamNumberRow
            label="Hidden size"
            value={params.hiddenSize}
            step={32}
            min={8}
            onChange={(v) => set({ hiddenSize: v })}
          />
        )}
        {fields.includes('epochs') && (
          <ParamNumberRow
            label="Max epochs"
            value={params.epochs}
            step={10}
            min={1}
            onChange={(v) => set({ epochs: v })}
          />
        )}
        {fields.length === 0 && (
          <p className="text-xs text-muted-foreground">
            Mass-mean is closed-form — no hyperparameters.
          </p>
        )}
        <div className="border-t pt-2">
          <ParamNumberRow
            label="Seed"
            value={params.seed}
            onChange={(v) => set({ seed: v })}
          />
          <div className="pt-2">
            <ParamNumberRow
              label="Train split"
              value={params.trainSplit}
              step={0.05}
              min={0.5}
              max={0.95}
              onChange={(v) => set({ trainSplit: v })}
            />
          </div>
        </div>
        <Button
          size="sm"
          variant="ghost"
          className="h-6 w-full text-xs text-muted-foreground"
          onClick={() => onChange(DEFAULT_PROBE_PARAMS)}
        >
          Reset to defaults
        </Button>
      </PopoverContent>
    </Popover>
  );
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
  const [kind, setKind] = useState<ProbeKind>('ridge');
  const [params, setParams] = useState<ProbeParams>(DEFAULT_PROBE_PARAMS);
  const [progress, setProgress] = useState<JobProgress | null>(null);

  // Numeric fields + binary categorical fields (trained as 0/1) are probe
  // targets; probe-derived fields never are (never probe a probe).
  const targetOptions = useMemo(
    () => (colorFieldOptions ?? []).filter(isProbeTargetOption),
    [colorFieldOptions],
  );

  const followedField =
    colorByField && targetOptions.some((o) => o.field === colorByField)
      ? colorByField
      : null;
  const effectiveField = selectedField ?? followedField ?? null;

  // Remember the last real (non-probe) target so follow-mode selection can be
  // pinned when training auto-recolors the map to a probe_* field.
  const lastTargetRef = useRef<string | null>(null);
  useEffect(() => {
    if (effectiveField) lastTargetRef.current = effectiveField;
  });

  // Follow the active color field by default, like CategoryBarList — but when
  // training auto-recolors to a probe_* field, keep (or pin) the just-probed
  // target so another kind can be fitted on it instead of disabling Fit.
  useEffect(() => {
    setSelectedField((prev) =>
      resolveProbeTargetSelection(colorByField, prev, lastTargetRef.current),
    );
  }, [colorByField]);

  // Binary-only kinds (logreg) need a two-class target; a numeric field with
  // exactly two distinct values qualifies too. Fall back to ridge when the
  // selected target stops being binary.
  const selectedOption = targetOptions.find((o) => o.field === effectiveField);
  const isBinaryTarget = selectedOption?.uniqueCount === 2;
  useEffect(() => {
    if (isBinaryKind(kind) && !isBinaryTarget) setKind('ridge');
  }, [kind, isBinaryTarget]);

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
              <Select value={kind} onValueChange={(v) => setKind(v as ProbeKind)}>
                <SelectTrigger size="sm" className="h-7 flex-1 text-xs">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {PROBE_KIND_OPTIONS.map((k) => (
                    <SelectItem
                      key={k.value}
                      value={k.value}
                      disabled={isBinaryKind(k.value) && !isBinaryTarget}
                    >
                      <span className="text-xs">{k.label}</span>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <ProbeSettingsPopover kind={kind} params={params} onChange={setParams} />
              <Button
                size="sm"
                className="h-7 shrink-0 text-xs"
                disabled={!effectiveField || probes.training}
                onClick={() => {
                  if (effectiveField && probes.collectionName) {
                    void probes.train(
                      buildTrainProbeInput(probes.collectionName, effectiveField, kind, params),
                    );
                  }
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
