'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import { useSubscription } from '@apollo/client/react';
import { Settings2, Trash2 } from 'lucide-react';

import { Card, CardContent, CardHeader, CardTitle } from '@/lib/ui-primitives/card';
import { Button } from '@/lib/ui-primitives/button';
import { Badge } from '@/lib/ui-primitives/badge';
import { Checkbox } from '@/lib/ui-primitives/checkbox';
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
  ProbeFieldNames,
  formatTargetMapping,
  isProbeTargetOption,
  probeTask,
  resolveProbeTargetSelection,
  type ProbeInfo,
} from '@/lib/utils/probeFields';
import {
  DEFAULT_PROBE_PARAMS,
  MLP_ACTIVATION_OPTIONS,
  PROBE_KIND_OPTIONS,
  buildTrainProbeInput,
  isClassificationKind,
  probeKindLabel,
  probeParamFields,
  type ProbeKind,
  type ProbeParams,
  type TrainedProbeKind,
} from '@/lib/utils/probeParams';
import {
  MAX_PROBE_CLASSES,
  ProbeTargetProfile,
  fallbackProbeKind,
  isProbeKindAvailable,
  mlpClassifierMode,
  resolveProbeKind,
  type ClassPreview,
} from '@/lib/utils/probeTargets';
import type { UseProbesReturn } from '@/lib/hooks/useProbes';
import { PROBE_SHOWCASE_EVENT, type ProbeShowcaseDetail } from '@/lib/utils/probeTourSteps';
import { DEMO_DISABLED_MESSAGE } from '@/lib/utils/demoMode';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/lib/ui-primitives/tooltip';

interface ProbeSectionProps {
  probes: UseProbesReturn;
  colorFieldOptions?: ColorFieldOption[];
  /**
   * Demo builds: show shipped probes with their recolor buttons (pure
   * client-side) but no training form or delete — the backend rejects those
   * mutations anyway (`ORRERY_READ_ONLY`).
   */
  readOnly?: boolean;
}

interface SubscriptionData {
  embeddingProgress: JobProgress;
}

const fmtMetric = (value: unknown) => (typeof value === 'number' ? value.toFixed(2) : '—');

/** Compact badge: multiclass probes show accuracy + macro-F1; binary
 * classifiers AUC + accuracy; regression kinds validation R² (massmean's is
 * calibrated) + Spearman ρ. */
function metricsBadge(probe: ProbeInfo): string {
  const m = probe.metrics ?? {};
  const parts: string[] = [];
  const task = probeTask(probe);
  if (task === 'multiclass') {
    if (m.val_accuracy !== undefined) parts.push(`acc ${fmtMetric(m.val_accuracy)}`);
    if (m.val_f1_macro !== undefined) parts.push(`F1 ${fmtMetric(m.val_f1_macro)}`);
  } else if (task === 'binary') {
    if (m.val_auc !== undefined) parts.push(`AUC ${fmtMetric(m.val_auc)}`);
    if (m.val_accuracy !== undefined) parts.push(`acc ${fmtMetric(m.val_accuracy)}`);
  } else {
    if (m.val_r2 !== undefined) parts.push(`R² ${fmtMetric(m.val_r2)}`);
    if (m.val_spearman !== undefined) parts.push(`ρ ${fmtMetric(m.val_spearman)}`);
  }
  return parts.join(' · ') || 'no metrics';
}

const METRIC_LABELS: [string, string][] = [
  ['val_r2', 'R² (validation)'],
  ['val_pearson', 'Pearson r (validation)'],
  ['val_spearman', 'Spearman ρ (validation)'],
  ['val_mse', 'MSE (validation)'],
  ['val_mae', 'MAE (validation)'],
  ['train_r2', 'R² (train)'],
  ['val_auc', 'ROC-AUC (validation)'],
  ['val_auc_ovr', 'ROC-AUC one-vs-rest (validation)'],
  ['val_accuracy', 'Accuracy (validation)'],
  ['val_balanced_accuracy', 'Balanced accuracy (validation)'],
  ['val_f1_macro', 'F1 macro (validation)'],
  ['val_f1_weighted', 'F1 weighted (validation)'],
  ['train_accuracy', 'Accuracy (train)'],
  ['val_majority_baseline', 'Majority-class baseline (validation)'],
];

/** Native tooltip: every stored metric + the classes the probe trained on. */
function metricsTitle(probe: ProbeInfo): string {
  const m = probe.metrics ?? {};
  const lines = METRIC_LABELS.filter(([key]) => m[key] !== undefined).map(
    ([key, label]) => `${label}: ${fmtMetric(m[key])}`,
  );
  const mapping = formatTargetMapping(probe.targetMapping);
  if (mapping) {
    lines.push(`Classes: ${mapping}`);
  } else if (probe.classes?.length) {
    lines.push(`Classes (${probe.classes.length}): ${probe.classes.join(', ')}`);
  }
  if (typeof probe.minClassFraction === 'number') {
    lines.push(`Min class: ${probe.minClassFraction * 100}% of labelled items`);
  }
  const excluded = Object.entries(probe.excludedClasses ?? {});
  if (excluded.length > 0) {
    const list = excluded.map(([label, count]) => `${label} (${count})`).join(', ');
    lines.push(`Excluded (too few items): ${list}`);
  }
  return lines.join('\n') || 'No metrics recorded';
}

/** One-line summary of which classes a classifier would train on. */
function classPreviewText(preview: ClassPreview): string {
  if (preview.tooFew) return 'Fewer than 2 classes kept — lower the min class %.';
  if (preview.tooMany) {
    return `${preview.kept} classes kept — at most ${MAX_PROBE_CLASSES}; raise the min class %.`;
  }
  const excluded = preview.excluded.length;
  return excluded === 0
    ? `${preview.kept} classes kept.`
    : `${preview.kept} classes kept · ${excluded} excluded (< ${preview.threshold} items).`;
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
  disabled = false,
  checked,
  onCheckedChange,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  step?: number;
  min?: number;
  max?: number;
  disabled?: boolean;
  /** When provided, a leading checkbox toggles the row on/off. */
  checked?: boolean;
  onCheckedChange?: (checked: boolean) => void;
}) {
  const [text, setText] = useState(String(value));
  useEffect(() => {
    setText(String(value));
  }, [value]);
  return (
    <div className="flex items-center justify-between gap-2">
      <div className="flex items-center gap-2">
        {onCheckedChange !== undefined && (
          <Checkbox
            checked={checked}
            onCheckedChange={(c) => onCheckedChange(c === true)}
            aria-label={`Enable ${label.toLowerCase()}`}
            className="size-3.5"
          />
        )}
        <Label
          className={`text-xs font-normal text-muted-foreground${disabled ? ' opacity-60' : ''}`}
        >
          {label}
        </Label>
      </div>
      <Input
        type="number"
        className="h-7 w-24 text-right text-xs"
        value={text}
        step={step}
        min={min}
        max={max}
        disabled={disabled}
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
  classifierMode,
  classPreview,
  open,
  onOpenChange,
}: {
  /** The kind that will train (the MLP resolves to `mlp_clf` as a classifier). */
  kind: TrainedProbeKind;
  params: ProbeParams;
  onChange: (p: ProbeParams) => void;
  /** MLP only: the Classifier switch state allowed by the target; null hides it. */
  classifierMode: ReturnType<typeof mlpClassifierMode> | null;
  /** Classifiers on 3+ value targets: classes kept at the current min class %. */
  classPreview: ClassPreview | null;
  /** Controlled open state — the probing tour opens the popover to showcase it. */
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
}) {
  const fields = probeParamFields(kind);
  const set = (patch: Partial<ProbeParams>) => onChange({ ...params, ...patch });

  return (
    <Popover open={open} onOpenChange={onOpenChange}>
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
      {/* data-tour: ring target for the probing tour's settings showcase */}
      <PopoverContent align="end" className="w-60 space-y-2 p-3" data-tour="probe-settings">
        <p className="text-xs font-medium">Probe parameters</p>
        {classifierMode !== null && (
          <div className="flex items-center gap-2">
            <Checkbox
              id="probe-mlp-classifier"
              checked={classifierMode === 'forced-on' || (classifierMode === 'free' && params.mlpClassifier)}
              disabled={classifierMode !== 'free'}
              onCheckedChange={(c) => set({ mlpClassifier: c === true })}
              className="size-3.5"
            />
            <Label
              htmlFor="probe-mlp-classifier"
              className={`text-xs font-normal text-muted-foreground${classifierMode !== 'free' ? ' opacity-60' : ''}`}
              title={
                classifierMode === 'forced-on'
                  ? 'Text targets with 3+ classes can only be classified'
                  : classifierMode === 'forced-off'
                    ? 'Continuous targets can only be regressed'
                    : 'Train on class labels instead of regressing the value'
              }
            >
              Classifier
            </Label>
          </div>
        )}
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
        {fields.includes('activation') && (
          <div className="flex items-center justify-between gap-2">
            <Label className="text-xs font-normal text-muted-foreground">Activation</Label>
            <Select
              value={params.activation}
              onValueChange={(v) => set({ activation: v as ProbeParams['activation'] })}
            >
              <SelectTrigger size="sm" className="h-7 w-24 text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {MLP_ACTIVATION_OPTIONS.map((a) => (
                  <SelectItem key={a} value={a}>
                    {a}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        )}
        {fields.includes('devSplit') && (
          <ParamNumberRow
            label="Dev split"
            value={params.devSplit}
            step={0.05}
            min={0.05}
            max={0.5}
            checked={params.devSplitEnabled}
            onCheckedChange={(c) => set({ devSplitEnabled: c })}
            disabled={!params.devSplitEnabled}
            onChange={(v) => set({ devSplit: v })}
          />
        )}
        {fields.includes('patience') && (
          <ParamNumberRow
            label="Patience"
            value={params.patience}
            min={1}
            // Patience only acts on the dev set; without one it is inert.
            disabled={!params.devSplitEnabled}
            onChange={(v) => set({ patience: v })}
          />
        )}
        {fields.includes('minClassPct') && (
          <div className="space-y-1">
            <ParamNumberRow
              label="Min class %"
              value={params.minClassPct}
              step={0.5}
              min={0}
              max={49.9}
              onChange={(v) => set({ minClassPct: v })}
            />
            {classPreview && (
              <p
                className={`text-[10px] ${classPreview.tooFew || classPreview.tooMany ? 'text-destructive' : 'text-muted-foreground'}`}
                title={classPreview.excluded.map((e) => `${e.label} (${e.count})`).join('\n') || undefined}
              >
                {classPreviewText(classPreview)}
              </p>
            )}
          </div>
        )}
        {fields.length === 0 && (
          <p className="text-xs text-muted-foreground">
            Mass-mean probes are closed-form — no hyperparameters.
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
 * "Direction probes" block for the Analytics sidebar: pick a metadata field +
 * probe kind, fit server-side, then color the map by the probe's outputs —
 * score (sequential) / residual (diverging) for regressors, confusion for
 * binary classifiers, predicted class / correctness / confidence / P(class)
 * for multiclass classifiers. Which kinds a field allows comes from its
 * ProbeTargetProfile. Training progress arrives over the shared job
 * subscription; completion is driven by the mutation promise inside useProbes
 * (the subscription is display-only).
 */
export function ProbeSection({ probes, colorFieldOptions, readOnly = false }: ProbeSectionProps) {
  const colorByField = useVisualizationStore((s) => s.colorByField);
  const setColorByField = useVisualizationStore((s) => s.setColorByField);
  const setCustomNumericRange = useVisualizationStore((s) => s.setCustomNumericRange);

  const [selectedField, setSelectedField] = useState<string | null>(null);
  const [kind, setKind] = useState<ProbeKind>('ridge');
  const [params, setParams] = useState<ProbeParams>(DEFAULT_PROBE_PARAMS);
  const [progress, setProgress] = useState<JobProgress | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);

  // Probing-tour showcase: select the requested kind and open the settings
  // popover so the demo can show what's configurable even though training
  // is disabled there. Dispatched by the tour's last step (see
  // PROBE_SHOWCASE_EVENT in probeTourSteps.ts).
  useEffect(() => {
    const onShowcase = (e: Event) => {
      const detail = (e as CustomEvent<ProbeShowcaseDetail>).detail;
      if (detail?.kind && PROBE_KIND_OPTIONS.some((k) => k.value === detail.kind)) {
        setKind(detail.kind as ProbeKind);
      }
      setSettingsOpen(true);
    };
    window.addEventListener(PROBE_SHOWCASE_EVENT, onShowcase);
    return () => window.removeEventListener(PROBE_SHOWCASE_EVENT, onShowcase);
  }, []);

  // Numeric fields + categorical fields (class labels; two-value fields also
  // regress as 0/1) are probe targets; probe-derived fields never are.
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

  // What the selected target can train: regressors need numbers (or two
  // values), classifiers need class labels. A kind the target can't train
  // falls back to ridge (regression) or logistic (classes only).
  const profile = useMemo(
    () =>
      effectiveField && probes.itemMetadata
        ? ProbeTargetProfile.fromMetadata(probes.itemMetadata, effectiveField)
        : null,
    [effectiveField, probes.itemMetadata],
  );
  useEffect(() => {
    const next = fallbackProbeKind(kind, profile);
    if (next !== kind) setKind(next);
  }, [kind, profile]);
  const trainedKind = resolveProbeKind(kind, params.mlpClassifier, profile);
  const classPreview =
    profile && isClassificationKind(trainedKind) && profile.nDistinct > 2
      ? profile.preview(params.minClassPct / 100)
      : null;
  const classesUnfit = classPreview !== null && (classPreview.tooFew || classPreview.tooMany);
  // e.g. a single-valued field: every kind is disabled, so is Fit.
  const noKindFits =
    profile !== null && !profile.supportsRegression && !profile.supportsClassification;

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

  // Nothing to show: no trained probes and (read-only, or no field to fit).
  if (probes.probes.length === 0 && (readOnly || targetOptions.length === 0)) {
    return null;
  }

  return (
    <Card className="gap-0 border-0 bg-transparent py-0 shadow-none">
      <CardHeader className="gap-1 px-0 pb-3">
        <CardTitle className="text-sm">Direction Probes</CardTitle>
        <p className="text-xs text-muted-foreground">
          {readOnly
            ? 'Probes fitted on the embedding space, shipped with the demo — ' +
              'color the map by their predictions. Training new ones needs the full platform.'
            : 'Fit a probe on a numeric or categorical field, then color the map by ' +
              'how well the embedding encodes it.'}
        </p>
      </CardHeader>
      <CardContent className="space-y-3 px-0">
        {/* Read-only builds keep the full fit form browsable (target, kind,
            per-kind settings) so visitors see what the platform can train —
            only the Fit button itself is disabled. */}
        {targetOptions.length > 0 && (
          <div className="space-y-2">
            {/* value falls back to '' (not undefined) so the Select stays
                controlled and never shows a stale internal selection. */}
            <Select
              value={effectiveField ?? ''}
              onValueChange={(val) => setSelectedField(val)}
            >
              <SelectTrigger size="sm" className="h-7 w-full text-xs">
                <SelectValue placeholder="Select target field..." />
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
                      disabled={!isProbeKindAvailable(k.value, profile)}
                    >
                      <span className="text-xs">{k.label}</span>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <ProbeSettingsPopover
                kind={trainedKind}
                params={params}
                onChange={setParams}
                classifierMode={kind === 'mlp' ? mlpClassifierMode(profile) : null}
                classPreview={classPreview}
                open={settingsOpen}
                onOpenChange={setSettingsOpen}
              />
              {readOnly ? (
                <Tooltip>
                  {/* span wrapper: disabled buttons swallow the hover events
                      the tooltip needs (same pattern as VisualizationControls'
                      save-default button). */}
                  <TooltipTrigger asChild>
                    <span>
                      <Button size="sm" className="h-7 shrink-0 text-xs" disabled>
                        Fit probe
                      </Button>
                    </span>
                  </TooltipTrigger>
                  <TooltipContent>{DEMO_DISABLED_MESSAGE}</TooltipContent>
                </Tooltip>
              ) : (
                <Button
                  size="sm"
                  className="h-7 shrink-0 text-xs"
                  disabled={!effectiveField || probes.training || classesUnfit || noKindFits}
                  title={classesUnfit && classPreview ? classPreviewText(classPreview) : undefined}
                  onClick={() => {
                    if (effectiveField && probes.collectionName) {
                      void probes.train(
                        buildTrainProbeInput(
                          probes.collectionName,
                          effectiveField,
                          trainedKind,
                          params,
                        ),
                      );
                    }
                  }}
                >
                  {probes.training ? <Spinner className="size-3" /> : 'Fit probe'}
                </Button>
              )}
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
              const names = new ProbeFieldNames(probe);
              const multiclass = probeTask(probe) === 'multiclass';
              // Derived views only render when their field option exists —
              // residuals can be absent despite a residualField (degenerate
              // mass-mean calibration persists all-null residuals), confusion
              // and correctness need a client-resolvable actual class.
              const hasField = (field: string | null) =>
                field !== null && probes.fieldOptions.some((o) => o.field === field);
              const hasResidual = hasField(names.residual);
              const hasAbsError = hasField(names.absError);
              const hasConfusion = hasField(names.confusion);
              const hasPredicted = hasField(names.predicted);
              const hasCorrect = hasField(names.correct);
              const isActive = colorByField !== null && names.matches(colorByField);
              const activeClass = colorByField !== null ? names.classIndexOf(colorByField) : null;
              return (
                <div
                  key={`${probe.targetField}::${probe.kind}`}
                  // data-tour: the probing tour circles the row whose field is
                  // the active colouring (isActive holds for at most one row).
                  data-tour={isActive ? 'probe-active' : undefined}
                  className="space-y-1 rounded-md border border-border/60 px-2 py-1.5"
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="min-w-0 truncate text-xs font-medium">
                      {probe.targetField} · {probeKindLabel(probe.kind)}
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
                    {hasPredicted && (
                      <Button
                        size="sm"
                        variant={colorByField === names.predicted ? 'secondary' : 'ghost'}
                        className="h-6 px-2 text-[11px]"
                        onClick={() => setColorByField(names.predicted!, 'categorical')}
                      >
                        Predicted
                      </Button>
                    )}
                    {hasCorrect && (
                      <Button
                        size="sm"
                        variant={colorByField === names.correct ? 'secondary' : 'ghost'}
                        className="h-6 px-2 text-[11px]"
                        onClick={() => setColorByField(names.correct!, 'categorical')}
                      >
                        Correct
                      </Button>
                    )}
                    <Button
                      size="sm"
                      variant={colorByField === probe.scoreField ? 'secondary' : 'ghost'}
                      className="h-6 px-2 text-[11px]"
                      title={multiclass ? 'Highest class probability' : undefined}
                      onClick={() => setColorByField(probe.scoreField, 'sequential')}
                    >
                      {multiclass ? 'Confidence' : 'Score'}
                    </Button>
                    {hasResidual && (
                      <Button
                        size="sm"
                        variant={colorByField === probe.residualField ? 'secondary' : 'ghost'}
                        className="h-6 px-2 text-[11px]"
                        onClick={() => {
                          setColorByField(probe.residualField!, 'diverging');
                          // Zero-center the diverging scale: residuals are
                          // rarely symmetric, and the neutral midpoint must
                          // mean "no error". (Cleared automatically on the
                          // next color-field change by the store subscription.)
                          setCustomNumericRange({ center: 0 });
                        }}
                      >
                        Residual
                      </Button>
                    )}
                    {hasAbsError && (
                      <Button
                        size="sm"
                        variant={colorByField === names.absError ? 'secondary' : 'ghost'}
                        className="h-6 px-2 text-[11px]"
                        onClick={() => setColorByField(names.absError!, 'sequential')}
                      >
                        |Err|
                      </Button>
                    )}
                    {hasConfusion && (
                      <Button
                        size="sm"
                        variant={colorByField === names.confusion ? 'secondary' : 'ghost'}
                        className="h-6 px-2 text-[11px]"
                        onClick={() => setColorByField(names.confusion!, 'categorical')}
                      >
                        Confusion
                      </Button>
                    )}
                    <span className="ml-auto text-[10px] text-muted-foreground">
                      n={probe.nTrain + probe.nVal}
                    </span>
                    {/* Disabled while training: deleting mid-run races the
                        server-side persist (the finished run would resurrect
                        the probe or strand score rows). */}
                    {!readOnly && (
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
                    )}
                  </div>
                  {multiclass && probe.classes && probe.classes.length > 0 && (
                    // P(class) values load on demand, one class per request.
                    <Select
                      value={activeClass !== null ? String(activeClass) : ''}
                      onValueChange={(v) => probes.showClassProbability(probe, Number(v))}
                    >
                      <SelectTrigger size="sm" className="h-6 w-full text-[11px]">
                        <SelectValue placeholder="Color by class probability…" />
                      </SelectTrigger>
                      <SelectContent>
                        {probe.classes.map((label, index) => (
                          <SelectItem key={label} value={String(index)}>
                            <span className="text-xs">P({label})</span>
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
