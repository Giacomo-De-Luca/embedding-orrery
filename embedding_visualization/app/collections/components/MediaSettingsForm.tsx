'use client';

import { useState } from 'react';
import { Badge } from '@/lib/ui-primitives/badge';
import { Button } from '@/lib/ui-primitives/button';
import { Input } from '@/lib/ui-primitives/input';
import { Label } from '@/lib/ui-primitives/label';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/lib/ui-primitives/select';
import type { MediaOptionsInput } from '@/lib/graphql/mutations';
import type { MediaKindCounts } from '../lib/mediaColumns';
import type { MediaOptionKey, MediaOptionsBudget } from '../lib/mediaOptions';
import { CollapsibleCard } from './CollapsibleCard';

interface NumberFieldProps {
  id: string;
  label: string;
  value: number;
  onCommit: (value: number) => void;
  step: number;
  min: number;
  max: number;
  hint: string;
}

/** A number input committed on blur or Enter, so a half-typed value never reaches the estimate. */
function NumberField({ id, label, value, onCommit, step, min, max, hint }: NumberFieldProps) {
  const [draft, setDraft] = useState<string | null>(null);
  const commit = () => {
    const parsed = Number(draft);
    if (draft !== null && draft.trim() !== '' && Number.isFinite(parsed)) onCommit(parsed);
    setDraft(null);
  };
  return (
    <div className="space-y-2">
      <Label htmlFor={id}>{label}</Label>
      <Input
        id={id}
        type="number"
        step={step}
        min={min}
        max={max}
        value={draft ?? String(value)}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => {
          if (e.key === 'Enter') {
            e.preventDefault();
            commit();
          }
        }}
      />
      <p className="text-xs text-muted-foreground">{hint}</p>
    </div>
  );
}

interface BudgetFieldProps {
  id: string;
  label: string;
  value: number;
  defaultValue: number;
  budgets: number[];
  onCommit: (value: number) => void;
  hint: string;
}

/** A soft-token budget: one of the sizes the model's processor takes. */
function BudgetField({ id, label, value, defaultValue, budgets, onCommit, hint }: BudgetFieldProps) {
  return (
    <div className="space-y-2">
      <Label htmlFor={id}>{label}</Label>
      <Select value={String(value)} onValueChange={(v) => onCommit(Number(v))}>
        <SelectTrigger id={id} className="w-full">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {budgets.map((budget) => (
            <SelectItem key={budget} value={String(budget)}>
              {budget} tokens{budget === defaultValue ? ' (default)' : ''}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      <p className="text-xs text-muted-foreground">{hint}</p>
    </div>
  );
}

interface MediaSettingsFormProps {
  /** Defaults, limits and estimate (null until the settings load). */
  budget: MediaOptionsBudget | null;
  /** Why the settings failed to load; `onRetry` refetches them. */
  settingsError?: string | null;
  onRetry?: () => void;
  /** The options changed from their defaults. */
  overrides: MediaOptionsInput;
  onChange: (overrides: MediaOptionsInput) => void;
  /** Kinds of the selected media columns: only their options are shown and sent. */
  kinds: MediaKindCounts;
  idPrefix?: string;
}

/**
 * Media options of an Image / Audio / Video / Mixed embed, prefilled with the
 * defaults of config/media_embedding.json. Only the options of the selected
 * kinds are shown, and only changed values are sent with the job. The header
 * shows a row's worst-case size against the model's context window.
 */
export function MediaSettingsForm({
  budget,
  settingsError,
  onRetry,
  overrides,
  onChange,
  kinds,
  idPrefix = '',
}: MediaSettingsFormProps) {
  const options = budget?.effective(overrides) ?? null;
  const estimate = budget && options ? budget.estimateTokens(options, kinds) : null;
  const contextTokens = budget?.settings.contextTokens ?? null;
  const issues = budget && options ? budget.issues(options, kinds) : [];
  const set = (key: MediaOptionKey) => (value: number) => budget && onChange(budget.withOption(overrides, key, value));
  const hasKinds = Boolean(kinds.image || kinds.video || kinds.audio);
  const hasChanges = Object.keys(overrides).length > 0;

  return (
    <CollapsibleCard
      title="Media Settings"
      badge={
        estimate !== null && contextTokens !== null && hasKinds ? (
          <Badge variant={estimate > contextTokens ? 'destructive' : 'secondary'} className="text-xs">
            ≈ {estimate.toLocaleString('en-US')} of {contextTokens.toLocaleString('en-US')} tokens per row
          </Badge>
        ) : null
      }
      description="How much of each image, video and audio clip the model reads"
      headerExtra={
        !budget && settingsError ? (
          <div className="flex flex-wrap items-center gap-2 pt-1">
            <p className="text-xs text-destructive">Could not load the media settings: {settingsError}</p>
            {onRetry && (
              <Button variant="outline" size="sm" onClick={onRetry}>
                Retry
              </Button>
            )}
          </div>
        ) : null
      }
    >
      {!budget || !options ? (
        settingsError ? null : <p className="text-xs text-muted-foreground">Loading the media settings…</p>
      ) : !hasKinds ? (
        <p className="text-xs text-muted-foreground">Select a media column to see its settings.</p>
      ) : (
        <>
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            {kinds.image ? (
              <BudgetField
                id={`${idPrefix}media-image-tokens`}
                label="Tokens per image"
                value={options.imageMaxSoftTokens}
                defaultValue={budget.settings.defaults.imageMaxSoftTokens}
                budgets={budget.settings.softTokenBudgets}
                onCommit={set('imageMaxSoftTokens')}
                hint="More tokens keep more detail and cost more time."
              />
            ) : null}
            {kinds.video ? (
              <>
                <NumberField
                  id={`${idPrefix}media-video-fps`}
                  label="Frames per second"
                  value={options.videoFps}
                  onCommit={set('videoFps')}
                  step={0.5}
                  min={0}
                  max={budget.settings.videoFpsLimit}
                  hint={`Frames sampled from each second of video (at most ${budget.settings.videoFpsLimit}).`}
                />
                <NumberField
                  id={`${idPrefix}media-video-frames`}
                  label="Max frames"
                  value={options.videoMaxFrames}
                  onCommit={set('videoMaxFrames')}
                  step={1}
                  min={1}
                  max={budget.settings.videoFramesLimit}
                  hint="Longer clips are sampled evenly down to this many frames."
                />
                <BudgetField
                  id={`${idPrefix}media-video-tokens`}
                  label="Tokens per frame"
                  value={options.videoMaxSoftTokens}
                  defaultValue={budget.settings.defaults.videoMaxSoftTokens}
                  budgets={budget.settings.softTokenBudgets}
                  onCommit={set('videoMaxSoftTokens')}
                  hint="Video embeds its frames only, not its audio track."
                />
              </>
            ) : null}
            {kinds.audio ? (
              <NumberField
                id={`${idPrefix}media-audio-seconds`}
                label="Max audio length (seconds)"
                value={options.maxAudioSeconds}
                onCommit={set('maxAudioSeconds')}
                step={1}
                min={1}
                max={budget.settings.audioSecondsLimit}
                hint={`Longer audio is cut; each second costs ${budget.settings.audioTokensPerSecond} tokens.`}
              />
            ) : null}
          </div>
          <p className="text-xs text-muted-foreground">
            A row&apos;s text and media share one {budget.settings.contextTokens.toLocaleString('en-US')}-token
            context ({budget.settings.reservedTokens} tokens are kept for the text). Changed values are
            sent with the job; the others keep their defaults.
          </p>
          {issues.map((issue) => (
            <p key={issue} className="text-xs text-destructive">
              {issue}
            </p>
          ))}
          {hasChanges && (
            <Button variant="outline" size="sm" onClick={() => onChange({})}>
              Reset to defaults
            </Button>
          )}
        </>
      )}
    </CollapsibleCard>
  );
}
