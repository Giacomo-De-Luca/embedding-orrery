'use client';

import { useState } from 'react';
import { Badge } from '@/lib/ui-primitives/badge';
import { Button } from '@/lib/ui-primitives/button';
import { Checkbox } from '@/lib/ui-primitives/checkbox';
import { Label } from '@/lib/ui-primitives/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/lib/ui-primitives/select';
import { IS_DEMO } from '@/lib/utils/demoMode';
import { REDUCTION_STRATEGY_OPTIONS, ReductionRules } from '@/lib/utils/topicReduction';
import type { TopicTreeLinkage } from '@/lib/graphql/mutations';
import type { UseTopicMergeTreeReturn } from '@/lib/hooks/useTopicMergeTree';
import { JobProgressBody } from '@/app/collections/components/JobProgressBody';
import { SliderRow } from './SliderRow';

/** The tree linkages, labelled and explained like the Collections page strategies. */
const LINKAGE_OPTIONS = REDUCTION_STRATEGY_OPTIONS.filter(
  (o): o is (typeof o & { value: TopicTreeLinkage }) => o.value === 'average' || o.value === 'ward' || o.value === 'density',
);

/**
 * Topic slider of the Color section (colouring by topic_label): previews a cut
 * of the collection's merge tree in the browser, and Apply saves it. Apply is
 * hidden in the read-only demo (the slider still works: it only reads).
 */
export function TopicTreeControls({ topicTree }: { topicTree: UseTopicMergeTreeReturn }) {
  const {
    view, linkage, setLinkage, absorbNoise, setAbsorbNoise, minTopics, maxTopics, topicCount,
    setTopicCount, previewing, liveSlider, reset, apply, applying, applyError, jobId, ready,
  } = topicTree;
  // The thumb follows the pointer; large collections recut only on release.
  const [draft, setDraft] = useState<number | null>(null);
  const [llmLabels, setLlmLabels] = useState(false);
  if (!view) return null;

  const offered = ReductionRules.strategies(view.availableLinkages.includes('density'));
  const linkages = LINKAGE_OPTIONS.filter((o) => offered.includes(o.value));
  const hint = LINKAGE_OPTIONS.find((o) => o.value === linkage)?.hint;
  const shown = draft ?? topicCount;
  // Live sliders already recut on every step; a click that moves nothing starts no preview.
  const commit = (n: number) => {
    setDraft(null);
    if (n !== topicCount) setTopicCount(n);
  };

  return (
    <div className="space-y-3 pt-2" data-tour="topic-tree">
      <div className="flex items-center justify-between gap-2">
        <Label htmlFor="topic-tree-linkage" className="text-sm font-normal" title={hint}>
          Topic tree
        </Label>
        {previewing && <Badge variant="secondary" className="text-xs">Previewing</Badge>}
      </div>
      <Select
        value={linkage}
        onValueChange={(v) => setLinkage(v as TopicTreeLinkage)}
        disabled={applying}
      >
        <SelectTrigger id="topic-tree-linkage" className="w-full">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {linkages.map((o) => (
            <SelectItem key={o.value} value={o.value} title={o.hint}>{o.shortLabel}</SelectItem>
          ))}
        </SelectContent>
      </Select>

      {linkage === 'density' && (
        <div className="flex items-center space-x-2">
          <Checkbox
            id="topic-tree-absorb-noise"
            checked={absorbNoise}
            disabled={applying}
            onCheckedChange={(checked) => setAbsorbNoise(checked === true)}
          />
          <Label htmlFor="topic-tree-absorb-noise" className="font-normal cursor-pointer text-sm">
            Absorb noise
          </Label>
        </div>
      )}

      <SliderRow
        label="Topics"
        valueLabel={`${shown} of ${maxTopics}`}
        title="Merge the collection's original topics along the tree; noise is not counted."
        min={minTopics}
        max={maxTopics}
        step={1}
        value={shown}
        disabled={applying}
        onChange={(n) => {
          setDraft(n);
          if (liveSlider) setTopicCount(n);
        }}
        onCommit={commit}
      />

      <div className="flex items-center gap-2">
        <Button variant="outline" size="sm" onClick={reset} disabled={!previewing || applying}>
          Reset
        </Button>
        {!IS_DEMO && (
          <Button
            size="sm"
            onClick={() => apply(llmLabels)}
            disabled={!previewing || applying || !ready}
            title={ready ? undefined : 'Loading the selected topic tree'}
          >
            Apply
          </Button>
        )}
      </div>
      {!IS_DEMO && (
        <div className="flex items-center space-x-2">
          <Checkbox
            id="topic-tree-llm-labels"
            checked={llmLabels}
            disabled={applying}
            onCheckedChange={(checked) => setLlmLabels(checked === true)}
          />
          <Label htmlFor="topic-tree-llm-labels" className="font-normal cursor-pointer text-sm">
            LLM labels for merged topics
          </Label>
        </div>
      )}

      {applying && jobId && (
        <div className="rounded-md border p-3">
          <JobProgressBody jobId={jobId} title="Saving topics" itemsLabel="topics" cancellable />
        </div>
      )}
      {applyError && !applying && (
        <p role="alert" className="text-xs text-destructive">{applyError}</p>
      )}
    </div>
  );
}
