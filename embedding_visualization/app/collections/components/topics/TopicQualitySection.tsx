'use client';

import { useMemo, useState } from 'react';
import { Button } from '@/lib/ui-primitives/button';
import { Badge } from '@/lib/ui-primitives/badge';
import { Separator } from '@/lib/ui-primitives/separator';
import { Spinner } from '@/lib/ui-primitives/spinner';
import { Label } from '@/lib/ui-primitives/label';
import { ToggleGroup, ToggleGroupItem } from '@/lib/ui-primitives/toggle-group';
import { ProgressModal } from '../EmbeddingProgressModal';
import { useTopicQuality } from '@/lib/hooks/useTopicQuality';
import {
  DEFAULT_SELECTED_METRICS,
  METRIC_OPTIONS,
  pickLevelMetrics,
  qualityBadges,
  qualityTitle,
  type QualityLevel,
  type QualityMetricsBlob,
} from '@/lib/utils/topicQualityFormat';

interface TopicQualitySectionProps {
  collectionName: string;
  hasSubtopics: boolean;
  /** Other topic operations running — scoring is disabled meanwhile */
  otherOpsLoading: boolean;
  /** Persisted scores from GET_COLLECTION_TOPICS: { topic: {...}, subtopic: {...} } */
  storedQualityMetrics?: Record<string, Record<string, unknown>> | null;
}

/**
 * "Topic Quality" controls + score badges. Self-contained: owns metric/level
 * selection and the evaluateTopics mutation (useTopicQuality); freshly computed
 * scores override the persisted ones per level.
 */
export function TopicQualitySection({
  collectionName,
  hasSubtopics,
  otherOpsLoading,
  storedQualityMetrics,
}: TopicQualitySectionProps) {
  const { evaluateTopics, loading } = useTopicQuality();
  const [selectedMetrics, setSelectedMetrics] = useState<string[]>(DEFAULT_SELECTED_METRICS);
  const [level, setLevel] = useState<QualityLevel>('topic');
  // Fresh results from this session, keyed by level; take precedence over stored.
  const [freshByLevel, setFreshByLevel] = useState<Partial<Record<QualityLevel, QualityMetricsBlob>>>({});

  const handleScore = async () => {
    const result = await evaluateTopics({
      collectionName,
      level,
      metrics: selectedMetrics,
    });
    if (result?.metrics && !result.error) {
      setFreshByLevel((prev) => ({ ...prev, [result.level as QualityLevel]: result.metrics! }));
    }
  };

  const levels: QualityLevel[] = hasSubtopics ? ['topic', 'subtopic'] : ['topic'];
  const displayed = useMemo(
    () =>
      levels
        .map((lvl) => ({
          level: lvl,
          metrics: freshByLevel[lvl] ?? pickLevelMetrics(storedQualityMetrics, lvl),
        }))
        .filter((entry) => entry.metrics),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [freshByLevel, storedQualityMetrics, hasSubtopics]
  );

  return (
    <div className="space-y-4">
      <h4 className="text-sm font-medium">Topic Quality</h4>
      <p className="text-xs text-muted-foreground">
        Score the current topics: cluster separation, keyword diversity and coherence.
      </p>

      <div className="space-y-3">
        <div className="space-y-2">
          <Label>Metrics</Label>
          <ToggleGroup
            type="multiple"
            variant="outline"
            value={selectedMetrics}
            onValueChange={(values) => {
              if (values.length > 0) setSelectedMetrics(values);
            }}
          >
            {METRIC_OPTIONS.map((option) => (
              <ToggleGroupItem
                key={option.name}
                value={option.name}
                className="text-xs"
                disabled={option.disabled}
                title={option.title}
              >
                {option.label}
                {option.hint && (
                  <span className="ml-1 text-[10px] text-muted-foreground">({option.hint})</span>
                )}
              </ToggleGroupItem>
            ))}
          </ToggleGroup>
        </div>

        {hasSubtopics && (
          <div className="space-y-2">
            <Label>Level</Label>
            <ToggleGroup
              type="single"
              variant="outline"
              value={level}
              onValueChange={(v) => {
                if (v) setLevel(v as QualityLevel);
              }}
            >
              <ToggleGroupItem value="topic" className="text-xs">
                Topics
              </ToggleGroupItem>
              <ToggleGroupItem
                value="subtopic"
                className="text-xs"
                title="Score the pre-reduction subtopics (the original density clusters)"
              >
                Subtopics
              </ToggleGroupItem>
            </ToggleGroup>
          </div>
        )}

        <Button
          variant="outline"
          size="sm"
          onClick={handleScore}
          disabled={loading || otherOpsLoading}
        >
          {loading ? (
            <>
              <Spinner className="h-4 w-4 mr-2" />
              Scoring Topics...
            </>
          ) : (
            'Score Topics'
          )}
        </Button>
      </div>

      {displayed.length > 0 && (
        <div className="space-y-3">
          <Separator />
          <div className="space-y-2">
            {displayed.map(({ level: lvl, metrics }) => (
              <div key={lvl} className="flex flex-wrap items-center gap-2">
                <span className="text-xs text-muted-foreground w-16 shrink-0 capitalize">{lvl}s</span>
                {qualityBadges(metrics).map((badge) => (
                  <Badge
                    key={badge.key}
                    variant="secondary"
                    className="font-mono text-[10px]"
                    title={qualityTitle(metrics)}
                  >
                    {badge.text}
                  </Badge>
                ))}
              </div>
            ))}
          </div>
        </div>
      )}

      {loading && (
        <ProgressModal
          jobId={`${collectionName}_evaluate`}
          title="Scoring Topic Quality"
          subtitle="Computing cluster and keyword quality metrics..."
          itemsLabel="stages"
        />
      )}
    </div>
  );
}
