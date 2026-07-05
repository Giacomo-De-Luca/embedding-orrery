'use client';

import { Button } from '@/lib/ui-primitives/button';
import { Spinner } from '@/lib/ui-primitives/spinner';

interface EmbedFooterBarProps {
  /** One-line config recap (from buildEmbedSummary) */
  summary: string;
  ctaLabel: string;
  onSubmit: () => void;
  loading: boolean;
  /** Reasons the CTA is disabled (from getEmbedValidationIssues); empty = ready */
  issues: string[];
}

/**
 * Sticky bottom bar carrying the primary embed CTA, a config recap, and the
 * first blocking validation issue — so the launch action is always visible
 * without scrolling and never silently disabled.
 */
export function EmbedFooterBar({ summary, ctaLabel, onSubmit, loading, issues }: EmbedFooterBarProps) {
  const blocked = issues.length > 0;

  return (
    <div className="sticky bottom-0 z-40 -mx-2 border-t bg-background/85 backdrop-blur-sm px-4 py-3">
      <div className="flex items-center justify-between gap-4">
        <div className="min-w-0">
          <p className="text-sm text-muted-foreground truncate">{summary}</p>
          {blocked && (
            <p className="text-xs text-destructive mt-0.5">{issues[0]}</p>
          )}
        </div>
        <Button
          onClick={onSubmit}
          disabled={loading || blocked}
          size="lg"
          className="shrink-0"
        >
          {loading ? <Spinner className="mr-2 h-4 w-4" /> : null}
          {ctaLabel}
        </Button>
      </div>
    </div>
  );
}
