'use client';

import { Card, CardContent, CardHeader, CardTitle } from '@/lib/ui-primitives/card';
import type { EmbedDatasetResult } from '@/lib/graphql/mutations';
import { isJobCancelled } from '@/lib/utils/jobCancellation';

interface EmbedResultCardProps {
  result: EmbedDatasetResult;
  isImportMode?: boolean;
}

export function EmbedResultCard({ result, isImportMode = false }: EmbedResultCardProps) {
  const failText = isImportMode ? 'Import Failed' : 'Embedding Failed';
  const successText = isImportMode ? 'Import Complete!' : 'Embedding Complete!';
  const cancelledText = isImportMode ? 'Import Cancelled' : 'Embedding Cancelled';
  const countLabel = isImportMode ? 'Total Imported:' : 'Total Embedded:';
  // A cancel is not a failure: the items embedded before it are kept
  const cancelled = isJobCancelled(result.error);
  const failed = Boolean(result.error) && !cancelled;

  return (
    <Card className={failed ? 'border-destructive' : cancelled ? 'border-muted-foreground/40' : 'border-green-500'}>
      <CardHeader>
        <CardTitle>
          {failed ? `\u274C ${failText}` : cancelled ? `\u23F9 ${cancelledText}` : `\u2705 ${successText}`}
        </CardTitle>
      </CardHeader>
      <CardContent>
        {failed ? (
          <p className="text-destructive">{result.error}</p>
        ) : (
          <div className="space-y-2">
            {cancelled && (
              <p className="text-muted-foreground">
                Stopped early. If the job is listed under Jobs, Resume finishes it.
              </p>
            )}
            <p><strong>Collection:</strong> {result.collectionName}</p>
            <p><strong>{countLabel}</strong> {result.totalEmbedded.toLocaleString()}</p>
            <p><strong>Embedding Dim:</strong> {result.embeddingDim}</p>
            <p><strong>Device:</strong> {result.device}</p>
            <p><strong>Duration:</strong> {result.durationSeconds.toFixed(2)}s</p>
            <p><strong>Projections:</strong> {result.projectionsComputed ? '\u2713 Computed' : 'Not computed'}</p>
            {result.embeddingProvider && (
              <p><strong>Model:</strong> {result.embeddingProvider} / {result.embeddingModel}</p>
            )}
            {/* Non-cancelling issues: truncated documents, media rows that could not be decoded */}
            {(result.warnings ?? []).length > 0 && (
              <div className="rounded-md border border-amber-500/40 bg-amber-500/10 p-3 text-sm">
                <p className="font-medium">Warnings</p>
                <ul className="mt-1 list-disc space-y-1 pl-5">
                  {result.warnings!.map((warning) => (
                    <li key={warning}>{warning}</li>
                  ))}
                </ul>
              </div>
            )}
            {(!cancelled || result.projectionsComputed) && (
              <div className="mt-4">
                <a
                  href={`/?collection=${result.collectionName}`}
                  className="text-blue-500 hover:underline font-medium"
                >
                  View in Visualization &rarr;
                </a>
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
