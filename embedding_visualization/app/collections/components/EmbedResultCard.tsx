'use client';

import { Card, CardContent, CardHeader, CardTitle } from '@/lib/ui-primitives/card';
import type { EmbedDatasetResult } from '@/lib/graphql/mutations';

interface EmbedResultCardProps {
  result: EmbedDatasetResult;
  isImportMode?: boolean;
}

export function EmbedResultCard({ result, isImportMode = false }: EmbedResultCardProps) {
  const failText = isImportMode ? 'Import Failed' : 'Embedding Failed';
  const successText = isImportMode ? 'Import Complete!' : 'Embedding Complete!';
  const countLabel = isImportMode ? 'Total Imported:' : 'Total Embedded:';

  return (
    <Card className={result.error ? 'border-destructive' : 'border-green-500'}>
      <CardHeader>
        <CardTitle>
          {result.error ? `\u274C ${failText}` : `\u2705 ${successText}`}
        </CardTitle>
      </CardHeader>
      <CardContent>
        {result.error ? (
          <p className="text-destructive">{result.error}</p>
        ) : (
          <div className="space-y-2">
            <p><strong>Collection:</strong> {result.collectionName}</p>
            <p><strong>{countLabel}</strong> {result.totalEmbedded.toLocaleString()}</p>
            <p><strong>Embedding Dim:</strong> {result.embeddingDim}</p>
            <p><strong>Device:</strong> {result.device}</p>
            <p><strong>Duration:</strong> {result.durationSeconds.toFixed(2)}s</p>
            <p><strong>Projections:</strong> {result.projectionsComputed ? '\u2713 Computed' : 'Not computed'}</p>
            {result.embeddingProvider && (
              <p><strong>Model:</strong> {result.embeddingProvider} / {result.embeddingModel}</p>
            )}
            <div className="mt-4">
              <a
                href={`/?collection=${result.collectionName}`}
                className="text-blue-500 hover:underline font-medium"
              >
                View in Visualization &rarr;
              </a>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
