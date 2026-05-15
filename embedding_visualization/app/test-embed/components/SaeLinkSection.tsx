'use client';

import { useCallback, useMemo } from 'react';
import { useQuery } from '@apollo/client/react';
import Link from 'next/link';
import { ExternalLink } from 'lucide-react';
import { Badge } from '@/lib/ui-primitives/badge';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/lib/ui-primitives/card';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/lib/ui-primitives/select';
import { GET_SAE_MODELS } from '@/lib/graphql/queries';
import type { SaeModelInfo } from '@/lib/types/types';

interface SaeLinkSectionProps {
  collectionName: string;
  currentModelId: string | null;
  currentSaeId: string | null;
  onUpdate: (metadata: Record<string, unknown>) => Promise<unknown>;
}

const NONE_VALUE = '__none__';

export function SaeLinkSection({
  collectionName,
  currentModelId,
  currentSaeId,
  onUpdate,
}: SaeLinkSectionProps) {
  const { data: modelsData } = useQuery<{ saeModels: SaeModelInfo[] }>(GET_SAE_MODELS);
  const models = useMemo(() => modelsData?.saeModels ?? [], [modelsData]);

  const currentValue = currentModelId && currentSaeId
    ? `${currentModelId}::${currentSaeId}`
    : NONE_VALUE;

  const handleChange = useCallback(
    async (value: string) => {
      if (value === NONE_VALUE) {
        await onUpdate({ sae_model_id: null, sae_id: null });
      } else {
        const [modelId, saeId] = value.split('::');
        await onUpdate({ sae_model_id: modelId, sae_id: saeId });
      }
    },
    [onUpdate],
  );

  const isLinked = currentModelId && currentSaeId;

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base">SAE Link</CardTitle>
        <CardDescription>
          Connect this collection to SAE feature data for cross-navigation
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="space-y-1.5">
          <label className="text-xs font-medium text-muted-foreground">Linked SAE Model</label>
          <Select value={currentValue} onValueChange={handleChange}>
            <SelectTrigger>
              <SelectValue placeholder="Select an SAE model..." />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value={NONE_VALUE}>
                <span className="text-muted-foreground">None (not an SAE collection)</span>
              </SelectItem>
              {models.map((m) => (
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
        </div>

        {isLinked && (
          <div className="flex items-center gap-2">
            <Badge className="bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400 text-xs">
              Linked
            </Badge>
            <Link
              href={`/features?modelId=${encodeURIComponent(currentModelId)}&saeId=${encodeURIComponent(currentSaeId)}`}
              className="text-xs text-muted-foreground hover:text-foreground flex items-center gap-1"
            >
              Open in Feature Explorer
              <ExternalLink className="h-3 w-3" />
            </Link>
          </div>
        )}

        {models.length === 0 && (
          <p className="text-xs text-muted-foreground">
            No SAE models ingested yet. Use the SAE tab to download feature data first.
          </p>
        )}
      </CardContent>
    </Card>
  );
}
