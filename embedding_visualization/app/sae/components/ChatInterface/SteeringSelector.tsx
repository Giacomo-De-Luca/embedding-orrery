'use client';

import { useMemo } from 'react';
import { useQuery } from '@apollo/client/react';
import { Cpu } from 'lucide-react';
import {
  Combobox,
  ComboboxInput,
  ComboboxContent,
  ComboboxList,
  ComboboxItem,
  ComboboxGroup,
  ComboboxLabel,
  ComboboxEmpty,
} from '@/lib/ui-primitives/combobox';
import { GET_SAE_MODELS } from '@/lib/graphql/queries';

interface SaeModelInfo {
  modelId: string;
  saeId: string;
  featureCount: number;
  activationCount: number;
}

interface SteeringSelectorProps {
  modelId: string | null;
  saeId: string | null;
  onSelect: (modelId: string, saeId: string) => void;
}

/** Extract a short layer label from saeId, e.g. "9-gemmascope-…" → "L9" */
function layerLabel(saeId: string): string {
  const match = saeId.match(/^(\d+)/);
  return match ? `L${match[1]}` : saeId.slice(0, 12);
}

export function SteeringSelector({ modelId, saeId, onSelect }: SteeringSelectorProps) {
  const { data } = useQuery<{ saeModels: SaeModelInfo[] }>(GET_SAE_MODELS);

  // Group by modelId
  const grouped = useMemo(() => {
    if (!data?.saeModels) return new Map<string, SaeModelInfo[]>();
    const map = new Map<string, SaeModelInfo[]>();
    for (const m of data.saeModels) {
      const list = map.get(m.modelId) ?? [];
      list.push(m);
      map.set(m.modelId, list);
    }
    return map;
  }, [data?.saeModels]);

  const currentLabel = modelId && saeId
    ? `${modelId} / ${layerLabel(saeId)}`
    : undefined;

  return (
    <Combobox
      value={currentLabel ?? ''}
      onValueChange={(val) => {
        // val format: "modelId::saeId"
        if (typeof val === 'string') {
          const [mId, sId] = val.split('::');
          if (mId && sId) onSelect(mId, sId);
        }
      }}
    >
      <ComboboxInput
        placeholder="Select model..."
        className="h-7 w-48 text-xs"
        showTrigger
      />
      <ComboboxContent>
        <ComboboxList>
          <ComboboxEmpty>No models available</ComboboxEmpty>
          {[...grouped.entries()].map(([model, items]) => (
            <ComboboxGroup key={model}>
              <ComboboxLabel>
                <span className="flex items-center gap-1.5">
                  <Cpu className="size-3" />
                  {model}
                </span>
              </ComboboxLabel>
              {items.map((item) => (
                <ComboboxItem
                  key={`${item.modelId}::${item.saeId}`}
                  value={`${item.modelId}::${item.saeId}`}
                >
                  <span className="flex items-center gap-2">
                    <span className="font-mono text-xs">{layerLabel(item.saeId)}</span>
                    <span className="text-muted-foreground text-[11px]">
                      {item.featureCount.toLocaleString()} features
                    </span>
                  </span>
                </ComboboxItem>
              ))}
            </ComboboxGroup>
          ))}
        </ComboboxList>
      </ComboboxContent>
    </Combobox>
  );
}
