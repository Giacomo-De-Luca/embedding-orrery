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
import { formatSaeLabel } from '@/lib/utils/saeCollections';

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
    ? `${modelId} / ${formatSaeLabel(saeId)}`
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
                    <span className="font-mono text-xs">{formatSaeLabel(item.saeId)}</span>
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
