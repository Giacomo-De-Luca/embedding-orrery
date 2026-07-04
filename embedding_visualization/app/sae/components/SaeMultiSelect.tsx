'use client';

import { Badge } from '@/lib/ui-primitives/badge';
import { Button } from '@/lib/ui-primitives/button';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/lib/ui-primitives/select';
import {
  Combobox,
  ComboboxChip,
  ComboboxChips,
  ComboboxChipsInput,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxItem,
  ComboboxList,
  useComboboxAnchor,
} from '@/lib/ui-primitives/combobox';
import { HOOK_TYPE_DISPLAY } from '@/lib/utils/saeCollections';
import type { SaeOption } from '../hooks/useSaeSelection';

interface SaeMultiSelectProps {
  modelId: string | null;
  modelOptions: string[];
  onModelChange: (modelId: string) => void;
  saeOptions: SaeOption[];
  selectedSaeIds: string[];
  onSaeIdsChange: (saeIds: string[]) => void;
}

function chipLabel(option: SaeOption): string {
  return `L${option.parsed.layerIndex} ${option.parsed.width}`;
}

/**
 * Model single-select (model load is the expensive operation) plus a
 * chips multi-select over that model's SAEs, with All/Clear shortcuts.
 */
export function SaeMultiSelect({
  modelId,
  modelOptions,
  onModelChange,
  saeOptions,
  selectedSaeIds,
  onSaeIdsChange,
}: SaeMultiSelectProps) {
  const chipsRef = useComboboxAnchor();
  const optionsById = new Map(saeOptions.map((o) => [o.saeId, o]));
  const allSelected = selectedSaeIds.length === saeOptions.length && saeOptions.length > 0;

  return (
    <div className="flex items-center gap-2 flex-wrap">
      {/* Model */}
      <Select value={modelId ?? ''} onValueChange={onModelChange}>
        <SelectTrigger className="w-44 h-8 text-xs">
          <SelectValue placeholder="Model" />
        </SelectTrigger>
        <SelectContent>
          {modelOptions.map((m) => (
            <SelectItem key={m} value={m}>
              <span className="font-mono text-xs">{m}</span>
            </SelectItem>
          ))}
        </SelectContent>
      </Select>

      {/* SAEs of the selected model */}
      <div className="flex-1 min-w-64 max-w-xl">
        <Combobox<string, true> multiple value={selectedSaeIds} onValueChange={onSaeIdsChange}>
          <ComboboxChips ref={chipsRef} className="min-h-8">
            {selectedSaeIds.map((saeId) => {
              const option = optionsById.get(saeId);
              return (
                <ComboboxChip key={saeId} className="font-mono text-[10px]">
                  {option ? chipLabel(option) : saeId}
                </ComboboxChip>
              );
            })}
            <ComboboxChipsInput
              placeholder={selectedSaeIds.length === 0 ? 'Select SAEs...' : 'Add...'}
              className="text-xs"
            />
          </ComboboxChips>
          <ComboboxContent anchor={chipsRef}>
            <ComboboxList>
              {saeOptions.map((option) => (
                <ComboboxItem key={option.saeId} value={option.saeId}>
                  <span className="truncate flex-1 text-xs">
                    L{option.parsed.layerIndex} · {HOOK_TYPE_DISPLAY[option.parsed.hookType]} ·{' '}
                    {option.parsed.width}
                  </span>
                  <span className="text-muted-foreground text-[10px] ml-auto shrink-0">
                    {option.featureCount.toLocaleString()}
                  </span>
                </ComboboxItem>
              ))}
            </ComboboxList>
            <ComboboxEmpty>No matching SAEs</ComboboxEmpty>
          </ComboboxContent>
        </Combobox>
      </div>

      {/* All / Clear shortcuts */}
      <Button
        variant="ghost"
        size="sm"
        className="h-7 px-2 text-xs"
        disabled={allSelected}
        onClick={() => onSaeIdsChange(saeOptions.map((o) => o.saeId))}
      >
        All
      </Button>
      <Button
        variant="ghost"
        size="sm"
        className="h-7 px-2 text-xs"
        disabled={selectedSaeIds.length === 0}
        onClick={() => onSaeIdsChange([])}
      >
        Clear
      </Button>

      {/* Selected count badge */}
      <Badge variant="secondary" className="text-[10px] shrink-0">
        {selectedSaeIds.length === 1 ? '1 SAE' : `${selectedSaeIds.length} SAEs`}
      </Badge>
    </div>
  );
}
