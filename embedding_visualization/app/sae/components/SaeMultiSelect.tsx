'use client';

import { useMemo } from 'react';
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
  ComboboxCollection,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxGroup,
  ComboboxItem,
  ComboboxLabel,
  ComboboxList,
  useComboboxAnchor,
} from '@/lib/ui-primitives/combobox';
import {
  HOOK_TYPE_DISPLAY,
  describeSae,
  formatSaeLabel,
  groupSaeIdsBySite,
  matchesSaeQuery,
  type SaeSiteGroup,
} from '@/lib/utils/saeCollections';
import type { SaeOption } from '../hooks/useSaeSelection';
import { SaeSiteDot } from './SaeSiteDot';

interface SaeMultiSelectProps {
  modelId: string | null;
  modelOptions: string[];
  onModelChange: (modelId: string) => void;
  saeOptions: SaeOption[];
  selectedSaeIds: string[];
  onSaeIdsChange: (saeIds: string[]) => void;
}

/**
 * Model single-select (model load is the expensive operation) plus a
 * chips multi-select over that model's SAEs, grouped by hook site, with
 * All/Clear shortcuts.
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
  // saeOptions are sorted site-first, so each group keeps layer → width order
  const siteGroups = useMemo(
    () => groupSaeIdsBySite(saeOptions.map((o) => o.saeId)),
    [saeOptions],
  );
  const allSelected = selectedSaeIds.length === saeOptions.length && saeOptions.length > 0;

  return (
    // `contents` makes the three children direct items of FeatureHeader's
    // grid (md+) / flex-wrap (below md) container, so the model select and
    // chips box land in the same columns as the index nav and search below.
    <div className="contents">
      {/* Model */}
      <Select value={modelId ?? ''} onValueChange={onModelChange}>
        <SelectTrigger className="w-44 md:w-full h-8 text-xs">
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
      <div className="flex-1 min-w-64 md:min-w-0 max-w-xl">
        <Combobox<string, true>
          multiple
          value={selectedSaeIds}
          onValueChange={onSaeIdsChange}
          items={siteGroups}
          itemToStringLabel={describeSae}
          filter={matchesSaeQuery}
        >
          <ComboboxChips ref={chipsRef} className="min-h-8">
            {selectedSaeIds.map((saeId) => {
              const option = optionsById.get(saeId);
              return (
                <ComboboxChip
                  key={saeId}
                  className="font-mono text-[10px]"
                  title={option ? describeSae(option.parsed) : saeId}
                >
                  {option && <SaeSiteDot hookType={option.parsed.hookType} />}
                  {option ? formatSaeLabel(option.parsed) : saeId}
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
              {(group: SaeSiteGroup) => (
                <ComboboxGroup key={group.value} items={group.items}>
                  <ComboboxLabel className="flex items-center gap-1.5">
                    <SaeSiteDot hookType={group.value} />
                    {HOOK_TYPE_DISPLAY[group.value]}
                  </ComboboxLabel>
                  <ComboboxCollection>
                    {(saeId: string) => {
                      const option = optionsById.get(saeId);
                      if (!option) return null;
                      return (
                        <ComboboxItem key={saeId} value={saeId}>
                          <span className="truncate flex-1 text-xs">
                            {formatSaeLabel(option.parsed, { hook: false })}
                          </span>
                          <span className="text-muted-foreground text-[10px] ml-auto shrink-0">
                            {option.featureCount.toLocaleString()}
                          </span>
                        </ComboboxItem>
                      );
                    }}
                  </ComboboxCollection>
                </ComboboxGroup>
              )}
            </ComboboxList>
            <ComboboxEmpty>No matching SAEs</ComboboxEmpty>
          </ComboboxContent>
        </Combobox>
      </div>

      {/* All / Clear shortcuts + count — one group so this component
          contributes exactly three cells to the parent grid */}
      <div className="flex items-center gap-2">
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
        <Badge variant="secondary" className="text-[10px] shrink-0">
          {selectedSaeIds.length === 1 ? '1 SAE' : `${selectedSaeIds.length} SAEs`}
        </Badge>
      </div>
    </div>
  );
}
