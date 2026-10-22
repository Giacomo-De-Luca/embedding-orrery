'use client';

import * as React from 'react';
import { Autocomplete as AutocompletePrimitive } from '@base-ui/react';

import { cn } from '@/lib/utils/utils';
import { matchesSuggestion, type SuggestionGroup, type SuggestionItem } from '@/lib/utils/suggestions';
import {
  ComboboxCollection,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxGroup,
  ComboboxInput,
  ComboboxItem,
  ComboboxLabel,
  ComboboxList,
} from '@/lib/ui-primitives/combobox';

// The suggestion shapes and matcher live in lib/utils (pure, tested there).
export { matchesSuggestion, type SuggestionGroup, type SuggestionItem };

const itemValue = (item: SuggestionItem) => item.value;

interface SuggestionComboboxProps {
  value: string;
  onChange: (value: string) => void;
  groups: SuggestionGroup[];
  id?: string;
  placeholder?: string;
  className?: string;
  disabled?: boolean;
  /** Shown when the typed text matches no suggestion (free text is still accepted). */
  emptyMessage?: string;
  'aria-invalid'?: boolean;
}

/**
 * A free-text field with grouped suggestions (base-ui Autocomplete: keyboard
 * navigation, ARIA and a portalled popup). Any text is accepted; picking a
 * suggestion writes its value. Opening the list shows every suggestion, and
 * typing filters them.
 */
export function SuggestionCombobox({
  value,
  onChange,
  groups,
  id,
  placeholder,
  className,
  disabled,
  emptyMessage = 'No suggestions: the typed text is used as is.',
  'aria-invalid': ariaInvalid,
}: SuggestionComboboxProps) {
  // Until the user types after opening, the list is not filtered: a field
  // holding a suggestion's value would otherwise list that suggestion alone.
  const [typed, setTyped] = React.useState(false);

  return (
    <AutocompletePrimitive.Root
      items={groups}
      value={value}
      onValueChange={(next, details) => {
        if (details.reason === 'input-change') setTyped(true);
        onChange(next);
      }}
      onOpenChange={(open, details) => {
        // Typing opens the list after the value change: keep filtering then.
        if (open && details.reason !== 'input-change') setTyped(false);
      }}
      itemToStringValue={itemValue}
      filter={typed ? matchesSuggestion : null}
      openOnInputClick
      disabled={disabled}
    >
      <ComboboxInput
        id={id}
        placeholder={placeholder}
        className={cn('w-full', className)}
        disabled={disabled}
        aria-invalid={ariaInvalid}
      />
      <ComboboxContent>
        <ComboboxEmpty>{emptyMessage}</ComboboxEmpty>
        <ComboboxList>
          {(group: SuggestionGroup) => (
            <ComboboxGroup key={group.label} items={group.items}>
              <ComboboxLabel>{group.label}</ComboboxLabel>
              <ComboboxCollection>
                {(item: SuggestionItem) => (
                  <ComboboxItem key={item.value} value={item} className="items-start">
                    <span className="flex min-w-0 flex-col">
                      <span className="flex min-w-0 items-baseline gap-2">
                        <span className="truncate">{item.label ?? item.value}</span>
                        {item.detail && (
                          <span className="shrink-0 text-[11px] text-muted-foreground">{item.detail}</span>
                        )}
                      </span>
                      {item.description && (
                        <span className="text-xs text-muted-foreground">{item.description}</span>
                      )}
                    </span>
                  </ComboboxItem>
                )}
              </ComboboxCollection>
            </ComboboxGroup>
          )}
        </ComboboxList>
      </ComboboxContent>
    </AutocompletePrimitive.Root>
  );
}
