'use client';

import { PromptPresets } from '@/lib/utils/promptPresets';
import { SuggestionCombobox, type SuggestionGroup } from './suggestion-combobox';

const QUERY_PROMPT_GROUPS = PromptPresets.queryGroups();

interface PromptComboboxProps {
  value: string;
  onChange: (value: string) => void;
  /** Suggestions; defaults to the query prompts (Auto + the shared presets). */
  groups?: SuggestionGroup[];
  placeholder?: string;
  className?: string;
  id?: string;
}

/** A prompt field: any prompt text, with prompt names suggested (lib/utils/promptPresets.ts). */
export function PromptCombobox({
  value,
  onChange,
  groups,
  placeholder = 'Select or type a prompt...',
  className,
  id,
}: PromptComboboxProps) {
  return (
    <SuggestionCombobox
      id={id}
      value={value}
      onChange={onChange}
      groups={groups ?? QUERY_PROMPT_GROUPS}
      placeholder={placeholder}
      className={className}
      emptyMessage="No preset matches: the typed text is sent as the prompt."
    />
  );
}
