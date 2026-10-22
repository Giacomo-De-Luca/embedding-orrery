/**
 * Suggestions of the free-text pickers (`lib/ui-primitives/suggestion-combobox.tsx`):
 * the model-name field (`modelSuggestions.ts`) and the prompt fields
 * (`promptPresets.ts`). Pure (no React).
 */

/** One suggestion: `value` is what picking it writes into the field. */
export interface SuggestionItem {
  value: string;
  /** Shown instead of the value (defaults to the value). */
  label?: string;
  /** Short muted facts on the same line, e.g. "768d · used by 2 collections". */
  detail?: string;
  /** A second, muted line. */
  description?: string;
}

export interface SuggestionGroup {
  label: string;
  items: SuggestionItem[];
}

/** Whether a suggestion matches the typed text (case-insensitive, any of its texts; a blank query matches all). */
export function matchesSuggestion(item: SuggestionItem, query: string): boolean {
  const needle = query.trim().toLowerCase();
  if (!needle) return true;
  return [item.value, item.label, item.detail, item.description].some(
    (text) => text?.toLowerCase().includes(needle),
  );
}
