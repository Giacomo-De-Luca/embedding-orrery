'use client';

import { useMemo } from 'react';
import type { EmbeddingProvider } from '@/lib/graphql/mutations';
import { useEmbeddingModelCatalog } from '@/lib/hooks/useEmbeddingModelCatalog';
import type { ModelSuggestionFilter } from '@/lib/utils/modelSuggestions';
import { SuggestionCombobox } from '@/lib/ui-primitives/suggestion-combobox';

interface ModelNameComboboxProps {
  id?: string;
  provider: EmbeddingProvider;
  value: string;
  onChange: (modelName: string) => void;
  placeholder?: string;
  /** Modalities every suggestion must embed (the selected media kinds). */
  requires?: readonly string[];
  /** Only media models (Image / Audio / Video / Mixed mode). */
  anyMedia?: boolean;
}

/**
 * The model-name field of the embed forms: any name (a Hub id or a local
 * path), with the provider's catalog models and the models existing
 * collections use as suggestions.
 */
export function ModelNameCombobox({
  id,
  provider,
  value,
  onChange,
  placeholder,
  requires,
  anyMedia,
}: ModelNameComboboxProps) {
  const { suggestions, error } = useEmbeddingModelCatalog();
  const requiresKey = (requires ?? []).join(',');
  const groups = useMemo(() => {
    const filter: ModelSuggestionFilter = {
      provider,
      requires: requiresKey ? requiresKey.split(',') : [],
      anyMedia,
    };
    return suggestions.groups(filter);
  }, [suggestions, provider, requiresKey, anyMedia]);

  return (
    <SuggestionCombobox
      id={id}
      value={value}
      onChange={onChange}
      groups={groups}
      placeholder={placeholder}
      emptyMessage={
        error
          ? `Could not load model suggestions (${error}): any model name or local path works.`
          : 'No suggested model matches: any model name or local path works.'
      }
    />
  );
}
