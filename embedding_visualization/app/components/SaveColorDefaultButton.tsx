'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { useMutation } from '@apollo/client/react';
import { Check, Save } from 'lucide-react';
import { Button } from '@/lib/ui-primitives/button';
import { UPDATE_COLLECTION_METADATA } from '../../lib/graphql/mutations';
import { GET_COLLECTIONS } from '../../lib/graphql/queries';
import {
  useVisualizationStore,
  selectColorByField,
  selectColorScale,
  selectCategoricalPalette,
} from '../../lib/stores/useVisualizationStore';
import { useCollections } from '../../lib/hooks/useCollections';
import { serializeDefaultColorScheme } from '../../lib/utils/colorScaleUrl';

interface SaveColorDefaultButtonProps {
  /** Collection to persist the default onto (null when none selected). */
  collectionName: string | null;
}

/**
 * Saves the current colouring (field + scale + palette) as the active
 * collection's default colour scheme, persisted into `extra_metadata` via
 * `updateCollectionMetadata`. Applied on collection load by page.tsx when no
 * URL `colorBy` is present. WYSIWYG: captures whatever the plot shows now.
 */
export function SaveColorDefaultButton({ collectionName }: SaveColorDefaultButtonProps) {
  const colorByField = useVisualizationStore(selectColorByField);
  const colorScale = useVisualizationStore(selectColorScale);
  const categoricalPalette = useVisualizationStore(selectCategoricalPalette);
  const { collections } = useCollections();

  const [updateMetadata, { loading }] = useMutation(UPDATE_COLLECTION_METADATA, {
    refetchQueries: [{ query: GET_COLLECTIONS }],
  });

  // Transient "Saved ✓" confirmation (Toaster/sonner is not mounted app-wide).
  const [justSaved, setJustSaved] = useState(false);
  const savedTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => () => { if (savedTimer.current) clearTimeout(savedTimer.current); }, []);

  const hasDefault = !!(collectionName && collections?.[collectionName]?.defaultColorScheme);

  const flashSaved = useCallback(() => {
    setJustSaved(true);
    if (savedTimer.current) clearTimeout(savedTimer.current);
    savedTimer.current = setTimeout(() => setJustSaved(false), 2000);
  }, []);

  const handleSave = useCallback(async () => {
    if (!collectionName || !colorByField) return;
    const scheme = serializeDefaultColorScheme(colorByField, colorScale, categoricalPalette);
    try {
      await updateMetadata({
        variables: { collectionName, metadata: { default_color_scheme: JSON.stringify(scheme) } },
      });
      flashSaved();
    } catch (err) {
      console.error('Failed to save default colour scheme:', err);
    }
  }, [collectionName, colorByField, colorScale, categoricalPalette, updateMetadata, flashSaved]);

  const handleClear = useCallback(async () => {
    if (!collectionName) return;
    try {
      // Sending null removes the key (backend update_dataset drops null values).
      await updateMetadata({
        variables: { collectionName, metadata: { default_color_scheme: null } },
      });
    } catch (err) {
      console.error('Failed to clear default colour scheme:', err);
    }
  }, [collectionName, updateMetadata]);

  const disabled = !collectionName || !colorByField || loading;

  return (
    <div className="flex items-center gap-2">
      <Button
        type="button"
        variant="outline"
        size="sm"
        onClick={handleSave}
        disabled={disabled}
        title={hasDefault
          ? 'Update this collection’s default colour scheme'
          : 'Save the current colouring as this collection’s default'}
      >
        {justSaved ? <Check className="text-green-600" /> : <Save />}
        {justSaved ? 'Saved' : hasDefault ? 'Update default' : 'Save as default'}
      </Button>
      {hasDefault && !justSaved && (
        <Button
          type="button"
          variant="ghost"
          size="xs"
          onClick={handleClear}
          disabled={loading}
          className="text-muted-foreground"
        >
          Clear
        </Button>
      )}
    </div>
  );
}
