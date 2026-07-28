'use client';

import { useEffect } from 'react';
import { useQuery } from '@apollo/client/react';
import { Microscope, GraduationCap, Telescope, Orbit, FlaskConical } from 'lucide-react';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/lib/ui-primitives/dialog';
import { Spinner } from '@/lib/ui-primitives/spinner';
import { GET_SAE_MODELS } from '@/lib/graphql/queries';
import type { SaeModelInfo } from '@/lib/types/types';
import { apolloClient } from '@/lib/utils/apollo-client';
import { IS_DEMO } from '@/lib/utils/demoMode';
import { markIntro, warmEmotionSearch, TOUR_MIN_VIEWPORT } from '@/lib/utils/demoOnboarding';
import {
  TOUR_PRESETS,
  TOUR_PRESET_ID,
  SAE_MAP_PRESET_ID,
  WORDNET_PRESET_ID,
  PROBE_PRESET_ID,
} from '@/lib/utils/tourPresets';
import { SAE_TOUR_MODEL_ID, SAE_TOUR_SAE_ID } from '@/lib/utils/saeTourSteps';

interface DemoIntroProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onStartTour: () => void;
  /** Starts the "Inspect SAE" tour's Explore segment (the label map). */
  onStartSaeTour: () => void;
  /** Starts the WordNet-galaxy tour (212k senses; the heavy one). */
  onStartWordnetTour: () => void;
  /** Starts the Glasgow embedding-probing tour. */
  onStartProbeTour: () => void;
  /** Manifest collection names; null while loading. Gates the entry buttons. */
  availableCollections: ReadonlySet<string> | null;
}

function EntryButton({
  icon,
  title,
  description,
  onClick,
  disabled,
}: {
  icon: React.ReactNode;
  title: string;
  description: string;
  onClick: () => void;
  disabled?: boolean;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className="flex w-full items-start gap-3 rounded-lg border p-3 text-left transition-colors hover:bg-accent hover:text-accent-foreground disabled:pointer-events-none disabled:opacity-50"
    >
      <span className="mt-0.5 text-muted-foreground">{icon}</span>
      <span>
        <span className="block text-sm font-medium">{title}</span>
        <span className="block text-xs text-muted-foreground">{description}</span>
      </span>
    </button>
  );
}

/**
 * First-visit welcome dialog for the demo. Guided-tour entry points plus
 * "explore on my own"; reopenable via `?intro=1` and the header Help button.
 * Static preset views stay reachable via `?preset=` deep links only — the
 * dialog lists guided experiences, not raw views.
 * Opening it pre-warms the emotion collection's search model.
 */
export function DemoIntro({
  open,
  onOpenChange,
  onStartTour,
  onStartSaeTour,
  onStartWordnetTour,
  onStartProbeTour,
  availableCollections,
}: DemoIntroProps) {
  useEffect(() => {
    if (open) warmEmotionSearch(apolloClient);
  }, [open]);

  // The spotlight tour is too cramped below tablet width — offer presets only.
  const tourFits = typeof window === 'undefined' || window.innerWidth >= TOUR_MIN_VIEWPORT;

  // Buttons stay disabled while the manifest loads or (outside the demo seed)
  // when their collection simply doesn't exist — a click would no-op silently.
  const hasCollection = (presetId: string) =>
    availableCollections?.has(TOUR_PRESETS[presetId].collection) ?? false;

  // The SAE entry needs the tour's exact pair in the DB — its own gate, since
  // SAE availability isn't part of the collections manifest.
  const { data: saeModelsData } = useQuery<{ saeModels: SaeModelInfo[] }>(GET_SAE_MODELS, {
    skip: !open,
  });
  const hasSaeTourPair = (saeModelsData?.saeModels ?? []).some(
    (m) => m.modelId === SAE_TOUR_MODEL_ID && m.saeId === SAE_TOUR_SAE_ID,
  );

  const handleOpenChange = (next: boolean) => {
    if (!next) markIntro('dismissed');
    onOpenChange(next);
  };

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Welcome to Orrery</DialogTitle>
          <DialogDescription>
            An observatory for embedding spaces: each point is a document, nearby points are semantically similar. Pick a starting point:
          </DialogDescription>
        </DialogHeader>
        <div className="flex flex-col gap-2">
          {tourFits && (
            <EntryButton
              icon={<GraduationCap className="h-4 w-4" />}
              title="Take the 90-second tour"
              description="A guided walk through the map, topics, search, and analytics, using the EMNLP abstracts dataset."
              disabled={!hasCollection(TOUR_PRESET_ID)}
              onClick={() => {
                markIntro('completed');
                onStartTour();
              }}
            />
          )}
          {tourFits && (
            <EntryButton
              icon={<FlaskConical className="h-4 w-4" />}
              title={TOUR_PRESETS[PROBE_PRESET_ID].label}
              description="Train and visualize probes for concreteness and valence on the embeddings of the Glasgow norms dataset."
              disabled={!hasCollection(PROBE_PRESET_ID)}
              onClick={() => {
                markIntro('completed');
                onStartProbeTour();
              }}
            />
          )}
          {tourFits && (
            <EntryButton
              icon={<Microscope className="h-4 w-4" />}
              title="Inspect SAE features"
              description="Visualize sparse-autoencoder features as constellations, inspect individual features and steer the model."
              disabled={!hasSaeTourPair || !hasCollection(SAE_MAP_PRESET_ID)}
              onClick={() => {
                markIntro('completed');
                onStartSaeTour();
              }}
            />
          )}
          {tourFits && (
            <EntryButton
              icon={<Orbit className="h-4 w-4" />}
              title="Explore in Nebula Mode"
              description="All WordNet 212,478 dictionary senses as a navigable galaxy with haze; expect higher loading times."
              disabled={!hasCollection(WORDNET_PRESET_ID)}
              onClick={() => {
                markIntro('completed');
                onStartWordnetTour();
              }}
            />
          )}
          <EntryButton
            icon={<Telescope className="h-4 w-4" />}
            title="Explore on my own"
            description="Close and wander; reopen it anytime from the ? button."
            onClick={() => handleOpenChange(false)}
          />
        </div>
        {availableCollections === null && (
          <p className="flex items-center gap-2 text-xs text-muted-foreground">
            <Spinner className="h-3 w-3" />
            Waking the backend — the options above enable in a moment.
          </p>
        )}
        {IS_DEMO && (
          <p className="text-xs text-muted-foreground">
            This public demo is read-only; the full platform embeds your own data.
          </p>
        )}
      </DialogContent>
    </Dialog>
  );
}
