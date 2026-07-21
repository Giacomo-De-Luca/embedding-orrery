'use client';

import { useEffect } from 'react';
import { Compass, Palette, GraduationCap, Telescope } from 'lucide-react';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/lib/ui-primitives/dialog';
import { apolloClient } from '@/lib/utils/apollo-client';
import { IS_DEMO } from '@/lib/utils/demoMode';
import { markIntro, warmEmotionSearch, TOUR_MIN_VIEWPORT } from '@/lib/utils/demoOnboarding';
import { TOUR_PRESETS, TOUR_PRESET_ID } from '@/lib/utils/tourPresets';

interface DemoIntroProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onStartTour: () => void;
  onApplyPreset: (presetId: string) => void;
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
 * First-visit welcome dialog for the demo. Three goal-oriented entry points
 * plus "explore on my own"; reopenable via `?intro=1` and the header Help
 * button. Opening it pre-warms the emotion collection's search model.
 */
export function DemoIntro({
  open,
  onOpenChange,
  onStartTour,
  onApplyPreset,
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
            An observatory for embedding spaces: each point is a document placed by meaning —
            nearby points say similar things. Pick a starting point:
          </DialogDescription>
        </DialogHeader>
        <div className="flex flex-col gap-2">
          {tourFits && (
            <EntryButton
              icon={<GraduationCap className="h-4 w-4" />}
              title="Take the 90-second tour"
              description="A guided walk through the map, topics, search, and analytics."
              disabled={!hasCollection(TOUR_PRESET_ID)}
              onClick={() => {
                markIntro('completed');
                onStartTour();
              }}
            />
          )}
          <EntryButton
            icon={<Compass className="h-4 w-4" />}
            title={TOUR_PRESETS['emnlp-topics'].label}
            description={TOUR_PRESETS['emnlp-topics'].description}
            disabled={!hasCollection('emnlp-topics')}
            onClick={() => {
              markIntro('completed');
              onApplyPreset('emnlp-topics');
            }}
          />
          <EntryButton
            icon={<Palette className="h-4 w-4" />}
            title={TOUR_PRESETS['xkcd-manifold'].label}
            description={TOUR_PRESETS['xkcd-manifold'].description}
            disabled={!hasCollection('xkcd-manifold')}
            onClick={() => {
              markIntro('completed');
              onApplyPreset('xkcd-manifold');
            }}
          />
          <EntryButton
            icon={<Telescope className="h-4 w-4" />}
            title="Explore on my own"
            description="Close this and wander — reopen it anytime from the ? button."
            onClick={() => handleOpenChange(false)}
          />
        </div>
        {IS_DEMO && (
          <p className="text-xs text-muted-foreground">
            This public demo is read-only; the full platform embeds your own data.
          </p>
        )}
      </DialogContent>
    </Dialog>
  );
}
