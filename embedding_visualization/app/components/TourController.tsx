'use client';

import { useEffect, useMemo } from 'react';
import { Joyride, EVENTS, STATUS, type EventData, type Step } from 'react-joyride';
import { apolloClient } from '@/lib/utils/apollo-client';
import { markTour, warmEmotionSearch } from '@/lib/utils/demoOnboarding';
import { TOUR_ANCHORS, TOUR_STEPS, type TourRuntime } from '@/lib/utils/tourSteps';

interface TourControllerProps {
  runtime: TourRuntime;
  /** Called when the tour ends for any reason (finished, skipped, closed). */
  onDone: () => void;
}

/**
 * Thin react-joyride v3 shell around the library-agnostic step definitions in
 * `lib/utils/tourSteps.ts`. Mounted lazily (next/dynamic in page.tsx) only
 * while a tour is requested, so regular visits never load the library.
 */
export function TourController({ runtime, onDone }: TourControllerProps) {
  // Direct `?tour=1` entries skip the welcome dialog — warm the search model
  // here as well (idempotent).
  useEffect(() => {
    warmEmotionSearch(apolloClient);
  }, []);

  const steps = useMemo<Step[]>(
    () =>
      TOUR_STEPS.map((def) => ({
        id: def.id,
        target: TOUR_ANCHORS[def.anchor],
        title: def.title,
        content: def.body,
        placement: def.placement ?? 'auto',
        blockTargetInteraction: !def.allowInteraction,
        before: def.prepare ? () => def.prepare!(runtime) : undefined,
        beforeTimeout: def.prepareTimeoutMs,
      })),
    [runtime],
  );

  const reducedMotion =
    typeof window !== 'undefined' &&
    window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  const handleEvent = (data: EventData) => {
    if (data.type !== EVENTS.TOUR_END) return;
    markTour(data.status === STATUS.FINISHED ? 'completed' : 'dismissed');
    onDone();
  };

  return (
    <Joyride
      steps={steps}
      run
      continuous
      onEvent={handleEvent}
      options={{
        skipBeacon: true,
        buttons: ['back', 'primary', 'skip', 'close'],
        showProgress: true,
        overlayClickAction: false,
        targetWaitTimeout: 10000,
        scrollDuration: reducedMotion ? 0 : 300,
        spotlightRadius: 12,
        zIndex: 100,
        backgroundColor: 'var(--popover)',
        arrowColor: 'var(--popover)',
        textColor: 'var(--popover-foreground)',
        primaryColor: 'var(--primary)',
      }}
    />
  );
}

export default TourController;
