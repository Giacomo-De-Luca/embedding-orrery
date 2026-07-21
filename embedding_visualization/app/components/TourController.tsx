'use client';

import { useMemo } from 'react';
import { X } from 'lucide-react';
import {
  Joyride,
  EVENTS,
  STATUS,
  type EventData,
  type Step,
  type TooltipRenderProps,
} from 'react-joyride';
import { markTour } from '@/lib/utils/demoOnboarding';
import { TOUR_ANCHORS, TOUR_STEPS, type TourRuntime } from '@/lib/utils/tourSteps';

interface TourControllerProps {
  runtime: TourRuntime;
  /** Called when the tour ends for any reason (finished, skipped, closed). */
  onDone: () => void;
}

/**
 * Frosted-glass tour tooltip matching the plot hover tooltip (`.frosted-tooltip`
 * in globals.css): same surface, typography, and `border-foreground/15` divider.
 * backdrop-filter is inline for the same reason as FrostedTooltip — it must
 * hold up over the WebGL canvas.
 */
function TourTooltip({
  step,
  index,
  size,
  isLastStep,
  backProps,
  closeProps,
  primaryProps,
  skipProps,
  tooltipProps,
}: TooltipRenderProps) {
  return (
    <div
      {...tooltipProps}
      className="frosted-tooltip tour-tooltip relative w-[340px] max-w-[90vw]"
      style={{
        backdropFilter: 'blur(12px) saturate(150%)',
        WebkitBackdropFilter: 'blur(12px) saturate(150%)',
      }}
    >
      <button
        {...closeProps}
        className="absolute right-2.5 top-2.5 rounded-sm opacity-50 transition-opacity hover:opacity-100"
      >
        <X className="h-3.5 w-3.5" />
      </button>

      {step.title != null && (
        <div className="pr-6 text-sm font-semibold break-words">{step.title}</div>
      )}
      <div className="mt-1.5 text-xs leading-relaxed opacity-80">{step.content}</div>

      <div className="mt-2.5 flex items-center justify-between border-t border-foreground/15 pt-2 text-xs">
        <button {...skipProps} className="opacity-60 transition-opacity hover:opacity-100">
          Skip tour
        </button>
        <div className="flex items-center gap-2.5">
          <span className="tabular-nums opacity-50">
            {index + 1} / {size}
          </span>
          {index > 0 && (
            <button {...backProps} className="opacity-60 transition-opacity hover:opacity-100">
              Back
            </button>
          )}
          <button
            {...primaryProps}
            className="rounded-md bg-foreground/80 px-2.5 py-1 font-medium text-background backdrop-blur-sm transition-colors hover:bg-foreground/90"
          >
            {isLastStep ? 'Done' : 'Next'}
          </button>
        </div>
      </div>
    </div>
  );
}

/**
 * Thin react-joyride v3 shell around the library-agnostic step definitions in
 * `lib/utils/tourSteps.ts`. Mounted lazily (next/dynamic in page.tsx) only
 * while a tour is requested, so regular visits never load the library.
 */
export function TourController({ runtime, onDone }: TourControllerProps) {
  const steps = useMemo<Step[]>(
    () =>
      TOUR_STEPS.map((def) => ({
        id: def.id,
        target: TOUR_ANCHORS[def.anchor],
        title: def.title,
        content: def.body,
        placement: def.placement ?? 'auto',
        // 'center' placement has no spotlight cutout, so the overlay would
        // block ALL input — interactive plot steps drop the overlay entirely.
        hideOverlay: def.allowInteraction === true,
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
    // A mid-tour skip must not strand the analytics step's topic isolation
    // (after a completed finale this is a no-op on the fresh collection).
    runtime.clearTopicSelection();
    markTour(data.status === STATUS.FINISHED ? 'completed' : 'dismissed');
    onDone();
  };

  return (
    <Joyride
      steps={steps}
      run
      continuous
      onEvent={handleEvent}
      tooltipComponent={TourTooltip}
      // No opacity fade on the floater wrapper: the animated layer is part of
      // what defeats backdrop-filter sampling under the tooltip.
      styles={{ floater: { transition: 'none' } }}
      options={{
        skipBeacon: true,
        overlayClickAction: false,
        targetWaitTimeout: 10000,
        scrollDuration: reducedMotion ? 0 : 300,
        spotlightRadius: 12,
        zIndex: 100,
        // The scene should stay alive behind the tour: light dim, spotlight
        // target always interactive, no library arrow on the frosted surface.
        overlayColor: 'rgba(0, 0, 0, 0.25)',
        blockTargetInteraction: false,
        arrowColor: 'transparent',
      }}
    />
  );
}

export default TourController;
