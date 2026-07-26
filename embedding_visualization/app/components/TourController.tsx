'use client';

import { useMemo } from 'react';
import { useTheme } from 'next-themes';
import { X } from 'lucide-react';
import {
  Joyride,
  EVENTS,
  STATUS,
  type EventData,
  type Step,
  type TooltipRenderProps,
} from 'react-joyride';
import { markTourKey, type OnboardingMark } from '@/lib/utils/demoOnboarding';
import type { TourStepDefinitionBase } from '@/lib/utils/tourSteps';

/*
 * The runtime is intentionally type-erased here: each tour's steps module
 * (tourSteps.ts, saeTourSteps.ts) pairs its own runtime interface with its
 * step definitions, and the page passes a matching pair — this shell only
 * threads the value through to prepare hooks.
 */
/* eslint-disable @typescript-eslint/no-explicit-any */
interface TourControllerProps {
  steps: ReadonlyArray<TourStepDefinitionBase<string, any>>;
  /** Anchor name → CSS selector map for the page hosting the tour. */
  anchors: Readonly<Record<string, string>>;
  runtime: any;
  /** localStorage key recording this tour's completed/dismissed outcome. */
  storageKey: string;
  /** Page-specific cleanup run when the tour ends for any reason. */
  onBeforeEnd?: (runtime: any) => void;
  /**
   * Called when the tour ends for any reason. `outcome` distinguishes a real
   * finish ('completed') from skip/close ('dismissed') — chained tours use it
   * to decide whether to hand off to their next segment.
   */
  onDone: (outcome: OnboardingMark) => void;
}
/* eslint-enable @typescript-eslint/no-explicit-any */

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
export function TourController({
  steps: stepDefs,
  anchors,
  runtime,
  storageKey,
  onBeforeEnd,
  onDone,
}: TourControllerProps) {
  const steps = useMemo<Step[]>(
    () =>
      stepDefs.map((def) => ({
        id: def.id,
        target: anchors[def.anchor],
        title: def.title,
        content: def.body,
        placement: def.placement ?? 'auto',
        // 'center' placement has no spotlight cutout, so the overlay would
        // block ALL input — interactive plot steps drop the overlay entirely.
        hideOverlay: def.allowInteraction === true,
        before: def.prepare ? () => def.prepare!(runtime) : undefined,
        beforeTimeout: def.prepareTimeoutMs,
        // Collection-switching steps sit behind the page's own full-screen
        // loader — joyride's waiting spinner on top of it reads as a bug.
        ...(def.suppressWaitLoader ? { loaderComponent: null } : {}),
      })),
    [stepDefs, anchors, runtime],
  );

  const reducedMotion =
    typeof window !== 'undefined' &&
    window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  const { resolvedTheme } = useTheme();
  const isDark = resolvedTheme === 'dark';

  const handleEvent = (data: EventData) => {
    if (data.type !== EVENTS.TOUR_END) return;
    // Page-specific cleanup (e.g. the Explore tour drops its topic isolation
    // so a mid-tour skip can't strand it).
    onBeforeEnd?.(runtime);
    const outcome: OnboardingMark =
      data.status === STATUS.FINISHED ? 'completed' : 'dismissed';
    markTourKey(storageKey, outcome);
    onDone(outcome);
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
      styles={{
        floater: { transition: 'none' },
        // Dark mode: a translucent black overlay is invisible over the pure-
        // black plot, so the light theme's "dim everything but the target"
        // effect vanishes. Draw the cutout explicitly instead — joyride
        // renders any non-empty `spotlight` as an outline path over the hole
        // (fill:none), so a soft ring + glow marks the highlighted area.
        ...(isDark
          ? {
              spotlight: {
                stroke: 'rgba(255, 255, 255, 0.5)',
                strokeWidth: 1.5,
                style: {
                  pointerEvents: 'none',
                  filter: 'drop-shadow(0 0 10px rgba(140, 170, 255, 0.65))',
                },
              },
            }
          : {}),
      }}
      options={{
        skipBeacon: true,
        overlayClickAction: false,
        targetWaitTimeout: 10000,
        scrollDuration: reducedMotion ? 0 : 300,
        spotlightRadius: 12,
        zIndex: 100,
        // The scene should stay alive behind the tour: light dim, spotlight
        // target always interactive, no library arrow on the frosted surface.
        // Dark needs a somewhat stronger dim — over a black background the
        // overlay itself is invisible, and the highlight reads through the
        // contrast between dimmed points/panels and the untouched cutout
        // (the spotlight ring above carries most of the effect; 0.55 muted
        // the scene too much).
        overlayColor: isDark ? 'rgba(0, 0, 0, 0.4)' : 'rgba(0, 0, 0, 0.25)',
        blockTargetInteraction: false,
        arrowColor: 'transparent',
      }}
    />
  );
}

export default TourController;
