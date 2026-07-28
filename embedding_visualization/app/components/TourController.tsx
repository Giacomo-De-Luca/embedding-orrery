'use client';

import { useEffect, useMemo, useState } from 'react';
import { createPortal } from 'react-dom';
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
  /**
   * Chained tours: this segment's steps are numbered `offset+1 … offset+n`
   * out of `progressTotal` in the step counter, so two joyride instances
   * separated by a navigation read as ONE tour (e.g. 4/7 instead of 1/4).
   */
  progressOffset?: number;
  progressTotal?: number;
}
/* eslint-enable @typescript-eslint/no-explicit-any */

/** Custom fields this controller adds to each joyride step for the tooltip. */
interface StepExtras {
  primaryLabel?: string;
  progressOffset?: number;
  progressTotal?: number;
  highlightSelectors?: string[];
}

interface HighlightRect {
  top: number;
  left: number;
  width: number;
  height: number;
}

/**
 * Spotlight-style rings over a step's `highlightAnchors` — for steps that
 * point at MORE controls than joyride's single spotlight can cut (or that
 * drop the overlay entirely via `allowInteraction`). Rendered from inside
 * `TourTooltip` so the rings live exactly as long as the step's card, but
 * portaled to <body>: joyride's floater wrapper is transformed (Floating-UI
 * positioning), which would re-root `position: fixed` onto the card itself.
 */
function AnchorHighlights({ selectors }: { selectors: string[] }) {
  const [rects, setRects] = useState<HighlightRect[]>([]);

  useEffect(() => {
    const measure = () => {
      const next: HighlightRect[] = [];
      for (const selector of selectors) {
        // Unmounted anchors are skipped, not errors — the `?` mission button
        // only exists in demo builds while the tour runs in any build.
        const rect = document.querySelector(selector)?.getBoundingClientRect();
        if (rect && rect.width > 0 && rect.height > 0) {
          next.push({ top: rect.top, left: rect.left, width: rect.width, height: rect.height });
        }
      }
      // Keep the previous array identity when nothing moved so the interval
      // doesn't re-render three times a second.
      setRects((prev) =>
        prev.length === next.length &&
        prev.every(
          (p, i) =>
            Math.abs(p.top - next[i].top) < 0.5 &&
            Math.abs(p.left - next[i].left) < 0.5 &&
            Math.abs(p.width - next[i].width) < 0.5 &&
            Math.abs(p.height - next[i].height) < 0.5,
        )
          ? prev
          : next,
      );
    };
    measure();
    // The header reflows without scroll/resize events (badges popping in,
    // panels wrapping rows) — a slow poll keeps the rings glued on.
    const intervalId = window.setInterval(measure, 300);
    window.addEventListener('resize', measure);
    return () => {
      window.clearInterval(intervalId);
      window.removeEventListener('resize', measure);
    };
  }, [selectors]);

  if (rects.length === 0) return null;
  const PAD = 4; // breathing room, mirroring joyride's spotlightPadding
  return createPortal(
    <>
      {rects.map((r, i) => (
        <div
          key={i}
          aria-hidden
          // Same visual language as the dark-mode spotlight ring (see the
          // `styles.spotlight` override below), works over light theme too.
          className="pointer-events-none fixed rounded-xl border-[1.5px] border-foreground/50"
          style={{
            top: r.top - PAD,
            left: r.left - PAD,
            width: r.width + PAD * 2,
            height: r.height + PAD * 2,
            zIndex: 100,
            filter: 'drop-shadow(0 0 10px rgba(140, 170, 255, 0.65))',
          }}
        />
      ))}
    </>,
    document.body,
  );
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
  // `?? 0` fallback: if a future joyride version stops threading custom
  // step fields through, the counter degrades to plain index/size — not NaN.
  const extras = step as unknown as StepExtras;
  const offset = extras.progressOffset ?? 0;
  const shownIndex = index + 1 + offset;
  const shownTotal = extras.progressTotal ?? size + offset;
  return (
    <div
      {...tooltipProps}
      className="frosted-tooltip tour-tooltip relative w-[340px] max-w-[90vw]"
      style={{
        backdropFilter: 'blur(12px) saturate(150%)',
        WebkitBackdropFilter: 'blur(12px) saturate(150%)',
      }}
    >
      {/* X = dismiss. Joyride's default close action ADVANCES the index
          (useControls.close() does `index: index + 1`), which made the X
          behave exactly like Next. Every step therefore sets
          `closeButtonAction: 'skip'` (see the steps map below), which the
          library's own close handler routes to skip — ending the tour. */}
      <button
        {...closeProps}
        className="absolute right-2.5 top-2.5 rounded-sm opacity-50 transition-opacity hover:opacity-100"
      >
        <X className="h-3.5 w-3.5" />
      </button>

      {extras.highlightSelectors && extras.highlightSelectors.length > 0 && (
        <AnchorHighlights selectors={extras.highlightSelectors} />
      )}

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
            {shownIndex} / {shownTotal}
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
            {extras.primaryLabel ?? (isLastStep ? 'Done' : 'Next')}
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
  progressOffset = 0,
  progressTotal,
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
        // X dismisses instead of advancing (joyride's default close action
        // does `index + 1` — indistinguishable from Next).
        closeButtonAction: 'skip',
        // Custom fields for TourTooltip (joyride threads unknown step keys
        // through to the tooltip's `step` prop untouched).
        primaryLabel: def.primaryLabel,
        highlightSelectors: def.highlightAnchors?.map((a) => anchors[a]),
        progressOffset,
        progressTotal,
        // Collection-switching steps sit behind the page's own full-screen
        // loader — joyride's waiting spinner on top of it reads as a bug.
        ...(def.suppressWaitLoader ? { loaderComponent: null } : {}),
      })),
    [stepDefs, anchors, runtime, progressOffset, progressTotal],
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
