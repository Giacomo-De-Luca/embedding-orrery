/**
 * Read-only demo deployments (e.g. the public HuggingFace Space) are built
 * with NEXT_PUBLIC_DEMO_MODE=1. Write-capable UI stays VISIBLE but disabled
 * (with `DEMO_DISABLED_MESSAGE` as tooltip) so visitors can see what the full
 * app offers — Collections/SAE nav, probe training, save-default buttons.
 * Enforcement is server-side (ORRERY_READ_ONLY blocks all GraphQL mutations)
 * — this flag is cosmetic.
 *
 * NEXT_PUBLIC_ vars are inlined at build time, so IS_DEMO is a constant.
 */
export const IS_DEMO = process.env.NEXT_PUBLIC_DEMO_MODE === '1';

/** One consistent explanation for every feature the demo disables. */
export const DEMO_DISABLED_MESSAGE =
  'Not available in the read-only demo — run the full app to use this.';
