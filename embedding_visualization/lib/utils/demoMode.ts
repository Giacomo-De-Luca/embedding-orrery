/**
 * Read-only demo deployments (e.g. the public HuggingFace Space) are built
 * with NEXT_PUBLIC_DEMO_MODE=1, which hides write-capable UI (Collections/SAE
 * nav, probe training, save-default buttons). Enforcement is server-side
 * (ORRERY_READ_ONLY blocks all GraphQL mutations) — this flag is cosmetic.
 *
 * NEXT_PUBLIC_ vars are inlined at build time, so IS_DEMO is a constant.
 */
export const IS_DEMO = process.env.NEXT_PUBLIC_DEMO_MODE === '1';
