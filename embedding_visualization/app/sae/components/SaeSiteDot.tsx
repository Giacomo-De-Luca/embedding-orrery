import { cn } from '@/lib/utils/utils';
import type { HookType } from '@/lib/utils/saeCollections';

/** One hue per hook site. A cue only: always shown next to the site's name or abbreviation. */
const SITE_DOT_CLASS: Record<HookType, string> = {
  RESID_POST: 'bg-sky-600',
  MLP_OUT: 'bg-amber-500',
  ATTN_OUT: 'bg-pink-400',
};

/** Small coloured dot marking an SAE's hook site (residual, MLP, attention). */
export function SaeSiteDot({ hookType, className }: { hookType: HookType; className?: string }) {
  return (
    <span
      aria-hidden
      className={cn('inline-block size-1.5 shrink-0 rounded-full', SITE_DOT_CLASS[hookType], className)}
    />
  );
}
