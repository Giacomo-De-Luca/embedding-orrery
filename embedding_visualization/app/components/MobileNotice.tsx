'use client';

import { Monitor } from 'lucide-react';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/lib/ui-primitives/dialog';
import { Button } from '@/lib/ui-primitives/button';
import { markMobileNoticeSeen } from '@/lib/utils/demoOnboarding';

interface MobileNoticeProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

/**
 * One-time "best viewed on desktop" card for phone-sized viewports in demo
 * builds. Deliberately dismissible rather than blocking — the panels do work on
 * mobile, they are just cramped — and deliberately dependency-free (no Apollo,
 * no store) so both `/` and `/sae` can mount it.
 *
 * Gating lives in `shouldShowMobileNotice`; the pages latch it once at mount.
 */
export function MobileNotice({ open, onOpenChange }: MobileNoticeProps) {
  // Records the dismissal for overlay/Esc closes too, not just the button.
  const handleOpenChange = (next: boolean) => {
    if (!next) markMobileNoticeSeen();
    onOpenChange(next);
  };

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Monitor className="h-4 w-4 text-muted-foreground" />
            Best viewed on desktop
          </DialogTitle>
          <DialogDescription>
            Orrery&apos;s map, side panels, and legend are built for a larger screen.
          </DialogDescription>
        </DialogHeader>
        <p className="text-sm text-muted-foreground">
          On a phone you can still pan and zoom the map, run a semantic search, and open the
          controls, search, and analytics panels — but they cover most of the view, and the
          guided tour is unavailable. For the full experience, open this page on a laptop.
        </p>
        <Button onClick={() => handleOpenChange(false)}>Got it — continue</Button>
      </DialogContent>
    </Dialog>
  );
}
