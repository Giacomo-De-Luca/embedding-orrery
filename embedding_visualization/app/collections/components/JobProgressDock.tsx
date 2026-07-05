'use client';

import { Card, CardContent } from '@/lib/ui-primitives/card';
import { JobProgressBody, type JobProgressBodyProps } from './JobProgressBody';

/**
 * Non-blocking floating progress card. Shows the same live progress as
 * ProgressModal but leaves the rest of the page usable, so long embeds don't
 * lock the user out. Positionless — render inside a JobProgressDockContainer
 * so multiple simultaneous docks stack instead of overlapping.
 */
export function JobProgressDock({
  subtitle = 'You can keep using the page while this runs.',
  ...props
}: JobProgressBodyProps) {
  return (
    <Card className="shadow-lg w-[420px] max-w-[calc(100vw-2rem)]">
      <CardContent className="pt-6">
        <JobProgressBody subtitle={subtitle} {...props} />
      </CardContent>
    </Card>
  );
}

/** Fixed bottom-right stack for one or more JobProgressDocks. */
export function JobProgressDockContainer({ children }: { children: React.ReactNode }) {
  return (
    <div className="fixed bottom-4 right-4 z-50 flex flex-col gap-2 items-end">
      {children}
    </div>
  );
}
