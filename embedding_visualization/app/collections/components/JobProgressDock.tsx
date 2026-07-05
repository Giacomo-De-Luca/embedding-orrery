'use client';

import { Card, CardContent } from '@/lib/ui-primitives/card';
import { JobProgressBody, type JobProgressBodyProps } from './JobProgressBody';

/**
 * Non-blocking floating progress card (bottom-right). Shows the same live
 * progress as ProgressModal but leaves the rest of the page usable, so long
 * embeds don't lock the user out.
 */
export function JobProgressDock({
  subtitle = 'You can keep using the page while this runs.',
  ...props
}: JobProgressBodyProps) {
  return (
    <div className="fixed bottom-4 right-4 z-50 w-[420px] max-w-[calc(100vw-2rem)]">
      <Card className="shadow-lg">
        <CardContent className="pt-6">
          <JobProgressBody subtitle={subtitle} {...props} />
        </CardContent>
      </Card>
    </div>
  );
}
