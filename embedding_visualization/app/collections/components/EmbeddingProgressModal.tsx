'use client';

import { Card, CardContent } from '@/lib/ui-primitives/card';
import { JobProgressBody, type JobProgressBodyProps } from './JobProgressBody';

/**
 * Blocking modal overlay displaying real-time job progress.
 * All progress logic lives in JobProgressBody / useJobProgress; this wrapper
 * only supplies the centered full-screen overlay layout.
 */
export function ProgressModal({
  subtitle = 'This may take several minutes for large datasets.',
  ...props
}: JobProgressBodyProps) {
  return (
    <div className="fixed inset-0 bg-background/80 backdrop-blur-sm flex items-center justify-center z-50">
      <Card className="w-[650px]">
        <CardContent className="pt-6">
          <JobProgressBody subtitle={subtitle} {...props} />
        </CardContent>
      </Card>
    </div>
  );
}
