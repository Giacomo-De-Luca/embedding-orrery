'use client';

import { Sparkles } from 'lucide-react';

export function AssistantAvatar() {
  return (
    <div className="flex size-7 shrink-0 items-center justify-center rounded-lg bg-muted/60 ring-1 ring-border/50">
      <Sparkles className="size-3.5 text-muted-foreground" />
    </div>
  );
}
