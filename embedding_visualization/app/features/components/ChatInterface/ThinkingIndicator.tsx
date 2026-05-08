'use client';

import { AssistantAvatar } from './AssistantAvatar';
import { Shimmer } from './Shimmer';

export function ThinkingIndicator() {
  return (
    <div className="flex w-full animate-[message-in_0.3s_cubic-bezier(0.16,1,0.3,1)] items-start gap-3">
      <div className="flex h-[calc(13px*1.65)] shrink-0 items-center">
        <AssistantAvatar />
      </div>
      <div className="flex h-[calc(13px*1.65)] items-center text-[13px] leading-[1.65]">
        <Shimmer className="font-medium" duration={1}>
          Thinking...
        </Shimmer>
      </div>
    </div>
  );
}
