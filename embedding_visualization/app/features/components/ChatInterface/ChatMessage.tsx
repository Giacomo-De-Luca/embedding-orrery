'use client';

import { cn } from '@/lib/utils/utils';
import type { ChatMessage as ChatMessageType } from '@/lib/types/types';
import { AssistantAvatar } from './AssistantAvatar';

interface ChatMessageProps {
  message: ChatMessageType;
}

/**
 * Split text on triple-backtick fences into alternating text / code segments.
 * Even indices are text, odd indices are code.
 */
function parseContent(content: string) {
  const parts = content.split(/```(?:\w*\n?)?/);
  return parts.map((text, i) => ({ text, isCode: i % 2 === 1 }));
}

export function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === 'user';
  const segments = parseContent(message.content);

  if (isUser) {
    return (
      <div className="flex w-full animate-[fade-up_0.25s_cubic-bezier(0.22,1,0.36,1)] flex-col items-end gap-2">
        <div
          className={cn(
            'w-fit max-w-[min(80%,56ch)] overflow-hidden break-words',
            'rounded-2xl rounded-br-lg',
            'border border-border/30 bg-gradient-to-br from-secondary to-muted',
            'px-3.5 py-2 shadow-[var(--shadow-chat-card)]',
            'text-[13px] leading-[1.65]',
          )}
        >
          {segments.map((seg, i) =>
            seg.isCode ? (
              <pre key={i} className="my-2 overflow-x-auto rounded-lg bg-background/50 p-3 text-xs font-mono">
                <code>{seg.text.trim()}</code>
              </pre>
            ) : (
              <span key={i} className="whitespace-pre-wrap">{seg.text}</span>
            ),
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="flex w-full animate-[message-in_0.3s_cubic-bezier(0.16,1,0.3,1)] items-start gap-3">
      {/* Avatar aligned to first line of text */}
      <div className="flex h-[calc(13px*1.65)] shrink-0 items-center">
        <AssistantAvatar />
      </div>

      {/* Content */}
      <div className="min-w-0 flex-1 text-[13px] leading-[1.65]">
        {segments.map((seg, i) =>
          seg.isCode ? (
            <pre key={i} className="my-2 overflow-x-auto rounded-lg bg-muted/50 p-3 text-xs font-mono">
              <code>{seg.text.trim()}</code>
            </pre>
          ) : (
            <span key={i} className="whitespace-pre-wrap">{seg.text}</span>
          ),
        )}
      </div>
    </div>
  );
}
