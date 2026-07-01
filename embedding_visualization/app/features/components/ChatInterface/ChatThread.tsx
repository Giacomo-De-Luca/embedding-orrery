'use client';

import { motion } from 'motion/react';
import { ChevronDown } from 'lucide-react';
import { cn } from '@/lib/utils/utils';
import { useScrollToBottom } from '@/lib/hooks/useScrollToBottom';
import type {
  ChatMessage as ChatMessageType,
  ChatStatus,
  MessageVote,
  SteeringFeature,
} from '@/lib/types/types';
import { ChatGreeting } from './ChatGreeting';
import { ChatMessage } from './ChatMessage';
import { ThinkingIndicator } from './ThinkingIndicator';

interface ChatThreadProps {
  messages: ChatMessageType[];
  status: ChatStatus;
  error: string | null;
  /** Drives the greeting feature count and the thinking-indicator identicon. */
  steeringFeatures: SteeringFeature[];
  votes: Map<string, MessageVote>;
  onVote: (messageId: string, isUpvoted: boolean) => void;
  /** Omitted (read-only) in compare mode. */
  onEdit?: (messageIndex: number, newContent: string) => void;
  /** Omitted (read-only) in compare mode. Receives the assistant message index. */
  onRegenerate?: (assistantIndex: number) => void;
}

/**
 * Presentational message column: greeting, message list, thinking indicator,
 * error banner, and its own scroll-to-bottom affordance. Owns a private
 * `useScrollToBottom` so multiple threads (compare mode) scroll independently.
 */
export function ChatThread({
  messages,
  status,
  error,
  steeringFeatures,
  votes,
  onVote,
  onEdit,
  onRegenerate,
}: ChatThreadProps) {
  const { containerRef, endRef, isAtBottom, scrollToBottom } = useScrollToBottom();

  const isEmpty = messages.length === 0;
  const isLoadingModel = status === 'loading_model';
  const isGenerating = status === 'generating';
  const isBusy = isLoadingModel || isGenerating;

  return (
    <div className="relative flex-1 overflow-hidden">
      {isEmpty && <ChatGreeting featureCount={steeringFeatures.length} />}

      <div ref={containerRef} className="absolute inset-0 overflow-y-auto">
        <div className="mx-auto flex min-h-full max-w-2xl flex-col gap-5 px-4 py-6 md:gap-7">
          {messages.map((msg, i) => {
            // Skip the trailing empty assistant placeholder while busy —
            // ThinkingIndicator already covers that slot, so rendering both
            // produces two avatars. Restricted to the last message so earlier
            // empty assistant turns (aborted, regen'd) still render.
            if (
              isBusy &&
              i === messages.length - 1 &&
              msg.role === 'assistant' &&
              msg.content === '' &&
              (!msg.parts || msg.parts.length === 0)
            ) {
              return null;
            }
            return (
              <ChatMessage
                key={msg.id}
                message={msg}
                messageIndex={i}
                isLast={i === messages.length - 1}
                isGenerating={isBusy}
                vote={votes.get(msg.id)}
                onVote={onVote}
                onEdit={msg.role === 'user' ? onEdit : undefined}
                onRegenerate={
                  msg.role === 'assistant' && onRegenerate ? () => onRegenerate(i) : undefined
                }
              />
            );
          })}

          {isBusy && messages[messages.length - 1]?.content === '' && (
            <ThinkingIndicator
              phase={isLoadingModel ? 'loading_model' : 'thinking'}
              features={steeringFeatures}
            />
          )}

          {error && (
            <motion.div
              className="rounded-lg border border-destructive/30 bg-destructive/5 px-3 py-2 text-xs text-destructive"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3, ease: [0.22, 1, 0.36, 1] }}
            >
              {error}
            </motion.div>
          )}

          <div ref={endRef} />
        </div>
      </div>

      {/* Scroll-to-bottom button */}
      <button
        onClick={() => scrollToBottom()}
        className={cn(
          'absolute bottom-2 left-1/2 z-10 flex -translate-x-1/2 items-center justify-center',
          'size-7 rounded-full border border-border/50 bg-card/90',
          'shadow-[var(--shadow-float)] backdrop-blur-lg',
          'transition-all duration-200',
          isAtBottom
            ? 'pointer-events-none scale-90 opacity-0'
            : 'pointer-events-auto scale-100 opacity-100',
        )}
        aria-label="Scroll to bottom"
      >
        <ChevronDown className="size-3 text-muted-foreground" />
      </button>
    </div>
  );
}
