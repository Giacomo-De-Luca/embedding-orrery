'use client';

import { useCallback, useState } from 'react';
import { motion } from 'motion/react';
import { ChevronDown, PanelRightIcon, RotateCcw } from 'lucide-react';
import { Button } from '@/lib/ui-primitives/button';
import { Slider } from '@/lib/ui-primitives/slider';
import { useScrollToBottom } from '@/lib/hooks/useScrollToBottom';
import { useSteeringChat } from '@/lib/hooks/useSteeringChat';
import { cn } from '@/lib/utils/utils';
import type { SaeFeature, SteeringConfig, SteeringFeature, MessageVote } from '@/lib/types/types';
import { ChatGreeting } from './ChatGreeting';
import { ChatInput } from './ChatInput';
import { ChatMessage } from './ChatMessage';
import { SteeringControls } from './SteeringControls';
import { ThinkingIndicator } from './ThinkingIndicator';

interface ChatPanelProps {
  steeringConfig: SteeringConfig;
  modelId: string | null;
  saeId: string | null;
  currentFeature: SaeFeature | null;
  onAddFeature: (feature: SteeringFeature) => void;
  onRemoveFeature: (key: string) => void;
  onUpdateStrength: (key: string, strength: number) => void;
  onClose?: () => void;
}

export function ChatPanel({
  steeringConfig,
  modelId,
  saeId,
  currentFeature,
  onAddFeature,
  onRemoveFeature,
  onUpdateStrength,
  onClose,
}: ChatPanelProps) {
  const [maxTokens, setMaxTokens] = useState(256);
  const { messages, status, error, send, stop, reset, regenerate, editAndResend } = useSteeringChat(steeringConfig, maxTokens);
  const { containerRef, endRef, isAtBottom, scrollToBottom } = useScrollToBottom();
  const [votes, setVotes] = useState<Map<string, MessageVote>>(new Map());

  const isEmpty = messages.length === 0;
  const isLoadingModel = status === 'loading_model';
  const isGenerating = status === 'generating';
  const isBusy = isLoadingModel || isGenerating;

  const handleVote = useCallback((messageId: string, isUpvoted: boolean) => {
    setVotes((prev) => {
      const next = new Map(prev);
      const existing = next.get(messageId);
      // Toggle off if same vote
      if (existing?.isUpvoted === isUpvoted) {
        next.delete(messageId);
      } else {
        next.set(messageId, { messageId, isUpvoted });
      }
      return next;
    });
  }, []);

  const handleRegenerate = useCallback((assistantMessageIndex: number) => {
    // Find the preceding user message and re-send via the hook's regenerate support
    for (let i = assistantMessageIndex - 1; i >= 0; i--) {
      if (messages[i].role === 'user') {
        regenerate(assistantMessageIndex);
        return;
      }
    }
  }, [messages, regenerate]);

  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <div className="flex items-center justify-between px-4 pt-4 pb-0">
        <h2 className="text-sm font-semibold">Steered Chat</h2>
        <div className="flex items-center gap-1">
          {!isEmpty && (
            <Button
              size="icon"
              variant="ghost"
              onClick={reset}
              className="size-7 text-muted-foreground"
            >
              <RotateCcw className="size-3.5" />
              <span className="sr-only">Reset chat</span>
            </Button>
          )}
          {onClose && (
            <Button
              size="icon"
              variant="ghost"
              onClick={onClose}
              className="size-7 text-muted-foreground"
            >
              <PanelRightIcon className="size-3.5" />
              <span className="sr-only">Close chat</span>
            </Button>
          )}
        </div>
      </div>
      {/* Steering controls */}
      <SteeringControls
        config={steeringConfig}
        onAddFeature={onAddFeature}
        onRemoveFeature={onRemoveFeature}
        onUpdateStrength={onUpdateStrength}
        currentFeature={currentFeature}
        currentModelId={modelId}
        currentSaeId={saeId}
      />

      {/* Max tokens control */}
      <div className="flex items-center gap-3 border-b border-border/30 px-4 py-2">
        <span className="shrink-0 text-[11px] text-muted-foreground">Max tokens</span>
        <Slider
          value={[maxTokens]}
          min={32}
          max={2048}
          step={32}
          onValueChange={([v]) => setMaxTokens(v)}
          className="flex-1"
        />
        <span className="w-10 shrink-0 text-right font-mono text-[10px] text-muted-foreground tabular-nums">
          {maxTokens}
        </span>
      </div>

      {/* Messages area */}
      <div className="relative flex-1 overflow-hidden">
        {isEmpty && (
          <ChatGreeting featureCount={steeringConfig.features.length} />
        )}

        <div
          ref={containerRef}
          className="absolute inset-0 overflow-y-auto"
        >
          <div className="mx-auto flex min-h-full max-w-2xl flex-col gap-5 px-4 py-6 md:gap-7">
            {messages.map((msg, i) => (
              <ChatMessage
                key={msg.id}
                message={msg}
                messageIndex={i}
                isGenerating={isBusy}
                vote={votes.get(msg.id)}
                onVote={handleVote}
                onEdit={msg.role === 'user' ? editAndResend : undefined}
                onRegenerate={msg.role === 'assistant' ? () => handleRegenerate(i) : undefined}
              />
            ))}

            {isBusy && messages[messages.length - 1]?.content === '' && (
              <ThinkingIndicator phase={isLoadingModel ? 'loading_model' : 'thinking'} />
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

      {/* Input area */}
      <ChatInput
        onSend={send}
        onStop={stop}
        isGenerating={isBusy}
        showSuggestions={isEmpty}
        onSuggest={send}
      />
    </div>
  );
}
