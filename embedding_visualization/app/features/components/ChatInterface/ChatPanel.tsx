'use client';

import { ChevronDown, RotateCcw, X } from 'lucide-react';
import { Button } from '@/lib/ui-primitives/button';
import { Separator } from '@/lib/ui-primitives/separator';
import { useScrollToBottom } from '@/lib/hooks/useScrollToBottom';
import { useSteeringChat } from '@/lib/hooks/useSteeringChat';
import { cn } from '@/lib/utils/utils';
import type { SaeFeature, SteeringConfig, SteeringFeature } from '@/lib/types/types';
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
  const { messages, status, error, send, stop, reset } = useSteeringChat(steeringConfig);
  const { containerRef, endRef, isAtBottom, scrollToBottom } = useScrollToBottom();

  const isEmpty = messages.length === 0;
  const isGenerating = status === 'generating';

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
              <X className="size-3.5" />
              <span className="sr-only">Close chat</span>
            </Button>
          )}
        </div>
      </div>
      {/* Steering controls *
      <Separator className="my-2" />
      */}

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

      {/* Messages area */}
      <div className="relative flex-1 overflow-hidden">
        {isEmpty ? (
          <ChatGreeting
            featureCount={steeringConfig.features.length}
            onSuggest={send}
          />
        ) : (
          <div
            ref={containerRef}
            className="absolute inset-0 overflow-y-auto"
          >
            <div className="mx-auto flex min-h-full max-w-2xl flex-col gap-5 px-4 py-6 md:gap-7">
              {messages.map((msg) => (
                <ChatMessage key={msg.id} message={msg} />
              ))}

              {isGenerating && <ThinkingIndicator />}

              {error && (
                <div className="animate-fade-up rounded-lg border border-destructive/30 bg-destructive/5 px-3 py-2 text-xs text-destructive">
                  {error}
                </div>
              )}

              <div ref={endRef} />
            </div>
          </div>
        )}

        {/* Scroll-to-bottom button */}
        <button
          onClick={() => scrollToBottom()}
          className={cn(
            'absolute bottom-2 left-1/2 z-10 flex -translate-x-1/2 items-center',
            'h-7 gap-1 rounded-full border border-border/50 bg-card/90 px-3.5',
            'text-[10px] text-muted-foreground',
            'shadow-[var(--shadow-float)] backdrop-blur-lg',
            'transition-all duration-200',
            isAtBottom
              ? 'pointer-events-none scale-90 opacity-0'
              : 'pointer-events-auto scale-100 opacity-100',
          )}
        >
          <ChevronDown className="size-3 text-muted-foreground" />
          <span>Scroll to bottom</span>
        </button>
      </div>

      {/* Input area */}
      <ChatInput
        onSend={send}
        onStop={stop}
        isGenerating={isGenerating}
      />
    </div>
  );
}
