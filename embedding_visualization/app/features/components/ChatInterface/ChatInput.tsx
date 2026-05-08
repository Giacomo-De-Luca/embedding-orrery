'use client';

import { useCallback, useRef, useState, type KeyboardEvent } from 'react';
import { ArrowUp, Paperclip, Square } from 'lucide-react';
import { cn } from '@/lib/utils/utils';

interface ChatInputProps {
  onSend: (content: string) => void;
  onStop: () => void;
  isGenerating: boolean;
  disabled?: boolean;
}

export function ChatInput({ onSend, onStop, isGenerating, disabled }: ChatInputProps) {
  const [input, setInput] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleSend = useCallback(() => {
    const trimmed = input.trim();
    if (!trimmed || isGenerating) return;
    onSend(trimmed);
    setInput('');
    requestAnimationFrame(() => textareaRef.current?.focus());
  }, [input, isGenerating, onSend]);

  const handleKeyDown = useCallback(
    (e: KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
      if (e.key === 'Escape') {
        textareaRef.current?.blur();
      }
    },
    [handleSend],
  );

  const canSend = input.trim().length > 0 && !isGenerating && !disabled;

  return (
    <div className="px-4 pb-4 pt-2">
      <div
        className={cn(
          'relative flex flex-col overflow-hidden rounded-2xl',
          'border border-border/30 bg-card/70',
          'shadow-[var(--shadow-composer)] transition-all duration-300',
          'focus-within:shadow-[var(--shadow-composer-focus)]',
          'focus-within:border-ring focus-within:ring-[3px] focus-within:ring-ring/20',
        )}
      >
        {/* Textarea */}
        <textarea
          ref={textareaRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Send a message..."
          disabled={disabled}
          rows={1}
          className={cn(
            'w-full resize-none border-0 bg-transparent outline-none',
            'min-h-[80px] max-h-48 px-4 pt-3.5 pb-1.5',
            'text-[13px] leading-relaxed',
            'placeholder:text-muted-foreground/35',
            'disabled:opacity-50',
          )}
          style={{ fieldSizing: 'content' } as React.CSSProperties}
        />

        {/* Footer bar */}
        <div className="flex items-center justify-between px-3 pb-3">
          {/* Attach — disabled, faint, no cursor change */}
          <div
            className="flex h-7 w-7 items-center justify-center rounded-lg border border-border/40 p-1 text-muted-foreground/30"
            aria-hidden="true"
          >
            <Paperclip style={{ width: 14, height: 14 }} />
          </div>

          {/* Send / Stop */}
          {isGenerating ? (
            <button
              type="button"
              onClick={onStop}
              className="flex h-7 w-7 items-center justify-center rounded-xl bg-foreground text-background transition-all duration-200 hover:opacity-85 active:scale-95"
              aria-label="Stop generating"
            >
              <Square style={{ width: 14, height: 14 }} />
            </button>
          ) : (
            <button
              type="button"
              onClick={handleSend}
              disabled={!canSend}
              className={cn(
                'flex h-7 w-7 items-center justify-center rounded-xl transition-all duration-200',
                canSend
                  ? 'bg-foreground text-background hover:opacity-85 active:scale-95'
                  : 'bg-muted text-muted-foreground/25',
              )}
              aria-label="Send message"
            >
              <ArrowUp style={{ width: 16, height: 16 }} />
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
