'use client';

import { useCallback, useEffect, useRef, useState, type KeyboardEvent, type ClipboardEvent } from 'react';
import { motion } from 'motion/react';
import { ArrowUp, Paperclip, Square } from 'lucide-react';
import { cn } from '@/lib/utils/utils';
import { ModelStatusButton } from './ModelStatusButton';

const DRAFT_KEY = 'steering-chat-draft';

const SUGGESTIONS = [
  'What is your favourite job?',
  'How do you feel?',
  'Tell me a story',
  'What are the important things in life?',
];

interface ChatInputProps {
  onSend: (content: string) => void;
  onStop: () => void;
  isGenerating: boolean;
  disabled?: boolean;
  showSuggestions?: boolean;
  onSuggest?: (prompt: string) => void;
  onSelectModel?: (modelId: string, saeId: string) => void;
}

export function ChatInput({ onSend, onStop, isGenerating, disabled, showSuggestions, onSuggest, onSelectModel }: ChatInputProps) {
  const [input, setInput] = useState(() => {
    if (typeof window === 'undefined') return '';
    return localStorage.getItem(DRAFT_KEY) ?? '';
  });
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const isComposingRef = useRef(false);

  // Persist draft to localStorage
  useEffect(() => {
    if (input) {
      localStorage.setItem(DRAFT_KEY, input);
    } else {
      localStorage.removeItem(DRAFT_KEY);
    }
  }, [input]);

  const handleSend = useCallback(() => {
    const trimmed = input.trim();
    if (!trimmed || isGenerating || disabled) return;
    onSend(trimmed);
    setInput('');
    localStorage.removeItem(DRAFT_KEY);
    requestAnimationFrame(() => textareaRef.current?.focus());
  }, [input, isGenerating, disabled, onSend]);

  const handleKeyDown = useCallback(
    (e: KeyboardEvent<HTMLTextAreaElement>) => {
      // Don't intercept during IME composition
      if (isComposingRef.current) return;

      // Enter (no shift) or Cmd/Ctrl+Enter → send
      if (e.key === 'Enter' && (!e.shiftKey || e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        handleSend();
      }
      if (e.key === 'Escape') {
        textareaRef.current?.blur();
      }
    },
    [handleSend],
  );

  const handlePaste = useCallback((e: ClipboardEvent<HTMLTextAreaElement>) => {
    // Strip rich text formatting — insert plain text only
    const plain = e.clipboardData.getData('text/plain');
    if (plain && e.clipboardData.types.includes('text/html')) {
      e.preventDefault();
      const textarea = e.currentTarget;
      textarea.setRangeText(plain, textarea.selectionStart, textarea.selectionEnd, 'end');
      setInput(textarea.value);
    }
  }, []);

  const canSend = input.trim().length > 0 && !isGenerating && !disabled;

  return (
    <div className="px-4 pb-4 pt-2">
      {/* Suggestion pills */}
      {showSuggestions && onSuggest && (
        <div
          className="mb-2 flex flex-wrap gap-2"
        >
          {SUGGESTIONS.map((text, i) => (
            <motion.button
              key={text}
              type="button"
              onClick={() => onSuggest(text)}
              className="flex-1 rounded-full border border-border/50 bg-card/30 px-4 py-1.5 text-[12px] whitespace-nowrap text-muted-foreground transition-all duration-200 hover:-translate-y-0.5 hover:bg-card/60 hover:text-foreground hover:shadow-[var(--shadow-chat-card)]"
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.06 * i, duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
            >
              {text}
            </motion.button>
          ))}
        </div>
      )}

      <div
        className={cn(
          'relative flex flex-col overflow-hidden rounded-2xl',
          'border border-border/30 bg-card/70',
          'shadow-[var(--shadow-composer)] transition-all duration-300',
          'focus-within:shadow-[var(--shadow-composer-focus)]',
          'focus-within:border-ring focus-within:ring-[3px] focus-within:ring-ring/50',
        )}
      >
        {/* Textarea */}
        <textarea
          ref={textareaRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          onPaste={handlePaste}
          onCompositionStart={() => { isComposingRef.current = true; }}
          onCompositionEnd={() => { isComposingRef.current = false; }}
          placeholder="Ask anything..."
          disabled={disabled}
          rows={1}
          className={cn(
            'w-full resize-none border-0 bg-transparent outline-none',
            'min-h-24 max-h-48 px-4 pt-3.5 pb-1.5',
            'text-[13px] leading-relaxed',
            'placeholder:text-muted-foreground/35',
            'disabled:opacity-50',
          )}
          style={{ fieldSizing: 'content' } as React.CSSProperties}
        />

        {/* Footer bar */}
        <div className="flex items-center justify-between px-3 pb-3">
          <div className="flex items-center gap-1">
            {/* Attach — disabled placeholder */}
            <div
              className="flex h-7 w-7 items-center justify-center rounded-lg border border-border/40 p-1 text-muted-foreground/30"
              aria-hidden="true"
            >
              <Paperclip style={{ width: 14, height: 14 }} />
            </div>
            {/* Model status indicator */}
            <ModelStatusButton onSelectModel={onSelectModel} />
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
