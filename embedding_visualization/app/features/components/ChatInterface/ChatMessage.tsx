'use client';

import { useCallback, useRef, useState, type KeyboardEvent } from 'react';
import { motion } from 'motion/react';
import { Check, X } from 'lucide-react';
import { cn } from '@/lib/utils/utils';
import { Button } from '@/lib/ui-primitives/button';
import type { ChatMessage as ChatMessageType, MessageVote } from '@/lib/types/types';
import { AssistantAvatar } from './AssistantAvatar';
import { MessageResponse } from './ai-elements/message';
import { MessageReasoning } from './MessageReasoning';
import { ChatMessageActions } from './ChatMessageActions';

interface ChatMessageProps {
  message: ChatMessageType;
  isGenerating?: boolean;
  vote?: MessageVote;
  onVote?: (messageId: string, isUpvoted: boolean) => void;
  onEdit?: (messageIndex: number, newContent: string) => void;
  messageIndex?: number;
  onRegenerate?: () => void;
}

export function ChatMessage({
  message,
  isGenerating = false,
  vote,
  onVote,
  onEdit,
  messageIndex,
  onRegenerate,
}: ChatMessageProps) {
  const isUser = message.role === 'user';
  const [isEditing, setIsEditing] = useState(false);
  const [editText, setEditText] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const startEditing = useCallback(() => {
    setEditText(message.content);
    setIsEditing(true);
    requestAnimationFrame(() => {
      const ta = textareaRef.current;
      if (ta) {
        ta.focus();
        ta.selectionStart = ta.value.length;
      }
    });
  }, [message.content]);

  const cancelEditing = useCallback(() => {
    setIsEditing(false);
    setEditText('');
  }, []);

  const submitEdit = useCallback(() => {
    const trimmed = editText.trim();
    if (!trimmed || trimmed === message.content) {
      cancelEditing();
      return;
    }
    if (onEdit && messageIndex !== undefined) {
      onEdit(messageIndex, trimmed);
    }
    setIsEditing(false);
    setEditText('');
  }, [editText, message.content, onEdit, messageIndex, cancelEditing]);

  const handleEditKeyDown = useCallback(
    (e: KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        submitEdit();
      }
      if (e.key === 'Escape') {
        cancelEditing();
      }
    },
    [submitEdit, cancelEditing],
  );

  if (isUser) {
    return (
      <motion.div
        className="group flex w-full flex-col items-end gap-1"
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.25, ease: [0.22, 1, 0.36, 1] }}
      >
        {isEditing ? (
          <div className="w-full max-w-[min(80%,56ch)]">
            <textarea
              ref={textareaRef}
              value={editText}
              onChange={(e) => setEditText(e.target.value)}
              onKeyDown={handleEditKeyDown}
              rows={1}
              className={cn(
                'w-full resize-none rounded-2xl rounded-br-lg',
                'border border-ring/50 bg-gradient-to-br from-secondary to-muted',
                'px-3.5 py-2 text-[13px] leading-[1.65] outline-none',
                'ring-[3px] ring-ring/20',
              )}
              style={{ fieldSizing: 'content' } as React.CSSProperties}
            />
            <div className="mt-1.5 flex justify-end gap-1">
              <Button
                size="icon-xs"
                variant="ghost"
                onClick={cancelEditing}
                className="size-6 text-muted-foreground"
              >
                <X className="size-3" />
                <span className="sr-only">Cancel edit</span>
              </Button>
              <Button
                size="icon-xs"
                variant="ghost"
                onClick={submitEdit}
                disabled={!editText.trim() || editText.trim() === message.content}
                className="size-6 text-foreground"
              >
                <Check className="size-3" />
                <span className="sr-only">Submit edit</span>
              </Button>
            </div>
          </div>
        ) : (
          <>
            <div
              className={cn(
                'w-fit max-w-[min(80%,56ch)] overflow-hidden break-words',
                'rounded-2xl rounded-br-lg',
                'border border-border/30 bg-gradient-to-br from-secondary to-muted',
                'px-3.5 py-2 shadow-[var(--shadow-chat-card)]',
                'text-[13px] leading-[1.65]',
              )}
            >
              <MessageResponse>{message.content}</MessageResponse>
            </div>
            <ChatMessageActions
              message={message}
              isGenerating={isGenerating}
              onEdit={onEdit ? () => startEditing() : undefined}
            />
          </>
        )}
      </motion.div>
    );
  }

  // Assistant message
  const parts = message.parts;

  return (
    <motion.div
      className="group flex w-full items-start gap-3"
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
    >
      {/* Avatar aligned to first line */}
      <div className="flex h-[calc(13px*1.65)] shrink-0 items-center">
        <AssistantAvatar />
      </div>

      {/* Content */}
      <div className="min-w-0 flex-1">
        <div className="flex flex-col gap-2 text-[13px] leading-[1.65]">
          {parts && parts.length > 0 ? (
            parts.map((part, i) => {
              if (part.type === 'reasoning') {
                return (
                  <MessageReasoning
                    key={`reasoning-${i}`}
                    reasoning={part.text}
                    isStreaming={part.state === 'streaming'}
                  />
                );
              }
              if (part.type === 'text') {
                return <MessageResponse key={`text-${i}`}>{part.text}</MessageResponse>;
              }
              if (part.type === 'error') {
                return (
                  <div
                    key={`error-${i}`}
                    className="rounded-lg border border-destructive/30 bg-destructive/5 px-3 py-2 text-xs text-destructive"
                  >
                    {part.error}
                  </div>
                );
              }
              return null;
            })
          ) : (
            <MessageResponse>{message.content}</MessageResponse>
          )}
        </div>

        {/* Actions toolbar */}
        <ChatMessageActions
          message={message}
          isGenerating={isGenerating}
          vote={vote}
          onVote={onVote}
          onRegenerate={onRegenerate}
          className="mt-2"
        />
      </div>
    </motion.div>
  );
}
