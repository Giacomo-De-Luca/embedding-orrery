'use client';

import { useCallback, useState } from 'react';
import { Copy, Check, ThumbsUp, ThumbsDown, Pencil, RotateCcw } from 'lucide-react';
import { toast } from 'sonner';
import { cn } from '@/lib/utils/utils';
import type { ChatMessage, MessageVote } from '@/lib/types/types';
import { MessageActions, MessageAction } from './ai-elements/message';

interface ChatMessageActionsProps {
  message: ChatMessage;
  isGenerating: boolean;
  vote?: MessageVote;
  onVote?: (messageId: string, isUpvoted: boolean) => void;
  onEdit?: (message: ChatMessage) => void;
  onRegenerate?: () => void;
  className?: string;
}

export function ChatMessageActions({
  message,
  isGenerating,
  vote,
  onVote,
  onEdit,
  onRegenerate,
  className,
}: ChatMessageActionsProps) {
  const [hasCopied, setHasCopied] = useState(false);

  const handleCopy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(message.content);
      setHasCopied(true);
      toast.success('Copied to clipboard');
      setTimeout(() => setHasCopied(false), 2000);
    } catch {
      toast.error('Failed to copy');
    }
  }, [message.content]);

  const isUser = message.role === 'user';

  return (
    <MessageActions
      className={cn(
        'opacity-0 transition-opacity group-hover:opacity-100',
        className,
      )}
    >
      {/* Copy */}
      <MessageAction
        tooltip="Copy"
        onClick={handleCopy}
        disabled={isGenerating}
      >
        {hasCopied ? <Check size={14} /> : <Copy size={14} />}
      </MessageAction>

      {/* Vote (assistant only) */}
      {!isUser && onVote && (
        <>
          <MessageAction
            tooltip="Good response"
            onClick={() => onVote(message.id, true)}
            disabled={isGenerating}
            className={cn(vote?.isUpvoted === true && 'text-green-500')}
          >
            <ThumbsUp size={14} />
          </MessageAction>
          <MessageAction
            tooltip="Bad response"
            onClick={() => onVote(message.id, false)}
            disabled={isGenerating}
            className={cn(vote?.isUpvoted === false && 'text-red-500')}
          >
            <ThumbsDown size={14} />
          </MessageAction>
        </>
      )}

      {/* Edit (user only) */}
      {isUser && onEdit && (
        <MessageAction
          tooltip="Edit"
          onClick={() => onEdit(message)}
          disabled={isGenerating}
        >
          <Pencil size={14} />
        </MessageAction>
      )}

      {/* Regenerate (assistant only) */}
      {!isUser && onRegenerate && (
        <MessageAction
          tooltip="Regenerate"
          onClick={onRegenerate}
          disabled={isGenerating}
        >
          <RotateCcw size={14} />
        </MessageAction>
      )}
    </MessageActions>
  );
}
