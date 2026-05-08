import { useCallback, useEffect, useRef, useState } from 'react';
import { toast } from 'sonner';
import type { ChatMessage, ChatStatus, SteeringConfig } from '@/lib/types/types';

export interface UseSteeringChatReturn {
  messages: ChatMessage[];
  status: ChatStatus;
  error: string | null;
  send: (content: string) => void;
  stop: () => void;
  reset: () => void;
}

/** Serialise config into a stable key for change detection. */
function configKey(config: SteeringConfig): string {
  const sorted = [...config.features]
    .sort((a, b) => a.featureIndex - b.featureIndex)
    .map((f) => `${f.modelId}/${f.saeId}/${f.featureIndex}:${f.strength}`);
  return sorted.join(',');
}

/**
 * Stub transport — returns a placeholder response.
 * Replace with SSE fetch when the backend endpoint is ready.
 */
async function fetchSteeringChat(
  turns: Array<{ role: string; content: string }>,
  config: SteeringConfig,
  signal: AbortSignal,
): Promise<string> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      if (signal.aborted) {
        reject(new DOMException('Aborted', 'AbortError'));
        return;
      }
      const n = config.features.length;
      const featureList =
        n > 0
          ? config.features
              .map((f) => `#${f.featureIndex} (layer ${f.layerIndex}, strength ${f.strength})`)
              .join(', ')
          : 'none';
      resolve(
        `[Stub] Steering chat is not yet connected to the backend.\n\n` +
          `Active features: ${featureList}\n` +
          `Conversation turns: ${turns.length}\n\n` +
          `When the backend endpoint is ready, this response will be replaced ` +
          `with real model output from Gemma 3 4b-it.`,
      );
    }, 1500);

    signal.addEventListener('abort', () => {
      clearTimeout(timer);
      reject(new DOMException('Aborted', 'AbortError'));
    });
  });
}

export function useSteeringChat(config: SteeringConfig): UseSteeringChatReturn {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [status, setStatus] = useState<ChatStatus>('idle');
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const messagesRef = useRef<ChatMessage[]>(messages);
  const prevKeyRef = useRef<string>(configKey(config));

  // Keep ref in sync with state
  useEffect(() => {
    messagesRef.current = messages;
  }, [messages]);

  // Auto-reset when steering config changes
  useEffect(() => {
    const key = configKey(config);
    if (key !== prevKeyRef.current) {
      prevKeyRef.current = key;
      if (messagesRef.current.length > 0) {
        abortRef.current?.abort();
        setMessages([]);
        setStatus('idle');
        setError(null);
        toast.info('Chat cleared — steering configuration changed');
      }
    }
  }, [config]);

  const reset = useCallback(() => {
    abortRef.current?.abort();
    setMessages([]);
    setStatus('idle');
    setError(null);
  }, []);

  const stop = useCallback(() => {
    abortRef.current?.abort();
    setStatus('idle');
  }, []);

  const send = useCallback(
    (content: string) => {
      const trimmed = content.trim();
      if (!trimmed || status === 'generating') return;

      const userMsg: ChatMessage = {
        id: crypto.randomUUID(),
        role: 'user',
        content: trimmed,
        timestamp: Date.now(),
      };

      // Build turns from current messages + the new user message
      const allMessages = [...messagesRef.current, userMsg];
      const turns = allMessages.map((m) => ({
        role: m.role === 'user' ? 'user' : 'model',
        content: m.content,
      }));

      setMessages(allMessages);
      setStatus('generating');
      setError(null);

      const controller = new AbortController();
      abortRef.current = controller;

      fetchSteeringChat(turns, config, controller.signal)
        .then((response) => {
          const assistantMsg: ChatMessage = {
            id: crypto.randomUUID(),
            role: 'assistant',
            content: response,
            timestamp: Date.now(),
          };
          setMessages((prev) => [...prev, assistantMsg]);
          setStatus('idle');
        })
        .catch((err) => {
          if (err instanceof DOMException && err.name === 'AbortError') {
            setStatus('idle');
            return;
          }
          setError(err instanceof Error ? err.message : 'Unknown error');
          setStatus('error');
        });
    },
    [config, status],
  );

  return { messages, status, error, send, stop, reset };
}
