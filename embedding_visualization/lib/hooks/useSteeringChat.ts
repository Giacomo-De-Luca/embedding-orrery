import { useCallback, useEffect, useRef, useState } from 'react';
import { toast } from 'sonner';
import { apolloClient } from '@/lib/utils/apollo-client';
import { GENERATE_STREAM } from '@/lib/graphql/queries';
import type { ChatMessage, ChatStatus, SteeringConfig } from '@/lib/types/types';

export interface UseSteeringChatReturn {
  messages: ChatMessage[];
  status: ChatStatus;
  error: string | null;
  send: (content: string) => void;
  stop: () => void;
  reset: () => void;
  regenerate: (assistantIndex: number) => void;
}

/** Serialise config into a stable key for change detection. */
function configKey(config: SteeringConfig): string {
  const sorted = [...config.features]
    .sort((a, b) => a.featureIndex - b.featureIndex)
    .map((f) => `${f.modelId}/${f.saeId}/${f.featureIndex}:${f.strength}`);
  return sorted.join(',');
}

interface TokenChunkData {
  generateStream: {
    streamId: string;
    tokenIndex: number;
    tokenId: number;
    text: string;
    done: boolean;
    error: string | null;
  };
}

const DEFAULT_OUTPUT_LEN = 256;
const DEFAULT_TOP_P = 0.95;
const DEFAULT_TOP_K = 64;

/** Map SteeringConfig features to the GraphQL [SteeringInput] list format. */
function buildSteeringInputs(config: SteeringConfig) {
  if (config.features.length === 0) return null;
  return config.features.map((f) => ({
    featureIndex: f.featureIndex,
    layer: f.layerIndex,
    strength: f.strength,
    hookType: 'RESID_POST',
    width: '16k',
  }));
}

export function useSteeringChat(config: SteeringConfig): UseSteeringChatReturn {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [status, setStatus] = useState<ChatStatus>('idle');
  const [error, setError] = useState<string | null>(null);
  const subscriptionRef = useRef<{ unsubscribe: () => void } | null>(null);
  const messagesRef = useRef<ChatMessage[]>(messages);
  const prevKeyRef = useRef<string>(configKey(config));
  const assistantIdRef = useRef<string | null>(null);

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
        subscriptionRef.current?.unsubscribe();
        subscriptionRef.current = null;
        setMessages([]);
        setStatus('idle');
        setError(null);
        toast.info('Chat cleared — steering configuration changed');
      }
    }
  }, [config]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      subscriptionRef.current?.unsubscribe();
    };
  }, []);

  const reset = useCallback(() => {
    subscriptionRef.current?.unsubscribe();
    subscriptionRef.current = null;
    setMessages([]);
    setStatus('idle');
    setError(null);
  }, []);

  const stop = useCallback(() => {
    subscriptionRef.current?.unsubscribe();
    subscriptionRef.current = null;
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

      // Build conversation turns
      const allMessages = [...messagesRef.current, userMsg];
      const turns = allMessages.map((m) => ({
        role: m.role === 'user' ? 'user' : 'model',
        content: m.content,
      }));

      // Create placeholder assistant message
      const assistantId = crypto.randomUUID();
      assistantIdRef.current = assistantId;
      const assistantMsg: ChatMessage = {
        id: assistantId,
        role: 'assistant',
        content: '',
        timestamp: Date.now(),
      };

      setMessages([...allMessages, assistantMsg]);
      setStatus('generating');
      setError(null);

      const steering = buildSteeringInputs(config);

      // Subscribe to streaming generation
      const observable = apolloClient.subscribe<TokenChunkData>({
        query: GENERATE_STREAM,
        variables: {
          input: {
            turns,
            steering,
            outputLen: DEFAULT_OUTPUT_LEN,
            topP: DEFAULT_TOP_P,
            topK: DEFAULT_TOP_K,
          },
        },
      });

      const sub = observable.subscribe({
        next({ data }) {
          if (!data) return;
          const chunk = data.generateStream;

          if (chunk.error) {
            setError(chunk.error);
            setStatus('error');
            subscriptionRef.current = null;
            return;
          }

          if (chunk.done) {
            setStatus('idle');
            subscriptionRef.current = null;
            return;
          }

          // Append token text to the assistant message
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantIdRef.current
                ? { ...m, content: m.content + chunk.text }
                : m
            )
          );
        },
        error(err) {
          // Don't report errors if we intentionally stopped
          if (subscriptionRef.current === null) return;
          setError(err instanceof Error ? err.message : 'Stream error');
          setStatus('error');
          subscriptionRef.current = null;
        },
        complete() {
          setStatus('idle');
          subscriptionRef.current = null;
        },
      });

      subscriptionRef.current = sub;
    },
    [config, status],
  );

  const regenerate = useCallback(
    (assistantIndex: number) => {
      if (status === 'generating') return;
      // Find the preceding user message
      let userContent: string | null = null;
      for (let i = assistantIndex - 1; i >= 0; i--) {
        if (messagesRef.current[i]?.role === 'user') {
          userContent = messagesRef.current[i].content;
          break;
        }
      }
      if (!userContent) return;

      // Remove the assistant message and everything after it, then re-send
      const truncated = messagesRef.current.slice(0, assistantIndex);
      setMessages(truncated);
      messagesRef.current = truncated;

      // Use queueMicrotask to ensure state is updated before send reads messagesRef
      const content = userContent;
      queueMicrotask(() => send(content));
    },
    [status, send],
  );

  return { messages, status, error, send, stop, reset, regenerate };
}
