import { useCallback, useEffect, useRef, useState } from 'react';
import { toast } from 'sonner';
import { apolloClient } from '@/lib/utils/apollo-client';
import { GENERATE_STREAM } from '@/lib/graphql/queries';
import { ensureModelLoaded } from '@/lib/utils/modelLoader';
import type { ChatMessage, ChatStatus, SteeringConfig } from '@/lib/types/types';

export interface UseSteeringChatReturn {
  messages: ChatMessage[];
  status: ChatStatus;
  error: string | null;
  send: (content: string) => void;
  stop: () => void;
  reset: () => void;
  regenerate: (assistantIndex: number) => void;
  editAndResend: (messageIndex: number, newContent: string) => void;
  loadMessages: (msgs: ChatMessage[]) => void;
}

export interface SteeringChatOptions {
  onUserMessageSent?: (message: ChatMessage) => void;
  onAssistantMessageComplete?: (message: ChatMessage) => void;
}

/** Serialise config into a stable key for change detection. */
function configKey(config: SteeringConfig): string {
  const features = config?.features ?? [];
  const sorted = [...features]
    .sort((a, b) => a.featureIndex - b.featureIndex)
    .map((f) => `${f.modelId}/${f.saeId}/${f.featureIndex}/${f.hookType}/${f.width}:${f.strength}`);
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
    hookType: f.hookType ?? 'RESID_POST',
    width: f.width ?? '16k',
  }));
}


export function useSteeringChat(
  config: SteeringConfig,
  maxTokens: number = DEFAULT_OUTPUT_LEN,
  options?: SteeringChatOptions,
): UseSteeringChatReturn {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [status, setStatus] = useState<ChatStatus>('idle');
  const [error, setError] = useState<string | null>(null);
  const subscriptionRef = useRef<{ unsubscribe: () => void } | null>(null);
  const messagesRef = useRef<ChatMessage[]>(messages);
  const prevKeyRef = useRef<string>(configKey(config));
  const assistantIdRef = useRef<string | null>(null);
  const cancelledRef = useRef(false);
  const optionsRef = useRef(options);
  optionsRef.current = options;

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
    cancelledRef.current = true;
    subscriptionRef.current?.unsubscribe();
    subscriptionRef.current = null;
    setStatus('idle');
  }, []);

  const send = useCallback(
    (content: string) => {
      const trimmed = content.trim();
      if (!trimmed || status !== 'idle') return;

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
      setStatus('loading_model');
      setError(null);
      cancelledRef.current = false;

      // Notify parent of the user message for persistence
      optionsRef.current?.onUserMessageSent?.(userMsg);

      const steering = buildSteeringInputs(config);

      /** Remove the empty assistant placeholder on failure. */
      const removePlaceholder = () => {
        setMessages((prev) => prev.filter((m) => m.id !== assistantId));
      };

      // Ensure model is loaded, then start streaming
      const startStreaming = async () => {
        try {
          const loadError = await ensureModelLoaded();
          if (cancelledRef.current) return;
          if (loadError) {
            removePlaceholder();
            setError(loadError);
            setStatus('error');
            return;
          }
        } catch (err) {
          if (cancelledRef.current) return;
          removePlaceholder();
          setError(err instanceof Error ? err.message : 'Failed to load model');
          setStatus('error');
          return;
        }

        if (cancelledRef.current) return;
        setStatus('generating');

        // Subscribe to streaming generation
        const observable = apolloClient.subscribe<TokenChunkData>({
          query: GENERATE_STREAM,
          variables: {
            input: {
              turns,
              steering,
              outputLen: maxTokens,
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
              // Notify parent of completed assistant message
              const completedMsg = messagesRef.current.find(
                (m) => m.id === assistantIdRef.current
              );
              if (completedMsg) {
                optionsRef.current?.onAssistantMessageComplete?.(completedMsg);
              }
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
      };

      startStreaming();
    },
    [config, status],
  );

  const regenerate = useCallback(
    (assistantIndex: number) => {
      if (status !== 'idle') return;
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

  const editAndResend = useCallback(
    (messageIndex: number, newContent: string) => {
      if (status !== 'idle') return;
      const trimmed = newContent.trim();
      if (!trimmed) return;

      // Truncate everything from the edited message onward, then re-send with new content
      const truncated = messagesRef.current.slice(0, messageIndex);
      setMessages(truncated);
      messagesRef.current = truncated;

      queueMicrotask(() => send(trimmed));
    },
    [status, send],
  );

  const loadMessages = useCallback((msgs: ChatMessage[], newConfig?: SteeringConfig) => {
    subscriptionRef.current?.unsubscribe();
    subscriptionRef.current = null;
    // Update the config key to prevent the auto-reset effect from firing
    if (newConfig) prevKeyRef.current = configKey(newConfig);
    setMessages(msgs);
    messagesRef.current = msgs;
    setStatus('idle');
    setError(null);
  }, []);

  return { messages, status, error, send, stop, reset, regenerate, editAndResend, loadMessages };
}
