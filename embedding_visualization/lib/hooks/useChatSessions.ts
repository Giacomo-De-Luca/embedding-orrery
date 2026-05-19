'use client';

import { useMutation, useQuery } from '@apollo/client/react';
import { useCallback, useRef, useState } from 'react';

import {
  GET_CHAT_SESSION,
  GET_CHAT_SESSIONS,
  type ChatSessionQueryResult,
} from '../graphql/queries';
import {
  CREATE_CHAT_SESSION,
  DELETE_CHAT_SESSION,
  SAVE_CHAT_MESSAGE,
} from '../graphql/mutations';
import type {
  ChatMessage,
  ChatSessionSummary,
  SteeringConfig,
} from '../types/types';

function generateId(): string {
  return crypto.randomUUID();
}

function deriveTitle(firstMessage: string): string {
  const trimmed = firstMessage.trim();
  if (trimmed.length <= 50) return trimmed;
  return trimmed.slice(0, 47) + '...';
}

export interface UseChatSessionsReturn {
  sessions: ChatSessionSummary[];
  loading: boolean;
  activeSessionId: string | null;
  createSession: (
    config: SteeringConfig,
    firstMessage: string
  ) => Promise<string>;
  loadSession: (
    id: string
  ) => Promise<{ messages: ChatMessage[]; config: SteeringConfig }>;
  saveMessage: (
    sessionId: string,
    message: ChatMessage,
    steeringSnapshot?: SteeringConfig | null
  ) => void;
  deleteSession: (id: string) => void;
  setActiveSessionId: (id: string | null) => void;
}

export function useChatSessions(): UseChatSessionsReturn {
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const activeSessionIdRef = useRef<string | null>(null);
  activeSessionIdRef.current = activeSessionId;

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const { data, loading, refetch } = useQuery<{ chatSessions: any[] }>(GET_CHAT_SESSIONS, {
    variables: { limit: 50 },
    fetchPolicy: 'cache-and-network',
  });

  const [createSessionMutation] = useMutation(CREATE_CHAT_SESSION);
  const [saveMessageMutation] = useMutation(SAVE_CHAT_MESSAGE);
  const [deleteSessionMutation] = useMutation(DELETE_CHAT_SESSION);

  const sessions: ChatSessionSummary[] = (data?.chatSessions ?? []).map(
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (s: any) => ({
      id: s.id,
      title: s.title,
      config: s.config,
      createdAt: s.createdAt,
      updatedAt: s.updatedAt,
    })
  );

  const createSession = useCallback(
    async (config: SteeringConfig, firstMessage: string): Promise<string> => {
      const id = generateId();
      const title = deriveTitle(firstMessage);
      await createSessionMutation({
        variables: {
          input: { id, title, config },
        },
      });
      setActiveSessionId(id);
      refetch();
      return id;
    },
    [createSessionMutation, refetch]
  );

  const loadSession = useCallback(
    async (
      id: string
    ): Promise<{ messages: ChatMessage[]; config: SteeringConfig }> => {
      const { apolloClient } = await import('../utils/apollo-client');
      const { data: detail } = await apolloClient.query<ChatSessionQueryResult>({
        query: GET_CHAT_SESSION,
        variables: { id },
        fetchPolicy: 'network-only',
      });

      if (!detail?.chatSession) {
        throw new Error(`Session ${id} not found`);
      }

      const session = detail.chatSession;
      const messages: ChatMessage[] = session.messages.map((m) => ({
        id: m.id,
        role: m.role,
        content: m.content,
        timestamp: new Date(m.createdAt).getTime(),
        parts: m.parts ?? undefined,
        steeringSnapshot: m.steeringSnapshot ?? undefined,
      }));

      setActiveSessionId(id);
      return { messages, config: session.config };
    },
    []
  );

  const saveMessage = useCallback(
    (
      sessionId: string,
      message: ChatMessage,
      steeringSnapshot?: SteeringConfig | null
    ) => {
      saveMessageMutation({
        variables: {
          input: {
            id: message.id,
            sessionId,
            role: message.role,
            content: message.content,
            parts: message.parts ?? null,
            steeringSnapshot: steeringSnapshot ?? null,
          },
        },
      }).catch((err: unknown) => {
        console.error('Failed to save chat message:', err);
      });
    },
    [saveMessageMutation]
  );

  const deleteSession = useCallback(
    (id: string) => {
      deleteSessionMutation({
        variables: { id },
      }).then(() => {
        if (activeSessionIdRef.current === id) {
          setActiveSessionId(null);
        }
        refetch();
      });
    },
    [deleteSessionMutation, refetch]
  );

  return {
    sessions,
    loading,
    activeSessionId,
    createSession,
    loadSession,
    saveMessage,
    deleteSession,
    setActiveSessionId,
  };
}
