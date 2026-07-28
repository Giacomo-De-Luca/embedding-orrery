import type {
  ChatMessage,
  ChatMessagePart,
  ChatSessionSummary,
  SteeringConfig,
} from '../types/types';

/**
 * Pre-generated steered-chat transcripts for the read-only demo.
 *
 * Live inference is disabled in the demo build, so the chat panel plays back
 * sessions recorded against the real engine and committed to the repo as a
 * static fixture. Each fixture entry is EXACTLY what the chat's
 * "Download chat as JSON" button produces (ChatPanel.handleDownload), so
 * recording a new session is: run locally with inference, steer, download,
 * drop the object into the array in `public/demo/chat-sessions.json`.
 */

export const DEMO_CHAT_FIXTURE_URL = '/demo/chat-sessions.json';

/** One "Download chat as JSON" export (the shape written by ChatPanel). */
interface DemoChatFixtureEntry {
  schemaVersion: number;
  session: {
    id: string;
    title: string;
    config: SteeringConfig;
    createdAt: string;
    updatedAt: string;
  };
  messages: Array<{
    id: string;
    role: 'user' | 'assistant';
    content: string;
    createdAt: string;
    parts?: ChatMessagePart[] | null;
    steeringSnapshot?: SteeringConfig | null;
  }>;
}

export interface DemoChatFixture {
  summaries: ChatSessionSummary[];
  messagesById: ReadonlyMap<string, ChatMessage[]>;
  configById: ReadonlyMap<string, SteeringConfig>;
}

export const EMPTY_DEMO_CHAT_FIXTURE: DemoChatFixture = {
  summaries: [],
  messagesById: new Map(),
  configById: new Map(),
};

function isFixtureEntry(value: unknown): value is DemoChatFixtureEntry {
  if (typeof value !== 'object' || value === null) return false;
  const entry = value as Partial<DemoChatFixtureEntry>;
  return (
    typeof entry.session === 'object' &&
    entry.session !== null &&
    typeof entry.session.id === 'string' &&
    typeof entry.session.title === 'string' &&
    Array.isArray(entry.messages)
  );
}

/**
 * Parse the raw fixture JSON into session summaries + per-session messages.
 * Tolerant by design: malformed entries and messages are skipped (a bad
 * hand-edited entry must not blank the whole chat), and an unparsable
 * timestamp falls back to 0 rather than propagating NaN (timestamps are
 * display-only here; sessions and messages keep their file order).
 */
export function parseChatFixture(raw: unknown): DemoChatFixture {
  if (!Array.isArray(raw)) return EMPTY_DEMO_CHAT_FIXTURE;

  const summaries: ChatSessionSummary[] = [];
  const messagesById = new Map<string, ChatMessage[]>();
  const configById = new Map<string, SteeringConfig>();

  for (const value of raw) {
    if (!isFixtureEntry(value)) continue;
    const { session, messages } = value;
    if (messagesById.has(session.id)) continue;

    summaries.push({
      id: session.id,
      title: session.title,
      config: session.config ?? { features: [] },
      createdAt: session.createdAt ?? '',
      updatedAt: session.updatedAt ?? '',
    });
    configById.set(session.id, session.config ?? { features: [] });
    messagesById.set(
      session.id,
      messages
        .filter(
          (m) =>
            typeof m?.id === 'string' &&
            typeof m?.content === 'string' &&
            (m.role === 'user' || m.role === 'assistant'),
        )
        .map((m) => ({
          id: m.id,
          role: m.role,
          content: m.content,
          timestamp: Date.parse(m.createdAt) || 0,
          parts: m.parts ?? undefined,
          steeringSnapshot: m.steeringSnapshot ?? undefined,
        })),
    );
  }

  return { summaries, messagesById, configById };
}

/**
 * Fetch + parse the committed fixture. Resolves to the empty fixture on any
 * failure (file not committed yet, network error, invalid JSON) — the demo
 * chat then simply opens with no saved sessions.
 */
export async function fetchDemoChatFixture(): Promise<DemoChatFixture> {
  try {
    const response = await fetch(DEMO_CHAT_FIXTURE_URL);
    if (!response.ok) return EMPTY_DEMO_CHAT_FIXTURE;
    return parseChatFixture(await response.json());
  } catch {
    return EMPTY_DEMO_CHAT_FIXTURE;
  }
}
