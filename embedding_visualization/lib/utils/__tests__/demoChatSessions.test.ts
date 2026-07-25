import { describe, it, expect } from 'vitest';
import { parseChatFixture, EMPTY_DEMO_CHAT_FIXTURE } from '../demoChatSessions';

const entry = (id: string, overrides: Record<string, unknown> = {}) => ({
  schemaVersion: 1,
  session: {
    id,
    title: `Session ${id}`,
    config: { features: [{ modelId: 'gemma-3-4b-it', saeId: '9', layerIndex: 9, featureIndex: 42, strength: 8 }] },
    createdAt: '2026-07-20T10:00:00Z',
    updatedAt: '2026-07-20T10:05:00Z',
  },
  messages: [
    { id: `${id}-u1`, role: 'user', content: 'Tell me a story', createdAt: '2026-07-20T10:00:00Z' },
    {
      id: `${id}-a1`,
      role: 'assistant',
      content: 'Once upon a time…',
      createdAt: '2026-07-20T10:00:10Z',
      parts: [{ type: 'text', text: 'Once upon a time…' }],
      steeringSnapshot: { features: [] },
    },
  ],
  ...overrides,
});

describe('parseChatFixture', () => {
  it('maps download-export entries into summaries + per-session messages', () => {
    const fixture = parseChatFixture([entry('a'), entry('b')]);
    expect(fixture.summaries.map((s) => s.id)).toEqual(['a', 'b']);
    expect(fixture.summaries[0].title).toBe('Session a');
    expect(fixture.configById.get('a')?.features).toHaveLength(1);

    const messages = fixture.messagesById.get('a')!;
    expect(messages).toHaveLength(2);
    expect(messages[0]).toMatchObject({ role: 'user', content: 'Tell me a story' });
    // createdAt (ISO) → epoch timestamp, GraphQL-null parts → undefined.
    expect(messages[0].timestamp).toBe(Date.parse('2026-07-20T10:00:00Z'));
    expect(messages[0].parts).toBeUndefined();
    expect(messages[1].parts).toEqual([{ type: 'text', text: 'Once upon a time…' }]);
  });

  it('skips malformed entries without dropping the valid ones', () => {
    const fixture = parseChatFixture([
      null,
      42,
      { schemaVersion: 1 }, // no session
      { session: { id: 'x' } }, // no title / messages
      entry('good'),
    ]);
    expect(fixture.summaries.map((s) => s.id)).toEqual(['good']);
  });

  it('drops messages with a bad role or missing content, keeps the rest', () => {
    const fixture = parseChatFixture([
      entry('a', {
        messages: [
          { id: '1', role: 'user', content: 'kept', createdAt: '2026-07-20T10:00:00Z' },
          { id: '2', role: 'system', content: 'dropped', createdAt: '2026-07-20T10:00:00Z' },
          { id: '3', role: 'assistant', createdAt: '2026-07-20T10:00:00Z' },
        ],
      }),
    ]);
    expect(fixture.messagesById.get('a')!.map((m) => m.id)).toEqual(['1']);
  });

  it('deduplicates by session id (first entry wins)', () => {
    const fixture = parseChatFixture([entry('a'), entry('a', { messages: [] })]);
    expect(fixture.summaries).toHaveLength(1);
    expect(fixture.messagesById.get('a')).toHaveLength(2);
  });

  it('returns the empty fixture for non-array roots', () => {
    for (const raw of [null, undefined, {}, 'nope', 7]) {
      expect(parseChatFixture(raw)).toEqual(EMPTY_DEMO_CHAT_FIXTURE);
    }
  });

  it('tolerates an unparsable timestamp (falls back to 0 instead of NaN)', () => {
    const fixture = parseChatFixture([
      entry('a', {
        messages: [{ id: '1', role: 'user', content: 'hi', createdAt: 'not-a-date' }],
      }),
    ]);
    expect(fixture.messagesById.get('a')![0].timestamp).toBe(0);
  });
});
