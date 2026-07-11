'use client';

import { useCallback, useEffect, useMemo, useRef, useState, type ComponentProps } from 'react';
import { AnimatePresence, motion } from 'motion/react';
import { Columns2, Dices, Download, History, PanelRightIcon, SquarePen } from 'lucide-react';
import { useLazyQuery } from '@apollo/client/react';
import { toast } from 'sonner';
import { cn } from '@/lib/utils/utils';
import { Button } from '@/lib/ui-primitives/button';
import { Input } from '@/lib/ui-primitives/input';
import { Slider } from '@/lib/ui-primitives/slider';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/lib/ui-primitives/tooltip';
import {
  configKey,
  useSteeringChat,
  type SteeringChatOptions,
} from '@/lib/hooks/useSteeringChat';
import { downloadJson } from '@/lib/utils/downloadJson';
import { isQwenModel } from '@/lib/utils/modelCheckpoints';
import { useModelIdentityStore } from '@/lib/stores/useModelIdentityStore';
import { STEERING_PRESETS } from '@/lib/utils/steeringPresets';
import { GET_CHAT_SESSION, type ChatSessionQueryResult } from '@/lib/graphql/queries';
import type {
  ChatMessage as ChatMessageType,
  ChatSessionSummary,
  MessageVote,
  SaeFeature,
  SteeringConfig,
} from '@/lib/types/types';
import { ChatHistory } from './ChatHistory';
import { ChatInput } from './ChatInput';
import { ChatThread } from './ChatThread';
import { SteeringControls } from './SteeringControls';

interface ChatPanelProps {
  currentFeature: SaeFeature | null;
  onClose?: () => void;
  // Chat history props
  sessions?: ChatSessionSummary[];
  sessionsLoading?: boolean;
  activeSessionId?: string | null;
  onSelectSession?: (id: string) => void;
  onDeleteSession?: (id: string) => void;
  onNewChat?: () => void;
  onUserMessageSent?: (message: ChatMessageType) => void;
  onAssistantMessageComplete?: (message: ChatMessageType) => void;
  /** When set, replaces current messages (used for loading a session). Reset to null after load. */
  loadedMessages?: ChatMessageType[] | null;
  onSelectModel?: (modelId: string, saeId: string) => void;
}

/**
 * TEMP (video recording): suppress auto-loading the steering presets so the
 * chat opens with an empty steering config. Set back to `false` to restore the
 * default preset bundle. The presets themselves are untouched in
 * `lib/utils/steeringPresets.ts`.
 */
const DISABLE_STEERING_PRESETS = true;

/** Baseline (no steering) config — stable identity so the baseline hook never auto-resets. */
const EMPTY_CONFIG: SteeringConfig = { features: [] };

/** Random seed for reproducible sampling (dice button + compare auto-roll). */
const rollSeed = () => Math.floor(Math.random() * 2 ** 31);

/** Chat-header icon button with a tooltip (the icons alone are cryptic). */
function HeaderIconButton({
  label,
  children,
  className,
  ...buttonProps
}: { label: string } & ComponentProps<typeof Button>) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button
          size="icon"
          {...buttonProps}
          className={cn('size-7 text-muted-foreground', className)}
        >
          {children}
          <span className="sr-only">{label}</span>
        </Button>
      </TooltipTrigger>
      <TooltipContent side="bottom">{label}</TooltipContent>
    </Tooltip>
  );
}

export function ChatPanel({
  currentFeature,
  onClose,
  sessions = [],
  sessionsLoading = false,
  activeSessionId = null,
  onSelectSession,
  onDeleteSession,
  onNewChat,
  onUserMessageSent,
  onAssistantMessageComplete,
  loadedMessages,
  onSelectModel,
}: ChatPanelProps) {
  // Read model identity + steering config from store
  const steeringConfig = useModelIdentityStore((s) => s.steeringConfig);
  const [maxTokens, setMaxTokens] = useState(256);
  const [temperature, setTemperature] = useState(0.7);
  // null = unseeded (natural sampling). Compare mode auto-rolls a seed.
  // seedText is the free-form input draft — it can transiently disagree with
  // `seed` while typing (e.g. "-5" is rejected → unseeded) and snaps back to
  // the committed value on blur, so the display never lies about the seed.
  const [seed, setSeed] = useState<number | null>(null);
  const [seedText, setSeedText] = useState('');
  // Qwen-only: stream the model's raw <think> reasoning (off = suppressed).
  const [enableThinking, setEnableThinking] = useState(false);

  const applySeed = useCallback((value: number | null) => {
    setSeed(value);
    setSeedText(value?.toString() ?? '');
  }, []);
  const [showHistory, setShowHistory] = useState(false);
  const [compareMode, setCompareMode] = useState(false);
  const prevLoadedRef = useRef<ChatMessageType[] | null | undefined>(undefined);

  const chatOptions: SteeringChatOptions = useMemo(
    () => ({
      onUserMessageSent,
      onAssistantMessageComplete,
    }),
    [onUserMessageSent, onAssistantMessageComplete]
  );

  // Steered thread — the persisted conversation (receives the persistence callbacks).
  const {
    messages: steeredMessages,
    status: steeredStatus,
    error: steeredError,
    send: steeredSend,
    stop: steeredStop,
    reset: steeredReset,
    regenerate: steeredRegenerate,
    editAndResend: steeredEdit,
    loadMessages: steeredLoad,
  } = useSteeringChat(steeringConfig, maxTokens, temperature, seed, chatOptions, enableThinking);

  // Baseline thread — no steering, no persistence callbacks (ephemeral).
  const {
    messages: baselineMessages,
    status: baselineStatus,
    error: baselineError,
    send: baselineSend,
    stop: baselineStop,
    reset: baselineReset,
  } = useSteeringChat(EMPTY_CONFIG, maxTokens, temperature, seed, undefined, enableThinking);

  const [votes, setVotes] = useState<Map<string, MessageVote>>(new Map());

  // Sync the steered thread with the parent's `loadedMessages` signal. An empty
  // array is the parent's new-chat/clear signal (see page.tsx `handleNewChat`);
  // a non-empty array is a real saved-session load. Only a real session load —
  // a single steered thread — drops out of compare mode and clears the baseline.
  // (Guarding on length is essential: `handleToggleCompare` calls `onNewChat`,
  // which sets `loadedMessages` to `[]`; an unguarded reset here would flip
  // compare mode straight back off.)
  useEffect(() => {
    if (loadedMessages && loadedMessages !== prevLoadedRef.current) {
      prevLoadedRef.current = loadedMessages;
      steeredLoad(loadedMessages, steeringConfig);
      if (loadedMessages.length > 0) {
        setCompareMode(false);
        baselineReset();
      }
    }
  }, [loadedMessages, steeredLoad, steeringConfig, baselineReset]);

  // Keep the baseline thread in lockstep with the steered thread's auto-reset:
  // the steered hook clears itself when the (strength-filtered) steering config
  // changes, so clear the baseline too. Constant "" on mount → no spurious fire.
  const steeringKey = configKey(steeringConfig);
  useEffect(() => {
    baselineReset();
  }, [steeringKey, baselineReset]);

  // Auto-load model-specific steering presets when the chat opens against
  // a model that has a preset bundle and no features are configured yet.
  // Presets ship at strength 0 — the user activates them via the slider.
  const modelId = useModelIdentityStore((s) => s.modelId);
  const isQwen = isQwenModel(modelId);
  useEffect(() => {
    if (DISABLE_STEERING_PRESETS) return;
    if (!modelId) return;
    const presets = STEERING_PRESETS[modelId];
    if (!presets) return;
    const { steeringConfig: cfg, setSteeringConfig } = useModelIdentityStore.getState();
    if (cfg.features.length > 0) return;
    setSteeringConfig({ features: presets });
  }, [modelId]);

  const steeredBusy = steeredStatus === 'loading_model' || steeredStatus === 'generating';
  const baselineBusy = baselineStatus === 'loading_model' || baselineStatus === 'generating';
  const isBusy = steeredBusy || (compareMode && baselineBusy);
  const steeredEmpty = steeredMessages.length === 0;

  const handleVote = useCallback((messageId: string, isUpvoted: boolean) => {
    setVotes((prev) => {
      const next = new Map(prev);
      const existing = next.get(messageId);
      if (existing?.isUpvoted === isUpvoted) {
        next.delete(messageId);
      } else {
        next.set(messageId, { messageId, isUpvoted });
      }
      return next;
    });
  }, []);

  // Fan the shared input out to both threads. Input is gated on !isBusy so both
  // start from idle; useSteeringChat.send also no-ops unless status === 'idle'.
  const handleSend = useCallback(
    (content: string) => {
      steeredSend(content);
      if (compareMode) baselineSend(content);
    },
    [steeredSend, baselineSend, compareMode]
  );

  const handleStop = useCallback(() => {
    steeredStop();
    if (compareMode) baselineStop();
  }, [steeredStop, baselineStop, compareMode]);

  // Clear both threads and start a fresh session.
  const handleNewChat = useCallback(() => {
    steeredReset();
    baselineReset();
    onNewChat?.();
  }, [steeredReset, baselineReset, onNewChat]);

  // Toggling compare mode resets both threads for a clean side-by-side comparison.
  // Entering compare with no seed rolls one, so steered-vs-baseline differences
  // stay attributable to steering rather than sampling noise. The rolled seed
  // deliberately survives compare exit (visible + reproducible); clear the
  // field manually to return to natural sampling.
  const handleToggleCompare = useCallback(() => {
    steeredReset();
    baselineReset();
    onNewChat?.();
    if (!compareMode && seed == null) applySeed(rollSeed());
    setCompareMode((v) => !v);
  }, [steeredReset, baselineReset, onNewChat, compareMode, seed, applySeed]);

  const handleRegenerate = useCallback(
    (assistantMessageIndex: number) => {
      for (let i = assistantMessageIndex - 1; i >= 0; i--) {
        if (steeredMessages[i].role === 'user') {
          steeredRegenerate(assistantMessageIndex);
          return;
        }
      }
    },
    [steeredMessages, steeredRegenerate]
  );

  // Download the active session as JSON (fetched fresh so per-message
  // steering snapshots are included even for messages saved this turn).
  const [fetchSessionForDownload] = useLazyQuery<ChatSessionQueryResult>(
    GET_CHAT_SESSION,
    { fetchPolicy: 'network-only' },
  );

  const handleDownload = useCallback(async () => {
    if (!activeSessionId) return;
    try {
      const { data } = await fetchSessionForDownload({ variables: { id: activeSessionId } });
      const session = data?.chatSession;
      if (!session) {
        toast.error('Session not found');
        return;
      }
      downloadJson(
        {
          schemaVersion: 1,
          exportedAt: new Date().toISOString(),
          session: {
            id: session.id,
            title: session.title,
            config: session.config,
            createdAt: session.createdAt,
            updatedAt: session.updatedAt,
          },
          messages: session.messages,
        },
        `chat-${session.id}.json`,
      );
    } catch (err) {
      console.error('Failed to download chat:', err);
      toast.error('Failed to download chat');
    }
  }, [activeSessionId, fetchSessionForDownload]);

  const canDownload = !!activeSessionId && steeredMessages.length > 0;

  return (
    <div className="flex h-full">
      {/* Lateral history panel */}
      <AnimatePresence>
        {showHistory && (
          <motion.div
            initial={{ width: 0, opacity: 0 }}
            animate={{ width: 200, opacity: 1 }}
            exit={{ width: 0, opacity: 0 }}
            transition={{ duration: 0.25, ease: [0.22, 1, 0.36, 1] }}
            className="flex h-full shrink-0 flex-col overflow-hidden border-r border-border/30"
          >
            <ChatHistory
              sessions={sessions}
              loading={sessionsLoading}
              activeSessionId={activeSessionId}
              onSelectSession={(id) => onSelectSession?.(id)}
              onDeleteSession={(id) => onDeleteSession?.(id)}
              onNewChat={handleNewChat}
            />
          </motion.div>
        )}
      </AnimatePresence>

      {/* Main chat column */}
      <div className="flex min-w-0 flex-1 flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-4 pt-4 pb-0">
          <h2 className="text-sm font-semibold">Steered Chat</h2>
          <div className="flex items-center gap-1">
            <HeaderIconButton
              label={compareMode ? 'Exit compare mode' : 'Compare with baseline'}
              variant={compareMode ? 'secondary' : 'ghost'}
              onClick={handleToggleCompare}
            >
              <Columns2 className="size-3.5" />
            </HeaderIconButton>
            <HeaderIconButton
              label="History"
              variant={showHistory ? 'secondary' : 'ghost'}
              onClick={() => setShowHistory((v) => !v)}
            >
              <History className="size-3.5" />
            </HeaderIconButton>
            <HeaderIconButton
              label="Download chat as JSON"
              variant="ghost"
              onClick={handleDownload}
              disabled={!canDownload}
            >
              <Download className="size-3.5" />
            </HeaderIconButton>
            <HeaderIconButton label="New chat" variant="ghost" onClick={handleNewChat}>
              <SquarePen className="size-3.5" />
            </HeaderIconButton>
            {onClose && (
              <HeaderIconButton label="Close chat" variant="ghost" onClick={onClose}>
                <PanelRightIcon className="size-3.5" />
              </HeaderIconButton>
            )}
          </div>
        </div>

        {/* Steering controls */}
        <SteeringControls
          currentFeature={currentFeature}
        />

        {/* Generation parameters */}
        <div className="flex flex-col gap-1.5 border-b border-border/30 px-4 py-2">
          <div className="flex items-center gap-3">
            <span className="w-16 shrink-0 text-[11px] text-muted-foreground">Max tokens</span>
            <Slider
              value={[maxTokens]}
              min={32}
              max={2048}
              step={32}
              onValueChange={([v]) => setMaxTokens(v)}
              className="flex-1"
            />
            <span className="w-10 shrink-0 text-right font-mono text-[10px] text-muted-foreground tabular-nums">
              {maxTokens}
            </span>
          </div>
          <div className="flex items-center gap-3">
            <span className="w-16 shrink-0 text-[11px] text-muted-foreground">Temperature</span>
            <Slider
              value={[temperature]}
              min={0}
              max={2}
              step={0.05}
              onValueChange={([v]) => setTemperature(v)}
              className="flex-1"
            />
            <span className="w-10 shrink-0 text-right font-mono text-[10px] text-muted-foreground tabular-nums">
              {temperature.toFixed(2)}
            </span>
          </div>
          {/* One shared seed for both threads, re-applied verbatim each turn
              (not advanced). Empty = natural sampling; entering compare mode
              auto-rolls one so differences stay attributable to steering. */}
          <div className="flex items-center gap-3">
            <span className="w-16 shrink-0 text-[11px] text-muted-foreground">Seed</span>
            <div className="flex-1" />
            <Button
              size="icon"
              variant="ghost"
              onClick={() => applySeed(rollSeed())}
              className="size-7 shrink-0 text-muted-foreground"
            >
              <Dices className="size-3.5" />
              <span className="sr-only">Randomize seed</span>
            </Button>
            <Input
              type="number"
              min={0}
              value={seedText}
              placeholder="random"
              onChange={(e) => {
                const raw = e.target.value;
                setSeedText(raw);
                const n = Math.floor(Number(raw));
                setSeed(raw !== '' && Number.isFinite(n) && n >= 0 ? n : null);
              }}
              onBlur={() => setSeedText(seed?.toString() ?? '')}
              className="no-spinner h-7 w-24 shrink-0 px-2 text-right font-mono text-[10px]! tabular-nums"
              aria-label="Generation seed (empty for natural sampling)"
            />
          </div>
        </div>

        {/* Messages area — single thread, or steered vs. baseline side-by-side */}
        {compareMode ? (
          <div className="flex min-h-0 flex-1">
            <div className="flex min-w-0 flex-1 flex-col border-r border-border/30">
              <div className="border-b border-border/30 px-4 py-1.5 text-center text-[11px] font-medium text-muted-foreground">
                Steered
              </div>
              <ChatThread
                messages={steeredMessages}
                status={steeredStatus}
                error={steeredError}
                steeringFeatures={steeringConfig.features}
                votes={votes}
                onVote={handleVote}
              />
            </div>
            <div className="flex min-w-0 flex-1 flex-col">
              <div className="border-b border-border/30 px-4 py-1.5 text-center text-[11px] font-medium text-muted-foreground">
                Baseline
              </div>
              <ChatThread
                messages={baselineMessages}
                status={baselineStatus}
                error={baselineError}
                steeringFeatures={EMPTY_CONFIG.features}
                votes={votes}
                onVote={handleVote}
                isBaseline
              />
            </div>
          </div>
        ) : (
          <ChatThread
            messages={steeredMessages}
            status={steeredStatus}
            error={steeredError}
            steeringFeatures={steeringConfig.features}
            votes={votes}
            onVote={handleVote}
            onEdit={steeredEdit}
            onRegenerate={handleRegenerate}
            onStartCompare={handleToggleCompare}
          />
        )}

        {/* Input area */}
        <ChatInput
          onSend={handleSend}
          onStop={handleStop}
          isGenerating={isBusy}
          showSuggestions={steeredEmpty}
          onSuggest={handleSend}
          onSelectModel={onSelectModel}
          showThinking={isQwen}
          thinkingEnabled={enableThinking}
          onToggleThinking={() => setEnableThinking((v) => !v)}
        />
      </div>
    </div>
  );
}
