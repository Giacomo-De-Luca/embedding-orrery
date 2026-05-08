'use client';

import { Sparkles } from 'lucide-react';

const SUGGESTIONS = [
  { title: 'Explain what you see', subtitle: 'Describe the model\'s behavior' },
  { title: 'Write a poem', subtitle: 'Test creative steering' },
  { title: 'Tell me a story', subtitle: 'Narrative generation' },
  { title: 'Describe your purpose', subtitle: 'Identity and self-model' },
];

interface ChatGreetingProps {
  featureCount: number;
  onSuggest: (prompt: string) => void;
}

export function ChatGreeting({ featureCount, onSuggest }: ChatGreetingProps) {
  return (
    <div className="flex flex-1 flex-col items-center justify-center px-6">
      {/* Icon */}
      <div
        className="animate-fade-up flex size-10 items-center justify-center rounded-xl bg-muted/60 ring-1 ring-border/50"
      >
        <Sparkles className="size-5 text-muted-foreground" />
      </div>

      {/* Heading */}
      <h3
        className="animate-fade-up mt-4 text-center text-2xl font-semibold tracking-tight text-foreground"
        style={{ animationDelay: '0.05s' }}
      >
        Chat with Gemma
      </h3>

      {/* Subtitle */}
      <p
        className="animate-fade-up mt-2 text-center text-sm text-muted-foreground/80"
        style={{ animationDelay: '0.1s' }}
      >
        {featureCount > 0
          ? `Steering with ${featureCount} feature${featureCount > 1 ? 's' : ''}`
          : 'Add features to steer the model'}
      </p>

      {/* Suggested actions — 2-column grid with staggered rise animation */}
      <div
        className="mt-6 grid w-full max-w-sm grid-cols-2 gap-2.5"
      >
        {SUGGESTIONS.map((s, i) => (
          <button
            key={s.title}
            type="button"
            onClick={() => onSuggest(s.title)}
            className="animate-fade-up h-auto w-full rounded-xl border border-border/50 bg-card/30 px-4 py-3 text-left transition-all duration-200 hover:-translate-y-0.5 hover:bg-card/60 hover:text-foreground hover:shadow-[var(--shadow-chat-card)]"
            style={{ animationDelay: `${0.15 + i * 0.06}s` }}
          >
            <span className="block text-[13px] leading-snug text-foreground">
              {s.title}
            </span>
            <span className="mt-0.5 block text-[11px] leading-snug text-muted-foreground/60">
              {s.subtitle}
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}
