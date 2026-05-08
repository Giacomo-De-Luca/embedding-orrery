'use client';

import {
  Reasoning,
  ReasoningContent,
  ReasoningTrigger,
} from './ai-elements/reasoning';

interface MessageReasoningProps {
  reasoning: string;
  isStreaming: boolean;
}

export function MessageReasoning({ reasoning, isStreaming }: MessageReasoningProps) {
  return (
    <Reasoning isStreaming={isStreaming}>
      <ReasoningTrigger />
      <ReasoningContent>{reasoning}</ReasoningContent>
    </Reasoning>
  );
}
