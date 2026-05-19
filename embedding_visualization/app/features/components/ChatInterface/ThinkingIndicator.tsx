'use client';

import { useState } from 'react';
import { motion } from 'motion/react';
import type { SteeringFeature } from '@/lib/types/types';
import { AssistantAvatar } from './AssistantAvatar';
import { Shimmer } from './Shimmer';

const MODEL_LOADING_PHRASES = [
  'Roaring the motors...',
  'Warming up the neurons...',
  'Loading the brain...',
  'Dusting off the weights...',
  'Charging the circuits...',
];

interface ThinkingIndicatorProps {
  phase?: 'thinking' | 'loading_model';
  features?: SteeringFeature[];
}

export function ThinkingIndicator({ phase = 'thinking', features = [] }: ThinkingIndicatorProps) {
  const [phraseIndex] = useState(() =>
    Math.floor(Math.random() * MODEL_LOADING_PHRASES.length)
  );

  const text = phase === 'loading_model'
    ? MODEL_LOADING_PHRASES[phraseIndex]
    : 'Thinking...';

  return (
    <motion.div
      className="flex w-full items-start gap-3"
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
    >
      <div className="flex h-[calc(13px*1.65)] shrink-0 items-center">
        <AssistantAvatar features={features} />
      </div>
      <div className="flex h-[calc(13px*1.65)] items-center text-[13px] leading-[1.65]">
        <Shimmer className="font-medium" duration={phase === 'loading_model' ? 2 : 1}>
          {text}
        </Shimmer>
      </div>
    </motion.div>
  );
}
