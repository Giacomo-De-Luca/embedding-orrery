'use client';

import { motion } from 'motion/react';
import { AssistantAvatar } from './AssistantAvatar';
import { Shimmer } from './Shimmer';

export function ThinkingIndicator() {
  return (
    <motion.div
      className="flex w-full items-start gap-3"
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
    >
      <div className="flex h-[calc(13px*1.65)] shrink-0 items-center">
        <AssistantAvatar />
      </div>
      <div className="flex h-[calc(13px*1.65)] items-center text-[13px] leading-[1.65]">
        <Shimmer className="font-medium" duration={1}>
          Thinking...
        </Shimmer>
      </div>
    </motion.div>
  );
}
