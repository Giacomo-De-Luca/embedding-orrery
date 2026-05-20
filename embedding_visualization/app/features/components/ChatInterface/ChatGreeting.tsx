'use client';

import { motion } from 'motion/react';
import { Sparkles } from 'lucide-react';
import { useModelIdentityStore } from '@/lib/stores/useModelIdentityStore';
import { SteeringIdenticon } from './SteeringIdenticon';

interface ChatGreetingProps {
  featureCount: number;
}

export function ChatGreeting({ featureCount }: ChatGreetingProps) {
  const steeringConfig = useModelIdentityStore((s) => s.steeringConfig);

  return (
    <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
      <div className="flex flex-col items-center px-6">
        {/* Icon */}
        <motion.div
          className="flex size-10 items-center justify-center rounded-xl bg-muted/60 ring-1 ring-border/50"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
        >
          <SteeringIdenticon
            features={steeringConfig.features}
            size={40}
            fallback={<Sparkles className="size-5 text-muted-foreground" />}
            crossfadeOnChange
          />
        </motion.div>

        {/* Heading */}
        <motion.h3
          className="mt-4 text-center text-2xl font-semibold tracking-tight text-foreground"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.35, duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
        >
          Chat with Steered Gemma
        </motion.h3>

        {/* Subtitle */}
        <motion.p
          className="mt-3 text-center text-sm text-muted-foreground/80"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5, duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
        >
          {featureCount > 0
            ? `Steering with ${featureCount} feature${featureCount > 1 ? 's' : ''}`
            : 'Add features to steer the model'}
        </motion.p>
      </div>
    </div>
  );
}
