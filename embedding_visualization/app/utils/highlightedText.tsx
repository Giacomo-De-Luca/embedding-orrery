import * as React from 'react';
import { cn } from '@/lib/utils/utils';

interface HighlightedTextProps {
  text: string;
  query: string;
  className?: string;
  highlightClassName?: string;
}

export const HighlightedText = React.memo(function HighlightedText({
  text,
  query,
  className,
  highlightClassName,
}: HighlightedTextProps) {
  // Return early if no text or query
  if (!query || !text) {
    return <span className={className}>{text}</span>;
  }

  // 1. Create a Regex with capturing groups. 
  // We still need a tiny bit of escaping to prevent the app from crashing 
  // if the user types symbols like "?", "(", or "*".
  const escapedQuery = query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const regex = new RegExp(`(${escapedQuery})`, 'gi');

  // 2. Split. Because of the capturing group (), the matches are included in the array.
  const parts = text.split(regex);

  return (
    <span className={className}>
      {parts.map((part, index) => {
        // 3. Simplify logic: Regex matches are always at odd indices (1, 3, 5...)
        const isMatch = index % 2 === 1;

        if (isMatch) {
          return (
            <mark
              key={index}
              className={cn(
                "bg-yellow-200 dark:bg-yellow-900/60 dark:text-yellow-100 rounded-[2px] px-0.5 text-inherit font-inherit box-decoration-clone",
                highlightClassName
              )}
            >
              {part}
            </mark>
          );
        }

        return <span key={index}>{part}</span>;
      })}
    </span>
  );
});