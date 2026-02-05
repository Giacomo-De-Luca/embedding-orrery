'use client';

import * as React from 'react';
import { CheckIcon, ChevronDownIcon, XIcon } from 'lucide-react';
import { cn } from '@/lib/utils/utils';

const PREDEFINED_PROMPTS = [
  { value: 'auto', label: 'Auto-detect from collection' },
  { value: 'Retrieval-query', label: 'Retrieval-query (RAG search)' },
  { value: 'Retrieval-document', label: 'Retrieval-document (RAG storage)' },
  { value: 'STS', label: 'STS (Sentence similarity)' },
  { value: 'Classification', label: 'Classification' },
  { value: 'Clustering', label: 'Clustering' },
];

interface PromptComboboxProps {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  className?: string;
  id?: string;
}

export function PromptCombobox({
  value,
  onChange,
  placeholder = 'Select or type a prompt...',
  className,
  id,
}: PromptComboboxProps) {
  const [open, setOpen] = React.useState(false);
  const [inputValue, setInputValue] = React.useState(value);
  const containerRef = React.useRef<HTMLDivElement>(null);
  const inputRef = React.useRef<HTMLInputElement>(null);

  // Sync when external value changes
  React.useEffect(() => {
    setInputValue(value);
  }, [value]);

  // Close on outside click
  React.useEffect(() => {
    if (!open) return;
    const handleClickOutside = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [open]);

  const filtered = PREDEFINED_PROMPTS.filter(
    (p) =>
      p.label.toLowerCase().includes(inputValue.toLowerCase()) ||
      p.value.toLowerCase().includes(inputValue.toLowerCase())
  );

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const v = e.target.value;
    setInputValue(v);
    onChange(v);
    if (!open) setOpen(true);
  };

  const handleSelect = (promptValue: string) => {
    setInputValue(promptValue);
    onChange(promptValue);
    setOpen(false);
  };

  const handleClear = () => {
    setInputValue('');
    onChange('');
    inputRef.current?.focus();
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Escape') setOpen(false);
  };

  return (
    <div ref={containerRef} className="relative">
      <div
        className={cn(
          'border-input flex items-center rounded-md border backdrop-blur-sm bg-white/80 dark:bg-input/30 shadow-xs transition-[color,box-shadow]',
          'focus-within:border-ring focus-within:ring-ring/50 focus-within:ring-[3px]',
          className
        )}
      >
        <input
          ref={inputRef}
          id={id}
          type="text"
          value={inputValue}
          onChange={handleInputChange}
          onFocus={() => setOpen(true)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          className="flex-1 min-w-0 bg-transparent px-3 py-1 text-sm outline-none placeholder:text-muted-foreground"
        />
        {value && (
          <button
            type="button"
            onClick={handleClear}
            className="text-muted-foreground hover:text-foreground px-1 shrink-0"
          >
            <XIcon className="size-3.5" />
          </button>
        )}
        <button
          type="button"
          onClick={() => setOpen(!open)}
          className="text-muted-foreground hover:text-foreground px-2 shrink-0"
        >
          <ChevronDownIcon className="size-4" />
        </button>
      </div>

      {open && filtered.length > 0 && (
        <div className="bg-popover text-popover-foreground absolute z-50 mt-1 w-full overflow-hidden rounded-md border shadow-md animate-in fade-in-0 zoom-in-95 slide-in-from-top-2">
          <div className="max-h-60 overflow-y-auto p-1">
            {filtered.map((prompt) => (
              <button
                key={prompt.value}
                type="button"
                onClick={() => handleSelect(prompt.value)}
                className={cn(
                  'relative flex w-full cursor-default items-center rounded-sm py-1.5 pr-8 pl-2 text-sm outline-hidden select-none',
                  'hover:bg-accent hover:text-accent-foreground',
                  value === prompt.value && 'bg-accent/50'
                )}
              >
                {prompt.label}
                {value === prompt.value && (
                  <span className="absolute right-2 flex size-3.5 items-center justify-center">
                    <CheckIcon className="size-4" />
                  </span>
                )}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
