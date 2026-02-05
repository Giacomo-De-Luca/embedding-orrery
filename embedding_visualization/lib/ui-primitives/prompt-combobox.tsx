'use client';

import * as React from 'react';
import { Input } from './input';
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
  const listId = React.useId();

  return (
    <>
      <Input
        id={id}
        list={listId}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        className={cn('w-full', className)}
      />
      <datalist id={listId}>
        {PREDEFINED_PROMPTS.map((prompt) => (
          <option key={prompt.value} value={prompt.value}>
            {prompt.label}
          </option>
        ))}
      </datalist>
    </>
  );
}
