'use client';

import type { LucideIcon } from 'lucide-react';
import { Type, Images, Target } from 'lucide-react';
import { Label } from '@/lib/ui-primitives/label';
import { RadioGroup, RadioGroupItem } from '@/lib/ui-primitives/radio-group';
import type { DataType } from '@/lib/graphql/mutations';
import { DATA_TYPE_LABELS } from '../lib/mediaColumns';

const DATA_TYPE_OPTIONS: Record<DataType, { icon: LucideIcon; summary: string; description: string }> = {
  TEXT: {
    icon: Type,
    summary: 'embed text columns',
    description: 'Embed text with sentence-transformers or another text embedding model',
  },
  MULTIMODAL: {
    icon: Images,
    summary: 'media columns, optionally with text',
    description:
      'Embed image, audio and video columns, alone or with text columns, into one space ' +
      '(EmbeddingGemma 2 by default). Text queries then find the media.',
  },
  VECTOR: {
    icon: Target,
    summary: 'load a column of embeddings',
    description: 'Load pre-computed embeddings from a vector column',
  },
};

interface DataTypeSelectorProps {
  dataType: DataType;
  onDataTypeChange: (type: DataType) => void;
  /** The data types this source supports (VECTOR is local files only). */
  options?: DataType[];
  /** Prefix of the radio ids, unique per form (two tabs render a selector). */
  idPrefix?: string;
  disabled?: boolean;
}

export function DataTypeSelector({
  dataType,
  onDataTypeChange,
  options = ['TEXT', 'MULTIMODAL', 'VECTOR'],
  idPrefix = '',
  disabled,
}: DataTypeSelectorProps) {
  return (
    <div className="space-y-3">
      <Label>Data Type</Label>
      <RadioGroup
        value={dataType}
        onValueChange={(value) => onDataTypeChange(value as DataType)}
        disabled={disabled}
      >
        {options.map((type) => {
          const { icon: Icon, summary } = DATA_TYPE_OPTIONS[type];
          const id = `${idPrefix}data-type-${type.toLowerCase()}`;
          return (
            <div key={type} className="flex items-center space-x-2">
              <RadioGroupItem value={type} id={id} />
              <Label htmlFor={id} className="font-normal cursor-pointer flex items-center gap-2">
                <Icon className="h-4 w-4" /> {DATA_TYPE_LABELS[type]}: {summary}
              </Label>
            </div>
          );
        })}
      </RadioGroup>
      <p className="text-xs text-muted-foreground">{DATA_TYPE_OPTIONS[dataType].description}</p>
    </div>
  );
}
