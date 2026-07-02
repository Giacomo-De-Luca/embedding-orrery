'use client';

import { Type, Image, Target } from 'lucide-react';
import { Label } from '@/lib/ui-primitives/label';
import { RadioGroup, RadioGroupItem } from '@/lib/ui-primitives/radio-group';
import type { DataType } from '@/lib/graphql/mutations';

interface DataTypeSelectorProps {
  dataType: DataType;
  onDataTypeChange: (type: DataType) => void;
  disabled?: boolean;
}

export function DataTypeSelector({ dataType, onDataTypeChange, disabled }: DataTypeSelectorProps) {
  return (
    <div className="space-y-3">
      <Label>Data Type</Label>
      <RadioGroup
        value={dataType}
        onValueChange={(value) => onDataTypeChange(value as DataType)}
        disabled={disabled}
      >
        <div className="flex items-center space-x-2">
          <RadioGroupItem value="TEXT" id="text" />
          <Label htmlFor="text" className="font-normal cursor-pointer flex items-center gap-2">
            <Type className="h-4 w-4" /> TEXT - Embed text columns with custom model
          </Label>
        </div>
        <div className="flex items-center space-x-2">
          <RadioGroupItem value="IMAGE" id="image" />
          <Label htmlFor="image" className="font-normal cursor-pointer flex items-center gap-2">
            <Image className="h-4 w-4" /> IMAGE - Embed images using ViT model
          </Label>
        </div>
        <div className="flex items-center space-x-2">
          <RadioGroupItem value="VECTOR" id="vector" />
          <Label htmlFor="vector" className="font-normal cursor-pointer flex items-center gap-2">
            <Target className="h-4 w-4" /> VECTOR - Use pre-computed embeddings
          </Label>
        </div>
      </RadioGroup>
      <p className="text-xs text-muted-foreground">
        {dataType === 'TEXT' && 'Embed text data using sentence transformers or other text embedding models'}
        {dataType === 'IMAGE' && 'Embed images using google/vit-base-patch16-384 (768 dimensions)'}
        {dataType === 'VECTOR' && 'Load pre-computed embeddings from a vector column'}
      </p>
    </div>
  );
}
