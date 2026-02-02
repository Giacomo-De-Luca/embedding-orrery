'use client';

import React, { useState, useMemo } from 'react';
import { Palette } from 'lucide-react';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/lib/ui-primitives/dialog';
import { Button } from '@/lib/ui-primitives/button';
import { Label } from '@/lib/ui-primitives/label';
import { RadioGroup, RadioGroupItem } from '@/lib/ui-primitives/radio-group';
import type { ColorScaleType } from '@/lib/types/types';
import { getSequentialScale, getDivergingScale, getMonochromeScale, generateCategoryColors, type SequentialScaleName, type DivergingScaleName } from '@/lib/utils/categoryColors';

// Human-readable labels for scale names
const SEQUENTIAL_SCALE_LABELS: Record<SequentialScaleName, string> = {
  sinebow: 'Sinebow',
  viridis: 'Viridis',
  cividis: 'Cividis',
  turbo: 'Turbo',
  plasma: 'Plasma',
  inferno: 'Inferno',
  magma: 'Magma',
};

const DIVERGING_SCALE_LABELS: Record<DivergingScaleName, string> = {
  blueGold: 'Blue-Purple-Gold',
  rdBu: 'Red-Blue',
  spectral: 'Spectral',
  piYG: 'Pink-Yellow-Green',
  puOr: 'Purple-Orange',
  brBG: 'Brown-Blue-Green',
};

// Get all scale names as arrays for iteration
const SEQUENTIAL_SCALE_NAMES = Object.keys(SEQUENTIAL_SCALE_LABELS) as SequentialScaleName[];
const DIVERGING_SCALE_NAMES = Object.keys(DIVERGING_SCALE_LABELS) as DivergingScaleName[];

interface ColorScaleSelectorProps {
  colorScaleType: ColorScaleType;
  onColorScaleTypeChange: (type: ColorScaleType) => void;
  monochromeColor?: string;
  onMonochromeColorChange?: (color: string) => void;
  sequentialScaleName?: SequentialScaleName;
  onSequentialScaleNameChange?: (name: SequentialScaleName) => void;
  divergingScaleName?: DivergingScaleName;
  onDivergingScaleNameChange?: (name: DivergingScaleName) => void;
}

interface ColorScalePreviewProps {
  type: ColorScaleType;
  baseColor?: string;
  sequentialScaleName?: SequentialScaleName;
  divergingScaleName?: DivergingScaleName;
}

function ColorScalePreview({
  type,
  baseColor = '#1f77b4',
  sequentialScaleName = 'sinebow',
  divergingScaleName = 'blueGold',
}: ColorScalePreviewProps) {
  const colors = useMemo(() => {
    if (type === 'categorical') {
      return generateCategoryColors(10);
    } else if (type === 'sequential') {
      const scale = getSequentialScale([0, 1], sequentialScaleName);
      return Array.from({ length: 10 }, (_, i) => scale(i / 9));
    } else if (type === 'monochrome') {
      const scale = getMonochromeScale(baseColor, [0, 1]);
      return Array.from({ length: 10 }, (_, i) => scale(i / 9));
    } else {
      const scale = getDivergingScale([-1, 0, 1], divergingScaleName);
      return Array.from({ length: 10 }, (_, i) => scale(-1 + (i * 2) / 9));
    }
  }, [type, baseColor, sequentialScaleName, divergingScaleName]);

  if (type === 'categorical') {
    return (
      <div className="flex gap-0.5">
        {colors.map((color, i) => (
          <div
            key={i}
            className="h-4 w-4 rounded-sm"
            style={{ backgroundColor: color }}
          />
        ))}
      </div>
    );
  }

  return (
    <div
      className="h-2 w-full rounded-sm"
      style={{
        background: `linear-gradient(to right, ${colors.join(', ')})`,
      }}
    />
  );
}

export function ColorScaleSelector({
  colorScaleType,
  onColorScaleTypeChange,
  monochromeColor = '#1f77b4',
  onMonochromeColorChange,
  sequentialScaleName = 'sinebow',
  onSequentialScaleNameChange,
  divergingScaleName = 'blueGold',
  onDivergingScaleNameChange,
}: ColorScaleSelectorProps) {
  const [open, setOpen] = useState(false);
  const [localMonochromeColor, setLocalMonochromeColor] = useState(monochromeColor);

  const handleOpen = (isOpen: boolean) => {
    if (!isOpen) {
      setLocalMonochromeColor(monochromeColor);
    }
    setOpen(isOpen);
  };

  const handleTypeChange = (value: ColorScaleType) => {
    onColorScaleTypeChange(value);
  };

  const handleColorChange = (color: string) => {
    setLocalMonochromeColor(color);
    if (onMonochromeColorChange) {
      onMonochromeColorChange(color);
    }
  };

  const handleSequentialScaleChange = (name: SequentialScaleName) => {
    if (onSequentialScaleNameChange) {
      onSequentialScaleNameChange(name);
    }
  };

  const handleDivergingScaleChange = (name: DivergingScaleName) => {
    if (onDivergingScaleNameChange) {
      onDivergingScaleNameChange(name);
    }
  };

  return (
    <Dialog open={open} onOpenChange={handleOpen}>
      <DialogTrigger asChild>
        <Button
          variant="ghost"
          size="icon"
          className="h-8 w-8"
          title="Color scale settings"
        >
          <Palette className="h-4 w-4" />
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Color Scale</DialogTitle>
        </DialogHeader>

        <div className="space-y-6 py-4">
          <RadioGroup
            value={colorScaleType}
            onValueChange={(value) => handleTypeChange(value as ColorScaleType)}
            className="space-y-4"
          >
            <div className="space-y-2">
              <div className="flex items-center space-x-3">
                <RadioGroupItem value="categorical" id="categorical" />
                <Label htmlFor="categorical" className="font-medium">
                  Categorical
                </Label>
              </div>
              <p className="text-muted-foreground ml-6 text-sm">
                Distinct colors for discrete categories
              </p>
              <div className="ml-6">
                <ColorScalePreview type="categorical" />
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex items-center space-x-3">
                <RadioGroupItem value="sequential" id="sequential" />
                <Label htmlFor="sequential" className="font-medium">
                  Sequential
                </Label>
              </div>
              <p className="text-muted-foreground ml-6 text-sm">
                Continuous scale for numeric values (low → high)
              </p>
              <div className="ml-6">
                <ColorScalePreview type="sequential" sequentialScaleName={sequentialScaleName} />
              </div>
              {colorScaleType === 'sequential' && (
                <div className="ml-6 mt-2 space-y-2">
                  {SEQUENTIAL_SCALE_NAMES.map((name) => (
                    <div
                      key={name}
                      className={`flex items-center gap-3 p-2 rounded-md cursor-pointer transition-colors ${
                        sequentialScaleName === name
                          ? 'bg-accent'
                          : 'hover:bg-accent/50'
                      }`}
                      onClick={() => handleSequentialScaleChange(name)}
                    >
                      <div className="w-24 h-2 rounded-sm overflow-hidden">
                        <ColorScalePreview type="sequential" sequentialScaleName={name} />
                      </div>
                      <span className="text-sm">{SEQUENTIAL_SCALE_LABELS[name]}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>

            <div className="space-y-2">
              <div className="flex items-center space-x-3">
                <RadioGroupItem value="diverging" id="diverging" />
                <Label htmlFor="diverging" className="font-medium">
                  Diverging
                </Label>
              </div>
              <p className="text-muted-foreground ml-6 text-sm">
                Centered scale for values with a midpoint
              </p>
              <div className="ml-6">
                <ColorScalePreview type="diverging" divergingScaleName={divergingScaleName} />
              </div>
              {colorScaleType === 'diverging' && (
                <div className="ml-6 mt-2 space-y-2">
                  {DIVERGING_SCALE_NAMES.map((name) => (
                    <div
                      key={name}
                      className={`flex items-center gap-3 p-2 rounded-md cursor-pointer transition-colors ${
                        divergingScaleName === name
                          ? 'bg-accent'
                          : 'hover:bg-accent/50'
                      }`}
                      onClick={() => handleDivergingScaleChange(name)}
                    >
                      <div className="w-24 h-2 rounded-sm overflow-hidden">
                        <ColorScalePreview type="diverging" divergingScaleName={name} />
                      </div>
                      <span className="text-sm">{DIVERGING_SCALE_LABELS[name]}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>

            <div className="space-y-2">
              <div className="flex items-center space-x-3">
                <RadioGroupItem value="monochrome" id="monochrome" />
                <Label htmlFor="monochrome" className="font-medium">
                  Monochrome (Single Color)
                </Label>
              </div>
              <p className="text-muted-foreground ml-6 text-sm">
                Opacity gradient using a single base color
              </p>
              <div className="ml-6">
                <ColorScalePreview type="monochrome" baseColor={monochromeColor} />
              </div>
              {colorScaleType === 'monochrome' && (
                <div className="ml-6 mt-2 flex items-center gap-2">
                  <Label htmlFor="mono-color" className="text-sm">Base Color:</Label>
                  <input
                    id="mono-color"
                    type="color"
                    value={localMonochromeColor}
                    onChange={(e) => handleColorChange(e.target.value)}
                    className="h-8 w-12 cursor-pointer rounded border"
                  />
                </div>
              )}
            </div>
          </RadioGroup>
        </div>
      </DialogContent>
    </Dialog>
  );
}
