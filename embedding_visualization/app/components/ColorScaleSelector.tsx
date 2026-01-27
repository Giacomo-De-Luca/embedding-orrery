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
import { getSequentialScale, getDivergingScale, getMonochromeScale, generateCategoryColors } from '@/lib/utils/categoryColors';

interface ColorScaleSelectorProps {
  colorScaleType: ColorScaleType;
  onColorScaleTypeChange: (type: ColorScaleType) => void;
  monochromeColor?: string;
  onMonochromeColorChange?: (color: string) => void;
}

function ColorScalePreview({ type, baseColor = '#1f77b4' }: { type: ColorScaleType; baseColor?: string }) {
  const colors = useMemo(() => {
    if (type === 'categorical') {
      return generateCategoryColors(10);
    } else if (type === 'sequential') {
      const scale = getSequentialScale([0, 1]);
      return Array.from({ length: 10 }, (_, i) => scale(i / 9));
    } else if (type === 'monochrome') {
      const scale = getMonochromeScale(baseColor, [0, 1]);
      return Array.from({ length: 10 }, (_, i) => scale(i / 9));
    } else {
      const scale = getDivergingScale([-1, 0, 1]);
      return Array.from({ length: 10 }, (_, i) => scale(-1 + (i * 2) / 9));
    }
  }, [type, baseColor]);

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
                  Sequential (Viridis)
                </Label>
              </div>
              <p className="text-muted-foreground ml-6 text-sm">
                Continuous scale for numeric values (low → high)
              </p>
              <div className="ml-6">
                <ColorScalePreview type="sequential" />
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex items-center space-x-3">
                <RadioGroupItem value="diverging" id="diverging" />
                <Label htmlFor="diverging" className="font-medium">
                  Diverging (Blue-Gold)
                </Label>
              </div>
              <p className="text-muted-foreground ml-6 text-sm">
                Centered scale for values with a midpoint
              </p>
              <div className="ml-6">
                <ColorScalePreview type="diverging" />
              </div>
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
