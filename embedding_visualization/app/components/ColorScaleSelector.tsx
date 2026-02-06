'use client';

import React, { useState, useMemo, useEffect, useCallback } from 'react';
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
import { ScrollArea } from '@/lib/ui-primitives/scroll-area';
import type { ColorScaleType } from '@/lib/types/types';
import {
  getSequentialScale,
  getDivergingScale,
  getMonochromeScale,
  generateCategoryColors,
  D3_SEQUENTIAL_NAMES,
  D3_DIVERGING_NAMES,
  type SequentialScaleName,
  type DivergingScaleName,
  type D3SequentialScaleName,
  type D3DivergingScaleName,
} from '@/lib/utils/categoryColors';
import {
  CRAMERI_SEQUENTIAL_NAMES,
  CRAMERI_DIVERGING_NAMES,
  CRAMERI_CATEGORICAL_NAMES,
  CRAMERI_SEQUENTIAL_LABELS,
  CRAMERI_DIVERGING_LABELS,
  CRAMERI_CATEGORICAL_LABELS,
  preloadCrameriColormaps,
  crameriGradientCSS,
  getCrameriColors,
  isCrameriLoaded,
} from '@/lib/colorMaps/crameriScales';

// Human-readable labels for D3 scale names
const D3_SEQUENTIAL_LABELS: Record<D3SequentialScaleName, string> = {
  sinebow: 'Sinebow',
  viridis: 'Viridis',
  cividis: 'Cividis',
  turbo: 'Turbo',
  plasma: 'Plasma',
  inferno: 'Inferno',
  magma: 'Magma',
};

const D3_DIVERGING_LABELS: Record<D3DivergingScaleName, string> = {
  blueGold: 'Blue-Purple-Gold',
  rdBu: 'Red-Blue',
  spectral: 'Spectral',
  piYG: 'Pink-Yellow-Green',
  puOr: 'Purple-Orange',
  brBG: 'Brown-Blue-Green',
};

interface ColorScaleSelectorProps {
  colorScaleType: ColorScaleType;
  onColorScaleTypeChange: (type: ColorScaleType) => void;
  monochromeColor?: string;
  onMonochromeColorChange?: (color: string) => void;
  sequentialScaleName?: SequentialScaleName;
  onSequentialScaleNameChange?: (name: SequentialScaleName) => void;
  divergingScaleName?: DivergingScaleName;
  onDivergingScaleNameChange?: (name: DivergingScaleName) => void;
  categoricalPalette?: string;
  onCategoricalPaletteChange?: (palette: string | undefined) => void;
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

/**
 * Renders a gradient preview from a Crameri colormap (from cache).
 * Shows a shimmer placeholder if the colormap is not yet loaded.
 */
function CrameriGradientPreview({ name }: { name: string }) {
  const gradient = isCrameriLoaded(name) ? crameriGradientCSS(name, 20) : null;

  if (!gradient) {
    return (
      <div className="h-2 w-full rounded-sm bg-muted animate-pulse" />
    );
  }

  return (
    <div
      className="h-2 w-full rounded-sm"
      style={{ background: gradient }}
    />
  );
}

/**
 * Renders a swatch preview from a Crameri categorical colormap.
 */
function CrameriSwatchPreview({ name }: { name: string }) {
  const colors = getCrameriColors(name);
  if (!colors) {
    return <div className="flex gap-0.5">{Array.from({ length: 8 }, (_, i) => (
      <div key={i} className="h-3 w-3 rounded-sm bg-muted animate-pulse" />
    ))}</div>;
  }
  // Show first 8 colors as swatches
  return (
    <div className="flex gap-0.5">
      {colors.slice(0, 8).map((c, i) => (
        <div key={i} className="h-3 w-3 rounded-sm" style={{ backgroundColor: c }} />
      ))}
    </div>
  );
}

/**
 * Section header for scale groups
 */
function ScaleGroupHeader({ children }: { children: React.ReactNode }) {
  return (
    <div className="text-xs font-medium text-muted-foreground uppercase tracking-wider pt-2 pb-1 border-b border-border/50">
      {children}
    </div>
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
  categoricalPalette,
  onCategoricalPaletteChange,
}: ColorScaleSelectorProps) {
  const [open, setOpen] = useState(false);
  const [localMonochromeColor, setLocalMonochromeColor] = useState(monochromeColor);
  // Counter to force re-render when Crameri colormaps finish loading
  const [, setLoadTick] = useState(0);

  // Preload Crameri colormaps when dialog opens
  useEffect(() => {
    if (!open) return;

    const toLoad: string[] = [];

    if (colorScaleType === 'sequential') {
      toLoad.push(...CRAMERI_SEQUENTIAL_NAMES);
    } else if (colorScaleType === 'diverging') {
      toLoad.push(...CRAMERI_DIVERGING_NAMES);
    } else if (colorScaleType === 'categorical') {
      toLoad.push(...CRAMERI_CATEGORICAL_NAMES);
    }

    if (toLoad.length > 0) {
      preloadCrameriColormaps(toLoad).then(() => setLoadTick(t => t + 1));
    }
  }, [open, colorScaleType]);

  const handleOpen = (isOpen: boolean) => {
    if (!isOpen) {
      setLocalMonochromeColor(monochromeColor);
    }
    setOpen(isOpen);
  };

  const handleTypeChange = useCallback((value: ColorScaleType) => {
    onColorScaleTypeChange(value);
  }, [onColorScaleTypeChange]);

  const handleColorChange = useCallback((color: string) => {
    setLocalMonochromeColor(color);
    onMonochromeColorChange?.(color);
  }, [onMonochromeColorChange]);

  const handleSequentialScaleChange = useCallback((name: SequentialScaleName) => {
    onSequentialScaleNameChange?.(name);
  }, [onSequentialScaleNameChange]);

  const handleDivergingScaleChange = useCallback((name: DivergingScaleName) => {
    onDivergingScaleNameChange?.(name);
  }, [onDivergingScaleNameChange]);

  const handleCategoricalPaletteChange = useCallback((palette: string | undefined) => {
    onCategoricalPaletteChange?.(palette);
  }, [onCategoricalPaletteChange]);

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
      <DialogContent className="sm:max-w-md max-h-[85vh] overflow-hidden flex flex-col">
        <DialogHeader>
          <DialogTitle>Color Scale</DialogTitle>
        </DialogHeader>

        <ScrollArea className="flex-1 overflow-y-auto pr-2">
          <div className="space-y-6 py-4">
            <RadioGroup
              value={colorScaleType}
              onValueChange={(value) => handleTypeChange(value as ColorScaleType)}
              className="space-y-4"
            >
              {/* ============ CATEGORICAL ============ */}
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
                {colorScaleType === 'categorical' && (
                  <div className="ml-6 mt-2 space-y-2">
                    <ScaleGroupHeader>D3 Palettes</ScaleGroupHeader>
                    <div
                      className={`flex items-center gap-3 p-2 rounded-md cursor-pointer transition-colors ${
                        !categoricalPalette ? 'bg-accent' : 'hover:bg-accent/50'
                      }`}
                      onClick={() => handleCategoricalPaletteChange(undefined)}
                    >
                      <div className="flex gap-0.5">
                        {generateCategoryColors(8).map((c, i) => (
                          <div key={i} className="h-3 w-3 rounded-sm" style={{ backgroundColor: c }} />
                        ))}
                      </div>
                      <span className="text-sm">D3 Category (default)</span>
                    </div>

                    <ScaleGroupHeader>Crameri Scientific (100 colors)</ScaleGroupHeader>
                    <ScrollArea className="max-h-48">
                      <div className="space-y-1">
                        {CRAMERI_CATEGORICAL_NAMES.map((name) => (
                          <div
                            key={name}
                            className={`flex items-center gap-3 p-2 rounded-md cursor-pointer transition-colors ${
                              categoricalPalette === name ? 'bg-accent' : 'hover:bg-accent/50'
                            }`}
                            onClick={() => handleCategoricalPaletteChange(name)}
                          >
                            <div className="w-24 shrink-0">
                              <CrameriSwatchPreview name={name} />
                            </div>
                            <span className="text-sm">{CRAMERI_CATEGORICAL_LABELS[name]}</span>
                          </div>
                        ))}
                      </div>
                    </ScrollArea>
                  </div>
                )}
              </div>

              {/* ============ SEQUENTIAL ============ */}
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
                    <ScaleGroupHeader>D3 Scales</ScaleGroupHeader>
                    {D3_SEQUENTIAL_NAMES.map((name) => (
                      <div
                        key={name}
                        className={`flex items-center gap-3 p-2 rounded-md cursor-pointer transition-colors ${
                          sequentialScaleName === name ? 'bg-accent' : 'hover:bg-accent/50'
                        }`}
                        onClick={() => handleSequentialScaleChange(name)}
                      >
                        <div className="w-24 h-2 rounded-sm overflow-hidden">
                          <ColorScalePreview type="sequential" sequentialScaleName={name} />
                        </div>
                        <span className="text-sm">{D3_SEQUENTIAL_LABELS[name]}</span>
                      </div>
                    ))}

                    <ScaleGroupHeader>Crameri Scientific</ScaleGroupHeader>
                    <ScrollArea className="max-h-64">
                      <div className="space-y-1">
                        {CRAMERI_SEQUENTIAL_NAMES.map((name) => (
                          <div
                            key={name}
                            className={`flex items-center gap-3 p-2 rounded-md cursor-pointer transition-colors ${
                              sequentialScaleName === name ? 'bg-accent' : 'hover:bg-accent/50'
                            }`}
                            onClick={() => handleSequentialScaleChange(name)}
                          >
                            <div className="w-24 h-2 rounded-sm overflow-hidden">
                              <CrameriGradientPreview name={name} />
                            </div>
                            <span className="text-sm">{CRAMERI_SEQUENTIAL_LABELS[name]}</span>
                          </div>
                        ))}
                      </div>
                    </ScrollArea>

                  </div>
                )}
              </div>

              {/* ============ DIVERGING ============ */}
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
                    <ScaleGroupHeader>D3 Scales</ScaleGroupHeader>
                    {D3_DIVERGING_NAMES.map((name) => (
                      <div
                        key={name}
                        className={`flex items-center gap-3 p-2 rounded-md cursor-pointer transition-colors ${
                          divergingScaleName === name ? 'bg-accent' : 'hover:bg-accent/50'
                        }`}
                        onClick={() => handleDivergingScaleChange(name)}
                      >
                        <div className="w-24 h-2 rounded-sm overflow-hidden">
                          <ColorScalePreview type="diverging" divergingScaleName={name} />
                        </div>
                        <span className="text-sm">{D3_DIVERGING_LABELS[name]}</span>
                      </div>
                    ))}

                    <ScaleGroupHeader>Crameri Scientific</ScaleGroupHeader>
                    <ScrollArea className="max-h-48">
                      <div className="space-y-1">
                        {CRAMERI_DIVERGING_NAMES.map((name) => (
                          <div
                            key={name}
                            className={`flex items-center gap-3 p-2 rounded-md cursor-pointer transition-colors ${
                              divergingScaleName === name ? 'bg-accent' : 'hover:bg-accent/50'
                            }`}
                            onClick={() => handleDivergingScaleChange(name)}
                          >
                            <div className="w-24 h-2 rounded-sm overflow-hidden">
                              <CrameriGradientPreview name={name} />
                            </div>
                            <span className="text-sm">{CRAMERI_DIVERGING_LABELS[name]}</span>
                          </div>
                        ))}
                      </div>
                    </ScrollArea>
                  </div>
                )}
              </div>

              {/* ============ MONOCHROME ============ */}
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
        </ScrollArea>
      </DialogContent>
    </Dialog>
  );
}
