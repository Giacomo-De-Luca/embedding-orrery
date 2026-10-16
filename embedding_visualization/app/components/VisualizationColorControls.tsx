'use client';

import { Save } from 'lucide-react';
import { useShallow } from 'zustand/react/shallow';
import { IS_DEMO, DEMO_DISABLED_MESSAGE } from '@/lib/utils/demoMode';
import { Button } from '@/lib/ui-primitives/button';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/lib/ui-primitives/tooltip';
import { Label } from '@/lib/ui-primitives/label';
import { Checkbox } from '@/lib/ui-primitives/checkbox';
import { Slider } from '@/lib/ui-primitives/slider';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/lib/ui-primitives/select';
import { useVisualizationStore } from '@/lib/stores/useVisualizationStore';
import { CATEGORY_PRESETS } from '@/lib/utils/categoryColors';
import type { ColorFieldOption } from '@/lib/utils/fieldAnalysis';
import { ColorScaleSelector } from './ColorScaleSelector';
import { SaveColorDefaultButton } from './SaveColorDefaultButton';

const SECTION_HEADER = 'text-xs font-medium uppercase tracking-wider text-muted-foreground';

/** Color section of the Controls panel: field, scale and hide/fade toggles. */
export function VisualizationColorControls({ colorFieldOptions, collectionName, nestedColorAvailable, hasMissingValues, canFadeMissing }: {
  colorFieldOptions: ColorFieldOption[]; collectionName?: string | null; nestedColorAvailable?: boolean;
  /** Some points have no value for the colour field. */
  hasMissingValues?: boolean;
  /** The view draws those points as a "Missing values" group, so their opacity can be set. */
  canFadeMissing?: boolean;
}) {
  const store = useVisualizationStore;
  const { colorByField, hideUnclustered, hideMissing, missingPointOpacity, nestedColorMode } = store(useShallow(s => ({
    colorByField: s.colorByField, hideUnclustered: s.hideUnclustered, hideMissing: s.hideMissing,
    missingPointOpacity: s.missingPointOpacity, nestedColorMode: s.nestedColorMode,
  })));
  const handleFieldChange = (field: string) => {
    if (field === 'none') store.getState().setColorByField(null);
    else {
      const option = colorFieldOptions.find(option => option.field === field);
      if (option) store.getState().setColorByField(field, option.recommendedScale);
    }
  };
  return <>
        {/* Color */}
        <div className="space-y-3">
          <Label htmlFor="color-by" className={SECTION_HEADER}>Color</Label>
          <div className="flex items-center gap-2">
            <Select
              value={colorByField ?? 'none'}
              onValueChange={handleFieldChange}
            >
              <SelectTrigger id="color-by" className="flex-1">
                <SelectValue placeholder="Select coloring" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="none">None (Single Color)</SelectItem>
                {colorFieldOptions.map((option) => (
                  <SelectItem key={option.field} value={option.field}>
                    {option.displayName}
                    <span className="ml-1 text-muted-foreground text-xs">
                      ({option.recommendedScale === 'sequential'
                        ? 'numeric'
                        : `${option.uniqueCount} values`})
                    </span>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            {/* Scale selector (override of auto-detected type) + save-as-default, when a field is selected */}
            {colorByField && <ColorScaleSelector />}
            {colorByField && (IS_DEMO ? (
              <Tooltip>
                {/* span: disabled buttons swallow the hover the tooltip needs */}
                <TooltipTrigger asChild>
                  <span tabIndex={0}>
                    <Button
                      variant="circularghost"
                      size="icon"
                      disabled
                      aria-label="Save as collection default"
                    >
                      <Save className="h-4 w-4" />
                    </Button>
                  </span>
                </TooltipTrigger>
                <TooltipContent>{DEMO_DISABLED_MESSAGE}</TooltipContent>
              </Tooltip>
            ) : (
              <SaveColorDefaultButton collectionName={collectionName ?? null} />
            ))}
          </div>

          {/* Hide Unclustered Checkbox - only show for fields with an Unclustered preset */}
          {colorByField &&
            CATEGORY_PRESETS[colorByField.toLowerCase()]?.labels &&
            Object.values(CATEGORY_PRESETS[colorByField.toLowerCase()].labels!).includes('Unclustered') && (
            <div className="flex items-center space-x-2 mt-2">
              <Checkbox
                id="hide-unclustered"
                checked={hideUnclustered ?? false}
                onCheckedChange={(checked) => store.getState().setFlag('hideUnclustered', checked === true)}
              />
              <Label
                htmlFor="hide-unclustered"
                className="font-normal cursor-pointer text-sm"
              >
                Hide unclustered points
              </Label>
            </div>
          )}

          {/* Missing values - only when some points have no value for the colour field */}
          {colorByField && hasMissingValues && (
            <div className="space-y-4 pt-2">
              {/* Opacity applies to the "Missing values" group; disabled (not removed) while
                  hidden so the checkbox below keeps its place */}
              {canFadeMissing && (
                <div className={`space-y-2 ${hideMissing ? 'opacity-50' : ''}`}>
                  <div className="flex items-center justify-between">
                    <Label className="text-sm font-normal">Missing value opacity</Label>
                    <span className="text-xs text-muted-foreground tabular-nums">
                      {Math.round(missingPointOpacity * 100)}%
                    </span>
                  </div>
                  <Slider
                    aria-label="Missing value opacity"
                    min={0}
                    max={100}
                    step={5}
                    disabled={hideMissing}
                    value={[Math.round(missingPointOpacity * 100)]}
                    onValueChange={([v]) => store.getState().setMissingPointOpacity(v / 100)}
                  />
                </div>
              )}
              <div className="flex items-center space-x-2">
                <Checkbox
                  id="hide-missing"
                  checked={hideMissing}
                  onCheckedChange={(checked) => store.getState().setFlag('hideMissing', checked === true)}
                />
                <Label
                  htmlFor="hide-missing"
                  className="font-normal cursor-pointer text-sm"
                >
                  Hide missing values
                </Label>
              </div>
            </div>
          )}

          {/* Nested subtopic coloring - only when topic_label is selected and subtopics exist */}
          {nestedColorAvailable && colorByField === 'topic_label' && (
            <div className="flex items-center space-x-2 mt-2">
              <Checkbox
                id="nested-color-mode"
                checked={nestedColorMode ?? false}
                onCheckedChange={(checked) => store.getState().setNestedColorMode(checked === true)}
              />
              <Label
                htmlFor="nested-color-mode"
                className="font-normal cursor-pointer text-sm"
              >
                Color by subtopics
              </Label>
            </div>
          )}
        </div>

  </>;
}
