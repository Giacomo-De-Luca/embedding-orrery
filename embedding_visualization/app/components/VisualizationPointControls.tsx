'use client';
import { useShallow } from 'zustand/react/shallow';
import { useVisualizationStore } from '@/lib/stores/useVisualizationStore';
import { Label } from '@/lib/ui-primitives/label';
import { Checkbox } from '@/lib/ui-primitives/checkbox';
import { Slider } from '@/lib/ui-primitives/slider';
const SECTION_HEADER = 'text-xs font-medium uppercase tracking-wider text-muted-foreground';
export function VisualizationPointControls({ hasActiveFilter, hasHighlights }: { hasActiveFilter?: boolean; hasHighlights?: boolean }) {
  const store = useVisualizationStore;
  const { pointOpacity, showOnlyHighlighted, hideFilteredPoints, mutedPointOpacity } = store(useShallow(s => ({
    pointOpacity: s.pointOpacity, showOnlyHighlighted: s.showOnlyHighlighted, hideFilteredPoints: s.hideFilteredPoints, mutedPointOpacity: s.mutedPointOpacity,
  })));
  return <>
        {/* Points */}
        <div className="space-y-3">
          <Label className={SECTION_HEADER}>Points</Label>

          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label className="text-sm font-normal">Point opacity</Label>
              <span className="text-xs text-muted-foreground tabular-nums">
                {Math.round((pointOpacity ?? 1.0) * 100)}%
              </span>
            </div>
            <Slider
              min={5}
              max={100}
              step={5}
              value={[Math.round((pointOpacity ?? 1.0) * 100)]}
              onValueChange={([v]) => store.getState().setPointOpacity(v / 100)}
            />
          </div>

          {hasHighlights && (
            <div className="flex items-center space-x-2">
              <Checkbox
                id="show-only-highlighted"
                checked={showOnlyHighlighted ?? false}
                onCheckedChange={(checked) => store.getState().setFlag('showOnlyHighlighted', checked === true)}
              />
              <Label htmlFor="show-only-highlighted" className="font-normal cursor-pointer text-sm">
                Show only highlighted
              </Label>
            </div>
          )}

          {hasActiveFilter && (
            <div className="flex items-center space-x-2">
              <Checkbox
                id="hide-filtered"
                checked={hideFilteredPoints ?? false}
                onCheckedChange={(checked) => store.getState().setFlag('hideFilteredPoints', checked === true)}
              />
              <Label htmlFor="hide-filtered" className="font-normal cursor-pointer text-sm">
                Hide filtered points
              </Label>
            </div>
          )}

          {hasActiveFilter && !hideFilteredPoints && (
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label className="text-sm font-normal">Filtered point opacity</Label>
                <span className="text-xs text-muted-foreground tabular-nums">
                  {Math.round((mutedPointOpacity ?? 0.20) * 100)}%
                </span>
              </div>
              <Slider
                min={0}
                max={100}
                step={5}
                value={[Math.round((mutedPointOpacity ?? 0.20) * 100)]}
                onValueChange={([v]) => store.getState().setMutedPointOpacity(v / 100)}
              />
            </div>
          )}
        </div>

  </>;
}
