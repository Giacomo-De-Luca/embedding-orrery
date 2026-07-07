'use client';

import React from 'react';
import { Label } from '@/lib/ui-primitives/label';
import { Input } from '@/lib/ui-primitives/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/lib/ui-primitives/select';
import { Separator } from '@/lib/ui-primitives/separator';
import { Checkbox } from '@/lib/ui-primitives/checkbox';
import { ToggleGroup, ToggleGroupItem } from '@/lib/ui-primitives/toggle-group';
import {
  Combobox,
  ComboboxChips,
  ComboboxChip,
  ComboboxChipsInput,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxItem,
  ComboboxList,
  useComboboxAnchor,
} from '@/lib/ui-primitives/combobox';
import { Slider } from '@/lib/ui-primitives/slider';
import type { ProjectionMethod, DimensionMode } from '../../lib/types/types';
import type { ColorFieldOption } from '../../lib/utils/fieldAnalysis';
import { ColorScaleSelector } from './ColorScaleSelector';
import { SaveColorDefaultButton } from './SaveColorDefaultButton';
import { CATEGORY_PRESETS } from '../../lib/utils/categoryColors';
import { useVisualizationStore } from '../../lib/stores/useVisualizationStore';
import { useShallow } from 'zustand/react/shallow';

const SECTION_HEADER = 'text-xs font-medium uppercase tracking-wider text-muted-foreground';
const SEGMENT_ITEM = 'text-xs h-7 px-3';

interface VisualizationControlsProps {
  embeddingDim: number;
  metadata?: {
    pca_2d_variance?: number[];
    pca_3d_variance?: number[];
  };
  colorFieldOptions?: ColorFieldOption[];
  availableFields?: string[];
  nestedColorAvailable?: boolean;
  /** Active collection — enables saving the colouring as its default. */
  collectionName?: string | null;
  /** Any points currently muted by search/temporal/category filters — gates the filtered-point controls. */
  hasActiveFilter?: boolean;
  /** Search highlights exist — gates the highlight-dependent controls. */
  hasHighlights?: boolean;
}

export function VisualizationControls({
  embeddingDim,
  metadata,
  colorFieldOptions = [],
  availableFields = [],
  nestedColorAvailable,
  collectionName,
  hasActiveFilter,
  hasHighlights,
}: VisualizationControlsProps) {
  const store = useVisualizationStore;
  const {
    method, mode, colorByField, selectedDimensions,
    nebulaMode, hideUnclustered, nestedColorMode, showAxes,
    showClusterLabels, showAllClusterLabels, hideFilteredPoints, mutedPointOpacity,
    pointOpacity, tooltipFields, showLabels, showOnlyHighlighted,
  } = store(useShallow((s) => ({
    method: s.method,
    mode: s.mode,
    colorByField: s.colorByField,
    selectedDimensions: s.selectedDimensions,
    nebulaMode: s.nebulaMode,
    showAxes: s.showAxes,
    hideUnclustered: s.hideUnclustered,
    nestedColorMode: s.nestedColorMode,
    showClusterLabels: s.showClusterLabels,
    showAllClusterLabels: s.showAllClusterLabels,
    hideFilteredPoints: s.hideFilteredPoints,
    mutedPointOpacity: s.mutedPointOpacity,
    pointOpacity: s.pointOpacity,
    tooltipFields: s.tooltipFields,
    showLabels: s.showLabels,
    showOnlyHighlighted: s.showOnlyHighlighted,
  })));

  // Handle field selection with auto-detection of scale type
  const handleFieldChange = (value: string) => {
    if (value === 'none') {
      store.getState().setColorByField(null);
      return;
    }

    const fieldOption = colorFieldOptions.find(f => f.field === value);
    if (!fieldOption) return;

    // Use the recommended scale from the field analysis
    store.getState().setColorByField(value, fieldOption.recommendedScale);
  };

  const showLabelsGroup = Boolean(hasHighlights || colorByField);

  return (
    <div className="space-y-6">
        {/* View: projection method + dimensions + display effects */}
        <div className="space-y-3">
          <Label className={SECTION_HEADER}>View</Label>
          <div className="flex items-center justify-between gap-2">
            <Label className="text-xs text-muted-foreground">Projection</Label>
            <ToggleGroup
              type="single"
              variant="outline"
              value={method}
              onValueChange={(v) => v && store.getState().setMethod(v as ProjectionMethod)}
            >
              <ToggleGroupItem
                value="pca"
                className={SEGMENT_ITEM}
                aria-label="PCA projection"
                title="Principal Component Analysis"
              >
                PCA
              </ToggleGroupItem>
              <ToggleGroupItem
                value="umap"
                className={SEGMENT_ITEM}
                aria-label="UMAP projection"
                title="Uniform Manifold Approximation and Projection"
              >
                UMAP
              </ToggleGroupItem>
              <ToggleGroupItem
                value="manual"
                className={SEGMENT_ITEM}
                aria-label="Manual dimension selection"
                title="Pick raw embedding dimensions as axes"
              >
                Manual
              </ToggleGroupItem>
            </ToggleGroup>
          </div>
          <div className="flex items-center justify-between gap-2">
            <Label className="text-xs text-muted-foreground">Dimensions</Label>
            <ToggleGroup
              type="single"
              variant="outline"
              value={mode}
              onValueChange={(v) => v && store.getState().setMode(v as DimensionMode)}
            >
              <ToggleGroupItem value="2d" className={SEGMENT_ITEM} aria-label="2D view">
                2D
              </ToggleGroupItem>
              <ToggleGroupItem value="3d" className={SEGMENT_ITEM} aria-label="3D view">
                3D
              </ToggleGroupItem>
            </ToggleGroup>
          </div>

          {metadata?.pca_2d_variance && mode === '2d' && method === 'pca' && (
            <p className="text-xs text-muted-foreground">
              Explained variance: {(metadata.pca_2d_variance.reduce((a, b) => a + b, 0) * 100).toFixed(2)}%
            </p>
          )}
          {metadata?.pca_3d_variance && mode === '3d' && method === 'pca' && (
            <p className="text-xs text-muted-foreground">
              Explained variance: {(metadata.pca_3d_variance.reduce((a, b) => a + b, 0) * 100).toFixed(2)}%
            </p>
          )}

          {/* Manual dimension selection (axis lines label the hand-picked dimensions) */}
          {method === 'manual' && (
            <div className="space-y-2">
              <Label className="text-xs text-muted-foreground">
                Dimensions (0–{embeddingDim - 1})
              </Label>
              <div className={mode === '3d' ? 'grid grid-cols-3 gap-2' : 'grid grid-cols-2 gap-2'}>
                {(['X', 'Y', 'Z'] as const).map((axis, i) => (
                  (i < 2 || mode === '3d') && (
                    <div key={axis} className="space-y-1">
                      <Label htmlFor={`dim-${axis.toLowerCase()}`} className="text-xs">
                        {axis}
                      </Label>
                      <Input
                        id={`dim-${axis.toLowerCase()}`}
                        type="number"
                        min={0}
                        max={embeddingDim - 1}
                        value={selectedDimensions?.[i] ?? i}
                        onChange={(e) => {
                          const dims = [...(selectedDimensions ?? [0, 1, 2])];
                          dims[i] = parseInt(e.target.value);
                          store.getState().setSelectedDimensions(dims);
                        }}
                      />
                    </div>
                  )
                ))}
              </div>
              <div className="flex items-center space-x-2 pt-1">
                <Checkbox
                  id="show-axes"
                  checked={showAxes}
                  onCheckedChange={(checked) => store.getState().setFlag('showAxes', checked === true)}
                />
                <Label htmlFor="show-axes" className="font-normal cursor-pointer text-sm">
                  Show axis lines
                </Label>
              </div>
            </div>
          )}

          {mode === '3d' && (
            <div className="flex items-center space-x-2">
              <Checkbox
                id="nebula-mode"
                checked={nebulaMode ?? false}
                onCheckedChange={(checked) => store.getState().setFlag('nebulaMode', checked === true)}
              />
              <Label htmlFor="nebula-mode" className="font-normal cursor-pointer text-sm">
                Nebula effects
              </Label>
            </div>
          )}
        </div>

        <Separator />

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
            {colorByField && <SaveColorDefaultButton collectionName={collectionName ?? null} />}
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

        {/* Labels */}
        {showLabelsGroup && (
          <>
            <Separator />
            <div className="space-y-3">
              <Label className={SECTION_HEADER}>Labels</Label>

              {hasHighlights && (
                <div className="flex items-center space-x-2">
                  <Checkbox
                    id="show-labels"
                    checked={showLabels ?? false}
                    onCheckedChange={(checked) => store.getState().setFlag('showLabels', checked === true)}
                  />
                  <Label htmlFor="show-labels" className="font-normal cursor-pointer text-sm">
                    Label search results
                  </Label>
                </div>
              )}

              {colorByField && (
                <div className="flex items-center space-x-2">
                  <Checkbox
                    id="show-cluster-labels"
                    checked={showClusterLabels ?? false}
                    onCheckedChange={(checked) => store.getState().setFlag('showClusterLabels', checked === true)}
                  />
                  <Label
                    htmlFor="show-cluster-labels"
                    className="font-normal cursor-pointer text-sm"
                  >
                    Show cluster labels
                  </Label>
                </div>
              )}
              {colorByField && showClusterLabels && (
                <div className="flex items-center space-x-2 mt-1 ml-6">
                  <Checkbox
                    id="show-all-cluster-labels"
                    checked={showAllClusterLabels ?? false}
                    onCheckedChange={(checked) => store.getState().setFlag('showAllClusterLabels', checked === true)}
                  />
                  <Label
                    htmlFor="show-all-cluster-labels"
                    className="font-normal cursor-pointer text-sm"
                  >
                    Show all labels
                  </Label>
                </div>
              )}
            </div>
          </>
        )}

        <Separator />

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

        {/* Show Contours
        <div className="flex items-center space-x-2">
          <Checkbox
            id="show-contours"
            checked={false}
            onCheckedChange={(checked) => store.getState().setFlag('showContours', checked === true)}
          />
          <Label
            htmlFor="show-contours"
            className="font-normal cursor-pointer"
          >
            Show contours
          </Label>
        </div>
        commented out at the moment until I manage to make the rust code work */}

        {/* Tooltip Fields */}
        {availableFields.length > 0 && (
          <>
            <Separator />
            <div className="space-y-3">
              <Label className={SECTION_HEADER}>Tooltip Fields</Label>
              <p className="text-xs text-muted-foreground">
                Extra metadata shown on hover (label + document always shown)
              </p>
              <TooltipFieldsCombobox
                availableFields={availableFields}
                selectedFields={tooltipFields ?? []}
                onChange={(fields) => store.getState().setTooltipFields(fields)}
              />
            </div>
          </>
        )}
    </div>
  );
}

/** Convert snake_case field names to Title Case for display */
function formatFieldLabel(field: string): string {
  return field
    .replace(/_/g, ' ')
    .replace(/\b\w/g, c => c.toUpperCase());
}

/** Multi-select combobox for choosing tooltip metadata fields */
function TooltipFieldsCombobox({
  availableFields,
  selectedFields,
  onChange,
}: {
  availableFields: string[];
  selectedFields: string[];
  onChange: (fields: string[]) => void;
}) {
  const chipsRef = useComboboxAnchor();

  return (
    <Combobox<string, true>
      multiple
      value={selectedFields}
      onValueChange={(newValue) => onChange(newValue ?? [])}
    >
      <ComboboxChips ref={chipsRef} className="min-h-9">
        {selectedFields.map((field) => (
          <ComboboxChip key={field}>
            {formatFieldLabel(field)}
          </ComboboxChip>
        ))}
        <ComboboxChipsInput placeholder="Add fields..." />
      </ComboboxChips>
      <ComboboxContent anchor={chipsRef}>
        <ComboboxList>
          {availableFields.map((field) => (
            <ComboboxItem key={field} value={field}>
              {formatFieldLabel(field)}
            </ComboboxItem>
          ))}
        </ComboboxList>
        <ComboboxEmpty>No matching fields</ComboboxEmpty>
      </ComboboxContent>
    </Combobox>
  );
}
