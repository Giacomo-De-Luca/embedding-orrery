'use client';

import React from 'react';
import { RadioGroup, RadioGroupItem } from '@/lib/ui-primitives/radio-group';
import { Label } from '@/lib/ui-primitives/label';
import { Input } from '@/lib/ui-primitives/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/lib/ui-primitives/select';
import { Separator } from '@/lib/ui-primitives/separator';
import { Checkbox } from '@/lib/ui-primitives/checkbox';
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
import type { ProjectionMethod, DimensionMode, DistanceMetric } from '../../lib/types/types';
import type { ColorFieldOption } from '../../lib/utils/fieldAnalysis';
import { ColorScaleSelector } from './ColorScaleSelector';
import { SaveColorDefaultButton } from './SaveColorDefaultButton';
import { CATEGORY_PRESETS } from '../../lib/utils/categoryColors';
import { useVisualizationStore } from '../../lib/stores/useVisualizationStore';
import { useShallow } from 'zustand/react/shallow';

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
}

export function VisualizationControls({
  embeddingDim,
  metadata,
  colorFieldOptions = [],
  availableFields = [],
  nestedColorAvailable,
  collectionName,
}: VisualizationControlsProps) {
  const store = useVisualizationStore;
  const {
    method, mode, colorByField, selectedDimensions,
    nebulaMode, hideUnclustered, nestedColorMode,
    showClusterLabels, showAllClusterLabels, hideFilteredPoints, mutedPointOpacity,
    pointOpacity, distanceMetric, tooltipFields,
  } = store(useShallow((s) => ({
    method: s.method,
    mode: s.mode,
    colorByField: s.colorByField,
    selectedDimensions: s.selectedDimensions,
    nebulaMode: s.nebulaMode,
    hideUnclustered: s.hideUnclustered,
    nestedColorMode: s.nestedColorMode,
    showClusterLabels: s.showClusterLabels,
    showAllClusterLabels: s.showAllClusterLabels,
    hideFilteredPoints: s.hideFilteredPoints,
    mutedPointOpacity: s.mutedPointOpacity,
    pointOpacity: s.pointOpacity,
    distanceMetric: s.distanceMetric,
    tooltipFields: s.tooltipFields,
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

  return (
    <div className="space-y-6">
        {/* Projection Method */}
        <div className="space-y-3">
          <Label className="text-base">Projection Method</Label>
          <RadioGroup
            value={method}
            onValueChange={(value) => store.getState().setMethod(value as ProjectionMethod)}
          >
            <div className="flex items-center space-x-2">
              <RadioGroupItem value="pca" id="method-pca" />
              <Label htmlFor="method-pca" className="font-normal cursor-pointer">
                PCA (Principal Component Analysis)
              </Label>
            </div>
            {metadata?.pca_2d_variance && mode === '2d' && method === 'pca' && (
              <p className="text-xs text-muted-foreground ml-6">
                Explained variance: {(metadata.pca_2d_variance.reduce((a, b) => a + b, 0) * 100).toFixed(2)}%
              </p>
            )}
            {metadata?.pca_3d_variance && mode === '3d' && method === 'pca' && (
              <p className="text-xs text-muted-foreground ml-6">
                Explained variance: {(metadata.pca_3d_variance.reduce((a, b) => a + b, 0) * 100).toFixed(2)}%
              </p>
            )}

            <div className="flex items-center space-x-2">
              <RadioGroupItem value="umap" id="method-umap" />
              <Label htmlFor="method-umap" className="font-normal cursor-pointer">
                UMAP (Uniform Manifold Approximation)
              </Label>
            </div>

            <div className="flex items-center space-x-2">
              <RadioGroupItem value="manual" id="method-manual" />
              <Label htmlFor="method-manual" className="font-normal cursor-pointer">
                Manual Dimension Selection
              </Label>
            </div>
          </RadioGroup>
        </div>

        <Separator />

        {/* Dimension Mode */}
        <div className="space-y-3">
          <Label className="text-base">Dimensions</Label>
          <RadioGroup
            value={mode}
            onValueChange={(value) => store.getState().setMode(value as DimensionMode)}
          >
            <div className="flex items-center space-x-2">
              <RadioGroupItem value="2d" id="mode-2d" />
              <Label htmlFor="mode-2d" className="font-normal cursor-pointer">
                2D
              </Label>
            </div>

            <div className="flex items-center space-x-2">
              <RadioGroupItem value="3d" id="mode-3d" />
              <Label htmlFor="mode-3d" className="font-normal cursor-pointer">
                3D
              </Label>
            </div>
          </RadioGroup>
        </div>

        {/* Nebula Cluster Effects (3D only) */}
        {mode === '3d' && (
          <>
            <Separator />
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
          </>
        )}

        {/* Manual Dimension Selection */}
        {method === 'manual' && (
          <>
            <Separator />
            <div className="space-y-3">
              <Label className="text-base">Select Dimensions (0-{embeddingDim - 1})</Label>
              <div className="space-y-3">
                <div className="space-y-1.5">
                  <Label htmlFor="dim-x" className="text-xs">Dimension 1 (X-axis)</Label>
                  <Input
                    id="dim-x"
                    type="number"
                    min={0}
                    max={embeddingDim - 1}
                    value={selectedDimensions?.[0] ?? 0}
                    onChange={(e) => {
                      const dims = [...(selectedDimensions ?? [0, 1, 2])];
                      dims[0] = parseInt(e.target.value);
                      store.getState().setSelectedDimensions(dims);
                    }}
                  />
                </div>

                <div className="space-y-1.5">
                  <Label htmlFor="dim-y" className="text-xs">Dimension 2 (Y-axis)</Label>
                  <Input
                    id="dim-y"
                    type="number"
                    min={0}
                    max={embeddingDim - 1}
                    value={selectedDimensions?.[1] ?? 1}
                    onChange={(e) => {
                      const dims = [...(selectedDimensions ?? [0, 1, 2])];
                      dims[1] = parseInt(e.target.value);
                      store.getState().setSelectedDimensions(dims);
                    }}
                  />
                </div>

                {mode === '3d' && (
                  <div className="space-y-1.5">
                    <Label htmlFor="dim-z" className="text-xs">Dimension 3 (Z-axis)</Label>
                    <Input
                      id="dim-z"
                      type="number"
                      min={0}
                      max={embeddingDim - 1}
                      value={selectedDimensions?.[2] ?? 2}
                      onChange={(e) => {
                        const dims = [...(selectedDimensions ?? [0, 1, 2])];
                        dims[2] = parseInt(e.target.value);
                        store.getState().setSelectedDimensions(dims);
                      }}
                    />
                  </div>
                )}
              </div>
            </div>
          </>
        )}

        <Separator />

        {/* Color By */}
        <div className="space-y-3">
          <Label htmlFor="color-by" className="text-base">Color By</Label>
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
            {/* Show scale selector when a field is selected (allows override of auto-detected type) */}
            {colorByField && (
              <ColorScaleSelector />
            )}
          </div>

          {/* Persist the current colouring as this collection's default */}
          {colorByField && (
            <SaveColorDefaultButton collectionName={collectionName ?? null} />
          )}

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

          {colorByField && (
            <div className="flex items-center space-x-2 mt-2">
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
          {showClusterLabels && (
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

        <Separator />

        {/* Filtered Points */}
        <div className="space-y-3">
          <Label className="text-base">Filtered Points</Label>

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

          {!(hideFilteredPoints) && (
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label className="text-sm font-normal">Muted opacity factor</Label>
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
        </div>

        <Separator />

        {/* Distance Metric */}
        <div className="space-y-3">
          <Label htmlFor="distance-metric" className="text-base">Distance Metric</Label>
          <Select
            value={distanceMetric ?? 'COSINE'}
            onValueChange={(value) => store.getState().setDistanceMetric(value as DistanceMetric)}
          >
            <SelectTrigger id="distance-metric">
              <SelectValue placeholder="Select metric" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="COSINE">Cosine Similarity</SelectItem>
              <SelectItem value="L2">Euclidean (L2)</SelectItem>
              <SelectItem value="IP">Inner Product</SelectItem>
            </SelectContent>
          </Select>
          <p className="text-xs text-muted-foreground">
            Used for semantic search similarity calculations
          </p>
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
              <Label className="text-base">Tooltip Fields</Label>
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
