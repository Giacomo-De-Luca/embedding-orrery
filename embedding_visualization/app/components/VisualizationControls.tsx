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
import type { ProjectionMethod, DimensionMode, DistanceMetric, VisualizationState } from '../../lib/types/types';
import type { ColorFieldOption } from '../../lib/utils/fieldAnalysis';
import { ColorScaleSelector } from './ColorScaleSelector';

interface VisualizationControlsProps {
  state: VisualizationState;
  onStateChange: (newState: Partial<VisualizationState>) => void;
  embeddingDim: number;
  metadata?: {
    pca_2d_variance?: number[];
    pca_3d_variance?: number[];
  };
  colorFieldOptions?: ColorFieldOption[];
  availableFields?: string[];
}

export function VisualizationControls({
  state,
  onStateChange,
  embeddingDim,
  metadata,
  colorFieldOptions = [],
  availableFields = [],
}: VisualizationControlsProps) {
  // Handle field selection with auto-detection of scale type
  const handleFieldChange = (value: string) => {
    if (value === 'none') {
      onStateChange({ colorByField: null, colorScaleType: 'categorical' });
      return;
    }

    const fieldOption = colorFieldOptions.find(f => f.field === value);
    if (!fieldOption) return;

    // Use the recommended scale from the field analysis
    onStateChange({ colorByField: value, colorScaleType: fieldOption.recommendedScale });
  };

  return (
    <div className="space-y-6">
        {/* Projection Method */}
        <div className="space-y-3">
          <Label className="text-base">Projection Method</Label>
          <RadioGroup
            value={state.method}
            onValueChange={(value) => onStateChange({ method: value as ProjectionMethod })}
          >
            <div className="flex items-center space-x-2">
              <RadioGroupItem value="pca" id="method-pca" />
              <Label htmlFor="method-pca" className="font-normal cursor-pointer">
                PCA (Principal Component Analysis)
              </Label>
            </div>
            {metadata?.pca_2d_variance && state.mode === '2d' && state.method === 'pca' && (
              <p className="text-xs text-muted-foreground ml-6">
                Explained variance: {(metadata.pca_2d_variance.reduce((a, b) => a + b, 0) * 100).toFixed(2)}%
              </p>
            )}
            {metadata?.pca_3d_variance && state.mode === '3d' && state.method === 'pca' && (
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
            value={state.mode}
            onValueChange={(value) => onStateChange({ mode: value as DimensionMode })}
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

        {/* Manual Dimension Selection */}
        {state.method === 'manual' && (
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
                    value={state.selectedDimensions?.[0] ?? 0}
                    onChange={(e) => {
                      const dims = state.selectedDimensions ?? [0, 1, 2];
                      dims[0] = parseInt(e.target.value);
                      onStateChange({ selectedDimensions: [...dims] });
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
                    value={state.selectedDimensions?.[1] ?? 1}
                    onChange={(e) => {
                      const dims = state.selectedDimensions ?? [0, 1, 2];
                      dims[1] = parseInt(e.target.value);
                      onStateChange({ selectedDimensions: [...dims] });
                    }}
                  />
                </div>

                {state.mode === '3d' && (
                  <div className="space-y-1.5">
                    <Label htmlFor="dim-z" className="text-xs">Dimension 3 (Z-axis)</Label>
                    <Input
                      id="dim-z"
                      type="number"
                      min={0}
                      max={embeddingDim - 1}
                      value={state.selectedDimensions?.[2] ?? 2}
                      onChange={(e) => {
                        const dims = state.selectedDimensions ?? [0, 1, 2];
                        dims[2] = parseInt(e.target.value);
                        onStateChange({ selectedDimensions: [...dims] });
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
              value={state.colorByField ?? 'none'}
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
                      ({option.uniqueCount === Infinity
                        ? option.valueType === 'string' ? '>100 categories' : 'numeric'
                        : option.recommendedScale === 'sequential'
                          ? 'numeric'
                          : `${option.uniqueCount} values`})
                    </span>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            {/* Show scale selector when a field is selected (allows override of auto-detected type) */}
            {state.colorByField && (
              <ColorScaleSelector
                colorScaleType={state.colorScaleType ?? 'categorical'}
                onColorScaleTypeChange={(type) => onStateChange({ colorScaleType: type })}
                monochromeColor={state.monochromeColor}
                onMonochromeColorChange={(color) => onStateChange({ monochromeColor: color })}
                sequentialScaleName={state.sequentialScaleName}
                onSequentialScaleNameChange={(name) => onStateChange({ sequentialScaleName: name })}
                divergingScaleName={state.divergingScaleName}
                onDivergingScaleNameChange={(name) => onStateChange({ divergingScaleName: name })}
              />
            )}
          </div>

          {/* Hide Unclustered Checkbox - only show for topic fields */}
          {state.colorByField && (
            state.colorByField === 'topic_id' ||
            state.colorByField === 'topic_label' ||
            state.colorByField === 'topic'
          ) && (
            <div className="flex items-center space-x-2 mt-2">
              <Checkbox
                id="hide-unclustered"
                checked={state.hideUnclustered ?? false}
                onCheckedChange={(checked) => onStateChange({ hideUnclustered: checked === true })}
              />
              <Label
                htmlFor="hide-unclustered"
                className="font-normal cursor-pointer text-sm"
              >
                Hide unclustered points
              </Label>
            </div>
          )}
        </div>

        <Separator />

        {/* Distance Metric */}
        <div className="space-y-3">
          <Label htmlFor="distance-metric" className="text-base">Distance Metric</Label>
          <Select
            value={state.distanceMetric ?? 'COSINE'}
            onValueChange={(value) => onStateChange({ distanceMetric: value as DistanceMetric })}
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
            checked={state.showContours ?? false}
            onCheckedChange={(checked) => onStateChange({ showContours: checked === true })}
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
                selectedFields={state.tooltipFields ?? []}
                onChange={(fields) => onStateChange({ tooltipFields: fields })}
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
