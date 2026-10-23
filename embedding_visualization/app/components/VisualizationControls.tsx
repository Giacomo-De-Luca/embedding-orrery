'use client';

import React from 'react';
import { Label } from '@/lib/ui-primitives/label';
import { dotControlIndentClassName } from '@/lib/ui-primitives/dot-control';
import { Input } from '@/lib/ui-primitives/input';
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
import type { ProjectionMethod, DimensionMode, HazeRendererKind } from '../../lib/types/types';
import type { ColorFieldOption } from '../../lib/utils/fieldAnalysis';
import type { UseTopicMergeTreeReturn } from '../../lib/hooks/useTopicMergeTree';
import { VisualizationColorControls } from './VisualizationColorControls';
import { VisualizationPointControls } from './VisualizationPointControls';
import { SliderRow } from './SliderRow';
import { supportedHazeRenderer } from '../../lib/utils/densityHaze/support';
import { DensityRenderer } from '../../lib/utils/densityRenderer/DensityRenderer';
import { useFloatRenderTargets } from '../../lib/hooks/useFloatRenderTargets';
import { useVisualizationStore } from '../../lib/stores/useVisualizationStore';
import { useShallow } from 'zustand/react/shallow';

const SECTION_HEADER = 'text-xs font-medium uppercase tracking-wider text-muted-foreground';
const SEGMENT_ITEM = 'text-xs h-7 px-3';
const NO_FLOAT_TARGETS_HINT = 'Needs WebGL2 float render targets, unavailable in this browser';

interface VisualizationControlsProps {
  embeddingDim: number;
  metadata?: {
    pca_2d_variance?: number[];
    pca_3d_variance?: number[];
  };
  colorFieldOptions?: ColorFieldOption[];
  availableFields?: string[];
  nestedColorAvailable?: boolean;
  /** Some points have no value for the colour field — gates the missing-value controls. */
  hasMissingValues?: boolean;
  /** Those points are drawn as a category group — gates the missing-value opacity slider. */
  canFadeMissing?: boolean;
  /** Active collection — enables saving the colouring as its default. */
  collectionName?: string | null;
  /** Any points currently muted by search/temporal/category filters — gates the filtered-point controls. */
  hasActiveFilter?: boolean;
  /** Search highlights exist — gates the highlight-dependent controls. */
  hasHighlights?: boolean;
  /** The topic slider (Color section, while colouring by topic). */
  topicTree?: UseTopicMergeTreeReturn | null;
}

export function VisualizationControls({
  embeddingDim,
  metadata,
  colorFieldOptions = [],
  availableFields = [],
  nestedColorAvailable,
  hasMissingValues,
  canFadeMissing,
  collectionName,
  hasActiveFilter,
  hasHighlights,
  topicTree,
}: VisualizationControlsProps) {
  const store = useVisualizationStore;
  // Density haze and the 2D density overlay both need float render targets
  // (false during SSR). Reset a persisted 'density' haze or density overlay
  // preference this browser can't draw, so no dead controls are shown here.
  const floatTargetsSupported = useFloatRenderTargets();
  React.useEffect(() => {
    const { hazeRenderer, setHazeRenderer, densityMode: densityOn, setFlag } = store.getState();
    const runnable = supportedHazeRenderer(hazeRenderer);
    if (runnable !== hazeRenderer) setHazeRenderer(runnable);
    if (densityOn && !DensityRenderer.isSupported()) setFlag('densityMode', false);
  }, [store]);
  const {
    method, mode, colorByField, selectedDimensions,
    nebulaMode, hazeRenderer, hazeStrength, hazeEqualize, densityMode, densityIntensity, showAxes,
    showClusterLabels, showAllClusterLabels,
    tooltipFields, showLabels,
  } = store(useShallow((s) => ({
    method: s.method,
    mode: s.mode,
    colorByField: s.colorByField,
    selectedDimensions: s.selectedDimensions,
    nebulaMode: s.nebulaMode,
    hazeRenderer: s.hazeRenderer,
    hazeStrength: s.hazeStrength,
    hazeEqualize: s.hazeEqualize,
    densityMode: s.densityMode,
    densityIntensity: s.densityIntensity,
    showAxes: s.showAxes,
    showClusterLabels: s.showClusterLabels,
    showAllClusterLabels: s.showAllClusterLabels,
    tooltipFields: s.tooltipFields,
    showLabels: s.showLabels,
  })));

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
            <div className="space-y-2">
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
              {nebulaMode && (
                <div className={`space-y-2 ${dotControlIndentClassName}`}>
                  <div className="flex items-center justify-between gap-2">
                    <Label className="text-xs text-muted-foreground">Haze renderer</Label>
                    <ToggleGroup
                      type="single"
                      variant="outline"
                      value={hazeRenderer}
                      onValueChange={(v) => v && store.getState().setHazeRenderer(v as HazeRendererKind)}
                    >
                      <ToggleGroupItem
                        value="sprites"
                        className={SEGMENT_ITEM}
                        aria-label="Sprite haze"
                        title="Feathered sprites at sampled cluster points"
                      >
                        Sprites
                      </ToggleGroupItem>
                      <ToggleGroupItem
                        value="density"
                        className={SEGMENT_ITEM}
                        aria-label="Density haze"
                        disabled={!floatTargetsSupported}
                        title={floatTargetsSupported
                          ? 'Per-point density splats (experimental)'
                          : NO_FLOAT_TARGETS_HINT}
                      >
                        Density
                      </ToggleGroupItem>
                    </ToggleGroup>
                  </div>
                  {hazeRenderer === 'density' && (
                    <>
                      <SliderRow
                        label="Haze strength"
                        valueLabel={`${hazeStrength.toFixed(2)}×`}
                        min={25}
                        max={300}
                        step={25}
                        value={Math.round(hazeStrength * 100)}
                        onChange={(v) => store.getState().setHazeStrength(v / 100)}
                      />
                      <SliderRow
                        label="Equalise clusters"
                        title="0 = raw density, 1 = every cluster equally bright"
                        valueLabel={`γ ${hazeEqualize.toFixed(1)}`}
                        min={0}
                        max={100}
                        step={10}
                        value={Math.round(hazeEqualize * 100)}
                        onChange={(v) => store.getState().setHazeEqualize(v / 100)}
                      />
                    </>
                  )}
                </div>
              )}
            </div>
          )}

          {mode === '2d' && (
            <div className="space-y-2">
              <div
                className="flex items-center space-x-2"
                title={floatTargetsSupported ? undefined : NO_FLOAT_TARGETS_HINT}
              >
                <Checkbox
                  id="density-mode"
                  checked={densityMode ?? false}
                  disabled={!floatTargetsSupported}
                  onCheckedChange={(checked) => store.getState().setFlag('densityMode', checked === true)}
                />
                <Label htmlFor="density-mode" className="font-normal cursor-pointer text-sm">
                  Density overlay
                </Label>
              </div>
              {densityMode && (
                <div className={`space-y-2 ${dotControlIndentClassName}`}>
                  <div className="flex items-center justify-between">
                    <Label className="text-sm font-normal">Density intensity</Label>
                    <span className="text-xs text-muted-foreground tabular-nums">
                      {(densityIntensity ?? 1.5).toFixed(2)}×
                    </span>
                  </div>
                  <Slider
                    min={25}
                    max={400}
                    step={25}
                    value={[Math.round((densityIntensity ?? 1.5) * 100)]}
                    onValueChange={([v]) => store.getState().setDensityIntensity(v / 100)}
                  />
                </div>
              )}
            </div>
          )}
        </div>

        <Separator />

        <VisualizationColorControls colorFieldOptions={colorFieldOptions} collectionName={collectionName} nestedColorAvailable={nestedColorAvailable} hasMissingValues={hasMissingValues} canFadeMissing={canFadeMissing} topicTree={topicTree} />

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
                <div className={`flex items-center space-x-2 mt-1 ${dotControlIndentClassName}`}>
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

        <VisualizationPointControls hasHighlights={hasHighlights} hasActiveFilter={hasActiveFilter} />

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
