import { create } from 'zustand';
import { persist, subscribeWithSelector } from 'zustand/middleware';
import type {
  ColorScale,
  ColorScaleType,
  CustomNumericRange,
  ProjectionMethod,
  DimensionMode,
  DistanceMetric,
  TemporalRange,
  TextSearchConfig,
  HazeRendererKind,
} from '../types/types';
import { DEFAULT_COLOR_SCALE, defaultColorScaleForType } from '../types/types';
import type { CategoryColorAlias, CategoryColorPin } from '../utils/categoryColors';
import type { TopicTreeLinkage } from '../graphql/mutations';
import { IS_DEMO } from '../utils/demoMode';

/**
 * The Explore topic slider's cut of the collection's merge tree. Previewing
 * replaces the topic fields in the browser (useTopicMergeTree); Apply saves
 * the cut through `reduceTopics`. Collection-scoped: never persisted, reset
 * on collection change and whenever the colour field changes.
 */
export interface TopicCutState {
  linkage: TopicTreeLinkage;
  /** Topics the slider shows, noise not counted; null = the saved topic count. */
  nTopics: number | null;
  previewing: boolean;
  /** Density cuts only: move noise into the topics that absorb it. */
  absorbNoise: boolean;
}

export const DEFAULT_TOPIC_CUT: TopicCutState = {
  linkage: 'average',
  nTopics: null,
  previewing: false,
  absorbNoise: false,
};

// ---------------------------------------------------------------------------
// State shape
// ---------------------------------------------------------------------------

export interface VisualizationStoreState {
  // Projection
  method: ProjectionMethod;
  mode: DimensionMode;
  selectedDimensions: number[];

  // Color
  colorByField: string | null;
  colorScale: ColorScale;
  customNumericRange: CustomNumericRange | null;
  categoricalPalette: string | undefined;
  nestedColorMode: boolean;
  /** Per-field custom color overrides. Outer key = field name, inner key = category value. */
  categoryColorOverrides: Record<string, Record<string, string>>;
  /**
   * Derived categorical fields that borrow another field's colors (a probe's
   * predicted class uses its target field's). Owned by useProbes, which
   * replaces the whole map whenever its probes or data change — so it is
   * deliberately not cleared by resetForCollectionChange. Not persisted.
   */
  categoryColorAliases: Record<string, CategoryColorAlias>;
  /**
   * Fields whose values take another value's colour (the topic-tree preview
   * pins each group to its representative base topic). Owned by
   * useTopicMergeTree, which replaces the whole map whenever its preview
   * changes; read by useCategoryColorMap and the nested colour map. Kept
   * separate from `categoryColorAliases` because useProbes replaces that map
   * wholesale. Not persisted.
   */
  categoryColorPins: Record<string, CategoryColorPin>;

  // Topic tree (Explore topic slider)
  topicCut: TopicCutState;

  // Search / filter
  searchQuery: string;
  textSearchConfig: TextSearchConfig;
  distanceMetric: DistanceMetric;

  // Visibility toggles
  showOnlyHighlighted: boolean;
  showLabels: boolean;
  showContours: boolean;
  hideUnclustered: boolean;
  /** Hide points whose colour-by field has no value. */
  hideMissing: boolean;
  showClusterLabels: boolean;
  showAllClusterLabels: boolean;
  nebulaMode: boolean;
  /** Haze implementation behind `nebulaMode` (sprites = current, density = experimental). */
  hazeRenderer: HazeRendererKind;
  /** Density haze exposure multiplier (1 = default). */
  hazeStrength: number;
  /** Density haze cluster-equalisation exponent gamma (0 = raw density, 1 = equalised). */
  hazeEqualize: number;
  densityMode: boolean;
  /** Brightness multiplier for the 2D density overlay (1 = Apple's tuning). */
  densityIntensity: number;
  showAxes: boolean;

  // Muting / filtering
  mutedCategories: string[];
  hideFilteredPoints: boolean;
  mutedPointOpacity: number;
  /** 0-1 multiplier on the opacity of points whose colour-by value is missing. */
  missingPointOpacity: number;
  pointOpacity: number;
  temporalRange: TemporalRange | null;

  // Tooltip
  tooltipFields: string[] | undefined;
}

// ---------------------------------------------------------------------------
// Actions
// ---------------------------------------------------------------------------

interface VisualizationStoreActions {
  // Projection
  setMethod: (method: ProjectionMethod) => void;
  setMode: (mode: DimensionMode) => void;
  setSelectedDimensions: (dims: number[]) => void;

  // Color
  setColorByField: (field: string | null, recommendedScaleType?: ColorScaleType) => void;
  setColorScale: (scale: ColorScale) => void;
  setCustomNumericRange: (range: CustomNumericRange | null) => void;
  setCategoricalPalette: (palette: string | undefined) => void;
  setNestedColorMode: (enabled: boolean) => void;
  setCategoryColorOverride: (field: string, category: string, color: string) => void;
  removeCategoryColorOverride: (field: string, category: string) => void;
  clearCategoryColorOverrides: (field?: string) => void;
  setCategoryColorAliases: (aliases: Record<string, CategoryColorAlias>) => void;
  setCategoryColorPins: (pins: Record<string, CategoryColorPin>) => void;

  // Topic tree
  setTopicCut: (patch: Partial<TopicCutState>) => void;

  // Search / filter
  setSearchQuery: (query: string) => void;
  setTextSearchConfig: (config: TextSearchConfig) => void;
  setDistanceMetric: (metric: DistanceMetric) => void;

  // Boolean toggles (generic setter)
  setFlag: (flag: keyof Pick<VisualizationStoreState,
    'showOnlyHighlighted' | 'showLabels' | 'showContours' |
    'hideUnclustered' | 'hideMissing' | 'showClusterLabels' | 'showAllClusterLabels' | 'nebulaMode' |
    'densityMode' | 'showAxes' | 'hideFilteredPoints'
  >, value: boolean) => void;

  // Muting / filtering
  setMutedCategories: (categories: string[]) => void;
  setDensityIntensity: (intensity: number) => void;
  setHazeRenderer: (kind: HazeRendererKind) => void;
  setHazeStrength: (strength: number) => void;
  setHazeEqualize: (gamma: number) => void;
  setMutedPointOpacity: (opacity: number) => void;
  setMissingPointOpacity: (opacity: number) => void;
  setPointOpacity: (opacity: number) => void;
  setTemporalRange: (range: TemporalRange | null) => void;

  // Tooltip
  setTooltipFields: (fields: string[]) => void;
  initTooltipFields: (fields: string[]) => void;

  // Lifecycle
  resetForCollectionChange: () => void;
}

export type VisualizationStore = VisualizationStoreState & VisualizationStoreActions;

// ---------------------------------------------------------------------------
// Store
// ---------------------------------------------------------------------------

export const useVisualizationStore = create<VisualizationStore>()(
  subscribeWithSelector(persist((set) => ({
    // ---- Initial state ----
    method: 'umap',
    mode: '3d',
    selectedDimensions: [0, 1, 2],
    colorByField: null,
    colorScale: DEFAULT_COLOR_SCALE,
    customNumericRange: null,
    categoricalPalette: undefined,
    nestedColorMode: false,
    categoryColorOverrides: {},
    categoryColorAliases: {},
    categoryColorPins: {},
    topicCut: DEFAULT_TOPIC_CUT,
    searchQuery: '',
    textSearchConfig: { fields: null, mode: 'CONTAINS', caseSensitive: false, filters: [] },
    distanceMetric: 'COSINE',
    showOnlyHighlighted: false,
    // Demo builds label search results out of the box — the glow alone doesn't
    // tell first-time visitors what matched.
    showLabels: IS_DEMO,
    showContours: false,
    hideUnclustered: false,
    hideMissing: false,
    showClusterLabels: false,
    showAllClusterLabels: false,
    nebulaMode: false,
    hazeRenderer: 'sprites',
    hazeStrength: 1.0,
    hazeEqualize: 0.5,
    densityMode: false,
    densityIntensity: 1.5,
    showAxes: false,
    mutedCategories: [],
    hideFilteredPoints: false,
    mutedPointOpacity: 0.20,
    missingPointOpacity: 1.0,
    pointOpacity: 1.0,
    temporalRange: null,
    tooltipFields: undefined,

    // ---- Actions ----
    setMethod: (method) => set({ method }),
    setMode: (mode) => set({ mode }),
    setSelectedDimensions: (dims) => set({ selectedDimensions: dims }),

    setColorByField: (field, recommendedScaleType) => set((prev) => ({
      colorByField: field,
      colorScale: field === null
        ? DEFAULT_COLOR_SCALE
        : recommendedScaleType
          ? defaultColorScaleForType(recommendedScaleType)
          : prev.colorScale,
    })),

    setColorScale: (scale) => set({ colorScale: scale }),
    setCustomNumericRange: (range) => set({ customNumericRange: range }),
    setCategoricalPalette: (palette) => set({ categoricalPalette: palette }),
    setNestedColorMode: (enabled) => set({ nestedColorMode: enabled }),

    setCategoryColorOverride: (field, category, color) => set((prev) => ({
      categoryColorOverrides: {
        ...prev.categoryColorOverrides,
        [field]: { ...prev.categoryColorOverrides[field], [category]: color },
      },
    })),
    removeCategoryColorOverride: (field, category) => set((prev) => {
      const fieldOverrides = { ...prev.categoryColorOverrides[field] };
      delete fieldOverrides[category];
      const next = { ...prev.categoryColorOverrides };
      if (Object.keys(fieldOverrides).length === 0) {
        delete next[field];
      } else {
        next[field] = fieldOverrides;
      }
      return { categoryColorOverrides: next };
    }),
    clearCategoryColorOverrides: (field) => set((prev) => {
      if (!field) return { categoryColorOverrides: {} };
      const next = { ...prev.categoryColorOverrides };
      delete next[field];
      return { categoryColorOverrides: next };
    }),
    setCategoryColorAliases: (aliases) => set({ categoryColorAliases: aliases }),
    setCategoryColorPins: (pins) => set({ categoryColorPins: pins }),

    setTopicCut: (patch) => set((prev) => ({ topicCut: { ...prev.topicCut, ...patch } })),

    setSearchQuery: (query) => set({ searchQuery: query }),
    setTextSearchConfig: (config) => set({ textSearchConfig: config }),
    setDistanceMetric: (metric) => set({ distanceMetric: metric }),

    setFlag: (flag, value) => set({ [flag]: value }),

    setMutedCategories: (categories) => set({ mutedCategories: categories }),
    setDensityIntensity: (intensity) => set({ densityIntensity: intensity }),
    setHazeRenderer: (kind) => set({ hazeRenderer: kind }),
    setHazeStrength: (strength) => set({ hazeStrength: strength }),
    setHazeEqualize: (gamma) => set({ hazeEqualize: gamma }),
    setMutedPointOpacity: (opacity) => set({ mutedPointOpacity: opacity }),
    setMissingPointOpacity: (opacity) => set({ missingPointOpacity: opacity }),
    setPointOpacity: (opacity) => set({ pointOpacity: opacity }),
    setTemporalRange: (range) => set({ temporalRange: range }),

    setTooltipFields: (fields) => set({ tooltipFields: fields }),
    initTooltipFields: (fields) => set((prev) =>
      prev.tooltipFields === undefined ? { tooltipFields: fields } : prev
    ),

    resetForCollectionChange: () => set({
      colorByField: null,
      colorScale: DEFAULT_COLOR_SCALE,
      customNumericRange: null,
      mutedCategories: [],
      categoryColorOverrides: {},
      tooltipFields: undefined,
      temporalRange: null,
      hideFilteredPoints: false,
      mutedPointOpacity: 0.20,
      missingPointOpacity: 1.0,
      textSearchConfig: { fields: null, mode: 'CONTAINS', caseSensitive: false, filters: [] },
      topicCut: DEFAULT_TOPIC_CUT,
    }),
  }), {
    name: 'viz-preferences',
    // Persist only global user preferences. Collection-scoped state (colorByField,
    // colorScale, mutes, temporalRange, tooltipFields, …) keeps its existing
    // lifecycle: URL params / per-collection defaults / resetForCollectionChange.
    partialize: (s) => ({
      method: s.method,
      mode: s.mode,
      nebulaMode: s.nebulaMode,
      hazeRenderer: s.hazeRenderer,
      hazeStrength: s.hazeStrength,
      hazeEqualize: s.hazeEqualize,
      densityMode: s.densityMode,
      densityIntensity: s.densityIntensity,
      showClusterLabels: s.showClusterLabels,
      showAllClusterLabels: s.showAllClusterLabels,
      showAxes: s.showAxes,
      pointOpacity: s.pointOpacity,
      distanceMetric: s.distanceMetric,
    }),
    // SSR renders defaults; rehydrate post-mount (Providers) to avoid hydration mismatch.
    skipHydration: true,
  }))
);

// ---------------------------------------------------------------------------
// Auto-reset subscription: clear muted categories when colorByField changes
// Replaces the useEffect in page.tsx that did the same thing.
// ---------------------------------------------------------------------------
useVisualizationStore.subscribe(
  (s) => s.colorByField,
  () => {
    useVisualizationStore.setState((s) => ({
      mutedCategories: [], hideUnclustered: false, hideMissing: false, customNumericRange: null,
      // The topic-tree preview lives in the topic_label colouring (its controls do).
      topicCut: { ...s.topicCut, previewing: false, nTopics: null },
    }));
  },
);

// Mutes are category labels: entering, moving or leaving a topic-tree preview
// renames the topic categories, so mutes from before would mute the wrong
// (or no) points.
useVisualizationStore.subscribe(
  (s) => (s.topicCut.previewing ? s.topicCut : null),
  () => {
    // Only when there is something to clear: a fresh array on every slider
    // step would re-render the plots ahead of the deferred preview.
    if (useVisualizationStore.getState().mutedCategories.length > 0) {
      useVisualizationStore.setState({ mutedCategories: [] });
    }
  },
);

// ---------------------------------------------------------------------------
// Selectors (for fine-grained subscriptions via useVisualizationStore(selector))
// ---------------------------------------------------------------------------
export const selectColorByField = (s: VisualizationStore) => s.colorByField;
export const selectColorScale = (s: VisualizationStore) => s.colorScale;
export const selectCustomNumericRange = (s: VisualizationStore) => s.customNumericRange;
export const selectCategoricalPalette = (s: VisualizationStore) => s.categoricalPalette;
export const selectMode = (s: VisualizationStore) => s.mode;
export const selectMethod = (s: VisualizationStore) => s.method;
