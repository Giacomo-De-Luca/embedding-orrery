import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';
import type {
  ColorScale,
  ColorScaleType,
  CustomNumericRange,
  ProjectionMethod,
  DimensionMode,
  DistanceMetric,
  TemporalRange,
  TextSearchConfig,
} from '../types/types';
import { DEFAULT_COLOR_SCALE, defaultColorScaleForType } from '../types/types';

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

  // Search / filter
  searchQuery: string;
  textSearchConfig: TextSearchConfig;
  distanceMetric: DistanceMetric;

  // Visibility toggles
  showOnlyHighlighted: boolean;
  showLabels: boolean;
  showContours: boolean;
  hideUnclustered: boolean;
  showClusterLabels: boolean;
  showAllClusterLabels: boolean;
  nebulaMode: boolean;
  showAxes: boolean;

  // Muting / filtering
  mutedCategories: string[];
  hideFilteredPoints: boolean;
  mutedPointOpacity: number;
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

  // Search / filter
  setSearchQuery: (query: string) => void;
  setTextSearchConfig: (config: TextSearchConfig) => void;
  setDistanceMetric: (metric: DistanceMetric) => void;

  // Boolean toggles (generic setter)
  setFlag: (flag: keyof Pick<VisualizationStoreState,
    'showOnlyHighlighted' | 'showLabels' | 'showContours' |
    'hideUnclustered' | 'showClusterLabels' | 'showAllClusterLabels' | 'nebulaMode' |
    'showAxes' | 'hideFilteredPoints'
  >, value: boolean) => void;

  // Muting / filtering
  setMutedCategories: (categories: string[]) => void;
  setMutedPointOpacity: (opacity: number) => void;
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
  subscribeWithSelector((set) => ({
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
    searchQuery: '',
    textSearchConfig: { fields: null, mode: 'CONTAINS', caseSensitive: false, filters: [] },
    distanceMetric: 'COSINE',
    showOnlyHighlighted: false,
    showLabels: false,
    showContours: false,
    hideUnclustered: false,
    showClusterLabels: false,
    showAllClusterLabels: false,
    nebulaMode: false,
    showAxes: false,
    mutedCategories: [],
    hideFilteredPoints: false,
    mutedPointOpacity: 0.20,
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

    setSearchQuery: (query) => set({ searchQuery: query }),
    setTextSearchConfig: (config) => set({ textSearchConfig: config }),
    setDistanceMetric: (metric) => set({ distanceMetric: metric }),

    setFlag: (flag, value) => set({ [flag]: value }),

    setMutedCategories: (categories) => set({ mutedCategories: categories }),
    setMutedPointOpacity: (opacity) => set({ mutedPointOpacity: opacity }),
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
      mutedPointOpacity: 0.15,
      textSearchConfig: { fields: null, mode: 'CONTAINS', caseSensitive: false, filters: [] },
    }),
  }))
);

// ---------------------------------------------------------------------------
// Auto-reset subscription: clear muted categories when colorByField changes
// Replaces the useEffect in page.tsx that did the same thing.
// ---------------------------------------------------------------------------
useVisualizationStore.subscribe(
  (s) => s.colorByField,
  () => {
    useVisualizationStore.setState({ mutedCategories: [], hideUnclustered: false, customNumericRange: null });
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
