// Types for embedding visualization
// Flexible design to support any embedding data source

export interface CollectionInfo {
  name: string;
  display_name: string;
  count: number;
  embedding_dim: number;
  embedding_provider?: string;
  embedding_model?: string;
  timestamp: string;
  // Source info (optional, depends on data source)
  source_dataset?: string;
  source_file?: string;
  embedded_columns?: string[];
  has_projections?: boolean;
  // Topic info (populated when topic extraction has been run)
  has_topics?: boolean;
  topics?: TopicInfo[];
}

// ============ Topic types (shared with mutations.ts) ============

export interface TopicKeyword {
  word: string;
  score: number;
}

export interface TopicInfo {
  topicId: number;
  keywords: TopicKeyword[];
  label: string | null;
  count: number;
  subtopics?: string[] | null;
}

// ============ Filter types (for backend ChromaDB where clauses) ============

export type FilterOperator = 'EQ' | 'NE' | 'GT' | 'GTE' | 'LT' | 'LTE' | 'IN' | 'NIN';

export interface FilterInput {
  field: string;
  operator: FilterOperator;
  value: unknown;
}

export interface CollectionsManifest {
  [key: string]: CollectionInfo;
}

export interface EmbeddingMetadata {
  total_items: number;
  embedding_dim: number;
  embedding_provider?: string;
  embedding_model?: string;
  pca_2d_variance?: number[];
  pca_3d_variance?: number[];
  timestamp: string;
  // Source info
  source_dataset?: string;
  source_file?: string;
  source_split?: string;
  embedded_columns?: string;
  has_projections?: boolean;
  // Prompt info (for models like Gemma Embedding)
  embedding_prompt?: string | null;  // Single field - can be predefined name or custom string
}

export interface ProjectionData {
  pca_2d: number[][] | null;
  pca_3d: number[][] | null;
  umap_2d: number[][] | null;
  umap_3d: number[][] | null;
}

/**
 * Generic embedding data structure.
 *
 * The backend returns flexible data where:
 * - ids: unique identifiers for each item
 * - documents: the main text content (what was embedded)
 * - itemMetadata: array of arbitrary key/value objects per item
 *
 * For visualization, we extract a label and category from metadata.
 */
export interface EmbeddingData {
  metadata: EmbeddingMetadata;
  ids: string[];
  documents: string[];
  // Raw metadata from ChromaDB - each item can have different fields
  itemMetadata: Record<string, unknown>[];
  projections: ProjectionData;
  // Schema info - what fields are available across items
  availableFields: string[];
  // Display configuration (auto-detected or user-specified)
  displayConfig: DisplayConfig;
}

/**
 * Configuration for how to display data in the visualization.
 * This allows the UI to adapt to different data sources.
 */
export interface DisplayConfig {
  // Field to use as display label (e.g., "word" for WordNet, "title" for articles)
  labelField: string | null;
  // Field to use as category for coloring (e.g., "pos" for WordNet, "topic" for articles)
  categoryField: string | null;
  // Available category values (for legend)
  categoryValues: string[];
  // Human-readable name for the category (e.g., "Part of Speech", "Topic")
  categoryName: string;
}

export type ProjectionMethod = 'pca' | 'umap' | 'manual';
export type DimensionMode = '2d' | '3d';
export type DistanceMetric = 'COSINE' | 'L2' | 'IP';

/**
 * Color scale types for visualization:
 * - categorical: Discrete colors for categories (D3 category10/20)
 * - sequential: Continuous scale for numeric values (Viridis)
 * - diverging: Centered scale for values with midpoint (Blue-Gold)
 * - monochrome: Single-color opacity gradient (10% → 100%)
 */
export type ColorScaleType = 'categorical' | 'sequential' | 'diverging' | 'monochrome';

// Re-export scale name types for convenience
export type { SequentialScaleName, DivergingScaleName } from '../utils/categoryColors';

/**
 * Discriminated union for color scale configuration.
 * Each variant carries only the parameters relevant to that scale type.
 * `categoricalPalette` is intentionally separate — chart components need it
 * regardless of which scale type the scatter plot uses.
 */
export type ColorScale =
  | { type: 'categorical' }
  | { type: 'sequential'; scaleName: import('../utils/categoryColors').SequentialScaleName }
  | { type: 'diverging'; scaleName: import('../utils/categoryColors').DivergingScaleName }
  | { type: 'monochrome'; baseColor: string };

export const DEFAULT_COLOR_SCALE: ColorScale = { type: 'categorical' };

/** Build a ColorScale with sensible defaults from just a type name */
export function defaultColorScaleForType(type: ColorScaleType): ColorScale {
  switch (type) {
    case 'categorical': return { type: 'categorical' };
    case 'sequential':  return { type: 'sequential', scaleName: 'sinebow' };
    case 'diverging':   return { type: 'diverging', scaleName: 'blueGold' };
    case 'monochrome':  return { type: 'monochrome', baseColor: '#1f77b4' };
  }
}

/**
 * Custom overrides for numeric color scale range (Tableau-style).
 * Undefined fields fall back to auto-detected data range.
 */
export interface CustomNumericRange {
  min?: number;
  max?: number;
  center?: number;  // midpoint for diverging scales
  logScale?: boolean;  // when true, map values through log10 before coloring
}

/** Histogram bin for NumericRangeChart. */
export interface HistogramBin {
  binStart: number;
  binEnd: number;
  count: number;
}

export interface TemporalRange {
  field: string;         // temporal field name (e.g. "year")
  startPeriod: string;   // inclusive start (e.g. "1990")
  endPeriod: string;     // inclusive end (e.g. "2010")
  allPeriods: string[];  // sorted period values for index lookup
}

// ---- Text Search ----

export type TextSearchMode = 'CONTAINS' | 'EXACT';

export interface TextSearchConfig {
  fields: string[] | null; // null = document only (default, fastest)
  mode: TextSearchMode;
  caseSensitive: boolean;
  filters: FilterInput[];  // metadata pre-filters (empty = no filters)
}

export interface TextSearchMatch {
  id: string;
  matchedField: string;
  snippet?: string | null;
}

// VisualizationState is now managed by Zustand — see lib/stores/useVisualizationStore.ts
// The store's VisualizationStoreState interface is the canonical shape.

/**
 * Information about a field that can be used for coloring.
 */
export interface CategoryFieldOption {
  field: string;
  uniqueCount: number;
  displayName: string;
}

/**
 * Generic point interface for visualization.
 *
 * Fields:
 * - id: unique identifier (from ChromaDB id)
 * - label: display text (extracted from metadata or id)
 * - document: full text content
 * - category: optional categorical value for coloring
 * - metadata: all raw metadata for the item
 */
export interface Point3D {
  x: number;
  y: number;
  z: number;
  id: string;
  label: string;
  document: string;
  category: string;
  index: number;
  metadata: Record<string, unknown>;
}

export interface Point2D {
  x: number;
  y: number;
  id: string;
  label: string;
  document: string;
  category: string;
  index: number;
  metadata: Record<string, unknown>;
}

/**
 * Semantic search result with flexible metadata.
 */
export interface SemanticSearchResult {
  id: string;
  label: string;
  document: string;
  category: string;
  similarity: number;
  distance: number;
  metadata: Record<string, unknown>;
}

// ============ Color configuration ============

/**
 * Preset color palettes for known category types.
 * Falls back to dynamic color generation for unknown categories.
 */
export interface CategoryColorPreset {
  name: string;
  colors: Record<string, string>;
  labels?: Record<string, string>;
}

/**
 * Map from point index to similarity score (0-1).
 * Used for similarity-aware highlighting where visual emphasis
 * can vary based on the similarity score.
 *
 * - Semantic search: actual similarity from vector distance
 * - Text search: 1.0 (perfect match)
 */
export type HighlightMap = Map<number, number>;

// ============ Nested color map (topic → subtopic hierarchy) ============

/**
 * Nested color map for Tableau-style topic/subtopic coloring.
 * Topics define base hues; subtopics get lightness variations within that hue.
 */
export interface NestedColorMap {
  /** Map from subtopic label → hex color */
  subtopicColors: Record<string, string>;
  /** Map from topic label → hex color (base hue) */
  topicColors: Record<string, string>;
  /** Map from topic label → array of subtopic labels */
  hierarchy: Record<string, string[]>;
  /** Map from topic label → total point count */
  topicCounts: Record<string, number>;
  /** Map from subtopic label → point count */
  subtopicCounts: Record<string, number>;
}

// ============ SAE (Sparse Autoencoder) types ============

export interface SaeLogitEntry {
  token: string;
  score: number;
}

export interface SaeFeature {
  modelId: string;
  saeId: string;
  featureIndex: number;
  density: number | null;
  label: string | null;
  topLogits: SaeLogitEntry[] | null;
  bottomLogits: SaeLogitEntry[] | null;
}

export interface SaeActivation {
  id: string;
  tokens: string[];
  values: number[];
  maxValue: number;
  maxValueTokenIndex: number;
}

export interface SaeModelInfo {
  modelId: string;
  saeId: string;
  featureCount: number;
  activationCount: number;
}

export interface SaeFeatureSearchResult {
  feature: SaeFeature;
  activationCount: number | null;
}

export interface SaeActivationQuantileGroup {
  quantile: number;
  binMin: number;
  binMax: number;
  activations: SaeActivation[];
}

// ── Chat / Steering types ──────────────────────────────────────────

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: number;
}

export interface SteeringFeature {
  modelId: string;
  saeId: string;
  layerIndex: number;
  featureIndex: number;
  strength: number;
  label?: string;
}

export interface SteeringConfig {
  features: SteeringFeature[];
}

export type ChatStatus = 'idle' | 'generating' | 'error';

