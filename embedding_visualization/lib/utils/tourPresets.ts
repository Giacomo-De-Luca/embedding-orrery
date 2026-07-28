import type { ColorScale, ProjectionMethod, DimensionMode } from '../types/types';

/**
 * Curated view presets for the demo (`?preset=<id>`), plus the initial-state
 * seeding helpers the Explore page uses to apply them race-free.
 *
 * Presets expand entirely client-side: a preset link carries only the id, and
 * the definitions here translate it into a collection, a colour scheme (fed
 * through the same initial-refs path as explicit URL colour params) and a set
 * of store flags. Precedence: explicit URL param > preset > collection
 * default > persisted preferences.
 */

/**
 * Bare-URL default collection in demo builds: the flagship EMNLP research-
 * topics map (same collection the guided tour opens on). NOTE the trade-off,
 * accepted deliberately: it is Gemini-embedded, so manual header searches on
 * a fresh visit now spend metered quota — the previous default (`emotion`)
 * was chosen precisely because its MiniLM search runs free inside the Space.
 */
export const DEMO_DEFAULT_COLLECTION = 'acl_abstracts_emnlp_findings';

/**
 * The collection whose search model (MiniLM) runs inside the Space — the
 * warm-up target. Was `DEMO_DEFAULT_COLLECTION` until that default moved to
 * the Gemini-embedded EMNLP map: a warm-up "query" against Gemini would spend
 * metered quota on every dialog open.
 */
export const SPACE_LOCAL_SEARCH_COLLECTION = 'emotion';

/** The preset the guided tour applies — the flagship research-topics view. */
export const TOUR_PRESET_ID = 'emnlp-topics';

/**
 * The query the tour's search step runs against the tour collection (EMNLP,
 * Gemini-embedded): a DELIBERATE cost of one Gemini embed call per tour run.
 * The step's guard keeps auto-search restricted to `TOUR_COLLECTION` so no
 * other Gemini collection is ever queried without user intent.
 */
export const TOUR_SEARCH_QUERY = 'hallucination in summarization';

/**
 * The label query the tour's SAE feature-search step resolves against the
 * tour collection's linked SAE (gemma-3-1b L22). Unlike the semantic step
 * this is FREE — feature-label match and document ranking are both DuckDB
 * queries over stored activations; no embedding call, no model in the loop.
 * "humor" resolves to the model's "humor and jokes" feature, whose
 * top-ranked EMNLP abstracts are the computational-humor papers.
 */
export const TOUR_FEATURE_QUERY = 'humor';

/**
 * The WordNet tour's search step query (the paper's Figure-1 example). The
 * collection is MiniLM-embedded — the exact model the demo Space bakes into
 * its image — so this search runs locally and free, safe to fire per tour.
 */
export const WORDNET_TOUR_QUERY = 'geometry';

/** Flags a preset may set (subset of the store's boolean toggles). */
export type PresetFlagName =
  | 'nebulaMode'
  | 'showClusterLabels'
  | 'showAllClusterLabels'
  | 'showLabels'
  | 'densityMode';

export interface PresetDefinition {
  id: string;
  collection: string;
  /** Button copy in the welcome dialog. */
  label: string;
  description: string;
  color?: { colorBy: string; scale?: ColorScale; palette?: string };
  method?: ProjectionMethod;
  mode?: DimensionMode;
  flags?: Partial<Record<PresetFlagName, boolean>>;
}

export const TOUR_PRESETS: Record<string, PresetDefinition> = {
  'emnlp-topics': {
    id: 'emnlp-topics',
    collection: 'acl_abstracts_emnlp_findings',
    label: 'Explore research topics',
    description: '14k EMNLP abstracts clustered into 60 LLM-labeled research topics.',
    color: { colorBy: 'topic_label' },
    method: 'umap',
    mode: '3d',
    // densityMode is persisted — pin it off so the curated 3D view is
    // deterministic (the tour's 2D density step re-enables it explicitly).
    flags: { nebulaMode: true, showClusterLabels: true, densityMode: false },
  },
  'xkcd-manifold': {
    id: 'xkcd-manifold',
    collection: 'xkcd_hilbert_gemini',
    label: 'Explore the color manifold',
    description: 'Color names embedded as text — the rainbow re-emerges from language alone.',
    color: {
      colorBy: 'mapped_colour',
      scale: { type: 'sequential', scaleName: 'xkcdColor' },
    },
    method: 'umap',
    mode: '3d',
    // Explicit false: the manifold reads as one continuous gradient — haze,
    // cluster labels, and density contours from a previous view would only
    // obscure it.
    flags: { nebulaMode: false, showClusterLabels: false, densityMode: false },
  },
  emotion: {
    id: 'emotion',
    collection: 'emotion',
    label: 'Emotion-labeled tweets',
    description: 'Tweets clustered into LLM-labeled topics; search runs inside the Space.',
    color: { colorBy: 'topic_label' },
    method: 'umap',
    mode: '3d',
    // The tour's finale lands here — densityMode:false undoes its 2D step.
    flags: { nebulaMode: true, showClusterLabels: true, densityMode: false },
  },
  'sae-map': {
    id: 'sae-map',
    collection: 'Gemma_9_16k_embedded',
    label: 'Map the model’s features',
    description:
      '16,384 SAE feature labels from gemma-3-4b-it embedded as their own map — right-click any point to inspect it.',
    // No colour block: the label map ships without topic extraction (LLM
    // topics are the planned 4b-pt upgrade), so the collection default rules.
    method: 'umap',
    mode: '3d',
    flags: { nebulaMode: false, showClusterLabels: false, densityMode: false },
  },
  'wordnet-pos': {
    id: 'wordnet-pos',
    collection: 'wordnet_senses_full',
    label: 'The whole dictionary',
    description:
      'All 212,478 WordNet senses in one map, colored by part of speech — the demo’s heaviest view.',
    // POS is the deep-link default; the WordNet tour deliberately arrives
    // uncoloured (its first step nulls the field) and reveals POS in step 2.
    color: { colorBy: 'pos' },
    method: 'umap',
    mode: '3d',
    // The tour's nebula step flips nebulaMode on; pinned off here so the
    // static preset (and a tour restart) always opens on the plain field.
    flags: { nebulaMode: false, showClusterLabels: false, densityMode: false },
  },
  'glasgow-norms': {
    id: 'glasgow-norms',
    collection: 'Glasgow_norm_all-gemini-2',
    label: 'Psycholinguistic Probes',
    description:
      '4,682 words rated by people on nine psychological dimensions — with probes that read the ratings back out of the embedding.',
    // Explicit scale: without it the recommended-scale path lands on the
    // sinebow (rainbow) default; managua is this collection's curated look
    // (it is also its saved default, which colors by imageability instead).
    color: { colorBy: 'concreteness', scale: { type: 'diverging', scaleName: 'managua' } },
    method: 'umap',
    mode: '3d',
    flags: { nebulaMode: false, showClusterLabels: false, densityMode: false },
  },
};

/** The "Inspect SAE" tour opens on this preset (the SAE label map). */
export const SAE_MAP_PRESET_ID = 'sae-map';
export const SAE_MAP_COLLECTION = TOUR_PRESETS[SAE_MAP_PRESET_ID].collection;

/** The WordNet nebula tour runs entirely on this preset's collection. */
export const WORDNET_PRESET_ID = 'wordnet-pos';
export const WORDNET_COLLECTION = TOUR_PRESETS[WORDNET_PRESET_ID].collection;

/** The probing tour runs entirely on this preset's collection. */
export const PROBE_PRESET_ID = 'glasgow-norms';
export const PROBE_COLLECTION = TOUR_PRESETS[PROBE_PRESET_ID].collection;

/** The collection the tour lands on; its search step may only query this. */
export const TOUR_COLLECTION = TOUR_PRESETS[TOUR_PRESET_ID].collection;

export function getPreset(id: string | null | undefined): PresetDefinition | null {
  if (!id) return null;
  return TOUR_PRESETS[id] ?? null;
}

/**
 * What to seed the Explore page's initial colour refs with at first render.
 * The colour block is atomic: any explicit URL `colorBy` wins outright and the
 * preset's colour is ignored entirely (a URL field must never be mixed with a
 * preset's scale). Without a preset colour block, everything stays null and
 * the collection default applies as before.
 */
export function seedInitialColorState(args: {
  urlColorBy: string | null;
  urlScale: ColorScale | null;
  urlPalette: string | null;
  preset: PresetDefinition | null;
}): { colorBy: string | null; scale: ColorScale | null; palette: string | null } {
  if (args.urlColorBy) {
    return { colorBy: args.urlColorBy, scale: args.urlScale, palette: args.urlPalette };
  }
  if (args.preset?.color) {
    return {
      colorBy: args.preset.color.colorBy,
      scale: args.preset.color.scale ?? null,
      palette: args.preset.color.palette ?? null,
    };
  }
  return { colorBy: null, scale: null, palette: null };
}

/**
 * Initial collection precedence: URL > preset > demo default (demo builds
 * only) > first manifest key. Candidates missing from the manifest are
 * skipped, so a stale link degrades gracefully.
 */
export function resolveInitialCollection(args: {
  urlCollection: string | null;
  presetCollection: string | null;
  isDemo: boolean;
  manifestKeys: string[];
}): string | null {
  const candidates = [
    args.urlCollection,
    args.presetCollection,
    args.isDemo ? DEMO_DEFAULT_COLLECTION : null,
    args.manifestKeys[0] ?? null,
  ];
  for (const candidate of candidates) {
    if (candidate && args.manifestKeys.includes(candidate)) return candidate;
  }
  return null;
}

/**
 * The store mutations a preset implies, as data — executed by callers against
 * `useVisualizationStore.getState()` (kept pure here so it's unit-testable).
 */
export type StoreOp =
  | { kind: 'method'; value: ProjectionMethod }
  | { kind: 'mode'; value: DimensionMode }
  | { kind: 'flag'; flag: PresetFlagName; value: boolean };

export function presetStoreOps(preset: PresetDefinition): StoreOp[] {
  const ops: StoreOp[] = [];
  if (preset.method) ops.push({ kind: 'method', value: preset.method });
  if (preset.mode) ops.push({ kind: 'mode', value: preset.mode });
  for (const [flag, value] of Object.entries(preset.flags ?? {})) {
    ops.push({ kind: 'flag', flag: flag as PresetFlagName, value: value as boolean });
  }
  return ops;
}
