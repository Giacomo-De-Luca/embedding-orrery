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

/** Bare-URL default collection in demo builds (small, locally searchable). */
export const DEMO_DEFAULT_COLLECTION = 'emotion';

/** The preset the guided tour applies — the flagship research-topics view. */
export const TOUR_PRESET_ID = 'emnlp-topics';

/**
 * The query the tour's search step runs against the tour collection (EMNLP,
 * Gemini-embedded): a DELIBERATE cost of one Gemini embed call per tour run.
 * The step's guard keeps auto-search restricted to `TOUR_COLLECTION` so no
 * other Gemini collection is ever queried without user intent.
 */
export const TOUR_SEARCH_QUERY = 'hallucination in summarization';

/** Flags a preset may set (subset of the store's boolean toggles). */
export type PresetFlagName =
  | 'nebulaMode'
  | 'showClusterLabels'
  | 'showAllClusterLabels'
  | 'showLabels';

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
    flags: { nebulaMode: true, showClusterLabels: true },
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
    // Explicit false: the manifold reads as one continuous gradient — haze and
    // cluster labels from a previous view would only obscure it.
    flags: { nebulaMode: false, showClusterLabels: false },
  },
  emotion: {
    id: 'emotion',
    collection: 'emotion',
    label: 'Emotion-labeled tweets',
    description: 'Tweets clustered into LLM-labeled topics; search runs inside the Space.',
    color: { colorBy: 'topic_label' },
    method: 'umap',
    mode: '3d',
    flags: { nebulaMode: true, showClusterLabels: true },
  },
};

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
