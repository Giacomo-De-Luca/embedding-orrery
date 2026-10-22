/**
 * Model-name suggestions of the embed forms, from the backend's
 * `embeddingModelCatalog` query: the catalog's models
 * (config/embedding_models.json, `suggested`) and the models existing
 * collections use (`collectionCount`; used-only entries, such as local
 * checkpoint paths, come with `suggested` false). Pure (no React).
 */
import type { EmbeddingModelCatalogEntry, EmbeddingProvider } from '../graphql/mutations';
import type { SuggestionGroup, SuggestionItem } from './suggestions';

export const SUGGESTED_GROUP = 'Suggested';
export const USED_GROUP = 'Used in your collections';

const TEXT = 'text';

export interface ModelSuggestionFilter {
  provider: EmbeddingProvider;
  /** Modalities every suggestion must embed (e.g. the selected media kinds). */
  requires?: readonly string[];
  /** Only models that embed some media (image, audio or video). */
  anyMedia?: boolean;
}

const sameName = (a: string, b: string) => a.trim().toLowerCase() === b.trim().toLowerCase();

export class ModelSuggestions {
  constructor(private readonly catalog: readonly EmbeddingModelCatalogEntry[]) {}

  /** Short facts about a model: its modalities beyond text, dimension and use. */
  static hint(entry: EmbeddingModelCatalogEntry): string {
    const parts: string[] = [];
    if (entry.modalities.some((m) => m !== TEXT)) parts.push(entry.modalities.join(' / '));
    if (entry.dimension) parts.push(`${entry.dimension}d`);
    if (entry.collectionCount > 0) {
      parts.push(`used by ${entry.collectionCount} collection${entry.collectionCount === 1 ? '' : 's'}`);
    }
    return parts.join(' · ');
  }

  static item(entry: EmbeddingModelCatalogEntry): SuggestionItem {
    const item: SuggestionItem = { value: entry.modelName };
    if (entry.description) item.description = entry.description;
    const detail = ModelSuggestions.hint(entry);
    if (detail) item.detail = detail;
    return item;
  }

  /**
   * The provider's models that embed what the filter asks for, each once
   * (names compare case-insensitively; catalog entries win over used ones).
   */
  entries(filter: ModelSuggestionFilter): EmbeddingModelCatalogEntry[] {
    const kept: EmbeddingModelCatalogEntry[] = [];
    const ordered = [...this.catalog].sort((a, b) => Number(b.suggested) - Number(a.suggested));
    for (const entry of ordered) {
      if (entry.provider !== filter.provider) continue;
      if (!entry.suggested && entry.collectionCount === 0) continue;
      if (!(filter.requires ?? []).every((m) => entry.modalities.includes(m))) continue;
      if (filter.anyMedia && !entry.modalities.some((m) => m !== TEXT)) continue;
      if (kept.some((k) => sameName(k.modelName, entry.modelName))) continue;
      kept.push(entry);
    }
    return kept;
  }

  /** "Suggested" (catalog models) then "Used in your collections" (the others); empty groups dropped. */
  groups(filter: ModelSuggestionFilter): SuggestionGroup[] {
    const entries = this.entries(filter);
    return [
      { label: SUGGESTED_GROUP, items: entries.filter((e) => e.suggested).map(ModelSuggestions.item) },
      { label: USED_GROUP, items: entries.filter((e) => !e.suggested).map(ModelSuggestions.item) },
    ].filter((group) => group.items.length > 0);
  }

  /** The catalog model listed as the default of a data type (`"multimodal"`), or null. */
  defaultFor(dataType: string): EmbeddingModelCatalogEntry | null {
    return this.catalog.find((e) => e.suggested && e.defaultFor.includes(dataType)) ?? null;
  }

  /**
   * A model's catalog entry, or null: models the catalog does not list (local
   * checkpoints) have unknown modalities, which the backend checks at load.
   */
  find(provider: EmbeddingProvider, modelName: string): EmbeddingModelCatalogEntry | null {
    return (
      this.catalog.find((e) => e.suggested && e.provider === provider && sameName(e.modelName, modelName)) ?? null
    );
  }

  /** The wanted modalities a catalog model does not embed ([] when it embeds them all, or is unknown). */
  missingModalities(provider: EmbeddingProvider, modelName: string, wanted: readonly string[]): string[] {
    const entry = this.find(provider, modelName);
    return entry ? wanted.filter((m) => !entry.modalities.includes(m)) : [];
  }
}
