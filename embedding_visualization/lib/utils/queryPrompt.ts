/**
 * Query-prompt resolution shared by every text-query search (semantic point
 * search and semantic topic search).
 *
 * "Auto" (and an empty field) sends no prompt: the backend picks the
 * collection's own query prompt (a prompt name's query counterpart, e.g.
 * Retrieval-document -> Retrieval-query, or a task-prefix collection's stored
 * query prefix).
 */

/**
 * The query prompt to send.
 *
 * @param queryPromptName null, blank or 'auto' = Auto (the collection's own,
 *   resolved by the backend); anything else = an explicit prompt name or
 *   prompt text, sent as typed (a prefix's trailing space is part of it).
 */
export function resolveQueryPrompt(queryPromptName: string | null | undefined): string | null {
  if (queryPromptName === 'auto') return null;
  return queryPromptName?.trim() ? queryPromptName : null;
}

/**
 * Why a collection takes no semantic (text-query) search, or null when it
 * does. The backend embeds a text query with the provider and model the
 * collection records (`chromadb_client._embedding_function_for`) and refuses to
 * fall back to another one, so collections that record none (e.g. SAE
 * decoder-vector maps) can only be searched from a point. Keyword text search
 * is unaffected. Nothing is blocked before the collection's metadata is known.
 */
export function textQueryUnavailableReason(
  collection: { embedding_provider?: string | null; embedding_model?: string | null } | null | undefined,
): string | null {
  if (!collection || (collection.embedding_provider && collection.embedding_model)) return null;
  return 'This collection records no embedding model, so semantic (text-query) search is unavailable: click a point to find similar items.';
}
