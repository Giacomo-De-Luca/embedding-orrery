/**
 * Query-prompt resolution shared by every text-query search (semantic point
 * search and semantic topic search).
 *
 * "Auto" sends no prompt: the backend picks the collection's own query
 * prompt (a prompt name's query counterpart, e.g. Retrieval-document ->
 * Retrieval-query, or a task-prefix collection's stored query prefix).
 */

/**
 * The query prompt to send.
 *
 * @param queryPromptName null/'' = none, 'auto' = the collection's own (backend),
 *   anything else = explicit prompt name or prompt text.
 */
export function resolveQueryPrompt(queryPromptName: string | null | undefined): string | null {
  if (queryPromptName === 'auto') return null;
  return queryPromptName || null;
}
