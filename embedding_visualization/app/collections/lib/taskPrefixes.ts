/**
 * Task prefixes of the embed forms: EmbeddingGemma and Gemini Embedding 2 read
 * their task from a text prefix (`task: search result | query: ` on queries,
 * `title: none | text: ` on documents). The backend's `embeddingTaskPrefixes`
 * query serves the table (config/embedding_models.json) and the models that
 * take it. Pure (no React).
 *
 * The document prefix is editable (`title: {title} | text: `): it reaches the
 * model input only, never the stored document. The query prefix follows the
 * chosen task. Both travel in the embedding model input (`prompt`, `task`).
 */
import type { EmbeddingProvider } from '@/lib/graphql/mutations';

export interface EmbeddingTaskPrefix {
  id: string;
  label: string;
  queryPrefix: string;
  documentPrefix: string;
  symmetric: boolean;
}

export interface TaskPrefixTable {
  defaultTask: string;
  tasks: EmbeddingTaskPrefix[];
  models: { provider: EmbeddingProvider; modelName: string }[];
}

/** The tasks a model offers (null: the model takes no task prefix). */
export interface TaskPrefixInfo {
  defaultTask: string;
  tasks: EmbeddingTaskPrefix[];
}

/** The form's choice: a task (or NO_TASK) and its, possibly edited, document prefix. */
export interface TaskPrefixValue {
  taskId: string;
  documentPrefix: string;
}

/** What the embedding model input carries (`prompt`, `task`). */
export interface TaskPrefixInput {
  documentPrefix: string;
  queryPrefix: string;
}

/** Task id of "no prefix". */
export const NO_TASK = 'none';

export function taskPrefixesFor(
  table: TaskPrefixTable | null,
  provider: EmbeddingProvider,
  modelName: string,
): TaskPrefixInfo | null {
  if (!table) return null;
  const wanted = modelName.trim().toLowerCase();
  const listed = table.models.some(
    (m) => m.provider === provider && m.modelName.toLowerCase() === wanted,
  );
  return listed ? { defaultTask: table.defaultTask, tasks: table.tasks } : null;
}

/** The value of picking `taskId`: its document prefix, or no prefix for NO_TASK. */
export function taskPrefixValue(info: TaskPrefixInfo, taskId: string): TaskPrefixValue {
  if (taskId === NO_TASK) return { taskId: NO_TASK, documentPrefix: '' };
  const task = info.tasks.find((t) => t.id === taskId)
    ?? info.tasks.find((t) => t.id === info.defaultTask);
  return task ? { taskId: task.id, documentPrefix: task.documentPrefix } : { taskId: NO_TASK, documentPrefix: '' };
}

/** The value in effect: the user's choice, or the default task until there is one. */
export function effectiveTaskPrefix(
  info: TaskPrefixInfo | null,
  value: TaskPrefixValue | null,
): TaskPrefixValue | null {
  if (!info) return null;
  if (value && (value.taskId === NO_TASK || info.tasks.some((t) => t.id === value.taskId))) {
    return value;
  }
  return taskPrefixValue(info, info.defaultTask);
}

/** The query prefix of the chosen task ('' for NO_TASK). */
export function queryPrefixOf(info: TaskPrefixInfo, value: TaskPrefixValue): string {
  return info.tasks.find((t) => t.id === value.taskId)?.queryPrefix ?? '';
}

export function taskPrefixInput(
  info: TaskPrefixInfo | null,
  value: TaskPrefixValue | null,
): TaskPrefixInput | null {
  const effective = effectiveTaskPrefix(info, value);
  if (!info || !effective) return null;
  return { documentPrefix: effective.documentPrefix, queryPrefix: queryPrefixOf(info, effective) };
}

/**
 * The `{column}` placeholders of a template, with the backend's grammar
 * (`utils/text_processing.ColumnTemplate`): `{{` and `}}` are literal braces,
 * names are taken literally (`{Conc.M}`), and empty placeholders,
 * conversions (`!r`) and format specs (`:>4`) are errors.
 */
export function templateFields(template: string): { fields: string[]; error: string | null } {
  const fields: string[] = [];
  let i = 0;
  while (i < template.length) {
    const ch = template[i];
    if (ch === '{' && template[i + 1] === '{') {
      i += 2;
    } else if (ch === '{') {
      const end = template.indexOf('}', i + 1);
      const nested = template.indexOf('{', i + 1);
      if (end === -1 || (nested !== -1 && nested < end)) {
        return { fields, error: "unmatched '{': close it with '}', or write '{{' for a brace" };
      }
      const name = template.slice(i + 1, end);
      if (!name) return { fields, error: 'empty {}: name a column' };
      if (/[!:]/.test(name)) return { fields, error: `{${name}}: placeholders take a column name only` };
      if (!fields.includes(name)) fields.push(name);
      i = end + 1;
    } else if (ch === '}' && template[i + 1] === '}') {
      i += 2;
    } else if (ch === '}') {
      return { fields, error: "unmatched '}': write '}}' for a brace" };
    } else {
      i += 1;
    }
  }
  return { fields, error: null };
}

/**
 * Problems the backend would reject the job for: the text template may name
 * the embedded columns, the prefix any column.
 */
export function placeholderIssues(input: {
  template?: string;
  embeddingColumns?: string[];
  prefix?: string;
  prefixColumns?: string[];
}): string[] {
  const issues: string[] = [];
  if (input.template) {
    const { fields, error } = templateFields(input.template);
    const unknown = fields.filter((f) => !(input.embeddingColumns ?? []).includes(f));
    if (error) issues.push(`Text template: ${error}`);
    else if (unknown.length > 0) {
      issues.push(`Text template references non-selected columns: ${unknown.join(', ')}`);
    }
  }
  if (input.prefix) {
    const { fields, error } = templateFields(input.prefix);
    const unknown = fields.filter((f) => !(input.prefixColumns ?? []).includes(f));
    if (error) issues.push(`Task prefix: ${error}`);
    else if (unknown.length > 0) {
      issues.push(`Task prefix references unknown columns: ${unknown.join(', ')}`);
    }
  }
  return issues;
}

/** What the model reads for a row, as a pattern: the prefix, then the template (or the columns joined). */
export function modelInputPattern(prefix: string, template: string, embeddingColumns: string[]): string {
  return prefix + (template || embeddingColumns.map((c) => `{${c}}`).join(' '));
}
