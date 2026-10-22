import { describe, it, expect } from 'vitest';
import {
  NO_TASK,
  effectiveTaskPrefix,
  modelInputPattern,
  placeholderIssues,
  queryPrefixOf,
  taskPrefixValue,
  taskPrefixesFor,
  templateFields,
  type TaskPrefixTable,
} from '../taskPrefixes';

const SEARCH_QUERY = 'task: search result | query: ';
const CLUSTERING = 'task: clustering | query: ';

const TABLE: TaskPrefixTable = {
  defaultTask: 'search',
  tasks: [
    { id: 'search', label: 'Search', queryPrefix: SEARCH_QUERY, documentPrefix: 'title: none | text: ', symmetric: false },
    { id: 'clustering', label: 'Clustering', queryPrefix: CLUSTERING, documentPrefix: CLUSTERING, symmetric: true },
  ],
  models: [
    { provider: 'SENTENCE_TRANSFORMERS', modelName: 'google/embeddinggemma-2' },
    { provider: 'GEMINI', modelName: 'gemini-embedding-2' },
  ],
};

describe('taskPrefixesFor', () => {
  it('matches a listed model ignoring case and surrounding spaces', () => {
    const info = taskPrefixesFor(TABLE, 'SENTENCE_TRANSFORMERS', ' Google/EmbeddingGemma-2 ');
    expect(info?.defaultTask).toBe('search');
    expect(info?.tasks.map((t) => t.id)).toEqual(['search', 'clustering']);
  });

  it('needs the provider to match too', () => {
    expect(taskPrefixesFor(TABLE, 'OPENAI', 'gemini-embedding-2')).toBeNull();
  });

  it('is null for other models and while the table is loading', () => {
    expect(taskPrefixesFor(TABLE, 'SENTENCE_TRANSFORMERS', 'all-MiniLM-L6-v2')).toBeNull();
    expect(taskPrefixesFor(null, 'GEMINI', 'gemini-embedding-2')).toBeNull();
  });
});

describe('task prefix values', () => {
  const info = taskPrefixesFor(TABLE, 'GEMINI', 'gemini-embedding-2')!;

  it('uses the default task until one is picked', () => {
    expect(effectiveTaskPrefix(info, null)).toEqual({
      taskId: 'search',
      documentPrefix: 'title: none | text: ',
    });
  });

  it('is null when the model takes no task prefix', () => {
    expect(effectiveTaskPrefix(null, { taskId: 'search', documentPrefix: 'x' })).toBeNull();
  });

  it('falls back to the default when the picked task is not offered', () => {
    expect(effectiveTaskPrefix(info, { taskId: 'gone', documentPrefix: 'x' })?.taskId).toBe('search');
  });

  it('keeps an edited document prefix', () => {
    const edited = { taskId: 'search', documentPrefix: 'title: {title} | text: ' };
    expect(effectiveTaskPrefix(info, edited)).toBe(edited);
  });

  it('picking a task fills in its document prefix', () => {
    expect(taskPrefixValue(info, 'clustering')).toEqual({ taskId: 'clustering', documentPrefix: CLUSTERING });
  });

  it('None clears both prefixes', () => {
    const none = taskPrefixValue(info, NO_TASK);
    expect(none).toEqual({ taskId: NO_TASK, documentPrefix: '' });
    expect(queryPrefixOf(info, none)).toBe('');
  });

  it('the query prefix follows the task, not the edited document prefix', () => {
    expect(queryPrefixOf(info, { taskId: 'search', documentPrefix: 'title: {title} | text: ' })).toBe(SEARCH_QUERY);
    expect(queryPrefixOf(info, { taskId: 'clustering', documentPrefix: CLUSTERING })).toBe(CLUSTERING);
  });
});

describe('templateFields', () => {
  it('lists placeholders in order without repeats', () => {
    expect(templateFields('{title}: {abstract} ({title})')).toEqual({ fields: ['title', 'abstract'], error: null });
  });

  it('reads column names literally', () => {
    expect(templateFields('{Conc.M} / {word}').fields).toEqual(['Conc.M', 'word']);
  });

  it('treats doubled braces as text', () => {
    expect(templateFields('{{x}} {a}')).toEqual({ fields: ['a'], error: null });
  });

  it.each(['{}', '{a', 'a}', '{a!r}', '{a:>4}', '{{a}'])('rejects %s like the backend', (template) => {
    expect(templateFields(template).error).not.toBeNull();
  });
});

describe('placeholderIssues', () => {
  it('flags template placeholders outside the embedded columns', () => {
    expect(placeholderIssues({ template: '{abstract} ({year})', embeddingColumns: ['abstract'] })).toEqual([
      'Text template references non-selected columns: year',
    ]);
  });

  it('lets the prefix read any column', () => {
    expect(
      placeholderIssues({
        template: '{abstract}',
        embeddingColumns: ['abstract'],
        prefix: 'title: {title} | text: ',
        prefixColumns: ['title', 'abstract'],
      }),
    ).toEqual([]);
  });

  it('flags prefix placeholders that name no column', () => {
    expect(
      placeholderIssues({ prefix: 'title: {headline} | text: ', prefixColumns: ['title'] }),
    ).toEqual(['Task prefix references unknown columns: headline']);
  });

  it('reports malformed templates', () => {
    const [issue] = placeholderIssues({ template: '{abstract', embeddingColumns: ['abstract'] });
    expect(issue).toMatch(/^Text template: /);
  });

  it('accepts a static prefix and an empty template', () => {
    expect(placeholderIssues({ template: '', embeddingColumns: [], prefix: CLUSTERING, prefixColumns: [] })).toEqual([]);
  });
});

describe('modelInputPattern', () => {
  it('puts the prefix before the template', () => {
    expect(modelInputPattern('title: {title} | text: ', '{abstract}', ['abstract'])).toBe(
      'title: {title} | text: {abstract}',
    );
  });

  it('without a template joins the columns with spaces, like the backend', () => {
    expect(modelInputPattern(CLUSTERING, '', ['title', 'abstract'])).toBe(`${CLUSTERING}{title} {abstract}`);
  });
});
