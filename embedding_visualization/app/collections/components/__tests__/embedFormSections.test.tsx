import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it, vi } from 'vitest';

import type { EmbeddingModelInput } from '@/lib/graphql/mutations';
import { EMBEDDING_PROVIDERS } from '@/lib/utils/embeddingProviders';
import { ColumnSelector } from '../ColumnSelector';
import { EmbeddingModelForm } from '../EmbeddingModelForm';
import { MediaSettingsForm } from '../MediaSettingsForm';
import { SaeActivationsSection } from '../SaeActivationsSection';
import { TopicExtractionSection } from '../TopicExtractionSection';
import { DEFAULT_TOPIC_CONFIG } from '../../lib/topicConfigInput';
import { STATIC_TASK_PREFIX_ROW, type EmbeddingTaskPrefix, type TaskPrefixRow } from '../../lib/taskPrefixes';
import type { EmbeddingModelState } from '../../lib/useEmbeddingModelState';

// The model form's two queries: an empty catalog and no length limit.
vi.mock('@/lib/hooks/useEmbeddingModelCatalog', async () => {
  const { ModelSuggestions } = await import('@/lib/utils/modelSuggestions');
  return {
    useEmbeddingModelCatalog: () => ({
      suggestions: new ModelSuggestions([]),
      loading: false,
      error: null,
      refetch: () => {},
    }),
  };
});
vi.mock('@/app/collections/lib/useModelMaxLength', () => ({
  useModelMaxLength: () => ({ limit: null, loading: false }),
}));
// The topic settings themselves are TopicConfigForm's business.
vi.mock('../TopicConfigForm', () => ({ TopicConfigForm: () => 'topic-settings-stub' }));
// SaeActivationsSection's SAE list: none loaded.
vi.mock('@apollo/client/react', () => ({ useQuery: () => ({ data: undefined }) }));

const TASKS: EmbeddingTaskPrefix[] = [
  { id: 'search', label: 'Search', queryPrefix: 'task: search result | query: ', documentPrefix: 'title: none | text: ', symmetric: false },
];

const noop = () => {};

function modelState(overrides: Partial<EmbeddingModelState> = {}): EmbeddingModelState {
  return {
    embeddingProvider: 'SENTENCE_TRANSFORMERS',
    modelName: 'sentence-transformers/all-MiniLM-L6-v2',
    effectiveModelName: 'sentence-transformers/all-MiniLM-L6-v2',
    ollamaUrl: '',
    qwenTask: '',
    geminiTaskType: 'SEMANTIC_SIMILARITY',
    batchSize: 32,
    maxLength: null,
    taskPrefixInfo: null,
    taskPrefix: null,
    taskPrefixesLoading: false,
    setModelName: noop,
    setOllamaUrl: noop,
    setQwenTask: noop,
    setGeminiTaskType: noop,
    setBatchSize: noop,
    setMaxLength: noop,
    setTaskPrefix: noop,
    enableTopics: false,
    setEnableTopics: noop,
    topicConfig: DEFAULT_TOPIC_CONFIG,
    setTopicConfig: noop,
    enableSaeActivations: false,
    setEnableSaeActivations: noop,
    saeSelection: null,
    setSaeSelection: noop,
    handleProviderChange: noop,
    buildEmbeddingModelInput: () => ({}) as EmbeddingModelInput,
    getTopicParams: () => ({}),
    getSaePostParams: () => ({ enabled: false, selection: null }),
    ...overrides,
  };
}

/** Gemini Embedding 2 with the search task and `documentPrefix`. */
function gemini2(documentPrefix = 'title: {title} | text: '): EmbeddingModelState {
  return modelState({
    embeddingProvider: 'GEMINI',
    modelName: 'gemini-embedding-2',
    effectiveModelName: 'gemini-embedding-2',
    batchSize: 100,
    taskPrefixInfo: { defaultTask: 'search', tasks: TASKS },
    taskPrefix: { taskId: 'search', documentPrefix },
  });
}

const ROW: TaskPrefixRow = { columns: ['title', 'text'], template: '{text}', embeddingColumns: ['text'] };

const count = (html: string, text: string) => html.split(text).length - 1;

describe('EmbeddingModelForm task prefix', () => {
  it('sits under the model name, before the batch size', () => {
    const html = renderToStaticMarkup(<EmbeddingModelForm model={gemini2()} taskPrefixRow={ROW} />);
    const modelName = html.indexOf('Model Name');
    const prefix = html.indexOf('Document prefix');
    expect(modelName).toBeGreaterThan(-1);
    expect(prefix).toBeGreaterThan(modelName);
    expect(html.indexOf('Batch Size')).toBeGreaterThan(prefix);
  });

  it('previews the model input once, from the row', () => {
    const html = renderToStaticMarkup(<EmbeddingModelForm model={gemini2()} taskPrefixRow={ROW} />);
    expect(count(html, 'Model input:')).toBe(1);
    expect(html).toContain('title: {title} | text: {text}');
    // Gemini Embedding 2 takes no task type.
    expect(html).not.toContain('gemini-task-type');
  });

  it('is absent for a model without task prefixes', () => {
    const html = renderToStaticMarkup(<EmbeddingModelForm model={modelState()} taskPrefixRow={ROW} />);
    expect(html).not.toContain('Document prefix');
    expect(html).not.toContain('Model input:');
  });

  it('notes that media get no prefix in multimodal mode only', () => {
    const media = { kinds: ['image' as const], defaultModel: 'google/embeddinggemma-2' };
    const note = 'images, audio and video get none';
    expect(renderToStaticMarkup(<EmbeddingModelForm model={gemini2()} taskPrefixRow={ROW} media={media} />))
      .toContain(note);
    expect(renderToStaticMarkup(<EmbeddingModelForm model={gemini2()} taskPrefixRow={ROW} />)).not.toContain(note);
  });

  it('flags a placeholder that names no column of the row', () => {
    const html = renderToStaticMarkup(
      <EmbeddingModelForm model={gemini2('title: {author} | text: ')} taskPrefixRow={ROW} />,
    );
    expect(html).toContain('Task prefix references unknown columns: author');
    expect(html).toContain('aria-invalid="true"');
  });

  it('explains column placeholders once when the row has columns', () => {
    const html = renderToStaticMarkup(<EmbeddingModelForm model={gemini2()} taskPrefixRow={ROW} />);
    expect(count(html, '{column}')).toBe(1);
  });

  it('offers static prefixes only without a column section (SAE label maps)', () => {
    const html = renderToStaticMarkup(
      <EmbeddingModelForm model={gemini2('title: none | text: ')} taskPrefixRow={STATIC_TASK_PREFIX_ROW} />,
    );
    expect(html).toContain('Document prefix');
    expect(html).not.toContain('Model input:');
    expect(html).not.toContain('{column}');
  });

  it('reads "No prefix" when the document prefix is emptied, as nothing is sent', () => {
    const html = renderToStaticMarkup(<EmbeddingModelForm model={gemini2('')} taskPrefixRow={ROW} />);
    expect(html).toContain('placeholder="No prefix"');
  });

  it('holds no topic option (TopicExtractionSection does)', () => {
    const html = renderToStaticMarkup(
      <EmbeddingModelForm model={gemini2()} taskPrefixRow={ROW} idPrefix="hf-" />,
    );
    expect(html).not.toContain('enable-topics');
  });
});

describe('EmbeddingModelForm task prompt section', () => {
  const TITLE = 'Task prompt';

  it('titles the task prefix once', () => {
    const html = renderToStaticMarkup(<EmbeddingModelForm model={gemini2()} taskPrefixRow={ROW} />);
    expect(count(html, TITLE)).toBe(1);
    expect(html).toContain('Document prefix');
  });

  it('holds the task type of gemini-embedding-001', () => {
    const model = modelState({
      embeddingProvider: 'GEMINI',
      modelName: 'gemini-embedding-001',
      effectiveModelName: 'gemini-embedding-001',
    });
    const html = renderToStaticMarkup(<EmbeddingModelForm model={model} taskPrefixRow={ROW} />);
    expect(count(html, TITLE)).toBe(1);
    expect(html).toContain('id="gemini-task-type"');
    expect(html).not.toContain('Document prefix');
  });

  it("holds Qwen's query instruction, under the model name, explaining its default", () => {
    const instruction = EMBEDDING_PROVIDERS.QWEN.queryInstruction ?? '';
    const model = modelState({
      embeddingProvider: 'QWEN',
      modelName: 'Qwen/Qwen3-Embedding-0.6B',
      effectiveModelName: 'Qwen/Qwen3-Embedding-0.6B',
      qwenTask: instruction,
    });
    const html = renderToStaticMarkup(<EmbeddingModelForm model={model} taskPrefixRow={ROW} />);
    expect(count(html, TITLE)).toBe(1);
    expect(html).toContain('id="qwen-task"');
    expect(html).toContain(`value="${instruction}"`);
    expect(html).toContain('trained with');
    const instructionAt = html.indexOf('id="qwen-task"');
    expect(instructionAt).toBeGreaterThan(html.indexOf('Model Name'));
    expect(html.indexOf('Batch Size')).toBeGreaterThan(instructionAt);
  });

  it('is absent for models without a task, with no free-text prompt', () => {
    const html = renderToStaticMarkup(<EmbeddingModelForm model={modelState()} taskPrefixRow={ROW} />);
    expect(html).not.toContain(TITLE);
    expect(html).not.toContain('id="prompt"');
  });
});

describe('TopicExtractionSection', () => {
  it('is a titled card that starts collapsed while topics are off', () => {
    const html = renderToStaticMarkup(<TopicExtractionSection model={modelState()} idPrefix="hf-" />);
    expect(html).toContain('Topic Extraction');
    expect(html).toContain('aria-expanded="false"');
    expect(html).not.toContain('hf-enable-topics');
    expect(html).not.toContain('>On<');
  });

  it('starts expanded with its settings while topics are on', () => {
    const html = renderToStaticMarkup(
      <TopicExtractionSection
        model={modelState({ enableTopics: true })}
        idPrefix="local-"
        label="Extract topics after import"
      />,
    );
    expect(html).toContain('aria-expanded="true"');
    expect(html).toContain('>On<');
    expect(html).toContain('id="local-enable-topics"');
    expect(html).toContain('aria-checked="true"');
    expect(html).toContain('Extract topics after import');
    expect(html).toContain('topic-settings-stub');
  });
});

describe('SaeActivationsSection', () => {
  it('is a titled card that starts collapsed while off', () => {
    const html = renderToStaticMarkup(<SaeActivationsSection model={modelState()} idPrefix="hf-" />);
    expect(html).toContain('SAE Activations');
    expect(html).toContain('aria-expanded="false"');
    expect(html).not.toContain('hf-enable-sae-activations');
  });

  it('starts expanded with the SAE picker while on', () => {
    const html = renderToStaticMarkup(
      <SaeActivationsSection model={modelState({ enableSaeActivations: true })} idPrefix="hf-" />,
    );
    expect(html).toContain('aria-expanded="true"');
    expect(html).toContain('>On<');
    expect(html).toContain('id="hf-enable-sae-activations"');
    expect(html).toContain('id="hf-sae-activations-model"');
  });
});

describe('MediaSettingsForm', () => {
  it('keeps its title and shows a load error while collapsed', () => {
    const html = renderToStaticMarkup(
      <MediaSettingsForm budget={null} settingsError="boom" onRetry={noop} overrides={{}} onChange={noop} kinds={{ image: 1 }} />,
    );
    expect(html).toContain('Media Settings');
    expect(html).toContain('aria-expanded="false"');
    expect(html).toContain('Could not load the media settings: boom');
    expect(html).toContain('Retry');
  });
});

describe('ColumnSelector', () => {
  it('explains a pre-selected column under the column list', () => {
    const html = renderToStaticMarkup(
      <ColumnSelector
        columns={[{ name: 'text', dtype: 'string' }, { name: 'label', dtype: 'string' }]}
        selectedEmbeddingColumns={['text']}
        selectedMetadataColumns={['label']}
        onEmbeddingColumnsChange={noop}
        onMetadataColumnsChange={noop}
        selectionNote='Pre-selected "text" by its name.'
        textTemplate="{text}"
        onTemplateChange={noop}
        idColumn="auto"
        onIdColumnChange={noop}
      />,
    );
    expect(html).toContain('Pre-selected &quot;text&quot; by its name.');
  });
});
