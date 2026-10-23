# Embedding task prefixes (EmbeddingGemma, Gemini Embedding 2)

`google/embeddinggemma-300m`, `google/embeddinggemma-2` (sentence-transformers) and
`gemini-embedding-2` (Gemini API) read their task from a **text prefix**. One system serves all
three: the same task table, the same storage, and the same form controls.

## The format

| Task | Query prefix | Document prefix |
|---|---|---|
| Search (default) | `task: search result \| query: ` | `title: none \| text: ` |
| Question answering | `task: question answering \| query: ` | `title: none \| text: ` |
| Fact checking | `task: fact checking \| query: ` | `title: none \| text: ` |
| Code retrieval | `task: code retrieval \| query: ` | `title: none \| text: ` |
| Classification | `task: classification \| query: ` | same as the query prefix |
| Clustering | `task: clustering \| query: ` | same as the query prefix |
| Sentence similarity | `task: sentence similarity \| query: ` | same as the query prefix |

- **Asymmetric tasks** (search, question answering, fact checking, code retrieval) prefix
  documents with `title: {title} | text: `, or `title: none | text: ` when there is no title.
- **Sentence similarity** is for comparing texts with each other. Both vendors' docs say not to
  use it for search, which is why the default is search.
- **Media** (images, audio, video) never gets a prefix.

**Sources:**
- the EmbeddingGemma 1 `config_sentence_transformers.json`;
- the EmbeddingGemma 2 model card;
- the Gemini embeddings docs. These say "You cannot use the `task_type` field for the
  `gemini-embedding-2` model".

The table is `task_prefixes` in `interpretability_backend/config/embedding_models.json`. Models
are marked with `"prompt_format": "task_prefix"`. Each task also lists `prompt_names`: the
sentence-transformers prompt names whose prefix is that task's query prefix
(`Retrieval-query`, `SearchQuery` → search; `STS` → sentence similarity).

## How a collection carries it

There is no new DuckDB column; existing fields are reused:

| Where | Field | Holds |
|---|---|---|
| `EmbeddingModelConfig.prompt` / `embedding_prompt` | document prefix template | `title: {title} \| text: ` |
| `EmbeddingModelConfig.task` / `embedding_task` | query prefix | `task: search result \| query: ` |
| `EmbeddingModelConfig.prompt_format` / ChromaDB `embedding_prompt_format` | marker | `task_prefix` |

`task` already meant "query-time only" (it is Qwen's query instruction), and the GraphQL
`EmbeddingModelInput` already carried `prompt` and `task`.

**The marker** comes from `TaskPrefixPolicy.prompt_format_for` (`backend/utils/embedding_tasks.py`).
The converter sets it for a catalog task-prefix model unless the prompt is a sentence-transformers
prompt name.
- **Prompt names keep today's path.** Older EmbeddingGemma collections and resumed jobs
  (`Retrieval-document`, `STS`) still resolve their names through sentence-transformers.
- **So do older collections without the key** (built before this change).
- **Local fine-tuned EmbeddingGemma checkpoints** are paths, not catalog entries, so they also
  keep prompt names.

## Documents

- **The prefix reaches the model input only.** `EmbeddingTexts` (`utils/embedding_tasks.py`)
  gives, for each row:
  - the stored document: the template, or the embedded columns joined;
  - the model input: `DocumentPrefix.model_input(row, document)`.
- **The stored document never contains the prefix.** That keeps it out of hover cards, text
  search, topic keywords, LLM labels and SAE document activations.
- **Placeholders** (`{title}`) may name any column of the row, not only the embedded ones; the
  title is usually a metadata column.
  - In a re-embed, the row is the item's metadata plus `__document__`.
  - Empty, missing and NaN values render as `none`.
- **Unknown placeholders fail early:** before the model loads, and before an overwrite deletes the
  old collection. The same check runs on the text template (`EmbeddingTexts.validate`).
- **Sentence-transformers:** for marked configs the factory passes no `prompt`, because the text
  already holds the prefix. EmbeddingGemma's pooling has `include_prompt: true`, so a prefix
  written into the text embeds exactly like sentence-transformers' `prompt=`.
- **Gemini:** `EmbedTextGemini` sends each text as its own `Content`. A plain string list would
  be merged into one Content for gemini-embedding-2, giving one aggregated vector (google-genai
  1.73.1, verified offline).

## Queries

`ChromaDBClient._embedding_function_for` rebuilds the config from ChromaDB metadata, and the
factory picks the query prefix (`TaskPrefixPolicy.query_prefix`):

1. **An explicit query prompt wins.**
   - Literal text is used as-is.
   - A prompt name maps through its task's `prompt_names`, so the Explore picker's
     `Retrieval-query` works on Gemini 2 too.
   - Any other prompt name raises; a document prompt such as `Retrieval-document` is not a query
     prefix.
2. **Otherwise** the collection's stored query prefix is used (empty means none).

**Gemini Embedding 2 never gets `task_type`.** The factory drops it for every task-prefix model,
including older collections that recorded `CLUSTERING`/`SEMANTIC_SIMILARITY`. Those older
collections were in fact embedded without any task instruction, and their queries stay unprefixed
unless an explicit prompt is given.

**Semantic search** embeds text queries through the same raising path as `embed_query` and
searches by vector. It used to attach an embedding function to the Chroma collection and silently
fall back to Chroma's stored embedder when ours failed to load.

## Collections page

- **Data:** `embeddingTaskPrefixes` (GraphQL, no arguments) returns the table and its models. The
  page fetches it once, cache-first (`useTaskPrefixTable`), and matches the typed model
  case-insensitively (`taskPrefixesFor`), so the embed button never races a lookup. While it
  loads, the button is blocked.
- **`TaskPrefixField`** offers:
  - a Task select (the tasks plus None);
  - an editable **Document prefix** (picking a task fills it in);
  - "Queries use: …" (the task's query prefix, not editable);
  - a "Model input: …" pattern, the form's only preview of what the model reads;
  - placeholder validation.

  It lives in **`EmbeddingModelForm`** ("5 · Embedding Model"), under Provider and Model Name, and
  only for these models. Every form uses this one place, through the required `taskPrefixRow`
  prop (`TaskPrefixRow`: the columns a placeholder may name, the text template and embedded
  columns for the "Model input" pattern):
  - **HuggingFace, local file, re-embed:** each form builds its row once with
    `form.taskPrefixRow(columns)` (`useEmbedFormState`) and validates with
    `form.placeholderIssues(row.columns)`, so the field and the embed button check the same
    columns. HuggingFace and local files pass the source's column names; re-embed passes the
    source dataset's fields (its metadata keys plus `__document__`).
  - **SaeTab label maps:** no column section, so `STATIC_TASK_PREFIX_ROW` (no columns): static
    prefixes only, no `{column}` hint, validated against the same empty list.

  An emptied document prefix reads "No prefix": documents then get none, while queries keep the
  task's query prefix. In multimodal mode a note says the prefix applies to a row's text only. The
  task prefix used to
  sit in Column Configuration, before the model was picked, with a pointer line in the model card
  and a "not used by the selected model" note for every other model.
- **`EmbeddingModelForm`** shows the task prefix as one kind of its "Task prompt" section (with
  Gemini's task type and Qwen's query instruction; see `COLLECTIONS_EMBED_FORM.md`), so these
  models get no Gemini task type. Other sentence-transformers models get no prompt field.
- **Model input:** `embeddingModelInputFrom` sends `prompt` = document prefix and `task` = query
  prefix.
- **Validation** (`placeholderIssues`, `embedValidation`) blocks the template and prefix problems
  the backend would reject. The text template's check now uses the same grammar as the backend
  (`templateFields`).

**Explore:** "Auto" sends no query prompt; the backend resolves it. The frontend used to map the
stored document prompt itself, which would have sent a document template as the query prompt.

## Limits

- **No multimodal pipeline yet.** A future media embedder must render the prefix into its input's
  `text` itself (`embed_inputs` applies the sentence-transformers prompt, which marked configs
  leave empty).
- **Placeholders hold column names only:** no format specs, no conversions.
- **Unchecked equivalence:** EmbeddingGemma 2's `include_prompt` hasn't been checked locally (it
  needs sentence-transformers ≥ 6.1, which isn't installed).

## Tests

- **Backend:**
  - `test_embedding_model_catalog.py` (the table and flags);
  - `test_text_processing.py` (`ColumnTemplate`);
  - `test_embedding_tasks.py` (marker, document prefix, query prefix);
  - `test_embedding_model_converter.py`;
  - `test_embed_gemini.py` (fake client);
  - `test_embedding_function_factory.py::TestTaskPrefixes`;
  - `test_chroma_embed_query.py` (query path, semantic search raises);
  - `test_embedding_job.py::TestLocalTextTaskPrefix`;
  - `test_job_cancellation.py::TestReEmbedTaskPrefix`;
  - `test_embedding_task_prefixes_query.py` (GraphQL).
- **Frontend:**
  - `app/collections/lib/__tests__/taskPrefixes.test.ts`;
  - `app/collections/components/__tests__/embedFormSections.test.tsx` (where the prefix renders,
    one model-input preview, the shared topics card);
  - `embeddingLimits.test.ts`;
  - `embedValidation.test.ts`;
  - `lib/utils/__tests__/queryPrompt.test.ts`.
