# Collections embed form

The embed forms of the Collections page (HuggingFace, local file, re-embed; SaeTab's label maps
reuse the model card). This page covers three things set in October 2026: the optional cards, the
model card's "Task prompt" section, and the columns a source starts with.

## Cards

The required steps are numbered cards: "1 · Data Source" to "5 · Embedding Model" (re-embed's
cards are unnumbered). The optional ones are **collapsible cards** with a title, a chevron and a
one-line description, all from one component, `CollapsibleCard`
(`app/collections/components/CollapsibleCard.tsx`):

| Card | Component | Shown |
|---|---|---|
| Media Settings | `MediaSettingsForm` | image / audio / video mode; its badge is the token estimate |
| Topic Extraction | `TopicExtractionSection` | HuggingFace, local file, re-embed |
| SAE Activations | `SaeActivationsSection` | the same, except media embeds |

- **Collapsed** while the step is off; a card whose step is on starts expanded and shows an "On"
  badge, so a collapsed card still tells its state.
- **The checkbox inside** turns the step on; its settings appear under it. Both step cards are
  `OptionalStepCard` (same file), so they share the badge, the open state and the checkbox row.
- Collapsed content unmounts: the settings live in the form state (`useEmbeddingModelState`).

Before, the topics option sat inside the HuggingFace model card but in cards of its own on the
local-file tab, and both optional steps were a lone checkbox in an untitled card.

## Task prompt

The model card shows **one titled section, "Task prompt"**, under Provider and Model Name, only
for models that read a task. Which control it holds comes from `modelTaskKind`
(`app/collections/lib/embeddingLimits.ts`), the rule `embeddingModelInputFrom` also follows, so
the form and the job input agree:

| Kind | Models | Control | Sent as |
|---|---|---|---|
| `prefix` | EmbeddingGemma, Gemini Embedding 2 | task + document prefix (`TaskPrefixField`, see `EMBEDDING_TASK_PREFIXES.md`) | `prompt` = document prefix, `task` = query prefix |
| `taskType` | other Gemini models (gemini-embedding-001) | task type select | `taskType` |
| `instruction` | Qwen (QWEN provider) | query instruction | `task` |
| none | every other model | nothing | nothing |

- **Qwen's instruction** starts as the one Qwen3-Embedding was trained with, `Given a web search
  query, retrieve relevant passages that answer the query` (its model card and its
  sentence-transformers `query` prompt), held in `EMBEDDING_PROVIDERS.QWEN.queryInstruction`
  (`lib/utils/embeddingProviders.ts`). Queries get it, documents get none; an emptied field sends
  the default, since the model card puts the loss without an instruction at about 1 to 5% of
  retrieval accuracy. The backend's `EmbedTextQWEN` keeps the same default.
- **No free-text prompt.** The form used to offer every other sentence-transformers model a
  prompt picker suggesting EmbeddingGemma's prompt names, which those models don't define (the
  backend rejects such names). Text to add goes in the text template, with one difference: the
  template's output is the stored document (shown in Explore, matched by text search), while the
  old prompt reached the model only, and queries get no matching prefix. Model-only prefixes such
  as e5's `passage: ` or nomic's `search_document: ` can no longer be set from the form. The
  backend still accepts `prompt` (resumed jobs replay their stored input), Explore's query prompt
  picker (`PromptPresets.queryGroups`) is unchanged, and the frontend no longer fetches the
  catalog's `prompts` field.

## Starting columns

When a source's columns and its 5 preview rows load (and again on a data-type switch), both tabs
call `form.applyColumnDefaults` (`useEmbedFormState`), which applies `ColumnDefaults.select`
(`app/collections/lib/columnDefaults.ts`, pure):

1. **Text by name:** the first column named like a document, in this order, ignoring case:
   `text`, `document`, `content`, `body`, `abstract`, `article`, `passage`, `context`, `review`,
   `description`, `summary`, `sentence`, `caption`, `question`. Its preview values must be
   strings (a tokenised list is skipped); without a preview the name decides.
2. **Else the longest prose:** among columns where at least half the preview values are prose
   (two words of letters, in any alphabet, and not a list or object), the longest on average.
   Ids, labels, urls, numbers and timestamps never qualify, nor do lists, objects or vectors
   stored as text (the previews stringify nested values). Text written without spaces (Chinese,
   Japanese) doesn't either: such a column is picked by name or by hand.
3. **Else nothing:** the form asks for a column instead of embedding, say, a number column.

Media columns and the id column are never picked as text. Every other column becomes metadata,
the template is `{column}`, and a column named like an id (`id`, `index`, `idx`, `_id`,
`row_id`, `item_id`, `doc_id`, `feature_id`) becomes the ID column. Image / audio / video mode
selects every media column and no text; vector mode takes a column whose name contains
`embedding` / `vector` / `emb`, else the first one. Re-embed keeps `__document__`.

A note under the column list says why the column was picked ("by its name" / "the longest text in
the preview") while it is still the whole selection (`ColumnDefaults.note`).

Before, HuggingFace took the first string column and local files the first column of any type, so
an `id` or `title` column was often embedded; the two tabs also had separate copies of this logic
with id-name lists that had drifted apart.

## Tests

- `app/collections/lib/__tests__/columnDefaults.test.ts`
- `app/collections/lib/__tests__/embeddingLimits.test.ts` (`modelTaskKind`, Qwen's default)
- `app/collections/components/__tests__/embedFormSections.test.tsx` (task prompt section,
  collapsible cards, the column note)
- `lib/utils/__tests__/promptPresets.test.ts`
