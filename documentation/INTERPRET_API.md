# Interpret API — Frontend Integration Guide

Backend API for SAE inference: prompt activation capture, steered generation, and scatter plot feature highlighting. The backend wraps the `interpret/` toolkit and exposes it via GraphQL mutations.

## Model Lifecycle

The Gemma3-4b-it model (~8GB) is loaded on demand and stays resident until explicitly unloaded. Only one model can be loaded at a time. All inference operations require a loaded model.

### Query: `modelStatus`

```graphql
query {
  modelStatus {
    loaded
    modelName
    device
  }
}
```

Returns `{ loaded: false }` when no model is loaded.

### Mutation: `loadModel`

```graphql
mutation {
  loadModel(checkpoint: "google/gemma-3-4b-it") {
    loaded
    modelName
    device
  }
}
```

- Downloads weights from HuggingFace on first call (~30s), cached thereafter
- Returns `loaded: true, device: "mps"` on success
- Returns `loaded: false, modelName: "<error message>"` on failure
- Timeout: 300s (covers download + load)
- Calling when already loaded returns an error

### Mutation: `unloadModel`

```graphql
mutation {
  unloadModel {
    loaded
  }
}
```

Frees GPU memory. Safe to call when no model is loaded (no-op).

## Use Case 1: Prompt Activations (Features Page)

Run a prompt through the model with SAE hooks attached at multiple layers. Returns per-token top-k feature activations with Neuronpedia labels.

### Mutation: `runPromptActivations`

```graphql
mutation RunPromptActivations($input: RunPromptActivationsInput!) {
  runPromptActivations(input: $input) {
    prompt
    tokenStrings
    layers {
      layer
      width
      tokens {
        token
        position
        features {
          index
          activation
          label
          density
        }
      }
    }
    error
  }
}
```

#### Input

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `prompt` | `String!` | — | Raw user prompt (chat template applied automatically) |
| `layers` | `[Int!]` | `[9, 17, 22, 29]` | Which decoder layers to attach SAE hooks |
| `width` | `String!` | `"16k"` | SAE width: `"16k"`, `"32k"`, `"65k"`, `"262k"` |
| `saes` | `[SaeLayerSpecInput!]` | `null` | Explicit `(layer, width)` SAE specs — takes precedence over `layers` + `width`. Supports mixed widths, **including two widths at the same layer** (co-attached at one hook site), all captured in a single forward pass. |
| `topK` | `Int!` | `10` | Number of top features per token |

`SaeLayerSpecInput`: `{ layer: Int!, width: String! = "16k" }`. Example — hook L9-16k, L9-65k and L22-16k at once:

```json
{ "saes": [ {"layer": 9, "width": "16k"}, {"layer": 9, "width": "65k"}, {"layer": 22, "width": "16k"} ] }
```

Each result entry in `layers[]` is identified by its `(layer, width)` pair. With multiple specs, leave `saeId` unset so each layer derives its own width-aware sae_id for DuckDB label lookup.

#### Response

- `tokenStrings`: The tokenized prompt as string pieces (including BOS, chat template tokens)
- `layers[]`: One entry per requested layer, each containing:
  - `tokens[]`: One entry per token position, each containing:
    - `features[]`: Top-k active features sorted by activation descending
      - `index`: Feature index (maps to `featureIndex` in the feature explorer)
      - `activation`: Raw activation value
      - `label`: Neuronpedia autointerpreter label (or "(no labels)" if labels unavailable)
      - `density`: Feature density (activation frequency), may be null

#### Frontend Integration

**Location**: Features page (`/features`), as a new `PromptExplorerPanel` component below `ActivationExamples`.

**Components to create**:
- `PromptExplorerPanel.tsx`: Prompt input + layer checkboxes + "Run" button + results display
- Results should render per-layer sections, each using the existing `TokenStrip` component for the token-activation heatmap
- Clicking a feature `index` in the results should navigate to that feature's detail view (update URL: `?featureIndex=<index>`)

**Reusable**: The existing `TokenStrip` component (`app/features/components/TokenStrip.tsx`) renders token heatmaps given `tokens[]`, `values[]`, and `maxValueTokenIndex`. For prompt explorer results, build these arrays from the response per-token features (use the top-1 activation value per token for the heatmap).

## Use Case 2: Steered Generation (Features Page)

Apply additive steering on an SAE feature direction and generate text. Returns both baseline (no steering) and steered output for comparison.

### Mutation: `generateSteeredResponse`

```graphql
mutation GenerateSteeredResponse($input: GenerateSteeredInput!) {
  generateSteeredResponse(input: $input) {
    baselineText
    steeredText
    steering { featureIndex layer hookType width strength }
    error
  }
}
```

#### Input

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `prompt` | `String!` | — | User prompt |
| `steering` | `[SteeringInput!]!` | — | One or more steering features (see below) |
| `outputLen` | `Int!` | `128` | Max tokens to generate |
| `temperature` | `Float` | `null` | Sampling temperature. `null` = greedy decoding |

#### SteeringInput (used by UC2 and UC4)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `featureIndex` | `Int!` | — | SAE feature index (row in `w_dec`) |
| `layer` | `Int!` | — | Decoder layer |
| `hookType` | `HookTypeEnum!` | `RESID_POST` | Hook site: `RESID_POST`, `MLP_OUT`, `ATTN_OUT` |
| `width` | `String!` | `"16k"` | SAE width |
| `strength` | `Float!` | `800.0` | Additive steering coefficient |

Multiple features compose additively in insertion order. Features can target the same or different layers. One SAE is loaded per unique `(layer, hookType, width)` combination.

#### Strength Guidelines (Gemma3-4b-it)

Gemma3's post-norm architecture pushes per-token residual norms into the thousands. Steering strengths must be on the same scale:

- `0` — No effect (baseline)
- `400-600` — Subtle influence
- `800-1200` — Moderate steering (good default range)
- `1600` — Strong steering
- `>2000` — Model may break down (incoherent output)

Negative strengths steer _away_ from the feature direction.

#### Frontend Integration

**Location**: Features page, as a new `SteeringPanel` component.

**UI design**:
- Feature list: add/remove steering features, each with index + layer + strength. Pre-fill from currently viewed feature.
- Strength slider per feature: range -2000 to 2000, default 800. Show numeric input alongside.
- Prompt text area
- Temperature and output length controls (collapsible "Advanced")
- "Generate" button
- Results: two text blocks side-by-side — "Baseline" and "Steered" — for easy comparison
- Loading state: generation takes 5-30s depending on output length

## Use Case 3: Prompt Highlight (Main Page Scatter Plot)

When an SAE collection (decoder vectors) is loaded in the scatter plot, run a prompt through the model, max-pool SAE activations across tokens, and return which features fired. These map to points in the scatter plot for highlighting.

### Mutation: `runPromptHighlight`

```graphql
mutation RunPromptHighlight($input: RunPromptHighlightInput!) {
  runPromptHighlight(input: $input) {
    features {
      featureIndex
      activation
    }
    error
  }
}
```

#### Input

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `prompt` | `String!` | — | User prompt |
| `layer` | `Int!` | — | Decoder layer to capture activations from |
| `width` | `String!` | `"16k"` | SAE width |
| `hookType` | `HookTypeEnum!` | `RESID_POST` | Hook site |

#### Response

- `features[]`: Nonzero features after max-pooling across token positions, sorted by activation descending
  - `featureIndex`: Maps to `metadata.index` on scatter plot points
  - `activation`: Max activation value across all tokens in the prompt

Typically returns a few hundred to a few thousand features (out of 16384 for 16k width).

#### Frontend Integration

**Location**: Main page, as a `PromptHighlightInput` component in the search sidebar. Only visible when the active collection is an SAE collection.

**Detection**: Use `getSaeInfo(selectedCollection)` from `lib/utils/saeCollections.ts`. Returns `{ modelId, saeId }` or `null`. When non-null, show the prompt highlight input.

**Layer/width extraction**: Parse from the `saeId` field. For example, `"9-gemmascope-2-res-16k"` → `layer=9`, `width="16k"`, `hookType="resid_post"` (the `"res"` maps to `resid_post`).

**Mapping to scatter plot points**:
```typescript
// Build HighlightMap from mutation result
const highlightMap = new Map<number, number>();
const maxActivation = Math.max(...features.map(f => f.activation));

for (const { featureIndex, activation } of features) {
  // Find the point index where metadata.index === featureIndex
  const pointIndex = data.ids.findIndex(
    (_, i) => data.item_metadata?.[i]?.index === featureIndex
  );
  if (pointIndex !== -1) {
    highlightMap.set(pointIndex, activation / maxActivation); // normalize to 0-1
  }
}
```

For performance, pre-build a `featureIndex → pointIndex` lookup map once when data loads (like `useHighlightedIndices` does for IDs).

**Feeding into the highlight pipeline**: The resulting `HighlightMap` should merge with or replace the semantic search results in `useHighlightedIndices`. Options:
1. Pass as an additional prop to `DashboardPanel` alongside existing `combinedHighlightedIndices`
2. Extend `useHighlightedIndices` to accept an optional `promptHighlights` parameter

## Use Case 4: Streaming Chat Generation (WebSocket Subscription)

Stream tokens from the Gemma model one at a time for a chatbot-like interface. Supports multi-turn conversations and optional SAE steering. Uses WebSocket via GraphQL subscription.

### Subscription: `generateStream`

```graphql
subscription GenerateStream($input: GenerateStreamInput!) {
  generateStream(input: $input) {
    streamId
    tokenIndex
    tokenId
    text
    done
    error
  }
}
```

#### Input

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `turns` | `[ChatTurnInput!]!` | — | Conversation history: `[{role: "user", content: "..."}, ...]` |
| `outputLen` | `Int!` | `256` | Maximum tokens to generate |
| `temperature` | `Float` | `null` | Sampling temperature. `null` = greedy decoding |
| `topP` | `Float!` | `0.95` | Nucleus sampling threshold |
| `topK` | `Int!` | `64` | Top-k sampling threshold |
| `steering` | `[SteeringInput!]` | `null` | Optional list of SAE steering features (same type as UC2, see above) |

#### ChatTurnInput

| Field | Type | Description |
|-------|------|-------------|
| `role` | `String!` | `"user"` or `"model"` |
| `content` | `String!` | Message content |

#### Response (streamed per-token)

Each `TokenChunk` event contains:
- `streamId`: UUID identifying this generation session
- `tokenIndex`: 0-based position in the generated output
- `tokenId`: Raw SentencePiece token ID
- `text`: Clean text delta for display — concatenate these for the full response
- `done`: `true` on the last token (EOS/EOT or output_len reached)
- `error`: Non-null on failure (model not loaded, timeout, etc.)

#### Architecture

```
Frontend (WebSocket) ← GraphQL subscription ← asyncio.Queue ← token_emitter.emit_token()
                                                                        ↑ (thread-safe put_nowait)
                                                              InterpretService.generate_stream()
                                                                        ↑ (runs in thread pool)
                                                              GemmaPytorchInference.generate_chat_stream()
                                                                        ↑ (sync Python generator)
                                                              Gemma3ForMultimodalLM.generate_stream()
                                                                        ↑ (yields per-token from decode loop)
```

- **Delta decoding**: SentencePiece tokens don't align with character boundaries. The wrapper decodes the full growing token list each step and diffs against the previous output to get clean `text` deltas.
- **GPU lock**: The subscription acquires `InterpretService._lock` for the entire generation, serialising GPU access.
- **Abort**: A `threading.Event` is checked per-token in the decode loop. On client disconnect (WebSocket close) or timeout (300s per-token), the event is set and generation stops after the current forward pass.
- **Steering**: When `SteeringInput` is provided, a `HookManager` with SAE + `SteeringOp` wraps the generation — same mechanism as `generateSteeredResponse`.

#### Frontend Integration

**Components**:
- Chat message list with streaming text append
- Input box with multi-turn conversation state
- Stop button: unsubscribe from the WebSocket (triggers abort via `GeneratorExit`)
- Optional steering controls (feature index, layer, strength)

**Example (conceptual)**:
```typescript
const subscription = client.subscribe({
  query: GENERATE_STREAM,
  variables: {
    input: {
      turns: [{ role: "user", content: "Explain quantum entanglement" }],
      outputLen: 256,
      temperature: 0.7,
    }
  }
});

let fullText = "";
subscription.subscribe({
  next: ({ data }) => {
    const chunk = data.generateStream;
    fullText += chunk.text;
    updateUI(fullText);
    if (chunk.done) closeStream();
  }
});
```

## Error Handling

All mutations return an `error: String | null` field instead of throwing. The streaming subscription includes `error` on the final `TokenChunk` (`done: true`). The frontend should:

1. Check `error` field first — if non-null, show a toast/alert with the message
2. Common errors:
   - `"Model not loaded. Call loadModel first."` — show a "Load Model" button
   - `"Generation timed out after Xs"` — suggest shorter output or simpler prompt
   - `"feature_index N out of range [0, M)"` — invalid feature index
   - `"Model already loaded (...). Call unloadModel first."` — model lifecycle error

## Model Status UX

**Recommended**: Add a `ModelStatusBadge` component to `FeatureHeader` showing:
- Red dot + "Model not loaded" with a "Load" button
- Green dot + "Gemma3-4b (mps)" when loaded
- "Unload" button to free memory

Poll `modelStatus` on page mount to initialize. The load/unload mutations return the updated status directly.

## GraphQL Type Reference

```graphql
enum HookTypeEnum {
  RESID_POST
  MLP_OUT
  ATTN_OUT
}

input RunPromptActivationsInput {
  prompt: String!
  layers: [Int!]
  width: String! = "16k"
  topK: Int! = 10
}

input SteeringInput {
  featureIndex: Int!
  layer: Int!
  hookType: HookTypeEnum! = RESID_POST
  width: String! = "16k"
  strength: Float! = 800.0
}

input GenerateSteeredInput {
  prompt: String!
  steering: [SteeringInput!]!
  outputLen: Int! = 128
  temperature: Float
}

input RunPromptHighlightInput {
  prompt: String!
  layer: Int!
  width: String! = "16k"
  hookType: HookTypeEnum! = RESID_POST
}

type ModelStatus {
  loaded: Boolean!
  modelName: String
  device: String
}

type InterpretActiveFeature {
  index: Int!
  activation: Float!
  label: String!
  density: Float
}

type InterpretTokenFeatures {
  token: String!
  position: Int!
  features: [InterpretActiveFeature!]!
}

type InterpretLayerResult {
  layer: Int!
  width: String!
  tokens: [InterpretTokenFeatures!]!
}

type PromptActivationsResponse {
  prompt: String!
  tokenStrings: [String!]!
  layers: [InterpretLayerResult!]!
  error: String
}

type AppliedSteering {
  featureIndex: Int!
  layer: Int!
  hookType: String!
  width: String!
  strength: Float!
}

type SteeredGenerationResponse {
  baselineText: String!
  steeredText: String!
  steering: [AppliedSteering!]!
  error: String
}

type PromptHighlightFeature {
  featureIndex: Int!
  activation: Float!
}

type PromptHighlightResponse {
  features: [PromptHighlightFeature!]!
  error: String
}

# Streaming chat generation (UC4)

input ChatTurnInput {
  role: String!
  content: String!
}

input GenerateStreamInput {
  turns: [ChatTurnInput!]!
  outputLen: Int! = 256
  temperature: Float
  topP: Float! = 0.95
  topK: Int! = 64
  steering: [SteeringInput!]
}

type TokenChunk {
  streamId: String!
  tokenIndex: Int!
  tokenId: Int!
  text: String!
  done: Boolean!
  error: String
}
```

## Frontend Integration Task List

> **STATUS (2026-07): IMPLEMENTED — kept for historical reference only.** The frontend described below has been fully built: `useSteeringChat.ts` is a complete streaming implementation over the `generateStream` subscription (with abort, regenerate, seeded compare mode, and steering-config change detection — no `fetchSteeringChat()` stub remains), and the components live under `app/sae/components/` (ModelStatusButton, PromptTokenActivations, SteeringControls, the ChatInterface family). Note also that the GraphQL type reference above predates several schema additions — e.g. `SteeringInput.directionName`, `GenerateStreamInput.seed`, `RunPromptActivationsInput.modelId/saeId/skipChatTemplate/filterMode`, `ModelStatus.variant/modelSize` — check `backend/API/types.py` for the current schema.

### 1. GraphQL operations

**File: `embedding_visualization/lib/graphql/queries.ts`** (append)

```graphql
query GetModelStatus {
  modelStatus { loaded modelName device }
}
```

**File: `embedding_visualization/lib/graphql/mutations.ts`** (create or append)

```graphql
mutation LoadModel($checkpoint: String!) {
  loadModel(checkpoint: $checkpoint) { loaded modelName device }
}

mutation UnloadModel {
  unloadModel { loaded modelName device }
}

mutation RunPromptActivations($input: RunPromptActivationsInput!) {
  runPromptActivations(input: $input) {
    prompt tokenStrings
    layers { layer width tokens { token position features { index activation label density } } }
    error
  }
}

mutation GenerateSteeredResponse($input: GenerateSteeredInput!) {
  generateSteeredResponse(input: $input) {
    baselineText steeredText
    steering { featureIndex layer hookType width strength }
    error
  }
}

mutation RunPromptHighlight($input: RunPromptHighlightInput!) {
  runPromptHighlight(input: $input) {
    features { featureIndex activation }
    error
  }
}
```

**File: `embedding_visualization/lib/graphql/subscriptions.ts`** (create)

```graphql
subscription GenerateStream($input: GenerateStreamInput!) {
  generateStream(input: $input) {
    streamId tokenIndex tokenId text done error
  }
}
```

### 2. TypeScript types

Add to `embedding_visualization/lib/types/types.ts`:

- `ModelStatus { loaded, modelName?, device? }`
- `InterpretActiveFeature { index, activation, label, density? }`
- `InterpretTokenFeatures { token, position, features[] }`
- `InterpretLayerResult { layer, width, tokens[] }`
- `PromptActivationsResponse { prompt, tokenStrings, layers[], error? }`
- `SteeringInput { featureIndex, layer, hookType, width, strength }`
- `AppliedSteering { featureIndex, layer, hookType, width, strength }`
- `SteeredGenerationResponse { baselineText, steeredText, steering[], error? }`
- `PromptHighlightFeature { featureIndex, activation }`
- `TokenChunk { streamId, tokenIndex, tokenId, text, done, error? }`

### 3. Features page: Model status badge

**File: `embedding_visualization/app/features/components/ModelStatusBadge.tsx`** (create)

- Small indicator in `FeatureHeader` showing loaded/unloaded state
- "Load" / "Unload" button
- Polls `modelStatus` on mount, updates after load/unload mutations
- All inference panels should be disabled when model is not loaded

### 4. Features page: Prompt Explorer panel

**File: `embedding_visualization/app/features/components/PromptExplorerPanel.tsx`** (create)

- Text input for prompt
- Layer multi-select (checkboxes, default `[9, 17, 22, 29]`)
- "Run" button fires `RunPromptActivations` mutation
- Results: per-layer sections using existing `TokenStrip` component
  - Build `tokens[]` and `values[]` from response (top-1 activation per token for the heatmap)
  - `maxValueTokenIndex` = index of token with highest activation
- Click on feature index → navigate to feature detail (update URL `?featureIndex=<index>`)
- Show loading spinner during inference (5-15s)

### 5. Features page: Steering panel

**File: `embedding_visualization/app/features/components/SteeringPanel.tsx`** (create)

- Multi-feature steering list: add/remove entries, each with `featureIndex`, `layer`, `hookType`, `width`, `strength`
- Pre-fill first entry from currently viewed feature
- Strength slider per entry: range -2000 to 2000, default 800
- Prompt text area + output length + temperature (collapsible "Advanced")
- "Generate" button fires `GenerateSteeredResponse` mutation
- Results: side-by-side "Baseline" vs "Steered" text comparison
- Loading state (5-30s depending on output length and number of features)

### 6. Features page: Wire `useSteeringChat` to streaming subscription

**File: `embedding_visualization/lib/hooks/useSteeringChat.ts`** (modify)

The hook currently has a stub `fetchSteeringChat()` that returns a hardcoded response after 1500ms. Replace with:

1. Apollo `useSubscription(GENERATE_STREAM_SUBSCRIPTION)` or manual `client.subscribe()`
2. Convert chat messages to `ChatTurnInput[]` (note: frontend uses `role: 'assistant'`, backend expects `role: 'model'`)
3. On each `TokenChunk`: append `text` delta to the last assistant message's `content`
4. On `done: true`: set status to `'idle'`
5. On `error`: set status to `'error'`, show error message
6. Stop button: unsubscribe from WebSocket (triggers abort on backend)
7. Pass `steering` list from `SteeringConfig` if features are configured

### 7. Main page: Prompt highlight input

**File: `embedding_visualization/app/components/PromptHighlightInput.tsx`** (create)

- Only visible when `getSaeInfo(selectedCollection)` returns non-null
- Text input + "Highlight" button
- Parse `layer` and `width` from `saeId` (e.g. `"9-gemmascope-2-res-16k"` → `layer=9, width="16k"`)
- Fire `RunPromptHighlight` mutation
- Convert result to `HighlightMap`:
  - Pre-build `featureIndex → pointIndex` lookup from `data.item_metadata[i].index`
  - Normalize activations to 0-1 by dividing by max
- Feed into existing `highlightedIndices` pipeline in `page.tsx`

### 8. Wire into page layouts

**`embedding_visualization/app/features/page.tsx`** (modify):
- Import and render `ModelStatusBadge` in `FeatureHeader` area
- Import and render `PromptExplorerPanel` after `ActivationExamples`
- Import and render `SteeringPanel` after `PromptExplorerPanel`
- Pass `modelId`, `saeId`, `featureIndex`, `currentFeature` as needed

**`embedding_visualization/app/page.tsx`** (modify):
- Import and render `PromptHighlightInput` in the search sidebar area
- Merge its `HighlightMap` output with `combinedHighlightedIndices`

### Dependency order

```
1. GraphQL operations + types (foundation for everything)
2. ModelStatusBadge (gates all inference panels)
3. PromptExplorerPanel (standalone, no dependencies beyond model status)
4. SteeringPanel (standalone, pre-fills from current feature)
5. useSteeringChat wiring (depends on streaming subscription + SteeringPanel config)
6. PromptHighlightInput (standalone, main page integration)
```

Steps 3-6 are independent of each other and can be built in parallel.

## Key References

- **SAE collection mapping**: `embedding_visualization/lib/utils/saeCollections.ts`
- **Highlight pipeline**: `embedding_visualization/lib/hooks/useHighlightedIndices.ts`
- **Token strip component**: `embedding_visualization/app/features/components/TokenStrip.tsx`
- **Chat interface**: `embedding_visualization/app/features/components/ChatInterface/` (ChatPanel, ChatMessage, ChatInput, ChatGreeting)
- **Chat hook (stub)**: `embedding_visualization/lib/hooks/useSteeringChat.ts` (replace `fetchSteeringChat()`)
- **Scroll hook**: `embedding_visualization/lib/hooks/useScrollToBottom.ts`
- **Features page**: `embedding_visualization/app/features/page.tsx`
- **Apollo client**: `embedding_visualization/lib/utils/apollo-client.ts` (WebSocket already configured)
- **Backend service**: `interpretability_backend/backend/services/interpret_service.py`
- **Backend mutations**: `interpretability_backend/backend/API/mutations.py` (search for "Interpret")
- **Backend subscription**: `interpretability_backend/backend/API/subscriptions.py` (search for "generate_stream")
- **Token event bus**: `interpretability_backend/backend/services/token_emitter.py`
