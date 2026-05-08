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
| `topK` | `Int!` | `10` | Number of top features per token |

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
    featureIndex
    layer
    hookType
    strength
    error
  }
}
```

#### Input

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `prompt` | `String!` | — | User prompt |
| `featureIndex` | `Int!` | — | SAE feature index to steer on |
| `layer` | `Int!` | — | Decoder layer for SAE + steering |
| `hookType` | `HookTypeEnum!` | `RESID_POST` | Hook site: `RESID_POST`, `MLP_OUT`, `ATTN_OUT` |
| `width` | `String!` | `"16k"` | SAE width |
| `strength` | `Float!` | `800.0` | Steering strength (additive coefficient) |
| `outputLen` | `Int!` | `128` | Max tokens to generate |
| `temperature` | `Float` | `null` | Sampling temperature. `null` = greedy decoding |

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
- Pre-fill `featureIndex` and `layer` from the currently viewed feature
- Strength slider: range -2000 to 2000, default 800. Show numeric input alongside
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

## Error Handling

All mutations return an `error: String | null` field instead of throwing. The frontend should:

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

input GenerateSteeredInput {
  prompt: String!
  featureIndex: Int!
  layer: Int!
  hookType: HookTypeEnum! = RESID_POST
  width: String! = "16k"
  strength: Float! = 800.0
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

type SteeredGenerationResponse {
  baselineText: String!
  steeredText: String!
  featureIndex: Int!
  layer: Int!
  hookType: String!
  strength: Float!
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
```

## Key References

- **SAE collection mapping**: `embedding_visualization/lib/utils/saeCollections.ts`
- **Highlight pipeline**: `embedding_visualization/lib/hooks/useHighlightedIndices.ts`
- **Token strip component**: `embedding_visualization/app/features/components/TokenStrip.tsx`
- **Features page**: `embedding_visualization/app/features/page.tsx`
- **Backend service**: `interpretability_backend/backend/services/interpret_service.py`
- **Backend mutations**: `interpretability_backend/backend/API/mutations.py` (search for "Interpret")
