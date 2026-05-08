# SAE Module

Hook-based system for attaching pretrained Sparse Autoencoders to raw PyTorch transformer models, capturing feature activations during inference, and (where label data exists) looking up Neuronpedia autointerpreter labels. No SAELens or TransformerLens dependency.

## Supported SAE families

| Family | Architecture | Model | Capture | Steering | Neuronpedia labels | Decoder-vector pipeline |
|---|---|---|---|---|---|---|
| [Gemma-scope](https://huggingface.co/google/gemma-scope-2-4b-it) | JumpReLU | Gemma3-4b | yes | yes | yes | yes |
| [Qwen-scope](https://huggingface.co/Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50) | TopK (L0_50, L0_100) | Qwen3-1.7B | yes | yes | not yet | not yet |

The hook / activation-store / steering machinery is family-agnostic. Adding Qwen-scope was a matter of adding a new `SAEBase` subclass (`TopKSAE`), a config dataclass (`QwenScopeSAEConfig`), and a loader path. `HookManager` accepts either family transparently. The Neuronpedia-coupled bits (`feature_labels`, `pipeline/`, `autointerpreter/`, `exploration/`, `extract_decoder_vectors`) remain Gemma-only — Qwen-scope is not indexed by Neuronpedia, so generalising those paths would be churn without a payoff today.

`HookManager` currently supports `RESID_POST`, `ATTN_OUT`, and `MLP_OUT` hook sites for both families. `POST_ATTN` is supported by the inference-side `Qwen3Inference.cache_activations` but not by `HookManager` (no SAE in either suite is trained at that site).

## Folder layout

The top level of this folder holds the **core library** (SAE model, config, loading, hooks, steering, activation store, feature labels). Larger tooling lives in subpackages:

- [`exploration/`](exploration/) — Interactive / notebook-facing feature exploration (`NeuronpediaExplorer`, `PromptExplorer`).
- [`pipeline/`](pipeline/) — Data preparation (`prepare_sae_data`, `extract_decoder_vectors`).
- [`diagnostics/`](diagnostics/) — Manual smoke tests and steering/alignment diagnostics. Not pytest.

## Quick Start — Gemma-scope on Gemma3

```python
from interpret.inference.gemma_pytorch import GemmaPytorchInference
from interpret.sae import HookManager, SAEConfig, FeatureLabelStore

# Load model
wrapper = GemmaPytorchInference("google/gemma-3-4b-it")

# Attach SAE to layer 29 (prefill only — skip decode tokens)
config = SAEConfig(layer_index=29, prefill_only=True)
manager = HookManager()
manager.add_sae(config)

# Run inference with SAE hooks
with manager.session(wrapper.model.model.layers) as store:
    wrapper.generate("The cat sat on the warm red mat", output_len=1)
    acts = store.prefill(layer=29).feature_acts[0]  # (seq_len, 16384)

# Look up labels for top features
label_store = FeatureLabelStore("resources/sae_labels/neuronpedia_gemma-3-4b-it")
model_id, layer, hook, width = label_store.params_from_config(config)

densities = label_store.get_densities(model_id, layer, hook, width)
mask = (densities > 0) & (densities < 0.01)  # exclude high-frequency features

results = label_store.label_top_k_per_token(
    acts, model_id, layer, hook, width, k=5, mask=mask,
)
```

## Quick Start — Qwen-scope on Qwen3

```python
from interpret.inference.qwen3_transformers import Qwen3Inference
from interpret.sae import HookManager, QwenScopeSAEConfig

# Load model
wrapper = Qwen3Inference("Qwen/Qwen3-1.7B")

# Attach the W32K-L0_50 SAE at layer 15. Use k=100 for the L0_100 sibling.
config = QwenScopeSAEConfig(
    layer_index=15, k=50, device=str(wrapper.device),
)
manager = HookManager()
manager.add_sae(config)

with manager.session(wrapper.decoder_layers) as store:
    wrapper.generate("The colour of the sky is", output_len=1)
    acts = store.prefill(layer=15).feature_acts[0]  # (seq_len, 32768)
    # per-token L0 == 50 exactly (hard TopK)
```

Steering works identically across families — `SteeringOp(layer_index=15, mode=SteeringMode.ADDITIVE, feature_index=top_feat, strength=10.0, normalise=True)` against either an `SAEConfig` (Gemma) or a `QwenScopeSAEConfig` registered at the same site.

## Files

| File | Class | Purpose |
|---|---|---|
| `sae_config.py` | `GemmaScopeSAEConfig` (alias: `SAEConfig`), `QwenScopeSAEConfig`, `HookType` | Per-family configuration dataclasses. Each derives its HF `repo_id` and weight filename. |
| `sae_model.py` | `SAEBase`, `JumpReLUSAE`, `TopKSAE` | `SAEBase` is the common interface (`encode/decode/forward` returning `(feature_acts, reconstruction)` + `w_dec` shape `(d_sae, d_in)`). `JumpReLUSAE` is the Gemma-scope architecture; `TopKSAE` is the Qwen-scope architecture. |
| `loading.py` | `load_sae()` | Downloads SAE weights from HuggingFace Hub. Dispatches on config type — Gemma-scope `params.safetensors` -> `JumpReLUSAE`, Qwen-scope `layer{N}.sae.pt` -> `TopKSAE` (transposes `W_enc` / `W_dec` to the Gemma orientation on load). |
| `hook_manager.py` | `HookManager` | Attaches/detaches SAEs as forward hooks on decoder layers. Supports `prefill_only`, read-only mode, and steering interventions composed alongside activation capture. |
| `activation_store.py` | `ActivationStore` | Captures feature activations per forward pass. Provides `prefill()`, `latest()`, `all_feature_acts()`. |
| `feature_labels.py` | `FeatureLabelStore` | SQLite-backed Neuronpedia label lookup. Stores labels, densities, 256-dim explanation embeddings, and top/bottom logits. Supports multiple labelling methods and feature-to-feature similarity search. |
| `steering.py` | `SteeringOp`, `SteeringMode`, `apply_steering()`, `resolve_op()` | Steering specs and math for additive / orthogonal / ablation / projection-cap interventions on SAE features or raw direction vectors. |

## Configs

```python
# Gemma-scope JumpReLU SAE config (alias: SAEConfig).
GemmaScopeSAEConfig(
    layer_index=29,         # 0-33 for Gemma3 4b
    hook_type=HookType.RESID_POST,  # RESID_POST, MLP_OUT, ATTN_OUT
    model_size="4b",        # derives repo_id + neuronpedia_model_id
    variant="it",           # "it" (instruction-tuned) or "pt" (base)
    width="16k",            # SAE feature count: "16k", "65k", "262k"
    l0_size="medium",       # sparsity level: "small", "medium", "big"
    d_in=2560,              # model hidden size
    prefill_only=False,     # only capture the first forward pass
    read_only=True,         # False enables activation steering
)
# .repo_id -> "google/gemma-scope-2-4b-it"
# .neuronpedia_model_id -> "gemma-3-4b-it"

# Qwen-scope TopK SAE config.
QwenScopeSAEConfig(
    layer_index=15,         # 0-27 for Qwen3-1.7B
    k=50,                   # 50 or 100 (selects the L0_50 / L0_100 trained variant)
    width="32k",            # only "32k" shipped today
    model_size="1.7B",      # only "1.7B" today
    variant="Base",         # only "Base" today
    hook_type=HookType.RESID_POST,
    d_in=2048,              # Qwen3-1.7B hidden size
    prefill_only=False,
    read_only=True,
)
# .repo_id -> "Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50"
# .weights_filename() -> "layer15.sae.pt"
```

## Steering

`HookManager` can apply steering interventions on the residual stream during inference. Four modes are supported, all broadcast across every token position. The direction `v` can be a row of `sae.w_dec` (via `feature_index`) or a raw vector.

| Mode | Formula | Notes |
|---|---|---|
| `ADDITIVE` | `h + strength * v` | Pure push along `v`. |
| `ORTHOGONAL` | `h + (strength - 1) * ((h · v) / (v · v)) * v` | Scales only the component parallel to `v`. `strength=1` is identity, `strength=0` removes the direction. |
| `ABLATION` | `h + (strength - 1) * (h · v) * v` | `v` always L2-normalised. `strength=0` fully ablates. |
| `PROJECTION_CAP` | `h + (clip(h · v, cap_min, cap_max) - h · v) * v` | `v` always L2-normalised. Conditional — no-op when `h · v` is inside the bounds. Either bound may be `None`. `strength_multiplier` is ignored. |

```python
from interpret.sae import HookManager, SAEConfig, SteeringOp, SteeringMode

manager = HookManager()
manager.add_sae(SAEConfig(layer_index=9))    # for feature lookup + capture
manager.add_sae(SAEConfig(layer_index=29))   # capture only

manager.add_steering([
    # amplify SAE feature 4287 at layer 9
    SteeringOp(layer_index=9, mode=SteeringMode.ADDITIVE,
               feature_index=4287, strength=6.0, normalise=True),
    # partially suppress a feature direction at layer 29
    SteeringOp(layer_index=29, mode=SteeringMode.ORTHOGONAL,
               feature_index=1234, strength=0.3),
    # fully ablate a custom direction at layer 20 (no SAE registered there)
    SteeringOp(layer_index=20, mode=SteeringMode.ABLATION,
               vector=my_direction, strength=0.0),
    # cap how strongly a feature can fire — only intervenes if proj > 5.0
    SteeringOp(layer_index=9, mode=SteeringMode.PROJECTION_CAP,
               feature_index=4287, cap_max=5.0),
])

with manager.session(wrapper.model.model.layers) as store:
    wrapper.generate("What colour is the sky?")
    feats_9 = store.prefill(layer=9).feature_acts  # post-steering activations
```

Notes:

- Multiple ops on the same layer compose in insertion order.
- A layer can have steering without an SAE registered; a lightweight steering-only hook is attached.
- Feature activations captured during a steered session reflect the **post-steering** hidden state — i.e. "given this intervention, what features are active?".
- `manager.set_strength_multiplier(m)` scales every additive / orthogonal / ablation op globally.
- **Warning**: combining steering with `read_only=False` on the same layer replaces the steered state with its (lossy) SAE reconstruction. A `warnings.warn` is raised at `attach()` time.

## FeatureLabelStore

Backed by a single SQLite database (`features.db`) in the labels directory. JSONL files are auto-imported on first query and re-imported when the source file changes.

### Label Methods

Each feature can have multiple labels under different method names. The Neuronpedia autointerpreter label is stored as `method="label"`. Custom methods can be written:

```python
store.write_labels({0: "my label", 1: "another"}, model_id, layer, hook, width, method="custom")
store.get_label(0, model_id, layer, hook, width, method="custom")  # "my label"
store.get_label(0, model_id, layer, hook, width, method="label")   # original Neuronpedia label
```

### Feature Similarity

The stored 256-dim explanation embeddings (from Neuronpedia's autointerpreter) enable feature-to-feature similarity search:

```python
# Find features with similar explanations to feature 4287 ("colors and hues")
similar = store.find_similar_features(4287, model_id, layer, hook, width, k=10)
# -> [(idx, cosine_similarity, label), ...]
```

Note: these are text embeddings of the label strings, not SAE activations. Since we don't know the embedding model, only feature-to-feature similarity is supported (not text queries).

### Logits

```python
logits = store.get_logits(0, model_id, layer, hook, width)
# {"top": [("token", score), ...], "bottom": [("token", score), ...]}
```

## Data Layout

```
resources/sae_labels/neuronpedia_gemma-3-4b-it/
    features.db                                          # single SQLite DB (auto-generated)
    gemma-3-4b-it_9-gemmascope-2-res-16k_features.jsonl  # source JSONL (~65 MB)
    gemma-3-4b-it_9-gemmascope-2-res-16k_activations.jsonl  # activation examples (~1.8 GB)
    activations/9-gemmascope-2-res-16k/batch-*.jsonl.gz  # raw activation batches
    ...
```

The features JSONL files contain density, labels, embeddings, and logits. The activation files contain ~20 token-level activation examples per feature (512 tokens each). Activations are NOT imported into the DB due to size.
