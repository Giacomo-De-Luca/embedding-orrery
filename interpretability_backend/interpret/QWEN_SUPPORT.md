# Qwen3 / Qwen3.5 support in `interpret/`

Status of the Qwen path through the toolkit (inference + activation capture +
Qwen-scope SAE download + connect + steering + streaming). The refusal/poetry
**experiment** adapters and the backend `InterpretService` / frontend wiring are
**not** covered here — see "Backend readiness" at the bottom for the seams.

## What works

| Capability | Where | Notes |
|---|---|---|
| Inference + raw activation capture | `inference/qwen3_transformers.py` (`Qwen3Inference`) | `RESID_POST` / `POST_ATTN` / `ATTN_OUT` / `MLP_OUT` via `cache_activations()`. |
| Qwen-scope SAE download + load | `sae/sae_config.py` (`QwenScopeSAEConfig`) + `sae/loading.py` (`_load_qwen_scope_sae`) | TopK SAEs; dims read from the weights (see below). |
| Connect Qwen → SAE (feature acts) | `sae/hook_manager.py` (`HookManager.session`) | Model-agnostic; hooks the whole decoder layer for `RESID_POST`. |
| Direction steering (actadd / ablation) | `sae/steering.py` (`apply_steering`, `SteeringOp`) | Pure linear algebra; family-agnostic. |
| SAE feature steering | `SteeringOp(feature_index=...)` → `sae.w_dec[idx]` | Exercised by the smoke test. |
| Streaming generation | `Qwen3Inference.generate_stream` / `generate_chat_stream` | Yields the shared `TokenStreamEvent`; cancellable. |

## Supported models (registry)

`QWEN_SCOPE_MODELS` in `sae/sae_config.py` is the single place to add a model.
All Qwen-scope SAE repos live under the **`Qwen/`** org; the family prefix
(`Qwen3` vs `Qwen3.5`) and the optional `-Base` segment live in the repo *name*.

| `model_size` | family | variant | d_in | width | k | n_layers | SAE repo | transformers | RAM (bf16) | Verified |
|---|---|---|---|---|---|---|---|---|---|---|
| `1.7B` | Qwen3 | Base | 2048 | 32k | 50/100 | 28 | `Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_{50,100}` | ≥4.51 | ~4 GB | unit ✓ / live ☐ |
| `8B` | Qwen3 | Base | 4096 | 64k | 50/100 | 36 | `Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_{50,100}` | ≥4.51 | ~16 GB | unit ✓ / live ☐ |
| `27B` | Qwen3.5 | — | 5120 | 80k | 50/100 | 64 | `Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_{50,100}` | ≥5.2 (Gated DeltaNet) | ~54 GB (server) | unit ✓ / live ☐ |

`d_in` / `n_layers` for 8B and 27B should be confirmed against the model card on
the first live download — the loader's shape check is the safety net for `d_in`,
and the smoke test's layer-count guard catches an `n_layers` / model mismatch.
To add a model (incl. Qwen3.5-2B/9B or the MoE 30B-A3B / 35B-A3B), add one
`QwenScopeModelInfo(...)` entry; confirm its `widths`/`ks` against the HF repo.

## Design notes

- **Dims come from the weights, names from the registry.** `_load_qwen_scope_sae`
  reads `d_sae, d_in = state["W_enc"].shape` (on-disk `W_enc` is `(d_sae, d_in)`,
  transposed on load to the Gemma convention) and builds the `TopKSAE` from those.
  `config.d_in` is validated against the tensor (catches a wrong `model_size`);
  `config.d_sae` (from the shared `WIDTH_TO_D_SAE`, which carries both families'
  width labels) is only a pre-download estimate used by the smoke test's shape
  asserts. Adding a model therefore needs no dim bookkeeping beyond the advisory
  registry value.
- **RESID_POST only.** Qwen-scope SAEs are trained on the residual stream;
  `QwenScopeSAEConfig.__post_init__` rejects any other `hook_type`. Because
  `HookManager` hooks the *whole decoder layer* for `RESID_POST`, attachment is
  architecture-agnostic — Qwen3.5's hybrid linear-attention (Gated DeltaNet)
  layers need no special handling for the SAE path.
- **`linear_attn` fallback.** `hook_manager.py` and `qwen3_transformers.py` resolve
  `ATTN_OUT` to `self_attn` *or* `linear_attn`. Not needed for the residual-stream
  SAEs above; it's there for the (out-of-scope) experiment `ATTN_OUT` path on
  Qwen3.5 hybrid layers.
- **SAE cache.** `loading._SAE_CACHE` keys Qwen entries on
  `(layer_index, hook_type, model_size, width, k, dtype, device)` — `model_size`
  is included so models that share a width (e.g. the MoE W128K pair) don't collide.

## Streaming contract

`Qwen3Inference.generate_stream(prompt, ..., cancel_event=None)` and
`generate_chat_stream(turns, ...)` mirror `GemmaPytorchInference` exactly: both
yield `interpret.inference.streaming.TokenStreamEvent(token_index, token_id,
text_delta, is_done)` and accept a `threading.Event` for cancellation, so a
consumer (e.g. a future `InterpretService.generate_stream` subscription) can use
either wrapper unchanged. Implementation: `model.generate` runs on a background
thread feeding a token-id queue; text deltas come from a full-decode + diff (so
multi-byte / BPE pieces aren't split); a `StoppingCriteria` checks the cancel
event per token (set automatically on early generator close). Chat formatting
uses `enable_thinking=False` to suppress `<think>` blocks in the visible output.

## Verifying

```bash
# Pure unit tests (CPU, no download)
uv run pytest interpretability_backend/unit_tests/test_qwen_scope_config.py \
              interpretability_backend/unit_tests/test_qwen_stream.py

# Raw activation capture (downloads the base model)
uv run python -m interpret.inference.qwen3_transformers \
    --model Qwen/Qwen3-1.7B --prompt "What colour is the sky?" --capture-layer 14

# End-to-end SAE smoke (downloads base model + SAE)
uv run python -m interpret.diagnostics.qwen_scope_smoke --model Qwen/Qwen3-1.7B   --size 1.7B
uv run python -m interpret.diagnostics.qwen_scope_smoke --model Qwen/Qwen3-8B     --size 8B
uv run python -m interpret.diagnostics.qwen_scope_smoke --model Qwen/Qwen3.5-27B  --size 27B  # server
```

## Backend readiness

**Phase 1 is wired** (2026-07): `backend/services/interpret_service.py` carries a
family switch — `load_model` dispatches on the checkpoint (`Qwen/Qwen3-1.7B` →
`Qwen3Inference`), `_make_sae_config()` builds `QwenScopeSAEConfig` (k=50) for
steering/highlight, streaming chat translates the Gemma `"model"` role to
`"assistant"` and exposes `enable_thinking` end-to-end (GraphQL
`GenerateStreamInput.enableThinking`, default off). The id scheme is
`model_id="qwen3-1.7B-base"` / `sae_id="{layer}-qwenscope-1-res-{width}"`
(`source_ids.qwen_source_id`). Verified live on MPS: smoke test, steered
generation, streaming with thinking on/off, prompt highlight.

**Phase 1.5 is wired** (2026-07): per-token prompt activations
(`PromptExplorer` / `run_prompt_activations`) now work for Qwen. `PromptExplorer`
is family-agnostic — the `wrapper` supplies the chat template (`format_prompt`)
and per-token pieces (a new symmetric `token_strings()` on both
`GemmaPytorchInference` and `Qwen3Inference`), and `PromptExplorerConfig.sae_config_factory`
(injected by the service from its `_make_sae_config`) builds the family's SAE
config, so the explorer no longer constructs Gemma-scope configs itself. The
service's `_find_prompt_token_range` gained a Qwen ChatML branch
(`<|im_start|>`/`<|im_end|>` markers, forward-scan robust to an injected system
turn; base offset derived from `wrapper.prepends_bos`). Features come back with
index + activation; **labels stay blank** (see Phase 2). Aligned to the prefill
sequence via each wrapper's `token_strings` (Gemma prepends BOS, Qwen does not).

Still Gemma-only: labels/densities/activation examples (Qwen-scope is **not** on
Neuronpedia, so labels need the autointerp pass — Phase 2). Bootstrap
decoder-vector ingestion: `scripts/extract_qwen_decoder_vectors.py`.
