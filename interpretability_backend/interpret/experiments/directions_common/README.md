# `directions_common` — backend-agnostic direction-experiment toolkit

Shared infrastructure for the activation-**direction** experiments
(`refusal_directions`, `poetry_directions`) so each runs on **Gemma-3**
(`GemmaPytorchInference`, fork PyTorch) **or Qwen3 / Qwen3.5**
(`Qwen3Inference`, HF transformers) from one pipeline — no per-model forks.

Both experiments do the same three things: extract mean-of-difference
candidate directions from raw activations, sweep `(site, position, layer[,
coeff])` scoring each via refusal-logit + KL, then evaluate completions under
steering. Only a narrow surface is backend-specific; this package isolates it.

## Structure

| Module | Contents |
|---|---|
| `model_adapter.py` | `DirectionModel` protocol + `GemmaDirectionModel`, `QwenDirectionModel`, and `build_direction_model(model_name)` factory. |
| `sites.py` | `CaptureSite` enum (canonical residual-stream points) + `QWEN_SITE_MAP` resolution. |
| `steering_ops.py` | `_additive_op`, `_ablation_ops`, `_make_manager` — emit `SteeringOp`s for `HookManager`. |
| `scoring.py` | `_refusal_score`, `_kl_div`, `_score_dataset` (talks to a `DirectionModel`). |

## `DirectionModel` surface

The pipelines depend only on:

- `n_layers`, `d_model`, `decoder_layers` — dims + the `ModuleList` for `HookManager.session`.
- `format_chat(instruction)` — apply the model's chat template.
- `eoi_token_ids()` — the constant end-of-instruction token window (slice size for mean capture).
- `refusal_token_ids(configured)` — resolve/verify the refusal-cue token ids.
- `capture_means(instructions, sites, n_eoi)` → `{site_name: (n_eoi, n_layers, d_model)}` fp64 CPU.
- `last_position_logits(prompt)` — last-token logits under any open steering session.
- `generate(prompt, max_new_tokens)` — greedy completion (honours an open session).

**Steering is not part of the adapter** — it stays on `HookManager`, which
takes `decoder_layers`. The adapter only *reads* (capture, logits) and formats.

## Capture sites

`CaptureSite` values are the canonical strings used in configs and
`mean_diffs_<site>.pt` artifacts (kept identical to Gemma's fork-cache names):

| `CaptureSite` | value | Gemma | Qwen `(HookType, offset)` |
|---|---|---|---|
| `RESID_PRE` | `pre_attn` | `pre_attn` | `(RESID_POST, -1)` |
| `POST_ATTN` | `post_attn` | `post_attn` | `(POST_ATTN, 0)` |
| `MLP_OUT` | `mlp_out` | `mlp_out` | `(MLP_OUT, 0)` |
| `RESID_POST` | `post_mlp` | `post_mlp` | `(RESID_POST, 0)` |

Qwen has no `pre_attn` capture; `RESID_PRE[L] = RESID_POST[L-1]`. Layer 0 has
no source and is left as zeros — matching Gemma, where `pre_attn[0]` over the
constant EOI positions is identically zero and is discarded by the downstream
zero-norm filter.

## Backend differences handled by the adapter

| Concern | Gemma | Qwen |
|---|---|---|
| Chat template | `format_prompt` (static Gemma markers) | `apply_chat_template(enable_thinking=False)` |
| EOI window | tokenize the known `<end_of_turn>…` suffix | longest common token suffix of two formatted prompts |
| Refusal ids | verify configured maps to `"I"` (Gemma default `236777`) | recompute first token of `"I"`/`"As"` for the Qwen tokenizer |
| Capture | fork `cache_activations(intermediates=…)` + `reset_prefill_cache` | re-enter `cache_activations(hook_types=…)` per sample |
| Logits | post-final-norm + tied embedding (+ softcap) | plain HF forward `logits[:, -1]` (no softcap) |

## Usage

```python
from interpret.experiments.directions_common import build_direction_model, CaptureSite

model = build_direction_model("Qwen/Qwen3-1.7B")
n_eoi = len(model.eoi_token_ids())
means = model.capture_means(instructions, (CaptureSite.RESID_PRE,), n_eoi)
```
