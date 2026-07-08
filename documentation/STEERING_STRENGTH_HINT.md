# Steering-strength hint from per-layer residual norms

A utility that measures the residual-stream L2 norm at every decoder layer of a
model and surfaces it in the chat steering UI as a hint for **how strong a
steering coefficient to use**.

## Why

Additive steering does, at layer *L* (`interpret/sae/steering.py::apply_steering`):

```
h' = h + strength · v
```

How disruptive that is depends on the dimensionless ratio

```
ρ = |strength| · ‖v‖ / ‖h_L‖
```

`ρ ≈ 0.1` is a gentle nudge; `ρ ≈ 0.5–1` is strong; `ρ ≫ 1` produces
word-salad. `‖h_L‖`, the residual-stream norm at layer *L*, **grows with
depth** (RMSNorm + attention-sink effects), so a fixed coefficient is far more
violent at an early layer than a late one. Measured on `gemma-3-1b-pt`,
`‖h‖` climbs from ~340 at layer 0 to ~29,000 at layer 24 — so strength 800 is
ρ≈0.14 at layer 9 but ρ≈0.03 at layer 22 (a ~4× difference in effect for the
same number). The hardcoded slider ranges (`±2000` for SAE features, `±5` for
directions in `SteeringControls.tsx`) were the only prior guidance; this
utility replaces guesswork with measurement.

**Key fact (measured, not assumed):** every Gemma-scope SAE decoder row is
pinned to **exactly unit norm** (`‖w_dec[feature]‖ = 1.0000`, std 0, across the
16k and 65k SAEs). So for SAE features `‖v‖ = 1` and `ρ = strength / ‖h_L‖` — a
single per-layer table is a complete, exact hint for *every* SAE feature.
Only the pre-extracted **direction** presets (raw activation-space `.pt`
vectors) are non-unit and carry their own `‖v‖`.

## Pieces

1. **`interpret/inference/residual_norm_profiler.py`** — `ResidualNormProfiler`,
   a family-agnostic class (Gemma *and* Qwen) that runs prompts through a
   wrapper in prefill and summarises per-layer `‖h_L‖`. It **reuses each
   wrapper's existing `cache_activations`** — no new hooks — reconciling the two
   capture shapes (`{"prefill": {L: {"post_mlp": …}}}` for Gemma, `{L:
   {RESID_POST: …}}` for Qwen; `post_mlp` == RESID_POST). It masks the BOS /
   attention-sink token (position 0) and pools per-token norms across prompts,
   reporting median/p25/p75/mean per layer. Pure helpers (`compute_token_norms`,
   `summarize_layer_norms`) are unit-tested without a model
   (`unit_tests/test_residual_norm_profiler.py`).

2. **`scripts/profile_residual_norms.py`** + **`profile_residual_norms_config.toml`**
   — config-driven offline runner. For each checkpoint it loads the wrapper,
   profiles all layers, records each registered steering direction's `‖v‖`
   (from `resources/directions/*.pt` via `DIRECTION_REGISTRY`), and writes the
   frontend asset. Merges in place (keyed by model id), so profiling one model
   never clobbers another. Prints a calibration report (ρ at strength 800 for
   the published SAE layers). Run from `interpretability_backend/`:

   ```bash
   uv run python -m scripts.profile_residual_norms
   ```

3. **`embedding_visualization/lib/utils/residualNorms.json`** — the generated
   asset, imported directly by the frontend (no GraphQL — it's just a small
   table of numbers). Keyed by model id, each entry has `layers[]`
   (`{layer, median, p25, p75, mean, count}`) and `directions{}`
   (`{name: {layer, vecNorm}}`).

4. **`embedding_visualization/lib/utils/steeringStrengthHint.ts`** — pure
   helper: `computeRho`, `suggestedStrength` (its inverse), `strengthBand`
   (subtle/medium/strong on ρ thresholds), and `steeringHint(...)` which reads
   the table and returns `{rho, band, residualNorm, vecNorm, layer,
   suggestedStrength}` or `null` when data is absent. Unit-tested with an
   injected mock table (`__tests__/steeringStrengthHint.test.ts`).

5. **`SteeringControls.tsx`** — per steering row shows `band · ≈ N% of ‖resid‖`
   with a "use \<recommended\>" one-click button, and newly-added features
   default to a **layer-aware** starting strength (`ρ = RHO_RECOMMENDED`)
   instead of the flat 800. Advisory only — the slider still sends a raw
   coefficient; steering math and the GraphQL contract are unchanged.

## Generating / regenerating the table

The shipped `residualNorms.json` is **empty until profiled** — the hint simply
hides when a model has no data. To populate it, run the profiler on a machine
with enough VRAM for the target model (the live demo model, `gemma-3-4b-it`, is
listed in the config; the 4B forward passes need more memory than a laptop
MPS). Stop the backend first if it holds the GPU. The console report's
"strength 800 as fraction of ‖h_L‖" lines let you sanity-check the ρ band
thresholds in `steeringStrengthHint.ts` against today's known-good default.

## Adding a model

Add its checkpoint to `checkpoints` in the config and re-run. For **Qwen**,
also re-check whether its TopK-SAE decoder rows are unit-norm; if not, that
model's SAE hint needs a per-feature `‖v‖` (Gemma-scope does not, because its
decoder is unit-norm).
