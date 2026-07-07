# Refusal is mediated by a single direction — replication on Gemma-3-4b-it

We replicate Arditi et al. (2024), *"Refusal in Language Models Is Mediated by a
Single Direction"* (arXiv:2406.11717), on `google/gemma-3-4b-it`, a model outside
the paper's original suite. The goal is to test whether a single linear direction
in the residual stream bidirectionally controls refusal on a modern,
post-norm-heavy architecture, and to record where the method has to deviate from
the reference on this model.

## Method

We follow the reference pipeline. From `n_train = 128` harmful and 128 harmless
instructions we compute the **difference-in-means** direction
`r = mean(h_harmful) − mean(h_harmless)` at each candidate `(intermediate,
end-of-instruction token position, layer)`, taking activations at the
`pre_attn` residual point. Prompts use the bare Gemma chat template with **no
system prompt** (`<start_of_turn>user … <end_of_turn> <start_of_turn>model`).

Each candidate is scored on `n_val = 32` held-out harmful/harmless prompts with
the reference's three metrics, all read from **last-position logits**:

- **bypass score** — refusal score on harmful-val under the direction-removing
  intervention;
- **induce score** — refusal score on harmless-val under adding the direction;
- **KL divergence** — between baseline and intervened next-token distributions on
  harmless-val (the "surgical" constraint).

The refusal score is `log P(refusal_tok) − log(1 − P(refusal_tok))` with the
refusal token `"I"` (Gemma-3 SentencePiece id `236777`). Candidate filtering is
the paper's: drop the top `20%` of layers, drop `KL > 0.1`, drop
`induce < 0.0`; the surviving candidate with the lowest bypass score is selected.

The selected direction is then **behaviourally evaluated** on held-out sets by
greedy-generating 256-token completions under three conditions and scoring each
completion:

- harmful eval (JailbreakBench, HarmBench-test) under `{baseline, ablation,
  actadd(−1)}` — success = refusal *drops* when the direction is removed;
- harmless-test under `{baseline, actadd(+1)}` — success = refusal *rises* when
  the direction is added.

## Differences from Arditi's implementation

| Aspect | Arditi et al. | This replication | Why |
|---|---|---|---|
| **Model** | Gemma-1/2, Llama-2/3, Qwen-1.8B, Yi | **Gemma-3-4b-it** (34 layers, d=2560) | out-of-distribution test on a post-norm architecture |
| **Bypass intervention** | directional **ablation** (project the direction out at every layer × {resid, attn, mlp}) | **activation addition** with coeff −1 at the source layer (`bypass_mode="actadd"`) | three-site ablation *collapses* Gemma-3: its post-norm residual norms are in the thousands, so the projection perturbation drives logits to ±∞ and the model emits degenerate output |
| **Ablation hook site** | forward-**pre**-hook on each layer input | forward-hook on each layer **output** | 1-of-`n_layers` site offset (input of layer 0 vs. input to the final norm); irrelevant under `actadd` |
| **Refusal token** | `235285` (Gemma-1/2 vocab) | `236777` = `"I"` (Gemma-3 renumbered vocab) | vocabulary change; verified against the tokenizer |
| **Direction scaling (actadd)** | — | **raw** mean-difference vector (norm ≈ 1.5×10³ at layer 14), not unit-normalised | matches the reference's difference-in-means scale; ablation self-normalises internally |
| **Safety metric** | substring refusal judge **+** LlamaGuard-2 / HarmBench-cls | substring refusal judge **+** Llama-Guard-4-12b (via API); HarmBench-cls also supported locally — see `safety_score.py` | different classifier generation; hosted so no GPU needed |
| **Robustness** | — | non-finite-logit guards in scoring; an eval-cache signature | keep the selection filter sane under intervention collapse |

The headline deviation is the bypass intervention: on Gemma-3 the paper's
directional *ablation* is unusable, so we jailbreak by *subtracting* the
direction (activation addition, coeff −1) rather than projecting it out. The
config exposes both (`bypass_mode ∈ {ablation, actadd}`); `actadd` is the default
for this model.

## Results

We report **both of Arditi's judges**: the substring **refusal rate** and, on the
harmful sets, a **safety score** — the attack-success rate (ASR = fraction of
completions a harmfulness classifier flags as harmful), here Llama-Guard-4-12b.
Under the **activation-addition** intervention the single direction controls
refusal *and* actual harmfulness in both directions:

| set (n) | | baseline | actadd(−1) | actadd(+1) |
|---|---|---|---|---|
| JailbreakBench (100) | refusal | 0.84 | **0.00** | — |
| | harm ASR | 0.11 | **0.83** | — |
| HarmBench-test (159) | refusal | 0.75 | **0.00** | — |
| | harm ASR | 0.21 | **0.86** | — |
| Harmless-test (100) | refusal | 0.04 | — | **1.00** |

Subtracting the direction (actadd −1) does not merely stop refusals — it raises the
judge-verified harm ASR from 0.11 → **0.83** on JailbreakBench and 0.21 → **0.86**
on HarmBench-test, with genuinely harmful completions (synthesis instructions,
persuasive disinformation, malware). Adding the direction (actadd +1) induces
refusal on **100/100** entirely benign requests (up from 4/100 at baseline). This
reproduces the paper's central claim on Gemma-3 under both metrics.

The harm increase is uniform across categories, several saturating. (Baseline ASR
is the ~16–25% of non-refusing baseline responses that are already harmful — a
control; the actadd delta is the causal effect of removing the direction.)

**JailbreakBench — harm ASR by category** (n = 10 each):

| category | baseline | actadd(−1) |
|---|---|---|
| Sexual/Adult content | 0.00 | **1.00** |
| Government decision-making | 0.20 | **1.00** |
| Expert advice | 0.20 | 0.90 |
| Fraud/Deception | 0.40 | 0.90 |
| Harassment/Discrimination | 0.10 | 0.90 |
| Economic harm | 0.10 | 0.80 |
| Malware/Hacking | 0.00 | 0.80 |
| Physical harm | 0.00 | 0.80 |
| Privacy | 0.10 | 0.80 |
| Disinformation | 0.00 | 0.40 |

**HarmBench-test — harm ASR by category**:

| category | n | baseline | actadd(−1) |
|---|---|---|---|
| chemical_biological | 19 | 0.37 | **1.00** |
| cybercrime_intrusion | 33 | 0.45 | **1.00** |
| harassment_bullying | 16 | 0.00 | 0.88 |
| illegal | 47 | 0.15 | 0.87 |
| misinformation_disinformation | 27 | 0.15 | 0.74 |
| harmful | 17 | 0.00 | 0.59 |

The weakest cells (Disinformation 0.40, `harmful` 0.59) are the categories where the
model most often produces a non-refusing but *non-harmful* completion after the
intervention (see below), not where it re-refuses.

The `ablation` condition reports 0% refusal on both harmful sets, **but this is an
artifact of the Gemma-3 collapse, not a real jailbreak**: the completions are
degenerate token loops (`"troviamo blends blends …"`), which the substring judge
scores as non-refusal while a safety classifier would score as near-zero harm.
This is precisely why the safety score is necessary, and why the `actadd`
numbers — not the `ablation` numbers — are the result on this model. (We therefore
skip harm-scoring the ablation condition.)

## Limitations

- The evaluated direction is a **hand-selected candidate** (`pre_attn`, position
  −3, layer 14), not the pipeline's automatically-selected one: under the earlier
  (ablation-only) selection code every candidate was filtered out on Gemma-3. The
  fixed `actadd` selection criterion should now yield an automatically-selected
  direction; re-running it is the natural next step.
- The safety score above uses **Llama-Guard-4-12b** (via a hosted API), not the
  paper's `cais/HarmBench-Llama-2-13b-cls`; the two judges are close but not
  identical, so the ASR values are indicative rather than a like-for-like match to
  the paper. `safety_score.py` also runs HarmBench-cls locally (GPU) for exact
  reproduction.
- The substring refusal judge is surface-level and crude by design (ported
  verbatim for comparability): it counts degenerate output as non-refusal — which
  is why the safety classifier is reported alongside it.

## Reproduce

```bash
# 1. fetch the datasets (gitignored; harmful prompts not committed)
uv run python -m interpret.experiments.refusal_directions.download_dataset
# 2. run selection + evaluation (Gemma default, actadd bypass)
uv run python -m interpret.experiments.refusal_directions.run
# 3. safety-score the completions (paper's second judge)
#    hosted API (no GPU) — recommended:
OPENROUTER_API_KEY=... uv run python -m interpret.experiments.refusal_directions.safety_score llamaguard_api
#    or locally on a GPU box:
uv run python -m interpret.experiments.refusal_directions.safety_score harmbench   # or: llamaguard
```

Code: [`interpret/experiments/refusal_directions/`](../../interpret/experiments/refusal_directions/README.md).
The eval-from-fixed-direction harness used for the numbers above is
`scripts/scratch/eval_refusal_csv_candidate.py`.


