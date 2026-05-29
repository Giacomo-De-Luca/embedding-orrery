# `interpret/` — interpretability toolkit

A self-contained subsystem for white-box interpretability work on small open-weights LLMs. Bundles raw-PyTorch and HuggingFace inference wrappers, the Gemma- and Qwen-scope SAE library, Neuronpedia data downloaders, two worked experimental applications (refusal-direction and poetry-direction extraction), a vendored fork of Google's `gemma_pytorch`, and a few pre-extracted steering vectors so you can play with the pipeline without re-running the whole extraction.

> **Status: portable.** No imports from outside the folder; all tooling resolves relative to the toolkit. The folder is intended to be either copied into another project as-is or extracted into its own repository. The name is provisional.

## Quickstart

```python
import torch
from interpret.inference.gemma_pytorch import GemmaPytorchInference
from interpret.experiments.refusal_directions.select_direction import _additive_op
from interpret.experiments.refusal_directions.tokens import format_chat
from interpret.sae import HookManager

direction = torch.load("interpret/directions/poetry_prose.pt").to(torch.float32)
LAYER, COEFF = 11, +1.0   # match interpret/directions/poetry_prose.json

wrapper = GemmaPytorchInference("google/gemma-3-4b-it")
manager = HookManager()
manager.add_steering([_additive_op(direction, LAYER, coeff=COEFF)])

with manager.session(wrapper.model.model.layers):
    print(wrapper.generate_from_template(
        format_chat(wrapper, "Recommend a chocolate cake recipe for two."),
        output_len=160,
    ))
```

For an interactive cell-by-cell version with a sweep-score browser, see [`notebooks/poetry_steer_tester.ipynb`](notebooks/poetry_steer_tester.ipynb).

## Layout

```
interpret/
├── inference/                    LLM wrappers exposing residual-stream hooks
│   ├── gemma_pytorch.py          Gemma3-4b-it (raw PyTorch, MPS/CUDA, bfloat16)
│   └── qwen3_transformers.py     Qwen3 / Qwen3.5 (HF transformers; no fork)
├── sae/                          SAE library (no SAELens / TransformerLens dependency)
│   ├── sae_model.py              JumpReLU / TopK
│   ├── hook_manager.py           Attach / detach, read-only, steering sessions
│   ├── steering.py               Additive / orthogonal / ablation / projection-cap
│   ├── activation_store.py       Sparse + dense capture buffers
│   ├── feature_labels.py         Lazy Neuronpedia label store + density mask
│   ├── loading.py                HF Hub SAE-weight resolver
│   ├── sae_config.py             SAEConfig + HookType (resid_post / attn_out / mlp_out)
│   ├── exploration/              Neuronpedia explorer + Jupyter PromptExplorer
│   ├── pipeline/                 prepare_sae_data — download + merge + decoder extract
│   └── autointerpreter/          WordNet-prompt → top-k features → LLM autointerp
├── download/                     Neuronpedia bulk-S3 + per-feature API downloaders
├── experiments/                  Worked applications of the toolkit
│   ├── refusal_directions/       Arditi et al. replication on Gemma-3-4b-it
│   └── poetry_directions/        Poetry-vs-prose direction extraction
├── directions/                   Pre-extracted steering vectors (load + steer)
│   ├── poetry_prose.pt + .json   ≈ 12 KB each; selected at post_attn pos=−2 layer=11
│   └── poems_paraphrase.pt + .json
├── diagnostics/                  Manual smoke tests (steering, index alignment, qwen-scope)
├── notebooks/                    Interactive testers (steer / refusal / poetry)
├── utils/
│   ├── wordnet_parser.py         English WordNet 2024 (auto-downloads, cached)
│   └── results_io.py             CSV / JSON I/O helpers
└── forked/
    └── gemma_pytorch/             Patched fork of github.com/google/gemma_pytorch
```

Each non-trivial subdirectory has its own README:
- [`QWEN_SUPPORT.md`](QWEN_SUPPORT.md) — Qwen3 / Qwen3.5 support: model registry, SAE-config design, streaming contract
- [`sae/README.md`](sae/README.md) — full SAE library tour
- [`experiments/refusal_directions/README.md`](experiments/refusal_directions/README.md)
- [`experiments/poetry_directions/README.md`](experiments/poetry_directions/README.md)
- [`directions/README.md`](directions/README.md) — load and steer
- [`diagnostics/README.md`](diagnostics/README.md) — smoke tests

## Porting to a new project

1. **Drop the folder in.** Copy the entire `interpret/` directory into the new project's repo root. Keeping it at the repo root lets `import interpret.X` resolve when the new project's CWD is the repo root, with no packaging step.

2. **Add the dependencies** to the new project's `pyproject.toml` (or `requirements.txt`). The toolkit needs:

   ```
   torch                  inference + SAE math
   transformers           Qwen3 path + autointerpreter tokenisers
   huggingface_hub        snapshot_download for Gemma weights + SAE weights
   tqdm                   long-loop progress
   requests               Neuronpedia downloads
   pandas, numpy, scipy   analysis + sparse activation store
   matplotlib             sweep plots in experiments
   pyarrow                parquet decoder-vector outputs
   pyyaml                 config loading

   # Required by the gemma_pytorch fork
   absl-py
   immutabledict
   sentencepiece
   pillow
   ```

   Optional:
   - `ipykernel` to run the notebooks under VS Code / JupyterLab.
   - `accelerate` if you want HF's faster model loading on CUDA.

3. **Override the resource paths.** The experiment configs default to CWD-relative `Path("resources/...")` for outputs and inputs (see [`experiments/refusal_directions/config.py`](experiments/refusal_directions/config.py) and [`experiments/poetry_directions/config.py`](experiments/poetry_directions/config.py)). When porting:

   ```python
   from interpret.experiments.refusal_directions import RefusalConfig, RefusalRunner
   cfg = RefusalConfig(
       output_dir=Path("/your/project/data/refusal"),
       splits_dir=Path("/your/project/data/refusal_splits"),
       eval_dir=Path("/your/project/data/refusal_eval"),
   )
   ```

   Or, easier, mirror the parent project's layout: create matching `resources/` folders at the new project's CWD and the defaults just work. Resources are gitignored in this project; same approach is fine elsewhere.

4. **(Optional) Pin the Gemma fork.** [`forked/gemma_pytorch/`](forked/gemma_pytorch/) carries local patches on top of upstream. Re-syncing with `github.com/google/gemma_pytorch` is a manual diff job — don't blindly pull. See the patches section below.

5. **Notebooks: re-run the bootstrap cell once.** Each notebook in `notebooks/` starts with a small cell that walks up from CWD until it finds `interpret/__init__.py` and inserts that on `sys.path` (and optionally `os.chdir`s there). It is idempotent and works regardless of where the kernel was launched.

## Patches in `forked/gemma_pytorch/`

Local edits on top of the vendored upstream — preserved across the refactor by dropping the inner `.git` so the patches enter the parent repo's history:

| File | What's patched | Why |
|---|---|---|
| `gemma/model.py` | ~134 lines: per-layer activation cache wired into the forward pass | `GemmaPytorchInference.cache_activations()` reads residual-stream intermediates per layer per step |
| `gemma/gemma3_model.py` | Minor edits to expose hookable layer outputs | hook attachment site for steering |
| `gemma/tokenizer.py` | Small helper additions | EOI suffix tokenisation |

Path resolution: [`inference/gemma_pytorch.py:49`](inference/gemma_pytorch.py#L49) computes `_GEMMA_PYTORCH_ROOT` as `parents[1] / "forked" / "gemma_pytorch"` — relative to the toolkit, not the repo. The fork moves with the folder.

## What ships

- **Inference wrappers** for two model families with clean residual-stream hook surfaces.
- **A self-contained SAE library** with attach / read / steer primitives, a Neuronpedia label store, an HF Hub weight resolver, and a Jupyter explorer. No SAELens / TransformerLens.
- **A Neuronpedia downloader** (bulk S3 + per-feature API) and an SAE-data pipeline that downloads, merges, and extracts decoder vectors to parquet.
- **An autointerpreter** that scores SAE features against WordNet prompts and ships them to an LLM agent for labels — see [`sae/autointerpreter/`](sae/autointerpreter/).
- **Two worked direction-extraction experiments** — refusal (Arditi et al. replication) and poetry-vs-prose — each with `extract → sweep / select → evaluate → summarise` pipelines and matplotlib outputs.
- **Pre-extracted steering vectors** at [`directions/`](directions/) so you can verify the pipeline without GPU time.
- **Diagnostic scripts** in [`diagnostics/`](diagnostics/) — manual smoke tests that exercise the steering injection, feature-index alignment vs Neuronpedia, and the Qwen-scope path end-to-end.
- **Three interactive notebooks** in [`notebooks/`](notebooks/) covering the SAE-feature steer, refusal-direction steer, and poetry-direction steer flows. Each starts with a small `sys.path`/`chdir` bootstrap cell so they work regardless of where the Jupyter kernel was launched.

## Internal utility split

Two utilities live inside the toolkit deliberately:

- [`utils/wordnet_parser.py`](utils/wordnet_parser.py) — only the autointerpreter uses it; lives here so the toolkit remains self-contained.
- [`utils/results_io.py`](utils/results_io.py) — incremental CSV / JSON checkpointing helpers. Intentional duplicate of any equivalent in the parent project.

No other utilities cross the toolkit boundary.

## Conventions

- **British spelling** for project text (`colour`, etc.) inherited from the parent — code symbols use the British form too where it shows up.
- **Run from the repo root.** All toolkit modules use `interpret.X` absolute imports. CLI entry points are typically `uv run python -m interpret.<module>`.
- **bfloat16** default on MPS/CUDA. fp16 overflows on MPS for Gemma3 due to post-norm scaling; bfloat16 is the only safe option there.
- **Idempotent pipelines.** Every multi-stage runner under `experiments/` skips a stage if its output artifacts already exist. Re-running is cheap.
