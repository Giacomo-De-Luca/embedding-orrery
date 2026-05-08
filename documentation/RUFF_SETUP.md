# Ruff Setup

Ruff replaces the would-be combination of `flake8 + isort + pyupgrade + Black` for the Python backend with a single Rust-based tool.

## Why Ruff

- One tool, one config, ~10–100× faster than the legacy stack.
- Astral-built — pairs naturally with the existing `uv` dependency manager.
- `ASYNC` rule family catches blocking calls inside `async def` resolvers (relevant for the FastAPI + Strawberry GraphQL layer and WebSocket subscriptions).
- `UP` rule family modernizes typing syntax for Python 3.12 (`Dict[str, X]` → `dict[str, X]`, `Optional[X]` → `X | None`).

## Active Rule Set

Defined in the root `pyproject.toml` under `[tool.ruff.lint]`:

| Family | What it covers |
|---|---|
| `E`, `W` | pycodestyle errors and warnings |
| `F`     | pyflakes — unused imports, undefined names |
| `I`     | isort — import sorting and grouping |
| `B`     | flake8-bugbear — likely-bug patterns |
| `UP`    | pyupgrade — modernize syntax for the `target-version` |
| `ASYNC` | flake8-async — blocking calls in `async def` |

Globally ignored:

- `E501` — line-too-long (the formatter already wraps; long strings/comments allowed).
- `B008` — function calls in argument defaults (FastAPI `Depends()`, Strawberry `field()` legitimately use this).

Per-file ignores:

- `unit_tests/**`, `test/**` — `F401`, `F811` (test fixtures look like unused imports / redefinitions).
- `**/__init__.py` — `F401` (re-exports are not "unused").

Excluded paths (`extend-exclude`):

- `interpretability_backend/sae_inference/gemma_pytorch` — git submodule, not our code.
- `interpretability_backend/interpretability_experiments/WordNet/wordnet_data` — large data dumps.
- `embedding-atlas`, `tensorboard` — vendored reference projects.
- `embedding_visualization` — JS/TS frontend (no Python).
- `*.ipynb` — Jupyter notebooks. Ruff lints notebooks per-cell and cannot resolve cross-cell variable definitions, so `F821` false positives dominate; one notebook also fails the formatter's parser.

## Commands

```bash
# Lint (read-only)
uv run ruff check interpretability_backend/

# Lint + apply safe autofixes
uv run ruff check interpretability_backend/ --fix

# Format
uv run ruff format interpretability_backend/

# Verify-only (CI-style, no rewrite)
uv run ruff format --check interpretability_backend/

# Statistics breakdown
uv run ruff check interpretability_backend/ --statistics
```

## Initial Rollout (one-time)

Applied 2026-05-05:
- 990 of 1044 baseline violations auto-fixed.
- 79 of 101 backend `.py` files reformatted.
- One real bug surfaced and fixed: missing `import uvicorn` at the top of [interpretability_backend/backend/main.py](../interpretability_backend/backend/main.py).
- Two pre-existing broken files isolated by the user during rollout: `extract_topics.py` (renamed to `.txt`, was vendored BERTopic with broken indentation) and `steer_tester.ipynb` (deleted, was empty stub).

## Residual Cleanup Queue

61 violations remain after the initial pass. None block landing the change; each requires human judgment to fix correctly. Address opportunistically:

| Code | Count | Notes |
|---|---|---|
| `B905` | 27 | `zip(...)` without explicit `strict=` — pick `True` (raise on length mismatch) or `False` (silent truncation, current behavior) per call site. |
| `F841` | 13 | Unused local variables — could be debug leftovers or assignment-with-side-effect. Check each. |
| `B904` | 5 | `raise X` inside `except` without `from`. Add `from err` (preserve chain) or `from None` (suppress). |
| `E731` | 4 | `f = lambda: ...` → `def f(): ...`. Trivial. |
| `B007` | 3 | Loop variable not used — rename to `_`. |
| `E741` | 3 | Ambiguous names (`l`, `I`, `O`). Often scientific code (length, identity matrix). Rename or per-file ignore. |
| `E402` | 2 | Import not at top — typically intentional `sys.path.insert` patterns. Likely add `# noqa: E402` with reason. |
| `F821` | 1 | `gemma_pytorch.py:220` references `torch.Tensor` in a forward-ref annotation but only imports `torch` inside a function (violates the project's "no in-function imports" rule). Fix by lifting `import torch` to module top during a separate cleanup of `sae_inference/inference/`. |
| `B006` | 1 | Mutable default argument — possibly intentional, needs review. |
| `B011` | 1 | `assert False` → `raise AssertionError(...)`. |
| `E712` | 1 | `== True` → `if x:`. |

## Extending the Rule Set

When the team is comfortable with the baseline, additional rule families to consider:

- `SIM` — flake8-simplify (collapses redundant patterns).
- `PERF` — perflint (loop/comprehension efficiency).
- `RUF` — ruff-specific lints, including `RUF100` (drop stale `# noqa`).
- `C4` — flake8-comprehensions.
- `PT` — flake8-pytest-style.
- `N` — pep8-naming.
- `D` — pydocstyle, paired with `[tool.ruff.lint.pydocstyle] convention = "google"` to match the codebase's existing docstring style.

Add new families incrementally, run `--fix --statistics` to size the change, fix or ignore as appropriate before merging the next tier.

## Project-Specific Rules This Enforces

- All Python imports go at the top of the file. The CLAUDE.md root rule "never import modules inside functions unless strictly necessary" is partially enforceable via Ruff's `PLC0415` (in the `PL` family, currently disabled). To turn this into a hard lint, add `"PLC0415"` to `select` and revisit existing in-function imports.
