# ACL Anthology Abstracts Dataset

Builds a flat papers dataset from the [ACL Anthology](https://aclanthology.org/) using the
[`acl-anthology`](https://github.com/acl-org/acl-anthology-py) package, for embedding and
visualization in the platform.

## Files

| File | Purpose |
|---|---|
| `build_acl_dataset.py` | Main script: clones the anthology metadata repo (one-time, shallow), extracts one row per paper, writes parquet. |
| `acl_dataset_config.toml` | Default configuration: repo/clone paths, output path, filters (require abstract, year range, venue allowlist). |
| `acl_dataset_config_emnlp.toml` | Variant config: EMNLP main conference only (`venues = ["emnlp"]`, excludes Findings). Dataset variants are separate config files passed as the script's single optional argument. |
| `acl_dataset_config_emnlp_findings.toml` | Variant config: EMNLP main conference + Findings of EMNLP. Uses `id_patterns = ["findings-emnlp"]` OR'd with the venue allowlist, since Findings volumes only carry the shared `findings` venue id. |
| `test_build_acl_dataset.py` | Unit tests for the pure helpers (`markup_to_text`, `format_authors`, `resolve_venues`, `paper_to_record`, `passes_filters`) using duck-typed fakes — no anthology data needed. Intentionally colocated with the script (not in `unit_tests/`) since it imports the sibling module directly. |

## Output schema

One parquet row per paper (`interpretability_backend/resources/datasets/acl_abstracts.parquet`, gitignored):

| Column | Type | Notes |
|---|---|---|
| `acl_id` | str | Anthology id, e.g. `2020.acl-main.1` |
| `title` | str | Plain text (LaTeX/markup stripped via `as_text()`) |
| `year` | Int64 (nullable) | Null if non-numeric in source |
| `authors` | str | `First Last; First Last` |
| `venue` | str | Comma-joined venue acronyms (e.g. `ACL`, `EMNLP, WMT`) |
| `abstract` | str | Plain text; rows without abstracts dropped by default |
| `url` | str | aclanthology.org landing page |

## Usage

```bash
# From the repository root
uv run python interpretability_backend/interpretability_experiments/ACL/build_acl_dataset.py

# Variant dataset (EMNLP-only)
uv run python interpretability_backend/interpretability_experiments/ACL/build_acl_dataset.py \
    interpretability_backend/interpretability_experiments/ACL/acl_dataset_config_emnlp.toml

# Tests
cd interpretability_backend/interpretability_experiments/ACL && uv run pytest test_build_acl_dataset.py
```

First run shallow-clones the anthology metadata repo (~250 MB) to
`interpretability_backend/resources/acl_anthology/` (gitignored). Subsequent runs reuse it;
refresh with `git -C interpretability_backend/resources/acl_anthology pull` or delete the folder.

The parquet is directly ingestible by the collections page local-file flow (embed the
`abstract` column; `title`/`year`/`authors`/`venue` become metadata for coloring/filtering,
`year` auto-detects for the temporal filter).

Frontmatter and deleted papers are always skipped. Papers without abstracts are skipped by
default (`require_abstract = true`) — abstracts only exist in the anthology from roughly the
2000s onward, so disabling that filter roughly doubles the row count with abstract-less rows.
