# ACL Anthology Abstracts Dataset

Builds a flat papers dataset from the [ACL Anthology](https://aclanthology.org/) using the
[`acl-anthology`](https://github.com/acl-org/acl-anthology-py) package, for embedding and
visualization in the platform.

## Files

| File | Purpose |
|---|---|
| `build_acl_dataset.py` | Main script: `AnthologyRepo` keeps a shallow clone of the anthology metadata repo pinned to the installed library's release tag (see "Anthology clone"); extracts one row per paper, writes parquet. `VenueNormalizer` resolves each paper's venue ids to one primary venue and groups the long tail (see "Venue columns"). |
| `acl_dataset_config.toml` | Default configuration, the **full corpus** (all venues): repo/clone paths, output path, filters (require abstract, year range, venue allowlist), `[venue]` grouping (`groups` acronyms keep their label, the rest become `other_label`). |
| `acl_dataset_config_emnlp.toml` | Variant config: EMNLP main conference only (`venues = ["emnlp"]`, excludes Findings). Dataset variants are separate config files passed as the script's single optional argument. No `[venue]` section, so `venue` is ungrouped. |
| `acl_dataset_config_emnlp_findings.toml` | Variant config: EMNLP main conference + Findings of EMNLP. Uses `id_patterns = ["findings-emnlp"]` OR'd with the venue allowlist, since Findings volumes only carry the shared `findings` venue id. |
| `test_build_acl_dataset.py` | Unit tests for the pure helpers (`markup_to_text`, `format_authors`, `VenueNormalizer`, `derive_track`, `paper_to_record`, `passes_filters`) using duck-typed fakes, plus `AnthologyRepo` against a local git fixture: no anthology data or network needed. Intentionally colocated with the script (not in `unit_tests/`) since it imports the sibling module directly. **Not tracked by git**: the root `.gitignore` rule `test*.py` matches it. |

## Output schema

One parquet row per paper (`interpretability_backend/resources/datasets/acl_abstracts.parquet`, gitignored):

| Column | Type | Notes |
|---|---|---|
| `acl_id` | str | Anthology id, e.g. `2020.acl-main.1` |
| `title` | str | Plain text (LaTeX/markup stripped via `as_text()`) |
| `year` | Int64 (nullable) | Null if non-numeric in source |
| `authors` | str | `First Last; First Last` |
| `venue` | str | Primary venue, grouped for coloring (e.g. `ACL`, `EMNLP`, `Other`); see "Venue columns" |
| `venue_detail` | str | Primary venue acronym, ungrouped (e.g. `BioNLP`, `BlackboxNLP`) |
| `track` | str | Volume type, derived from the parent volume title + flags (Findings also from the volume id, since `2025.findings-ijcnlp` is titled without "Findings"): `main`, `findings`, `demo`, `industry`, `tutorial`, `srw`, `shared_task`, `workshop`, `journal` |
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

## Anthology clone

First run shallow-clones the anthology metadata repo (~250 MB) to
`interpretability_backend/resources/acl_anthology/` (gitignored); later runs reuse it.

The clone is pinned to the repo tag of the installed `acl-anthology` release
(`py-v<version>`, e.g. `py-v1.3.3`), not to the default branch. The data's schema
evolves together with the library, and the default branch is regularly ahead of the latest
PyPI release (in October 2026, venue YAML `related_ids` broke 1.3.3, and July data had
already broken 1.2.0 with `AnthologyXMLError: unsupported element for <paper>`). The data at
a release tag always parses with that release. **To get newer papers, upgrade the library**
(`uv lock --upgrade-package acl-anthology`); the next run moves the clone to the new tag with
a depth-1 fetch + `reset --hard` (local edits in the clone are discarded). Don't `git pull`
the clone by hand: on a shallow clone that downloads every intermediate version of the
large XML files (~0.5 GB for one quarter of history). Each run prints the tag and commit
it built from.

`AnthologyRepo` refuses to run git in a folder that has `data/` but no `.git` of its own:
`git -C` would otherwise resolve to this project's repository and the reset would wipe its
working tree. Delete such a folder and rerun to re-clone.

The parquet is directly ingestible by the collections page local-file flow (embed the
`abstract` column; `title`/`year`/`authors`/`venue`/`track` become metadata for
coloring/filtering, `year` auto-detects for the temporal filter).

## Venue columns

The anthology gives each paper a list of venue ids, which is awkward as a color field
(~560 distinct comma-joined combinations): joint events list several ids (`emnlp`,
`ijcnlp`), workshop papers also carry the umbrella `ws` id, and Findings papers carry only
the shared `findings` id. `VenueNormalizer` picks one primary venue per paper, first match wins:

1. Findings: the conference in the id (`2023.findings-emnlp.42` → `EMNLP`). Main vs Findings
   stays visible in `track`.
2. The id's volume prefix, when it is one of the paper's venue ids (`2024.lrec-main.5` →
   `LREC` for LREC-COLING 2024; `2021.acl-long.3` → `ACL` for ACL-IJCNLP 2021).
3. The first venue id other than `ws`/`findings` (old-style ids such as `D19-1001` →
   `EMNLP`; `W19-5001` with `ws, bionlp` → `BioNLP`).
4. The first venue id. `WS` in `venue_detail` is therefore a catch-all for workshop volumes
   tagged only with the umbrella id (~840 papers: W16–W19 and D19-5xxx workshops, older
   MT Summit/AMTA workshops such as `2005.mtsummit-ebmt`).

Old-style joint ids follow the venue order in the XML (COLING-ACL 1998/2006 → `COLING`,
HLT/EMNLP 2005 → `HLT`).

The result is `venue_detail` (~420 distinct acronyms). `venue` keeps only the acronyms in the
config's `[venue] groups` and maps the rest (mostly workshops) to `other_label`, which keeps
the category count within one categorical palette. With no `groups`, `venue == venue_detail`.
No paper with an `emnlp` venue id resolves to another venue, so `venue == "EMNLP"` selects
the same papers as `acl_dataset_config_emnlp_findings.toml` (main conference + Findings).

Datasets built before October 2026 have the old `venue` column (comma-joined acronyms of
every venue id, Findings papers as `Findings`), including the Space's EMNLP collection.

Frontmatter and deleted papers are always skipped. Papers without abstracts are skipped by
default (`require_abstract = true`) — abstracts only exist in the anthology from roughly the
2000s onward, so disabling that filter roughly doubles the row count with abstract-less rows.
