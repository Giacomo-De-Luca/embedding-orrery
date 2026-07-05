"""Build a papers dataset (title, year, authors, venue, abstract) from the ACL Anthology.

Uses the `acl-anthology` package (https://github.com/acl-org/acl-anthology-py)
over a shallow clone of the official anthology metadata repository, and writes
a parquet file that the platform's local-file embedding flow can ingest directly.

Configuration lives in acl_dataset_config.toml next to this script; dataset
variants are separate config files passed as the single optional argument.

Run from the repository root:
    uv run python interpretability_backend/interpretability_experiments/ACL/build_acl_dataset.py
    uv run python interpretability_backend/interpretability_experiments/ACL/build_acl_dataset.py \\
        interpretability_backend/interpretability_experiments/ACL/acl_dataset_config_emnlp.toml
"""

import subprocess
import sys
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
from acl_anthology import Anthology
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "acl_dataset_config.toml"


# ---------- Pure extraction helpers (unit-tested, duck-typed) ----------


def markup_to_text(markup: Any) -> str:
    """Convert an acl-anthology MarkupText (or None) to a plain string."""
    if markup is None:
        return ""
    return markup.as_text().strip()


def format_authors(namespecs: Any) -> str:
    """Join author names as 'First Last; First Last'."""
    return "; ".join(spec.name.as_first_last() for spec in namespecs)


def resolve_venues(venue_ids: Any, venue_index: Mapping[str, Any]) -> str:
    """Map venue ids to their acronyms, falling back to the raw id."""
    names = []
    for vid in venue_ids:
        venue = venue_index.get(vid)
        names.append(venue.acronym if venue is not None else vid)
    return ", ".join(names)


# Ordered (keyword, track) rules for classifying a volume title; first hit wins.
# "student research" and "shared task" must precede "workshop"/the workshop flag,
# since their titles usually contain the word "workshop" too.
_TRACK_TITLE_RULES = (
    ("findings", "findings"),
    ("demonstration", "demo"),
    ("demo", "demo"),
    ("tutorial", "tutorial"),
    ("industry", "industry"),
    ("industrial", "industry"),
    ("student research", "srw"),
    ("shared task", "shared_task"),
    ("workshop", "workshop"),
)


def derive_track(volume_title: str, is_journal: bool, is_workshop: bool) -> str:
    """Classify a paper's volume into a track (main/findings/demo/industry/...).

    Title keywords work for both old-style (D19-3xxx) and new-style
    (2023.emnlp-demo.5) anthology ids, unlike parsing the id itself.
    """
    if is_journal:
        return "journal"
    lowered = volume_title.lower()
    for keyword, track in _TRACK_TITLE_RULES:
        if keyword in lowered:
            return track
    return "workshop" if is_workshop else "main"


def paper_to_record(paper: Any, venue_index: Mapping[str, Any]) -> dict[str, Any]:
    """Extract one flat dataset row from an anthology Paper."""
    year_str = paper.year
    year = int(year_str) if year_str and year_str.isdigit() else None
    volume = paper.parent
    track = derive_track(
        markup_to_text(volume.title),
        is_journal=volume.type.value == "journal",
        is_workshop=volume.is_workshop,
    )
    return {
        "acl_id": paper.full_id,
        "title": markup_to_text(paper.title),
        "year": year,
        "authors": format_authors(paper.namespecs),
        "venue": resolve_venues(paper.venue_ids, venue_index),
        "track": track,
        "abstract": markup_to_text(paper.abstract),
        "url": paper.web_url,
    }


def passes_filters(record: dict[str, Any], filters: dict[str, Any], venue_ids: Any) -> bool:
    """Apply the [filters] section of the config to an extracted record."""
    if filters.get("require_abstract", True) and not record["abstract"]:
        return False
    min_year = filters.get("min_year")
    max_year = filters.get("max_year")
    if (min_year is not None or max_year is not None) and record["year"] is None:
        return False
    if min_year is not None and record["year"] < min_year:
        return False
    if max_year is not None and record["year"] > max_year:
        return False
    allowed_venues = filters.get("venues") or []
    id_patterns = filters.get("id_patterns") or []
    if allowed_venues or id_patterns:
        # OR semantics: a paper is kept if its venue ids intersect the allowlist
        # or its anthology id contains any pattern (e.g. "findings-emnlp", which
        # is the only way to identify per-conference Findings volumes).
        venue_hit = bool(set(venue_ids) & set(allowed_venues))
        id_hit = any(pattern in record["acl_id"] for pattern in id_patterns)
        if not (venue_hit or id_hit):
            return False
    return True


# ---------- Repo + build orchestration ----------


def ensure_anthology_repo(url: str, path: Path) -> None:
    """Shallow-clone the anthology metadata repo if not already present."""
    if (path / "data").is_dir():
        print(f"Reusing existing anthology clone at {path}")
        return
    if path.exists():
        raise RuntimeError(
            f"{path} exists but has no data/ directory (interrupted clone?). "
            "Delete the folder and rerun."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Shallow-cloning {url} -> {path} (~250 MB, one-time)...")
    subprocess.run(
        ["git", "clone", "--depth", "1", url, str(path)],
        check=True,
    )


def build_dataset(config: dict[str, Any]) -> pd.DataFrame:
    repo_path = REPO_ROOT / config["repo"]["path"]
    ensure_anthology_repo(config["repo"]["url"], repo_path)

    print("Loading anthology index...")
    anthology = Anthology(datadir=repo_path / "data")
    venue_index = anthology.venues
    filters = config.get("filters", {})

    records = []
    skipped = 0
    for paper in tqdm(anthology.papers(), desc="Extracting papers", unit=" papers"):
        if paper.is_frontmatter or paper.is_deleted:
            skipped += 1
            continue
        record = paper_to_record(paper, venue_index)
        if not passes_filters(record, filters, paper.venue_ids):
            skipped += 1
            continue
        records.append(record)

    print(f"Kept {len(records)} papers, skipped {skipped} (frontmatter/deleted/filtered).")
    df = pd.DataFrame.from_records(records)
    if not df.empty:
        # Nullable Int64 keeps the parquet schema stable even when some years are null
        # (e.g. with require_abstract = false); plain from_records would coerce to float64.
        df["year"] = df["year"].astype("Int64")
    return df


def main() -> int:
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CONFIG_PATH
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    print(f"Using config: {config_path}")

    df = build_dataset(config)
    if df.empty:
        print("No papers matched the configured filters; nothing written.", file=sys.stderr)
        return 1

    output_path = REPO_ROOT / config["output"]["path"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Wrote {len(df)} rows to {output_path}")
    print(df[["year", "venue"]].describe(include="all").to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
