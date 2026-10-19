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

import re
import subprocess
import sys
import tomllib
import warnings
from collections.abc import Iterable, Mapping, Sequence
from importlib.metadata import version
from pathlib import Path
from typing import Any, Self

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


class VenueNormalizer:
    """Collapse a paper's anthology venue ids into one primary venue.

    Joint events list several venue ids (EMNLP-IJCNLP, LREC-COLING), workshop
    papers also carry the umbrella "ws" id, and Findings volumes carry only the
    shared "findings" id. The primary venue is resolved, in order, from:
    the Findings suffix of the anthology id (2023.findings-emnlp -> emnlp);
    the id's volume prefix when it is one of the paper's venue ids
    (2024.lrec-main -> lrec); the first non-umbrella venue id (old-style ids
    such as D19-1001 have no prefix); the first venue id.

    `group()` then keeps the configured `groups` acronyms and maps every other
    venue to `other_label`, so the venue column stays usable as a color field.
    """

    DEFAULT_OTHER_LABEL = "Other"
    _UMBRELLA_IDS = frozenset({"ws", "findings"})
    _NEW_STYLE_ID = re.compile(r"^\d{4}\.(?P<prefix>[^.-]+)-(?P<suffix>[^.]+)\.")

    def __init__(
        self,
        venue_index: Mapping[str, Any],
        groups: Iterable[str] = (),
        other_label: str = DEFAULT_OTHER_LABEL,
    ) -> None:
        self._venue_index = venue_index
        self._groups = frozenset(groups)
        self._other_label = other_label

    @classmethod
    def from_config(cls, section: Mapping[str, Any], venue_index: Mapping[str, Any]) -> Self:
        """Build from the config's [venue] section (missing keys = no grouping).

        Warns about group acronyms the anthology doesn't know, since a typo
        would otherwise silently send that venue to `other_label`.
        """
        groups = section.get("groups", [])
        if not isinstance(groups, list) or not all(isinstance(g, str) for g in groups):
            raise ValueError(f"[venue] groups must be a list of acronyms, got {groups!r}")
        known = {venue.acronym for venue in venue_index.values()}
        unknown = sorted(set(groups) - known)
        if unknown:
            warnings.warn(f"[venue] groups not in the anthology: {unknown}", stacklevel=2)
        return cls(
            venue_index,
            groups=groups,
            other_label=section.get("other_label", cls.DEFAULT_OTHER_LABEL),
        )

    def primary_id(self, acl_id: str, venue_ids: Sequence[str]) -> str:
        """Venue id of the paper's primary venue ("" when it has none)."""
        match = self._NEW_STYLE_ID.match(acl_id)
        if match:
            if match["prefix"] == "findings":
                return match["suffix"]
            if match["prefix"] in venue_ids:
                return match["prefix"]
        specific = [vid for vid in venue_ids if vid not in self._UMBRELLA_IDS]
        if specific:
            return specific[0]
        return venue_ids[0] if venue_ids else ""

    def detail(self, acl_id: str, venue_ids: Sequence[str]) -> str:
        """Acronym of the primary venue, ungrouped."""
        return self._acronym(self.primary_id(acl_id, venue_ids))

    def group(self, detail: str) -> str:
        """Keep a configured group's acronym; map anything else to `other_label`."""
        if not self._groups or detail in self._groups:
            return detail
        return self._other_label

    def _acronym(self, venue_id: str) -> str:
        """Map a venue id to its acronym, falling back to the raw id."""
        venue = self._venue_index.get(venue_id)
        return venue.acronym if venue is not None else venue_id


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


def derive_track(
    volume_title: str, is_journal: bool, is_workshop: bool, is_findings: bool = False
) -> str:
    """Classify a paper's volume into a track (main/findings/demo/industry/...).

    Title keywords work for both old-style (D19-3xxx) and new-style
    (2023.emnlp-demo.5) anthology ids, unlike parsing the id itself. Findings
    volumes are also flagged from their id (`is_findings`), since some titles
    omit the word (2025.findings-ijcnlp).
    """
    if is_journal:
        return "journal"
    if is_findings:
        return "findings"
    lowered = volume_title.lower()
    for keyword, track in _TRACK_TITLE_RULES:
        if keyword in lowered:
            return track
    return "workshop" if is_workshop else "main"


def paper_to_record(paper: Any, normalizer: VenueNormalizer) -> dict[str, Any]:
    """Extract one flat dataset row from an anthology Paper."""
    year_str = paper.year
    year = int(year_str) if year_str and year_str.isdigit() else None
    volume = paper.parent
    track = derive_track(
        markup_to_text(volume.title),
        is_journal=volume.type.value == "journal",
        is_workshop=volume.is_workshop,
        is_findings=".findings-" in volume.full_id,
    )
    venue_detail = normalizer.detail(paper.full_id, paper.venue_ids)
    return {
        "acl_id": paper.full_id,
        "title": markup_to_text(paper.title),
        "year": year,
        "authors": format_authors(paper.namespecs),
        "venue": normalizer.group(venue_detail),
        "venue_detail": venue_detail,
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


class AnthologyRepo:
    """Shallow clone of the anthology metadata repo, pinned to a library release.

    The schema of the data/ files moves with the `acl-anthology` package, so
    the repo's default branch is often ahead of what the latest PyPI release
    can parse. Each release is tagged `py-v<version>` in the repo and its data parses
    with that release, so the clone follows the installed package's tag:
    upgrading the package (`uv lock --upgrade-package acl-anthology`) moves the
    data forward.

    Moving an existing clone fetches only the tagged commit (depth 1) and resets
    the checkout to it, discarding local edits; a plain `git pull` would
    download every intermediate version of the large XML files instead.
    """

    def __init__(self, url: str, path: Path, tag: str) -> None:
        self.url = url
        self.path = path
        self.tag = tag

    @staticmethod
    def library_tag() -> str:
        """Repo tag of the installed acl-anthology release, e.g. `py-v1.3.3`."""
        return f"py-v{version('acl-anthology')}"

    def ensure(self) -> None:
        """Clone at the tag, reuse a clone already there, or move a clone to it."""
        if not (self.path / "data").is_dir():
            self._clone()
        else:
            self._require_own_checkout()
            if self._commit("HEAD") != self._commit(f"{self.tag}^{{commit}}"):
                print(f"Moving anthology clone at {self.path} to {self.tag} (depth-1 fetch)...")
                self._git("fetch", "--depth", "1", "origin", "tag", self.tag)
                self._git("reset", "--hard", self.tag)
        print(f"Anthology data: {self.tag} ({self._commit('HEAD')[:10]}) at {self.path}")

    def _clone(self) -> None:
        if self.path.exists():
            raise RuntimeError(
                f"{self.path} exists but has no data/ directory (interrupted clone?). "
                "Delete the folder and rerun."
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Shallow-cloning {self.url} at {self.tag} -> {self.path} (~250 MB, one-time)...")
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", self.tag, self.url, str(self.path)],
            check=True,
        )

    def _require_own_checkout(self) -> None:
        """Refuse to run git unless `path` is itself the top of a checkout.

        The clone lives inside this project's repository (gitignored): without
        its own .git, `git -C path` would act on the project repository, and the
        `reset --hard` would wipe uncommitted work there.
        """
        top = self._git("rev-parse", "--show-toplevel", check=False)
        if not top or Path(top).resolve() != self.path.resolve():
            raise RuntimeError(
                f"{self.path} has a data/ directory but is not its own git clone. "
                "Delete the folder and rerun to re-clone."
            )

    def _commit(self, rev: str) -> str:
        """Commit id of `rev`, or "" when it isn't available locally."""
        return self._git("rev-parse", "--verify", "--quiet", rev, check=False)

    def _git(self, *args: str, check: bool = True) -> str:
        result = subprocess.run(
            ["git", "-C", str(self.path), *args], capture_output=True, text=True
        )
        if check and result.returncode != 0:
            raise RuntimeError(
                f"git {' '.join(args)} failed in {self.path}: {result.stderr.strip()}"
            )
        return result.stdout.strip() if result.returncode == 0 else ""


def build_dataset(config: dict[str, Any]) -> pd.DataFrame:
    repo = config["repo"]
    repo_path = REPO_ROOT / repo["path"]
    AnthologyRepo(repo["url"], repo_path, AnthologyRepo.library_tag()).ensure()

    print("Loading anthology index...")
    anthology = Anthology(datadir=repo_path / "data")
    normalizer = VenueNormalizer.from_config(config.get("venue", {}), anthology.venues)
    filters = config.get("filters", {})

    records = []
    skipped = 0
    for paper in tqdm(anthology.papers(), desc="Extracting papers", unit=" papers"):
        if paper.is_frontmatter or paper.is_deleted:
            skipped += 1
            continue
        record = paper_to_record(paper, normalizer)
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
    print(df["venue"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
