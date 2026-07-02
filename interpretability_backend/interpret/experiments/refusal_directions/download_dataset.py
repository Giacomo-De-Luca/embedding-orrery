"""Download the refusal-direction datasets from the upstream Arditi et al. repo.

The harmful/harmless splits and processed eval sets used by this experiment are
*not* tracked in this repo (they contain harmful jailbreak prompts, and the whole
`resources/` tree is gitignored). They are, however, committed upstream at
github.com/andyrdt/refusal_direction under `dataset/{splits,processed}/`. This
module fetches them at a pinned commit into the gitignored target directories
that `RefusalConfig` already points at — so a clean checkout can reproduce the
experiment without committing the prompts.

Filenames mirror upstream exactly (note: upstream's `malicious_instruct.json` is
referenced as `maliciousinstruct` in the reference repo's `load_dataset.py`; this
project addresses eval sets by their on-disk filename, so we keep the underscore).

Run with defaults:
    uv run python -m interpret.experiments.refusal_directions.download_dataset
"""

from __future__ import annotations

import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from interpret.experiments.refusal_directions.config import RefusalConfig

# Six harm × split files + eight processed eval sets, named exactly as upstream.
SPLIT_FILES: tuple[str, ...] = tuple(
    f"{harmtype}_{split}.json"
    for harmtype in ("harmful", "harmless")
    for split in ("train", "val", "test")
)
PROCESSED_FILES: tuple[str, ...] = (
    "advbench.json",
    "alpaca.json",
    "harmbench_test.json",
    "harmbench_val.json",
    "jailbreakbench.json",
    "malicious_instruct.json",
    "strongreject.json",
    "tdc2023.json",
)


@dataclass
class RefusalDownloadConfig:
    """Where to fetch the refusal datasets from, and where to write them.

    Defaults pin the upstream commit that this project's data was vendored from,
    and target the same gitignored directories `RefusalConfig` reads
    (`resources/refusal_direction/{splits,processed}`).
    """

    repo: str = "andyrdt/refusal_direction"
    # Pinned to the commit the local `references/refusal_direction` clone was at.
    commit: str = "9d852fae1a9121c78b29142de733cb1340770cc3"

    splits_dir: Path = field(default_factory=lambda: RefusalConfig().splits_dir)
    eval_dir: Path = field(default_factory=lambda: RefusalConfig().eval_dir)

    overwrite: bool = False  # re-download files that already exist

    def raw_url(self, subdir: str, filename: str) -> str:
        return (
            f"https://raw.githubusercontent.com/{self.repo}/{self.commit}"
            f"/dataset/{subdir}/{filename}"
        )


def _download_file(url: str, dest: Path, *, overwrite: bool) -> bool:
    """Fetch `url` to `dest`. Returns True if downloaded, False if skipped.

    Reads the full response before writing, so a failure never leaves a
    truncated file (re-running cleanly resumes with overwrite=False).
    """
    if dest.exists() and not overwrite:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url) as resp:  # noqa: S310 — pinned https GitHub URL
            data = resp.read()
    except (urllib.error.HTTPError, urllib.error.URLError) as e:
        raise RuntimeError(f"Failed to download {dest.name} from {url}: {e}") from e
    dest.write_bytes(data)
    return True


def download_refusal_datasets(config: RefusalDownloadConfig | None = None) -> None:
    """Download all split + processed files described by `config`."""
    cfg = config or RefusalDownloadConfig()
    targets = [
        ("splits", name, cfg.splits_dir / name) for name in SPLIT_FILES
    ] + [
        ("processed", name, cfg.eval_dir / name) for name in PROCESSED_FILES
    ]

    print(f"Downloading refusal datasets from {cfg.repo}@{cfg.commit[:8]}")
    downloaded = skipped = 0
    for subdir, name, dest in targets:
        if _download_file(cfg.raw_url(subdir, name), dest, overwrite=cfg.overwrite):
            print(f"  + {subdir}/{name}")
            downloaded += 1
        else:
            print(f"  = {subdir}/{name} (exists)")
            skipped += 1
    print(f"Done: {downloaded} downloaded, {skipped} skipped.")


if __name__ == "__main__":
    download_refusal_datasets()
