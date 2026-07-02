"""XKCD colour survey manifest: 954 named colours with RGB + LAB targets.

Single-source manifest. `get_rated_samples(source, column)` accepts
``"xkcd"`` as the source string and any of ``{R, G, B, L, a, b}`` as
column. Probe targets are the six colour channels.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from interpret.probing.manifests.manifest_base import ManifestBuilder

_RGB_COLS = ("R", "G", "B")
_LAB_COLS = ("L", "a", "b")  # mapped from Lab_L, Lab_a, Lab_b


@dataclass
class XKCDPaths:
    """Path to the XKCD survey CSV."""

    csv: Path = Path("resources/colour_names_to_codes/xkcd_colours.csv")


class XKCDColourManifestBuilder(ManifestBuilder):
    """954 XKCD colour names with RGB + LAB regression targets."""

    SOURCE = "xkcd"

    def __init__(
        self,
        paths: XKCDPaths | None = None,
        lowercase: bool = True,
    ) -> None:
        cfg = paths if paths is not None else XKCDPaths()
        df = pd.read_csv(cfg.csv)

        names = df["Name"].astype(str).str.strip()
        if lowercase:
            names = names.str.lower()

        df = df.rename(
            columns={"Lab_L": "L", "Lab_a": "a", "Lab_b": "b"},
        )
        df["colour_name"] = names
        df = df[df["colour_name"].str.len() > 0].reset_index(drop=True)

        # Duplicate detection (raise loudly — no silent drop).
        dups = df["colour_name"].value_counts()
        dups = dups[dups > 1]
        if not dups.empty:
            raise ValueError(
                f"XKCD: duplicate colour names: "
                f"{dups.head(5).to_dict()}",
            )

        self._df = df[["colour_name", *_RGB_COLS, *_LAB_COLS]].copy()
        self._samples: list[str] = self._df["colour_name"].tolist()
        print(
            f"XKCDColourManifestBuilder: {len(self._samples)} colours",
        )

    @property
    def prompt_column(self) -> str:
        return "colour_name"

    @property
    def target_columns(self) -> list[str]:
        return [*_RGB_COLS, *_LAB_COLS]

    @property
    def samples(self) -> list[str]:
        return self._samples

    def build_dataframe(self) -> pd.DataFrame:
        return self._df.copy()

    def get_rated_samples(
        self, source: str, column: str,
    ) -> tuple[list[str], np.ndarray]:
        if source != self.SOURCE:
            raise ValueError(
                f"XKCD manifest only knows source {self.SOURCE!r}, "
                f"got {source!r}",
            )
        if column not in self._df.columns:
            raise ValueError(
                f"Unknown column {column!r}. "
                f"Valid: {self.target_columns}",
            )
        return (
            self._df["colour_name"].tolist(),
            self._df[column].to_numpy(dtype=np.float32),
        )
