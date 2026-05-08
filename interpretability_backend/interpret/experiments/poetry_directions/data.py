"""CSV loaders for the poetry-direction experiments.

Two shapes:

- ``load_pairs`` reads ``paraphrased_poems_aligned.csv`` and returns two
  same-length lists ``(poems, paraphrases)`` aligned by row order.
- ``load_filtered_subset`` reads a prompts file (CSV or TSV — separator is
  inferred from the suffix) with column-pruned ``read_csv``, filters by
  exact-match column values, deduplicates by the configured text column,
  and optionally subsamples.
"""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd


def load_pairs(
    csv_path: Path,
    *,
    max_n: int | None = None,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Read a paired-poem CSV and return ``(class_a_texts, class_b_texts)``.

    Class A is the ``poem`` column, class B is the ``paraphrase`` column.
    Pairing is preserved by row order (caller can use it for variance
    reduction; the extraction code does not exploit it).
    """
    df = pd.read_csv(csv_path, usecols=["item_id", "poem", "paraphrase"])
    df = df.dropna(subset=["poem", "paraphrase"])
    if max_n is not None and max_n < len(df):
        df = df.sample(n=max_n, random_state=seed).reset_index(drop=True)
    return df["poem"].tolist(), df["paraphrase"].tolist()


def load_filtered_subset(
    csv_path: Path,
    *,
    filters: dict[str, str],
    max_n: int | None = None,
    seed: int = 42,
    text_column: str = "prompt_text",
) -> list[str]:
    """Load `text_column` values from a prompts file matching `filters`.

    Separator is inferred from the file suffix (``.tsv`` → tab, else comma).
    We use ``usecols`` to load only what we need, then filter by exact-match
    on each ``filters`` key, deduplicate by ``text_column``, and optionally
    subsample.
    """
    sep = "\t" if Path(csv_path).suffix.lower() == ".tsv" else ","
    needed_cols = list({text_column, *filters.keys()})
    df = pd.read_csv(csv_path, sep=sep, usecols=needed_cols)
    for key, value in filters.items():
        df = df[df[key] == value]
    df = df.drop_duplicates(text_column)
    df = df.dropna(subset=[text_column])
    if max_n is not None and max_n < len(df):
        df = df.sample(n=max_n, random_state=seed)
    return df[text_column].tolist()


def load_classes_for_experiment(
    name: str,
    spec: dict,
    *,
    poems_csv: Path,
    prompts_csv: Path,
    max_per_class: int | None,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Dispatch on `EXPERIMENTS[name]['loader']` and return (class_a, class_b)."""
    loader = spec["loader"]
    args = spec["args"]
    if loader == "pairs":
        return load_pairs(poems_csv, max_n=max_per_class, seed=seed)
    if loader == "filtered":
        text_column = args["text_column"]
        a = load_filtered_subset(
            prompts_csv,
            filters=args["class_a_filter"],
            max_n=max_per_class,
            seed=seed,
            text_column=text_column,
        )
        b = load_filtered_subset(
            prompts_csv,
            filters=args["class_b_filter"],
            max_n=max_per_class,
            seed=seed + 1,
            text_column=text_column,
        )
        return a, b
    raise ValueError(f"unknown loader: {loader!r} for experiment {name!r}")


def shuffle_into(items: list[str], n: int, seed: int) -> list[str]:
    """Deterministic random sample of `n` items, full list if `n >= len(items)`."""
    if n >= len(items):
        return list(items)
    rng = random.Random(seed)
    return rng.sample(items, n)
