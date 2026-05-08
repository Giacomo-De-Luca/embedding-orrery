"""Format the prior_refusal_eval TSV into a readable markdown document.

Reads the TSV produced by ``interpret.experiments.poetry_directions.prior_refusal_eval`` and
emits a markdown file with one section per prompt, grouped by prior safety
status and sorted by (hazard, sample_id). Each section shows the prompt and
both responses (unsteered and steered) in fenced code blocks so formatting
is preserved and any markdown / backticks in the content can't break the
surrounding doc.

Edit the constants near the top to filter or change paths.

Run::

    uv run python -m interpret.experiments.poetry_directions.prior_refusal_to_md
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


INPUT_TSV = Path(
    "resources/experiments/poetry_directions/poetry_prose/"
    "prior_refusal_steered_responses.tsv"
)
OUTPUT_MD: Path | None = None  # None -> derive from INPUT_TSV (.md sibling)
STATUS_FILTER: str | None = None  # None | "safe" | "unsafe"
HAZARD_FILTER: str | None = None  # None | exact-match hazard string

_REQUIRED_COLUMNS = (
    "sample_id",
    "hazard",
    "prompt",
    "prior_safety_status",
    "response_unsteered",
    "response_steered",
)


def _safe_text(value) -> str:
    """Render a possibly-NaN cell as a plain string."""
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value)


def _fence_for(text: str) -> str:
    """Choose a code-fence wide enough to dodge any backtick run inside `text`."""
    longest = max((len(m.group(0)) for m in re.finditer(r"`+", text)), default=0)
    return "`" * max(longest + 1, 3)


def _format_block(label: str, text: str) -> list[str]:
    fence = _fence_for(text)
    return [f"**{label}**", "", fence, text, fence, ""]


def _build_markdown(df: pd.DataFrame, source_path: Path) -> str:
    lines: list[str] = []
    lines.append("# Prior-refusal steered responses")
    lines.append("")
    lines.append(f"Source: `{source_path}`")
    lines.append(f"Total rows: **{len(df)}**")
    if STATUS_FILTER is not None:
        lines.append(f"Filter: `prior_safety_status == {STATUS_FILTER!r}`")
    if HAZARD_FILTER is not None:
        lines.append(f"Filter: `hazard == {HAZARD_FILTER!r}`")
    lines.append("")
    lines.append(
        "Sections are grouped by `prior_safety_status` and then sorted by "
        "`(hazard, sample_id)`. Prompt and responses are in fenced code blocks "
        "so any markdown / backticks inside them render verbatim."
    )
    lines.append("")

    for status, group in df.groupby("prior_safety_status", sort=False):
        lines.append(f"## prior_safety_status: `{status}` ({len(group)} rows)")
        lines.append("")
        for _, row in group.iterrows():
            sid = _safe_text(row["sample_id"])
            hazard = _safe_text(row["hazard"]) or "(unspecified)"
            lines.append(f"### {sid} — hazard: {hazard}")
            lines.append("")
            lines.extend(_format_block("Prompt", _safe_text(row["prompt"])))
            lines.extend(
                _format_block("Unsteered response", _safe_text(row["response_unsteered"]))
            )
            lines.extend(
                _format_block("Steered response", _safe_text(row["response_steered"]))
            )
            lines.append("---")
            lines.append("")

    return "\n".join(lines)


def main() -> None:
    if not INPUT_TSV.exists():
        raise FileNotFoundError(
            f"input TSV not found at {INPUT_TSV}. Run "
            "`interpret.experiments.poetry_directions.prior_refusal_eval` first."
        )

    df = pd.read_csv(INPUT_TSV, sep="\t", dtype=str)
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"input TSV missing required columns: {missing}")

    if STATUS_FILTER is not None:
        df = df[df["prior_safety_status"] == STATUS_FILTER]
    if HAZARD_FILTER is not None:
        df = df[df["hazard"] == HAZARD_FILTER]
    df = df.sort_values(
        ["prior_safety_status", "hazard", "sample_id"], na_position="last"
    )

    out_path = OUTPUT_MD if OUTPUT_MD is not None else INPUT_TSV.with_suffix(".md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    markdown = _build_markdown(df, INPUT_TSV)
    out_path.write_text(markdown)
    print(f"wrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
