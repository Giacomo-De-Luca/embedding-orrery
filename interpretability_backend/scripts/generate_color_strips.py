"""
Generate colorscale JSON files for direct color column visualization.

Produces Crameri-compatible JSON colorscales that map 3D color spaces to 1D strips
using Hilbert curves for locality preservation.

Usage:
    uv run python interpretability_backend/scripts/generate_color_strips.py
    uv run python interpretability_backend/scripts/generate_color_strips.py --custom-colors path/to/colors.csv
"""

import argparse
import colorsys
import csv
import json
import math
from pathlib import Path

from hilbertcurve.hilbertcurve import HilbertCurve

# Output directory: frontend colormaps
COLORMAPS_DIR = (
    Path(__file__).resolve().parents[2]
    / "embedding_visualization"
    / "lib"
    / "colorMaps"
    / "colormaps"
)


# ---------------------------------------------------------------------------
# Colorscale generators
# ---------------------------------------------------------------------------


def rgb_to_plotly_string(r: int, g: int, b: int) -> str:
    return f"rgb({r},{g},{b})"


def build_crameri_json(name: str, colors_rgb: list[tuple[int, int, int]]) -> dict:
    """Build a Crameri-compatible JSON dict from a list of RGB tuples."""
    n = len(colors_rgb)
    colors = [rgb_to_plotly_string(*c) for c in colors_rgb]
    plotly = []
    for i, c in enumerate(colors):
        pos = round(i / (n - 1), 6) if n > 1 else 0
        plotly.append([pos, c])
    return {
        "name": name,
        "type": "sequential",
        "colors": colors,
        "plotly": plotly,
    }


def generate_hilbert_rgb(grid_size: int = 8) -> list[tuple[int, int, int]]:
    """
    Generate an RGB colorscale by walking a 3D Hilbert curve through
    an n×n×n RGB grid. grid_size must be a power of 2.
    """
    p = int(math.log2(grid_size))
    hc = HilbertCurve(p, 3)  # p iterations, 3 dimensions
    step = 255 / (grid_size - 1)

    points = []
    for x in range(grid_size):
        for y in range(grid_size):
            for z in range(grid_size):
                r = int(round(x * step))
                g = int(round(y * step))
                b = int(round(z * step))
                idx = hc.distance_from_point([x, y, z])
                points.append((idx, r, g, b))
    points.sort(key=lambda t: t[0])
    return [(r, g, b) for _, r, g, b in points]


def generate_hue_sat(hue_steps: int = 20, sat_steps: int = 18) -> list[tuple[int, int, int]]:
    """
    Generate a Hue×Saturation colorscale with fixed luminosity=0.5,
    ordered by a 2D Hilbert curve through the HS grid.
    """
    # Grid size for Hilbert must be power of 2 >= max(hue_steps, sat_steps)
    hilbert_n = 1
    while hilbert_n < max(hue_steps, sat_steps):
        hilbert_n *= 2

    p = int(math.log2(hilbert_n))
    hc = HilbertCurve(p, 2)  # p iterations, 2 dimensions

    points = []
    for hi in range(hue_steps):
        for si in range(sat_steps):
            hue = hi / hue_steps  # 0-1
            sat = 0.1 + (si / (sat_steps - 1)) * 0.9  # 0.1-1.0 (skip very low saturation)
            r, g, b = colorsys.hls_to_rgb(hue, 0.5, sat)
            idx = hc.distance_from_point([hi, si])
            points.append((idx, int(round(r * 255)), int(round(g * 255)), int(round(b * 255))))
    points.sort(key=lambda t: t[0])
    return [(r, g, b) for _, r, g, b in points]


def generate_xkcd_strip(csv_path: str) -> list[tuple[int, int, int]]:
    """
    Load colors from the XKCD color survey CSV and order them
    by a 3D Hilbert curve through RGB space.
    """
    colors = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            r, g, b = int(row["R"]), int(row["G"]), int(row["B"])
            colors.append((r, g, b))

    # 8-bit RGB: values 0-255, need p=8 for 256 divisions per axis
    hc = HilbertCurve(8, 3)
    indexed = [(hc.distance_from_point([r, g, b]), r, g, b) for r, g, b in colors]
    indexed.sort(key=lambda t: t[0])
    return [(r, g, b) for _, r, g, b in indexed]


# ---------------------------------------------------------------------------
# Index.json updater
# ---------------------------------------------------------------------------


def update_index(names_and_counts: list[tuple[str, int]]) -> None:
    """Add entries to index.json for the generated colormaps."""
    index_path = COLORMAPS_DIR / "index.json"
    with open(index_path) as f:
        index = json.load(f)
    for name, num_colors in names_and_counts:
        index[name] = {"type": "sequential", "numColors": num_colors}
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
        f.write("\n")
    print(f"Updated {index_path} with {len(names_and_counts)} new entries")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate color strip JSON files for Plotly colorscales"
    )
    parser.add_argument(
        "--custom-colors",
        type=str,
        default=None,
        help="Path to CSV with columns: Name,Hex,R,G,B,...",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=8,
        help="RGB grid size for Hilbert strip (power of 2, default 8 → 512 colors)",
    )
    parser.add_argument(
        "--hue-steps", type=int, default=20, help="Hue steps for hue-sat strip (default 20)"
    )
    parser.add_argument(
        "--sat-steps", type=int, default=18, help="Saturation steps for hue-sat strip (default 18)"
    )
    args = parser.parse_args()

    COLORMAPS_DIR.mkdir(parents=True, exist_ok=True)
    generated = []

    # 1. Hilbert RGB strip
    print(f"Generating hilbertColor ({args.grid_size}^3 = {args.grid_size**3} colors)...")
    hilbert_colors = generate_hilbert_rgb(args.grid_size)
    hilbert_json = build_crameri_json("hilbertColor", hilbert_colors)
    out_path = COLORMAPS_DIR / "hilbertColor.json"
    with open(out_path, "w") as f:
        json.dump(hilbert_json, f)
    print(f"  Wrote {out_path} ({len(hilbert_colors)} colors)")
    generated.append(("hilbertColor", len(hilbert_colors)))

    # 2. Hue-Saturation strip
    print(
        f"Generating hueSatColor ({args.hue_steps}×{args.sat_steps} = {args.hue_steps * args.sat_steps} colors)..."
    )
    huesat_colors = generate_hue_sat(args.hue_steps, args.sat_steps)
    huesat_json = build_crameri_json("hueSatColor", huesat_colors)
    out_path = COLORMAPS_DIR / "hueSatColor.json"
    with open(out_path, "w") as f:
        json.dump(huesat_json, f)
    print(f"  Wrote {out_path} ({len(huesat_colors)} colors)")
    generated.append(("hueSatColor", len(huesat_colors)))

    # 3. XKCD color list (if provided or default path exists)
    xkcd_default = (
        Path(__file__).resolve().parents[1] / "resources" / "uploads" / "xkcd_colours.csv"
    )
    custom_path = args.custom_colors or (str(xkcd_default) if xkcd_default.exists() else None)
    if custom_path:
        print(f"Generating xkcdColor from {custom_path}...")
        xkcd_colors = generate_xkcd_strip(custom_path)
        xkcd_json = build_crameri_json("xkcdColor", xkcd_colors)
        out_path = COLORMAPS_DIR / "xkcdColor.json"
        with open(out_path, "w") as f:
            json.dump(xkcd_json, f)
        print(f"  Wrote {out_path} ({len(xkcd_colors)} colors)")
        generated.append(("xkcdColor", len(xkcd_colors)))
    else:
        print("Skipping xkcdColor (no custom colors file found)")

    # Update index.json
    update_index(generated)
    print("Done!")


if __name__ == "__main__":
    main()
