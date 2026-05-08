"""
Color column preprocessing for embedding pipelines.

Detects hex color columns in source data and maps each color to a float 0-1
position on a pre-built colorscale strip. The resulting `mapped_colour` value
can be used with Plotly's native colorscale path for GPU-accelerated rendering.
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("star_map")

# Column names that trigger color preprocessing
COLOR_COLUMN_NAMES = {"colour_code", "color_code", "hex_color", "hex_colour", "color_hex"}

# Frontend colormaps directory
_COLORMAPS_DIR = (
    Path(__file__).resolve().parents[3]
    / "embedding_visualization"
    / "lib"
    / "colorMaps"
    / "colormaps"
)

# Cache for loaded strips: name -> Nx3 numpy array
_strip_cache: dict[str, np.ndarray] = {}


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Parse #RRGGBB or #RGB hex string to (R, G, B) tuple."""
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = h[0] * 2 + h[1] * 2 + h[2] * 2
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def load_color_strip(name: str) -> np.ndarray:
    """
    Load a colorscale JSON and return an Nx3 numpy array of RGB values.
    Results are cached.
    """
    if name in _strip_cache:
        return _strip_cache[name]

    json_path = _COLORMAPS_DIR / f"{name}.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Colorscale '{name}' not found at {json_path}")

    with open(json_path) as f:
        data = json.load(f)

    rgb_list = []
    for color_str in data["colors"]:
        # Parse "rgb(r,g,b)" format
        inner = color_str[4:-1]  # strip "rgb(" and ")"
        r, g, b = (int(x) for x in inner.split(","))
        rgb_list.append((r, g, b))

    arr = np.array(rgb_list, dtype=np.float32)
    _strip_cache[name] = arr
    return arr


def map_color_to_strip(hex_color: str, strip_rgb: np.ndarray) -> float:
    """
    Map a hex color to its nearest position on a color strip.
    Returns a float in [0, 1].
    """
    r, g, b = hex_to_rgb(hex_color)
    point = np.array([r, g, b], dtype=np.float32)
    distances = np.sum((strip_rgb - point) ** 2, axis=1)
    idx = int(np.argmin(distances))
    n = len(strip_rgb)
    return idx / (n - 1) if n > 1 else 0.0


def map_color_to_rainbow(hex_color: str) -> float:
    """
    Map a hex color to a rainbow position (hue-only, 0-1).
    Lossy: ignores lightness and saturation — white, black, and red all map to 0.
    """
    import colorsys

    r, g, b = hex_to_rgb(hex_color)
    h, _, _ = colorsys.rgb_to_hls(r / 255.0, g / 255.0, b / 255.0)
    return h


def preprocess_color_metadata(
    metadata: dict[str, Any],
    row: dict[str, Any],
    strip_name: str = "hilbertColor",
    ## other are "rainbow", "xkcdColor", "hilbertColor"
) -> dict[str, Any]:
    """
    Check if the row has a color column and add mapped_colour to metadata.

    Args:
        metadata: Existing metadata dict (modified in place and returned)
        row: Full row dict from the source data
        strip_name: Which colorscale strip to map against.
                    Use "rainbow" for hue-only mapping (no strip file needed).

    Returns:
        The metadata dict with `mapped_colour` and `mapped_colour_scale` added
        if a color column was found. Unchanged otherwise.
    """
    # Find color column in the row
    color_value: str | None = None
    for col_name in COLOR_COLUMN_NAMES:
        val = row.get(col_name)
        if val and isinstance(val, str) and val.startswith("#"):
            color_value = val
            break

    if color_value is None:
        return metadata

    try:
        if strip_name == "rainbow":
            mapped = map_color_to_rainbow(color_value)
        else:
            strip_rgb = load_color_strip(strip_name)
            mapped = map_color_to_strip(color_value, strip_rgb)
        metadata["mapped_colour"] = round(mapped, 6)
        metadata["mapped_colour_scale"] = strip_name
    except Exception as e:
        logger.warning(f"Failed to map color '{color_value}': {e}")

    return metadata
