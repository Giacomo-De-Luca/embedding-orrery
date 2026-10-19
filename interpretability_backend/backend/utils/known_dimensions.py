"""Known embedding dimensions per model name (``utils/known_dimensions.json``).

Knowing a model's dimension up front avoids a test embedding (and a model load).
The embedding-function factory records each newly probed model here; the
model-catalog query reads it to show dimensions next to model suggestions.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger("orrery")

KNOWN_DIMENSIONS_FILE = Path(__file__).with_name("known_dimensions.json")


class KnownDimensions:
    """Read and extend the model -> dimension JSON file."""

    def __init__(self, path: Path = KNOWN_DIMENSIONS_FILE):
        self.path = path

    def load(self) -> dict[str, int]:
        """All known dimensions; a missing or unreadable file means none."""
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def get(self, model_name: str) -> int | None:
        """The dimension of ``model_name``: exact name first, then ignoring case."""
        dimensions = self.load()
        if model_name in dimensions:
            return dimensions[model_name]
        wanted = model_name.casefold()
        return next((d for name, d in dimensions.items() if name.casefold() == wanted), None)

    def record(self, model_name: str, dimension: int) -> None:
        """Add (or update) one model; a write failure is logged, never raised."""
        dimensions = self.load()
        dimensions[model_name] = dimension
        try:
            self.path.write_text(json.dumps(dimensions, indent=1, sort_keys=True), encoding="utf-8")
            logger.info("Recorded embedding dimension %s for %s", dimension, model_name)
        except OSError as error:
            logger.warning("Could not save the dimension of %s: %s", model_name, error)
