"""Cache-key helpers.

The cache layer reads `cache_filename()` directly off each stage's config
dataclass. This module is a thin home for helpers shared across configs
(none right now — kept to centralise additions).
"""

from __future__ import annotations
