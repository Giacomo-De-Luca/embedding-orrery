"""Utility modules for embedding operations.

The convenience re-exports below resolve lazily (PEP 562, like
``topic_extraction/__init__.py``): importing any ``backend.utils.*`` submodule
runs this file, and an eager ``compute_projections`` import here pulls ChromaDB
into every such import and closes an import cycle with ``services.job_state``
(job_state → utils.resource_paths → utils → compute_projections →
services.cancel_registry → job_state).
"""

from importlib import import_module

_LAZY_EXPORTS = {
    "format_text_for_embedding": ".text_processing",
    "extract_metadata": ".text_processing",
    "compute_projections_for_collection": ".compute_projections",
    "IDDeduplicator": ".id_utils",
    "sort_items_by_length": ".batch_utils",
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str):
    module = _LAZY_EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(module, __name__), name)
