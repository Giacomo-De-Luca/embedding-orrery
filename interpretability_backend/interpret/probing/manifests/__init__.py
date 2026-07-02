"""Manifest builders for the probing engine.

Only the abstract `ManifestBuilder` base lives in the toolkit. Concrete,
dataset-specific builders live alongside their data (e.g. the parent repo's
`scripts.interpretability.probing.manifests`) and are referenced from experiment
YAMLs by dotted path (``"module.path:ClassName"``), resolved at run time by
`interpret.probing.configs.experiment.ManifestSpec.resolve`.
"""

from interpret.probing.manifests.manifest_base import ManifestBuilder

__all__ = ["ManifestBuilder"]
