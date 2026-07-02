"""Manifest builders for the probing engine.

The abstract ``ManifestBuilder`` base plus a few concrete, self-contained
builders (generic ``feature_csv``; the ``glasgow`` psycholinguistic-norms and
``xkcd`` colour-survey CSV mappers). All are referenced from experiment YAMLs by
dotted path (``"module.path:ClassName"``), resolved at run time by
``interpret.probing.configs.experiment.ManifestSpec.resolve``.

Dataset-specific builders that need rendered image assets (colour patches,
THINGS-coloured) live with their host project rather than the toolkit.
"""

from interpret.probing.manifests.feature_csv import FeatureCSVManifestBuilder
from interpret.probing.manifests.glasgow import (
    GlasgowManifestBuilder,
    GlasgowPaths,
    RatingSource,
)
from interpret.probing.manifests.manifest_base import ManifestBuilder
from interpret.probing.manifests.xkcd import (
    XKCDColourManifestBuilder,
    XKCDPaths,
)

__all__ = [
    "ManifestBuilder",
    "FeatureCSVManifestBuilder",
    "GlasgowManifestBuilder",
    "GlasgowPaths",
    "RatingSource",
    "XKCDColourManifestBuilder",
    "XKCDPaths",
]
