"""Tests for the Glasgow psycholinguistic probing experiment.

Validates the merged GlasgowManifestBuilder against the real datasets placed at
resources/psycolinguistics/, and that the experiment YAML parses + resolves.
No model inference — fast, offline. Skips cleanly if the data files are absent
(they are gitignored, so a fresh clone won't have them).
"""

from pathlib import Path

import numpy as np
import pytest

from interpret.probing.configs.experiment import ExperimentConfig
from interpret.probing.manifests.glasgow import GlasgowManifestBuilder, GlasgowPaths

BACKEND = Path(__file__).resolve().parents[1]
PSYCO = BACKEND / "resources" / "psycolinguistics"
GLASGOW_CSV = PSYCO / "glasgow_norm.csv"
CONCRETENESS_TSV = PSYCO / "concreteness.tsv"
EXPERIMENT_YAML = BACKEND / "experiments" / "glasgow_psycholinguistic" / "experiment.yaml"

GLASGOW_TARGETS = [
    "concreteness", "imageability", "valence", "arousal", "dominance",
    "familiarity", "aoa", "semsize", "gender",
]

pytestmark = pytest.mark.skipif(
    not (GLASGOW_CSV.exists() and CONCRETENESS_TSV.exists()),
    reason="psycholinguistic datasets not present (gitignored)",
)


def _builder(glasgow_only=True):
    return GlasgowManifestBuilder(
        paths=GlasgowPaths(concreteness=CONCRETENESS_TSV, glasgow=GLASGOW_CSV),
        default_targets=GLASGOW_TARGETS,
        glasgow_only=glasgow_only,
    )


def test_glasgow_only_samples_and_prompt_column():
    b = _builder(glasgow_only=True)
    assert b.prompt_column == "word"
    # Glasgow norms are ~4.7k words; sample list is unique + non-empty.
    assert 4000 < len(b.samples) < 6000
    assert len(b.samples) == len(set(b.samples))
    assert b.target_columns == GLASGOW_TARGETS


def test_get_rated_samples_glasgow_aligned():
    b = _builder()
    words, values = b.get_rated_samples("glasgow", "concreteness")
    assert len(words) == len(values) > 0
    assert values.dtype == np.float32
    assert np.isfinite(values).all()


def test_get_rated_samples_concreteness_source():
    # Brysbaert concreteness source (Conc.M) is loaded and queryable.
    b = _builder()
    words, values = b.get_rated_samples("concreteness", "Conc.M")
    assert len(words) == len(values) > 0
    assert np.isfinite(values).all()


def test_build_dataframe_has_prompt_and_targets():
    b = _builder()
    df = b.build_dataframe()
    assert "word" in df.columns
    for col in GLASGOW_TARGETS:
        assert col in df.columns
    assert len(df) > 0


def test_unknown_source_raises():
    b = _builder()
    with pytest.raises(ValueError):
        b.get_rated_samples("not_a_source", "concreteness")


@pytest.mark.skipif(not EXPERIMENT_YAML.exists(), reason="experiment.yaml absent")
def test_experiment_yaml_parses_and_resolves():
    # Full schema validation: manifest import, unique extraction names, target
    # names, probe specs — all without running extraction.
    config = ExperimentConfig.from_yaml(EXPERIMENT_YAML)
    assert config.name
    assert config.output_dir
    assert config.manifest.resolve() is GlasgowManifestBuilder
    assert len(config.extractions) >= 1
    assert len(config.targets) == len(GLASGOW_TARGETS)
    assert len(config.probes) >= 1
