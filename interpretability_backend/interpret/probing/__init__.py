"""Experiment-agnostic probing engine.

Reads a YAML experiment config and runs:
    manifest -> extract -> (optional SAE encode) -> probes -> (optional SAE analysis) -> report -> figures

Per-experiment Python lives only in manifest builders (referenced from the YAML
by dotted path) and in `ExperimentalDistance` plug-ins (see
`interpret.utils.distances`). Everything else is configuration. See
`orchestrator.run_experiment` for the entry point.

Public visualisers live in `interpret.probing.visualisations`
(`ExperimentVisualiser`, `ConsolidatedVisualiser`). They are intentionally
*not* re-exported here — importing them eagerly would conflict with
`python -m interpret.probing.visualisations` execution.
"""
