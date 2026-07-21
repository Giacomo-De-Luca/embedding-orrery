"""Shared constants for the SAE analysis stage.

`SAE_INTERMEDIATES` names every intermediate key that marks a dataset's
`(layer, intermediate)` entry as SAE feature activations: the classic
pre-pooled path writes `"sae_feat"`, the token-level pooled path writes
`"sae_max"` / `"sae_last"`. Analyses iterate a dataset's keys and skip
anything not in this set.
"""

SAE_INTERMEDIATES = frozenset({"sae_feat", "sae_max", "sae_last"})
