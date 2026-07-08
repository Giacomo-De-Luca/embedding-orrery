"""Canonical Neuronpedia source-string derivation for SAE collections.

A Neuronpedia "source" identifies one SAE within a model, e.g.
``"9-gemmascope-2-res-65k"`` (layer 9, residual-stream hook, 65k-wide).

Qwen-scope SAEs are **not** on Neuronpedia; ``qwen_source_id`` mints an
analogous 5-part string (``"14-qwenscope-1-res-32k"``) used purely as the
DuckDB ``sae_id`` key. Both schemes share the positional shape
``{layer}-{scope}-{version}-{hookAbbrev}-{width}`` that the API layer and
frontend parse by dash-split positions 0/3/4.

All code that needs these strings should import from here rather than
constructing them ad-hoc.
"""

from interpret.sae.sae_config import GemmaScopeSAEConfig, QwenScopeSAEConfig

# Mapping from HookType enum values to the abbreviations Neuronpedia uses
# in S3 bucket paths and source identifiers.
HOOK_TO_NEURONPEDIA: dict[str, str] = {
    "resid_post": "res",
    "mlp_out": "mlp",
    "attn_out": "att",
}


def neuronpedia_source_id(config: GemmaScopeSAEConfig) -> str:
    """Bare source string: ``'9-gemmascope-2-res-65k'``.

    This is the identifier used by Neuronpedia S3 paths, JSONL filenames,
    and the DuckDB ``sae_id`` column.
    """
    hook_value = config.hook_type.value
    hook_abbrev = HOOK_TO_NEURONPEDIA.get(hook_value)
    if hook_abbrev is None:
        raise ValueError(
            f"Hook type '{hook_value}' is not indexed by Neuronpedia. "
            f"Valid: {list(HOOK_TO_NEURONPEDIA.keys())}"
        )
    return f"{config.layer_index}-gemmascope-2-{hook_abbrev}-{config.width}"


def neuronpedia_source_id_prefixed(config: GemmaScopeSAEConfig) -> str:
    """Source string with model prefix: ``'gemma-3-4b-it_9-gemmascope-2-res-65k'``.

    Used by ``FeatureLabelStore`` as the SQLite source key.
    """
    return f"{config.neuronpedia_model_id}_{neuronpedia_source_id(config)}"


def qwen_source_id(config: QwenScopeSAEConfig) -> str:
    """Canonical qwen-scope source string: ``'14-qwenscope-1-res-32k'``.

    Qwen-scope SAEs are residual-stream only, so the hook abbreviation is
    always ``res``. The ``1`` version segment exists only to preserve the
    5-part positional shape shared with the gemma-scope scheme.
    """
    return f"{config.layer_index}-qwenscope-1-res-{config.width}"
