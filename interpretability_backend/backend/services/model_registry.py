"""Model-id → HuggingFace-checkpoint registry (torch-free).

Maps the stable ``model_id`` strings stored in DuckDB (``sae_features``,
collection ``sae_model_id`` metadata) to the checkpoint ``load_model``
should load for them.

Gemma ids are Neuronpedia model ids and derive by rule (``google/{id}``);
non-Gemma families need explicit entries — e.g. the qwen-scope id names the
SAE's *training* checkpoint (Base) while chat deliberately loads the
instruct model (see the Phase-1 qwen integration plan).

This module is imported by the GraphQL layer (``backend.main`` import
graph), so it must stay a literal-only module: no ``torch`` and no
``interpret.*`` imports (both are banned by ``test_torch_free_import.py``).
The frontend mirror lives in ``lib/utils/modelCheckpoints.ts``.
"""

MODEL_ID_TO_CHECKPOINT: dict[str, str] = {
    # Keep in sync with QwenScopeSAEConfig.neuronpedia_model_id output.
    "qwen3-1.7B-base": "Qwen/Qwen3-1.7B",
}


def checkpoint_for_model_id(model_id: str) -> str:
    """Resolve the HF checkpoint to load for a stored model id.

    Falls back to the Gemma rule (``google/{model_id}``) for unregistered
    ids without an org prefix; ids that already carry one pass through.
    Unregistered qwen-family ids fail fast — the Gemma rule would mint a
    nonexistent ``google/qwen…`` path and surface as a confusing HF 404.
    """
    registered = MODEL_ID_TO_CHECKPOINT.get(model_id)
    if registered is not None:
        return registered
    if model_id.lower().startswith("qwen"):
        raise ValueError(
            f"No checkpoint registered for model id {model_id!r} — "
            "add it to MODEL_ID_TO_CHECKPOINT."
        )
    return model_id if "/" in model_id else f"google/{model_id}"


def model_id_for_checkpoint(checkpoint: str) -> str:
    """Inverse of ``checkpoint_for_model_id``: the stored model id for a checkpoint.

    Falls back to the checkpoint basename (the inverse of the Gemma
    ``google/{id}`` rule) for checkpoints without an explicit registry entry —
    e.g. ``google/gemma-3-4b-it`` -> ``gemma-3-4b-it``, while
    ``Qwen/Qwen3-1.7B`` -> ``qwen3-1.7B-base`` via the registry.
    """
    for model_id, registered in MODEL_ID_TO_CHECKPOINT.items():
        if registered == checkpoint:
            return model_id
    return checkpoint.rsplit("/", 1)[-1]
