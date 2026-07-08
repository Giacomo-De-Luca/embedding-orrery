"""Bootstrap qwen-scope SAE feature data (Phase 1: label-free).

For each configured layer this script:

1. Downloads the qwen-scope TopK SAE weights (HF) and extracts the decoder
   directions into a features parquet (``index``, 2048-d ``vector``,
   ``density=0.0``, empty ``label``/logits) — qwen-scope has no Neuronpedia
   data, so labels stay empty until the Phase-2 autointerp pass backfills
   them.
2. Ingests the parquet into DuckDB ``sae_features`` (+ a ChromaDB
   decoder-vector collection for the feature scatter) under the canonical
   ids ``model_id="qwen3-1.7B-base"`` / ``sae_id="{layer}-qwenscope-1-res-32k"``.

The ingest step makes the model appear in the frontend's ``saeModels``-driven
pickers. **Run with the backend stopped** — DuckDB is single-writer.

    uv run python -m interpretability_backend.scripts.extract_qwen_decoder_vectors
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from backend.embedding_functions.ingest_sae import ingest_sae_features
from interpret.sae.extract_decoder_vectors import extract_and_merge
from interpret.sae.loading import clear_sae_cache
from interpret.sae.paths import vectors_dir
from interpret.sae.sae_config import QwenScopeSAEConfig
from interpret.sae.source_ids import qwen_source_id

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_SIZE = "1.7B"
# L14: verified live by the qwen_scope_smoke gate; L24: the layer the existing
# qwen autointerp configs target (keeps Phase-2 label data compatible).
LAYERS = [14, 24]
INGEST = True  # False → extract parquets only


def main() -> None:
    for layer in LAYERS:
        config = QwenScopeSAEConfig(layer_index=layer, model_size=MODEL_SIZE, device="cpu")
        model_id = config.neuronpedia_model_id
        sae_id = qwen_source_id(config)
        out_path = (
            vectors_dir()
            / f"w_dec_{model_id}_layer{layer}_{config.hook_type.value}_w{config.width}.parquet"
        )

        print(f"\n=== Layer {layer}: {model_id} / {sae_id} ===")
        if out_path.exists():
            print(f"  Parquet already exists: {out_path}")
        else:
            extract_and_merge(config, out_path, skip_labels=True)
            clear_sae_cache()  # free the ~0.5 GB cpu tensors between layers

        if INGEST:
            result = ingest_sae_features(
                str(out_path),
                model_id=model_id,
                sae_id=sae_id,
                store_vectors=True,
            )
            if result.get("error"):
                print(f"  Ingest FAILED: {result['error']}")
                sys.exit(1)
            print(
                f"  Ingested {result['records_inserted']} features in {result['duration_seconds']}s"
            )


if __name__ == "__main__":
    main()
