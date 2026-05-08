# `interpret/sae/pipeline/`

Data preparation pipeline for Gemma-Scope SAEs. Three stages: download from Neuronpedia S3 → merge activation batches → extract decoder vectors + labels into parquet.

| File | Class / entry point | Purpose |
|---|---|---|
| `prepare_sae_data.py` | `PrepareSAEConfig`, `PrepareSAEItem`, `PrepareSAERunner` | Orchestrates all three stages for a list of SAEs. Config-driven (no CLI) — edit `main()` or import the runner. |
| `extract_decoder_vectors.py` | `extract_and_merge`, `load_feature_labels` | Loads SAE decoder weights from HuggingFace and merges them with Neuronpedia labels/logits/densities into a per-layer parquet in `resources/sae_vectors/`. |

Run with:

```
uv run python -m interpret.sae.pipeline.prepare_sae_data
uv run python -m interpret.sae.pipeline.extract_decoder_vectors --layers 9 22
```
