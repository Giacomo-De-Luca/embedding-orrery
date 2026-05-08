# `interpret/sae/pipeline/`

Data preparation pipeline for Gemma-Scope SAEs. Three stages: download from Neuronpedia S3, merge activation batches, extract decoder vectors + labels into parquet.

## Main API

| Class | Purpose |
|---|---|
| `SAEPipelineConfig` | Config: takes a `GemmaScopeSAEConfig` + stage skip flags. All paths/identifiers are derived automatically. |
| `SAEPipelineRunner` | Orchestrates the three stages, returns `SAEPipelineResult` with output file paths. |
| `SAEPipelineResult` | Output: paths to features parquet, activations JSONL, features JSONL, plus model/sae IDs. |

Decoder vector extraction logic lives in the parent module: `interpret.sae.extract_decoder_vectors`.

## Usage

```bash
# Download + extract for layer 9, 16k width (default: skip activations)
uv run python -m interpret.sae.pipeline.prepare_sae_data --layer 9 --width 16k

# Include activation examples (~336 MB download)
uv run python -m interpret.sae.pipeline.prepare_sae_data --layer 9 --width 65k --with-activations

# Skip download (use already-downloaded labels)
uv run python -m interpret.sae.pipeline.prepare_sae_data --layer 9 --width 16k --skip-download
```

## Programmatic usage

```python
from interpret.sae.sae_config import GemmaScopeSAEConfig
from interpret.sae.pipeline import SAEPipelineConfig, SAEPipelineRunner

config = SAEPipelineConfig(
    sae=GemmaScopeSAEConfig(layer_index=9, width="65k", device="cpu"),
    skip_activations=True,
)
result = SAEPipelineRunner(config).run()
print(result.features_parquet)  # Path to output parquet
```
