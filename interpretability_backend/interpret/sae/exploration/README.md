# `interpret/sae/exploration/`

Interactive / notebook-facing tools for exploring SAE features.

| File | Class / entry point | Purpose |
|---|---|---|
| `explore_neuronpedia.py` | `NeuronpediaExplorer`, `FeatureMatch`, `FeatureInfo`, `ActivationExample` | Reader over the raw Neuronpedia JSONL files in `resources/sae_labels/...`. Label search across layers, density filtering, top-activation-document retrieval. Features JSONL is cached in memory per layer; the ~1.9 GB activations JSONL is streamed. |
| `prompt_explorer.py` | `PromptExplorer`, `PromptExplorerConfig`, `PromptResult`, `LayerResult`, `TokenFeatures`, `ActiveFeature`, `FeatureDetail` | Runs a prompt through Gemma with SAE hooks and returns per-token top-k features with Neuronpedia labels. Rich `_repr_html_` rendering for Jupyter. Wraps `HookManager` + `FeatureLabelStore` + `NeuronpediaExplorer`. |
| `explore_neuronpedia_book.ipynb` | — | Exploratory notebook. |

Run the explorers standalone with:

```
uv run python -m interpret.sae.exploration.explore_neuronpedia
```
