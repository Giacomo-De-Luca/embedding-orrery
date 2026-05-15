# StarMap: Core Points for EMNLP Demo Paper

Annotated outline for a 6-page system demonstration paper.

---

## Title Options

- "StarMap: An Interactive Platform for Embedding Analysis and Mechanistic Interpretability"
- "StarMap: Unifying Embedding Visualization, Topic Extraction, and SAE Interpretability"

---

## 1. Introduction (0.75 pages)

**Opening problem:** Researchers working with embeddings face fragmented tooling. Visualization (TensorBoard Projector), topic analysis (BERTopic), and mechanistic interpretability (Neuronpedia) exist as separate systems. Moving between them requires data export, format conversion, and context switching. No existing tool lets a researcher go from "what does my embedding space look like?" to "which model features drive this structure?" in a single workflow.

**System contribution:** StarMap is an open-source platform that integrates:
1. Interactive embedding visualization (2D/3D, 250k+ points)
2. Automatic topic extraction with hierarchical reduction
3. Sparse Autoencoder analysis with live model inference and steering

**Motivating example (1 paragraph):** A safety researcher embeds HarmBench prompts, sees them cluster by harm category (Fig. 1), identifies that "cybersecurity hacking" and "red team testing" form adjacent clusters, runs a prompt through Gemma with SAE hooks, discovers shared feature activations across both clusters, then steers the model by suppressing the refusal feature to study vulnerability — all without leaving the platform.

**Key claims:**
- First system to unify embedding visualization with SAE mechanistic interpretability
- Reveals emergent structure: color-word embeddings mirror perceptual color space (Mantel r=0.4); psycholinguistic dimensions are linearly decodable from embedding geometry (R^2 up to 0.9)
- Open-source, runs locally, supports multiple embedding providers

---

## 2. System Overview (1.5 pages)

### 2.1 Architecture (0.5 pages)

**Dual-database design (1 paragraph + diagram):**
- DuckDB: central orchestrator (documents, metadata, projections, topics, SAE data)
- ChromaDB: dense vector storage only (similarity search)
- Rationale: one dataset can have multiple embeddings (different models); DuckDB handles relational queries, ChromaDB handles ANN search

**Data flow diagram:** Data Sources -> Embedding Providers -> Dual Storage -> Topic Extraction -> GraphQL API -> Frontend

**Multi-provider embedding (1 paragraph):** SentenceTransformers (default, local), Gemini, OpenAI, Cohere, Ollama, QWEN, BGE. Hardware auto-detection (MPS/CUDA/CPU). Core features work with local models only, no API keys required.

### 2.2 Topic Extraction Pipeline (0.5 pages)

**Three-stage pipeline:**
1. Clustering: HDBSCAN on UMAP projections (following BERTopic's approach). Also supports KMeans, GMM, Spectral.
2. Keyword extraction: c-TF-IDF (class-based TF-IDF with optional BM25 weighting)
3. Optional LLM labeling (Gemini/OpenAI) with resume support

**Topic reduction:** Hierarchical merging via AgglomerativeClustering or auto-HDBSCAN. Preserves original topics as subtopics with full hierarchy tracking. Enables nested Tableau-style coloring.

**Cluster quality:** Silhouette score, Davies-Bouldin index reported alongside extraction results.

### 2.3 SAE Integration (0.5 pages)

**This is the primary novel contribution. Emphasize.**

**SAE implementation:** From-scratch JumpReLU and TopK SAE architectures. Not wrapping SAELens/TransformerLens — lightweight, portable, ~135 lines.

**Hook manager:** Attaches SAE hooks to specific model layers (resid_post, attn_out, mlp_out). Handles prefill-only semantics, steering composition, multi-SAE attachment.

**Four use cases:**
1. **Prompt activations** -- Run prompt through model, return per-token top-k feature activations with Neuronpedia labels
2. **Scatter plot highlighting** -- Max-pool SAE activations across tokens, highlight corresponding features on the embedding scatter plot
3. **Feature steering** -- Additive intervention on decoder directions with configurable strength. Four modes: additive, orthogonal, ablation, projection-cap
4. **Streaming chat** -- Token-by-token generation with optional steering, exposed via WebSocket subscription

**Key architectural decision:** Internal activation cache via forked gemma_pytorch (not standard forward hooks) enables mid-layer state capture required for SAE analysis.

---

## 3. Visualization & Interaction (1 page)

### 3.1 Rendering (0.3 pages)

- WebGL-accelerated Plotly.js scatter plots (2D and 3D)
- 250k+ points at 60fps on 8GB RAM
- Forked Plotly.js to fix O(n*m) trace update bug (critical for real-time overlays)
- Adaptive marker sizing and opacity (logarithmic scaling with point count)

### 3.2 Interactive Features (0.4 pages)

- **Semantic search** with topic and temporal scoping, glow-effect highlighting
- **Topic filtering** with synchronized legend, search panel, and cluster label clicks
- **Temporal filtering** with auto-detected year/date fields and draggable range picker
- **Analytical coloring** by any metadata field (categorical, sequential, diverging, monochrome scales)
- **Text search** (server-side ILIKE) with field selection and match mode controls

### 3.3 Novel Rendering Features (0.3 pages)

- **Nebula cluster effects** (3D): Sprite-based translucent halos around topic clusters with adaptive opacity, distance-based fade, Poisson-disk sampling. Creates visually distinctive "cosmic" rendering that aids cluster identification.
- **Label collision avoidance** (3D): Projects 3D label positions to 2D screen space via MVP matrices from Plotly's WebGL internals, then uses spatial grid acceleration for greedy set packing. Similarity-weighted priority ensures semantically relevant labels survive.
- **Deferred selected point**: Architectural pattern preventing Plotly O(n) diffs during camera fly-to animation.

---

## 4. Empirical Findings (1 page)

**These findings demonstrate the platform's value as a research tool. They are enabled specifically by the analytical coloring + interactive exploration workflow.**

### 4.1 Embedding Spaces Mirror Perceptual Color Space (0.4 pages)

**Dataset:** XKCD color survey (~950 named colors with hex values)
**Method:** Embed color names with SentenceTransformers, project via UMAP 3D, color each point by its actual hex value using Hilbert-curve color mapping
**Finding:** Visually, color names cluster by perceptual similarity in embedding space (greens together, blues together, warm tones together). Quantified via Mantel test on pairwise embedding distances vs. CIELab perceptual distances: r=0.4 (p<0.001).
**Significance:** The embedding model was never trained on color perception. This demonstrates that semantic representations of color words encode perceptual color structure as an emergent property.

**Figure:** XKCD scatter plot (gallery/XKCD_embedded_colourspace.png)

### 4.2 Psycholinguistic Dimensions as Spatial Gradients (0.4 pages)

**Dataset:** Glasgow Norms (19,537 words with psycholinguistic ratings: concreteness, imageability, valence, arousal, dominance, familiarity, age of acquisition, semantic size, gender association)
**Method:** Embed words with Gemma-4b embeddings, fit linear probes from embedding vectors to each psycholinguistic dimension
**Finding:** R^2 up to 0.9 for concreteness/imageability. Multiple dimensions are linearly decodable, confirming that the embedding space self-organizes along psycholinguistic axes without supervision.
**Significance:** Extends prior work on embedding probing (e.g., Conneau et al., 2018) to psycholinguistic dimensions. The visualization makes these gradients directly visible — researchers can *see* concreteness emerge as a spatial gradient.

**Figure:** Glasgow Norms concreteness (gallery/concreteness.jpg)

**Table:** R^2 for linear probes across Glasgow Norms dimensions, across 2-3 embedding models

### 4.3 SAE Validation: Refusal Direction Recovery (0.2 pages)

**Experiment:** Replicate Arditi et al. (2024) refusal direction extraction on Gemma-3-4b-it using the platform's SAE infrastructure
**Finding:** Extracted refusal vector passes every HarmBench prompt (100% bypass rate), confirming the phenomenon generalizes to Gemma-3-4b-it
**Steering presets:** Curated feature presets (poetic language, formal register, etc.) demonstrate that SAE steering produces interpretable behavioral changes

---

## 5. Comparison with Existing Systems (0.5 pages)

**Table:**

| Feature | StarMap | TensorBoard Projector | BERTopic | Neuronpedia | Nomic Atlas |
|---------|---------|----------------------|----------|-------------|-------------|
| Interactive 2D/3D visualization | Yes | Yes | No | No | Yes |
| Custom dataset embedding | Yes (multi-provider) | No (pre-computed only) | No (headless) | No | Yes (proprietary) |
| Topic extraction + LLM labels | Yes | No | Yes | No | No |
| Hierarchical topic reduction | Yes | No | Yes | No | No |
| SAE feature analysis | Yes | No | No | Yes | No |
| SAE steering + chat | Yes | No | No | No | No |
| Scatter plot <-> SAE linking | Yes | No | No | No | No |
| Metadata-driven analytical coloring | Yes (any field) | Limited | No | No | Limited |
| Temporal filtering | Yes | No | No | No | No |
| Open source | Yes | Yes | Yes | Partial | No |
| Local/offline operation | Yes | Yes | Yes | No | No |
| Scale (points) | 250k+ | ~50k | N/A | N/A | 1M+ (cloud) |

**Positioning:** StarMap occupies a unique intersection: it is the only system that combines general-purpose embedding visualization with mechanistic interpretability (SAE analysis). TensorBoard Projector visualizes but cannot analyze model internals. BERTopic extracts topics but has no interactive visualization layer. Neuronpedia explores SAE features but cannot connect them to embedding-level structure. Nomic Atlas scales further but is closed-source and lacks SAE integration.

---

## 6. Demo Plan (0.5 pages)

**Live demo walkthrough (3-5 minutes):**
1. Open pre-loaded HarmBench dataset -- show topic clustering with LLM labels
2. Switch coloring to different metadata fields -- demonstrate analytical coloring flexibility
3. Run semantic search, show topic-scoped results with glow highlighting
4. Switch to XKCD colors dataset -- show color-word self-organization
5. Navigate to Feature Explorer -- browse SAE features, view activation heatmaps
6. Enter prompt, show per-token feature activations on scatter plot
7. Activate steering preset (e.g., "Poetic") -- show behavioral change in chat
8. Switch to refusal vector preset -- demonstrate safety-relevant steering

**Hosted instance:** [URL] with pre-loaded datasets (HarmBench, XKCD Colors, Glasgow Norms, WordNet subset)

**Docker:** `docker compose up` for local reproduction

**Video:** 3-5 minute recorded walkthrough at [URL]

---

## 7. Conclusion (0.25 pages)

StarMap unifies embedding visualization, topic extraction, and SAE interpretability in a single open-source platform. Through analytical coloring and linear probing, we demonstrate that embedding spaces exhibit emergent alignment with perceptual color space (Mantel r=0.4) and encode psycholinguistic dimensions as linear directions (R^2 up to 0.9). The SAE integration enables researchers to trace embedding-level structure back to individual model features, and to interactively steer model behavior — a workflow not available in any existing tool.

---

## Appendix: Figures List

1. **Fig 1**: HarmBench topic clustering (2D, light mode, LLM labels) -- system overview figure
2. **Fig 2**: Architecture diagram (data flow from sources through dual DB to frontend)
3. **Fig 3**: XKCD color-word embedding (3D, dark mode, hex-colored points)
4. **Fig 4**: Glasgow Norms concreteness gradient (with R^2 table inset)
5. **Fig 5**: SAE workflow: prompt activations -> scatter plot highlighting -> steering chat (composite screenshot)
6. **Fig 6**: WordNet 212k with nebula effects (scale demonstration)
7. **Fig 7**: Comparison table (Section 5)

**Page budget for figures:** ~1.5 pages total across 6 pages. Figures 1, 3, 4, 5 are essential. Figure 6 is optional (visually striking but not scientifically necessary). Figure 2 can be compact (0.25 pages).

---

## Notes on Framing

**Lead with the SAE integration as the primary contribution.** This is what no other system does. The topic extraction is BERTopic-inspired engineering (acknowledge openly), the visualization is Plotly-based (acknowledge), but the end-to-end loop from embedding space to model features to steering is genuinely novel.

**The empirical findings are evidence, not the main contribution.** They demonstrate that the platform produces real scientific insight. Frame as: "To validate the platform's utility, we report two findings enabled by the analytical coloring workflow that, to our knowledge, have not been previously demonstrated visually."

**Be honest about limitations:**
- SAE support is currently Gemma-focused (core machinery is model-agnostic, inference wrapper is Gemma-specific)
- Topic extraction follows BERTopic's approach (HDBSCAN + c-TF-IDF), contribution is integration not algorithm
- Most SAE features do not produce clearly interpretable steering effects (consistent with published findings)
- Scale ceiling at ~250-500k points (vs. Nomic Atlas cloud at 1M+)

**Don't oversell the nebula effects.** Mention as "custom rendering for cluster identification" — visually memorable but not a scientific contribution.
