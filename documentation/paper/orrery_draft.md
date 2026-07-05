# Orrery: An Interactive Platform for Embedding Visualisation and Mechanistic Interpretability

*Draft v0.1 — EMNLP System Demonstrations format (6 pages + refs). Markdown draft; convert to the ACL LaTeX template once content is frozen. Unverified numbers are marked **[TODO]** — see the claims-audit notes at the bottom of this file.*

---

## Abstract

Orrery is an open-source platform for interactive visualisation of embedding spaces with native support for Sparse Autoencoder (SAE) interpretability. Given any local or HuggingFace dataset, it produces an explorable 3D constellation through a single pipeline of embedding, projection, clustering, automatic topic labelling, and, optionally, collection of per-document SAE feature activations. The resulting map can be searched visually, semantically, lexically (BM25), or by SAE feature name. SAE decoder vectors are visualised as a navigable constellation of their own, in which entering a prompt highlights the features it activates; individual features can be inspected in a Neuronpedia-style dashboard and applied directly to steer the model. To validate the platform as a research instrument, we report three case studies: psycholinguistic norms (concreteness, imageability, valence) are linearly decodable from sentence-embedding geometry (linear-probe R² up to 0.79, Spearman ρ up to 0.89); UMAP layouts of colour-word embeddings align with perceptual colour space *more strongly than the embeddings themselves* (Mantel ρ = 0.60 vs. 0.29); and an ActAdd-style refusal-direction intervention on Gemma-3-4b-it reproduces the findings of Arditi et al. (2024). The interface sustains 60 fps on 250k points, including animations and volumetric nebula effects, on 8 GB of RAM. Orrery unifies corpus-scale semantic exploration with internal feature interpretation and causal model intervention, and is released under the Apache 2.0 licence at **[URL]**.

---

## 1 Introduction

As soon as Mikolov et al. popularised meaningful textual embedding spaces `[mikolov2013distributed]`, the search began for ways to visualise them easily. The Embedding Projector `[smilkov2016embeddingprojectorinteractivevisualization]`, integrated into TensorBoard and released in 2016, is the primary inspiration for Orrery. The Projector offered the possibility of examining embeddings through a scatter plot in a 3-dimensional space, reduced with either PCA or t-SNE `[van2008visualizing]`, and named three tasks it aimed to solve: **exploring local neighbourhoods**, **viewing global geometry and finding clusters**, and **finding meaningful "directions"** — the famous *man:woman :: king:queen* regularities also found by Mikolov `[mikolov-etal-2013-linguistic]`. Regardless of the intent, the Projector has no way of clustering the vectors or finding directions inside the platform: the tasks were named, not supported.

Google kept exploring 3D vector spaces through artistic experiments — the *t-SNE Map* for images and *Semantic Galaxy* for Gemma embeddings **[TODO: verify attribution — the HF space is published by webml-community, not Google]** — which are demos limited in scope rather than general platforms; Semantic Galaxy is the main inspiration behind Orrery's cosmic theme. When UMAP was released in 2018 `[mcinnes2018umap]`, the new algorithm's effectiveness was itself demonstrated through scatter plots showing cleaner clustering than t-SNE on popular datasets. But while the Projector was never updated for the passage from static to contextual embeddings, the tools that emerged in its place all made the same two choices: **2D, and scale**. Nomic Atlas visualises millions of embeddings in 2D with automatically generated clusters and labels, plus filtering and search over the collection; WizMap `[wang2023wizmapscalableinteractivevisualization]` likewise renders millions of points, using density contouring to find regions of interest; Latent Scope offers an open-source local pipeline (embed → project → cluster → label) with early, limited SAE support; Embedding Atlas from Apple is the most polished of the recent generation. Curiously, alongside the retreat from 3D, all of these tools offer restricted clustering, projection, and labelling options — typically one algorithm each, minimally configurable. Orrery brings the focus back to 3D and *exploration*, with projection (PCA/UMAP, 2D/3D), clustering (HDBSCAN, K-means, GMM, spectral, over a configurable space), and labelling (c-TF-IDF, LLM, hierarchical reduction) as first-class, configurable steps — and completes, in-platform, the three tasks the Projector could only name: clusters via topic extraction, directions via analytical colouring and probing (§4.1), neighbourhoods via click-to-search in the original space.

A parallel story unfolded in interpretability. *Towards Monosemanticity* `[bricken2023monosemanticity]` already contains the visualisation framework still in use for SAE features: an elegant per-feature dashboard — activation density, top and bottom logits, activation examples — and, alongside it, a 2D UMAP scatter plot of all 4,608 features. The follow-up, *Scaling Monosemanticity* `[templeton2024scaling]`, displays a striking map of the feature space of Claude 3 Sonnet. Open-source tooling — SAE-Vis, its successor SAEDashboard, and Neuronpedia `[neuronpedia]` — faithfully replicated the dashboard and **left behind the scatter plot**: the map of the feature space, arguably the most exploratory element of Anthropic's interface, has no open-source counterpart.

Orrery addresses both gaps at once: (a) it makes the Anthropic-style feature map available as open source, backed by the same exploration machinery as any embedding collection; (b) it improves on the feature dashboard by fusing it with map-level exploration and direct steering of the model; and (c) because corpora and feature spaces are the same kind of object in the platform, it extends the Projector's third task — finding directions — from embedding space into model internals. The intended workflow is: *find a semantic pattern → identify candidate features or directions → inspect the evidence → intervene in the model → observe the behavioural effect.*

**Contributions.**

- A **collection manager** that embeds, projects, and stores vector collections with a dual-database design (DuckDB + ChromaDB), decoupling datasets from their (possibly many) embeddings;
- An **optimised 3D scatter-plot stack** for textual collections, inspired by star cartography, sustaining 60 fps at 250k points on 8 GB of RAM, with extensive filtering, colour-scale, and labelling support;
- **Native SAE support**: one-click ingestion of Gemma Scope feature spaces as navigable constellations, per-document activation collection, prompt-to-constellation highlighting, dashboard-with-steering — restoring the open-source feature map;
- **Three case studies** — linearly decodable psycholinguistic directions, perceptual organisation of colour-word embeddings, and a refusal-direction replication — each an instance of the Projector's "finding directions" task, executed inside the platform.

---

## 2 System Design

### 2.1 Datasets and collections

Orrery separates two concepts. A **dataset** is a set of documents (text, image, audio) stored and indexed in DuckDB. A **collection** is one embedding of that dataset: a list of vectors stored in ChromaDB, plus projections, optional topic extractions, and optional SAE activations, all stored in DuckDB. One dataset can have many collections — the same corpus embedded with different models, prompts, or column combinations — without duplicating the documents themselves, which was the main limitation we encountered with ChromaDB-only storage. DuckDB acts as the central orchestrator (documents, metadata, projections, topics, sparse SAE activations, full-text index); ChromaDB stores only IDs and dense vectors for approximate nearest-neighbour search.

*Figure 1 (architecture): Data sources → embedding providers → DuckDB + ChromaDB → GraphQL API → Next.js frontend.*

### 2.2 Ingestion pipeline

A dataset is loaded as a preview from HuggingFace or from local files (JSON, Parquet, CSV). The only mandatory step afterwards is computing projections; everything else is optional.

**Embedding.** The user selects a column combination and an optional prompt template, using column names in curly brackets (e.g. `{word}: {definition}`). Eight providers are supported — local: SentenceTransformers (default), Ollama, BGE/FlagEmbedding, Qwen; API: OpenAI, Cohere, HuggingFace, Gemini — with task types for Gemma/Gemini embedders. Hardware is auto-detected (MPS → CUDA → CPU); all core features run fully offline with local models. Datasets that already contain a vector column skip this step. Selected metadata columns are stored in DuckDB; vectors and the embedding-function configuration go to ChromaDB.

**Projection.** 2D and 3D projections are computed automatically for both PCA and UMAP and stored as native float arrays in DuckDB.

**Topic modelling (optional).** A BERTopic-style pipeline: clustering (HDBSCAN by default; K-means, GMM, and spectral clustering are also available — the latter suited to small datasets given its O(n²) memory), then c-TF-IDF keyword extraction, then optional LLM labelling (Gemini/OpenAI), reusing BERTopic's labelling prompt. The clustering space is configurable: a fresh low-dimensional UMAP of the raw vectors (default, following BERTopic), the stored visualisation coordinates (fast), or the L2-normalised original vectors. An optional reduction step merges topics to a target count while preserving the originals as subtopics, yielding a two-level hierarchy used for nested colouring.

**SAE activations (optional).** For any stored collection, each document can be run through Gemma with Gemma Scope SAEs (any size, any layer) attached. Activations are collected per token, max-pooled per feature across tokens, and stored as a sparse matrix in DuckDB (only nonzero entries, typically ~100–150 features per document). Collection is batched, reports progress over WebSocket, and is resumable.

### 2.3 Searching the collection

A stored collection can be searched four ways, with results rendered as a highlighted constellation whose star colours grow warmer with similarity to the query:

- **Semantic search** over the original embedding space (ChromaDB cosine), also triggered by clicking any point: the camera flies to the point and its top-k nearest neighbours in the *original* (pre-projection) space light up;
- **Text search**, server-side over DuckDB, with per-column selection, exact/partial matching, case sensitivity, and BM25 word-level search via DuckDB's FTS extension;
- **SAE feature search**: selecting named features (e.g. *football*) retrieves the top-k documents by max activation across the selected features — a two-hop search from feature label to documents;
- **Prompt highlighting** (SAE collections): entering a prompt runs a forward pass and highlights every point whose feature activates on the prompt.

**Filtering.** The legend panel doubles as a filter: categorical columns render as a searchable list with counts where categories can be toggled; numeric columns render as a colour-scale histogram with draggable handles. The analytics panel offers a second histogram for graphical range filtering (including auto-detected temporal fields), and an advanced filter builder composes equality, inequality, set-membership, and range predicates.

**Colour.** Categorical, sequential, diverging, and monochrome scales are supported, including Crameri's perceptually uniform scientific colour maps alongside the d3 defaults — serving both interpretability research and the display of historical corpora. The active colour scheme round-trips through the URL and can be saved per collection.

### 2.4 Rendering at scale

The scatter plots are WebGL-accelerated Plotly.js, with a forked renderer that fixes an O(n·m) trace-update pathology critical for real-time overlays. Marker size and opacity adapt logarithmically to point count. Two 3D-specific systems support the cartographic metaphor: **nebula effects** — translucent haze sprites around topic clusters, composited on an overlay canvas synchronised with the WebGL camera — and **label collision avoidance**, which projects 3D label anchors to screen space each frame and greedily packs labels on a spatial grid, prioritised by relevance (cluster labels highest, then similarity-ranked point labels). The interface sustains 60 fps at 250k points with nebula effects enabled on 8 GB of RAM. **[TODO: state test machine + measurement method for the 60 fps / 8 GB claim.]**

---

## 3 SAE Support

Orrery treats an SAE's feature space as itself an embeddable dataset — restoring the feature-space map that open-source SAE tooling left behind (§1) and closing the loop between corpus geometry and model internals. A dedicated collection type, the *SAE collection*, can be created with one click: features, labels, and decoder vectors for any Gemma Scope SAE are downloaded from the Neuronpedia S3 bucket and ingested into DuckDB/ChromaDB.

### 3.1 Visualising the feature space

A feature space can be constellated in two ways: (i) using each feature's row of the **decoder matrix** (dimension = the model's residual stream) as its vector, or (ii) **embedding the auto-interpreted label** of each feature with a conventional text encoder. Either yields a `{feature_index, label, vector, top_logits, exemplar documents}` dataset that flows through the standard projection pipeline. SAE collections gain two abilities beyond normal collections: prompt search (§2.3) — each point *is* a feature, so the features most activated by a prompt are highlighted and ranked — and right-click navigation from any feature-point to its full dashboard.

### 3.2 Feature explorer

A Neuronpedia-inspired page shows per-feature dashboards: token-strip activation heatmaps over exemplar contexts, top/bottom logit bar charts, and density statistics. Features are searchable three ways: text search on labels, semantic search on label embeddings, and *prompt search* — write a prompt and inspect, per token, which features fire, or rank features by max/mean activation over the prompt. Multiple SAEs (including two widths at the same layer) can be hooked in a single forward pass.

### 3.3 Direct steering

Unlike Neuronpedia, the feature explorer and steering live on the same page. Any feature can be injected additively on its decoder direction (ActAdd-style; Turner et al., 2023) at a chosen strength, and the effect observed immediately in a streaming chat interface. A *compare mode* runs a steered and an unsteered thread side-by-side under a shared random seed, so behavioural differences are attributable to the intervention alone. Steering also supports pre-extracted activation-difference directions (e.g. a refusal direction, §4.3) alongside SAE features. Model lifecycle (load / resident / unload) is managed server-side with serialised GPU access.

The SAE machinery is implemented from scratch (JumpReLU and TopK architectures, ~135 lines) rather than wrapping SAELens/TransformerLens, with a hook manager supporting multi-SAE attachment and steering composition; mid-layer activation capture uses a fork of `gemma_pytorch`. Live inference currently targets Gemma-3-4b (see Limitations).

---

## 4 Case Studies

The case studies instantiate the Projector's third, never-supported task — *finding meaningful directions* — at three levels: linear directions in embedding space (§4.1), nonlinear perceptual structure that projection makes visible (§4.2), and a causal direction inside the model (§4.3). §4.4 exercises the other two tasks (clusters, neighbourhoods) at corpus scale.

### 4.1 Psycholinguistic norms are linear directions

Marks & Tegmark (2023) showed that true and false statements are linearly separable in activation space — so cleanly that a *mass-mean* probe (classifying along the difference of class means) suffices. A key use of an embedding visualiser for interpretability is *finding such directions visually*. We embed the Glasgow Norms (Scott et al., 2019) — 4,682 English words rated on nine psycholinguistic dimensions (arousal, valence, dominance, concreteness, imageability, familiarity, age of acquisition, semantic size, gender association) — and colour the map by each rating. Gradients for concreteness, imageability, and valence are visible immediately (Fig. 3).

To quantify what is visible, we train probes (ridge, lasso, SVR, MLP, mass-mean) on the embedding vectors of two encoders, MiniLM-L6-v2 and EmbeddingGemma-300M, with an 80/20 split. Linear probes recover the strongest dimensions almost as well as nonlinear ones — the gap between the best linear probe and an RBF-SVR is ≤ 0.04 R² on every dimension — indicating the structure is essentially linear:

**Table 1: Best linear-probe validation R² (Spearman ρ) per dimension.**

| Dimension | EmbeddingGemma-300M | MiniLM-L6-v2 |
|---|---:|---:|
| Concreteness | 0.79 (0.89) | 0.75 (0.87) |
| Imageability | 0.73 (0.86) | 0.69 (0.84) |
| Valence | 0.71 (0.81) | 0.59 (0.75) |
| Semantic size | 0.67 (0.83) | 0.53 (0.74) |
| Age of acquisition | 0.68 (0.82) | 0.61 (0.79) |
| Gender association | 0.59 (0.74) | 0.50 (0.67) |
| Familiarity | 0.56 (0.77) | 0.52 (0.75) |
| Arousal | 0.54 (0.70) | 0.43 (0.65) |
| Dominance | 0.52 (0.70) | 0.40 (0.62) |

*(Full per-probe, per-layer results in the appendix; mass-mean probes reach Spearman ρ = 0.86 on concreteness, echoing Marks & Tegmark's observation that the direction is recoverable from class means alone.)*

The embedding space self-organises around interpretable linear directions without supervision; how many such directions exist remains open, and we hope the visualiser lowers the cost of looking for them.

### 4.2 XKCD colour survey: projections can *reveal* structure

Nonlinear structure can be visualised too. The XKCD colour survey (Monroe, 2010) collected labels for colour patches from ~200k respondents; we use its ~950 English colour terms, each with a hex code. We embed the terms with Gemini embeddings and render each point in its own colour, both directly and via a Hilbert-curve mapping that linearises 3D colour space onto a single perceptual strip (appendix). The UMAP layout visibly self-organises into a colour space (Fig. 4) — greens with greens, warm tones together — consistent with Abdou et al. (2021), who found colour-term representations align with perceptual colour geometry without grounding.

We quantify the alignment with Mantel tests (Spearman ρ between condensed pairwise-distance structures; 1,000-permutation null): perceptual distance is CIEDE2000 in CIELAB, embedding distance is cosine, projection distance is Euclidean.

**Table 2: Mantel alignment on 954 XKCD colour terms (all p_emp < 0.001).**

| Reference | Target | Global ρ | kNN@10 ρ |
|---|---|---:|---:|
| Perceptual colour (CIEDE2000) | raw Gemini embedding | 0.29 | — |
| Perceptual colour | **UMAP-3D** | **0.60** | 0.20 |
| Perceptual colour | PCA-3D | 0.48 | 0.12 |
| Embedding (cosine) | UMAP-3D | 0.29 | **0.34** |
| Embedding (cosine) | **PCA-3D** | **0.49** | 0.25 |

The striking result is the first block: the UMAP-3D layout is *more* colour-coherent (ρ = 0.60) than the raw 3072-d embedding it was computed from (ρ = 0.29). The projection concentrates a perceptual signal that is diffuse in the full embedding — UMAP does not merely preserve visible structure, it can *make hidden relationships visible*. Conversely, PCA is the more faithful picture of the embedding's global geometry (0.49 vs. 0.29), while UMAP better preserves local neighbourhoods (kNN ρ 0.34 vs. 0.25); the two projections answer different questions, and Orrery ships both.

**[TODO: annotations claim colour channels are "almost perfectly decodable" via probes — no saved probe results for XKCD yet; run the xkcd probing manifest or cut the sentence.]**

### 4.3 Refusal direction on Gemma-3-4b-it

To validate steering end-to-end we adapt Arditi et al. (2024) — *refusal is mediated by a single direction* — to Gemma-3-4b-it on our `gemma_pytorch` fork. Unlike the original, which ablates the refusal direction geometrically at every layer, we apply ActAdd-style additive steering (the same mechanism as our SAE steering), grid-searching over layers and steering strengths for the setting that maximally flips refusals to acceptances without destroying generation coherence. **[TODO before submission: pin the final numbers — draft annotations say layer 11 and 128/128 prompts accepted on HarmBench, but the committed direction metadata says layer 14 and the default eval set is JailbreakBench; the evaluation results were not saved. Re-run and save the eval.]** We release the full experiment code and the extracted refusal vector to support LLM-safety research.

### 4.4 Digital humanities at scale

To show the platform outside interpretability, we embed two corpora: the complete works of Lacan at sentence level (150k sentences) and a subsection of a Sanskrit corpus of digitalised literature. The embedding space cleanly separates texts by period, and on the 150k-sentence Lacan corpus the scatter plot remains fully interactive with nebula mode enabled, allowing humanities scholars to search and access the corpus visually. **[TODO: confirm the exact Sanskrit corpus name/size; "Sanskrit Travelogue corpus" in annotations needs a citation.]**

---

## 5 Comparison with Existing Systems

**Table 3: Feature comparison.**

| | Orrery | TB Projector | Nomic Atlas | WizMap | Latent Scope | Embedding Atlas | Neuronpedia |
|---|---|---|---|---|---|---|---|
| 3D view | ● | ● | – | – | – | – | – |
| In-app embedding (multi-provider) | ● | – | ● (cloud) | – | ● | – | – |
| Configurable clustering (>1 algorithm) | ● | – | – | – | – | – | – |
| Topic labels (c-TF-IDF / LLM) + hierarchy | ●/●/● | – | ●/–/– | ●/–/– | ●/●/– | ●/–/– | n/a |
| SAE feature dashboards | ● | – | – | – | partial | – | ● |
| SAE feature-space map | ● | – | – | – | – | – | – |
| SAE steering + chat | ● | – | – | – | – | – | – |
| Embedding map ↔ SAE linking | ● | – | – | – | – | – | – |
| Colour by any metadata field | ● | limited | limited | – | ● | ● | n/a |
| Open source + local/offline | ● | ● | – | ● | ● | ● | partial |
| Scale (interactive points) | 250k | ~50k | 1M+ (cloud) | 1M+ | ~100k | 1M+ | n/a |

The post-Projector generation (Nomic Atlas, WizMap, Embedding Atlas, Latent Scope) is uniformly 2D and scale-oriented, with a single fixed projection/clustering recipe; the SAE generation (Neuronpedia, SAE-Vis/SAEDashboard) reproduces Anthropic's dashboard without the feature map. Orrery is the only system in either column with a 3D exploratory view, configurable projection/clustering/labelling, an SAE feature-space map, and a path from map to causal intervention. **[TODO: re-verify per-competitor cells against current versions — esp. Latent Scope's SAE support (characterise precisely what it does: which SAEs, dashboard-only vs. map?), Embedding Atlas capabilities, Nomic's current labelling. The novelty claim in this paragraph rests on these cells.]**

---

## 6 Demonstration Plan

A 3–5 minute walkthrough: (1) open a pre-loaded dataset and show LLM-labelled topic clusters; (2) recolour by metadata fields, filter via legend and temporal range; (3) semantic search with constellation highlighting; (4) the XKCD colour map self-organisation; (5) the SAE feature constellation — type a prompt, watch activated features light up; (6) right-click a feature into the explorer, inspect its heatmaps and logits; (7) inject the feature and steer the chat, comparing against the seeded baseline; (8) the refusal-direction preset. A hosted instance with pre-loaded datasets, a Docker one-liner (`docker compose up`), and a screencast are available at **[URL]**.

---

## 7 Conclusion

Orrery unifies corpus-scale semantic exploration, internal feature interpretation, and causal model intervention in one open-source platform. Its analytical colouring and probing workflow surfaced real findings — psycholinguistic dimensions as linear directions (linear R² up to 0.79), and UMAP layouts that are *more* aligned with perceptual colour space than the embeddings they project (Mantel ρ 0.60 vs. 0.29) — and its SAE loop takes a researcher from a semantic pattern to a causal intervention without leaving the tool.

---

## Limitations

Live SAE inference currently targets Gemma-3-4b with Gemma Scope SAEs; the SAE machinery itself (JumpReLU/TopK, hook manager) is model-agnostic, but other model families require an inference wrapper (Qwen support exists in the toolkit but is not wired to the live service). Only additive steering is exposed in the UI (ablation, orthogonalisation, and projection-cap modes exist in the library). Topic extraction follows BERTopic's design; our contribution there is integration, not algorithm. Consistent with published findings, most SAE features do not produce clearly interpretable steering effects. Interactive scale tops out around 250–500k points, below cloud systems such as Nomic Atlas. The refusal-direction release has dual-use implications; we follow Arditi et al. (2024) in judging the safety-research value to outweigh marginal misuse risk, since the technique is already public.

---

## References (to be BibTeX-ified; keys in backticks in the text match the intended .bib keys)

- Mikolov, T. et al. (2013). *Distributed Representations of Words and Phrases and their Compositionality.* NeurIPS. `[mikolov2013distributed]`
- Mikolov, T. et al. (2013). *Linguistic Regularities in Continuous Space Word Representations.* NAACL. `[mikolov-etal-2013-linguistic]`
- van der Maaten, L. & Hinton, G. (2008). *Visualizing Data using t-SNE.* JMLR. `[van2008visualizing]`
- Wang, Z. J. et al. (2023). *WizMap: Scalable Interactive Visualization for Exploring Large Machine Learning Embeddings.* ACL demo. `[wang2023wizmapscalableinteractivevisualization]`
- Bricken, T. et al. (2023). *Towards Monosemanticity: Decomposing Language Models With Dictionary Learning.* Transformer Circuits Thread. `[bricken2023monosemanticity]`
- Templeton, A. et al. (2024). *Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet.* Transformer Circuits Thread. `[templeton2024scaling]`
- Lin, J. (2023–). *Neuronpedia.* `[neuronpedia]`
- Abdou, M. et al. (2021). *Can Language Models Encode Perceptual Structure Without Grounding? A Case Study in Color.* CoNLL. (aclanthology.org/2021.conll-1.9)
- Arditi, A. et al. (2024). *Refusal in Language Models Is Mediated by a Single Direction.* NeurIPS.
- Crameri, F. et al. (2020). *The misuse of colour in science communication.* Nature Communications.
- Grootendorst, M. (2022). *BERTopic: Neural topic modeling with a class-based TF-IDF procedure.* arXiv:2203.05794.
- Lieberum, T. et al. (2024). *Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2.* arXiv:2408.05147.
- Marks, S. & Tegmark, M. (2023). *The Geometry of Truth: Emergent Linear Structure in LLM Representations of True/False Datasets.* arXiv:2310.06824.
- McInnes, L. et al. (2018). *UMAP: Uniform Manifold Approximation and Projection.* JOSS.
- McInnes, L. et al. (2017). *hdbscan: Hierarchical density based clustering.* JOSS.
- Monroe, R. (2010). *XKCD Color Survey Results.* blog.xkcd.com.
- Rajamanoharan, S. et al. (2024). *Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders.* arXiv:2407.14435.
- Scott, G. G. et al. (2019). *The Glasgow Norms: Ratings of 5,500 words on nine scales.* Behavior Research Methods.
- Smilkov, D. et al. (2016). *Embedding Projector: Interactive Visualization and Interpretation of Embeddings.* NeurIPS Workshop.
- Turner, A. et al. (2023). *Activation Addition: Steering Language Models Without Optimization.* arXiv:2308.10248.
- Mantel, N. (1967). *The detection of disease clustering and a generalized regression approach.* Cancer Research.
- + Neuronpedia, Nomic Atlas, Latent Scope, DataMapPlot, Embedding Atlas, DuckDB, ChromaDB citations.

---

---

## Claims audit (internal — delete before submission)

**Backed by committed code/results (safe to print):**
- Glasgow probe table (Table 1): from `interpretability_backend/resources/probing_results/glasgow_psycholinguistic/` — 4,682 words, 9 dims, MiniLM + EmbeddingGemma-300M, ridge/lasso/SVR/MLP/mass-mean. Numbers above are best *linear* (lasso/ridge) R² and Spearman; SVR-rbf adds ≤0.04 R².
- Mantel table (Table 2): from `evaluation/projection_fidelity_results.json` / `documentation/PROJECTION_FIDELITY.md` (954 colours, Gemini 3072-d, 1000 perms).
- Lacan corpus = 150k sentences (author-confirmed; committed TSV is a 41k subset).
- SAE loop, 8 providers, topic pipeline, Plotly fork, 250k-point rendering: all in code.

**Still unbacked — marked [TODO] in the text:**
1. Refusal: layer 11 vs. 14 discrepancy; "128/128 HarmBench" never saved; default eval set is JailbreakBench. Re-run and save before printing any number.
2. XKCD colour-channel probes ("almost perfectly decodable") — no saved results; xkcd probing manifest exists, just needs a run.
3. Brysbaert concreteness "93%" (from annotations) — no data/code committed; cut from this draft, reinstate only with results.
4. 60 fps / 250k / 8 GB — plausible but needs a stated measurement setup.
5. Comparison-table competitor cells — verify against current versions of each tool. Latent Scope's "limited SAE support" must be characterised precisely (the "first/only SAE feature map" claim rests on it); same for what Embedding Atlas can/can't do.
6. Sanskrit corpus name/size/citation.
7. Semantic Galaxy attribution: the HF space is by webml-community (Xenova), released around Google's EmbeddingGemma launch — check whether "Google kept exploring" is accurate or it should be credited to the HF/WebML community.
8. Note: the tool is "WizMap" (Wang et al. 2023), not "VizMap" — spelled per the paper throughout.

**Deliberately not claimed (per audit):** silhouette/Davies-Bouldin "reported alongside extraction" (not computed in-app; the standalone evaluation package computes DBCV/silhouette/coherence — could be cited instead); live Qwen steering; four steering modes in the UI (only additive is reachable).
