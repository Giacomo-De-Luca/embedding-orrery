# Issues to Fix Before EMNLP Submission

Issues identified during comprehensive codebase audit (May 2026). Organized by subsystem.
For pre-existing issues (code smells, duplication), see [ISSUES_REPORT.md](ISSUES_REPORT.md).

**Severity scale**: Critical > High > Medium > Low
**Effort scale**: Trivial (minutes) < Easy (< 1hr) < Moderate (1-3hrs) < Hard (3hrs+)

---

## Backend

### B-NEW-1: ChromaDB metadata coupling in topic reduction

| Field | Value |
|-------|-------|
| **Severity** | High |
| **Effort** | Easy |
| **File** | `interpretability_backend/backend/topic_extraction/topic_reducer.py` (~line 102-150) |

**Description:**
`_compute_semantic_embeddings()` queries ChromaDB with a `topic_id` metadata filter to retrieve per-topic embeddings for similarity computation during topic reduction. After the DuckDB migration, `topic_id` may not exist as ChromaDB metadata for newer collections — ChromaDB now stores only IDs and dense vectors.

**Impact:** Topic reduction with `use_ctfidf=False` (semantic mode) silently fails or returns zero vectors for topics whose items lack ChromaDB metadata. Falls back to empty embedding, producing meaningless cosine similarities.

**Fix:** Retrieve item IDs per topic from DuckDB (`topic_assignments` table), then batch-fetch embeddings from ChromaDB by ID. This removes the metadata filter dependency entirely.

---

### B-NEW-2: LLM label output not validated

| Field | Value |
|-------|-------|
| **Severity** | Medium |
| **Effort** | Easy |
| **File** | `interpretability_backend/backend/topic_extraction/llm_labeling.py` (~line 157-160) |

**Description:**
Labels returned by LLM providers undergo minimal post-processing — only a `"topic: "` prefix strip. No validation for:
- Empty labels (LLM returns whitespace or "topic: " only)
- Excessively long labels (prompt says "three words at most" but LLMs often ignore this)
- Duplicate labels across topics

**Impact:** Empty or overly long labels appear in the frontend legend and scatter plot cluster labels, hurting readability. Duplicate labels across topics make visual disambiguation impossible.

**Fix:** Add post-processing: strip whitespace, truncate to N words (e.g., 6) if exceeding limit, detect duplicates and append topic ID suffix. Log a warning when the LLM label is discarded or modified.

---

### B-NEW-3: No cluster quality metrics in topic extraction output

| Field | Value |
|-------|-------|
| **Severity** | Medium |
| **Effort** | Easy |
| **File** | `interpretability_backend/backend/topic_extraction/cluster_and_label.py` |

**Description:**
The clustering pipeline returns topic labels and assignments but no quality indicators. Researchers have no way to assess whether the clustering is meaningful without external analysis.

**Fix:** Compute and return silhouette score, Davies-Bouldin index, and Calinski-Harabasz index (all available in `sklearn.metrics`) after clustering. Store in `topic_extractions` metadata. Expose via GraphQL `collectionTopics` query. Display in frontend topic extraction UI.

---

### B-NEW-4: c-TF-IDF integer truncation (minor)

| Field | Value |
|-------|-------|
| **Severity** | Low |
| **Effort** | Trivial |
| **File** | `interpretability_backend/backend/topic_extraction/class_tfidf.py:72` |

**Description:**
`avg_nr_samples = int(X.sum(axis=1).mean())` truncates the average document count to integer. The IDF formula `log((avg / df) + 1)` uses this value. For datasets with sufficient documents per class (100+), the truncation has negligible effect on the log output (< 0.01 difference). For very small clusters (< 10 docs), the rounding error could affect keyword ranking marginally.

**Assessment:** Not a functional bug in practice. Worth fixing as a one-character change (`int` -> `float`) for correctness, but does not affect output quality for realistic datasets.

---

### B-NEW-5: InterpretService does not batch SAE loads

| Field | Value |
|-------|-------|
| **Severity** | Medium |
| **Effort** | Moderate |
| **File** | `interpretability_backend/backend/services/interpret_service.py` (~line 354-366) |

**Description:**
When generating steered responses with multiple `SteeringSpec` entries, each spec triggers a separate SAE load even if multiple specs share the same `(layer, hook_type, width)`. For example, steering on 5 features from the same layer loads the same SAE 5 times.

**Fix:** Group steering specs by `(layer, hook_type, width)` tuple, load each unique SAE once, then attach all specs from that group.

---

### B-NEW-6: Seed not propagated to all clustering paths

| Field | Value |
|-------|-------|
| **Severity** | Low |
| **Effort** | Easy |
| **File** | `interpretability_backend/backend/topic_extraction/cluster_and_label.py` |

**Description:**
`random_state` is set in some clustering calls (KMeans, GMM) but not consistently across all paths. UMAP, HDBSCAN, and topic reduction clustering should all accept and propagate a seed from the extraction config for full reproducibility.

**Fix:** Add `random_state` parameter to `TopicExtractionConfig`, propagate to all sklearn/UMAP/HDBSCAN calls.

---

## Frontend

### F-NEW-1: No error boundaries around scatter plots

| Field | Value |
|-------|-------|
| **Severity** | High |
| **Effort** | Easy |
| **File** | `embedding_visualization/app/components/DashboardPanel.tsx` |

**Description:**
Scatter plots use Plotly.js with WebGL, dynamic imports, and complex trace building. Any uncaught error (malformed data, WebGL context loss, Plotly internal error) crashes the entire application with a white screen. No React Error Boundaries exist in the app.

**Fix:** Add `react-error-boundary` around scatter plot area with a fallback UI and retry button.

---

### F-NEW-2: Errors not surfaced to users

| Field | Value |
|-------|-------|
| **Severity** | Medium |
| **Effort** | Easy |
| **Files** | `useSemanticSearch.ts`, `useEmbeddingData.ts`, `useAppSearch.ts` |

**Description:**
Error handling across hooks follows `console.error()` with no user-visible feedback. Sonner (toast library) is in dependencies but not used for error notifications. Failed semantic searches produce no visible response.

**Fix:** Import `toast` from Sonner, call `toast.error()` in catch blocks for user-facing operations.

---

### F-NEW-3: No frontend test coverage

| Field | Value |
|-------|-------|
| **Severity** | Medium |
| **Effort** | Moderate-Hard |
| **Files** | All frontend code |

**Description:**
Vitest is configured in the project but no test files exist. Key areas that need coverage:
- Zustand store state transitions (color field changes, mute/unmute, reset on collection change)
- Data transformation hooks (`useVisualizationPoints`, `useCategoryData`)
- Color map building (`buildCategoryColorMap`, nested color mode)
- Temporal analysis field detection

**Fix:** Add unit tests for pure logic (store, hooks, utilities). Component tests for scatter plots are lower priority due to WebGL dependency.

---

## Documentation & Packaging

### D-1: README is stale

| Field | Value |
|-------|-------|
| **Severity** | High (for submission) |
| **Effort** | Medium |

**Description:**
README contains phrases like "Polish is not there" and "the code was... raw." Describes ChromaDB as primary storage (pre-migration). Does not mention SAE integration, topic extraction, or key capabilities.

**Fix:** Complete rewrite. See `updated_readme.md` for draft.

---

### D-2: No Docker setup

| Field | Value |
|-------|-------|
| **Severity** | High (for submission) |
| **Effort** | Easy-Medium |

**Description:**
No `Dockerfile` or `docker-compose.yml`. Reviewers and researchers must manually install Python 3.12, Node.js, uv, and configure environment variables.

**Fix:** Create multi-stage Docker Compose: backend (Python/FastAPI), frontend (Next.js), with pre-loaded demo datasets (WordNet subset, XKCD colors, Glasgow Norms).

---

### D-3: No pre-loaded demo datasets

| Field | Value |
|-------|-------|
| **Severity** | High (for submission) |
| **Effort** | Easy |

**Description:**
First run shows empty state. No sample data to explore. WordNet pipeline takes ~8 min.

**Fix:** Bundle pre-computed DuckDB + ChromaDB snapshots with 2-3 curated datasets. Load on first startup if no existing data found.

---

### D-4: No demo video

| Field | Value |
|-------|-------|
| **Severity** | Medium (for submission) |
| **Effort** | Easy |

**Description:**
EMNLP demo track expects a short video demonstrating the system. No video exists.

**Fix:** Record 3-5 min walkthrough: load dataset -> explore topics -> semantic search -> SAE prompt activations -> steering chat.

---

### D-5: No license file

| Field | Value |
|-------|-------|
| **Severity** | Medium |
| **Effort** | Trivial |

**Fix:** Add MIT or Apache 2.0 LICENSE file to repo root.

---

## Suggested Fix Order

| Priority | Issues | Why |
|----------|--------|-----|
| 1 | D-1, D-2, D-3 | Submission blockers: README, Docker, demo data |
| 2 | B-NEW-1 | Silent failure in topic reduction (high severity) |
| 3 | F-NEW-1, F-NEW-2 | Error handling prevents white-screen crashes |
| 4 | B-NEW-2, B-NEW-3 | Label quality + cluster metrics for research credibility |
| 5 | D-4, D-5 | Demo video + license for submission package |
| 6 | B-NEW-5, F-NEW-3 | Performance + test coverage for robustness |
| 7 | B-NEW-4, B-NEW-6 | Minor correctness improvements |
