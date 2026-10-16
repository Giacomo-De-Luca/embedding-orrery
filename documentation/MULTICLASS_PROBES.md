# Multi-class probes

Embedding-space probes (Analytics sidebar → Direction Probes) can train
**classifiers** on categorical metadata fields — two classes or many — next to
the regression kinds. Motivating case: `combined.category` (8 irony
categories, 1,421 of 14,213 tweets labelled): the predicted class extends the
labels to the ~12.8k unlabelled tweets.

## Probe kinds

| Kind (stored) | UI | Trains on | Score |
|---|---|---|---|
| `ridge`, `lasso`, `massmean`, `massmean_cov`, `svr`, `mlp` | as before | numeric fields, or two-value fields as 0/1 | prediction / projection |
| `logreg` | Logistic | class labels (2…50 kept classes) | binary: P(class 1); multiclass: confidence |
| `svc` | SVC (RBF) | class labels | same; Platt-scaled `predict_proba` (seeded) |
| `mlp_clf` | MLP with **Classifier** checked | class labels | same; softmax of the classifier head |

`mlp_clf` is a distinct stored kind so an MLP regressor and an MLP classifier
on the same field never overwrite each other (probes are keyed
`(collection, field, kind)`). The settings popover's **Classifier** checkbox is
locked on for text targets with 3+ classes (regression is impossible), locked
off for continuous targets, and free for two-value and integer-valued targets
(default off — existing behaviour).

## Targets and labels

Classifier kinds read the field via `DuckDBClient.get_categorical_metadata_field`
(`id, text, json_type`) and canonicalise labels (`CategoricalLabel.canonical`,
`backend/services/probe_targets.py`) so they equal what the browser renders
(`String(JSON.parse(v))`), which keeps predicted labels aligned with the
target field's categories:

- integral JSON numbers → `"3"` (DuckDB renders a JSON `3.0` as `"3.0"`);
- JSON strings verbatim (`"03"`, `"3.0"` stay distinct classes);
- `""`, null, arrays and objects are unlabelled (fixes a latent bug where the
  old `binary_target_mapping` counted `""` as a class).

Class order: numeric ascending when every label is a number, else code-point
order — identical to the old binary mapping (larger number → 1, alphabetical
text). Integer numeric fields with 3+ values (e.g. `trec.coarse_label`) are
valid class targets; non-integer numeric fields with 3+ values are
continuous and rejected by classifiers.

The client mirror is `embedding_visualization/lib/utils/probeTargets.ts`
(`ProbeTargetProfile`, same strict numeric regex and threshold rule — keep in
sync).

## Min class threshold

Setting **Min class %** (default 1 %, per probe) excludes classes with fewer
than `max(5, ceil(fraction × n_labelled − 1e-9))` labelled items, where
`n_labelled` counts **labelled** items only (a share of all items would drop
three `category` classes). Two-value targets are never thresholded. At most
50 classes may survive. Excluded classes are recorded on the probe
(`excludedClasses`); their items are not trained on but are still scored, and
the Correct field marks them `Excluded`. The popover previews the outcome
(e.g. `combined.source` at 1 %: threshold 143, 10 kept, 6 excluded).

## Training

- **Split**: multiclass targets use `ProbeSplitter`'s per-class split
  (`backend/utils/probe_split.py`): every class keeps ≥ 1 validation and ≥ 1
  training row; when the training cap applies the subsample is stratified,
  keeping ≥ min(count, 5) rows per class. Regression and binary probes keep
  the original random split bit-for-bit (comparable metrics across kinds).
- **Caps**: SVC's training pool (train + validation) holds at most 5,000 rows (`_SVC_MAX_TRAIN`) — RBF is
  O(n²) and Platt scaling refits ~5×; SVR stays at 10,000.
- **Metrics** (`backend/utils/classification_metrics.py`,
  `ClassificationMetrics`): computed from the persisted probabilities for every
  classifier kind, so they agree with the predicted class the UI shows (SVC's
  own `predict` votes one-vs-one and can disagree with argmax):
  `val_accuracy`, `val_balanced_accuracy`, `val_f1_macro`, `val_f1_weighted`,
  `train_accuracy`, `val_majority_baseline` (train-majority class on
  validation) and `val_auc` (binary) / `val_auc_ovr` (macro one-vs-rest).

## Storage and API

- `probe_scores` gained `predicted INTEGER` and `probs FLOAT[]` (appended,
  nullable; guarded `pragma_table_info` + ALTER + CHECKPOINT migration). Inserts
  go through Arrow with `INSERT … BY NAME`. Multiclass rows store
  `score = max probability`, the predicted class index and all probabilities;
  binary rows keep `score = P(class 1)` only.
- `probes.direction` is NULL for multiclass logreg — the `[C, D]` coefficients
  stay in the on-disk `.npz` artifact. Binary logreg still persists its
  hyperplane normal.
- Config snapshot keys: `task` (`regression` / `binary` / `multiclass`),
  `classes`, `excluded_classes`, `class_threshold`, `min_class_fraction`,
  `target_mapping` (binary text targets only, legacy UI tooltip).
- GraphQL: `ProbeInfo.task/classes/excludedClasses/minClassFraction` (task is
  derived for legacy rows: `logreg` → binary, else regression — via
  `converters.probe_info_from_record`, shared by `collectionProbes` and
  `trainProbe`); `ProbeScores.predicted`; `TrainProbeInput.minClassFraction`
  (validated to [0, 0.5)); new async query
  `probeClassProbabilities(collectionName, targetField, kind, classIndex)` →
  `{itemIds, probabilities}` (0-based class index, rounded to 4 decimals, null
  when out of range or the probe has no probabilities).

## UI (embedding_visualization)

- **Fields per multiclass probe** (names derived from the score field by
  `ProbeFieldNames`, exact-suffix matching):
  - `_pred`: predicted label, categorical, **coloured like the target field** —
    `useProbes` publishes a store alias (`categoryColorAliases`) whose domain is
    the target field's full value list, and every categorical view resolves
    colours through `useCategoryColorMap`, so a never-predicted class can't
    shift the others' palette slots;
  - `_correct`: Correct / Wrong / Excluded (`CORRECTNESS_PRESET`);
  - `_score`: confidence;
  - `_p<k>`: P(class k), fetched **lazily** (one class per request) from the
    probe row's class select or when a `colorBy` URL param names it.
- Binary classifiers (now also SVC / MLP classifier) keep the TP/TN/FP/FN
  confusion view.
- After training, a multiclass probe recolours the map to its predicted class. Auto-recolours (after training, after a P(class) fetch) are dropped if the user picks another colour field while the values load (`pendingRecolorAction`), and a per-probe generation counter discards class-probability fetches that started before a retrain or delete.
- Badge: `acc · F1 (macro)` for multiclass; tooltip lists all metrics, the
  majority baseline, the classes and the excluded classes.

## Payload size

Measured on `combined` (14,213 items): probe scores ≈ 2.1 MB raw / 0.8 MB
gzipped; one class's probabilities ≈ 1.9 MB / 0.7 MB. Shipping all class
probabilities eagerly would multiply the score payload by the class count
(≈ 81 MB raw for 212k items × 16 classes) — hence the lazy per-class query.

## HTTP compression

`backend/utils/http_compression.py` (`JsonGzipMiddleware`, settings in
`interpretability_backend/config/http_compression.json`) gzips JSON responses
for every deployment — local dev hits uvicorn directly, so nginx's gzip never
applied there. It compresses allow-listed JSON media types only (binary
responses such as Arrow pass through), buffers single-message bodies only (streams pass
through), never re-encodes an already-encoded response, and compresses bodies
above 1 MB in a worker thread so WebSocket progress keeps flowing. Default
level 4. On the HF Space, nginx strips `Accept-Encoding` towards the backend
(`deploy/hf-space/nginx.conf`) so compression stays in nginx's worker.

## Caveats

- `_p<k>` field names use the class index; retraining with a different
  threshold can shift indices, so an old shared link may show another class.
- Integers above 2^53 lose precision in the browser and won't match their
  backend labels.
- Classes are the labels as stored: list-like strings such as
  `"[IRONITA-TW-BS, SENTIPOLC_16]"` are single classes (no multi-label probing).

## Tests

Backend (`interpretability_backend/unit_tests/`): `test_probe_targets.py`,
`test_probe_split.py`, `test_classification_metrics.py`,
`test_sklearn_svc_probability.py`, `test_http_compression.py`,
`test_probe_graphql.py`, plus additions to `test_probing_service.py`,
`test_duckdb_probes.py` (incl. the first file-based migration test),
`test_probe_converters.py` and `test_seed_snapshot.py`.
Frontend (`embedding_visualization/lib/utils/__tests__/`):
`probeTargets.test.ts`, plus additions to `probeFields.test.ts`,
`probeParams.test.ts` and `categoryColors.test.ts`.
