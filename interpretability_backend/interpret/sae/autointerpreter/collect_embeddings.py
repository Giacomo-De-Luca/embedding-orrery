"""Stage 1 (embedding variant) — collect sentence-transformer vectors.

For every ``(word, synset)`` pair in WordNet, format the configured prompt
template, embed it with a sentence-transformer (one pooled, dense, **signed**
vector per prompt), and append to a :class:`DenseActivationStore`. This is the
embedding-model counterpart to :class:`ActivationCollector` (which runs an LM +
SAE hooks); both share :class:`WordNetSampleIterator` for the corpus walk and
write the same ``index.parquet`` row schema so the downstream extract / label /
eval / score stages are agnostic to the source.

The embedder is the project's prompt-aware
:class:`SentenceTransformerEmbeddingFunction` (it mean-pools internally, so no
per-token aggregation applies). Gated checkpoints (e.g. EmbeddingGemma) trigger
a one-shot Hugging Face login via the factory's ``_ensure_hf_login`` helper.

Resume: a sample is skipped when the store already has its ``(word, synset_id)``
key, mirroring the SAE collector.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from tqdm import tqdm

from interpret.sae.autointerpreter.config import (
    PROJECT_ROOT,
    AutoInterpretCollectConfig,
    EmbeddingSourceSpec,
    dump_yaml,
)
from interpret.sae.autointerpreter.dense_activation_store import DenseActivationStore
from interpret.sae.autointerpreter.wordnet_samples import WordNetSampleIterator
from interpret.utils.wordnet_parser import WordNetParser


class EmbeddingCollector:
    """Embed every WordNet entry with a sentence-transformer and persist.

    ``embedder`` and ``parser`` are injectable for tests; ``run_dir`` lets the
    orchestrator override the model-derived default.
    """

    def __init__(
        self,
        config: AutoInterpretCollectConfig,
        embedder=None,
        parser: WordNetParser | None = None,
        run_dir: Path | None = None,
    ) -> None:
        self.config = config
        self.spec: EmbeddingSourceSpec = config.resolve_embedding()
        self.embedder = embedder if embedder is not None else self._build_embedder(self.spec)

        if parser is not None:
            self.parser = parser
        elif config.wordnet_xml_path:
            xml_path = Path(config.wordnet_xml_path)
            if not xml_path.is_absolute():
                xml_path = (PROJECT_ROOT / xml_path).resolve()
            self.parser = WordNetParser(xml_file_path=str(xml_path))
        else:
            self.parser = WordNetParser()

        self.run_dir = run_dir if run_dir is not None else config.run_dir()
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.dtype = np.dtype(self.spec.activation_dtype)
        self.n_features = self._probe_dim()
        self.store = DenseActivationStore(
            self.run_dir, n_features=self.n_features, dtype=self.dtype,
        )
        self._seen = self.store.existing_row_keys() if config.resume else set()

    # ── Embedder factory ───────────────────────────────────────────────

    @staticmethod
    def _build_embedder(spec: EmbeddingSourceSpec):
        """Construct the prompt-aware sentence-transformer wrapper.

        Retries once behind a Hugging Face login for gated checkpoints
        (e.g. EmbeddingGemma), reusing the embedding factory's helper.
        """
        from huggingface_hub.errors import GatedRepoError
        from scripts.utils.embedding_database.create_embedding_function import (
            _ensure_hf_login,
        )
        from scripts.utils.embedding_database.specific_functions.embed_sentence_transformer import (  # noqa: E501
            SentenceTransformerEmbeddingFunction,
        )

        kwargs = dict(
            model_name=spec.model_name,
            device=spec.device,
            normalize_embeddings=spec.normalize,
            prompt=spec.prompt,
            batch_size=spec.embed_batch_size,
        )
        try:
            return SentenceTransformerEmbeddingFunction(**kwargs)
        except GatedRepoError:
            _ensure_hf_login()
            return SentenceTransformerEmbeddingFunction(**kwargs)

    def _probe_dim(self) -> int:
        vec = self.embedder(["probe"])[0]
        return int(np.asarray(vec).shape[-1])

    # ── Sample iteration ───────────────────────────────────────────────

    def _iter_samples(self):
        iterator = WordNetSampleIterator(
            self.parser,
            pos_filter=self.config.pos_filter,
            prompt_template=self.config.prompt_template,
            limit=self.config.limit,
        )
        return iterator.iter_samples(self._seen)

    def _meta(self, sample: dict) -> dict:
        # Mirror the SAE row schema so index.parquet stays uniform across
        # sources. layer/hook/width carry embedding-appropriate sentinels.
        return {
            **sample,
            "n_tokens": 0,
            "layer": -1,
            "hook_type": "embedding",
            "width": self.spec.model_name,
            "aggregation": "pooled",
        }

    # ── Run ────────────────────────────────────────────────────────────

    def run(self) -> Path:
        """Execute the full embedding pass. Returns the run directory."""
        from interpret.sae.autointerpreter.config import AutoInterpretConfig

        dump_yaml(
            AutoInterpretConfig(run_slug=self.config.run_slug(), collect=self.config),
            self.run_dir / "experiment.yaml",
        )

        flush_every = max(1, self.config.flush_every)
        batch_size = max(1, self.spec.embed_batch_size)
        rows_since_flush = 0
        batch: list[dict] = []
        pbar = tqdm(desc="embed-collect", unit="sample")

        def process(samples: list[dict]) -> None:
            nonlocal rows_since_flush
            if not samples:
                return
            vecs = self.embedder([s["prompt"] for s in samples])
            for sample, vec in zip(samples, vecs, strict=True):
                self.store.append(np.asarray(vec, dtype=self.dtype), self._meta(sample))
                rows_since_flush += 1
            pbar.update(len(samples))
            if rows_since_flush >= flush_every:
                self.store.flush()
                rows_since_flush = 0

        for sample in self._iter_samples():
            batch.append(sample)
            if len(batch) >= batch_size:
                process(batch)
                batch = []
        process(batch)
        self.store.flush()
        pbar.close()
        return self.run_dir
