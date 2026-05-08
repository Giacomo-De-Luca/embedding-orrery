"""Stage 1 — collect SAE feature activations for WordNet prompts.

For every ``(word, synset)`` pair in WordNet, format ``"{word}: {definition}."``,
run it through Gemma3-4b-it with a single SAE hook attached, aggregate the
per-token feature activations into one vector, and append to a sparse CSR
store (see :class:`SparseActivationStore`).

Reuses:

- :class:`WordNetParser` (interpret/utils/wordnet_parser.py)
- :class:`GemmaPytorchInference` (interpret/inference/gemma_pytorch.py)
- :class:`HookManager` + :class:`SAEConfig` (interpret/sae/)

Aggregation is configurable (``last_token`` / ``mean_prefill`` / ``max_prefill``)
per user requirement.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from tqdm import tqdm

from interpret.inference.gemma_pytorch import GemmaPytorchInference
from interpret.sae.autointerpreter.config import (
    AutoInterpretCollectConfig,
    dump_yaml,
)
from interpret.sae.autointerpreter.sparse_activation_store import SparseActivationStore
from interpret.sae.hook_manager import HookManager
from interpret.sae.loading import WIDTH_TO_D_SAE
from interpret.utils.wordnet_parser import WordNetParser


class ActivationCollector:
    """Run Gemma + SAE on every WordNet entry and persist sparse activations."""

    def __init__(
        self,
        config: AutoInterpretCollectConfig,
        wrapper: GemmaPytorchInference | None = None,
        parser: WordNetParser | None = None,
    ) -> None:
        self.config = config
        self.wrapper = wrapper or GemmaPytorchInference(
            config.checkpoint, precision=config.dtype,
        )
        self.parser = parser or WordNetParser()

        self.sae_config = config.to_sae_config()
        self.manager = HookManager()
        self.manager.add_sae(self.sae_config)

        self.run_dir = config.run_dir()
        self.run_dir.mkdir(parents=True, exist_ok=True)

        d_sae = WIDTH_TO_D_SAE[config.width]
        self.store = SparseActivationStore(
            self.run_dir,
            n_features=d_sae,
            dtype=np.dtype(config.activation_dtype),
        )

    # ── Sample iteration ────────────────────────────────────────────────

    def _iter_samples(self) -> Iterator[dict]:
        """Yield ``{word, synset_id, pos, definition, prompt}`` dicts."""
        words = self.parser.get_all_words()
        pos_filter = (
            set(self.config.pos_filter) if self.config.pos_filter else None
        )
        seen_keys: set[tuple[str, str]] = (
            self.store.existing_row_keys() if self.config.resume else set()
        )
        yielded = 0
        limit = self.config.limit

        for word in words:
            for synset in self.parser.get_synsets_for_word(word):
                if pos_filter and synset.part_of_speech not in pos_filter:
                    continue
                definition = (synset.definition or "").strip()
                if not definition:
                    continue
                if (word, synset.id) in seen_keys:
                    continue
                prompt = self.config.prompt_template.format(
                    word=word, definition=definition,
                )
                yield {
                    "word": word,
                    "synset_id": synset.id,
                    "pos": synset.part_of_speech,
                    "definition": definition,
                    "prompt": prompt,
                }
                yielded += 1
                if limit is not None and yielded >= limit:
                    return

    # ── Forward pass ────────────────────────────────────────────────────

    def _aggregate(self, feats: torch.Tensor) -> torch.Tensor:
        """Reduce ``(seq_len, d_sae)`` to ``(d_sae,)`` per aggregation mode."""
        mode = self.config.aggregation
        if mode == "last_token":
            return feats[-1]
        if mode == "mean_prefill":
            return feats.mean(dim=0)
        if mode == "max_prefill":
            return feats.amax(dim=0)
        raise ValueError(f"unknown aggregation: {mode!r}")

    def _capture_vector(self, prompt: str) -> tuple[np.ndarray, int]:
        """Run one sample and return the aggregated activation + seq_len."""
        self.manager.reset()
        # Output_len=1 triggers a single decode step after prefill; cheaper
        # than longer generation and keeps the SAE hook's prefill capture intact.
        # We use generate_from_template to avoid the chat-wrapping of generate()
        # when use_chat_template is False.
        if self.config.use_chat_template:
            run = lambda: self.wrapper.generate(prompt, output_len=1)
        else:
            run = lambda: self.wrapper.generate_from_template(prompt, output_len=1)

        with self.manager.session(self.wrapper.model.model.layers):
            run()
            record = self.manager.store.prefill(
                layer=self.sae_config.layer_index,
                hook_type=self.sae_config.hook_type,
            )
        if record is None:
            raise RuntimeError("SAE hook did not capture a prefill record")

        feats = record.feature_acts[0]  # (seq_len, d_sae)
        seq_len = int(feats.shape[0])
        vec = self._aggregate(feats).to(torch.float32).cpu().numpy()
        vec[np.abs(vec) < self.config.sparse_threshold] = 0.0
        return vec.astype(self.config.activation_dtype, copy=False), seq_len

    # ── Run ─────────────────────────────────────────────────────────────

    def run(self) -> Path:
        """Execute the full collection pass. Returns the run directory."""
        dump_yaml_path = self.run_dir / "experiment.yaml"
        # Minimal reproducibility stamp (collect-only; full experiment is
        # dumped by the orchestrator).
        from interpret.sae.autointerpreter.config import AutoInterpretConfig
        dump_yaml(
            AutoInterpretConfig(run_slug=self.config.run_slug(), collect=self.config),
            dump_yaml_path,
        )

        flush_every = max(1, self.config.flush_every)
        samples = self._iter_samples()
        for i, sample in enumerate(tqdm(samples, desc="collect", unit="sample")):
            vec, seq_len = self._capture_vector(sample["prompt"])
            meta = {
                **sample,
                "n_tokens": seq_len,
                "layer": self.config.layer_index,
                "width": self.config.width,
                "aggregation": self.config.aggregation,
            }
            self.store.append(vec, meta)
            if (i + 1) % flush_every == 0:
                self.store.flush()

        self.store.flush()
        return self.run_dir
