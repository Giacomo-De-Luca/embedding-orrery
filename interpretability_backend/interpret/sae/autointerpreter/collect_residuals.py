"""Stage 1 (residual variant) — collect raw residual-stream activations.

For every ``(word, synset)`` pair in WordNet, run the prompt through the
base LM with **no SAE attached** and persist the raw hidden state at one or
more residual-stream sites (e.g. ``post_mlp`` at layer 29 — the very point
the Gemma-scope ``resid_post`` SAEs read). Each site gets its own
:class:`DenseActivationStore`, so the downstream stages treat every site
exactly like a dense embedding run: signed dimensions, ``signed``/``split``
extract modes, embed-axis/embed-dim agents.

Only the Gemma family is implemented: capture relies on the forked
``gemma_pytorch`` internal activation cache exposed by
:meth:`GemmaPytorchInference.cache_activations` (Qwen would go through the
hook-based ``Qwen3Inference.cache_activations`` — not wired yet; the config
validator rejects it up front).

Multi-site layout and resume mirror :class:`ActivationCollector`:

- one site  → store directly in ``run_dir/`` (single-source layout);
- many      → ``run_dir/<residual_subdir>/`` per site, one forward pass
  feeding all sites;
- a sample is re-run only when at least one store is missing it.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from interpret.inference.gemma_pytorch import GemmaPytorchInference
from interpret.sae.autointerpreter.collect_activations import aggregate_tokens
from interpret.sae.autointerpreter.config import (
    PROJECT_ROOT,
    AutoInterpretCollectConfig,
    BaseModelSpec,
    ResidualSiteSpec,
    ResidualSourceSpec,
    dump_yaml,
    normalize_aggregations,
    residual_subdir,
)
from interpret.sae.autointerpreter.dense_activation_store import DenseActivationStore
from interpret.sae.autointerpreter.wordnet_samples import WordNetSampleIterator
from interpret.utils.wordnet_parser import WordNetParser


def capture_residual_prefill(
    wrapper: GemmaPytorchInference,
    sites: list[ResidualSiteSpec],
    aggregations: list[str],
    dtype: np.dtype,
) -> list[tuple[np.ndarray, int]]:
    """Read the fork's prefill activation cache and aggregate per (site, mode).

    Returns one ``(vec, seq_len)`` per (site, aggregation) in **site-major**
    order — aligned with :func:`residual_unit_layout` — so a single-mode
    call yields one vector per site (the historical shape). Each site's
    cached tensor is read once and aggregated for every mode (the cache
    read, not the aggregation, is what the forward pass paid for).

    Shared by :class:`ResidualCollector` and the SAE collector's
    side-capture path so both read the cache and treat BOS identically.
    Call after a forward pass inside ``wrapper.cache_activations(...)``.
    """
    prefill = wrapper.get_cached_activations().get("prefill")
    if not prefill:
        raise RuntimeError("activation cache captured no prefill record")
    out: list[tuple[np.ndarray, int]] = []
    for site in sites:
        if site.intermediate == "final_norm":
            tensor = prefill.get("final_norm")
        else:
            tensor = prefill.get(site.layer_index, {}).get(site.intermediate)
        if tensor is None:
            raise RuntimeError(
                f"no cached activation for site {residual_subdir(site)}"
            )
        feats = tensor[0].to(torch.float32)  # (seq_len, hidden_size)
        seq_len = int(feats.shape[0])
        # Raw BOS residuals are extreme outliers — keep them out of
        # mean/max aggregates (last_token is unaffected).
        for agg in aggregations:
            vec = aggregate_tokens(
                feats, agg, skip_first_token=wrapper.prepends_bos,
            ).cpu().numpy()
            out.append((vec.astype(dtype, copy=False), seq_len))
    return out


class ResidualCollector:
    """Run the base LM on every WordNet entry and persist raw residuals.

    ``wrapper`` and ``parser`` are injectable for tests; ``run_dir`` lets
    the orchestrator override the derived default so a top-level
    ``run_slug`` in the umbrella experiment config wins.
    """

    def __init__(
        self,
        config: AutoInterpretCollectConfig,
        wrapper: GemmaPytorchInference | None = None,
        parser: WordNetParser | None = None,
        run_dir: Path | None = None,
    ) -> None:
        self.config = config
        self.base_spec: BaseModelSpec = config.resolve_base_model()
        self.spec: ResidualSourceSpec = config.resolve_residual()
        self.sites: list[ResidualSiteSpec] = list(self.spec.sites)
        if not self.sites:
            raise ValueError("ResidualSourceSpec produced an empty site list")
        if wrapper is None and self.base_spec.family != "gemma":
            raise NotImplementedError(
                "source_kind='residual' currently supports only the Gemma "
                "wrapper (forked gemma_pytorch activation cache); got "
                f"family={self.base_spec.family!r}"
            )
        self.wrapper = (
            wrapper
            if wrapper is not None
            else GemmaPytorchInference(
                self.base_spec.checkpoint, precision=self.base_spec.dtype,
            )
        )

        n_layers = len(self.wrapper.decoder_layers)
        for site in self.sites:
            if site.intermediate != "final_norm" and not (
                0 <= site.layer_index < n_layers
            ):
                raise ValueError(
                    f"residual site {residual_subdir(site)} is out of range "
                    f"for a {n_layers}-layer model"
                )

        if parser is not None:
            self.parser = parser
        elif config.wordnet_xml_path:
            xml_path = Path(config.wordnet_xml_path)
            if not xml_path.is_absolute():
                xml_path = (PROJECT_ROOT / xml_path).resolve()
            self.parser = WordNetParser(xml_file_path=str(xml_path))
        else:
            self.parser = WordNetParser()

        # Single-store standalone path: the validator guarantees one mode.
        # normalize_aggregations coerces a str / one-element list / None
        # (inherit collect-level) uniformly.
        self.aggregation = normalize_aggregations(
            self.spec.aggregation, config.resolve_aggregations(),
        )[0]

        self.run_dir = run_dir if run_dir is not None else config.run_dir()
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # One DenseActivationStore per site; multi-site runs land each site
        # in its own subdirectory so downstream stages can treat it as a
        # self-contained single-source run_dir (mirrors the multi-SAE layout).
        self.multi = len(self.sites) > 1
        self.n_features = int(self.wrapper.config.hidden_size)
        self.dtype = np.dtype(self.spec.activation_dtype)
        self.stores: list[DenseActivationStore] = []
        for site in self.sites:
            store_dir = self.run_dir / residual_subdir(site) if self.multi else self.run_dir
            store_dir.mkdir(parents=True, exist_ok=True)
            self.stores.append(
                DenseActivationStore(
                    store_dir, n_features=self.n_features, dtype=self.dtype,
                ),
            )
        self._per_store_seen: list[set[tuple[str, str]]] = [
            store.existing_row_keys() if config.resume else set()
            for store in self.stores
        ]

    # ── Sample iteration ───────────────────────────────────────────────

    def _iter_samples(self) -> Iterator[dict]:
        """Skip a sample only when **every** store already has it (resume)."""
        if self._per_store_seen:
            skip_keys = set.intersection(*self._per_store_seen)
        else:
            skip_keys = set()
        iterator = WordNetSampleIterator(
            self.parser,
            pos_filter=self.config.pos_filter,
            prompt_template=self.config.prompt_template,
            limit=self.config.limit,
        )
        return iterator.iter_samples(skip_keys)

    # ── Forward pass ───────────────────────────────────────────────────

    def _capture_vectors(self, prompt: str) -> list[tuple[np.ndarray, int]]:
        """One forward pass; return ``(vec, seq_len)`` per site (spec order).

        ``reset_prefill_cache`` re-arms the fork's one-shot prefill capture
        between samples; ``output_len=1`` keeps the pass minimal (the single
        decode step is not captured since ``last=False``).
        """
        self.wrapper.reset_prefill_cache()
        if self.base_spec.use_chat_template:
            self.wrapper.generate(prompt, output_len=1)
        else:
            self.wrapper.generate_from_template(prompt, output_len=1)
        return capture_residual_prefill(
            self.wrapper, self.sites, [self.aggregation], self.dtype,
        )

    # ── Run ────────────────────────────────────────────────────────────

    def run(self) -> Path:
        """Execute the full collection pass. Returns the run directory."""
        from interpret.sae.autointerpreter.config import AutoInterpretConfig

        dump_yaml(
            AutoInterpretConfig(run_slug=self.config.run_slug(), collect=self.config),
            self.run_dir / "experiment.yaml",
        )

        # One cache configuration serves every site: per-layer intermediates
        # are captured for the union of requested layers, final_norm (when
        # requested) is captured top-level regardless of the layer set.
        layer_set = {
            s.layer_index for s in self.sites if s.intermediate != "final_norm"
        }
        inter_set = {s.intermediate for s in self.sites}
        flush_every = max(1, self.config.flush_every)
        samples = self._iter_samples()
        with self.wrapper.cache_activations(
            layers=layer_set, intermediates=inter_set, prefill=True, last=False,
        ):
            for i, sample in enumerate(
                tqdm(samples, desc="collect-resid", unit="sample"),
            ):
                captured = self._capture_vectors(sample["prompt"])
                key = (sample["word"], sample["synset_id"])
                for site, store, seen, (vec, seq_len) in zip(
                    self.sites, self.stores, self._per_store_seen, captured, strict=True,
                ):
                    if key in seen:
                        continue
                    meta: dict[str, Any] = {
                        **sample,
                        "n_tokens": seq_len,
                        "layer": (
                            -1 if site.intermediate == "final_norm"
                            else site.layer_index
                        ),
                        "hook_type": site.intermediate,
                        "width": self.base_spec.checkpoint,
                        "aggregation": self.aggregation,
                    }
                    store.append(vec, meta)
                    seen.add(key)
                if (i + 1) % flush_every == 0:
                    for store in self.stores:
                        store.flush()

        for store in self.stores:
            store.flush()
        return self.run_dir
