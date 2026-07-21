"""Stage 1 — collect SAE feature activations for WordNet prompts.

For every ``(word, synset)`` pair in WordNet, format the configured
prompt template, run it through the base LM with N SAE hooks attached,
aggregate the per-token feature activations into one vector per SAE, and
append each to its own sparse CSR store (see
:class:`SparseActivationStore`).

Reuses:

- :class:`WordNetParser` (interpret/utils/wordnet_parser.py)
- :class:`GemmaPytorchInference` / :class:`Qwen3Inference`
  (interpret/inference/) — selected by ``BaseModelSpec.family``.
- :class:`HookManager` + SAE configs (interpret/sae/) — co-attaches
  every SAE in the resolved spec list.

Aggregation is configurable (``last_token`` / ``mean_prefill`` /
``max_prefill``), as a single mode or a list — a list writes one store
per (SAE, aggregation) from the same forward pass. ``mean_prefill`` and
``max_prefill`` exclude the BOS position when the wrapper prepends one
(``wrapper.prepends_bos``); BOS activations are extreme outliers that
would otherwise dominate the aggregate.

Store layout (see ``sae_unit_layout`` in config.py):
- 1 SAE × 1 aggregation  → directly into ``run_dir/`` (legacy layout —
  preserves backwards compat with existing single-SAE runs).
- N SAEs × 1 aggregation → ``run_dir/<sae_subdir>/`` per SAE (historical
  multi-SAE layout).
- any × M aggregations   → ``run_dir/<sae_subdir>_<aggregation>/``.

Resume:
A sample is re-run only when at least one store is missing it. Each
store skips its own append when it already has the key, so multi-SAE
runs that crashed mid-flush converge without duplicate rows.
"""

from __future__ import annotations

import contextlib
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from interpret.inference.gemma_pytorch import GemmaPytorchInference
from interpret.inference.qwen3_transformers import Qwen3Inference
from interpret.sae.autointerpreter.config import (
    PROJECT_ROOT,
    AutoInterpretCollectConfig,
    BaseModelSpec,
    ResidualSiteSpec,
    SAESpec,
    dump_yaml,
    residual_unit_layout,
    sae_unit_layout,
)
from interpret.sae.autointerpreter.dense_activation_store import DenseActivationStore
from interpret.sae.autointerpreter.sparse_activation_store import SparseActivationStore
from interpret.sae.autointerpreter.wordnet_samples import WordNetSampleIterator
from interpret.sae.hook_manager import HookManager
from interpret.sae.sae_config import HookType
from interpret.utils.wordnet_parser import WordNetParser


def aggregate_tokens(
    feats: torch.Tensor, mode: str, skip_first_token: bool = False,
) -> torch.Tensor:
    """Reduce a per-token ``(seq_len, d)`` tensor to ``(d,)``.

    Shared by the SAE and residual collectors so both interpret the
    ``aggregation`` config knob identically.

    ``skip_first_token`` excludes position 0 from ``mean_prefill`` and
    ``max_prefill`` — pass ``wrapper.prepends_bos`` so BOS activations
    (notorious outliers that would dominate an unmasked max) never enter
    the aggregate. ``last_token`` is unaffected, and a length-1 sequence
    is never reduced to empty.
    """
    if mode == "last_token":
        return feats[-1]
    if skip_first_token and feats.shape[0] > 1:
        feats = feats[1:]
    if mode == "mean_prefill":
        return feats.mean(dim=0)
    if mode == "max_prefill":
        return feats.amax(dim=0)
    raise ValueError(f"unknown aggregation: {mode!r}")


class ActivationCollector:
    """Run base LM + N SAEs on every WordNet entry and persist activations.

    ``wrapper`` and ``parser`` are injectable for tests; ``run_dir`` lets
    the orchestrator override the SAE-derived default so a top-level
    ``run_slug`` in the umbrella experiment config wins.
    """

    def __init__(
        self,
        config: AutoInterpretCollectConfig,
        wrapper: GemmaPytorchInference | Qwen3Inference | None = None,
        parser: WordNetParser | None = None,
        run_dir: Path | None = None,
    ) -> None:
        self.config = config
        self.base_spec: BaseModelSpec = config.resolve_base_model()
        self.specs: list[SAESpec] = config.resolve_saes()
        if not self.specs:
            raise ValueError("AutoInterpretCollectConfig produced an empty SAE list")
        self.wrapper = wrapper if wrapper is not None else self._build_wrapper(self.base_spec)

        if parser is not None:
            self.parser = parser
        elif config.wordnet_xml_path:
            xml_path = Path(config.wordnet_xml_path)
            if not xml_path.is_absolute():
                xml_path = (PROJECT_ROOT / xml_path).resolve()
            self.parser = WordNetParser(xml_file_path=str(xml_path))
        else:
            self.parser = WordNetParser()

        # Build per-spec SAEConfigs with the collect-time hook policy
        # baked in (read-only, prefill-only, collect-last-only). Register
        # each on the HookManager. add_sae() enforces read_only=True at
        # shared (layer, hook_type) sites.
        self.manager = HookManager()
        self.sae_configs = []
        for spec in self.specs:
            sae_cfg = spec.to_sae_config()
            sae_cfg.dtype = self.base_spec.dtype
            sae_cfg.device = self.base_spec.device
            sae_cfg.collect_last_only = True
            sae_cfg.prefill_only = True
            sae_cfg.read_only = True
            self.manager.add_sae(sae_cfg)
            self.sae_configs.append(sae_cfg)

        self.run_dir = run_dir if run_dir is not None else config.run_dir()
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Whether prefills start with a BOS token. mean/max aggregation
        # must exclude it (BOS activations are extreme outliers that would
        # dominate an unmasked max); last_token is unaffected. Probed from
        # the wrapper: True for Gemma (fork always prepends <bos>), False
        # for Qwen3/3.5 (no BOS token — position 0 is real content).
        self._skip_first_token = bool(getattr(self.wrapper, "prepends_bos", False))

        # One SparseActivationStore per (SAE, aggregation). The layout —
        # legacy root for a single combo, one subdir per combo otherwise —
        # is shared with the runner via ``sae_unit_layout`` so downstream
        # stages always find the stores.
        self.aggregations = config.resolve_aggregations()
        self.unit_layout = sae_unit_layout(self.specs, self.aggregations)
        # Per-(layer, hook_type) SAE count. HookManager._make_sae_site_hook
        # stores records under sae_id="" when a site has exactly one SAE,
        # and under each SAE's identity() when a site has multiple. The
        # collector must mirror that decision per spec when reading the
        # prefill back, otherwise multi-SAE configs with some sites
        # singly-occupied (e.g. L9 alone + L29 shared) read None.
        self._site_count = Counter(
            (s.layer_index, s.hook_type) for s in self.specs
        )
        self.stores: list[SparseActivationStore] = []
        for unit_idx, (_spec, _agg, sub) in enumerate(self.unit_layout):
            # unit_layout is spec-major: units i*len(aggs)..(i+1)*len(aggs)-1
            # all belong to specs[i] / sae_configs[i].
            sae_cfg = self.sae_configs[unit_idx // len(self.aggregations)]
            store_dir = self.run_dir / sub if sub else self.run_dir
            store_dir.mkdir(parents=True, exist_ok=True)
            store = SparseActivationStore(
                store_dir,
                n_features=sae_cfg.d_sae,
                dtype=np.dtype(config.activation_dtype),
            )
            self.stores.append(store)
        # Optional side-capture: a `residual:` block on the SAE path
        # collects raw base-model embeddings (e.g. final_norm last_token)
        # from the same forward pass via the Gemma fork's activation
        # cache (validated Gemma-only). One dense store per (site,
        # aggregation), always under a named subdir — the run_dir root
        # belongs to the (single-)SAE layout. The residual aggregations
        # may differ from (and out-number) the SAE aggregations, e.g.
        # raw {last_token, max_prefill} alongside SAE max_prefill.
        self.residual_spec = config.residual
        self.residual_sites = (
            list(self.residual_spec.sites) if self.residual_spec else []
        )
        self.residual_layout: list[tuple[ResidualSiteSpec, str, str]] = []
        self.residual_stores: list[DenseActivationStore] = []
        if self.residual_sites:
            self._residual_aggs = config.resolve_residual_aggregations()
            self.residual_layout = residual_unit_layout(
                self.residual_sites, self._residual_aggs,
            )
            self._residual_dtype = np.dtype(self.residual_spec.activation_dtype)
            hidden = int(self.wrapper.config.hidden_size)
            for _site, _agg, sub in self.residual_layout:
                store_dir = self.run_dir / sub
                store_dir.mkdir(parents=True, exist_ok=True)
                self.residual_stores.append(
                    DenseActivationStore(
                        store_dir, n_features=hidden, dtype=self._residual_dtype,
                    ),
                )

        # Per-store in-memory tracking of (word, synset_id) keys already
        # on disk. Updated as we append so we don't re-issue store.append
        # for a sample one store already has.
        self._per_store_seen: list[set[tuple[str, str]]] = [
            store.existing_row_keys() if config.resume else set()
            for store in self.stores
        ]
        self._residual_seen: list[set[tuple[str, str]]] = [
            store.existing_row_keys() if config.resume else set()
            for store in self.residual_stores
        ]

    # ── Wrapper factory ────────────────────────────────────────────────

    @staticmethod
    def _build_wrapper(base: BaseModelSpec):
        if base.family == "gemma":
            return GemmaPytorchInference(base.checkpoint, precision=base.dtype)
        if base.family == "qwen":
            return Qwen3Inference(base.checkpoint, dtype=base.dtype, device=base.device)
        raise ValueError(f"Unsupported base model family: {base.family!r}")

    # ── Sample iteration ───────────────────────────────────────────────

    def _iter_samples(self) -> Iterator[dict]:
        """Yield ``{word, synset_id, pos, definition, prompt}`` dicts.

        Skips a sample only when **every** store (SAE and residual
        side-capture alike) already has it (intersection of per-store
        ``existing_row_keys``). When at least one store is missing the
        sample, the forward pass re-runs and the writer-side per-store
        check decides which stores append.

        The WordNet traversal itself is shared with the embedding collector
        via :class:`WordNetSampleIterator`.
        """
        all_seen = self._per_store_seen + self._residual_seen
        if all_seen:
            skip_keys = set.intersection(*all_seen)
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

    def _aggregate(self, feats: torch.Tensor, aggregation: str) -> torch.Tensor:
        """Reduce ``(seq_len, d_sae)`` to ``(d_sae,)`` per aggregation mode."""
        return aggregate_tokens(
            feats, aggregation, skip_first_token=self._skip_first_token,
        )

    def _capture_vectors(self, prompt: str) -> list[tuple[np.ndarray, int]]:
        """One forward pass; return ``(vec, seq_len)`` per (SAE, aggregation) unit.

        Output_len=1 keeps the model running one decode step after the
        prefill so the prefill sentinel fires and prefill-only hooks
        auto-skip subsequent decodes. ``use_chat_template`` toggles
        between chat-formatted ``generate`` and raw ``generate_from_template``;
        both wrappers expose ``generate_from_template`` after the Phase B
        edits.
        """
        self.manager.reset()
        if self.residual_sites:
            # Re-arm the fork's one-shot prefill cache for the side-capture.
            self.wrapper.reset_prefill_cache()
        if self.base_spec.use_chat_template:
            self.wrapper.generate(prompt, output_len=1)
        else:
            self.wrapper.generate_from_template(prompt, output_len=1)

        out: list[tuple[np.ndarray, int]] = []
        for spec, sae_cfg in zip(self.specs, self.sae_configs, strict=True):
            # HookManager writes records under sae_id="" when only one
            # SAE is at a site, and under config.identity() when more
            # than one. Use the per-site count so configs that mix
            # singly-occupied and multi-occupied sites (e.g. L9 alone
            # + L29 with two widths) read the right key on every spec.
            site_multi = self._site_count[(spec.layer_index, spec.hook_type)] > 1
            read_sae_id = sae_cfg.identity() if site_multi else ""
            record = self.manager.store.prefill(
                layer=spec.layer_index,
                hook_type=HookType(spec.hook_type),
                sae_id=read_sae_id,
            )
            if record is None:
                raise RuntimeError(
                    f"SAE hook did not capture a prefill record for spec "
                    f"L{spec.layer_index} {spec.hook_type} ({sae_cfg.identity()})"
                )
            feats = record.feature_acts[0]  # (seq_len, d_sae)
            seq_len = int(feats.shape[0])
            # One aggregation per unit, all from the same captured feats —
            # this is what makes multi-aggregation nearly free.
            for agg in self.aggregations:
                vec = self._aggregate(feats, agg).to(torch.float32).cpu().numpy()
                vec[np.abs(vec) < self.config.sparse_threshold] = 0.0
                out.append((vec.astype(self.config.activation_dtype, copy=False), seq_len))
        return out

    # ── Run ────────────────────────────────────────────────────────────

    def run(self) -> Path:
        """Execute the full collection pass. Returns the run directory."""
        from interpret.sae.autointerpreter.config import AutoInterpretConfig

        dump_yaml(
            AutoInterpretConfig(run_slug=self.config.run_slug(), collect=self.config),
            self.run_dir / "experiment.yaml",
        )

        # Co-site SAEs share one forward hook (HookManager groups by
        # site at attach time). The session() context manager attaches
        # once for the whole pass; reset() between samples clears the
        # store and re-arms prefill-only hooks. The side-capture (when
        # configured) additionally arms the fork's activation cache for
        # the whole pass — one cache config serves every site.
        if self.residual_sites:
            from interpret.sae.autointerpreter.collect_residuals import (
                capture_residual_prefill,
            )

            cache_ctx = self.wrapper.cache_activations(
                layers={
                    s.layer_index
                    for s in self.residual_sites
                    if s.intermediate != "final_norm"
                },
                intermediates={s.intermediate for s in self.residual_sites},
                prefill=True,
                last=False,
            )
        else:
            cache_ctx = contextlib.nullcontext()

        flush_every = max(1, self.config.flush_every)
        samples = self._iter_samples()
        all_stores = self.stores + self.residual_stores
        with self.manager.session(self.wrapper.decoder_layers), cache_ctx:
            for i, sample in enumerate(tqdm(samples, desc="collect", unit="sample")):
                captured = self._capture_vectors(sample["prompt"])
                key = (sample["word"], sample["synset_id"])
                for (spec, agg, _sub), store, seen, (vec, seq_len) in zip(
                    self.unit_layout,
                    self.stores,
                    self._per_store_seen,
                    captured, strict=True,
                ):
                    if key in seen:
                        continue
                    meta: dict[str, Any] = {
                        **sample,
                        "n_tokens": seq_len,
                        "layer": spec.layer_index,
                        "hook_type": spec.hook_type,
                        "width": spec.width,
                        "aggregation": agg,
                    }
                    store.append(vec, meta)
                    seen.add(key)
                if self.residual_sites:
                    resid_captured = capture_residual_prefill(
                        self.wrapper,
                        self.residual_sites,
                        self._residual_aggs,
                        self._residual_dtype,
                    )
                    for (site, r_agg, _sub), store, seen, (vec, seq_len) in zip(
                        self.residual_layout,
                        self.residual_stores,
                        self._residual_seen,
                        resid_captured, strict=True,
                    ):
                        if key in seen:
                            continue
                        meta = {
                            **sample,
                            "n_tokens": seq_len,
                            "layer": (
                                -1 if site.intermediate == "final_norm"
                                else site.layer_index
                            ),
                            "hook_type": site.intermediate,
                            "width": self.base_spec.checkpoint,
                            "aggregation": r_agg,
                        }
                        store.append(vec, meta)
                        seen.add(key)
                if (i + 1) % flush_every == 0:
                    for store in all_stores:
                        store.flush()

        for store in all_stores:
            store.flush()
        return self.run_dir
