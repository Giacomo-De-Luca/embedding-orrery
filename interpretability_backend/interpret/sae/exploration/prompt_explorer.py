"""Per-token SAE feature explorer for Gemma3 / Qwen3.

Runs a text prompt through the model with SAE hooks attached, then returns
per-token top-k feature activations with Neuronpedia labels (Gemma; Qwen
features are label-free until autointerp). Designed for interactive use in
Jupyter notebooks and reused by the backend prompt-activations service.

Usage::

    from interpret.inference.gemma_pytorch import GemmaPytorchInference
    from interpret.sae.exploration.prompt_explorer import PromptExplorer, PromptExplorerConfig

    wrapper = GemmaPytorchInference("google/gemma-3-4b-it")
    explorer = PromptExplorer(PromptExplorerConfig(wrapper=wrapper))

    result = explorer.run_prompt("The cat sat on the warm red mat")
    result                          # rich HTML table in Jupyter
    result.layer(29)                # single layer
    result.token(5)                 # all layers for one position

    detail = explorer.inspect_feature(14525, layer=29)
    detail                          # label, logits, top activation docs
"""

from __future__ import annotations

import html as html_module
import warnings
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from interpret.sae.exploration.explore_neuronpedia import ActivationExample, NeuronpediaExplorer
from interpret.sae.feature_labels import FeatureLabelStore
from interpret.sae.hook_manager import HookManager
from interpret.sae.sae_config import SAEConfig

if TYPE_CHECKING:
    from collections.abc import Callable

    from interpret.inference.gemma_pytorch import GemmaPytorchInference
    from interpret.inference.qwen3_transformers import Qwen3Inference

_DEFAULT_LABELS_DIR = Path("resources/sae_labels/neuronpedia_gemma-3-4b-it")


# ── Configuration ────────────────────────────────────────────────────────────


@dataclass
class PromptExplorerConfig:
    """Configuration for PromptExplorer. Edit fields or pass to constructor.

    The ``wrapper`` must be a loaded :class:`GemmaPytorchInference` or
    :class:`Qwen3Inference` instance. The wrapper supplies the chat template
    (``format_prompt``) and per-token pieces (``token_strings``), so this class
    stays family-agnostic.

    SAEs to hook are given either as ``layers`` (all at the shared ``width``)
    or, for mixed widths — including two widths at the same layer — as
    explicit ``saes`` (layer, width) specs, which take precedence.

    ``sae_config_factory`` maps a ``(layer, width)`` pair to the SAE config to
    hook. When ``None`` (the default, used by notebooks), a Gemma-scope config
    is built. The service injects a family-aware factory (its ``_make_sae_config``)
    so Qwen-scope configs are built for Qwen models.
    """

    wrapper: GemmaPytorchInference | Qwen3Inference
    layers: list[int] = field(default_factory=lambda: [9, 17, 22, 29])
    width: str = "16k"
    saes: list[tuple[int, str]] | None = None
    top_k: int = 10
    density_threshold: float = 0.01
    labels_dir: Path = _DEFAULT_LABELS_DIR
    skip_labels: bool = False
    model_size: str = "4b"
    variant: str = "it"
    sae_config_factory: Callable[[int, str], SAEConfig] | None = None

    def effective_saes(self) -> list[tuple[int, str]]:
        """The (layer, width) specs to hook: explicit ``saes`` or layers × width."""
        if self.saes:
            return list(self.saes)
        return [(layer, self.width) for layer in self.layers]


def _store_read_plan(
    configs: list[SAEConfig],
) -> list[tuple[SAEConfig, str]]:
    """Per-config ActivationStore read key, mirroring HookManager's write rule.

    A ``(layer, hook_type)`` site with a single SAE records under
    ``sae_id=""`` (legacy fast path); a shared site records each SAE under
    its ``identity()`` slug. Reads must match, or ``store.prefill`` returns
    ``None``.
    """
    site_counts = Counter((c.layer_index, c.hook_type) for c in configs)
    return [
        (c, c.identity() if site_counts[(c.layer_index, c.hook_type)] > 1 else "") for c in configs
    ]


# ── Result dataclasses ───────────────────────────────────────────────────────


@dataclass
class ActiveFeature:
    """A single SAE feature active at a token position."""

    index: int
    activation: float
    label: str
    density: float | None = None


@dataclass
class TokenFeatures:
    """Features active at one token position within a layer."""

    token: str
    position: int
    features: list[ActiveFeature]

    def __repr__(self) -> str:
        top3 = self.features[:3]
        feats = ", ".join(f"F{f.index}={f.activation:.1f}" for f in top3)
        suffix = f" (+{len(self.features) - 3} more)" if len(self.features) > 3 else ""
        return f"TokenFeatures({self.token!r} @{self.position}: {feats}{suffix})"

    def _repr_html_(self) -> str:
        esc = html_module.escape
        rows = []
        for f in self.features:
            density_str = f"{f.density:.4f}" if f.density is not None else ""
            bar_width = min(int(f.activation / max(self.features[0].activation, 1) * 100), 100)
            rows.append(
                f"<tr>"
                f"<td style='font-family:monospace'>{f.index}</td>"
                f"<td style='text-align:right'>{f.activation:.2f}</td>"
                f"<td><div style='background:#4a90d9;height:12px;width:{bar_width}%'></div></td>"
                f"<td style='font-size:0.85em'>{esc(f.label)}</td>"
                f"<td style='color:#888;font-size:0.85em'>{density_str}</td>"
                f"</tr>"
            )
        return (
            f"<div style='margin:2px 0'>"
            f"<b>{esc(self.token)}</b> <span style='color:#888'>@{self.position}</span>"
            f"<table style='border-collapse:collapse;margin:2px 0;width:100%'>"
            f"<tr><th>Feature</th><th>Act</th><th></th><th>Label</th><th>Density</th></tr>"
            + "\n".join(rows)
            + "</table></div>"
        )


@dataclass
class LayerResult:
    """Per-token features for one layer."""

    layer: int
    width: str
    tokens: list[TokenFeatures]
    feature_acts: torch.Tensor  # raw (seq_len, d_sae) for further analysis

    def token(self, position: int) -> TokenFeatures:
        return self.tokens[position]

    def __iter__(self):
        return iter(self.tokens)

    def __len__(self):
        return len(self.tokens)

    def __repr__(self) -> str:
        n_active = sum(len(t.features) for t in self.tokens)
        return (
            f"LayerResult(layer={self.layer}, tokens={len(self.tokens)}, "
            f"total_active_features={n_active})"
        )

    def _repr_html_(self) -> str:
        esc = html_module.escape
        header = (
            f"<h4 style='margin:8px 0 4px'>Layer {self.layer} "
            f"({self.width}, {len(self.tokens)} tokens)</h4>"
        )
        rows = []
        for tf in self.tokens:
            top_feats = tf.features[:5]
            feat_parts = []
            for f in top_feats:
                feat_parts.append(
                    f"<span style='background:#e8f0fe;padding:1px 4px;"
                    f"border-radius:3px;margin:1px;display:inline-block;"
                    f"font-size:0.85em'>"
                    f"F{f.index} <b>{f.activation:.1f}</b>"
                    f"</span>"
                )
            extra = f" +{len(tf.features) - 5}" if len(tf.features) > 5 else ""
            rows.append(
                f"<tr>"
                f"<td style='color:#888;text-align:right;padding-right:6px'>{tf.position}</td>"
                f"<td style='font-family:monospace;white-space:pre'>{esc(tf.token)}</td>"
                f"<td>{''.join(feat_parts)}{extra}</td>"
                f"</tr>"
            )
        return (
            header
            + "<table style='border-collapse:collapse;width:100%'>"
            + "<tr><th>Pos</th><th>Token</th><th>Top features</th></tr>"
            + "\n".join(rows)
            + "</table>"
        )


@dataclass
class PromptResult:
    """Top-level result from :meth:`PromptExplorer.run_prompt`.

    ``layers`` is keyed by ``(layer_index, width)`` so two SAEs at the same
    layer (e.g. L9 16k + L9 65k) hold separate entries.
    """

    prompt: str
    token_strings: list[str]
    layers: dict[tuple[int, str], LayerResult]
    generated_text: str | None = None

    def layer(self, idx: int) -> LayerResult:
        """The unique LayerResult for a layer index.

        Raises ``ValueError`` when the layer was hooked at several widths —
        index ``layers[(layer, width)]`` directly in that case.
        """
        matches = [lr for (layer, _w), lr in self.layers.items() if layer == idx]
        if not matches:
            raise KeyError(idx)
        if len(matches) > 1:
            widths = sorted(w for layer, w in self.layers if layer == idx)
            raise ValueError(
                f"Layer {idx} has {len(matches)} SAEs (widths: {', '.join(widths)}); "
                f"use PromptResult.layers[(layer, width)] instead."
            )
        return matches[0]

    def token(self, position: int) -> dict[tuple[int, str], TokenFeatures]:
        """All hooked SAEs' features for one token position."""
        return {key: lr.token(position) for key, lr in self.layers.items()}

    @staticmethod
    def _key_str(key: tuple[int, str]) -> str:
        layer, width = key
        return f"{layer}/{width}"

    def __repr__(self) -> str:
        layer_strs = ", ".join(self._key_str(k) for k in sorted(self.layers))
        return f"PromptResult({len(self.token_strings)} tokens, layers=[{layer_strs}])"

    def _repr_html_(self) -> str:
        esc = html_module.escape
        layer_strs = ", ".join(self._key_str(k) for k in sorted(self.layers))
        parts = [
            f"<div style='font-family:sans-serif'>"
            f"<h3>PromptExplorer result</h3>"
            f"<p><b>Prompt:</b> {esc(self.prompt)}</p>"
            f"<p><b>Tokens:</b> {len(self.token_strings)} | "
            f"<b>Layers:</b> {layer_strs}</p>"
        ]
        if self.generated_text:
            parts.append(f"<p><b>Generated:</b> {esc(self.generated_text)}</p>")
        for key in sorted(self.layers):
            parts.append(self.layers[key]._repr_html_())
        parts.append("</div>")
        return "\n".join(parts)


@dataclass
class FeatureDetail:
    """Detailed information about a single SAE feature."""

    index: int
    layer: int
    label: str | None
    density: float | None
    top_logits: list[tuple[str, float]]
    bottom_logits: list[tuple[str, float]]
    similar_features: list[tuple[int, float, str]]
    activation_examples: list[ActivationExample]

    def __repr__(self) -> str:
        return (
            f"FeatureDetail(F{self.index} @layer {self.layer}, "
            f"label={self.label!r}, density={self.density}, "
            f"{len(self.activation_examples)} examples)"
        )

    def _repr_html_(self) -> str:
        esc = html_module.escape
        density_str = f"{self.density:.5f}" if self.density is not None else "(unknown)"
        parts = [
            "<div style='font-family:sans-serif'>",
            f"<h3>Feature {self.index} — Layer {self.layer}</h3>",
            f"<p><b>Label:</b> {esc(self.label or '(none)')}</p>",
            f"<p><b>Density:</b> {density_str}</p>",
        ]

        # Logits tables
        if self.top_logits or self.bottom_logits:
            parts.append("<div style='display:flex;gap:24px;margin:8px 0'>")
            for title, logits in [
                ("Top logits", self.top_logits),
                ("Bottom logits", self.bottom_logits),
            ]:
                if not logits:
                    continue
                rows = "".join(
                    f"<tr><td style='font-family:monospace'>{esc(tok)}</td>"
                    f"<td style='text-align:right'>{score:.3f}</td></tr>"
                    for tok, score in logits[:10]
                )
                parts.append(
                    f"<div><b>{title}</b>"
                    f"<table style='border-collapse:collapse;margin:4px 0'>"
                    f"<tr><th>Token</th><th>Score</th></tr>{rows}</table></div>"
                )
            parts.append("</div>")

        # Similar features
        if self.similar_features:
            rows = "".join(
                f"<tr><td>F{idx}</td><td>{sim:.3f}</td>"
                f"<td style='font-size:0.85em'>{esc(lbl)}</td></tr>"
                for idx, sim, lbl in self.similar_features[:10]
            )
            parts.append(
                f"<b>Similar features</b>"
                f"<table style='border-collapse:collapse;margin:4px 0'>"
                f"<tr><th>Feature</th><th>Cosine</th><th>Label</th></tr>"
                f"{rows}</table>"
            )

        # Activation examples
        if self.activation_examples:
            parts.append(f"<b>Top activation examples ({len(self.activation_examples)})</b>")
            for i, ex in enumerate(self.activation_examples):
                # Highlight the peak token in the context
                parts.append(
                    f"<div style='margin:6px 0;padding:6px;background:#f8f8f8;"
                    f"border-left:3px solid #4a90d9;font-size:0.9em'>"
                    f"<b>#{i + 1}</b> max={ex.max_value:.1f} "
                    f"@ token {ex.max_token_index}<br>"
                    f"<span style='font-family:monospace'>{esc(ex.context)}</span>"
                    f"</div>"
                )

        parts.append("</div>")
        return "\n".join(parts)


# ── Explorer ─────────────────────────────────────────────────────────────────


class PromptExplorer:
    """Run prompts through a model + SAE hooks and explore per-token features.

    Family-agnostic: the ``wrapper`` (Gemma or Qwen) supplies the chat template
    and per-token pieces, and ``config.sae_config_factory`` builds the family's
    SAE configs. The notebook default (no factory, Gemma wrapper) is unchanged.

    Example::

        explorer = PromptExplorer(PromptExplorerConfig(wrapper=model))
        result = explorer.run_prompt("The sky is blue")
        result.layer(29)   # LayerResult for layer 29
        result.token(3)    # all layers for token at position 3

        detail = explorer.inspect_feature(14525, layer=29)
    """

    def __init__(self, config: PromptExplorerConfig) -> None:
        self._config = config
        self._wrapper = config.wrapper
        self._label_store: FeatureLabelStore | None = None
        self._neuronpedia: NeuronpediaExplorer | None = None
        self._density_masks: dict[tuple[int, str], torch.Tensor] = {}

    @property
    def config(self) -> PromptExplorerConfig:
        return self._config

    @property
    def label_store(self) -> FeatureLabelStore:
        if self._label_store is None:
            self._label_store = FeatureLabelStore(self._config.labels_dir)
        return self._label_store

    @property
    def neuronpedia(self) -> NeuronpediaExplorer:
        if self._neuronpedia is None:
            from interpret.sae.feature_labels import _width_as_int

            specs = self._config.effective_saes()
            widths = {w for _, w in specs}
            if len(widths) != 1:
                raise ValueError(
                    "NeuronpediaExplorer supports a single width; config resolves "
                    f"to widths {sorted(widths)}. Use a single-width "
                    "PromptExplorerConfig for feature inspection."
                )
            self._neuronpedia = NeuronpediaExplorer(
                layers=[layer for layer, _ in specs],
                width=_width_as_int(next(iter(widths))),
                labels_dir=self._config.labels_dir,
            )
        return self._neuronpedia

    def __repr__(self) -> str:
        specs = ", ".join(f"{layer}/{w}" for layer, w in self._config.effective_saes())
        return (
            f"PromptExplorer(saes=[{specs}], "
            f"top_k={self._config.top_k}, "
            f"density_threshold={self._config.density_threshold})"
        )

    # ── Prompt execution ─────────────────────────────────────────────────

    def _build_sae_config(self, layer: int, width: str) -> SAEConfig:
        """Build the SAE config to hook for one ``(layer, width)`` spec.

        Uses the injected ``sae_config_factory`` when present (the service
        supplies a family-aware one) so Qwen-scope configs are built for Qwen
        models. Falls back to a Gemma-scope config for standalone/notebook use.
        """
        factory = self._config.sae_config_factory
        if factory is not None:
            return factory(layer, width)
        return SAEConfig(
            layer_index=layer,
            width=width,
            model_size=self._config.model_size,
            variant=self._config.variant,
            device=str(self._wrapper.device),
            prefill_only=True,
            read_only=True,
        )

    def _get_density_mask(self, sae_config: SAEConfig) -> torch.Tensor:
        """Get or build a boolean density mask for a layer (True = keep)."""
        key = (sae_config.layer_index, sae_config.width)
        if key not in self._density_masks:
            params = FeatureLabelStore.params_from_config(sae_config)
            densities = self.label_store.get_densities(*params)
            mask = (densities > 0) & (densities < self._config.density_threshold)
            self._density_masks[key] = mask
        return self._density_masks[key]

    @property
    def _is_base_model(self) -> bool:
        """True if using a base (pretrained) model without chat template."""
        return self._config.variant == "pt"

    def run_prompt(
        self,
        prompt: str,
        output_len: int = 1,
        top_k: int | None = None,
    ) -> PromptResult:
        """Run a prompt through the model with SAE hooks and collect features.

        Args:
            prompt: Raw user prompt. Chat template is applied for IT models;
                    base (pt) models receive the raw text with BOS only.
            output_len: Tokens to generate (1 is enough for activation capture).
            top_k: Override config top_k for this call. 0 = all non-zero features.

        Returns:
            PromptResult with per-token features for each layer.
        """
        k = top_k if top_k is not None else self._config.top_k

        # Build SAE configs and hook manager — one per (layer, width) spec.
        # Two widths at the same layer co-attach at one site (read_only).
        sae_configs: list[SAEConfig] = []
        manager = HookManager()
        for layer, width in self._config.effective_saes():
            cfg = self._build_sae_config(layer, width)
            sae_configs.append(cfg)
            manager.add_sae(cfg)
        read_plan = _store_read_plan(sae_configs)

        # Base models: pass raw text (no chat template).
        # IT models: the wrapper applies its family's chat template.
        # token_strings is aligned to the prefill sequence by the wrapper
        # (see GemmaPytorchInference/Qwen3Inference.token_strings).
        if self._is_base_model:
            formatted = prompt
        else:
            formatted = self._wrapper.format_prompt(prompt)
        token_strings = self._wrapper.token_strings(formatted)

        # Run inference with hooks
        with manager.session(self._wrapper.model.model.layers) as store:
            generated = self._wrapper.generate_from_template(
                formatted,
                output_len=output_len,
            )

            # Collect per-SAE results, keyed by (layer, width)
            layer_results: dict[tuple[int, str], LayerResult] = {}
            for cfg, read_sae_id in read_plan:
                record = store.prefill(
                    layer=cfg.layer_index,
                    hook_type=cfg.hook_type,
                    sae_id=read_sae_id,
                )
                if record is None:
                    continue

                # feature_acts: (batch, seq_len, d_sae) → (seq_len, d_sae)
                feature_acts = record.feature_acts[0]
                if feature_acts.shape[0] != len(token_strings):
                    warnings.warn(
                        f"Layer {cfg.layer_index}: feature_acts has "
                        f"{feature_acts.shape[0]} positions but tokenizer "
                        f"produced {len(token_strings)} tokens — token labels "
                        f"may be misaligned.",
                        stacklevel=2,
                    )
                if self._config.skip_labels:
                    # Service mode: no JSONL/SQLite access.  Density filtering
                    # is handled by the service layer via DuckDB (authoritative).
                    mask = None
                    if k > 0:
                        per_token = self._unlabelled_top_k_per_token(
                            feature_acts,
                            k=k,
                            mask=mask,
                        )
                    else:
                        per_token = self._unlabelled_all_nonzero_per_token(
                            feature_acts,
                            mask=mask,
                        )
                    densities = torch.zeros(feature_acts.shape[1])
                else:
                    # Notebook mode: JSONL/SQLite labels with fallback
                    params = FeatureLabelStore.params_from_config(cfg)
                    try:
                        mask = self._get_density_mask(cfg)
                    except FileNotFoundError:
                        mask = None

                    try:
                        if k > 0:
                            per_token = self.label_store.label_top_k_per_token(
                                feature_acts,
                                *params,
                                k=k,
                                mask=mask,
                            )
                        else:
                            per_token = self._all_nonzero_per_token(
                                feature_acts,
                                params,
                                mask,
                            )
                    except FileNotFoundError:
                        if k > 0:
                            per_token = self._unlabelled_top_k_per_token(
                                feature_acts,
                                k=k,
                                mask=mask,
                            )
                        else:
                            per_token = self._unlabelled_all_nonzero_per_token(
                                feature_acts,
                                mask=mask,
                            )

                    try:
                        densities = self.label_store.get_densities(*params)
                    except FileNotFoundError:
                        densities = torch.zeros(feature_acts.shape[1])
                tokens_list: list[TokenFeatures] = []
                for pos, features_at_pos in enumerate(per_token):
                    active = [
                        ActiveFeature(
                            index=idx,
                            activation=act_val,
                            label=label,
                            density=float(densities[idx]) if idx < len(densities) else None,
                        )
                        for idx, act_val, label in features_at_pos
                    ]
                    tok_str = token_strings[pos] if pos < len(token_strings) else f"[{pos}]"
                    tokens_list.append(
                        TokenFeatures(
                            token=tok_str,
                            position=pos,
                            features=active,
                        )
                    )

                layer_results[(cfg.layer_index, cfg.width)] = LayerResult(
                    layer=cfg.layer_index,
                    width=cfg.width,
                    tokens=tokens_list,
                    feature_acts=feature_acts.cpu(),
                )

        return PromptResult(
            prompt=prompt,
            token_strings=token_strings,
            layers=layer_results,
            generated_text=generated if output_len > 0 else None,
        )

    @staticmethod
    def _unlabelled_top_k_per_token(
        feature_acts: torch.Tensor,
        k: int,
        mask: torch.Tensor | None = None,
    ) -> list[list[tuple[int, float, str]]]:
        """Return top-k features per token without labels (fallback when no label file)."""
        result = []
        for pos in range(feature_acts.shape[0]):
            acts = feature_acts[pos].detach().float().cpu()
            if mask is not None:
                acts = torch.where(mask.cpu(), acts, torch.tensor(float("-inf")))
            topk = torch.topk(acts, k=min(k, acts.shape[0]))
            token_feats = []
            for val, idx in zip(topk.values, topk.indices, strict=True):
                if val.item() == float("-inf") or val.item() <= 0:
                    break
                token_feats.append((idx.item(), float(val), ""))
            result.append(token_feats)
        return result

    @staticmethod
    def _unlabelled_all_nonzero_per_token(
        feature_acts: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> list[list[tuple[int, float, str]]]:
        """Return all nonzero features per token without labels."""
        mask_cpu = mask.cpu() if mask is not None else None
        zeros = torch.zeros(feature_acts.shape[1])
        result = []
        for pos in range(feature_acts.shape[0]):
            acts = feature_acts[pos].detach().float().cpu()
            if mask_cpu is not None:
                acts = torch.where(mask_cpu, acts, zeros)
            nonzero_idx = torch.nonzero(acts, as_tuple=True)[0]
            if len(nonzero_idx) == 0:
                result.append([])
                continue
            vals = acts[nonzero_idx]
            order = vals.argsort(descending=True)
            sorted_idx = nonzero_idx[order]
            sorted_vals = vals[order]
            result.append(
                [(idx.item(), float(sorted_vals[i]), "") for i, idx in enumerate(sorted_idx)]
            )
        return result

    def _all_nonzero_per_token(
        self,
        feature_acts: torch.Tensor,
        params: tuple[str, int, str, str],
        mask: torch.Tensor | None,
    ) -> list[list[tuple[int, float, str]]]:
        """Return all non-zero features per token (when top_k=0)."""
        mask_cpu = mask.cpu() if mask is not None else None
        zeros = torch.zeros(feature_acts.shape[1])
        result = []
        for pos in range(feature_acts.shape[0]):
            acts = feature_acts[pos].detach().float().cpu()
            if mask_cpu is not None:
                acts = torch.where(mask_cpu, acts, zeros)
            nonzero_idx = torch.nonzero(acts, as_tuple=True)[0]
            if len(nonzero_idx) == 0:
                result.append([])
                continue
            # Sort by activation descending
            vals = acts[nonzero_idx]
            order = vals.argsort(descending=True)
            sorted_idx = nonzero_idx[order]
            sorted_vals = vals[order]
            # Batch label lookup
            indices = sorted_idx.tolist()
            labels = self.label_store.get_labels(indices, *params)
            result.append(
                [(idx, float(sorted_vals[i]), labels.get(idx, "")) for i, idx in enumerate(indices)]
            )
        return result

    # ── Feature inspection ───────────────────────────────────────────────

    def inspect_feature(
        self,
        feature_index: int,
        layer: int,
        top_k_docs: int = 5,
        top_k_similar: int = 10,
    ) -> FeatureDetail:
        """Get detailed information about a specific SAE feature.

        Args:
            feature_index: The feature index within the SAE.
            layer: Which layer the feature belongs to.
            top_k_docs: Number of top-activating documents to retrieve.
            top_k_similar: Number of similar features to find.

        Returns:
            FeatureDetail with label, logits, similar features, and examples.
        """
        cfg = SAEConfig(
            layer_index=layer,
            width=self._config.width,
            model_size=self._config.model_size,
            variant=self._config.variant,
        )
        params = FeatureLabelStore.params_from_config(cfg)

        # Label and density
        feature_record = self.label_store.get_feature(feature_index, *params)
        label = feature_record["label"] if feature_record else None
        density = feature_record["density"] if feature_record else None

        # Logits
        logits = self.label_store.get_logits(feature_index, *params)
        top_logits = logits.get("top", [])
        bottom_logits = logits.get("bottom", [])

        # Similar features
        try:
            similar = self.label_store.find_similar_features(
                feature_index,
                *params,
                k=top_k_similar,
            )
        except (ValueError, RuntimeError):
            similar = []

        # Top activation documents from Neuronpedia
        try:
            examples = self.neuronpedia.get_top_activations(
                feature_index,
                layer=layer,
                k=top_k_docs,
            )
        except FileNotFoundError:
            examples = []

        return FeatureDetail(
            index=feature_index,
            layer=layer,
            label=label,
            density=density,
            top_logits=top_logits,
            bottom_logits=bottom_logits,
            similar_features=similar,
            activation_examples=examples,
        )
