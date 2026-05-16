"""Per-token SAE feature explorer for Gemma3.

Runs a text prompt through Gemma with SAE hooks attached, then returns
per-token top-k feature activations with Neuronpedia labels. Designed
for interactive use in Jupyter notebooks.

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
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from interpret.sae.exploration.explore_neuronpedia import ActivationExample, NeuronpediaExplorer
from interpret.sae.feature_labels import FeatureLabelStore
from interpret.sae.hook_manager import HookManager
from interpret.sae.sae_config import SAEConfig

if TYPE_CHECKING:
    from interpret.inference.gemma_pytorch import GemmaPytorchInference

_DEFAULT_LABELS_DIR = Path("resources/sae_labels/neuronpedia_gemma-3-4b-it")


# ── Configuration ────────────────────────────────────────────────────────────

@dataclass
class PromptExplorerConfig:
    """Configuration for PromptExplorer. Edit fields or pass to constructor.

    The ``wrapper`` must be a loaded :class:`GemmaPytorchInference` instance.
    """

    wrapper: GemmaPytorchInference
    layers: list[int] = field(default_factory=lambda: [9, 17, 22, 29])
    width: str = "16k"
    top_k: int = 10
    density_threshold: float = 0.01
    labels_dir: Path = _DEFAULT_LABELS_DIR


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
    """Top-level result from :meth:`PromptExplorer.run_prompt`."""

    prompt: str
    token_strings: list[str]
    layers: dict[int, LayerResult]
    generated_text: str | None = None

    def layer(self, idx: int) -> LayerResult:
        return self.layers[idx]

    def token(self, position: int) -> dict[int, TokenFeatures]:
        """All layers' features for one token position."""
        return {
            layer_idx: lr.token(position)
            for layer_idx, lr in self.layers.items()
        }

    def __repr__(self) -> str:
        layer_strs = ", ".join(str(l) for l in sorted(self.layers))
        return (
            f"PromptResult({len(self.token_strings)} tokens, "
            f"layers=[{layer_strs}])"
        )

    def _repr_html_(self) -> str:
        esc = html_module.escape
        parts = [
            f"<div style='font-family:sans-serif'>"
            f"<h3>PromptExplorer result</h3>"
            f"<p><b>Prompt:</b> {esc(self.prompt)}</p>"
            f"<p><b>Tokens:</b> {len(self.token_strings)} | "
            f"<b>Layers:</b> {', '.join(str(l) for l in sorted(self.layers))}</p>"
        ]
        if self.generated_text:
            parts.append(f"<p><b>Generated:</b> {esc(self.generated_text)}</p>")
        for layer_idx in sorted(self.layers):
            parts.append(self.layers[layer_idx]._repr_html_())
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
            for title, logits in [("Top logits", self.top_logits), ("Bottom logits", self.bottom_logits)]:
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
                    f"<b>#{i+1}</b> max={ex.max_value:.1f} "
                    f"@ token {ex.max_token_index}<br>"
                    f"<span style='font-family:monospace'>{esc(ex.context)}</span>"
                    f"</div>"
                )

        parts.append("</div>")
        return "\n".join(parts)


# ── Explorer ─────────────────────────────────────────────────────────────────

class PromptExplorer:
    """Run prompts through Gemma + SAE hooks and explore per-token features.

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
            self._neuronpedia = NeuronpediaExplorer(
                layers=self._config.layers,
                width=_width_as_int(self._config.width),
                labels_dir=self._config.labels_dir,
            )
        return self._neuronpedia

    def __repr__(self) -> str:
        return (
            f"PromptExplorer(layers={self._config.layers}, "
            f"width={self._config.width!r}, "
            f"top_k={self._config.top_k}, "
            f"density_threshold={self._config.density_threshold})"
        )

    # ── Prompt execution ─────────────────────────────────────────────────

    @staticmethod
    def _format_prompt(prompt: str) -> str:
        """Apply the standard Gemma chat template.

        Must match the formatting in GemmaPytorchInference.generate().
        """
        return (
            f"<start_of_turn>user\n{prompt}<end_of_turn>\n"
            f"<start_of_turn>model"
        )

    def _tokenize_to_strings(self, formatted_prompt: str) -> list[str]:
        """Tokenize a formatted prompt and return per-token string pieces."""
        sp = self._wrapper.tokenizer.sp_model
        token_ids = self._wrapper.tokenize(formatted_prompt, bos=True)
        return [sp.IdToPiece(tid) for tid in token_ids]

    def _get_density_mask(self, sae_config: SAEConfig) -> torch.Tensor:
        """Get or build a boolean density mask for a layer (True = keep)."""
        key = (sae_config.layer_index, sae_config.width)
        if key not in self._density_masks:
            params = FeatureLabelStore.params_from_config(sae_config)
            densities = self.label_store.get_densities(*params)
            mask = (densities > 0) & (densities < self._config.density_threshold)
            self._density_masks[key] = mask
        return self._density_masks[key]

    def run_prompt(
        self, prompt: str, output_len: int = 1, top_k: int | None = None,
    ) -> PromptResult:
        """Run a prompt through Gemma with SAE hooks and collect features.

        Args:
            prompt: Raw user prompt (chat template is applied automatically).
            output_len: Tokens to generate (1 is enough for activation capture).
            top_k: Override config top_k for this call. 0 = all non-zero features.

        Returns:
            PromptResult with per-token features for each layer.
        """
        k = top_k if top_k is not None else self._config.top_k

        # Build SAE configs and hook manager
        sae_configs: dict[int, SAEConfig] = {}
        manager = HookManager()
        for layer in self._config.layers:
            cfg = SAEConfig(
                layer_index=layer,
                width=self._config.width,
                device=str(self._wrapper.device),
                prefill_only=True,
                read_only=True,
            )
            sae_configs[layer] = cfg
            manager.add_sae(cfg)

        # Format prompt and get token strings
        formatted = self._format_prompt(prompt)
        token_strings = self._tokenize_to_strings(formatted)

        # Run inference with hooks
        with manager.session(self._wrapper.model.model.layers) as store:
            generated = self._wrapper.generate_from_template(
                formatted, output_len=output_len,
            )

            # Collect per-layer results
            layer_results: dict[int, LayerResult] = {}
            for layer, cfg in sae_configs.items():
                record = store.prefill(layer=layer)
                if record is None:
                    continue

                # feature_acts: (batch, seq_len, d_sae) → (seq_len, d_sae)
                feature_acts = record.feature_acts[0]
                if feature_acts.shape[0] != len(token_strings):
                    warnings.warn(
                        f"Layer {layer}: feature_acts has {feature_acts.shape[0]} "
                        f"positions but tokenizer produced {len(token_strings)} "
                        f"tokens — token labels may be misaligned.",
                        stacklevel=2,
                    )
                params = FeatureLabelStore.params_from_config(cfg)
                try:
                    mask = self._get_density_mask(cfg)
                except FileNotFoundError:
                    # No label file for this layer/hook/width — skip density masking
                    mask = None

                # Get per-token top-k labels
                try:
                    if k > 0:
                        per_token = self.label_store.label_top_k_per_token(
                            feature_acts, *params, k=k, mask=mask,
                        )
                    else:
                        # All non-zero features per token
                        per_token = self._all_nonzero_per_token(
                            feature_acts, params, mask,
                        )
                except FileNotFoundError:
                    # No label file — fall back to unlabelled top-k
                    per_token = self._unlabelled_top_k_per_token(
                        feature_acts, k=k, mask=mask,
                    )

                # Build TokenFeatures for each position
                # Densities for annotation
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
                    tokens_list.append(TokenFeatures(
                        token=tok_str, position=pos, features=active,
                    ))

                layer_results[layer] = LayerResult(
                    layer=layer,
                    width=self._config.width,
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
            for val, idx in zip(topk.values, topk.indices):
                if val.item() == float("-inf") or val.item() <= 0:
                    break
                token_feats.append((idx.item(), float(val), ""))
            result.append(token_feats)
        return result

    def _all_nonzero_per_token(
        self,
        feature_acts: torch.Tensor,
        params: tuple[str, int, str, str],
        mask: torch.Tensor,
    ) -> list[list[tuple[int, float, str]]]:
        """Return all non-zero features per token (when top_k=0)."""
        mask_cpu = mask.cpu()
        zeros = torch.zeros(feature_acts.shape[1])
        result = []
        for pos in range(feature_acts.shape[0]):
            acts = feature_acts[pos].detach().float().cpu()
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
            result.append([
                (idx, float(sorted_vals[i]), labels.get(idx, ""))
                for i, idx in enumerate(indices)
            ])
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
        cfg = SAEConfig(layer_index=layer, width=self._config.width)
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
                feature_index, *params, k=top_k_similar,
            )
        except (ValueError, RuntimeError):
            similar = []

        # Top activation documents from Neuronpedia
        try:
            examples = self.neuronpedia.get_top_activations(
                feature_index, layer=layer, k=top_k_docs,
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
