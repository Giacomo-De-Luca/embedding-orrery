"""Verify SAE feature-index alignment against Neuronpedia activation data.

For each (layer, feature) under test, this script:

1. Loads one of Neuronpedia's top-activating token sequences for that
   feature from the downloaded activation JSONL.
2. Reconstructs the full text (chat template + user prompt + model
   response) from the token list and feeds it directly via the wrapper's
   internal `_generate` path so the chat template is NOT re-applied.
3. Captures the feature-activation tensor via our SAE hook on the prefill.
4. Reports the max activation of the target feature across all positions,
   the token it peaks on, and how the target feature ranks among all
   features at that token.

If our SAE indices are correctly aligned with Neuronpedia's, the target
feature should peak with a value in the same order of magnitude as
Neuronpedia's reported `maxValue`, ideally on the same context.

Run with::

    uv run python -m interpret.sae.diagnostics.check_feature_index_alignment
"""

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from interpret.inference.gemma_pytorch import GemmaPytorchInference
from interpret.sae import HookManager, SAEConfig

LABELS_DIR = Path("resources/sae_labels/neuronpedia_gemma-3-4b-it")


@dataclass
class AlignmentCase:
    layer: int
    feature: int
    label: str  # free-form description for the printout


CASES = [
    AlignmentCase(layer=29, feature=14525, label="pirate and treasure"),
    AlignmentCase(layer=17, feature=9511, label="'crypto pirates' (autointerp label)"),
    AlignmentCase(layer=9, feature=2721, label="'sequence, cascade, decades, pirate'"),
]


def load_top_record(layer: int, feature: int) -> dict | None:
    """Return the highest-maxValue activation record for a (layer, feature)."""
    path = LABELS_DIR / f"gemma-3-4b-it_{layer}-gemmascope-2-res-16k_activations.jsonl"
    best: dict | None = None
    with open(path) as f:
        for line in f:
            e = json.loads(line)
            if e.get("index") != feature:
                continue
            if best is None or e.get("maxValue", 0) > best.get("maxValue", 0):
                best = e
    return best


def reconstruct_text(tokens: list[str]) -> str:
    """Reconstruct the verbatim text string from a SentencePiece token list.

    Joins tokens directly. Works for Gemma because its tokenizer encodes
    leading whitespace inside the token strings already (via the
    SentencePiece ``▁`` marker which Neuronpedia exports as a literal
    space prefix). Special tokens like ``<bos>`` are kept verbatim and
    will be re-tokenized to themselves.
    """
    return "".join(tokens)


def run_alignment_check(
    wrapper: GemmaPytorchInference,
    case: AlignmentCase,
) -> None:
    print(f"\n===== layer {case.layer} / feature {case.feature}  ({case.label}) =====")

    record = load_top_record(case.layer, case.feature)
    if record is None:
        print("  no activation records found.")
        return

    np_max = record["maxValue"]
    np_max_idx = record["maxValueTokenIndex"]
    np_tokens = record["tokens"]
    np_top_tok = np_tokens[np_max_idx]
    np_context = "".join(
        np_tokens[max(0, np_max_idx - 4) : np_max_idx + 5]
    )
    print(f"  Neuronpedia: max={np_max:.1f} on tok {np_max_idx}={np_top_tok!r}")
    print(f"              context: {np_context!r}")

    full_text = reconstruct_text(np_tokens)
    print(f"  reconstructed text: {len(np_tokens)} tokens, {len(full_text)} chars")

    manager = HookManager()
    config = SAEConfig(layer_index=case.layer, device="mps", prefill_only=True)
    manager.add_sae(config)

    # Bypass the chat template wrapper in `generate()` — feed the
    # reconstructed sequence directly via the lower-level `_generate`.
    with manager.session(wrapper.model.model.layers) as store:
        wrapper._generate([full_text], output_len=1)
        feats = store.prefill(layer=case.layer).feature_acts[0]  # (seq, d_sae)

    feats = feats.detach().float().cpu()
    seq_len, d_sae = feats.shape

    our_col = feats[:, case.feature]
    our_max = float(our_col.max().item())
    our_pos = int(our_col.argmax().item())

    # What feature actually wins at our top position?
    topk_at_pos = torch.topk(feats[our_pos], k=5)
    # And what's the rank of the target feature at its own peak position?
    sorted_at_pos = torch.argsort(feats[our_pos], descending=True)
    target_rank = int((sorted_at_pos == case.feature).nonzero(as_tuple=True)[0].item())

    print(f"  Our SAE capture: {seq_len} tokens, {d_sae} features")
    print(
        f"  Feature {case.feature} peak: {our_max:.1f} "
        f"at position {our_pos} (rank {target_rank + 1} of {d_sae} at that token)"
    )
    print(f"  Top-5 features at position {our_pos}:")
    for val, idx in zip(topk_at_pos.values.tolist(), topk_at_pos.indices.tolist(), strict=True):
        marker = " <-- target" if idx == case.feature else ""
        print(f"    feat {idx:6d}  act={val:8.1f}{marker}")


def main() -> None:
    wrapper = GemmaPytorchInference("google/gemma-3-4b-it")
    for case in CASES:
        run_alignment_check(wrapper, case)


if __name__ == "__main__":
    main()
