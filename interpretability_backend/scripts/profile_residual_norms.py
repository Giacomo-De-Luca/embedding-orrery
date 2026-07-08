"""Offline profiler: per-layer residual-stream norms → frontend steering hint.

Runs a fixed set of test prompts through each configured model (Gemma / Qwen),
measures ``||h_L||`` at every decoder layer's RESID_POST site via
``ResidualNormProfiler``, also records each registered steering *direction*'s
vector norm, and writes a small JSON the frontend imports directly to turn a
raw steering coefficient into a "fraction of the residual stream" hint.

Because Gemma-scope SAE decoder rows are unit-norm, the per-layer table alone
is a complete hint for every SAE feature (rho = strength / ||h_L||); only the
direction presets carry their own ``||v||`` (emitted under ``directions``).

The output JSON is keyed by frontend model id and merged in place, so
re-running for one model never clobbers another's entry.

Run from the ``interpretability_backend/`` directory (so ``interpret`` and
``backend`` are importable as top-level packages), with the backend stopped
if it holds the GPU:

    uv run python -m scripts.profile_residual_norms

Config: ``scripts/profile_residual_norms_config.toml`` (or the path in
``ORRERY_RESIDUAL_NORMS_CONFIG``).
"""

import json
import os
import tomllib
from datetime import datetime
from pathlib import Path

import torch

from backend.services.interpret_service import DIRECTION_REGISTRY, InterpretService
from backend.services.model_registry import model_id_for_checkpoint
from backend.utils.resource_paths import DIRECTIONS_DIR
from interpret.inference.gemma_pytorch import GemmaPytorchInference
from interpret.inference.qwen3_transformers import Qwen3Inference
from interpret.inference.residual_norm_profiler import ResidualNormProfiler

# scripts/ -> interpretability_backend/ -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = Path(__file__).with_name("profile_residual_norms_config.toml")
DEFAULT_OUTPUT_PATH = REPO_ROOT / "embedding_visualization" / "lib" / "utils" / "residualNorms.json"

# Published GemmaScope-2 residual SAE layers per size (see interpret_service
# _DEFAULT_LAYERS_BY_SIZE) — used only for the calibration report below.
_ANCHOR_LAYERS = (9, 17, 22, 29)
_ANCHOR_STRENGTH = 800.0  # SteeringControls' current hardcoded SAE default


def _build_wrapper(checkpoint: str):
    """Instantiate the right inference wrapper; return (wrapper, frontend_model_id)."""
    family, model_size, variant = InterpretService._parse_checkpoint(checkpoint)
    checkpoint = InterpretService._normalize_checkpoint(checkpoint, family, model_size, variant)
    # Key by the stored/frontend model id (the store passes this to the hint):
    # "google/gemma-3-4b-it" -> "gemma-3-4b-it", "Qwen/Qwen3-1.7B" -> "qwen3-1.7B-base".
    model_id = model_id_for_checkpoint(checkpoint)
    if family == "qwen":
        wrapper = Qwen3Inference(checkpoint, dtype="bfloat16")
    else:
        wrapper = GemmaPytorchInference(checkpoint, model_size=model_size, precision="bfloat16")
    return wrapper, model_id


def _direction_norms(model_id: str) -> dict[str, dict]:
    """L2 norm + layer of each pre-extracted direction bound to this model."""
    out: dict[str, dict] = {}
    for preset in DIRECTION_REGISTRY.values():
        if preset.model_id != model_id:
            continue
        path = DIRECTIONS_DIR / preset.file
        if not path.exists():
            continue
        vec = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(vec, torch.Tensor) or vec.ndim != 1:
            continue
        out[preset.name] = {
            "layer": preset.layer,
            "vecNorm": float(torch.linalg.vector_norm(vec.to(torch.float32))),
        }
    return out


def profile_checkpoint(checkpoint: str, prompts: list[str], output_len: int):
    """Profile one checkpoint; return (frontend_model_id, json_entry)."""
    wrapper, model_id = _build_wrapper(checkpoint)
    profiler = ResidualNormProfiler(wrapper)
    layer_stats = profiler.profile(prompts, output_len=output_len)
    layers = [{"layer": layer, **layer_stats[layer]} for layer in sorted(layer_stats)]
    entry = {
        "checkpoint": checkpoint,
        "dModel": profiler.d_model,
        "nLayers": profiler.n_layers,
        "promptCount": len(prompts),
        "droppedBos": profiler.drop_bos,
        "generatedAt": datetime.now().isoformat(timespec="seconds"),
        "layers": layers,
        "directions": _direction_norms(model_id),
    }
    return model_id, entry


def _print_report(model_id: str, entry: dict) -> None:
    """Console table + rho calibration against the current strength-800 default."""
    by_layer = {layer_row["layer"]: layer_row for layer_row in entry["layers"]}
    print(f"\n=== {model_id}  (dModel={entry['dModel']}, {entry['nLayers']} layers) ===")
    print(f"  {'layer':>5} {'median':>10} {'p25':>10} {'p75':>10}")
    for layer_row in entry["layers"]:
        print(
            f"  {layer_row['layer']:>5} {layer_row['median']:>10.2f} "
            f"{layer_row['p25']:>10.2f} {layer_row['p75']:>10.2f}"
        )
    print(f"  strength {_ANCHOR_STRENGTH:.0f} as fraction of ||h_L|| (SAE v is unit-norm):")
    for layer in _ANCHOR_LAYERS:
        if layer in by_layer:
            rho = _ANCHOR_STRENGTH / by_layer[layer]["median"]
            print(
                f"    L{layer}: rho = {_ANCHOR_STRENGTH:.0f}/{by_layer[layer]['median']:.1f} = {rho:.3f}"
            )
    for name, meta in entry["directions"].items():
        layer = meta["layer"]
        if layer in by_layer:
            rho3 = 3.0 * meta["vecNorm"] / by_layer[layer]["median"]
            print(
                f"  direction {name}: ||v||={meta['vecNorm']:.2f} @ L{layer}; "
                f"strength 3 => rho = {rho3:.3f}"
            )


def main() -> None:
    config_path = Path(os.getenv("ORRERY_RESIDUAL_NORMS_CONFIG", DEFAULT_CONFIG_PATH))
    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    checkpoints = config.get("checkpoints", [])
    prompts = config.get("prompts", [])
    output_len = int(config.get("output_len", 1))
    output_path = Path(config.get("output_path") or DEFAULT_OUTPUT_PATH)
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path

    if not checkpoints or not prompts:
        print(f"Config {config_path} needs both `checkpoints` and `prompts`.")
        return

    # Merge into any existing table so profiling one model keeps the others.
    table: dict = {}
    if output_path.exists():
        table = json.loads(output_path.read_text())

    for checkpoint in checkpoints:
        print(f"Profiling {checkpoint} on {len(prompts)} prompts (output_len={output_len}) ...")
        model_id, entry = profile_checkpoint(checkpoint, prompts, output_len)
        table[model_id] = entry
        _print_report(model_id, entry)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(table, indent=2) + "\n")
    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
