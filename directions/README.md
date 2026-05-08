# `interpret/directions/`

Curated steering vectors extracted from Gemma3-4b-it residual stream — small enough to commit and pre-selected so you can load and steer without rerunning the full extraction pipeline.

Each direction is a pair `<name>.pt` (a 1-D `torch.Tensor` of shape `(d_model,)`, `d_model=2560`) and `<name>.json` (the `(intermediate, position, layer, coefficient)` it was selected at, plus refusal/KL scores from the sweep).

| Direction | Source | Selected at |
|---|---|---|
| `poetry_prose` | poetry-vs-prose mean-of-difference (1154 poetry / 1151 prose prompts) | `post_attn`, pos=−2, layer=11, coeff=+1.0 |
| `poems_paraphrase` | paired (poem, paraphrase) mean-of-difference (1000 pairs) | per `poems_paraphrase.json` |

Origin: copied from `resources/experiments/poetry_directions/<name>/` — those folders remain in place so the experiment runners' idempotency checks still find them. These copies are the stable "play with it" set; if you re-extract a direction with different hyperparams, copy the new artefact here and overwrite.

## Load and steer

```python
import torch
from interpret.inference.gemma_pytorch import GemmaPytorchInference
from interpret.experiments.refusal_directions.select_direction import _additive_op
from interpret.experiments.refusal_directions.tokens import format_chat
from interpret.sae import HookManager

direction = torch.load("interpret/directions/poetry_prose.pt").to(torch.float32)
LAYER = 11   # match the metadata
COEFF = +1.0

wrapper = GemmaPytorchInference("google/gemma-3-4b-it")
manager = HookManager()
manager.add_steering([_additive_op(direction, LAYER, coeff=COEFF)])

with manager.session(wrapper.model.model.layers):
    out = wrapper.generate_from_template(
        format_chat(wrapper, "Recommend a chocolate cake recipe for two."),
        output_len=160,
    )
print(out)
```

The interactive notebook at [`interpret/notebooks/poetry_steer_tester.ipynb`](../notebooks/poetry_steer_tester.ipynb) builds the same pipeline cell-by-cell and includes a sweep-score browser if you want to pick a different `(pos, layer, coeff)`.

## Refusal direction

Not yet stored here — the refusal-direction pipeline at `interpret/experiments/refusal_directions/` hasn't been run to completion. Run

```
uv run python -m interpret.experiments.refusal_directions.run
```

then copy `resources/experiments/refusal_directions/direction.pt` to `interpret/directions/refusal.pt` (and the metadata).
