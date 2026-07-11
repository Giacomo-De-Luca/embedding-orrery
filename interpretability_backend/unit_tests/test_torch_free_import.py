"""Guard the heavy-import boundaries of the GraphQL schema.

Nothing importable from ``backend.main`` (the FastAPI app plus the full
GraphQL schema) may transitively import torch or the interpret/ toolkit at
module level: torch ships in the image (CPU-only on linux) but must stay out
of the import graph so startup RSS stays small and inference-only
dependencies load lazily on first use. Heavy imports must stay behind the documented lazy
boundaries: ``API/interpret_instance.get_interpret_service()`` and the
deferred ``probing_service`` import inside the ``train_probe`` mutation.

The clustering stack (hdbscan → sklearn → scipy, plus umap) is equally
deferred for RAM (~94 MB RSS; startup ~180 MB vs ~250 MB eager) behind the
local import in ``topic_extraction_service.extract_topics()`` and the lazy
PEP 562 re-export in ``topic_extraction/__init__.py``.

Runs in a subprocess so an already-imported module in the test session can't
mask a regression.
"""

import subprocess
import sys

_CHECK_SNIPPET = """
import sys

import interpretability_backend.backend.main  # noqa: F401  (full app + schema)

BANNED = ("torch", "interpret", "hdbscan", "sklearn", "scipy", "umap")
heavy = sorted(
    m for m in sys.modules
    if any(m == b or m.startswith(b + ".") for b in BANNED)
)
if heavy:
    print("heavy modules loaded at import time:", heavy[:10])
    sys.exit(1)
"""


def test_schema_import_loads_no_heavy_modules():
    result = subprocess.run(
        [sys.executable, "-c", _CHECK_SNIPPET],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"Importing backend.main pulled in a heavy module (torch/interpret.*/"
        f"hdbscan/sklearn/scipy/umap) at module level.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
