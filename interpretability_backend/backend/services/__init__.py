"""Services module for embedding backend."""

import sys
from pathlib import Path

# The interpret/ toolkit uses `interpret.*` absolute imports internally.
# Ensure its parent directory is on sys.path before any service module
# (interpret_service, probing_service) does `from interpret... import`.
_INTERPRET_PARENT = str(Path(__file__).resolve().parents[2])
if _INTERPRET_PARENT not in sys.path:
    sys.path.insert(0, _INTERPRET_PARENT)

from .job_state import JobState, JobStateService, JobStatus  # noqa: E402

__all__ = ["JobStatus", "JobState", "JobStateService"]
