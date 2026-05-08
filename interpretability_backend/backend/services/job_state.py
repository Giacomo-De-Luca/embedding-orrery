"""
Job state service for tracking embedding job progress.

Provides persistent storage of job state in a JSON file, enabling:
- Progress tracking (items embedded, batches completed)
- Resume capability after interruption
- Frontend discovery of running/interrupted jobs
"""

import json
import threading
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class JobStatus(Enum):
    """Status of an embedding job."""

    RUNNING = "running"
    INTERRUPTED = "interrupted"
    COMPLETED = "completed"


@dataclass
class JobState:
    """State of an embedding job."""

    collection_name: str
    status: JobStatus
    job_type: str  # "huggingface" or "local_file"

    # Progress tracking
    items_embedded: int
    total_expected: int
    batches_completed: int
    total_batches: int

    # Full configuration (stored as dict for flexibility)
    config: dict[str, Any]

    started_at: str

    @property
    def percent_complete(self) -> float:
        """Calculate completion percentage."""
        if self.total_expected == 0:
            return 0.0
        return round(self.items_embedded / self.total_expected * 100, 1)

    @property
    def source(self) -> str:
        """Extract source (dataset_id or file_path) from config."""
        return self.config.get("dataset_id") or self.config.get("file_path", "")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "collection_name": self.collection_name,
            "status": self.status.value,
            "job_type": self.job_type,
            "items_embedded": self.items_embedded,
            "total_expected": self.total_expected,
            "batches_completed": self.batches_completed,
            "total_batches": self.total_batches,
            "config": self.config,
            "started_at": self.started_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JobState":
        """Create JobState from dictionary."""
        return cls(
            collection_name=data["collection_name"],
            status=JobStatus(data["status"]),
            job_type=data["job_type"],
            items_embedded=data["items_embedded"],
            total_expected=data["total_expected"],
            batches_completed=data["batches_completed"],
            total_batches=data["total_batches"],
            config=data["config"],
            started_at=data["started_at"],
        )


# Default path for job state file (alongside vector_db)
DEFAULT_JOB_STATE_PATH = Path(__file__).parent.parent.parent / "resources" / "job_state.json"


class JobStateService:
    """Service for managing embedding job state persistence.

    Thread-safe service that stores job state in a JSON file.
    On initialization, marks any 'running' jobs as 'interrupted'.
    """

    def __init__(self, state_file: Path | None = None):
        """Initialize the job state service.

        Args:
            state_file: Path to the job state JSON file. Defaults to resources/job_state.json.
        """
        self.state_file = state_file or DEFAULT_JOB_STATE_PATH
        self._lock = threading.Lock()
        self._ensure_file_exists()
        self._mark_running_as_interrupted()

    def _ensure_file_exists(self) -> None:
        """Ensure the state file and its directory exist."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.state_file.exists():
            self._save({"jobs": {}})

    def _load(self) -> dict[str, Any]:
        """Load state from file."""
        try:
            return json.loads(self.state_file.read_text())
        except (json.JSONDecodeError, FileNotFoundError):
            return {"jobs": {}}

    def _save(self, data: dict[str, Any]) -> None:
        """Save state to file."""
        self.state_file.write_text(json.dumps(data, indent=2, default=str))

    def _mark_running_as_interrupted(self) -> None:
        """On startup, mark any 'running' jobs as 'interrupted'.

        This handles the case where the server crashed or was stopped
        while jobs were in progress.
        """
        with self._lock:
            data = self._load()
            changed = False
            for job in data.get("jobs", {}).values():
                if job.get("status") == JobStatus.RUNNING.value:
                    job["status"] = JobStatus.INTERRUPTED.value
                    changed = True
            if changed:
                self._save(data)

    def start_job(
        self,
        collection_name: str,
        job_type: str,
        total_expected: int,
        total_batches: int,
        config: dict[str, Any],
    ) -> None:
        """Record a new job start with full configuration.

        Args:
            collection_name: Name of the ChromaDB collection
            job_type: Type of job ("huggingface" or "local_file")
            total_expected: Total number of items to embed
            total_batches: Total number of batches
            config: Full embedding configuration (for resume verification)
        """
        with self._lock:
            data = self._load()
            data["jobs"][collection_name] = {
                "collection_name": collection_name,
                "status": JobStatus.RUNNING.value,
                "job_type": job_type,
                "items_embedded": 0,
                "total_expected": total_expected,
                "batches_completed": 0,
                "total_batches": total_batches,
                "config": config,
                "started_at": datetime.now().isoformat(),
            }
            self._save(data)

    def update_total_expected(
        self,
        collection_name: str,
        total_expected: int,
        total_batches: int | None = None,
    ) -> None:
        """Update total_expected (and optionally total_batches) for a job.

        Useful when the total work is not known at job start time.

        Args:
            collection_name: Name of the collection / job ID
            total_expected: Updated total expected items
            total_batches: Updated total batches (if provided)
        """
        with self._lock:
            data = self._load()
            if collection_name in data["jobs"]:
                data["jobs"][collection_name]["total_expected"] = total_expected
                if total_batches is not None:
                    data["jobs"][collection_name]["total_batches"] = total_batches
                self._save(data)

    def update_progress(
        self, collection_name: str, items_embedded: int, batches_completed: int
    ) -> None:
        """Update progress counters for a job.

        Args:
            collection_name: Name of the collection
            items_embedded: Total items embedded so far
            batches_completed: Total batches completed so far
        """
        with self._lock:
            data = self._load()
            if collection_name in data["jobs"]:
                data["jobs"][collection_name]["items_embedded"] = items_embedded
                data["jobs"][collection_name]["batches_completed"] = batches_completed
                self._save(data)

    def complete_job(self, collection_name: str, remove: bool = True) -> None:
        """Mark a job as completed.

        Args:
            collection_name: Name of the collection
            remove: If True, remove the job from state. If False, mark as completed.
        """
        with self._lock:
            data = self._load()
            if collection_name in data["jobs"]:
                if remove:
                    del data["jobs"][collection_name]
                else:
                    data["jobs"][collection_name]["status"] = JobStatus.COMPLETED.value
                self._save(data)

    def fail_job(self, collection_name: str, error: str | None = None) -> None:
        """Mark a job as interrupted/failed.

        Args:
            collection_name: Name of the collection
            error: Optional error message
        """
        with self._lock:
            data = self._load()
            if collection_name in data["jobs"]:
                data["jobs"][collection_name]["status"] = JobStatus.INTERRUPTED.value
                if error:
                    data["jobs"][collection_name]["error"] = error
                self._save(data)

    def get_job(self, collection_name: str) -> JobState | None:
        """Get job state by collection name.

        Args:
            collection_name: Name of the collection

        Returns:
            JobState if found, None otherwise
        """
        with self._lock:
            data = self._load()
            job_data = data["jobs"].get(collection_name)
            if job_data:
                return JobState.from_dict(job_data)
            return None

    def list_jobs(self, status: JobStatus | None = None) -> list[JobState]:
        """List all jobs, optionally filtered by status.

        Args:
            status: Optional status filter

        Returns:
            List of JobState objects
        """
        with self._lock:
            data = self._load()
            jobs = []
            for job_data in data["jobs"].values():
                job = JobState.from_dict(job_data)
                if status is None or job.status == status:
                    jobs.append(job)
            return jobs

    def remove_job(self, collection_name: str) -> None:
        """Remove a job from the state file.

        Args:
            collection_name: Name of the collection
        """
        with self._lock:
            data = self._load()
            if collection_name in data["jobs"]:
                del data["jobs"][collection_name]
                self._save(data)


# Global singleton instance
_job_state_service: JobStateService | None = None


def get_job_state_service() -> JobStateService:
    """Get the global JobStateService singleton.

    This ensures all parts of the application share the same instance,
    which is important for thread safety and consistent state.
    """
    global _job_state_service
    if _job_state_service is None:
        _job_state_service = JobStateService()
    return _job_state_service
