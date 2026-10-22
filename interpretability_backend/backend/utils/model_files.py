"""Read a model's small JSON config files without loading the model.

``config.json``, ``sentence_bert_config.json`` and ``tokenizer_config.json``
tell a model's checkpoint dtype and input-length limit. ``ModelConfigFiles``
reads them from a local model folder or the Hugging Face Hub: the local cache
first, a download otherwise (a few KB). Any failure (offline, gated, missing
file) reads as "no file". Results are cached per (model, file) for the life of
the process.

Torch-free: safe to import from the GraphQL schema.
"""

import json
import threading
from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import Self

from huggingface_hub import hf_hub_download, try_to_load_from_cache
from huggingface_hub.errors import (
    EntryNotFoundError,
    GatedRepoError,
    HFValidationError,
    RepositoryNotFoundError,
)

# sentence-transformers resolves bare model names ("all-MiniLM-L6-v2") to this org.
_BARE_NAME_ORG = "sentence-transformers"


def _no_download(repo_id: str, filename: str) -> str:
    raise LookupError("downloads are disabled for this reader")


class ModelConfigFiles:
    """JSON config files of models, each read at most once.

    Only definitive answers are cached: a parsed file, or a file the model
    does not have. Failures that may pass (offline, timeouts, gated repos,
    disabled downloads) return None and are retried on the next read.
    """

    def __init__(
        self,
        cache_lookup: Callable[[str, str], object] = try_to_load_from_cache,
        downloader: Callable[[str, str], str] = hf_hub_download,
    ):
        self._cache_lookup = cache_lookup
        self._downloader = downloader
        self._read: dict[tuple[str, str], dict | None] = {}
        self._lock = threading.Lock()

    @classmethod
    @cache
    def shared(cls) -> Self:
        """The process-wide reader (Hub cache, then download)."""
        return cls()

    @classmethod
    @cache
    def cache_only(cls) -> Self:
        """A reader that never downloads (local folders and the Hub cache only).

        For the public read-only demo, where any visitor could otherwise make
        the server fetch files for arbitrary repo ids.
        """
        return cls(downloader=_no_download)

    def read(self, model_name: str, filename: str) -> dict | None:
        """The parsed file, or None when the model has no such file or it cannot be read."""
        key = (model_name, filename)
        with self._lock:
            if key in self._read:
                return self._read[key]
        parsed, definitive = self._load(model_name, filename)
        if definitive:
            with self._lock:
                self._read[key] = parsed
        return parsed

    def _load(self, model_name: str, filename: str) -> tuple[dict | None, bool]:
        """``(parsed file or None, whether that answer is definitive)``."""
        path, definitive = self._locate(model_name, filename)
        if path is None:
            return None, definitive
        try:
            parsed = json.loads(Path(path).read_text())
        except (OSError, ValueError):
            return None, True
        return (parsed if isinstance(parsed, dict) else None), True

    def _locate(self, model_name: str, filename: str) -> tuple[str | None, bool]:
        local = Path(model_name)
        if local.is_dir():
            candidate = local / filename
            return (str(candidate) if candidate.is_file() else None), True
        repo_id = model_name if "/" in model_name else f"{_BARE_NAME_ORG}/{model_name}"
        try:
            cached = self._cache_lookup(repo_id, filename)
            if isinstance(cached, str):
                return cached, True
            if cached is not None:  # the Hub cache records the file as absent
                return None, True
            return self._downloader(repo_id, filename), True
        except GatedRepoError:  # readable once a token is configured
            return None, False
        except (EntryNotFoundError, RepositoryNotFoundError, HFValidationError):
            return None, True  # no such file, repo or repo id
        except Exception:  # offline, timeout, HTTP error, downloads disabled: retry later
            return None, False
