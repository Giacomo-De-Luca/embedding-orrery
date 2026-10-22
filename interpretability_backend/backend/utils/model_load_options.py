"""How a SentenceTransformer model is loaded: dtype, and which encoder towers to skip.

``EmbeddingDtypePolicy`` follows the EmbeddingGemma model cards: bfloat16 on a
bf16-capable CUDA device, float32 everywhere else, never float16 (it overflows
EmbeddingGemma's activations into NaN or silently degraded vectors). On Apple
GPUs (MPS, macOS 14+) a checkpoint saved in bfloat16 (EmbeddingGemma 2) keeps
bfloat16 instead of being upcast, which doubled its weights and every attention
tensor; float32 checkpoints (EmbeddingGemma 1) stay float32, so existing
collections see no drift. CPU is always float32.

``SentenceTransformerLoadOptions`` disables the towers a job does not need. The
catalog's ``encoders`` map names, per media modality, the model-config
attribute holding its tower (EmbeddingGemma 2: image and video share
``vision_config``, audio uses ``audio_config``). Setting it to ``None`` through
``config_kwargs`` skips loading it, so text datasets and text queries load the
text tower only.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Self

from .embedding_model_catalog import EmbeddingModelCatalog
from .model_files import ModelConfigFiles

SENTENCE_TRANSFORMERS = "sentence_transformers"
BFLOAT16 = "bfloat16"


def _cuda_bf16_supported() -> bool:
    # Lazy: torch must stay out of the schema import graph (torch-free boundary).
    import torch

    return bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())


def _mps_bf16_supported() -> bool:
    # Lazy for the same reason; MPS supports bfloat16 from macOS 14.
    import torch

    return bool(torch.backends.mps.is_macos_or_newer(14, 0))


def _checkpoint_dtype(model_name: str) -> str | None:
    """The dtype a checkpoint was saved in, from its config.json (None if unknown)."""
    config = ModelConfigFiles.shared().read(model_name, "config.json") or {}
    text_config = config.get("text_config")
    nested = text_config if isinstance(text_config, dict) else {}
    dtype = config.get("dtype") or config.get("torch_dtype") or nested.get("dtype")
    return dtype if isinstance(dtype, str) else None


class EmbeddingDtypePolicy:
    """The dtype to load embedding models in, per device (and checkpoint, on MPS)."""

    def __init__(
        self,
        cuda_bf16_supported: Callable[[], bool] = _cuda_bf16_supported,
        mps_bf16_supported: Callable[[], bool] = _mps_bf16_supported,
        checkpoint_dtype: Callable[[str], str | None] = _checkpoint_dtype,
    ):
        self._cuda_bf16_supported = cuda_bf16_supported
        self._mps_bf16_supported = mps_bf16_supported
        self._checkpoint_dtype = checkpoint_dtype

    def for_device(self, device: str, model_name: str | None = None) -> str:
        if device.startswith("cuda") and self._cuda_bf16_supported():
            return BFLOAT16
        if (
            device == "mps"
            and model_name is not None
            and self._mps_bf16_supported()
            and self._checkpoint_dtype(model_name) == BFLOAT16
        ):
            return BFLOAT16
        return "float32"


@dataclass(frozen=True)
class SentenceTransformerLoadOptions:
    """Everything that makes one loaded model different from another."""

    model_name: str
    device: str
    dtype: str
    disabled_encoders: tuple[str, ...] = ()

    @classmethod
    def for_modalities(
        cls,
        model_name: str,
        device: str,
        modalities: Iterable[str],
        catalog: EmbeddingModelCatalog | None = None,
        dtype_policy: EmbeddingDtypePolicy | None = None,
    ) -> Self:
        """Options loading only the towers ``modalities`` need (all of them for unknown models)."""
        catalog = catalog or EmbeddingModelCatalog.shipped()
        dtype_policy = dtype_policy or EmbeddingDtypePolicy()
        entry = catalog.find(SENTENCE_TRANSFORMERS, model_name)
        encoders = entry.encoders if entry is not None else {}
        needed = {encoders[m] for m in set(modalities) if m in encoders}
        disabled = tuple(sorted(set(encoders.values()) - needed))
        return cls(model_name, device, dtype_policy.for_device(device, model_name), disabled)

    @property
    def cache_key(self) -> tuple[str, str, str, tuple[str, ...]]:
        return (self.model_name, self.device, self.dtype, self.disabled_encoders)

    def covers(self, other: "SentenceTransformerLoadOptions") -> bool:
        """True when a model loaded with these options can serve ``other`` too."""
        same_model = (self.model_name, self.device, self.dtype) == (
            other.model_name,
            other.device,
            other.dtype,
        )
        return same_model and set(self.disabled_encoders) <= set(other.disabled_encoders)

    def config_kwargs(self) -> dict[str, None]:
        return dict.fromkeys(self.disabled_encoders)

    def model_kwargs(self) -> dict[str, str]:
        return {"dtype": self.dtype}

    def constructor_kwargs(self) -> dict[str, dict]:
        """Keyword arguments for ``SentenceTransformer(...)`` (besides name, device, token)."""
        kwargs: dict[str, dict] = {"model_kwargs": self.model_kwargs()}
        if self.disabled_encoders:
            kwargs["config_kwargs"] = self.config_kwargs()
        return kwargs
