"""Decode dataset media cells into the inputs a multimodal embedding model takes.

``MediaDecoder.decode(value, kind)`` accepts every cell shape ``media_kinds``
recognises (raw bytes, file paths relative to the data file, http(s) URLs when
``allow_remote_urls`` is on, HuggingFace ``{"bytes", "path"}`` dicts, decoded
``{"array", "sampling_rate"}`` audio and PIL images) and returns:

- image: an RGB ``PIL.Image``;
- audio: ``{"array": float32 mono, "sampling_rate": audio_sample_rate}``, capped
  at ``max_audio_seconds``. Files are decoded with PyAV (its wheels bundle
  FFmpeg, so mp3/m4a/aac work without a system install) and every waveform is
  resampled with scipy: sentence-transformers does not resample arrays and the
  EmbeddingGemma 2 feature extractor does not check the rate;
- video: ``{"array": uint8 (frames, height, width, 3), "video_metadata": ...}``,
  frames sampled at ``video_fps`` (at most ``video_max_frames``, spread evenly
  over longer clips) and downscaled to ``video_frame_max_side``. Only the
  pictures are embedded, not the soundtrack. The frames are already sampled,
  so the processor is told not to sample them again
  (``MediaOptions.processing_kwargs``).

Failures raise ``MediaDecodeError``; the embedder skips the row and counts it.
PIL, PyAV and scipy are imported inside the methods: this module is reachable
from the GraphQL schema, which must stay torch- and scipy-free (see
``unit_tests/test_torch_free_import.py``).
"""

import io
import math
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .media_kinds import URL_SCHEMES, MediaKind
from .media_settings import MediaEmbeddingSettings, MediaOptions

MAX_REMOTE_BYTES = 200 * 1024 * 1024


class MediaDecodeError(ValueError):
    """One media cell could not be decoded."""


def fetch_url(url: str, timeout: float) -> bytes:
    """Download ``url`` (at most ``MAX_REMOTE_BYTES``)."""
    with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310 (http(s) only)
        data = response.read(MAX_REMOTE_BYTES + 1)
    if len(data) > MAX_REMOTE_BYTES:
        raise MediaDecodeError(f"{url} is larger than {MAX_REMOTE_BYTES // (1024 * 1024)} MB")
    return data


class MediaDecoder:
    """Turn media cells of one job into model inputs (see module docstring)."""

    def __init__(
        self,
        settings: MediaEmbeddingSettings,
        options: MediaOptions,
        base_dir: Path | None = None,
        fetch: Callable[[str, float], bytes] | None = None,
        open_path: Callable[[str], bytes] | None = None,
    ):
        """``base_dir``: where relative paths resolve (the data file's folder).
        ``open_path``: reads paths that are not local files, the repo files a
        streamed Hugging Face dataset points at (``hf://…``, ``https://…``); the
        dataset is trusted like ``datasets`` trusts it, so ``allow_remote_urls``
        does not apply to them."""
        self.settings = settings
        self.options = options
        self.base_dir = base_dir
        self.fetch = fetch or fetch_url
        self.open_path = open_path

    def decode(self, value: Any, kind: MediaKind) -> Any:
        """The model input for one cell; raises ``MediaDecodeError``."""
        decoders = {
            MediaKind.IMAGE: self.image,
            MediaKind.AUDIO: self.audio,
            MediaKind.VIDEO: self.video,
        }
        try:
            return decoders[MediaKind(kind)](value)
        except MediaDecodeError:
            raise
        except Exception as error:  # codec errors: one bad cell must not stop the job
            raise MediaDecodeError(f"could not decode {kind}: {error}") from error

    # ------------------------------------------------------------------ image

    def image(self, value: Any) -> Any:
        from PIL import Image

        if type(value).__module__.startswith("PIL."):
            return value.convert("RGB")
        source = self._source(value)
        with Image.open(source if isinstance(source, Path) else io.BytesIO(source)) as image:
            return image.convert("RGB")

    # ------------------------------------------------------------------ audio

    def audio(self, value: Any) -> dict[str, Any]:
        if isinstance(value, Mapping) and "array" in value and "sampling_rate" in value:
            waveform, rate = self._mono(np.asarray(value["array"])), int(value["sampling_rate"])
        else:
            waveform, rate = self._decode_audio_stream(self._source(value))
        waveform = self._resample(waveform, rate)
        waveform = waveform[: int(self.options.max_audio_seconds * self.settings.audio_sample_rate)]
        if waveform.size == 0:
            raise MediaDecodeError("the audio has no samples")
        return {
            "array": waveform.astype(np.float32),
            "sampling_rate": self.settings.audio_sample_rate,
        }

    def _decode_audio_stream(self, source: bytes | Path) -> tuple[np.ndarray, int]:
        """Mono float waveform at the file's own rate, read up to the duration cap."""
        import av

        with av.open(self._av_source(source)) as container:
            if not container.streams.audio:
                raise MediaDecodeError("the file has no audio stream")
            stream = container.streams.audio[0]
            rate = stream.rate or stream.sample_rate
            resampler = av.AudioResampler(format="flt", layout="mono", rate=rate)
            limit = int(self.options.max_audio_seconds * rate) + 1
            chunks: list[np.ndarray] = []
            total = 0
            for frame in container.decode(stream):
                for converted in resampler.resample(frame):
                    chunk = converted.to_ndarray().reshape(-1)
                    chunks.append(chunk)
                    total += chunk.size
                if total >= limit:
                    break
        if not chunks:
            raise MediaDecodeError("the audio has no samples")
        return np.concatenate(chunks), rate

    @staticmethod
    def _mono(waveform: np.ndarray) -> np.ndarray:
        """Average channels: ``(channels, samples)`` or ``(samples, channels)``."""
        if waveform.ndim == 1:
            return waveform.astype(np.float32)
        if waveform.ndim != 2:
            raise MediaDecodeError(f"audio arrays must be 1-D or 2-D, got {waveform.ndim}-D")
        channel_axis = 0 if waveform.shape[0] <= waveform.shape[1] else 1
        return waveform.mean(axis=channel_axis).astype(np.float32)

    def _resample(self, waveform: np.ndarray, rate: int) -> np.ndarray:
        target = self.settings.audio_sample_rate
        if rate == target or waveform.size == 0:
            return waveform
        if rate <= 0:
            raise MediaDecodeError(f"invalid sampling rate {rate}")
        from scipy.signal import resample_poly

        divisor = math.gcd(rate, target)
        return resample_poly(waveform, target // divisor, rate // divisor).astype(np.float32)

    # ------------------------------------------------------------------ video

    def video(self, value: Any) -> dict[str, Any]:
        import av

        with av.open(self._av_source(self._source(value))) as container:
            if not container.streams.video:
                raise MediaDecodeError("the file has no video stream")
            stream = container.streams.video[0]
            duration = self._duration(container, stream)
            step = self._frame_step(duration)
            count = self._frame_count(duration, step)
            frames = []
            for frame in container.decode(stream):
                time = frame.time if frame.time is not None else 0.0
                if time + 1e-6 >= len(frames) * step:
                    frames.append(self._rgb(frame))
                    if len(frames) == count:
                        break
        if not frames:
            raise MediaDecodeError("the video has no frames")
        return {
            "array": np.stack(frames),
            "video_metadata": {
                "fps": 1.0 / step,
                "total_num_frames": len(frames),
                "duration": duration if duration is not None else len(frames) * step,
            },
        }

    @staticmethod
    def _duration(container: Any, stream: Any) -> float | None:
        if stream.duration is not None and stream.time_base is not None:
            return float(stream.duration * stream.time_base)
        if container.duration is not None:
            return container.duration / 1_000_000  # av.time_base: microseconds
        return None

    def _frame_step(self, duration: float | None) -> float:
        """Seconds between sampled frames: 1 / fps, wider when the cap would be exceeded."""
        step = 1.0 / self.options.video_fps
        if duration:
            step = max(step, duration / self.options.video_max_frames)
        return step

    def _frame_count(self, duration: float | None, step: float) -> int:
        if not duration:
            return self.options.video_max_frames
        return max(1, min(self.options.video_max_frames, math.ceil(duration / step - 1e-9)))

    def _rgb(self, frame: Any) -> np.ndarray:
        scale = min(1.0, self.settings.video_frame_max_side / max(frame.width, frame.height))
        width = max(1, round(frame.width * scale))
        height = max(1, round(frame.height * scale))
        return frame.reformat(width=width, height=height, format="rgb24").to_ndarray()

    # ---------------------------------------------------------------- sources

    def _source(self, value: Any) -> bytes | Path:
        """Raw bytes or an existing local path for one cell."""
        if isinstance(value, (bytes, bytearray, memoryview)):
            return bytes(value)
        if isinstance(value, Mapping):
            data = value.get("bytes")
            if isinstance(data, (bytes, bytearray, memoryview)) and len(data):
                return bytes(data)
            value = value.get("path")
        if isinstance(value, str) and value:
            path = None
            if not value.startswith(URL_SCHEMES):
                path = Path(value).expanduser()
                if not path.is_absolute() and self.base_dir is not None:
                    path = self.base_dir / path
                if path.is_file():
                    return path
            if self.open_path is not None:
                return self._dataset_file(value)
            if path is not None:
                raise MediaDecodeError(f"file not found: {path}")
            if not self.settings.allow_remote_urls:
                raise MediaDecodeError(
                    f"{value} is a remote URL; set allow_remote_urls in "
                    "config/media_embedding.json to fetch it"
                )
            return self.fetch(value, self.settings.remote_timeout_seconds)
        raise MediaDecodeError(f"unsupported media value of type {type(value).__name__}")

    def _dataset_file(self, path: str) -> bytes:
        """A dataset repo file through ``open_path`` (at most ``MAX_REMOTE_BYTES``)."""
        try:
            data = self.open_path(path)
        except Exception as error:
            raise MediaDecodeError(f"cannot read {path}: {error}") from error
        if len(data) > MAX_REMOTE_BYTES:
            raise MediaDecodeError(f"{path} is larger than {MAX_REMOTE_BYTES // (1024 * 1024)} MB")
        return data

    @staticmethod
    def _av_source(source: bytes | Path) -> Any:
        return str(source) if isinstance(source, Path) else io.BytesIO(source)
