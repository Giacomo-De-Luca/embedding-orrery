"""Decoding dataset media cells into model inputs (backend/utils/media_decoding.py).

Images become RGB PIL images; audio a 16 kHz mono float32 waveform (decoded
with PyAV, resampled, capped at ``max_audio_seconds``); video a stack of RGB
frames sampled at ``video_fps`` (at most ``video_max_frames``, spread over long
clips) with its metadata. A cell that cannot be decoded raises
``MediaDecodeError``; the embedder skips that row and counts it.
"""

import io
import wave

import numpy as np
import pytest
from PIL import Image

from backend.utils.media_decoding import MediaDecodeError, MediaDecoder
from backend.utils.media_kinds import MediaKind
from backend.utils.media_settings import MediaEmbeddingSettings, MediaOptions


def _png(size=(40, 30), mode="RGB") -> bytes:
    buffer = io.BytesIO()
    Image.new(mode, size, color=(200, 10, 10, 255)[: len(mode)]).save(buffer, format="PNG")
    return buffer.getvalue()


def _wav(seconds=1.0, rate=8000, channels=2) -> bytes:
    t = np.arange(int(seconds * rate)) / rate
    tone = (0.5 * np.sin(2 * np.pi * 440 * t) * 32767).astype("<i2")
    frames = np.repeat(tone[:, None], channels, axis=1).tobytes()
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(channels)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(frames)
    return buffer.getvalue()


def _decoder(tmp_path=None, settings=None, options=None, fetch=None) -> MediaDecoder:
    return MediaDecoder(
        settings or MediaEmbeddingSettings(),
        options or MediaOptions(),
        base_dir=tmp_path,
        fetch=fetch,
    )


class TestImages:
    def test_png_bytes_decode_to_rgb(self):
        image = _decoder().decode(_png(), MediaKind.IMAGE)
        assert image.mode == "RGB" and image.size == (40, 30)

    def test_hf_dicts_and_pil_images(self):
        decoder = _decoder()
        assert decoder.decode({"bytes": _png(), "path": None}, MediaKind.IMAGE).size == (40, 30)
        rgba = Image.new("RGBA", (8, 8))
        assert decoder.decode(rgba, MediaKind.IMAGE).mode == "RGB"

    def test_relative_paths_resolve_against_the_data_file(self, tmp_path):
        (tmp_path / "pics").mkdir()
        (tmp_path / "pics" / "cat.png").write_bytes(_png((12, 9)))
        assert _decoder(tmp_path).decode("pics/cat.png", MediaKind.IMAGE).size == (12, 9)
        hf_style = {"bytes": None, "path": "pics/cat.png"}
        assert _decoder(tmp_path).decode(hf_style, MediaKind.IMAGE).size == (12, 9)

    def test_missing_file(self, tmp_path):
        with pytest.raises(MediaDecodeError, match="not found"):
            _decoder(tmp_path).decode("nope.png", MediaKind.IMAGE)

    def test_corrupt_bytes(self):
        with pytest.raises(MediaDecodeError):
            _decoder().decode(b"\x89PNG\r\n\x1a\nbroken", MediaKind.IMAGE)

    def test_unsupported_value(self):
        with pytest.raises(MediaDecodeError):
            _decoder().decode(42, MediaKind.IMAGE)

    def test_remote_urls_are_refused_by_default(self):
        with pytest.raises(MediaDecodeError, match="allow_remote_urls"):
            _decoder().decode("https://example.org/cat.png", MediaKind.IMAGE)

    def test_remote_urls_are_fetched_when_allowed(self):
        fetched = []

        def fetch(url, timeout):
            fetched.append((url, timeout))
            return _png((5, 4))

        settings = MediaEmbeddingSettings(allow_remote_urls=True, remote_timeout_seconds=3)
        image = _decoder(settings=settings, fetch=fetch).decode(
            "https://example.org/cat.png", MediaKind.IMAGE
        )
        assert image.size == (5, 4) and fetched == [("https://example.org/cat.png", 3)]


class TestAudioArrays:
    """Decoded HF audio needs no codec: resampling and the cap only."""

    def test_resamples_to_the_model_rate(self):
        value = {"array": np.zeros(8000, dtype=np.float64), "sampling_rate": 8000}
        audio = _decoder().decode(value, MediaKind.AUDIO)
        assert audio["sampling_rate"] == 16000
        assert audio["array"].dtype == np.float32 and len(audio["array"]) == 16000

    def test_stereo_is_downmixed(self):
        stereo = np.stack([np.ones(1600), -np.ones(1600)])  # (channels, samples)
        audio = _decoder().decode({"array": stereo, "sampling_rate": 16000}, MediaKind.AUDIO)
        assert audio["array"].ndim == 1 and np.allclose(audio["array"], 0.0)

    def test_duration_cap(self):
        value = {"array": np.zeros(16000 * 5, dtype=np.float32), "sampling_rate": 16000}
        audio = _decoder(options=MediaOptions(max_audio_seconds=2.0)).decode(value, MediaKind.AUDIO)
        assert len(audio["array"]) == 32000

    def test_empty_audio_is_an_error(self):
        with pytest.raises(MediaDecodeError):
            _decoder().decode({"array": np.zeros(0), "sampling_rate": 16000}, MediaKind.AUDIO)


class TestAudioFiles:
    @pytest.fixture(autouse=True)
    def _pyav(self):
        pytest.importorskip("av")

    def test_wav_bytes_decode_to_16k_mono(self):
        audio = _decoder().decode(_wav(seconds=1.0, rate=8000, channels=2), MediaKind.AUDIO)
        assert audio["sampling_rate"] == 16000 and audio["array"].ndim == 1
        assert abs(len(audio["array"]) - 16000) < 400
        assert audio["array"].dtype == np.float32 and np.abs(audio["array"]).max() > 0.1

    def test_wav_files_and_the_cap(self, tmp_path):
        (tmp_path / "tone.wav").write_bytes(_wav(seconds=3.0, rate=16000, channels=1))
        decoder = _decoder(tmp_path, options=MediaOptions(max_audio_seconds=1.0))
        assert len(decoder.decode("tone.wav", MediaKind.AUDIO)["array"]) == 16000

    def test_not_audio(self):
        with pytest.raises(MediaDecodeError):
            _decoder().decode(b"RIFF\x00\x00\x00\x00WAVEjunk", MediaKind.AUDIO)


class TestVideo:
    @pytest.fixture
    def clip(self, tmp_path):
        """A 2-second 64x48 MPEG-4 clip at 10 fps."""
        av = pytest.importorskip("av")
        path = tmp_path / "clip.mp4"
        with av.open(str(path), mode="w") as container:
            stream = container.add_stream("mpeg4", rate=10)
            stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
            for i in range(20):
                pixels = np.full((48, 64, 3), i * 12, dtype=np.uint8)
                for packet in stream.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        return path

    def test_samples_frames_at_the_configured_fps(self, clip):
        video = _decoder(clip.parent).decode(clip.name, MediaKind.VIDEO)
        frames = video["array"]
        assert frames.dtype == np.uint8 and frames.shape[1:] == (48, 64, 3)
        assert frames.shape[0] == 2  # 2 s at 1 fps
        assert video["video_metadata"]["total_num_frames"] == 2
        assert video["video_metadata"]["fps"] == pytest.approx(1.0)

    def test_long_clips_spread_the_frame_cap(self, clip):
        options = MediaOptions(video_fps=4.0, video_max_frames=3)
        video = _decoder(clip.parent, options=options).decode(clip.name, MediaKind.VIDEO)
        assert video["array"].shape[0] == 3
        # spread over the whole clip, not its first 0.75 s
        assert video["video_metadata"]["fps"] < 4.0

    def test_frames_are_downscaled_to_the_max_side(self, clip):
        settings = MediaEmbeddingSettings(video_frame_max_side=32)
        video = _decoder(clip.parent, settings=settings).decode(clip.name, MediaKind.VIDEO)
        assert video["array"].shape[1:] == (24, 32, 3)

    def test_bytes_decode_too(self, clip):
        video = _decoder().decode(clip.read_bytes(), MediaKind.VIDEO)
        assert video["array"].shape[0] == 2

    def test_not_a_video(self):
        pytest.importorskip("av")
        with pytest.raises(MediaDecodeError):
            _decoder().decode(b"\x00\x00\x00\x18ftypisomjunk", MediaKind.VIDEO)


class TestDatasetFiles:
    """Streamed Hugging Face datasets point media cells at the repo's own files
    (``hf://…``, ``zip://…::https://…``, ``https://huggingface.co/…``)."""

    def _decoder(self, open_path, base_dir=None) -> MediaDecoder:
        return MediaDecoder(
            MediaEmbeddingSettings(), MediaOptions(), base_dir=base_dir, open_path=open_path
        )

    def test_non_local_paths_go_to_the_dataset_reader(self):
        opened = []
        decoder = self._decoder(lambda path: opened.append(path) or _png())
        cell = {"bytes": None, "path": "hf://datasets/org/pets@main/cat.png"}

        assert decoder.decode(cell, MediaKind.IMAGE).mode == "RGB"
        assert opened == ["hf://datasets/org/pets@main/cat.png"]

    def test_dataset_urls_need_no_remote_url_opt_in(self):
        url = "https://huggingface.co/datasets/org/pets/resolve/main/cat.png"
        assert self._decoder(lambda path: _png()).decode(url, MediaKind.IMAGE).mode == "RGB"

    def test_local_files_are_read_directly(self, tmp_path):
        (tmp_path / "cat.png").write_bytes(_png())

        def unexpected(path):
            raise AssertionError(f"dataset reader used for a local file: {path}")

        assert self._decoder(unexpected, tmp_path).decode("cat.png", MediaKind.IMAGE).mode == "RGB"

    def test_reader_failures_skip_the_row(self):
        def missing(path):
            raise FileNotFoundError(path)

        with pytest.raises(MediaDecodeError, match="cat.png"):
            self._decoder(missing).decode("hf://datasets/org/pets/cat.png", MediaKind.IMAGE)
