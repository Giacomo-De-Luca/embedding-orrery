"""HuggingFace media features read undecoded (clients/huggingface_client.py).

datasets 4.x decodes Audio and Video with torchcodec, which is not installed
(it needs a system FFmpeg), so any dataset with such a column failed to load.
``HFMediaFeatures`` casts every Image / Audio / Video feature to
``decode=False``: cells arrive as ``{"bytes", "path"}`` and the media decoder
reads them. The datasets here are built from plain structs and then cast,
because encoding into an ``Audio()`` feature would import torchcodec.
"""

import datasets
import pytest

from backend.clients import huggingface_client
from backend.clients.huggingface_client import HFMediaFeatures

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
WAV = b"RIFF\x10\x00\x00\x00WAVEfmt " + b"\x00" * 8


def _media_dataset() -> datasets.Dataset:
    plain = datasets.Dataset.from_dict(
        {
            "caption": ["a cat", "a dog"],
            "image": [{"bytes": PNG, "path": "cat.png"}, {"bytes": PNG, "path": "dog.png"}],
            "speech": [{"bytes": WAV, "path": "a.wav"}, {"bytes": WAV, "path": "b.wav"}],
        }
    )
    return plain.cast_column("image", datasets.Image()).cast_column("speech", datasets.Audio())


class TestHFMediaFeatures:
    def test_kinds_of_the_media_features(self):
        assert HFMediaFeatures.kinds(_media_dataset().features) == {
            "image": "image",
            "speech": "audio",
        }

    def test_no_features(self):
        assert HFMediaFeatures.kinds(None) == {}

    def test_media_cells_arrive_undecoded(self):
        rows = list(HFMediaFeatures.undecoded(_media_dataset()))
        assert rows[0]["image"] == {"bytes": PNG, "path": "cat.png"}
        assert rows[0]["speech"] == {"bytes": WAV, "path": "a.wav"}
        assert rows[0]["caption"] == "a cat"

    def test_streaming_datasets_too(self):
        streamed = _media_dataset().to_iterable_dataset()
        row = next(iter(HFMediaFeatures.undecoded(streamed)))
        assert row["speech"]["bytes"] == WAV


class TestLoading:
    @pytest.fixture(autouse=True)
    def fake_hub(self, monkeypatch):
        def load(dataset_id, name=None, split=None, streaming=False):
            dataset = _media_dataset()
            return dataset.to_iterable_dataset() if streaming else dataset

        monkeypatch.setattr(datasets, "load_dataset", load)
        monkeypatch.setattr(huggingface_client, "get_dataset_info", lambda _id: None)

    @pytest.mark.parametrize(
        "portion",
        [
            None,
            huggingface_client.PortionConfig(huggingface_client.PortionStrategy.FIRST_N, n=1),
            huggingface_client.PortionConfig(
                huggingface_client.PortionStrategy.RANDOM_SAMPLE, n=1, seed=1
            ),
            huggingface_client.PortionConfig(
                huggingface_client.PortionStrategy.ROW_RANGE, start=0, end=1
            ),
        ],
    )
    def test_every_portion_loads_audio_without_torchcodec(self, portion):
        rows, _ = huggingface_client.load_dataset_portion("org/pets", portion=portion)
        assert rows and rows[0]["speech"]["bytes"] == WAV

    def test_preview_shows_placeholders(self):
        preview = huggingface_client.get_dataset_preview("org/pets", n_rows=1)
        assert preview.error is None
        assert preview.rows[0]["image"] == "cat.png"
        assert preview.rows[0]["speech"] == "a.wav"


def test_video_dtype_label():
    assert huggingface_client._dtype_to_string(datasets.Video(decode=False)) == "video"
