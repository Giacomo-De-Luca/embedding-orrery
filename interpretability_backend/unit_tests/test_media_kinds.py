"""Media-kind detection for dataset cells (backend/utils/media_kinds.py).

Pure Python: no PIL / audio / video libraries are imported to classify a value.
"""

import numpy as np
import pytest

from backend.utils.embedding_model_catalog import MEDIA_MODALITIES
from backend.utils.media_kinds import MediaKind, MediaKindDetector, MediaReference

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 16
GIF = b"GIF89a" + b"\x00" * 16
WEBP = b"RIFF\x10\x00\x00\x00WEBPVP8 " + b"\x00" * 8
WAV = b"RIFF\x10\x00\x00\x00WAVEfmt " + b"\x00" * 8
AVI = b"RIFF\x10\x00\x00\x00AVI LIST" + b"\x00" * 8
MP3_ID3 = b"ID3\x03\x00" + b"\x00" * 16
MP3_FRAME = b"\xff\xfb\x90\x64" + b"\x00" * 16
FLAC = b"fLaC" + b"\x00" * 16
OGG = b"OggS" + b"\x00" * 16
MP4 = b"\x00\x00\x00\x18ftypisom" + b"\x00" * 8
MOV = b"\x00\x00\x00\x14ftypqt  " + b"\x00" * 8
M4A = b"\x00\x00\x00\x18ftypM4A " + b"\x00" * 8
HEIC = b"\x00\x00\x00\x18ftypheic" + b"\x00" * 8
WEBM = b"\x1a\x45\xdf\xa3" + b"\x00" * 16


class FakePilImage:
    """Stands in for a PIL image: detection goes by the defining module."""

    size = (640, 480)


FakePilImage.__module__ = "PIL.PngImagePlugin"


@pytest.fixture
def detector() -> MediaKindDetector:
    return MediaKindDetector()


class TestMediaKind:
    def test_values_match_the_catalog_media_modalities(self):
        assert tuple(kind.value for kind in MediaKind) == MEDIA_MODALITIES


class TestValueDetection:
    @pytest.mark.parametrize(
        "data, kind",
        [
            (PNG, MediaKind.IMAGE),
            (JPEG, MediaKind.IMAGE),
            (GIF, MediaKind.IMAGE),
            (WEBP, MediaKind.IMAGE),
            (HEIC, MediaKind.IMAGE),
            (WAV, MediaKind.AUDIO),
            (MP3_ID3, MediaKind.AUDIO),
            (MP3_FRAME, MediaKind.AUDIO),
            (FLAC, MediaKind.AUDIO),
            (OGG, MediaKind.AUDIO),
            (M4A, MediaKind.AUDIO),
            (AVI, MediaKind.VIDEO),
            (MP4, MediaKind.VIDEO),
            (MOV, MediaKind.VIDEO),
            (WEBM, MediaKind.VIDEO),
        ],
    )
    def test_magic_bytes(self, detector, data, kind):
        assert detector.kind_of(data) is kind
        assert detector.kind_of(bytearray(data)) is kind
        assert detector.kind_of(memoryview(data)) is kind

    def test_unknown_bytes(self, detector):
        assert detector.kind_of(b"plain text bytes") is None
        assert detector.kind_of(b"") is None

    @pytest.mark.parametrize(
        "value, kind",
        [
            ("photos/cat.JPG", MediaKind.IMAGE),
            ("/abs/clip.webm", MediaKind.VIDEO),
            ("speech.wav", MediaKind.AUDIO),
            ("song.m4a", MediaKind.AUDIO),
            ("https://example.org/a/b.png?size=large#x", MediaKind.IMAGE),
            ("https://example.org/video.mp4", MediaKind.VIDEO),
        ],
    )
    def test_paths_and_urls_by_extension(self, detector, value, kind):
        assert detector.kind_of(value) is kind

    @pytest.mark.parametrize("value", ["a sentence about a cat.jpg lover", "notes.txt", "", "jpg"])
    def test_plain_strings_are_not_media(self, detector, value):
        assert detector.kind_of(value) is None

    def test_hf_undecoded_dicts(self, detector):
        assert detector.kind_of({"bytes": PNG, "path": None}) is MediaKind.IMAGE
        assert detector.kind_of({"bytes": None, "path": "clip.mp4"}) is MediaKind.VIDEO
        # the bytes win over a misleading or missing extension
        assert detector.kind_of({"bytes": WAV, "path": "track"}) is MediaKind.AUDIO

    def test_decoded_audio_dict(self, detector):
        value = {"array": np.zeros(160, dtype=np.float32), "sampling_rate": 16000}
        assert detector.kind_of(value) is MediaKind.AUDIO

    def test_pil_images_by_module(self, detector):
        assert detector.kind_of(FakePilImage()) is MediaKind.IMAGE

    @pytest.mark.parametrize("value", [None, 3, 2.5, True, [1, 2], {"a": 1}, np.zeros(3)])
    def test_other_values_are_not_media(self, detector, value):
        assert detector.kind_of(value) is None

    def test_extensions_come_from_the_constructor(self):
        detector = MediaKindDetector(extensions={MediaKind.IMAGE: (".raw",)})
        assert detector.kind_of("frame.raw") is MediaKind.IMAGE
        assert detector.kind_of("frame.png") is None


class TestColumnDetection:
    def test_majority_vote(self, detector):
        values = [PNG, None, JPEG, "cat.png", "not media"]
        assert detector.kind_of_column(values) is MediaKind.IMAGE

    def test_mostly_text_is_not_a_media_column(self, detector):
        values = ["a", "b", "c.png", "d"]
        assert detector.kind_of_column(values) is None

    def test_empty_or_all_null(self, detector):
        assert detector.kind_of_column([]) is None
        assert detector.kind_of_column([None, None]) is None

    def test_hf_dtype_hint_wins(self, detector):
        assert detector.kind_of_column(["x"], hint="audio") is MediaKind.AUDIO
        assert detector.kind_of_column([PNG], hint="string") is MediaKind.IMAGE
        assert detector.kind_of_column([PNG], hint=None) is MediaKind.IMAGE

    def test_only_the_sample_is_inspected(self, detector):
        values = iter([PNG] * 3 + ["text"] * 100)
        assert detector.kind_of_column(values, sample_size=3) is MediaKind.IMAGE


class TestMediaReference:
    def test_paths_and_urls_are_kept(self):
        assert MediaReference.of("photos/cat.jpg") == "photos/cat.jpg"
        assert MediaReference.of({"bytes": PNG, "path": "cat.png"}) == "cat.png"
        assert MediaReference.of(PNG) is None
        assert MediaReference.of({"bytes": PNG, "path": None}) is None

    def test_placeholders_never_contain_bytes(self):
        assert MediaReference.placeholder(PNG + b"\x00" * 2048, MediaKind.IMAGE) == "[image 2 KB]"
        assert (
            MediaReference.placeholder({"bytes": WAV, "path": None}, MediaKind.AUDIO)
            == "[audio 24 B]"
        )
        assert MediaReference.placeholder(FakePilImage(), MediaKind.IMAGE) == "[image 640x480]"
        audio = {"array": np.zeros(32000, dtype=np.float32), "sampling_rate": 16000}
        assert MediaReference.placeholder(audio, MediaKind.AUDIO) == "[audio 2.0 s]"
        assert MediaReference.placeholder("clips/a.mp4", MediaKind.VIDEO) == "[video clips/a.mp4]"

    def test_label_prefers_the_file_name(self):
        assert MediaReference.label("photos/cat.jpg", MediaKind.IMAGE) == "image: cat.jpg"
        assert (
            MediaReference.label("https://example.org/x/dog.png?s=1", MediaKind.IMAGE)
            == "image: dog.png"
        )
        assert MediaReference.label(PNG, MediaKind.IMAGE) == "image"


BMP = b"BM" + b"\x00" * 24
# ISO base media with a generic brand: an .m4a and an .mp4 can both carry it
ISOM = b"\x00\x00\x00\x18ftypisom" + b"\x00" * 8


class TestAmbiguousContainers:
    def test_bmp_bytes_are_images(self, detector):
        assert detector.kind_of(BMP) is MediaKind.IMAGE

    def test_a_generic_iso_brand_defers_to_the_file_extension(self, detector):
        assert detector.kind_of({"bytes": ISOM, "path": "clips/song.m4a"}) is MediaKind.AUDIO
        assert detector.kind_of({"bytes": ISOM, "path": "clips/walk.mp4"}) is MediaKind.VIDEO

    def test_a_generic_iso_brand_without_a_path_reads_as_video(self, detector):
        assert detector.kind_of(ISOM) is MediaKind.VIDEO

    def test_a_dataset_hint_wins_over_sniffing(self, detector):
        assert detector.kind_of_column([{"bytes": ISOM, "path": None}], hint="audio") is (
            MediaKind.AUDIO
        )


class TestDescribe:
    def test_path_else_placeholder(self):
        assert MediaReference.describe({"bytes": PNG, "path": "a/cat.png"}, MediaKind.IMAGE) == (
            "a/cat.png"
        )
        assert MediaReference.describe(PNG, MediaKind.IMAGE) == "[image 24 B]"

    def test_for_metadata_covers_every_media_cell(self):
        assert MediaReference.for_metadata({"bytes": None, "path": "a/cat.png"}) == "a/cat.png"
        assert MediaReference.for_metadata(PNG) == "[image 24 B]"
        assert MediaReference.for_metadata(b"\x00\x01") == "[binary 2 B]"
        assert MediaReference.for_metadata(FakePilImage()) == "[image 640x480]"

    def test_for_metadata_leaves_other_values(self):
        for value in ({"lang": "en"}, {"path": "notes.txt"}, 3, [1, 2], np.zeros(3)):
            assert MediaReference.for_metadata(value) is None

    def test_unrecognised_bytes_are_binary(self):
        assert MediaReference.describe(b"\x00" * 2048, None) == "[binary 2 KB]"
        assert MediaReference.describe({"bytes": b"\x01\x02", "path": None}, None) == "[binary 2 B]"
