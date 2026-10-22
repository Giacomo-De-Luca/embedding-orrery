"""Building one multimodal model input per row (backend/utils/media_rows.py)."""

import pytest

from backend.utils.embedding_model_catalog import TASK_PREFIX_FORMAT
from backend.utils.embedding_tasks import EmbeddingTexts
from backend.utils.media_decoding import MediaDecodeError
from backend.utils.media_kinds import MediaKind
from backend.utils.media_rows import MediaColumn, MediaRowBuilder

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
WAV = b"RIFF\x10\x00\x00\x00WAVEfmt " + b"\x00" * 8


class FakeDecoder:
    """Returns a tagged value instead of decoding, and fails on b"bad"."""

    def __init__(self):
        self.calls = []

    def decode(self, value, kind):
        self.calls.append((value, kind))
        if value == b"bad":
            raise MediaDecodeError("broken")
        return f"decoded {kind}"


IMAGE = MediaColumn("img", MediaKind.IMAGE)
CLIP = MediaColumn("clip", MediaKind.VIDEO)
SOUND = MediaColumn("sound", MediaKind.AUDIO)


def _builder(columns=(IMAGE,), texts=None):
    return MediaRowBuilder(list(columns), FakeDecoder(), texts)


class TestInputs:
    def test_media_only_row(self):
        row = _builder().build({"img": PNG})
        assert row.embed_input == {"image": "decoded image"}
        assert row.has_text is False

    def test_text_comes_first_then_image_video_audio(self):
        texts = EmbeddingTexts(["caption"], None, None, None)
        row = _builder((SOUND, CLIP, IMAGE), texts).build(
            {"caption": "a cat", "img": PNG, "clip": "a.mp4", "sound": WAV}
        )
        assert list(row.embed_input) == ["text", "image", "video", "audio"]
        assert row.embed_input["text"] == "a cat" and row.has_text

    def test_several_columns_of_one_kind_become_a_list(self):
        builder = _builder((IMAGE, MediaColumn("img2", MediaKind.IMAGE)))
        row = builder.build({"img": PNG, "img2": PNG})
        assert row.embed_input == {"image": ["decoded image", "decoded image"]}

    def test_missing_cells_are_left_out(self):
        texts = EmbeddingTexts(["caption"], None, None, None)
        row = _builder((IMAGE,), texts).build({"caption": "only text", "img": None})
        assert row.embed_input == {"text": "only text"} and row.has_text

    def test_empty_rows_build_nothing(self):
        texts = EmbeddingTexts(["caption"], None, None, None)
        assert _builder((IMAGE,), texts).build({"caption": "  ", "img": None}) is None

    def test_a_bad_cell_fails_the_row(self):
        with pytest.raises(MediaDecodeError):
            _builder().build({"img": b"bad"})


class TestTaskPrefixes:
    def test_the_prefix_goes_on_the_text_only(self):
        texts = EmbeddingTexts(["caption"], None, TASK_PREFIX_FORMAT, "title: {title} | text: ")
        row = _builder((IMAGE,), texts).build({"caption": "a cat", "title": "Pets", "img": PNG})
        assert row.embed_input["text"] == "title: Pets | text: a cat"
        assert row.document == "a cat"  # the stored document never holds the prefix

    def test_media_only_rows_get_no_prefix(self):
        texts = EmbeddingTexts(["caption"], None, TASK_PREFIX_FORMAT, "title: none | text: ")
        row = _builder((IMAGE,), texts).build({"caption": "", "img": PNG})
        assert row.embed_input == {"image": "decoded image"} and not row.has_text


class TestDocumentsAndReferences:
    def test_document_is_the_text_when_there_is_one(self):
        texts = EmbeddingTexts(["caption"], "{caption}!", None, None)
        assert _builder((IMAGE,), texts).build({"caption": "hi", "img": PNG}).document == "hi!"

    def test_media_only_documents_are_labels(self):
        row = _builder((IMAGE, CLIP)).build({"img": "pics/cat.jpg", "clip": PNG})
        assert row.document == "image: cat.jpg; video"

    def test_references_keep_paths_and_summarise_bytes(self):
        row = _builder((IMAGE, SOUND)).build({"img": "pics/cat.jpg", "sound": WAV})
        assert row.references == {"img": "pics/cat.jpg", "sound": "[audio 24 B]"}

    def test_kinds(self):
        assert _builder((IMAGE, CLIP, MediaColumn("i2", MediaKind.IMAGE))).kinds == (
            MediaKind.IMAGE,
            MediaKind.VIDEO,
        )


class TestColumnDetection:
    def test_detects_kinds_from_values_and_hints(self):
        rows = [{"img": PNG, "clip": "a.mp4", "s": "x"}]
        columns = MediaColumn.detect(rows, ["img", "clip", "s"], hints={"s": "audio"})
        assert columns == [IMAGE, CLIP, MediaColumn("s", MediaKind.AUDIO)]

    def test_undetectable_column_raises(self):
        with pytest.raises(ValueError, match="'caption'"):
            MediaColumn.detect([{"caption": "a cat"}], ["caption"])

    def test_unknown_column_raises(self):
        with pytest.raises(ValueError, match="'missing'"):
            MediaColumn.detect([{"img": PNG}], ["missing"])

    def test_no_columns_raises(self):
        with pytest.raises(ValueError, match="media column"):
            MediaColumn.detect([{"img": PNG}], [])
