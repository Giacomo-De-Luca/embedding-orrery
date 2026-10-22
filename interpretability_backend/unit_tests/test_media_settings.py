"""Media embedding settings (config/media_embedding.json, utils/media_settings.py).

The config holds the defaults and limits of the per-job media options (token
budgets, video sampling, audio length). The embed forms show the defaults and
send overrides, validated here, plus the per-row token-budget check.
"""

import json

import pytest

from backend.utils.media_kinds import DEFAULT_EXTENSIONS, MediaKind
from backend.utils.media_settings import (
    MEDIA_SETTINGS_FILE,
    MediaEmbeddingSettings,
    MediaOptions,
)


@pytest.fixture
def settings() -> MediaEmbeddingSettings:
    return MediaEmbeddingSettings()


class TestShippedSettings:
    def test_shipped_file_loads_with_the_documented_defaults(self):
        shipped = MediaEmbeddingSettings.from_file(MEDIA_SETTINGS_FILE)
        assert shipped.defaults == MediaOptions(
            image_max_soft_tokens=280,
            video_fps=1.0,
            video_max_frames=32,
            video_max_soft_tokens=140,
            max_audio_seconds=120.0,
        )
        assert shipped.soft_token_budgets == (70, 140, 280, 560, 1120)
        assert shipped.context_tokens == 8192
        assert shipped.audio_sample_rate == 16000
        assert shipped.allow_remote_urls is False

    def test_extensions_default_to_the_detector_lists(self, settings):
        assert settings.extensions() == DEFAULT_EXTENSIONS


class TestOverrides:
    def test_no_overrides_means_the_defaults(self, settings):
        assert settings.with_overrides(None) == settings.defaults
        assert settings.with_overrides({}) == settings.defaults

    def test_partial_overrides_merge_into_the_defaults(self, settings):
        options = settings.with_overrides({"video_max_frames": 16, "image_max_soft_tokens": 70})
        assert options.video_max_frames == 16
        assert options.image_max_soft_tokens == 70
        assert options.video_fps == settings.defaults.video_fps

    def test_none_values_keep_the_default(self, settings):
        options = settings.with_overrides({"video_fps": None, "max_audio_seconds": 30})
        assert options.video_fps == settings.defaults.video_fps
        assert options.max_audio_seconds == 30

    @pytest.mark.parametrize(
        "overrides, message",
        [
            ({"image_max_soft_tokens": 300}, "image_max_soft_tokens"),
            ({"video_max_soft_tokens": 0}, "video_max_soft_tokens"),
            ({"video_fps": 0}, "video_fps"),
            ({"video_fps": 100}, "video_fps"),
            ({"video_max_frames": 0}, "video_max_frames"),
            ({"video_max_frames": 10_000}, "video_max_frames"),
            ({"max_audio_seconds": -1}, "max_audio_seconds"),
            ({"max_audio_seconds": 1000}, "max_audio_seconds"),
            ({"video_max_frames": True}, "video_max_frames"),
            ({"video_max_frames": "8"}, "video_max_frames"),
            ({"colour": 1}, "colour"),
        ],
    )
    def test_invalid_overrides_are_rejected(self, settings, overrides, message):
        with pytest.raises(ValueError, match=message):
            settings.with_overrides(overrides)

    def test_overrides_round_trip_as_a_dict(self, settings):
        options = settings.with_overrides({"video_fps": 2})
        assert MediaOptions(**options.to_dict()) == options
        assert options.to_dict()["video_fps"] == 2.0


class TestProcessingKwargs:
    def test_only_the_kinds_of_the_job(self):
        options = MediaOptions(image_max_soft_tokens=70, video_max_soft_tokens=280)
        assert options.processing_kwargs({MediaKind.IMAGE}) == {"image": {"max_soft_tokens": 70}}
        assert options.processing_kwargs({MediaKind.AUDIO}) == {}

    def test_pre_sampled_video_frames_are_not_sampled_again(self):
        kwargs = MediaOptions().processing_kwargs({"video"})
        assert kwargs == {"video": {"max_soft_tokens": 140, "do_sample_frames": False}}

    def test_image_and_video_together(self):
        kwargs = MediaOptions().processing_kwargs({MediaKind.IMAGE, MediaKind.VIDEO})
        assert set(kwargs) == {"image", "video"}


class TestTokenBudget:
    def test_estimate_counts_each_media_column(self, settings):
        options = settings.defaults
        kinds = {MediaKind.IMAGE: 2, MediaKind.VIDEO: 1, MediaKind.AUDIO: 1}
        expected = (
            settings.reserved_tokens
            + 2 * options.image_max_soft_tokens
            + options.video_max_frames * options.video_max_soft_tokens
            + int(options.max_audio_seconds * settings.audio_tokens_per_second)
        )
        assert settings.estimate_tokens(options, kinds) == expected

    def test_kinds_can_be_given_as_strings(self, settings):
        options = settings.defaults
        assert settings.estimate_tokens(options, {"image": 1}) == settings.estimate_tokens(
            options, {MediaKind.IMAGE: 1}
        )

    def test_within_budget_passes(self, settings):
        settings.check_budget(settings.defaults, {MediaKind.VIDEO: 1, MediaKind.AUDIO: 1})

    def test_over_budget_names_the_estimate(self, settings):
        with pytest.raises(ValueError, match=r"\d+ tokens.*8192"):
            settings.check_budget(settings.defaults, {MediaKind.VIDEO: 2})

    def test_lower_frame_budget_fits_two_videos(self, settings):
        options = settings.with_overrides({"video_max_frames": 16})
        settings.check_budget(options, {MediaKind.VIDEO: 2})

    def test_text_gets_what_the_media_leave(self, settings):
        """A row's text is cut to this, so text plus worst-case media fits the context."""
        options = settings.defaults
        kinds = {MediaKind.VIDEO: 1, MediaKind.AUDIO: 1}
        media = settings.estimate_tokens(options, kinds) - settings.reserved_tokens
        assert settings.text_token_budget(options, kinds) == settings.context_tokens - media
        assert settings.text_token_budget(options, {"image": 1}) == (
            settings.context_tokens - options.image_max_soft_tokens
        )

    def test_the_text_budget_never_drops_below_the_reserve(self, settings):
        """check_budget guarantees it; the budget follows from the same numbers."""
        kinds = {MediaKind.VIDEO: 1, MediaKind.AUDIO: 1, MediaKind.IMAGE: 1}
        options = settings.with_overrides({"max_audio_seconds": 100})
        settings.check_budget(options, kinds)
        assert settings.text_token_budget(options, kinds) >= settings.reserved_tokens


class TestSettingsValidation:
    def test_unknown_file_keys_are_rejected(self, tmp_path):
        path = tmp_path / "media.json"
        path.write_text(json.dumps({"vidoe_fps": 2}))
        with pytest.raises(ValueError, match="vidoe_fps"):
            MediaEmbeddingSettings.from_file(path)

    def test_unknown_default_keys_are_rejected(self, tmp_path):
        path = tmp_path / "media.json"
        path.write_text(json.dumps({"defaults": {"fps": 2}}))
        with pytest.raises(ValueError, match="fps"):
            MediaEmbeddingSettings.from_file(path)

    def test_defaults_must_respect_the_limits(self):
        with pytest.raises(ValueError, match="video_max_frames"):
            MediaEmbeddingSettings(defaults={"video_max_frames": 500}, video_frames_limit=128)

    def test_budgets_must_be_positive_integers(self):
        with pytest.raises(ValueError, match="soft_token_budgets"):
            MediaEmbeddingSettings(soft_token_budgets=(70, -1))

    def test_extension_overrides_must_be_dotted(self):
        with pytest.raises(ValueError, match="extensions"):
            MediaEmbeddingSettings(image_extensions=("png",))

    def test_extension_overrides_replace_one_kind(self):
        custom = MediaEmbeddingSettings(image_extensions=(".PNG",))
        assert custom.extensions()[MediaKind.IMAGE] == (".png",)
        assert custom.extensions()[MediaKind.AUDIO] == DEFAULT_EXTENSIONS[MediaKind.AUDIO]
