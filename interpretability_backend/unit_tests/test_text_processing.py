"""Column templates (utils/text_processing.py): "{title}: {abstract}" over one row."""

import math

import pytest

from backend.utils.text_processing import ColumnTemplate, format_text_for_embedding


class TestColumnTemplate:
    def test_fields_in_order_without_repeats(self):
        template = ColumnTemplate("{title}: {abstract} ({title})")
        assert template.fields == ("title", "abstract")

    def test_renders_columns_literally_named(self):
        template = ColumnTemplate("{Conc.M} / {word}")
        assert template.render({"Conc.M": 4.5, "word": "cat"}) == "4.5 / cat"

    def test_doubled_braces_are_literal(self):
        assert ColumnTemplate("{{x}} {a}").render({"a": "1"}) == "{x} 1"

    def test_missing_values_render_as_the_missing_text(self):
        template = ColumnTemplate("title: {title} | text: ")
        expected = "title: none | text: "
        for row in ({}, {"title": None}, {"title": ""}, {"title": "  "}, {"title": math.nan}):
            assert template.render(row, missing="none") == expected
        assert template.render({"title": None}) == "title:  | text: "

    def test_lists_join_like_the_concatenation(self):
        assert ColumnTemplate("{tags}").render({"tags": ["a", "b"]}) == "a b"

    def test_missing_from_lists_unknown_columns(self):
        template = ColumnTemplate("{title}: {abstract}")
        assert template.missing_from(["abstract", "year"]) == ["title"]
        assert template.missing_from(["title", "abstract"]) == []

    def test_require_columns_names_the_unknown_ones(self):
        with pytest.raises(ValueError, match=r"Task prefix .*tittle"):
            ColumnTemplate("title: {tittle} | text: ").require_columns(["title"], "Task prefix")

    @pytest.mark.parametrize("bad", ["{}", "{0}{", "{a!r}", "{a:>4}", "}"])
    def test_rejects_malformed_templates(self, bad):
        with pytest.raises(ValueError):
            ColumnTemplate(bad)

    def test_parse_is_cached(self):
        assert ColumnTemplate.parse("{a}") is ColumnTemplate.parse("{a}")


class TestFormatTextForEmbedding:
    def test_template_renders_the_embedded_columns(self):
        row = {"title": "Attention", "abstract": "We propose", "year": 2017}
        text = format_text_for_embedding(row, ["title", "abstract"], "{title}: {abstract}")
        assert text == "Attention: We propose"

    def test_none_renders_empty_instead_of_the_word_none(self):
        row = {"title": None, "abstract": "We propose"}
        text = format_text_for_embedding(row, ["title", "abstract"], "{title}: {abstract}")
        assert text == ": We propose"

    def test_a_placeholder_outside_the_embedded_columns_raises(self):
        """It used to fall back to plain concatenation without a word."""
        with pytest.raises(ValueError, match="year"):
            format_text_for_embedding(
                {"abstract": "x", "year": 1}, ["abstract"], "{abstract} {year}"
            )

    def test_without_a_template_columns_are_concatenated(self):
        row = {"title": "A", "abstract": "B", "tags": ["c", "d"], "n": 3}
        assert format_text_for_embedding(row, ["title", "abstract", "tags", "n"]) == "A B c d 3"
