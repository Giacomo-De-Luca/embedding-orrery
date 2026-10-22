"""Task prefixes of EmbeddingGemma / Gemini Embedding 2 (utils/embedding_tasks.py).

Documents carry a per-row prefix template (``title: {title} | text: ``) that
reaches the model input only; queries get the task's query prefix stored on
the collection. Runs on the shipped catalog.
"""

import pytest

from backend.utils.embedding_model_catalog import TASK_PREFIX_FORMAT
from backend.utils.embedding_tasks import DocumentPrefix, TaskPrefixPolicy

ST = "sentence_transformers"
EG2 = "google/embeddinggemma-2"
SEARCH_QUERY = "task: search result | query: "


@pytest.fixture
def policy() -> TaskPrefixPolicy:
    return TaskPrefixPolicy()


class TestPromptFormat:
    @pytest.mark.parametrize(
        "provider, model, prompt",
        [
            (ST, EG2, "title: {title} | text: "),
            (ST, "google/embeddinggemma-300m", "task: clustering | query: "),
            ("gemini", "gemini-embedding-2", None),
            ("gemini", "Gemini-Embedding-2", ""),
        ],
    )
    def test_task_prefix_models_get_the_marker(self, policy, provider, model, prompt):
        assert policy.prompt_format_for(provider, model, prompt) == TASK_PREFIX_FORMAT

    @pytest.mark.parametrize(
        "provider, model, prompt",
        [
            # Prompt names: collections and resumes from before task prefixes.
            (ST, EG2, "Retrieval-document"),
            (ST, "google/embeddinggemma-300m", "STS"),
            # Not a task-prefix model.
            (ST, "all-MiniLM-L6-v2", "title: {title} | text: "),
            ("gemini", "gemini-embedding-001", None),
            (ST, "/models/my-gemma-checkpoint", "Retrieval-document"),
        ],
    )
    def test_other_configs_keep_the_old_path(self, policy, provider, model, prompt):
        assert policy.prompt_format_for(provider, model, prompt) is None

    def test_uses_task_prefixes_follows_the_catalog(self, policy):
        assert policy.uses_task_prefixes("gemini", "gemini-embedding-2")
        assert not policy.uses_task_prefixes("gemini", "gemini-embedding-001")


class TestDocumentPrefix:
    def test_renders_the_row_into_the_model_input(self):
        prefix = DocumentPrefix("title: {title} | text: ")
        text = prefix.model_input({"title": "Attention", "abstract": "x"}, "We propose")
        assert text == "title: Attention | text: We propose"

    def test_a_missing_title_reads_none(self):
        prefix = DocumentPrefix("title: {title} | text: ")
        assert prefix.model_input({"title": None}, "doc") == "title: none | text: doc"

    def test_a_static_prefix_is_prepended(self):
        assert DocumentPrefix("task: clustering | query: ").model_input({}, "doc") == (
            "task: clustering | query: doc"
        )

    def test_validate_names_unknown_columns(self):
        prefix = DocumentPrefix("title: {headline} | text: ")
        assert prefix.columns == ("headline",)
        prefix.validate(["headline", "abstract"])
        with pytest.raises(ValueError, match="headline"):
            prefix.validate(["title", "abstract"])

    def test_policy_builds_it_only_for_marked_configs(self, policy):
        prefix = policy.document_prefix(TASK_PREFIX_FORMAT, "title: none | text: ")
        assert isinstance(prefix, DocumentPrefix)
        assert policy.document_prefix(TASK_PREFIX_FORMAT, None) is None
        assert policy.document_prefix(TASK_PREFIX_FORMAT, "") is None
        assert policy.document_prefix(None, "title: none | text: ") is None


class TestQueryPrefix:
    def test_stored_prefix_without_an_explicit_prompt(self, policy):
        assert policy.query_prefix(stored=SEARCH_QUERY, explicit=None) == SEARCH_QUERY
        assert policy.query_prefix(stored=SEARCH_QUERY, explicit="") == SEARCH_QUERY
        assert policy.query_prefix(stored="", explicit=None) is None
        assert policy.query_prefix(stored=None, explicit=None) is None

    def test_explicit_literal_text_wins(self, policy):
        explicit = "task: question answering | query: "
        assert policy.query_prefix(stored=SEARCH_QUERY, explicit=explicit) == explicit

    @pytest.mark.parametrize(
        "name, prefix",
        [
            ("Retrieval-query", SEARCH_QUERY),
            ("SearchQuery", SEARCH_QUERY),
            ("STS", "task: sentence similarity | query: "),
            ("Clustering", "task: clustering | query: "),
        ],
    )
    def test_explicit_prompt_names_resolve_through_their_task(self, policy, name, prefix):
        assert policy.query_prefix(stored=None, explicit=name) == prefix

    def test_other_prompt_names_raise(self, policy):
        with pytest.raises(ValueError, match="Retrieval-document"):
            policy.query_prefix(stored=SEARCH_QUERY, explicit="Retrieval-document")
