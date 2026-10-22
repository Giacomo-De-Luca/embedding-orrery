"""GraphQL ``embeddingTaskPrefixes``: the task-prefix table for the embed forms.

One no-argument query (fetched once by the frontend, which matches models
case-insensitively), served from the shipped catalog.
"""

import asyncio

import pytest

QUERY = """
{
  embeddingTaskPrefixes {
    defaultTask
    tasks { id label queryPrefix documentPrefix symmetric }
    models { provider modelName }
  }
}
"""


@pytest.fixture(scope="module")
def table() -> dict:
    from backend.API import schema

    result = asyncio.run(schema.execute(QUERY))
    assert result.errors is None, result.errors
    return result.data["embeddingTaskPrefixes"]


def test_default_task_is_search(table):
    assert table["defaultTask"] == "search"


def test_tasks_carry_both_prefixes(table):
    tasks = {task["id"]: task for task in table["tasks"]}
    assert list(tasks)[0] == "search"
    assert tasks["search"] == {
        "id": "search",
        "label": "Search",
        "queryPrefix": "task: search result | query: ",
        "documentPrefix": "title: none | text: ",
        "symmetric": False,
    }
    assert tasks["clustering"]["symmetric"] is True
    assert tasks["clustering"]["documentPrefix"] == "task: clustering | query: "


def test_models_that_take_task_prefixes(table):
    assert {(m["provider"], m["modelName"]) for m in table["models"]} == {
        ("SENTENCE_TRANSFORMERS", "google/embeddinggemma-300m"),
        ("SENTENCE_TRANSFORMERS", "google/embeddinggemma-2"),
        ("GEMINI", "gemini-embedding-2"),
    }
