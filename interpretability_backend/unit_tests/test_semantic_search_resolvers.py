"""Semantic-search resolvers, executed against the real schema.

DuckDB is the in-memory fixture (real filter SQL, real topic assignments);
ChromaDB is a fake that records the arguments it receives and ranks items in a
fixed order. The point under test: filters are resolved in DuckDB and handed
to the vector store as an ``ids`` scope, so a filtered search returns the full
``nResults`` instead of whatever survived a post-filter of a 5x over-fetch.
"""

import asyncio

import numpy as np
import pytest

from backend.API import queries, schema

COLLECTION = "ds_dense"
# Fake nearest-neighbour order for every query.
RANKING = ["a", "b", "c", "d", "e"]
YEARS = {"a": 2021, "b": 2021, "c": 2020, "d": 2021, "e": 2020}
TOPICS = {"a": 1, "b": 1, "c": 0, "d": 1, "e": 0}


class FakeCollection:
    def get(self, ids, include):
        return {"ids": ids, "embeddings": [[0.5, 0.5] for _ in ids]}


class FakeChroma:
    def __init__(self):
        self.calls: list[dict] = []

    def semantic_search(
        self,
        collection_name,
        query_texts=None,
        query_embeddings=None,
        n_results=10,
        distance_metric="cosine",
        query_prompt=None,
        ids=None,
    ):
        self.calls.append(
            {
                "query_texts": query_texts,
                "query_embeddings": query_embeddings,
                "n_results": n_results,
                "ids": ids,
            }
        )
        pool = RANKING if ids is None else [i for i in RANKING if i in set(ids)]
        hits = pool[:n_results]
        distances = [0.1 * k for k in range(len(hits))]
        return {
            "ids": [hits],
            "distances": [distances],
            "similarities": [[1 - d for d in distances]],
            "embeddings": None,
        }

    def get_collection(self, name, load_embedding_function=False):
        return FakeCollection()

    def embed_query(self, collection_name, text, query_prompt=None):
        return np.array([1.0, 0.0])


@pytest.fixture
def seeded(db, monkeypatch):
    db.create_dataset("ds")
    db.insert_items_batch(
        "ds", RANKING, [f"doc {i}" for i in RANKING], [{"year": YEARS[i]} for i in RANKING]
    )
    db.register_vector_collection("ds", "chromadb", COLLECTION, "dense")
    ext = db.create_topic_extraction(COLLECTION, "ds")
    db.insert_topic_assignments_batch(
        ext,
        [{"item_id": i, "topic_id": TOPICS[i], "topic_label": f"T{TOPICS[i]}"} for i in RANKING],
    )
    db.replace_topic_centroids(
        ext,
        [
            {"level": "topic", "topic_id": 0, "member_count": 2, "centroid": [1.0, 0.0]},
            {"level": "topic", "topic_id": 1, "member_count": 3, "centroid": [-1.0, 0.0]},
        ],
    )
    chroma = FakeChroma()
    monkeypatch.setattr(queries, "get_duckdb_client", lambda: db)
    monkeypatch.setattr(queries, "get_chromadb_client", lambda: chroma)
    return chroma


def _execute(query: str, **variables):
    result = asyncio.run(schema.execute(query, variable_values=variables))
    assert result.errors is None, result.errors
    return result.data


SEMANTIC = """
query($c: String!, $q: String!, $n: Int, $f: [FilterInput!]) {
  semanticSearch(collectionName: $c, query: $q, nResults: $n, filters: $f) {
    id similarity metadata
  }
}
"""

BY_ID = """
query($c: String!, $id: String!, $n: Int, $f: [FilterInput!]) {
  semanticSearchById(collectionName: $c, itemId: $id, nResults: $n, filters: $f) { id }
}
"""


def _ids(rows):
    return [r["id"] for r in rows]


class TestSemanticSearch:
    def test_unfiltered_search_is_unscoped(self, seeded):
        data = _execute(SEMANTIC, c=COLLECTION, q="cats", n=2)

        assert _ids(data["semanticSearch"]) == ["a", "b"]
        assert data["semanticSearch"][0]["metadata"] == {"year": 2021}
        assert seeded.calls[-1]["ids"] is None
        assert seeded.calls[-1]["query_texts"] == ["cats"]

    def test_metadata_filter_scopes_the_vector_search(self, seeded):
        filters = [{"field": "year", "operator": "EQ", "value": 2020}]
        data = _execute(SEMANTIC, c=COLLECTION, q="cats", n=2, f=filters)

        assert sorted(seeded.calls[-1]["ids"]) == ["c", "e"]
        # Both 2020 items come back even though they rank 3rd and 5th overall.
        assert _ids(data["semanticSearch"]) == ["c", "e"]

    def test_topic_filter_resolves_through_assignments(self, seeded):
        filters = [{"field": "topic_id", "operator": "IN", "value": [0]}]
        data = _execute(SEMANTIC, c=COLLECTION, q="cats", n=10, f=filters)

        assert sorted(seeded.calls[-1]["ids"]) == ["c", "e"]
        assert _ids(data["semanticSearch"]) == ["c", "e"]

    def test_filter_matching_nothing_skips_the_vector_store(self, seeded):
        filters = [{"field": "year", "operator": "EQ", "value": 1999}]
        data = _execute(SEMANTIC, c=COLLECTION, q="cats", n=5, f=filters)

        assert data["semanticSearch"] == []
        assert seeded.calls == []


class TestSemanticSearchById:
    def test_filter_scopes_the_vector_search(self, seeded):
        filters = [{"field": "topic_id", "operator": "EQ", "value": 1}]
        data = _execute(BY_ID, id="c", c=COLLECTION, n=3, f=filters)

        call = seeded.calls[-1]
        assert sorted(call["ids"]) == ["a", "b", "d"]
        assert call["query_embeddings"] == [[0.5, 0.5]]
        assert _ids(data["semanticSearchById"]) == ["a", "b", "d"]


class TestTextSearch:
    def test_topic_filter_is_honoured(self, seeded):
        query = """
        query($c: String!, $f: [FilterInput!]) {
          textSearch(collectionName: $c, query: "doc", filters: $f) {
            totalMatches matches { id }
          }
        }
        """
        filters = [{"field": "topic_id", "operator": "EQ", "value": 0}]
        data = _execute(query, c=COLLECTION, f=filters)

        assert data["textSearch"]["totalMatches"] == 2
        assert sorted(m["id"] for m in data["textSearch"]["matches"]) == ["c", "e"]


class TestTopicSemanticSearch:
    def test_returns_ranked_topics(self, seeded):
        query = """
        query($c: String!, $q: String!) {
          topicSemanticSearch(collectionName: $c, query: $q, limit: 5) {
            topicId similarity matchedSubtopicId matchedSubtopicLabel
          }
        }
        """
        data = _execute(query, c=COLLECTION, q="cats")

        results = data["topicSemanticSearch"]
        assert [r["topicId"] for r in results] == [0, 1]
        assert results[0]["similarity"] == pytest.approx(1.0)
        assert results[0]["matchedSubtopicId"] is None
        assert results[0]["matchedSubtopicLabel"] is None
