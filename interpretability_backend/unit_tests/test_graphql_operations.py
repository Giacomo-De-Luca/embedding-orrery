"""Guard RootFieldAllowList (backend/utils/graphql_operations.py).

The response cache only considers queries whose root selections are all
allow-listed fields. The same check runs twice: on the raw query text before
execution (so other operations skip the cache path entirely) and on the parsed
document after execution (the extension's final decision).
"""

import pytest
from graphql import parse

from backend.utils.graphql_operations import RootFieldAllowList

ALLOW = RootFieldAllowList(("collection",))


@pytest.mark.parametrize(
    "query, operation_name",
    [
        ('{ collection(name: "a") { ids } }', None),
        ('query Q { collection(name: "a") { ids } __typename }', None),
        ('query Q { collection(name: "a") { ids } }', "Q"),
        ('query A { other } query B { collection(name: "a") { ids } }', "B"),
        ('{ c1: collection(name: "a") { ids } c2: collection(name: "b") { ids } }', None),
    ],
    ids=["anonymous", "typename", "named", "picks-named-operation", "aliases-of-allowed"],
)
def test_allows_queries_of_allow_listed_fields(query, operation_name):
    assert ALLOW.allows_query_text(query, operation_name)
    assert ALLOW.allows_document(parse(query), operation_name)


@pytest.mark.parametrize(
    "query, operation_name",
    [
        ("{ other }", None),
        ("{ collection: other }", None),
        ('{ collection(name: "a") { ids } other }', None),
        ("{ __typename }", None),
        ('{ ...F } fragment F on Query { collection(name: "a") { ids } }', None),
        ('{ ... on Query { collection(name: "a") { ids } } }', None),
        ("mutation { collection }", None),
        ("subscription { collection }", None),
        ('query A { collection(name: "a") { ids } } query B { other }', None),
        ('query A { collection(name: "a") { ids } }', "Missing"),
    ],
    ids=[
        "other-field",
        "alias-hides-other-field",
        "mixed",
        "typename-only",
        "fragment-spread",
        "inline-fragment",
        "mutation",
        "subscription",
        "ambiguous-operation",
        "unknown-operation-name",
    ],
)
def test_rejects_everything_else(query, operation_name):
    assert not ALLOW.allows_query_text(query, operation_name)
    assert not ALLOW.allows_document(parse(query), operation_name)


def test_unparsable_query_text_is_rejected():
    assert not ALLOW.allows_query_text("{ collection(", None)


def test_missing_document_is_rejected():
    assert not ALLOW.allows_document(None, None)
