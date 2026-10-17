"""Which root fields a GraphQL operation selects.

``RootFieldAllowList`` answers one question for the response cache: is this
operation a query whose root selections are all allow-listed fields? Fields
are matched by name, never by response alias; ``__typename`` is ignored but
at least one allow-listed field is required; fragments at the root are
rejected rather than resolved.

The cache asks twice: ``GraphQLResponseCacheMiddleware`` on the raw query
text before execution, so every other operation skips the cache path, and
``ResponseCacheMarkerExtension`` on the parsed document after execution,
together with the error check only execution can answer.
"""

from collections.abc import Iterable

from graphql import (
    DocumentNode,
    FieldNode,
    GraphQLError,
    OperationDefinitionNode,
    OperationType,
    parse,
)

# Bounds parsing work on untrusted request text; collection queries use ~100.
_MAX_QUERY_TOKENS = 5_000


class RootFieldAllowList:
    """Allow-list of query root field names (see module docstring)."""

    def __init__(self, fields: Iterable[str]):
        self.fields = frozenset(fields)

    def allows_query_text(self, query: str, operation_name: str | None) -> bool:
        """Parse ``query`` and check it; unparsable text is never allowed."""
        try:
            document = parse(query, no_location=True, max_tokens=_MAX_QUERY_TOKENS)
        except (GraphQLError, RecursionError):  # syntax errors, token limit, deep nesting
            return False
        return self.allows_document(document, operation_name)

    def allows_document(self, document: DocumentNode | None, operation_name: str | None) -> bool:
        operation = _select_operation(document, operation_name) if document is not None else None
        if operation is None or operation.operation != OperationType.QUERY:
            return False
        names = []
        for selection in operation.selection_set.selections:
            if not isinstance(selection, FieldNode):
                return False
            if selection.name.value != "__typename":
                names.append(selection.name.value)
        return bool(names) and all(name in self.fields for name in names)


def _select_operation(
    document: DocumentNode, operation_name: str | None
) -> OperationDefinitionNode | None:
    """The operation a request executes, as GraphQL resolves it (None if ambiguous)."""
    operations = [d for d in document.definitions if isinstance(d, OperationDefinitionNode)]
    if operation_name is None:
        return operations[0] if len(operations) == 1 else None
    return next(
        (op for op in operations if op.name is not None and op.name.value == operation_name),
        None,
    )
