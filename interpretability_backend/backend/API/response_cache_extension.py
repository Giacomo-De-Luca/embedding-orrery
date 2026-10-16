"""Mark GraphQL results that the read-only response cache may store.

``GraphQLResponseCacheMiddleware`` (``backend/utils/response_cache.py``)
works on HTTP bytes and cannot see GraphQL semantics, so this extension
decides after execution whether a response is safe to replay, and says so
with the ``CACHEABLE_HEADER`` response header (the middleware strips it
before the response leaves the server). A result is cacheable when:

- the cache is active: enabled in ``config/response_cache.json`` and the
  instance is read-only, so the data cannot change;
- the operation is a query;
- every root selection is a field in ``root_fields``, matched by field name,
  never by response alias (``__typename`` is ignored);
- execution produced no errors.
"""

from graphql import FieldNode, OperationDefinitionNode
from strawberry.extensions import SchemaExtension
from strawberry.types import ExecutionContext
from strawberry.types.graphql import OperationType

from ..utils.response_cache import CACHEABLE_HEADER, ResponseCacheSettings
from .read_only import is_read_only


def response_cache_active(settings: ResponseCacheSettings) -> bool:
    """The cache only runs where data is immutable: enabled and read-only."""
    return settings.enabled and is_read_only()


class ResponseCacheMarkerExtension(SchemaExtension):
    """Set ``CACHEABLE_HEADER`` on responses the cache may store (see module docstring)."""

    def __init__(
        self,
        *,
        settings: ResponseCacheSettings,
        execution_context: ExecutionContext | None = None,
    ):
        super().__init__(execution_context=execution_context)
        self.settings = settings

    def on_execute(self):
        # Strawberry reuses one extension instance and reassigns
        # ``execution_context`` for every operation; another request can run
        # while this one executes, so keep this operation's context locally.
        context = self.execution_context
        yield
        if not self._cacheable(context):
            return
        response = _http_response(context.context)
        if response is not None:
            response.headers[CACHEABLE_HEADER] = "1"

    def _cacheable(self, context: ExecutionContext) -> bool:
        if not response_cache_active(self.settings):
            return False
        if context.operation_type != OperationType.QUERY:
            return False
        if context.result is None or context.result.errors:
            return False
        return _root_fields_allowed(context, frozenset(self.settings.root_fields))


def _root_fields_allowed(context: ExecutionContext, allowed: frozenset[str]) -> bool:
    operation = _executed_operation(context)
    if operation is None:
        return False
    names = []
    for selection in operation.selection_set.selections:
        if not isinstance(selection, FieldNode):
            return False  # fragments at the root: not worth resolving here
        names.append(selection.name.value)
    fields = [name for name in names if name != "__typename"]
    return bool(fields) and all(name in allowed for name in fields)


def _executed_operation(context: ExecutionContext) -> OperationDefinitionNode | None:
    document = context.graphql_document
    if document is None:
        return None
    operations = [d for d in document.definitions if isinstance(d, OperationDefinitionNode)]
    name = context.operation_name
    if name is None:
        return operations[0] if len(operations) == 1 else None
    return next((op for op in operations if op.name is not None and op.name.value == name), None)


def _http_response(request_context: object):
    """The Starlette sub-response Strawberry's FastAPI context exposes, if any."""
    if isinstance(request_context, dict):
        response = request_context.get("response")
    else:
        response = getattr(request_context, "response", None)
    return response if hasattr(response, "headers") else None
