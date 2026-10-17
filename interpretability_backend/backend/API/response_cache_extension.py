"""Mark GraphQL results that the read-only response cache may store.

``GraphQLResponseCacheMiddleware`` (``backend/utils/response_cache.py``)
works on HTTP bytes and cannot see execution results, so this extension makes
the final call after execution and signals it with the ``CACHEABLE_HEADER``
response header (the middleware strips it before the response leaves the
server). A result is cacheable when:

- the cache is active: enabled in ``config/response_cache.json`` and the
  instance is read-only, so the data cannot change;
- the operation is a query whose root selections are all allow-listed fields
  (``RootFieldAllowList``: field names, never aliases);
- execution produced no errors.
"""

from strawberry.extensions import SchemaExtension
from strawberry.types import ExecutionContext

from ..utils.graphql_operations import RootFieldAllowList
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
        self.allow_list = RootFieldAllowList(settings.root_fields)

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
        if context.result is None or context.result.errors:
            return False
        return self.allow_list.allows_document(context.graphql_document, context.operation_name)


def _http_response(request_context: object):
    """The Starlette sub-response Strawberry's FastAPI context exposes, if any."""
    if isinstance(request_context, dict):
        response = request_context.get("response")
    else:
        response = getattr(request_context, "response", None)
    return response if hasattr(response, "headers") else None
