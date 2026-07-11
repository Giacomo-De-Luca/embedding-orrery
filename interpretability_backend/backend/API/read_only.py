"""Read-only demo mode for public deployments (e.g. the HuggingFace Space).

When ``ORRERY_READ_ONLY=1`` the GraphQL schema rejects every mutation before
execution — hiding write UI is not enough on a public endpoint, where anyone
can open the playground and call ``deleteCollection`` or launch an embedding
job. Queries and subscriptions pass through unchanged.

The env var is read per-operation (not at import) so tests can toggle it with
monkeypatch and a single build works for both demo and normal deployments.
"""

import os

from graphql import ExecutionResult as GraphQLExecutionResult, GraphQLError
from strawberry.extensions import SchemaExtension
from strawberry.types.graphql import OperationType

READ_ONLY_ENV = "ORRERY_READ_ONLY"
READ_ONLY_MESSAGE = "This demo instance is read-only: mutations are disabled."

# Accept the common truthy spellings — a hand-configured public deployment
# setting ORRERY_READ_ONLY=true must not silently run with writes enabled.
_TRUTHY = {"1", "true", "yes", "on"}


def is_read_only() -> bool:
    """Whether the read-only demo flag is set (checked per call, not cached)."""
    return os.getenv(READ_ONLY_ENV, "").strip().lower() in _TRUTHY


class ReadOnlyExtension(SchemaExtension):
    """Short-circuit mutation operations when the read-only flag is set.

    Setting ``execution_context.result`` before yielding skips the real
    executor entirely, so blocked mutations never touch resolvers or the DB.
    """

    def on_execute(self):
        if is_read_only():
            context = self.execution_context
            if context.operation_type == OperationType.MUTATION:
                context.result = GraphQLExecutionResult(
                    data=None,
                    errors=[GraphQLError(READ_ONLY_MESSAGE)],
                )
        yield
