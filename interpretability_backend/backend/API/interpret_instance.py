"""Lazy singleton for the InterpretService.

The import of ``InterpretService`` is deliberately deferred to first use:
its module pulls in torch and the whole interpret/ toolkit (~1-2 s import,
hundreds of MB resident), and the GraphQL schema imports this module at
startup. Keeping the heavy import out of module scope lets the backend run
without torch installed as long as no SAE endpoint is called (e.g. the
torch-free demo image).
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..services.interpret_service import InterpretService

_interpret_service: "InterpretService | None" = None


def get_interpret_service() -> "InterpretService":
    """Get the shared InterpretService instance, creating it on first call."""
    global _interpret_service
    if _interpret_service is None:
        from ..services.interpret_service import InterpretService

        _interpret_service = InterpretService()
    return _interpret_service
