"""Lazy singleton for the InterpretService."""

from ..services.interpret_service import InterpretService

_interpret_service: InterpretService | None = None


def get_interpret_service() -> InterpretService:
    """Get the shared InterpretService instance, creating it on first call."""
    global _interpret_service
    if _interpret_service is None:
        _interpret_service = InterpretService()
    return _interpret_service
