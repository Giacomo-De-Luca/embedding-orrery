"""Torch-free steering types shared between the GraphQL layer and InterpretService.

Lives in its own module so ``mutations.py``/``subscriptions.py`` can import
``SteeringSpec`` without pulling in ``interpret_service`` (and with it torch and
the interpret/ toolkit) at schema-build time. Heavy imports happen lazily on
the first SAE request via ``interpret_instance.get_interpret_service()``.
"""

from dataclasses import dataclass


@dataclass
class SteeringSpec:
    """A single steering specification (service-internal).

    Two mutually exclusive flavours:
      - SAE feature: ``feature_index`` + ``layer`` + ``hook_type`` + ``width``
        resolve a direction from ``sae.w_dec``.
      - Pre-extracted direction: ``direction_name`` resolves a 1-D vector
        from ``DIRECTION_REGISTRY``. The other fields are ignored.
    """

    feature_index: int
    layer: int
    hook_type: str
    width: str
    strength: float
    direction_name: str | None = None
