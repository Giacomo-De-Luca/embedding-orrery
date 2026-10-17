"""Fast JSON encoding for HTTP responses: orjson, with a stdlib fallback.

GraphQL responses here reach tens of MB (a full collection load: 81 MB of
JSON for WordNet). The stdlib path builds a Python ``str`` and Starlette then
encodes it to ``bytes`` — two full copies, at a fraction of orjson's speed.
``encode_json_bytes`` writes UTF-8 bytes in one pass.

Deliberate differences from ``json.dumps`` defaults:
- compact separators and raw UTF-8 instead of ``\\uXXXX`` escapes (both valid
  JSON; the output is smaller);
- NaN/Infinity become ``null``. ``json.dumps`` emits bare ``NaN``, which is
  not JSON and makes the browser's ``JSON.parse`` fail;
- NumPy arrays and scalars, UUIDs and enums are serialized natively.

Kept like the stdlib on purpose: ``datetime``/``date``/``time`` values and
dataclasses still raise ``TypeError`` (via the passthrough options and the
fallback). Strawberry's WebSocket path keeps ``json.dumps``, so an
unsanitized value must fail on both transports, not only on subscriptions
(see ``_sanitize_for_json`` in ``duckdb_client.py``).

Fallback: values orjson refuses but the stdlib accepts (integers beyond 64
bits, strings holding lone surrogates) go through ``json.dumps``, so nothing
that encoded before fails now. That path writes the whole payload with stdlib
semantics, including bare ``NaN``. ``orjson.JSONEncodeError`` subclasses
``TypeError``.
"""

import json

import orjson

_ORJSON_OPTIONS = (
    orjson.OPT_NON_STR_KEYS
    | orjson.OPT_SERIALIZE_NUMPY
    | orjson.OPT_PASSTHROUGH_DATETIME
    | orjson.OPT_PASSTHROUGH_DATACLASS
)


def encode_json_bytes(data: object) -> bytes:
    """Serialize ``data`` to UTF-8 JSON bytes (see the module docstring)."""
    try:
        return orjson.dumps(data, option=_ORJSON_OPTIONS)
    except orjson.JSONEncodeError:
        return json.dumps(data).encode("utf-8")
