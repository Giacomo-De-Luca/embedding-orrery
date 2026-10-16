"""Fast JSON encoding for HTTP responses: orjson, with a stdlib fallback.

GraphQL responses here reach tens of MB (a full collection load: 81 MB of
JSON for WordNet). The stdlib path builds a Python ``str`` and Starlette then
encodes it to ``bytes`` — two full copies, at a fraction of orjson's speed.
``encode_json_bytes`` writes UTF-8 bytes in one pass. Deliberate differences
from ``json.dumps`` defaults:

- compact separators and raw UTF-8 instead of ``\\uXXXX`` escapes (both valid
  JSON; the output is smaller);
- NaN/Infinity become ``null`` (``json.dumps`` emits bare ``NaN``, which is
  not JSON and makes the browser's ``JSON.parse`` fail);
- NumPy arrays and scalars are serialized natively.

Values orjson refuses but the stdlib accepts (integers beyond 64 bits,
strings holding lone surrogates) fall back to ``json.dumps``, so nothing that
encoded before fails now. ``orjson.JSONEncodeError`` subclasses ``TypeError``;
truly unsupported types still raise ``TypeError`` from the fallback.
"""

import json

import orjson

_ORJSON_OPTIONS = orjson.OPT_NON_STR_KEYS | orjson.OPT_SERIALIZE_NUMPY


def encode_json_bytes(data: object) -> bytes:
    """Serialize ``data`` to UTF-8 JSON bytes (see the module docstring)."""
    try:
        return orjson.dumps(data, option=_ORJSON_OPTIONS)
    except orjson.JSONEncodeError:
        return json.dumps(data).encode("utf-8")
