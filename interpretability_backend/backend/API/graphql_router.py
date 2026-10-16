"""GraphQL router that encodes responses with orjson.

Strawberry's FastAPI router builds each response body with ``json.dumps``
and Starlette then encodes that ``str`` to ``bytes``. For a full collection
load (WordNet: 81 MB of JSON) that is two complete copies plus the slow
stdlib encoder. ``OrreryGraphQLRouter`` writes the body as bytes in one pass
via ``encode_json_bytes`` (see ``backend/utils/json_encoding.py``).

``create_response`` is overridden rather than ``encode_json`` because
Strawberry's multipart (incremental delivery) framing concatenates
``encode_json`` output as ``str``; that path keeps the stdlib encoder.
"""

from fastapi import Response, status
from strawberry.fastapi import GraphQLRouter
from strawberry.http import GraphQLHTTPResponse

from ..utils.json_encoding import encode_json_bytes


class OrreryGraphQLRouter(GraphQLRouter):
    """``GraphQLRouter`` with orjson-encoded response bodies."""

    def create_response(
        self,
        response_data: GraphQLHTTPResponse | list[GraphQLHTTPResponse],
        sub_response: Response,
    ) -> Response:
        response = Response(
            encode_json_bytes(response_data),
            media_type="application/json",
            status_code=sub_response.status_code or status.HTTP_200_OK,
        )
        response.headers.raw.extend(sub_response.headers.raw)
        return response
