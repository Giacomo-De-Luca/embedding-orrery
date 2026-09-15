"""Native Cosmograph protocol tests: no live database or model access."""

import asyncio
import json
from dataclasses import replace

import pyarrow as pa
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.services.graph_query_service import GraphQueryService, GraphRegistry, GraphSettings
from backend.API.graphs import create_graph_router
from backend.utils.graph_datasets.contract import GraphBundleWriter
from test_graph_datasets import fixture_tables


@pytest.fixture
def bundle(tmp_path):
    nodes, edges, manifest = fixture_tables()
    path = GraphBundleWriter().write(tmp_path, nodes, edges, manifest)
    return tmp_path, path


@pytest.fixture
def graph_api(bundle):
    root, _ = bundle
    service = GraphQueryService(GraphRegistry(root), GraphSettings())
    app = FastAPI()
    app.include_router(create_graph_router(service))
    with TestClient(app) as client:
        descriptor = service.registry.list()[0]
        yield client, descriptor, service
    service.close()


def test_native_protocol_and_edge_rowids(graph_api):
    client, descriptor, _ = graph_api
    url = descriptor['queryUrl']
    queries = ['SHOW ALL TABLES', "PRAGMA table_info('graph_nodes')", 'SUMMARIZE graph_nodes',
               'SELECT * FROM graph_nodes ORDER BY index',
               'SELECT rowid, sourceIndex, targetIndex FROM graph_edges',
               'SELECT index FROM graph_nodes WHERE index IN (SELECT sourceIndex FROM graph_edges UNION SELECT targetIndex FROM graph_edges)']
    for sql in queries:
        response = client.post(url, json={'type': 'arrow', 'sql': sql})
        assert response.status_code == 200, response.text
        table = pa.ipc.open_stream(response.content).read_all()
        assert table.num_rows > 0
    rows = client.post(url, json={'type': 'json', 'sql': queries[3]}).json()
    assert rows[0]['featureIndex'] == 2
    assert rows[2]['activeEventCount'] == 0
    edges = client.post(url, json={'type': 'json', 'sql': queries[4]}).json()
    assert [e['rowid'] for e in edges] == [0]


@pytest.mark.parametrize('sql', ['SELECT 1; SELECT 2', 'DROP TABLE graph_nodes',
    'DELETE FROM graph_edges', "ATTACH ':memory:' AS stolen", "COPY graph_nodes TO '/tmp/stolen'",
    'SET enable_external_access=true', "PRAGMA version", 'SELECT * FROM sae_features',
    "SELECT * FROM read_csv('/tmp/absent-graph-test.csv')"])
def test_rejects_unsupported_and_external_queries(graph_api, sql):
    client, descriptor, _ = graph_api
    assert client.post(descriptor['queryUrl'], json={'type': 'arrow', 'sql': sql}).status_code in (400, 403)


def test_read_only_demo_and_request_limits(graph_api, monkeypatch):
    client, descriptor, service = graph_api
    monkeypatch.setenv('ORRERY_READ_ONLY', '1')
    url = descriptor['queryUrl']
    assert client.post(url, json={'type': 'json', 'sql': 'SELECT COUNT(*) AS n FROM graph_nodes'}).json() == [{'n': 4}]
    assert client.post(url, json={'type': 'exec', 'sql': 'SELECT 1'}).status_code == 400
    assert client.post(url, content=b'x' * (service.settings.max_request_bytes + 1)).status_code == 413
    assert client.post('/graphs/missing/revision/query', json={'type': 'json', 'sql': 'SELECT 1'}).status_code == 404


def test_timeout_finishes_worker_before_reuse(bundle):
    root, _ = bundle
    service = GraphQueryService(GraphRegistry(root), replace(GraphSettings(), query_timeout_seconds=.01))
    descriptor = service.registry.list()[0]
    manifest = descriptor['manifest']
    async def run():
        with pytest.raises(TimeoutError):
            await service.query(manifest['graphId'], manifest['revision'], 'SELECT SUM(a.i*b.i) FROM range(1000000) a(i), range(1000000) b(i)', 'json')
        result, _ = await service.query(manifest['graphId'], manifest['revision'], 'SELECT COUNT(*) AS n FROM graph_nodes', 'json')
        assert json.loads(result) == [{'n': 4}]
    try:
        asyncio.run(run())
    finally:
        service.close()
