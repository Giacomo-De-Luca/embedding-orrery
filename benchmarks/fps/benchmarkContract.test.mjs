import assert from 'node:assert/strict';
import test from 'node:test';

import {
  BenchmarkResultFile,
  GraphqlPayloadTelemetry,
  SyntheticCollectionPayload,
} from './lib/benchmarkContract.mjs';

test('synthetic 3D core payload contains only the active projection and matching signatures', () => {
  const payload = JSON.parse(new SyntheticCollectionPayload({
    pointCount: 2,
    clusters: 2,
    projectionType: 'umap_3d',
    includeCore: true,
  }).build());
  const collection = payload.data.collection;

  assert.equal(collection.umap2d, null);
  assert.equal(collection.umap3d.length, 2);
  assert.equal(collection.pca2d, null);
  assert.equal(collection.pca3d, null);
  assert.deepEqual(collection.ids, ['s0', 's1']);
  assert.equal(
    collection.itemSignature,
    '7d5b780074ece4bea0d5101770aaf21e9b13902353d591051d387cd0043b5784',
  );
  assert.deepEqual(collection.projectionSignatures, {
    umap_3d: collection.itemSignature,
  });
});

test('synthetic projection-only payload omits core arrays and inactive dimensions', () => {
  const payload = JSON.parse(new SyntheticCollectionPayload({
    pointCount: 3,
    clusters: 2,
    projectionType: 'umap_2d',
    includeCore: false,
  }).build());
  const collection = payload.data.collection;

  assert.deepEqual(collection.ids, []);
  assert.deepEqual(collection.documents, []);
  assert.deepEqual(collection.itemMetadata, []);
  assert.deepEqual(collection.availableFields, []);
  assert.equal(collection.itemSignature, null);
  assert.equal(collection.umap2d.length, 3);
  assert.equal(collection.umap3d, null);
  assert.match(collection.projectionSignatures.umap_2d, /^[a-f0-9]{64}$/);
});

test('synthetic payload rejects unsupported projection names', () => {
  assert.throws(
    () => new SyntheticCollectionPayload({
      pointCount: 1,
      clusters: 1,
      projectionType: 'tsne_3d',
      includeCore: true,
    }),
    /Unsupported projection type/,
  );
});

test('benchmark results use a labeled file and preserve the historical baseline', () => {
  assert.equal(
    BenchmarkResultFile.path('/tmp/results', '3d', 'heap reduction after'),
    '/tmp/results/results_3d_heap-reduction-after.json',
  );
  assert.equal(
    BenchmarkResultFile.path('/tmp/results/', '2d'),
    '/tmp/results/results_2d_current.json',
  );
  assert.notEqual(
    BenchmarkResultFile.path('/tmp/results/', '3d'),
    '/tmp/results/results_3d.json',
  );
  assert.equal(
    BenchmarkResultFile.artifactPath(
      '/tmp/results/',
      '3d',
      'heap reduction after',
      'wordnet senses',
    ),
    '/tmp/results/shot_3d_heap-reduction-after_wordnet-senses.png',
  );
});

test('GraphQL telemetry records core and projection-only response bytes', () => {
  const telemetry = new GraphqlPayloadTelemetry();
  telemetry.record({
    variables: {
      projectionTypes: ['umap_3d'],
      includeCore: true,
    },
    responseBytes: 2 * 1024 * 1024,
  });
  telemetry.record({
    variables: {
      projectionTypes: ['umap_2d'],
      includeCore: false,
    },
    responseBytes: 512 * 1024,
  });

  assert.deepEqual(telemetry.snapshot(), {
    totalResponseBytes: 2621440,
    totalResponseMB: 2.5,
    requests: [
      {
        projectionTypes: ['umap_3d'],
        includeCore: true,
        responseBytes: 2097152,
        responseMB: 2,
      },
      {
        projectionTypes: ['umap_2d'],
        includeCore: false,
        responseBytes: 524288,
        responseMB: 0.5,
      },
    ],
  });
});

test('pass-through requests are measured later by re-sending them, outside the browser', async () => {
  const telemetry = new GraphqlPayloadTelemetry();
  const postData = JSON.stringify({ operationName: 'GetCollectionData', variables: { name: 'wordnet' } });
  telemetry.recordPassThrough({
    variables: { projectionTypes: ['umap_3d'], includeCore: true },
    url: 'http://localhost:8000/graphql',
    postData,
  });
  // Nothing is measured until the run asks for it.
  assert.equal(telemetry.snapshot().requests.length, 0);

  const calls = [];
  const fakeFetch = async (url, init) => {
    calls.push({ url, init });
    return { ok: true, arrayBuffer: async () => new ArrayBuffer(3 * 1024 * 1024) };
  };
  await telemetry.measurePending(fakeFetch);

  assert.deepEqual(calls, [{
    url: 'http://localhost:8000/graphql',
    init: { method: 'POST', headers: { 'content-type': 'application/json' }, body: postData },
  }]);
  assert.deepEqual(telemetry.snapshot(), {
    totalResponseBytes: 3145728,
    totalResponseMB: 3,
    requests: [{ projectionTypes: ['umap_3d'], includeCore: true, responseBytes: 3145728, responseMB: 3 }],
  });

  // Measuring again does not re-send or double-count.
  await telemetry.measurePending(fakeFetch);
  assert.equal(calls.length, 1);
  assert.equal(telemetry.snapshot().requests.length, 1);
});

test('a failed re-send is recorded as unmeasured instead of being dropped', async () => {
  const telemetry = new GraphqlPayloadTelemetry();
  telemetry.recordPassThrough({
    variables: { projectionTypes: ['umap_2d'], includeCore: false },
    url: 'http://localhost:8000/graphql',
    postData: '{}',
  });
  await telemetry.measurePending(async () => { throw new Error('ECONNREFUSED'); });

  assert.deepEqual(telemetry.snapshot(), {
    totalResponseBytes: 0,
    totalResponseMB: 0,
    requests: [{
      projectionTypes: ['umap_2d'],
      includeCore: false,
      responseBytes: null,
      responseMB: null,
      measureError: 'Error: ECONNREFUSED',
    }],
  });
});

test('a non-OK re-send response is recorded as unmeasured', async () => {
  const telemetry = new GraphqlPayloadTelemetry();
  telemetry.recordPassThrough({
    variables: { projectionTypes: ['umap_3d'] },
    url: 'http://localhost:8000/graphql',
    postData: '{}',
  });
  await telemetry.measurePending(async () => ({ ok: false, status: 502, arrayBuffer: async () => new ArrayBuffer(10) }));
  assert.equal(telemetry.snapshot().requests[0].responseBytes, null);
  assert.equal(telemetry.snapshot().requests[0].measureError, 'HTTP 502');
});
