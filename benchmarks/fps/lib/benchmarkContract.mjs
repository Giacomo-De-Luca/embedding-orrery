import { createHash } from 'node:crypto';

const PROJECTION_FIELDS = {
  pca_2d: 'pca2d',
  pca_3d: 'pca3d',
  umap_2d: 'umap2d',
  umap_3d: 'umap3d',
};

/** Builds memory-conscious synthetic GraphQL responses matching the live schema. */
export class SyntheticCollectionPayload {
  constructor({ pointCount, clusters, projectionType, includeCore }) {
    if (!Number.isInteger(pointCount) || pointCount < 0) {
      throw new Error('pointCount must be a non-negative integer');
    }
    if (!Number.isInteger(clusters) || clusters < 1) {
      throw new Error('clusters must be a positive integer');
    }
    if (!(projectionType in PROJECTION_FIELDS)) {
      throw new Error(`Unsupported projection type: ${projectionType}`);
    }

    this.pointCount = pointCount;
    this.clusters = clusters;
    this.projectionType = projectionType;
    this.includeCore = includeCore;
  }

  static #random(seed) {
    return function () {
      let value = (seed += 0x6d2b79f5);
      value = Math.imul(value ^ (value >>> 15), value | 1);
      value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
      return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
    };
  }

  static #updateSignature(digest, itemId) {
    const encoded = Buffer.from(itemId, 'utf8');
    const length = Buffer.allocUnsafe(8);
    length.writeBigUInt64BE(BigInt(encoded.length));
    digest.update(length);
    digest.update(encoded);
  }

  build() {
    const random = SyntheticCollectionPayload.#random(42);
    const gaussian = () => {
      let u = 0;
      let v = 0;
      while (u === 0) u = random();
      while (v === 0) v = random();
      return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    };
    const centers = Array.from({ length: this.clusters }, () => [
      random() * 14 - 7,
      random() * 14 - 7,
      random() * 14 - 7,
    ]);
    const dimensions = this.projectionType.endsWith('_3d') ? 3 : 2;
    const ids = this.includeCore ? new Array(this.pointCount) : null;
    const documents = this.includeCore ? new Array(this.pointCount) : null;
    const metadata = this.includeCore ? new Array(this.pointCount) : null;
    const coordinates = new Array(this.pointCount);
    const digest = createHash('sha256');

    for (let index = 0; index < this.pointCount; index++) {
      const cluster = index % this.clusters;
      const [centerX, centerY, centerZ] = centers[cluster];
      const x = (centerX + gaussian() * 0.9).toFixed(3);
      const y = (centerY + gaussian() * 0.9).toFixed(3);
      const z = (centerZ + gaussian() * 0.9).toFixed(3);
      const itemId = `s${index}`;
      SyntheticCollectionPayload.#updateSignature(digest, itemId);

      if (this.includeCore) {
        ids[index] = JSON.stringify(itemId);
        documents[index] = JSON.stringify(`Synthetic point ${index} (cluster ${cluster})`);
        metadata[index] = `{"topic_id":${cluster},"topic_label":"Cluster ${cluster < 10 ? `0${cluster}` : cluster}"}`;
      }
      coordinates[index] = dimensions === 3 ? `[${x},${y},${z}]` : `[${x},${y}]`;
    }

    const itemSignature = digest.digest('hex');
    const projectionFields = {
      pca2d: 'null',
      pca3d: 'null',
      umap2d: 'null',
      umap3d: 'null',
    };
    projectionFields[PROJECTION_FIELDS[this.projectionType]] = `[${coordinates.join(',')}]`;
    const collectionMetadata = JSON.stringify({
      totalItems: this.pointCount,
      embeddingDim: 384,
      timestamp: 'synthetic',
      pca2dVariance: null,
      pca3dVariance: null,
      sourceDataset: 'synthetic-benchmark',
      sourceSplit: null,
      sourceFile: null,
      hasProjections: true,
      embeddingProvider: 'synthetic',
      embeddingModel: 'synthetic',
      embeddingPrompt: null,
      fieldAnalysis: null,
      saeModelId: null,
      saeId: null,
    });
    const coreIds = this.includeCore ? `[${ids.join(',')}]` : '[]';
    const coreDocuments = this.includeCore ? `[${documents.join(',')}]` : '[]';
    const coreMetadata = this.includeCore ? `[${metadata.join(',')}]` : '[]';
    const availableFields = this.includeCore ? '["topic_id","topic_label"]' : '[]';
    const coreSignature = this.includeCore ? JSON.stringify(itemSignature) : 'null';
    const projectionSignatures = JSON.stringify({
      [this.projectionType]: itemSignature,
    });

    return `{"data":{"collection":{` +
      `"ids":${coreIds},"documents":${coreDocuments},` +
      `"itemMetadata":${coreMetadata},"availableFields":${availableFields},` +
      `"itemSignature":${coreSignature},` +
      `"projectionSignatures":${projectionSignatures},` +
      `"pca2d":${projectionFields.pca2d},"pca3d":${projectionFields.pca3d},` +
      `"umap2d":${projectionFields.umap2d},"umap3d":${projectionFields.umap3d},` +
      `"metadata":${collectionMetadata}}}}`;
  }
}

/** Keeps new benchmark runs separate from historical baseline JSON files. */
export class BenchmarkResultFile {
  static path(resultsDirectory, pass, label = 'current') {
    const directory = resultsDirectory.replace(/\/$/, '');
    const safePass = this.#safeSegment(pass, 'pass');
    const safeLabel = this.#safeSegment(label, 'current');
    return `${directory}/results_${safePass}_${safeLabel}.json`;
  }

  static artifactPath(resultsDirectory, pass, label, collectionName) {
    const directory = resultsDirectory.replace(/\/$/, '');
    const safePass = this.#safeSegment(pass, 'pass');
    const safeLabel = this.#safeSegment(label, 'current');
    const safeCollection = this.#safeSegment(collectionName, 'collection');
    return `${directory}/shot_${safePass}_${safeLabel}_${safeCollection}.png`;
  }

  static #safeSegment(value, fallback) {
    const safe = String(value ?? '')
      .trim()
      .toLowerCase()
      .replace(/[^a-z0-9_-]+/g, '-')
      .replace(/^-+|-+$/g, '');
    return safe || fallback;
  }
}

/** Captures collection-query transfer sizes alongside heap measurements. */
export class GraphqlPayloadTelemetry {
  constructor() {
    this.requests = [];
  }

  record({ variables, responseBytes }) {
    this.requests.push({
      projectionTypes: [...(variables.projectionTypes ?? [])],
      includeCore: variables.includeCore !== false,
      responseBytes,
      responseMB: GraphqlPayloadTelemetry.#megabytes(responseBytes),
    });
  }

  snapshot() {
    const totalResponseBytes = this.requests.reduce(
      (total, request) => total + request.responseBytes,
      0,
    );
    return {
      totalResponseBytes,
      totalResponseMB: GraphqlPayloadTelemetry.#megabytes(totalResponseBytes),
      requests: this.requests.map((request) => ({ ...request })),
    };
  }

  static #megabytes(bytes) {
    return +(bytes / 1048576).toFixed(2);
  }
}
