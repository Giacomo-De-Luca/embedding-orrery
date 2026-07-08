// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import * as cluster from "./pkg/density_clustering_wasm.js";

// The .wasm binary is served from public/ and fetched at runtime; passing an
// explicit URL keeps both bundlers (Turbopack + webpack) out of wasm handling
// entirely (the glue's `new URL(..., import.meta.url)` fallback never runs).
// Keep public/wasm/density_clustering_wasm_bg.wasm in sync with pkg/.
let initPromise: Promise<unknown> | null = null;

function ensureInit(): Promise<unknown> {
  initPromise ??= cluster.default("/wasm/density_clustering_wasm_bg.wasm");
  return initPromise;
}

/** A resulting cluster from the find clusters function */
export interface Cluster {
  /** Cluster identifier */
  identifier: number;
  /** The total density */
  sumDensity: number;
  /** The mean x location (weighted by density) */
  meanX: number;
  /** The mean y location (weighted by density) */
  meanY: number;
  /** The maximum density */
  maxDensity: number;
  /** The location with the maximum density */
  maxDensityLocation: [number, number];
  /** The number of pixels in the cluster */
  pixelCount: number;
  /** The cluster's boundary represented as a list of polygons */
  boundary?: [number, number][][];
  /** The cluster's boundary approximated with a list of rectangles, each rectangle is given as an array `[x1, y1, x2, y2]` */
  boundaryRectApproximation?: [number, number, number, number][];
}

/** Options of the find clusters function */
export interface FindClustersOptions {
  /** The threshold for unioning two clusters */
  unionThreshold?: number;
}

/**
 * Find clusters from a density map
 * @param densityMap the density map, a `Float32Array` with `width * height` elements
 * @param width the width of the density map
 * @param height the height of the density map
 * @param options algorithm options
 * @returns
 */
export async function findClusters(
  densityMap: Float32Array,
  width: number,
  height: number,
  options: Partial<FindClustersOptions> = {}
): Promise<Cluster[]> {
  await ensureInit();
  const t0 = new Date().getTime();
  const input = new cluster.DensityMap(width, height, densityMap);
  const result = cluster.find_clusters(input, {
    clustering_options: {
      use_disjoint_set: true,
      truncate_to_max_density: true,
      perform_neighbor_map_grouping: false,
      union_threshold: options.unionThreshold ?? 10,
      density_upperbound_scaler: 0.2,
      density_lowerbound_scaler: 0.2,
      ...options,
    },
    return_boundary_rects: true,
    smooth_boundaries: true,
  });
  input.free();
  let clusters: Cluster[] = [];
  for (const [id, summary] of result.summaries) {
    clusters.push({
      identifier: id,
      sumDensity: summary.sum_density,
      meanX: summary.sum_x_density / summary.sum_density,
      meanY: summary.sum_y_density / summary.sum_density,
      maxDensity: summary.max_density,
      maxDensityLocation: summary.max_density_location,
      pixelCount: summary.num_pixels,
      boundary: result.boundaries.get(id),
      boundaryRectApproximation: result.boundary_rects.get(id),
    });
  }
  clusters = clusters.filter((x) => x.boundary != null);
  const t1 = new Date().getTime();
  console.debug(`find clusters complete, time: ${(t1 - t0).toFixed(0)}ms, count: ${result.summaries.size}`);
  return clusters;
}
