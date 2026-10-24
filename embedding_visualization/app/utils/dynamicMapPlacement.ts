// Copyright (c) 2025 Apple Inc. Licensed under MIT License.
// Adapted from Embedding Atlas; license notice in THIRD_PARTY_NOTICES.md (frontend root).
// Modified for speed, with the same placements as the original (pinned by
// __tests__/dynamicMapPlacement.test.ts against an unmodified copy):
// - Label pairs are built only where they can touch below the global max scale,
//   found through a grid over the anchors. The original built all n² pairs, though
//   the sweep never uses a pair that only touches above that scale.
// - The level sweep visits only the labels whose state changes at a level (a
//   conflict ending, their scale range starting or ending) instead of all labels at
//   every level, and a label that stops being visible updates only its neighbours.

// The algorithm is based on the following research paper:
// Been, Ken, Eli Daiches, and Chee Yap. "Dynamic Map Labeling." IEEE Transactions on Visualization and Computer Graphics 12, no. 5 (2006): 773–80. https://doi.org/10.1109/TVCG.2006.136.

import { PriorityQueue } from "./priorityQueue";

interface Point {
  x: number;
  y: number;
}

/** A rectangle with min, max coordinate for each dimension.
 * It is required that xMin <= xMax and yMin <= yMax. */
interface Rectangle {
  xMin: number;
  yMin: number;
  xMax: number;
  yMax: number;
}

/** Description of a label for placement. */
export interface Label {
  /** The bounds of the label at scale = 1. */
  bounds: Rectangle;
  /** The location of the label at scale = 0. Usually you can set this to the center of `bounds`. */
  locationAtZero: Point;
  /** The minimum scale this label can appear in. */
  minScale?: number;
  /** The maximum scale this label can appear in. */
  maxScale?: number;
  /** The label's priority. */
  priority?: number;
}

/** The placement of a label. */
export interface Placement {
  /** The minimum scale. */
  minScale: number;
  /** The maximum scale. */
  maxScale: number;
}

export interface Options {
  /** The global max scale for all labels. */
  globalMaxScale: number;
}

/** Touch scales are discretized to steps of 1/100 in log space, so there aren't too many levels. */
const LOG_STEPS = 100;
/**
 * Discretizing lowers a scale by less than one step (a factor e^(1/100)), so a pair
 * whose raw touch scale is at least the max scale times this factor never touches
 * below the max scale. Kept a little wider than one step to stay clear of rounding.
 */
const DISCRETIZATION_SLACK = Math.exp(1.5 / LOG_STEPS);

/** Label geometry as flat arrays, read in the pair loop. */
class LabelGeometry {
  readonly x: Float64Array;
  readonly y: Float64Array;
  readonly xMin: Float64Array;
  readonly xMax: Float64Array;
  readonly yMin: Float64Array;
  readonly yMax: Float64Array;

  constructor(labels: Label[]) {
    const n = labels.length;
    this.x = new Float64Array(n);
    this.y = new Float64Array(n);
    this.xMin = new Float64Array(n);
    this.xMax = new Float64Array(n);
    this.yMin = new Float64Array(n);
    this.yMax = new Float64Array(n);
    labels.forEach((label, i) => {
      this.x[i] = label.locationAtZero.x;
      this.y[i] = label.locationAtZero.y;
      this.xMin[i] = label.bounds.xMin;
      this.xMax[i] = label.bounds.xMax;
      this.yMin[i] = label.bounds.yMin;
      this.yMax[i] = label.bounds.yMax;
    });
  }

  /**
   * The discretized scale at which two labels touch (overlapping above it, apart
   * below), as the original computes it; Infinity when they never touch below
   * `rejectAbove` (raw scale), when they never touch at all, or when the scale is
   * undefined (boxes with no area).
   */
  touchScale(i: number, j: number, rejectAbove: number): number {
    const xi = this.x[i], yi = this.y[i], xj = this.x[j], yj = this.y[j];
    if (xi == xj && yi == yj) {
      const overlap = this.xMin[i] < this.xMax[j] && this.xMax[i] > this.xMin[j]
        && this.yMin[i] < this.yMax[j] && this.yMax[i] > this.yMin[j];
      return overlap ? 0 : Infinity;
    }
    const k1 = xi < xj ? (xi - xj) / (this.xMin[j] - this.xMax[i] + xi - xj) : (xi - xj) / (this.xMax[j] - this.xMin[i] + xi - xj);
    const k2 = yi < yj ? (yi - yj) / (this.yMin[j] - this.yMax[i] + yi - yj) : (yi - yj) / (this.yMax[j] - this.yMin[i] + yi - yj);
    const scale = Math.max(k1, k2);
    if (!(scale > 0) || scale >= rejectAbove) return Infinity;
    return Math.exp(Math.floor(Math.log(scale) * LOG_STEPS) / LOG_STEPS);
  }

  /** Whether every box contains its anchor, which the grid's reach bound assumes. */
  boxesContainAnchors(): boolean {
    for (let i = 0; i < this.x.length; i++) {
      if (!(this.xMin[i] <= this.x[i] && this.x[i] <= this.xMax[i] && this.yMin[i] <= this.y[i] && this.y[i] <= this.yMax[i])) return false;
    }
    return true;
  }
}

/** The labels each label conflicts with below the max scale, and the scales they touch at. */
class ConflictGraph {
  readonly neighbors: number[][];
  readonly scales: number[][];

  constructor(geometry: LabelGeometry, maxScale: number) {
    const n = geometry.x.length;
    this.neighbors = Array.from({ length: n }, () => []);
    this.scales = Array.from({ length: n }, () => []);
    const rejectAbove = maxScale * DISCRETIZATION_SLACK;
    const addPair = (i: number, j: number) => {
      const scale = geometry.touchScale(i, j, rejectAbove);
      if (!(scale < maxScale)) return;
      this.neighbors[i].push(j);
      this.scales[i].push(scale);
      this.neighbors[j].push(i);
      this.scales[j].push(scale);
    };
    if (!this.addNearbyPairs(geometry, rejectAbove, addPair)) {
      for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) addPair(i, j);
    }
  }

  /**
   * Visits only pairs whose anchors are close enough to touch below the max scale:
   * along an axis, two boxes touch at |distance| / (the extents facing each other),
   * so anchors further apart than `rejectAbove` × (the widest facing extents) never
   * do. Buckets anchors into a grid of cells that size and pairs each label with the
   * labels in its own and the 8 neighbouring cells. False when the bound does not
   * hold (a box not containing its anchor), so the caller visits every pair.
   */
  private addNearbyPairs(geometry: LabelGeometry, rejectAbove: number, addPair: (i: number, j: number) => void): boolean {
    const n = geometry.x.length;
    if (n < 2) return true;
    if (!geometry.boxesContainAnchors()) return false;

    let left = 0, right = 0, top = 0, bottom = 0;
    let minX = Infinity, minY = Infinity, maxX = -Infinity;
    for (let i = 0; i < n; i++) {
      left = Math.max(left, geometry.x[i] - geometry.xMin[i]);
      right = Math.max(right, geometry.xMax[i] - geometry.x[i]);
      top = Math.max(top, geometry.y[i] - geometry.yMin[i]);
      bottom = Math.max(bottom, geometry.yMax[i] - geometry.y[i]);
      minX = Math.min(minX, geometry.x[i]);
      minY = Math.min(minY, geometry.y[i]);
      maxX = Math.max(maxX, geometry.x[i]);
    }
    const reachX = rejectAbove * (left + right);
    const reachY = rejectAbove * (top + bottom);
    if (!(reachX > 0 && reachY > 0 && Number.isFinite(reachX) && Number.isFinite(reachY) && Number.isFinite(maxX - minX))) return false;

    const columns = Math.floor((maxX - minX) / reachX) + 1;
    const cellX = new Int32Array(n);
    const cellY = new Int32Array(n);
    const cells = new Map<number, number[]>();
    for (let i = 0; i < n; i++) {
      cellX[i] = Math.floor((geometry.x[i] - minX) / reachX);
      cellY[i] = Math.floor((geometry.y[i] - minY) / reachY);
      const key = cellY[i] * columns + cellX[i];
      const cell = cells.get(key);
      if (cell) cell.push(i);
      else cells.set(key, [i]);
    }
    for (let i = 0; i < n; i++) {
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const cx = cellX[i] + dx;
          if (cx < 0 || cx >= columns) continue;
          const cell = cells.get((cellY[i] + dy) * columns + cx);
          if (!cell) continue;
          for (const j of cell) if (j > i) addPair(i, j);
        }
      }
    }
    return true;
  }
}

/** A min-heap of label indices: the order a level's pass visits labels in. */
class IndexHeap {
  private readonly items: number[] = [];

  get size(): number {
    return this.items.length;
  }

  push(value: number): void {
    const items = this.items;
    let i = items.length;
    items.push(value);
    while (i > 0) {
      const parent = (i - 1) >> 1;
      if (items[parent] <= value) break;
      items[i] = items[parent];
      i = parent;
    }
    items[i] = value;
  }

  pop(): number {
    const items = this.items;
    const top = items[0];
    const last = items.pop()!;
    if (items.length > 0) {
      let i = 0;
      while (true) {
        let child = i * 2 + 1;
        if (child >= items.length) break;
        if (child + 1 < items.length && items[child + 1] < items[child]) child++;
        if (items[child] >= last) break;
        items[i] = items[child];
        i = child;
      }
      items[i] = last;
    }
    return top;
  }
}

export function dynamicLabelPlacement(labels: Label[], options: Partial<Options> = {}): (Placement | null)[] {
  const globalMaxScale = options.globalMaxScale ?? 1;
  const n = labels.length;
  const graph = new ConflictGraph(new LabelGeometry(labels), globalMaxScale);
  const ranges: [number, number][] = labels.map((label) => [
    Math.min(globalMaxScale, label.minScale ?? 0),
    Math.min(globalMaxScale, label.maxScale ?? globalMaxScale),
  ]);
  const priorities = labels.map((label) => label.priority ?? 0);

  // Every level where something can change, from the largest down. Levels above
  // the max scale change nothing: no label's range reaches them.
  const levelSet = new Set<number>([0]);
  for (const scales of graph.scales) for (const scale of scales) levelSet.add(scale);
  for (const [min, max] of ranges) {
    levelSet.add(min);
    levelSet.add(max);
  }
  const levels = Array.from(levelSet).sort((a, b) => b - a);
  const levelIndex = new Map(levels.map((level, index) => [level, index]));

  // What happens at each level: the conflicts that end there (a pair stops
  // overlapping at levels at or below its touch scale), and the labels whose range
  // starts or ends there.
  const conflictsEnding: number[][] = levels.map(() => []);
  graph.neighbors.forEach((neighbors, i) => {
    neighbors.forEach((j, k) => conflictsEnding[levelIndex.get(graph.scales[i][k])!].push(i, j));
  });
  const rangeEntered: number[][] = levels.map(() => []);
  const rangeLeft: number[][] = levels.map(() => []);
  ranges.forEach(([min, max], i) => {
    rangeEntered[levelIndex.get(max)!].push(i);
    rangeLeft[levelIndex.get(min)!].push(i);
  });

  const placements: (Placement | null)[] = labels.map(() => null);
  const queue = new PriorityQueue();
  const numConflicts: Set<number>[] = Array.from({ length: n }, () => new Set());
  const inQueue = new Uint8Array(n);
  const isVisible = new Uint8Array(n);
  const visitedAt = new Int32Array(n).fill(-1);
  let carried: number[] = [];

  levels.forEach((level, li) => {
    // A label can only become eligible when a conflict of it ends, when its range
    // starts, or when a visible neighbour leaves its range; a visible label whose
    // range starts here stops being visible. The original checked every label at
    // every level, in index order, and a label leaving updated the conflicts of
    // labels it had not checked yet in this pass: visiting these labels in index
    // order, and carrying those it updates below its index to the next level,
    // reproduces it.
    const pass = new IndexHeap();
    const ending = conflictsEnding[li];
    for (let k = 0; k < ending.length; k += 2) {
      const i = ending[k];
      if (numConflicts[i].delete(ending[k + 1]) && numConflicts[i].size === 0) pass.push(i);
    }
    for (const i of rangeEntered[li]) pass.push(i);
    for (const i of rangeLeft[li]) pass.push(i);
    for (const i of carried) pass.push(i);
    carried = [];

    while (pass.size > 0) {
      const i = pass.pop();
      if (visitedAt[i] === li) continue;
      visitedAt[i] = li;
      if (ranges[i][0] < level && level <= ranges[i][1] && numConflicts[i].size == 0 && !inQueue[i]) {
        inQueue[i] = 1;
        queue.insert(i, priorities[i]);
      }
      if (isVisible[i] && placements[i]!.minScale >= level) {
        isVisible[i] = 0;
        for (const j of graph.neighbors[i]) {
          if (numConflicts[j].delete(i) && numConflicts[j].size === 0) {
            if (j > i) pass.push(j);
            else carried.push(j);
          }
        }
      }
    }

    while (true) {
      const i = queue.popMax();
      if (i == null) {
        break;
      }
      placements[i] = { minScale: ranges[i][0], maxScale: level };
      isVisible[i] = 1;
      // Pairs touching below this level overlap at it
      const neighbors = graph.neighbors[i];
      const scales = graph.scales[i];
      for (let c = 0; c < neighbors.length; c++) {
        if (scales[c] >= level) continue;
        const j = neighbors[c];
        numConflicts[j].add(i);
        if (inQueue[j]) {
          queue.delete(j);
          inQueue[j] = 0;
        }
      }
    }
  });
  return placements;
}
