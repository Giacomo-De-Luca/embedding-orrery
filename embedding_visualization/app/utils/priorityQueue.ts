// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

export class PriorityQueue {
  private heap: [number, number][];
  private deleted: Set<number>;

  constructor() {
    this.heap = [];
    this.deleted = new Set();
  }

  insert(key: number, priority: number) {
    let heap = this.heap;
    let i = heap.length;
    while (i > 0) {
      let j = (i - 1) >> 1;
      if (priority <= heap[j][0]) {
        break;
      }
      heap[i] = heap[j];
      i = j;
    }
    heap[i] = [priority, key];
  }

  delete(key: number) {
    this.deleted.add(key);
  }

  _popMax(): number | null {
    let heap = this.heap;
    if (heap.length == 0) {
      return null;
    }
    let result = heap[0][1];
    let value = heap.pop()!;
    if (heap.length == 0) {
      return result;
    }
    let priority = value[0];
    let i = 0;
    while (true) {
      let j = i * 2 + 1;
      if (j + 1 < heap.length && heap[j][0] < heap[j + 1][0]) {
        j++;
      }
      if (j >= heap.length || priority >= heap[j][0]) {
        break;
      }
      heap[i] = heap[j];
      i = j;
    }
    heap[i] = value;
    return result;
  }

  popMax(): number | null {
    while (true) {
      let k = this._popMax();
      if (k == null) {
        return null;
      }
      if (!this.deleted.has(k)) {
        return k;
      }
      this.deleted.delete(k);
    }
  }
}
