/**
 * Spatial grid for fast label collision detection.
 * Ported from TensorBoard's embedding projector (Apache 2.0).
 */

export interface BoundingBox {
  loX: number;
  loY: number;
  hiX: number;
  hiY: number;
}

/**
 * Accelerates label placement by dividing the view into a uniform grid.
 * Labels only need to be tested for collision with other labels that overlap
 * the same grid cells.
 */
export class CollisionGrid {
  private numHorizCells: number;
  private numVertCells: number;
  private grid: (BoundingBox[] | undefined)[];
  private bound: BoundingBox;
  private cellWidth: number;
  private cellHeight: number;

  constructor(bound: BoundingBox, cellWidth: number, cellHeight: number) {
    this.bound = bound;
    this.cellWidth = cellWidth;
    this.cellHeight = cellHeight;
    this.numHorizCells = Math.ceil((bound.hiX - bound.loX) / cellWidth);
    this.numVertCells = Math.ceil((bound.hiY - bound.loY) / cellHeight);
    this.grid = new Array(this.numHorizCells * this.numVertCells);
  }

  /**
   * Checks if a bounding box can be placed without collisions.
   * If `justTest` is false (default), inserts it on success.
   * Returns true if no collision was found.
   */
  insert(bound: BoundingBox, justTest = false): boolean {
    // Reject out-of-bounds labels
    if (
      bound.hiX < this.bound.loX ||
      bound.loX > this.bound.hiX ||
      bound.hiY < this.bound.loY ||
      bound.loY > this.bound.hiY
    ) {
      return false;
    }

    const minCellX = this.getCellX(bound.loX);
    const maxCellX = this.getCellX(bound.hiX);
    const minCellY = this.getCellY(bound.loY);
    const maxCellY = this.getCellY(bound.hiY);

    // Check all overlapped cells for conflicts
    const baseIdx = minCellY * this.numHorizCells + minCellX;
    let idx = baseIdx;
    for (let j = minCellY; j <= maxCellY; j++) {
      for (let i = minCellX; i <= maxCellX; i++) {
        const cell = this.grid[idx++];
        if (cell) {
          for (let k = 0; k < cell.length; k++) {
            if (this.boundsIntersect(bound, cell[k])) {
              return false;
            }
          }
        }
      }
      idx += this.numHorizCells - (maxCellX - minCellX + 1);
    }

    if (justTest) return true;

    // Insert into overlapped cells
    idx = baseIdx;
    for (let j = minCellY; j <= maxCellY; j++) {
      for (let i = minCellX; i <= maxCellX; i++) {
        if (!this.grid[idx]) {
          this.grid[idx] = [bound];
        } else {
          this.grid[idx]!.push(bound);
        }
        idx++;
      }
      idx += this.numHorizCells - (maxCellX - minCellX + 1);
    }
    return true;
  }

  private boundsIntersect(a: BoundingBox, b: BoundingBox): boolean {
    return !(a.loX > b.hiX || a.loY > b.hiY || a.hiX < b.loX || a.hiY < b.loY);
  }

  private getCellX(x: number): number {
    return Math.max(0, Math.min(
      this.numHorizCells - 1,
      Math.floor((x - this.bound.loX) / this.cellWidth),
    ));
  }

  private getCellY(y: number): number {
    return Math.max(0, Math.min(
      this.numVertCells - 1,
      Math.floor((y - this.bound.loY) / this.cellHeight),
    ));
  }
}
