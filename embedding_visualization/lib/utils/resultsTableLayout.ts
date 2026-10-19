/**
 * Layout of the search results table (SimilarItemsTable).
 *
 * Phones keep every cell on one line and scroll its overflow horizontally inside
 * the cell, so they get narrower columns: ID, Label and Similarity fit a 360px
 * phone side by side, and the rest is one horizontal swipe away.
 */
export type ResultsColumn = 'id' | 'label' | 'similarity' | 'category' | 'document' | 'metadata';

export interface ColumnWidth {
  size: number;
  minSize: number;
  maxSize: number;
}

/** Measured geometry (px) behind the header title's alignment over the Label column. */
export interface TitleAlignmentMeasure {
  tableLeft: number;
  tableBorder: number;
  firstColumnWidth: number;
  cellPaddingLeft: number;
  headerLeft: number;
  headerPaddingLeft: number;
  headerGap: number;
}

const DESKTOP_WIDTHS: Record<ResultsColumn, ColumnWidth> = {
  id: { size: 130, minSize: 80, maxSize: 150 },
  label: { size: 150, minSize: 100, maxSize: 250 },
  similarity: { size: 200, minSize: 180, maxSize: 250 },
  category: { size: 120, minSize: 80, maxSize: 150 },
  document: { size: 300, minSize: 200, maxSize: 500 },
  metadata: { size: 180, minSize: 150, maxSize: 350 },
};

const MOBILE_WIDTHS: Record<ResultsColumn, ColumnWidth> = {
  id: { size: 80, minSize: 60, maxSize: 150 },
  label: { size: 120, minSize: 80, maxSize: 250 },
  similarity: { size: 115, minSize: 110, maxSize: 250 },
  category: { size: 90, minSize: 60, maxSize: 150 },
  document: { size: 220, minSize: 120, maxSize: 500 },
  metadata: { size: 140, minSize: 80, maxSize: 350 },
};

export class ResultsTableLayout {
  static columnWidths(isMobile: boolean): Record<ResultsColumn, ColumnWidth> {
    return isMobile ? MOBILE_WIDTHS : DESKTOP_WIDTHS;
  }

  /**
   * Minimum width of the header title cell so the description starts over the
   * Label column's cell text. Every offset is measured rather than assumed, so
   * breakpoint-specific padding and gaps keep the alignment.
   */
  static titleColumnWidth(m: TitleAlignmentMeasure): number {
    const labelTextLeft = m.tableLeft + m.tableBorder + m.firstColumnWidth + m.cellPaddingLeft;
    const titleLeft = m.headerLeft + m.headerPaddingLeft;
    return Math.max(labelTextLeft - titleLeft - m.headerGap, 0);
  }
}
