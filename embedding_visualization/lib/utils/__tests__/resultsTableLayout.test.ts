import { describe, expect, it } from 'vitest';
import { ResultsTableLayout, type ResultsColumn } from '../resultsTableLayout';

const COLUMNS: ResultsColumn[] = ['id', 'label', 'similarity', 'category', 'document', 'metadata'];

describe('ResultsTableLayout.columnWidths', () => {
  it.each([false, true])('keeps every size within its resize bounds (mobile=%s)', (isMobile) => {
    // tanstack clamps getSize() to [minSize, maxSize], so a size outside its own
    // bounds would silently render at the bound instead.
    const widths = ResultsTableLayout.columnWidths(isMobile);
    for (const column of COLUMNS) {
      const { size, minSize, maxSize } = widths[column];
      expect(minSize).toBeLessThanOrEqual(size);
      expect(size).toBeLessThanOrEqual(maxSize);
    }
  });

  it('never widens a column on mobile', () => {
    const desktop = ResultsTableLayout.columnWidths(false);
    const mobile = ResultsTableLayout.columnWidths(true);
    for (const column of COLUMNS) {
      expect(mobile[column].size).toBeLessThanOrEqual(desktop[column].size);
    }
  });

  it('fits ID, Label and Similarity side by side on a 360px phone', () => {
    // 360 minus DashboardPanel px-2 (16), panel border (2), CardContent px-2 (16)
    // and table wrapper border (2) leaves 324px for the table.
    const { id, label, similarity } = ResultsTableLayout.columnWidths(true);
    expect(id.size + label.size + similarity.size).toBeLessThanOrEqual(324);
  });
});

describe('ResultsTableLayout.titleColumnWidth', () => {
  const desktop = {
    tableLeft: 100,
    tableBorder: 1,
    firstColumnWidth: 130,
    cellPaddingLeft: 8,
    headerLeft: 76,
    headerPaddingLeft: 32,
    headerGap: 16,
  };

  it('matches the former hard-coded desktop formula', () => {
    // Previously: max(tableLeft + 1 + firstCol + 8 - (headerLeft + 32) - 16, 0)
    const legacy = Math.max(100 + 1 + 130 + 8 - (76 + 32) - 16, 0);
    expect(ResultsTableLayout.titleColumnWidth(desktop)).toBe(legacy);
  });

  it('follows measured padding and gap, so responsive spacing stays aligned', () => {
    const mobile = { ...desktop, tableLeft: 92, firstColumnWidth: 90, headerPaddingLeft: 16, headerGap: 8 };
    // Label text sits at 92 + 1 + 90 + 8 = 191; title starts at 76 + 16 = 92; minus the gap.
    expect(ResultsTableLayout.titleColumnWidth(mobile)).toBe(191 - 92 - 8);
  });

  it('clamps to zero when the header starts right of the label text', () => {
    expect(ResultsTableLayout.titleColumnWidth({ ...desktop, headerLeft: 400 })).toBe(0);
  });
});
