'use client';

import { Download, Hash, Loader2 } from 'lucide-react';
import { Button } from '@/lib/ui-primitives/button';
import { Toggle } from '@/lib/ui-primitives/toggle';

interface HeatmapActionsProps {
  showValues: boolean;
  onShowValuesChange: (show: boolean) => void;
  exporting: boolean;
  onExport: () => void;
  /** Nothing to export yet (empty table, no matrix). */
  exportDisabled: boolean;
}

/** The Values toggle and Export PNG button shared by the heatmap tabs. */
export function HeatmapActions({
  showValues,
  onShowValuesChange,
  exporting,
  onExport,
  exportDisabled,
}: HeatmapActionsProps) {
  return (
    <>
      <Toggle
        variant="outline"
        size="sm"
        className="h-7 text-xs"
        pressed={showValues}
        disabled={exporting}
        onPressedChange={onShowValuesChange}
        aria-label="Show values in cells"
      >
        <Hash /> Values
      </Toggle>
      <Button
        variant="outline"
        size="sm"
        className="h-7 text-xs"
        onClick={onExport}
        disabled={exporting || exportDisabled}
        aria-busy={exporting}
      >
        {exporting ? <Loader2 className="animate-spin" /> : <Download />}
        {exporting ? 'Exporting…' : 'Export PNG'}
      </Button>
    </>
  );
}
