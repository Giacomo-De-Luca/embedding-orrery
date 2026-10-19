'use client';

import { useCallback, useState } from 'react';
import { toast } from 'sonner';
import { HeatmapCapture } from '../utils/heatmapCapture';
import { downloadCanvasPng } from '../utils/downloadJson';

/** First non-transparent background at or above `element` (the dialog surface), white as a last resort. */
function opaqueBackground(element: HTMLElement): string {
  for (let node: HTMLElement | null = element; node; node = node.parentElement) {
    const color = getComputedStyle(node).backgroundColor;
    if (color && color !== 'transparent' && !/rgba\(.*,\s*0\)$/.test(color)) return color;
  }
  return '#ffffff';
}

/**
 * PNG export of a heatmap: its heading, legend and full grid stacked on the
 * surrounding surface colour (see `HeatmapCapture`). `exporting` disables the
 * controls meanwhile; closing the dialog mid-export cancels quietly.
 */
export function useHeatmapExport() {
  const [exporting, setExporting] = useState(false);

  const exportPng = useCallback(async (sections: readonly (HTMLElement | null)[], filename: string) => {
    if (exporting || sections.some(section => !section)) return;
    const elements = sections as HTMLElement[];
    setExporting(true);
    try {
      const canvas = await HeatmapCapture.toCanvas(elements, opaqueBackground(elements[0]));
      const safeName = filename.replace(/[<>:"/\\|?*\u0000-\u001f]/g, '_');
      if (!await downloadCanvasPng(canvas, safeName)) throw new Error('PNG encoding failed.');
    } catch (error) {
      if (error instanceof DOMException && error.name === 'AbortError') return;
      console.error('Heatmap PNG export failed:', error);
      toast.error('Could not export the heatmap. Please try again.');
    } finally {
      setExporting(false);
    }
  }, [exporting]);

  return { exporting, exportPng };
}
