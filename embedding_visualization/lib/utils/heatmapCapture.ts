/** Capture a full DOM heatmap with its heading and legend, outside its scroll viewport. */
export class HeatmapCapture {
  private static readonly PIXEL_RATIO = 2;
  private static readonly PADDING = 24;
  private static readonly GAP = 16;

  private static ensureConnected(sections: readonly HTMLElement[]): void {
    if (sections.some(section => !section.isConnected)) {
      throw new DOMException('The heatmap was closed during export.', 'AbortError');
    }
  }

  static async toCanvas(
    sections: readonly HTMLElement[],
    backgroundColor: string,
  ): Promise<HTMLCanvasElement> {
    const { toCanvas, getFontEmbedCSS } = await import('html-to-image');
    HeatmapCapture.ensureConnected(sections);
    // Embed fonts once for all three sections, preserving the application's typeface.
    const fontEmbedCSS = await getFontEmbedCSS(sections[0]);
    HeatmapCapture.ensureConnected(sections);
    const images = await Promise.all(sections.map(section => toCanvas(section, {
      width: section.scrollWidth,
      height: section.scrollHeight,
      pixelRatio: HeatmapCapture.PIXEL_RATIO,
      fontEmbedCSS,
      filter: node => !(node instanceof HTMLElement && 'exportExclude' in node.dataset),
    })));
    HeatmapCapture.ensureConnected(sections);

    const padding = HeatmapCapture.PADDING * HeatmapCapture.PIXEL_RATIO;
    const gap = HeatmapCapture.GAP * HeatmapCapture.PIXEL_RATIO;
    const canvas = document.createElement('canvas');
    canvas.width = Math.max(...images.map(image => image.width)) + 2 * padding;
    canvas.height = images.reduce((height, image) => height + image.height, 0)
      + gap * (images.length - 1) + 2 * padding;
    const context = canvas.getContext('2d');
    if (!context) throw new Error('Could not create a canvas for the heatmap export.');

    context.fillStyle = backgroundColor;
    context.fillRect(0, 0, canvas.width, canvas.height);
    let y = padding;
    for (const image of images) {
      context.drawImage(image, padding, y);
      y += image.height + gap;
    }
    return canvas;
  }
}
