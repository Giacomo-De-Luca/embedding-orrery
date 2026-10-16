import { beforeEach, describe, expect, it, vi } from 'vitest';
import { HeatmapCapture } from '../heatmapCapture';

const { getFontEmbedCSS, toCanvas } = vi.hoisted(() => ({
  getFontEmbedCSS: vi.fn(),
  toCanvas: vi.fn(),
}));

vi.mock('html-to-image', () => ({ getFontEmbedCSS, toCanvas }));

describe('HeatmapCapture cancellation', () => {
  beforeEach(() => {
    vi.resetAllMocks();
    getFontEmbedCSS.mockResolvedValue('');
  });

  it('does not read styles from a dialog closed before the lazy import finishes', async () => {
    const section = { isConnected: false } as HTMLElement;
    await expect(HeatmapCapture.toCanvas([section], '#fff'))
      .rejects.toMatchObject({ name: 'AbortError' });
    expect(getFontEmbedCSS).not.toHaveBeenCalled();
    expect(toCanvas).not.toHaveBeenCalled();
  });

  it('cancels when the dialog closes while fonts are embedding', async () => {
    const source = { isConnected: true };
    getFontEmbedCSS.mockImplementation(async () => {
      source.isConnected = false;
      return '';
    });
    await expect(HeatmapCapture.toCanvas([source as HTMLElement], '#fff'))
      .rejects.toMatchObject({ name: 'AbortError' });
    expect(toCanvas).not.toHaveBeenCalled();
  });

  it('does not return a partial image if the dialog closes during rasterization', async () => {
    const source = { isConnected: true, scrollWidth: 800, scrollHeight: 1200 };
    toCanvas.mockImplementation(async () => {
      source.isConnected = false;
      return { width: 1600, height: 2400 };
    });
    await expect(HeatmapCapture.toCanvas([source as HTMLElement], '#fff'))
      .rejects.toMatchObject({ name: 'AbortError' });
  });
});
