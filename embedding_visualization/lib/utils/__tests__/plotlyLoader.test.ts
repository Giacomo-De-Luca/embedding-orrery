import { describe, expect, it, vi } from 'vitest';

import { PlotlyLoader } from '../plotlyLoader';

type Importer = ConstructorParameters<typeof PlotlyLoader>[0];
type PlotlyModule = Awaited<ReturnType<Importer>>;

const lib = { newPlot: () => undefined } as unknown as PlotlyModule['default'];
const plotlyModule = { default: lib } as unknown as PlotlyModule;

const settle = () => new Promise<void>((resolve) => setTimeout(resolve, 0));

describe('PlotlyLoader', () => {
  it('imports the bundle once and hands every caller the library', async () => {
    const importer = vi.fn<Importer>().mockResolvedValue(plotlyModule);
    const loader = new PlotlyLoader(importer);

    const [first, second] = await Promise.all([loader.load(), loader.load()]);
    const third = await loader.load();

    expect(first).toBe(lib);
    expect(second).toBe(lib);
    expect(third).toBe(lib);
    expect(importer).toHaveBeenCalledTimes(1);
  });

  it('preload starts the import that a later load shares', async () => {
    const importer = vi.fn<Importer>().mockResolvedValue(plotlyModule);
    const loader = new PlotlyLoader(importer);

    loader.preload();
    expect(importer).toHaveBeenCalledTimes(1);

    expect(await loader.load()).toBe(lib);
    expect(importer).toHaveBeenCalledTimes(1);
  });

  it('a failed preload is silent and the next load retries the import', async () => {
    const importer = vi
      .fn<Importer>()
      .mockRejectedValueOnce(new Error('chunk load failed'))
      .mockResolvedValueOnce(plotlyModule);
    const loader = new PlotlyLoader(importer);

    loader.preload();
    // Lets the rejection settle: an uncaught one would fail the run.
    await settle();

    expect(await loader.load()).toBe(lib);
    expect(importer).toHaveBeenCalledTimes(2);
  });

  it('load reports a failure and retries on the next call', async () => {
    const error = new Error('offline');
    const importer = vi
      .fn<Importer>()
      .mockRejectedValueOnce(error)
      .mockResolvedValueOnce(plotlyModule);
    const loader = new PlotlyLoader(importer);

    await expect(loader.load()).rejects.toBe(error);

    expect(await loader.load()).toBe(lib);
    expect(importer).toHaveBeenCalledTimes(2);
  });
});
