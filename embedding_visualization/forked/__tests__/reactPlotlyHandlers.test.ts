/**
 * react-plotly.js fork change 2 (`../plotly_changes.md`): Plotly event handlers are
 * re-synced after every update, not only after a figure change, so a handler whose
 * closure changed (new tooltip fields, a new search callback, a click handler that
 * appears once the plot is ready) is never left bound to its old copy.
 *
 * Runs the fork's real `factory.js` against a stand-in for the Plotly API. Its
 * `react` installs `on` / `removeListener` / `emit` on the graph div the way Plotly's
 * `Events.init` does on the first plot, and `purge` removes them again. The
 * component lifecycle is driven by hand (no React renderer in the node environment).
 */
import { EventEmitter } from 'node:events';
import { createRequire } from 'node:module';
import { describe, expect, it, vi } from 'vitest';

const require = createRequire(import.meta.url);
const plotComponentFactory = require('../react-plotly.js/factory.js').default;

type Listener = (...args: unknown[]) => void;

/** The graph div: event methods exist only between the first `react` and `purge`, as in Plotly. */
class FakeGraphDiv {
  data: unknown[] = [];
  layout: Record<string, unknown> = {};
  on?: (event: string, fn: Listener) => void;
  removeListener?: (event: string, fn: Listener) => void;
  private emitter: EventEmitter | null = null;

  installEvents() {
    if (this.emitter) return;
    const emitter = new EventEmitter();
    this.emitter = emitter;
    this.on = (event, fn) => { emitter.on(event, fn); };
    this.removeListener = (event, fn) => { emitter.removeListener(event, fn); };
  }

  removeEvents() {
    this.emitter = null;
    delete this.on;
    delete this.removeListener;
  }

  emit(event: string, payload: unknown = {}) {
    this.emitter?.emit(event, payload);
  }

  listenerCount(event: string): number {
    return this.emitter?.listenerCount(event) ?? 0;
  }
}

/** The parts of the Plotly API react-plotly calls; `hold` keeps the next `react` pending. */
class FakePlotly {
  reactCalls = 0;
  private gate: Promise<void> | null = null;

  react = async (gd: FakeGraphDiv) => {
    this.reactCalls += 1;
    const gate = this.gate;
    if (gate) await gate;
    gd.installEvents();
  };

  purge = (gd: FakeGraphDiv) => gd.removeEvents();

  /** Holds every `react` that starts from now on until the returned function is called. */
  hold(): () => void {
    let release!: () => void;
    this.gate = new Promise<void>(resolve => { release = resolve; });
    return () => {
      this.gate = null;
      release();
    };
  }
}

interface PlotInstance {
  props: Record<string, unknown>;
  p: Promise<void>;
  getRef(el: FakeGraphDiv): void;
  componentDidMount(): void;
  componentDidUpdate(prevProps: Record<string, unknown>): void;
  componentWillUnmount(): void;
}

const figure = { data: [{ type: 'scattergl' }], layout: { width: 400 }, config: {} };

/** Mounts the fork's component on a fake graph div and waits for the first plot. */
async function mount(props: Record<string, unknown> = {}) {
  const plotly = new FakePlotly();
  const Plot = plotComponentFactory(plotly);
  const initial = { ...figure, ...props };
  const plot: PlotInstance = new Plot(initial);
  const gd = new FakeGraphDiv();
  plot.getRef(gd);
  plot.componentDidMount();
  await plot.p;

  /** Re-renders with new props, the way React calls componentDidUpdate after a commit. */
  const rerender = (next: Record<string, unknown>) => {
    const prev = plot.props;
    plot.props = next;
    plot.componentDidUpdate(prev);
  };
  return { plot, gd, plotly, rerender };
}

describe('react-plotly fork: handlers re-synced after every update', () => {
  it('rebinds a changed handler without replotting when the figure is unchanged', async () => {
    const first = vi.fn();
    const second = vi.fn();
    const { plot, gd, plotly, rerender } = await mount({ onClick: first });

    rerender({ ...figure, onClick: second });
    await plot.p;
    gd.emit('plotly_click');

    expect(second).toHaveBeenCalledTimes(1);
    expect(first).not.toHaveBeenCalled();
    expect(gd.listenerCount('plotly_click')).toBe(1);
    expect(plotly.reactCalls).toBe(1);
  });

  it('binds a handler that appears after the first plot', async () => {
    // ScatterPlot2D passes `onClick` only once the plot reports it is ready
    const onClick = vi.fn();
    const { plot, gd, rerender } = await mount({ onClick: undefined });

    rerender({ ...figure, onClick });
    await plot.p;
    gd.emit('plotly_click');

    expect(onClick).toHaveBeenCalledTimes(1);
  });

  it('removes a handler that goes away', async () => {
    const onClick = vi.fn();
    const { plot, gd, rerender } = await mount({ onClick });

    rerender({ ...figure, onClick: undefined });
    await plot.p;
    gd.emit('plotly_click');

    expect(onClick).not.toHaveBeenCalled();
    expect(gd.listenerCount('plotly_click')).toBe(0);
  });

  it('still replots and rebinds on a figure change', async () => {
    const first = vi.fn();
    const second = vi.fn();
    const { plot, gd, plotly, rerender } = await mount({ onClick: first });

    rerender({ ...figure, data: [{ type: 'scattergl' }], onClick: second });
    await plot.p;
    gd.emit('plotly_click');

    expect(plotly.reactCalls).toBe(2);
    expect(second).toHaveBeenCalledTimes(1);
    expect(first).not.toHaveBeenCalled();
  });

  it('syncs only after a pending react has finished', async () => {
    const first = vi.fn();
    const second = vi.fn();
    const { plot, gd, plotly, rerender } = await mount({ onClick: first });

    const release = plotly.hold();
    const replotted = { ...figure, data: [{ type: 'scattergl' }], onClick: first };
    rerender(replotted);
    rerender({ ...replotted, onClick: second });
    await Promise.resolve();
    gd.emit('plotly_click');
    expect(second).not.toHaveBeenCalled();

    release();
    await plot.p;
    gd.emit('plotly_click');
    expect(second).toHaveBeenCalledTimes(1);
  });

  it('skips the sync when the plot unmounts first', async () => {
    const onError = vi.fn();
    const { plot, gd, plotly, rerender } = await mount({ onClick: vi.fn(), onError });

    const release = plotly.hold();
    const replotted = { ...figure, data: [{ type: 'scattergl' }], onClick: vi.fn(), onError };
    rerender(replotted);
    rerender({ ...replotted, onClick: vi.fn() });
    plot.componentWillUnmount();
    release();
    await plot.p;

    expect(onError).not.toHaveBeenCalled();
    expect(gd.on).toBeUndefined();
  });

  it('reports a failing sync to onError, and later updates still sync', async () => {
    const onError = vi.fn();
    const { plot, gd, rerender } = await mount({ onClick: vi.fn(), onError });

    const on = gd.on!;
    gd.on = () => { throw new Error('listener rejected'); };
    rerender({ ...figure, onClick: vi.fn(), onError });
    await plot.p;
    expect(onError).toHaveBeenCalledWith(expect.objectContaining({ message: 'listener rejected' }));

    gd.on = on;
    const later = vi.fn();
    rerender({ ...figure, onClick: later, onError });
    await plot.p;
    gd.emit('plotly_click');
    expect(later).toHaveBeenCalledTimes(1);
  });

  it('calls the current onUpdate on plotly_relayouting, without rebinding it', async () => {
    // ScatterPlot2D redraws its overlays from onUpdate during pan and zoom
    const first = vi.fn();
    const second = vi.fn();
    const { plot, gd, rerender } = await mount({ onUpdate: first });

    rerender({ ...figure, onUpdate: second });
    await plot.p;
    gd.emit('plotly_relayouting');

    expect(second).toHaveBeenCalledTimes(1);
    expect(first).not.toHaveBeenCalled();
  });
});
