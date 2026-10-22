import { describe, expect, it, vi } from 'vitest';

import { CoalescedRedraw } from '../coalescedRedraw';

/** Stands in for the microtask queue: callbacks wait until the test flushes them. */
class ManualQueue {
  private callbacks: (() => void)[] = [];

  readonly enqueue = (callback: () => void) => {
    this.callbacks.push(callback);
  };

  get size(): number {
    return this.callbacks.length;
  }

  flush(): void {
    const callbacks = this.callbacks;
    this.callbacks = [];
    for (const callback of callbacks) callback();
  }
}

describe('CoalescedRedraw', () => {
  it('runs one redraw for a burst of requests, after the burst', () => {
    // A fast pinch frame replays as several wheel events, each emitting plotly_relayouting
    const queue = new ManualQueue();
    const scheduler = new CoalescedRedraw(queue.enqueue);
    const redraw = vi.fn();

    scheduler.request(redraw);
    scheduler.request(redraw);
    scheduler.request(redraw);

    expect(redraw).not.toHaveBeenCalled();
    expect(queue.size).toBe(1);
    queue.flush();
    expect(redraw).toHaveBeenCalledTimes(1);
  });

  it('runs the latest redraw of the burst', () => {
    // React hands over a new closure when the plot size or theme changes
    const queue = new ManualQueue();
    const scheduler = new CoalescedRedraw(queue.enqueue);
    const stale = vi.fn();
    const latest = vi.fn();

    scheduler.request(stale);
    scheduler.request(latest);
    queue.flush();

    expect(stale).not.toHaveBeenCalled();
    expect(latest).toHaveBeenCalledTimes(1);
  });

  it('runs again for a request after the redraw', () => {
    const queue = new ManualQueue();
    const scheduler = new CoalescedRedraw(queue.enqueue);
    const redraw = vi.fn();

    scheduler.request(redraw);
    queue.flush();
    scheduler.request(redraw);
    queue.flush();

    expect(redraw).toHaveBeenCalledTimes(2);
  });

  it('never drops a request made during the redraw', () => {
    const queue = new ManualQueue();
    const scheduler = new CoalescedRedraw(queue.enqueue);
    let calls = 0;
    const redraw = () => {
      calls += 1;
      if (calls === 1) scheduler.request(redraw);
    };

    scheduler.request(redraw);
    queue.flush();
    expect(calls).toBe(1);
    expect(queue.size).toBe(1);
    queue.flush();
    expect(calls).toBe(2);
  });

  it('drops a cancelled redraw, and a later request still runs once', () => {
    const queue = new ManualQueue();
    const scheduler = new CoalescedRedraw(queue.enqueue);
    const redraw = vi.fn();

    scheduler.request(redraw);
    scheduler.cancel();
    queue.flush();
    expect(redraw).not.toHaveBeenCalled();

    // The cancelled callback can still be queued when the next request lands
    scheduler.request(redraw);
    scheduler.cancel();
    scheduler.request(redraw);
    queue.flush();
    expect(redraw).toHaveBeenCalledTimes(1);
  });

  it('stays usable after a redraw throws', () => {
    const queue = new ManualQueue();
    const scheduler = new CoalescedRedraw(queue.enqueue);
    const redraw = vi.fn().mockImplementationOnce(() => {
      throw new Error('lost GL context');
    });

    scheduler.request(redraw);
    expect(() => queue.flush()).toThrow('lost GL context');
    scheduler.request(redraw);
    queue.flush();

    expect(redraw).toHaveBeenCalledTimes(2);
  });

  it('flushes in a microtask by default, before the next task', async () => {
    const scheduler = new CoalescedRedraw();
    const redraw = vi.fn();
    const nextTask = new Promise<number>(resolve => setTimeout(() => resolve(redraw.mock.calls.length), 0));

    scheduler.request(redraw);
    scheduler.request(redraw);
    expect(redraw).not.toHaveBeenCalled();

    await Promise.resolve();
    expect(redraw).toHaveBeenCalledTimes(1);
    expect(await nextTask).toBe(1);
  });
});
