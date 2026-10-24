import { describe, expect, it, vi } from 'vitest';

import { ClusterLabelClicks, LabelHitBoxes } from '../labelHitBoxes';

describe('LabelHitBoxes', () => {
  it('finds the labels whose box holds a point, edges included, in drawing order', () => {
    const hits = new LabelHitBoxes();
    hits.add({ loX: 0, loY: 0, hiX: 10, hiY: 10 }, 'a');
    hits.add({ loX: 5, loY: 5, hiX: 20, hiY: 20 }, 'b');
    expect(hits.at(7, 7)).toEqual(['a', 'b']);
    expect(hits.at(10, 0)).toEqual(['a']);
    expect(hits.at(15, 15)).toEqual(['b']);
    expect(hits.at(30, 30)).toEqual([]);
    expect(hits.size).toBe(2);
  });

  it('starts over when cleared', () => {
    const hits = new LabelHitBoxes();
    hits.add({ loX: 0, loY: 0, hiX: 10, hiY: 10 }, 'a');
    hits.clear();
    expect(hits.at(5, 5)).toEqual([]);
    expect(hits.size).toBe(0);
  });
});

describe('ClusterLabelClicks', () => {
  it('toggles the topic when the label names one', () => {
    const onTopic = vi.fn();
    const onCategory = vi.fn(() => true);
    const handled = ClusterLabelClicks.route(['Poetry'], { topicIdOf: () => 7, onTopic, onCategory }, false);
    expect(handled).toBe(true);
    expect(onTopic).toHaveBeenCalledWith(7);
    expect(onCategory).not.toHaveBeenCalled();
  });

  it('falls back to the category handler, passing shift', () => {
    const onCategory = vi.fn(() => true);
    const handled = ClusterLabelClicks.route(['EMNLP'], { topicIdOf: () => undefined, onTopic: vi.fn(), onCategory }, true);
    expect(handled).toBe(true);
    expect(onCategory).toHaveBeenCalledWith('EMNLP', true);
  });

  it('tries the next label when one is declined, and reports a click nobody took', () => {
    const onCategory = vi.fn((label: string) => label === 'second');
    expect(ClusterLabelClicks.route(['first', 'second'], { onCategory }, false)).toBe(true);
    expect(onCategory).toHaveBeenCalledTimes(2);
    expect(ClusterLabelClicks.route(['first'], { onCategory }, false)).toBe(false);
    expect(ClusterLabelClicks.route(['first'], {}, false)).toBe(false);
  });

  it('needs a topic handler to use a topic id', () => {
    const onCategory = vi.fn(() => false);
    expect(ClusterLabelClicks.route(['Poetry'], { topicIdOf: () => 7, onCategory }, false)).toBe(false);
    expect(onCategory).toHaveBeenCalledWith('Poetry', false);
  });

  it('treats a release more than 5 px from its press as a drag, not a click', () => {
    expect(ClusterLabelClicks.isDrag({ x: 0, y: 0 }, { x: 3, y: 4 })).toBe(false);
    expect(ClusterLabelClicks.isDrag({ x: 0, y: 0 }, { x: 4, y: 4 })).toBe(true);
    expect(ClusterLabelClicks.isDrag(null, { x: 100, y: 0 })).toBe(false);
  });

  it('knows whether any click handler is set', () => {
    expect(ClusterLabelClicks.isClickable({})).toBe(false);
    expect(ClusterLabelClicks.isClickable({ topicIdOf: () => 1 })).toBe(false);
    expect(ClusterLabelClicks.isClickable({ topicIdOf: () => 1, onTopic: vi.fn() })).toBe(true);
    expect(ClusterLabelClicks.isClickable({ onCategory: vi.fn() })).toBe(true);
  });
});
