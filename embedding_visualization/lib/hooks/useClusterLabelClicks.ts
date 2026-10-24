'use client';

import { useCallback, useEffect, useMemo, useRef, type MouseEvent, type RefObject } from 'react';
import { ClusterLabelClicks, type ClusterLabelClickHandlers, type LabelHitBoxes } from '../utils/labels/labelHitBoxes';

interface UseClusterLabelClicksArgs {
  /** The plot container: label hit boxes are in its coordinates. */
  containerRef: RefObject<HTMLElement | null>;
  /** The clickable labels the plot drew last. */
  hits: LabelHitBoxes;
  /** Whether the plot draws cluster labels. */
  enabled: boolean;
  topicLabelToIdMap?: Map<string, number> | null;
  /** Toggles a topic, for labels that name one. */
  onTopic?: (topicId: number) => void;
  /** Toggles a category (shift: add to the selection); returns whether the label was one. */
  onCategory?: (label: string, shiftKey: boolean) => boolean;
  /** True while the labels are hidden without being redrawn (a 3D camera animation). */
  isSuspended?: () => boolean;
}

/**
 * Clicks on a plot's cluster labels, shared by both scatter plots. A click on a
 * clickable label toggles its topic or category and is kept from the plot; a press
 * that moved more than 5 px before its release is a drag (pan, rotate), and a click
 * on the modebar is the modebar's. Plotly reports a point click before (2D) or
 * during (3D) the press that a label takes, so the plots ask `isOnLabel` before
 * selecting a point.
 */
export function useClusterLabelClicks({
  containerRef,
  hits,
  enabled,
  topicLabelToIdMap,
  onTopic,
  onCategory,
  isSuspended,
}: UseClusterLabelClicksArgs) {
  const handlers = useMemo<ClusterLabelClickHandlers>(() => ({
    topicIdOf: topicLabelToIdMap ? (label) => topicLabelToIdMap.get(label) : undefined,
    onTopic,
    onCategory,
  }), [topicLabelToIdMap, onTopic, onCategory]);
  const clickable = enabled && ClusterLabelClicks.isClickable(handlers);

  // Where the last press started, to tell clicks from drags
  const pressRef = useRef<{ x: number; y: number } | null>(null);
  useEffect(() => {
    const container = containerRef.current;
    if (!container || !clickable) return;
    const onPointerDown = (e: PointerEvent) => {
      pressRef.current = { x: e.clientX, y: e.clientY };
    };
    container.addEventListener('pointerdown', onPointerDown, { capture: true });
    return () => container.removeEventListener('pointerdown', onPointerDown, { capture: true });
  }, [containerRef, clickable]);

  /** The clickable labels under a pointer, from client coordinates (none while suspended). */
  const labelsAt = useCallback((clientX: number, clientY: number): string[] => {
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect || isSuspended?.()) return [];
    return hits.at(clientX - rect.left, clientY - rect.top);
  }, [containerRef, hits, isSuspended]);

  /** Whether a pointer position is on a clickable label, so a point click there is the label's. */
  const isOnLabel = useCallback(
    (clientX: number, clientY: number) => clickable && labelsAt(clientX, clientY).length > 0,
    [clickable, labelsAt],
  );

  const handleClickCapture = useCallback((e: MouseEvent) => {
    if (e.target instanceof Element && e.target.closest('.modebar-container')) return;
    if (ClusterLabelClicks.isDrag(pressRef.current, { x: e.clientX, y: e.clientY })) return;
    if (ClusterLabelClicks.route(labelsAt(e.clientX, e.clientY), handlers, e.shiftKey)) e.stopPropagation();
  }, [labelsAt, handlers]);

  return useMemo(
    () => ({ isOnLabel, onClickCapture: clickable ? handleClickCapture : undefined }),
    [isOnLabel, clickable, handleClickCapture],
  );
}
