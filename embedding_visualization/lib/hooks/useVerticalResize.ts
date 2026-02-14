'use client';

import { useRef, useState, useCallback, useEffect } from 'react';

interface UseVerticalResizeOptions {
  initialHeight: number;
  minHeight: number;
  maxHeight?: number;
  onCollapse: () => void;
}

/**
 * Hook for drag-to-resize with collapse-on-release behavior.
 * Attach `handleRef` to the drag handle element.
 * Uses a callback ref so listeners are registered when the element actually mounts,
 * even if the element is conditionally rendered.
 */
export function useVerticalResize({
  initialHeight,
  minHeight,
  maxHeight = 600,
  onCollapse,
}: UseVerticalResizeOptions) {
  const [height, setHeight] = useState(initialHeight);
  const [isDragging, setIsDragging] = useState(false);
  // Callback ref: triggers re-render when element mounts/unmounts
  const [handleEl, setHandleEl] = useState<HTMLDivElement | null>(null);
  const handleRef = useCallback((node: HTMLDivElement | null) => setHandleEl(node), []);

  const heightRef = useRef(height);
  heightRef.current = height;
  const onCollapseRef = useRef(onCollapse);
  onCollapseRef.current = onCollapse;

  const startRef = useRef<{ startY: number; startHeight: number } | null>(null);

  const handleMove = useCallback((clientY: number) => {
    if (!startRef.current) return;
    const { startY, startHeight } = startRef.current;
    const newHeight = Math.min(maxHeight, Math.max(minHeight / 2, startHeight + (clientY - startY)));
    setHeight(newHeight);
  }, [maxHeight, minHeight]);

  const handleEnd = useCallback(() => {
    if (!startRef.current) return;
    startRef.current = null;
    setIsDragging(false);
    if (heightRef.current < minHeight) {
      onCollapseRef.current();
      setHeight(initialHeight);
    }
  }, [minHeight, initialHeight]);

  const onMouseMove = useCallback((e: MouseEvent) => handleMove(e.clientY), [handleMove]);
  const onTouchMoveHandler = useCallback((e: TouchEvent) => {
    if (e.touches.length === 1) handleMove(e.touches[0].clientY);
  }, [handleMove]);

  const onMouseUp = useCallback(() => {
    handleEnd();
    document.removeEventListener('mousemove', onMouseMove);
    document.removeEventListener('mouseup', onMouseUp);
  }, [handleEnd, onMouseMove]);

  const onTouchEndHandler = useCallback(() => {
    handleEnd();
    document.removeEventListener('touchmove', onTouchMoveHandler);
    document.removeEventListener('touchend', onTouchEndHandler);
  }, [handleEnd, onTouchMoveHandler]);

  // Re-runs when handleEl changes (element mounts/unmounts) or handlers change
  useEffect(() => {
    if (!handleEl) return;

    const onMouseDown = (e: MouseEvent) => {
      e.preventDefault();
      startRef.current = { startY: e.clientY, startHeight: heightRef.current };
      setIsDragging(true);
      document.addEventListener('mousemove', onMouseMove);
      document.addEventListener('mouseup', onMouseUp);
    };

    const onTouchStart = (e: TouchEvent) => {
      if (e.touches.length !== 1) return;
      startRef.current = { startY: e.touches[0].clientY, startHeight: heightRef.current };
      setIsDragging(true);
      document.addEventListener('touchmove', onTouchMoveHandler, { passive: true });
      document.addEventListener('touchend', onTouchEndHandler);
    };

    handleEl.addEventListener('mousedown', onMouseDown);
    handleEl.addEventListener('touchstart', onTouchStart, { passive: true });

    return () => {
      handleEl.removeEventListener('mousedown', onMouseDown);
      handleEl.removeEventListener('touchstart', onTouchStart);
      document.removeEventListener('mousemove', onMouseMove);
      document.removeEventListener('mouseup', onMouseUp);
      document.removeEventListener('touchmove', onTouchMoveHandler);
      document.removeEventListener('touchend', onTouchEndHandler);
    };
  }, [handleEl, onMouseMove, onMouseUp, onTouchMoveHandler, onTouchEndHandler]);

  const reset = useCallback(() => setHeight(initialHeight), [initialHeight]);

  return { height, handleRef, isDragging, reset };
}
