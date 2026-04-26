'use client';

import { useRef, useCallback, useEffect } from 'react';
import type { MutableRefObject, RefObject } from 'react';
import type { Point3D } from '../types/types';
import { easeInOutCubic, lerp, cartesianToSpherical, sphericalToCartesian, getSceneNormalization, type DataBounds } from '../../app/utils/rendeding';

interface CameraState {
  eye: { x: number; y: number; z: number };
  center: { x: number; y: number; z: number };
}

/**
 * Encapsulates the camera fly-to animation for 3D scatter plots.
 * Owns animation frame and isAnimating state; shared refs are passed in.
 */
export function useCameraFlyTo(
  bounds: DataBounds | null,
  graphDivRef: MutableRefObject<any>,
  currentCameraRef: MutableRefObject<CameraState>,
  plotlyLibRef: MutableRefObject<any>,
  renderLabelsRef: MutableRefObject<(() => void) | null>,
  labelCanvasRef: RefObject<HTMLCanvasElement | null>,
) {
  const animationFrameRef = useRef<number | undefined>(undefined);
  const isAnimatingRef = useRef(false);

  // Cancel any in-flight animation on unmount
  useEffect(() => {
    return () => {
      if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
    };
  }, []);

  // Start camera fly-to animation. Called imperatively after Plotly.redraw so the
  // main thread is free and the animation frames aren't blocked.
  const startFlyTo = useCallback((target: Point3D) => {
    if (!bounds || !graphDivRef.current) return;
    if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);

    const norm = getSceneNormalization(graphDivRef.current, bounds);
    if (!norm) return;

    const targetCenterX = (target.x - norm.centerX) / norm.normRange;
    const targetCenterY = (target.y - norm.centerY) / norm.normRange;
    const targetCenterZ = (target.z - norm.centerZ) / norm.normRange;

    const targetR = 0.15;
    const targetPhi = 1.3;
    const duration = 2000;

    let startEye: any, startCenter: any, startSpherical: any, targetSpherical: any, startTime: number;
    let initialized = false;

    const animate = (currentTime: number) => {
      if (!isAnimatingRef.current || !graphDivRef.current) return;

      if (!initialized) {
        const scene = graphDivRef.current._fullLayout?.scene;
        const layoutCamera = scene?.camera;
        if (layoutCamera?.eye) {
          startEye = { ...layoutCamera.eye };
          startCenter = layoutCamera.center ? { ...layoutCamera.center } : { x: 0, y: 0, z: 0 };
        } else {
          startEye = { ...currentCameraRef.current.eye };
          startCenter = { ...currentCameraRef.current.center };
        }
        startSpherical = cartesianToSpherical(
          startEye.x - startCenter.x,
          startEye.y - startCenter.y,
          startEye.z - startCenter.z
        );
        targetSpherical = { r: targetR, theta: startSpherical.theta + 0.5, phi: targetPhi };
        startTime = currentTime;
        initialized = true;
      }

      const elapsed = currentTime - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const ease = easeInOutCubic(progress);

      const newCenter = {
        x: lerp(startCenter.x, targetCenterX, ease),
        y: lerp(startCenter.y, targetCenterY, ease),
        z: lerp(startCenter.z, targetCenterZ, ease),
      };

      const curR = lerp(startSpherical.r, targetSpherical.r, ease);
      const curTheta = lerp(startSpherical.theta, targetSpherical.theta, ease);
      const curPhi = lerp(startSpherical.phi, targetSpherical.phi, ease);

      const relativeEye = sphericalToCartesian(curR, curTheta, curPhi);

      const newEye = {
        x: relativeEye.x + newCenter.x,
        y: relativeEye.y + newCenter.y,
        z: relativeEye.z + newCenter.z,
      };

      currentCameraRef.current = { eye: newEye, center: newCenter };
      const currentScene = graphDivRef.current._fullLayout?.scene?._scene;
      const glplot = currentScene?.glplot as any;

      if (glplot?.camera) {
        if (Array.isArray(glplot.camera.eye)) {
          glplot.camera.eye = [newEye.x, newEye.y, newEye.z];
          glplot.camera.center = [newCenter.x, newCenter.y, newCenter.z];
          glplot.camera.up = [0, 0, 1];
        } else if (glplot.camera.eye && typeof glplot.camera.eye === 'object') {
          glplot.camera.eye.x = newEye.x;
          glplot.camera.eye.y = newEye.y;
          glplot.camera.eye.z = newEye.z;
          glplot.camera.center.x = newCenter.x;
          glplot.camera.center.y = newCenter.y;
          glplot.camera.center.z = newCenter.z;
        }
        if (typeof glplot.camera.update === 'function') glplot.camera.update();
        if (typeof glplot.draw === 'function') glplot.draw();
      }

      if (progress < 1) {
        animationFrameRef.current = requestAnimationFrame(animate);
      } else {
        isAnimatingRef.current = false;
        if (plotlyLibRef.current && graphDivRef.current) {
          plotlyLibRef.current.relayout(graphDivRef.current, {
            'scene.camera': { eye: newEye, center: newCenter, up: { x: 0, y: 0, z: 1 } },
          });
        }
        renderLabelsRef.current?.();
      }
    };

    isAnimatingRef.current = true;
    const labelCtx = labelCanvasRef.current?.getContext('2d');
    if (labelCtx && labelCanvasRef.current) {
      labelCtx.clearRect(0, 0, labelCanvasRef.current.width, labelCanvasRef.current.height);
    }
    animationFrameRef.current = requestAnimationFrame(animate);
  }, [bounds]);

  return { startFlyTo, isAnimatingRef, animationFrameRef };
}

/**
 * Smoothly animate the camera from its current position to a target eye+center.
 * Uses spherical interpolation (like flyTo) for natural arc motion.
 * Defers start by one rAF frame so Plotly.react() settles before we read camera state.
 * Shares animationFrameRef/isAnimatingRef with useCameraFlyTo so animations cancel each other.
 */
export function animateCameraToRegion(opts: {
  targetEye: { x: number; y: number; z: number };
  targetCenter: { x: number; y: number; z: number };
  duration: number;
  graphDivRef: MutableRefObject<any>;
  currentCameraRef: MutableRefObject<CameraState>;
  plotlyLibRef: MutableRefObject<any>;
  isAnimatingRef: MutableRefObject<boolean>;
  animationFrameRef: MutableRefObject<number | undefined>;
  renderLabelsRef: MutableRefObject<(() => void) | null>;
  labelCanvasRef: RefObject<HTMLCanvasElement | null>;
}): void {
  const {
    targetEye, targetCenter, duration,
    graphDivRef, currentCameraRef, plotlyLibRef,
    isAnimatingRef, animationFrameRef,
    renderLabelsRef, labelCanvasRef,
  } = opts;

  if (!graphDivRef.current) return;
  if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);

  isAnimatingRef.current = true;
  const labelCtx = labelCanvasRef.current?.getContext('2d');
  if (labelCtx && labelCanvasRef.current) {
    labelCtx.clearRect(0, 0, labelCanvasRef.current.width, labelCanvasRef.current.height);
  }

  // Defer by one frame so Plotly.react() from the trace update effect has settled
  animationFrameRef.current = requestAnimationFrame(() => {
    if (!isAnimatingRef.current || !graphDivRef.current) return;

    // Read live camera state AFTER Plotly.react has completed
    const scene = graphDivRef.current._fullLayout?.scene;
    const layoutCamera = scene?.camera;
    const startEye = layoutCamera?.eye
      ? { ...layoutCamera.eye }
      : { ...currentCameraRef.current.eye };
    const startCenter = layoutCamera?.center
      ? { ...layoutCamera.center }
      : { ...currentCameraRef.current.center };

    // Convert start/target eye positions (relative to their centers) to spherical
    const startSpherical = cartesianToSpherical(
      startEye.x - startCenter.x,
      startEye.y - startCenter.y,
      startEye.z - startCenter.z,
    );
    const targetSpherical = cartesianToSpherical(
      targetEye.x - targetCenter.x,
      targetEye.y - targetCenter.y,
      targetEye.z - targetCenter.z,
    );

    let startTime: number | null = null;

    const animate = (currentTime: number) => {
      if (!isAnimatingRef.current || !graphDivRef.current) return;

      if (startTime === null) startTime = currentTime;
      const elapsed = currentTime - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const ease = easeInOutCubic(progress);

      // Interpolate center linearly
      const newCenter = {
        x: lerp(startCenter.x, targetCenter.x, ease),
        y: lerp(startCenter.y, targetCenter.y, ease),
        z: lerp(startCenter.z, targetCenter.z, ease),
      };

      // Interpolate eye in spherical coordinates for smooth arc
      const curR = lerp(startSpherical.r, targetSpherical.r, ease);
      const curTheta = lerp(startSpherical.theta, targetSpherical.theta, ease);
      const curPhi = lerp(startSpherical.phi, targetSpherical.phi, ease);
      const relativeEye = sphericalToCartesian(curR, curTheta, curPhi);

      const newEye = {
        x: relativeEye.x + newCenter.x,
        y: relativeEye.y + newCenter.y,
        z: relativeEye.z + newCenter.z,
      };

      currentCameraRef.current = { eye: newEye, center: newCenter };
      const currentScene = graphDivRef.current._fullLayout?.scene?._scene;
      const glplot = currentScene?.glplot as any;

      if (glplot?.camera) {
        if (Array.isArray(glplot.camera.eye)) {
          glplot.camera.eye = [newEye.x, newEye.y, newEye.z];
          glplot.camera.center = [newCenter.x, newCenter.y, newCenter.z];
          glplot.camera.up = [0, 0, 1];
        } else if (glplot.camera.eye && typeof glplot.camera.eye === 'object') {
          glplot.camera.eye.x = newEye.x;
          glplot.camera.eye.y = newEye.y;
          glplot.camera.eye.z = newEye.z;
          glplot.camera.center.x = newCenter.x;
          glplot.camera.center.y = newCenter.y;
          glplot.camera.center.z = newCenter.z;
        }
        if (typeof glplot.camera.update === 'function') glplot.camera.update();
        if (typeof glplot.draw === 'function') glplot.draw();
      }

      if (progress < 1) {
        animationFrameRef.current = requestAnimationFrame(animate);
      } else {
        isAnimatingRef.current = false;
        if (plotlyLibRef.current && graphDivRef.current) {
          plotlyLibRef.current.relayout(graphDivRef.current, {
            'scene.camera': { eye: newEye, center: newCenter, up: { x: 0, y: 0, z: 1 } },
          });
        }
        renderLabelsRef.current?.();
      }
    };

    animationFrameRef.current = requestAnimationFrame(animate);
  });
}
