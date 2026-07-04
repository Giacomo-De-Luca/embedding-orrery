/**
 * Custom Plotly modebar buttons using Lucide icon SVG paths.
 *
 * Lucide icons are stroke-based (24x24 viewBox). Circles are converted to arc paths
 * so everything fits in Plotly's single-path icon format. CSS overrides in globals.css
 * switch Plotly's default `fill` rendering to `stroke` for these icons.
 */

import type { ModeBarButtonAny } from 'plotly.js';

// ---------------------------------------------------------------------------
// Lucide SVG path data (24x24 viewBox, stroke-based)
// Circle elements converted to arc paths: M(cx-r) cy a r r 0 1 0 (2r) 0 a r r 0 1 0 -(2r) 0
// ---------------------------------------------------------------------------



const ICON_MOVE = {
  width: 24,
  height: 24,
  path: 'M12 2v20 M15 19l-3 3-3-3 M19 9l3 3-3 3 M2 12h20 M5 9l-3 3 3 3 M9 5l3-3 3 3',
};

const ICON_ORBIT = {
  width: 24,
  height: 24,
  path: 'M20.341 6.484A10 10 0 0 1 10.266 21.85 M3.659 17.516A10 10 0 0 1 13.74 2.152 M9 12a3 3 0 1 0 6 0a3 3 0 1 0-6 0 M17 5a2 2 0 1 0 4 0a2 2 0 1 0-4 0 M3 19a2 2 0 1 0 4 0a2 2 0 1 0-4 0',
};

const ICON_UNDO = {
  width: 24,
  height: 24,
  // Lucide "undo-2" icon: curved arrow pointing back
  path: 'M9 14l-5-5 5-5 M4 9h10.5a5.5 5.5 0 0 1 5.5 5.5a5.5 5.5 0 0 1-5.5 5.5H11',
};

const ICON_BOX_SELECT = {
  width: 24,
  height: 24,
  path: 'M5 3a2 2 0 0 0-2 2 M19 3a2 2 0 0 1 2 2 M21 19a2 2 0 0 1-2 2 M5 21a2 2 0 0 1-2-2 M9 3h1 M9 21h1 M14 3h1 M14 21h1 M3 9v1 M21 9v1 M3 14v1 M21 14v1',
};

const ICON_LASSO = {
  width: 24,
  height: 24,
  path: 'M3.704 14.467A10 8 0 0 1 2 10a10 8 0 0 1 20 0 10 8 0 0 1-10 8 10 8 0 0 1-5.181-1.158 M7 22a5 5 0 0 1-2-3.994 M3 16a2 2 0 1 0 4 0a2 2 0 1 0-4 0',
};

const ICON_MAXIMIZE = {
  width: 24,
  height: 24,
  path: 'M8 3H5a2 2 0 0 0-2 2v3 M21 8V5a2 2 0 0 0-2-2h-3 M3 16v3a2 2 0 0 0 2 2h3 M16 21h3a2 2 0 0 0 2-2v-3',
};

const ICON_CAMERA = {
  width: 24,
  height: 24,
  path: 'M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3l-2.5-3z M9 13a3 3 0 1 0 6 0a3 3 0 1 0-6 0',
};

/** Minimal ref shape so this util stays free of React imports. */
type RefLike<T> = { current: T | null };

/** Screenshot button appended to a group when a handler ref is provided. */
function screenshotButton(onScreenshotRef: RefLike<(gd: any) => void>): ModeBarButtonAny {
  return {
    name: 'downloadImage',
    title: '',
    icon: ICON_CAMERA,
    click: (gd: any) => {
      onScreenshotRef.current?.(gd);
    },
  } as ModeBarButtonAny;
}

// ---------------------------------------------------------------------------
// Button builders
// ---------------------------------------------------------------------------

export function build3DModeBarButtons(
  plotlyLib: any,
  onScreenshotRef?: RefLike<(gd: any) => void>,
): ModeBarButtonAny[][] {
  // Helper: read the live camera from glplot so relayout preserves the current view
  const getCurrentCamera = (gd: any) => {
    const scene = gd._fullLayout?.scene;
    const glplot = scene?._scene?.glplot;
    const cam = glplot?.camera;
    if (!cam) return undefined;
    // glplot stores eye/center as arrays — convert to {x,y,z} for Plotly layout
    const toXYZ = (v: any) => {
      if (Array.isArray(v)) return { x: v[0], y: v[1], z: v[2] };
      return v;
    };
    return {
      eye: toXYZ(cam.eye),
      center: toXYZ(cam.center || [0, 0, 0]),
      up: toXYZ(cam.up || [0, 0, 1]),
    };
  };

  return [
    [
      {
        name: 'pan3d',
        title: '',
        icon: ICON_MOVE,
        click: (gd: any) => {
          if (!plotlyLib) return;
          // Preserve current camera when switching drag mode
          const cam = getCurrentCamera(gd);
          plotlyLib.relayout(gd, {
            'scene.dragmode': 'pan',
            ...(cam && { 'scene.camera': cam }),
          });
        },
      },
      {
        name: 'orbitRotation',
        title: '',
        icon: ICON_ORBIT,
        click: (gd: any) => {
          if (!plotlyLib) return;
          // Preserve current camera when switching drag mode
          const cam = getCurrentCamera(gd);
          plotlyLib.relayout(gd, {
            'scene.dragmode': 'orbit',
            ...(cam && { 'scene.camera': cam }),
          });
        },
      },
      {
        name: 'resetCameraDefault3d',
        title: '',
        icon: ICON_UNDO,
        click: (gd: any) => {
          if (plotlyLib) plotlyLib.relayout(gd, { 'scene.camera': gd._fullLayout?.scene?._scene?.viewInitial?.camera });
        },
      },
      ...(onScreenshotRef ? [screenshotButton(onScreenshotRef)] : []),
    ] as ModeBarButtonAny[],
  ];
}

export function build2DModeBarButtons(
  plotlyLib: any,
  onScreenshotRef?: RefLike<(gd: any) => void>,
): ModeBarButtonAny[][] {
  return [
    [
      {
        name: 'pan2d',
        title: '',
        icon: ICON_MOVE,
        click: (gd: any) => {
          if (plotlyLib) plotlyLib.relayout(gd, { dragmode: 'pan' });
        },
      },
      {
        name: 'select2d',
        title: '',
        icon: ICON_BOX_SELECT,
        click: (gd: any) => {
          if (plotlyLib) plotlyLib.relayout(gd, { dragmode: 'select' });
        },
      },
      {
        name: 'lasso2d',
        title: '',
        icon: ICON_LASSO,
        click: (gd: any) => {
          if (plotlyLib) plotlyLib.relayout(gd, { dragmode: 'lasso' });
        },
      },
      {
        name: 'resetScale2d',
        title: '',
        icon: ICON_MAXIMIZE,
        click: (gd: any) => {
          if (plotlyLib) plotlyLib.relayout(gd, { 'xaxis.autorange': true, 'yaxis.autorange': true });
        },
      },
      ...(onScreenshotRef ? [screenshotButton(onScreenshotRef)] : []),
    ] as ModeBarButtonAny[],
  ];
}
