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

const ICON_HOME = {
  width: 24,
  height: 24,
  path: 'M15 21v-8a1 1 0 0 0-1-1h-4a1 1 0 0 0-1 1v8 M3 10a2 2 0 0 1 .709-1.528l7-6a2 2 0 0 1 2.582 0l7 6A2 2 0 0 1 21 10v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z',
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

// ---------------------------------------------------------------------------
// Button builders
// ---------------------------------------------------------------------------

export function build3DModeBarButtons(
  plotlyLib: any,
): ModeBarButtonAny[][] {
  return [
    [
      {
        name: 'pan3d',
        title: '',
        icon: ICON_MOVE,
        click: (gd: any) => {
          if (plotlyLib) plotlyLib.relayout(gd, { 'scene.dragmode': 'pan' });
        },
      },
      {
        name: 'orbitRotation',
        title: '',
        icon: ICON_ORBIT,
        click: (gd: any) => {
          if (plotlyLib) plotlyLib.relayout(gd, { 'scene.dragmode': 'orbit' });
        },
      },
      {
        name: 'resetCameraDefault3d',
        title: '',
        icon: ICON_HOME,
        click: (gd: any) => {
          if (plotlyLib) plotlyLib.relayout(gd, { 'scene.camera': gd._fullLayout?.scene?._scene?.viewInitial?.camera });
        },
      },
    ] as ModeBarButtonAny[],
  ];
}

export function build2DModeBarButtons(
  plotlyLib: any,
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
    ] as ModeBarButtonAny[],
  ];
}
