import type { Point3D } from '../../lib/types/types';

// --- Animation Helpers ---
export const easeInOutCubic = (t: number): number => {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
};

export const lerp = (start: number, end: number, t: number) => {
  return start + (end - start) * t;
};

export function cartesianToSpherical(x: number, y: number, z: number) {
  const r = Math.sqrt(x * x + y * y + z * z);
  const theta = Math.atan2(y, x);
  const phi = Math.acos(z / (r || 1));
  return { r, theta, phi };
}

export function sphericalToCartesian(r: number, theta: number, phi: number) {
  return {
    x: r * Math.sin(phi) * Math.cos(theta),
    y: r * Math.sin(phi) * Math.sin(theta),
    z: r * Math.cos(phi),
  };
}

export const getZoomLevel = (
  eye: { x: number; y: number; z: number },
  center: { x: number; y: number; z: number } = { x: 0, y: 0, z: 0 }
): number => {
  const dx = eye.x - center.x;
  const dy = eye.y - center.y;
  const dz = eye.z - center.z;
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
};

export const getZoomMultiplier = (
  eye: { x: number; y: number; z: number },
  center: { x: number; y: number; z: number } = { x: 0, y: 0, z: 0 },
  defaultDistance: number = Math.sqrt(0.9**2 * 3) // ~1.56 for your defaults
): number => {
  const distance = getZoomLevel(eye, center);
  return defaultDistance / distance;
};


export function formatHoverText(point: Point3D): string {
  const label = point.label || point.id;
  const doc = point.document || '';
  const truncatedDoc = doc.length > 100 ? doc.substring(0, 100) + '...' : doc;
  return `${label}<br>${truncatedDoc}`;
}
