/**
 * Framebuffer format per float precision: phones without 32-bit float
 * blending or filtering get half-float targets (see floatRenderTargets.ts).
 */

import { describe, it, expect } from 'vitest';
import { createFramebuffer } from '../glResources';

/** WebGL2 enum values the helper reads, plus a texImage2D/texParameteri recorder. */
function recordingContext() {
  const calls = { texImage2D: [] as unknown[][], texParameteri: new Map<number, number>() };
  const gl = {
    TEXTURE_2D: 0x0de1,
    FRAMEBUFFER: 0x8d40,
    COLOR_ATTACHMENT0: 0x8ce0,
    TEXTURE_MIN_FILTER: 0x2801,
    TEXTURE_MAG_FILTER: 0x2800,
    TEXTURE_WRAP_S: 0x2802,
    TEXTURE_WRAP_T: 0x2803,
    LINEAR: 0x2601,
    CLAMP_TO_EDGE: 0x812f,
    RED: 0x1903,
    RGBA: 0x1908,
    FLOAT: 0x1406,
    HALF_FLOAT: 0x140b,
    R32F: 0x822e,
    RGBA32F: 0x8814,
    R16F: 0x822d,
    RGBA16F: 0x881a,
    createTexture: () => ({}),
    createFramebuffer: () => ({}),
    bindTexture: () => {},
    bindFramebuffer: () => {},
    framebufferTexture2D: () => {},
    texImage2D: (...args: unknown[]) => { calls.texImage2D.push(args); },
    texParameteri: (_target: number, name: number, value: number) => { calls.texParameteri.set(name, value); },
  };
  return { gl: gl as unknown as WebGL2RenderingContext, enums: gl, calls };
}

describe('createFramebuffer', () => {
  it.each([
    ['float32', 4, 'RGBA32F', 'RGBA', 'FLOAT'],
    ['float32', 1, 'R32F', 'RED', 'FLOAT'],
    ['float16', 4, 'RGBA16F', 'RGBA', 'HALF_FLOAT'],
    ['float16', 1, 'R16F', 'RED', 'HALF_FLOAT'],
  ] as const)('%s × %d channels allocates %s', (precision, channels, internal, format, type) => {
    const { gl, enums, calls } = recordingContext();
    const fb = createFramebuffer(gl, 64, 32, channels, precision);
    const [, level, internalFormat, width, height, border, fmt, texelType, data] = calls.texImage2D[0];
    expect({ level, width, height, border, data }).toEqual({ level: 0, width: 64, height: 32, border: 0, data: null });
    expect(internalFormat).toBe(enums[internal]);
    expect(fmt).toBe(enums[format]);
    expect(texelType).toBe(enums[type]);
    expect(fb).toMatchObject({ width: 64, height: 32, channels });
  });

  it.each(['float32', 'float16'] as const)('%s targets sample with LINEAR filtering', (precision) => {
    const { gl, enums, calls } = recordingContext();
    createFramebuffer(gl, 8, 8, 4, precision);
    expect(calls.texParameteri.get(enums.TEXTURE_MIN_FILTER)).toBe(enums.LINEAR);
    expect(calls.texParameteri.get(enums.TEXTURE_MAG_FILTER)).toBe(enums.LINEAR);
  });
});
