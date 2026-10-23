import { describe, it, expect, vi, afterEach } from 'vitest';
import { FloatRenderTargets } from '../floatRenderTargets';

const FLOAT32_SET = ['EXT_color_buffer_float', 'EXT_float_blend', 'OES_texture_float_linear'];

/** A context exposing exactly `extensions`, recording every request. */
function stubContext(extensions: string[]) {
  const requested: string[] = [];
  const loseContext = vi.fn();
  const gl = {
    getExtension: (name: string) => {
      requested.push(name);
      if (name === 'WEBGL_lose_context') return { loseContext };
      return extensions.includes(name) ? {} : null;
    },
  };
  return { gl, requested, loseContext };
}

describe('FloatRenderTargets.enable', () => {
  it('picks float32 where 32-bit targets blend and filter (desktop)', () => {
    expect(FloatRenderTargets.enable(stubContext(FLOAT32_SET).gl)).toBe('float32');
  });

  it('falls back to float16 without 32-bit blending (Apple phone GPUs)', () => {
    const { gl } = stubContext(['EXT_color_buffer_float', 'OES_texture_float_linear']);
    expect(FloatRenderTargets.enable(gl)).toBe('float16');
  });

  it('falls back to float16 without 32-bit linear filtering (iOS, many Android GPUs)', () => {
    const { gl } = stubContext(['EXT_color_buffer_float', 'EXT_float_blend']);
    expect(FloatRenderTargets.enable(gl)).toBe('float16');
  });

  it('falls back to float16 with renderable float textures only', () => {
    expect(FloatRenderTargets.enable(stubContext(['EXT_color_buffer_float']).gl)).toBe('float16');
  });

  it('returns null without renderable float textures', () => {
    const { gl } = stubContext(['EXT_float_blend', 'OES_texture_float_linear']);
    expect(FloatRenderTargets.enable(gl)).toBeNull();
  });

  it('enables every extension the chosen tier uses on the given context', () => {
    const { gl, requested } = stubContext(FLOAT32_SET);
    FloatRenderTargets.enable(gl);
    expect(requested).toEqual(expect.arrayContaining(FLOAT32_SET));
  });
});

describe('FloatRenderTargets.probe', () => {
  afterEach(() => vi.unstubAllGlobals());

  // The probe caches its first answer per module instance, so every test
  // imports a fresh copy after arranging the environment.
  async function loadFresh() {
    vi.resetModules();
    return (await import('../floatRenderTargets')).FloatRenderTargets;
  }

  function stubDocument(context: unknown) {
    const createElement = vi.fn(() => ({ getContext: () => context }));
    vi.stubGlobal('document', { createElement });
    return createElement;
  }

  it('reports the precision of a throwaway context and frees it', async () => {
    const { gl, loseContext } = stubContext(['EXT_color_buffer_float']);
    stubDocument(gl);
    const targets = await loadFresh();
    expect(targets.probe()).toBe('float16');
    expect(targets.available()).toBe(true);
    expect(loseContext).toHaveBeenCalledTimes(1);
  });

  it('probes once and caches the answer', async () => {
    const createElement = stubDocument(stubContext(FLOAT32_SET).gl);
    const targets = await loadFresh();
    expect(targets.probe()).toBe('float32');
    expect(targets.probe()).toBe('float32');
    expect(createElement).toHaveBeenCalledTimes(1);
  });

  it('is unavailable without WebGL2', async () => {
    stubDocument(null);
    const targets = await loadFresh();
    expect(targets.probe()).toBeNull();
    expect(targets.available()).toBe(false);
  });

  it('is unavailable when context creation throws', async () => {
    vi.stubGlobal('document', {
      createElement: () => ({ getContext: () => { throw new Error('blocked'); } }),
    });
    const targets = await loadFresh();
    expect(targets.probe()).toBeNull();
  });

  it('is unavailable during SSR (no document)', async () => {
    const targets = await loadFresh();
    expect(targets.probe()).toBeNull();
  });
});
