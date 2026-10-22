import { describe, it, expect, vi, afterEach } from 'vitest';

// The capability probe caches its first answer per module instance, so every
// test imports a fresh copy after arranging the environment.
async function loadSupport() {
  vi.resetModules();
  return import('../densityHaze/support');
}

/** A document whose canvases offer WebGL2 with renderable float textures. */
function stubFloatTargets(ok: boolean) {
  vi.stubGlobal('document', {
    createElement: () => ({
      getContext: () => ({
        getExtension: (name: string) =>
          name === 'EXT_color_buffer_float' ? (ok ? {} : null) : { loseContext: () => {} },
      }),
    }),
  });
}

describe('supportedHazeRenderer', () => {
  afterEach(() => vi.unstubAllGlobals());

  it('keeps the sprite renderer everywhere', async () => {
    const { supportedHazeRenderer } = await loadSupport();
    expect(supportedHazeRenderer('sprites')).toBe('sprites');
  });

  it('keeps density where float render targets exist', async () => {
    stubFloatTargets(true);
    const { supportedHazeRenderer } = await loadSupport();
    expect(supportedHazeRenderer('density')).toBe('density');
  });

  it('degrades density to sprites without float render targets', async () => {
    stubFloatTargets(false);
    const { supportedHazeRenderer } = await loadSupport();
    expect(supportedHazeRenderer('density')).toBe('sprites');
  });

  it('degrades density to sprites during SSR (no document)', async () => {
    const { supportedHazeRenderer } = await loadSupport();
    expect(supportedHazeRenderer('density')).toBe('sprites');
  });
});
